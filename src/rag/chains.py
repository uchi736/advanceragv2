from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableLambda, RunnableBranch, Runnable, ConfigurableField
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from operator import itemgetter
from typing import List, Dict, Any
from .prompts import (
    get_jargon_extraction_prompt,
    get_query_augmentation_prompt,
    get_query_expansion_prompt,
    get_reranking_prompt,
    get_answer_generation_prompt,
    get_semantic_router_prompt,
    get_multi_table_text_to_sql_prompt,
    get_sql_answer_generation_prompt,
    get_synthesis_prompt
)

# Forward declaration to avoid circular import
class JapaneseHybridRetriever:
    pass
class JargonDictionaryManager:
    pass

def _format_docs(docs: List[Any]) -> str:
    """Helper function to format documents for context."""
    if not docs:
        return "コンテキスト情報が見つかりませんでした。"
    
    formatted_docs = []
    for doc in docs:
        if hasattr(doc, 'page_content'):
            content = doc.page_content.strip()
            if content:
                formatted_docs.append(content)
    
    if not formatted_docs:
        return "ドキュメントが空です。"
    
    return "\n\n".join(formatted_docs)

def _reciprocal_rank_fusion(doc_lists: List[List[Any]], k=60) -> List[Any]:
    """Performs Reciprocal Rank Fusion on a list of document lists."""
    if not doc_lists:
        return []
    
    fused_scores: Dict[str, float] = {}
    doc_map: Dict[str, Any] = {}

    for docs in doc_lists:
        for rank, doc in enumerate(docs):
            doc_id = doc.metadata.get("chunk_id")
            if not doc_id:
                continue
            
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
            
            previous_score = fused_scores.get(doc_id, 0.0)
            fused_scores[doc_id] = previous_score + 1.0 / (k + rank + 1)

    sorted_docs = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
    return [doc_map[doc_id] for doc_id in sorted_docs]

def _combine_documents_simple(list_of_document_lists: List[List[Any]]) -> List[Any]:
    """A simple combination of documents, preserving order and removing duplicates."""
    if not list_of_document_lists:
        return []
    
    all_docs: List[Any] = []
    seen_chunk_ids = set()
    for doc_list in list_of_document_lists:
        for doc in doc_list:
            chunk_id = doc.metadata.get('chunk_id', '')
            if chunk_id and chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                all_docs.append(doc)
    
    return all_docs

def create_retrieval_chain(
    llm: Runnable, 
    retriever: JapaneseHybridRetriever, 
    jargon_manager: JargonDictionaryManager, 
    config_obj: Any
) -> Runnable:
    """
    Creates a traceable chain that handles up to the document retrieval and reranking steps.
    """
    jargon_extraction_prompt = get_jargon_extraction_prompt(config_obj.max_jargon_terms_per_query)
    jargon_extraction_chain = jargon_extraction_prompt | llm | StrOutputParser()

    query_augmentation_prompt = get_query_augmentation_prompt()
    query_augmentation_chain = query_augmentation_prompt | llm | StrOutputParser()

    def augment_query_with_jargon(input_dict: dict) -> dict:
        original_question = input_dict["question"]
        jargon_terms = jargon_extraction_chain.invoke({"question": original_question, "max_terms": config_obj.max_jargon_terms_per_query}).split('\n')
        jargon_terms = [t.strip() for t in jargon_terms if t.strip()]
        augmented_query = original_question
        augmentation_payload = input_dict.get("jargon_augmentation", {})

        if jargon_terms:
            jargon_defs = jargon_manager.lookup_terms(jargon_terms)
            if jargon_defs:
                defs_text = "\n".join([f"- {term}: {info['definition']}" for term, info in jargon_defs.items()])
                augmented_query = query_augmentation_chain.invoke({"original_question": original_question, "jargon_definitions": defs_text})
                augmentation_payload = {
                    "extracted_terms": jargon_terms,
                    "matched_terms": jargon_defs,  # Dictionary, not list
                    "augmented_query": augmented_query
                }
            else:
                augmentation_payload = {"extracted_terms": jargon_terms, "matched_terms": {}, "augmented_query": original_question}

        updated = {**input_dict, "retrieval_query": augmented_query}
        if augmentation_payload:
            updated["jargon_augmentation"] = augmentation_payload
        return updated

    query_expansion_prompt = get_query_expansion_prompt()
    
    def expand_query(input_dict):
        result = (query_expansion_prompt | llm | StrOutputParser()).invoke(input_dict)
        expanded_queries = [q.strip() for q in result.split('\n') if q.strip()]
        # Ensure the original question is always available for downstream logging/UI.
        if input_dict.get("question") and input_dict["question"] not in expanded_queries:
            expanded_queries.insert(0, input_dict["question"])
        return expanded_queries
    
    query_expansion_chain = RunnableLambda(expand_query)

    def get_retriever_with_search_type(input_or_config: Any):
        search_type = "ハイブリッド検索"
        if isinstance(input_or_config, dict) and "configurable" in input_or_config:
             search_type = input_or_config.get("configurable", {}).get("search_type", "ハイブリッド検索")
        return retriever.with_config(configurable={"search_type": search_type})

    def get_docs_for_batch(input_dict):
        """Batch retrieve documents for expanded queries"""
        expanded_queries = input_dict.get("expanded_queries", [])
        search_type = input_dict.get("search_type", "ハイブリッド検索")
        config = input_dict.get("config")
        
        if not expanded_queries:
            return []
        
        retriever_with_config = retriever.with_config(configurable={"search_type": search_type})
        doc_lists = retriever_with_config.batch(expanded_queries, config)
        return doc_lists

    expansion_retrieval_chain = RunnablePassthrough.assign(
        expanded_queries=query_expansion_chain
    ).assign(
        doc_lists=RunnableLambda(get_docs_for_batch)
    ).assign(
        documents=RunnableBranch(
            (lambda x: x.get("use_rag_fusion"), itemgetter("doc_lists") | RunnableLambda(_reciprocal_rank_fusion)),
            itemgetter("doc_lists") | RunnableLambda(_combine_documents_simple)
        )
    )

    standard_retrieval_chain = RunnablePassthrough.assign(
        documents=lambda x: retriever.with_config(configurable={"search_type": x.get("search_type")}).invoke(x["retrieval_query"], x.get("config"))
    )

    reranking_prompt = get_reranking_prompt()
    reranking_chain = reranking_prompt | llm | StrOutputParser()

    def rerank_documents(input_dict: dict) -> List[Any]:
        docs = input_dict["documents"]
        if not docs: return []
        docs_for_rerank = [f"ドキュメント {i}:\n{doc.page_content}" for i, doc in enumerate(docs)]
        rerank_input = {"question": input_dict["question"], "documents": "\n\n---\n\n".join(docs_for_rerank)}
        try:
            reranked_indices_str = reranking_chain.invoke(rerank_input)
            reranked_indices = [int(i.strip()) for i in reranked_indices_str.split(',') if i.strip().isdigit()]
            reranked_docs = [docs[i] for i in reranked_indices if i < len(docs)]
            if reranked_indices:
                input_dict["reranking"] = {
                    "original_order": list(range(len(docs))),
                    "reranked_order": reranked_indices,
                    "applied": True
                }
            return reranked_docs if reranked_docs else docs
        except Exception:
            input_dict.setdefault("reranking", {"applied": False})
            return docs

    retrieval_and_rerank_chain = (
        RunnableBranch(
            (lambda x: x.get("use_jargon_augmentation"), RunnableLambda(augment_query_with_jargon)),
            RunnablePassthrough.assign(retrieval_query=itemgetter("question"))
        )
        | RunnableBranch(
            (lambda x: x.get("use_rag_fusion") or x.get("use_query_expansion"), expansion_retrieval_chain),
            standard_retrieval_chain
        )
        | RunnablePassthrough.assign(
            documents=RunnableBranch(
                (lambda x: x.get("use_reranking"), RunnableLambda(rerank_documents)),
                itemgetter("documents")
            )
        )
    )
    return retrieval_and_rerank_chain

def create_full_rag_chain(retrieval_chain: Runnable, llm: Runnable) -> Runnable:
    """Creates the final answer generation part of the RAG chain."""
    answer_generation_prompt = get_answer_generation_prompt()
    
    full_chain = (
        retrieval_chain
        | RunnablePassthrough.assign(
            answer=(
                RunnablePassthrough.assign(context=itemgetter("documents") | RunnableLambda(_format_docs))
                | answer_generation_prompt
                | llm
                | StrOutputParser()
            )
        )
    )
    return full_chain

def create_chains(llm, max_sql_results: int) -> dict:
    """Creates and returns a dictionary of all LangChain runnables for SQL and Synthesis."""
    semantic_router_prompt = get_semantic_router_prompt()
    semantic_router_chain = semantic_router_prompt | llm | JsonOutputParser()

    multi_table_text_to_sql_prompt = get_multi_table_text_to_sql_prompt(max_sql_results)
    multi_table_sql_chain = multi_table_text_to_sql_prompt | llm | StrOutputParser()

    sql_answer_generation_prompt = get_sql_answer_generation_prompt()
    sql_answer_generation_chain = sql_answer_generation_prompt | llm | StrOutputParser()

    synthesis_prompt = get_synthesis_prompt()
    synthesis_chain = synthesis_prompt | llm | StrOutputParser()

    return {
        "semantic_router": semantic_router_chain,
        "multi_table_sql": multi_table_sql_chain,
        "sql_answer_generation": sql_answer_generation_chain,
        "synthesis": synthesis_chain,
    }
