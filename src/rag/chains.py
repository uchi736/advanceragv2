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
    get_answer_generation_prompt
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
            doc_key = doc.page_content if hasattr(doc, 'page_content') else str(doc)

            if doc_key not in fused_scores:
                fused_scores[doc_key] = 0
                doc_map[doc_key] = doc

            fused_scores[doc_key] += 1 / (rank + k)

    sorted_docs = sorted(
        doc_map.values(),
        key=lambda d: fused_scores[d.page_content if hasattr(d, 'page_content') else str(d)],
        reverse=True
    )

    return sorted_docs

def _rerank_documents_with_llm(docs: List[Any], question: str, llm, reranking_prompt) -> List[Any]:
    """Reranks documents based on relevance to the question using LLM."""
    if not docs:
        return []

    # Limit to top 20 docs for reranking to control cost
    docs_to_rerank = docs[:20]

    reranking_chain = reranking_prompt | llm | JsonOutputParser()

    reranking_input = {
        "question": question,
        "documents": [
            {"index": i, "content": doc.page_content[:500] if hasattr(doc, 'page_content') else str(doc)[:500]}
            for i, doc in enumerate(docs_to_rerank)
        ]
    }

    try:
        rankings = reranking_chain.invoke(reranking_input)
        reranked_indices = rankings.get("rankings", list(range(len(docs_to_rerank))))
        reranked_docs = [docs_to_rerank[i] for i in reranked_indices if i < len(docs_to_rerank)]
        # Add any remaining docs that weren't reranked
        reranked_docs.extend(docs[20:])
        return reranked_docs
    except:
        # If reranking fails, return original order
        return docs

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
        from .reverse_lookup import ReverseLookupEngine

        original_question = input_dict["question"]
        jargon_terms = jargon_extraction_chain.invoke({"question": original_question, "max_terms": config_obj.max_jargon_terms_per_query}).split('\n')
        jargon_terms = [t.strip() for t in jargon_terms if t.strip()]

        # Initialize reverse lookup engine
        reverse_engine = ReverseLookupEngine(
            jargon_manager=jargon_manager,
            vector_store=getattr(retriever, 'vector_store', None)
        )

        # Perform reverse lookup to find technical terms from descriptions
        reverse_results = reverse_engine.reverse_lookup(original_question, top_k=5)
        reverse_terms = [r.term for r in reverse_results if r.confidence > 0.5]

        # Combine forward extracted terms and reverse lookup terms
        all_terms = list(set(jargon_terms + reverse_terms))

        augmented_query = original_question
        augmentation_payload = input_dict.get("jargon_augmentation", {})

        if all_terms:
            jargon_defs = jargon_manager.lookup_terms(all_terms)
            if jargon_defs:
                defs_text = "\n".join([f"- {term}: {info['definition']}" for term, info in jargon_defs.items()])
                augmented_query = query_augmentation_chain.invoke({"original_question": original_question, "jargon_definitions": defs_text})
                augmentation_payload = {
                    "extracted_terms": jargon_terms,
                    "reverse_terms": reverse_terms,
                    "matched_terms": jargon_defs,
                    "augmented_query": augmented_query
                }
            else:
                augmentation_payload = {
                    "extracted_terms": jargon_terms,
                    "reverse_terms": reverse_terms,
                    "matched_terms": {},
                    "augmented_query": original_question
                }

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

    def retrieve_with_optional_fusion(input_dict):
        use_query_expansion = input_dict.get("use_query_expansion", False)
        use_rag_fusion = input_dict.get("use_rag_fusion", False)
        search_type = input_dict.get("search_type", "hybrid")

        query = input_dict.get("retrieval_query", input_dict["question"])

        if use_rag_fusion:
            # RAG Fusion: expand query and perform RRF
            expanded_queries = expand_query(input_dict)
            doc_lists = []
            for q in expanded_queries[:5]:  # Limit to 5 queries
                # retriever.invoke() expects string, not dict
                docs = retriever.invoke(q)
                doc_lists.append(docs)
            fused_docs = _reciprocal_rank_fusion(doc_lists)
            input_dict["query_expansion"] = {"queries": expanded_queries[:5], "method": "rag_fusion"}
            return fused_docs
        elif use_query_expansion:
            # Simple query expansion (concatenate)
            expanded_queries = expand_query(input_dict)
            combined_query = " ".join(expanded_queries[:3])
            input_dict["query_expansion"] = {"queries": expanded_queries[:3], "method": "simple"}
            # retriever.invoke() expects string, not dict
            return retriever.invoke(combined_query)
        else:
            # Direct retrieval
            # retriever.invoke() expects string, not dict
            return retriever.invoke(query)

    def rerank_if_enabled(input_dict):
        docs = input_dict["documents"]
        use_reranking = input_dict.get("use_reranking", False)

        if use_reranking and docs:
            reranking_prompt = get_reranking_prompt()
            question = input_dict.get("question")
            reranked = _rerank_documents_with_llm(docs, question, llm, reranking_prompt)
            input_dict["reranking"] = {"enabled": True, "original_count": len(docs), "reranked_count": len(reranked)}
            return reranked
        else:
            input_dict["reranking"] = {"enabled": False}
            return docs

    # Conditional jargon augmentation
    jargon_branch = RunnableBranch(
        (
            lambda x: x.get("use_jargon_augmentation", False),
            RunnableLambda(augment_query_with_jargon)
        ),
        # Default: pass through
        RunnablePassthrough()
    )

    # Build the retrieval chain
    retrieval_chain = (
        RunnablePassthrough()
        | jargon_branch
        | RunnablePassthrough.assign(
            documents=RunnableLambda(retrieve_with_optional_fusion)
        )
        | RunnablePassthrough.assign(
            documents=RunnableLambda(rerank_if_enabled)
        )
    )

    return retrieval_chain


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