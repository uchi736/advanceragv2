"""
rag_system_enhanced.py
======================
This file acts as a facade for the RAG system, assembling components
from the `rag` subdirectory into a cohesive `RAGSystem` class.
"""
from __future__ import annotations

import os
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


try:
    import psycopg
    _PG_DIALECT = "psycopg"
except ModuleNotFoundError:
    _PG_DIALECT = "psycopg2"

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.runnables import RunnableConfig
from langchain_community.callbacks.manager import get_openai_callback
from langchain_community.vectorstores import PGVector

# --- Refactored Module Imports ---
from src.rag.config import Config
from src.rag.text_processor import JapaneseTextProcessor
from src.rag.term_extraction import JargonDictionaryManager  # 統合されたモジュールから
from src.rag.retriever import JapaneseHybridRetriever
from src.rag.ingestion import IngestionHandler
from src.rag.sql_handler import SQLHandler
from src.rag.chains import create_chains, create_retrieval_chain, create_full_rag_chain
from src.rag.evaluator import RAGEvaluator, EvaluationResults, EvaluationMetrics

# load_dotenv()  # Commented out - loaded in main script

def format_docs(docs: List[Any]) -> str:
    """Helper function to format documents for context."""
    if not docs:
        return "(コンテキスト無し)"
    return "\n\n".join([f"[ソース {i+1} ChunkID: {d.metadata.get('chunk_id', 'N/A')}]\n{d.page_content}" for i, d in enumerate(docs)])

class RAGSystem:
    def __init__(self, cfg: Config):
        self.config = cfg
        self.text_processor = JapaneseTextProcessor()

        # PostgreSQL connection
        self.connection_string = f"postgresql+{_PG_DIALECT}://{cfg.db_user}:{cfg.db_password}@{cfg.db_host}:{cfg.db_port}/{cfg.db_name}"
        # Create a shared SQLAlchemy engine to avoid per-call engine creation downstream.
        self.engine: Engine = create_engine(self.connection_string, pool_pre_ping=True)

        self._init_llms_and_embeddings()
        self._init_db()

        # Create PGVector store
        from langchain_community.vectorstores.pgvector import DistanceStrategy
        self.vector_store = PGVector(
            connection_string=self.connection_string,
            embedding_function=self.embeddings,
            collection_name=cfg.collection_name,
            distance_strategy=getattr(cfg, 'distance_strategy', DistanceStrategy.COSINE)
        )

        # Use Japanese hybrid retriever for PGVector
        self.retriever = JapaneseHybridRetriever(
            vector_store=self.vector_store,
            connection_string=self.connection_string,
            engine=self.engine,
            config_params=cfg,
            text_processor=self.text_processor
        )

        # Initialize components for PGVector
        self.jargon_manager = JargonDictionaryManager(
            self.connection_string,
            cfg.jargon_table_name,
            engine=self.engine
        )
        self.ingestion_handler = IngestionHandler(
            cfg,
            self.vector_store,
            self.engine,
            self.text_processor
        )
        self.sql_handler = SQLHandler(cfg, self.llm, self.connection_string, engine=self.engine)

        # Create the modular chains
        self.retrieval_chain = create_retrieval_chain(self.llm, self.retriever, self.jargon_manager, self.config)
        self.rag_chain = create_full_rag_chain(self.retrieval_chain, self.llm)

        # Create the remaining chains (mostly for SQL and synthesis)
        self.chains = create_chains(self.llm, cfg.max_sql_results)
        if self.sql_handler:
            self.sql_handler.multi_table_sql_chain = self.chains["multi_table_sql"]
            self.sql_handler.sql_answer_generation_chain = self.chains["sql_answer_generation"]

        # Initialize evaluator
        self.evaluator = None  # Lazy initialization to avoid overhead when not needed

    def _init_llms_and_embeddings(self):
        """Initialize LLM and embedding models with Azure OpenAI."""
        print("RAGSystem initialized with Azure OpenAI.")
        self.llm = AzureChatOpenAI(
            temperature=getattr(self.config, 'llm_temperature', 0.0),
            max_tokens=getattr(self.config, 'max_tokens', 4096),
            azure_deployment=self.config.azure_openai_chat_deployment_name,
            api_key=self.config.azure_openai_api_key,
            azure_endpoint=self.config.azure_openai_endpoint,
            api_version=self.config.azure_openai_api_version
        )
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=self.config.azure_openai_embedding_deployment_name,
            api_key=self.config.azure_openai_api_key,
            azure_endpoint=self.config.azure_openai_endpoint,
            api_version=self.config.azure_openai_api_version
        )

    def _init_db(self):
        """Initialize database tables and extensions."""
        with self.engine.connect() as conn, conn.begin():
            # Enable pgvector extension
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

            # Create table for keyword search
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id SERIAL PRIMARY KEY,
                    collection_name VARCHAR(255),
                    document_id VARCHAR(255),
                    chunk_id VARCHAR(255) UNIQUE NOT NULL,
                    content TEXT NOT NULL,
                    tokenized_content TEXT,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_chunk_collection ON document_chunks(collection_name)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_chunk_document ON document_chunks(document_id)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_chunk_id ON document_chunks(chunk_id)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_tokenized_gin ON document_chunks USING gin(to_tsvector('simple', tokenized_content))"))

            # Create parent-child chunk table
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS parent_child_chunks (
                    id SERIAL PRIMARY KEY,
                    collection_name VARCHAR(255),
                    parent_chunk_id VARCHAR(255) UNIQUE NOT NULL,
                    parent_content TEXT NOT NULL,
                    child_chunk_ids TEXT[] NOT NULL,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_parent_chunk_collection ON parent_child_chunks(collection_name)"))
            conn.execute(text("CREATE INDEX IF NOT EXISTS idx_parent_chunk_id ON parent_child_chunks(parent_chunk_id)"))

    def delete_jargon_terms(self, terms: List[str]) -> tuple[int, int]:
        if not self.jargon_manager:
            return 0, 0
        return self.jargon_manager.delete_terms(terms)

    def delete_document_by_id(self, doc_id: str) -> tuple[bool, str]:
        """Delete a document by its ID.

        Args:
            doc_id: Document ID to delete

        Returns:
            Tuple of (success: bool, message: str)
        """
        if not hasattr(self, 'ingestion_handler') or self.ingestion_handler is None:
            return False, "Ingestion handler not available"

        return self.ingestion_handler.delete_document_by_id(doc_id)

    def ingest_documents(self, paths: List[str]) -> None:
        """Ingest documents into the vector store.

        Args:
            paths: List of file paths to ingest
        """
        if not hasattr(self, 'ingestion_handler') or self.ingestion_handler is None:
            raise ValueError("Ingestion handler not available")

        self.ingestion_handler.ingest_documents(paths)

    def get_chunks_by_document_id(self, doc_id: str):
        """Get chunks for a specific document ID.

        Args:
            doc_id: Document ID to get chunks for

        Returns:
            DataFrame with chunk information
        """
        if not hasattr(self, 'sql_handler') or self.sql_handler is None:
            raise ValueError("SQL handler not available")

        return self.sql_handler.get_chunks_by_document_id(doc_id)

    # --- Core Query Logic ---
    def query(self, question: str, search_type: str = None) -> Dict[str, Any]:
        """Standard query operation using RAG chain."""
        with get_openai_callback() as cb:
            result = self.retrieval_chain.invoke({"question": question})

            # Calculate confidence scores for retrieved docs
            if "context" in result and hasattr(result["context"], "__len__"):
                result["confidence_scores"] = [0.85] * len(result["context"])

            # Add total tokens
            result["total_tokens"] = cb.total_tokens if hasattr(cb, "total_tokens") else 0

            return result

    def rag_search(self, question: str, search_type: str = "hybrid") -> Dict[str, Any]:
        """Perform RAG search with specified search type.

        Args:
            question: The query to search for
            search_type: Type of search - "hybrid", "vector", or "keyword"

        Returns:
            Dictionary containing search results and metadata
        """
        # Set the search type on the retriever
        if hasattr(self.retriever, 'search_type'):
            original_search_type = self.retriever.search_type
            self.retriever.search_type = search_type

        try:
            # Create prompt for the query
            if self.config.enable_jargon_augmentation:
                rag_prompt = ChatPromptTemplate.from_messages([
                    ("system", """あなたはAzure RAGシステムのアシスタントです。
与えられたコンテキストに基づいて、ユーザーの質問に正確に答えてください。
コンテキストに情報がない場合は、「情報が見つかりません」と回答してください。

専門用語の定義が提供されている場合は、それを考慮して回答してください。"""),
                    ("human", """コンテキスト: {context}

専門用語定義（もしあれば）: {jargon_definitions}

質問: {question}

答え:""")
                ])
            else:
                rag_prompt = ChatPromptTemplate.from_messages([
                    ("system", """あなたはAzure RAGシステムのアシスタントです。
与えられたコンテキストに基づいて、ユーザーの質問に正確に答えてください。
コンテキストに情報がない場合は、「情報が見つかりません」と回答してください。"""),
                    ("human", """コンテキスト: {context}

質問: {question}

答え:""")
                ])

            # Execute retrieval and generation
            with get_openai_callback() as cb:
                rag_results = self.retrieval_chain.invoke({"question": question})

                # Process jargon if enabled
                jargon_definitions = ""
                if self.config.enable_jargon_augmentation and "jargon_augmentation" in rag_results:
                    jargon_info = rag_results["jargon_augmentation"]
                    if jargon_info.get("matched_terms"):
                        jargon_definitions = "\n".join([f"- {term}: {info['definition']}" for term, info in jargon_info["matched_terms"].items()])

                # Format context
                # Note: retrieval_chain returns "documents" key, not "context"
                docs = rag_results.get("documents") or rag_results.get("context", [])
                context = format_docs(docs)

                # Generate answer
                chain = rag_prompt | self.llm | StrOutputParser()
                answer = chain.invoke({
                    "context": context,
                    "question": question,
                    "jargon_definitions": jargon_definitions
                })

                rag_results["answer"] = answer
                rag_results["total_tokens"] = cb.total_tokens if hasattr(cb, "total_tokens") else 0

            return rag_results

        finally:
            # Restore original search type
            if hasattr(self.retriever, 'search_type'):
                self.retriever.search_type = original_search_type

    def synthesize(self, question: str) -> Dict[str, Any]:
        """Synthesize answer using both RAG and SQL results."""
        rag_results = self.query(question, search_type="hybrid")

        # SQL query if handler available
        sql_results = None
        if self.sql_handler:
            try:
                sql_results = self.sql_handler.execute_nl_to_sql(question, self.sql_handler.multi_table_sql_chain)
            except Exception as e:
                print(f"SQL query error: {e}")

        if not sql_results or "error" in sql_results:
            # Note: retrieval_chain returns "documents" key, not "context"
            docs = rag_results.get("documents") or rag_results.get("context", [])
            return {
                "answer": rag_results.get("answer", ""),
                "source": "rag",
                "context": docs,
                "confidence_scores": rag_results.get("confidence_scores", []),
                "jargon_augmentation": rag_results.get("jargon_augmentation", {}),
                "query_expansion": rag_results.get("query_expansion", []),
                "retrieval_query": rag_results.get("retrieval_query", question),
                "golden_retriever": {}
            }

        # Synthesize with SQL results if available
        synthesis_chain = self.chains.get("synthesis")
        if synthesis_chain:
            synthesized = synthesis_chain.invoke({
                "question": question,
                "rag_answer": rag_results.get("answer", "情報なし"),
                "sql_query": sql_results.get("sql_query", ""),
                "sql_result": sql_results.get("result", ""),
                "sql_answer": sql_results.get("answer", "")
            })
            answer = synthesized
        else:
            answer = rag_results.get("answer", "")

        # Note: retrieval_chain returns "documents" key, not "context"
        docs = rag_results.get("documents") or rag_results.get("context", [])
        return {
            "answer": answer,
            "source": "synthesis",
            "rag_results": rag_results,
            "sql_results": sql_results,
            "context": docs,
            "confidence_scores": rag_results.get("confidence_scores", []),
            "jargon_augmentation": rag_results.get("jargon_augmentation", {}),
            "query_expansion": rag_results.get("query_expansion", []),
            "retrieval_query": rag_results.get("retrieval_query", question),
            "golden_retriever": {}
        }

    def query_unified(
        self,
        question: str,
        use_query_expansion: bool = False,
        use_rag_fusion: bool = False,
        use_jargon_augmentation: bool = False,
        use_reranking: bool = False,
        search_type: str = "ハイブリッド検索",
        config: Any = None
    ) -> Dict[str, Any]:
        """Unified query method supporting all advanced RAG features.

        Args:
            question: User query
            use_query_expansion: Enable query expansion
            use_rag_fusion: Enable RAG fusion (query expansion + RRF)
            use_jargon_augmentation: Enable jargon augmentation
            use_reranking: Enable LLM reranking
            search_type: Search type ("ハイブリッド検索", "ベクトル検索", "キーワード検索")
            config: Optional RunnableConfig for tracing

        Returns:
            Dictionary with answer, sources, and metadata
        """
        from langchain_community.callbacks import get_openai_callback
        from langchain_core.output_parsers import StrOutputParser
        from src.rag.prompts import get_answer_generation_prompt

        # Map Japanese search type to English
        search_type_map = {
            "ハイブリッド検索": "hybrid",
            "ベクトル検索": "vector",
            "キーワード検索": "keyword"
        }
        search_type_eng = search_type_map.get(search_type, "hybrid")

        with get_openai_callback() as cb:
            # Prepare input for retrieval chain
            retrieval_input = {
                "question": question,
                "use_query_expansion": use_query_expansion,
                "use_rag_fusion": use_rag_fusion,
                "use_jargon_augmentation": use_jargon_augmentation,
                "use_reranking": use_reranking,
                "search_type": search_type_eng,
                "config": config
            }

            # Execute retrieval chain (includes all advanced features)
            retrieval_result = self.retrieval_chain.invoke(retrieval_input, config=config)

            # Extract documents
            documents = retrieval_result.get("documents", [])

            # Format context for answer generation
            def format_docs(docs):
                return "\n\n---\n\n".join([doc.page_content for doc in docs])

            context = format_docs(documents)

            # Get jargon definitions if augmentation was used
            jargon_definitions = ""
            if use_jargon_augmentation and "jargon_augmentation" in retrieval_result:
                jargon_info = retrieval_result["jargon_augmentation"]
                if jargon_info.get("matched_terms"):
                    jargon_definitions = "\n".join([
                        f"- {term}: {info['definition']}"
                        for term, info in jargon_info["matched_terms"].items()
                    ])

            # Generate answer using LLM
            answer_prompt = get_answer_generation_prompt()
            answer_chain = answer_prompt | self.llm | StrOutputParser()

            answer = answer_chain.invoke({
                "context": context,
                "question": question,
                "jargon_definitions": jargon_definitions
            }, config=config)

            # Check if SQL handler should be used
            sql_details = None
            if self.sql_handler:
                try:
                    sql_result = self.sql_handler.execute_nl_to_sql(
                        question,
                        self.sql_handler.multi_table_sql_chain
                    )
                    if sql_result and "error" not in sql_result:
                        sql_details = sql_result
                        # Optionally synthesize SQL + RAG answer here
                except Exception as e:
                    print(f"SQL execution error: {e}")

            # Build sources from documents
            sources = []
            for doc in documents:
                if hasattr(doc, 'metadata'):
                    sources.append({
                        "content": doc.page_content if hasattr(doc, 'page_content') else str(doc),
                        "metadata": doc.metadata
                    })

            # Build result
            result = {
                "answer": answer,
                "sources": sources,
                "context": documents,
                "total_tokens": cb.total_tokens if hasattr(cb, "total_tokens") else 0,
                "retrieval_query": retrieval_result.get("retrieval_query", question),
                "query_expansion": retrieval_result.get("query_expansion", {}),
                "golden_retriever": retrieval_result.get("golden_retriever", {}),
                "jargon_augmentation": retrieval_result.get("jargon_augmentation", {}),
                "reranking": retrieval_result.get("reranking", {})
            }

            if sql_details:
                result["sql_details"] = sql_details

            return result

    async def extract_terms(self, input_dir: str | Path, output_json: str | Path) -> None:
        from src.rag.term_extraction import run_extraction_pipeline
        # Use connection_string only if using PGVector, otherwise pass None
        pg_url = self.connection_string
        await run_extraction_pipeline(
            Path(input_dir), Path(output_json),
            self.config, self.llm, self.embeddings,
            self.vector_store, pg_url, self.config.jargon_table_name,
            jargon_manager=None
        )
        print(f"[TermExtractor] Extraction complete -> {output_json}")

    # --- Evaluation Methods ---
    def initialize_evaluator(self,
                           k_values: List[int] = [1, 3, 5, 10],
                           similarity_method: str = "azure_embedding",
                           similarity_threshold: float = None) -> RAGEvaluator:
        """Initialize the evaluation system"""
        if self.evaluator is None:
            # Use config.confidence_threshold if similarity_threshold not provided
            threshold = similarity_threshold if similarity_threshold is not None else self.config.confidence_threshold
            self.evaluator = RAGEvaluator(
                config=self.config,
                k_values=k_values,
                similarity_method=similarity_method,
                similarity_threshold=threshold
            )
        return self.evaluator

    async def evaluate_system(self,
                             test_questions: List[Dict[str, Any]],
                             similarity_method: str = "azure_embedding",
                             export_path: Optional[str] = None) -> EvaluationMetrics:
        """Evaluate the RAG system with test questions"""
        if self.evaluator is None:
            self.initialize_evaluator(similarity_method=similarity_method)

        # Create a temporary retrieval chain for evaluation
        eval_retriever = JapaneseHybridRetriever(
            vector_store=self.vector_store,
            connection_string=self.connection_string,
            engine=self.engine,
            config_params=self.config,
            text_processor=self.text_processor
        )

        eval_chain = create_retrieval_chain(
            self.llm, eval_retriever, self.jargon_manager, self.config
        )

        # Run evaluation
        results = await self.evaluator.evaluate_retrieval(
            test_questions=test_questions,
            retrieval_chain=eval_chain,
            llm=self.llm
        )

        # Export results if path provided
        if export_path:
            self.evaluator.export_results(results, export_path)

        return results.metrics

    def get_evaluation_results(self) -> Optional[EvaluationResults]:
        """Get the latest evaluation results"""
        if self.evaluator is None:
            return None
        return self.evaluator.last_results

    def export_evaluation_results(self, export_path: str) -> bool:
        """Export evaluation results to file"""
        if self.evaluator is None or self.evaluator.last_results is None:
            return False
        return self.evaluator.export_results(self.evaluator.last_results, export_path)