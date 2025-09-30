"""
Hybrid Retriever that works with both PGVector and ChromaDB
Combines vector search with BM25 keyword search
"""

from typing import List, Dict, Any, Optional, Tuple
from pydantic import Field
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_community.retrievers import BM25Retriever
from langchain_core.runnables import RunnableConfig


class UniversalHybridRetriever(BaseRetriever):
    """
    Universal hybrid retriever that works with any vector store
    Combines vector search with BM25 keyword search using RRF
    """
    # Define fields for Pydantic model
    vector_store: Any
    config_params: Any = None
    text_processor: Any = None
    search_type: str = "hybrid"
    bm25_retriever: Optional[BM25Retriever] = None

    def __init__(
        self,
        vector_store,
        documents: List[Document] = None,
        config_params: Any = None,
        text_processor: Any = None,
        search_type: str = "hybrid",
        **kwargs
    ):
        """
        Initialize the universal hybrid retriever

        Args:
            vector_store: Vector store adapter (PGVector or ChromaDB)
            documents: Documents for BM25 index (will be fetched if not provided)
            config_params: Configuration parameters
            text_processor: Text processor for tokenization
            search_type: "vector", "keyword", or "hybrid"
        """
        super().__init__(
            vector_store=vector_store,
            config_params=config_params,
            text_processor=text_processor,
            search_type=search_type,
            **kwargs
        )

        # Initialize BM25 retriever if documents are provided or can be fetched
        if documents or hasattr(vector_store, 'get_all_documents'):
            self._initialize_bm25(documents)

    def _initialize_bm25(self, documents: List[Document] = None):
        """Initialize BM25 retriever with documents"""
        try:
            # If documents not provided, try to fetch from vector store
            if documents is None and hasattr(self.vector_store, 'get_all_documents'):
                documents = self.vector_store.get_all_documents()

            if documents:
                # Create preprocessing function using text processor if available
                if self.text_processor and hasattr(self.text_processor, 'tokenize'):
                    preprocess_func = lambda text: self.text_processor.tokenize(text)
                else:
                    # Use default preprocessing (splits on whitespace)
                    preprocess_func = lambda text: text.lower().split()

                # Initialize BM25 retriever
                self.bm25_retriever = BM25Retriever.from_documents(
                    documents,
                    preprocess_func=preprocess_func,
                    k=self.config_params.keyword_search_k if self.config_params else 15
                )
                print(f"BM25 retriever initialized with {len(documents)} documents")
        except Exception as e:
            print(f"Failed to initialize BM25 retriever: {e}")
            self.bm25_retriever = None

    def update_documents(self, documents: List[Document]):
        """Update the BM25 index with new documents"""
        self._initialize_bm25(documents)

    def _vector_search(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """Perform vector search"""
        if not self.vector_store:
            return []

        k = k or (self.config_params.vector_search_k if self.config_params else 15)

        try:
            # Check if it's an adapter or native vector store
            if hasattr(self.vector_store, 'similarity_search_with_score'):
                return self.vector_store.similarity_search_with_score(query, k=k)
            elif hasattr(self.vector_store, 'vector_store'):
                # It's wrapped in an adapter
                return self.vector_store.vector_store.similarity_search_with_score(query, k=k)
        except Exception as exc:
            print(f"Vector search error: {exc}")
            return []

    def _keyword_search(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """Perform BM25 keyword search"""
        if not self.bm25_retriever:
            return []

        k = k or (self.config_params.keyword_search_k if self.config_params else 15)

        try:
            # BM25Retriever returns documents without scores
            docs = self.bm25_retriever.get_relevant_documents(query)[:k]

            # Add pseudo-scores based on ranking (higher rank = higher score)
            scored_docs = []
            for i, doc in enumerate(docs):
                # Score decreases with rank
                score = 1.0 / (i + 1)
                scored_docs.append((doc, score))

            return scored_docs
        except Exception as exc:
            print(f"Keyword search error: {exc}")
            return []

    @staticmethod
    def _rrf_hybrid(rank: int, k: int = 60) -> float:
        """Calculate RRF score for a given rank"""
        return 1.0 / (k + rank)

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Tuple[Document, float]],
        keyword_results: List[Tuple[Document, float]],
        k: int = None
    ) -> List[Document]:
        """Apply Reciprocal Rank Fusion to combine results"""
        k = k or (self.config_params.final_k if self.config_params else 15)
        rrf_k = self.config_params.rrf_k_for_fusion if self.config_params else 60

        score_map: Dict[str, Dict[str, Any]] = {}

        # Helper to get document ID
        def get_doc_id(doc: Document) -> str:
            return doc.metadata.get("chunk_id", doc.page_content[:100])

        # Process vector search results
        for rank, (doc, _) in enumerate(vector_results, 1):
            doc_id = get_doc_id(doc)
            score_map.setdefault(doc_id, {"doc": doc, "score": 0.0})
            score_map[doc_id]["score"] += self._rrf_hybrid(rank, rrf_k)

        # Process keyword search results
        for rank, (doc, _) in enumerate(keyword_results, 1):
            doc_id = get_doc_id(doc)
            score_map.setdefault(doc_id, {"doc": doc, "score": 0.0})
            score_map[doc_id]["score"] += self._rrf_hybrid(rank, rrf_k)

        # Sort by combined score
        ranked = sorted(score_map.values(), key=lambda x: x["score"], reverse=True)
        return [x["doc"] for x in ranked[:k]]

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
        **kwargs: Any
    ) -> List[Document]:
        """Get relevant documents based on search type"""
        config = kwargs.get("config")

        if self.search_type == "vector":
            # Vector search only
            vector_results = self._vector_search(query)
            return [doc for doc, _ in vector_results]

        elif self.search_type == "keyword":
            # Keyword search only
            keyword_results = self._keyword_search(query)
            return [doc for doc, _ in keyword_results]

        else:  # hybrid
            # Combine vector and keyword search
            vector_results = self._vector_search(query)
            keyword_results = self._keyword_search(query)
            return self._reciprocal_rank_fusion(vector_results, keyword_results)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
        **kwargs: Any
    ) -> List[Document]:
        """Async version - uses sync implementation for now"""
        return self._get_relevant_documents(query, run_manager=run_manager, **kwargs)


class ChromaDBWithBM25Adapter:
    """
    Adapter that adds BM25 capability to ChromaDB adapter
    """

    def __init__(self, chromadb_adapter, documents: List[Document] = None):
        """
        Initialize ChromaDB with BM25 support

        Args:
            chromadb_adapter: ChromaDB vector store adapter
            documents: Initial documents for BM25 index
        """
        self.chromadb_adapter = chromadb_adapter
        self.bm25_retriever = None
        self.documents = documents or []

        if documents:
            self._update_bm25_index(documents)

    def _update_bm25_index(self, documents: List[Document]):
        """Update BM25 index with documents"""
        try:
            self.bm25_retriever = BM25Retriever.from_documents(
                documents,
                k=15  # Default k value
            )
            self.documents = documents
        except Exception as e:
            print(f"Failed to create BM25 index: {e}")

    def add_documents(self, documents: List[Document], ids: Optional[List[str]] = None) -> List[str]:
        """Add documents to both ChromaDB and BM25 index"""
        # Add to ChromaDB
        result_ids = self.chromadb_adapter.add_documents(documents, ids)

        # Update BM25 index
        self.documents.extend(documents)
        self._update_bm25_index(self.documents)

        return result_ids

    def similarity_search_with_score(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        """Vector search using ChromaDB"""
        return self.chromadb_adapter.similarity_search_with_score(query, k)

    def keyword_search(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        """Keyword search using BM25"""
        if not self.bm25_retriever:
            return []

        docs = self.bm25_retriever.get_relevant_documents(query)[:k]
        # Add pseudo-scores
        return [(doc, 1.0 / (i + 1)) for i, doc in enumerate(docs)]

    def delete(self, ids: List[str]) -> None:
        """Delete documents from ChromaDB (BM25 index needs full rebuild)"""
        self.chromadb_adapter.delete(ids)
        # Note: BM25 index would need to be rebuilt after deletion
        # This requires fetching all remaining documents from ChromaDB

    def get_by_ids(self, ids: List[str]) -> List[Document]:
        """Get documents by IDs from ChromaDB"""
        return self.chromadb_adapter.get_by_ids(ids)