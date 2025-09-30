"""
Vector Store Adapter - Abstraction layer for vector databases
Supports both PGVector and ChromaDB
"""
from __future__ import annotations

import os
# Disable ChromaDB telemetry to avoid posthog errors
os.environ["ANONYMIZED_TELEMETRY"] = "false"

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import uuid

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


class VectorStoreAdapter(ABC):
    """Abstract base class for vector store implementations"""

    @abstractmethod
    def add_documents(self, documents: List[Document], ids: Optional[List[str]] = None) -> List[str]:
        """Add documents to the vector store"""
        pass

    @abstractmethod
    def similarity_search_with_score(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        """Search for similar documents with scores"""
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """Delete documents by IDs"""
        pass

    @abstractmethod
    def get_by_ids(self, ids: List[str]) -> List[Document]:
        """Retrieve documents by IDs"""
        pass


class PGVectorAdapter(VectorStoreAdapter):
    """Adapter for PGVector (PostgreSQL with vector extension)"""

    def __init__(
        self,
        connection_string: str,
        collection_name: str,
        embedding_function: Embeddings,
        use_jsonb: bool = True,
        distance_strategy: str = "cosine"
    ):
        from langchain_community.vectorstores import PGVector, DistanceStrategy

        # Map string to DistanceStrategy enum
        strategy_map = {
            "cosine": DistanceStrategy.COSINE,
            "euclidean": DistanceStrategy.EUCLIDEAN,  # Fixed: EUCLIDEAN_DISTANCE -> EUCLIDEAN
            "max_inner_product": DistanceStrategy.MAX_INNER_PRODUCT
        }

        self.vector_store = PGVector(
            connection_string=connection_string,
            collection_name=collection_name,
            embedding_function=embedding_function,
            use_jsonb=use_jsonb,
            distance_strategy=strategy_map.get(distance_strategy, DistanceStrategy.COSINE)
        )
        self.collection_name = collection_name

    def add_documents(self, documents: List[Document], ids: Optional[List[str]] = None) -> List[str]:
        return self.vector_store.add_documents(documents, ids=ids)

    def similarity_search_with_score(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        return self.vector_store.similarity_search_with_score(query, k=k)

    def delete(self, ids: List[str]) -> None:
        self.vector_store.delete(ids=ids)

    def get_by_ids(self, ids: List[str]) -> List[Document]:
        # PGVector doesn't have direct get_by_ids, so we implement it
        # This would need to be implemented based on the actual PGVector table structure
        # For now, return empty list as this method isn't used in the current codebase
        return []


class ChromaDBAdapter(VectorStoreAdapter):
    """Adapter for ChromaDB"""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: str = "documents",
        embedding_function: Embeddings = None,
        chroma_server_host: Optional[str] = None,
        chroma_server_port: Optional[int] = None,
        chroma_server_ssl_enabled: bool = False
    ):
        from langchain_community.vectorstores import Chroma

        # If server host is provided, use client-server mode
        if chroma_server_host:
            import chromadb
            from chromadb.config import Settings

            settings = Settings(
                chroma_server_host=chroma_server_host,
                chroma_server_port=chroma_server_port or 8000,
                chroma_server_ssl_enabled=chroma_server_ssl_enabled
            )
            client = chromadb.HttpClient(settings=settings)

            self.vector_store = Chroma(
                client=client,
                collection_name=collection_name,
                embedding_function=embedding_function
            )
        else:
            # Use persistent local storage
            if not persist_directory:
                persist_directory = "./chroma_db"

            Path(persist_directory).mkdir(parents=True, exist_ok=True)

            self.vector_store = Chroma(
                persist_directory=persist_directory,
                collection_name=collection_name,
                embedding_function=embedding_function
            )

        self.collection_name = collection_name
        self.embedding_function = embedding_function

    def add_documents(self, documents: List[Document], ids: Optional[List[str]] = None) -> List[str]:
        # ChromaDB requires string IDs
        if not ids:
            ids = [str(uuid.uuid4()) for _ in documents]
        return self.vector_store.add_documents(documents, ids=ids)

    def similarity_search_with_score(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        return self.vector_store.similarity_search_with_score(query, k=k)

    def delete(self, ids: List[str]) -> None:
        # ChromaDB's delete method works with IDs
        self.vector_store.delete(ids=ids)

    def get_by_ids(self, ids: List[str]) -> List[Document]:
        # ChromaDB has a get method
        results = self.vector_store.get(ids=ids)
        if not results or not results.get('documents'):
            return []

        documents = []
        for i, doc_content in enumerate(results['documents']):
            metadata = results.get('metadatas', [{}])[i] if results.get('metadatas') else {}
            documents.append(Document(page_content=doc_content, metadata=metadata))

        return documents


class VectorStoreFactory:
    """Factory class to create vector store adapters"""

    @staticmethod
    def create_vector_store(
        store_type: str,
        config: Any,
        embedding_function: Embeddings,
        connection_string: Optional[str] = None,
        **kwargs
    ) -> VectorStoreAdapter:
        """
        Create a vector store adapter based on the specified type

        Args:
            store_type: Type of vector store ("pgvector" or "chromadb")
            config: Configuration object
            embedding_function: Embedding function to use
            connection_string: Database connection string (for PGVector)
            **kwargs: Additional arguments for the specific vector store

        Returns:
            VectorStoreAdapter instance
        """
        store_type = store_type.lower()

        if store_type == "pgvector":
            if not connection_string:
                raise ValueError("Connection string is required for PGVector")

            return PGVectorAdapter(
                connection_string=connection_string,
                collection_name=config.collection_name,
                embedding_function=embedding_function,
                use_jsonb=kwargs.get("use_jsonb", True),
                distance_strategy=kwargs.get("distance_strategy", "cosine")
            )

        elif store_type == "chromadb":
            return ChromaDBAdapter(
                persist_directory=kwargs.get("persist_directory", config.chroma_persist_directory if hasattr(config, 'chroma_persist_directory') else "./chroma_db"),
                collection_name=config.collection_name,
                embedding_function=embedding_function,
                chroma_server_host=kwargs.get("chroma_server_host", getattr(config, 'chroma_server_host', None)),
                chroma_server_port=kwargs.get("chroma_server_port", getattr(config, 'chroma_server_port', None)),
                chroma_server_ssl_enabled=kwargs.get("chroma_server_ssl_enabled", getattr(config, 'chroma_server_ssl_enabled', False))
            )

        else:
            raise ValueError(f"Unsupported vector store type: {store_type}")


# For backward compatibility - expose the actual langchain PGVector class
# This allows existing code to work without modification
def get_native_vector_store(adapter: VectorStoreAdapter):
    """Get the native vector store instance from the adapter"""
    if hasattr(adapter, 'vector_store'):
        return adapter.vector_store
    return adapter