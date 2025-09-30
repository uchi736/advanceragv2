"""
ChromaDB Embedding Function Adapter
Wraps LangChain embeddings to work with ChromaDB's new interface
"""

from typing import List, Union
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings


class LangChainEmbeddingAdapter(EmbeddingFunction):
    """
    Adapter to make LangChain embeddings work with ChromaDB
    Implements ChromaDB's EmbeddingFunction interface
    """

    def __init__(self, langchain_embeddings):
        """
        Initialize the adapter with LangChain embeddings

        Args:
            langchain_embeddings: LangChain embedding function (e.g., AzureOpenAIEmbeddings)
        """
        self.langchain_embeddings = langchain_embeddings

    def __call__(self, input: Documents) -> Embeddings:
        """
        Generate embeddings for the given documents

        Args:
            input: List of documents to embed

        Returns:
            List of embeddings
        """
        # Handle both single string and list of strings
        if isinstance(input, str):
            input = [input]

        # Use LangChain's embed_documents method
        embeddings = self.langchain_embeddings.embed_documents(input)
        return embeddings


def create_chromadb_embedding_function(langchain_embeddings):
    """
    Factory function to create a ChromaDB-compatible embedding function

    Args:
        langchain_embeddings: LangChain embedding instance

    Returns:
        ChromaDB-compatible embedding function
    """
    return LangChainEmbeddingAdapter(langchain_embeddings)