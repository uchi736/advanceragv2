"""
ChromaDB Extensions for keyword search and jargon dictionary
Provides additional functionality for ChromaDB to match PGVector features
"""

import json
import re
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import uuid

from langchain_core.documents import Document
import chromadb
from chromadb.config import Settings
from .chromadb_embedding_adapter import create_chromadb_embedding_function


class ChromaDBKeywordSearchMixin:
    """Mixin to add keyword search capabilities to ChromaDB adapter"""

    def keyword_search(self, query: str, k: int = 10, collection_name: str = None) -> List[Tuple[Document, float]]:
        """
        Perform keyword search on ChromaDB collection

        Args:
            query: Search query
            k: Number of results to return
            collection_name: Collection to search (defaults to main collection)

        Returns:
            List of (Document, score) tuples
        """
        if not hasattr(self, 'vector_store') or not self.vector_store:
            return []

        # Get the collection
        collection = self.vector_store._collection if hasattr(self.vector_store, '_collection') else None
        if not collection:
            return []

        # Get all documents from collection (for small-medium datasets)
        # For large datasets, consider implementing pagination
        try:
            results = collection.get(include=["documents", "metadatas"])

            if not results or not results['documents']:
                return []

            # Normalize query for matching
            query_lower = query.lower()
            query_tokens = set(query_lower.split())

            scored_docs = []
            for i, (doc_text, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
                if not doc_text:
                    continue

                # Calculate keyword match score
                doc_lower = doc_text.lower()

                # Score based on exact phrase match
                phrase_score = doc_lower.count(query_lower) * 10

                # Score based on individual token matches
                doc_tokens = set(doc_lower.split())
                token_overlap = len(query_tokens & doc_tokens)
                token_score = token_overlap / len(query_tokens) if query_tokens else 0

                # Combined score
                score = phrase_score + token_score

                if score > 0:
                    doc = Document(page_content=doc_text, metadata=metadata or {})
                    scored_docs.append((doc, float(score)))

            # Sort by score and return top k
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            return scored_docs[:k]

        except Exception as e:
            print(f"Error in keyword search: {e}")
            return []


class ChromaDBJargonManager:
    """Jargon Dictionary Manager using ChromaDB as storage backend"""

    def __init__(self,
                 persist_directory: str = "./chroma_jargon_db",
                 collection_name: str = "jargon_dictionary",
                 embedding_function=None):
        """
        Initialize ChromaDB-based Jargon Manager

        Args:
            persist_directory: Directory for ChromaDB persistence
            collection_name: Name of the collection for jargon terms
            embedding_function: Embedding function for semantic search
        """
        self.collection_name = collection_name
        self.original_embedding_function = embedding_function

        # Wrap the embedding function for ChromaDB compatibility
        if embedding_function:
            self.embedding_function = create_chromadb_embedding_function(embedding_function)
        else:
            self.embedding_function = None

        # Initialize ChromaDB client
        # Disable telemetry to avoid posthog errors
        import os
        os.environ["ANONYMIZED_TELEMETRY"] = "false"
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Get or create collection for jargon terms
        try:
            self.collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "Jargon dictionary for technical terms"}
            )

    def add_term(self, term: str, definition: str, domain: str = None,
                 aliases: List[str] = None, related_terms: List[str] = None,
                 embeddings: List[float] = None) -> bool:
        """Add a new term to the jargon dictionary"""
        try:
            # Create unique ID for the term
            term_id = f"jargon_{term.lower().replace(' ', '_')}_{uuid.uuid4().hex[:8]}"

            # Prepare metadata
            metadata = {
                "term": term,
                "definition": definition,
                "domain": domain or "",
                "aliases": json.dumps(aliases or []),
                "related_terms": json.dumps(related_terms or []),
                "created_at": datetime.now().isoformat()
            }

            # Combine term and definition for embedding
            content = f"{term}\n{definition}"
            if aliases:
                content += f"\nAliases: {', '.join(aliases)}"

            # Add to collection
            self.collection.add(
                documents=[content],
                metadatas=[metadata],
                ids=[term_id],
                embeddings=[embeddings] if embeddings else None
            )

            return True
        except Exception as e:
            print(f"Error adding term to ChromaDB jargon dictionary: {e}")
            return False

    def get_term(self, term: str) -> Optional[Dict[str, Any]]:
        """Get a specific term from the dictionary"""
        try:
            # Search by term in metadata
            results = self.collection.get(
                where={"term": term},
                include=["metadatas", "documents"]
            )

            if results and results['metadatas']:
                metadata = results['metadatas'][0]
                return {
                    "term": metadata.get("term"),
                    "definition": metadata.get("definition"),
                    "domain": metadata.get("domain"),
                    "aliases": json.loads(metadata.get("aliases", "[]")),
                    "related_terms": json.loads(metadata.get("related_terms", "[]")),
                    "created_at": metadata.get("created_at")
                }
            return None
        except Exception as e:
            print(f"Error getting term from ChromaDB: {e}")
            return None

    def search_terms(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for terms semantically similar to the query"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                include=["metadatas", "distances"]
            )

            terms = []
            if results and results['metadatas']:
                for metadata_list in results['metadatas']:
                    for metadata in metadata_list:
                        terms.append({
                            "term": metadata.get("term"),
                            "definition": metadata.get("definition"),
                            "domain": metadata.get("domain"),
                            "aliases": json.loads(metadata.get("aliases", "[]")),
                            "related_terms": json.loads(metadata.get("related_terms", "[]"))
                        })
            return terms
        except Exception as e:
            print(f"Error searching terms in ChromaDB: {e}")
            return []

    def get_all_terms(self) -> List[Dict[str, Any]]:
        """Get all terms from the dictionary"""
        try:
            results = self.collection.get(
                include=["metadatas"]
            )

            terms = []
            if results and results['metadatas']:
                for metadata in results['metadatas']:
                    terms.append({
                        "term": metadata.get("term"),
                        "definition": metadata.get("definition"),
                        "domain": metadata.get("domain"),
                        "aliases": json.loads(metadata.get("aliases", "[]")),
                        "related_terms": json.loads(metadata.get("related_terms", "[]")),
                        "created_at": metadata.get("created_at")
                    })

            # Sort by term name
            terms.sort(key=lambda x: x.get("term", "").lower())
            return terms
        except Exception as e:
            print(f"Error getting all terms from ChromaDB: {e}")
            return []

    def delete_terms(self, terms: List[str]) -> Tuple[int, int]:
        """Delete multiple terms from the dictionary"""
        deleted = 0
        failed = 0

        for term in terms:
            try:
                # First try exact match (ids are always included by default)
                results = self.collection.get(
                    where={"term": term}
                )

                ids_to_delete = []
                if results and results.get('ids'):
                    ids_to_delete.extend(results['ids'])

                # If not found, try case-insensitive search through all terms
                if not ids_to_delete:
                    all_results = self.collection.get(
                        include=["metadatas"]
                    )

                    if all_results and all_results.get('metadatas') and all_results.get('ids'):
                        for i, metadata in enumerate(all_results['metadatas']):
                            if metadata.get("term", "").lower() == term.lower():
                                ids_to_delete.append(all_results['ids'][i])

                if ids_to_delete:
                    self.collection.delete(ids=ids_to_delete)
                    deleted += 1
                    print(f"Deleted term: {term} (IDs: {ids_to_delete})")
                else:
                    failed += 1
                    print(f"Term not found: {term}")
            except Exception as e:
                print(f"Error deleting term {term}: {e}")
                failed += 1

        return deleted, failed

    def find_related_terms(self, query: str, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Find terms related to the query using semantic similarity"""
        try:
            # Search for semantically similar terms
            results = self.collection.query(
                query_texts=[query],
                n_results=10,
                include=["metadatas", "distances"]
            )

            related = []
            if results and results['metadatas'] and results['distances']:
                for metadata_list, distance_list in zip(results['metadatas'], results['distances']):
                    for metadata, distance in zip(metadata_list, distance_list):
                        # ChromaDB distance is lower for more similar items
                        # Convert to similarity score (1 - distance for cosine)
                        similarity = 1 - distance if distance < 1 else 0

                        if similarity >= threshold:
                            term_info = {
                                "term": metadata.get("term"),
                                "definition": metadata.get("definition"),
                                "similarity": similarity
                            }

                            # Check if query matches any aliases
                            aliases = json.loads(metadata.get("aliases", "[]"))
                            if any(query.lower() in alias.lower() for alias in aliases):
                                term_info["matched_alias"] = True

                            related.append(term_info)

            return related
        except Exception as e:
            print(f"Error finding related terms: {e}")
            return []

    def lookup_terms(self, terms: List[str]) -> Dict[str, Dict[str, Any]]:
        """Look up multiple terms from the dictionary

        Args:
            terms: List of terms to look up

        Returns:
            Dictionary mapping term names to their information
        """
        if not terms:
            return {}

        results = {}
        try:
            # Search for each term
            for term in terms:
                # First try exact match by term metadata
                exact_results = self.collection.get(
                    where={"term": term},
                    include=["metadatas"]
                )

                if exact_results and exact_results['metadatas']:
                    metadata = exact_results['metadatas'][0]
                    results[term] = {
                        "definition": metadata.get("definition"),
                        "domain": metadata.get("domain"),
                        "aliases": json.loads(metadata.get("aliases", "[]")),
                        "related_terms": json.loads(metadata.get("related_terms", "[]")),
                        "confidence_score": 1.0
                    }
                else:
                    # Try case-insensitive search or semantic search
                    all_terms = self.get_all_terms()
                    for stored_term in all_terms:
                        if stored_term['term'].lower() == term.lower():
                            results[term] = {
                                "definition": stored_term["definition"],
                                "domain": stored_term["domain"],
                                "aliases": stored_term["aliases"],
                                "related_terms": stored_term["related_terms"],
                                "confidence_score": 1.0
                            }
                            break
                        # Check aliases
                        elif term.lower() in [alias.lower() for alias in stored_term.get("aliases", [])]:
                            results[term] = {
                                "definition": stored_term["definition"],
                                "domain": stored_term["domain"],
                                "aliases": stored_term["aliases"],
                                "related_terms": stored_term["related_terms"],
                                "confidence_score": 0.9
                            }
                            break

        except Exception as e:
            print(f"Error looking up terms: {e}")

        return results


class ChromaDBHybridRetriever:
    """Hybrid retriever for ChromaDB combining vector and keyword search"""

    def __init__(self, vector_store, text_processor=None, config=None):
        """
        Initialize hybrid retriever for ChromaDB

        Args:
            vector_store: ChromaDB vector store adapter
            text_processor: Text processor for tokenization
            config: Configuration object
        """
        self.vector_store = vector_store
        self.text_processor = text_processor
        self.config = config

        # Add keyword search capability
        if not hasattr(vector_store, 'keyword_search'):
            # Monkey-patch the keyword search method
            import types
            mixin = ChromaDBKeywordSearchMixin()
            vector_store.keyword_search = types.MethodType(mixin.keyword_search, vector_store)

    def hybrid_search(self, query: str, k: int = 10) -> List[Document]:
        """
        Perform hybrid search combining vector and keyword results

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of documents ranked by RRF
        """
        # Perform vector search
        vector_results = []
        if hasattr(self.vector_store, 'similarity_search_with_score'):
            vector_results = self.vector_store.similarity_search_with_score(query, k=k)

        # Perform keyword search
        keyword_results = []
        if hasattr(self.vector_store, 'keyword_search'):
            keyword_results = self.vector_store.keyword_search(query, k=k)

        # Apply Reciprocal Rank Fusion
        return self._reciprocal_rank_fusion(vector_results, keyword_results, k)

    def _reciprocal_rank_fusion(self, vector_results: List[Tuple[Document, float]],
                                keyword_results: List[Tuple[Document, float]],
                                k: int, rrf_k: int = 60) -> List[Document]:
        """Apply Reciprocal Rank Fusion to combine results"""
        score_map = {}

        # Process vector search results
        for rank, (doc, _) in enumerate(vector_results, 1):
            doc_id = doc.metadata.get("chunk_id", doc.page_content[:100])
            score_map.setdefault(doc_id, {"doc": doc, "score": 0})["score"] += 1 / (rrf_k + rank)

        # Process keyword search results
        for rank, (doc, _) in enumerate(keyword_results, 1):
            doc_id = doc.metadata.get("chunk_id", doc.page_content[:100])
            score_map.setdefault(doc_id, {"doc": doc, "score": 0})["score"] += 1 / (rrf_k + rank)

        # Sort by combined score
        ranked = sorted(score_map.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in ranked[:k]]