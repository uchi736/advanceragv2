import json
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from typing import List, Dict, Any, Optional, Tuple

from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.runnables import RunnableConfig

from .config import Config
from .text_processor import JapaneseTextProcessor

class JapaneseHybridRetriever(BaseRetriever):
    """
    A retriever that combines vector search and keyword search (FTS)
    with Reciprocal Rank Fusion (RRF) for hybrid search, optimized for
    Japanese language.
    """
    vector_store: PGVector
    connection_string: str
    config_params: Config
    text_processor: JapaneseTextProcessor
    search_type: str = "ハイブリッド検索"
    engine: Optional[Engine] = None
    
    def __init__(self, engine: Optional[Engine] = None, **kwargs):
        super().__init__(**kwargs)
        self.text_processor = JapaneseTextProcessor()
        # Reuse shared engine when provided to avoid recreating connections.
        object.__setattr__(self, "engine", engine or create_engine(self.connection_string))

    def _vector_search(self, q: str, config: Optional[RunnableConfig] = None) -> List[Tuple[Document, float]]:
        if not self.vector_store:
            return []
        try:
            # Check if it's an adapter or native vector store
            if hasattr(self.vector_store, 'similarity_search_with_score'):
                return self.vector_store.similarity_search_with_score(q, k=self.config_params.vector_search_k)
            elif hasattr(self.vector_store, 'vector_store'):
                # It's wrapped in an adapter
                return self.vector_store.vector_store.similarity_search_with_score(q, k=self.config_params.vector_search_k)
        except Exception as exc:
            print(f"[HybridRetriever] vector search error: {exc}")
            return []

    def _keyword_search(self, q: str, config: Optional[RunnableConfig] = None) -> List[Tuple[Document, float]]:
        """Performs keyword-based search with Japanese tokenization support."""
        res: List[Tuple[Document, float]] = []

        normalized_query = self.text_processor.normalize_text(q)
        is_japanese = self.text_processor.is_japanese(normalized_query)

        try:
            with self.engine.connect() as conn:
                if is_japanese and self.config_params.enable_japanese_search:
                    tokens = self.text_processor.tokenize(normalized_query)
                    if not tokens: return []
                    
                    conditions = []
                    params = {}
                    for i, token in enumerate(tokens[:5]):
                        if len(token) >= self.config_params.japanese_min_token_length:
                            conditions.append(f"(content LIKE :token{i} OR tokenized_content LIKE :token{i})")
                            params[f"token{i}"] = f"%{token}%"
                    
                    if not conditions: return []
                    
                    where_clause = " AND ".join(conditions)
                    sql = f"""
                        SELECT chunk_id, content, metadata, 
                               (LENGTH(content) - LENGTH(REPLACE(LOWER(content), LOWER(:original_query), ''))) / LENGTH(:original_query) AS score
                        FROM document_chunks 
                        WHERE {where_clause} AND collection_name = :collection_name
                        ORDER BY score DESC LIMIT :k;
                    """
                    params.update({"original_query": normalized_query, "collection_name": self.config_params.collection_name, "k": self.config_params.keyword_search_k})
                    db_result = conn.execute(text(sql), params)
                else:
                    sql = f"""
                        SELECT chunk_id, content, metadata, 
                               ts_rank(to_tsvector('{self.config_params.fts_language}', content), plainto_tsquery('{self.config_params.fts_language}', :q)) AS score 
                        FROM document_chunks 
                        WHERE to_tsvector('{self.config_params.fts_language}', content) @@ plainto_tsquery('{self.config_params.fts_language}', :q) 
                        AND collection_name = :collection_name 
                        ORDER BY score DESC LIMIT :k;
                    """
                    db_result = conn.execute(text(sql), {"q": normalized_query, "k": self.config_params.keyword_search_k, "collection_name": self.config_params.collection_name})
                
                for row in db_result:
                    md = row.metadata if isinstance(row.metadata, dict) else json.loads(row.metadata or "{}")
                    res.append((Document(page_content=row.content, metadata=md), float(row.score)))
                    
        except Exception as exc:
            print(f"[HybridRetriever] keyword search error: {exc}")
        return res

    @staticmethod
    def _rrf_hybrid(rank: int, k: int = 60) -> float:
        return 1.0 / (k + rank)

    def _reciprocal_rank_fusion_hybrid(self, vres: List[Tuple[Document, float]], kres: List[Tuple[Document, float]]) -> List[Document]:
        score_map: Dict[str, Dict[str, Any]] = {}
        _id = lambda d: d.metadata.get("chunk_id", d.page_content[:100])
        
        for r, (d, _) in enumerate(vres, 1):
            doc_id_val = _id(d)
            score_map.setdefault(doc_id_val, {"doc": d, "s": 0.0})["s"] += self._rrf_hybrid(r, self.config_params.rrf_k_for_fusion)
            
        for r, (d, _) in enumerate(kres, 1):
            doc_id_val = _id(d)
            score_map.setdefault(doc_id_val, {"doc": d, "s": 0.0})["s"] += self._rrf_hybrid(r, self.config_params.rrf_k_for_fusion)
            
        ranked = sorted(score_map.values(), key=lambda x: x["s"], reverse=True)
        return [x["doc"] for x in ranked[:self.config_params.final_k]]

    def _fetch_parent_chunks(self, child_docs: List[Document]) -> List[Document]:
        """Fetches parent chunks for a list of child documents."""
        parent_ids_to_fetch = []
        for doc in child_docs:
            if "parent_chunk_id" in doc.metadata:
                parent_ids_to_fetch.append(doc.metadata["parent_chunk_id"])
        
        if not parent_ids_to_fetch:
            return child_docs

        unique_parent_ids = list(set(parent_ids_to_fetch))
        sql = text("""
            SELECT content, metadata 
            FROM document_chunks 
            WHERE chunk_id = ANY(:parent_ids) AND collection_name = :collection_name
        """)

        try:
            with self.engine.connect() as conn:
                db_result = conn.execute(sql, {"parent_ids": unique_parent_ids, "collection_name": self.config_params.collection_name})
                parent_docs_map = {
                    json.loads(row.metadata).get("chunk_id"): Document(page_content=row.content, metadata=json.loads(row.metadata or "{}"))
                    for row in db_result
                }
        except Exception as e:
            print(f"Error fetching parent chunks: {e}")
            return child_docs # Fallback to child docs on error

        # Replace child docs with their parents, maintaining order and handling misses
        final_docs = []
        fetched_parent_ids = set()
        for doc in child_docs:
            parent_id = doc.metadata.get("parent_chunk_id")
            if parent_id and parent_id in parent_docs_map and parent_id not in fetched_parent_ids:
                final_docs.append(parent_docs_map[parent_id])
                fetched_parent_ids.add(parent_id)
            elif not parent_id: # It's a regular chunk, not a child
                final_docs.append(doc)
        
        return final_docs

    def _get_relevant_documents(self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None, **kwargs: Any) -> List[Document]:
        config = kwargs.get("config")
        
        if self.search_type == 'ベクトル検索':
            vres = self._vector_search(query, config=config)
            retrieved_docs = [doc for doc, score in vres]
        else: # Hybrid search
            vres = self._vector_search(query, config=config)
            kres = self._keyword_search(query, config=config)
            retrieved_docs = self._reciprocal_rank_fusion_hybrid(vres, kres)

        if self.config_params.enable_parent_child_chunking:
            return self._fetch_parent_chunks(retrieved_docs)
        
        return retrieved_docs[:self.config_params.final_k]

    async def _aget_relevant_documents(self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None, **kwargs: Any) -> List[Document]:
        # For simplicity, using the sync version. For production, implement async I/O.
        return self._get_relevant_documents(query, run_manager=run_manager, **kwargs)
