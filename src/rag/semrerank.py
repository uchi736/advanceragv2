"""
SemRe-Rank: Semantic Relatedness-based Re-ranking for Automatic Term Extraction

Implementation of the SemRe-Rank method from:
Zhang et al., 2017. "SemRe-Rank: Improving Automatic Term Extraction
By Incorporating Semantic Relatedness With Personalised PageRank"

主要機能:
1. 候補用語の埋め込みベクトル取得（キャッシュ機構付き）
2. 意味的関連性グラフ構築
3. シード選定（パーセンタイルベース）
4. Personalized PageRank実行
5. スコア改訂: final_score = base_score × (1 + avg_importance)
"""

import logging
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """
    候補用語の埋め込みベクトルをpgvectorでキャッシュ管理

    初回: Azure OpenAI Embeddingsで計算
    2回目以降: pgvectorキャッシュから高速取得
    """

    def __init__(
        self,
        embeddings,  # AzureOpenAIEmbeddings
        connection_string: str,
        cache_table: str = "term_embeddings"
    ):
        """
        Args:
            embeddings: AzureOpenAIEmbeddingsインスタンス
            connection_string: PostgreSQL接続文字列
            cache_table: キャッシュテーブル名
        """
        self.embeddings = embeddings
        self.connection_string = connection_string
        self.cache_table = cache_table
        self.engine: Engine = create_engine(connection_string)
        self._init_cache_table()

    def _init_cache_table(self):
        """キャッシュテーブルの初期化"""
        try:
            with self.engine.connect() as conn:
                # 埋め込み次元数を取得（動的に設定）
                # Azure OpenAI text-embedding-3-small: 1536次元
                conn.execute(text(f"""
                    CREATE TABLE IF NOT EXISTS {self.cache_table} (
                        term TEXT PRIMARY KEY,
                        embedding VECTOR(1536),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                conn.execute(text(f"""
                    CREATE INDEX IF NOT EXISTS idx_{self.cache_table}_term
                    ON {self.cache_table} (term)
                """))
                conn.commit()
                logger.info(f"Embedding cache table '{self.cache_table}' initialized")
        except Exception as e:
            logger.error(f"Failed to initialize cache table: {e}")
            raise

    def get_embeddings(self, terms: List[str]) -> Dict[str, np.ndarray]:
        """
        候補用語の埋め込みを取得（キャッシュ優先）

        Args:
            terms: 候補用語のリスト

        Returns:
            用語 -> 埋め込みベクトルのマッピング
        """
        embeddings_map = {}
        terms_to_compute = []

        # 1. キャッシュから取得
        if terms:
            try:
                with self.engine.connect() as conn:
                    placeholders = ', '.join([f':term_{i}' for i in range(len(terms))])
                    params = {f'term_{i}': term for i, term in enumerate(terms)}

                    result = conn.execute(
                        text(f"SELECT term, embedding FROM {self.cache_table} WHERE term IN ({placeholders})"),
                        params
                    )

                    for row in result:
                        # pgvector の vector型は文字列として返される: "[0.1, 0.2, ...]"
                        embedding_str = row.embedding
                        if isinstance(embedding_str, str):
                            # "[0.1, 0.2, ...]" -> [0.1, 0.2, ...]
                            embedding_list = eval(embedding_str)
                            embeddings_map[row.term] = np.array(embedding_list, dtype=np.float32)
                        else:
                            embeddings_map[row.term] = np.array(row.embedding, dtype=np.float32)

                logger.info(f"Loaded {len(embeddings_map)} embeddings from cache")
            except Exception as e:
                logger.warning(f"Failed to load embeddings from cache: {e}")

        # 2. 未キャッシュの用語を特定
        terms_to_compute = [term for term in terms if term not in embeddings_map]

        # 3. 未キャッシュの用語の埋め込みを計算
        if terms_to_compute:
            logger.info(f"Computing embeddings for {len(terms_to_compute)} terms")

            try:
                # Azure OpenAI Embeddingsで計算（バッチ処理）
                computed_embeddings = self.embeddings.embed_documents(terms_to_compute)

                # 結果を保存
                for term, embedding in zip(terms_to_compute, computed_embeddings):
                    embedding_array = np.array(embedding, dtype=np.float32)
                    embeddings_map[term] = embedding_array

                    # キャッシュに保存
                    self._save_to_cache(term, embedding_array)

                logger.info(f"Computed and cached {len(terms_to_compute)} embeddings")
            except Exception as e:
                logger.error(f"Failed to compute embeddings: {e}")
                raise

        return embeddings_map

    def _save_to_cache(self, term: str, embedding: np.ndarray):
        """埋め込みをキャッシュに保存"""
        try:
            with self.engine.connect() as conn:
                # numpy配列をリストに変換してPostgreSQLに保存
                embedding_list = embedding.tolist()
                embedding_str = str(embedding_list)

                conn.execute(
                    text(f"""
                        INSERT INTO {self.cache_table} (term, embedding)
                        VALUES (:term, CAST(:embedding AS vector))
                        ON CONFLICT (term) DO UPDATE
                        SET embedding = EXCLUDED.embedding,
                            created_at = CURRENT_TIMESTAMP
                    """),
                    {"term": term, "embedding": embedding_str}
                )
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to save embedding to cache for '{term}': {e}")


def build_semantic_graph(
    embeddings: Dict[str, np.ndarray],
    relmin: float = 0.5,
    reltop: float = 0.15
) -> nx.Graph:
    """
    意味的関連性グラフの構築

    Args:
        embeddings: 用語 -> 埋め込みベクトルのマッピング
        relmin: 最小類似度閾値（0.5）
        reltop: 各ノードの上位関連語の割合（0.15 = 15%）

    Returns:
        networkx.Graph
    """
    terms = list(embeddings.keys())
    n_terms = len(terms)

    if n_terms < 2:
        logger.warning("Not enough terms to build graph")
        return nx.Graph()

    # 埋め込み行列を作成
    embedding_matrix = np.array([embeddings[term] for term in terms])

    # コサイン類似度計算（全ペア）
    similarity_matrix = cosine_similarity(embedding_matrix)

    # グラフ構築
    graph = nx.Graph()
    graph.add_nodes_from(terms)

    # 各ノードについて、上位reltop%の関連語とエッジを作成
    top_k = max(1, int(n_terms * reltop))

    for i, term1 in enumerate(terms):
        # 自分自身を除く類似度を取得
        similarities = [(j, similarity_matrix[i][j]) for j in range(n_terms) if i != j]

        # relmin以上の類似度のみ
        similarities = [(j, sim) for j, sim in similarities if sim >= relmin]

        # 類似度降順でソート
        similarities.sort(key=lambda x: x[1], reverse=True)

        # 上位top_k件とエッジを作成
        for j, sim in similarities[:top_k]:
            term2 = terms[j]
            if not graph.has_edge(term1, term2):
                graph.add_edge(term1, term2, weight=sim)

    logger.info(f"Built semantic graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    return graph


def select_seeds_by_percentile(
    seed_scores: Dict[str, float],
    percentile: float = 15.0
) -> List[str]:
    """
    上位N%でシード選定

    Args:
        seed_scores: シード選定用スコア
        percentile: 上位何%を選定するか（デフォルト15%）

    Returns:
        シード用語のリスト
    """
    if not seed_scores:
        return []

    # 上位N%の件数を計算
    n_seeds = max(1, int(len(seed_scores) * percentile / 100))

    # スコア降順でソート
    sorted_terms = sorted(seed_scores.items(), key=lambda x: x[1], reverse=True)

    # 上位n_seeds件を選択
    seeds = [term for term, score in sorted_terms[:n_seeds]]

    logger.info(f"Selected {len(seeds)} seeds (top {percentile}%) from {len(seed_scores)} candidates")

    return seeds


def personalized_pagerank(
    graph: nx.Graph,
    seeds: List[str],
    alpha: float = 0.85
) -> Dict[str, float]:
    """
    Personalized PageRankの実行

    Args:
        graph: 意味的関連性グラフ
        seeds: シード用語のリスト
        alpha: ダンピング係数（デフォルト0.85）

    Returns:
        用語 -> 重要度スコアのマッピング
    """
    if not graph.nodes() or not seeds:
        logger.warning("Empty graph or no seeds for PageRank")
        return {}

    # personalizationベクトル作成（シードに重み1.0、他は0.0）
    personalization = {node: 1.0 if node in seeds else 0.0 for node in graph.nodes()}

    # シードが存在しない場合は均等分布
    if sum(personalization.values()) == 0:
        personalization = {node: 1.0 for node in graph.nodes()}

    try:
        # networkxのpagerank関数を使用
        importance_scores = nx.pagerank(
            graph,
            alpha=alpha,
            personalization=personalization,
            max_iter=100,
            tol=1e-06
        )

        logger.info(f"Computed Personalized PageRank for {len(importance_scores)} nodes")

        return importance_scores
    except Exception as e:
        logger.error(f"PageRank computation failed: {e}")
        return {}


def revise_scores(
    base_scores: Dict[str, float],
    importance_scores: Dict[str, float]
) -> Dict[str, float]:
    """
    スコア改訂: final_score = base_score × (1 + avg_importance)

    Args:
        base_scores: TF-IDF + C-valueの基底スコア
        importance_scores: PageRankによる重要度スコア

    Returns:
        改訂後のスコア
    """
    if not importance_scores:
        logger.warning("No importance scores available, returning base scores")
        return base_scores

    revised_scores = {}

    for term, base_score in base_scores.items():
        importance = importance_scores.get(term, 0.0)

        # 式: final_score = base_score × (1 + importance)
        revised_score = base_score * (1 + importance)
        revised_scores[term] = revised_score

    logger.info(f"Revised scores for {len(revised_scores)} terms")

    return revised_scores


class SemReRank:
    """
    SemRe-Rankの完全実装

    低頻度だが意味的に重要な専門用語を救い上げる
    """

    def __init__(
        self,
        embeddings,  # AzureOpenAIEmbeddings
        connection_string: str,
        relmin: float = 0.5,
        reltop: float = 0.15,
        alpha: float = 0.85,
        seed_percentile: float = 15.0
    ):
        """
        Args:
            embeddings: AzureOpenAIEmbeddingsインスタンス
            connection_string: PostgreSQL接続文字列
            relmin: 最小類似度閾値（デフォルト0.5）
            reltop: 上位関連語の割合（デフォルト0.15 = 15%）
            alpha: PageRankダンピング係数（デフォルト0.85）
            seed_percentile: シード選定の上位何%（デフォルト15.0）
        """
        self.embedding_cache = EmbeddingCache(embeddings, connection_string)
        self.relmin = relmin
        self.reltop = reltop
        self.alpha = alpha
        self.seed_percentile = seed_percentile

        logger.info(
            f"SemReRank initialized: relmin={relmin}, reltop={reltop}, "
            f"alpha={alpha}, seed_percentile={seed_percentile}%"
        )

    def enhance_scores(
        self,
        candidates: List[str],
        base_scores: Dict[str, float],
        seed_scores: Dict[str, float]
    ) -> Dict[str, float]:
        """
        SemRe-Rankによるスコア強化

        Args:
            candidates: 候補用語のリスト
            base_scores: TF-IDF + C-valueの最終スコア
            seed_scores: シード選定用スコア（Stage A）

        Returns:
            改訂後のスコア
        """
        if not candidates or not base_scores:
            logger.warning("No candidates or base scores provided")
            return base_scores

        logger.info(f"Enhancing scores for {len(candidates)} candidates with SemRe-Rank")

        # 1. 埋め込み取得（キャッシュ活用）
        embeddings = self.embedding_cache.get_embeddings(candidates)

        if len(embeddings) < 2:
            logger.warning("Not enough embeddings, returning base scores")
            return base_scores

        # 2. 意味的関連性グラフ構築
        graph = build_semantic_graph(embeddings, self.relmin, self.reltop)

        if graph.number_of_nodes() < 2:
            logger.warning("Graph too small, returning base scores")
            return base_scores

        # 3. シード選定（上位N%）
        seeds = select_seeds_by_percentile(seed_scores, self.seed_percentile)

        if not seeds:
            logger.warning("No seeds selected, returning base scores")
            return base_scores

        # 4. Personalized PageRank実行
        importance_scores = personalized_pagerank(graph, seeds, self.alpha)

        if not importance_scores:
            logger.warning("PageRank failed, returning base scores")
            return base_scores

        # 5. スコア改訂
        enhanced_scores = revise_scores(base_scores, importance_scores)

        logger.info(
            f"SemRe-Rank completed: {len(seeds)} seeds, "
            f"{len(enhanced_scores)} scores enhanced"
        )

        return enhanced_scores


__all__ = [
    "EmbeddingCache",
    "build_semantic_graph",
    "select_seeds_by_percentile",
    "personalized_pagerank",
    "revise_scores",
    "SemReRank"
]