#!/usr/bin/env python3
"""term_clustering_analyzer.py
専門用語のクラスタリング分析ツール
------------------------------------------
HDBSCANを使用して専門用語を自動的にカテゴリ分類
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
from datetime import datetime

import hdbscan
import umap
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI
from sqlalchemy import create_engine, text

# Project imports
sys.path.append(str(Path(__file__).resolve().parents[1]))
from rag.config import Config
from rag.prompts import get_synonym_judgment_single_definition_prompt, get_synonym_judgment_with_definitions_prompt

# ── ENV ───────────────────────────────────────────
load_dotenv()
cfg = Config()

PG_URL = f"postgresql+psycopg://{cfg.db_user}:{cfg.db_password}@{cfg.db_host}:{cfg.db_port}/{cfg.db_name}"
JARGON_TABLE_NAME = cfg.jargon_table_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Azure OpenAI Embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=cfg.azure_openai_embedding_deployment_name,
    api_version=cfg.azure_openai_api_version,
    azure_endpoint=cfg.azure_openai_endpoint,
    api_key=cfg.azure_openai_api_key
)

# Azure OpenAI LLM for naming (4.1-mini使用)
llm = AzureChatOpenAI(
    azure_deployment=cfg.azure_openai_chat_mini_deployment_name,
    api_version=cfg.azure_openai_api_version,
    azure_endpoint=cfg.azure_openai_endpoint,
    api_key=cfg.azure_openai_api_key,
    temperature=0.1
)

# ── Term Clustering Analyzer ──────────────────────
class TermClusteringAnalyzer:
    """専門用語のクラスタリング分析クラス"""

    def __init__(self, connection_string: str, min_terms: int = 3, jargon_table_name: str = None, embeddings=None):
        """
        Args:
            connection_string: PostgreSQL接続文字列
            min_terms: クラスタリングを実行する最小用語数
            jargon_table_name: 専門用語テーブル名（デフォルトはJARGON_TABLE_NAME）
            embeddings: AzureOpenAIEmbeddings instance（デフォルトはグローバル変数）
        """
        self.connection_string = connection_string
        self.min_terms = min_terms
        self.jargon_table_name = jargon_table_name or JARGON_TABLE_NAME
        self.embeddings = embeddings or globals()['embeddings']
        self.terms_data = []
        self.embeddings_matrix = None
        self.clusters = None
        self.clusterer = None
        
    def load_terms_from_db(self) -> List[Dict[str, Any]]:
        """データベースから専門用語を読み込み"""
        engine = create_engine(self.connection_string)
        query = text(f"""
            SELECT term, definition, domain, aliases, related_terms
            FROM {self.jargon_table_name}
            ORDER BY term
        """)

        terms = []
        try:
            with engine.connect() as conn:
                result = conn.execute(query)
                for row in result:
                    terms.append({
                        'term': row.term,
                        'definition': row.definition,
                        'domain': row.domain,
                        'aliases': row.aliases or [],
                        'related_terms': row.related_terms or [],
                        'text_for_embedding': f"{row.term}: {row.definition}"
                    })
            logger.info(f"Loaded {len(terms)} terms from database")
        except Exception as e:
            logger.error(f"Error loading terms: {e}")
            
        self.terms_data = terms
        return terms
    
    def generate_embeddings(self) -> np.ndarray:
        """用語+定義のテキストからエンベディングを生成"""
        if not self.terms_data:
            logger.warning("No terms loaded")
            return np.array([])
        
        texts = [t['text_for_embedding'] for t in self.terms_data]
        
        logger.info(f"Generating embeddings for {len(texts)} terms...")
        try:
            embeddings_list = embeddings.embed_documents(texts)
            self.embeddings_matrix = np.array(embeddings_list)
            logger.info(f"Generated embeddings with shape: {self.embeddings_matrix.shape}")
            return self.embeddings_matrix
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return np.array([])
    
    def perform_clustering(self, min_cluster_size: int = 3) -> Dict[str, Any]:
        """HDBSCANによるクラスタリング実行"""
        if self.embeddings_matrix is None or len(self.embeddings_matrix) == 0:
            logger.warning("No embeddings available for clustering")
            return {}
        
        # 用語数チェック
        if len(self.terms_data) < self.min_terms:
            logger.warning(f"Not enough terms for meaningful clustering. Have {len(self.terms_data)}, need at least {self.min_terms}")
            return {
                'status': 'skipped',
                'reason': f'Insufficient terms (have {len(self.terms_data)}, need {self.min_terms})',
                'terms_count': len(self.terms_data)
            }
        
        # ベクトルを正規化（コサイン類似度のため）
        normalized_embeddings = normalize(self.embeddings_matrix, norm='l2')
        
        # UMAP次元圧縮
        logger.info(f"Applying UMAP dimensional reduction: {normalized_embeddings.shape[1]} -> 20 dimensions")
        umap_reducer = umap.UMAP(
            n_components=20,  # 20次元に圧縮
            n_neighbors=15,  # 近傍サンプル数
            min_dist=0.1,  # クラスタ内の密度制御
            metric='cosine',  # コサイン距離
            random_state=42
        )
        reduced_embeddings = umap_reducer.fit_transform(normalized_embeddings)
        logger.info(f"UMAP reduction complete: shape {reduced_embeddings.shape}")
        
        # HDBSCAN実行（改善された設定）
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=2,  # 最小クラスタサイズ
            min_samples=1,
            cluster_selection_epsilon=0.3,  # クラスタ選択の柔軟性
            cluster_selection_method='leaf',  # より多くの点を含む
            metric='euclidean',  # 圧縮後のユークリッド距離
            allow_single_cluster=True,
            prediction_data=True
        )
        
        self.clusters = self.clusterer.fit_predict(reduced_embeddings)
        
        # 圧縮後のデータも保存（可視化用）
        self.reduced_embeddings = reduced_embeddings
        
        # クラスタリング結果の統計
        unique_clusters = set(self.clusters)
        n_clusters = len([c for c in unique_clusters if c >= 0])
        n_noise = sum(1 for c in self.clusters if c == -1)
        
        logger.info(f"Found {n_clusters} clusters, {n_noise} noise points")
        
        # シルエット係数計算（ノイズ点を除外、圧縮後のデータで計算）
        silhouette = None
        if n_clusters >= 2:
            mask = self.clusters >= 0
            if sum(mask) >= 2:
                silhouette = silhouette_score(
                    self.reduced_embeddings[mask], 
                    self.clusters[mask]
                )
                logger.info(f"Silhouette score: {silhouette:.3f}")
        
        return {
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'silhouette_score': silhouette,
            'cluster_labels': self.clusters.tolist()
        }
    
    async def name_clusters_with_llm(self, cluster_terms: Dict[int, List[str]]) -> Dict[int, str]:
        """LLMを使用してクラスタに名前を付ける"""
        cluster_names = {}
        
        for cluster_id, terms in cluster_terms.items():
            if cluster_id == -1:  # ノイズクラスタはスキップ
                continue
            
            # クラスタ内の用語と定義を準備
            cluster_info = []
            for term in terms[:10]:  # 最大10個の代表的な用語
                term_data = next(t for t in self.terms_data if t['term'] == term)
                cluster_info.append(f"- {term}: {term_data['definition'][:100]}")
            
            prompt = f"""
以下の専門用語グループに適切なカテゴリ名を付けてください。
カテゴリ名は短く（1-3語）、日本語で、技術分野を表すものにしてください。

用語グループ:
{chr(10).join(cluster_info)}

カテゴリ名のみを返してください:
"""
            
            try:
                response = await llm.ainvoke(prompt)
                cluster_names[cluster_id] = response.content.strip()
                logger.info(f"Cluster {cluster_id} named: {cluster_names[cluster_id]}")
            except Exception as e:
                logger.error(f"Error naming cluster {cluster_id}: {e}")
                cluster_names[cluster_id] = f"クラスタ{cluster_id}"
        
        return cluster_names

    async def llm_judge_candidate_synonym(
        self,
        spec_term: str,
        spec_def: str,
        candidate_term: str
    ) -> bool:
        """
        LLMで候補用語が類義語かどうかを判定（定義なし候補用語向け）

        Args:
            spec_term: 専門用語
            spec_def: 専門用語の定義
            candidate_term: 候補用語

        Returns:
            True: 類義語, False: 非類義語（包含関係・関連語）
        """
        import json

        prompt_template = get_synonym_judgment_single_definition_prompt()

        try:
            response = await llm.ainvoke(
                prompt_template.format(
                    spec_term=spec_term,
                    spec_def=spec_def,
                    candidate_term=candidate_term
                )
            )

            # JSON解析
            content = response.content.strip()
            # コードブロックを除去
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            result = json.loads(content)
            is_synonym = result.get("is_synonym", True)
            reason = result.get("reason", "")

            logger.debug(f"LLM判定: '{spec_term}' ↔ '{candidate_term}': {is_synonym} ({reason})")
            return is_synonym

        except Exception as e:
            logger.warning(f"LLM判定失敗 '{spec_term}' ↔ '{candidate_term}': {e}")
            # エラー時は保守的に類義語と判定（偽陰性を避ける）
            return True

    async def llm_judge_synonyms_bulk(
        self,
        term: str,
        definition: str,
        candidates: List[Dict[str, Any]],
        specialized_terms: List[Dict[str, Any]],
        spec_count: int
    ) -> List[int]:
        """
        LLMで1つの専門用語に対して複数の候補用語をまとめて類義語判定（バルク処理）

        Args:
            term: 専門用語
            definition: 専門用語の定義
            candidates: 候補用語リスト [{"term": "候補1", "similarity": 0.9, "is_specialized": True}]
            specialized_terms: 全専門用語リスト（定義取得用）
            spec_count: 専門用語の数

        Returns:
            類義語と判定された候補のインデックスリスト [0, 2, 5]
        """
        import json
        import re

        # 候補リストを番号付きで作成
        candidate_list = []
        for i, cand in enumerate(candidates):
            cand_term = cand['term']
            is_specialized = cand.get('is_specialized', False)

            # 定義取得
            if is_specialized:
                cand_def = next(
                    (t.get('definition', '') for t in specialized_terms if t['term'] == cand_term),
                    ''
                )
            else:
                # 候補用語は定義なし
                cand_def = ''

            cand_def_str = cand_def if cand_def else "（定義なし）"
            candidate_list.append(f"{i+1}. {cand_term}: {cand_def_str}")

        candidates_text = "\n".join(candidate_list)

        prompt = f"""専門用語: {term}
定義: {definition or "（定義なし）"}

以下の候補用語の中で、上記の専門用語の類義語を全て選んでください。

【類義語の判定基準】
- ほぼ同じ意味を持つ用語（同義語）
- 括弧書きで示される別名表記（例: 「ガス軸受（空気軸受）」の場合、「空気軸受」は類義語）
- 異なる呼び方だが実質的に同じものを指す用語

【除外する用語】
- 包含関係（一方が他方の一部や上位概念）
- 関連語（関連はあるが同じものではない）

候補用語:
{candidates_text}

類義語の番号のみをカンマ区切りで返してください（例: 1,3,5）
類義語がない場合は「なし」と返してください。
番号以外の説明は不要です。"""

        try:
            response = await llm.ainvoke(prompt)
            content = response.content.strip()

            # パース処理
            if content in ["なし", "無し", "None", "none", ""]:
                return []

            # 番号を抽出（カンマ区切りまたはスペース区切り）
            numbers = re.findall(r'\d+', content)
            synonym_indices = [int(num) - 1 for num in numbers if 0 < int(num) <= len(candidates)]

            logger.debug(f"LLM判定（バルク）: '{term}' → {len(synonym_indices)}/{len(candidates)}件が類義語")
            return synonym_indices

        except Exception as e:
            logger.warning(f"LLMバルク判定失敗 '{term}': {e}")
            # エラー時は空リスト（保守的に類義語なしとする）
            return []

    async def llm_judge_synonym_with_definitions(
        self,
        term1: str,
        def1: str,
        term2: str,
        def2: str
    ) -> bool:
        """
        LLMで2つの用語が類義語かどうかを判定（両方の定義あり）
        ※この関数は後方互換性のために残すが、バルク処理の方が効率的

        Args:
            term1: 用語1
            def1: 用語1の定義
            term2: 用語2
            def2: 用語2の定義

        Returns:
            True: 類義語, False: 非類義語（包含関係・関連語）
        """
        import json

        prompt_template = get_synonym_judgment_with_definitions_prompt()

        try:
            response = await llm.ainvoke(
                prompt_template.format(
                    term1=term1,
                    def1=def1 or "（定義なし）",
                    term2=term2,
                    def2=def2 or "（定義なし）"
                )
            )

            # JSON解析
            content = response.content.strip()
            # コードブロックを除去
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            result = json.loads(content)
            is_synonym = result.get("is_synonym", True)
            reason = result.get("reason", "")

            logger.debug(f"LLM判定: '{term1}' ↔ '{term2}': {is_synonym} ({reason})")
            return is_synonym

        except Exception as e:
            logger.warning(f"LLM判定失敗 '{term1}' ↔ '{term2}': {e}")
            # エラー時は保守的に類義語と判定（偽陰性を避ける）
            return True

    async def extract_semantic_synonyms_hybrid(
        self,
        specialized_terms: List[Dict[str, Any]],
        candidate_terms: List[Dict[str, Any]],
        similarity_threshold: float = None,  # Deprecated: 互換性のため残すが未使用
        max_synonyms: int = 5,
        use_llm_naming: bool = True,
        use_llm_for_candidates: bool = True
    ) -> Dict[str, Any]:
        """
        2段階エンベディングによる意味ベース類義語抽出

        フィルタリング: HDBSCANクラスタリング + LLM判定のみ
        (コサイン類似度フィルタは削除: クラスタリング済みなので冗長)

        Args:
            specialized_terms: 専門用語リスト [{"term": "ETC", "definition": "...", "text": "ETC: ..."}]
            candidate_terms: 候補用語リスト [{"term": "過給機", "text": "過給機"}]
            similarity_threshold: 非推奨（互換性のため残存、未使用）
            max_synonyms: 各用語の最大類義語数
            use_llm_naming: LLMによるクラスタ命名を使用するかどうか

        Returns:
            {
                "synonyms": {
                    "ETC": [
                        {"term": "電動ターボチャージャ", "similarity": 0.92, "is_specialized": True},
                        {"term": "過給機", "similarity": 0.87, "is_specialized": False}
                    ]
                },
                "clusters": {"ETC": 0, "過給機": 0},
                "cluster_names": {0: "軸受技術", 1: "電動化システム"}
            }
        """
        # 互換性のための警告
        if similarity_threshold is not None:
            logger.warning(f"similarity_threshold={similarity_threshold} is deprecated and ignored. "
                          "Filtering now uses HDBSCAN clustering + LLM judgment only.")
        logger.info(f"Starting hybrid semantic synonym extraction: {len(specialized_terms)} specialized, {len(candidate_terms)} candidates")

        # terms_dataに専門用語を保存（LLM命名用）
        self.terms_data = specialized_terms

        # 1. エンベディング生成（統一形式: term + context/definition）
        logger.info("Generating embeddings for specialized terms (term + definition)...")
        spec_texts = [t['text'] for t in specialized_terms]
        spec_embeddings_list = self.embeddings.embed_documents(spec_texts)
        spec_embeddings = np.array(spec_embeddings_list)

        # 候補用語: 'text'フィールドを使用（周辺テキスト or 用語のみ）
        logger.info("Generating embeddings for candidate terms (term + context)...")
        cand_texts = [t.get('text', t['term']) for t in candidate_terms]
        cand_embeddings_list = self.embeddings.embed_documents(cand_texts)
        cand_embeddings = np.array(cand_embeddings_list)

        # 2. 統合
        all_embeddings = np.vstack([spec_embeddings, cand_embeddings])
        all_terms = specialized_terms + candidate_terms
        logger.info(f"Combined embeddings shape: {all_embeddings.shape}")

        # 3. 正規化（コサイン類似度用）
        normalized_embeddings = normalize(all_embeddings, norm='l2')

        # 4. UMAP次元圧縮
        logger.info("Applying UMAP dimensional reduction...")
        # n_componentsはデータ数より小さくする必要がある
        n_samples = len(all_embeddings)
        n_components = min(20, max(2, n_samples // 2))  # データ数の半分以下、最低2
        n_neighbors = min(15, max(2, n_samples // 3))   # データ数の1/3以下、最低2

        logger.info(f"UMAP params: n_samples={n_samples}, n_components={n_components}, n_neighbors={n_neighbors}")

        umap_reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric='cosine',
            random_state=42
        )
        reduced_embeddings = umap_reducer.fit_transform(normalized_embeddings)
        logger.info(f"UMAP reduction complete: shape {reduced_embeddings.shape}")

        # 5. HDBSCANクラスタリング
        logger.info("Performing HDBSCAN clustering...")
        # min_cluster_sizeを動的に調整: データ数の3%以上、最低2
        min_cluster_size = max(2, int(n_samples * 0.03))
        logger.info(f"HDBSCAN min_cluster_size: {min_cluster_size}")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=1,
            cluster_selection_epsilon=0.5,
            cluster_selection_method='leaf',
            metric='euclidean',
            allow_single_cluster=True,
            prediction_data=True
        )
        clusters = clusterer.fit_predict(reduced_embeddings)

        n_clusters = len([c for c in set(clusters) if c >= 0])
        n_noise = sum(1 for c in clusters if c == -1)
        logger.info(f"Clustering complete: {n_clusters} clusters, {n_noise} noise points")

        # クラスタリング結果の詳細ログを出力
        logger.info("\n" + "="*60)
        logger.info("Clustering Details:")
        logger.info("="*60)
        spec_count = len(specialized_terms)

        # 専門用語のクラスタ配置を表示
        for cluster_id in sorted(set(clusters)):
            cluster_terms = []
            for idx, c_id in enumerate(clusters):
                if c_id == cluster_id:
                    term_obj = all_terms[idx]
                    term_name = term_obj['term']
                    is_spec = idx < spec_count
                    marker = "★" if is_spec else "○"
                    cluster_terms.append(f"{marker}{term_name}")

            cluster_label = f"Cluster {cluster_id}" if cluster_id >= 0 else "Noise"
            logger.info(f"\n{cluster_label} ({len(cluster_terms)} terms):")
            logger.info("  " + ", ".join(cluster_terms))

        logger.info("\n" + "="*60)

        # 6. 専門用語ごとに類義語抽出とクラスタマッピング
        synonyms_dict = {}
        cluster_mapping = {}  # term -> cluster_id のマッピング

        # 全専門用語の類義語候補を一度に収集（バルク処理用）
        all_llm_tasks = []
        all_task_metadata = []
        term_to_candidates = {}  # term_name -> [(sim_item, task_idx), ...]

        for idx in range(spec_count):
            spec_term = specialized_terms[idx]
            term_name = spec_term['term']
            cluster_id = clusters[idx]

            # クラスタIDを記録（ノイズクラスタも含む）
            cluster_mapping[term_name] = int(cluster_id)

            # ノイズクラスタはスキップ（類義語抽出のみ）
            if cluster_id == -1:
                logger.debug(f"Term '{term_name}' is in noise cluster, skipping synonym extraction")
                continue

            # 同一クラスタ内の他の用語を検索
            same_cluster_indices = [
                i for i, c in enumerate(clusters)
                if c == cluster_id and i != idx
            ]

            if not same_cluster_indices:
                continue

            # 類似度計算（記録用のみ、フィルタリングには使用しない）
            term_embedding = normalized_embeddings[idx]
            similarities = []

            for other_idx in same_cluster_indices:
                other_embedding = normalized_embeddings[other_idx]
                # コサイン類似度を計算（記録のみ、フィルタリングなし）
                similarity = float(np.dot(term_embedding, other_embedding))

                other_term = all_terms[other_idx]
                other_term_name = other_term['term']
                is_specialized = other_idx < spec_count

                # 自分自身は除外
                if other_term_name == term_name:
                    continue

                # 包含関係（related_terms）に含まれる用語は類義語から除外
                related_terms = spec_term.get('related_terms', [])
                if other_term_name in related_terms:
                    logger.debug(f"Skipping '{other_term_name}' for '{term_name}': in related_terms (inclusion relationship)")
                    continue

                # コサイン類似度フィルタを削除: クラスタリング済みなので全て候補に追加
                similarities.append({
                    'term': other_term_name,
                    'similarity': similarity,  # 記録のみ
                    'is_specialized': is_specialized
                })

            # LLM判定用タスクを収集（バルク判定: 1専門用語につき1タスク）
            if use_llm_for_candidates and similarities:
                spec_def = spec_term.get('definition', '')

                # 全候補をまとめて1回のLLM呼び出しで判定するタスクを作成
                task = self.llm_judge_synonyms_bulk(
                    term_name,
                    spec_def,
                    similarities,
                    specialized_terms,
                    spec_count
                )
                all_llm_tasks.append(task)
                all_task_metadata.append((term_name, similarities))

        # 全LLM判定をセマフォで制限付き並列実行（TPM/RPM制限対策）
        if use_llm_for_candidates and all_llm_tasks:
            import asyncio

            # 同時実行数制限を設定から取得
            max_concurrent = getattr(cfg, 'max_concurrent_llm_requests', 10)

            logger.info(f"Running LLM judgment: {len(all_llm_tasks)} tasks (max {max_concurrent} concurrent)")

            # セマフォを使用して同時実行数を制限
            semaphore = asyncio.Semaphore(max_concurrent)

            async def run_with_semaphore(task):
                async with semaphore:
                    return await task

            # 全タスクをセマフォ付きで実行
            results = await asyncio.gather(
                *[run_with_semaphore(task) for task in all_llm_tasks],
                return_exceptions=True
            )

            # 結果を各専門用語に振り分け
            for (term_name, similarities), result in zip(all_task_metadata, results):
                if isinstance(result, Exception):
                    logger.warning(f"LLM bulk judgment failed for '{term_name}': {result}")
                    term_to_candidates[term_name] = []
                    continue

                # resultは類義語と判定された候補のインデックスリスト
                synonym_indices = result
                llm_filtered_similarities = [similarities[idx] for idx in synonym_indices if idx < len(similarities)]

                # 結果を保存（後でmax_synonyms適用）
                term_to_candidates[term_name] = llm_filtered_similarities
                logger.debug(f"After LLM filtering for '{term_name}': {len(llm_filtered_similarities)}/{len(similarities)} synonyms remain")

        # LLM判定結果をsynonyms_dictに反映
        for term_name, filtered_similarities in term_to_candidates.items():
            # 類似度順にソート
            filtered_similarities.sort(key=lambda x: x['similarity'], reverse=True)

            # 上位N個のみ保存
            if filtered_similarities:
                synonyms_dict[term_name] = filtered_similarities[:max_synonyms]
                logger.debug(f"Found {len(filtered_similarities[:max_synonyms])} synonyms for '{term_name}'")

        logger.info(f"Extracted semantic synonyms for {len(synonyms_dict)} terms")
        logger.info(f"Mapped {len(cluster_mapping)} terms to clusters")

        # LLMによるクラスタ命名（オプション）
        cluster_names = {}
        if use_llm_naming:
            try:
                # クラスタIDごとに用語をグループ化
                cluster_terms_map = {}
                for idx, cluster_id in enumerate(clusters[:spec_count]):
                    if cluster_id >= 0:  # ノイズクラスタを除外
                        if cluster_id not in cluster_terms_map:
                            cluster_terms_map[cluster_id] = []
                        cluster_terms_map[cluster_id].append(specialized_terms[idx]['term'])

                logger.info(f"Naming {len(cluster_terms_map)} clusters with LLM...")
                cluster_names = await self.name_clusters_with_llm(cluster_terms_map)
                logger.info(f"LLM naming complete: {cluster_names}")
            except Exception as e:
                logger.error(f"LLM cluster naming failed: {e}. Using default names.", exc_info=True)
                # フォールバック: クラスタN
                unique_clusters = set(c for c in cluster_mapping.values() if c >= 0)
                cluster_names = {cid: f"クラスタ{cid}" for cid in unique_clusters}

        # クラスタ情報とクラスタ名を含む結果を返す
        return {
            'synonyms': synonyms_dict,
            'clusters': cluster_mapping,
            'cluster_names': cluster_names
        }

    def _ensure_bidirectional_synonyms(
        self,
        synonyms_dict: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        類義語関係の双方向性を保証する

        A→Bの類義語関係がある場合、B→Aも追加する
        例: "ETC" → ["電動ターボチャージャ"] の場合、
            "電動ターボチャージャ" → ["ETC"] も追加

        Args:
            synonyms_dict: 元の類義語辞書 {term: [{"term": ..., "similarity": ...}]}

        Returns:
            双方向性を保証した類義語辞書
        """
        from collections import defaultdict

        # 全ての類義語関係を収集
        synonym_pairs = defaultdict(dict)  # {term1: {term2: similarity, ...}}

        for term1, synonyms in synonyms_dict.items():
            for syn_info in synonyms:
                term2 = syn_info['term']
                similarity = syn_info.get('similarity', 0.85)
                is_specialized = syn_info.get('is_specialized', False)

                # 双方向に登録
                synonym_pairs[term1][term2] = {
                    'similarity': similarity,
                    'is_specialized': is_specialized
                }
                # 逆方向も登録（存在しない場合のみ）
                if term2 not in synonym_pairs or term1 not in synonym_pairs[term2]:
                    synonym_pairs[term2][term1] = {
                        'similarity': similarity,
                        'is_specialized': is_specialized
                    }

        # Dict[str, List[Dict]]形式に変換
        bidirectional_dict = {}
        for term, syn_map in synonym_pairs.items():
            bidirectional_dict[term] = [
                {
                    'term': syn_term,
                    'similarity': info['similarity'],
                    'is_specialized': info['is_specialized']
                }
                for syn_term, info in syn_map.items()
            ]

        logger.info(f"Ensured bidirectional synonyms: {len(synonyms_dict)} → {len(bidirectional_dict)} terms")
        return bidirectional_dict

    def update_semantic_synonyms_to_db(
        self,
        synonyms_dict: Dict[str, List[Dict[str, Any]]],
        cluster_mapping: Dict[str, int] = None,
        cluster_names: Dict[int, str] = None,
        collection_name: str = None
    ):
        """
        抽出した意味的類義語とクラスタ情報をDBに保存

        重要: cluster_mappingの全用語を処理するため、類義語がない用語もdomainが更新される

        Args:
            synonyms_dict: 類義語辞書 {term: [{"term": ..., "similarity": ...}]}
            cluster_mapping: クラスタマッピング {term: cluster_id} ※全用語を含む
            cluster_names: クラスタ名マッピング {cluster_id: "軸受技術"}
            collection_name: コレクション名（指定されない場合はデフォルト値を使用）
        """
        if not cluster_mapping:
            logger.warning("No cluster_mapping provided, skipping domain update")
            return 0

        # collection_nameが指定されていない場合はデフォルト値を使用
        if collection_name is None:
            from rag.config import Config
            cfg = Config()
            collection_name = cfg.collection_name
            logger.warning(f"⚠️ collection_name not provided, using default: '{collection_name}'")
        else:
            logger.info(f"✓ Using provided collection_name: '{collection_name}'")

        engine = create_engine(self.connection_string)

        # データベース状態確認（UPDATE前）
        with engine.connect() as conn:
            check_result = conn.execute(
                text(f"SELECT COUNT(*) FROM {self.jargon_table_name} WHERE collection_name = :cname"),
                {"cname": collection_name}
            ).scalar()
            logger.info(f"📊 Target collection '{collection_name}' has {check_result} terms in database before update")

            if check_result == 0:
                logger.error(f"❌ No terms found in collection '{collection_name}' - UPDATE will fail!")
                return 0

        updated_count = 0
        synonyms_count = 0
        no_synonyms_count = 0
        noise_count = 0

        # 類義語の双方向性を保証: A→Bの関係があればB→Aも追加
        bidirectional_synonyms = self._ensure_bidirectional_synonyms(synonyms_dict)

        with engine.begin() as conn:
            # cluster_mappingの全用語をループ（類義語なしでもdomain更新）
            for term, cluster_id in cluster_mapping.items():
                # 1. 類義語を取得（双方向性保証済み）
                synonyms = bidirectional_synonyms.get(term, [])
                synonym_terms = [s['term'] for s in synonyms]

                # 2. domainを決定（全用語に必ず値を設定）
                if cluster_id >= 0:
                    # LLM命名があればそれを使用、なければフォールバック
                    if cluster_names and cluster_id in cluster_names:
                        domain = cluster_names[cluster_id]
                    else:
                        domain = f"クラスタ{cluster_id}"
                else:
                    domain = "未分類"
                    noise_count += 1

                # 統計カウント
                if synonyms:
                    synonyms_count += 1
                else:
                    no_synonyms_count += 1

                try:
                    # 3. 無条件で上書き（COALESCEなし）
                    # Note: 'aliases'列に類義語を保存（semantic_synonymsは存在しない）
                    # WHERE句を複合キーに修正
                    result = conn.execute(
                        text(f"""
                            UPDATE {self.jargon_table_name}
                            SET aliases = :synonyms,
                                domain = :domain
                            WHERE collection_name = :collection_name AND term = :term
                        """),
                        {
                            "collection_name": collection_name,
                            "term": term,
                            "synonyms": synonym_terms,
                            "domain": domain
                        }
                    )

                    # 実際に更新された行数を確認
                    if result.rowcount == 0:
                        logger.warning(f"⚠️ UPDATE failed for term='{term}', collection='{collection_name}' (0 rows updated - term not found in DB)")
                    else:
                        updated_count += 1
                        logger.debug(f"✓ Updated term='{term}': domain='{domain}', synonyms={len(synonym_terms)} ({synonym_terms})")

                except Exception as e:
                    logger.error(f"Error updating term '{term}': {e}", exc_info=True)

        logger.info(f"Updated domain field for {updated_count} terms in database:")
        logger.info(f"  - With synonyms: {synonyms_count} terms")
        logger.info(f"  - Without synonyms: {no_synonyms_count} terms")
        logger.info(f"  - Noise cluster (未分類): {noise_count} terms")

        return updated_count

    async def analyze_and_save(self, output_path: str = "output/term_clusters.json", include_hierarchy: bool = True, use_llm_naming: bool = False) -> Dict[str, Any]:
        """完全な分析を実行して結果を保存"""
        
        # 1. 用語を読み込み
        self.load_terms_from_db()
        
        if len(self.terms_data) == 0:
            logger.warning("No terms found in database")
            return {'status': 'error', 'message': 'No terms found'}
        
        # 2. エンベディング生成
        self.generate_embeddings()
        
        # 3. クラスタリング実行
        clustering_result = self.perform_clustering()
        
        if clustering_result.get('status') == 'skipped':
            # 用語数が少ない場合は簡易カテゴリ分類のみ
            logger.info("Using manual categories due to insufficient terms")
            
            # 既存のdomainフィールドを使用
            domain_groups = {}
            for term in self.terms_data:
                domain = term.get('domain', 'その他')
                if domain not in domain_groups:
                    domain_groups[domain] = []
                domain_groups[domain].append(term['term'])
            
            result = {
                'status': 'manual_categorization',
                'timestamp': datetime.now().isoformat(),
                'total_terms': len(self.terms_data),
                'categories': domain_groups,
                'message': f'Using manual categorization. Will switch to clustering when {self.min_terms}+ terms available.'
            }
        else:
            # 4. クラスタごとの用語をグループ化
            cluster_terms = {}
            for i, cluster_id in enumerate(self.clusters):
                if cluster_id not in cluster_terms:
                    cluster_terms[cluster_id] = []
                cluster_terms[cluster_id].append(self.terms_data[i]['term'])
            
            # 5. クラスタに名前を付ける
            if use_llm_naming:
                cluster_names = await self.name_clusters_with_llm(cluster_terms)
            else:
                # LLMを使わない場合は簡単な名前を付ける
                cluster_names = {}
                for cluster_id in cluster_terms.keys():
                    cluster_names[cluster_id] = f"クラスタ{cluster_id}"
            
            # 6. 結果を整理
            categorized_terms = {}
            for cluster_id, terms in cluster_terms.items():
                category_name = cluster_names.get(cluster_id, f"クラスタ{cluster_id}" if cluster_id >= 0 else "未分類")
                categorized_terms[category_name] = {
                    'terms': terms,
                    'count': len(terms),
                    'cluster_id': int(cluster_id)  # Convert numpy int64 to Python int
                }
            
            result = {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'total_terms': len(self.terms_data),
                'clustering_stats': clustering_result,
                'categories': categorized_terms
            }
            
            # 階層構造を追加
            if include_hierarchy:
                hierarchy_structure = self.build_hierarchical_structure()
                result['hierarchy'] = hierarchy_structure
        
        # 7. 結果を保存
        output_file = Path(output_path)
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Analysis results saved to {output_path}")
        return result
    
    def suggest_related_terms(self, term: str, n_suggestions: int = 5) -> List[Dict[str, Any]]:
        """指定用語の関連語を提案（同一クラスタから）"""
        if self.clusters is None or self.embeddings_matrix is None:
            logger.warning("Clustering not performed yet")
            return []
        
        # 指定用語のインデックスを検索
        term_idx = None
        for i, t in enumerate(self.terms_data):
            if t['term'] == term:
                term_idx = i
                break
        
        if term_idx is None:
            logger.warning(f"Term '{term}' not found")
            return []
        
        # 同じクラスタの用語を検索
        cluster_id = self.clusters[term_idx]
        if cluster_id == -1:
            logger.info(f"Term '{term}' is in noise cluster")
            return []
        
        # 同一クラスタ内の他の用語を距離順にソート
        same_cluster_indices = [
            i for i, c in enumerate(self.clusters) 
            if c == cluster_id and i != term_idx
        ]
        
        if not same_cluster_indices:
            return []
        
        # 距離計算
        term_embedding = self.embeddings_matrix[term_idx]
        distances = []
        for idx in same_cluster_indices:
            dist = np.linalg.norm(term_embedding - self.embeddings_matrix[idx])
            distances.append((idx, dist))
        
        # 距離でソート
        distances.sort(key=lambda x: x[1])
        
        # 上位N個を返す
        suggestions = []
        for idx, dist in distances[:n_suggestions]:
            suggestions.append({
                'term': self.terms_data[idx]['term'],
                'definition': self.terms_data[idx]['definition'],
                'distance': float(dist),
                'similarity': float(1 / (1 + dist))  # 簡易的な類似度スコア
            })
        
        return suggestions
    
    def analyze_condensed_tree(self) -> Dict[str, Any]:
        """HDBSCANのCondensed Treeから階層構造を抽出"""
        if self.clusterer is None:
            logger.warning("Clustering not performed yet")
            return {}
        
        # Condensed Treeを取得
        tree = self.clusterer.condensed_tree_
        tree_df = tree.to_pandas()
        
        # 階層情報を整理
        hierarchy_info = {
            'tree_data': [],
            'cluster_hierarchy': {},
            'term_hierarchy': []
        }
        
        # 1. ツリー構造の解析
        for _, row in tree_df.iterrows():
            parent_id = int(row['parent'])
            child_id = int(row['child'])
            lambda_val = float(row['lambda_val'])
            child_size = int(row['child_size'])
            
            # クラスタの分離情報を記録
            if child_size > 1:  # クラスタの場合
                hierarchy_info['tree_data'].append({
                    'parent': parent_id,
                    'child': child_id,
                    'lambda': lambda_val,
                    'size': child_size,
                    'type': 'cluster'
                })
                
                # クラスタの親子関係を記録
                if child_id not in hierarchy_info['cluster_hierarchy']:
                    hierarchy_info['cluster_hierarchy'][child_id] = {
                        'parent': parent_id,
                        'lambda': lambda_val,
                        'size': child_size,
                        'depth': self._calculate_depth(child_id, tree_df)
                    }
        
        # 2. 最終クラスタとCondensed Tree IDのマッピングを構築
        # Condensed Treeの末端ノード（他のクラスタの親になっていない）を特定
        all_parents = set([row['parent'] for row in hierarchy_info['tree_data']])
        final_clusters = []
        for cluster_id, info in hierarchy_info['cluster_hierarchy'].items():
            if cluster_id not in all_parents:
                final_clusters.append(cluster_id)
        
        # 最終クラスタをソートして、ラベル（0,1,2...）にマッピング
        final_clusters.sort()
        cluster_id_mapping = {i: tree_id for i, tree_id in enumerate(final_clusters)}
        
        logger.info(f"Cluster ID mapping: {cluster_id_mapping}")
        
        # 3. 各用語の階層レベルを判定
        if self.clusters is not None:
            for idx, term_data in enumerate(self.terms_data):
                cluster_label = self.clusters[idx]
                if cluster_label >= 0:
                    # 実際のCondensed Tree IDを取得
                    actual_tree_id = cluster_id_mapping.get(cluster_label)
                    
                    if actual_tree_id:
                        # クラスタ形成時のlambda値を取得
                        cluster_info = hierarchy_info['cluster_hierarchy'].get(actual_tree_id, {})
                        lambda_val = cluster_info.get('lambda', 0.0)
                        depth = cluster_info.get('depth', 0)
                    else:
                        # マッピングが見つからない場合のフォールバック
                        lambda_val = 0.0
                        depth = 0
                    
                    # lambda値に基づいて階層レベルを判定
                    if lambda_val > 0.89:  # より具体的な閾値
                        level = "具体的概念"
                    elif lambda_val > 0.88:
                        level = "中間概念"
                    else:
                        level = "上位概念"
                    
                    hierarchy_info['term_hierarchy'].append({
                        'term': term_data['term'],
                        'cluster': int(cluster_label),
                        'tree_cluster_id': actual_tree_id if actual_tree_id else -1,
                        'lambda': lambda_val,
                        'depth': depth,
                        'level': level
                    })
        
        # 3. クラスタの永続性情報
        if hasattr(self.clusterer, 'cluster_persistence_'):
            persistence = self.clusterer.cluster_persistence_
            for cluster_id, persist_val in enumerate(persistence):
                if cluster_id in hierarchy_info['cluster_hierarchy']:
                    hierarchy_info['cluster_hierarchy'][cluster_id]['persistence'] = float(persist_val)
        
        logger.info(f"Analyzed condensed tree with {len(hierarchy_info['tree_data'])} nodes")
        return hierarchy_info
    
    def _calculate_depth(self, node_id: int, tree_df) -> int:
        """ノードの階層深さを計算"""
        depth = 0
        current_node = node_id
        
        # 親を辿って深さを計算
        while True:
            parent_rows = tree_df[tree_df['child'] == current_node]
            if parent_rows.empty:
                break
            current_node = int(parent_rows.iloc[0]['parent'])
            depth += 1
            
            # 無限ループ防止
            if depth > 100:
                break
        
        return depth
    
    def build_hierarchical_structure(self) -> Dict[str, Any]:
        """階層構造を構築して整理"""
        if self.clusterer is None or self.clusters is None:
            logger.warning("Clustering not performed yet")
            return {}
        
        # Condensed Tree解析
        hierarchy_info = self.analyze_condensed_tree()
        
        # 階層ごとに用語をグループ化
        hierarchical_groups = {
            '上位概念': [],
            '中間概念': [],
            '具体的概念': [],
            '未分類': []
        }
        
        # 用語を階層レベルごとに分類
        for term_info in hierarchy_info.get('term_hierarchy', []):
            level = term_info.get('level', '未分類')
            hierarchical_groups[level].append({
                'term': term_info['term'],
                'lambda': term_info.get('lambda', 0),
                'depth': term_info.get('depth', 0)
            })
        
        # 未分類の用語（ノイズポイント）を追加
        for idx, term_data in enumerate(self.terms_data):
            if self.clusters[idx] == -1:
                hierarchical_groups['未分類'].append({
                    'term': term_data['term'],
                    'lambda': 0,
                    'depth': -1
                })
        
        # 結果を整理
        result = {
            'hierarchical_groups': hierarchical_groups,
            'tree_statistics': {
                'total_clusters': len(hierarchy_info.get('cluster_hierarchy', {})),
                'max_depth': max([info['depth'] for info in hierarchy_info.get('term_hierarchy', [{'depth': 0}])]),
                'persistence_scores': {
                    cid: info.get('persistence', 0) 
                    for cid, info in hierarchy_info.get('cluster_hierarchy', {}).items()
                }
            },
            'condensed_tree_raw': hierarchy_info
        }
        
        return result

# ── Main Execution ────────────────────────────────
async def main():
    """メイン実行関数"""
    analyzer = TermClusteringAnalyzer(PG_URL, min_terms=3)
    
    # 分析実行
    result = await analyzer.analyze_and_save()
    
    # 結果表示
    print("\n" + "="*50)
    print("Term Clustering Analysis Results")
    print("="*50)
    
    if result['status'] == 'manual_categorization':
        print(f"Status: Manual categorization (not enough terms)")
        print(f"Total terms: {result['total_terms']}")
        print(f"\nCategories:")
        for category, terms in result['categories'].items():
            print(f"\n[{category}]: {len(terms)} terms")
            for term in terms[:5]:
                print(f"  - {term}")
            if len(terms) > 5:
                print(f"  ... and {len(terms)-5} more")
    
    elif result['status'] == 'success':
        stats = result['clustering_stats']
        print(f"Total terms: {result['total_terms']}")
        print(f"Clusters found: {stats['n_clusters']}")
        print(f"Noise points: {stats['n_noise']}")
        if stats['silhouette_score']:
            print(f"Silhouette score: {stats['silhouette_score']:.3f}")
        
        print(f"\nCategories:")
        for category, info in result['categories'].items():
            print(f"\n[{category}]: {info['count']} terms")
            for term in info['terms'][:5]:
                print(f"  - {term}")
            if len(info['terms']) > 5:
                print(f"  ... and {len(info['terms'])-5} more")
        
        # 階層構造の表示
        if 'hierarchy' in result:
            print(f"\n" + "="*50)
            print("Hierarchical Structure (based on HDBSCAN Condensed Tree)")
            print("="*50)
            
            hierarchy = result['hierarchy']
            groups = hierarchy.get('hierarchical_groups', {})
            
            for level_name in ['上位概念', '中間概念', '具体的概念', '未分類']:
                if level_name in groups and groups[level_name]:
                    print(f"\n[{level_name}]:")
                    for item in groups[level_name][:10]:
                        depth_indicator = "  " * max(0, item.get('depth', 0))
                        lambda_str = f"(λ={item.get('lambda', 0):.3f})" if item.get('lambda', 0) > 0 else ""
                        print(f"{depth_indicator}- {item['term']} {lambda_str}")
                    if len(groups[level_name]) > 10:
                        print(f"  ... and {len(groups[level_name])-10} more")
            
            # ツリー統計の表示
            tree_stats = hierarchy.get('tree_statistics', {})
            if tree_stats:
                print(f"\nTree Statistics:")
                print(f"  Total clusters: {tree_stats.get('total_clusters', 0)}")
                print(f"  Max depth: {tree_stats.get('max_depth', 0)}")
    
    print(f"\nFull results saved to: output/term_clusters.json")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())