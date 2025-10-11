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

# Azure OpenAI LLM for naming
llm = AzureChatOpenAI(
    azure_deployment=cfg.azure_openai_chat_deployment_name,
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
            SELECT term, definition, domain, aliases
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

    def extract_semantic_synonyms_hybrid(
        self,
        specialized_terms: List[Dict[str, Any]],
        candidate_terms: List[Dict[str, Any]],
        similarity_threshold: float = 0.85,
        max_synonyms: int = 5
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        2段階エンベディングによる意味ベース類義語抽出

        Args:
            specialized_terms: 専門用語リスト [{"term": "ETC", "definition": "...", "text": "ETC: ..."}]
            candidate_terms: 候補用語リスト [{"term": "過給機", "text": "過給機"}]
            similarity_threshold: コサイン類似度の閾値
            max_synonyms: 各用語の最大類義語数

        Returns:
            {
                "ETC": [
                    {"term": "電動ターボチャージャ", "similarity": 0.92, "is_specialized": True},
                    {"term": "過給機", "similarity": 0.87, "is_specialized": False}
                ]
            }
        """
        logger.info(f"Starting hybrid semantic synonym extraction: {len(specialized_terms)} specialized, {len(candidate_terms)} candidates")

        # 1. エンベディング生成（2段階）
        logger.info("Generating embeddings for specialized terms (term + definition)...")
        spec_texts = [t['text'] for t in specialized_terms]
        spec_embeddings_list = self.embeddings.embed_documents(spec_texts)
        spec_embeddings = np.array(spec_embeddings_list)

        logger.info("Generating embeddings for candidate terms (term only)...")
        cand_texts = [t['text'] for t in candidate_terms]
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
        umap_reducer = umap.UMAP(
            n_components=20,
            n_neighbors=15,
            min_dist=0.1,
            metric='cosine',
            random_state=42
        )
        reduced_embeddings = umap_reducer.fit_transform(normalized_embeddings)
        logger.info(f"UMAP reduction complete: shape {reduced_embeddings.shape}")

        # 5. HDBSCANクラスタリング
        logger.info("Performing HDBSCAN clustering...")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=2,
            min_samples=1,
            cluster_selection_epsilon=0.3,
            cluster_selection_method='leaf',
            metric='euclidean',
            allow_single_cluster=True,
            prediction_data=True
        )
        clusters = clusterer.fit_predict(reduced_embeddings)

        n_clusters = len([c for c in set(clusters) if c >= 0])
        n_noise = sum(1 for c in clusters if c == -1)
        logger.info(f"Clustering complete: {n_clusters} clusters, {n_noise} noise points")

        # 6. 専門用語ごとに類義語抽出
        synonyms_dict = {}
        spec_count = len(specialized_terms)

        for idx in range(spec_count):
            spec_term = specialized_terms[idx]
            term_name = spec_term['term']
            cluster_id = clusters[idx]

            # ノイズクラスタはスキップ
            if cluster_id == -1:
                logger.debug(f"Term '{term_name}' is in noise cluster, skipping")
                continue

            # 同一クラスタ内の他の用語を検索
            same_cluster_indices = [
                i for i, c in enumerate(clusters)
                if c == cluster_id and i != idx
            ]

            if not same_cluster_indices:
                continue

            # コサイン類似度を計算
            term_embedding = normalized_embeddings[idx]
            similarities = []

            for other_idx in same_cluster_indices:
                other_embedding = normalized_embeddings[other_idx]
                # コサイン類似度（正規化済みなので内積）
                similarity = float(np.dot(term_embedding, other_embedding))

                if similarity >= similarity_threshold:
                    other_term = all_terms[other_idx]
                    is_specialized = other_idx < spec_count

                    similarities.append({
                        'term': other_term['term'],
                        'similarity': similarity,
                        'is_specialized': is_specialized
                    })

            # 類似度順にソート
            similarities.sort(key=lambda x: x['similarity'], reverse=True)

            # 上位N個のみ保存
            if similarities:
                synonyms_dict[term_name] = similarities[:max_synonyms]
                logger.debug(f"Found {len(similarities[:max_synonyms])} synonyms for '{term_name}'")

        logger.info(f"Extracted semantic synonyms for {len(synonyms_dict)} terms")
        return synonyms_dict

    def update_semantic_synonyms_to_db(self, synonyms_dict: Dict[str, List[Dict[str, Any]]]):
        """
        抽出した意味的類義語をDBに保存

        Args:
            synonyms_dict: extract_semantic_synonyms_hybrid()の出力
        """
        engine = create_engine(self.connection_string)

        updated_count = 0
        with engine.begin() as conn:
            for term, synonyms in synonyms_dict.items():
                # 類義語のリスト（用語名のみ）
                synonym_terms = [s['term'] for s in synonyms]

                try:
                    conn.execute(
                        text(f"""
                            UPDATE {self.jargon_table_name}
                            SET semantic_synonyms = :synonyms
                            WHERE term = :term
                        """),
                        {"term": term, "synonyms": synonym_terms}
                    )
                    updated_count += 1
                except Exception as e:
                    logger.error(f"Error updating semantic synonyms for term '{term}': {e}")

        logger.info(f"Updated semantic synonyms for {updated_count} terms in database")
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