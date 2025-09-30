#!/usr/bin/env python3
"""
Knowledge Graph Builder
既存の専門用語辞書とクラスタリング結果からナレッジグラフを構築
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import numpy as np

import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from rag.config import Config

# ============================================
# Configuration
# ============================================
load_dotenv()
cfg = Config()

PG_URL = f"host={cfg.db_host} port={cfg.db_port} dbname={cfg.db_name} user={cfg.db_user} password={cfg.db_password}"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Provenance types
PROVENANCE_TYPES = {
    "clustering": "HDBSCANクラスタリング結果から",
    "definition": "定義文パターンマッチングから",
    "cooccurrence": "共起関係から",
    "llm": "LLMによる抽出",
    "inference": "推論による導出",
    "manual": "手動入力",
    "hierarchy": "階層推定から"
}

# ============================================
# Database Helper Functions
# ============================================

class KnowledgeGraphDB:
    """ナレッジグラフのデータベース操作クラス"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.conn = None
    
    def __enter__(self):
        self.conn = psycopg.connect(self.connection_string, row_factory=dict_row)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()
    
    def create_node(self, node_type: str, term: str, definition: str = None, 
                   properties: Dict = None, embedding: np.ndarray = None) -> str:
        """ノードを作成"""
        properties = properties or {}
        
        query = """
        INSERT INTO knowledge_nodes (node_type, term, definition, properties, embedding)
        VALUES (%(type)s, %(term)s, %(def)s, %(props)s, %(emb)s)
        ON CONFLICT (term) WHERE node_type = 'Term'
        DO UPDATE SET
            definition = COALESCE(EXCLUDED.definition, knowledge_nodes.definition),
            properties = knowledge_nodes.properties || EXCLUDED.properties,
            embedding = COALESCE(EXCLUDED.embedding, knowledge_nodes.embedding)
        RETURNING id;
        """
        
        with self.conn.cursor() as cur:
            cur.execute(query, {
                'type': node_type,
                'term': term,
                'def': definition,
                'props': json.dumps(properties),
                'emb': embedding.tolist() if embedding is not None else None
            })
            self.conn.commit()
            return cur.fetchone()['id']
    
    def get_node_by_term(self, term: str) -> Optional[Dict]:
        """用語からノードを取得"""
        query = "SELECT * FROM knowledge_nodes WHERE term = %s"
        with self.conn.cursor() as cur:
            cur.execute(query, (term,))
            return cur.fetchone()
    
    def create_edge(self, source_id: str, target_id: str, edge_type: str,
                   weight: float = 1.0, confidence: float = 1.0,
                   provenance: str = "manual", evidence: str = None) -> str:
        """エッジを作成（重複時は最大値採用）"""
        if source_id == target_id:
            logger.warning(f"Skipping self-loop: {source_id}")
            return None
        
        properties = {'provenance': provenance}
        if evidence:
            properties['evidence'] = evidence
        
        query = """
        INSERT INTO knowledge_edges 
        (source_id, target_id, edge_type, weight, confidence, properties)
        VALUES (%(src)s, %(tgt)s, %(type)s, %(weight)s, %(conf)s, %(props)s)
        ON CONFLICT (source_id, target_id, edge_type)
        DO UPDATE SET
            weight = GREATEST(knowledge_edges.weight, EXCLUDED.weight),
            confidence = GREATEST(knowledge_edges.confidence, EXCLUDED.confidence),
            properties = knowledge_edges.properties || EXCLUDED.properties
        RETURNING id;
        """
        
        with self.conn.cursor() as cur:
            cur.execute(query, {
                'src': source_id,
                'tgt': target_id,
                'type': edge_type,
                'weight': weight,
                'conf': confidence,
                'props': json.dumps(properties)
            })
            self.conn.commit()
            result = cur.fetchone()
            return result['id'] if result else None
    
    def get_or_create_node(self, term: str, node_type: str = 'Term', 
                          definition: str = None, properties: Dict = None) -> str:
        """ノードを取得、なければ作成"""
        node = self.get_node_by_term(term)
        if node:
            return node['id']
        return self.create_node(node_type, term, definition, properties)

# ============================================
# Graph Building Functions
# ============================================

def load_terms_from_json(filepath: str) -> List[Dict]:
    """JSONファイルから用語を読み込み"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Format check
    if 'terms' in data:
        return data['terms']
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Unknown JSON format in {filepath}")

def load_clustering_results(filepath: str) -> Dict:
    """クラスタリング結果を読み込み"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def build_nodes_from_terms(db: KnowledgeGraphDB, terms: List[Dict]) -> Dict[str, str]:
    """用語からノードを構築"""
    logger.info(f"Building nodes from {len(terms)} terms...")
    term_to_id = {}
    
    for term_data in terms:
        # Extract term information
        if isinstance(term_data, dict):
            term = term_data.get('headword', term_data.get('term', ''))
            definition = term_data.get('definition', '')
            synonyms = term_data.get('synonyms', term_data.get('aliases', []))
            
            # Properties
            properties = {
                'synonyms': synonyms,
                'c_value': term_data.get('c_value', 0),
                'frequency': term_data.get('frequency', 0)
            }
            
            # Create main term node
            node_id = db.create_node('Term', term, definition, properties)
            term_to_id[term] = node_id
            
            # Create synonym relationships
            for synonym in synonyms:
                if synonym and synonym != term:
                    syn_id = db.get_or_create_node(synonym, 'Term')
                    db.create_edge(syn_id, node_id, 'SYNONYM', 
                                 weight=0.95, confidence=0.9, 
                                 provenance='definition')
    
    logger.info(f"Created {len(term_to_id)} term nodes")
    return term_to_id

def build_category_nodes_from_clusters(db: KnowledgeGraphDB, 
                                      clustering_results: Dict) -> Dict[str, str]:
    """クラスタリング結果からカテゴリノードを構築"""
    logger.info("Building category nodes from clustering results...")
    category_to_id = {}
    
    if 'categories' not in clustering_results:
        logger.warning("No categories found in clustering results")
        return category_to_id
    
    for category_name, category_data in clustering_results['categories'].items():
        if category_name == '未分類':
            continue
        
        properties = {
            'cluster_id': category_data.get('cluster_id', -1),
            'member_count': category_data.get('count', 0)
        }
        
        # Create category node
        cat_id = db.create_node('Category', category_name, 
                              f"クラスタ{category_data.get('cluster_id')}のカテゴリ",
                              properties)
        category_to_id[category_name] = cat_id
    
    logger.info(f"Created {len(category_to_id)} category nodes")
    return category_to_id

def build_hierarchy_from_clustering(db: KnowledgeGraphDB, 
                                   clustering_results: Dict,
                                   term_to_id: Dict[str, str]) -> int:
    """クラスタリング結果から階層関係を構築"""
    logger.info("Building hierarchy from clustering results...")
    edge_count = 0
    
    # Extract hierarchy information
    if 'hierarchy' not in clustering_results:
        logger.warning("No hierarchy information in clustering results")
        return edge_count
    
    hierarchy = clustering_results['hierarchy']
    
    # Build IS_A relationships based on lambda values
    if 'term_hierarchy' in hierarchy:
        term_hierarchy = hierarchy['term_hierarchy']
        
        # Sort by lambda value (lower = more general)
        sorted_terms = sorted(term_hierarchy, key=lambda x: x.get('lambda', 999))
        
        # Create IS_A relationships based on lambda difference
        for i, term1 in enumerate(sorted_terms):
            term1_name = term1.get('term')
            if term1_name not in term_to_id:
                continue
            
            # Find potential parent (more general term)
            for j in range(i):
                term2 = sorted_terms[j]
                term2_name = term2.get('term')
                
                if term2_name not in term_to_id:
                    continue
                
                # Check lambda difference
                lambda_diff = term1['lambda'] - term2['lambda']
                
                # Create IS_A if significant difference
                if 0.1 < lambda_diff < 0.5:
                    db.create_edge(
                        term_to_id[term1_name],
                        term_to_id[term2_name],
                        'IS_A',
                        weight=max(0.5, 1.0 - lambda_diff),
                        confidence=0.7,
                        provenance='hierarchy',
                        evidence=f"Lambda difference: {lambda_diff:.3f}"
                    )
                    edge_count += 1
    
    logger.info(f"Created {edge_count} hierarchy edges")
    return edge_count

def build_similarity_from_clusters(db: KnowledgeGraphDB,
                                  clustering_results: Dict,
                                  term_to_id: Dict[str, str]) -> int:
    """同一クラスタから類似関係を構築"""
    logger.info("Building similarity relationships from clusters...")
    edge_count = 0
    
    if 'categories' not in clustering_results:
        return edge_count
    
    for category_name, category_data in clustering_results['categories'].items():
        if category_name == '未分類':
            continue
        
        terms = category_data.get('terms', [])
        
        # Create SIMILAR_TO relationships within cluster
        for i, term1 in enumerate(terms):
            if term1 not in term_to_id:
                continue
            
            for term2 in terms[i+1:]:
                if term2 not in term_to_id:
                    continue
                
                # Bidirectional similarity
                db.create_edge(
                    term_to_id[term1],
                    term_to_id[term2],
                    'SIMILAR_TO',
                    weight=0.8,
                    confidence=0.75,
                    provenance='clustering',
                    evidence=f"Same cluster: {category_name}"
                )
                edge_count += 1
    
    logger.info(f"Created {edge_count} similarity edges")
    return edge_count

def build_term_category_relationships(db: KnowledgeGraphDB,
                                     clustering_results: Dict,
                                     term_to_id: Dict[str, str],
                                     category_to_id: Dict[str, str]) -> int:
    """用語とカテゴリの関係を構築"""
    logger.info("Building term-category relationships...")
    edge_count = 0
    
    for category_name, category_data in clustering_results['categories'].items():
        if category_name == '未分類' or category_name not in category_to_id:
            continue
        
        cat_id = category_to_id[category_name]
        
        for term in category_data.get('terms', []):
            if term in term_to_id:
                db.create_edge(
                    term_to_id[term],
                    cat_id,
                    'BELONGS_TO',
                    weight=0.9,
                    confidence=0.95,
                    provenance='clustering'
                )
                edge_count += 1
    
    logger.info(f"Created {edge_count} term-category edges")
    return edge_count

# ============================================
# Main Function
# ============================================

def main():
    """メイン処理"""
    # File paths
    terms_file = Path("output/terms_100.json")
    clustering_file = Path("output/term_clusters.json")
    
    if not terms_file.exists():
        logger.error(f"Terms file not found: {terms_file}")
        return
    
    if not clustering_file.exists():
        logger.error(f"Clustering file not found: {clustering_file}")
        return
    
    # Load data
    terms = load_terms_from_json(str(terms_file))
    clustering_results = load_clustering_results(str(clustering_file))
    
    logger.info(f"Loaded {len(terms)} terms")
    logger.info(f"Loaded clustering results with {len(clustering_results.get('categories', {}))} categories")
    
    # Build graph
    with KnowledgeGraphDB(PG_URL) as db:
        # 1. Create term nodes
        term_to_id = build_nodes_from_terms(db, terms)
        
        # 2. Create category nodes
        category_to_id = build_category_nodes_from_clusters(db, clustering_results)
        
        # 3. Build hierarchy from clustering
        hierarchy_edges = build_hierarchy_from_clustering(db, clustering_results, term_to_id)
        
        # 4. Build similarity relationships
        similarity_edges = build_similarity_from_clusters(db, clustering_results, term_to_id)
        
        # 5. Build term-category relationships
        category_edges = build_term_category_relationships(
            db, clustering_results, term_to_id, category_to_id
        )
        
        # Summary
        logger.info("=" * 50)
        logger.info("Graph Building Complete!")
        logger.info(f"Created {len(term_to_id)} term nodes")
        logger.info(f"Created {len(category_to_id)} category nodes")
        logger.info(f"Created {hierarchy_edges} hierarchy edges")
        logger.info(f"Created {similarity_edges} similarity edges")
        logger.info(f"Created {category_edges} term-category edges")
        logger.info(f"Total edges: {hierarchy_edges + similarity_edges + category_edges}")
        logger.info("=" * 50)

if __name__ == "__main__":
    main()