#!/usr/bin/env python3
"""
Query Expansion using Knowledge Graph
Expands search queries using term relationships from the knowledge graph
"""

import json
import logging
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
import psycopg
from psycopg.rows import dict_row
from dotenv import load_dotenv
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from rag.config import Config

# ============================================
# Configuration
# ============================================
load_dotenv()
cfg = Config()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================
# Data Classes
# ============================================

@dataclass
class ExpandedTerm:
    """Expanded term with weight and relationship"""
    term: str
    weight: float
    relationship: str
    distance: int  # Graph distance from original term

@dataclass
class QueryExpansion:
    """Result of query expansion"""
    original_query: str
    expanded_terms: List[ExpandedTerm]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'original_query': self.original_query,
            'expanded_terms': [
                {
                    'term': t.term,
                    'weight': t.weight,
                    'relationship': t.relationship,
                    'distance': t.distance
                }
                for t in self.expanded_terms
            ]
        }
    
    def get_weighted_terms(self, min_weight: float = 0.0) -> List[Tuple[str, float]]:
        """Get terms with weights above threshold"""
        return [
            (t.term, t.weight) 
            for t in self.expanded_terms 
            if t.weight >= min_weight
        ]

# ============================================
# Query Expander Class
# ============================================

class KnowledgeGraphQueryExpander:
    """Query expander using knowledge graph relationships"""
    
    def __init__(self, connection_string: str = None):
        """Initialize with database connection"""
        if connection_string is None:
            connection_string = (
                f"host={cfg.db_host} port={cfg.db_port} "
                f"dbname={cfg.db_name} user={cfg.db_user} password={cfg.db_password}"
            )
        self.connection_string = connection_string
        
        # Weight decay factors for different relationships
        self.relationship_weights = {
            'SYNONYM': 0.95,
            'IS_A': 0.8,
            'PART_OF': 0.7,
            'SIMILAR_TO': 0.6,
            'BELONGS_TO': 0.5,
            'RELATED_TO': 0.4,
            'CO_OCCURS_WITH': 0.3
        }
        
        # Distance decay factor
        self.distance_decay = 0.7
    
    def expand_query(self, 
                    query_term: str, 
                    max_depth: int = 2,
                    min_weight: float = 0.3,
                    max_terms: int = 20) -> QueryExpansion:
        """
        Expand a query term using knowledge graph relationships
        
        Args:
            query_term: Term to expand
            max_depth: Maximum graph traversal depth
            min_weight: Minimum weight threshold for included terms
            max_terms: Maximum number of expanded terms
            
        Returns:
            QueryExpansion object with expanded terms
        """
        expanded_terms = []
        visited = set()
        
        with psycopg.connect(self.connection_string, row_factory=dict_row) as conn:
            # Find the query term node
            node = self._get_node_by_term(conn, query_term)
            if not node:
                logger.warning(f"Term '{query_term}' not found in knowledge graph")
                return QueryExpansion(query_term, [])
            
            # Add original term
            expanded_terms.append(
                ExpandedTerm(query_term, 1.0, 'ORIGINAL', 0)
            )
            visited.add(node['id'])
            
            # BFS traversal
            current_level = [(node['id'], 1.0, 'ORIGINAL', 0)]
            
            for depth in range(1, max_depth + 1):
                next_level = []
                
                for node_id, current_weight, _, _ in current_level:
                    # Get connected nodes
                    edges = self._get_edges_for_node(conn, node_id)
                    
                    for edge in edges:
                        # Determine target node
                        if edge['direction'] == 'outgoing':
                            target_id = edge['target_id']
                        else:
                            target_id = edge['source_id']
                        
                        # Skip if already visited
                        if target_id in visited:
                            continue
                        
                        # Get target node details
                        target_node = self._get_node_by_id(conn, target_id)
                        if not target_node or target_node['node_type'] != 'Term':
                            continue
                        
                        # Calculate weight
                        rel_weight = self.relationship_weights.get(
                            edge['edge_type'], 0.2
                        )
                        edge_weight = edge['weight'] * edge['confidence']
                        distance_factor = self.distance_decay ** depth
                        
                        final_weight = (
                            current_weight * rel_weight * 
                            edge_weight * distance_factor
                        )
                        
                        # Add if above threshold
                        if final_weight >= min_weight:
                            expanded_terms.append(
                                ExpandedTerm(
                                    target_node['term'],
                                    final_weight,
                                    edge['edge_type'],
                                    depth
                                )
                            )
                            next_level.append(
                                (target_id, final_weight, edge['edge_type'], depth)
                            )
                            visited.add(target_id)
                
                current_level = next_level
                
                # Stop if we have enough terms
                if len(expanded_terms) >= max_terms:
                    break
        
        # Sort by weight and limit
        expanded_terms.sort(key=lambda x: x.weight, reverse=True)
        expanded_terms = expanded_terms[:max_terms]
        
        return QueryExpansion(query_term, expanded_terms)
    
    def expand_multiple_terms(self, 
                             query_terms: List[str],
                             **kwargs) -> Dict[str, QueryExpansion]:
        """
        Expand multiple query terms
        
        Args:
            query_terms: List of terms to expand
            **kwargs: Arguments for expand_query
            
        Returns:
            Dictionary mapping terms to their expansions
        """
        expansions = {}
        for term in query_terms:
            expansions[term] = self.expand_query(term, **kwargs)
        return expansions
    
    def merge_expansions(self, 
                        expansions: List[QueryExpansion],
                        aggregation: str = 'max') -> List[Tuple[str, float]]:
        """
        Merge multiple query expansions
        
        Args:
            expansions: List of QueryExpansion objects
            aggregation: How to aggregate weights ('max', 'mean', 'sum')
            
        Returns:
            List of (term, weight) tuples
        """
        term_weights = {}
        
        for expansion in expansions:
            for term in expansion.expanded_terms:
                if term.term not in term_weights:
                    term_weights[term.term] = []
                term_weights[term.term].append(term.weight)
        
        # Aggregate weights
        result = []
        for term, weights in term_weights.items():
            if aggregation == 'max':
                weight = max(weights)
            elif aggregation == 'mean':
                weight = sum(weights) / len(weights)
            elif aggregation == 'sum':
                weight = min(sum(weights), 1.0)  # Cap at 1.0
            else:
                weight = max(weights)
            
            result.append((term, weight))
        
        # Sort by weight
        result.sort(key=lambda x: x[1], reverse=True)
        return result
    
    # ============================================
    # Database Helper Methods
    # ============================================
    
    def _get_node_by_term(self, conn, term: str) -> Optional[Dict]:
        """Get node by term"""
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM knowledge_nodes WHERE term = %s LIMIT 1",
                (term,)
            )
            return cur.fetchone()
    
    def _get_node_by_id(self, conn, node_id: str) -> Optional[Dict]:
        """Get node by ID"""
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM knowledge_nodes WHERE id = %s",
                (node_id,)
            )
            return cur.fetchone()
    
    def _get_edges_for_node(self, conn, node_id: str) -> List[Dict]:
        """Get all edges connected to a node"""
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    e.*,
                    'outgoing' as direction
                FROM knowledge_edges e
                WHERE e.source_id = %s
                
                UNION ALL
                
                SELECT 
                    e.*,
                    'incoming' as direction
                FROM knowledge_edges e
                WHERE e.target_id = %s
            """, (node_id, node_id))
            return cur.fetchall()

# ============================================
# API Functions
# ============================================

def expand_query_api(query: str, 
                    max_depth: int = 2,
                    min_weight: float = 0.3,
                    max_terms: int = 20) -> Dict:
    """
    API function for query expansion
    
    Args:
        query: Query string (can contain multiple terms)
        max_depth: Maximum graph traversal depth
        min_weight: Minimum weight threshold
        max_terms: Maximum expanded terms per query term
        
    Returns:
        Dictionary with expanded terms and weights
    """
    expander = KnowledgeGraphQueryExpander()
    
    # Split query into terms
    terms = [t.strip() for t in query.split() if t.strip()]
    
    if not terms:
        return {'error': 'No query terms provided'}
    
    # Expand each term
    expansions = []
    for term in terms:
        expansion = expander.expand_query(
            term, 
            max_depth=max_depth,
            min_weight=min_weight,
            max_terms=max_terms
        )
        expansions.append(expansion)
    
    # Merge expansions
    merged = expander.merge_expansions(expansions, aggregation='max')
    
    return {
        'original_query': query,
        'query_terms': terms,
        'expanded_terms': [
            {'term': term, 'weight': weight}
            for term, weight in merged
        ],
        'total_terms': len(merged)
    }

# ============================================
# Main Function
# ============================================

def main():
    """Test query expansion"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Expand query using knowledge graph')
    parser.add_argument('query', help='Query term(s) to expand')
    parser.add_argument('--depth', type=int, default=2, help='Maximum traversal depth')
    parser.add_argument('--min-weight', type=float, default=0.3, help='Minimum weight threshold')
    parser.add_argument('--max-terms', type=int, default=20, help='Maximum expanded terms')
    parser.add_argument('--format', choices=['json', 'text'], default='text', help='Output format')
    
    args = parser.parse_args()
    
    # Expand query
    result = expand_query_api(
        args.query,
        max_depth=args.depth,
        min_weight=args.min_weight,
        max_terms=args.max_terms
    )
    
    # Output results
    if args.format == 'json':
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"\n原クエリ: {result['original_query']}")
        print(f"拡張用語数: {result['total_terms']}")
        print("\n拡張結果:")
        print("-" * 50)
        
        for item in result['expanded_terms']:
            term = item['term']
            weight = item['weight']
            bar = '█' * int(weight * 20)
            print(f"{term:30} {weight:.3f} {bar}")

if __name__ == "__main__":
    main()