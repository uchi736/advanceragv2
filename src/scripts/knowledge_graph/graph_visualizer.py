#!/usr/bin/env python3
"""
Knowledge Graph Visualizer
ナレッジグラフをインタラクティブに可視化
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
import psycopg
from psycopg.rows import dict_row
from pyvis.network import Network
from pyvis.options import Layout
import streamlit as st
import streamlit.components.v1 as components
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).resolve().parents[2]))
from rag.config import Config
from dotenv import load_dotenv

# Configuration
load_dotenv()
cfg = Config()
PG_URL = f"host={cfg.db_host} port={cfg.db_port} dbname={cfg.db_name} user={cfg.db_user} password={cfg.db_password}"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================
# Visual Settings
# ============================================

# ノードの色設定
NODE_COLORS = {
    'Term': '#4a90e2',       # 青（専門用語）
    'Category': '#e74c3c',    # 赤（カテゴリ）
    'Domain': '#27ae60',      # 緑（ドメイン）
    'System': '#9b59b6',      # 紫（システム）
    'Component': '#f39c12'    # 黄（コンポーネント）
}

# エッジの色設定
EDGE_COLORS = {
    # 階層関係
    'IS_A': '#7f8c8d',           # グレー
    'HAS_SUBTYPE': '#95a5a6',    # 薄グレー
    'BELONGS_TO': '#e91e63',     # ピンク
    
    # 構成関係  
    'PART_OF': '#3498db',        # 青
    'HAS_COMPONENT': '#2980b9',  # 濃青
    'INCLUDES': '#1abc9c',       # ターコイズ
    
    # 機能関係
    'USED_FOR': '#f39c12',       # オレンジ
    'CONTROLS': '#e67e22',       # 濃オレンジ
    'MEASURES': '#d35400',       # 茶
    'PERFORMS': '#c0392b',       # 濃赤
    
    # 関連関係
    'SIMILAR_TO': '#27ae60',     # 緑
    'SYNONYM': '#16a085',        # 濃緑
    'RELATED_TO': '#8e44ad',     # 紫
    'CO_OCCURS_WITH': '#2ecc71', # 薄緑
    
    # その他
    'DEPENDS_ON': '#34495e',     # 濃グレー
    'CAUSES': '#c0392b',         # 赤
    'PREVENTS': '#27ae60',       # 緑
    'GENERATES': '#f1c40f'       # 黄
}

# ============================================
# Database Functions
# ============================================

def get_subgraph_from_db(center_term: str, depth: int = 2, 
                         edge_types: Optional[List[str]] = None) -> Dict:
    """
    指定用語を中心としたサブグラフをDBから取得
    """
    with psycopg.connect(PG_URL, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            # 中心ノードを取得
            cur.execute("""
                SELECT id, node_type, term, definition, properties
                FROM knowledge_nodes
                WHERE term = %s
                LIMIT 1
            """, (center_term,))
            
            center_node = cur.fetchone()
            if not center_node:
                return {'nodes': [], 'edges': []}
            
            center_id = center_node['id']
            
            # BFSでサブグラフのノードIDを取得
            # edge_typesの有無で条件分岐
            if edge_types:
                cur.execute("""
                    WITH RECURSIVE bfs AS (
                        SELECT id, 0 as depth
                        FROM knowledge_nodes
                        WHERE id = %s
                        
                        UNION
                        
                        SELECT 
                            CASE 
                                WHEN e.source_id = b.id THEN e.target_id
                                ELSE e.source_id
                            END as id,
                            b.depth + 1
                        FROM bfs b
                        JOIN knowledge_edges e ON (e.source_id = b.id OR e.target_id = b.id)
                        WHERE b.depth < %s
                        AND e.edge_type = ANY(%s)
                    )
                    SELECT DISTINCT id, MIN(depth) as depth
                    FROM bfs
                    GROUP BY id
                """, (center_id, depth, edge_types))
            else:
                cur.execute("""
                    WITH RECURSIVE bfs AS (
                        SELECT id, 0 as depth
                        FROM knowledge_nodes
                        WHERE id = %s
                        
                        UNION
                        
                        SELECT 
                            CASE 
                                WHEN e.source_id = b.id THEN e.target_id
                                ELSE e.source_id
                            END as id,
                            b.depth + 1
                        FROM bfs b
                        JOIN knowledge_edges e ON (e.source_id = b.id OR e.target_id = b.id)
                        WHERE b.depth < %s
                    )
                    SELECT DISTINCT id, MIN(depth) as depth
                    FROM bfs
                    GROUP BY id
                """, (center_id, depth))
            
            node_ids = [(row['id'], row['depth']) for row in cur.fetchall()]
            node_id_list = [nid for nid, _ in node_ids]
            depth_map = {nid: d for nid, d in node_ids}
            
            # ノード情報を取得
            cur.execute("""
                SELECT id, node_type, term, definition, properties
                FROM knowledge_nodes
                WHERE id = ANY(%s)
            """, (node_id_list,))
            
            nodes = []
            for row in cur.fetchall():
                node = dict(row)
                node['depth'] = depth_map.get(node['id'], 0)
                node['is_center'] = (node['id'] == center_id)
                nodes.append(node)
            
            # エッジ情報を取得
            if edge_types:
                cur.execute("""
                    SELECT 
                        e.id, e.source_id, e.target_id, e.edge_type, 
                        e.weight, e.confidence, e.properties,
                        n1.term as source_term, n2.term as target_term
                    FROM knowledge_edges e
                    JOIN knowledge_nodes n1 ON e.source_id = n1.id
                    JOIN knowledge_nodes n2 ON e.target_id = n2.id
                    WHERE e.source_id = ANY(%s) AND e.target_id = ANY(%s)
                    AND e.edge_type = ANY(%s)
                """, (node_id_list, node_id_list, edge_types))
            else:
                cur.execute("""
                    SELECT 
                        e.id, e.source_id, e.target_id, e.edge_type, 
                        e.weight, e.confidence, e.properties,
                        n1.term as source_term, n2.term as target_term
                    FROM knowledge_edges e
                    JOIN knowledge_nodes n1 ON e.source_id = n1.id
                    JOIN knowledge_nodes n2 ON e.target_id = n2.id
                    WHERE e.source_id = ANY(%s) AND e.target_id = ANY(%s)
                """, (node_id_list, node_id_list))
            
            edges = [dict(row) for row in cur.fetchall()]
            
            return {'nodes': nodes, 'edges': edges}

def apply_subgraph_filters(
    subgraph: Dict,
    edge_types: Optional[List[str]] = None,
    min_weight: Optional[float] = None,
    node_types: Optional[List[str]] = None,
    hide_isolated: bool = False,
    include_terms: Optional[List[str]] = None,
    exclude_terms: Optional[List[str]] = None,
) -> Dict:
    """指定の条件でサブグラフを絞り込み"""
    nodes = list(subgraph.get('nodes', []))
    edges = list(subgraph.get('edges', []))

    # エッジタイプ
    if edge_types:
        edge_type_set: Set[str] = set(edge_types)
        edges = [e for e in edges if e.get('edge_type') in edge_type_set]

    # 重みの下限
    if min_weight is not None:
        try:
            thr = float(min_weight)
        except Exception:
            thr = None
        if thr is not None:
            edges = [e for e in edges if (e.get('weight') or 0.0) >= thr]

    # ノードタイプ
    if node_types:
        node_type_set: Set[str] = set(node_types)
        nodes = [n for n in nodes if n.get('node_type') in node_type_set]

    # ラベル（用語名）での包含/除外
    if include_terms:
        inc = [s.lower() for s in include_terms if s]
        if inc:
            nodes = [n for n in nodes if any(k in str(n.get('term','')).lower() for k in inc)]
    if exclude_terms:
        exc = [s.lower() for s in exclude_terms if s]
        if exc:
            nodes = [n for n in nodes if not any(k in str(n.get('term','')).lower() for k in exc)]

    # 孤立ノード除去（中心ノードは常に残す）
    present_ids = {e['source_id'] for e in edges} | {e['target_id'] for e in edges}
    center_ids = {n['id'] for n in subgraph.get('nodes', []) if n.get('is_center')}
    if hide_isolated:
        nodes = [n for n in nodes if (n['id'] in present_ids) or (n['id'] in center_ids)]

    # ノード側で落ちたものに合わせてエッジを再度整合
    valid_ids = {n['id'] for n in nodes}
    edges = [e for e in edges if e['source_id'] in valid_ids and e['target_id'] in valid_ids]

    return {'nodes': nodes, 'edges': edges}

def get_all_terms() -> List[str]:
    """全ての用語を取得"""
    with psycopg.connect(PG_URL, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT term FROM knowledge_nodes 
                WHERE node_type = 'Term'
                ORDER BY term
            """)
            return [row['term'] for row in cur.fetchall()]

def get_graph_statistics() -> Dict:
    """グラフの統計情報を取得"""
    with psycopg.connect(PG_URL, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    (SELECT COUNT(*) FROM knowledge_nodes) as total_nodes,
                    (SELECT COUNT(*) FROM knowledge_nodes WHERE node_type = 'Term') as term_nodes,
                    (SELECT COUNT(*) FROM knowledge_edges) as total_edges,
                    (SELECT COUNT(DISTINCT edge_type) FROM knowledge_edges) as edge_types,
                    (SELECT AVG(weight) FROM knowledge_edges) as avg_weight
            """)
            return dict(cur.fetchone())

def get_global_subgraph(limit_edges: int = 200,
                        edge_types: Optional[List[str]] = None,
                        min_weight: Optional[float] = None) -> Dict:
    """全体ビュー用のサブグラフ（中心なし）。重み順で上位エッジを取得"""
    with psycopg.connect(PG_URL, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            # エッジ取得（重み順）
            where_clauses = []
            params: List[Any] = []
            if edge_types:
                where_clauses.append("e.edge_type = ANY(%s)")
                params.append(edge_types)
            if min_weight is not None:
                where_clauses.append("e.weight >= %s")
                params.append(min_weight)
            where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

            sql = f"""
                SELECT 
                    e.id, e.source_id, e.target_id, e.edge_type,
                    e.weight, e.confidence, e.properties,
                    n1.term as source_term, n2.term as target_term
                FROM knowledge_edges e
                JOIN knowledge_nodes n1 ON e.source_id = n1.id
                JOIN knowledge_nodes n2 ON e.target_id = n2.id
                {where_sql}
                ORDER BY e.weight DESC NULLS LAST, e.id
                LIMIT %s
            """
            params2 = params + [limit_edges]
            cur.execute(sql, params2)
            edges = [dict(row) for row in cur.fetchall()]

            if not edges:
                return {'nodes': [], 'edges': []}

            node_ids: Set[Any] = set()
            for e in edges:
                node_ids.add(e['source_id'])
                node_ids.add(e['target_id'])

            cur.execute(
                """
                SELECT id, node_type, term, definition, properties
                FROM knowledge_nodes
                WHERE id = ANY(%s)
                """,
                (list(node_ids),)
            )
            nodes = [dict(row) for row in cur.fetchall()]
            # 中心なし
            for n in nodes:
                n['is_center'] = False
            return {'nodes': nodes, 'edges': edges}

# ============================================
# Visualization Functions
# ============================================

def create_pyvis_network(subgraph: Dict, 
                        layout: str = 'hierarchical',
                        physics: bool = True) -> Network:
    """
    PyvisネットワークオブジェクトをY作成
    """
    # ネットワーク初期化
    net = Network(
        height="750px", 
        width="100%",
        bgcolor="#ffffff",
        font_color="#333333",
        directed=True,
        notebook=False,
        cdn_resources='in_line'  # 自己完結型HTML
    )
    
    # レイアウト/物理設定（set_optionsは使わずオブジェクトで指定）
    if layout == 'hierarchical':
        # レイアウト
        net.options.layout = Layout()
        net.options.layout.hierarchical.enabled = True
        net.options.layout.hierarchical.sortMethod = 'directed'
        net.options.layout.hierarchical.levelSeparation = 200
        # Vis.jsの拡張キーも追加（Optionsに未定義でもJSON化される）
        net.options.layout.hierarchical.direction = 'UD'
        # 物理
        net.options.physics.enabled = bool(physics)
        net.options.physics.use_hrepulsion({
            'node_distance': 150,
            'central_gravity': 0.3,
            'spring_length': 200,
            'spring_strength': 0.001,
            'damping': 0.09
        })
    else:
        net.options.physics.enabled = bool(physics)
        if physics:
            net.barnes_hut(
                gravity=-80000,
                central_gravity=0.3,
                spring_length=200,
                spring_strength=0.001,
                damping=0.09
            )
        # 物理OFF時はenabledがFalseのままでOK
    
    # ノード追加
    for node in subgraph['nodes']:
        # サイズ設定（中心ノードは大きく）
        if node.get('is_center'):
            size = 35
            border_width = 3
        else:
            depth = node.get('depth', None)
            if depth is None:
                size = 20
            else:
                size = 20 + max(0, (3 - depth)) * 5
            border_width = 1
        
        # ラベル作成
        label = node['term']
        if node['node_type'] == 'Category':
            label = f"[{label}]"
        
        # ツールチップ（ホバー時表示）
        title = f"<b>{node['term']}</b><br>"
        if node.get('definition'):
            title += f"{node['definition'][:200]}...<br>"
        title += f"<i>Type: {node['node_type']}</i><br>"
        title += f"<i>Depth: {node.get('depth', 0)}</i>"
        
        # ノード追加
        # PyVis requires node ids to be str or int. Ensure string IDs (e.g., for UUIDs).
        net.add_node(
            str(node['id']),
            label=label,
            title=title,
            color={
                'background': NODE_COLORS.get(node['node_type'], '#888888'),
                'border': '#2B2B2B' if node.get('is_center') else '#666666',
                'highlight': {
                    'background': '#FFA500',
                    'border': '#FF6600'
                }
            },
            size=size,
            borderWidth=border_width,
            font={'size': 14 if node.get('is_center') else 12},
            level=(node.get('depth', None) if layout == 'hierarchical' else None)
        )
    
    # エッジ追加
    for edge in subgraph['edges']:
        # ラベル（重みが高い場合のみ表示）
        label = edge['edge_type'] if edge['weight'] > 0.7 else ""
        
        # ツールチップ
        title = f"{edge['source_term']} → {edge['target_term']}<br>"
        title += f"Type: {edge['edge_type']}<br>"
        title += f"Weight: {edge['weight']:.2f}<br>"
        title += f"Confidence: {edge['confidence']:.2f}"
        
        # エッジ追加
        # Match edge endpoints to the stringified node ids
        net.add_edge(
            str(edge['source_id']),
            str(edge['target_id']),
            title=title,
            label=label,
            color={
                'color': EDGE_COLORS.get(edge['edge_type'], '#999999'),
                'opacity': min(0.3 + edge['weight'] * 0.7, 1.0)
            },
            width=1 + edge['weight'] * 2,
            arrows={
                'to': {
                    'enabled': True, 
                    'scaleFactor': 0.5
                }
            },
            smooth={
                'enabled': True,
                'type': 'curvedCW',
                'roundness': 0.1
            }
        )
    
    # インタラクション設定（オブジェクトに直接設定）
    net.options.interaction.hover = True
    net.options.interaction.tooltipDelay = 100
    net.options.interaction.navigationButtons = True
    net.options.interaction.keyboard = True
    net.options.manipulation = {'enabled': False}
    
    return net

def visualize_knowledge_graph(center_term: Optional[str] = None,
                            depth: int = 2,
                            edge_types: Optional[List[str]] = None,
                            layout: str = 'barnes_hut',
                            physics: bool = True,
                            min_weight: Optional[float] = None,
                            node_types: Optional[List[str]] = None,
                            hide_isolated: bool = True,
                            include_terms: Optional[List[str]] = None,
                            exclude_terms: Optional[List[str]] = None,
                            mode: str = 'centered',
                            limit_edges: int = 200) -> str:
    """
    ナレッジグラフを可視化してHTMLを返す
    """
    # サブグラフ取得
    if mode == 'global':
        subgraph = get_global_subgraph(limit_edges=limit_edges,
                                       edge_types=edge_types,
                                       min_weight=min_weight)
    else:
        if not center_term:
            return "<h3>Please select a center term</h3>"
        subgraph = get_subgraph_from_db(center_term, depth, edge_types)
    subgraph = apply_subgraph_filters(
        subgraph,
        edge_types=edge_types,
        min_weight=min_weight,
        node_types=node_types,
        hide_isolated=hide_isolated,
        include_terms=include_terms,
        exclude_terms=exclude_terms,
    )
    
    if not subgraph['nodes']:
        return "<h3>No data found for the specified term</h3>"
    
    # Pyvisネットワーク作成
    net = create_pyvis_network(subgraph, layout, physics)
    
    # HTML生成
    html = net.generate_html()
    
    # カスタムCSS追加
    custom_css = """
    <style>
        #mynetwork {
            border: 2px solid #ddd;
            border-radius: 5px;
        }
        .vis-navigation {
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 5px;
        }
    </style>
    """
    
    # HTMLに埋め込み
    html = html.replace('</head>', f'{custom_css}</head>')
    
    return html

def export_graph(subgraph: Dict, format: str = 'html') -> Any:
    """
    グラフを指定形式でエクスポート
    """
    if format == 'html':
        net = create_pyvis_network(subgraph)
        return net.generate_html()
    
    elif format == 'json':
        # Cytoscape.js形式
        return {
            'elements': {
                'nodes': [
                    {
                        'data': {
                            'id': str(node['id']),
                            'label': node['term'],
                            'type': node['node_type']
                        }
                    }
                    for node in subgraph['nodes']
                ],
                'edges': [
                    {
                        'data': {
                            'id': str(edge['id']),
                            'source': str(edge['source_id']),
                            'target': str(edge['target_id']),
                            'label': edge['edge_type'],
                            'weight': edge['weight']
                        }
                    }
                    for edge in subgraph['edges']
                ]
            }
        }
    
    elif format == 'dot':
        # Graphviz DOT形式
        lines = ['digraph KnowledgeGraph {']
        lines.append('  rankdir=TB;')
        lines.append('  node [shape=box, style=rounded];')
        
        # ノード
        for node in subgraph['nodes']:
            color = {
                'Term': 'lightblue',
                'Category': 'pink',
                'Domain': 'lightgreen'
            }.get(node['node_type'], 'white')
            
            lines.append(f'  "{node["term"]}" [fillcolor={color}, style="filled,rounded"];')
        
        # エッジ
        for edge in subgraph['edges']:
            style = 'solid' if edge['weight'] > 0.7 else 'dashed'
            lines.append(
                f'  "{edge["source_term"]}" -> "{edge["target_term"]}" '
                f'[label="{edge["edge_type"]}", style={style}];'
            )
        
        lines.append('}')
        return '\n'.join(lines)
    
    else:
        raise ValueError(f"Unsupported format: {format}")

# ============================================
# Streamlit Integration
# ============================================

def render_graph_explorer():
    """
    Streamlit用のグラフエクスプローラーUI
    """
    st.title("🕸️ ナレッジグラフ エクスプローラー")
    
    # サイドバー設定
    with st.sidebar:
        st.header("グラフ設定")
        
        # モード選択
        mode = st.radio(
            "モード",
            ["起点から探索", "全体ビュー"],
            help="起点＝中心語から深さで探索 / 全体＝重み上位の関係を俯瞰"
        )

        center_term = None
        if mode == "起点から探索":
            # 中心用語選択
            terms = get_all_terms()
            if not terms:
                st.warning("用語が登録されていません")
                return
            center_term = st.selectbox(
                "中心用語（起点）",
                terms,
                help="グラフの中心となる用語を選択"
            )
        
        # 探索深度 or 全体ビュー上限
        if mode == "起点から探索":
            depth = st.slider(
                "探索深度",
                min_value=1,
                max_value=4,
                value=3,
                help="中心用語からの探索深度"
            )
            edge_limit = None
        else:
            depth = 0
            edge_limit = st.slider(
                "最大エッジ数（全体ビュー）",
                min_value=50,
                max_value=1000,
                value=200,
                step=50,
                help="重みの高い順に取得するエッジ数の上限"
            )
        
        # エッジタイプフィルタ
        all_edge_types = [
            'IS_A', 'PART_OF', 'HAS_COMPONENT', 
            'SIMILAR_TO', 'RELATED_TO', 'BELONGS_TO',
            'USED_FOR', 'CONTROLS', 'MEASURES'
        ]
        
        selected_edges = st.multiselect(
            "表示する関係",
            all_edge_types,
            default=[],
            help="空のまま＝全関係を表示。必要な場合のみ絞り込み。"
        )
        
        # エッジ重み（下限）
        min_weight = st.slider(
            "エッジ重み（最小）",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05,
            help="この重み以上のエッジのみ表示"
        )

        # ノードタイプフィルタ
        all_node_types = list(NODE_COLORS.keys())
        selected_node_types = st.multiselect(
            "ノードタイプ",
            all_node_types,
            default=all_node_types,
            help="表示するノードタイプを選択"
        )

        # 孤立ノードの非表示
        hide_isolated = st.checkbox("孤立ノードを非表示", value=True)

        # ノード名の包含/除外
        include_kw_text = st.text_input(
            "ノード名に含めるキーワード（任意・カンマ区切り）",
            value="",
            help="指定した語を含むノードのみ表示。未入力なら制限なし"
        )
        exclude_kw_text = st.text_input(
            "ノード名の除外キーワード（カンマ区切り）",
            value="クラスタ,cluster",
            help="指定した語を含むノードを非表示にします（例: クラスタ, sample）"
        )
        include_terms = [s.strip() for s in include_kw_text.split(',') if s.strip()]
        exclude_terms = [s.strip() for s in exclude_kw_text.split(',') if s.strip()]

        # レイアウトは階層表示で固定
        layout = 'hierarchical'
        st.caption("レイアウト: 階層表示（固定）")

        # 物理シミュレーション（既定OFF）
        physics = st.checkbox("物理シミュレーション", value=False, help="重なり解消に有効ですが重くなる場合があります")
        
        # エクスポート
        st.divider()
        st.subheader("エクスポート")
        export_format = st.selectbox(
            "形式",
            ['html', 'json', 'dot']
        )
    
    # メインエリア
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        if st.button("🔍 グラフを生成", type="primary", use_container_width=True):
            st.session_state.generate_graph = True
    
    with col2:
        if st.button("🔄 リセット", use_container_width=True):
            st.session_state.generate_graph = False
            st.rerun()
    
    # グラフ生成・表示
    if st.session_state.get('generate_graph', False):
        with st.spinner("グラフを生成中..."):
            html = visualize_knowledge_graph(
                center_term if mode == "起点から探索" else None,
                depth,
                selected_edges if selected_edges else None,
                layout,
                physics,
                min_weight,
                selected_node_types if selected_node_types else None,
                hide_isolated,
                include_terms if include_terms else None,
                exclude_terms if exclude_terms else None,
                mode='centered' if mode == "起点から探索" else 'global',
                limit_edges=(edge_limit or 200)
            )
            
            # グラフ表示
            components.html(html, height=800, scrolling=True)
            
            # サブグラフ概要（デバッグ/確認用）
            if mode == "起点から探索":
                raw_subgraph = get_subgraph_from_db(center_term, depth, selected_edges if selected_edges else None)
            else:
                raw_subgraph = get_global_subgraph(
                    limit_edges=(edge_limit or 200),
                    edge_types=selected_edges if selected_edges else None,
                    min_weight=min_weight,
                )
            subgraph = apply_subgraph_filters(
                raw_subgraph,
                edge_types=selected_edges if selected_edges else None,
                min_weight=min_weight,
                node_types=selected_node_types if selected_node_types else None,
                hide_isolated=hide_isolated,
                include_terms=include_terms if include_terms else None,
                exclude_terms=exclude_terms if exclude_terms else None,
            )
            node_count = len(subgraph.get('nodes', []))
            edge_count = len(subgraph.get('edges', []))
            st.caption(f"サブグラフ: ノード {node_count} / エッジ {edge_count}")
            if edge_count == 0:
                st.info("選択中の関係タイプや探索深度ではエッジが見つかりません。関係を増やす/全解除、深度を上げる、別の用語をお試しください。")
            
            # エクスポートボタン
            with col3:
                if mode == "起点から探索":
                    raw_subgraph = get_subgraph_from_db(center_term, depth, selected_edges)
                else:
                    raw_subgraph = get_global_subgraph(
                        limit_edges=(edge_limit or 200),
                        edge_types=selected_edges if selected_edges else None,
                        min_weight=min_weight,
                    )
                filtered = apply_subgraph_filters(
                    raw_subgraph,
                    edge_types=selected_edges if selected_edges else None,
                    min_weight=min_weight,
                    node_types=selected_node_types if selected_node_types else None,
                    hide_isolated=hide_isolated,
                    include_terms=include_terms if include_terms else None,
                    exclude_terms=exclude_terms if exclude_terms else None,
                )
                exported = export_graph(filtered, export_format)
                
                if export_format == 'html':
                    st.download_button(
                        "📥 ダウンロード",
                        exported,
                        file_name=f"graph_{center_term}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html",
                        use_container_width=True
                    )
                elif export_format == 'json':
                    st.download_button(
                        "📥 ダウンロード",
                        json.dumps(exported, ensure_ascii=False, indent=2),
                        file_name=f"graph_{center_term}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                elif export_format == 'dot':
                    st.download_button(
                        "📥 ダウンロード",
                        exported,
                        file_name=f"graph_{center_term}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.dot",
                        mime="text/plain",
                        use_container_width=True
                    )
    
    # 統計情報
    with st.expander("📊 グラフ統計", expanded=False):
        stats = get_graph_statistics()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("総ノード数", f"{stats['total_nodes']:,}")
        with col2:
            st.metric("用語ノード数", f"{stats['term_nodes']:,}")
        with col3:
            st.metric("総エッジ数", f"{stats['total_edges']:,}")
        with col4:
            st.metric("平均重み", f"{stats['avg_weight']:.2f}" if stats['avg_weight'] else "N/A")

# ============================================
# Main
# ============================================

def main():
    """スタンドアロン実行用"""
    st.set_page_config(
        page_title="Knowledge Graph Explorer",
        page_icon="🕸️",
        layout="wide"
    )
    
    render_graph_explorer()

if __name__ == "__main__":
    main()
