# ナレッジグラフ機能実装計画書 v2
*実装時の落とし穴を回避した実践版*

## 1. 概要

既存の専門用語辞書とクラスタリング結果を活用し、専門知識の関係性を可視化・活用するナレッジグラフシステムを構築する。

## 2. 改訂の要点

- pgvectorの適切な設定とHNSWインデックス
- 日本語形態素解析による高精度な関係抽出
- エッジ方向の一貫性とprovenance管理
- 推論の暴走防止と信頼度減衰
- 現実的な実装ロードマップ

## 3. データベース設計（改訂版）

### 3.1 必須拡張とスキーマ

```sql
-- 必須拡張
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ノードテーブル
CREATE TABLE knowledge_nodes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    node_type VARCHAR(50) NOT NULL CHECK (node_type IN ('Term', 'Category', 'Domain', 'Component', 'System')),
    term VARCHAR(255),
    definition TEXT,
    properties JSONB DEFAULT '{}',
    embedding vector(1536),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Term型のみにUNIQUE制約
CREATE UNIQUE INDEX uniq_term_only_term_nodes
ON knowledge_nodes (term)
WHERE node_type = 'Term';

-- HNSWインデックス（高精度・メモリ多）
CREATE INDEX idx_nodes_embedding_hnsw
ON knowledge_nodes
USING hnsw (embedding vector_l2_ops)
WITH (m = 16, ef_construction = 64);

-- その他のインデックス
CREATE INDEX idx_nodes_type ON knowledge_nodes(node_type);
CREATE INDEX idx_nodes_properties ON knowledge_nodes USING GIN(properties);

-- エッジテーブル
CREATE TABLE knowledge_edges (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id UUID REFERENCES knowledge_nodes(id) ON DELETE CASCADE,
    target_id UUID REFERENCES knowledge_nodes(id) ON DELETE CASCADE,
    edge_type VARCHAR(50) NOT NULL,
    weight FLOAT DEFAULT 1.0 CHECK (weight >= 0 AND weight <= 1),
    confidence FLOAT DEFAULT 1.0 CHECK (confidence >= 0 AND confidence <= 1),
    properties JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    CONSTRAINT no_self_loop CHECK (source_id != target_id),
    UNIQUE(source_id, target_id, edge_type)
);

-- エッジインデックス
CREATE INDEX idx_edges_source ON knowledge_edges(source_id);
CREATE INDEX idx_edges_target ON knowledge_edges(target_id);
CREATE INDEX idx_edges_type ON knowledge_edges(edge_type);
CREATE INDEX idx_edges_weight ON knowledge_edges(weight DESC);

-- updated_at自動更新トリガー
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS trigger AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_nodes_updated
BEFORE UPDATE ON knowledge_nodes
FOR EACH ROW EXECUTE FUNCTION set_updated_at();
```

### 3.2 provenance（出典）管理

```python
PROVENANCE_TYPES = {
    "clustering": "HDBSCANクラスタリング結果から",
    "definition": "定義文パターンマッチングから",
    "cooccurrence": "共起関係から",
    "llm": "LLMによる抽出",
    "inference": "推論による導出",
    "manual": "手動入力"
}
```

## 4. エッジタイプと方向性の厳密化

### 4.1 階層関係（下位→上位）

| エッジタイプ | 方向 | 例 |
|------------|------|-----|
| IS_A | 具体→抽象 | ピストン→エンジン部品 |
| BELONGS_TO | メンバー→カテゴリ | ピストン→エンジン部品カテゴリ |

### 4.2 構成関係（部分→全体）

| エッジタイプ | 方向 | 例 |
|------------|------|-----|
| PART_OF | 部分→全体 | ピストン→エンジン |
| HAS_COMPONENT | 全体→部分 | エンジン→ピストン |

### 4.3 機能関係（主体→対象）

| エッジタイプ | 方向 | 例 |
|------------|------|-----|
| USED_FOR | ツール→目的 | ターボチャージャー→過給 |
| CONTROLS | 制御主体→制御対象 | ガバナ→エンジン回転数 |
| MEASURES | 測定器→測定対象 | センサー→圧力 |

## 5. 日本語定義文からの関係抽出

### 5.1 形態素解析 + 係り受け解析

```python
import spacy
from sudachipy import Dictionary, SplitMode

# 初期化
nlp = spacy.load("ja_ginza")
sudachi = Dictionary().create()

# 日本語パターン（述語項構造）
JAPANESE_PATTERNS = [
    # 階層関係
    ("IS_A", [
        r"(?P<target>.+?)の一種(?:である|です|だ)",
        r"(?P<target>.+?)に分類される",
        r"(?P<target>.+?)の一つ"
    ]),
    
    # 構成関係
    ("PART_OF", [
        r"(?P<target>.+?)の(?:部品|部分|一部)(?:である|です)",
        r"(?P<target>.+?)に(?:含まれる|属する)",
        r"(?P<target>.+?)を構成する"
    ]),
    
    # 機能関係
    ("USED_FOR", [
        r"(?P<target>.+?)(?:に|のために)(?:使用|利用|用いら)れる",
        r"(?P<target>.+?)を(?:目的と|用途と)する"
    ]),
    
    ("CONTROLS", [
        r"(?P<target>.+?)を(?:制御|調整|管理)する",
        r"(?P<target>.+?)の(?:制御|調整)を行う"
    ]),
    
    ("MEASURES", [
        r"(?P<target>.+?)を(?:測定|計測|検出)する",
        r"(?P<target>.+?)の(?:測定|計測)を行う"
    ]),
    
    ("PREVENTS", [
        r"(?P<target>.+?)を(?:防止|防ぐ|抑制)する",
        r"(?P<target>.+?)の(?:防止|抑制)を行う"
    ])
]

def extract_relations_from_japanese(term, definition, term_vocabulary):
    """日本語定義文から関係を抽出"""
    relations = []
    doc = nlp(definition)
    
    # 1. パターンマッチング
    for edge_type, patterns in JAPANESE_PATTERNS:
        for pattern in patterns:
            matches = re.finditer(pattern, definition)
            for match in matches:
                if 'target' in match.groupdict():
                    target = match.group('target')
                    # 正規化と辞書チェック
                    normalized = normalize_term(target)
                    if normalized in term_vocabulary and normalized != term:
                        relations.append({
                            'source': term,
                            'target': normalized,
                            'type': edge_type,
                            'confidence': 0.8,
                            'provenance': 'definition',
                            'evidence': definition[max(0, match.start()-20):min(len(definition), match.end()+20)]
                        })
    
    # 2. 係り受け解析による補完
    for token in doc:
        if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
            # 主語-動詞関係から推定
            verb = token.head.lemma_
            if "制御" in verb:
                # token（主語）が制御する対象を探す
                for obj in token.head.children:
                    if obj.dep_ == "dobj" and obj.text in term_vocabulary:
                        relations.append({
                            'source': term,
                            'target': obj.text,
                            'type': 'CONTROLS',
                            'confidence': 0.7,
                            'provenance': 'definition'
                        })
    
    return relations

def normalize_term(text):
    """用語の正規化（表記ゆれ対応）"""
    # Sudachiで正規化形を取得
    tokens = sudachi.tokenize(text, SplitMode.C)
    normalized = ''.join([t.normalized_form() for t in tokens])
    return normalized
```

### 5.2 略語展開テーブル

```python
# 製造業固有の略語辞書
ABBREVIATION_DICT = {
    "VIT": "Variable Injection Timing",
    "CPP": "Controllable Pitch Propeller",
    "FGSS": "Fuel Gas Supply System",
    "EGR": "Exhaust Gas Recirculation",
    "SCR": "Selective Catalytic Reduction",
    "DPF": "Diesel Particulate Filter",
    "EEDI": "Energy Efficiency Design Index",
    "BOG": "Boil Off Gas",
    # ... 業界固有の略語を追加
}

def expand_abbreviations(text):
    """略語を展開して検索性を向上"""
    for abbr, full in ABBREVIATION_DICT.items():
        if abbr in text:
            # 略語と展開形の両方を保持
            text = f"{text} ({full})"
    return text
```

## 6. 階層推定の改善（λ値の誤解を修正）

### 6.1 C-valueベースの一般性推定

```python
def estimate_generality(term_data):
    """
    用語の一般性を推定（階層の上位ほど高い）
    λ値ではなく、C-value、頻度、文脈エントロピーを使用
    """
    score = 0.0
    
    # 1. C-value（低いほど一般的）
    if term_data['c_value'] < 10:
        score += 0.3
    elif term_data['c_value'] < 50:
        score += 0.2
    else:
        score += 0.1
    
    # 2. 出現頻度（高いほど一般的）
    freq_normalized = min(term_data['frequency'] / 100, 1.0)
    score += freq_normalized * 0.3
    
    # 3. 文脈エントロピー（高いほど一般的）
    context_entropy = calculate_context_entropy(term_data['contexts'])
    score += context_entropy * 0.4
    
    return score

def build_hierarchy_from_generality(terms):
    """一般性スコアから階層を構築"""
    # 一般性でソート
    sorted_terms = sorted(terms, key=lambda t: estimate_generality(t), reverse=True)
    
    hierarchy = []
    for i, term in enumerate(sorted_terms):
        # 文字列包含関係もチェック
        for j in range(i):
            parent = sorted_terms[j]
            if term['term'] in parent['term'] or is_hyponym(term['term'], parent['term']):
                hierarchy.append({
                    'source': term['term'],
                    'target': parent['term'],
                    'type': 'IS_A',
                    'confidence': 0.7,
                    'provenance': 'hierarchy_estimation'
                })
                break
    
    return hierarchy
```

## 7. 推論の安全策

### 7.1 信頼度減衰と深さ制限

```python
class SafeInferenceEngine:
    def __init__(self, graph, decay_factor=0.7, max_depth=2):
        self.graph = graph
        self.decay_factor = decay_factor
        self.max_depth = max_depth
    
    def infer_transitive_relations(self):
        """推移律による推論（減衰あり）"""
        new_relations = []
        
        # A IS_A B かつ B IS_A C なら A IS_A C
        for edge_ab in self.graph.get_edges(edge_type='IS_A'):
            a, b = edge_ab['source'], edge_ab['target']
            
            for edge_bc in self.graph.get_edges(source=b, edge_type='IS_A'):
                c = edge_bc['target']
                
                # 既存チェック
                if not self.graph.has_edge(a, c, 'IS_A'):
                    # 信頼度減衰
                    confidence = min(edge_ab['confidence'], edge_bc['confidence']) * self.decay_factor
                    
                    if confidence > 0.5:  # 閾値以下は追加しない
                        new_relations.append({
                            'source': a,
                            'target': c,
                            'type': 'IS_A',
                            'confidence': confidence,
                            'weight': confidence,
                            'provenance': 'inference',
                            'properties': {
                                'inference_type': 'transitive',
                                'path': [a, b, c],
                                'depth': 2
                            }
                        })
        
        return new_relations
    
    def infer_symmetric_relations(self):
        """対称関係の推論"""
        new_relations = []
        symmetric_types = ['SIMILAR_TO', 'CO_OCCURS_WITH', 'RELATED_TO']
        
        for edge_type in symmetric_types:
            for edge in self.graph.get_edges(edge_type=edge_type):
                # 逆方向をチェック
                if not self.graph.has_edge(edge['target'], edge['source'], edge_type):
                    new_relations.append({
                        'source': edge['target'],
                        'target': edge['source'],
                        'type': edge_type,
                        'confidence': edge['confidence'] * self.decay_factor,
                        'weight': edge['weight'] * self.decay_factor,
                        'provenance': 'inference',
                        'properties': {
                            'inference_type': 'symmetric',
                            'original_edge': edge['id']
                        }
                    })
        
        return new_relations
```

## 8. クエリ拡張の重み設計

### 8.1 関係別の重み係数

```python
# 関係タイプ別の拡張重み
EXPANSION_WEIGHTS = {
    # 類似・同義
    'SIMILAR_TO': 0.9,
    'SYNONYM': 0.95,
    
    # 階層（IS_A）
    'IS_A_UP_1': 0.7,      # 1階層上
    'IS_A_UP_2': 0.5,      # 2階層上
    'IS_A_DOWN_1': 0.8,    # 1階層下
    'IS_A_DOWN_2': 0.6,    # 2階層下
    
    # 構成
    'PART_OF': 0.5,        # 全体
    'HAS_COMPONENT': 0.7,  # 構成要素
    
    # 関連
    'RELATED_TO': 0.4,
    'CO_OCCURS_WITH': 0.5,
    
    # 機能
    'USED_FOR': 0.6,
    'CONTROLS': 0.5,
    
    # 推論由来はさらに減衰
    'INFERENCE_PENALTY': 0.5
}

class WeightedQueryExpander:
    def __init__(self, graph):
        self.graph = graph
    
    def expand(self, query_term, max_depth=2):
        """重み付きクエリ拡張"""
        expanded = {query_term: 1.0}
        visited = set()
        
        # BFS with weighted expansion
        queue = [(query_term, 1.0, 0)]
        
        while queue:
            term, weight, depth = queue.pop(0)
            
            if term in visited or depth >= max_depth:
                continue
            visited.add(term)
            
            # 各関係タイプで拡張
            for edge in self.graph.get_edges(source=term):
                target = edge['target']
                edge_type = edge['type']
                provenance = edge.get('provenance', 'unknown')
                
                # 基本重み
                if edge_type == 'IS_A':
                    key = f'IS_A_UP_{depth+1}'
                elif edge_type in EXPANSION_WEIGHTS:
                    key = edge_type
                else:
                    key = 'RELATED_TO'
                
                new_weight = weight * EXPANSION_WEIGHTS.get(key, 0.3)
                
                # 推論由来はペナルティ
                if provenance == 'inference':
                    new_weight *= EXPANSION_WEIGHTS['INFERENCE_PENALTY']
                
                # エッジ自体の重みも考慮
                new_weight *= edge.get('weight', 1.0)
                
                # 既存の重みと比較して大きい方を採用
                if target not in expanded or expanded[target] < new_weight:
                    expanded[target] = new_weight
                    
                    # 閾値以上なら次の探索に追加
                    if new_weight > 0.3:
                        queue.append((target, new_weight, depth + 1))
        
        return expanded
    
    def to_query_vector(self, expanded_terms):
        """拡張結果をクエリベクトルに変換"""
        # 重み付き平均でベクトル合成
        vectors = []
        weights = []
        
        for term, weight in expanded_terms.items():
            if embedding := self.graph.get_embedding(term):
                vectors.append(embedding)
                weights.append(weight)
        
        if not vectors:
            return None
        
        # 重み付き平均
        weights = np.array(weights) / np.sum(weights)
        weighted_vector = np.average(vectors, weights=weights, axis=0)
        
        return weighted_vector
```

## 9. エッジ作成ヘルパー（重複防止）

```python
def upsert_edge(conn, src_id, dst_id, edge_type, weight=1.0, confidence=1.0,
                provenance="definition", evidence=None, properties=None):
    """エッジの作成/更新（重複時は最大値採用）"""
    assert src_id != dst_id, "Self-loop is prohibited"
    
    properties = properties or {}
    properties['provenance'] = provenance
    if evidence:
        properties['evidence'] = evidence
    
    query = """
    INSERT INTO knowledge_edges (source_id, target_id, edge_type, weight, confidence, properties)
    VALUES (%(src)s, %(dst)s, %(type)s, %(weight)s, %(conf)s, %(props)s)
    ON CONFLICT (source_id, target_id, edge_type)
    DO UPDATE SET
        weight = GREATEST(knowledge_edges.weight, EXCLUDED.weight),
        confidence = GREATEST(knowledge_edges.confidence, EXCLUDED.confidence),
        properties = knowledge_edges.properties || EXCLUDED.properties
    RETURNING id;
    """
    
    with conn.cursor() as cur:
        cur.execute(query, {
            'src': src_id,
            'dst': dst_id,
            'type': edge_type,
            'weight': weight,
            'conf': confidence,
            'props': json.dumps(properties)
        })
        return cur.fetchone()[0]
```

## 10. 最小実装ロードマップ（現実版）

### Week 1: 基盤とPoC

#### Day 1-2: DB設定とデータロード
```bash
# 1. DDL適用
psql -d $DATABASE_URL -f schema.sql

# 2. 100件のTermロード
python scripts/load_terms.py --input output/terms_100.json
```

#### Day 3-4: 基本3関係の抽出
```python
# IS_A（クラスタ階層から）
python scripts/extract_hierarchy.py --method clustering

# SIMILAR_TO（同一クラスタから）
python scripts/extract_similarity.py --threshold 0.7

# PART_OF（定義文から）
python scripts/extract_composition.py --method definition
```

#### Day 5: 簡易API実装
```python
# FastAPI
from fastapi import FastAPI
app = FastAPI()

@app.get("/expand")
def expand_query(q: str, depth: int = 2):
    expander = WeightedQueryExpander(graph)
    expanded = expander.expand(q, max_depth=depth)
    return {"query": q, "expanded": expanded}
```

#### Day 6-7: 可視化
```python
# Pyvis最小実装
def visualize_subgraph(center: str, depth: int = 2):
    from pyvis.network import Network
    net = Network(height="750px", width="100%", directed=True)
    
    # サブグラフ抽出
    nodes = graph.get_subgraph_nodes(center, depth)
    edges = graph.get_subgraph_edges(nodes)
    
    # ノード追加
    for node in nodes:
        color = '#97c2fc' if node['type'] == 'Term' else '#fb7e81'
        net.add_node(node['id'], label=node['term'], color=color)
    
    # エッジ追加
    for edge in edges:
        net.add_edge(edge['source'], edge['target'], 
                    label=edge['type'], value=edge['weight'])
    
    return net.generate_html()
```

### Week 2-3: 拡張と精度向上

- 日本語定義文抽出の実装（ginza + パターン）
- 共起関係の計算（PMI/NPMI）
- 推論エンジンの実装（減衰・深さ制限付き）
- オフライン評価（Recall@K, MRR）

### Week 4: 統合と最適化

- RAGシステムへの統合
- クエリ拡張係数のA/Bテスト
- パフォーマンス最適化
- ドキュメント作成

## 11. 評価とモニタリング

### 11.1 オフライン評価

```python
def evaluate_query_expansion(test_queries, ground_truth):
    """クエリ拡張の効果測定"""
    metrics = {
        'recall_improvement': [],
        'mrr_improvement': [],
        'coverage_increase': []
    }
    
    for query in test_queries:
        # ベースライン（拡張なし）
        base_results = search(query)
        base_recall = calculate_recall(base_results, ground_truth[query])
        
        # 拡張あり
        expanded = expander.expand(query)
        exp_results = search_with_expansion(expanded)
        exp_recall = calculate_recall(exp_results, ground_truth[query])
        
        # 改善率
        improvement = (exp_recall - base_recall) / base_recall
        metrics['recall_improvement'].append(improvement)
    
    return {
        'mean_recall_improvement': np.mean(metrics['recall_improvement']),
        'median_recall_improvement': np.median(metrics['recall_improvement'])
    }
```

### 11.2 オンラインモニタリング

```python
# Prometheus metrics
from prometheus_client import Counter, Histogram, Gauge

expansion_requests = Counter('kg_expansion_requests_total', 'Total expansion requests')
expansion_latency = Histogram('kg_expansion_latency_seconds', 'Expansion latency')
expanded_terms_count = Histogram('kg_expanded_terms_count', 'Number of expanded terms')
graph_size = Gauge('kg_graph_size', 'Graph size', ['type'])

@expansion_latency.time()
def monitored_expand(query):
    expansion_requests.inc()
    result = expander.expand(query)
    expanded_terms_count.observe(len(result))
    return result
```

## 12. トラブルシューティング

### よくある問題と解決策

| 問題 | 原因 | 解決策 |
|------|------|--------|
| pgvectorエラー | 拡張未インストール | `CREATE EXTENSION vector;` |
| 日本語文字化け | エンコーディング | DB作成時に `ENCODING 'UTF8'` |
| 推論の暴走 | 減衰なし・深さ無制限 | decay_factor=0.7, max_depth=2 |
| クエリ拡張で精度低下 | 重み設計が不適切 | A/Bテストで係数調整 |
| HNSWインデックスが遅い | パラメータ不適切 | m=16, ef_construction=64 推奨 |

## 参考実装

- [pg_vector examples](https://github.com/pgvector/pgvector)
- [GiNZA (spaCy Japanese)](https://megagonlabs.github.io/ginza/)
- [Pyvis Network Visualization](https://pyvis.readthedocs.io/)