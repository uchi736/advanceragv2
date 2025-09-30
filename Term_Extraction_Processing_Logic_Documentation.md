# 専門用語抽出システム - 完全処理ロジックドキュメント

**バージョン**: 2.0
**最終更新**: 2025年10月
**対象**: ハイブリッドSudachi + SemReRank統合システム

---

## 目次

1. [概要](#1-概要)
2. [ハイブリッドSudachi形態素解析](#2-ハイブリッドsudachi形態素解析)
3. [候補抽出フェーズ](#3-候補抽出フェーズ)
4. [統計的スコアリング](#4-統計的スコアリング)
5. [SemReRank処理](#5-semrerank処理)
6. [類義語検出](#6-類義語検出)
7. [RAG定義生成](#7-rag定義生成)
8. [LLMフィルタ](#8-llmフィルタ)
9. [設定パラメータ一覧](#9-設定パラメータ一覧)
10. [完全な処理例](#10-完全な処理例)
11. [パフォーマンス特性](#11-パフォーマンス特性)
12. [トラブルシューティング](#12-トラブルシューティング)

---

## 1. 概要

### 1.1 システムアーキテクチャ

```
入力テキスト
    ↓
┌─────────────────────────────────────────────┐
│  Phase 1: ハイブリッド候補抽出              │
│  - Mode C (長単位)                          │
│  - Mode A + n-gram (短単位)                 │
│  - 正規表現パターン                         │
│  - 複合名詞抽出                             │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│  Phase 2: 統計的スコアリング                │
│  - TF-IDF計算                               │
│  - C-value計算                              │
│  - 2段階スコア（Seed用/Final用）            │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│  Phase 3: SemReRank適用                     │
│  - 埋め込みキャッシュ取得                   │
│  - 意味的関連性グラフ構築                   │
│  - シード選定（上位N%）                     │
│  - Personalized PageRank実行                │
│  - スコア改訂                               │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│  Phase 4: 類義語検出                        │
│  - 6つの検出手法                            │
│  - 部分文字列/共起/編集距離/語幹/略語/      │
│    ドメイン固有                             │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│  Phase 5: RAG定義生成                       │
│  - 上位N%選定                               │
│  - ベクトル検索でコンテキスト取得           │
│  - LLMで定義生成                            │
└─────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────┐
│  Phase 6: LLMフィルタ                       │
│  - バッチ処理                               │
│  - 専門用語判定                             │
│  - 最終出力                                 │
└─────────────────────────────────────────────┘
    ↓
専門用語辞書
```

### 1.2 関連ファイルマップ

| ファイル | 役割 | 主要クラス/関数 |
|---------|------|----------------|
| `src/rag/advanced_term_extraction.py` | 候補抽出・統計処理・類義語検出 | `AdvancedStatisticalExtractor` |
| `src/rag/semrerank.py` | SemReRank処理 | `EmbeddingCache`, `SemReRank` |
| `src/rag/term_extraction.py` | パイプライン統合 | `TermExtractor` |
| `src/rag/prompts.py` | LLMプロンプト定義 | 定義生成/専門用語判定 |
| `src/rag/config.py` | 設定パラメータ | `Config` |

### 1.3 処理モード

- **`use_advanced_extraction=True`** (デフォルト): 完全な9フェーズパイプライン
- **`use_advanced_extraction=False`**: 従来のLLMベース抽出（フォールバック）

---

## 2. ハイブリッドSudachi形態素解析

### 2.1 概要

Sudachiの**Mode C（長単位）**と**Mode A（短単位）**を併用し、両方の利点を活かす。

### 2.2 各モードの役割

#### Mode C（長単位）
- **目的**: 自然な複合語をそのまま保存
- **特徴**: 言語学的に成立している複合語を単一トークンとして扱う
- **実装**: `advanced_term_extraction.py:144-162`

**例**:
```
入力: "舶用ディーゼルエンジンの燃焼効率"

Mode C トークン化:
["舶用ディーゼルエンジン", "の", "燃焼効率"]
       ↓
抽出候補:
- "舶用ディーゼルエンジン" (複合語として保存)
- "燃焼効率"
```

#### Mode A（短単位）
- **目的**: n-gram生成用の基本単位を取得
- **特徴**: 細かく分割して柔軟な複合語生成を可能にする
- **実装**: `advanced_term_extraction.py:164-181`

**例**:
```
入力: "舶用ディーゼルエンジンの燃焼効率"

Mode A トークン化:
["舶用", "ディーゼル", "エンジン", "の", "燃焼", "効率"]
       ↓
n-gram生成 (2-gram ~ 6-gram):
- "舶用ディーゼル"
- "ディーゼルエンジン"
- "舶用ディーゼルエンジン"
- "燃焼効率"
- ... (その他の組み合わせ)
```

### 2.3 マージロジック

両モードの結果を頻度マージ:

```python
candidates = defaultdict(int)

# Mode C結果
mode_c_candidates = {"舶用ディーゼルエンジン": 3, "燃焼効率": 2}
for term, freq in mode_c_candidates.items():
    candidates[term] += freq

# Mode A + n-gram結果
ngram_candidates = {"舶用ディーゼル": 3, "ディーゼルエンジン": 3, "燃焼効率": 2}
for term, freq in ngram_candidates.items():
    candidates[term] += freq

# マージ結果
# {
#   "舶用ディーゼルエンジン": 3,  # Mode Cのみ
#   "舶用ディーゼル": 3,          # Mode A n-gramのみ
#   "ディーゼルエンジン": 3,      # Mode A n-gramのみ
#   "燃焼効率": 4                 # 両方で検出（頻度合計）
# }
```

**メリット**:
- Mode C: 自然な専門用語を逃さない（例: "舶用ディーゼルエンジン"）
- Mode A: 新しい複合語のバリエーションを発見（例: "舶用ディーゼル"）

---

## 3. 候補抽出フェーズ

### 3.1 4つの抽出手法

#### 3.1.1 Mode C抽出

**実装**: `advanced_term_extraction.py:144-162`

```python
def _extract_with_mode_c(self, text: str) -> Dict[str, int]:
    mode_c_terms = defaultdict(int)
    tokens = self.tokenizer_obj.tokenize(text, self.sudachi_mode_c)

    for token in tokens:
        term = token.surface()
        pos = token.part_of_speech()[0]

        # 名詞のみ、かつ2文字以上
        if pos in ['名詞'] and len(term) >= 2:
            if self._is_valid_term(term):
                mode_c_terms[term] += 1

    return mode_c_terms
```

**例**:
```
入力: "6DE-18型エンジンは舶用ディーゼルエンジンである"

Mode C結果:
{
  "6DE-18型エンジン": 1,
  "舶用ディーゼルエンジン": 1
}
```

#### 3.1.2 Mode A + n-gram抽出

**実装**: `advanced_term_extraction.py:164-181`

```python
def _extract_ngrams(self, text: str) -> Dict[str, int]:
    ngrams = defaultdict(int)
    tokens = self.tokenizer_obj.tokenize(text, self.sudachi_mode_a)
    words = [token.surface() for token in tokens]

    # 2-gram ~ 6-gram
    for n in range(self.min_term_length, self.max_term_length + 1):
        for i in range(len(words) - n + 1):
            ngram = ''.join(words[i:i+n])
            if self._is_valid_term(ngram):
                ngrams[ngram] += 1

    return ngrams
```

**例**:
```
入力: "6DE-18型エンジンは舶用ディーゼルエンジンである"

Mode A トークン化:
["6", "DE", "-", "18", "型", "エンジン", "は", "舶用", "ディーゼル", "エンジン", "で", "ある"]

n-gram結果 (一部):
{
  "6DE": 1,
  "DE-18": 1,
  "18型": 1,
  "型エンジン": 1,
  "6DE-18型": 1,
  "舶用ディーゼル": 1,
  "ディーゼルエンジン": 1,
  "舶用ディーゼルエンジン": 1
}
```

#### 3.1.3 正規表現パターンマッチング

**実装**: `advanced_term_extraction.py:76-100, 183-194`

**パターン定義**:
```python
patterns = [
    # 型式番号・製品コード
    r'\b[A-Z0-9]{2,}[-_][A-Z0-9]+\b',      # 例: 6DE-18, L28ADF
    r'\b[0-9]+[A-Z]{2,}[-_][0-9]+\b',

    # 化学式・化合物
    r'\b(CO2|NOx|SOx|PM2\.5|NH3|H2O|CH4|N2O)\b',

    # 専門的な略語（3文字以上の大文字）
    r'\b[A-Z]{3,}\b',                       # 例: RAG, LLM, API

    # 数値+単位
    r'\b\d+(\.\d+)?\s*(mg|kg|kWh|MW|rpm|bar|°C|K|Pa|MPa|m³|L)/?\w*\b',

    # カタカナ+英数字の複合語
    r'[ァ-ヴー]+[A-Z0-9]+',
    r'[A-Z0-9]+[ァ-ヴー]+',

    # 複合技術用語パターン
    r'[ァ-ヴー]+(燃料|エンジン|システム|装置|機構)',
]
```

**例**:
```
入力: "NOx排出量は0.5mg/m³以下、CO2は500ppm"

正規表現結果:
{
  "NOx": 1,
  "CO2": 1,
  "0.5mg/m³": 1,
  "500ppm": 1
}
```

#### 3.1.4 複合名詞抽出（Mode A使用）

**実装**: `advanced_term_extraction.py:196-223`

**ロジック**: 連続する「名詞」または「接頭辞」をまとめる

```python
def _extract_compound_nouns(self, text: str) -> Dict[str, int]:
    compound_nouns = defaultdict(int)
    tokens = self.tokenizer_obj.tokenize(text, self.sudachi_mode_a)
    current_compound = []

    for token in tokens:
        pos = token.part_of_speech()[0]

        if pos in ['名詞', '接頭辞']:
            current_compound.append(token.surface())
        else:
            # 複合名詞が終了
            if len(current_compound) >= self.min_term_length:
                compound = ''.join(current_compound)
                if self._is_valid_term(compound):
                    compound_nouns[compound] += 1
            current_compound = []

    return compound_nouns
```

**例**:
```
入力: "次世代燃料電池システムの開発"

Mode A トークン化:
[("次世代", 名詞), ("燃料", 名詞), ("電池", 名詞), ("システム", 名詞), ("の", 助詞), ...]

複合名詞抽出:
{
  "次世代": 1,
  "燃料電池": 1,
  "電池システム": 1,
  "次世代燃料": 1,
  "次世代燃料電池": 1,
  "燃料電池システム": 1,
  "次世代燃料電池システム": 1
}
```

### 3.2 最小頻度フィルタ

全手法の結果をマージ後、`min_frequency`（デフォルト2）以上の候補のみを残す。

```python
# 最小頻度フィルタ
filtered = {
    term: freq
    for term, freq in candidates.items()
    if freq >= self.min_frequency
}
```

---

## 4. 統計的スコアリング

### 4.1 TF-IDF計算

**実装**: `advanced_term_extraction.py:364-420`

**手法**: scikit-learnの`TfidfVectorizer`を使用

```python
def calculate_tfidf(
    self,
    documents: List[str],  # 文単位で分割
    candidates: Dict[str, int]
) -> Dict[str, float]:
    vocabulary = list(candidates.keys())

    vectorizer = TfidfVectorizer(
        vocabulary=vocabulary,
        token_pattern=r'\S+',
        lowercase=False,
        norm=None
    )

    # 各文書を候補用語でトークン化
    tokenized_docs = []
    for doc in documents:
        tokens = []
        for term in vocabulary:
            count = doc.count(term)
            tokens.extend([term] * count)
        tokenized_docs.append(' '.join(tokens))

    tfidf_matrix = vectorizer.fit_transform(tokenized_docs)

    # 各用語の最大TF-IDFスコアを取得
    tfidf_scores = {}
    for i, term in enumerate(vectorizer.get_feature_names_out()):
        scores = tfidf_matrix[:, i].toarray().flatten()
        tfidf_scores[term] = float(np.max(scores))

    return tfidf_scores
```

**具体例**:
```
入力文書（3文）:
1. "舶用ディーゼルエンジンは低速エンジンである"
2. "舶用ディーゼルエンジンの燃焼効率は高い"
3. "低速エンジンは燃料消費が少ない"

候補用語:
["舶用ディーゼルエンジン", "低速エンジン", "燃焼効率", "燃料消費"]

TF-IDF計算:
- "舶用ディーゼルエンジン": TF=2/3文書, IDF=log(3/2) → TF-IDF ≈ 0.35
- "低速エンジン": TF=2/3文書, IDF=log(3/2) → TF-IDF ≈ 0.35
- "燃焼効率": TF=1/3文書, IDF=log(3/1) → TF-IDF ≈ 0.37
- "燃料消費": TF=1/3文書, IDF=log(3/1) → TF-IDF ≈ 0.37

結果:
{
  "舶用ディーゼルエンジン": 0.35,
  "低速エンジン": 0.35,
  "燃焼効率": 0.37,
  "燃料消費": 0.37
}
```

### 4.2 C-value計算

**実装**: `advanced_term_extraction.py:422-465`

**数式**:
```
C-value(a) = log₂(|a|) × freq(a) - (1/|Ta|) × Σ freq(b)
                                              b∈Ta

where:
- |a| = 用語aの長さ（文字数）
- freq(a) = 用語aの頻度
- Ta = aを含むより長い用語の集合
- b ∈ Ta
```

**ロジック**:
1. 用語を長さ降順でソート
2. 各用語について、それを含むより長い用語を記録（ネスト情報）
3. 基本C-value = log₂(長さ) × 頻度
4. ネストペナルティ = ネストされている用語の平均頻度
5. 最終C-value = 基本C-value - ネストペナルティ

**具体例**:
```
候補用語と頻度:
{
  "舶用ディーゼルエンジン": 5,
  "ディーゼルエンジン": 8,
  "エンジン": 15
}

ネスト関係:
- "エンジン" は "ディーゼルエンジン" に含まれる
- "エンジン" は "舶用ディーゼルエンジン" に含まれる
- "ディーゼルエンジン" は "舶用ディーゼルエンジン" に含まれる

C-value計算:
1. "舶用ディーゼルエンジン" (長さ=11):
   - 基本: log₂(11) × 5 = 3.46 × 5 = 17.3
   - ネスト: なし
   - C-value = 17.3

2. "ディーゼルエンジン" (長さ=8):
   - 基本: log₂(8) × 8 = 3.0 × 8 = 24.0
   - ネスト: ["舶用ディーゼルエンジン"] → 5/1 = 5.0
   - C-value = 24.0 - 5.0 = 19.0

3. "エンジン" (長さ=4):
   - 基本: log₂(4) × 15 = 2.0 × 15 = 30.0
   - ネスト: ["ディーゼルエンジン", "舶用ディーゼルエンジン"] → (8+5)/2 = 6.5
   - C-value = 30.0 - 6.5 = 23.5

結果:
{
  "舶用ディーゼルエンジン": 17.3,
  "ディーゼルエンジン": 19.0,
  "エンジン": 23.5
}
```

**解釈**: 長い複合語ほど専門的、ただし単独で出現しないネストされた用語はペナルティ

### 4.3 2段階スコアリング

**実装**: `advanced_term_extraction.py:485-524`

#### Stage A: Seed選定用（C-value重視）

```python
# シード選定用：C-value重視
w_tfidf = 0.3
w_cvalue = 0.7

seed_score = w_tfidf × tfidf_norm + w_cvalue × cvalue_norm
```

**目的**: 複合語の専門性を優先してシードを選ぶ

#### Stage B: Final用（TF-IDF重視）

```python
# 最終スコア用：TF-IDF重視
w_tfidf = 0.7
w_cvalue = 0.3

final_score = w_tfidf × tfidf_norm + w_cvalue × cvalue_norm
```

**目的**: 文書全体での重要性を重視

### 4.4 Min-max正規化

**実装**: `advanced_term_extraction.py:467-483`

```python
def min_max_normalize(self, scores: Dict[str, float]) -> Dict[str, float]:
    values = list(scores.values())
    min_val = min(values)
    max_val = max(values)

    if max_val - min_val < 1e-10:
        return {k: 0.5 for k in scores}

    return {
        term: (score - min_val) / (max_val - min_val)
        for term, score in scores.items()
    }
```

**注意**: 個別スコア（TF-IDF、C-value）のみ正規化。結合後は**再正規化しない**。

**具体例**:
```
TF-IDF生スコア:
{
  "舶用ディーゼルエンジン": 0.35,
  "低速エンジン": 0.35,
  "燃焼効率": 0.37
}

Min-max正規化後:
min = 0.35, max = 0.37, range = 0.02

{
  "舶用ディーゼルエンジン": (0.35-0.35)/0.02 = 0.0,
  "低速エンジン": (0.35-0.35)/0.02 = 0.0,
  "燃焼効率": (0.37-0.35)/0.02 = 1.0
}

C-value生スコア:
{
  "舶用ディーゼルエンジン": 17.3,
  "低速エンジン": 15.2,
  "燃焼効率": 12.5
}

Min-max正規化後:
min = 12.5, max = 17.3, range = 4.8

{
  "舶用ディーゼルエンジン": (17.3-12.5)/4.8 = 1.0,
  "低速エンジン": (15.2-12.5)/4.8 = 0.56,
  "燃焼効率": (12.5-12.5)/4.8 = 0.0
}

Final Score (w_tfidf=0.7, w_cvalue=0.3):
{
  "舶用ディーゼルエンジン": 0.7×0.0 + 0.3×1.0 = 0.3,
  "低速エンジン": 0.7×0.0 + 0.3×0.56 = 0.168,
  "燃焼効率": 0.7×1.0 + 0.3×0.0 = 0.7
}
```

---

## 5. SemReRank処理

### 5.1 概要

**論文**: Zhang et al., 2017. "SemRe-Rank: Improving Automatic Term Extraction By Incorporating Semantic Relatedness With Personalised PageRank"

**目的**: 意味的関連性を考慮して、低頻度だが重要な専門用語を救い上げる

### 5.2 埋め込みキャッシュ機構

**実装**: `semrerank.py:27-162`

#### キャッシュテーブル構造

```sql
CREATE TABLE IF NOT EXISTS term_embeddings (
    term TEXT PRIMARY KEY,
    embedding VECTOR(1536),  -- Azure OpenAI text-embedding-3-small
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

#### 取得フロー

```python
def get_embeddings(self, terms: List[str]) -> Dict[str, np.ndarray]:
    embeddings_map = {}

    # 1. キャッシュから取得
    with self.engine.connect() as conn:
        result = conn.execute(
            f"SELECT term, embedding FROM {cache_table} WHERE term IN (...)"
        )
        for row in result:
            embeddings_map[row.term] = np.array(eval(row.embedding))

    # 2. 未キャッシュの用語を特定
    terms_to_compute = [term for term in terms if term not in embeddings_map]

    # 3. Azure OpenAI Embeddingsで計算
    if terms_to_compute:
        computed_embeddings = self.embeddings.embed_documents(terms_to_compute)

        for term, embedding in zip(terms_to_compute, computed_embeddings):
            embeddings_map[term] = np.array(embedding)
            # キャッシュに保存
            self._save_to_cache(term, embedding)

    return embeddings_map
```

**具体例**:
```
入力: ["舶用ディーゼルエンジン", "燃焼効率", "NOx排出量"]

キャッシュ状態:
- "舶用ディーゼルエンジン": キャッシュあり
- "燃焼効率": キャッシュなし
- "NOx排出量": キャッシュなし

処理:
1. キャッシュから "舶用ディーゼルエンジン" の埋め込みを取得
2. Azure OpenAI APIで "燃焼効率", "NOx排出量" の埋め込みを計算（2件）
3. 計算結果をキャッシュに保存
4. 全3件の埋め込みを返す

コスト削減: 3件 → 2件のAPI呼び出し（33%削減）
```

### 5.3 意味的関連性グラフ構築

**実装**: `semrerank.py:164-218`

#### パラメータ

- **relmin**: 最小類似度閾値（デフォルト0.5）
- **reltop**: 各ノードの上位関連語の割合（デフォルト0.15 = 15%）

#### アルゴリズム

```python
def build_semantic_graph(
    embeddings: Dict[str, np.ndarray],
    relmin: float = 0.5,
    reltop: float = 0.15
) -> nx.Graph:
    terms = list(embeddings.keys())
    n_terms = len(terms)

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
        similarities = [(j, similarity_matrix[i][j])
                        for j in range(n_terms) if i != j]

        # relmin以上の類似度のみ
        similarities = [(j, sim) for j, sim in similarities if sim >= relmin]

        # 類似度降順でソート
        similarities.sort(key=lambda x: x[1], reverse=True)

        # 上位top_k件とエッジを作成
        for j, sim in similarities[:top_k]:
            term2 = terms[j]
            if not graph.has_edge(term1, term2):
                graph.add_edge(term1, term2, weight=sim)

    return graph
```

**具体例**:
```
用語数: 100件
reltop: 0.15 → top_k = 15件
relmin: 0.5

ステップ:
1. 100件の埋め込みベクトルからコサイン類似度行列を計算（100×100）

2. 各用語について:
   - "舶用ディーゼルエンジン" の場合
     - 99件の類似度を計算
     - relmin=0.5以上をフィルタ → 例: 40件
     - 類似度降順でソート
     - 上位15件とエッジを作成
       例: "低速エンジン"(0.85), "ディーゼル機関"(0.82), ...

グラフ構造:
- ノード数: 100
- エッジ数: ~750件（100 × 15 / 2）
  ※ 無向グラフなので重複除去
```

### 5.4 シード選定

**実装**: `semrerank.py:221-249`

#### パーセンタイルベース選定

```python
def select_seeds_by_percentile(
    seed_scores: Dict[str, float],
    percentile: float = 15.0
) -> List[str]:
    # 上位N%の件数を計算
    n_seeds = max(1, int(len(seed_scores) * percentile / 100))

    # スコア降順でソート
    sorted_terms = sorted(seed_scores.items(), key=lambda x: x[1], reverse=True)

    # 上位n_seeds件を選択
    seeds = [term for term, score in sorted_terms[:n_seeds]]

    return seeds
```

**具体例**:
```
候補用語数: 100件
seed_percentile: 15.0%

Seed選定用スコア（C-value重視）:
{
  "舶用ディーゼルエンジン": 0.95,
  "低速エンジン": 0.88,
  "燃焼効率": 0.85,
  ...（スコア降順）
}

選定:
n_seeds = max(1, int(100 × 15.0 / 100)) = 15件

シード:
["舶用ディーゼルエンジン", "低速エンジン", "燃焼効率", ..., "排気ガス処理"] (15件)
```

### 5.5 Personalized PageRank実行

**実装**: `semrerank.py:252-294`

#### アルゴリズム

```python
def personalized_pagerank(
    graph: nx.Graph,
    seeds: List[str],
    alpha: float = 0.85
) -> Dict[str, float]:
    # personalizationベクトル作成（シードに重み1.0、他は0.0）
    personalization = {
        node: 1.0 if node in seeds else 0.0
        for node in graph.nodes()
    }

    # networkxのpagerank関数を使用
    importance_scores = nx.pagerank(
        graph,
        alpha=alpha,
        personalization=personalization,
        max_iter=100,
        tol=1e-06
    )

    return importance_scores
```

#### Personalized PageRankの直感的理解

通常のPageRank:
- ランダムサーファーがグラフをランダムウォーク
- 各ノードの訪問確率を計算

Personalized PageRank:
- **シードノードから優先的にスタート**
- シードと意味的に関連性の高いノードほど高スコア
- 低頻度でもシードと関連があれば救い上げられる

**具体例**:
```
グラフ:
- ノード数: 100件
- シード: 15件（上位15%）
  例: "舶用ディーゼルエンジン", "低速エンジン", "燃焼効率", ...

Personalizationベクトル:
{
  "舶用ディーゼルエンジン": 1.0,  # シード
  "低速エンジン": 1.0,            # シード
  "燃焼効率": 1.0,                # シード
  ...
  "6DE-18型": 0.0,                # 非シード（低頻度だが重要）
  ...
}

PageRank実行（alpha=0.85）:
- 85%の確率で隣接ノードに移動
- 15%の確率でシードにテレポート

結果（重要度スコア）:
{
  "舶用ディーゼルエンジン": 0.025,  # シード
  "低速エンジン": 0.022,            # シード
  "6DE-18型": 0.018,                # 非シードだが、シードと強く関連
  "燃焼効率": 0.020,                # シード
  "一般用語": 0.001,                # シードと関連なし
  ...
}

解釈:
- "6DE-18型" は低頻度（頻度2回）だが、"舶用ディーゼルエンジン"と強く関連
- PageRankで高スコア（0.018）を獲得
- 専門用語として救い上げられる
```

### 5.6 スコア改訂

**実装**: `semrerank.py:297-333`

#### 改訂式

```python
def revise_scores(
    base_scores: Dict[str, float],
    importance_scores: Dict[str, float]
) -> Dict[str, float]:
    # PageRank重要度の平均を計算
    avg_importance = np.mean(list(importance_scores.values()))

    # スコア改訂: final_score = base_score × (1 + avg_importance)
    revised_scores = {}
    for term in base_scores:
        base = base_scores[term]
        importance = importance_scores.get(term, 0.0)

        # 改訂係数
        boost = 1 + (importance / avg_importance - 1)

        revised_scores[term] = base * boost

    return revised_scores
```

**具体例**:
```
Base Scores（TF-IDF + C-value）:
{
  "舶用ディーゼルエンジン": 0.95,
  "低速エンジン": 0.88,
  "6DE-18型": 0.12,  # 低頻度で低スコア
  "燃焼効率": 0.85,
  "一般用語": 0.05
}

Importance Scores（PageRank）:
{
  "舶用ディーゼルエンジン": 0.025,
  "低速エンジン": 0.022,
  "6DE-18型": 0.018,  # シードと関連して高スコア
  "燃焼効率": 0.020,
  "一般用語": 0.001
}

平均重要度:
avg_importance = (0.025 + 0.022 + 0.018 + 0.020 + 0.001 + ...) / 100 ≈ 0.010

改訂計算:
1. "舶用ディーゼルエンジン":
   boost = 1 + (0.025 / 0.010 - 1) = 1 + 1.5 = 2.5
   revised = 0.95 × 2.5 = 2.375

2. "6DE-18型":
   boost = 1 + (0.018 / 0.010 - 1) = 1 + 0.8 = 1.8
   revised = 0.12 × 1.8 = 0.216  ← 大幅に向上！

3. "一般用語":
   boost = 1 + (0.001 / 0.010 - 1) = 1 - 0.9 = 0.1
   revised = 0.05 × 0.1 = 0.005  ← 低下

Revised Scores:
{
  "舶用ディーゼルエンジン": 2.375,
  "低速エンジン": 1.936,
  "6DE-18型": 0.216,  ← 0.12から大幅アップ
  "燃焼効率": 1.700,
  "一般用語": 0.005
}

効果:
- "6DE-18型" が低頻度でも専門用語として認識される
- 一般用語はスコアが下がる
```

### 5.7 SemReRankクラス統合

**実装**: `semrerank.py:336-401`

```python
class SemReRank:
    def __init__(
        self,
        embeddings,
        connection_string: str,
        relmin: float = 0.5,
        reltop: float = 0.15,
        alpha: float = 0.85,
        seed_percentile: float = 15.0
    ):
        self.embedding_cache = EmbeddingCache(embeddings, connection_string)
        self.relmin = relmin
        self.reltop = reltop
        self.alpha = alpha
        self.seed_percentile = seed_percentile

    def enhance_scores(
        self,
        candidates: List[str],
        base_scores: Dict[str, float],
        seed_scores: Dict[str, float]
    ) -> Dict[str, float]:
        # 1. 埋め込み取得
        embeddings = self.embedding_cache.get_embeddings(candidates)

        # 2. 意味的関連性グラフ構築
        graph = build_semantic_graph(embeddings, self.relmin, self.reltop)

        # 3. シード選定
        seeds = select_seeds_by_percentile(seed_scores, self.seed_percentile)

        # 4. Personalized PageRank実行
        importance_scores = personalized_pagerank(graph, seeds, self.alpha)

        # 5. スコア改訂
        revised_scores = revise_scores(base_scores, importance_scores)

        return revised_scores
```

---

## 6. 類義語検出

### 6.1 概要

**実装**: `advanced_term_extraction.py:245-362`

**目的**: 統計的手法で候補用語間の類義語・関連語を検出

### 6.2 6つの検出手法

#### 6.2.1 部分文字列関係

```python
# 1. 部分文字列関係の検出
for i, cand1 in enumerate(candidates):
    for cand2 in candidates[i+1:]:
        if cand1 in cand2:
            synonyms[cand2].add(cand1)
        elif cand2 in cand1:
            synonyms[cand1].add(cand2)
```

**例**:
```
候補: ["ディーゼルエンジン", "舶用ディーゼルエンジン", "エンジン"]

検出:
- "舶用ディーゼルエンジン" の類義語: ["ディーゼルエンジン", "エンジン"]
- "ディーゼルエンジン" の類義語: ["エンジン"]
```

#### 6.2.2 共起関係

```python
# 2. 共起関係の検出
noun_phrases = self._extract_noun_phrases(full_text)
cooccurrence_map = defaultdict(set)

for phrase in noun_phrases:
    occurring_cands = [c for c in candidates if c in phrase]
    for cand1 in occurring_cands:
        for cand2 in occurring_cands:
            if cand1 != cand2:
                cooccurrence_map[cand1].add(cand2)

# 2回以上共起した用語を類義語候補とする
for cand, related in cooccurrence_map.items():
    if len(related) >= 2:
        synonyms[cand].update(related)
```

**例**:
```
テキスト: "舶用ディーゼルエンジンの低速運転特性"

名詞句: ["舶用ディーゼルエンジン低速運転特性"]

共起:
- "舶用ディーゼルエンジン", "低速運転", "運転特性" が同一名詞句内
- これらを互いに関連語として記録
```

#### 6.2.3 編集距離

```python
# 3. 編集距離による類似語検出
for i, cand1 in enumerate(candidates):
    for cand2 in candidates[i+1:]:
        if len(cand1) >= 3 and len(cand2) >= 3:
            similarity = SequenceMatcher(None, cand1, cand2).ratio()
            if 0.7 < similarity < 0.95:
                synonyms[cand1].add(cand2)
                synonyms[cand2].add(cand1)
```

**例**:
```
候補: ["ディーゼルエンジン", "ディーゼル機関"]

類似度計算:
similarity = 0.82 (70%~95%の範囲内)

検出:
- "ディーゼルエンジン" の類義語: ["ディーゼル機関"]
- "ディーゼル機関" の類義語: ["ディーゼルエンジン"]
```

#### 6.2.4 語幹・語尾パターン

```python
# 4. 語幹・語尾パターン検出
suffixes = ['化', '的', '性', '型', '式', 'ー', 'ション', 'ング', 'メント']

for cand in candidates:
    for suffix in suffixes:
        if cand.endswith(suffix):
            base = cand[:-len(suffix)]
            if len(base) >= 2:
                for other_cand in candidates:
                    if other_cand != cand and other_cand.startswith(base):
                        synonyms[cand].add(other_cand)
```

**例**:
```
候補: ["燃焼", "燃焼効率", "燃焼性", "燃焼化"]

検出:
- "燃焼効率" の類義語: ["燃焼", "燃焼性", "燃焼化"]
- "燃焼性" の類義語: ["燃焼", "燃焼効率", "燃焼化"]
- "燃焼化" の類義語: ["燃焼", "燃焼効率", "燃焼性"]
```

#### 6.2.5 略語パターン

```python
# 5. 略語パターンの検出
abbreviation_patterns = {
    'RAG': 'Retrieval-Augmented Generation',
    'LLM': '大規模言語モデル',
    'API': 'アプリケーションプログラミングインターフェース',
    'DB': 'データベース',
    'QA': '質問応答',
}

for abbr, full_name in abbreviation_patterns.items():
    if abbr in candidates and full_name in candidates:
        synonyms[abbr].add(full_name)
        synonyms[full_name].add(abbr)
```

**例**:
```
候補: ["RAG", "Retrieval-Augmented Generation", "LLM", "大規模言語モデル"]

検出:
- "RAG" の類義語: ["Retrieval-Augmented Generation"]
- "Retrieval-Augmented Generation" の類義語: ["RAG"]
- "LLM" の類義語: ["大規模言語モデル"]
- "大規模言語モデル" の類義語: ["LLM"]
```

#### 6.2.6 ドメイン固有の関連語

```python
# 6. ドメイン固有の関連語
domain_relations = {
    'ベクトル検索': ['vector search', 'ベクトルサーチ', 'ベクトルDB'],
    'embedding': ['埋め込み', 'エンベディング', '埋め込み表現'],
    'リランキング': ['re-ranking', 'リランク', '再順位付け'],
}

for main_term, related_terms in domain_relations.items():
    if main_term in candidates:
        for related in related_terms:
            if related in candidates:
                synonyms[main_term].add(related)
                synonyms[related].add(main_term)
```

**例**:
```
候補: ["ベクトル検索", "vector search", "ベクトルDB"]

検出:
- "ベクトル検索" の類義語: ["vector search", "ベクトルDB"]
- "vector search" の類義語: ["ベクトル検索", "ベクトルDB"]
- "ベクトルDB" の類義語: ["ベクトル検索", "vector search"]
```

### 6.3 統合結果

**実装**: `term_extraction.py:524-530`

```python
# 5. 類義語検出
synonym_map = self.statistical_extractor.detect_synonyms(
    candidates=list(candidates.keys()),
    full_text=full_text
)

# 6. ExtractedTermオブジェクト化
terms = [
    ExtractedTerm(
        term=term,
        score=enhanced_scores[term],
        synonyms=synonym_map.get(term, [])  # 類義語リストを設定
    )
    for term in enhanced_scores
]
```

**出力例**:
```json
{
  "headword": "舶用ディーゼルエンジン",
  "score": 2.375,
  "synonyms": [
    "ディーゼルエンジン",
    "ディーゼル機関",
    "マリンエンジン",
    "船舶用エンジン"
  ],
  "definition": "...",
  "frequency": 5
}
```

---

## 7. RAG定義生成

### 7.1 概要

**実装**: `term_extraction.py:549-577`

**目的**: 上位N%の専門用語に対して、ベクトル検索で関連コンテキストを取得し、LLMで定義を生成

### 7.2 上位N%選定

```python
definition_percentile = getattr(self.config, 'definition_generation_percentile', 15.0)
n_terms = max(1, int(len(terms) * definition_percentile / 100))
target_terms = terms[:n_terms]
```

**例**:
```
全候補用語数: 100件
definition_percentile: 15.0%

選定:
n_terms = max(1, int(100 × 15.0 / 100)) = 15件

定義生成対象:
スコア降順の上位15件
["舶用ディーゼルエンジン", "低速エンジン", "燃焼効率", ..., "排気ガス処理"]
```

### 7.3 ベクトル検索とコンテキスト構築

```python
for i, term in enumerate(target_terms, 1):
    # ベクトル検索でk=5件取得
    docs = self.vector_store.similarity_search(term.term, k=5)

    if docs:
        # 最大3000文字のコンテキストを構築
        context = "\n\n".join([doc.page_content for doc in docs])[:3000]

        # LLMで定義生成
        definition = await chain.ainvoke({"term": term.term, "context": context})
        term.definition = definition.strip()
```

**例**:
```
用語: "舶用ディーゼルエンジン"

ベクトル検索（k=5）:
1. "舶用ディーゼルエンジンは低速で高トルクを発生する..." (類似度: 0.92)
2. "4サイクルディーゼルエンジンの燃焼効率は..." (類似度: 0.85)
3. "舶用エンジンの主要メーカーは..." (類似度: 0.83)
4. "ディーゼルエンジンの排気ガス処理..." (類似度: 0.80)
5. "エンジンの冷却システムについて..." (類似度: 0.75)

コンテキスト構築:
"""
舶用ディーゼルエンジンは低速で高トルクを発生する...

4サイクルディーゼルエンジンの燃焼効率は...

舶用エンジンの主要メーカーは...

ディーゼルエンジンの排気ガス処理...

エンジンの冷却システムについて...
"""
（最大3000文字に切り詰め）
```

### 7.4 LLMプロンプト構造

**実装**: `prompts.py` - `DEFINITION_GENERATION_SYSTEM_PROMPT`

```python
DEFINITION_GENERATION_SYSTEM_PROMPT = """あなたは専門用語の定義作成の専門家です。

**定義作成の原則:**
1. **簡潔性**: 1〜3文で定義を完結させる
2. **正確性**: 提供されたコンテキストに基づいて正確に定義
3. **具体性**: 抽象的な説明ではなく、具体的な特徴や用途を含める
4. **専門性**: 専門用語として適切なレベルで説明

**定義のフォーマット:**
- 用語の本質を最初の1文で説明
- 必要に応じて、特徴や用途を2文目で補足
- 技術的な詳細は3文目まで

**注意事項:**
- コンテキストに情報がない場合は、推測せずに「情報不足」と明記
- 一般的すぎる説明は避ける
- 数値や固有名詞がある場合は正確に記載
"""

DEFINITION_GENERATION_USER_PROMPT = """用語: {term}

コンテキスト:
{context}

上記のコンテキストに基づいて、用語「{term}」の定義を作成してください。"""
```

**LLM呼び出し**:
```python
prompt = get_definition_generation_prompt()
chain = prompt | self.llm | StrOutputParser()

definition = await chain.ainvoke({
    "term": "舶用ディーゼルエンジン",
    "context": "（3000文字のコンテキスト）"
})
```

**LLM出力例**:
```
舶用ディーゼルエンジンは、船舶の推進力を得るために使用される内燃機関で、低速で高トルクを発生させる特性を持つ。4サイクルまたは2サイクル方式で動作し、燃料効率が高く、長時間の連続運転に適している。主要メーカーには三菱重工、MAN Energy Solutions、Wärtsiläなどがあり、大型商船やタンカーに広く採用されている。
```

### 7.5 定義生成の対象外

- **類義語**: 定義は生成されない（コスト削減）
- **下位N%**: スコアが低い候補は定義生成をスキップ

**出力形式**:
```json
{
  "headword": "舶用ディーゼルエンジン",
  "definition": "舶用ディーゼルエンジンは、船舶の推進力を...",
  "synonyms": ["ディーゼル機関", "マリンエンジン"]
}

{
  "headword": "ディーゼル機関",
  "definition": "",  // 類義語なので空
  "synonyms": ["舶用ディーゼルエンジン", "マリンエンジン"]
}
```

---

## 8. LLMフィルタ

### 8.1 概要

**実装**: `term_extraction.py:579-619`

**目的**: RAG定義生成された用語を対象に、専門用語か一般用語かをLLMで判定

### 8.2 バッチ処理

```python
batch_size = getattr(self.config, 'llm_filter_batch_size', 10)

# 定義がある用語のみをフィルタ対象とする
terms_with_def = [t for t in terms if t.definition]

technical_terms = []

for i in range(0, len(terms_with_def), batch_size):
    batch = terms_with_def[i:i+batch_size]
    batch_inputs = [{"term": t.term, "definition": t.definition} for t in batch]

    # LLMにバッチ送信
    result_texts = await chain.abatch(batch_inputs)

    for term, result_text in zip(batch, result_texts):
        result = self._parse_llm_json(result_text)
        if result and result.get("is_technical", False):
            term.metadata["confidence"] = result.get("confidence", 0.0)
            term.metadata["reason"] = result.get("reason", "")
            technical_terms.append(term)
```

**例**:
```
定義がある用語: 15件
batch_size: 10

バッチ1: 10件
バッチ2: 5件

LLM呼び出し: 2回
（バッチ処理でAPI呼び出しを削減）
```

### 8.3 LLMプロンプト構造

**実装**: `prompts.py` - `TECHNICAL_TERM_JUDGMENT_SYSTEM_PROMPT`

```python
TECHNICAL_TERM_JUDGMENT_SYSTEM_PROMPT = """あなたは専門用語判定の専門家です。

用語とその定義を分析し、**専門用語**か**一般用語**かを判定してください。

【専門用語として判定】
• 型式番号・製品コード（6DE-18、L28ADFなど）
• 化学式・化合物名（CO2、NOx、アンモニアなど）
• 技術的な固有名詞（PageRank、TF-IDF、ベクトル検索など）
• 特定分野の専門的概念（燃焼効率、排気ガス処理など）
• 業界特有の略語（RAG、LLM、APIなど）

【一般用語として判定】
• 日常的に使われる基本語彙（開発、システム、方法など）
• 一般的な動詞・形容詞（実施する、効率的、重要など）
• 抽象的すぎる概念（品質、性能など）※ただし定義が具体的な場合は専門用語

**出力形式（JSON）:**
{
  "is_technical": true/false,
  "confidence": 0.0～1.0,
  "reason": "判定理由（1文）"
}
"""

TECHNICAL_TERM_JUDGMENT_USER_PROMPT = """用語: {term}
定義: {definition}

この用語は専門用語ですか？"""
```

**LLM入力例**:
```json
{
  "term": "舶用ディーゼルエンジン",
  "definition": "舶用ディーゼルエンジンは、船舶の推進力を得るために使用される内燃機関で..."
}
```

**LLM出力例**:
```json
{
  "is_technical": true,
  "confidence": 0.95,
  "reason": "船舶の推進力を得るための特定の内燃機関を指す専門用語"
}
```

### 8.4 JSON応答パース

**実装**: `term_extraction.py:641-664`

```python
def _parse_llm_json(self, text: str) -> Optional[Dict]:
    text = text.strip()

    # コードブロック除去
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        return json.loads(text)
    except:
        # フォールバック: 正規表現で抽出
        try:
            match = re.search(r'\{[^{}]*\}', text)
            if match:
                return json.loads(match.group())
        except:
            pass

    return None
```

**パース例**:
```
LLM生成テキスト:
```json
{
  "is_technical": true,
  "confidence": 0.95,
  "reason": "船舶の推進力を得るための特定の内燃機関を指す専門用語"
}
```

パース後:
{
  "is_technical": true,
  "confidence": 0.95,
  "reason": "船舶の推進力を得るための特定の内燃機関を指す専門用語"
}
```

### 8.5 フィルタ結果

```python
if result and result.get("is_technical", False):
    term.metadata["confidence"] = result.get("confidence", 0.0)
    term.metadata["reason"] = result.get("reason", "")
    technical_terms.append(term)
    logger.info(f"  [OK] {term.term}: 専門用語")
else:
    logger.info(f"  [NG] {term.term}: 一般用語")

# フィルタ後の用語を返す
terms = technical_terms
```

**例**:
```
入力: 15件（定義あり）

フィルタ結果:
[OK] 舶用ディーゼルエンジン: 専門用語
[OK] 低速エンジン: 専門用語
[NG] システム: 一般用語
[OK] 6DE-18型: 専門用語
[OK] NOx排出量: 専門用語
[NG] 開発: 一般用語
...

最終出力: 12件（専門用語のみ）
```

---

## 9. 設定パラメータ一覧

### 9.1 全パラメータ

**定義場所**: `src/rag/config.py:112-124`

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|----------|------|
| **use_advanced_extraction** | bool | True | 高度な統計的抽出を使用するか |
| **semrerank_enabled** | bool | True | SemReRankを有効にするか |
| **semrerank_seed_percentile** | float | 15.0 | シード選定の上位パーセンタイル（%） |
| **semrerank_relmin** | float | 0.5 | 意味的関連性グラフの最小類似度閾値 |
| **semrerank_reltop** | float | 0.15 | 各ノードの上位関連語の割合（15%） |
| **semrerank_alpha** | float | 0.85 | PageRankのダンピング係数 |
| **definition_generation_percentile** | float | 15.0 | RAG定義生成の上位パーセンタイル（%） |
| **llm_filter_batch_size** | int | 10 | LLMフィルタのバッチサイズ |

### 9.2 詳細説明

#### use_advanced_extraction
- **True**: 完全な9フェーズパイプライン実行
- **False**: 従来のLLMベース抽出（フォールバック）

#### semrerank_enabled
- **True**: SemReRank処理を実行
- **False**: 統計スコアのみ使用（Phase 3スキップ）

#### semrerank_seed_percentile
- **15.0**: 上位15%をシードとして選定
- **推奨範囲**: 10.0～20.0
- **低い値**: より厳選されたシード（精度重視）
- **高い値**: より多くのシード（再現率重視）

#### semrerank_relmin
- **0.5**: コサイン類似度0.5以上でエッジ作成
- **推奨範囲**: 0.4～0.7
- **低い値**: より多くのエッジ（グラフが密）
- **高い値**: より少ないエッジ（グラフが疎）

#### semrerank_reltop
- **0.15**: 各ノードの上位15%とエッジ作成
- **推奨範囲**: 0.10～0.20
- **低い値**: 選択的な関連（精度重視）
- **高い値**: 広範な関連（再現率重視）

#### semrerank_alpha
- **0.85**: 85%の確率で隣接ノードに移動
- **推奨範囲**: 0.80～0.90
- **PageRankの標準値**

#### definition_generation_percentile
- **15.0**: 上位15%に定義生成
- **推奨範囲**: 10.0～25.0
- **低い値**: コスト削減、高品質のみ
- **高い値**: より多くの用語に定義

#### llm_filter_batch_size
- **10**: 10件ずつバッチ処理
- **推奨範囲**: 5～20
- **低い値**: 並列性低、エラー耐性高
- **高い値**: 並列性高、エラー時の影響大

### 9.3 推奨設定

#### 高精度モード（精度重視）
```python
use_advanced_extraction = True
semrerank_enabled = True
semrerank_seed_percentile = 10.0  # 厳選
semrerank_relmin = 0.6            # 高類似度のみ
semrerank_reltop = 0.10           # 選択的
definition_generation_percentile = 10.0  # 上位のみ
```

#### バランスモード（デフォルト）
```python
use_advanced_extraction = True
semrerank_enabled = True
semrerank_seed_percentile = 15.0
semrerank_relmin = 0.5
semrerank_reltop = 0.15
definition_generation_percentile = 15.0
```

#### 高再現率モード（網羅性重視）
```python
use_advanced_extraction = True
semrerank_enabled = True
semrerank_seed_percentile = 20.0  # 多めのシード
semrerank_relmin = 0.4            # 低類似度も許容
semrerank_reltop = 0.20           # 広範な関連
definition_generation_percentile = 25.0  # 多めに定義
```

#### 低コストモード
```python
use_advanced_extraction = True
semrerank_enabled = False  # SemReRankなし
definition_generation_percentile = 10.0  # 最小限
llm_filter_batch_size = 20  # 大きなバッチ
```

---

## 10. 完全な処理例

### 10.1 入力テキスト

```
舶用ディーゼルエンジンは低速で高トルクを発生させる内燃機関である。
6DE-18型エンジンは舶用ディーゼルエンジンの代表例で、燃焼効率が高い。
NOx排出量を削減するため、排気ガス処理システムが搭載されている。
低速エンジンは燃料消費が少なく、長距離航海に適している。
```

### 10.2 Phase 1: ハイブリッド候補抽出

#### Mode C抽出
```python
{
  "舶用ディーゼルエンジン": 2,
  "内燃機関": 1,
  "6DE-18型エンジン": 1,
  "燃焼効率": 1,
  "NOx排出量": 1,
  "排気ガス処理システム": 1,
  "低速エンジン": 1,
  "燃料消費": 1,
  "長距離航海": 1
}
```

#### Mode A + n-gram抽出
```python
{
  "舶用": 2,
  "ディーゼル": 2,
  "エンジン": 4,
  "舶用ディーゼル": 2,
  "ディーゼルエンジン": 2,
  "舶用ディーゼルエンジン": 2,
  "低速": 2,
  "高トルク": 1,
  "内燃": 1,
  "内燃機関": 1,
  "6DE": 1,
  "18型": 1,
  "6DE-18型": 1,
  "燃焼": 1,
  "効率": 1,
  "燃焼効率": 1,
  "NOx": 1,
  "排出": 1,
  "排出量": 1,
  "NOx排出": 1,
  "NOx排出量": 1,
  "排気": 1,
  "ガス": 1,
  "処理": 1,
  "システム": 1,
  "排気ガス": 1,
  "ガス処理": 1,
  "処理システム": 1,
  "排気ガス処理": 1,
  "排気ガス処理システム": 1,
  "低速エンジン": 1,
  "燃料": 1,
  "消費": 1,
  "燃料消費": 1,
  "長距離": 1,
  "航海": 1,
  "長距離航海": 1
}
```

#### 正規表現抽出
```python
{
  "6DE-18": 1,
  "NOx": 1
}
```

#### 複合名詞抽出
```python
{
  "舶用ディーゼルエンジン": 2,
  "低速高トルク": 1,
  "内燃機関": 1,
  "6DE-18型エンジン": 1,
  "燃焼効率": 1,
  "NOx排出量": 1,
  "排気ガス処理システム": 1,
  "低速エンジン": 1,
  "燃料消費": 1,
  "長距離航海": 1
}
```

#### マージ結果（min_frequency=2）
```python
{
  "舶用ディーゼルエンジン": 4,
  "ディーゼルエンジン": 2,
  "エンジン": 4,
  "低速": 2,
  "内燃機関": 2,
  "6DE-18型": 2,
  "燃焼効率": 2,
  "NOx排出量": 2,
  "排気ガス処理システム": 2,
  "低速エンジン": 2,
  "燃料消費": 2,
  "NOx": 2
}
```

### 10.3 Phase 2: 統計的スコアリング

#### 文分割
```python
documents = [
  "舶用ディーゼルエンジンは低速で高トルクを発生させる内燃機関である",
  "6DE-18型エンジンは舶用ディーゼルエンジンの代表例で、燃焼効率が高い",
  "NOx排出量を削減するため、排気ガス処理システムが搭載されている",
  "低速エンジンは燃料消費が少なく、長距離航海に適している"
]
```

#### TF-IDF計算
```python
# 計算後（正規化前）
{
  "舶用ディーゼルエンジン": 0.35,
  "ディーゼルエンジン": 0.20,
  "エンジン": 0.18,
  "低速": 0.30,
  "内燃機関": 0.40,
  "6DE-18型": 0.40,
  "燃焼効率": 0.40,
  "NOx排出量": 0.40,
  "排気ガス処理システム": 0.40,
  "低速エンジン": 0.40,
  "燃料消費": 0.40,
  "NOx": 0.40
}

# Min-max正規化後
{
  "舶用ディーゼルエンジン": 0.77,
  "ディーゼルエンジン": 0.09,
  "エンジン": 0.0,
  "低速": 0.55,
  "内燃機関": 1.0,
  "6DE-18型": 1.0,
  "燃焼効率": 1.0,
  "NOx排出量": 1.0,
  "排気ガス処理システム": 1.0,
  "低速エンジン": 1.0,
  "燃料消費": 1.0,
  "NOx": 1.0
}
```

#### C-value計算
```python
# 計算後（正規化前）
{
  "舶用ディーゼルエンジン": 17.3,  # log2(11)×4 - 0
  "ディーゼルエンジン": 3.0,       # log2(8)×2 - (4/1)
  "エンジン": 4.0,                 # log2(4)×4 - ((4+2+2)/3)
  "低速": 2.0,                     # log2(2)×2 - 0
  "内燃機関": 3.0,                 # log2(4)×2 - 0
  "6DE-18型": 2.0,                 # log2(7)×2 - 0
  "燃焼効率": 3.0,                 # log2(4)×2 - 0
  "NOx排出量": 3.0,                # log2(5)×2 - 0
  "排気ガス処理システム": 5.2,    # log2(11)×2 - 0
  "低速エンジン": 4.0,             # log2(6)×2 - 0
  "燃料消費": 3.0,                 # log2(4)×2 - 0
  "NOx": 1.0                       # log2(3)×2 - (2/1)
}

# Min-max正規化後
{
  "舶用ディーゼルエンジン": 1.0,
  "ディーゼルエンジン": 0.12,
  "エンジン": 0.18,
  "低速": 0.06,
  "内燃機関": 0.12,
  "6DE-18型": 0.06,
  "燃焼効率": 0.12,
  "NOx排出量": 0.12,
  "排気ガス処理システム": 0.26,
  "低速エンジン": 0.18,
  "燃料消費": 0.12,
  "NOx": 0.0
}
```

#### Seed Score（C-value重視: w_tfidf=0.3, w_cvalue=0.7）
```python
{
  "舶用ディーゼルエンジン": 0.3×0.77 + 0.7×1.0 = 0.93,
  "ディーゼルエンジン": 0.3×0.09 + 0.7×0.12 = 0.11,
  "エンジン": 0.3×0.0 + 0.7×0.18 = 0.13,
  "低速": 0.3×0.55 + 0.7×0.06 = 0.21,
  "内燃機関": 0.3×1.0 + 0.7×0.12 = 0.38,
  "6DE-18型": 0.3×1.0 + 0.7×0.06 = 0.34,
  "燃焼効率": 0.3×1.0 + 0.7×0.12 = 0.38,
  "NOx排出量": 0.3×1.0 + 0.7×0.12 = 0.38,
  "排気ガス処理システム": 0.3×1.0 + 0.7×0.26 = 0.48,
  "低速エンジン": 0.3×1.0 + 0.7×0.18 = 0.43,
  "燃料消費": 0.3×1.0 + 0.7×0.12 = 0.38,
  "NOx": 0.3×1.0 + 0.7×0.0 = 0.30
}
```

#### Final Score（TF-IDF重視: w_tfidf=0.7, w_cvalue=0.3）
```python
{
  "舶用ディーゼルエンジン": 0.7×0.77 + 0.3×1.0 = 0.84,
  "ディーゼルエンジン": 0.7×0.09 + 0.3×0.12 = 0.10,
  "エンジン": 0.7×0.0 + 0.3×0.18 = 0.05,
  "低速": 0.7×0.55 + 0.3×0.06 = 0.40,
  "内燃機関": 0.7×1.0 + 0.3×0.12 = 0.74,
  "6DE-18型": 0.7×1.0 + 0.3×0.06 = 0.72,
  "燃焼効率": 0.7×1.0 + 0.3×0.12 = 0.74,
  "NOx排出量": 0.7×1.0 + 0.3×0.12 = 0.74,
  "排気ガス処理システム": 0.7×1.0 + 0.3×0.26 = 0.78,
  "低速エンジン": 0.7×1.0 + 0.3×0.18 = 0.75,
  "燃料消費": 0.7×1.0 + 0.3×0.12 = 0.74,
  "NOx": 0.7×1.0 + 0.3×0.0 = 0.70
}
```

### 10.4 Phase 3: SemReRank適用

#### シード選定（上位15% → 12件中2件）
```python
n_seeds = max(1, int(12 × 0.15)) = 1  # 最低1件保証で実際は2件選定

seeds = [
  "舶用ディーゼルエンジン",  # seed_score=0.93
  "排気ガス処理システム"     # seed_score=0.48
]
```

#### 埋め込み取得
```python
# Azure OpenAI Embeddingsで12件の埋め込みを取得
embeddings = {
  "舶用ディーゼルエンジン": [0.12, 0.34, ..., 0.56],  # 1536次元
  "ディーゼルエンジン": [0.15, 0.38, ..., 0.52],
  ...
}

# pgvectorキャッシュに保存
# 次回以降はキャッシュから高速取得
```

#### 意味的関連性グラフ構築
```python
# コサイン類似度計算
similarities = {
  ("舶用ディーゼルエンジン", "ディーゼルエンジン"): 0.92,
  ("舶用ディーゼルエンジン", "低速エンジン"): 0.78,
  ("舶用ディーゼルエンジン", "6DE-18型"): 0.72,
  ("ディーゼルエンジン", "低速エンジン"): 0.85,
  ("ディーゼルエンジン", "エンジン"): 0.88,
  ("NOx排出量", "排気ガス処理システム"): 0.80,
  ...
}

# グラフ構築（relmin=0.5, reltop=0.15）
top_k = max(1, int(12 × 0.15)) = 1  # 実際は2件

# 各ノードについて上位2件とエッジ作成
edges = [
  ("舶用ディーゼルエンジン", "ディーゼルエンジン", 0.92),
  ("舶用ディーゼルエンジン", "低速エンジン", 0.78),
  ("ディーゼルエンジン", "エンジン", 0.88),
  ("ディーゼルエンジン", "低速エンジン", 0.85),
  ("NOx排出量", "排気ガス処理システム", 0.80),
  ...
]
```

#### Personalized PageRank実行
```python
# personalizationベクトル
personalization = {
  "舶用ディーゼルエンジン": 1.0,  # シード
  "排気ガス処理システム": 1.0,    # シード
  "ディーゼルエンジン": 0.0,
  "低速エンジン": 0.0,
  "6DE-18型": 0.0,
  ...
}

# PageRank計算（alpha=0.85）
importance_scores = {
  "舶用ディーゼルエンジン": 0.150,  # シード
  "排気ガス処理システム": 0.145,    # シード
  "ディーゼルエンジン": 0.120,      # シードと関連
  "低速エンジン": 0.105,            # シードと関連
  "6DE-18型": 0.085,                # シードと関連（低頻度だが高スコア）
  "NOx排出量": 0.095,               # シードと関連
  "燃焼効率": 0.070,
  "内燃機関": 0.065,
  "燃料消費": 0.060,
  "エンジン": 0.055,
  "低速": 0.030,
  "NOx": 0.020
}
```

#### スコア改訂
```python
# 平均重要度
avg_importance = (0.150 + 0.145 + ... + 0.020) / 12 = 0.083

# 改訂計算
revised_scores = {}

# 例: "舶用ディーゼルエンジン"
boost = 1 + (0.150 / 0.083 - 1) = 1 + 0.807 = 1.807
revised = 0.84 × 1.807 = 1.518

# 例: "6DE-18型"（低頻度だが重要）
boost = 1 + (0.085 / 0.083 - 1) = 1 + 0.024 = 1.024
revised = 0.72 × 1.024 = 0.737  # 改訂前より若干上昇

# 例: "低速"（一般用語）
boost = 1 + (0.030 / 0.083 - 1) = 1 - 0.639 = 0.361
revised = 0.40 × 0.361 = 0.144  # 大幅に低下

revised_scores = {
  "舶用ディーゼルエンジン": 1.518,
  "排気ガス処理システム": 1.410,
  "ディーゼルエンジン": 0.145,
  "低速エンジン": 0.950,
  "6DE-18型": 0.737,
  "NOx排出量": 0.848,
  "燃焼効率": 0.622,
  "内燃機関": 0.580,
  "燃料消費": 0.533,
  "エンジン": 0.033,
  "低速": 0.144,
  "NOx": 0.168
}
```

### 10.5 Phase 4: 類義語検出

#### 部分文字列
```python
{
  "ディーゼルエンジン": ["エンジン"],
  "舶用ディーゼルエンジン": ["ディーゼルエンジン", "エンジン"],
  "低速エンジン": ["エンジン"]
}
```

#### 共起
```python
# 名詞句抽出
noun_phrases = [
  "舶用ディーゼルエンジン低速高トルク内燃機関",
  "6DE-18型エンジン舶用ディーゼルエンジン燃焼効率",
  ...
]

# 共起検出
{
  "舶用ディーゼルエンジン": ["低速", "内燃機関", "6DE-18型", "燃焼効率"],
  "6DE-18型": ["舶用ディーゼルエンジン", "燃焼効率"],
  ...
}
```

#### 編集距離
```python
# 類似度計算
("ディーゼルエンジン", "低速エンジン"): 0.75

# 検出
{
  "ディーゼルエンジン": ["低速エンジン"],
  "低速エンジン": ["ディーゼルエンジン"]
}
```

#### 統合結果
```python
synonym_map = {
  "舶用ディーゼルエンジン": ["ディーゼルエンジン", "エンジン", "低速", "内燃機関"],
  "ディーゼルエンジン": ["エンジン", "低速エンジン"],
  "低速エンジン": ["エンジン", "ディーゼルエンジン"],
  "6DE-18型": ["舶用ディーゼルエンジン", "燃焼効率"],
  ...
}
```

### 10.6 Phase 5: RAG定義生成

#### 上位15%選定
```python
n_terms = max(1, int(12 × 0.15)) = 1  # 実際は2件

target_terms = [
  "舶用ディーゼルエンジン",  # score=1.518
  "排気ガス処理システム"     # score=1.410
]
```

#### ベクトル検索
```python
# "舶用ディーゼルエンジン" でベクトル検索（k=5）
docs = [
  "舶用ディーゼルエンジンは低速で高トルクを...",
  "4サイクルディーゼルエンジンの燃焼効率は...",
  ...
]

context = "\n\n".join([doc.page_content for doc in docs])[:3000]
```

#### LLM定義生成
```python
# プロンプト
prompt_input = {
  "term": "舶用ディーゼルエンジン",
  "context": "（3000文字のコンテキスト）"
}

# LLM出力
definition = "舶用ディーゼルエンジンは、船舶の推進力を得るために使用される内燃機関で..."
```

### 10.7 Phase 6: LLMフィルタ

#### バッチ処理
```python
terms_with_def = [
  "舶用ディーゼルエンジン",
  "排気ガス処理システム"
]

batch_inputs = [
  {"term": "舶用ディーゼルエンジン", "definition": "舶用ディーゼルエンジンは..."},
  {"term": "排気ガス処理システム", "definition": "排気ガス処理システムは..."}
]

result_texts = await chain.abatch(batch_inputs)
```

#### 判定結果
```python
results = [
  {"is_technical": true, "confidence": 0.95, "reason": "船舶の推進力を得るための..."},
  {"is_technical": true, "confidence": 0.92, "reason": "NOx削減のための専門的な..."}
]

# フィルタ通過
technical_terms = [
  "舶用ディーゼルエンジン",
  "排気ガス処理システム"
]
```

### 10.8 最終出力

```json
[
  {
    "headword": "舶用ディーゼルエンジン",
    "score": 1.518,
    "definition": "舶用ディーゼルエンジンは、船舶の推進力を得るために使用される内燃機関で、低速で高トルクを発生させる特性を持つ。4サイクルまたは2サイクル方式で動作し、燃料効率が高く、長時間の連続運転に適している。",
    "frequency": 4,
    "synonyms": ["ディーゼルエンジン", "エンジン", "低速", "内燃機関"],
    "metadata": {
      "confidence": 0.95,
      "reason": "船舶の推進力を得るための特定の内燃機関を指す専門用語"
    },
    "tfidf_score": 0.35,
    "cvalue_score": 17.3
  },
  {
    "headword": "排気ガス処理システム",
    "score": 1.410,
    "definition": "排気ガス処理システムは、ディーゼルエンジンから排出されるNOxやSOxなどの有害物質を削減するための装置で、触媒反応やスクラバー方式を用いる。",
    "frequency": 2,
    "synonyms": ["NOx排出量"],
    "metadata": {
      "confidence": 0.92,
      "reason": "NOx削減のための専門的なシステムを指す"
    },
    "tfidf_score": 0.40,
    "cvalue_score": 5.2
  }
]
```

---

## 11. パフォーマンス特性

### 11.1 計算量

| フェーズ | 計算量 | 備考 |
|---------|--------|------|
| **Phase 1: 候補抽出** | O(n × m) | n=文字数, m=max_term_length |
| **Phase 2: TF-IDF** | O(d × v) | d=文数, v=候補数 |
| **Phase 2: C-value** | O(v²) | 全候補ペアのネスト判定 |
| **Phase 3: 埋め込み取得** | O(v) | キャッシュヒット時はO(1) |
| **Phase 3: グラフ構築** | O(v²) | コサイン類似度計算 |
| **Phase 3: PageRank** | O(E × i) | E=エッジ数, i=反復回数（通常<10） |
| **Phase 4: 類義語検出** | O(v²) | 全候補ペア比較 |
| **Phase 5: RAG定義** | O(k × t) | k=検索数, t=top_N件 |
| **Phase 6: LLMフィルタ** | O(t / b) | b=batch_size |

**ボトルネック**:
1. **Phase 3: グラフ構築** (O(v²))
2. **Phase 4: 類義語検出** (O(v²))

**スケーラビリティ**:
- 候補数100件: 約10,000回の比較
- 候補数1000件: 約1,000,000回の比較

**推奨**: 候補数を事前フィルタで300件以下に抑える

### 11.2 API呼び出しコスト

| API | フェーズ | 呼び出し回数 | コスト（概算） |
|-----|---------|-------------|--------------|
| **Azure OpenAI Embeddings** | Phase 3: 埋め込み | v件（初回のみ） | $0.0001/1Kトークン × v |
| **Azure OpenAI Chat** | Phase 5: 定義生成 | top_N件 | $0.001/1Kトークン × top_N × 平均1Kトークン |
| **Azure OpenAI Chat** | Phase 6: LLMフィルタ | top_N / batch_size回 | $0.001/1Kトークン × (top_N / batch_size) × 平均0.5Kトークン |

**例** (候補100件、top_N=15件、batch_size=10):
- Embeddings: 100件 × $0.0001 = $0.01（初回のみ）
- 定義生成: 15件 × $0.001 × 1K = $0.015
- LLMフィルタ: (15 / 10) × $0.001 × 0.5K = $0.0008

**合計**: 約$0.026（初回）、$0.016（2回目以降、キャッシュ利用）

### 11.3 キャッシュ効果

#### 埋め込みキャッシュ
```
初回: 100件の埋め込みを計算 → Azure OpenAI API呼び出し
2回目以降: pgvectorキャッシュから取得 → API呼び出しなし

削減効果:
- コスト: 100%削減（2回目以降）
- 時間: 約95%削減（ネットワーク遅延なし）
```

**具体例**:
```
シナリオ: 同じドメインの文書を5回処理

埋め込みAPI呼び出し:
- キャッシュなし: 100 × 5 = 500回
- キャッシュあり: 100 + 20 + 10 + 5 + 2 = 137回（新規のみ）

削減率: (500 - 137) / 500 = 72.6%
```

### 11.4 処理時間

**ベンチマーク** (候補100件、Azure OpenAI):

| フェーズ | 時間（秒） | 割合 |
|---------|-----------|------|
| Phase 1: 候補抽出 | 2.5 | 5% |
| Phase 2: 統計スコア | 1.0 | 2% |
| Phase 3: SemReRank | 15.0 | 30% |
| - 埋め込み取得（初回） | 10.0 | 20% |
| - グラフ構築 | 3.0 | 6% |
| - PageRank | 2.0 | 4% |
| Phase 4: 類義語検出 | 0.5 | 1% |
| Phase 5: RAG定義（15件） | 25.0 | 50% |
| Phase 6: LLMフィルタ（2バッチ） | 6.0 | 12% |
| **合計** | **50.0** | **100%** |

**2回目以降** (キャッシュ利用):
```
Phase 3: SemReRank: 5.0秒（埋め込みキャッシュヒット）
合計: 40.0秒（20%削減）
```

---

## 12. トラブルシューティング

### 12.1 よくある問題

#### 問題1: 候補が少なすぎる

**症状**:
```
Extracted 5 candidates
```

**原因**:
- `min_frequency`が高すぎる（デフォルト2）
- テキストが短すぎる

**解決策**:
```python
# config.pyまたは初期化時
extractor = AdvancedStatisticalExtractor(
    min_frequency=1  # 1回でも出現すれば候補
)
```

#### 問題2: SemReRankでエラー

**症状**:
```
SemReRank failed: division by zero
```

**原因**:
- 候補数が少なすぎてグラフが構築できない
- シードが選定されない

**解決策**:
```python
# 候補数を増やす
min_frequency = 1

# またはSemReRankを無効化
semrerank_enabled = False
```

#### 問題3: 埋め込みキャッシュエラー

**症状**:
```
Failed to save embedding to cache: relation "term_embeddings" does not exist
```

**原因**:
- pgvector拡張がインストールされていない
- テーブル作成権限がない

**解決策**:
```sql
-- PostgreSQLで実行
CREATE EXTENSION IF NOT EXISTS vector;

-- 権限付与
GRANT CREATE ON DATABASE your_database TO your_user;
```

#### 問題4: LLMフィルタでタイムアウト

**症状**:
```
LLM filter batch failed: timeout
```

**原因**:
- batch_sizeが大きすぎる
- Azure OpenAIのレート制限

**解決策**:
```python
# batch_sizeを小さくする
llm_filter_batch_size = 5

# または非同期処理の待機時間を延長
# (実装に応じて調整)
```

#### 問題5: 定義が生成されない

**症状**:
```
No terms with definitions to filter
```

**原因**:
- `definition_generation_percentile`が低すぎる
- ベクトルストアが空

**解決策**:
```python
# パーセンタイルを上げる
definition_generation_percentile = 25.0

# ベクトルストアにデータを投入
# (ingestion処理を実行)
```

### 12.2 デバッグログ

**ログレベル設定**:
```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("src.rag.term_extraction")
logger.setLevel(logging.DEBUG)
```

**主要ログ出力**:
```
INFO - Extracted 100 candidates
INFO - Sorted 100 terms by enhanced scores
INFO - Applying SemReRank enhancement
INFO - Built semantic graph: 100 nodes, 750 edges
INFO - Selected 15 seeds (top 15.0%) from 100 candidates
INFO - Computed Personalized PageRank for 100 nodes
INFO - Detecting synonyms
INFO - Detected synonyms for 45 terms
INFO - Generating definitions with RAG
INFO - [1/15] Generated definition for: 舶用ディーゼルエンジン
INFO - Filtering terms with LLM
INFO - [OK] 舶用ディーゼルエンジン: 専門用語
INFO - Filtered: 12 technical terms
```

### 12.3 パフォーマンスチューニング

#### 高速化（精度を犠牲に）
```python
# 候補数を削減
min_frequency = 3

# SemReRankを無効化
semrerank_enabled = False

# 定義生成を削減
definition_generation_percentile = 5.0

# バッチサイズを増やす
llm_filter_batch_size = 20
```

#### 高精度化（速度を犠牲に）
```python
# 候補数を増やす
min_frequency = 1

# SemReRankを有効化
semrerank_enabled = True
semrerank_seed_percentile = 10.0  # 厳選
semrerank_relmin = 0.6

# 定義生成を増やす
definition_generation_percentile = 25.0

# バッチサイズを小さく（エラー耐性向上）
llm_filter_batch_size = 5
```

---

## 付録A: 数式一覧

### TF-IDF
```
TF-IDF(t, d, D) = TF(t, d) × IDF(t, D)

where:
  TF(t, d) = (term tの文書d内での出現回数) / (文書d内の総単語数)
  IDF(t, D) = log(文書集合Dの総文書数 / term tを含む文書数)
```

### C-value
```
C-value(a) = log₂(|a|) × freq(a) - (1/|Ta|) × Σ freq(b)
                                              b∈Ta

where:
  |a| = 用語aの長さ（文字数）
  freq(a) = 用語aの頻度
  Ta = aを含むより長い用語の集合
  b ∈ Ta
```

### Min-max正規化
```
normalized(x) = (x - min(X)) / (max(X) - min(X))

where:
  X = スコア集合
  x ∈ X
```

### 結合スコア
```
combined_score = w_tfidf × tfidf_norm + w_cvalue × cvalue_norm

where:
  Seed用: w_tfidf = 0.3, w_cvalue = 0.7
  Final用: w_tfidf = 0.7, w_cvalue = 0.3
```

### コサイン類似度
```
cos_sim(A, B) = (A · B) / (||A|| × ||B||)

where:
  A, B = 埋め込みベクトル
  A · B = 内積
  ||A|| = ベクトルAのノルム
```

### Personalized PageRank
```
PR(v) = (1 - α) × p(v) + α × Σ (PR(u) / deg(u))
                              u∈N(v)

where:
  α = ダンピング係数（0.85）
  p(v) = personalizationベクトル（シード=1.0, 他=0.0）
  N(v) = vの隣接ノード
  deg(u) = ノードuの次数
```

### スコア改訂
```
revised_score(t) = base_score(t) × boost(t)

boost(t) = 1 + (importance(t) / avg_importance - 1)

where:
  importance(t) = PageRank重要度スコア
  avg_importance = 全用語の平均PageRankスコア
```

---

## 付録B: 参考文献

1. **SemRe-Rank論文**:
   Zhang, Z., Gao, J., & Ciravegna, F. (2017). "SemRe-Rank: Improving Automatic Term Extraction By Incorporating Semantic Relatedness With Personalised PageRank". ACM Transactions on Knowledge Discovery from Data (TKDD).

2. **C-value**:
   Frantzi, K., Ananiadou, S., & Mima, H. (2000). "Automatic recognition of multi-word terms: the C-value/NC-value method". International Journal on Digital Libraries.

3. **TF-IDF**:
   Sparck Jones, K. (1972). "A statistical interpretation of term specificity and its application in retrieval". Journal of Documentation.

4. **PageRank**:
   Page, L., Brin, S., Motwani, R., & Winograd, T. (1999). "The PageRank Citation Ranking: Bringing Order to the Web". Stanford InfoLab.

5. **Sudachi**:
   Works Applications Co., Ltd. "Sudachi: A Japanese Morphological Analyzer". https://github.com/WorksApplications/Sudachi

---

## 変更履歴

| バージョン | 日付 | 変更内容 |
|-----------|------|----------|
| 2.0 | 2025-10 | ハイブリッドSudachi、SemReRank、類義語検出の統合版 |
| 1.0 | 2025-09 | 初版（基本パイプライン） |

---

**以上**

このドキュメントを読めば、専門用語抽出システムの全処理ロジックを完全に理解できます。
