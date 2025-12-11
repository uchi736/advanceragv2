# 専門用語抽出・類義語抽出 チューニングガイド

本ドキュメントでは、専門用語抽出から類義語抽出までのパイプライン全体で調整可能なパラメータを整理します。

## 目次

1. [候補抽出段階](#候補抽出段階)
   - [形態素解析 (Sudachi)](#形態素解析-sudachi)
   - [正規表現パターン抽出](#正規表現パターン抽出)
   - [TF-IDF + C-value スコアリング](#tf-idf--c-value-スコアリング)
   - [テキスト分割](#テキスト分割)
2. [専門用語抽出 (SemReRank)](#専門用語抽出-semrerank)
3. [LLMフィルタリング](#llmフィルタリング)
4. [次元削減 (UMAP)](#次元削減-umap)
5. [クラスタリング (HDBSCAN)](#クラスタリング-hdbscan)
6. [類義語判定 (LLM)](#類義語判定-llm)
7. [パラメータチューニングの指針](#パラメータチューニングの指針)

---

## 候補抽出段階

専門用語候補を文書から抽出する最初のステージです。形態素解析、正規表現、統計的スコアリングを組み合わせたハイブリッドアプローチを採用しています。

### 形態素解析 (Sudachi)

Sudachi形態素解析器を使用して、日本語テキストを単語単位に分割します。

#### パラメータ一覧

| パラメータ | デフォルト値 | 設定場所 | 説明 |
|----------|------------|---------|------|
| `min_term_length` | `2` | `term_extraction.py:275` | 最小用語長（単語数） |
| `max_term_length` | `6` | `term_extraction.py:276` | 最大用語長（単語数） |
| `min_frequency` | `2` | `term_extraction.py:277` | 最小出現頻度 |
| `use_regex_patterns` | `True` | `term_extraction.py:278` | 正規表現パターンを使用するか |
| Sudachi Mode | Hybrid (A + C) | `advanced_term_extraction.py:74-75` | Mode A（短単位）+ Mode C（長単位）のハイブリッド |

#### パラメータの影響

##### `min_term_length`
- **小さくする（例: 1）**: 1単語の用語も抽出、ノイズ増加の可能性
- **大きくする（例: 3）**: 複合語のみ抽出、単純語を見逃す

**推奨範囲**: 2〜3

##### `max_term_length`
- **小さくする（例: 4）**: 短い用語のみ、処理速度向上
- **大きくする（例: 8）**: 長い複合語も抽出、計算コスト増加

**推奨範囲**: 4〜8

##### `min_frequency`
- **小さくする（例: 1）**: 1回だけ出現する用語も抽出、ノイズ増加
- **大きくする（例: 3）**: 頻出用語のみ抽出、レアな専門用語を見逃す

**推奨範囲**: 2〜5

##### Sudachi Mode
- **Mode A（短単位）**: n-gram生成に最適、詳細な分割
- **Mode C（長単位）**: 自然な複合語を捉える、粗い分割
- **Hybrid**: 両方を組み合わせて網羅性と精度を両立（推奨）

---

### 正規表現パターン抽出

形態素解析では捉えきれない技術用語を正規表現で抽出します。

#### 抽出パターン一覧

| パターン種類 | 正規表現例 | マッチ例 |
|------------|----------|---------|
| 括弧内略語 | `[（(][A-Z]{2,5}[）)]` | （BMS）、(AVR)、（EMS） |
| 型式番号 | `\b[0-9]+[A-Z]+[-_][0-9]+[A-Z]*\b` | 6DE-18、L28ADF、12V-170 |
| 化学式 | `\b(CO2\|NOx\|SOx\|PM2\.5)\b` | CO2、NOx、PM2.5 |
| 略語+用語 | `\b[A-Z]{2,5}(制御\|装置\|システム)\b` | MPPT制御、BMS装置 |
| 数値+単位 | `\b\d+(\.\d+)?\s*(kWh\|MW\|rpm)\b` | 5kWh、1800rpm |
| カタカナ複合 | `[ァ-ヴー]+(燃料\|エンジン\|システム)` | ディーゼルエンジン、ハイブリッドシステム |

#### パラメータの影響

##### `use_regex_patterns`
- **`True`**: 正規表現パターンを使用、技術用語の網羅性向上
- **`False`**: 形態素解析のみ、処理速度向上、技術用語の見逃し増加

**推奨**: `True`（技術文書では必須）

**カスタマイズ**: [advanced_term_extraction.py:149-180](src/rag/advanced_term_extraction.py#L149-180)でパターンを追加・修正可能

---

### TF-IDF + C-value スコアリング

統計的手法で候補用語の重要度をスコアリングします。

#### TF-IDF (Term Frequency - Inverse Document Frequency)

文書集合内での用語の特徴的重要度を計算します。

**計算式**:
```
TF-IDF = (1 + log(TF)) × (log((N + 1) / (DF + 1)) + 1)
```

- `TF`: 用語頻度（サブリニア圧縮で頻度10と100の差を緩和）
- `DF`: 文書頻度
- `N`: 総文書数
- Laplace平滑化でゼロ除算を防止

#### C-value (Nested Term Penalty)

複合語の重要性を評価し、他の用語に含まれる非独立語を抑制します。

**計算式**:
```
C-value = log2(|a| + 1) × freq(a) - (1/|Ta|) × Σ freq(b)
```

- `|a|`: 用語の長さ（形態素数）
- `freq(a)`: 用語の頻度
- `Ta`: aを含むより長い用語の集合
- **独立出現率フィルタ**: 30%未満の独立出現率の用語は除外

#### スコア結合

TF-IDFとC-valueを重み付き結合してファイナルスコアを計算します。

| ステージ | TF-IDF重み | C-value重み | 目的 |
|---------|-----------|------------|------|
| シード選定用 | `0.3` | `0.7` | 複合語を優先してシード選択 |
| 最終スコア用 | `0.4` | `0.6` | バランス型で最終候補を選定 |

**設定場所**: [advanced_term_extraction.py:1118-1125](src/rag/advanced_term_extraction.py#L1118-1125)

#### パラメータの影響

##### TF-IDF重み / C-value重み
- **TF-IDF重み↑**: 文書特異的な用語を優先
- **C-value重み↑**: 複合語・長い用語を優先

**推奨**: シード選定時は複合語優先（0.3/0.7）、最終スコアはバランス型（0.4/0.6）

##### 独立出現率閾値
- **デフォルト**: 30%（70%以上が複合語内出現の場合は除外）
- **小さくする（例: 20%）**: より厳格なフィルタ、非独立語を積極的に除外
- **大きくする（例: 40%）**: 緩いフィルタ、より多くの用語を保持

**設定場所**: [advanced_term_extraction.py:1042](src/rag/advanced_term_extraction.py#L1042)

---

### テキスト分割

長文書を扱いやすいチャンクに分割します。

#### パラメータ一覧

| パラメータ | デフォルト値 | 設定場所 | 説明 |
|----------|------------|---------|------|
| `chunk_size` | `2000` | `term_extraction.py:266` | チャンクの最大文字数 |
| `chunk_overlap` | `200` | `term_extraction.py:267` | チャンク間のオーバーラップ文字数 |

#### パラメータの影響

##### `chunk_size`
- **小さくする（例: 1000）**: より細かい分割、処理速度向上、文脈の断片化
- **大きくする（例: 3000）**: より広い文脈を保持、処理時間増加

**推奨範囲**: 1500〜3000

##### `chunk_overlap`
- **小さくする（例: 100）**: 重複削減、処理速度向上、チャンク境界での用語見逃しリスク
- **大きくする（例: 300）**: チャンク境界をまたぐ用語を確実に捕捉、重複増加

**推奨範囲**: 100〜300

---

## 専門用語抽出 (SemReRank)

SemReRankはSemantic Relatedness-based Re-rankingを用いて、文書から専門用語候補を抽出します。

### パラメータ一覧

| パラメータ | デフォルト値 | 設定場所 | 説明 |
|----------|------------|---------|------|
| `semrerank_enabled` | `True` | `config.py` | SemReRankの有効/無効 |
| `semrerank_seed_percentile` | `15.0` | `config.py` | シード選択のパーセンタイル（上位何%をシードとするか） |
| `semrerank_relmin` | `0.5` | `config.py` | 最小関連性閾値（これ以下の候補は除外） |
| `semrerank_reltop` | `0.15` | `config.py` | 上位関連性パーセンテージ（上位何%を選択するか） |
| `semrerank_alpha` | `0.85` | `config.py` | PageRankの減衰係数（0.85が標準） |
| `max_semrerank_candidates` | `1500` | `config.py` | 処理する候補の最大数 |

### パラメータの影響

#### `semrerank_seed_percentile`
- **小さくする（例: 10.0）**: より厳選されたシードのみ使用 → 精度向上、再現率低下
- **大きくする（例: 20.0）**: より多くのシードを使用 → 再現率向上、精度低下の可能性

#### `semrerank_relmin`
- **小さくする（例: 0.3）**: 関連性の低い用語も含める → 候補数増加、ノイズ増加
- **大きくする（例: 0.7）**: 関連性の高い用語のみ → 候補数減少、精度向上

#### `semrerank_reltop`
- **小さくする（例: 0.10）**: 上位10%のみ選択 → より厳選された用語
- **大きくする（例: 0.20）**: 上位20%を選択 → より多くの用語を抽出

#### `max_semrerank_candidates`
- **小さくする（例: 1000）**: 処理速度向上、網羅性低下
- **大きくする（例: 2000）**: 網羅性向上、処理時間増加

---

## LLMフィルタリング

SemReRankで選定された候補に対して、LLMを使って定義を生成し、真の専門用語かを判定します。

### パラメータ一覧

| パラメータ | デフォルト値 | 設定場所 | 説明 |
|----------|------------|---------|------|
| `definition_generation_percentile` | `50.0` | `config.py:127` | 定義生成を行う候補の上位パーセンテージ |
| `llm_filter_batch_size` | `50` | `config.py:129` | LLM処理のバッチサイズ |
| `max_concurrent_llm_requests` | `30` | `config.py:132` | LLM並列リクエストの最大数（TPM/RPM制限対策） |
| `enable_lightweight_filter` | `True` | `config.py:128` | 軽量フィルタを有効化するか |

### パラメータの影響

#### `definition_generation_percentile`
- **小さくする（例: 30.0）**: 上位30%のみ定義生成、LLMコスト削減、低スコア用語の見逃し
- **大きくする（例: 70.0）**: 上位70%まで定義生成、網羅性向上、LLMコスト増加

**推奨範囲**: 40.0〜60.0

#### `llm_filter_batch_size`
- **小さくする（例: 20）**: バッチサイズ削減、リクエスト数増加、RPM制限に注意
- **大きくする（例: 100）**: バッチサイズ増加、1リクエストあたりのトークン数増加、TPM制限に注意

**推奨範囲**: 30〜50（Azure OpenAI TPM/RPM制限を考慮）

#### `max_concurrent_llm_requests`
- **小さくする（例: 10）**: 並列度低下、処理時間増加、RPM制限に余裕
- **大きくする（例: 50）**: 並列度向上、処理速度向上、RPM制限に注意

**推奨範囲**:
- Azure OpenAI: 20〜30
- オンプレミスLLM: GPU数に応じて調整（例: GPU 4枚なら40〜50）

**注意**: Azure OpenAI TPM（Tokens Per Minute）とRPM（Requests Per Minute）の制限を確認してください。

#### `enable_lightweight_filter`
- **`True`**: 軽量フィルタを有効化、明らかなノイズを事前除去、LLMコスト削減
- **`False`**: 全候補をLLMに送信、網羅性向上、LLMコスト増加

**推奨**: `True`（コスト最適化のため）

---

## 次元削減 (UMAP)

UMAPは高次元の埋め込みベクトル（1536次元）を低次元空間（通常20次元）に削減します。

### パラメータ一覧

| パラメータ | デフォルト値 | 設定場所 | 説明 |
|----------|------------|---------|------|
| `n_components` | `min(20, max(2, n_samples // 2))` | `term_clustering_analyzer.py:497` | 削減後の次元数 |
| `n_neighbors` | `min(15, max(2, n_samples // 3))` | `term_clustering_analyzer.py:498` | 局所的な近傍点の数 |
| `min_dist` | `0.1` | `term_clustering_analyzer.py:499` | 埋め込み後の点間の最小距離 |
| `metric` | `'cosine'` | `term_clustering_analyzer.py:500` | 高次元空間での距離メトリック |
| `random_state` | `42` | `term_clustering_analyzer.py:501` | 再現性のための乱数シード |

### パラメータの影響

#### `n_components`
- **小さくする（例: 10）**: よりコンパクトな表現、グローバル構造を保持、細部の情報損失
- **大きくする（例: 30）**: より詳細な情報を保持、計算コスト増加

**推奨範囲**: 10〜30（データ数に応じて動的調整）

#### `n_neighbors`
- **小さくする（例: 5）**: 局所構造を重視、小さなクラスタを検出しやすい
- **大きくする（例: 30）**: グローバル構造を重視、大きなクラスタを優先

**推奨範囲**: 5〜50（データ数の10〜30%程度）

#### `min_dist`
- **小さくする（例: 0.0）**: 点がより密集、クラスタが明確に分離
- **大きくする（例: 0.5）**: 点がより分散、クラスタ境界が曖昧

**推奨範囲**: 0.0〜0.5（0.1が標準的なバランス）

#### `metric`
- **`'cosine'`**: 方向の類似性を重視（テキスト埋め込みに最適）
- **`'euclidean'`**: 絶対的な距離を重視
- **`'manhattan'`**: L1ノルム距離

**推奨**: テキスト埋め込みには`'cosine'`が最適

---

## クラスタリング (HDBSCAN)

HDBSCANは階層的密度ベースクラスタリングアルゴリズムで、クラスタ数を事前に指定する必要がありません。

### パラメータ一覧

| パラメータ | デフォルト値 | 設定場所 | 説明 |
|----------|------------|---------|------|
| `min_cluster_size` | `max(2, int(n_samples * 0.03))` | `term_clustering_analyzer.py:512` | クラスタの最小サイズ |
| `min_samples` | `1` | `term_clustering_analyzer.py:513` | コアポイントとなるための最小近傍点数 |
| `cluster_selection_epsilon` | `0.5` | `term_clustering_analyzer.py:514` | クラスタマージの距離閾値 |
| `cluster_selection_method` | `'leaf'` | `term_clustering_analyzer.py:515` | クラスタ選択方法 |
| `metric` | `'euclidean'` | `term_clustering_analyzer.py:516` | 距離メトリック（UMAP後） |
| `allow_single_cluster` | `True` | `term_clustering_analyzer.py:517` | 全点が1つのクラスタになることを許可 |
| `prediction_data` | `True` | `term_clustering_analyzer.py:518` | 予測用データの保存 |

### パラメータの影響

#### `min_cluster_size`
- **小さくする（例: 2〜3）**: 小さなクラスタも検出、細かい分類が可能、ノイズクラスタ増加
- **大きくする（例: 5〜10）**: 大きなクラスタのみ検出、安定性向上、小規模グループ見逃し

**推奨範囲**: データ数の2〜5%程度
**影響**: 最も重要なパラメータの1つ

#### `min_samples`
- **小さくする（例: 1）**: より多くの点をコアポイントとして扱う、クラスタが形成されやすい
- **大きくする（例: 5）**: ノイズに対してロバスト、クラスタ形成が厳格

**推奨範囲**: 1〜5（小規模データセットでは1が適切）

#### `cluster_selection_epsilon`
- **小さくする（例: 0.0）**: クラスタをマージしない、より多くの小さなクラスタ
- **大きくする（例: 1.0）**: 類似クラスタを積極的にマージ、クラスタ数減少

**推奨範囲**: 0.0〜1.0（0.5が標準的なバランス）

#### `cluster_selection_method`
- **`'leaf'`**: 階層の葉ノードを選択、より多くの細かいクラスタ
- **`'eom'`（Excess of Mass）**: 最も安定したクラスタを選択、クラスタ数少なめ

**推奨**: 細かい分類が必要な場合は`'leaf'`

#### `metric`
- **`'euclidean'`**: ユークリッド距離（UMAP後の低次元空間に適切）
- **`'manhattan'`**: マンハッタン距離
- **`'cosine'`**: コサイン類似度（UMAP前の高次元空間で使用済み）

**推奨**: UMAP後は`'euclidean'`が標準

---

## 類義語判定 (LLM)

LLMを用いてクラスタ内の用語が真に類義語かを判定します。

### パラメータ一覧

| パラメータ | デフォルト値 | 設定場所 | 説明 |
|----------|------------|---------|------|
| `max_synonyms` | `10` | `extract_semantic_synonyms.py:217` | 1つの専門用語に対する最大類義語数 |
| `use_llm_naming` | `True` | `term_clustering_analyzer.py:579` | LLMでクラスタ名を生成するか |
| `use_llm_for_candidates` | `True` | `extract_semantic_synonyms.py:217` | LLMで類義語候補を判定するか |
| `llm_model` | `gpt-4.1-mini` | 環境変数 | 使用するLLMモデル |
| `embedding_model` | `text-embedding-3-small` | 環境変数 | 使用する埋め込みモデル（1536次元） |

### パラメータの影響

#### `max_synonyms`
- **小さくする（例: 5）**: 高品質な類義語のみ保存、データベース容量削減
- **大きくする（例: 20）**: より多くの類義語を保存、網羅性向上

**推奨範囲**: 5〜15

#### `use_llm_naming`
- **`True`**: より意味的に適切なクラスタ名、LLMコスト増加
- **`False`**: 頻出用語をクラスタ名に使用、コスト削減

#### `use_llm_for_candidates`
- **`True`**: LLMで類義語関係を精査、高精度、コスト増加
- **`False`**: クラスタリング結果のみで判定、低精度、高速

---

## パラメータチューニングの指針

### シナリオ別推奨設定

#### 1. 高精度・厳密な類義語抽出

専門用語の品質を最優先し、ノイズを最小化する設定です。

**候補抽出段階**:
- `min_frequency`: `3`（頻出用語のみ）
- `min_term_length`: `2`
- `max_term_length`: `6`
- TF-IDF重み / C-value重み: `0.3 / 0.7`（複合語優先）

**SemReRank**:
- `semrerank_relmin`: `0.7`（高関連性のみ）
- `semrerank_reltop`: `0.10`（上位10%のみ）
- `max_semrerank_candidates`: `1000`

**LLMフィルタリング**:
- `definition_generation_percentile`: `40.0`（上位40%のみ）
- `llm_filter_batch_size`: `30`

**HDBSCAN**:
- `min_cluster_size`: `max(3, int(n_samples * 0.05))`（大きめのクラスタのみ）
- `cluster_selection_epsilon`: `0.3`（マージを控えめに）
- `max_synonyms`: `5`（厳選された類義語のみ）

#### 2. 網羅性重視・多くの候補抽出

レアな専門用語も含めて網羅的に抽出する設定です。

**候補抽出段階**:
- `min_frequency`: `2`（低頻度用語も含む）
- `min_term_length`: `2`
- `max_term_length`: `8`（長い複合語も抽出）
- TF-IDF重み / C-value重み: `0.4 / 0.6`（バランス型）

**SemReRank**:
- `semrerank_relmin`: `0.3`（低関連性も許容）
- `semrerank_reltop`: `0.20`（上位20%まで）
- `max_semrerank_candidates`: `2000`

**LLMフィルタリング**:
- `definition_generation_percentile`: `60.0`（上位60%まで）
- `llm_filter_batch_size`: `50`

**HDBSCAN**:
- `min_cluster_size`: `max(2, int(n_samples * 0.02))`（小さなクラスタも検出）
- `cluster_selection_epsilon`: `0.7`（積極的にマージ）
- `max_synonyms`: `15`（多くの類義語を保存）

#### 3. バランス型（デフォルト推奨）

精度と網羅性のバランスを取った標準設定です。

**候補抽出段階**:
- `min_frequency`: `2`
- `min_term_length`: `2`
- `max_term_length`: `6`
- TF-IDF重み / C-value重み: `0.3 / 0.7`（シード）、`0.4 / 0.6`（最終）

**SemReRank**:
- `semrerank_relmin`: `0.5`
- `semrerank_reltop`: `0.15`
- `max_semrerank_candidates`: `1500`

**LLMフィルタリング**:
- `definition_generation_percentile`: `50.0`
- `llm_filter_batch_size`: `50`

**HDBSCAN**:
- `min_cluster_size`: `max(2, int(n_samples * 0.03))`
- `cluster_selection_epsilon`: `0.5`
- `max_synonyms`: `10`

### パフォーマンスとコストの最適化

#### 処理速度を重視する場合

**候補抽出段階**:
- `min_frequency`: `3`（候補数削減）
- `max_term_length`: `4`（n-gram計算削減）
- `chunk_size`: `1500`（チャンク数削減）

**SemReRank**:
- `max_semrerank_candidates`: `1000`（候補数削減）

**LLMフィルタリング**:
- `definition_generation_percentile`: `40.0`（LLM処理対象削減）
- `llm_filter_batch_size`: `50`（バッチサイズ最大化）
- `max_concurrent_llm_requests`: `30`（並列度向上）

**UMAP/HDBSCAN**:
- `n_components`: `10`（次元数削減）
- `use_llm_naming`: `False`（LLM呼び出し削減）

#### LLMコストを削減する場合

**LLMフィルタリング**:
- `definition_generation_percentile`: `30.0`（上位30%のみ定義生成）
- `llm_filter_batch_size`: `50`（バッチ効率化）
- `enable_lightweight_filter`: `True`（事前フィルタリング）

**HDBSCAN類義語**:
- `use_llm_naming`: `False`（クラスタ名生成をスキップ）
- `max_synonyms`: `5`（LLM処理対象を削減）

**非推奨**:
- `use_llm_for_candidates`: `False`（精度が大幅低下するため非推奨）

### トラブルシューティング

#### 専門用語が抽出されない / 候補数が少なすぎる
1. **形態素解析**: `min_frequency`を`1`に下げる
2. **正規表現**: `use_regex_patterns`: `True`を確認、パターンを追加
3. **スコアリング**: TF-IDF重みを上げる（`0.5 / 0.5`）
4. **SemReRank**: `semrerank_relmin`を下げる（例: `0.3`）、`semrerank_reltop`を上げる（例: `0.20`）

#### ノイズが多すぎる / 無意味な候補が多い
1. **形態素解析**: `min_frequency`を上げる（例: `3`）、`min_term_length`を上げる（例: `3`）
2. **スコアリング**: C-value重みを上げる（`0.2 / 0.8`）で複合語優先
3. **SemReRank**: `semrerank_relmin`を上げる（例: `0.7`）、`semrerank_reltop`を下げる（例: `0.10`）
4. **LLMフィルタリング**: `definition_generation_percentile`を下げる（例: `30.0`）

#### クラスタが形成されない（全てノイズ）
1. **HDBSCAN**: `min_cluster_size`を小さくする（例: `2`）
2. **HDBSCAN**: `cluster_selection_epsilon`を大きくする（例: `0.7`）
3. **UMAP**: `n_neighbors`を小さくする（例: `5`）
4. **候補数確認**: 候補数が少なすぎないか確認（最低20件以上推奨）

#### クラスタが大きすぎる（全て1つのクラスタ）
1. **HDBSCAN**: `min_cluster_size`を大きくする（例: データ数の5%）
2. **HDBSCAN**: `cluster_selection_epsilon`を小さくする（例: `0.3`）
3. **HDBSCAN**: `cluster_selection_method`を`'eom'`に変更
4. **UMAP**: `n_neighbors`を大きくする（例: `30`）

#### 類義語の精度が低い
1. **SemReRank**: `semrerank_relmin`を上げる（例: `0.7`）
2. **HDBSCAN**: `use_llm_for_candidates`: `True`を確認
3. **HDBSCAN**: `max_synonyms`を減らす（例: `5`）で低スコア類義語を除外
4. **UMAP**: `metric`: `'cosine'`を確認（意味的類似度に最適）

#### 処理が遅すぎる
1. **候補抽出**: `min_frequency`を上げる（例: `3`）、`max_term_length`を下げる（例: `4`）
2. **SemReRank**: `max_semrerank_candidates`を減らす（例: `1000`）
3. **LLMフィルタリング**: `llm_filter_batch_size`を増やす（例: `50`）、`max_concurrent_llm_requests`を増やす（例: `30`）
4. **UMAP**: `n_components`を減らす（例: `10`）、`n_neighbors`を減らす（例: `5`）

#### LLMのTPM/RPM制限エラー
1. **並列度**: `max_concurrent_llm_requests`を減らす（例: `10`）
2. **バッチサイズ**: `llm_filter_batch_size`を減らす（例: `20`）
3. **処理対象**: `definition_generation_percentile`を下げる（例: `30.0`）
4. **Azure OpenAI**: デプロイメントのTPM/RPM割り当てを確認・増強

---

## 設定ファイルの場所

### コード内パラメータ

| カテゴリ | ファイルパス | 主要パラメータ |
|---------|------------|--------------|
| 形態素解析・候補抽出 | `src/rag/term_extraction.py` | `min_term_length`, `max_term_length`, `min_frequency`, `use_regex_patterns`, `chunk_size`, `chunk_overlap` |
| TF-IDF・C-value | `src/rag/advanced_term_extraction.py` | TF-IDF重み, C-value重み, 独立出現率閾値 |
| 正規表現パターン | `src/rag/advanced_term_extraction.py:149-180` | 括弧内略語、型式番号、化学式などのパターン |
| SemReRank | `src/rag/config.py` | `semrerank_enabled`, `semrerank_relmin`, `semrerank_reltop`, `semrerank_alpha`, `semrerank_seed_percentile`, `max_semrerank_candidates` |
| LLMフィルタリング | `src/rag/config.py` | `definition_generation_percentile`, `llm_filter_batch_size`, `max_concurrent_llm_requests`, `enable_lightweight_filter` |
| UMAP/HDBSCAN | `src/scripts/term_clustering_analyzer.py` | `n_components`, `n_neighbors`, `min_dist`, `min_cluster_size`, `min_samples`, `cluster_selection_epsilon` |
| 類義語判定 | `src/scripts/extract_semantic_synonyms.py` | `max_synonyms`, `use_llm_naming`, `use_llm_for_candidates` |

### 環境変数（.env）

| 環境変数 | 説明 | デフォルト値 |
|---------|------|-------------|
| `EMBEDDING_MODEL_IDENTIFIER` | 埋め込みモデル | `text-embedding-3-small` |
| `LLM_MODEL_IDENTIFIER` | LLMモデル | `gpt-4.1-mini` |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI APIキー | - |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAIエンドポイント | - |
| `AZURE_OPENAI_CHAT_DEPLOYMENT_NAME` | チャットモデルデプロイ名 | - |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME` | 埋め込みモデルデプロイ名 | - |

## 関連リソース

- HDBSCAN公式ドキュメント: https://hdbscan.readthedocs.io/
- UMAP公式ドキュメント: https://umap-learn.readthedocs.io/
- SemReRank論文: "Semantic Relatedness-based Re-ranking for Term Extraction"
