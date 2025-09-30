# 高度なRAGシステム (iRAG)

## 概要

このシステムは、文書検索とSQL検索を統合した高度なRAG（Retrieval-Augmented Generation）システムです。Streamlitベースの直感的なUIを提供し、Azure OpenAI Serviceを活用して、多言語対応の自然言語処理による強力な情報検索と質問応答を実現します。データ（CSV/Excel）に対するSQL検索も同時にします。

## 🎯 リファクタリング履歴

### 2025年9月23日 - コード統合と最適化
- **ファイル削減**: 45ファイル → 25ファイル（45%削減）
- **コード統合**: 専門用語処理を`term_extraction.py`に統合
- **不要コード除去**: 約2000行の重複コードを削除
- **6つの類義語検出手法**: すべて`term_extraction.py`に集約

### 2025年8月29日 - フォルダ構成リファクタリング
- **関心の分離**: 機能別にフォルダを整理し、各モジュールの責任を明確化
- **改修性向上**: 関連ファイルを近接配置し、変更影響範囲を明確化
- **テスト容易性**: 構造化によりユニットテスト作成が容易に
- **新規開発者対応**: 直感的な構造で理解しやすく

## 主な機能

- **ハイブリッド検索**: ベクトル検索とキーワード検索を組み合わせ、Reciprocal Rank Fusion (RRF) によって検索精度を向上させます。
- **日本語特化**: SudachiPyによる日本語形態素解析を使用し、日本語の文書処理に最適化されたハイブリッド検索を実現します。
- **柔軟なベクトルストア**: PGVector (PostgreSQL) とChromaDBの両方をサポート。PostgreSQLが利用できない環境でもChromaDBで動作可能です。
- **Text-to-SQL**: 自然言語のクエリを解析し、アップロードされたCSV/Excelファイルからのデータベースファイルに対して自動的にSQLクエリを生成します。
- **専門用語辞書 (Golden-Retriever)**: アップロードから専門用語をその定義を抽出し、辞書を構築。この辞書を用いてクエリの理解を深め、よりコンテキストに沿った回答を生成します。
- **複数のPDF処理エンジン**: PyMuPDF（高速）とAzure Document Intelligence（高精度）を選択可能。用途に応じて最適な処理方式を選択できます。
- **評価システム**: Recall、Precision、MRR、nDCG、Hit Rateなどの指標でRAGシステムの検索精度を定量的に評価できます。
- **ユーザーフレンドリーなUI**: タブ構成のインターフェース、メッセージ履歴、ドキュメント管理など、使いやすいストリームリットアプリケーションと洗練された設計です。

## システム構成

システムは大きく以下のコンポーネントから構成されています：

```
.
├── app.py                      # Streamlitアプリケーションのエントリポイント
├── requirements.txt            # 必要なPythonライブラリ
├── .env.example                # 環境変数の設定テンプレート
├── src/                        # メインソースコード
│   ├── core/                   # コアビジネスロジック
│   │   └── rag_system.py       # RAGシステムのファサード
│   ├── rag/                    # RAGシステムのコアモジュール
│   │   ├── chains.py           # LangChainのチェーンとプロンプト設定
│   │   ├── config.py           # 設定ファイル(Config)
│   │   ├── document_parser.py  # レガシーPDF処理（PyMuPDF）
│   │   ├── evaluator.py        # 評価システムモジュール
│   │   ├── ingestion.py        # ドキュメントの取り込みと処理
│   │   ├── term_extraction.py  # 専門用語抽出と類義語検出（統合版）
│   │   ├── retriever.py        # ハイブリッド検索リトリーバー
│   │   ├── sql_handler.py      # Text-to-SQL機能の処理
│   │   ├── text_processor.py   # 日本語テキスト処理
│   │   └── pdf_processors/     # PDF処理プロセッサ
│   │       ├── base_processor.py      # 抽象基底クラス
│   │       ├── pymupdf_processor.py   # PyMuPDF実装
│   │       └── azure_di_processor.py  # Azure Document Intelligence実装
│   ├── ui/                     # UIコンポーネント
│   │   ├── chat_tab.py         # チャット画面
│   │   ├── data_tab.py         # データ管理画面
│   │   ├── dictionary_tab.py   # 辞書管理画面
│   │   ├── documents_tab.py    # ドキュメント管理画面
│   │   ├── evaluation_tab.py   # 評価システム画面
│   │   ├── settings_tab.py     # 設定画面
│   │   ├── sidebar.py          # サイドバー
│   │   └── state.py            # セッション状態管理
│   ├── scripts/                # 拡張スクリプト
│   │   ├── term_extractor_embeding.py  # 互換性レイヤー（deprecated）
│   │   ├── term_clustering_analyzer.py # クラスタリング分析
│   │   └── knowledge_graph/            # ナレッジグラフ機能
│   └── utils/                  # ユーティリティ関数
│       ├── helpers.py          # ヘルパー関数
│       └── style.py            # UIスタイル設定
├── docs/                       # ドキュメント
│   ├── evaluation/             # 評価関連ドキュメント
│   └── architecture/           # アーキテクチャドキュメント
├── output/                     # 出力ファイル
│   ├── images/                 # 生成された画像
│   └── terms.json              # 抽出された専門用語
└── old/                        # アーカイブ（不要なファイル）
```

## インストール手順

1. **仮想環境の作成と有効化**:
    ```bash
    python -m venv myenv
    source myenv/bin/activate   # Linux/macOS
    myenv\Scripts\activate       # Windows
    ```

2. **依存関係のインストール**:
    ```bash
    pip install -r requirements.txt
    ```

3. **環境変数の設定**:
    `.env.example` ファイルをコピーして `.env` ファイルを作成し、以下の設定を記入してください。最低限、以下の設定が必要です。
    - `AZURE_OPENAI_API_KEY`
    - `AZURE_OPENAI_ENDPOINT`
    - `AZURE_OPENAI_CHAT_DEPLOYMENT_NAME`
    - `AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME`

    **ベクトルストア設定**:
    - `VECTOR_STORE_TYPE`: "pgvector" | "chromadb" (デフォルト: "pgvector")

    PGVector使用時:
    - `PG_URL` (PostgreSQLの接続URL) または `DB_*` の各項目
    - PostgreSQLにpgvector拡張が必要

    ChromaDB使用時:
    - `CHROMA_PERSIST_DIRECTORY`: ローカルストレージパス (デフォルト: "./chroma_db")
    - `CHROMA_SERVER_HOST`: ChromaDBサーバーホスト（オプション）
    - `CHROMA_SERVER_PORT`: ChromaDBサーバーポート（オプション）

    **PDF処理方式の設定（オプション）**:
    - `PDF_PROCESSOR_TYPE`: "legacy" | "pymupdf" | "azure_di" (デフォルト: "legacy")

    **Azure Document Intelligence設定（azure_di使用時のみ）**:
    - `AZURE_DI_ENDPOINT`: Azure Document IntelligenceのエンドポイントURL
    - `AZURE_DI_API_KEY`: APIキー
    - `AZURE_DI_MODEL`: 使用モデル（デフォルト: "prebuilt-layout"）
    - `SAVE_MARKDOWN`: Markdown保存フラグ（デフォルト: "false"）

## 使い方

以下のコマンドでStreamlitアプリケーションを起動します。

```bash
streamlit run app.py
```

### ベクトルストアの切り替え

PostgreSQLが使用できない環境でChromaDBを使用する場合：

1. **環境変数の更新**: `.env`ファイルで`VECTOR_STORE_TYPE=chromadb`を設定

2. **自動切り替えスクリプト**:
```bash
python scripts/vector_store_migration.py chromadb
```

3. **手動切り替え**: `.env`ファイルを直接編集してアプリケーションを再起動

**注意**: ベクトルストアを切り替えた後は、ドキュメントを再インデックス化する必要があります。

詳細は [docs/vector-store-configuration.md](docs/vector-store-configuration.md) を参照してください。

### PDF処理方式の選択

本システムは3つのPDF処理方式をサポートしています：

1. **レガシー (既存のDocumentParser)**
   - デフォルトの処理方式
   - 既存の安定した実装

2. **PyMuPDF (高速・軽量)**
   - 高速なPDF処理
   - メモリ効率が良い
   - テキスト、画像、テーブルの抽出

3. **Azure Document Intelligence (高精度)**
   - 高精度なレイアウト解析
   - Markdown形式での出力
   - 複雑な文書構造の正確な抽出
   - クラウドベースの処理

処理方式は以下の方法で選択できます：
- **UIから**: サイドバーまたは詳細設定タブで選択
- **環境変数**: `.env`ファイルで`PDF_PROCESSOR_TYPE`を設定
- **プログラム**: Configオブジェクトで`pdf_processor_type`を指定

## ナレッジグラフ エクスプローラー（新機能）

Streamlitアプリに「ナレッジグラフ エクスプローラー」を統合しました。PostgreSQLの `knowledge_nodes` / `knowledge_edges` を可視化し、関係を対話的に探索できます。

- モード
  - 起点から探索: 指定した中心用語（起点）からBFSで深さ `depth` までを可視化
  - 全体ビュー: 重みの高い順に上位エッジを俯瞰（上限はスライダーで調整、既定200）
- レイアウト
  - 階層表示（固定）。必要に応じて物理シミュレーションをON/OFF可能（既定OFF）
- フィルタ（グラフとエクスポートの両方に反映）
  - 関係タイプ: 該当するエッジのみ表示（空＝全関係）
  - エッジ重み（最小）: 指定以上の重みのエッジのみ表示
  - ノードタイプ: `Term / Category / Domain / System / Component`
  - 孤立ノードを非表示: どのエッジにも接続しないノードを隠す（中心は残す）
  - ノード名の包含/除外キーワード: ラベルに一致するノードを抽出/除外
    - 既定で「クラスタ, cluster」を除外
- エクスポート
  - HTML / JSON（Cytoscape形式）/ DOT（Graphviz）をダウンロード

環境準備（参考）

1) PostgreSQL + pgvector を有効化し、`src/scripts/knowledge_graph/schema.sql` を適用

```bash
python src/scripts/knowledge_graph/setup_database.py
```

2) 用語やクラスタリング結果からノード・エッジを作成（例）

```python
# 例: src/scripts/knowledge_graph/graph_builder.py のユーティリティを利用
```

3) アプリ起動後、サイドバーの「グラフ設定」から各種フィルタを調整して「グラフを生成」を実行

トラブルシュート

- 単一ノードしか表示されない: フィルタ（関係タイプ、重み、ノードタイプ、除外語、孤立ノード非表示）や探索深度を緩める→再生成
- ID型エラー: UUIDをPyVisに渡す際は文字列化済み（修正済）。引き続き問題がある場合はご連絡ください

## 評価システムの使用方法

RAGシステムの検索精度を評価するには、以下のスクリプトを実行します：

```bash
python src/evaluation/evaluator.py
```

### 評価機能の特徴

- **複数の評価指標**: 
  - Recall@K: 関連文書の再現率
  - Precision@K: 検索結果の精度
  - MRR (Mean Reciprocal Rank): 平均逆順位
  - nDCG (Normalized Discounted Cumulative Gain): 正規化減損累積利得
  - Hit Rate@K: ヒット率

- **複数の類似度計算手法**:
  - Azure Embedding: 埋め込みベースの類似度
  - Azure LLM: LLMベースの類似度判定
  - Text Overlap: テキストの重複度
  - Hybrid: 複数手法の組み合わせ

- **柔軟な評価方法**:
  - CSVファイルからの評価データ読み込み
  - プログラムでの直接評価
  - 結果のCSVエクスポート

### 評価データの形式

CSVファイルは以下の形式で準備してください：
- `質問`: 評価用の質問
- `想定の引用元1`, `想定の引用元2`, ...: 期待される情報源
- `チャンク1`, `チャンク2`, ...: 検索結果（オプション）

## アーキテクチャ概要

### システム全体構成

```mermaid
graph TB
    subgraph "Frontend Layer"
        UI[Streamlit UI]
        UI --> ST[State Manager]
    end
    
    subgraph "Application Layer"
        ST --> APP[app.py]
        APP --> TABS[UI Tabs]
        TABS --> CT[Chat Tab]
        TABS --> DT[Data Tab]
        TABS --> DICT[Dictionary Tab]
        TABS --> DOC[Documents Tab]
        TABS --> SET[Settings Tab]
    end
    
    subgraph "RAG Core Engine"
        CT --> CHAIN[LangChain Chains]
        CHAIN --> RET[Hybrid Retriever]
        RET --> VS[Vector Store]
        RET --> KS[Keyword Search]
        RET --> RRF[RRF Fusion]
        
        CHAIN --> SQL[SQL Handler]
        SQL --> CSV[CSV/Excel Parser]
        
        CHAIN --> JARGON[Jargon Dictionary]
        JARGON --> TERMEX[Term Extractor]
    end
    
    subgraph "Data Layer"
        VS --> |PGVector| PG[(PostgreSQL + pgvector)]
        VS --> |ChromaDB| CHROMA[(ChromaDB)]
        SQL --> SQLITE[(SQLite)]
        JARGON --> JSON[(JSON Store)]
    end
    
    subgraph "External Services"
        CHAIN --> AZURE[Azure OpenAI]
        AZURE --> GPT[GPT-4o]
        AZURE --> EMBED[text-embedding-ada-002]
    end
    
    subgraph "Evaluation System"
        EVAL[Evaluator] --> METRICS[Metrics Calculator]
        METRICS --> RECALL[Recall@K]
        METRICS --> PRECISION[Precision@K]
        METRICS --> MRR[MRR]
        METRICS --> NDCG[nDCG]
        METRICS --> HR[Hit Rate@K]
    end
```

### ハイブリッド検索の仕組み

```mermaid
graph LR
    Q[Query] --> QP[Query Processing]
    QP --> VS[Vector Search]
    QP --> KS[Keyword Search]
    
    VS --> VSR[Vector Results]
    KS --> KSR[Keyword Results]
    
    VSR --> RRF[Reciprocal Rank Fusion]
    KSR --> RRF
    
    RRF --> FR[Fused Results]
    FR --> RE[Reranker]
    RE --> FINAL[Final Results]
```

### 専門用語辞書システム

```mermaid
graph TB
    DOC[Documents] --> TE[Term Extractor]
    TE --> CV[C-Value Calculation]
    TE --> EMBED[Embedding Analysis]
    
    CV --> TERMS[Extracted Terms]
    EMBED --> TERMS
    
    TERMS --> DICT[Jargon Dictionary]
    DICT --> QE[Query Enhancement]
    QE --> SEARCH[Enhanced Search]
```

## 技術仕様

- **フロントエンド**: Streamlit
- **バックエンド**: Python 3.9+
- **ベクトルデータベース**:
  - PostgreSQL + pgvector (デフォルト)
  - ChromaDB (代替オプション)
- **言語モデル**: Azure OpenAI (GPT-4o, text-embedding-ada-002)
- **日本語処理**: SudachiPy
- **検索エンジン**: LangChain + カスタムハイブリッドリトリーバー
- **PDF処理**: 
  - PyMuPDF (fitz): 高速・軽量処理
  - Azure Document Intelligence: 高精度レイアウト解析・Markdown出力

## 専門用語抽出と自動クラスタリング機能

### 概要
本システムには、PDFなどの文書から専門用語を自動抽出し、クラスタリングによって自動分類する高度な機能が実装されています。

### 専門用語抽出機能 (`term_extraction.py`)

#### 主な機能
1. **C値アルゴリズムによる専門用語抽出**
   - 複合語の出現頻度と文脈を考慮したスコアリング
   - ネストされた用語（部分文字列）の検出と評価
   - **TF-IDFは使用せず**、専門用語抽出に特化したC値を採用
   ```python
   # C値計算式
   C-value(a) = log2(|a|) × freq(a)  # 単独出現の場合
   C-value(a) = log2(|a|) × (freq(a) - (1/|Ta|) × Σ freq(b))  # 他の用語に含まれる場合

   # |a|: 候補語aの文字数
   # freq(a): 候補語aの出現頻度
   # Ta: aを部分文字列として含むより長い候補語の集合
   ```

2. **類義語・関連語検出メソッド（6つの手法）**
   - **部分文字列関係**: 包含関係による上位・下位概念の検出
   - **共起関係分析**: 同一文脈での共起パターン分析（window_size=5）
   - **編集距離**: 類似度70-95%の語を関連語として検出
   - **語幹・語尾パターン**: 同じベースを持つ派生語の検出
   - **略語辞書マッチング**: GMP/適正製造規範などの対応関係
   - **ドメイン辞書**: 分野固有の関連語マッピング

3. **SudachiPy Mode.Cによる高精度形態素解析**
   - 最長一致による複合語の正確な抽出
   - 専門用語に多い未知語への対応
   - N-gram生成による候補語の網羅的抽出

4. **Azure OpenAI GPT-4による定義生成**
   - 抽出された専門用語に対する自動定義生成
   - 文脈を考慮した高品質な説明文の作成

### 専門用語クラスタリング機能 (`term_clustering_analyzer.py`)

#### アーキテクチャ
```
入力用語 → ベクトル化 → UMAP次元圧縮 → HDBSCAN → 階層クラスタ → カテゴリ出力
```

#### 主要コンポーネント

1. **UMAP次元圧縮**
   - 1536次元 → 20次元への非線形次元削減
   - コサイン類似度の保持により意味的関係を維持
   - **PCAではなくUMAPを採用**：非線形構造を保持し、局所的・大域的構造を両立
   ```python
   umap.UMAP(
       n_components=20,      # 圧縮後の次元数
       n_neighbors=15,       # 近傍サンプル数
       min_dist=0.1,        # クラスタ内密度制御
       metric='cosine'      # コサイン距離
   )
   ```

2. **HDBSCAN階層的密度ベースクラスタリング**
   - 自動的なクラスタ数決定（K-meansと異なり事前指定不要）
   - 任意形状のクラスタを検出可能
   - ノイズ点の自動検出と適切な処理
   - Condensed Treeによる階層構造から専門用語間の階層関係を抽出
   ```python
   hdbscan.HDBSCAN(
       min_cluster_size=2,              # 最小クラスタサイズ
       cluster_selection_epsilon=0.3,   # クラスタ選択の柔軟性
       cluster_selection_method='leaf', # より多くの点を含む
       metric='euclidean',              # 圧縮後のユークリッド距離
       allow_single_cluster=True        # 単一クラスタ許可
   )
   ```

3. **階層構造分析**
   - λ値（ラムダ値）による概念の一般性・具体性の評価
   - 最大11階層の深さで概念の粒度を表現
   - 上位概念・中間概念・具体的概念への自動分類

4. **LLMによる自動カテゴリ命名（オプション）**
   - Azure OpenAI GPT-4を使用した意味的なクラスタ名生成
   - 各クラスタ内の用語を分析して適切な名前を付与

### 実装結果（2025年9月6日）

#### 改善前後の比較
| 指標 | 改善前（次元圧縮なし） | 改善後（UMAP適用） | 改善率 |
|------|------------------------|-------------------|--------|
| クラスタ数 | 12 | 30 | +150% |
| ノイズ点 | 39 (39.8%) | 6 (6.1%) | -84.6% |
| シルエットスコア | 0.089 | 0.346 | +289% |
| 階層深度 | 10 | 11 | +10% |

#### 実際のクラスタリング例（舶用エンジン専門用語）
- **エンジン部品**: ピストン、コンロッド、カムシャフト、クランクシャフト
- **燃焼制御**: ノッキング、ミスファイア、先燃え、燃料噴射装置
- **排ガス制御**: EGRシステム、SCRシステム、NOx、水噴射
- **船舶エネルギー効率**: EEDI、EEXI、CII、SEEMP
- **海事規制**: IMO、IACS、MARPOL条約、船級協会

### 使用方法

#### 専門用語抽出

**UIから実行（推奨）**:
サイドバーの「📚 用語辞書生成」ボタンから実行

**コマンドラインから**:
```bash
# 新しい統合モジュール使用（推奨）
python -c "from src.rag.term_extraction import run_extraction_pipeline; import asyncio; asyncio.run(run_extraction_pipeline(Path('docs'), Path('output/terms.json'), ...))"

# 互換性レイヤー経由（非推奨）
python src/scripts/term_extractor_embeding.py docs output/terms.json
```

#### クラスタリング分析
```bash
python src/scripts/term_clustering_analyzer.py
```

#### データベースへのインポート
```bash
python src/scripts/import_terms_to_db.py
```

### 技術的な特徴

1. **高精度な用語抽出**
   - C-valueアルゴリズムによる統計的重要度評価
   - 複数の同義語検出手法の組み合わせ
   - 日本語特化の形態素解析

2. **意味的クラスタリング**
   - ベクトル埋め込みによる意味的類似性の捕捉
   - 次元圧縮による「次元の呪い」の回避
   - 密度ベースによる自然なグループ形成

3. **階層的概念構造**
   - Condensed Treeによる統計的階層抽出
   - λ値による概念の抽象度評価
   - 自動的な上位・下位概念の識別

4. **スケーラビリティ**
   - 数千〜数万の用語に対応可能
   - バッチ処理による効率的な処理
   - PostgreSQLによる永続化

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。
