# 高度なRAGシステム - ドキュメント

## 📚 ドキュメント構成

### ⚙️ [features/](./features/) - 機能別ドキュメント
各機能の詳細仕様と使用方法。

#### 📝 [term-extraction/](./features/term-extraction/) - 専門用語抽出
- **[extraction-logic.md](./features/term-extraction/extraction-logic.md)** - 抽出ロジック詳細（シーケンス図付き）
- **[synonym-detection.md](./features/term-extraction/synonym-detection.md)** - 類義語検出アルゴリズム

#### 🕸️ [knowledge-graph/](./features/knowledge-graph/) - ナレッジグラフ
- **[planning.md](./features/knowledge-graph/planning.md)** - 実装計画とシステム設計
- **[clustering-and-relations.md](./features/knowledge-graph/clustering-and-relations.md)** - クラスタリングと関係抽出

#### 📊 [evaluation/](./features/evaluation/) - 評価システム
- **[csv-format.md](./features/evaluation/csv-format.md)** - 評価データCSVフォーマット
- **[ui-guide.md](./features/evaluation/ui-guide.md)** - 評価UI操作ガイド

#### 🔧 [semantic-synonyms.md](./features/semantic-synonyms.md) - 意味的類義語抽出
HDBSCAN密度ベースクラスタリングによる類義語抽出機能の詳細。

### 🎛️ [tuning_guide.md](./tuning_guide.md) - チューニングガイド
専門用語抽出から類義語抽出までのパイプライン全体で調整可能なパラメータを網羅的に解説。
- 形態素解析（Sudachi）
- TF-IDF + C-value スコアリング
- SemReRank
- LLMフィルタリング
- UMAP次元削減
- HDBSCANクラスタリング
- シナリオ別推奨設定
- トラブルシューティング

### 📋 [term_extraction_logic.md](./term_extraction_logic.md) - 専門用語抽出処理ロジック
専門用語抽出システムの完全な処理フローと実装詳細。
- ハイブリッドSudachi形態素解析
- 候補抽出フェーズ
- 統計的スコアリング
- SemReRank処理
- RAG定義生成
- LLMフィルタ
- パフォーマンス特性

### 🔧 [semrerank_guide.md](./semrerank_guide.md) - SemReRank実装ガイド
Semantic Relatedness-based Re-rankingの完全実装ガイド。

### 📖 [guides/](./guides/) - 実装ガイド
技術実装のベストプラクティスとガイドライン。

- **[azure-openai.md](./guides/azure-openai.md)** - Azure OpenAI統合ガイド
- **[japanese-nlp.md](./guides/japanese-nlp.md)** - 日本語NLPの課題と対策
- **[vector-search.md](./guides/vector-search.md)** - ベクトル検索実装ガイド
- **[reranking.md](./guides/reranking.md)** - リランキング技術
- **[logging.md](./guides/logging.md)** - ロギングガイド
- **[database_setup.md](./guides/database_setup.md)** - データベースセットアップガイド

### 🔬 [research/](./research/) - 研究・実験
研究成果と実験レポート。

- **[research-plan.md](./research/research-plan.md)** - 研究計画
- **[lexical-mismatch.md](./research/lexical-mismatch.md)** - 語彙ミスマッチの検証

## 🚀 クイックリンク

### よく参照されるドキュメント
1. **[チューニングガイド](./tuning_guide.md)** - パラメータ調整の完全ガイド
2. **[専門用語抽出処理ロジック](./term_extraction_logic.md)** - 処理フローの詳細
3. [専門用語抽出ロジック（シーケンス図付き）](./features/term-extraction/extraction-logic.md)
4. [評価システムUI操作ガイド](./features/evaluation/ui-guide.md)
5. [意味的類義語抽出](./features/semantic-synonyms.md)

### 開発者向け
1. [Azure OpenAI統合](./guides/azure-openai.md)
2. [日本語NLP実装](./guides/japanese-nlp.md)
3. [ベクトル検索ガイド](./guides/vector-search.md)

## 📝 ドキュメント更新履歴

- **2025-12-11**: チューニングガイドを追加、不要なドキュメントを整理
- **2025-01-22**: ドキュメント構成を整理、シーケンス図を追加
- **2025-01-22**: 類義語検出ロジックを実装に合わせて更新
- **2025-01-20**: ナレッジグラフ計画書を作成