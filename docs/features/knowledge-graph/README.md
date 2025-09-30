# ナレッジグラフ機能

## 概要
専門用語の関係性を可視化し、知識の構造を表現するナレッジグラフ機能です。

## ドキュメント一覧

### [planning.md](./planning.md)
ナレッジグラフ機能の実装計画書。
- システム設計
- データベース構造
- UI/UX設計
- 実装スケジュール

## 関連ファイル

### 実装
- `/src/scripts/knowledge_graph/graph_builder.py` - グラフ構築
- `/src/scripts/knowledge_graph/graph_visualizer.py` - 可視化
- `/src/scripts/knowledge_graph/query_expander.py` - クエリ拡張
- `/src/scripts/knowledge_graph/setup_database.py` - DB初期化

### データベース
- PostgreSQL テーブル:
  - `knowledge_nodes` - ノード情報
  - `knowledge_edges` - エッジ情報

## 使用方法
Streamlitアプリのサイドバー「グラフ設定」から：
1. モード選択（起点から探索/全体ビュー）
2. フィルタ設定
3. 「グラフを生成」をクリック