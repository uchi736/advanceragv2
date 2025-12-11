# データベースセットアップガイド

本ガイドでは、Advanced RAGシステムのデータベース初期セットアップと保守について説明します。

## 目次

1. [前提条件](#前提条件)
2. [新規環境のセットアップ](#新規環境のセットアップ)
3. [既存環境のマイグレーション](#既存環境のマイグレーション)
4. [埋め込みモデル変更時の対応](#埋め込みモデル変更時の対応)
5. [トラブルシューティング](#トラブルシューティング)
6. [スキーマ詳細](#スキーマ詳細)

---

## 前提条件

### 必須要件

- **PostgreSQL 12以上**
- **pgvector拡張** - ベクトル検索に必要
- **uuid-ossp拡張** - UUID生成に必要
- **Python 3.9以上**
- **psycopg 3.x** - PostgreSQLドライバー

### マネージドデータベース（AWS RDS、Azure PostgreSQL等）の場合

pgvector拡張の有効化には特別な権限が必要です：

#### AWS RDS PostgreSQL
```sql
-- rds_superuser ロールで実行
CREATE EXTENSION vector;
CREATE EXTENSION "uuid-ossp";
```

#### Azure Database for PostgreSQL
```sql
-- azure_pg_admin ロールで実行
CREATE EXTENSION vector;
CREATE EXTENSION "uuid-ossp";
```

---

## 新規環境のセットアップ

### ステップ1: 環境変数の設定

`.env`ファイルにデータベース接続情報を設定します：

```env
DB_HOST=your-db-host.rds.amazonaws.com
DB_PORT=5432
DB_NAME=your_database
DB_USER=your_user
DB_PASSWORD=your_password
```

### ステップ2: pgvector拡張の有効化（マネージドDBの場合）

マネージドデータベースでは、スクリプト実行前に拡張を手動で有効化する必要があります：

```bash
# psqlで接続
psql -h your-db-host.rds.amazonaws.com -U your_user -d your_database

# 拡張を有効化
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

# 確認
\dx
```

### ステップ3: セットアップスクリプトの実行

```bash
python setup_database.py
```

#### 出力例（成功時）

```
============================================================
Advanced RAG System - Database Setup
============================================================
2025-12-11 19:00:00 - INFO - Connecting to database: your-db-host:5432/your_database
2025-12-11 19:00:01 - INFO - ✓ Database connection established
2025-12-11 19:00:01 - INFO - Checking required extensions...
2025-12-11 19:00:01 - INFO - ✓ pgvector extension found
2025-12-11 19:00:01 - INFO - ✓ uuid-ossp extension found
2025-12-11 19:00:01 - INFO - Executing schema from: database_schema.sql
2025-12-11 19:00:03 - INFO - ✓ Schema executed successfully
2025-12-11 19:00:03 - INFO - Verifying created tables...
2025-12-11 19:00:03 - INFO - Created tables:
2025-12-11 19:00:03 - INFO -   ✓ jargon_dictionary
2025-12-11 19:00:03 - INFO -   ✓ knowledge_edges
2025-12-11 19:00:03 - INFO -   ✓ knowledge_nodes
2025-12-11 19:00:03 - INFO - Created views:
2025-12-11 19:00:03 - INFO -   ✓ v_graph_statistics
2025-12-11 19:00:03 - INFO -   ✓ v_jargon_statistics
2025-12-11 19:00:03 - INFO -   ✓ v_term_relationships
============================================================
✅ Database setup completed successfully!
============================================================

Next steps:
  1. Run the application: streamlit run app.py
  2. Verify tables: SELECT * FROM v_graph_statistics;
```

### ステップ4: セットアップの確認

```sql
-- テーブル一覧の確認
\dt

-- 統計情報の確認
SELECT * FROM v_graph_statistics;
SELECT * FROM v_jargon_statistics;

-- カラム定義の確認
\d jargon_dictionary
\d knowledge_nodes
\d knowledge_edges
```

---

## 既存環境のマイグレーション

既存のデータベースに対しては、アプリケーション起動時に自動マイグレーションが実行されます。

### 自動マイグレーション内容

`JargonDictionaryManager`の初期化時に以下が自動実行されます：

1. **collection_name列の追加**（旧バージョンからの移行）
2. **domain列の追加**（HDBSCAN類義語抽出用）
3. **confidence_score列の削除**（廃止済み）
4. **インデックスの作成/更新**

### 手動マイグレーション（必要に応じて）

特定の列を手動で追加・削除する場合：

```sql
-- domain列の追加
ALTER TABLE jargon_dictionary ADD COLUMN IF NOT EXISTS domain TEXT;

-- 廃止されたconfidence_score列の削除
ALTER TABLE jargon_dictionary DROP COLUMN IF EXISTS confidence_score;

-- インデックスの作成
CREATE INDEX IF NOT EXISTS idx_jargon_domain ON jargon_dictionary(domain) WHERE domain IS NOT NULL;
```

---

## 埋め込みモデル変更時の対応

### 問題

`knowledge_nodes`テーブルの`embedding`列は次元数が固定されています：

```sql
embedding vector(1536)  -- text-embedding-3-small専用
```

別の埋め込みモデルを使用する場合、次元数が異なると`dimension mismatch`エラーが発生します。

### 埋め込みモデルと次元数

| モデル | 次元数 |
|--------|--------|
| text-embedding-ada-002 | 1536 |
| text-embedding-3-small | 1536 |
| text-embedding-3-large | 3072 |

### 解決方法

#### 方法1: スキーマファイルを事前修正

**ファイル**: `database_schema.sql`

```sql
-- Line 82を変更
embedding vector(3072),  -- text-embedding-3-largeの場合
```

その後、`python setup_database.py`を実行。

#### 方法2: 既存テーブルの変更

```sql
-- 既存データを削除してから列を変更
TRUNCATE TABLE knowledge_nodes CASCADE;

ALTER TABLE knowledge_nodes
ALTER COLUMN embedding TYPE vector(3072);

-- インデックスを再作成
DROP INDEX IF EXISTS idx_nodes_embedding_hnsw;
CREATE INDEX idx_nodes_embedding_hnsw
ON knowledge_nodes
USING hnsw (embedding vector_l2_ops)
WITH (m = 16, ef_construction = 64);
```

#### 方法3: テーブル再作成（推奨）

```sql
-- 既存のテーブルを削除
DROP TABLE IF EXISTS knowledge_edges CASCADE;
DROP TABLE IF EXISTS knowledge_nodes CASCADE;
```

その後、修正した`database_schema.sql`を実行。

---

## トラブルシューティング

### エラー1: "permission denied to create extension"

**原因**: pgvector拡張の作成権限がない

**解決方法**:
```sql
-- DBAに依頼して実行してもらう
CREATE EXTENSION vector;
CREATE EXTENSION "uuid-ossp";
```

**AWS RDS**の場合:
```sql
-- rds_superuserロールが必要
GRANT rds_superuser TO your_user;
```

---

### エラー2: "relation does not exist" (knowledge_nodes/knowledge_edges)

**原因**: グラフテーブルが作成されていない

**解決方法**:
```bash
# セットアップスクリプトを実行
python setup_database.py
```

または手動で：
```bash
psql -U your_user -d your_database -f database_schema.sql
```

---

### エラー3: "column confidence_score does not exist"

**原因**: UIヘルパーが廃止済みの列を参照している（修正済み）

**解決方法**:
```sql
-- 古い列が残っている場合は削除
ALTER TABLE jargon_dictionary DROP COLUMN IF EXISTS confidence_score;
```

または、アプリケーションを再起動すると自動削除されます（`JargonDictionaryManager`の初期化時）。

---

### エラー4: "dimension mismatch" (埋め込みモデル変更時)

**原因**: `embedding`列の次元数とモデルの次元数が一致しない

**解決方法**: [埋め込みモデル変更時の対応](#埋め込みモデル変更時の対応)を参照

---

### エラー5: "could not create unique index" (重複データ)

**原因**: UNIQUE制約に違反するデータが存在

**解決方法**:
```sql
-- 重複データの確認
SELECT term, collection_name, COUNT(*)
FROM jargon_dictionary
GROUP BY term, collection_name
HAVING COUNT(*) > 1;

-- 重複データを削除（IDが大きい方を削除）
DELETE FROM jargon_dictionary a
USING jargon_dictionary b
WHERE a.id > b.id
AND a.term = b.term
AND a.collection_name = b.collection_name;
```

---

## スキーマ詳細

### テーブル一覧

| テーブル名 | 説明 | 使用箇所 |
|-----------|------|----------|
| `jargon_dictionary` | 専門用語辞書 | 用語抽出、HDBSCAN類義語、UI |
| `knowledge_nodes` | ナレッジグラフのノード | グラフ構築、可視化 |
| `knowledge_edges` | ナレッジグラフのエッジ | グラフ構築、可視化 |

### jargon_dictionaryスキーマ

```sql
CREATE TABLE jargon_dictionary (
    id SERIAL PRIMARY KEY,
    collection_name VARCHAR(255) NOT NULL DEFAULT 'documents',
    term TEXT NOT NULL,
    definition TEXT NOT NULL,
    domain TEXT,                    -- 分野（HDBSCAN類義語抽出で設定）
    aliases TEXT[],                 -- 類義語リスト（HDBSCAN類義語抽出で設定）
    related_terms TEXT[],           -- 関連用語リスト
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(collection_name, term)
);
```

**重要**:
- `confidence_score`列は廃止されました
- `domain`と`aliases`はHDBSCAN類義語抽出で自動設定されます

### knowledge_nodesスキーマ

```sql
CREATE TABLE knowledge_nodes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    node_type VARCHAR(50) NOT NULL,  -- 'Term', 'Category', 'Domain', etc.
    term VARCHAR(255),
    definition TEXT,
    properties JSONB DEFAULT '{}',
    embedding vector(1536),          -- モデル変更時は次元数を調整
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

### knowledge_edgesスキーマ

```sql
CREATE TABLE knowledge_edges (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id UUID REFERENCES knowledge_nodes(id) ON DELETE CASCADE,
    target_id UUID REFERENCES knowledge_nodes(id) ON DELETE CASCADE,
    edge_type VARCHAR(50) NOT NULL,  -- 'IS_A', 'PART_OF', 'SYNONYM', etc.
    weight FLOAT DEFAULT 1.0,
    confidence FLOAT DEFAULT 1.0,
    properties JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

## ビュー一覧

### v_graph_statistics

グラフの統計情報を提供します。

```sql
SELECT * FROM v_graph_statistics;
```

**出力例**:
```
 total_nodes | term_nodes | category_nodes | total_edges | edge_types | avg_edge_weight | avg_edge_confidence
-------------+------------+----------------+-------------+------------+-----------------+--------------------
         250 |        220 |             15 |         450 |         12 |            0.85 |                0.90
```

### v_jargon_statistics

専門用語辞書の統計情報を提供します。

```sql
SELECT * FROM v_jargon_statistics;
```

**出力例**:
```
 collection_name | total_terms | terms_with_domain | terms_with_aliases | avg_aliases_per_term | terms_with_related_terms
-----------------+-------------+-------------------+--------------------+----------------------+-------------------------
 documents       |         125 |               118 |                 65 |                 3.2 |                       82
```

### v_term_relationships

用語間の関係を人間が読みやすい形式で表示します。

```sql
SELECT * FROM v_term_relationships LIMIT 10;
```

---

## メンテナンス

### インデックスの再構築

```sql
REINDEX TABLE jargon_dictionary;
REINDEX TABLE knowledge_nodes;
REINDEX TABLE knowledge_edges;
```

### VACUUM実行

```sql
VACUUM ANALYZE jargon_dictionary;
VACUUM ANALYZE knowledge_nodes;
VACUUM ANALYZE knowledge_edges;
```

### バックアップ

```bash
# 全体バックアップ
pg_dump -U your_user -d your_database > backup_$(date +%Y%m%d).sql

# 専門用語辞書のみバックアップ
pg_dump -U your_user -d your_database -t jargon_dictionary > jargon_backup.sql
```

---

## 関連ドキュメント

- [チューニングガイド](../tuning_guide.md) - パラメータ調整
- [専門用語抽出処理ロジック](../term_extraction_logic.md) - 処理フロー詳細
- [ログングガイド](./logging.md) - ログ設定

---

## バージョン履歴

- **1.0** (2025-12-11): 初版作成
  - 統合スキーマファイル作成
  - セットアップスクリプト作成
  - confidence_score列の廃止
