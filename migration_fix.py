#!/usr/bin/env python3
"""
migration_fix.py
================
jargon_dictionary テーブルのマイグレーション修正スクリプト

実行内容:
1. 重複データのクリーンアップ
2. collection_name カラムの追加
3. UNIQUE制約の追加
"""

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv
import os
import logging

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def fix_jargon_dictionary_migration():
    """jargon_dictionary テーブルのマイグレーションを修正"""

    # 環境変数の読み込み
    load_dotenv()

    # データベース接続情報
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432'),
        'database': os.getenv('DB_NAME', 'postgres'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', 'your-password')
    }

    logger.info("=" * 60)
    logger.info("jargon_dictionary テーブル マイグレーション修正")
    logger.info("=" * 60)

    try:
        # データベースに接続
        conn = psycopg2.connect(**db_config)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()

        logger.info(f"データベースに接続しました: {db_config['host']}:{db_config['port']}/{db_config['database']}")

        # Step 1: 現在のテーブル構造を確認
        logger.info("\nStep 1: 現在のテーブル構造を確認")
        cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'jargon_dictionary'
            ORDER BY ordinal_position
        """)
        existing_columns = [row[0] for row in cur.fetchall()]
        logger.info(f"  現在のカラム: {existing_columns}")

        # Step 2: 重複データのクリーンアップ
        logger.info("\nStep 2: 重複データのクリーンアップ")

        # 重複データを確認
        cur.execute("""
            SELECT term, COUNT(*) as count
            FROM jargon_dictionary
            GROUP BY term
            HAVING COUNT(*) > 1
            ORDER BY count DESC, term
        """)
        duplicates = cur.fetchall()

        if duplicates:
            logger.info(f"  重複データが {len(duplicates)} 件見つかりました:")
            for term, count in duplicates[:5]:  # 最初の5件を表示
                logger.info(f"    - {term}: {count}件")

            # 重複を削除（最初のレコードのみ残す）
            cur.execute("""
                DELETE FROM jargon_dictionary
                WHERE id NOT IN (
                    SELECT MIN(id)
                    FROM jargon_dictionary
                    GROUP BY term
                )
            """)
            deleted_count = cur.rowcount
            logger.info(f"  ✅ {deleted_count} 件の重複データを削除しました")
        else:
            logger.info("  ✅ 重複データはありません")

        # Step 3: collection_name カラムの追加
        if 'collection_name' not in existing_columns:
            logger.info("\nStep 3: collection_name カラムの追加")
            try:
                cur.execute("""
                    ALTER TABLE jargon_dictionary
                    ADD COLUMN collection_name VARCHAR(255) NOT NULL DEFAULT 'documents'
                """)
                logger.info("  ✅ collection_name カラムを追加しました")
            except psycopg2.Error as e:
                if 'already exists' in str(e):
                    logger.info("  ℹ️ collection_name カラムは既に存在します")
                else:
                    raise
        else:
            logger.info("\nStep 3: collection_name カラムは既に存在します")

        # Step 4: 既存のUNIQUE制約を確認
        logger.info("\nStep 4: 既存のUNIQUE制約を確認")
        cur.execute("""
            SELECT constraint_name, constraint_type
            FROM information_schema.table_constraints
            WHERE table_name = 'jargon_dictionary'
            AND constraint_type = 'UNIQUE'
        """)
        constraints = cur.fetchall()
        logger.info(f"  現在のUNIQUE制約: {[c[0] for c in constraints]}")

        # Step 5: 新しいUNIQUE制約の追加
        logger.info("\nStep 5: UNIQUE制約の追加/更新")

        # 既存の関連する制約を削除
        constraints_to_drop = [
            'jargon_dictionary_term_key',
            'jargon_dictionary_collection_term_key',
            'jargon_dictionary_collection_name_term_key'
        ]

        for constraint_name in constraints_to_drop:
            try:
                cur.execute(f"ALTER TABLE jargon_dictionary DROP CONSTRAINT IF EXISTS {constraint_name}")
                if cur.rowcount > 0:
                    logger.info(f"  削除: {constraint_name}")
            except psycopg2.Error:
                pass  # 制約が存在しない場合は無視

        # 新しい複合UNIQUE制約を追加
        try:
            cur.execute("""
                ALTER TABLE jargon_dictionary
                ADD CONSTRAINT jargon_dictionary_collection_term_key
                UNIQUE(collection_name, term)
            """)
            logger.info("  ✅ UNIQUE制約 (collection_name, term) を追加しました")
        except psycopg2.Error as e:
            if 'already exists' in str(e):
                logger.info("  ℹ️ UNIQUE制約は既に存在します")
            else:
                logger.error(f"  ❌ UNIQUE制約の追加に失敗: {e}")
                raise

        # Step 6: インデックスの確認と作成
        logger.info("\nStep 6: インデックスの確認と作成")

        indexes_to_create = [
            ('idx_jargon_collection', 'collection_name'),
            ('idx_jargon_term', 'term'),
            ('idx_jargon_domain', 'domain')
        ]

        for index_name, column_name in indexes_to_create:
            try:
                if column_name == 'domain':
                    # domain カラムはNULL値を除外
                    cur.execute(f"""
                        CREATE INDEX IF NOT EXISTS {index_name}
                        ON jargon_dictionary({column_name})
                        WHERE {column_name} IS NOT NULL
                    """)
                else:
                    cur.execute(f"""
                        CREATE INDEX IF NOT EXISTS {index_name}
                        ON jargon_dictionary({column_name})
                    """)
                logger.info(f"  ✅ インデックス {index_name} を確認/作成しました")
            except psycopg2.Error as e:
                logger.warning(f"  ⚠️ インデックス {index_name} の作成に失敗: {e}")

        # Step 7: 最終的なテーブル構造の確認
        logger.info("\nStep 7: 最終的なテーブル構造の確認")
        cur.execute("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'jargon_dictionary'
            ORDER BY ordinal_position
        """)
        final_columns = cur.fetchall()
        logger.info("  最終的なカラム構造:")
        for col_name, data_type, is_nullable in final_columns:
            nullable = "NULL" if is_nullable == 'YES' else "NOT NULL"
            logger.info(f"    - {col_name}: {data_type} {nullable}")

        # Step 8: データ件数の確認
        cur.execute("SELECT COUNT(*) FROM jargon_dictionary")
        total_count = cur.fetchone()[0]
        logger.info(f"\n  総レコード数: {total_count} 件")

        # クリーンアップ
        cur.close()
        conn.close()

        logger.info("\n" + "=" * 60)
        logger.info("✅ マイグレーション修正が完了しました！")
        logger.info("=" * 60)

        return True

    except Exception as e:
        logger.error(f"\n❌ マイグレーション修正中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = fix_jargon_dictionary_migration()
    exit(0 if success else 1)