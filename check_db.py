#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""データベースの内容を確認するスクリプト"""

import sys
import io
from pathlib import Path

# Windows環境でのUnicode出力設定
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent / "src"))

from sqlalchemy import create_engine, text
from rag.config import Config
from dotenv import load_dotenv
import time

def main():
    # .envファイルを読み込む
    load_dotenv()

    cfg = Config()

    print("=" * 80)
    print("PostgreSQL接続確認")
    print("=" * 80)
    print(f"Host: {cfg.db_host}")
    print(f"Port: {cfg.db_port}")
    print(f"Database: {cfg.db_name}")
    print(f"User: {cfg.db_user}")
    print()

    try:
        engine = create_engine(
            cfg.pgvector_connection_string,
            connect_args={"connect_timeout": 5}  # 5秒タイムアウト
        )

        print("🔌 データベースに接続中...")
        start_time = time.time()

        with engine.connect() as conn:
            elapsed = time.time() - start_time
            print(f"✓ 接続成功 ({elapsed:.2f}秒)")
            print()

            # jargon_dictionaryテーブルの内容確認
            print("=" * 80)
            print("jargon_dictionary テーブルの内容")
            print("=" * 80)

            result = conn.execute(
                text("""
                    SELECT term, domain, aliases
                    FROM jargon_dictionary
                    WHERE collection_name = :cname
                    ORDER BY term
                """),
                {"cname": "documents"}
            )

            rows = list(result)

            if not rows:
                print("❌ collection_name='documents' のデータが存在しません")
                print()

                # 他のコレクションを確認
                print("他のコレクション名を確認:")
                result2 = conn.execute(
                    text("SELECT DISTINCT collection_name FROM jargon_dictionary")
                )
                collections = [row[0] for row in result2]
                if collections:
                    for coll in collections:
                        print(f"  - {coll}")
                else:
                    print("  (データなし)")
            else:
                print(f"📊 {len(rows)}件のデータが存在")
                print()

                # ヘッダー
                print(f"{'用語':<30} {'分野(domain)':<25} {'類義語(aliases)'}")
                print("-" * 85)

                # データ表示
                for row in rows:
                    term = row.term or "(null)"
                    domain = row.domain or "(null)"
                    aliases = str(row.aliases) if row.aliases else "[]"

                    # 長い場合は省略
                    if len(aliases) > 30:
                        aliases = aliases[:27] + "..."

                    print(f"{term:<30} {domain:<25} {aliases}")

                print()
                print("=" * 80)
                print("統計:")
                print(f"  - domain が設定されている: {sum(1 for r in rows if r.domain)} 件")
                print(f"  - domain が NULL: {sum(1 for r in rows if not r.domain)} 件")
                print(f"  - aliases が設定されている: {sum(1 for r in rows if r.aliases)} 件")
                print(f"  - aliases が空: {sum(1 for r in rows if not r.aliases)} 件")

    except Exception as e:
        print(f"❌ エラー: {type(e).__name__}")
        print(f"   {e}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
