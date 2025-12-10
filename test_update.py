#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UPDATE処理のテスト"""

import sys
import io
from pathlib import Path

# Windows環境でのUnicode出力設定
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import create_engine, text
from rag.config import Config

def main():
    cfg = Config()
    engine = create_engine(cfg.pgvector_connection_string)

    # テスト用のダミーデータでUPDATE
    print("テストUPDATE実行:")
    print("=" * 80)

    test_updates = [
        ("ガス軸受", "機械要素（テスト）", []),
        ("スラスト軸受", "未分類（テスト）", []),
        ("電動ターボチャージャー", "未分類（テスト）", []),
    ]

    with engine.begin() as conn:
        for term, domain, aliases in test_updates:
            result = conn.execute(
                text("""
                    UPDATE jargon_dictionary
                    SET domain = :domain, aliases = :aliases
                    WHERE collection_name = :collection_name AND term = :term
                """),
                {
                    "collection_name": "documents",
                    "term": term,
                    "domain": domain,
                    "aliases": aliases
                }
            )

            if result.rowcount == 0:
                print(f"❌ UPDATE失敗: {term} (0 rows updated)")
            else:
                print(f"✓ UPDATE成功: {term} → domain='{domain}' ({result.rowcount} rows)")

    print()
    print("=" * 80)
    print("確認:")
    print("=" * 80)

    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT term, domain FROM jargon_dictionary WHERE collection_name = :collection_name ORDER BY term"),
            {"collection_name": "documents"}
        )
        for row in result:
            print(f"  {row.term:<30} {row.domain or '(null)'}")

if __name__ == "__main__":
    main()
