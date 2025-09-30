#!/usr/bin/env python3
"""100件の専門用語をクラスタリング結果に基づいてデータベースに投入するスクリプト"""

import json
import sys
from pathlib import Path
from datetime import datetime
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parents[1]))
from rag.config import Config

load_dotenv()
cfg = Config()

PG_URL = f"postgresql+psycopg://{cfg.db_user}:{cfg.db_password}@{cfg.db_host}:{cfg.db_port}/{cfg.db_name}"

# JSONファイルから用語を読み込み
with open("output/terms_100.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# クラスタリング結果を読み込み
with open("output/term_clusters.json", "r", encoding="utf-8") as f:
    clusters = json.load(f)

# 用語とカテゴリのマッピングを作成
term_to_category = {}
for category, info in clusters["categories"].items():
    for term in info["terms"]:
        term_to_category[term] = category if category != "未分類" else None

engine = create_engine(PG_URL)

# 既存データをクリア
with engine.begin() as conn:
    conn.execute(text("TRUNCATE TABLE jargon_dictionary"))
    print("Cleared existing data from jargon_dictionary table")

# データベースに投入
with engine.begin() as conn:
    imported_count = 0
    for term in data["terms"]:
        domain = term_to_category.get(term["headword"], None)
        
        query = text("""
            INSERT INTO jargon_dictionary (term, definition, domain, aliases, created_at, updated_at)
            VALUES (:term, :definition, :domain, :aliases, :created_at, :updated_at)
            ON CONFLICT (term) DO UPDATE SET
                definition = EXCLUDED.definition,
                domain = EXCLUDED.domain,
                aliases = EXCLUDED.aliases,
                updated_at = EXCLUDED.updated_at
        """)
        
        conn.execute(query, {
            "term": term["headword"],
            "definition": term["definition"],
            "domain": domain,
            "aliases": term["synonyms"],
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        })
        imported_count += 1
    
    # カテゴリ別の統計を表示
    category_stats = {}
    for category, info in clusters["categories"].items():
        category_stats[category] = info["count"]
    
    print(f"\nSuccessfully imported {imported_count} terms to database")
    print("\nCategory distribution:")
    for category, count in sorted(category_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {category}: {count} terms")