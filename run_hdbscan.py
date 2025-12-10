#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HDBSCAN類義語抽出を実行"""

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

import asyncio
from rag.config import Config
from langchain_openai import AzureOpenAIEmbeddings

async def main():
    cfg = Config()
    print(f"=== HDBSCAN類義語抽出開始 ===")
    print(f"Collection: {cfg.collection_name}")
    print(f"Database: {cfg.db_host}")
    print()

    # Import after config loaded
    from scripts.extract_semantic_synonyms import (
        load_specialized_terms,
        load_candidate_terms_from_extraction,
        extract_and_save_semantic_synonyms
    )

    # Azure OpenAI Embeddings
    embeddings = AzureOpenAIEmbeddings(
        model=cfg.azure_openai_embedding_deployment_name,
        azure_endpoint=cfg.azure_openai_endpoint,
        api_key=cfg.azure_openai_api_key,
        api_version=cfg.azure_openai_api_version
    )

    pg_url = cfg.pgvector_connection_string
    jargon_table_name = "jargon_dictionary"
    current_collection = cfg.collection_name
    debug_file = Path("output/term_extraction_debug.json")

    print(f"Debug file: {debug_file}")
    print(f"Exists: {debug_file.exists()}")
    print()

    if not debug_file.exists():
        print(f"❌ Debug file not found: {debug_file}")
        return

    # 専門用語と候補用語を読み込み
    print("📖 Loading specialized terms from database...")
    specialized_terms = load_specialized_terms(pg_url, jargon_table_name, current_collection)
    print(f"   Loaded {len(specialized_terms)} specialized terms")

    print("📖 Loading candidate terms from debug file...")
    candidate_terms = load_candidate_terms_from_extraction(debug_file)
    print(f"   Loaded {len(candidate_terms)} candidate terms")
    print()

    if not specialized_terms:
        print("❌ No specialized terms found")
        return

    if not candidate_terms:
        print("❌ No candidate terms found")
        return

    # 類義語抽出と保存
    print("🔍 Starting HDBSCAN semantic synonym extraction...")
    print()
    synonyms_dict = await extract_and_save_semantic_synonyms(
        specialized_terms=specialized_terms,
        candidate_terms=candidate_terms,
        pg_url=pg_url,
        jargon_table_name=jargon_table_name,
        embeddings=embeddings,
        collection_name=current_collection
    )

    print()
    print("=== 完了 ===")
    print(f"Extracted semantic synonyms for {len(synonyms_dict)} terms")

if __name__ == "__main__":
    asyncio.run(main())
