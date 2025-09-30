"""
term_extractor_embeding.py
==========================
互換性レイヤー - 新しい統合モジュールへのリダイレクト
"""

import warnings
from pathlib import Path
from src.rag.term_extraction import run_extraction_pipeline
from src.rag.config import Config
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import PGVector

# 互換性のための警告
warnings.warn(
    "term_extractor_embeding.py is deprecated. "
    "Please use src.rag.term_extraction module instead.",
    DeprecationWarning,
    stacklevel=2
)

# 既存のコードとの互換性維持のため、必要な関数を提供
async def run_pipeline(input_dir: Path, output_json: Path):
    """旧インターフェースとの互換性のためのラッパー"""
    from dotenv import load_dotenv
    load_dotenv()

    cfg = Config()

    # LLMとEmbeddingsの初期化
    llm = AzureChatOpenAI(
        azure_endpoint=cfg.azure_openai_endpoint,
        api_key=cfg.azure_openai_api_key,
        api_version=cfg.azure_openai_api_version,
        azure_deployment=cfg.azure_openai_chat_deployment_name,
        temperature=0.1,
    )

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=cfg.azure_openai_endpoint,
        api_key=cfg.azure_openai_api_key,
        api_version=cfg.azure_openai_api_version,
        azure_deployment=cfg.azure_openai_embedding_deployment_name
    )

    # ベクトルストアの初期化
    pg_url = f"postgresql+psycopg://{cfg.db_user}:{cfg.db_password}@{cfg.db_host}:{cfg.db_port}/{cfg.db_name}"
    vector_store = PGVector(
        collection_name=cfg.collection_name,
        connection_string=pg_url,
        embedding_function=embeddings,
        pre_delete_collection=False
    )

    # 新しいモジュールを呼び出し
    await run_extraction_pipeline(
        input_dir, output_json,
        cfg, llm, embeddings, vector_store,
        pg_url, cfg.jargon_table_name
    )


# クラスのエクスポート（互換性のため）
from src.rag.term_extraction import (
    SynonymDetector,
    TermExtractor as TermScorer,  # 名前の互換性
    JargonDictionaryManager
)