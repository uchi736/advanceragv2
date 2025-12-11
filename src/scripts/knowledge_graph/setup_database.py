#!/usr/bin/env python3
"""
Setup Knowledge Graph Database
Execute schema.sql to create tables and indexes
"""

import psycopg
from pathlib import Path
import os
from dotenv import load_dotenv
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _resolve_embedding_dim(default_dim: int = 1536) -> int:
    """
    埋め込み次元を Config から取得し、失敗時はデフォルトを返す。
    """
    try:
        # 遅延インポートで循環依存を避ける
        from rag.config import Config

        cfg = Config()
        if hasattr(cfg, "get_embedding_dimensions"):
            return int(cfg.get_embedding_dimensions())
    except Exception as e:
        logger.warning(f"Failed to resolve embedding dimension from Config, fallback to {default_dim}: {e}")
    return default_dim


def setup_database():
    """Execute schema.sql to set up database"""
    load_dotenv()
    
    # Connection parameters
    conn_params = {
        'host': os.getenv('DB_HOST'),
        'port': os.getenv('DB_PORT'),
        'dbname': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD')
    }
    
    # Read schema file
    schema_path = Path(__file__).parent / 'schema.sql'
    with open(schema_path, 'r', encoding='utf-8') as f:
        schema_sql = f.read()

    # 動的に埋め込み次元を差し込む（スキーマ側は {{EMBED_DIM}} プレースホルダを想定）
    embed_dim = _resolve_embedding_dim()
    schema_sql = schema_sql.replace("{{EMBED_DIM}}", str(embed_dim))
    
    try:
        # Connect and execute
        with psycopg.connect(**conn_params) as conn:
            with conn.cursor() as cur:
                logger.info("Executing schema.sql...")
                cur.execute(schema_sql)
                conn.commit()
                logger.info("Database schema created successfully!")
                
                # Verify tables
                cur.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name IN ('knowledge_nodes', 'knowledge_edges')
                    ORDER BY table_name
                """)
                tables = cur.fetchall()
                logger.info(f"Created tables: {[t[0] for t in tables]}")
                
    except Exception as e:
        logger.error(f"Failed to setup database: {e}")
        raise

if __name__ == "__main__":
    setup_database()
