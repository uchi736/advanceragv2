#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database Setup Script
=====================
Sets up the complete database schema for the Advanced RAG system.

Usage:
    python setup_database.py

Requirements:
    - PostgreSQL 12+ with pgvector extension
    - .env file with database credentials
"""

import sys
import psycopg
from pathlib import Path
from dotenv import load_dotenv
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_extensions(conn):
    """Check if required extensions are available"""
    logger.info("Checking required extensions...")

    try:
        # Check pgvector
        with conn.cursor() as cur:
            cur.execute("""
                SELECT EXISTS(
                    SELECT 1 FROM pg_extension WHERE extname = 'vector'
                )
            """)
            has_vector = cur.fetchone()[0]

            if not has_vector:
                logger.warning("⚠️  pgvector extension not found")
                logger.warning("    Attempting to create extension...")
                try:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                    conn.commit()
                    logger.info("✓ pgvector extension created successfully")
                except psycopg.errors.InsufficientPrivilege:
                    logger.error("❌ Failed to create pgvector extension (insufficient privileges)")
                    logger.error("   Please ask your DBA to run: CREATE EXTENSION vector;")
                    logger.error("   On AWS RDS, use rds_superuser role")
                    return False
            else:
                logger.info("✓ pgvector extension found")

            # Check uuid-ossp
            cur.execute("""
                SELECT EXISTS(
                    SELECT 1 FROM pg_extension WHERE extname = 'uuid-ossp'
                )
            """)
            has_uuid = cur.fetchone()[0]

            if not has_uuid:
                logger.warning("⚠️  uuid-ossp extension not found")
                logger.warning("    Attempting to create extension...")
                try:
                    cur.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
                    conn.commit()
                    logger.info("✓ uuid-ossp extension created successfully")
                except psycopg.errors.InsufficientPrivilege:
                    logger.error('❌ Failed to create uuid-ossp extension (insufficient privileges)')
                    logger.error('   Please ask your DBA to run: CREATE EXTENSION "uuid-ossp";')
                    return False
            else:
                logger.info("✓ uuid-ossp extension found")

        return True

    except Exception as e:
        logger.error(f"❌ Extension check failed: {e}")
        return False


def execute_schema(conn, schema_path: Path):
    """Execute the schema SQL file"""
    logger.info(f"Executing schema from: {schema_path}")

    try:
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema_sql = f.read()

        with conn.cursor() as cur:
            cur.execute(schema_sql)
            conn.commit()

        logger.info("✓ Schema executed successfully")
        return True

    except Exception as e:
        logger.error(f"❌ Schema execution failed: {e}")
        conn.rollback()
        return False


def verify_tables(conn):
    """Verify that all required tables were created"""
    logger.info("Verifying created tables...")

    try:
        with conn.cursor() as cur:
            # Check for required tables
            cur.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name IN ('jargon_dictionary', 'knowledge_nodes', 'knowledge_edges')
                ORDER BY table_name
            """)
            tables = [row[0] for row in cur.fetchall()]

            logger.info("Created tables:")
            for table in tables:
                logger.info(f"  ✓ {table}")

            # Check for views
            cur.execute("""
                SELECT table_name
                FROM information_schema.views
                WHERE table_schema = 'public'
                AND table_name IN ('v_term_relationships', 'v_graph_statistics', 'v_jargon_statistics')
                ORDER BY table_name
            """)
            views = [row[0] for row in cur.fetchall()]

            if views:
                logger.info("Created views:")
                for view in views:
                    logger.info(f"  ✓ {view}")

            # Verify jargon_dictionary columns
            cur.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = 'jargon_dictionary'
                ORDER BY ordinal_position
            """)
            columns = [row[0] for row in cur.fetchall()]
            expected_columns = ['id', 'collection_name', 'term', 'definition', 'domain',
                              'aliases', 'related_terms', 'created_at', 'updated_at']

            missing_columns = set(expected_columns) - set(columns)
            if missing_columns:
                logger.warning(f"⚠️  Missing columns in jargon_dictionary: {missing_columns}")

            # Check for obsolete confidence_score column
            if 'confidence_score' in columns:
                logger.warning("⚠️  Obsolete column 'confidence_score' found in jargon_dictionary")
                logger.warning("    This should be removed. Run:")
                logger.warning("    ALTER TABLE jargon_dictionary DROP COLUMN confidence_score;")

        return len(tables) == 3  # Should have all 3 tables

    except Exception as e:
        logger.error(f"❌ Table verification failed: {e}")
        return False


def main():
    """Main setup function"""
    logger.info("=" * 60)
    logger.info("Advanced RAG System - Database Setup")
    logger.info("=" * 60)

    # Load environment variables
    load_dotenv()

    # Connection parameters
    conn_params = {
        'host': os.getenv('DB_HOST'),
        'port': os.getenv('DB_PORT'),
        'dbname': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD')
    }

    # Validate connection parameters
    if not all(conn_params.values()):
        logger.error("❌ Missing database connection parameters in .env file")
        logger.error("   Required: DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD")
        return 1

    logger.info(f"Connecting to database: {conn_params['host']}:{conn_params['port']}/{conn_params['dbname']}")

    try:
        # Connect to database
        with psycopg.connect(**conn_params) as conn:
            logger.info("✓ Database connection established")

            # Check extensions
            if not check_extensions(conn):
                logger.error("❌ Extension check failed. Cannot proceed.")
                return 1

            # Execute schema
            schema_path = Path(__file__).parent / 'database_schema.sql'
            if not schema_path.exists():
                logger.error(f"❌ Schema file not found: {schema_path}")
                return 1

            if not execute_schema(conn, schema_path):
                logger.error("❌ Schema execution failed. Cannot proceed.")
                return 1

            # Verify tables
            if not verify_tables(conn):
                logger.warning("⚠️  Some tables may not have been created correctly")
                logger.warning("    Please check the logs above for details")
                return 1

            logger.info("=" * 60)
            logger.info("✅ Database setup completed successfully!")
            logger.info("=" * 60)
            logger.info("")
            logger.info("Next steps:")
            logger.info("  1. Run the application: streamlit run app.py")
            logger.info("  2. Verify tables: SELECT * FROM v_graph_statistics;")
            logger.info("")

            return 0

    except psycopg.OperationalError as e:
        logger.error(f"❌ Database connection failed: {e}")
        logger.error("   Please check your database credentials in .env file")
        return 1
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
