"""
Vector Store Migration Script
Helps switch between PGVector and ChromaDB implementations
"""

import os
import sys
from pathlib import Path
from typing import List
import argparse
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings
from src.rag.config import Config
from src.rag.vector_store_adapter import VectorStoreFactory


def setup_environment(target_store: str, config: Config):
    """Update environment for target vector store"""
    env_file = Path(".env")

    if not env_file.exists():
        print("⚠ Warning: .env file not found. Creating from .env.example")
        example_file = Path(".env.example")
        if example_file.exists():
            env_file.write_text(example_file.read_text())
        else:
            print("✗ Error: .env.example not found")
            return False

    # Read current .env
    lines = env_file.read_text().splitlines()

    # Update VECTOR_STORE_TYPE
    updated_lines = []
    found = False
    for line in lines:
        if line.startswith("VECTOR_STORE_TYPE="):
            updated_lines.append(f"VECTOR_STORE_TYPE={target_store}")
            found = True
        else:
            updated_lines.append(line)

    if not found:
        updated_lines.append(f"VECTOR_STORE_TYPE={target_store}")

    # Write back
    env_file.write_text("\n".join(updated_lines))
    print(f"✓ Updated .env file to use {target_store}")

    return True


def check_requirements(store_type: str, config: Config):
    """Check if requirements are met for the target store"""
    print(f"\nChecking requirements for {store_type}...")

    if store_type == "pgvector":
        # Check PostgreSQL connection
        if not config.db_host or config.db_password == "your-password":
            print("✗ PostgreSQL configuration missing in .env")
            print("  Please configure DB_HOST, DB_USER, DB_PASSWORD, etc.")
            return False

        try:
            import psycopg2
            print("✓ psycopg2 installed")
        except ImportError:
            try:
                import psycopg
                print("✓ psycopg installed")
            except ImportError:
                print("✗ Neither psycopg2 nor psycopg installed")
                print("  Run: pip install psycopg2-binary")
                return False

        try:
            import pgvector
            print("✓ pgvector Python package installed")
        except ImportError:
            print("✗ pgvector Python package not installed")
            print("  Run: pip install pgvector")
            return False

    elif store_type == "chromadb":
        try:
            import chromadb
            print("✓ ChromaDB installed")
        except ImportError:
            print("✗ ChromaDB not installed")
            print("  Run: pip install chromadb")
            return False

        # Check if using server mode
        if config.chroma_server_host:
            print(f"  Will connect to ChromaDB server at {config.chroma_server_host}:{config.chroma_server_port}")
        else:
            print(f"  Will use local storage at {config.chroma_persist_directory}")
            Path(config.chroma_persist_directory).mkdir(parents=True, exist_ok=True)

    return True


def test_connection(store_type: str, config: Config):
    """Test connection to the target vector store"""
    print(f"\nTesting {store_type} connection...")

    try:
        # Initialize embeddings
        embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=config.azure_openai_endpoint,
            api_key=config.azure_openai_api_key,
            api_version=config.azure_openai_api_version,
            azure_deployment=config.azure_openai_embedding_deployment_name
        )

        # Create connection string for PGVector
        connection_string = None
        if store_type == "pgvector":
            try:
                import psycopg
                pg_dialect = "psycopg"
            except ModuleNotFoundError:
                pg_dialect = "psycopg2"

            connection_string = f"postgresql+{pg_dialect}://{config.db_user}:{config.db_password}@{config.db_host}:{config.db_port}/{config.db_name}"

        # Create vector store
        vector_store = VectorStoreFactory.create_vector_store(
            store_type=store_type,
            config=config,
            embedding_function=embeddings,
            connection_string=connection_string
        )

        # Test with a simple operation
        test_doc = Document(
            page_content="Connection test document",
            metadata={"test": True, "chunk_id": "test_connection"}
        )

        # Add and delete test document
        ids = vector_store.add_documents([test_doc], ids=["test_connection"])
        vector_store.delete(ids=["test_connection"])

        print(f"✓ Successfully connected to {store_type}")
        return True

    except Exception as e:
        print(f"✗ Failed to connect to {store_type}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Switch between vector store implementations")
    parser.add_argument(
        "target",
        choices=["pgvector", "chromadb"],
        help="Target vector store type"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check requirements without switching"
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip connection test"
    )

    args = parser.parse_args()

    # Load environment
    load_dotenv()
    config = Config()

    print(f"\nVector Store Migration Tool")
    print("="*50)
    print(f"Current vector store: {config.vector_store_type}")
    print(f"Target vector store: {args.target}")

    if config.vector_store_type == args.target:
        print(f"\n✓ Already using {args.target}")
        return 0

    # Check requirements
    if not check_requirements(args.target, config):
        print("\n✗ Requirements check failed")
        return 1

    if args.check_only:
        print("\n✓ Requirements check passed")
        return 0

    # Test connection
    if not args.skip_test:
        if not test_connection(args.target, config):
            print("\n✗ Connection test failed")
            print("  Fix the issues above and try again")
            print("  Or use --skip-test to skip this check")
            return 1

    # Update environment
    if not setup_environment(args.target, config):
        print("\n✗ Failed to update environment")
        return 1

    print("\n" + "="*50)
    print(f"✓ Successfully switched to {args.target}")
    print("\nNext steps:")
    print("1. Restart your application")
    print("2. Re-index your documents using the ingestion pipeline")
    print("\nNote: Existing vector data was not migrated.")
    print("You'll need to re-ingest your documents.")

    return 0


if __name__ == "__main__":
    sys.exit(main())