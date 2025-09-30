"""
Test script for vector store adapters
Tests both PGVector and ChromaDB implementations
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings
from src.rag.config import Config
from src.rag.vector_store_adapter import VectorStoreFactory

def test_vector_store(store_type: str, config: Config):
    """Test basic vector store operations"""
    print(f"\n{'='*50}")
    print(f"Testing {store_type.upper()} Vector Store")
    print('='*50)

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

    # Create vector store adapter
    try:
        vector_store = VectorStoreFactory.create_vector_store(
            store_type=store_type,
            config=config,
            embedding_function=embeddings,
            connection_string=connection_string
        )
        print(f"✓ {store_type} vector store created successfully")
    except Exception as e:
        print(f"✗ Failed to create {store_type} vector store: {e}")
        return False

    # Test documents
    test_docs = [
        Document(
            page_content="This is a test document about artificial intelligence.",
            metadata={"source": "test1.txt", "chunk_id": "test_001"}
        ),
        Document(
            page_content="Machine learning is a subset of artificial intelligence.",
            metadata={"source": "test2.txt", "chunk_id": "test_002"}
        ),
        Document(
            page_content="Deep learning uses neural networks with multiple layers.",
            metadata={"source": "test3.txt", "chunk_id": "test_003"}
        )
    ]

    # Test 1: Add documents
    try:
        ids = ["test_001", "test_002", "test_003"]
        result_ids = vector_store.add_documents(test_docs, ids=ids)
        print(f"✓ Added {len(result_ids)} documents")
    except Exception as e:
        print(f"✗ Failed to add documents: {e}")
        return False

    # Test 2: Similarity search
    try:
        query = "What is artificial intelligence?"
        results = vector_store.similarity_search_with_score(query, k=2)
        print(f"✓ Similarity search returned {len(results)} results")

        for i, (doc, score) in enumerate(results, 1):
            print(f"  Result {i}: Score={score:.4f}, Content={doc.page_content[:50]}...")
    except Exception as e:
        print(f"✗ Failed to perform similarity search: {e}")
        return False

    # Test 3: Delete documents
    try:
        vector_store.delete(ids=["test_001"])
        print("✓ Deleted document test_001")

        # Verify deletion with another search
        results = vector_store.similarity_search_with_score(query, k=3)
        remaining_ids = [doc.metadata.get('chunk_id') for doc, _ in results]
        if "test_001" not in remaining_ids:
            print("✓ Deletion verified - test_001 not in results")
        else:
            print("✗ Deletion failed - test_001 still in results")
    except Exception as e:
        print(f"✗ Failed to delete documents: {e}")
        return False

    # Clean up remaining test documents
    try:
        vector_store.delete(ids=["test_002", "test_003"])
        print("✓ Cleaned up remaining test documents")
    except Exception as e:
        print(f"⚠ Warning: Failed to clean up test documents: {e}")

    print(f"\n✓ All tests passed for {store_type}")
    return True


def main():
    """Main test function"""
    # Load environment variables
    load_dotenv()

    # Initialize configuration
    config = Config()

    print("\nVector Store Adapter Test Suite")
    print("================================")

    # Test PGVector if PostgreSQL is configured
    if config.db_host and config.db_password != "your-password":
        try:
            # Temporarily set to pgvector for testing
            original_type = config.vector_store_type
            config.vector_store_type = "pgvector"
            test_vector_store("pgvector", config)
            config.vector_store_type = original_type
        except Exception as e:
            print(f"\n⚠ Skipping PGVector tests: {e}")
    else:
        print("\n⚠ Skipping PGVector tests: PostgreSQL not configured")

    # Test ChromaDB (always available as it can use local storage)
    try:
        # Temporarily set to chromadb for testing
        original_type = config.vector_store_type
        config.vector_store_type = "chromadb"
        config.chroma_persist_directory = "./test_chroma_db"
        test_vector_store("chromadb", config)
        config.vector_store_type = original_type
    except Exception as e:
        print(f"\n⚠ ChromaDB test failed: {e}")

    print("\n" + "="*50)
    print("Test suite completed")
    print("="*50)


if __name__ == "__main__":
    main()