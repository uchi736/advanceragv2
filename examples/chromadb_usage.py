"""
Example: Using ChromaDB as Vector Store
This example demonstrates how to use ChromaDB instead of PGVector
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag.config import Config
from src.core.rag_system import RAGSystem


def setup_chromadb_environment():
    """Configure environment for ChromaDB usage"""
    # Set vector store type to ChromaDB
    os.environ["VECTOR_STORE_TYPE"] = "chromadb"

    # Set ChromaDB specific settings
    os.environ["CHROMA_PERSIST_DIRECTORY"] = "./example_chroma_db"

    # For server mode (optional):
    # os.environ["CHROMA_SERVER_HOST"] = "localhost"
    # os.environ["CHROMA_SERVER_PORT"] = "8000"

    print("Environment configured for ChromaDB")


def main():
    """Main example function"""
    # Load environment variables
    load_dotenv()

    # Override to use ChromaDB
    setup_chromadb_environment()

    # Initialize configuration
    config = Config()
    print(f"Vector Store Type: {config.vector_store_type}")
    print(f"ChromaDB Directory: {config.chroma_persist_directory}")

    try:
        # Create RAG system with ChromaDB
        rag_system = RAGSystem(config)
        print("✓ RAG System initialized with ChromaDB")

        # Example: Ingest documents
        sample_docs = ["path/to/document1.pdf", "path/to/document2.txt"]
        # Uncomment to actually ingest:
        # rag_system.ingest_documents(sample_docs)

        # Example: Query the system
        question = "What is machine learning?"
        response = rag_system.query(
            question,
            use_query_expansion=False,
            use_rag_fusion=False
        )

        print(f"\nQuestion: {question}")
        print(f"Answer: {response['answer'][:200]}...")  # Show first 200 chars

    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure to:")
        print("1. Install chromadb: pip install chromadb")
        print("2. Configure Azure OpenAI credentials in .env")


if __name__ == "__main__":
    main()