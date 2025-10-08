"""Test script to check if query_unified works"""
import os
import logging
from dotenv import load_dotenv
from src.core.rag_system import RAGSystem, Config

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_query():
    print("Initializing RAG system...")
    cfg = Config()
    rag = RAGSystem(cfg)

    print("\nTesting query_unified...")
    try:
        response = rag.query_unified(
            question="アンモニア燃料の特徴は何ですか？",
            use_query_expansion=False,
            use_rag_fusion=False,
            use_jargon_augmentation=False,
            use_reranking=False,
            search_type="ハイブリッド検索"
        )

        print(f"\nResponse received:")
        print(f"Answer: {response.get('answer', 'No answer')[:200]}...")
        print(f"Sources: {len(response.get('sources', []))} documents")

    except Exception as e:
        print(f"Error during query: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_query()