"""Test all RAG modes to ensure answer generation works correctly"""
import os
import logging
from dotenv import load_dotenv
from src.core.rag_system import RAGSystem, Config

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_all_modes():
    print("=== Testing All RAG Modes ===\n")
    cfg = Config()
    rag = RAGSystem(cfg)

    test_question = "アンモニア燃料の特徴を教えてください"

    modes = [
        ("基本モード", False, False, False, False),
        ("クエリ拡張のみ", True, False, False, False),
        ("RAG-Fusion", False, True, False, False),
        ("専門用語補強", False, False, True, False),
        ("LLMリランク", False, False, False, True),
        ("全機能ON", True, True, True, True),
    ]

    for mode_name, use_qe, use_rf, use_ja, use_rr in modes:
        print(f"\n--- {mode_name} ---")
        print(f"  クエリ拡張: {use_qe}")
        print(f"  RAG-Fusion: {use_rf}")
        print(f"  専門用語補強: {use_ja}")
        print(f"  LLMリランク: {use_rr}")

        try:
            response = rag.query_unified(
                question=test_question,
                use_query_expansion=use_qe,
                use_rag_fusion=use_rf,
                use_jargon_augmentation=use_ja,
                use_reranking=use_rr,
                search_type="ハイブリッド検索"
            )

            answer = response.get("answer", "No answer")
            if answer and answer != "No answer":
                print(f"  [OK] Answer generated successfully (length: {len(answer)})")
                # Print first 50 chars in ASCII
                first_chars = answer[:50].encode('ascii', 'replace').decode('ascii')
                print(f"      First 50 chars: {first_chars}...")
            else:
                print(f"  [FAIL] Answer generation failed")

            print(f"  Sources: {len(response.get('sources', []))} documents")

        except Exception as e:
            print(f"  [ERROR] {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_all_modes()