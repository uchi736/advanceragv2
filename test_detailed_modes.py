"""Test each RAG mode in detail to see query expansion, sources, and answers"""
import os
import logging
from dotenv import load_dotenv
from src.core.rag_system import RAGSystem, Config
import json

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def print_mode_details(mode_name, response):
    print(f"\n{'='*80}")
    print(f"MODE: {mode_name}")
    print(f"{'='*80}")

    # Query Expansion
    if response.get("query_expansion"):
        qe = response["query_expansion"]
        print("\n[QUERY EXPANSION]")
        if isinstance(qe, dict):
            if "original_query" in qe:
                print(f"  Original: {qe.get('original_query', 'N/A')}")
            if "expanded_queries" in qe:
                print(f"  Expanded Queries:")
                for i, eq in enumerate(qe.get("expanded_queries", [])[:3], 1):
                    print(f"    {i}. {eq}")
        else:
            print(f"  Info: {qe}")

    # Jargon Augmentation
    if response.get("jargon_augmentation"):
        ja = response["jargon_augmentation"]
        print("\n[JARGON AUGMENTATION]")
        if "extracted_terms" in ja:
            print(f"  Extracted Terms: {', '.join(ja['extracted_terms'][:5])}")
        if "matched_terms" in ja:
            if isinstance(ja["matched_terms"], dict):
                print(f"  Matched Terms with Definitions:")
                for term, info in list(ja["matched_terms"].items())[:3]:
                    definition = info.get('definition', 'N/A') if isinstance(info, dict) else str(info)
                    print(f"    - {term}: {definition[:100]}...")
        if "augmented_query" in ja:
            aug_query = ja["augmented_query"]
            print(f"  Augmented Query: {aug_query[:150]}...")

    # Reranking
    if response.get("reranking"):
        rr = response["reranking"]
        print("\n[LLM RERANKING]")
        if isinstance(rr, dict):
            if "original_order" in rr:
                print(f"  Original Order: {rr.get('original_order', [])[:5]}")
            if "reranked_order" in rr:
                print(f"  Reranked Order: {rr.get('reranked_order', [])[:5]}")
            if "relevance_scores" in rr:
                print(f"  Relevance Scores: {dict(list(rr.get('relevance_scores', {}).items())[:3])}")

    # Sources
    print(f"\n[SOURCES]")
    sources = response.get("sources", [])
    print(f"  Total Retrieved: {len(sources)} documents")
    for i, source in enumerate(sources[:3], 1):
        if isinstance(source, dict):
            metadata = source.get("metadata", {})
        elif hasattr(source, 'metadata'):
            metadata = source.metadata
        else:
            metadata = {}

        doc_id = metadata.get("document_id", "Unknown")
        chunk_id = metadata.get("chunk_id", "N/A")
        score = metadata.get("score", "N/A")
        print(f"  {i}. Doc: {doc_id}, Chunk: {chunk_id}, Score: {score}")

    # Answer
    print(f"\n[GENERATED ANSWER]")
    answer = response.get("answer", "No answer")
    if answer and answer != "No answer":
        print(f"  Length: {len(answer)} characters")
        # Print first 300 chars
        answer_preview = answer[:300].replace('\n', '\n  ')
        print(f"  Content Preview:\n  {answer_preview}...")
    else:
        print(f"  [FAILED] No answer generated")

def test_all_modes_detailed():
    print("=== DETAILED RAG MODE TESTING ===")
    cfg = Config()
    rag = RAGSystem(cfg)

    test_question = "アンモニア燃料の環境影響について教えてください"

    modes = [
        ("基本モード", False, False, False, False),
        ("クエリ拡張", True, False, False, False),
        ("RAG-Fusion", False, True, False, False),
        ("専門用語補強", False, False, True, False),
        ("LLMリランク", False, False, False, True),
        ("全機能ON", True, True, True, True),
    ]

    for mode_name, use_qe, use_rf, use_ja, use_rr in modes:
        try:
            response = rag.query_unified(
                question=test_question,
                use_query_expansion=use_qe,
                use_rag_fusion=use_rf,
                use_jargon_augmentation=use_ja,
                use_reranking=use_rr,
                search_type="ハイブリッド検索"
            )
            print_mode_details(mode_name, response)
        except Exception as e:
            print(f"\n{'='*80}")
            print(f"MODE: {mode_name}")
            print(f"{'='*80}")
            print(f"[ERROR] {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_all_modes_detailed()