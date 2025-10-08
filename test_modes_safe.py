"""Test each RAG mode with safe encoding"""
import os
import sys
import io
import logging
from dotenv import load_dotenv
from src.core.rag_system import RAGSystem, Config

# Set UTF-8 encoding for stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def safe_print(text):
    """Print with safe encoding"""
    try:
        print(text)
    except UnicodeEncodeError:
        # Replace problematic characters
        safe_text = text.encode('ascii', 'replace').decode('ascii')
        print(safe_text)

def print_mode_details(mode_name, response):
    safe_print(f"\n{'='*80}")
    safe_print(f"MODE: {mode_name}")
    safe_print(f"{'='*80}")

    # Query Expansion
    if response.get("query_expansion"):
        qe = response["query_expansion"]
        safe_print("\n[QUERY EXPANSION]")
        if isinstance(qe, dict):
            if "original_query" in qe:
                safe_print(f"  Original: {qe.get('original_query', 'N/A')}")
            if "expanded_queries" in qe:
                safe_print(f"  Expanded Queries:")
                for i, eq in enumerate(qe.get("expanded_queries", [])[:3], 1):
                    safe_print(f"    {i}. {eq}")

    # Jargon Augmentation
    if response.get("jargon_augmentation"):
        ja = response["jargon_augmentation"]
        safe_print("\n[JARGON AUGMENTATION]")
        if "extracted_terms" in ja:
            terms_str = ', '.join(ja['extracted_terms'][:5])
            safe_print(f"  Extracted Terms: {terms_str}")
        if "matched_terms" in ja:
            if isinstance(ja["matched_terms"], dict):
                safe_print(f"  Matched Terms with Definitions:")
                for term, info in list(ja["matched_terms"].items())[:3]:
                    definition = info.get('definition', 'N/A') if isinstance(info, dict) else str(info)
                    safe_print(f"    - {term}: {definition[:100]}...")
        if "augmented_query" in ja:
            aug_query = ja["augmented_query"]
            safe_print(f"  Augmented Query: {aug_query[:150]}...")

    # Reranking
    if response.get("reranking"):
        rr = response["reranking"]
        safe_print("\n[LLM RERANKING]")
        if isinstance(rr, dict):
            if "original_order" in rr:
                safe_print(f"  Original Order: {rr.get('original_order', [])[:5]}")
            if "reranked_order" in rr:
                safe_print(f"  Reranked Order: {rr.get('reranked_order', [])[:5]}")

    # Sources
    safe_print(f"\n[SOURCES]")
    sources = response.get("sources", [])
    safe_print(f"  Total Retrieved: {len(sources)} documents")
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
        safe_print(f"  {i}. Doc: {doc_id}, Chunk: {chunk_id}")

    # Answer
    safe_print(f"\n[GENERATED ANSWER]")
    answer = response.get("answer", "No answer")
    if answer and answer != "No answer":
        safe_print(f"  Length: {len(answer)} characters")
        # Print first 200 chars safely
        answer_preview = answer[:200].replace('\n', ' ')
        safe_print(f"  Preview: {answer_preview}...")
    else:
        safe_print(f"  [FAILED] No answer generated")

def test_all_modes_detailed():
    safe_print("=== DETAILED RAG MODE TESTING ===")
    cfg = Config()
    rag = RAGSystem(cfg)

    test_question = "アンモニア燃料の環境影響について教えてください"

    modes = [
        ("1. 基本モード", False, False, False, False),
        ("2. クエリ拡張", True, False, False, False),
        ("3. RAG-Fusion", False, True, False, False),
        ("4. 専門用語補強", False, False, True, False),
        ("5. LLMリランク", False, False, False, True),
        ("6. 全機能ON", True, True, True, True),
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
            safe_print(f"\n{'='*80}")
            safe_print(f"MODE: {mode_name}")
            safe_print(f"{'='*80}")
            safe_print(f"[ERROR] {type(e).__name__}: {str(e)[:100]}")

if __name__ == "__main__":
    test_all_modes_detailed()