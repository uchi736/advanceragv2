"""Test script for reverse lookup functionality"""
import os
import sys
import io
from dotenv import load_dotenv
from src.rag.reverse_lookup import ReverseLookupEngine, ReverseLookupResult
from src.rag.term_extraction import JargonDictionaryManager
from src.core.rag_system import Config

# Set UTF-8 encoding for stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load environment variables
load_dotenv()

def test_reverse_lookup():
    print("=== Testing Reverse Lookup Functionality ===\n")

    # Initialize components
    cfg = Config()
    connection_string = f"postgresql://{cfg.db_user}:{cfg.db_password}@{cfg.db_host}:{cfg.db_port}/{cfg.db_name}"

    # Create JargonManager
    jargon_manager = JargonDictionaryManager(connection_string)

    # Create ReverseLookupEngine
    reverse_engine = ReverseLookupEngine(jargon_manager=jargon_manager)

    # Test cases
    test_queries = [
        "亜酸化窒素の削減方法",
        "温室効果ガスを減らす技術",
        "二酸化炭素の265倍の温室効果",
        "国際海事機関の規制",
        "アンモニア燃料の環境影響"
    ]

    print("Testing reverse lookup for various queries:\n")

    for query in test_queries:
        print(f"Query: {query}")
        print("-" * 50)

        # Perform reverse lookup
        results = reverse_engine.reverse_lookup(query, top_k=5)

        if results:
            print("Found terms:")
            for result in results:
                print(f"  - {result.term} (confidence: {result.confidence:.2f}, source: {result.source})")
        else:
            print("  No terms found")

        # Test augmentation
        augmentation = reverse_engine.augment_query_with_reverse_lookup(
            original_query=query,
            extracted_terms=[]  # Empty for testing reverse only
        )

        print(f"\nAugmented queries:")
        for i, aug_query in enumerate(augmentation['augmented_queries'][:3], 1):
            print(f"  {i}. {aug_query[:100]}...")

        print("\n")

def test_pattern_matching():
    print("=== Testing Pattern Matching ===\n")

    # Create a simple reverse engine
    reverse_engine = ReverseLookupEngine()

    # Test pattern dictionary
    test_cases = [
        ("亜酸化窒素", ["N2O", "nitrous oxide"]),
        ("温室効果ガス", ["GHG", "greenhouse gas"]),
        ("二酸化炭素", ["CO2", "carbon dioxide"]),
        ("アンモニア", ["NH3", "ammonia"])
    ]

    for japanese_term, expected_results in test_cases:
        query = f"{japanese_term}について教えてください"
        results = reverse_engine.reverse_lookup(query, top_k=5)

        found_terms = [r.term for r in results]
        print(f"Query: {query}")
        print(f"  Expected: {expected_results}")
        print(f"  Found: {found_terms}")

        # Check if expected results are in found terms
        for expected in expected_results:
            if expected in found_terms:
                print(f"    ✓ Found '{expected}'")
            else:
                print(f"    ✗ Missing '{expected}'")
        print()

if __name__ == "__main__":
    try:
        print("1. Testing Pattern Matching (no database needed)")
        print("=" * 60)
        test_pattern_matching()

        print("\n2. Testing Full Reverse Lookup (with database)")
        print("=" * 60)
        test_reverse_lookup()

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()