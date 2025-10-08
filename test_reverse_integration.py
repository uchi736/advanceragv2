#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test reverse lookup integration with RAG system."""

import os
import sys
import asyncio
from datetime import datetime

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import RAG components
from src.core.rag_system import RAGSystem, Config
from src.rag.reverse_lookup import ReverseLookupEngine

def safe_print(text):
    """Print text safely regardless of encoding."""
    try:
        print(text)
    except UnicodeEncodeError:
        # Replace problematic characters
        safe_text = text.encode('utf-8', errors='replace').decode('utf-8')
        print(safe_text)

async def test_reverse_integration():
    """Test reverse lookup integrated with RAG queries."""

    # Initialize RAG system
    config = Config()
    rag_system = RAGSystem(config)

    # Initialize reverse lookup separately for testing
    reverse_engine = ReverseLookupEngine(
        jargon_manager=rag_system.jargon_manager
    )

    # Test queries that need reverse lookup
    test_queries = [
        "亜酸化窒素の削減方法について教えてください",
        "温室効果ガスの排出規制について",
        "二酸化炭素の265倍の温室効果を持つガスは何ですか",
        "国際海事機関の環境規制について"
    ]

    safe_print("\n" + "="*60)
    safe_print("Testing Reverse Lookup Integration with RAG")
    safe_print("="*60)

    for query in test_queries:
        safe_print(f"\n{'='*50}")
        safe_print(f"Query: {query}")
        safe_print("-"*50)

        # Perform reverse lookup
        reverse_results = reverse_engine.reverse_lookup(query, top_k=3)

        safe_print("Reverse Lookup Results:")
        for result in reverse_results:
            safe_print(f"  - {result.term} (confidence: {result.confidence:.2f}, source: {result.source})")

        # Build augmentation
        extracted_terms = []  # No forward extraction for this test
        augmentation = reverse_engine.augment_query_with_reverse_lookup(
            query, extracted_terms
        )

        safe_print("\nAugmentation Details:")
        safe_print(f"  Combined terms: {augmentation['combined_terms']}")
        safe_print(f"  Augmented queries generated: {len(augmentation['augmented_queries'])}")

        # Test with RAG system (jargon augmentation ON)
        config.use_jargon_augmentation = True
        config.use_query_expansion = False
        config.use_rag_fusion = False
        config.use_llm_rerank = False

        try:
            result = await rag_system.query_unified(query)

            if result['status'] == 'success':
                safe_print("\nRAG System Results:")
                safe_print(f"  Answer length: {len(result['answer'])} chars")
                safe_print(f"  Sources retrieved: {len(result['sources'])}")

                # Show jargon augmentation details if available
                if 'jargon_augmentation' in result:
                    aug = result['jargon_augmentation']
                    safe_print(f"  Extracted terms: {aug.get('extracted_terms', [])}")
                    safe_print(f"  Matched terms: {list(aug.get('matched_terms', {}).keys())}")
            else:
                safe_print(f"\nRAG query failed: {result.get('error', 'Unknown error')}")

        except Exception as e:
            safe_print(f"\nError during RAG query: {str(e)}")

    safe_print("\n" + "="*60)
    safe_print("Test Complete")
    safe_print("="*60)

if __name__ == "__main__":
    asyncio.run(test_reverse_integration())