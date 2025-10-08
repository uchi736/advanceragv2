#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test reverse lookup integrated in chains.py"""

import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

from src.core.rag_system import RAGSystem, Config

def test_reverse_query():
    """Test queries that should trigger reverse lookup."""

    config = Config()
    rag_system = RAGSystem(config)

    # Test queries
    test_queries = [
        ("亜酸化窒素の削減方法", "Should find N2O through reverse lookup"),
        ("温室効果ガスの排出規制", "Should find GHG through reverse lookup"),
        ("N2Oの削減方法", "Should find N2O directly"),
    ]

    print("\n" + "="*60)
    print("Testing Reverse Lookup in Jargon Augmentation")
    print("="*60)

    for query, description in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print(f"Expected: {description}")
        print("-"*50)

        try:
            result = rag_system.query_unified(
                query,
                use_jargon_augmentation=True,
                use_query_expansion=False,
                use_rag_fusion=False,
                use_reranking=False
            )

            if result['status'] == 'success':
                jargon_info = result.get('jargon_augmentation', {})

                print(f"Extracted terms: {jargon_info.get('extracted_terms', [])}")
                print(f"Reverse terms: {jargon_info.get('reverse_terms', [])}")
                print(f"Matched terms: {list(jargon_info.get('matched_terms', {}).keys())}")
                print(f"Answer length: {len(result['answer'])} chars")

                # Check if reverse lookup worked
                if "亜酸化窒素" in query and "N2O" in str(jargon_info.get('reverse_terms', [])):
                    print("✓ Reverse lookup successful: Found N2O from 亜酸化窒素")
                elif "温室効果ガス" in query and "GHG" in str(jargon_info.get('reverse_terms', [])):
                    print("✓ Reverse lookup successful: Found GHG from 温室効果ガス")

            else:
                print(f"Query failed: {result.get('error', 'Unknown error')}")

        except Exception as e:
            print(f"Error: {str(e)}")

    print("\n" + "="*60)
    print("Test Complete")
    print("="*60)

if __name__ == "__main__":
    test_reverse_query()