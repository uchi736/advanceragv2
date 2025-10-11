#!/usr/bin/env python3
"""extract_semantic_synonyms.py
HDBSCANによる意味ベース類義語抽出スクリプト
--------------------------------------------
専門用語 + LLM簡易フィルタ前の候補用語でクラスタリングし、
意味的に類似した用語を類義語として抽出してDBに保存する
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Any

sys.path.append(str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from rag.config import Config
from sqlalchemy import create_engine, text

# TermClusteringAnalyzer をインポート
from scripts.term_clustering_analyzer import TermClusteringAnalyzer

load_dotenv()
cfg = Config()

PG_URL = f"postgresql+psycopg://{cfg.db_user}:{cfg.db_password}@{cfg.db_host}:{cfg.db_port}/{cfg.db_name}"
JARGON_TABLE_NAME = cfg.jargon_table_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_specialized_terms(connection_string: str, jargon_table_name: str = None) -> List[Dict[str, Any]]:
    """
    DBから専門用語を読み込み（定義あり）

    Returns:
        [{"term": "ETC", "definition": "...", "text": "ETC: ..."}]
    """
    if jargon_table_name is None:
        jargon_table_name = JARGON_TABLE_NAME

    engine = create_engine(connection_string)
    query = text(f"""
        SELECT term, definition
        FROM {jargon_table_name}
        WHERE definition IS NOT NULL AND definition != ''
        ORDER BY term
    """)

    terms = []
    try:
        with engine.connect() as conn:
            result = conn.execute(query)
            for row in result:
                terms.append({
                    'term': row.term,
                    'definition': row.definition,
                    'text': f"{row.term}: {row.definition}"
                })
        logger.info(f"Loaded {len(terms)} specialized terms with definitions")
    except Exception as e:
        logger.error(f"Error loading specialized terms: {e}")

    return terms


def load_candidate_terms_from_extraction(file_path: str = "output/term_extraction_debug.json") -> List[Dict[str, Any]]:
    """
    用語抽出の中間結果（LLM簡易フィルタ前の候補）を読み込み

    Args:
        file_path: term_extraction.pyが出力するデバッグJSONのパス

    Returns:
        [{"term": "過給機", "text": "過給機"}]
    """
    import json

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # candidates_for_semrerankまたはcandidatesキーから候補用語を取得
        candidates_data = data.get('candidates_for_semrerank', data.get('candidates', {}))

        if isinstance(candidates_data, dict):
            # {term: score}形式の場合
            candidate_terms = [
                {'term': term, 'text': term}
                for term in candidates_data.keys()
            ]
        elif isinstance(candidates_data, list):
            # [{term: ..., score: ...}]形式の場合
            candidate_terms = [
                {'term': item.get('term', item.get('headword', '')), 'text': item.get('term', item.get('headword', ''))}
                for item in candidates_data
            ]
        else:
            logger.warning(f"Unexpected candidates format in {file_path}")
            return []

        logger.info(f"Loaded {len(candidate_terms)} candidate terms from {file_path}")
        return candidate_terms

    except FileNotFoundError:
        logger.warning(f"File not found: {file_path}. Using empty candidate list.")
        return []
    except Exception as e:
        logger.error(f"Error loading candidate terms: {e}")
        return []


async def extract_and_save_semantic_synonyms(
    specialized_terms: List[Dict[str, Any]],
    candidate_terms: List[Dict[str, Any]],
    pg_url: str,
    jargon_table_name: str,
    embeddings
) -> Dict[str, List[Dict[str, Any]]]:
    """
    意味ベース類義語を抽出してDBに保存（term_extraction.pyから呼ばれる統合関数）

    Args:
        specialized_terms: 専門用語リスト [{"term": "ETC", "definition": "...", "text": "ETC: ..."}]
        candidate_terms: 候補用語リスト [{"term": "過給機", "text": "過給機"}]
        pg_url: PostgreSQL接続URL
        jargon_table_name: 専門用語テーブル名
        embeddings: AzureOpenAIEmbeddings instance

    Returns:
        {term: [{"term": "類義語", "similarity": 0.92, "is_specialized": True}]}
    """
    logger.info(f"  - Specialized terms: {len(specialized_terms)}")
    logger.info(f"  - Candidate terms: {len(candidate_terms)}")
    logger.info(f"  - Total: {len(specialized_terms) + len(candidate_terms)}")

    analyzer = TermClusteringAnalyzer(pg_url, min_terms=3, jargon_table_name=jargon_table_name, embeddings=embeddings)

    try:
        synonyms_dict = analyzer.extract_semantic_synonyms_hybrid(
            specialized_terms=specialized_terms,
            candidate_terms=candidate_terms,
            similarity_threshold=0.85,
            max_synonyms=5
        )

        # DBに保存
        analyzer.update_semantic_synonyms_to_db(synonyms_dict)
        logger.info(f"Extracted semantic synonyms for {len(synonyms_dict)} terms")

        return synonyms_dict

    except Exception as e:
        logger.error(f"Error extracting semantic synonyms: {e}", exc_info=True)
        return {}


async def main():
    """メイン実行関数"""
    logger.info("="*50)
    logger.info("Semantic Synonym Extraction (HDBSCAN)")
    logger.info("="*50)

    # 1. 専門用語を読み込み
    logger.info("\n[Step 1] Loading specialized terms from database...")
    specialized_terms = load_specialized_terms(PG_URL)

    if len(specialized_terms) < 3:
        logger.warning(f"Not enough specialized terms ({len(specialized_terms)}). Need at least 3 terms.")
        return

    # 2. 候補用語を読み込み（LLM簡易フィルタ前）
    logger.info("\n[Step 2] Loading candidate terms (pre-LLM filter)...")
    candidate_terms = load_candidate_terms_from_extraction()

    if len(candidate_terms) == 0:
        logger.warning("No candidate terms found. Using only specialized terms.")
        candidate_terms = []

    # 3. TermClusteringAnalyzerで類義語抽出
    logger.info(f"\n[Step 3] Extracting semantic synonyms...")
    logger.info(f"  - Specialized terms: {len(specialized_terms)}")
    logger.info(f"  - Candidate terms: {len(candidate_terms)}")
    logger.info(f"  - Total: {len(specialized_terms) + len(candidate_terms)}")

    analyzer = TermClusteringAnalyzer(PG_URL, min_terms=3)

    try:
        synonyms_dict = analyzer.extract_semantic_synonyms_hybrid(
            specialized_terms=specialized_terms,
            candidate_terms=candidate_terms,
            similarity_threshold=0.85,
            max_synonyms=5
        )

        # 4. DBに保存
        logger.info(f"\n[Step 4] Updating database...")
        updated_count = analyzer.update_semantic_synonyms_to_db(synonyms_dict)

        # 5. 結果サマリー
        logger.info("\n" + "="*50)
        logger.info("Extraction Complete!")
        logger.info("="*50)
        logger.info(f"Total terms processed: {len(specialized_terms)}")
        logger.info(f"Terms with synonyms: {len(synonyms_dict)}")
        logger.info(f"Database updated: {updated_count} terms")

        # 上位5件を表示
        logger.info("\nTop 5 examples:")
        for i, (term, synonyms) in enumerate(list(synonyms_dict.items())[:5]):
            logger.info(f"\n[{i+1}] {term}")
            for syn in synonyms[:3]:
                spec_mark = "★" if syn['is_specialized'] else "○"
                logger.info(f"    {spec_mark} {syn['term']} (similarity: {syn['similarity']:.3f})")

    except Exception as e:
        logger.error(f"Error during extraction: {e}", exc_info=True)
        return

    logger.info("\nDone!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
