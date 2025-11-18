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
        SELECT term, definition, related_terms
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
                    'related_terms': row.related_terms or [],
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
            # [{term: ..., text: ..., score: ...}]形式の場合
            candidate_terms = [
                {
                    'term': item.get('term', item.get('headword', '')),
                    'text': item.get('text', item.get('term', item.get('headword', '')))  # textフィールドを優先
                }
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


async def generate_definitions_for_candidates(
    candidate_terms: List[Dict[str, Any]],
    llm
) -> List[Dict[str, Any]]:
    """
    候補用語にLLMで定義を生成（バルク処理: 1リクエストで複数用語）

    Args:
        candidate_terms: 候補用語リスト [{"term": "...", "text": "..."}]
        llm: Azure ChatOpenAI インスタンス

    Returns:
        定義付き候補用語リスト [{"term": "...", "text": "用語: 定義"}]
    """
    import asyncio
    import re

    logger.info(f"Generating definitions for {len(candidate_terms)} candidate terms with bulk processing...")

    # バルクプロンプトテンプレート（1リクエストで複数用語）
    def create_bulk_prompt(terms: List[str]) -> str:
        term_list = "\n".join([f"{i+1}. {term}" for i, term in enumerate(terms)])
        return f"""以下の{len(terms)}個の専門用語の定義を生成してください。
各定義は1-2文（40-50文字）で、技術的文脈や関連概念を含めてください。

用語リスト:
{term_list}

以下の形式で必ず回答してください:
1. 用語1: 定義1
2. 用語2: 定義2
3. 用語3: 定義3
..."""

    # レスポンスパース関数
    def parse_bulk_response(response_text: str, expected_terms: List[str]) -> Dict[str, str]:
        definitions = {}
        lines = response_text.strip().split('\n')

        for line in lines:
            # "番号. 用語: 定義" 形式をパース
            match = re.match(r'^\d+\.\s*(.+?):\s*(.+)$', line.strip())
            if match:
                term, definition = match.groups()
                term_clean = term.strip()
                # 期待される用語リストと照合
                for expected_term in expected_terms:
                    if expected_term in term_clean or term_clean in expected_term:
                        definitions[expected_term] = definition.strip()
                        break

        return definitions

    # バッチ処理（1リクエストで15用語ずつ）
    bulk_batch_size = 15  # トークン制限とのバランス
    enriched = []

    for i in range(0, len(candidate_terms), bulk_batch_size):
        batch = candidate_terms[i:i+bulk_batch_size]
        terms = [cand['term'] for cand in batch]

        logger.debug(f"Processing bulk batch {i//bulk_batch_size + 1}/{(len(candidate_terms) + bulk_batch_size - 1)//bulk_batch_size} ({len(terms)} terms)")

        try:
            # 1リクエストで複数用語の定義を生成
            prompt = create_bulk_prompt(terms)
            response = await llm.ainvoke(prompt)

            # レスポンスをパース
            definitions = parse_bulk_response(response.content, terms)

            # 結果をマージ
            for cand in batch:
                term = cand['term']
                if term in definitions:
                    definition = definitions[term]
                    text = f"{term}: {definition}"
                    logger.debug(f"Generated definition for '{term}': {definition[:50]}...")
                else:
                    # パース失敗時のフォールバック
                    logger.warning(f"Failed to parse definition for '{term}', using fallback")
                    text = cand.get('text', term)

                enriched.append({"term": term, "text": text})

        except Exception as e:
            logger.warning(f"Bulk definition generation failed for batch {i//bulk_batch_size + 1}: {e}")
            # エラー時は全てフォールバック
            for cand in batch:
                text = cand.get('text', cand['term'])
                enriched.append({"term": cand['term'], "text": text})

    logger.info(f"Successfully generated {len(enriched)} definitions (bulk processing)")
    return enriched


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

    # LLMインスタンス作成（定義生成用）
    from langchain_openai import AzureChatOpenAI
    from rag.config import Config
    cfg = Config()

    llm = AzureChatOpenAI(
        azure_deployment=cfg.azure_openai_chat_mini_deployment_name,  # 4.1-mini使用
        api_version=cfg.azure_openai_api_version,
        openai_api_key=cfg.azure_openai_api_key,
        temperature=0
    )

    # 候補用語は用語名のみで使用（定義生成をスキップしてパフォーマンス向上）
    logger.info("\n[Step 1.5] Skipping definition generation for candidate terms (performance optimization)")
    logger.info("Using term names only - embedding model can understand meaning from term names alone")
    enriched_candidates = candidate_terms  # 用語名のみで使用（定義なし）
    logger.info(f"Using {len(enriched_candidates)} candidates without additional definition generation")

    analyzer = TermClusteringAnalyzer(pg_url, min_terms=3, jargon_table_name=jargon_table_name, embeddings=embeddings)

    try:
        result = await analyzer.extract_semantic_synonyms_hybrid(
            specialized_terms=specialized_terms,
            candidate_terms=enriched_candidates,  # 定義付き候補用語を使用
            # similarity_threshold: 削除（クラスタリング+LLM判定のみ使用）
            max_synonyms=10,
            use_llm_naming=True,  # LLMによるクラスタ命名を有効化
            use_llm_for_candidates=True  # LLM判定を有効化（バッチ並列処理で高速化済み）
        )

        # 結果から類義語辞書、クラスタマッピング、クラスタ名を取得
        synonyms_dict = result.get('synonyms', {})
        cluster_mapping = result.get('clusters', {})
        cluster_names = result.get('cluster_names', {})

        # DBに保存（クラスタ名も含む）
        analyzer.update_semantic_synonyms_to_db(synonyms_dict, cluster_mapping, cluster_names)
        logger.info(f"Extracted semantic synonyms for {len(synonyms_dict)} terms")
        logger.info(f"Updated domain field with cluster info for {len(cluster_mapping)} terms")
        if cluster_names:
            logger.info(f"Applied LLM-generated cluster names: {cluster_names}")

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

    # パフォーマンス最適化: 候補用語を上位150件に制限
    if len(candidate_terms) > 150:
        logger.info(f"Limiting candidate terms from {len(candidate_terms)} to 150 (performance optimization)")
        candidate_terms = candidate_terms[:150]

    # 3. TermClusteringAnalyzerで類義語抽出
    logger.info(f"\n[Step 3] Extracting semantic synonyms...")
    logger.info(f"  - Specialized terms: {len(specialized_terms)}")
    logger.info(f"  - Candidate terms: {len(candidate_terms)}")
    logger.info(f"  - Total: {len(specialized_terms) + len(candidate_terms)}")

    analyzer = TermClusteringAnalyzer(PG_URL, min_terms=3)

    try:
        result = await analyzer.extract_semantic_synonyms_hybrid(
            specialized_terms=specialized_terms,
            candidate_terms=candidate_terms,
            # similarity_threshold: 削除（クラスタリング+LLM判定のみ使用）
            max_synonyms=10,
            use_llm_naming=True  # LLMによるクラスタ命名を有効化
        )

        # 結果から類義語辞書、クラスタマッピング、クラスタ名を取得
        synonyms_dict = result.get('synonyms', {})
        cluster_mapping = result.get('clusters', {})
        cluster_names = result.get('cluster_names', {})

        # 4. DBに保存
        logger.info(f"\n[Step 4] Updating database...")
        updated_count = analyzer.update_semantic_synonyms_to_db(synonyms_dict, cluster_mapping, cluster_names)

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
