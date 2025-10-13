#!/usr/bin/env python3
"""
既存のDBデータの類義語を双方向性が保証されるように修正するスクリプト

実行方法:
    python src/scripts/fix_bidirectional_synonyms.py

機能:
- jargon_dictionaryテーブルの全用語をスキャン
- A→Bの類義語関係がある場合、B→Aも追加
- 変更をDBに反映
"""

import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set

sys.path.append(str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv
from rag.config import Config
from sqlalchemy import create_engine, text
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()
cfg = Config()

PG_URL = f"postgresql+psycopg://{cfg.db_user}:{cfg.db_password}@{cfg.db_host}:{cfg.db_port}/{cfg.db_name}"
JARGON_TABLE_NAME = cfg.jargon_table_name


def load_all_terms_with_aliases(engine) -> Dict[str, List[str]]:
    """
    DBから全用語とその類義語を読み込む

    Returns:
        {term: [alias1, alias2, ...]}
    """
    query = text(f"""
        SELECT term, aliases
        FROM {JARGON_TABLE_NAME}
        ORDER BY term
    """)

    terms_aliases = {}
    with engine.connect() as conn:
        result = conn.execute(query)
        for row in result:
            term = row.term
            aliases = row.aliases or []
            terms_aliases[term] = aliases

    logger.info(f"Loaded {len(terms_aliases)} terms from database")
    return terms_aliases


def ensure_bidirectional_synonyms(terms_aliases: Dict[str, List[str]]) -> Dict[str, Set[str]]:
    """
    類義語の双方向性を保証する

    Args:
        terms_aliases: {term: [alias1, alias2, ...]}

    Returns:
        {term: {alias1, alias2, ...}} (双方向性保証済み、重複なし)
    """
    bidirectional = defaultdict(set)

    # 全ての類義語関係を双方向に登録
    for term, aliases in terms_aliases.items():
        for alias in aliases:
            # A → B
            bidirectional[term].add(alias)
            # B → A （逆方向も追加）
            bidirectional[alias].add(term)

    logger.info(f"Created bidirectional synonym mapping for {len(bidirectional)} terms")
    return bidirectional


def find_changes(
    original: Dict[str, List[str]],
    bidirectional: Dict[str, Set[str]]
) -> Dict[str, List[str]]:
    """
    変更が必要な用語を検出

    Returns:
        {term: [new_aliases]} (変更があった用語のみ)
    """
    changes = {}

    for term, new_aliases_set in bidirectional.items():
        original_aliases = set(original.get(term, []))
        new_aliases = sorted(new_aliases_set)  # ソートして一貫性を保つ

        # 変更があるかチェック
        if original_aliases != new_aliases_set:
            changes[term] = new_aliases
            added = new_aliases_set - original_aliases
            removed = original_aliases - new_aliases_set
            if added:
                logger.debug(f"  [{term}] 追加: {added}")
            if removed:
                logger.debug(f"  [{term}] 削除: {removed}")

    logger.info(f"Found {len(changes)} terms with changes")
    return changes


def update_database(engine, changes: Dict[str, List[str]]):
    """
    変更をDBに反映

    Args:
        changes: {term: [new_aliases]}
    """
    if not changes:
        logger.info("No changes to apply")
        return

    updated_count = 0

    with engine.begin() as conn:
        for term, new_aliases in changes.items():
            try:
                conn.execute(
                    text(f"""
                        UPDATE {JARGON_TABLE_NAME}
                        SET aliases = :aliases
                        WHERE term = :term
                    """),
                    {"term": term, "aliases": new_aliases}
                )
                updated_count += 1
            except Exception as e:
                logger.error(f"Error updating term '{term}': {e}")

    logger.info(f"Successfully updated {updated_count} terms")


def main():
    """メイン処理"""
    logger.info("="*80)
    logger.info("類義語双方向性修正スクリプト開始")
    logger.info("="*80)

    engine = create_engine(PG_URL)

    # 1. 現在のデータを読み込み
    logger.info("\n[Step 1] Loading current data from database...")
    terms_aliases = load_all_terms_with_aliases(engine)

    # 2. 双方向性を保証
    logger.info("\n[Step 2] Ensuring bidirectional synonyms...")
    bidirectional = ensure_bidirectional_synonyms(terms_aliases)

    # 3. 変更を検出
    logger.info("\n[Step 3] Detecting changes...")
    changes = find_changes(terms_aliases, bidirectional)

    # 4. 変更をプレビュー
    if changes:
        logger.info("\n[Preview] Changes to be applied:")
        preview_count = min(10, len(changes))
        for i, (term, new_aliases) in enumerate(list(changes.items())[:preview_count]):
            original = terms_aliases.get(term, [])
            logger.info(f"  {term}")
            logger.info(f"    Before: {original}")
            logger.info(f"    After:  {new_aliases}")
        if len(changes) > preview_count:
            logger.info(f"  ... and {len(changes) - preview_count} more")

    # 5. 確認プロンプト
    if changes:
        print(f"\n{len(changes)}件の用語を更新します。続行しますか？ (y/n): ", end="")
        response = input().strip().lower()
        if response != 'y':
            logger.info("Cancelled by user")
            return

        # 6. DBに反映
        logger.info("\n[Step 4] Updating database...")
        update_database(engine, changes)

    logger.info("\n" + "="*80)
    logger.info("処理完了")
    logger.info("="*80)


if __name__ == "__main__":
    main()
