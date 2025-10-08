"""
専門用語抽出精度テストスクリプト

複数ドキュメントから用語抽出し、正解データと比較してPrecision/Recall/F1を計算
"""

import asyncio
import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Set

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# .envファイルの読み込み
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

from src.rag.config import Config
from src.core.rag_system import RAGSystem


def calculate_metrics(predicted: Set[str], ground_truth: Set[str]) -> Dict[str, float]:
    """Precision, Recall, F1スコアを計算"""
    tp = len(predicted & ground_truth)  # True Positives
    fp = len(predicted - ground_truth)  # False Positives
    fn = len(ground_truth - predicted)  # False Negatives

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn
    }


async def main():
    print("=" * 80)
    print("専門用語抽出精度テスト")
    print("=" * 80)
    print()

    # 1. テストデータのパス
    test_data_dir = project_root / "test_data" / "sample_docs"
    ground_truth_path = project_root / "test_data" / "ground_truth.json"
    output_path = project_root / "test_data" / "extracted_terms.json"

    print(f"[DATA] テストデータディレクトリ: {test_data_dir}")
    print(f"[DATA] 正解データ: {ground_truth_path}")
    print()

    # 2. 正解データの読み込み
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        ground_truth_data = json.load(f)

    ground_truth_all = set(ground_truth_data["all_documents"])
    print(f"[OK] 正解用語数: {len(ground_truth_all)} 用語")
    print()

    # 3. RAGシステムの初期化
    print("[INIT] RAGシステムを初期化中...")
    config = Config()
    rag = RAGSystem(config)

    if not rag.jargon_manager:
        print("[ERROR] JargonManager が初期化されていません")
        return

    print("[OK] RAGシステム初期化完了")
    print()

    # 4. 専門用語抽出の実行
    print("[START] 専門用語抽出を開始...")
    print("-" * 80)

    try:
        await rag.extract_terms(str(test_data_dir), str(output_path))
        print()
        print("[OK] 用語抽出完了")
        print()
    except Exception as e:
        print(f"[ERROR] 用語抽出エラー: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. 抽出結果の読み込み
    if not output_path.exists():
        print(f"[ERROR] 抽出結果ファイルが見つかりません: {output_path}")
        return

    with open(output_path, "r", encoding="utf-8") as f:
        extracted_data = json.load(f)

    extracted_terms = extracted_data.get("terms", [])
    extracted_set = set([term["headword"] for term in extracted_terms])

    print(f"[RESULT] 抽出用語数: {len(extracted_set)} 用語")
    print()

    # 6. 精度評価
    print("=" * 80)
    print("精度評価結果")
    print("=" * 80)
    print()

    metrics = calculate_metrics(extracted_set, ground_truth_all)

    print(f"[PRECISION] 適合率: {metrics['precision']:.2%}")
    print(f"   - 抽出した用語のうち、正解だった割合")
    print(f"   - True Positives: {metrics['tp']}")
    print(f"   - False Positives: {metrics['fp']}")
    print()

    print(f"[RECALL] 再現率: {metrics['recall']:.2%}")
    print(f"   - 正解用語のうち、抽出できた割合")
    print(f"   - True Positives: {metrics['tp']}")
    print(f"   - False Negatives: {metrics['fn']}")
    print()

    print(f"[F1] F1スコア: {metrics['f1']:.2%}")
    print(f"   - PrecisionとRecallの調和平均")
    print()

    # 7. 詳細分析
    print("=" * 80)
    print("詳細分析")
    print("=" * 80)
    print()

    # 正しく抽出できた用語（True Positives）
    tp_terms = extracted_set & ground_truth_all
    print(f"[TP] 正しく抽出できた用語（True Positives: {len(tp_terms)}件）:")
    for term in sorted(list(tp_terms))[:20]:  # 最初の20件のみ表示
        print(f"   [OK] {term}")
    if len(tp_terms) > 20:
        print(f"   ... 他 {len(tp_terms) - 20} 件")
    print()

    # 誤って抽出した用語（False Positives）
    fp_terms = extracted_set - ground_truth_all
    print(f"[FP] 誤って抽出した用語（False Positives: {len(fp_terms)}件）:")
    for term in sorted(list(fp_terms))[:20]:  # 最初の20件のみ表示
        print(f"   [NG] {term}")
    if len(fp_terms) > 20:
        print(f"   ... 他 {len(fp_terms) - 20} 件")
    print()

    # 抽出できなかった用語（False Negatives）
    fn_terms = ground_truth_all - extracted_set
    print(f"[FN] 抽出できなかった用語（False Negatives: {len(fn_terms)}件）:")
    for term in sorted(list(fn_terms))[:20]:  # 最初の20件のみ表示
        print(f"   [MISS] {term}")
    if len(fn_terms) > 20:
        print(f"   ... 他 {len(fn_terms) - 20} 件")
    print()

    # 8. ドキュメントごとの分析
    print("=" * 80)
    print("ドキュメントごとの抽出カバレッジ")
    print("=" * 80)
    print()

    for doc_key in ["document1_marine_engine", "document2_power_generation", "document3_renewable_energy"]:
        doc_ground_truth = set(ground_truth_data[doc_key])
        doc_extracted = extracted_set & doc_ground_truth
        coverage = len(doc_extracted) / len(doc_ground_truth) * 100 if doc_ground_truth else 0

        print(f"[DOC] {doc_key}:")
        print(f"   正解用語数: {len(doc_ground_truth)}")
        print(f"   抽出できた数: {len(doc_extracted)}")
        print(f"   カバレッジ: {coverage:.1f}%")
        print()

    print("=" * 80)
    print("テスト完了")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
