"""
Ground Truth用語の脱落追跡分析モジュール
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Set, Optional
from collections import defaultdict, Counter
from datetime import datetime

logger = logging.getLogger(__name__)


class TermDropoutAnalyzer:
    """
    Ground Truthの各用語が抽出プロセスのどの段階で脱落したかを追跡

    使用方法:
        analyzer = TermDropoutAnalyzer(ground_truth_terms)
        analyzer.track_stage("候補抽出", current_terms, all_candidates_dict)
        analyzer.track_stage("C-valueフィルタ", filtered_terms, removed_terms_with_reasons)
        ...
        report = analyzer.generate_report()
        analyzer.save_report("output/dropout_report.json")
    """

    def __init__(self, ground_truth_terms: List[str]):
        """
        Args:
            ground_truth_terms: Ground Truthの用語リスト
        """
        self.ground_truth = set(ground_truth_terms)
        self.traces: Dict[str, List[Dict[str, Any]]] = {term: [] for term in ground_truth_terms}
        self.stage_order: List[str] = []

        logger.info(f"TermDropoutAnalyzer initialized with {len(ground_truth_terms)} ground truth terms")

    def track_stage(
        self,
        stage_name: str,
        current_terms: List[str],
        metadata: Optional[Dict[str, Any]] = None,
        removed_terms_with_reasons: Optional[Dict[str, str]] = None
    ):
        """
        抽出プロセスの各段階での用語状態を記録

        Args:
            stage_name: 段階名（例: "候補抽出", "C-valueフィルタ"）
            current_terms: この段階で残っている用語リスト
            metadata: 用語ごとのメタデータ（スコア、頻度など）
            removed_terms_with_reasons: 除外された用語とその理由 {term: reason}
        """
        if stage_name not in self.stage_order:
            self.stage_order.append(stage_name)

        current_set = set(current_terms)
        metadata = metadata or {}
        removed_reasons = removed_terms_with_reasons or {}

        for term in self.ground_truth:
            trace_entry = {
                "stage": stage_name,
                "stage_index": len(self.stage_order) - 1
            }

            if term in current_set:
                # 用語が存在
                trace_entry["status"] = "present"

                # メタデータがあれば追加
                if term in metadata:
                    trace_entry.update(metadata[term])

            else:
                # 用語が不在
                trace_entry["status"] = "absent"

                # この段階で除外された場合は理由を記録
                if term in removed_reasons:
                    trace_entry["removed"] = True
                    trace_entry["reason"] = removed_reasons[term]

            self.traces[term].append(trace_entry)

        logger.info(f"Tracked stage '{stage_name}': {len(current_set)} terms present, "
                   f"{len(self.ground_truth & current_set)} ground truth terms present")

    def track_candidates_with_scores(
        self,
        stage_name: str,
        candidates: Dict[str, int],
        tfidf_scores: Optional[Dict[str, float]] = None,
        cvalue_scores: Optional[Dict[str, float]] = None,
        base_scores: Optional[Dict[str, float]] = None,
        enhanced_scores: Optional[Dict[str, float]] = None
    ):
        """
        候補用語とスコアを記録（統計情報付き）

        Args:
            stage_name: 段階名
            candidates: {term: frequency}
            tfidf_scores: TF-IDFスコア
            cvalue_scores: C-valueスコア
            base_scores: 基底スコア
            enhanced_scores: SemReRank後のスコア
        """
        current_terms = list(candidates.keys())

        # メタデータを構築
        metadata = {}
        for term in candidates:
            term_meta = {"frequency": candidates[term]}

            if tfidf_scores and term in tfidf_scores:
                term_meta["tfidf_score"] = tfidf_scores[term]
            if cvalue_scores and term in cvalue_scores:
                term_meta["cvalue_score"] = cvalue_scores[term]
            if base_scores and term in base_scores:
                term_meta["base_score"] = base_scores[term]
            if enhanced_scores and term in enhanced_scores:
                term_meta["enhanced_score"] = enhanced_scores[term]

            metadata[term] = term_meta

        self.track_stage(stage_name, current_terms, metadata)

    def track_llm_filter(
        self,
        stage_name: str,
        passed_terms: List[str],
        rejected_terms: List[str],
        rejection_reasons: Optional[Dict[str, str]] = None
    ):
        """
        LLMフィルタの結果を記録

        Args:
            stage_name: 段階名（例: "軽量LLMフィルタ", "重量LLMフィルタ"）
            passed_terms: フィルタを通過した用語
            rejected_terms: 除外された用語
            rejection_reasons: 除外理由 {term: reason}
        """
        rejection_reasons = rejection_reasons or {}

        # デフォルトの除外理由
        removed_with_reasons = {
            term: rejection_reasons.get(term, f"{stage_name}で除外")
            for term in rejected_terms
        }

        self.track_stage(stage_name, passed_terms, removed_terms_with_reasons=removed_with_reasons)

    def generate_report(self) -> Dict[str, Any]:
        """
        追跡レポートを生成

        Returns:
            {
                "ground_truth_terms": [...],
                "summary": {...},
                "dropout_by_stage": {...},
                "extraction_funnel": [...],
                "missed_terms": [...]
            }
        """
        # 各用語の最終状態を判定
        extracted_terms = []
        missed_terms = []

        for term, trace in self.traces.items():
            if not trace:
                # トレースがない（バグ）
                missed_terms.append({
                    "term": term,
                    "final_status": "no_trace",
                    "dropout_stage": "unknown",
                    "trace": []
                })
                continue

            # 最終段階で存在するか
            final_trace = trace[-1]
            if final_trace.get("status") == "present":
                extracted_terms.append({
                    "term": term,
                    "final_status": "extracted",
                    "trace": trace
                })
            else:
                # 脱落した段階を特定
                dropout_stage = self._find_dropout_stage(trace)
                missed_terms.append({
                    "term": term,
                    "final_status": "missed",
                    "dropout_stage": dropout_stage,
                    "trace": trace
                })

        # 段階別脱落数を集計
        dropout_by_stage = self._count_dropout_by_stage(missed_terms)

        # 抽出ファネル（各段階での残存数）
        extraction_funnel = self._build_extraction_funnel()

        # サマリー
        summary = {
            "total_ground_truth": len(self.ground_truth),
            "extracted": len(extracted_terms),
            "missed": len(missed_terms),
            "recall": len(extracted_terms) / len(self.ground_truth) if self.ground_truth else 0.0,
            "stages": self.stage_order
        }

        report = {
            "generation_time": datetime.now().isoformat(),
            "ground_truth_terms": extracted_terms + missed_terms,
            "summary": summary,
            "dropout_by_stage": dropout_by_stage,
            "extraction_funnel": extraction_funnel,
            "missed_terms": missed_terms
        }

        return report

    def _find_dropout_stage(self, trace: List[Dict[str, Any]]) -> str:
        """
        用語が脱落した段階を特定

        Args:
            trace: 用語のトレース履歴

        Returns:
            脱落した段階名
        """
        for i, entry in enumerate(trace):
            if entry.get("status") == "absent":
                # この段階で初めて不在になった
                if i > 0 and trace[i-1].get("status") == "present":
                    return entry["stage"]
                elif i == 0:
                    # 最初から不在
                    return entry["stage"]

        # 最終段階で不在
        if trace and trace[-1].get("status") == "absent":
            return trace[-1]["stage"]

        return "unknown"

    def _count_dropout_by_stage(self, missed_terms: List[Dict]) -> Dict[str, int]:
        """段階別の脱落数をカウント"""
        dropout_counts = Counter(term["dropout_stage"] for term in missed_terms)
        return dict(dropout_counts)

    def _build_extraction_funnel(self) -> List[Dict[str, Any]]:
        """
        抽出ファネルを構築（各段階での残存数）

        Returns:
            [{"stage": "候補抽出", "remaining": 45, "dropout": 5}, ...]
        """
        funnel = []

        for stage_idx, stage_name in enumerate(self.stage_order):
            # この段階でpresentな用語数をカウント
            remaining = 0
            for term, trace in self.traces.items():
                if stage_idx < len(trace) and trace[stage_idx].get("status") == "present":
                    remaining += 1

            # 前段階からの脱落数
            if stage_idx == 0:
                dropout = len(self.ground_truth) - remaining
            else:
                prev_remaining = funnel[-1]["remaining"]
                dropout = prev_remaining - remaining

            funnel.append({
                "stage": stage_name,
                "stage_index": stage_idx,
                "remaining": remaining,
                "dropout": dropout
            })

        return funnel

    def save_report(self, output_path: str) -> str:
        """
        レポートをJSON形式で保存

        Args:
            output_path: 出力ファイルパス

        Returns:
            保存されたファイルパス
        """
        report = self.generate_report()

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"Dropout report saved to: {output_file}")
        return str(output_file)

    def print_summary(self):
        """サマリーをコンソールに出力"""
        report = self.generate_report()
        summary = report["summary"]

        print("=" * 80)
        print("Ground Truth追跡レポート - サマリー")
        print("=" * 80)
        print(f"Ground Truth総数: {summary['total_ground_truth']}")
        print(f"抽出成功: {summary['extracted']} ({summary['recall']:.1%})")
        print(f"脱落: {summary['missed']} ({1-summary['recall']:.1%})")
        print()

        print("段階別脱落数:")
        dropout_by_stage = report["dropout_by_stage"]
        for stage in summary['stages']:
            count = dropout_by_stage.get(stage, 0)
            if count > 0:
                print(f"  {stage}: {count}件")
        print()

        print("抽出ファネル:")
        for entry in report["extraction_funnel"]:
            print(f"  {entry['stage']}: {entry['remaining']}件残存 "
                  f"({entry['dropout']}件脱落)")
        print("=" * 80)
