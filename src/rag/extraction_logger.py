"""
専門用語抽出過程のログ収集・保存機能
"""
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Set
from collections import defaultdict

logger = logging.getLogger(__name__)


class ExtractionLogger:
    """
    専門用語抽出の各段階でログを収集し、JSONとテキストで保存

    保存される情報:
    - 各段階での用語リスト
    - 除外された用語とその理由
    - 統計情報
    - タイムスタンプ
    """

    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ログデータ
        self.stages = []
        self.current_stage = None
        self.start_time = datetime.now()

        # 各段階での用語セット（変化を追跡）
        self.stage_terms = {}

    def start_stage(self, stage_name: str, description: str = ""):
        """新しい段階を開始"""
        self.current_stage = {
            "name": stage_name,
            "description": description,
            "timestamp": datetime.now().isoformat(),
            "terms_before": [],
            "terms_after": [],
            "removed_terms": [],
            "added_terms": [],
            "statistics": {},
            "details": []
        }
        logger.info(f"Stage started: {stage_name}")

    def log_terms_before(self, terms: List[str]):
        """処理前の用語リストを記録"""
        if self.current_stage:
            self.current_stage["terms_before"] = sorted(terms)

    def log_terms_after(self, terms: List[str]):
        """処理後の用語リストを記録"""
        if self.current_stage:
            self.current_stage["terms_after"] = sorted(terms)

            # 変化を計算
            before = set(self.current_stage["terms_before"])
            after = set(terms)
            self.current_stage["removed_terms"] = sorted(before - after)
            self.current_stage["added_terms"] = sorted(after - before)

    def log_statistics(self, stats: Dict[str, Any]):
        """統計情報を記録"""
        if self.current_stage:
            self.current_stage["statistics"].update(stats)

    def log_detail(self, message: str, data: Any = None):
        """詳細情報を記録"""
        if self.current_stage:
            detail = {
                "timestamp": datetime.now().isoformat(),
                "message": message
            }
            if data is not None:
                detail["data"] = data
            self.current_stage["details"].append(detail)

    def log_removed_terms_with_reason(self, terms_with_reasons: Dict[str, str]):
        """除外された用語とその理由を記録"""
        if self.current_stage:
            if "removed_terms_details" not in self.current_stage:
                self.current_stage["removed_terms_details"] = {}
            self.current_stage["removed_terms_details"].update(terms_with_reasons)

    def end_stage(self):
        """現在の段階を終了"""
        if self.current_stage:
            # 統計を追加
            if not self.current_stage["statistics"]:
                self.current_stage["statistics"] = {
                    "terms_before_count": len(self.current_stage["terms_before"]),
                    "terms_after_count": len(self.current_stage["terms_after"]),
                    "removed_count": len(self.current_stage["removed_terms"]),
                    "added_count": len(self.current_stage["added_terms"])
                }

            self.stages.append(self.current_stage)
            logger.info(f"Stage ended: {self.current_stage['name']}")
            self.current_stage = None

    def save(self, base_filename: str = "extraction_log"):
        """ログをJSONとテキストで保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # JSONファイルに保存
        json_path = self.output_dir / f"{base_filename}_{timestamp}.json"
        log_data = {
            "extraction_date": self.start_time.isoformat(),
            "total_duration_seconds": (datetime.now() - self.start_time).total_seconds(),
            "stages": self.stages,
            "summary": self._generate_summary()
        }

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Extraction log saved: {json_path}")

        # テキストファイルに保存（人間が読みやすい形式）
        txt_path = self.output_dir / f"{base_filename}_{timestamp}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            self._write_text_log(f, log_data)

        logger.info(f"Extraction log (text) saved: {txt_path}")

        return str(json_path), str(txt_path)

    def _generate_summary(self) -> Dict[str, Any]:
        """全体サマリーを生成"""
        if not self.stages:
            return {}

        first_stage_terms = set(self.stages[0].get("terms_before", []))
        last_stage_terms = set(self.stages[-1].get("terms_after", []))

        return {
            "initial_term_count": len(first_stage_terms),
            "final_term_count": len(last_stage_terms),
            "total_removed": len(first_stage_terms - last_stage_terms),
            "total_stages": len(self.stages),
            "stage_names": [s["name"] for s in self.stages]
        }

    def _write_text_log(self, f, log_data: Dict):
        """テキスト形式でログを出力"""
        f.write("=" * 80 + "\n")
        f.write("専門用語抽出ログ\n")
        f.write("=" * 80 + "\n")
        f.write(f"抽出日時: {log_data['extraction_date']}\n")
        f.write(f"処理時間: {log_data['total_duration_seconds']:.2f}秒\n")
        f.write("\n")

        # サマリー
        summary = log_data['summary']
        f.write("=" * 80 + "\n")
        f.write("サマリー\n")
        f.write("=" * 80 + "\n")
        f.write(f"初期候補数: {summary.get('initial_term_count', 0)}個\n")
        f.write(f"最終用語数: {summary.get('final_term_count', 0)}個\n")
        f.write(f"総除外数: {summary.get('total_removed', 0)}個\n")
        f.write(f"処理段階数: {summary.get('total_stages', 0)}段階\n")
        f.write("\n")

        # 各段階の詳細
        for i, stage in enumerate(log_data['stages'], 1):
            f.write("=" * 80 + "\n")
            f.write(f"段階 {i}: {stage['name']}\n")
            f.write("=" * 80 + "\n")

            if stage.get('description'):
                f.write(f"説明: {stage['description']}\n")

            f.write(f"タイムスタンプ: {stage['timestamp']}\n")
            f.write("\n")

            # 統計
            stats = stage.get('statistics', {})
            f.write("統計:\n")
            f.write(f"  処理前: {stats.get('terms_before_count', 0)}個\n")
            f.write(f"  処理後: {stats.get('terms_after_count', 0)}個\n")
            f.write(f"  除外: {stats.get('removed_count', 0)}個\n")
            f.write(f"  追加: {stats.get('added_count', 0)}個\n")

            # 追加の統計情報
            for key, value in stats.items():
                if key not in ['terms_before_count', 'terms_after_count', 'removed_count', 'added_count']:
                    f.write(f"  {key}: {value}\n")
            f.write("\n")

            # 除外された用語
            removed = stage.get('removed_terms', [])
            if removed:
                f.write(f"除外された用語 ({len(removed)}個):\n")
                removed_details = stage.get('removed_terms_details', {})

                # 理由別にグループ化
                by_reason = defaultdict(list)
                for term in removed:
                    reason = removed_details.get(term, "理由不明")
                    by_reason[reason].append(term)

                for reason, terms in sorted(by_reason.items()):
                    f.write(f"\n  理由: {reason} ({len(terms)}個)\n")
                    # 最大20個まで表示
                    for term in sorted(terms)[:20]:
                        f.write(f"    - {term}\n")
                    if len(terms) > 20:
                        f.write(f"    ... 他 {len(terms) - 20}個\n")

            # 追加された用語
            added = stage.get('added_terms', [])
            if added:
                f.write(f"\n追加された用語 ({len(added)}個):\n")
                for term in sorted(added)[:20]:
                    f.write(f"  - {term}\n")
                if len(added) > 20:
                    f.write(f"  ... 他 {len(added) - 20}個\n")

            # 詳細情報
            details = stage.get('details', [])
            if details:
                f.write(f"\n詳細 ({len(details)}件):\n")
                for detail in details[:10]:
                    f.write(f"  [{detail['timestamp']}] {detail['message']}\n")
                    if 'data' in detail:
                        f.write(f"    データ: {detail['data']}\n")
                if len(details) > 10:
                    f.write(f"  ... 他 {len(details) - 10}件\n")

            f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("ログ終了\n")
        f.write("=" * 80 + "\n")
