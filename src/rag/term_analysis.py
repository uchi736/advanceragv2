"""
専門用語の特徴分析モジュール

Ground Truthとの比較により、TF-IDF+C-valueアプローチの有効性を検証
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict, Counter
import numpy as np

logger = logging.getLogger(__name__)


class TermFeatureAnalyzer:
    """専門用語の特徴分析クラス"""

    def __init__(
        self,
        ground_truth: Dict[str, List[str]],
        extracted_terms: List[Dict],
        documents: List[str]
    ):
        """
        Args:
            ground_truth: {"document1": ["用語1", "用語2"], ...} or {"all_documents": [...]}
            extracted_terms: [{"headword": "用語", ...}, ...] or [{"term": "用語", ...}, ...]
            documents: ドキュメントテキストのリスト
        """
        self.ground_truth = ground_truth
        self.extracted_terms = extracted_terms
        self.documents = documents

        # Ground truthの正規化（all_documentsキーを優先）
        if "all_documents" in ground_truth:
            self.gt_terms = set(ground_truth["all_documents"])
        else:
            # 全ドキュメントの用語を統合
            all_terms = []
            for doc_terms in ground_truth.values():
                all_terms.extend(doc_terms)
            self.gt_terms = set(all_terms)

        # 抽出用語のセット（headwordとterm両方に対応）
        self.extracted_set = set(
            term.get("headword") or term.get("term")
            for term in extracted_terms
        )

        # 全テキスト結合
        self.full_text = "\n\n".join(documents)

        logger.info(f"Ground Truth: {len(self.gt_terms)} terms")
        logger.info(f"Extracted: {len(self.extracted_set)} terms")

    def _get_term_name(self, term_dict: Dict) -> str:
        """用語辞書から用語名を取得（headwordまたはterm）"""
        return term_dict.get("headword") or term_dict.get("term") or ""

    def analyze(self) -> Dict[str, Any]:
        """分析実行のメインメソッド"""
        logger.info("Starting term feature analysis...")

        results = {
            'overall_metrics': self._calculate_overall_metrics(),
            'category_analysis': self._analyze_by_category(),
            'frequency_analysis': self._analyze_by_frequency(),
            'score_analysis': self._analyze_scores(),
            'missed_terms': self._find_missed_terms(),
            'false_positives': self._find_false_positives(),
            'term_details': self._create_term_details()
        }

        logger.info("Analysis completed")
        return results

    def _calculate_overall_metrics(self) -> Dict:
        """Recall/Precision/F1の計算"""
        true_positives = self.gt_terms & self.extracted_set
        false_negatives = self.gt_terms - self.extracted_set
        false_positives = self.extracted_set - self.gt_terms

        recall = len(true_positives) / len(self.gt_terms) if self.gt_terms else 0
        precision = len(true_positives) / len(self.extracted_set) if self.extracted_set else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'total_ground_truth': len(self.gt_terms),
            'total_extracted': len(self.extracted_set),
            'true_positives': len(true_positives),
            'false_negatives': len(false_negatives),
            'false_positives': len(false_positives),
            'recall': recall,
            'precision': precision,
            'f1_score': f1
        }

    def _count_frequency(self, term: str) -> int:
        """用語の出現回数をカウント"""
        return self.full_text.count(term)

    def _classify_morphology(self, term: str) -> str:
        """形態分類: 略語/数値付き/複合語/単一名詞"""
        # 略語判定
        if re.match(r'^[A-Z]{2,5}$', term):
            return 'acronym'

        # 数値付き判定
        if re.search(r'\d', term):
            return 'numeric'

        # 語数カウント（簡易的）
        word_count = len(term)  # 日本語は文字数で近似

        if word_count == 1:
            return 'single_char'
        elif word_count <= 3:
            return 'short_compound'
        elif word_count <= 6:
            return 'medium_compound'
        else:
            return 'long_compound'

    def _categorize_term(self, term: str, frequency: int) -> str:
        """カテゴリ分類"""
        morphology = self._classify_morphology(term)

        # 略語は特別カテゴリ
        if morphology == 'acronym':
            return 'acronym'

        # 数値付きも特別カテゴリ
        if morphology == 'numeric':
            return 'numeric'

        # 語数と頻度で分類
        word_count = len(term)

        if word_count == 1:
            return 'single_char'
        elif frequency >= 3:
            return 'high_freq_compound'
        else:
            return 'low_freq_compound'

    def _analyze_by_category(self) -> Dict:
        """カテゴリ別Recall分析"""
        # Ground truth用語をカテゴリ分類
        category_gt = defaultdict(set)
        for term in self.gt_terms:
            freq = self._count_frequency(term)
            category = self._categorize_term(term, freq)
            category_gt[category].add(term)

        # カテゴリ別Recall計算
        category_analysis = {}
        for category, terms in category_gt.items():
            extracted_in_category = terms & self.extracted_set
            recall = len(extracted_in_category) / len(terms) if terms else 0

            category_analysis[category] = {
                'total': len(terms),
                'extracted': len(extracted_in_category),
                'missed': len(terms - self.extracted_set),
                'recall': recall,
                'terms': list(terms)[:10]  # サンプル10件
            }

        return category_analysis

    def _analyze_by_frequency(self) -> Dict:
        """頻度別Recall分析"""
        freq_bins = [
            (1, 1, "1回"),
            (2, 2, "2回"),
            (3, 5, "3-5回"),
            (6, 10, "6-10回"),
            (11, float('inf'), "11回以上")
        ]

        freq_analysis = {}
        for min_freq, max_freq, label in freq_bins:
            terms_in_bin = set()
            for term in self.gt_terms:
                freq = self._count_frequency(term)
                if min_freq <= freq <= max_freq:
                    terms_in_bin.add(term)

            if terms_in_bin:
                extracted_in_bin = terms_in_bin & self.extracted_set
                recall = len(extracted_in_bin) / len(terms_in_bin)

                freq_analysis[label] = {
                    'total': len(terms_in_bin),
                    'extracted': len(extracted_in_bin),
                    'recall': recall
                }

        return freq_analysis

    def _analyze_scores(self) -> Dict:
        """TF-IDF/C-valueスコアの分析"""
        # 抽出用語のスコア情報を取得
        extracted_dict = {self._get_term_name(term): term for term in self.extracted_terms}

        # Ground truth用語のスコア分布
        gt_scores = {
            'tfidf': [],
            'cvalue': [],
            'final': []
        }

        for term in self.gt_terms:
            if term in extracted_dict:
                info = extracted_dict[term]
                gt_scores['tfidf'].append(info.get('tfidf_score', 0))
                gt_scores['cvalue'].append(info.get('cvalue_score', 0))
                gt_scores['final'].append(info.get('score', 0))
            else:
                # 抽出されなかった用語はスコア0
                gt_scores['tfidf'].append(0)
                gt_scores['cvalue'].append(0)
                gt_scores['final'].append(0)

        # 統計量計算
        score_stats = {}
        for score_type, scores in gt_scores.items():
            score_stats[score_type] = {
                'mean': np.mean(scores) if scores else 0,
                'median': np.median(scores) if scores else 0,
                'std': np.std(scores) if scores else 0,
                'min': np.min(scores) if scores else 0,
                'max': np.max(scores) if scores else 0
            }

        return {
            'score_statistics': score_stats,
            'ground_truth_scores': gt_scores
        }

    def _find_missed_terms(self) -> List[Dict]:
        """見逃された用語（False Negatives）の詳細"""
        missed = self.gt_terms - self.extracted_set

        missed_details = []
        for term in sorted(missed):
            freq = self._count_frequency(term)
            category = self._categorize_term(term, freq)
            morphology = self._classify_morphology(term)

            missed_details.append({
                'term': term,
                'frequency': freq,
                'category': category,
                'morphology': morphology,
                'word_count': len(term)
            })

        # 頻度順にソート
        missed_details.sort(key=lambda x: x['frequency'], reverse=True)

        return missed_details

    def _find_false_positives(self) -> List[Dict]:
        """誤検出された用語（False Positives）の詳細"""
        false_pos = self.extracted_set - self.gt_terms

        extracted_dict = {self._get_term_name(term): term for term in self.extracted_terms}

        fp_details = []
        for term in sorted(false_pos):
            info = extracted_dict.get(term, {})
            freq = self._count_frequency(term)

            fp_details.append({
                'term': term,
                'frequency': freq,
                'score': info.get('score', 0),
                'tfidf_score': info.get('tfidf_score', 0),
                'cvalue_score': info.get('cvalue_score', 0)
            })

        # スコア順にソート
        fp_details.sort(key=lambda x: x['score'], reverse=True)

        return fp_details

    def _create_term_details(self) -> List[Dict]:
        """全Ground truth用語の詳細情報"""
        extracted_dict = {self._get_term_name(term): term for term in self.extracted_terms}

        details = []
        for term in sorted(self.gt_terms):
            freq = self._count_frequency(term)
            category = self._categorize_term(term, freq)
            morphology = self._classify_morphology(term)
            is_extracted = term in self.extracted_set

            info = extracted_dict.get(term, {})

            details.append({
                'term': term,
                'is_extracted': is_extracted,
                'frequency': freq,
                'category': category,
                'morphology': morphology,
                'word_count': len(term),
                'tfidf_score': info.get('tfidf_score', 0),
                'cvalue_score': info.get('cvalue_score', 0),
                'final_score': info.get('score', 0)
            })

        return details

    def generate_markdown_report(self, results: Dict[str, Any]) -> str:
        """Markdownレポート生成"""
        metrics = results['overall_metrics']
        category_analysis = results['category_analysis']
        freq_analysis = results['frequency_analysis']

        md = f"""# 専門用語抽出 特徴分析レポート

## 1. 全体メトリクス

| 指標 | 値 |
|------|-----|
| Ground Truth用語数 | {metrics['total_ground_truth']} |
| 抽出された用語数 | {metrics['total_extracted']} |
| True Positives | {metrics['true_positives']} |
| False Negatives (見逃し) | {metrics['false_negatives']} |
| False Positives (誤検出) | {metrics['false_positives']} |
| **Recall** | **{metrics['recall']:.2%}** |
| **Precision** | **{metrics['precision']:.2%}** |
| **F1 Score** | **{metrics['f1_score']:.2%}** |

---

## 2. カテゴリ別Recall分析

TF-IDF+C-valueアプローチが**どのタイプの専門用語を捉えているか**を検証

| カテゴリ | Ground Truth数 | 抽出数 | Recall |
|---------|---------------|-------|---------|
"""

        for category, data in sorted(category_analysis.items(), key=lambda x: x[1]['recall'], reverse=True):
            md += f"| {category} | {data['total']} | {data['extracted']} | **{data['recall']:.1%}** |\n"

        md += "\n### カテゴリの定義\n\n"
        md += "- `acronym`: 略語（例: SFOC, EGR, BMS）\n"
        md += "- `numeric`: 数値付き（例: 4ストロークエンジン）\n"
        md += "- `high_freq_compound`: 高頻度複合語（頻度≥3, 語数≥2）\n"
        md += "- `low_freq_compound`: 低頻度複合語（頻度<3, 語数≥2）\n"
        md += "- `single_char`: 単一文字（語数=1）\n"

        md += "\n---\n\n## 3. 頻度別Recall分析\n\n"
        md += "| 頻度範囲 | Ground Truth数 | 抽出数 | Recall |\n"
        md += "|---------|---------------|-------|--------|\n"

        for freq_label, data in freq_analysis.items():
            md += f"| {freq_label} | {data['total']} | {data['extracted']} | **{data['recall']:.1%}** |\n"

        md += "\n---\n\n## 4. 見逃された用語 (Top 20)\n\n"
        missed = results['missed_terms'][:20]
        md += "| 用語 | 頻度 | カテゴリ | 形態 |\n"
        md += "|------|-----|---------|------|\n"
        for term_info in missed:
            md += f"| {term_info['term']} | {term_info['frequency']} | {term_info['category']} | {term_info['morphology']} |\n"

        md += "\n---\n\n## 5. 誤検出された用語 (Top 20)\n\n"
        false_pos = results['false_positives'][:20]
        md += "| 用語 | 頻度 | 最終スコア | TF-IDF | C-value |\n"
        md += "|------|-----|-----------|--------|--------|\n"
        for term_info in false_pos:
            md += f"| {term_info['term']} | {term_info['frequency']} | {term_info['score']:.3f} | {term_info['tfidf_score']:.2f} | {term_info['cvalue_score']:.2f} |\n"

        md += "\n---\n\n## 6. 考察\n\n"

        # カテゴリ別Recallから考察
        high_recall_categories = [cat for cat, data in category_analysis.items() if data['recall'] > 0.7]
        low_recall_categories = [cat for cat, data in category_analysis.items() if data['recall'] < 0.5]

        md += "### 強み（Recall > 70%）\n\n"
        if high_recall_categories:
            md += "TF-IDF+C-valueは以下のタイプの用語を効果的に抽出:\n"
            for cat in high_recall_categories:
                recall = category_analysis[cat]['recall']
                md += f"- **{cat}**: {recall:.1%}\n"
        else:
            md += "高Recallのカテゴリなし\n"

        md += "\n### 弱み（Recall < 50%）\n\n"
        if low_recall_categories:
            md += "以下のタイプの用語は見逃されやすい:\n"
            for cat in low_recall_categories:
                recall = category_analysis[cat]['recall']
                md += f"- **{cat}**: {recall:.1%}\n"
        else:
            md += "低Recallのカテゴリなし\n"

        md += "\n### 改善提案\n\n"

        if 'single_char' in low_recall_categories or 'low_freq_compound' in low_recall_categories:
            md += "1. **低頻度・単一名詞対策**:\n"
            md += "   - 定義パターン検出（「〜とは」「〜により」）\n"
            md += "   - 文脈ベーススコアリング（周辺の技術用語密度）\n\n"

        if metrics['false_positives'] > metrics['true_positives'] * 0.3:
            md += "2. **False Positive削減**:\n"
            md += "   - C-valueフィルタの厳格化\n"
            md += "   - LLM判定の強化\n\n"

        md += "\n---\n\n"
        md += f"*Generated by TermFeatureAnalyzer*\n"

        return md


def load_ground_truth(file_path: Path) -> Dict[str, List[str]]:
    """Ground Truthファイルの読み込み"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_extracted_terms(file_path: Path) -> List[Dict]:
    """抽出結果ファイルの読み込み"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data.get('terms', [])


def load_documents(file_paths: List[Path]) -> List[str]:
    """ドキュメントファイルの読み込み"""
    documents = []
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            documents.append(f.read())
    return documents
