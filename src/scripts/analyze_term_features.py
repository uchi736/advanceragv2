"""
専門用語抽出の特徴分析CLIスクリプト

Usage:
    python src/scripts/analyze_term_features.py

デフォルト:
    - Ground Truth: test_data/ground_truth.json
    - 抽出結果: test_data/extracted_terms.json
    - サンプル文書: test_data/sample_docs/*.txt
    - 出力先: output/analysis/
"""

import sys
import json
import logging
from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')  # GUIなし環境対応
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rag.term_analysis import (
    TermFeatureAnalyzer,
    load_ground_truth,
    load_extracted_terms,
    load_documents
)

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def plot_recall_by_category(results: dict, output_path: Path):
    """カテゴリ別Recallの棒グラフ"""
    category_analysis = results['category_analysis']

    categories = []
    recalls = []
    totals = []

    for category, data in sorted(category_analysis.items(), key=lambda x: x[1]['recall'], reverse=True):
        categories.append(category)
        recalls.append(data['recall'] * 100)  # パーセント表示
        totals.append(data['total'])

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(categories, recalls, color='steelblue', alpha=0.8)

    # 値をバーの上に表示
    for i, (bar, recall, total) in enumerate(zip(bars, recalls, totals)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{recall:.1f}%\n(n={total})',
                ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Category', fontsize=12)
    ax.set_ylabel('Recall (%)', fontsize=12)
    ax.set_title('Recall by Category', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved: {output_path}")


def plot_recall_by_frequency(results: dict, output_path: Path):
    """頻度別Recallの棒グラフ"""
    freq_analysis = results['frequency_analysis']

    # 頻度範囲の順序を保持
    freq_order = ["1回", "2回", "3-5回", "6-10回", "11回以上"]
    freq_labels = []
    recalls = []
    totals = []

    for label in freq_order:
        if label in freq_analysis:
            data = freq_analysis[label]
            freq_labels.append(label)
            recalls.append(data['recall'] * 100)
            totals.append(data['total'])

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(freq_labels, recalls, color='coral', alpha=0.8)

    # 値をバーの上に表示
    for bar, recall, total in zip(bars, recalls, totals):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{recall:.1f}%\n(n={total})',
                ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Frequency Range', fontsize=12)
    ax.set_ylabel('Recall (%)', fontsize=12)
    ax.set_title('Recall by Frequency', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved: {output_path}")


def plot_score_vs_frequency(results: dict, output_dir: Path):
    """TF-IDF/C-value vs 頻度の散布図"""
    term_details = results['term_details']

    df = pd.DataFrame(term_details)

    # TF-IDF vs 頻度
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 抽出された用語 vs 見逃された用語
    extracted_df = df[df['is_extracted']]
    missed_df = df[~df['is_extracted']]

    ax1.scatter(extracted_df['frequency'], extracted_df['tfidf_score'],
                alpha=0.6, label='Extracted', color='green', s=50)
    ax1.scatter(missed_df['frequency'], missed_df['tfidf_score'],
                alpha=0.6, label='Missed', color='red', s=50, marker='x')
    ax1.set_xlabel('Frequency', fontsize=11)
    ax1.set_ylabel('TF-IDF Score', fontsize=11)
    ax1.set_title('TF-IDF vs Frequency', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # C-value vs 頻度
    ax2.scatter(extracted_df['frequency'], extracted_df['cvalue_score'],
                alpha=0.6, label='Extracted', color='green', s=50)
    ax2.scatter(missed_df['frequency'], missed_df['cvalue_score'],
                alpha=0.6, label='Missed', color='red', s=50, marker='x')
    ax2.set_xlabel('Frequency', fontsize=11)
    ax2.set_ylabel('C-value Score', fontsize=11)
    ax2.set_title('C-value vs Frequency', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'score_vs_frequency.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved: {output_path}")


def plot_category_score_distribution(results: dict, output_path: Path):
    """カテゴリ別スコア分布（箱ひげ図）"""
    term_details = results['term_details']

    df = pd.DataFrame(term_details)

    # 抽出された用語のみ（スコアがあるもの）
    extracted_df = df[df['is_extracted']]

    if extracted_df.empty:
        logger.warning("No extracted terms found for box plot")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # カテゴリ別に最終スコアの分布
    categories = extracted_df['category'].unique()
    data_to_plot = [extracted_df[extracted_df['category'] == cat]['final_score'].values
                    for cat in categories]

    bp = ax.boxplot(data_to_plot, labels=categories, patch_artist=True)

    # 色付け
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)

    ax.set_xlabel('Category', fontsize=12)
    ax.set_ylabel('Final Score', fontsize=12)
    ax.set_title('Score Distribution by Category (Extracted Terms Only)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved: {output_path}")


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description='専門用語抽出の特徴分析')
    parser.add_argument('--ground-truth', type=str, default='test_data/ground_truth.json',
                        help='Ground TruthファイルのPATH')
    parser.add_argument('--extracted', type=str, default='test_data/extracted_terms.json',
                        help='抽出結果ファイルのPATH')
    parser.add_argument('--docs-dir', type=str, default='test_data/sample_docs',
                        help='サンプル文書ディレクトリのPATH')
    parser.add_argument('--output-dir', type=str, default='output/analysis',
                        help='出力ディレクトリのPATH')

    args = parser.parse_args()

    # パス設定
    gt_path = Path(args.ground_truth)
    extracted_path = Path(args.extracted)
    docs_dir = Path(args.docs_dir)
    output_dir = Path(args.output_dir)

    # 出力ディレクトリ作成
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)

    # データ読み込み
    logger.info("Loading data...")
    ground_truth = load_ground_truth(gt_path)
    extracted_terms = load_extracted_terms(extracted_path)

    # サンプル文書読み込み
    doc_files = sorted(docs_dir.glob('*.txt'))
    if not doc_files:
        logger.error(f"No .txt files found in {docs_dir}")
        sys.exit(1)

    documents = load_documents(doc_files)
    logger.info(f"Loaded {len(documents)} documents")

    # 分析実行
    logger.info("Running analysis...")
    analyzer = TermFeatureAnalyzer(ground_truth, extracted_terms, documents)
    results = analyzer.analyze()

    # 結果出力
    logger.info("Generating outputs...")

    # 1. JSON出力
    json_path = output_dir / 'term_analysis.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        # numpy型をJSONシリアライズ可能に変換
        def convert(o):
            if isinstance(o, (np.int64, np.int32)):
                return int(o)
            if isinstance(o, (np.float64, np.float32)):
                return float(o)
            raise TypeError

        json.dump(results, f, ensure_ascii=False, indent=2, default=convert)
    logger.info(f"Saved: {json_path}")

    # 2. Markdownレポート
    md_report = analyzer.generate_markdown_report(results)
    md_path = output_dir / 'analysis_report.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_report)
    logger.info(f"Saved: {md_path}")

    # 3. グラフ生成
    logger.info("Generating figures...")
    plot_recall_by_category(results, figures_dir / 'recall_by_category.png')
    plot_recall_by_frequency(results, figures_dir / 'recall_by_frequency.png')
    plot_score_vs_frequency(results, figures_dir)
    plot_category_score_distribution(results, figures_dir / 'category_score_distribution.png')

    # 4. サマリー表示
    metrics = results['overall_metrics']
    logger.info("\n" + "="*60)
    logger.info("ANALYSIS SUMMARY")
    logger.info("="*60)
    logger.info(f"Ground Truth: {metrics['total_ground_truth']} terms")
    logger.info(f"Extracted: {metrics['total_extracted']} terms")
    logger.info(f"Recall: {metrics['recall']:.2%}")
    logger.info(f"Precision: {metrics['precision']:.2%}")
    logger.info(f"F1 Score: {metrics['f1_score']:.2%}")
    logger.info("="*60)
    logger.info(f"\nFull report: {md_path}")
    logger.info(f"Figures: {figures_dir}")
    logger.info("\nAnalysis completed successfully!")


if __name__ == '__main__':
    main()
