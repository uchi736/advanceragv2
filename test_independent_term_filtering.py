"""
独立出現率フィルタリング機能のテスト
"""

import sys
sys.path.insert(0, 'src')

from rag.advanced_term_extraction import AdvancedStatisticalExtractor

# テストテキスト：「システム」は常に複合語の一部、「エンジン」は独立使用も多い
sample_text = """
製造システムは品質管理システムと連携している。
品質管理システムでは、製造システムからのデータを分析する。
管理システムの導入により、製造システムの効率が向上した。
製造システムと管理システムを統合し、品質管理システムを強化する。
システムの統合により、製造と品質管理の両システムが改善された。

一方、エンジンの性能は重要である。
エンジンを改良し、エンジンの燃費を向上させる。
エンジンは高性能である。エンジン開発チームがエンジンを設計した。
エンジン技術の進歩により、エンジン効率が向上した。
新しいエンジンは、従来のエンジンより優れている。
ディーゼルエンジンとガスエンジンを比較する。
エンジンの改善が進んでいる。
"""

def main():
    print("=" * 60)
    print("独立出現率フィルタリング機能のテスト")
    print("=" * 60)

    # 抽出器の初期化
    extractor = AdvancedStatisticalExtractor(
        min_term_length=2,
        max_term_length=6,
        min_frequency=2
    )

    # 用語抽出
    print("\n用語抽出中...")
    terms = extractor.extract_terms(sample_text, top_n=30)

    print(f"\n抽出された用語数: {len(terms)}")
    print("\n" + "=" * 60)
    print("抽出結果:")
    print("=" * 60)

    # 結果表示
    for i, term in enumerate(terms, 1):
        independent_ratio = term.metadata.get('independent_ratio', 0.0)
        print(f"{i:2}. {term.term:20} | Score: {term.score:.4f} | "
              f"独立率: {independent_ratio:.1%} | Freq: {term.frequency}")

    # 検証ポイント
    print("\n" + "=" * 60)
    print("検証ポイント:")
    print("=" * 60)

    extracted_terms_set = {t.term for t in terms}

    # 期待される動作
    checks = [
        ("エンジン", True, "独立して使われる重要語 → 抽出されるべき"),
        ("ディーゼルエンジン", True, "複合語として重要 → 抽出されるべき"),
        ("ガスエンジン", True, "複合語として重要 → 抽出されるべき"),
        ("システム", False, "単独で使われない → 除外されるべき"),
        ("製造システム", True, "複合語として重要 → 抽出されるべき"),
        ("管理システム", True, "複合語として重要 → 抽出されるべき"),
        ("品質管理システム", True, "複合語として重要 → 抽出されるべき"),
    ]

    all_passed = True
    for term, should_exist, reason in checks:
        exists = term in extracted_terms_set
        status = "[OK]" if exists == should_exist else "[NG]"

        if exists != should_exist:
            all_passed = False

        print(f"{status} {term:20} | Expected: {'Extract' if should_exist else 'Filter':7} | "
              f"Actual: {'Extract' if exists else 'Filter':7} | {reason}")

    print("\n" + "=" * 60)
    if all_passed:
        print("[PASS] All checks passed!")
    else:
        print("[FAIL] Some checks failed.")
    print("=" * 60)

if __name__ == "__main__":
    main()
