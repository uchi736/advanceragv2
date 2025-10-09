"""
独立出現率の計算ロジックをデバッグ
"""

import sys
sys.path.insert(0, 'src')

from rag.advanced_term_extraction import AdvancedStatisticalExtractor

# テストテキスト
sample_text = """
エンジンの性能は重要である。
ディーゼルエンジンとガスエンジンを比較すると、エンジン効率が異なる。
エンジンを改良し、ディーゼルエンジンの燃費を向上させる。
新型エンジンは、従来のガスエンジンより高性能である。
エンジン開発チームが、次世代ディーゼルエンジンを設計した。
"""

def debug_independent_ratio():
    print("=" * 60)
    print("独立出現率の計算デバッグ")
    print("=" * 60)

    # 手動カウント
    print("\n【手動カウント】")
    print(f"「エンジン」の全出現回数: {sample_text.count('エンジン')}")
    print(f"「ディーゼルエンジン」の出現回数: {sample_text.count('ディーゼルエンジン')}")
    print(f"「ガスエンジン」の出現回数: {sample_text.count('ガスエンジン')}")
    print(f"「新型エンジン」の出現回数: {sample_text.count('新型エンジン')}")
    print(f"「次世代ディーゼルエンジン」の出現回数: {sample_text.count('次世代ディーゼルエンジン')}")

    # 複合語内出現 = 3 + 2 + 1 + 1 = 7
    # 独立出現 = 10 - 7 = 3
    # 独立率 = 3 / 10 = 30%

    print("\n【期待される計算】")
    print("複合語内出現: 3(ディーゼル) + 2(ガス) + 1(新型) + 1(次世代) = 7")
    print("独立出現: 10 - 7 = 3")
    print("独立率: 3 / 10 = 30.0%")

    # 実際の計算
    extractor = AdvancedStatisticalExtractor()
    candidates_list = ['エンジン', 'ディーゼルエンジン', 'ガスエンジン', '新型エンジン', '次世代ディーゼルエンジン']

    ratio = extractor._calculate_independent_occurrence_ratio(
        'エンジン',
        sample_text,
        candidates_list
    )

    print("\n【実際の計算結果】")
    print(f"独立率: {ratio:.1%}")

    if abs(ratio - 0.3) < 0.01:
        print("✓ 計算結果が正しい")
    else:
        print("✗ 計算結果が期待と異なる")

if __name__ == "__main__":
    debug_independent_ratio()
