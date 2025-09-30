"""
高度な統計的手法による専門用語抽出のテスト
"""

import asyncio
from pathlib import Path
from src.rag.advanced_term_extraction import AdvancedStatisticalExtractor

# サンプルテキスト（技術文書風）
sample_text = """
アンモニア燃料エンジンは、次世代の環境対応技術として注目されている。
このアンモニア燃料エンジンは、従来のディーゼルエンジンと比較して、
CO2排出量を大幅に削減できる。6DE-18型エンジンでは、NOx排出量も
50mg/kWh以下に抑えられている。

舶用ディーゼルエンジンの分野では、MARPOL規制に対応するため、
脱硝装置の搭載が必須となっている。特に、SCR（選択的触媒還元）システムは、
NOx削減技術の主流となっており、尿素水を還元剤として使用する。

燃料噴射システムの高圧化により、燃焼効率が向上し、
PM（粒子状物質）の排出量も低減される。コモンレール式燃料噴射装置では、
最大噴射圧力が200MPaを超える高圧化が進んでいる。

また、排ガス再循環（EGR）システムの採用により、
燃焼温度を制御し、NOx生成を抑制する技術も実用化されている。
これらの環境対応技術は、IMOの規制強化に対応するために不可欠である。

次世代船舶では、LNG燃料エンジンや水素燃料電池の採用も検討されており、
ゼロエミッション船舶の実現に向けた研究開発が加速している。
特に、アンモニア燃料は、カーボンニュートラルな代替燃料として期待されている。
"""

def test_advanced_extraction():
    """高度な統計的抽出のテスト"""
    print("=" * 60)
    print("高度な統計的手法による専門用語抽出テスト")
    print("=" * 60)

    # 抽出器の初期化
    extractor = AdvancedStatisticalExtractor(
        min_term_length=2,
        max_term_length=6,
        min_frequency=1,  # テスト用に低く設定
        use_regex_patterns=True
    )

    # 候補用語の抽出
    print("\n1. 候補用語の抽出")
    candidates = extractor.extract_candidates(sample_text)
    print(f"   候補用語数: {len(candidates)}")
    print(f"   上位10件: {list(candidates.keys())[:10]}")

    # 文書分割
    documents = extractor._split_into_sentences(sample_text)
    print(f"\n2. 文書分割")
    print(f"   文数: {len(documents)}")

    # TF-IDF計算
    print("\n3. TF-IDF計算")
    tfidf_scores = extractor.calculate_tfidf(documents, candidates)
    tfidf_sorted = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
    print("   TF-IDFスコア上位5件:")
    for term, score in tfidf_sorted[:5]:
        print(f"      {term:30} {score:.4f}")

    # C-value計算
    print("\n4. C-value計算")
    cvalue_scores = extractor.calculate_cvalue(candidates)
    cvalue_sorted = sorted(cvalue_scores.items(), key=lambda x: x[1], reverse=True)
    print("   C-valueスコア上位5件:")
    for term, score in cvalue_sorted[:5]:
        print(f"      {term:30} {score:.4f}")

    # 2段階スコアリング
    print("\n5. 2段階スコアリング")

    # シード選定用スコア（C-value重視）
    seed_scores = extractor.calculate_combined_scores(
        tfidf_scores, cvalue_scores, stage="seed"
    )
    seed_sorted = sorted(seed_scores.items(), key=lambda x: x[1], reverse=True)
    print("   シード選定スコア上位5件 (C-value重視: 0.7):")
    for term, score in seed_sorted[:5]:
        print(f"      {term:30} {score:.4f}")

    # 最終スコア（TF-IDF重視）
    final_scores = extractor.calculate_combined_scores(
        tfidf_scores, cvalue_scores, stage="final"
    )
    final_sorted = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    print("   最終スコア上位5件 (TF-IDF重視: 0.7):")
    for term, score in final_sorted[:5]:
        print(f"      {term:30} {score:.4f}")

    # 統合的な抽出
    print("\n6. 統合的な専門用語抽出")
    extracted_terms = extractor.extract_terms(sample_text, top_n=20)

    print(f"\n抽出された専門用語 (上位20件):")
    print("-" * 60)
    for i, term in enumerate(extracted_terms, 1):
        print(f"{i:2}. {term.term:30} Score: {term.score:.4f}")
        print(f"    TF-IDF: {term.tfidf_score:.4f}, C-value: {term.cvalue_score:.4f}")
        print(f"    Frequency: {term.frequency}, Seed Score: {term.metadata.get('seed_score', 0):.4f}")
        print()

    # 統計情報
    print("=" * 60)
    print("統計情報:")
    print(f"  総候補用語数: {len(candidates)}")
    print(f"  抽出用語数: {len(extracted_terms)}")
    print(f"  平均スコア: {sum(t.score for t in extracted_terms) / len(extracted_terms):.4f}")
    print(f"  最高スコア: {extracted_terms[0].score:.4f}")
    print(f"  最低スコア: {extracted_terms[-1].score:.4f}")

    # 専門用語の分類
    print("\n専門用語の分類:")

    # 型式番号・製品コード
    product_codes = [t for t in extracted_terms if any(c.isdigit() for c in t.term) and '-' in t.term]
    print(f"  型式番号・製品コード: {[t.term for t in product_codes]}")

    # 化学式・化合物
    chemicals = [t for t in extracted_terms if t.term in ['CO2', 'NOx', 'SOx', 'PM', 'NH3', 'H2O']]
    print(f"  化学式・化合物: {[t.term for t in chemicals]}")

    # 略語（3文字以上の大文字）
    acronyms = [t for t in extracted_terms if t.term.isupper() and len(t.term) >= 3]
    print(f"  略語: {[t.term for t in acronyms]}")

    # 複合技術用語
    compound_terms = [t for t in extracted_terms if 'エンジン' in t.term or 'システム' in t.term or '装置' in t.term]
    print(f"  複合技術用語: {[t.term for t in compound_terms]}")

    # JSON出力
    print("\n7. JSON出力テスト")
    output_path = "test_output_terms.json"
    extractor.export_to_json(extracted_terms, output_path)
    print(f"   JSONファイルに出力: {output_path}")

if __name__ == "__main__":
    test_advanced_extraction()