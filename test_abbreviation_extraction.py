"""略語抽出のデバッグテスト"""
from src.rag.advanced_term_extraction import AdvancedStatisticalExtractor
from src.rag.config import Config
import asyncio

async def test():
    extractor = AdvancedStatisticalExtractor(
        min_term_length=2,
        max_term_length=6,
        min_frequency=1,  # 低頻度でも抽出
        use_regex_patterns=True
    )

    # テストテキスト（実際のドキュメントから）
    text = """
    発電設備の運用管理

    自動電圧調整器（AVR）により、端子電圧を一定に保ちます。

    再生可能エネルギーシステム

    バッテリーマネジメントシステム（BMS）は、
    各セルの電圧・温度を監視し、充放電を最適制御します。

    エネルギーマネジメントシステム（EMS）が、各設備の運転を最適化し、
    ピークカットとピークシフトを実現します。

    舶用ディーゼルエンジン技術資料

    燃料消費率（SFOC: Specific Fuel Oil Consumption）は、180g/kWh以下を達成しています。
    """

    print("=== Testing candidate extraction ===")
    candidates = extractor.extract_candidates(text)

    print(f"\nTotal candidates: {len(candidates)}")

    # 略語をフィルタ
    import re
    abbr_pattern = re.compile(r'^[A-Z]{2,5}[0-9x]?$')
    abbreviations = {term: freq for term, freq in candidates.items() if abbr_pattern.match(term)}

    print(f"\nAbbreviations found: {len(abbreviations)}")
    for term, freq in sorted(abbreviations.items()):
        print(f"  {term}: {freq}")

    # 目標略語の存在確認
    target_abbreviations = ["BMS", "AVR", "EMS", "SFOC"]
    print(f"\nTarget abbreviations:")
    for abbr in target_abbreviations:
        status = "✅ FOUND" if abbr in candidates else "❌ MISSING"
        print(f"  {abbr}: {status}")

if __name__ == "__main__":
    asyncio.run(test())
