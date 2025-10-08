import re

# テストテキスト
text1 = "自動電圧調整器（AVR）により、端子電圧を一定に保ちます。"
text2 = "バッテリーマネジメントシステム（BMS）は、"
text3 = "エネルギーマネジメントシステム（EMS）が、"
text4 = "燃料消費率（SFOC: Specific Fuel Oil Consumption）は、180g/kWh以下を達成しています。"

# パターン
pattern1 = re.compile(r'[（(][A-Z]{2,5}[）)]')  # 括弧内略語
pattern2 = re.compile(r'\b[A-Z]{2,5}:\s*[A-Z]')  # コロン形式
pattern3 = re.compile(r'\b[A-Z]{2,5}\b')  # シンプルな略語

print("=== Pattern 1: Bracket-enclosed abbreviations ===")
for text in [text1, text2, text3, text4]:
    matches = pattern1.findall(text)
    print(f"Text: {text[:50]}...")
    print(f"Matches: {matches}")
    # 括弧除去
    cleaned = [m.strip('（）()') for m in matches]
    print(f"Cleaned: {cleaned}")
    print()

print("=== Pattern 2: Colon format ===")
for text in [text4]:
    matches = pattern2.findall(text)
    print(f"Text: {text[:50]}...")
    print(f"Matches: {matches}")
    # コロン除去
    cleaned = [m.split(':')[0].strip() for m in matches]
    print(f"Cleaned: {cleaned}")
    print()

print("=== Pattern 3: Simple abbreviations ===")
for text in [text1, text2, text3, text4]:
    matches = pattern3.findall(text)
    print(f"Text: {text[:50]}...")
    print(f"Matches: {matches}")
    print()
