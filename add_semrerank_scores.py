"""
extracted_terms.jsonにSemReRankスコアフィールドを追加するスクリプト
"""
import json

# ファイル読み込み
with open('test_data/extracted_terms.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 各用語にSemReRankスコアを追加
for term in data['terms']:
    tfidf = term.get('tfidf_score', 0)
    cvalue = term.get('cvalue_score', 0)
    final_score = term.get('score', 0)

    # base_score = tfidf × cvalue
    base_score = tfidf * cvalue

    # revised_score = 既存のscoreフィールド
    revised_score = final_score

    # importance_score = (revised / base) - 1
    if base_score > 0:
        importance_score = (revised_score / base_score) - 1
    else:
        importance_score = 0.0

    # 新しいフィールドを追加
    term['base_score'] = base_score
    term['revised_score'] = revised_score
    term['importance_score'] = importance_score

# ファイル書き込み
with open('test_data/extracted_terms.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Complete: Added SemReRank scores to {len(data['terms'])} terms")
