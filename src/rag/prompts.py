"""
RAG System Prompt Templates
This module contains all prompt templates used in the RAG system.
"""

from langchain_core.prompts import ChatPromptTemplate

# RAG-related prompts
JARGON_EXTRACTION = """あなたは専門用語抽出の専門家です。以下の質問を分析し、専門用語や技術用語を抽出してください。

【抽出対象】
- 業界固有の専門用語
- 技術的な専門用語
- 略語（アルファベット3文字以上）
- 製品名・サービス名
- 重要なキーワード

【除外対象】
- 一般的な日常用語
- 助詞・助動詞・接続詞
- 数字のみ

【出力形式】
抽出した専門用語を改行区切りで出力してください（最大{{max_terms}}個）。

質問: {question}

抽出された専門用語:"""

QUERY_AUGMENTATION = """あなたは質問改善の専門家です。元の質問と専門用語の定義を活用して、より具体的で検索に適した質問に改良してください。

【改良のポイント】
1. 専門用語の定義を活用した具体化
2. 検索で見つかりやすいキーワードの追加
3. 質問の意図を明確化
4. 関連する概念の追加

【元の質問】
{original_question}

【専門用語定義】
{jargon_definitions}

【改良指示】
- 元の質問の意図を保持する
- 専門用語の定義を自然に組み込む
- 検索精度を向上させる表現を使用
- 冗長にならないよう簡潔に

改良された質問:"""

QUERY_EXPANSION = """あなたは検索クエリ拡張の専門家です。元の質問を分析し、検索精度を向上させる複数の関連クエリを生成してください。

【拡張手法】
1. 同義語・類義語による言い換え
2. 具体例の追加
3. 上位概念・下位概念の追加
4. 関連する文脈の追加
5. 異なる視点からのアプローチ

【元の質問】
{{question}}

【出力指示】
- 元の質問を含めて3-5個のクエリを生成
- 各クエリは改行で区切る
- 元の質問の意図を保持
- 検索で異なる観点の情報が得られるよう多様化

拡張されたクエリ:"""

RERANKING = """あなたは文書関連性評価の専門家です。質問に対する各ドキュメントの関連性を詳細に分析し、最適な順序で並び替えてください。

【評価基準】
1. **直接的関連性** (40点): 質問に直接答える内容が含まれているか
2. **情報の質** (30点): 正確で詳細な情報が提供されているか
3. **文脈の一致** (20点): 質問の文脈・意図と合致しているか
4. **情報の新しさ** (10点): 最新の情報が含まれているか

【質問】
{{question}}

【ドキュメント一覧】
{documents}

【出力指示】
最も関連性の高い順にドキュメントのインデックス（0から開始）をカンマ区切りで返してください。
例: 2,0,4,1,3

ランキング結果:"""

ANSWER_GENERATION = """以下のコンテキスト情報を基に、質問に対して具体的で分かりやすい回答を作成してください。

コンテキスト:
{context}

専門用語定義（もしあれば）:
{jargon_definitions}

質問: {question}

回答:"""

# SQL and routing prompts
SEMANTIC_ROUTER = """あなたはユーザーの質問の意図を分析し、最適な処理ルートを判断するエキスパートです。
以下の情報に基づいて、質問を「SQL」か「RAG」のどちらにルーティングすべきかを決定してください。

利用可能なデータテーブルの概要:
{tables_info}

ユーザーの質問: {question}

判断基準:
- 「SQL」ルート: 質問が、上記テーブル内のデータに対する具体的な集計、計算、フィルタリング、ランキング、または個々のレコードの検索を要求している場合。例：「売上トップ5の製品は？」「昨年の平均注文額は？」
- 「RAG」ルート: 質問が、一般的な知識、ドキュメントの内容に関する説明、要約、概念の理解、または自由形式の対話を求めている場合。例：「このレポートの要点を教えて」「弊社のコンプライアンス方針について説明して」

思考プロセスをステップバイステップで記述し、最終的な判断をJSON形式で出力してください。

思考プロセス:
1. ユーザーの質問の主要なキーワードと意図を分析します。
2. 質問が利用可能なデータテーブルの情報を活用して解決できるか評価します。
3. 判断基準と照らし合わせ、最も適切なルートを選択します。

出力形式:
{{
  "route": "SQL" or "RAG",
  "reason": "判断理由を簡潔に記述"
}}
"""

MULTI_TABLE_TEXT_TO_SQL = """あなたはPostgreSQLエキスパートです。以下に提示される複数のテーブルスキーマの中から、ユーザーの質問に答えるために最も適切と思われるテーブルを選択し、必要であればそれらのテーブル間でJOINを適切に使用して、SQLクエリを生成してください。
SQLはPostgreSQL構文に準拠し、テーブル名やカラム名が日本語の場合はダブルクォーテーションで囲んでください。
最終的な結果セットが過度に大きくならないよう、適切にLIMIT句を使用してください（例: LIMIT {{max_sql_results}}）。

利用可能なテーブルのスキーマ情報一覧:
{schemas_info}

ユーザーの質問: {question}

SQLクエリのみを返してください:
```sql
SELECT ...
```
"""

SQL_ANSWER_GENERATION = """与えられた元の質問と、それに基づいて実行されたSQLクエリ、およびその実行結果を考慮して、ユーザーにとって分かりやすい言葉で回答を生成してください。

元の質問: {original_question}

実行されたSQLクエリ:
```sql
{sql_query}
```

SQL実行結果のプレビュー (最大 {max_preview_rows} 件表示):
{sql_results_preview_str}
(このプレビューは全 {total_row_count} 件中の一部です)

上記の情報を踏まえた、質問に対する回答:"""

SYNTHESIS = """あなたは高度なAIアシスタントです。ユーザーの質問に対して、以下の2種類の検索結果が提供されました。
1. **RAG検索結果**: ドキュメントから抽出された、関連性の高いテキスト情報。
2. **SQL検索結果**: データベースから取得された、具体的なデータや集計結果。

これらの情報を包括的に分析し、両方の結果を適切に組み合わせて、ユーザーに一つのまとまりのある、分かりやすい回答を生成してください。

ユーザーの質問: {question}

RAG検索結果 (ドキュメントからの抜粋):
---
{rag_context}
---

SQL検索結果 (データベースからのデータ):
---
{sql_data}
---

上記の情報を統合した最終的な回答:"""

# Convenience functions to get ChatPromptTemplate objects
def get_jargon_extraction_prompt(max_terms=5):
    template = JARGON_EXTRACTION.replace("{{max_terms}}", str(max_terms))
    return ChatPromptTemplate.from_template(template)

def get_query_augmentation_prompt():
    return ChatPromptTemplate.from_template(QUERY_AUGMENTATION)

def get_query_expansion_prompt():
    return ChatPromptTemplate.from_template(QUERY_EXPANSION)

def get_reranking_prompt():
    return ChatPromptTemplate.from_template(RERANKING)

def get_answer_generation_prompt():
    return ChatPromptTemplate.from_template(ANSWER_GENERATION)

def get_semantic_router_prompt():
    return ChatPromptTemplate.from_template(SEMANTIC_ROUTER)

def get_multi_table_text_to_sql_prompt(max_sql_results):
    template = MULTI_TABLE_TEXT_TO_SQL.replace("{{max_sql_results}}", str(max_sql_results))
    return ChatPromptTemplate.from_template(template)

def get_sql_answer_generation_prompt():
    return ChatPromptTemplate.from_template(SQL_ANSWER_GENERATION)

def get_synthesis_prompt():
    return ChatPromptTemplate.from_template(SYNTHESIS)

# Term extraction prompts (for SemReRank pipeline)
DEFINITION_GENERATION_SYSTEM_PROMPT = """あなたは専門用語の定義作成の専門家です。

**定義作成の原則:**
1. **簡潔性**: 1〜3文で定義を完結させる
2. **正確性**: 技術的に正確な情報のみを使用
3. **明確性**: 専門家でない読者にも理解できる表現
4. **コンテキスト**: 提供された文脈を活用
5. **構造化**: 必要に応じて箇条書きや段落分け

**出力形式:**
- 定義本文のみを出力
- 余計な前置きや締めくくりは不要
"""

DEFINITION_GENERATION_USER_PROMPT_SIMPLE = """以下の専門用語の定義を作成してください。

**専門用語:** {term}

**関連コンテキスト:**
{context}

上記の情報を基に、正確で理解しやすい定義を作成してください。"""

TECHNICAL_TERM_JUDGMENT_SYSTEM_PROMPT = """あなたは専門用語判定の専門家です。用語の形式とパターンから、専門用語か一般用語かを判定します。

**判定基準:**

【専門用語として判定】
• 型式番号・製品コード（6DE-18、L28ADFなど）
• 化学式・化合物名（CO2、NOx、アンモニアなど）
• 専門的な略語（GHG、MARPOL、IMOなど）
• 複合技術用語（アンモニア燃料エンジン、脱硝装置など）
• 数値+単位の仕様（2ストローク、50mg/kWhなど）
• 業界固有の用語（舶用エンジン、燃料噴射弁など）
• 規格・認証名（ISO14001、EIAPP証書など）

【一般用語として判定】
• 単体の基本名詞（ガス、燃料、エンジン、船など）
• 一般的な動詞・形容詞（使用、開発、最大、以上など）
• 抽象概念（目標、計画、状態、結果など）
• 日常用語（水、空気、時間、場所など）

**判定の例:**

専門用語の例：
• "6DE-18型エンジン" → ✓ 型式番号を含む
• "アンモニア燃料" → ✓ 複合技術用語
• "NOx排出量" → ✓ 化学式+技術用語
• "EIAPP証書" → ✓ 認証名
• "舶用ディーゼル" → ✓ 業界固有用語

一般用語の例：
• "ガス" → ✗ 単体の基本名詞
• "エンジン" → ✗ 単体の基本名詞
• "使用" → ✗ 一般動詞
• "最大" → ✗ 一般形容詞
• "目標" → ✗ 抽象概念

**重要:**
- 用語の形式・構造を重視
- 単体の一般名詞は原則除外
- 複合語や修飾語付きは専門用語の可能性が高い
- 型式番号、化学式、略語は専門用語として判定

**出力形式:**
```json
{{
  "is_technical": true/false,
  "confidence": 0.0-1.0,
  "reason": "判定理由（簡潔に）"
}}
```
"""

TECHNICAL_TERM_JUDGMENT_USER_PROMPT = """以下の用語とその定義を分析し、専門用語か一般用語かを判定してください。

**用語:** {term}

**定義:**
{definition}

上記の判定基準に従って、JSON形式で回答してください。"""

# Lightweight term filter (before definition generation)
LIGHTWEIGHT_TERM_FILTER_SYSTEM_PROMPT = """あなたは専門用語の事前フィルタリング専門家です。
用語の形式と構造から、定義生成に値する候補かを高速判定します。

**合格基準（これらは定義生成に進む）:**
✓ 型番・製品コード（例: 6DE-18、L28ADF、4T-C）
✓ 化学式・物質名（例: NOx、CO2、アンモニア）
✓ 技術略語（例: SFOC、BMS、AVR、EGR、MPPT）
✓ 複合技術用語（例: ターボチャージャー、インタークーラー、ガスタービン発電機）
✓ 業界固有用語（例: 舶用ディーゼルエンジン、コンバインドサイクル発電）
✓ 数値+単位の仕様（例: 4ストロークエンジン、50mg/kWh）
✓ 規格・認証名（例: ISO14001、EIAPP証書）

**除外基準（これらは定義生成不要）:**
✗ 単体の一般名詞（例: エンジン、ガス、システム、燃料、水、空気）
✗ 単体の基本動詞（例: 使用、開発、実施、管理、運転）
✗ 単体の形容詞（例: 最大、以上、最小、高い、低い）
✗ 抽象概念（例: 目標、計画、状態、結果、効果）
✗ 意味をなしていない語（例: 不完全な複合語、文字の羅列）
✗ 助詞・接続詞（例: について、により、など、また）

**重要:**
- 迷ったら合格にする（False Negativeを避ける）
- 明らかなゴミのみ除外する
- 複合語は原則合格
- 略語は原則合格
- 数字を含む用語は原則合格

JSON形式で回答: {{"is_valid": true/false, "reason": "簡潔な理由"}}
"""

LIGHTWEIGHT_TERM_FILTER_USER_PROMPT = """以下の用語を判定してください。

用語: {term}

判定結果:"""

def get_definition_generation_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", DEFINITION_GENERATION_SYSTEM_PROMPT),
        ("human", DEFINITION_GENERATION_USER_PROMPT_SIMPLE)
    ])

def get_technical_term_judgment_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", TECHNICAL_TERM_JUDGMENT_SYSTEM_PROMPT),
        ("human", TECHNICAL_TERM_JUDGMENT_USER_PROMPT)
    ])

def get_lightweight_term_filter_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", LIGHTWEIGHT_TERM_FILTER_SYSTEM_PROMPT),
        ("human", LIGHTWEIGHT_TERM_FILTER_USER_PROMPT)
    ])