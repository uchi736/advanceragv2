"""
高度な統計的手法による専門用語抽出
SemRe-Rank論文の手法を実装

主要な改善点:
1. TF-IDF + C-value による候補抽出
2. 2段階スコアリング（シード選定用と最終スコア用）
3. Min-max正規化（個別のみ、結合後は再正規化しない）
4. n-gramと正規表現による包括的な候補抽出
"""

import re
import math
import json
import logging
from typing import Dict, List, Optional, Tuple, Set, Any
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import numpy as np
from sudachipy import tokenizer, dictionary
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


@dataclass
class ExtractedTerm:
    """抽出された専門用語"""
    term: str
    score: float
    tfidf_score: float = 0.0
    cvalue_score: float = 0.0
    frequency: int = 0
    nested_terms: List[str] = field(default_factory=list)
    synonyms: List[str] = field(default_factory=list)  # 類義語リスト
    definition: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class AdvancedStatisticalExtractor:
    """高度な統計的手法による専門用語抽出器"""

    def __init__(
        self,
        min_term_length: int = 2,
        max_term_length: int = 6,
        min_frequency: int = 2,
        use_regex_patterns: bool = True
    ):
        """
        Args:
            min_term_length: 最小用語長（単語数）
            max_term_length: 最大用語長（単語数）
            min_frequency: 最小出現頻度
            use_regex_patterns: 正規表現パターンを使用するか
        """
        self.min_term_length = min_term_length
        self.max_term_length = max_term_length
        self.min_frequency = min_frequency
        self.use_regex_patterns = use_regex_patterns

        # Sudachiの初期化
        self.tokenizer_obj = dictionary.Dictionary().create()
        # Hybrid approach: Use both Mode A and Mode C
        # Mode A (短単位): Best for n-gram generation
        # Mode C (長単位): Captures natural compound terms
        self.sudachi_mode_a = tokenizer.Tokenizer.SplitMode.A
        self.sudachi_mode_c = tokenizer.Tokenizer.SplitMode.C

        # 専門用語パターン（正規表現）
        self.term_patterns = self._compile_term_patterns()

    def _compile_term_patterns(self) -> List[re.Pattern]:
        """専門用語を抽出するための正規表現パターン"""
        patterns = [
            # 型式番号・製品コード
            r'\b[A-Z0-9]{2,}[-_][A-Z0-9]+\b',
            r'\b[0-9]+[A-Z]{2,}[-_][0-9]+\b',

            # 化学式・化合物
            r'\b(CO2|NOx|SOx|PM2\.5|NH3|H2O|CH4|N2O)\b',

            # 専門的な略語（3文字以上の大文字）
            r'\b[A-Z]{3,}\b',

            # 数値+単位の仕様
            r'\b\d+(\.\d+)?\s*(mg|kg|kWh|MW|rpm|bar|°C|K|Pa|MPa|m³|L)/?\w*\b',

            # カタカナ+英数字の複合語
            r'[ァ-ヴー]+[A-Z0-9]+',
            r'[A-Z0-9]+[ァ-ヴー]+',

            # 複合技術用語パターン
            r'[ァ-ヴー]+(燃料|エンジン|システム|装置|機構)',
            r'(高|低|新|旧|次世代|環境)[ァ-ヴー]+',
        ]
        return [re.compile(p) for p in patterns]

    def extract_candidates(self, text: str) -> Dict[str, int]:
        """
        候補用語の抽出（ハイブリッドアプローチ）

        Args:
            text: 入力テキスト

        Returns:
            候補用語と出現頻度の辞書
        """
        candidates = defaultdict(int)

        # 1. Mode C: 自然な複合語を長単位で抽出
        mode_c_candidates = self._extract_with_mode_c(text)
        for term, freq in mode_c_candidates.items():
            candidates[term] += freq

        # 2. Mode A + n-gram: 短単位からn-gram生成
        ngram_candidates = self._extract_ngrams(text)
        for term, freq in ngram_candidates.items():
            candidates[term] += freq

        # 3. 正規表現パターンマッチング
        if self.use_regex_patterns:
            pattern_candidates = self._extract_by_patterns(text)
            for term, freq in pattern_candidates.items():
                candidates[term] += freq

        # 4. 複合名詞の抽出（Mode A使用）
        compound_candidates = self._extract_compound_nouns(text)
        for term, freq in compound_candidates.items():
            candidates[term] += freq

        # 最小頻度フィルタ
        filtered = {
            term: freq
            for term, freq in candidates.items()
            if freq >= self.min_frequency
        }

        return filtered

    def _extract_with_mode_c(self, text: str) -> Dict[str, int]:
        """
        Mode C（長単位）による複合語抽出
        自然な複合語をそのまま取得（例: "舶用ディーゼルエンジン"）
        """
        mode_c_terms = defaultdict(int)

        tokens = self.tokenizer_obj.tokenize(text, self.sudachi_mode_c)

        for token in tokens:
            term = token.surface()
            pos = token.part_of_speech()[0]

            # 名詞系のみを対象とし、複合語として成立しているもの
            if pos in ['名詞'] and len(term) >= 2:
                if self._is_valid_term(term):
                    mode_c_terms[term] += 1

        return mode_c_terms

    def _extract_ngrams(self, text: str) -> Dict[str, int]:
        """n-gram抽出（Mode A使用）"""
        ngrams = defaultdict(int)

        # Sudachiでトークン化（Mode A: 短単位）
        tokens = self.tokenizer_obj.tokenize(text, self.sudachi_mode_a)
        words = [token.surface() for token in tokens]

        # n-gram生成（2-gram から max_term_length-gram まで）
        for n in range(self.min_term_length, self.max_term_length + 1):
            for i in range(len(words) - n + 1):
                ngram = ''.join(words[i:i+n])

                # 基本フィルタ（ひらがなのみ、記号のみを除外）
                if self._is_valid_term(ngram):
                    ngrams[ngram] += 1

        return ngrams

    def _extract_by_patterns(self, text: str) -> Dict[str, int]:
        """正規表現パターンによる抽出"""
        pattern_terms = defaultdict(int)

        for pattern in self.term_patterns:
            matches = pattern.finditer(text)
            for match in matches:
                term = match.group()
                if self._is_valid_term(term):
                    pattern_terms[term] += 1

        return pattern_terms

    def _extract_compound_nouns(self, text: str) -> Dict[str, int]:
        """複合名詞の抽出（Mode A使用）"""
        compound_nouns = defaultdict(int)

        tokens = self.tokenizer_obj.tokenize(text, self.sudachi_mode_a)
        current_compound = []

        for token in tokens:
            pos = token.part_of_speech()[0]

            # 名詞または接頭辞の場合
            if pos in ['名詞', '接頭辞']:
                current_compound.append(token.surface())
            else:
                # 複合名詞が終了
                if len(current_compound) >= self.min_term_length:
                    compound = ''.join(current_compound)
                    if self._is_valid_term(compound):
                        compound_nouns[compound] += 1
                current_compound = []

        # 最後の複合名詞
        if len(current_compound) >= self.min_term_length:
            compound = ''.join(current_compound)
            if self._is_valid_term(compound):
                compound_nouns[compound] += 1

        return compound_nouns

    def _is_valid_term(self, term: str) -> bool:
        """用語の妥当性チェック"""
        # 空文字列チェック
        if not term or len(term) < 2:
            return False

        # ひらがなのみを除外
        if re.match(r'^[ぁ-ん]+$', term):
            return False

        # 記号のみを除外
        if re.match(r'^[!-/:-@\[-`{-~\s]+$', term):
            return False

        # 数字のみを除外（ただし単位付きは許可）
        if re.match(r'^\d+$', term):
            return False

        return True

    def detect_synonyms(self, candidates: List[str], full_text: str) -> Dict[str, List[str]]:
        """
        候補用語から類義語・関連語を検出

        Args:
            candidates: 候補用語リスト
            full_text: 全文テキスト

        Returns:
            用語と類義語のマッピング
        """
        synonyms = defaultdict(set)

        # テキストから名詞句を抽出（共起検出用）
        noun_phrases = self._extract_noun_phrases(full_text)

        # 1. 部分文字列関係の検出
        for i, cand1 in enumerate(candidates):
            if len(cand1) < 2:
                continue
            for cand2 in candidates[i+1:]:
                if cand1 != cand2:
                    if cand1 in cand2:
                        synonyms[cand2].add(cand1)
                    elif cand2 in cand1:
                        synonyms[cand1].add(cand2)

        # 2. 共起関係の検出
        cooccurrence_map = defaultdict(set)
        for phrase in noun_phrases:
            occurring_cands = [c for c in candidates if c in phrase]
            for cand1 in occurring_cands:
                for cand2 in occurring_cands:
                    if cand1 != cand2:
                        cooccurrence_map[cand1].add(cand2)

        for cand, related in cooccurrence_map.items():
            if len(related) >= 2:
                synonyms[cand].update(related)

        # 3. 編集距離による類似語検出
        for i, cand1 in enumerate(candidates):
            for cand2 in candidates[i+1:]:
                if len(cand1) >= 3 and len(cand2) >= 3:
                    similarity = SequenceMatcher(None, cand1, cand2).ratio()
                    if 0.7 < similarity < 0.95:
                        synonyms[cand1].add(cand2)
                        synonyms[cand2].add(cand1)

        # 4. 語幹・語尾パターン検出
        stem_groups = defaultdict(list)
        suffixes = ['化', '的', '性', '型', '式', 'ー', 'ション', 'ング', 'メント']

        for cand in candidates:
            for suffix in suffixes:
                if cand.endswith(suffix):
                    base = cand[:-len(suffix)]
                    if len(base) >= 2:
                        stem_groups[base].append(cand)
                        for other_cand in candidates:
                            if other_cand != cand and other_cand.startswith(base):
                                synonyms[cand].add(other_cand)

        for group in stem_groups.values():
            if len(group) > 1:
                for word in group:
                    synonyms[word].update(w for w in group if w != word)

        # 5. 略語パターンの検出
        abbreviation_patterns = {
            'RAG': 'Retrieval-Augmented Generation',
            'LLM': '大規模言語モデル',
            'API': 'アプリケーションプログラミングインターフェース',
            'DB': 'データベース',
            'QA': '質問応答',
        }

        for abbr, full_name in abbreviation_patterns.items():
            if abbr in candidates and full_name in candidates:
                synonyms[abbr].add(full_name)
                synonyms[full_name].add(abbr)

        # 6. ドメイン固有の関連語
        domain_relations = {
            'ベクトル検索': ['vector search', 'ベクトルサーチ', 'ベクトルDB'],
            'embedding': ['埋め込み', 'エンベディング', '埋め込み表現'],
            'リランキング': ['re-ranking', 'リランク', '再順位付け'],
        }

        for main_term, related_terms in domain_relations.items():
            if main_term in candidates:
                for related in related_terms:
                    if related in candidates:
                        synonyms[main_term].add(related)
                        synonyms[related].add(main_term)

        return {k: list(v) for k, v in synonyms.items() if v}

    def _extract_noun_phrases(self, text: str) -> List[str]:
        """名詞句を抽出（共起検出用）"""
        phrases = []

        # Mode Aで短単位トークン化
        tokens = self.tokenizer_obj.tokenize(text, self.sudachi_mode_a)
        current_phrase = []

        for token in tokens:
            if token.part_of_speech()[0] == '名詞':
                current_phrase.append(token.surface())
            else:
                if len(current_phrase) >= 2:
                    phrases.append(''.join(current_phrase))
                current_phrase = []

        if len(current_phrase) >= 2:
            phrases.append(''.join(current_phrase))

        return phrases

    def calculate_tfidf(
        self,
        documents: List[str],
        candidates: Dict[str, int]
    ) -> Dict[str, float]:
        """
        TF-IDF計算

        Args:
            documents: 文書リスト（文単位で分割）
            candidates: 候補用語と頻度

        Returns:
            TF-IDFスコアの辞書
        """
        # 候補用語のみを対象にしたTF-IDF計算
        vocabulary = list(candidates.keys())

        # TfidfVectorizerを使用
        vectorizer = TfidfVectorizer(
            vocabulary=vocabulary,
            token_pattern=r'\S+',
            lowercase=False,
            norm=None
        )

        try:
            # 文書を候補用語でトークン化
            tokenized_docs = []
            for doc in documents:
                tokens = []
                for term in vocabulary:
                    count = doc.count(term)
                    tokens.extend([term] * count)
                tokenized_docs.append(' '.join(tokens))

            # TF-IDF計算
            tfidf_matrix = vectorizer.fit_transform(tokenized_docs)

            # 各用語の最大TF-IDFスコアを取得
            tfidf_scores = {}
            feature_names = vectorizer.get_feature_names_out()

            for i, term in enumerate(feature_names):
                scores = tfidf_matrix[:, i].toarray().flatten()
                tfidf_scores[term] = float(np.max(scores))

            return tfidf_scores

        except Exception as e:
            logger.warning(f"TF-IDF calculation error: {e}")
            # フォールバック：頻度ベースのスコア
            max_freq = max(candidates.values()) if candidates else 1
            return {
                term: freq / max_freq
                for term, freq in candidates.items()
            }

    def calculate_cvalue(self, candidates: Dict[str, int]) -> Dict[str, float]:
        """
        C-value計算（ネストした用語の考慮）

        C-value = log2(|a|) * freq(a) - (1/|Ta|) * Σ freq(b)

        where:
        - |a| = 用語aの長さ（単語数）
        - freq(a) = 用語aの頻度
        - Ta = aを含むより長い用語の集合
        - b ∈ Ta
        """
        cvalues = {}

        # 用語を長さの降順でソート
        sorted_terms = sorted(candidates.keys(), key=len, reverse=True)

        # 各用語について、それを含むより長い用語を記録
        nested_info = defaultdict(list)

        for i, longer_term in enumerate(sorted_terms):
            for j, shorter_term in enumerate(sorted_terms[i+1:], i+1):
                if shorter_term in longer_term and shorter_term != longer_term:
                    nested_info[shorter_term].append(longer_term)

        # C-value計算
        for term in candidates:
            freq = candidates[term]
            term_length = len(term.split()) if ' ' in term else len([c for c in term if c.isalpha()])
            term_length = max(term_length, 1)

            # 基本C-value
            cvalue = math.log2(term_length + 1) * freq

            # ネストペナルティ
            if term in nested_info:
                nested_terms = nested_info[term]
                if nested_terms:
                    nested_freq_sum = sum(candidates.get(t, 0) for t in nested_terms)
                    cvalue -= nested_freq_sum / len(nested_terms)

            cvalues[term] = max(cvalue, 0.0)

        return cvalues

    def min_max_normalize(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Min-max正規化"""
        if not scores:
            return {}

        values = list(scores.values())
        min_val = min(values)
        max_val = max(values)

        # すべての値が同じ場合
        if max_val - min_val < 1e-10:
            return {k: 0.5 for k in scores}

        return {
            term: (score - min_val) / (max_val - min_val)
            for term, score in scores.items()
        }

    def calculate_combined_scores(
        self,
        tfidf_scores: Dict[str, float],
        cvalue_scores: Dict[str, float],
        stage: str = "final"
    ) -> Dict[str, float]:
        """
        TF-IDFとC-valueの重み付き結合

        Args:
            tfidf_scores: TF-IDFスコア
            cvalue_scores: C-valueスコア
            stage: "seed" (シード選定用) or "final" (最終スコア用)

        Returns:
            結合スコア（再正規化なし）
        """
        # 個別に正規化
        tfidf_norm = self.min_max_normalize(tfidf_scores)
        cvalue_norm = self.min_max_normalize(cvalue_scores)

        # ステージ別の重み設定
        if stage == "seed":
            # シード選定用：C-value重視
            w_tfidf = 0.3
            w_cvalue = 0.7
        else:
            # 最終スコア用：TF-IDF重視
            w_tfidf = 0.7
            w_cvalue = 0.3

        # 重み付き結合（重要：再正規化しない）
        combined = {}
        all_terms = set(tfidf_scores.keys()) | set(cvalue_scores.keys())

        for term in all_terms:
            tfidf = tfidf_norm.get(term, 0.0)
            cvalue = cvalue_norm.get(term, 0.0)
            combined[term] = w_tfidf * tfidf + w_cvalue * cvalue

        return combined

    def extract_terms(
        self,
        text: str,
        top_n: int = 100,
        return_all_scores: bool = False
    ) -> List[ExtractedTerm]:
        """
        高度な統計的手法による専門用語抽出

        Args:
            text: 入力テキスト
            top_n: 上位N件を返す
            return_all_scores: 全スコアを含めるか

        Returns:
            抽出された専門用語のリスト
        """
        # 1. 候補用語抽出
        candidates = self.extract_candidates(text)

        if not candidates:
            return []

        # 2. 文書分割（文単位）
        documents = self._split_into_sentences(text)

        # 3. TF-IDF計算
        tfidf_scores = self.calculate_tfidf(documents, candidates)

        # 4. C-value計算
        cvalue_scores = self.calculate_cvalue(candidates)

        # 5. シード選定用スコア（Stage A）
        seed_scores = self.calculate_combined_scores(
            tfidf_scores, cvalue_scores, stage="seed"
        )

        # 6. 最終スコア（Stage B）
        final_scores = self.calculate_combined_scores(
            tfidf_scores, cvalue_scores, stage="final"
        )

        # 7. ExtractedTermオブジェクト作成
        terms = []
        for term, score in final_scores.items():
            extracted_term = ExtractedTerm(
                term=term,
                score=score,
                tfidf_score=tfidf_scores.get(term, 0.0),
                cvalue_score=cvalue_scores.get(term, 0.0),
                frequency=candidates.get(term, 0),
                metadata={
                    "seed_score": seed_scores.get(term, 0.0)
                }
            )
            terms.append(extracted_term)

        # 8. スコア降順でソート
        terms.sort(key=lambda x: x.score, reverse=True)

        # 9. 上位N件を返す
        return terms[:top_n]

    def _split_into_sentences(self, text: str) -> List[str]:
        """文単位で分割"""
        # 日本語の文末記号で分割
        sentences = re.split(r'[。！？\n]+', text)
        # 空文字列を除外
        return [s.strip() for s in sentences if s.strip()]

    def export_to_json(
        self,
        terms: List[ExtractedTerm],
        output_path: str
    ) -> None:
        """JSON形式でエクスポート"""
        data = []
        for term in terms:
            data.append({
                "term": term.term,
                "score": term.score,
                "tfidf_score": term.tfidf_score,
                "cvalue_score": term.cvalue_score,
                "frequency": term.frequency,
                "metadata": term.metadata
            })

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


# 統合用のラッパークラス
class EnhancedTermExtractor:
    """既存のTermExtractorとの統合用ラッパー"""

    def __init__(
        self,
        config=None,
        llm=None,
        embeddings=None,
        vector_store=None
    ):
        """
        Args:
            config: 設定オブジェクト
            llm: LLMインスタンス
            embeddings: 埋め込みモデル
            vector_store: ベクトルストア
        """
        self.config = config
        self.llm = llm
        self.embeddings = embeddings
        self.vector_store = vector_store

        # 高度な統計抽出器
        self.statistical_extractor = AdvancedStatisticalExtractor(
            min_term_length=2,
            max_term_length=6,
            min_frequency=2,
            use_regex_patterns=True
        )

    async def extract_from_text(
        self,
        text: str,
        use_llm_filter: bool = False,
        use_definition_generation: bool = False,
        top_n: int = 100
    ) -> List[Dict[str, Any]]:
        """
        テキストから専門用語を抽出

        Args:
            text: 入力テキスト
            use_llm_filter: LLMフィルタを使用するか
            use_definition_generation: 定義生成を行うか
            top_n: 上位N件

        Returns:
            専門用語のリスト
        """
        # 1. 統計的抽出
        terms = self.statistical_extractor.extract_terms(text, top_n=top_n)

        # 2. 定義生成（オプション）
        if use_definition_generation and self.vector_store and self.llm:
            terms = await self._generate_definitions(terms[:30])  # 上位30件のみ

        # 3. LLMフィルタ（オプション）
        if use_llm_filter and self.llm:
            terms = await self._filter_by_llm(terms)

        # 辞書形式に変換
        return [
            {
                "headword": term.term,
                "score": term.score,
                "definition": term.definition,
                "frequency": term.frequency,
                "synonyms": [],
                "metadata": term.metadata
            }
            for term in terms
        ]

    async def _generate_definitions(
        self,
        terms: List[ExtractedTerm],
        top_percentile: float = 15.0
    ) -> List[ExtractedTerm]:
        """
        RAG定義生成（パーセンタイルベース）

        Args:
            terms: 抽出された専門用語のリスト
            top_percentile: 上位何%に定義を生成するか（デフォルト15%）

        Returns:
            定義が追加された専門用語のリスト
        """
        if not terms or not self.vector_store or not self.llm:
            logger.warning("Cannot generate definitions: missing terms, vector_store, or llm")
            return terms

        # 上位N%の件数を計算
        n_terms = max(1, int(len(terms) * top_percentile / 100))
        target_terms = terms[:n_terms]

        logger.info(f"Generating definitions for top {top_percentile}% ({n_terms} terms)")

        # プロンプトテンプレートを取得
        from .prompts import get_definition_generation_prompt
        from langchain_core.output_parsers import StrOutputParser

        prompt = get_definition_generation_prompt()
        chain = prompt | self.llm | StrOutputParser()

        # 各用語について定義を生成
        for i, term in enumerate(target_terms, 1):
            try:
                # ベクトル検索で関連文書を取得（k=5）
                docs = self.vector_store.similarity_search(term.term, k=5)

                if not docs:
                    logger.warning(f"No documents found for term: {term.term}")
                    continue

                # 文脈を結合（最大3000文字）
                context = "\n\n".join([doc.page_content for doc in docs])
                context = context[:3000]

                # 定義を生成
                definition = await chain.ainvoke({
                    "term": term.term,
                    "context": context
                })

                term.definition = definition.strip()
                logger.info(f"[{i}/{n_terms}] Generated definition for: {term.term}")

            except Exception as e:
                logger.error(f"Failed to generate definition for '{term.term}': {e}")
                term.definition = ""

        return terms

    async def _filter_by_llm(
        self,
        terms: List[ExtractedTerm],
        batch_size: int = 10
    ) -> List[ExtractedTerm]:
        """
        LLM専門用語判定フィルタ（バッチ処理）

        Args:
            terms: 抽出された専門用語のリスト
            batch_size: バッチサイズ（デフォルト10件）

        Returns:
            専門用語と判定された用語のリスト
        """
        if not self.llm:
            logger.warning("Cannot filter by LLM: llm not available")
            return terms

        # 定義がある用語のみ対象
        terms_with_def = [t for t in terms if t.definition]

        if not terms_with_def:
            logger.warning("No terms with definitions to filter")
            return []

        logger.info(f"Filtering {len(terms_with_def)} terms with LLM (batch_size={batch_size})")

        # プロンプトテンプレートを取得
        from .prompts import get_technical_term_judgment_prompt
        from langchain_core.output_parsers import StrOutputParser

        prompt = get_technical_term_judgment_prompt()
        chain = prompt | self.llm | StrOutputParser()

        technical_terms = []

        # バッチ処理
        for i in range(0, len(terms_with_def), batch_size):
            batch = terms_with_def[i:i+batch_size]

            # バッチ入力作成
            batch_inputs = [
                {"term": t.term, "definition": t.definition}
                for t in batch
            ]

            try:
                # バッチ実行
                result_texts = await chain.abatch(batch_inputs)

                # 結果をパース
                for term, result_text in zip(batch, result_texts):
                    result = self._parse_json(result_text)

                    if result and result.get("is_technical", False):
                        # メタデータを保存
                        term.metadata["confidence"] = result.get("confidence", 0.0)
                        term.metadata["reason"] = result.get("reason", "")
                        technical_terms.append(term)
                        logger.info(f"  [OK] {term.term}: 専門用語 (信頼度: {result.get('confidence', 0):.2f})")
                    else:
                        logger.info(f"  [NG] {term.term}: 一般用語")

            except Exception as e:
                logger.error(f"LLM filter batch processing failed: {e}")

        logger.info(f"Filtered: {len(technical_terms)} technical terms from {len(terms_with_def)} candidates")

        return technical_terms

    def _parse_json(self, text: str) -> Optional[Dict]:
        """
        JSON応答のパース（堅牢版）

        Args:
            text: LLMの応答テキスト

        Returns:
            パースされた辞書、失敗時はNone
        """
        text = text.strip()

        # マークダウンコードブロック除去
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]

        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # {}で囲まれた部分を抽出
            try:
                match = re.search(r'\{[^{}]*\}', text)
                if match:
                    return json.loads(match.group())
            except:
                pass

            logger.warning(f"Failed to parse JSON: {text[:100]}...")
            return None


# テスト用のmain関数
if __name__ == "__main__":
    # サンプルテキスト
    sample_text = """
    アンモニア燃料エンジンは、次世代の環境対応技術として注目されている。
    このアンモニア燃料エンジンは、従来のディーゼルエンジンと比較して、
    CO2排出量を大幅に削減できる。6DE-18型エンジンでは、NOx排出量も
    50mg/kWh以下に抑えられている。舶用ディーゼルエンジンの分野では、
    MARPOL規制に対応するため、脱硝装置の搭載が必須となっている。
    """

    # 抽出実行
    extractor = AdvancedStatisticalExtractor()
    terms = extractor.extract_terms(sample_text, top_n=20)

    # 結果表示
    print("専門用語抽出結果:")
    print("=" * 60)
    for i, term in enumerate(terms, 1):
        print(f"{i:2}. {term.term:30} Score: {term.score:.4f}")
        print(f"    TF-IDF: {term.tfidf_score:.4f}, C-value: {term.cvalue_score:.4f}")
        print(f"    Frequency: {term.frequency}")