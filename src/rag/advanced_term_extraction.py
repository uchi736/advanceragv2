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

from sudachipy import tokenizer, dictionary

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

    # 用語関係の3カテゴリ
    synonyms: List[str] = field(default_factory=list)       # 類義語（LLM確定済み）
    variants: List[str] = field(default_factory=list)       # 表記ゆれ
    related_terms: List[str] = field(default_factory=list)  # 関連語（上位/下位語・共起語）

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

        # Sudachi制限: 49149バイト
        self.max_tokenize_bytes = 49000  # 安全マージン

    def _safe_tokenize(self, text: str, mode):
        """大きなテキストを安全にトークン化（Sudachi制限回避）"""
        text_bytes = text.encode('utf-8')

        # 制限以下ならそのまま処理
        if len(text_bytes) <= self.max_tokenize_bytes:
            return self.tokenizer_obj.tokenize(text, mode)

        # 制限を超える場合は文単位で分割処理
        tokens = []

        # 文単位で分割（句点・改行・感嘆符・疑問符）
        sentences = re.split(r'([。！？\n]+)', text)

        current_chunk = []
        current_bytes = 0

        for sentence in sentences:
            sentence_bytes = len(sentence.encode('utf-8'))

            # 1文が制限を超える場合は文字数で強制分割
            if sentence_bytes > self.max_tokenize_bytes:
                # 安全な文字数を計算（バイト数ベース）
                safe_char_count = len(sentence) * self.max_tokenize_bytes // sentence_bytes
                # 余裕を持たせる（90%）
                safe_char_count = int(safe_char_count * 0.9)

                for i in range(0, len(sentence), safe_char_count):
                    chunk = sentence[i:i+safe_char_count]
                    if chunk.strip():
                        try:
                            tokens.extend(self.tokenizer_obj.tokenize(chunk, mode))
                        except Exception as e:
                            logger.warning(f"Failed to tokenize chunk: {e}")
                continue

            # チャンク蓄積
            if current_bytes + sentence_bytes > self.max_tokenize_bytes:
                # 現在のチャンクをトークン化
                if current_chunk:
                    chunk_text = ''.join(current_chunk)
                    try:
                        tokens.extend(self.tokenizer_obj.tokenize(chunk_text, mode))
                    except Exception as e:
                        logger.warning(f"Failed to tokenize chunk: {e}")

                # 新しいチャンク開始
                current_chunk = [sentence]
                current_bytes = sentence_bytes
            else:
                current_chunk.append(sentence)
                current_bytes += sentence_bytes

        # 残りのチャンクを処理
        if current_chunk:
            chunk_text = ''.join(current_chunk)
            try:
                tokens.extend(self.tokenizer_obj.tokenize(chunk_text, mode))
            except Exception as e:
                logger.warning(f"Failed to tokenize final chunk: {e}")

        return tokens

    def _compile_term_patterns(self) -> List[re.Pattern]:
        """専門用語を抽出するための正規表現パターン"""
        patterns = [
            # 括弧内の略語（最優先で抽出）
            r'[（(][A-Z]{2,5}[）)]',  # （BMS）、(AVR)、（EMS）形式

            # 型式番号・製品コード（6DE-18、L28ADF、4T-C、12V170など）
            r'\b[0-9]+[A-Z]+[-_][0-9]+[A-Z]*\b',  # 6DE-18, 12V-170
            r'\b[A-Z]+[0-9]+[A-Z]+[-_]?[0-9]*\b',  # L28ADF, 4T-C
            r'\b[0-9]+[A-Z]{2,}[-_]?[0-9]*\b',     # 6DE, 12V170

            # 化学式・化合物
            r'\b(CO2|NOx|SOx|PM2\.5|NH3|H2O|CH4|N2O)\b',

            # 略語パターン（SFOC、MPPT制御など）
            r'\b[A-Z]{2,5}:\s*[A-Z]',  # SFOC: Specific形式
            r'\b[A-Z]{2,5}\b',  # シンプルな略語（2-5文字の大文字）BMS, AVR, EMS, SFOC対応
            r'\b[A-Z]{2,5}(制御|装置|システム|方式|機能|技術|モード)\b',  # 略語+用語の複合語（MPPT制御など）

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

        # 1. 正規表現パターンマッチング（最優先：括弧付き略語、型式番号など）
        if self.use_regex_patterns:
            pattern_candidates = self._extract_by_patterns(text)
            for term, freq in pattern_candidates.items():
                candidates[term] += freq

        # 2. Mode C: 自然な複合語を長単位で抽出
        mode_c_candidates = self._extract_with_mode_c(text)
        for term, freq in mode_c_candidates.items():
            candidates[term] += freq

        # 3. Mode A + n-gram: 短単位からn-gram生成
        ngram_candidates = self._extract_ngrams(text)
        for term, freq in ngram_candidates.items():
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

        tokens = self._safe_tokenize(text, self.sudachi_mode_c)

        for token in tokens:
            term = token.surface()
            pos = token.part_of_speech()[0]

            # 名詞系のみを対象とし、複合語として成立しているもの
            if pos in ['名詞'] and len(term) >= 2:
                if self._is_valid_term(term):
                    mode_c_terms[term] += 1

        return mode_c_terms

    def _extract_ngrams(self, text: str) -> Dict[str, int]:
        """n-gram抽出（品詞ベース：名詞連続のみ）"""
        ngrams = defaultdict(int)

        # Sudachiでトークン化（Mode A: 短単位）
        tokens = self._safe_tokenize(text, self.sudachi_mode_a)

        # 名詞・接頭辞の連続を抽出
        noun_sequences = []
        current_sequence = []

        for token in tokens:
            pos = token.part_of_speech()[0]

            # 名詞または接頭辞の場合は連続に追加
            if pos in ['名詞', '接頭辞']:
                current_sequence.append(token.surface())
            else:
                # 連続が終了したら保存
                if len(current_sequence) >= self.min_term_length:
                    noun_sequences.append(current_sequence)
                current_sequence = []

        # 最後の連続も保存
        if len(current_sequence) >= self.min_term_length:
            noun_sequences.append(current_sequence)

        # 名詞連続からn-gramを生成
        for sequence in noun_sequences:
            for n in range(self.min_term_length, min(self.max_term_length + 1, len(sequence) + 1)):
                for i in range(len(sequence) - n + 1):
                    ngram = ''.join(sequence[i:i+n])

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

                # 括弧を除去（（BMS） → BMS）
                term = term.strip('（）()')

                # コロン形式から略語部分のみ抽出（SFOC: Specific → SFOC）
                if ':' in term:
                    term = term.split(':')[0].strip()

                if self._is_valid_term(term):
                    pattern_terms[term] += 1

        return pattern_terms

    def _extract_compound_nouns(self, text: str) -> Dict[str, int]:
        """複合名詞の抽出（Mode A使用）"""
        compound_nouns = defaultdict(int)

        tokens = self._safe_tokenize(text, self.sudachi_mode_a)
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
        """用語の妥当性チェック（文字列 + 品詞ベース）"""
        # 空文字列チェック
        if not term or len(term) < 2:
            return False

        # HTMLタグ除外（小文字のタグ名）
        html_tags = {
            'div', 'span', 'img', 'svg', 'td', 'tr', 'th', 'tbody', 'thead', 'table',
            'ul', 'ol', 'li', 'br', 'hr', 'pre', 'code', 'html', 'body', 'head',
            'meta', 'link', 'script', 'style', 'form', 'input', 'button', 'label',
            'select', 'option', 'iframe', 'nav', 'header', 'footer', 'section',
            'article', 'aside', 'main', 'figure', 'figcaption', 'video', 'audio',
            'source', 'canvas', 'embed', 'object', 'param', 'a', 'p', 'h1', 'h2',
            'h3', 'h4', 'h5', 'h6', 'strong', 'em', 'b', 'i', 'u', 's', 'small'
        }
        if term.lower() in html_tags:
            return False

        # ひらがなのみを除外
        if re.match(r'^[ぁ-ん]+$', term):
            return False

        # 記号のみを除外
        if re.match(r'^[!-/:-@\[-`{-~\s]+$', term):
            return False

        # 記号で始まる/終わる用語を除外（例：「（％」「，アンモニア」）
        if re.match(r'^[!-/:-@\[-`{-~、。，．・「」『』（）\s]', term):
            return False
        if re.match(r'[!-/:-@\[-`{-~、。，．・「」『』（）\s]$', term):
            return False

        # 数字のみを除外（ただし単位付きは許可）
        if re.match(r'^\d+$', term):
            return False

        # 超汎用的な単語のみ除外（最小限）
        generic_terms = {'エリア', 'モード'}
        if term in generic_terms:
            return False

        # 英字略語は品詞チェックをスキップして自動承認（BMS、AVR、EMS、SFOC、NOx、CO2など）
        abbreviation_pattern = re.compile(r'^[A-Z]{2,5}[0-9x]?$')
        if abbreviation_pattern.match(term):
            return True

        # 品詞チェック：名詞・接頭辞・接尾辞・記号（一部）のみ許可
        try:
            tokens = self._safe_tokenize(term, self.sudachi_mode_a)
            for token in tokens:
                pos = token.part_of_speech()[0]
                pos_sub = token.part_of_speech()[1] if len(token.part_of_speech()) > 1 else ''

                # 許可する品詞
                allowed_pos = ['名詞', '接頭辞', '接尾辞']
                # 許可する記号（単位など）
                allowed_symbols = ['％', '°', '/', '-', '・']

                if pos not in allowed_pos:
                    # 記号の場合は許可リストをチェック
                    if pos == '補助記号' and token.surface() in allowed_symbols:
                        continue
                    # それ以外（助詞、動詞、形容詞など）は除外
                    return False

        except Exception:
            # 形態素解析エラーの場合は基本チェックのみで通す
            pass

        return True

    def detect_variants(self, candidates: List[str]) -> Dict[str, List[str]]:
        """
        表記ゆれ検出（編集距離ベース）

        Args:
            candidates: 候補用語リスト

        Returns:
            用語と表記ゆれのマッピング
        """
        variants = defaultdict(set)

        for i, cand1 in enumerate(candidates):
            for cand2 in candidates[i+1:]:
                if len(cand1) >= 3 and len(cand2) >= 3:
                    # Levenshtein比率で類似度計算
                    similarity = SequenceMatcher(None, cand1, cand2).ratio()
                    # 閾値: 0.75-0.98（完全一致1.0は除外）
                    if 0.75 < similarity < 0.98:
                        variants[cand1].add(cand2)
                        variants[cand2].add(cand1)

        return {k: list(v) for k, v in variants.items() if v}

    def detect_related_terms(
        self,
        candidates: List[str],
        full_text: str
    ) -> Dict[str, List[str]]:
        """
        関連語検出（包含関係・PMI共起）

        Args:
            candidates: 候補用語リスト
            full_text: 全文テキスト

        Returns:
            用語と関連語のマッピング
        """
        related = defaultdict(set)

        # 1. 包含関係（上位語/下位語）
        for i, cand1 in enumerate(candidates):
            if len(cand1) < 2:
                continue
            for cand2 in candidates[i+1:]:
                if cand1 != cand2:
                    # 完全包含（短い方が長い方に含まれる）
                    if cand1 in cand2 and len(cand2) > len(cand1) + 1:
                        related[cand1].add(cand2)
                        related[cand2].add(cand1)
                    elif cand2 in cand1 and len(cand1) > len(cand2) + 1:
                        related[cand2].add(cand1)
                        related[cand1].add(cand2)

        # 2. PMI共起分析（改善版：完全一致のみ）
        cooccurrence_map = defaultdict(lambda: defaultdict(int))
        window_size = 10

        # Sudachiで形態素解析
        tokens = self._safe_tokenize(full_text, self.sudachi_mode_a)
        words = [token.surface() for token in tokens]

        # 候補用語の頻度カウント
        word_freq = defaultdict(int)
        for word in words:
            if word in candidates:
                word_freq[word] += 1

        # 共起カウント（完全一致のみ）
        for i, word in enumerate(words):
            if word in candidates:
                window_start = max(0, i - window_size)
                window_end = min(len(words), i + window_size + 1)

                for j in range(window_start, window_end):
                    if i != j and words[j] in candidates:
                        cooccurrence_map[word][words[j]] += 1

        # PMI（相互情報量）で共起の強さを評価
        total_words = len(words)
        cooccurrence_threshold = 3  # 最低3回共起
        pmi_threshold = 2.0  # PMIスコア閾値

        for cand1, related_counts in cooccurrence_map.items():
            for cand2, cooccur_count in related_counts.items():
                if cooccur_count >= cooccurrence_threshold:
                    # PMI計算: log2(P(x,y) / (P(x) * P(y)))
                    p_xy = cooccur_count / total_words
                    p_x = word_freq[cand1] / total_words
                    p_y = word_freq[cand2] / total_words

                    if p_x > 0 and p_y > 0:
                        pmi = math.log2(p_xy / (p_x * p_y))

                        if pmi >= pmi_threshold:
                            related[cand1].add(cand2)

        return {k: list(v) for k, v in related.items() if v}

    def _extract_noun_phrases(self, text: str) -> List[str]:
        """名詞句を抽出（共起検出用）"""
        phrases = []

        # Mode Aで短単位トークン化
        tokens = self._safe_tokenize(text, self.sudachi_mode_a)
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
        TF-IDF計算（形態素ベース・完全一致）

        Args:
            documents: 文書リスト（文単位で分割）
            candidates: 候補用語と頻度

        Returns:
            TF-IDFスコアの辞書
        """
        vocabulary = list(candidates.keys())
        N = len(documents)  # 文書数

        # 事前に全用語をトークン化（効率化）
        term_token_map = {}
        for term in vocabulary:
            try:
                term_tokens = tuple([t.surface() for t in self._safe_tokenize(term, self.sudachi_mode_a)])
                term_token_map[term] = term_tokens
            except:
                term_token_map[term] = (term,)  # フォールバック

        # 1. 各用語の文書頻度（DF）を計算
        df = defaultdict(int)
        term_freq_per_doc = []  # 各文書での用語頻度

        for doc in documents:
            # 形態素解析してトークン化（Mode A）
            tokens = self._safe_tokenize(doc, self.sudachi_mode_a)
            token_surfaces = tuple([t.surface() for t in tokens])

            # n-gramで用語マッチング（完全一致のみ）
            doc_term_count = defaultdict(int)

            for term, term_tokens in term_token_map.items():
                term_len = len(term_tokens)

                # トークン列を走査して用語を検索
                for i in range(len(token_surfaces) - term_len + 1):
                    if token_surfaces[i:i+term_len] == term_tokens:
                        doc_term_count[term] += 1

            # DF計算（この文書に出現するか）
            for term in doc_term_count:
                df[term] += 1

            term_freq_per_doc.append(doc_term_count)

        # 2. TF-IDF計算
        tfidf_scores = {}
        for term in vocabulary:
            # 各文書でのTF-IDF合計
            total_tfidf = 0.0
            for doc_tf in term_freq_per_doc:
                tf = doc_tf.get(term, 0)
                if tf > 0 and df.get(term, 0) > 0:
                    # TF-IDF = TF * log(N / DF)
                    idf = math.log(N / df[term])
                    total_tfidf += tf * idf

            tfidf_scores[term] = total_tfidf

        return tfidf_scores

    def calculate_cvalue(self, candidates: Dict[str, int]) -> Dict[str, float]:
        """
        C-value計算（ネストした用語の考慮）- 形態素数ベース

        C-value = log2(|a|) * freq(a) - (1/|Ta|) * Σ freq(b)

        where:
        - |a| = 用語aの長さ（形態素数）
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

            # 形態素数を正確に計算
            try:
                term_tokens = self._safe_tokenize(term, self.sudachi_mode_a)
                term_length = len(term_tokens)
            except:
                # フォールバック：文字数
                term_length = len(term)

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
        TF-IDFとC-valueの重み付き結合（複合語優先）

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
            # シード選定用：C-value重視（複合語優先）
            w_tfidf = 0.3
            w_cvalue = 0.7
        else:
            # 最終スコア用：バランス型（C-value少し優先で複合語促進）
            w_tfidf = 0.4
            w_cvalue = 0.6

        # 重み付き結合
        combined = {}
        all_terms = set(tfidf_scores.keys()) | set(cvalue_scores.keys())

        for term in all_terms:
            tfidf = tfidf_norm.get(term, 0.0)
            cvalue = cvalue_norm.get(term, 0.0)

            # 基本スコア
            base_score = w_tfidf * tfidf + w_cvalue * cvalue

            # 複合語ボーナス（形態素数に応じて）
            try:
                term_tokens = self._safe_tokenize(term, self.sudachi_mode_a)
                morpheme_count = len(term_tokens)

                # 2形態素以上でボーナス（最大1.5倍）
                if morpheme_count >= 2:
                    compound_bonus = min(1.0 + (morpheme_count - 1) * 0.15, 1.5)
                    base_score *= compound_bonus
            except:
                pass  # トークン化失敗時はボーナスなし

            combined[term] = base_score

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

    async def detect_semantic_synonyms_pgvector(
        self,
        candidates: List[str],
        similarity_threshold: float = 0.85
    ) -> Dict[str, List[str]]:
        """
        PGVectorで意味的類義語を検出（案2: 一括ベクトル比較）

        Args:
            candidates: 候補用語リスト
            similarity_threshold: コサイン類似度の閾値

        Returns:
            用語と類義語候補のマッピング
        """
        if not self.embeddings or len(candidates) < 2:
            logger.warning("Cannot detect synonyms: missing embeddings or insufficient candidates")
            return {}

        if not self.config or not hasattr(self.config, 'pgvector_connection_string'):
            logger.warning("Cannot detect synonyms: missing PGVector connection string")
            return {}

        try:
            from sqlalchemy import create_engine, text
            from sqlalchemy.pool import NullPool

            logger.info(f"Generating embeddings for {len(candidates)} candidates...")

            # 1. embeddings一括取得
            vectors = await self.embeddings.aembed_documents(candidates)

            # 2. PostgreSQL接続
            engine = create_engine(
                self.config.pgvector_connection_string,
                poolclass=NullPool
            )

            with engine.begin() as conn:
                # 3. 既存の候補用語を削除（重複回避）
                conn.execute(text("""
                    DELETE FROM knowledge_nodes
                    WHERE node_type = 'Term'
                      AND term = ANY(:terms)
                """), {'terms': candidates})

                logger.info(f"Inserting {len(candidates)} terms into knowledge_nodes...")

                # 4. 一括挿入
                for term, vec in zip(candidates, vectors):
                    vec_str = '[' + ','.join(map(str, vec)) + ']'
                    # SQLAlchemyの名前付きパラメータ形式に統一
                    conn.execute(text("""
                        INSERT INTO knowledge_nodes (node_type, term, embedding)
                        VALUES (:node_type, :term, cast(:embedding as vector))
                    """), {
                        'node_type': 'Term',
                        'term': term,
                        'embedding': vec_str
                    })

                # 5. PGVectorで全ペアの類似度を一括計算
                logger.info(f"Computing similarities with threshold={similarity_threshold}...")

                result = conn.execute(text("""
                    WITH term_vectors AS (
                        SELECT term, embedding
                        FROM knowledge_nodes
                        WHERE node_type = 'Term'
                          AND term = ANY(:terms)
                    )
                    SELECT
                        t1.term as term1,
                        t2.term as term2,
                        1 - (t1.embedding <=> t2.embedding) as similarity
                    FROM term_vectors t1
                    CROSS JOIN term_vectors t2
                    WHERE t1.term < t2.term
                      AND 1 - (t1.embedding <=> t2.embedding) >= :threshold
                    ORDER BY similarity DESC
                """), {
                    'terms': candidates,
                    'threshold': similarity_threshold
                })

                # 6. 対称的な辞書を構築
                synonyms = defaultdict(set)
                for row in result:
                    synonyms[row.term1].add(row.term2)
                    synonyms[row.term2].add(row.term1)

                logger.info(f"Found {len(synonyms)} terms with semantic similarity candidates")

                return {k: list(v) for k, v in synonyms.items()}

        except Exception as e:
            logger.error(f"Failed to detect semantic synonyms with PGVector: {e}")
            return {}

    async def validate_synonyms_with_llm(
        self,
        synonym_candidates: Dict[str, List[str]],
        batch_size: int = 10
    ) -> Dict[str, List[str]]:
        """
        LLMで類義語候補を最終判定（必須）

        Args:
            synonym_candidates: 類義語候補のマッピング
            batch_size: バッチサイズ

        Returns:
            LLMで確定された類義語のマッピング
        """
        if not self.llm:
            logger.error("LLM is required for synonym validation")
            return {}

        if not synonym_candidates:
            logger.info("No synonym candidates to validate")
            return {}

        # 候補ペアをフラット化（重複排除）
        pairs = []
        seen = set()
        for term1, candidates in synonym_candidates.items():
            for term2 in candidates:
                # 辞書順でペアを作成（重複排除）
                pair = tuple(sorted([term1, term2]))
                if pair not in seen:
                    seen.add(pair)
                    pairs.append(pair)

        logger.info(f"Validating {len(pairs)} synonym pairs with LLM (batch_size={batch_size})...")

        confirmed_synonyms = defaultdict(set)

        # バッチ処理
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]

            try:
                # LLMプロンプト作成
                prompt = self._create_synonym_validation_prompt(batch)

                # LLM実行
                result_text = await self.llm.ainvoke(prompt)

                # 応答をパース
                result_text = result_text.content if hasattr(result_text, 'content') else str(result_text)
                results = self._parse_synonym_validation_result(result_text, len(batch))

                # 結果を反映
                for (term1, term2), is_synonym in zip(batch, results):
                    if is_synonym:
                        confirmed_synonyms[term1].add(term2)
                        confirmed_synonyms[term2].add(term1)
                        logger.info(f"  [OK] 「{term1}」↔「{term2}」: 類義語")
                    else:
                        logger.info(f"  [NG] 「{term1}」↔「{term2}」: 非類義語")

            except Exception as e:
                logger.error(f"LLM validation batch processing failed: {e}")

        logger.info(f"Confirmed {len(confirmed_synonyms)} terms with synonyms")

        return {k: list(v) for k, v in confirmed_synonyms.items()}

    def _create_synonym_validation_prompt(
        self,
        pairs: List[Tuple[str, str]]
    ) -> str:
        """LLM判定用プロンプト"""
        pairs_text = '\n'.join([
            f"{i+1}. 「{t1}」と「{t2}」"
            for i, (t1, t2) in enumerate(pairs)
        ])

        return f"""以下の用語ペアが類義語（同じ意味を持つ語）かどうか判定してください。

【用語ペア】
{pairs_text}

【判定基準】
- 類義語: ほぼ同じ意味を持つ（例: 「データベース」と「DB」、「最適化」と「optimization」）
- 非類義語: 関連はあるが意味が異なる（例: 「エンジン」と「ディーゼルエンジン」は上位語/下位語なので非類義語）

【回答形式】
各ペアについて、類義語なら1、非類義語なら0を返してください。
形式: [1, 0, 1, ...]（カンマ区切りの数値リスト）

回答:"""

    def _parse_synonym_validation_result(
        self,
        result_text: str,
        expected_length: int
    ) -> List[bool]:
        """LLM応答をパース"""
        try:
            # [1, 0, 1, ...] 形式を抽出
            match = re.search(r'\[([0-9,\s]+)\]', result_text)
            if match:
                numbers_str = match.group(1)
                numbers = [int(n.strip()) for n in numbers_str.split(',') if n.strip()]
                if len(numbers) == expected_length:
                    return [bool(n) for n in numbers]

            # フォールバック: 0/1を順に抽出
            numbers = re.findall(r'[01]', result_text)
            if len(numbers) >= expected_length:
                return [bool(int(n)) for n in numbers[:expected_length]]

            logger.warning(f"Failed to parse LLM result, using all False: {result_text[:100]}")
            return [False] * expected_length

        except Exception as e:
            logger.error(f"Error parsing LLM result: {e}")
            return [False] * expected_length

    async def classify_term_relationships(
        self,
        candidates: List[str],
        full_text: str
    ) -> Dict[str, Dict[str, List[str]]]:
        """
        用語関係を3カテゴリに分類

        Args:
            candidates: 候補用語リスト
            full_text: 全文テキスト

        Returns:
            {
                'term1': {
                    'synonyms': [...],      # Phase 3+4で確定
                    'variants': [...],      # Phase 2
                    'related_terms': [...]  # Phase 2
                }
            }
        """
        result = defaultdict(lambda: {
            'synonyms': [],
            'variants': [],
            'related_terms': []
        })

        # Phase 2: ヒューリスティック分類
        logger.info("Phase 2: Heuristic classification...")
        variants = self.statistical_extractor.detect_variants(candidates)
        related = self.statistical_extractor.detect_related_terms(candidates, full_text)

        # Phase 3: PGVector意味的類似度
        logger.info("Phase 3: Semantic similarity with PGVector...")
        synonym_candidates = await self.detect_semantic_synonyms_pgvector(
            candidates,
            similarity_threshold=0.85
        )

        # Phase 4: LLM最終判定（必須）
        logger.info("Phase 4: LLM validation (required)...")
        synonyms = await self.validate_synonyms_with_llm(synonym_candidates)

        # 統合
        for term in candidates:
            if term in synonyms:
                result[term]['synonyms'] = synonyms[term]
            if term in variants:
                result[term]['variants'] = variants[term]
            if term in related:
                result[term]['related_terms'] = related[term]

        return dict(result)


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