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
from src.utils.profiler import get_profiler, timer

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
        # プロファイラーでカウント
        get_profiler().count_sudachi_call()

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
            r'(?<![A-Za-z0-9])[A-Z][0-9]{1,2}(?![A-Za-z0-9])',  # 英字1文字+数字1-2桁（G0, C1, T10など）
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

    def extract_candidates_with_spans(self, text: str) -> Dict[str, Set[Tuple[int, int]]]:
        """
        候補用語とスパン位置を抽出（ハイブリッドアプローチ + スパン追跡）

        Args:
            text: 入力テキスト

        Returns:
            候補用語 -> スパン位置集合のマッピング
            例: {"舶用エンジン": {(50, 55), (120, 125)}}
        """
        candidates_with_spans = defaultdict(set)

        # 1. 正規表現パターンマッチング（スパン付き）
        if self.use_regex_patterns:
            pattern_spans = self._extract_by_patterns_with_spans(text)
            pattern_count = sum(len(spans) for spans in pattern_spans.values())
            logger.info(f"[1/4] 正規表現パターン: {len(pattern_spans)}種類、{pattern_count}個の出現")
            for term, spans in pattern_spans.items():
                candidates_with_spans[term].update(spans)

        # 2. Mode C: 自然な複合語
        mode_c_spans = self._extract_with_mode_c_with_spans(text)
        mode_c_count = sum(len(spans) for spans in mode_c_spans.values())
        logger.info(f"[2/4] Mode C（長単位）: {len(mode_c_spans)}種類、{mode_c_count}個の出現")
        for term, spans in mode_c_spans.items():
            candidates_with_spans[term].update(spans)

        # 3. Mode A + n-gram
        ngram_spans = self._extract_ngrams_with_spans(text)
        ngram_count = sum(len(spans) for spans in ngram_spans.values())
        logger.info(f"[3/4] Mode A + n-gram: {len(ngram_spans)}種類、{ngram_count}個の出現")
        for term, spans in ngram_spans.items():
            candidates_with_spans[term].update(spans)

        # 4. 複合名詞
        compound_spans = self._extract_compound_nouns_with_spans(text)
        compound_count = sum(len(spans) for spans in compound_spans.values())
        logger.info(f"[4/4] 複合名詞: {len(compound_spans)}種類、{compound_count}個の出現")
        for term, spans in compound_spans.items():
            candidates_with_spans[term].update(spans)

        return candidates_with_spans

    def merge_candidates_by_spans(
        self,
        candidates_with_spans: Dict[str, Set[Tuple[int, int]]]
    ) -> Dict[str, int]:
        """
        スパン重複を排除して真の頻度を計算

        Args:
            candidates_with_spans: 候補用語とスパン位置の集合

        Returns:
            候補用語とユニークなスパン数（真の頻度）
        """
        # 頻度フィルタ前の統計
        freq_distribution = defaultdict(int)
        for term, spans in candidates_with_spans.items():
            freq = len(spans)
            freq_distribution[freq] += 1

        # 最小頻度フィルタ
        filtered = {
            term: len(spans)
            for term, spans in candidates_with_spans.items()
            if len(spans) >= self.min_frequency
        }

        # フィルタされた用語を記録
        filtered_out = {
            term: len(spans)
            for term, spans in candidates_with_spans.items()
            if len(spans) < self.min_frequency
        }

        if filtered_out:
            logger.info(f"頻度フィルタで除外: {len(filtered_out)}個（頻度<{self.min_frequency}）")
            # 頻度1の用語例を記録
            freq_1_terms = [term for term, freq in filtered_out.items() if freq == 1]
            if freq_1_terms:
                sample = freq_1_terms[:10]
                logger.debug(f"  頻度1の除外例: {', '.join(sample)}{'...' if len(freq_1_terms) > 10 else ''}")

        return filtered

    def extract_candidates(self, text: str) -> Dict[str, int]:
        """
        候補用語の抽出（スパンベース重複排除版）

        Args:
            text: 入力テキスト

        Returns:
            候補用語と出現頻度の辞書（スパン重複排除済み）
        """
        logger.info("=" * 70)
        logger.info("候補用語抽出開始")
        logger.info("=" * 70)

        # スパン付きで抽出
        candidates_with_spans = self.extract_candidates_with_spans(text)

        # 抽出手法別のカウント
        total_before_filter = sum(len(spans) for spans in candidates_with_spans.values())
        logger.info(f"抽出された候補（重複含む）: {total_before_filter}個")
        logger.info(f"ユニークな候補用語数: {len(candidates_with_spans)}個")

        # スパン重複排除して頻度計算
        candidates = self.merge_candidates_by_spans(candidates_with_spans)

        logger.info(f"頻度フィルタ後（min_frequency={self.min_frequency}）: {len(candidates)}個")
        logger.info("=" * 70)

        return candidates

    def _extract_with_mode_c_with_spans(self, text: str) -> Dict[str, Set[Tuple[int, int]]]:
        """
        Mode C（長単位）による複合語抽出（スパン付き）
        自然な複合語をそのまま取得（例: "舶用ディーゼルエンジン"）
        """
        mode_c_terms = defaultdict(set)

        tokens = self._safe_tokenize(text, self.sudachi_mode_c)

        offset = 0  # テキスト内の累積オフセット
        for token in tokens:
            term = token.surface()
            pos = token.part_of_speech()[0]

            # 名詞系のみを対象とし、複合語として成立しているもの
            if pos in ['名詞'] and len(term) >= 2:
                if self._is_valid_term(term):
                    # findを使ってスパンを計算
                    start = text.find(term, offset)
                    if start != -1:
                        span = (start, start + len(term))
                        mode_c_terms[term].add(span)
                        offset = start + len(term)
                    else:
                        offset += len(term)
            else:
                offset += len(term)

        return mode_c_terms

    def _extract_with_mode_c(self, text: str) -> Dict[str, int]:
        """Mode C複合語抽出（後方互換用）"""
        mode_c_spans = self._extract_with_mode_c_with_spans(text)
        return {term: len(spans) for term, spans in mode_c_spans.items()}

    def _extract_ngrams_with_spans(self, text: str) -> Dict[str, Set[Tuple[int, int]]]:
        """n-gram抽出（文単位処理、品詞ベース：名詞連続のみ、スパン付き）"""
        ngrams = defaultdict(set)

        # ===== 追加: 文単位で分割 =====
        sentences = self._split_into_sentences(text)

        current_offset = 0  # テキスト全体でのオフセット

        for sentence in sentences:
            # 文の開始位置を計算
            sentence_start = text.find(sentence, current_offset)
            if sentence_start == -1:
                continue

            # 文内でトークン化（Mode A: 短単位）
            tokens = self._safe_tokenize(sentence, self.sudachi_mode_a)

            # 名詞・接頭辞の連続を抽出（文内のみ）
            noun_sequences = []
            current_sequence = []
            current_start = 0
            offset = 0

            for token in tokens:
                pos = token.part_of_speech()[0]
                surface = token.surface()

                # 名詞または接頭辞の場合は連続に追加
                if pos in ['名詞', '接頭辞']:
                    if not current_sequence:
                        current_start = offset
                    current_sequence.append(surface)
                else:
                    # 連続が終了したら保存
                    if len(current_sequence) >= self.min_term_length:
                        noun_sequences.append((current_sequence, current_start))
                    current_sequence = []

                offset += len(surface)

            # 最後の連続も保存
            if len(current_sequence) >= self.min_term_length:
                noun_sequences.append((current_sequence, current_start))

            # n-gramを生成（文内のみ）
            for sequence, start_pos in noun_sequences:
                for n in range(self.min_term_length, min(self.max_term_length + 1, len(sequence) + 1)):
                    for i in range(len(sequence) - n + 1):
                        ngram_parts = sequence[i:i+n]
                        ngram = ''.join(ngram_parts)

                        # 元テキスト照合: 存在しない複合語を排除
                        if ngram not in sentence:
                            continue

                        if self._is_valid_term(ngram):
                            # 絶対位置を計算
                            ngram_start = sentence_start + start_pos + len(''.join(sequence[:i]))
                            ngram_end = ngram_start + len(ngram)
                            span = (ngram_start, ngram_end)
                            ngrams[ngram].add(span)

            # 次の文の検索開始位置を更新
            current_offset = sentence_start + len(sentence)

        return ngrams

    def _extract_ngrams(self, text: str) -> Dict[str, int]:
        """n-gram抽出（後方互換用）"""
        ngram_spans = self._extract_ngrams_with_spans(text)
        return {term: len(spans) for term, spans in ngram_spans.items()}

    def _extract_by_patterns_with_spans(self, text: str) -> Dict[str, Set[Tuple[int, int]]]:
        """正規表現パターンによる抽出（スパン付き）"""
        pattern_terms = defaultdict(set)

        for pattern in self.term_patterns:
            for match in pattern.finditer(text):
                term = match.group()

                # 括弧を除去（（BMS） → BMS）
                term = term.strip('（）()')

                # コロン形式から略語部分のみ抽出（SFOC: Specific → SFOC）
                if ':' in term:
                    term = term.split(':')[0].strip()

                if self._is_valid_term(term):
                    span = (match.start(), match.end())
                    pattern_terms[term].add(span)

        return pattern_terms

    def _extract_by_patterns(self, text: str) -> Dict[str, int]:
        """正規表現パターンによる抽出（後方互換用）"""
        pattern_spans = self._extract_by_patterns_with_spans(text)
        return {term: len(spans) for term, spans in pattern_spans.items()}

    def _extract_compound_nouns_with_spans(self, text: str) -> Dict[str, Set[Tuple[int, int]]]:
        """複合名詞の抽出（文単位処理、Mode A使用、スパン付き）"""
        compound_nouns = defaultdict(set)

        # ===== 追加: 文単位で分割 =====
        sentences = self._split_into_sentences(text)

        current_offset = 0  # テキスト全体でのオフセット

        for sentence in sentences:
            # 文の開始位置を計算
            sentence_start = text.find(sentence, current_offset)
            if sentence_start == -1:
                continue

            # 文内でトークン化（Mode A）
            tokens = self._safe_tokenize(sentence, self.sudachi_mode_a)
            current_compound = []
            current_start = 0
            offset = 0

            for idx, token in enumerate(tokens):
                pos = token.part_of_speech()[0]
                surface = token.surface()

                # 名詞または接頭辞の場合
                if pos in ['名詞', '接頭辞']:
                    # 数字のみのトークンは複合語を区切る（18, 2050などを除外）
                    if re.match(r'^\d+\.?\d*$', surface):
                        # 次のトークンが助数詞かチェック（2段、3層など）
                        if idx + 1 < len(tokens):
                            next_token = tokens[idx + 1]
                            next_pos = next_token.part_of_speech()
                            # pos[2]が"助数詞可能"なら数字を含める
                            if len(next_pos) > 2 and next_pos[2] and '助数詞' in next_pos[2]:
                                if not current_compound:
                                    current_start = offset
                                current_compound.append(surface)
                                offset += len(surface)
                                continue

                        # 助数詞でない場合は複合語を保存して区切る
                        if len(current_compound) >= self.min_term_length:
                            compound = ''.join(current_compound)
                            # 元テキスト照合: 存在しない複合語を排除
                            if compound in sentence:
                                # 複合語の最大長チェック（15文字以内）
                                if len(compound) <= 15 and self._is_valid_term(compound):
                                    # 絶対位置を計算
                                    abs_start = sentence_start + current_start
                                    span = (abs_start, abs_start + len(compound))
                                    compound_nouns[compound].add(span)
                        current_compound = []
                    else:
                        # 通常の名詞は複合語に追加
                        if not current_compound:
                            current_start = offset
                        current_compound.append(surface)
                else:
                    # 複合名詞が終了
                    if len(current_compound) >= self.min_term_length:
                        compound = ''.join(current_compound)
                        # 元テキスト照合: 存在しない複合語を排除
                        if compound in sentence:
                            # 複合語の最大長チェック（15文字以内）
                            if len(compound) <= 15 and self._is_valid_term(compound):
                                # 絶対位置を計算
                                abs_start = sentence_start + current_start
                                span = (abs_start, abs_start + len(compound))
                                compound_nouns[compound].add(span)
                    current_compound = []

                offset += len(surface)

            # 最後の複合名詞
            if len(current_compound) >= self.min_term_length:
                compound = ''.join(current_compound)
                # 元テキスト照合: 存在しない複合語を排除
                if compound in sentence:
                    # 複合語の最大長チェック（15文字以内）
                    if len(compound) <= 15 and self._is_valid_term(compound):
                        # 絶対位置を計算
                        abs_start = sentence_start + current_start
                        span = (abs_start, abs_start + len(compound))
                        compound_nouns[compound].add(span)

            # 次の文の検索開始位置を更新
            current_offset = sentence_start + len(sentence)

        return compound_nouns

    def _extract_compound_nouns(self, text: str) -> Dict[str, int]:
        """複合名詞の抽出（後方互換用）"""
        compound_spans = self._extract_compound_nouns_with_spans(text)
        return {term: len(spans) for term, spans in compound_spans.items()}

    def _is_valid_term(self, term: str) -> bool:
        """用語の妥当性チェック（文字列 + 品詞ベース）"""
        # 空文字列チェック
        if not term:
            return False

        # パターンベースのブラックリスト（図表・ページ番号など）
        blacklist_patterns = [
            r'^[pP]\d+$',        # p123, P456（ページ番号）
            r'^[pP][pP]\d+$',    # pp123
            r'^[fF]ig\d+$',      # fig1, Fig2（図番号）
            r'^[tT](able|bl)\d+$',  # table1, tbl2（表番号）
            r'^[eE]q\d+$',       # eq1, Eq2（式番号）
            r'^[vV]\d*$',        # v, v1, v2, V3（バージョン）
            r'^[vV]er\d*$',      # ver, ver1, ver2（バージョン）
            r'^[nN]o\.?\d+$',    # no.1, No2（番号）
            r'^[cC]h\d+$',       # ch1, Ch2（章番号）
            r'^[sS]ec\d+$',      # sec1, Sec2（節番号）
        ]

        for pattern in blacklist_patterns:
            if re.match(pattern, term):
                return False

        # 英数字コードの自動承認（ブラックリスト除外後）
        if self._is_alphanumeric_code(term):
            return True  # 品詞チェックをスキップして承認

        # 最小文字数チェック（3文字未満除外）- 英数字コード以外に適用
        if len(term) < 3:
            return False

        # 図表番号パターンを除外（第3図、第1表、第2式など）
        if re.match(r'^第\d+[図表式]', term):
            return False

        # 図表番号を含む用語を除外（第7図単段ETC等、テキスト処理で連結された場合）
        if re.search(r'第\d+[図表式]', term):
            return False

        # 助詞・助動詞ブラックリスト
        particles = {
            'の', 'に', 'が', 'を', 'は', 'で', 'と', 'から', 'まで', 'より',
            'へ', 'や', 'も', 'て', 'ば', 'ない', 'だ', 'です', 'ます', 'である'
        }
        if term in particles:
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

        # 数字で始まる一般複合語を除外（型番・規格は除く）
        # 型番パターン: 6DE-18, A-100X, ISO9001など（英数字+ハイフン）
        # 除外パターン: 18コンプレッサ, 2050年カーボンニュートラル, 3.5コンプレッサ圧力比
        if re.match(r'^[\d.]+', term):
            # 型番・規格パターンかチェック
            # - ハイフンを含む: 6DE-18, A-100
            # - 英字で始まり数字を含む: ISO9001, L28ADF
            if not re.match(r'^[A-Z]+[\dA-Z-]+$', term):  # 型番・規格パターン
                return False

        # 汎用語ブラックリスト拡充
        generic_terms = {
            # 日本語汎用語
            'エリア', 'モード', 'こと', 'もの', 'ため', 'よう', 'など',
            'について', 'により', 'における', 'として', 'による',
            'それ', 'これ', 'あれ', 'その', 'この', 'あの',
            'とき', 'とこ', 'ところ', 'ほか', 'さらに', 'また', 'および',
            # 英語汎用語・略語（小文字も含む）
            'and', 'the', 'of', 'in', 'for', 'with', 'from', 'to', 'at', 'by',
            'stage', 'fig', 'figure', 'table', 'exterior', 'interior', 'single',
            'system', 'Fig', 'Table', 'Stage', 'System', 'Exterior', 'Interior', 'Single',
            # 図表・ページ関連（パターンに引っかからないもの）
            'page', 'Page', 'ref', 'Ref', 'reference', 'Reference',
            'ver', 'version', 'Ver', 'Version', 'chapter', 'Chapter', 'section', 'Section',
            # 不完全な用語（語尾切れ）
            'システ', 'インタ', 'クーラ', 'アシスト', 'モータ', 'ロータ',
            'タービン', 'コンプレッサ', 'スタック', 'カモータ',
            # 図表関連（接尾辞ではなく名詞として認識されるため）
            '図', '表', '式'
        }
        if term in generic_terms:
            return False

        # 英字略語は品詞チェックをスキップして自動承認（BMS、AVR、EMS、SFOC、NOx、CO2など）
        abbreviation_pattern = re.compile(r'^[A-Z]{2,5}[0-9x]?$')
        if abbreviation_pattern.match(term):
            return True

        # 品詞チェック：名詞・接頭辞・接尾辞・記号（一部）のみ許可
        try:
            tokens = self._safe_tokenize(term, self.sudachi_mode_a)

            # 不完全語検出: 形態素で再構成できるかチェック
            reconstructed = ''.join([t.surface() for t in tokens])
            if reconstructed != term:
                logger.debug(f"Incomplete term detected: {term} != {reconstructed}")
                return False

            # 先頭トークンのチェック
            if tokens:
                first_token = tokens[0]
                first_pos = first_token.part_of_speech()
                first_surface = first_token.surface()

                # 図/表/式で始まる複合語を除外（図単段ETC等）
                # ※「図」は接尾辞ではなく名詞として認識されるため、表面形でチェック
                if first_surface in ['図', '表', '式'] and len(tokens) > 1:
                    return False

                # 助数詞で始まる場合は除外（段コンプレッサ等）
                if len(first_pos) > 2 and first_pos[2] and '助数詞' in first_pos[2]:
                    return False

            for token in tokens:
                pos = token.part_of_speech()[0]
                pos_sub = token.part_of_speech()[1] if len(token.part_of_speech()) > 1 else ''

                # 助詞・助動詞を明示的除外
                if pos in ['助詞', '助動詞']:
                    return False

                # 名詞-非自立（こと、もの等）を除外
                if pos == '名詞' and pos_sub == '非自立':
                    return False

                # 許可する品詞
                allowed_pos = ['名詞', '接頭辞', '接尾辞']
                # 許可する記号（単位など）
                allowed_symbols = ['％', '°', '/', '-', '・']

                if pos not in allowed_pos:
                    # 記号の場合は許可リストをチェック
                    if pos == '補助記号' and token.surface() in allowed_symbols:
                        continue
                    # それ以外（動詞、形容詞など）は除外
                    return False

        except Exception as e:
            # 形態素解析エラーの場合は除外（通さない）
            logger.debug(f"Tokenization failed for term: {term}, rejecting: {e}")
            return False

        return True

    def _is_alphanumeric_code(self, term: str) -> bool:
        """英数字コード（型式番号など）かどうかを判定"""

        # 条件1: 英字と数字が両方含まれる
        has_alpha = bool(re.search(r'[A-Za-z]', term))
        has_digit = bool(re.search(r'[0-9]', term))
        if not (has_alpha and has_digit):
            return False

        # 条件2: 全体が英数字+記号のみ
        if not re.match(r'^[A-Za-z0-9\-_.]+$', term):
            return False

        # 条件3: 長さが2-15文字
        if len(term) < 2 or len(term) > 15:
            return False

        # 条件4: ひらがな・カタカナを含まない
        if re.search(r'[ぁ-んァ-ヴ]', term):
            return False

        return True

    def detect_variants(self, candidates: List[str]) -> Dict[str, List[str]]:
        """
        表記ゆれ検出（編集距離ベース）

        Args:
            candidates: 候補用語リスト

        Returns:
            用語と表記ゆれのマッピング
        """
        # 候補を事前フィルタリング（不完全用語を除外）
        valid_candidates = [c for c in candidates if self._is_valid_term(c)]

        variants = defaultdict(set)

        for i, cand1 in enumerate(valid_candidates):
            for cand2 in valid_candidates[i+1:]:
                if len(cand1) >= 3 and len(cand2) >= 3:
                    # Levenshtein比率で類似度計算
                    similarity = SequenceMatcher(None, cand1, cand2).ratio()
                    # 閾値: 0.85-0.98（完全一致1.0は除外）
                    if 0.85 < similarity < 0.98:
                        # 両方の用語が有効かチェック
                        if self._is_valid_term(cand1) and self._is_valid_term(cand2):
                            variants[cand1].add(cand2)
                            variants[cand2].add(cand1)

        return {k: list(v) for k, v in variants.items() if v}

    def detect_related_terms(
        self,
        candidates: List[str],
        full_text: str,
        max_related: int = 5,
        min_term_length: int = 4,
        independent_ratio_threshold: float = 0.3
    ) -> Dict[str, List[str]]:
        """
        関連語検出（包含関係・PMI共起）with C-value based filtering

        Args:
            candidates: 候補用語リスト
            full_text: 全文テキスト
            max_related: 1用語あたりの最大関連語数（デフォルト5）
            min_term_length: 関連語として認める最小文字数（デフォルト4）
            independent_ratio_threshold: 独立出現率の閾値（デフォルト0.3 = 30%）

        Returns:
            用語と関連語のマッピング
        """
        # 候補を事前フィルタリング（不完全用語を除外）
        valid_candidates = [c for c in candidates if self._is_valid_term(c)]

        related = defaultdict(set)

        # 1. 包含関係（上位語/下位語）with C-value filtering
        for i, cand1 in enumerate(valid_candidates):
            # ===== 追加: 極短い部分語はスキップ =====
            if len(cand1) < min_term_length:
                continue
            for cand2 in valid_candidates[i+1:]:
                if cand1 != cand2:
                    # 完全包含（短い方が長い方に含まれる）
                    if cand1 in cand2 and len(cand2) > len(cand1) + 1:
                        # ===== C-value based filtering: 短い方が独立した用語かチェック =====
                        independent_ratio = self._calculate_independent_occurrence_ratio(
                            cand1, full_text, valid_candidates
                        )
                        # 独立出現率が閾値以上なら有効な関連語として追加
                        if independent_ratio >= independent_ratio_threshold:
                            related[cand1].add(cand2)
                            related[cand2].add(cand1)
                    elif cand2 in cand1 and len(cand1) > len(cand2) + 1:
                        # ===== C-value based filtering: 短い方が独立した用語かチェック =====
                        independent_ratio = self._calculate_independent_occurrence_ratio(
                            cand2, full_text, valid_candidates
                        )
                        # 独立出現率が閾値以上なら有効な関連語として追加
                        if independent_ratio >= independent_ratio_threshold:
                            related[cand2].add(cand1)
                            related[cand1].add(cand2)

        # 2. PMI共起分析（改善版：完全一致のみ）
        cooccurrence_map = defaultdict(lambda: defaultdict(int))
        window_size = 10

        # Sudachiで形態素解析
        tokens = self._safe_tokenize(full_text, self.sudachi_mode_a)
        words = [token.surface() for token in tokens]

        # 候補用語の頻度カウント（valid_candidatesのみ）
        word_freq = defaultdict(int)
        for word in words:
            if word in valid_candidates:
                word_freq[word] += 1

        # 共起カウント（完全一致のみ、valid_candidatesのみ）
        for i, word in enumerate(words):
            if word in valid_candidates:
                window_start = max(0, i - window_size)
                window_end = min(len(words), i + window_size + 1)

                for j in range(window_start, window_end):
                    if i != j and words[j] in valid_candidates:
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

        # ===== 修正: 関連語のフィルタリング強化 =====
        filtered_related = {}
        for term, related_set in related.items():
            related_list = list(related_set)
            # 各関連語が有効かつmin_term_length以上かチェック
            valid_related = [
                r for r in related_list
                if self._is_valid_term(r) and len(r) >= min_term_length
            ]

            # 上位max_related個のみ採用（頻度順でソートするのが理想だが、簡易版として先頭から）
            if valid_related:
                filtered_related[term] = valid_related[:max_related]

        return filtered_related

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

        # 2. TF-IDF計算（平滑化版）
        tfidf_scores = {}
        for term in vocabulary:
            # 各文書でのTF-IDF合計
            total_tfidf = 0.0
            for doc_tf in term_freq_per_doc:
                raw_tf = doc_tf.get(term, 0)
                if raw_tf > 0 and df.get(term, 0) > 0:
                    # サブリニアTF（log圧縮で頻度10と100の差を緩和）
                    tf = 1 + math.log(raw_tf)
                    # Laplace平滑化IDF（df=Nで0、df=0でエラーを防ぐ）
                    idf = math.log((N + 1) / (df[term] + 1)) + 1
                    total_tfidf += tf * idf

            tfidf_scores[term] = total_tfidf

        return tfidf_scores

    def _calculate_independent_occurrence_ratio(
        self,
        term: str,
        full_text: str,
        candidates: List[str]
    ) -> float:
        """
        用語が独立して使われる比率を計算

        Args:
            term: 対象用語
            full_text: 全文テキスト
            candidates: 全候補用語リスト

        Returns:
            独立出現回数 / 全出現回数 (0.0 - 1.0)
        """
        # 全出現回数
        total_occurrences = full_text.count(term)

        if total_occurrences == 0:
            return 0.0

        # 複合語内での出現をカウント
        compound_occurrences = 0
        for candidate in candidates:
            if candidate != term and term in candidate:
                compound_occurrences += full_text.count(candidate)

        # 独立出現回数
        independent = total_occurrences - compound_occurrences

        # 独立出現率を返す
        ratio = independent / total_occurrences if total_occurrences > 0 else 0.0

        return max(0.0, ratio)  # 負にならないように

    def calculate_cvalue(
        self,
        candidates: Dict[str, int],
        full_text: str = ""
    ) -> Dict[str, float]:
        """
        C-value計算（ネストした用語の考慮）- 形態素数ベース
        独立出現率による非独立語のフィルタリング機能を追加

        C-value = log2(|a|) * freq(a) - (1/|Ta|) * Σ freq(b)

        where:
        - |a| = 用語aの長さ（形態素数）
        - freq(a) = 用語aの頻度
        - Ta = aを含むより長い用語の集合
        - b ∈ Ta

        Args:
            candidates: 候補用語と頻度の辞書
            full_text: 全文テキスト（独立出現率計算用、オプション）
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
        candidates_list = list(candidates.keys())
        for term in candidates:
            freq = candidates[term]

            # 独立出現率チェック（full_textが提供されている場合）
            if full_text and term in nested_info:
                # 他の用語に包含されている場合のみチェック
                independent_ratio = self._calculate_independent_occurrence_ratio(
                    term, full_text, candidates_list
                )

                # 独立出現率が30%未満（70%以上が複合語内）の場合は非独立語として除外
                if independent_ratio < 0.3:
                    logger.debug(
                        f"Filtering non-independent term: {term} "
                        f"(independent ratio: {independent_ratio:.2%})"
                    )
                    cvalues[term] = 0.0
                    continue

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
        C-valueスコアが0の用語（非独立語）は除外

        Args:
            tfidf_scores: TF-IDFスコア
            cvalue_scores: C-valueスコア
            stage: "seed" (シード選定用) or "final" (最終スコア用)

        Returns:
            結合スコア（再正規化なし）
        """
        # C-valueスコアが0の用語を除外（非独立語フィルタリング）
        valid_cvalue_scores = {k: v for k, v in cvalue_scores.items() if v > 0}

        # 個別に正規化
        tfidf_norm = self.min_max_normalize(tfidf_scores)
        cvalue_norm = self.min_max_normalize(valid_cvalue_scores)

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
        # C-valueスコアが有効な用語のみを対象とする
        all_terms = set(tfidf_scores.keys()) | set(valid_cvalue_scores.keys())

        for term in all_terms:
            # C-valueスコアが0（除外対象）の用語はスキップ
            if cvalue_scores.get(term, 0.0) == 0:
                continue

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

        # 4. C-value計算（独立出現率フィルタリング付き）
        cvalue_scores = self.calculate_cvalue(candidates, full_text=text)

        # 5. シード選定用スコア（Stage A）
        seed_scores = self.calculate_combined_scores(
            tfidf_scores, cvalue_scores, stage="seed"
        )

        # 6. 最終スコア（Stage B）
        final_scores = self.calculate_combined_scores(
            tfidf_scores, cvalue_scores, stage="final"
        )

        # 7. ExtractedTermオブジェクト作成（スコア0を除外）
        terms = []
        candidates_list = list(candidates.keys())
        for term, score in final_scores.items():
            # スコア0の用語（非独立語）は除外
            if score == 0:
                continue

            # 独立出現率を計算（メタデータ記録用）
            independent_ratio = self._calculate_independent_occurrence_ratio(
                term, text, candidates_list
            )

            extracted_term = ExtractedTerm(
                term=term,
                score=score,
                tfidf_score=tfidf_scores.get(term, 0.0),
                cvalue_score=cvalue_scores.get(term, 0.0),
                frequency=candidates.get(term, 0),
                metadata={
                    "seed_score": seed_scores.get(term, 0.0),
                    "independent_ratio": independent_ratio
                }
            )
            terms.append(extracted_term)

        # 8. スコア降順でソート
        terms.sort(key=lambda x: x.score, reverse=True)

        # 9. 上位N件を返す
        return terms[:top_n]

    def _split_into_sentences(self, text: str) -> List[str]:
        """文単位で厳格に分割（改行・句読点・括弧を境界とする）"""
        # 強い区切り: 句点、改行、括弧、読点
        # ※改行を「強い文境界」として扱い、誤結合を防止
        # ※句点後のスペースも境界として扱う（Azure DIのMarkdown変換対応）
        sentences = re.split(r'[。！？\n「」『』、]+|(?<=[。！？])\s+', text)
        # 極端に短い断片を除外（3文字未満）
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) >= 3]

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

        # 候補ペアをフラット化（重複排除 + 包含関係除外）
        pairs = []
        seen = set()
        for term1, candidates in synonym_candidates.items():
            for term2 in candidates:
                # 包含関係チェック（部分文字列）
                if term1 in term2 or term2 in term1:
                    logger.debug(f"Skipping substring pair: 「{term1}」⊂「{term2}」")
                    continue

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
                from .prompts import get_synonym_validation_prompt
                prompt = get_synonym_validation_prompt(batch)

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
        related = self.statistical_extractor.detect_related_terms(
            candidates,
            full_text,
            max_related=self.config.max_related_terms_per_candidate,
            min_term_length=self.config.min_related_term_length
        )

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