"""
term_extraction.py
==================
専門用語抽出と類義語検出を統合したモジュール
JargonDictionaryManagerとSynonymDetectorの機能を統合
"""

from __future__ import annotations

import asyncio
import json
import logging
import pandas as pd
from collections import defaultdict, Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sudachipy import tokenizer, dictionary

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import PGVector
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
from pydantic import BaseModel, Field

# 高度な統計的手法のインポート
try:
    from .advanced_term_extraction import (
        AdvancedStatisticalExtractor,
        ExtractedTerm
    )
    HAS_ADVANCED_EXTRACTION = True
except ImportError:
    HAS_ADVANCED_EXTRACTION = False
    logger = logging.getLogger(__name__)
    logger.warning("Advanced term extraction not available")

logger = logging.getLogger(__name__)

from src.rag.pdf_processors import AzureDocumentIntelligenceProcessor


# ========== Pydantic Models ==========
class Term(BaseModel):
    """専門用語の構造"""
    headword: str = Field(description="専門用語の見出し語")
    synonyms: List[str] = Field(default_factory=list, description="類義語・別名のリスト")
    definition: str = Field(description="30-50字程度の簡潔な定義")


class TermList(BaseModel):
    """用語リストの構造"""
    terms: List[Term] = Field(default_factory=list, description="専門用語のリスト")


class _InMemoryLoader:
    """Simple loader that returns preloaded LangChain Documents."""

    def __init__(self, docs: List[Document]):
        self._docs = docs

    def load(self) -> List[Document]:
        return self._docs


# ========== JargonDictionaryManager ==========
class JargonDictionaryManager:
    """専門用語辞書の管理クラス"""

    def __init__(self, connection_string: str, table_name: str = "jargon_dictionary", engine: Optional[Engine] = None):
        self.connection_string = connection_string
        self.table_name = table_name
        self.engine: Engine = engine or create_engine(connection_string)
        self._init_jargon_table()

    def _init_jargon_table(self):
        """専門用語辞書テーブルの初期化"""
        with self.engine.connect() as conn:
            conn.execute(text(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id SERIAL PRIMARY KEY,
                    term TEXT UNIQUE NOT NULL,
                    definition TEXT NOT NULL,
                    domain TEXT,
                    aliases TEXT[],
                    related_terms TEXT[],
                    confidence_score FLOAT DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_jargon_term ON {self.table_name} (LOWER(term))"))
            conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_jargon_aliases ON {self.table_name} USING GIN(aliases)"))
            conn.commit()

    def add_term(self, term: str, definition: str, domain: Optional[str] = None,
                 aliases: Optional[List[str]] = None, related_terms: Optional[List[str]] = None,
                 confidence_score: float = 1.0) -> bool:
        """用語を辞書に追加または更新"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text(f"""
                    INSERT INTO {self.table_name}
                    (term, definition, domain, aliases, related_terms, confidence_score)
                    VALUES (:term, :definition, :domain, :aliases, :related_terms, :confidence_score)
                    ON CONFLICT (term) DO UPDATE SET
                        definition = EXCLUDED.definition,
                        domain = EXCLUDED.domain,
                        aliases = EXCLUDED.aliases,
                        related_terms = EXCLUDED.related_terms,
                        confidence_score = EXCLUDED.confidence_score,
                        updated_at = CURRENT_TIMESTAMP
                """), {
                    "term": term, "definition": definition, "domain": domain,
                    "aliases": aliases or [], "related_terms": related_terms or [],
                    "confidence_score": confidence_score
                })
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error adding term: {e}")
            return False

    def lookup_terms(self, terms: List[str]) -> Dict[str, Dict[str, Any]]:
        """複数の用語を辞書から検索"""
        if not terms:
            return {}

        results = {}
        try:
            with self.engine.connect() as conn:
                placeholders = ', '.join([f':term_{i}' for i in range(len(terms))])
                query = text(f"""
                    SELECT term, definition, domain, aliases, related_terms, confidence_score
                    FROM {self.table_name}
                    WHERE LOWER(term) IN ({placeholders})
                    OR term = ANY(:aliases_check)
                """)
                params = {f"term_{i}": term.lower() for i, term in enumerate(terms)}
                params["aliases_check"] = terms

                rows = conn.execute(query, params).fetchall()
                for row in rows:
                    results[row.term] = {
                        "definition": row.definition, "domain": row.domain,
                        "aliases": row.aliases or [], "related_terms": row.related_terms or [],
                        "confidence_score": row.confidence_score
                    }
        except Exception as e:
            logger.error(f"Error looking up terms: {e}")
        return results

    def delete_terms(self, terms: List[str]) -> tuple[int, int]:
        """複数の用語を一括削除"""
        if not terms:
            return 0, 0

        deleted = 0
        errors = 0
        try:
            with self.engine.connect() as conn, conn.begin():
                for term in terms:
                    if not term:
                        errors += 1
                        continue
                    result = conn.execute(
                        text(f"DELETE FROM {self.table_name} WHERE term = :term"),
                        {"term": term}
                    )
                    deleted += result.rowcount or 0
        except Exception as e:
            logger.error(f"Bulk delete error: {e}")
            return deleted, len(terms) - deleted
        return deleted, errors

    def get_all_terms(self) -> List[Dict[str, Any]]:
        """全ての用語を取得"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(f"SELECT * FROM {self.table_name} ORDER BY term")).fetchall()
                return [dict(row._mapping) for row in result]
        except Exception as e:
            logger.error(f"Error getting all terms: {e}")
            return []


# ========== TermExtractor ==========
class TermExtractor:
    """専門用語抽出の統合クラス"""

    def __init__(self, config, llm, embeddings, vector_store, pg_url, jargon_table_name):
        self.config = config
        self.llm = llm
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.pg_url = pg_url
        self.jargon_table_name = jargon_table_name

        # 高度な統計的抽出を使用する場合、形態素解析器は不要
        # (AdvancedStatisticalExtractorが内部でハイブリッドMode処理を行う)
        self.use_advanced_extraction = getattr(config, 'use_advanced_extraction', True) and HAS_ADVANCED_EXTRACTION

        # 従来の抽出方法用（use_advanced_extraction=Falseの場合のみ）
        if not self.use_advanced_extraction:
            self.tokenizer_obj = dictionary.Dictionary().create()
            self.sudachi_mode_a = tokenizer.Tokenizer.SplitMode.A
        else:
            self.tokenizer_obj = None
            self.sudachi_mode_a = None

        # MarkdownTextSplitterを使用してMarkdown構造を保持
        from langchain.text_splitter import MarkdownTextSplitter
        self.text_splitter = MarkdownTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )
        self.json_parser = JsonOutputParser(pydantic_object=TermList)
        self._init_prompts()

        # 高度な統計的抽出器の初期化
        if self.use_advanced_extraction:
            self.statistical_extractor = AdvancedStatisticalExtractor(
                min_term_length=2,
                max_term_length=6,
                min_frequency=2,
                use_regex_patterns=True
            )

        # SemReRankの初期化
        self.semrerank = None
        if getattr(config, 'semrerank_enabled', True) and self.use_advanced_extraction and pg_url:
            try:
                from .semrerank import SemReRank
                self.semrerank = SemReRank(
                    embeddings=embeddings,
                    connection_string=pg_url,
                    relmin=getattr(config, 'semrerank_relmin', 0.5),
                    reltop=getattr(config, 'semrerank_reltop', 0.15),
                    alpha=getattr(config, 'semrerank_alpha', 0.85),
                    seed_percentile=getattr(config, 'semrerank_seed_percentile', 15.0)
                )
                logger.info("SemReRank initialized successfully")
            except Exception as e:
                logger.warning(f"SemReRank initialization failed: {e}. Continuing without SemReRank.")
                self.semrerank = None

        # Initialize Azure Document Intelligence processor
        try:
            self.pdf_processor = AzureDocumentIntelligenceProcessor(config)
        except Exception as exc:
            logger.error(f"Azure Document Intelligence processor initialization failed: {exc}")
            self.pdf_processor = None

    def _init_prompts(self):
        """プロンプトテンプレートの初期化"""
        from .prompts import get_term_extraction_validation_prompt
        self.validation_prompt = get_term_extraction_validation_prompt().partial(
            format_instructions=self.json_parser.get_format_instructions()
        )

    async def extract_from_documents(self, file_paths: List[Path]) -> Dict[str, Any]:
        """複数の文書から専門用語を抽出（ドキュメントごとに候補抽出）

        Returns:
            Dict with keys:
                - "terms": List[Dict] - 専門用語リスト
                - "candidates": List[Dict] - 候補用語リスト（HDBSCAN用）
        """
        all_chunks = []
        per_document_texts = []  # ドキュメントごとのテキスト

        # 文書の読み込みと分割
        for file_path in file_paths:
            try:
                loader = self._get_loader(file_path)
                docs = loader.load()
                chunks = self.text_splitter.split_documents(docs)
                all_chunks.extend([c.page_content for c in chunks])

                # ドキュメントごとのテキストを保存
                doc_text = "\n".join([c.page_content for c in chunks])
                per_document_texts.append({
                    "file_path": str(file_path),
                    "text": doc_text
                })
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")

        if not all_chunks:
            logger.error("No text chunks generated")
            return {"terms": [], "candidates": []}

        # 高度な統計的抽出を使用
        if self.use_advanced_extraction and per_document_texts:
            logger.info("Using advanced statistical extraction (per-document candidate extraction)")
            result = await self._extract_with_advanced_method_per_document(per_document_texts, all_chunks)
            # 新しい辞書形式で返される
            return result
        else:
            # 従来の方法で抽出（候補なし）
            logger.info("Using traditional extraction")
            all_terms = []
            for chunk in all_chunks:
                terms = await self._extract_from_chunk(chunk)
                all_terms.extend(terms)
            terms = self._merge_duplicate_terms(all_terms)
            return {"terms": terms, "candidates": []}

    def _get_loader(self, file_path: Path):
        """ファイルタイプに応じたローダーを返す"""
        suffix = file_path.suffix.lower()

        # PDFはAzure Document Intelligenceで処理
        if suffix == '.pdf':
            docs = self._load_pdf_documents(file_path)
        # .txt/.mdはシンプルなテキストローダー
        elif suffix in ['.txt', '.md']:
            docs = self._load_text_documents(file_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        return _InMemoryLoader(docs)

    def _load_pdf_documents(self, file_path: Path) -> List[Document]:
        """Azure Document IntelligenceでPDFを読み込む"""
        docs: List[Document] = []

        if self.pdf_processor is not None:
            try:
                parsed = self.pdf_processor.parse_pdf(str(file_path))
                for text, metadata in parsed.get("texts", []):
                    if text and text.strip():
                        docs.append(Document(page_content=text, metadata=metadata or {}))
            except Exception as exc:
                logger.error(f"Azure Document Intelligence failed for {file_path}: {exc}")
                raise
        else:
            raise ValueError("Azure Document Intelligence processor not initialized")

        if not docs:
            raise ValueError(f"No text extracted from PDF: {file_path}")

        return docs

    def _load_text_documents(self, file_path: Path) -> List[Document]:
        """テキストファイル(.txt/.md)を読み込む"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            if not text.strip():
                raise ValueError(f"Empty text file: {file_path}")

            # メタデータを設定
            metadata = {
                "source": str(file_path),
                "file_type": file_path.suffix.lower()
            }

            docs = [Document(page_content=text, metadata=metadata)]
            logger.info(f"Loaded text file: {file_path} ({len(text)} characters)")

            return docs
        except Exception as exc:
            logger.error(f"Failed to load text file {file_path}: {exc}")
            raise

    async def _extract_from_chunk(self, chunk_text: str) -> List[Dict]:
        """単一チャンクから専門用語を抽出"""
        # 候補語生成
        candidates = self._generate_candidates(chunk_text)
        if not candidates:
            return []

        # LLMで検証（類義語ヒントなし - シンプル実装）
        try:
            prompt_input = {
                "chunk": chunk_text[:1500],
                "candidates": "\n".join(candidates[:100]),
                "synonym_hints": ""
            }

            chain = self.validation_prompt | self.llm | self.json_parser
            result = await chain.ainvoke(prompt_input)

            if hasattr(result, "terms"):
                terms_payload = result.terms
            elif isinstance(result, dict):
                terms_payload = result.get("terms", [])
            else:
                terms_payload = []

            normalized = []
            for term in terms_payload:
                if hasattr(term, "headword"):
                    headword = term.headword
                    synonyms = list(term.synonyms or [])
                    definition = term.definition
                elif isinstance(term, dict):
                    headword = term.get("headword")
                    synonyms = term.get("synonyms", [])
                    definition = term.get("definition", "")
                else:
                    continue

                if not headword:
                    continue

                normalized.append({
                    "headword": headword,
                    "synonyms": synonyms,
                    "definition": definition
                })

            return normalized
        except Exception as e:
            logger.error(f"LLM extraction error: {e}")
            return []

    def _generate_candidates(self, text: str) -> List[str]:
        """テキストから候補語を生成（Mode A使用）"""
        tokens = self.tokenizer_obj.tokenize(text, self.sudachi_mode_a)
        candidates = Counter()

        # 名詞句の抽出
        current_nouns = []
        for token in tokens:
            if token.part_of_speech()[0] == '名詞':
                current_nouns.append(token.surface())
            else:
                if current_nouns:
                    # 単語と複合語を候補に追加
                    for i in range(len(current_nouns)):
                        for j in range(i + 1, min(i + 5, len(current_nouns) + 1)):
                            compound = ''.join(current_nouns[i:j])
                            if 2 <= len(compound) <= 20:
                                candidates[compound] += 1
                    current_nouns = []

        # 頻度でフィルタリング
        return [term for term, freq in candidates.items() if freq >= 2]

    async def _extract_with_advanced_method_per_document(
        self, per_document_texts: List[Dict[str, str]], all_chunks: List[str]
    ) -> List[Dict]:
        """
        ドキュメントごとの候補抽出 + 全体統合パイプライン:
        1. 各ドキュメントで候補抽出
        2. 全ドキュメントで統計計算（TF-IDF, C-value）
        3. SemReRank
        4. RAG定義生成
        5. LLMフィルタ
        """
        logger.info("Starting per-document candidate extraction with global scoring")

        # 1. 各ドキュメントで候補抽出
        all_candidates = defaultdict(int)
        document_candidate_map = {}  # ドキュメントごとの候補リスト

        for doc_info in per_document_texts:
            file_path = doc_info["file_path"]
            text = doc_info["text"]

            logger.info(f"Extracting candidates from: {Path(file_path).name}")
            doc_candidates = self.statistical_extractor.extract_candidates(text)

            document_candidate_map[file_path] = doc_candidates

            # 全体候補に統合
            for term, freq in doc_candidates.items():
                all_candidates[term] += freq

        if not all_candidates:
            logger.warning("No candidates extracted from any document")
            return []

        logger.info(f"Total candidates across all documents: {len(all_candidates)}")
        logger.info(f"Processed {len(per_document_texts)} documents")

        # 2. 全ドキュメントでTF-IDF + C-value計算
        # 全テキストを結合して文単位に分割
        full_text = "\n".join([doc["text"] for doc in per_document_texts])
        documents = self._split_into_sentences(full_text)

        tfidf_scores = self.statistical_extractor.calculate_tfidf(documents, all_candidates)
        cvalue_scores = self.statistical_extractor.calculate_cvalue(all_candidates, full_text=full_text)

        # 3. 基底スコア計算（2段階）
        seed_scores = self.statistical_extractor.calculate_combined_scores(
            tfidf_scores, cvalue_scores, stage="seed"  # Stage A: シード選定用（C-value重視）
        )
        base_scores = self.statistical_extractor.calculate_combined_scores(
            tfidf_scores, cvalue_scores, stage="final"  # Stage B: 最終スコア用（TF-IDF重視）
        )

        # 3.5. 全候補でSemReRank実行（候補数制限を削除）
        # 以前はMAX_SEMRERANK_CANDIDATES=50に制限していたが、
        # 計算コストは許容範囲内（+10-15秒）なので全候補で実行
        candidates_for_semrerank = all_candidates
        seed_scores_for_semrerank = seed_scores
        base_scores_for_semrerank = base_scores

        logger.info(f"Processing {len(all_candidates)} candidates with SemReRank (no limit)")

        # 3.6. 略語にボーナススコアを付与
        logger.info("Applying abbreviation bonus scores")
        import re
        abbreviation_pattern = re.compile(r'^[A-Z]{2,5}$')
        abbreviation_count = 0

        for term in candidates_for_semrerank.keys():
            if abbreviation_pattern.match(term):
                # 略語には1.3倍のボーナス（辞書に存在する場合のみ）
                if term in base_scores_for_semrerank:
                    base_scores_for_semrerank[term] *= 1.3
                if term in seed_scores_for_semrerank:
                    seed_scores_for_semrerank[term] *= 1.3
                abbreviation_count += 1
                logger.info(f"  [BONUS] {term}: abbreviation bonus applied (×1.3)")

        logger.info(f"Applied bonus to {abbreviation_count} abbreviations")

        # 4. SemReRank適用（オプション）
        if self.semrerank:
            logger.info("Applying SemReRank enhancement")
            try:
                enhanced_scores = self.semrerank.enhance_scores(
                    candidates=list(candidates_for_semrerank.keys()),
                    base_scores=base_scores_for_semrerank,
                    seed_scores=seed_scores_for_semrerank
                )
            except Exception as e:
                logger.error(f"SemReRank failed: {e}. Using base scores.")
                enhanced_scores = base_scores
        else:
            logger.info("SemReRank disabled, using base scores")
            enhanced_scores = base_scores

        # 5. 類義語・関連語検出
        logger.info("Detecting synonyms (variants)")
        synonym_map = self.statistical_extractor.detect_variants(
            candidates=list(candidates_for_semrerank.keys())
        )
        logger.info(f"Detected synonyms for {len(synonym_map)} terms")

        logger.info("Detecting related terms (inclusion & co-occurrence)")
        related_map = self.statistical_extractor.detect_related_terms(
            candidates=list(candidates_for_semrerank.keys()),
            full_text=full_text,
            max_related=self.config.max_related_terms_per_candidate,
            min_term_length=self.config.min_related_term_length
        )
        logger.info(f"Detected related terms for {len(related_map)} terms")

        # 6. ExtractedTermオブジェクト化
        from .advanced_term_extraction import ExtractedTerm
        terms = [
            ExtractedTerm(
                term=term,
                score=enhanced_scores[term],
                tfidf_score=tfidf_scores.get(term, 0.0),
                cvalue_score=cvalue_scores.get(term, 0.0),
                frequency=all_candidates.get(term, 0),
                variants=synonym_map.get(term, []),  # 表記ゆれ
                related_terms=related_map.get(term, [])  # 関連語（包含・共起）
            )
            for term in enhanced_scores
        ]
        terms.sort(key=lambda x: x.score, reverse=True)

        logger.info(f"Sorted {len(terms)} terms by enhanced scores")

        # 7. 軽量LLMフィルタ（定義生成前、コスト削減）
        # 略語は問答無用で定義生成に進ませる
        abbreviations = [t for t in terms if self._is_abbreviation(t.term)]
        non_abbreviations = [t for t in terms if not self._is_abbreviation(t.term)]

        logger.info(f"Found {len(abbreviations)} abbreviations (will auto-include for definition generation)")

        enable_lightweight_filter = getattr(self.config, 'enable_lightweight_filter', True)

        if enable_lightweight_filter and self.llm:
            logger.info("Applying lightweight LLM filter (before definition generation)")
            try:
                # 上位N%を選択してから軽量フィルタ（略語以外）
                definition_percentile = getattr(self.config, 'definition_generation_percentile', 50.0)
                n_candidates = max(1, int(len(non_abbreviations) * definition_percentile / 100))
                candidate_terms = non_abbreviations[:n_candidates]

                logger.info(f"Lightweight filtering {len(candidate_terms)} candidates (top {definition_percentile}%)")

                filtered_terms = await self._lightweight_llm_filter(candidate_terms)
                logger.info(f"Lightweight filter passed: {len(filtered_terms)}/{len(candidate_terms)} terms")

                # フィルタ通過した用語 + 全略語を定義生成対象にする
                terms_for_definition = abbreviations + filtered_terms
            except Exception as e:
                logger.error(f"Lightweight filter failed: {e}. Proceeding without filtering.")
                # フィルタ失敗時は上位N%をそのまま使用 + 全略語
                definition_percentile = getattr(self.config, 'definition_generation_percentile', 50.0)
                n_candidates = max(1, int(len(non_abbreviations) * definition_percentile / 100))
                terms_for_definition = abbreviations + non_abbreviations[:n_candidates]
        else:
            # 軽量フィルタ無効時は上位N%をそのまま使用 + 全略語
            definition_percentile = getattr(self.config, 'definition_generation_percentile', 50.0)
            n_candidates = max(1, int(len(non_abbreviations) * definition_percentile / 100))
            terms_for_definition = abbreviations + non_abbreviations[:n_candidates]
            logger.info(f"Lightweight filter disabled, using top {definition_percentile}% ({len(terms_for_definition)} terms including {len(abbreviations)} abbreviations)")

        # 8. RAG定義生成
        if self.vector_store and self.llm:
            logger.info(f"Generating definitions with RAG for {len(terms_for_definition)} terms")
            try:
                from .prompts import get_definition_generation_prompt
                from langchain_core.output_parsers import StrOutputParser

                prompt = get_definition_generation_prompt()
                chain = prompt | self.llm | StrOutputParser()

                for i, term in enumerate(terms_for_definition, 1):
                    try:
                        # 略語の場合、ハイブリッド検索のために拡張クエリを使用
                        is_abbr = self._is_abbreviation(term.term)
                        search_query = f"{term.term} 略語" if is_abbr else term.term

                        docs = self.vector_store.similarity_search(search_query, k=5)
                        if docs:
                            context = "\n\n".join([doc.page_content for doc in docs])[:3000]
                            definition = await chain.ainvoke({"term": term.term, "context": context})
                            term.definition = definition.strip()
                            logger.info(f"[{i}/{len(terms_for_definition)}] Generated definition for: {term.term}")
                        else:
                            # コンテキストが見つからない場合、略語なら仮定義を設定
                            if is_abbr:
                                term.definition = f"{term.term}（専門用語の略語）"
                                logger.info(f"[{i}/{len(terms_for_definition)}] Set placeholder definition for abbreviation: {term.term}")
                    except Exception as e:
                        logger.error(f"Failed to generate definition for '{term.term}': {e}")
                        # エラーの場合も略語なら仮定義を設定
                        if self._is_abbreviation(term.term):
                            term.definition = f"{term.term}（専門用語の略語）"
                        else:
                            term.definition = ""
            except Exception as e:
                logger.error(f"Definition generation failed: {e}")
        else:
            logger.warning("RAG definition generation skipped (vector_store or llm not available)")

        # 9. 重量LLMフィルタ（定義がある用語のみ）
        if self.llm:
            logger.info("Filtering terms with LLM")
            try:
                from .prompts import get_technical_term_judgment_prompt
                from langchain_core.output_parsers import StrOutputParser
                import re

                terms_with_def = [t for t in terms if t.definition]
                if terms_with_def:
                    prompt = get_technical_term_judgment_prompt()
                    chain = prompt | self.llm | StrOutputParser()

                    batch_size = getattr(self.config, 'llm_filter_batch_size', 10)
                    technical_terms = []

                    for i in range(0, len(terms_with_def), batch_size):
                        batch = terms_with_def[i:i+batch_size]
                        batch_inputs = [{"term": t.term, "definition": t.definition} for t in batch]

                        try:
                            result_texts = await chain.abatch(batch_inputs)
                            for term, result_text in zip(batch, result_texts):
                                result = self._parse_llm_json(result_text)
                                if result and result.get("is_technical", False):
                                    term.metadata["confidence"] = result.get("confidence", 0.0)
                                    term.metadata["reason"] = result.get("reason", "")
                                    technical_terms.append(term)
                                    logger.info(f"  [OK] {term.term}: 専門用語")
                                else:
                                    logger.info(f"  [NG] {term.term}: 一般用語")
                        except Exception as e:
                            logger.error(f"LLM filter batch failed: {e}")

                    terms = technical_terms
                    logger.info(f"Filtered: {len(technical_terms)} technical terms")
                else:
                    logger.warning("No terms with definitions to filter")
            except Exception as e:
                logger.error(f"LLM filtering failed: {e}")
        else:
            logger.warning("LLM filter skipped (llm not available)")

        # 10. 辞書形式に変換して返す
        terms_list = [
            {
                "headword": term.term,
                "score": term.score,
                "definition": term.definition,
                "frequency": term.frequency,
                "synonyms": term.variants if hasattr(term, 'variants') else term.synonyms,  # 表記ゆれ
                "related_terms": term.related_terms if hasattr(term, 'related_terms') else [],  # 関連語
                "metadata": term.metadata,
                "tfidf_score": term.tfidf_score,
                "cvalue_score": term.cvalue_score
            }
            for term in terms
        ]

        # 候補用語もリスト化（HDBSCAN類義語抽出用）
        # 周辺テキストを抽出して text フィールドに追加
        candidates_list = []
        for term, freq in candidates_for_semrerank.items():
            # 用語の周辺テキストを抽出（最初の出現箇所から前後100文字）
            context = self._extract_context(term, full_text, window=100)
            candidates_list.append({
                "term": term,
                "frequency": freq,
                "tfidf_score": tfidf_scores.get(term, 0.0),
                "cvalue_score": cvalue_scores.get(term, 0.0),
                "text": context  # 周辺テキストを追加
            })

        # 辞書形式で返す（後方互換性のため、termsキーで取得可能に）
        return {
            "terms": terms_list,
            "candidates": candidates_list
        }

    def _extract_context(self, term: str, text: str, window: int = 100) -> str:
        """
        用語の周辺テキストを抽出

        Args:
            term: 抽出する用語
            text: 全体テキスト
            window: 前後の文字数

        Returns:
            "用語: 周辺テキスト" 形式の文字列
        """
        # 用語の最初の出現位置を探す
        idx = text.find(term)
        if idx == -1:
            # 見つからない場合は用語のみ返す
            return term

        # 前後window文字を抽出
        start = max(0, idx - window)
        end = min(len(text), idx + len(term) + window)
        context = text[start:end].strip()

        # "用語: 周辺テキスト" 形式で返す
        return f"{term}: {context}"

    def _is_abbreviation(self, term: str) -> bool:
        """
        略語判定ヘルパー関数

        Args:
            term: 判定する用語

        Returns:
            略語であればTrue
        """
        import re
        # 2-5文字の大文字のみ（BMS、AVR、EMS、SFOC、NOx、CO2など）
        abbreviation_pattern = re.compile(r'^[A-Z]{2,5}[0-9x]?$')
        return bool(abbreviation_pattern.match(term))

    async def _lightweight_llm_filter(self, terms: List) -> List:
        """
        軽量LLMフィルタ（定義生成前）
        明らかなゴミ（単体の一般名詞、動詞、形容詞など）を除外してコスト削減

        **重要:** 略語（2-5文字の大文字）は問答無用で通過させる

        Args:
            terms: ExtractedTermオブジェクトのリスト

        Returns:
            フィルタ通過した用語のリスト
        """
        from .prompts import get_lightweight_term_filter_prompt
        from langchain_core.output_parsers import StrOutputParser

        if not terms:
            return []

        # 略語とその他を分離
        abbreviations = []
        non_abbreviations = []

        for term in terms:
            if self._is_abbreviation(term.term):
                abbreviations.append(term)
                logger.info(f"  [AUTO-PASS] {term.term}: abbreviation (skipped lightweight filter)")
            else:
                non_abbreviations.append(term)

        logger.info(f"Auto-passed {len(abbreviations)} abbreviations, filtering {len(non_abbreviations)} non-abbreviations")

        # 略語は自動通過
        valid_terms = abbreviations.copy()

        # 非略語のみLLMフィルタ実行
        if non_abbreviations:
            prompt = get_lightweight_term_filter_prompt()
            chain = prompt | self.llm | StrOutputParser()

            batch_size = getattr(self.config, 'llm_filter_batch_size', 10)

            for i in range(0, len(non_abbreviations), batch_size):
                batch = non_abbreviations[i:i+batch_size]
                batch_inputs = [{"term": t.term} for t in batch]

                try:
                    result_texts = await chain.abatch(batch_inputs)
                    for term, result_text in zip(batch, result_texts):
                        result = self._parse_llm_json(result_text)
                        if result and result.get("is_valid", False):
                            valid_terms.append(term)
                            logger.info(f"  [PASS] {term.term}: {result.get('reason', '')}")
                        else:
                            logger.info(f"  [SKIP] {term.term}: {result.get('reason', '') if result else 'Invalid JSON'}")
                except Exception as e:
                    logger.error(f"Lightweight filter batch failed: {e}")
                    # エラー時はバッチ全体を通過させる（False Negativeを避ける）
                    valid_terms.extend(batch)

        return valid_terms

    def _split_into_sentences(self, text: str) -> List[str]:
        """文単位で分割"""
        import re
        sentences = re.split(r'[。！？\n]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _parse_llm_json(self, text: str) -> Optional[Dict]:
        """JSON応答のパース"""
        import re
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        try:
            return json.loads(text)
        except:
            try:
                match = re.search(r'\{[^{}]*\}', text)
                if match:
                    return json.loads(match.group())
            except:
                pass
        return None

    def _find_relevant_chunks(self, term: str, chunks: List[str]) -> List[str]:
        """用語を含むチャンクを検索"""
        relevant = []
        for chunk in chunks:
            if term in chunk:
                relevant.append(chunk)
                if len(relevant) >= 3:  # 最大3つのチャンク
                    break
        return relevant

    def _merge_duplicate_terms(self, terms: List[Dict]) -> List[Dict]:
        """重複する用語を統合"""
        merged = {}

        for term in terms:
            headword = term.get("headword", "")
            if not headword:
                continue

            # 既存の用語と類似度をチェック
            best_match = None
            best_score = 0

            for existing_key, existing_term in merged.items():
                score = SequenceMatcher(None, headword, existing_term["headword"]).ratio()
                if score > best_score and score > 0.8:
                    best_match = existing_key
                    best_score = score

            if best_match:
                # 既存の用語と統合
                existing = merged[best_match]
                existing["synonyms"] = list(set(
                    existing.get("synonyms", []) +
                    term.get("synonyms", []) +
                    [headword]
                ))
                # より長い定義を採用
                if len(term.get("definition", "")) > len(existing.get("definition", "")):
                    existing["definition"] = term["definition"]
            else:
                # 新規追加
                merged[headword] = term

        # 類義語リストから重複を除去
        for term in merged.values():
            if "synonyms" in term:
                term["synonyms"] = [
                    syn for syn in term["synonyms"]
                    if syn and syn != term["headword"]
                ]
                term["synonyms"] = list(set(term["synonyms"]))

        return list(merged.values())

    def save_to_database(self, terms: List[Dict]) -> int:
        """抽出した用語をデータベースに保存"""
        if not self.pg_url:
            logger.warning("No PostgreSQL connection available")
            return 0

        engine = create_engine(self.pg_url)
        saved_count = 0

        with engine.begin() as conn:
            for term in terms:
                try:
                    conn.execute(
                        text(f"""
                            INSERT INTO {self.jargon_table_name} (term, definition, aliases, related_terms)
                            VALUES (:term, :definition, :aliases, :related_terms)
                            ON CONFLICT (term) DO UPDATE
                            SET definition = EXCLUDED.definition,
                                aliases = EXCLUDED.aliases,
                                related_terms = EXCLUDED.related_terms,
                                updated_at = CURRENT_TIMESTAMP
                        """),
                        {
                            "term": term.get("headword"),
                            "definition": term.get("definition", ""),
                            "aliases": term.get("synonyms", []),
                            "related_terms": term.get("related_terms", [])
                        }
                    )
                    saved_count += 1
                except Exception as e:
                    logger.error(f"Error saving term {term.get('headword')}: {e}")

        logger.info(f"Saved {saved_count} terms to database")
        return saved_count


# ========== Utility Functions ==========
async def run_extraction_pipeline(input_dir: Path, output_json: Path, config, llm, embeddings, vector_store, pg_url, jargon_table_name, jargon_manager=None):
    """専門用語抽出パイプラインの実行"""
    # jargon_manager は将来の拡張用（現在は未使用）
    if jargon_manager is not None:
        logger.info("Using provided jargon_manager")

    extractor = TermExtractor(config, llm, embeddings, vector_store, pg_url, jargon_table_name)

    # ファイルの検索
    supported_exts = ['.txt', '.md', '.pdf']
    files = [p for ext in supported_exts for p in input_dir.glob(f"**/*{ext}")]

    if not files:
        logger.error(f"No supported files found in {input_dir}")
        return

    logger.info(f"Found {len(files)} files to process")

    # 用語抽出（新しい辞書形式で返される）
    result = await extractor.extract_from_documents(files)
    terms = result.get("terms", [])
    candidates = result.get("candidates", [])

    # JSONファイルに保存（専門用語のみ）
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({"terms": terms}, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved {len(terms)} terms to {output_json}")

    # デバッグファイルに候補用語を保存（HDBSCAN用）
    if candidates:
        debug_file = output_json.parent / "term_extraction_debug.json"
        with open(debug_file, "w", encoding="utf-8") as f:
            json.dump({"candidates": candidates}, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(candidates)} candidate terms to {debug_file}")

    # PostgreSQLに保存
    if terms:
        extractor.save_to_database(terms)

        # HDBSCAN意味ベース類義語抽出
        logger.info("Starting semantic synonym extraction with HDBSCAN...")
        try:
            from ..scripts.extract_semantic_synonyms import (
                load_specialized_terms,
                load_candidate_terms_from_extraction,
                extract_and_save_semantic_synonyms
            )

            # 候補用語の読み込み（デバッグファイルから）
            debug_file = output_json.parent / "term_extraction_debug.json"
            if debug_file.exists():
                # 専門用語と候補用語を読み込み
                specialized_terms = load_specialized_terms(pg_url, jargon_table_name)
                candidate_terms = load_candidate_terms_from_extraction(debug_file)

                if specialized_terms and candidate_terms:
                    # 類義語抽出と保存
                    synonyms_dict = await extract_and_save_semantic_synonyms(
                        specialized_terms=specialized_terms,
                        candidate_terms=candidate_terms,
                        pg_url=pg_url,
                        jargon_table_name=jargon_table_name,
                        embeddings=embeddings
                    )
                    logger.info(f"Extracted semantic synonyms for {len(synonyms_dict)} terms")
                else:
                    logger.warning("No specialized or candidate terms found for semantic synonym extraction")
            else:
                logger.warning(f"Debug file not found: {debug_file}")
        except Exception as e:
            logger.error(f"Error in semantic synonym extraction: {e}", exc_info=True)


__all__ = [
    "JargonDictionaryManager",
    "TermExtractor",
    "run_extraction_pipeline"
]
