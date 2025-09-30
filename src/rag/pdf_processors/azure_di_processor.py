"""
Azure Document Intelligence を使用したPDF処理プロセッサ
"""

import os
import logging
from typing import List, Dict, Any
from pathlib import Path
import base64
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI

from .base_processor import BasePDFProcessor

logger = logging.getLogger(__name__)


class AzureDocumentIntelligenceProcessor(BasePDFProcessor):
    """
    Azure Document Intelligence を使用したPDF処理クラス
    
    Layout モデルを使用してPDFをMarkdown形式に変換し、
    高精度な文書構造の抽出を行います。
    """
    
    def __init__(self, config: Any, image_output_dir: str = "output/images"):
        """
        Azure Document Intelligence プロセッサの初期化
        
        Args:
            config: 設定オブジェクト
            image_output_dir: 画像出力ディレクトリ
        """
        super().__init__(config, image_output_dir)
        
        # Azure Document Intelligence設定の確認
        self.endpoint = getattr(config, 'azure_di_endpoint', None)
        self.api_key = getattr(config, 'azure_di_api_key', None)
        self.model = getattr(config, 'azure_di_model', 'prebuilt-layout')
        
        if not self.endpoint or not self.api_key:
            raise ValueError(
                "Azure Document Intelligence requires 'azure_di_endpoint' and 'azure_di_api_key' in config"
            )
        
        # Azure Document Intelligenceクライアントの初期化（遅延インポート）
        try:
            from azure.core.credentials import AzureKeyCredential
            from azure.ai.documentintelligence import DocumentIntelligenceClient
            
            self.client = DocumentIntelligenceClient(
                endpoint=self.endpoint,
                credential=AzureKeyCredential(self.api_key)
            )
        except ImportError:
            raise ImportError(
                "Azure Document Intelligence library not installed. "
                "Please run: pip install azure-ai-documentintelligence"
            )
        
        # 画像要約用のLLM初期化
        if all([config.azure_openai_api_key, config.azure_openai_endpoint, config.azure_openai_chat_deployment_name]):
            self.llm = AzureChatOpenAI(
                azure_endpoint=config.azure_openai_endpoint,
                api_key=config.azure_openai_api_key,
                api_version=config.azure_openai_api_version,
                azure_deployment=config.azure_openai_chat_deployment_name,
                temperature=0.1,
                max_tokens=512
            )
        else:
            self.llm = None
            logger.warning("Azure OpenAI not configured. Image summarization will be disabled.")
        
        if not os.path.exists(self.image_output_dir):
            os.makedirs(self.image_output_dir)
        
        logger.info(f"Azure Document Intelligence processor initialized with model: {self.model}")
    
    def parse_pdf(self, file_path: str) -> Dict[str, List[Any]]:
        """
        PDFファイルを処理してMarkdown形式のテキストを抽出
        
        Args:
            file_path: 処理するPDFファイルのパス
            
        Returns:
            抽出された要素を含む辞書:
                - "texts": Markdown形式のテキストとメタデータ
                - "images": 抽出された画像情報
                - "tables": 抽出されたテーブル情報
        """
        try:
            from azure.ai.documentintelligence.models import (
                AnalyzeDocumentRequest,
                ContentFormat,
                AnalyzeResult,
            )
            
            logger.info(f"Processing PDF with Azure Document Intelligence: {file_path}")
            
            # ファイルを読み込み
            with open(file_path, "rb") as f:
                file_content = f.read()
            
            # Azure Document Intelligence で解析
            poller = self.client.begin_analyze_document(
                self.model,
                AnalyzeDocumentRequest(bytes_source=file_content),
                output_content_format=ContentFormat.MARKDOWN,
            )
            
            result: AnalyzeResult = poller.result()
            
            extracted_elements = {
                "texts": [],
                "images": [],
                "tables": []
            }
            
            # Markdown形式のコンテンツを取得
            if result.content:
                metadata = {
                    "source": file_path,
                    "type": "markdown",
                    "processor": "azure_document_intelligence",
                    "model": self.model,
                    "content_format": result.content_format
                }
                extracted_elements["texts"].append((result.content, metadata))
            
            # ページ情報を処理
            if hasattr(result, 'pages') and result.pages:
                for page_num, page in enumerate(result.pages, 1):
                    # ページのテキストを追加（必要に応じて）
                    if hasattr(page, 'lines'):
                        page_text = "\n".join([line.content for line in page.lines if hasattr(line, 'content')])
                        if page_text:
                            page_metadata = {
                                "source": file_path,
                                "page_number": page_num,
                                "type": "text",
                                "processor": "azure_document_intelligence"
                            }
                            extracted_elements["texts"].append((page_text, page_metadata))
            
            # テーブル情報を抽出
            if hasattr(result, 'tables') and result.tables:
                for table_idx, table in enumerate(result.tables):
                    table_data = self._extract_table_data(table)
                    metadata = {
                        "source": file_path,
                        "type": "table",
                        "table_number": table_idx,
                        "processor": "azure_document_intelligence",
                        "row_count": table.row_count,
                        "column_count": table.column_count
                    }
                    extracted_elements["tables"].append((table_data, metadata))
            
            # Markdownファイルとして保存（オプション）
            if getattr(self.config, 'save_markdown', False):
                output_path = self._save_markdown(file_path, result.content)
                logger.info(f"Markdown saved to: {output_path}")
            
            logger.info(f"Successfully processed PDF: {len(result.content)} characters extracted")
            
            return extracted_elements
            
        except Exception as e:
            logger.error(f"Error processing PDF with Azure Document Intelligence: {e}")
            raise
    
    def _extract_table_data(self, table) -> List[List[str]]:
        """
        Azure Document Intelligenceのテーブルオブジェクトからデータを抽出
        
        Args:
            table: Azure Document Intelligenceのテーブルオブジェクト
            
        Returns:
            テーブルデータ（行と列のリスト）
        """
        if not hasattr(table, 'cells'):
            return []
        
        # テーブルの行列を初期化
        rows = [['' for _ in range(table.column_count)] for _ in range(table.row_count)]
        
        # セルの内容を配置
        for cell in table.cells:
            row_idx = cell.row_index
            col_idx = cell.column_index
            content = cell.content if hasattr(cell, 'content') else ''
            
            # セルの範囲を考慮
            row_span = cell.row_span if hasattr(cell, 'row_span') else 1
            col_span = cell.column_span if hasattr(cell, 'column_span') else 1
            
            # セルの内容を適切な位置に配置
            for r in range(row_span):
                for c in range(col_span):
                    if row_idx + r < table.row_count and col_idx + c < table.column_count:
                        rows[row_idx + r][col_idx + c] = content
        
        return rows
    
    def summarize_image(self, image_path: str) -> str:
        """
        画像の内容を要約（Azure OpenAIを使用）
        
        Args:
            image_path: 画像ファイルのパス
            
        Returns:
            画像の要約テキスト
        """
        if not self.llm:
            return "画像要約機能が利用できません（Azure OpenAI未設定）"
        
        try:
            with open(image_path, "rb") as image_file:
                image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

            message = HumanMessage(
                content=[
                    {"type": "text", "text": "この画像について、内容を詳細に説明してください。グラフであれば、その傾向や読み取れる重要な数値を具体的に記述してください。図であれば、その構造や要素間の関係性を説明してください。"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}",
                            "detail": "high"
                        }
                    },
                ]
            )
            
            response = self.llm.invoke([message])
            return response.content
        except Exception as e:
            logger.error(f"Error summarizing image {image_path}: {e}")
            return "画像の内容を要約できませんでした。"
    
    def format_table_as_markdown(self, table_data: List[List[str]]) -> str:
        """
        テーブルデータをMarkdown形式に変換
        
        Args:
            table_data: テーブルデータ（行と列のリスト）
            
        Returns:
            Markdown形式のテーブル文字列
        """
        if not table_data:
            return ""

        # セルの内容をクリーンアップ
        def clean_cell(cell):
            if cell is None:
                return ""
            return str(cell).replace("\n", " ").strip()

        header = "| " + " | ".join(map(clean_cell, table_data[0])) + " |"
        separator = "| " + " | ".join(["---"] * len(table_data[0])) + " |"
        body = "\n".join([
            "| " + " | ".join(map(clean_cell, row)) + " |"
            for row in table_data[1:]
        ])
        
        return f"{header}\n{separator}\n{body}"
    
    def _save_markdown(self, original_path: str, content: str) -> Path:
        """
        Markdown形式のコンテンツをファイルに保存
        
        Args:
            original_path: 元のPDFファイルパス
            content: Markdown形式のコンテンツ
            
        Returns:
            保存したファイルのパス
        """
        # 出力ディレクトリの作成
        output_dir = Path(getattr(self.config, 'markdown_output_dir', 'output/markdown'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # ファイル名の生成
        original_name = Path(original_path).stem
        output_path = output_dir / f"{original_name}_azure_di.md"
        
        # ファイルに書き込み
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"# Document processed by Azure Document Intelligence\n")
            f.write(f"## Source: {original_path}\n")
            f.write(f"## Model: {self.model}\n\n")
            f.write(content)
        
        return output_path