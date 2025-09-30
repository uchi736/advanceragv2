"""
PDF処理プロセッサの基底クラス
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List


class BasePDFProcessor(ABC):
    """
    PDF処理プロセッサの抽象基底クラス
    
    すべてのPDF処理実装はこのインターフェースを実装する必要があります。
    """
    
    def __init__(self, config: Any, image_output_dir: str = "output/images"):
        """
        プロセッサの初期化
        
        Args:
            config: 設定オブジェクト
            image_output_dir: 画像出力ディレクトリ
        """
        self.config = config
        self.image_output_dir = image_output_dir
    
    @abstractmethod
    def parse_pdf(self, file_path: str) -> Dict[str, List[Any]]:
        """
        PDFファイルを解析してコンテンツを抽出
        
        Args:
            file_path: 処理するPDFファイルのパス
            
        Returns:
            抽出された要素を含む辞書:
                - "texts": テキスト要素のリスト [(content, metadata), ...]
                - "images": 画像要素のリスト [(path, metadata), ...]
                - "tables": テーブル要素のリスト [(data, metadata), ...]
        """
        pass
    
    @abstractmethod
    def summarize_image(self, image_path: str) -> str:
        """
        画像の内容を要約
        
        Args:
            image_path: 画像ファイルのパス
            
        Returns:
            画像の要約テキスト
        """
        pass
    
    @abstractmethod
    def format_table_as_markdown(self, table_data: Any) -> str:
        """
        テーブルデータをMarkdown形式に変換
        
        Args:
            table_data: テーブルデータ
            
        Returns:
            Markdown形式のテーブル文字列
        """
        pass