import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    # Placeholder values - will be populated in __post_init__
    db_host: str = "localhost"
    db_port: str = "5432"
    db_name: str = "postgres"
    db_user: str = "postgres"
    db_password: str = "your-password"
    pgvector_connection_string: str = ""

    openai_api_key: Optional[str] = None
    embedding_model_identifier: str = "text-embedding-3-small"
    llm_model_identifier: str = "gpt-4.1-mini"

    azure_openai_api_key: Optional[str] = None
    azure_openai_endpoint: Optional[str] = None
    azure_openai_api_version: str = "2024-02-01"
    azure_openai_chat_deployment_name: Optional[str] = None
    azure_openai_embedding_deployment_name: Optional[str] = None

    llm_temperature: float = 0.0
    max_tokens: int = 4096

    def __post_init__(self):
        """Load environment variables dynamically after load_dotenv() has been called"""
        # Database settings
        self.db_host = os.getenv("DB_HOST", "localhost")
        self.db_port = os.getenv("DB_PORT", "5432")
        self.db_name = os.getenv("DB_NAME", "postgres")
        self.db_user = os.getenv("DB_USER", "postgres")
        self.db_password = os.getenv("DB_PASSWORD", "your-password")

        # Build PGVector connection string
        try:
            import psycopg
            _PG_DIALECT = "psycopg"
        except ModuleNotFoundError:
            _PG_DIALECT = "psycopg2"

        self.pgvector_connection_string = (
            f"postgresql+{_PG_DIALECT}://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

        # OpenAI settings
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.embedding_model_identifier = os.getenv("EMBEDDING_MODEL_IDENTIFIER", "text-embedding-3-small")
        self.llm_model_identifier = os.getenv("LLM_MODEL_IDENTIFIER", "gpt-4.1-mini")

        # Azure OpenAI settings
        self.azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
        self.azure_openai_chat_deployment_name = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
        self.azure_openai_embedding_deployment_name = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")

        # LLM settings
        self.llm_temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))
        self.max_tokens = int(os.getenv("MAX_TOKENS", "4096"))

        # Azure Document Intelligence settings
        self.azure_di_endpoint = os.getenv("AZURE_DI_ENDPOINT")
        self.azure_di_api_key = os.getenv("AZURE_DI_API_KEY")
        self.azure_di_model = os.getenv("AZURE_DI_MODEL", "prebuilt-layout")
        self.save_markdown = os.getenv("SAVE_MARKDOWN", "false").lower() == "true"
        self.markdown_output_dir = os.getenv("MARKDOWN_OUTPUT_DIR", "output/markdown")

    # RAG and Search settings
    chunk_size: int = 1000
    chunk_overlap: int = 200
    vector_search_k: int = 15
    keyword_search_k: int = 15
    final_k: int = 15
    collection_name: str = "documents"
    fts_language: str = "english"
    rrf_k_for_fusion: int = 60
    distance_strategy: str = "COSINE"
    vector_store_type: str = "pgvector"

    # Japanese search settings
    enable_japanese_search: bool = True
    japanese_min_token_length: int = 2

    # Jargon dictionary settings
    enable_jargon_extraction: bool = True
    enable_jargon_augmentation: bool = True
    jargon_table_name: str = "jargon_dictionary"
    max_jargon_terms_per_query: int = 5

    # Document processing settings
    enable_doc_summarization: bool = True
    enable_metadata_enrichment: bool = True
    confidence_threshold: float = 0.2

    # Azure Document Intelligence settings
    azure_di_endpoint: Optional[str] = None
    azure_di_api_key: Optional[str] = None
    azure_di_model: str = "prebuilt-layout"
    save_markdown: bool = False
    markdown_output_dir: str = "output/markdown"

    # Term extraction settings (SemReRank)
    use_advanced_extraction: bool = True
    semrerank_enabled: bool = True
    semrerank_seed_percentile: float = 15.0
    semrerank_relmin: float = 0.5
    semrerank_reltop: float = 0.15
    semrerank_alpha: float = 0.85
    definition_generation_percentile: float = 50.0
    enable_lightweight_filter: bool = True
    llm_filter_batch_size: int = 10
