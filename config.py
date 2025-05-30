import os
from typing import Optional
from pydantic import BaseSettings

class Settings(BaseSettings):
    # Elasticsearch Configuration
    elasticsearch_host: str = os.getenv("ELASTICSEARCH_HOST", "localhost")
    elasticsearch_port: int = int(os.getenv("ELASTICSEARCH_PORT", "9200"))
    elasticsearch_scheme: str = os.getenv("ELASTICSEARCH_SCHEME", "http")
    elasticsearch_username: Optional[str] = os.getenv("ELASTICSEARCH_USERNAME")
    elasticsearch_password: Optional[str] = os.getenv("ELASTICSEARCH_PASSWORD")
    
    # API Configuration
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # Embedding Configuration
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "384"))
    
    # Search Configuration
    default_max_results: int = int(os.getenv("DEFAULT_MAX_RESULTS", "10"))
    default_min_score: float = float(os.getenv("DEFAULT_MIN_SCORE", "0.7"))
    default_max_context: int = int(os.getenv("DEFAULT_MAX_CONTEXT", "3"))
    
    # Index Configuration
    documents_index: str = "vector_rag_documents"
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    @property
    def elasticsearch_url(self) -> str:
        """Construct Elasticsearch URL"""
        if self.elasticsearch_username and self.elasticsearch_password:
            return f"{self.elasticsearch_scheme}://{self.elasticsearch_username}:{self.elasticsearch_password}@{self.elasticsearch_host}:{self.elasticsearch_port}"
        return f"{self.elasticsearch_scheme}://{self.elasticsearch_host}:{self.elasticsearch_port}"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()