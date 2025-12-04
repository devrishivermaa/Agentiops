"""
Centralized configuration management for AgentOps.
"""

import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()


class Config:
    """Application configuration."""
    
    # LLM Settings
    MISTRAL_API_KEY: str = os.getenv("MISTRAL_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "mistral-small-latest")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "5"))
    
    # Rate Limiting
    MAX_REQUESTS_PER_MINUTE: int = int(os.getenv("MAX_REQUESTS_PER_MINUTE", "50"))
    MAX_REQUESTS_PER_DAY: int = int(os.getenv("MAX_REQUESTS_PER_DAY", "5000"))
    
    # MongoDB
    MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    MONGO_DB: str = os.getenv("MONGO_DB", "agentops")
    MONGO_MASTER_COLLECTION: str = os.getenv("MONGO_MASTER_COLLECTION", "master_agent")
    MONGO_METADATA_COLLECTION: str = os.getenv("MONGO_METADATA_COLLECTION", "metadata")
    
    # Processing
    MAX_PARALLEL_SUBMASTERS: int = int(os.getenv("MAX_PARALLEL_SUBMASTERS", "3"))
    NUM_WORKERS_PER_SUBMASTER: int = int(os.getenv("NUM_WORKERS_PER_SUBMASTER", "4"))
    FEEDBACK_REQUIRED: bool = os.getenv("FEEDBACK_REQUIRED", "True").lower() == "true"
    
    # Ray
    RAY_NUM_CPUS: int = int(os.getenv("RAY_NUM_CPUS", "4"))
    RAY_INCLUDE_DASHBOARD: bool = os.getenv("RAY_INCLUDE_DASHBOARD", "False").lower() == "true"
    
    # Vector DB
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "agentops_documents")
    
    # Paths
    DATA_DIR: str = os.getenv("DATA_DIR", "./data")
    OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "./output")
    LOGS_DIR: str = os.getenv("LOGS_DIR", "./logs")
    
    # API
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_RELOAD: bool = os.getenv("API_RELOAD", "True").lower() == "true"
    
    # Feature Flags
    ENABLE_VECTOR_DB: bool = os.getenv("ENABLE_VECTOR_DB", "True").lower() == "true"
    ENABLE_RESIDUAL_AGENT: bool = os.getenv("ENABLE_RESIDUAL_AGENT", "True").lower() == "true"
    ENABLE_LLM_MERGE: bool = os.getenv("ENABLE_LLM_MERGE", "True").lower() == "true"
    
    @classmethod
    def validate(cls):
        """Validate critical configuration."""
        errors = []
        
        if not cls.MISTRAL_API_KEY:
            errors.append("MISTRAL_API_KEY is required")
        
        if cls.MAX_PARALLEL_SUBMASTERS < 1:
            errors.append("MAX_PARALLEL_SUBMASTERS must be >= 1")
        
        if cls.NUM_WORKERS_PER_SUBMASTER < 1:
            errors.append("NUM_WORKERS_PER_SUBMASTER must be >= 1")
        
        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")
    
    @classmethod
    def get_user_config(cls) -> dict:
        """Get user configuration for pipeline."""
        return {
            "document_type": "research_paper",
            "processing_requirements": [
                "summary_generation",
                "entity_extraction",
                "keyword_indexing"
            ],
            "preferred_model": cls.LLM_MODEL,
            "max_parallel_submasters": cls.MAX_PARALLEL_SUBMASTERS,
            "num_workers_per_submaster": cls.NUM_WORKERS_PER_SUBMASTER,
            "feedback_required": cls.FEEDBACK_REQUIRED,
            "complexity_level": "high"
        }


# Validate on import
Config.validate()
