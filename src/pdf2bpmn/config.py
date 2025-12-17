"""Configuration management."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()


class Config:
    """Application configuration."""
    
    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    
    # Neo4j
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "1234567bpmn")
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent.parent
    OUTPUT_DIR: Path = BASE_DIR / "output"
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    TEMPLATES_DIR: Path = Path(__file__).parent / "templates"
    
    # Processing
    CONFIDENCE_THRESHOLD: float = 0.8
    SIMILARITY_MERGE_THRESHOLD: float = 0.90
    SIMILARITY_REVIEW_THRESHOLD: float = 0.80
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Performance optimization options
    EVIDENCE_MODE: str = os.getenv("EVIDENCE_MODE", "full")  # "full", "reference_only", "off"
    CHUNKING_STRATEGY: str = os.getenv("CHUNKING_STRATEGY", "fixed")  # "fixed", "semantic"
    
    @classmethod
    def ensure_dirs(cls):
        """Ensure output directories exist."""
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# Initialize directories
Config.ensure_dirs()




