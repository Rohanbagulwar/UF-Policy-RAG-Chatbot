import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""
    # Hugging face embedding: https://huggingface.co/BAAI/bge-small-en
    model_name: str = "BAAI/bge-base-en-v1.5"
    dimension: Optional[int] = None


@dataclass
class ChromaDBConfig:
    """Configuration for ChromaDB."""
    persist_directory: str = str(PROJECT_ROOT / "data" / "chroma_db")
    collection_name: str = "policies"
    description: str = "UF Policy documents with embeddings"


def _get_openai_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY in the environment before starting the application.")
    return api_key


# models=['llama-3.1-70b-instruct','gemma-3-27b-it']
@dataclass
class OpenAIConfig:
    """Configuration for OpenAI client."""
    api_key: str = field(default_factory=_get_openai_api_key)
    base_url: str = "https://api.ai.it.ufl.edu"
    model: str = "gpt-oss-20b"
    temperature: float = 0.1
    max_tokens: int = 1000


@dataclass
class QueryConfig:
    """Configuration for query operations."""
    n_results: int = 5
    filter_by_type: Optional[str] = None
