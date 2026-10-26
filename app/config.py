import os
from functools import lru_cache
from pathlib import Path
from dotenv import load_dotenv
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Ensure .env is explicitly loaded into os.environ
load_dotenv(override=True)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Base Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = Path("./data")

    # Storage settings
    DATABASE_URL: str = ""
    FAISS_INDEX_PATH: str = ""

    # Embedding settings
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384

    # LLM Settings
    LLM_PROVIDER: str = "groq"  # "groq" or "gemini"
    GROQ_API_KEY: str = Field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    GROQ_MODEL: str = "openai/gpt-oss-120b"
    GROQ_FALLBACK_MODEL: str = "openai/gpt-oss-20b"
    GEMINI_API_KEY: str = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    GEMINI_MODEL: str = "gemini-2.5-flash"

    # Adaptive retrieval weights (must sum to 1.0)
    WEIGHT_SIMILARITY: float = 0.45
    WEIGHT_IMPORTANCE: float = 0.20
    WEIGHT_RECENCY: float = 0.20
    WEIGHT_CONFIDENCE: float = 0.15

    # Memory decay & conflict thresholds
    RECENCY_DECAY_LAMBDA: float = 0.015  # exponential decay rate per hour
    CONFLICT_SIMILARITY_THRESHOLD: float = 0.35

    @model_validator(mode="after")
    def assemble_paths_and_validate(self) -> "Settings":
        # Resolve data directory relative to base dir if relative
        if not self.DATA_DIR.is_absolute():
            resolved_data_dir = (self.BASE_DIR / self.DATA_DIR).resolve()
        else:
            resolved_data_dir = self.DATA_DIR.resolve()

        resolved_data_dir.mkdir(parents=True, exist_ok=True)
        self.DATA_DIR = resolved_data_dir

        if not self.DATABASE_URL:
            db_path = (self.DATA_DIR / "memory_system.db").resolve()
            # Normalize path for SQLite URL
            self.DATABASE_URL = f"sqlite:///{db_path.as_posix()}"

        if not self.FAISS_INDEX_PATH:
            index_path = (self.DATA_DIR / "faiss_memories.index").resolve()
            self.FAISS_INDEX_PATH = str(index_path.as_posix())

        # Validate weights sum
        weights_sum = (
            self.WEIGHT_SIMILARITY
            + self.WEIGHT_IMPORTANCE
            + self.WEIGHT_RECENCY
            + self.WEIGHT_CONFIDENCE
        )
        if abs(weights_sum - 1.0) > 1e-4:
            raise ValueError(
                f"Adaptive retrieval weights must sum to 1.0, got {weights_sum:.4f}"
            )

        return self


@lru_cache()
def get_settings() -> Settings:
    """Return cached Settings instance."""
    return Settings()


settings = get_settings()
