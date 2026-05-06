from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    OPENAI_API_KEY: str
    TTC_API_KEY: str
    NEBULA_API_KEY: str

    HF_REVISION: str | None = None
    TTC_RPM: float = 10.0
    NEBULA_RPM: float = 30.0
    NEBULA_BASE_URL: str = "https://api.trynebula.ai"
    OPENAI_CONCURRENCY: int = 32
    HTTP_TIMEOUT_S: int = 120

    READER_MODEL: str = "gpt-4o-mini-2024-07-18"
    JUDGE_MODEL: str = "gpt-4o-2024-08-06"
    EMBED_MODEL: str = "text-embedding-3-small"
    TTC_MODEL: str = "bear-1.2"

    READER_TEMPERATURE: float = 0.7
    READER_SEED: int = 42

    DATA_DIR: Path = Field(default=ROOT / "data")
    CACHE_DIR: Path = Field(default=ROOT / "cache")
    OUTPUT_DIR: Path = Field(default=ROOT / "outputs")

    @property
    def hf_cache_dir(self) -> Path:
        return self.DATA_DIR / "hf"

    @property
    def subset_path(self) -> Path:
        return self.DATA_DIR / "subset_v1.jsonl"

    @property
    def revision_path(self) -> Path:
        return self.DATA_DIR / "dataset_revision.txt"

    @property
    def embeddings_db(self) -> Path:
        return self.CACHE_DIR / "embeddings.sqlite"

    @property
    def ttc_db(self) -> Path:
        return self.CACHE_DIR / "ttc.sqlite"

    @property
    def nebula_db(self) -> Path:
        return self.CACHE_DIR / "nebula.sqlite"

    @property
    def runs_jsonl(self) -> Path:
        return self.OUTPUT_DIR / "runs.jsonl"

    @property
    def runs_parquet(self) -> Path:
        return self.OUTPUT_DIR / "runs.parquet"

    def ensure_dirs(self) -> None:
        for d in (self.DATA_DIR, self.hf_cache_dir, self.CACHE_DIR, self.OUTPUT_DIR):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
