from dataclasses import dataclass
import os
from pathlib import Path

class ConfigError(Exception):
    """Raised when required project configuration is missing or invalid."""

@dataclass(frozen=True)
class Config:
    docs_dir_raw: str | None
    storage_dir: Path
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    collection_name: str = "rag_study_docs"
    log_level: str = "INFO"
    active_profile: str | None = None
    ocr_min_chars: int = 50
    env_locked: frozenset[str] = frozenset()

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            docs_dir_raw=os.environ.get("RAG_DOCS_DIR"),
            storage_dir=Path(os.environ.get("RAG_STORAGE_DIR", "storage")),
            embedding_model=os.environ.get(
                "RAG_EMBEDDING_MODEL",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            ),
            chunk_size=int(os.environ.get("RAG_CHUNK_SIZE", "512")),
            chunk_overlap=int(os.environ.get("RAG_CHUNK_OVERLAP", "50")),
            log_level=os.environ.get("RAG_LOG_LEVEL", "INFO")
        )

    def get_docs_dir(self) -> Path:
        value = self.docs_dir_raw
        if not value:
            raise ConfigError(
                "No documents folder configured. Set the RAG_DOCS_DIR "
                'environment variable, or add "docs_dir": "<path>" to '
                "rag-study.config.json, "
                "e.g. RAG_DOCS_DIR=C:\\Users\\you\\Documents\\my-pdfs"
            )
        path = Path(value)
        if not path.is_dir():
            raise ConfigError(
                f"RAG_DOCS_DIR points to a path that does not exist or is not a folder: {path}"
            )
        return path

    def get_embed_model(self):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        return HuggingFaceEmbedding(model_name=self.embedding_model)

    @classmethod
    def load(cls, config_file: Path | None = None) -> "Config":
        from rag_study.settings_file import load_settings

        if config_file is None:
            config_file = cls.config_file_path()
        loaded_config_file = load_settings(config_file)
        locked: set[str] = set()

        def resolve(key: str, default, env_name: str | None = None, cast=lambda v: v):
            if env_name:
                env_value = os.environ.get(env_name)
                if env_value:
                    locked.add(key)
                    return cast(env_value)
            return loaded_config_file.get(key, default)

        doc_dir = resolve("docs_dir", None, env_name="RAG_DOCS_DIR")
        chunk_size = resolve("chunk_size", 512, env_name="RAG_CHUNK_SIZE", cast=int)
        chunk_overlap = resolve("chunk_overlap", 50, env_name="RAG_CHUNK_OVERLAP", cast=int)
        embedding_model = resolve(
            "embedding_model",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            env_name="RAG_EMBEDDING_MODEL",
        )
        active_profile = resolve("active_profile", None)
        ocr_min_chars = resolve("ocr_min_chars", 50)

        return cls(
            docs_dir_raw=doc_dir,
            storage_dir=Path(os.environ.get("RAG_STORAGE_DIR", "storage")),
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            log_level=os.environ.get("RAG_LOG_LEVEL", "INFO"),
            active_profile=active_profile,
            ocr_min_chars=ocr_min_chars,
            env_locked=frozenset(locked),
        )

    @staticmethod
    def config_file_path() -> Path:
        env_value = os.environ.get("RAG_CONFIG_FILE")

        if env_value:
            return Path(env_value)

        return Path("rag-study.config.json")
