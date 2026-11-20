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

        doc_dir_env = os.environ.get("RAG_DOCS_DIR")
        chunk_size_env = os.environ.get("RAG_CHUNK_SIZE")
        chunk_overlap_env = os.environ.get("RAG_CHUNK_OVERLAP")
        embedding_model_env = os.environ.get("RAG_EMBEDDING_MODEL")

        locked = set()

        if doc_dir_env:
            doc_dir = doc_dir_env
            locked.add("docs_dir")
        elif "docs_dir" in loaded_config_file:
            doc_dir = loaded_config_file["docs_dir"]
        else:
            doc_dir = None

        if chunk_overlap_env:
            chunk_overlap = int(chunk_overlap_env)
            locked.add("chunk_overlap")
        elif "chunk_overlap" in loaded_config_file:
            chunk_overlap = loaded_config_file["chunk_overlap"]
        else:
            chunk_overlap = 50

        if chunk_size_env:
            chunk_size = int(chunk_size_env)
            locked.add("chunk_size")
        elif "chunk_size" in loaded_config_file:
            chunk_size = loaded_config_file["chunk_size"]
        else:
            chunk_size = 512

        if embedding_model_env:
            embedding_model = embedding_model_env
            locked.add("embedding_model")
        elif "embedding_model" in loaded_config_file:
            embedding_model = loaded_config_file["embedding_model"]
        else:
            embedding_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

        return cls(
            docs_dir_raw=doc_dir,
            storage_dir=Path(os.environ.get("RAG_STORAGE_DIR", "storage")),
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            log_level=os.environ.get("RAG_LOG_LEVEL", "INFO"),
            active_profile=loaded_config_file.get("active_profile"),
            ocr_min_chars=loaded_config_file.get("ocr_min_chars", 50),
            env_locked=frozenset(locked),
        )

    @staticmethod
    def config_file_path() -> Path:
        env_value = os.environ.get("RAG_CONFIG_FILE")

        if env_value:
            return Path(env_value)

        return Path("rag-study.config.json")
