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
        value = os.environ.get("RAG_DOCS_DIR")
        if not value:
            raise ConfigError(
                "RAG_DOCS_DIR is not set. Set it to the path of the folder "
                "containing your documents, "
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