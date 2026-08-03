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
        )

    def get_docs_dir(self) -> Path:
        value = os.environ.get("RAG_DOCS_DIR")
        if not value:
            raise ConfigError(
                "RAG_DOCS_DIR não está definida. Configure o caminho da pasta "
                "com seus documentos, "
                "ex: RAG_DOCS_DIR=C:\\Users\\voce\\Documents\\meus-pdfs"
            )
        path = Path(value)
        if not path.is_dir():
            raise ConfigError(
                f"RAG_DOCS_DIR aponta para um caminho que não existe ou não é uma pasta: {path}"
            )
        return path

    def get_embed_model(self):
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        return HuggingFaceEmbedding(model_name=self.embedding_model)