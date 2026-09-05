# src/rag_study/ingest.py
from pathlib import Path

from llama_index.core.embeddings import BaseEmbedding

from rag_study.chunking import split_into_nodes
from rag_study.config import Config
from rag_study.indexing import build_index
from rag_study.loader import load_documents


def run_ingestion(
    docs_dir: Path,
    storage_dir: Path,
    embed_model: BaseEmbedding,
    chunk_size: int,
    chunk_overlap: int,
    collection_name: str,
) -> int:
    documents = load_documents(docs_dir)
    if not documents:
        print(f"Nenhum documento legível encontrado em '{docs_dir}'.")
        return 0

    nodes = split_into_nodes(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    build_index(nodes, storage_dir, embed_model, collection_name)

    print(
        f"{len(documents)} documento(s) indexado(s) em {len(nodes)} chunk(s). "
        f"Índice salvo em '{storage_dir}'."
    )
    return len(nodes)


def main() -> None:
    cfg = Config.from_env()
    docs_dir = cfg.get_docs_dir()
    run_ingestion(
        docs_dir=docs_dir,
        storage_dir=cfg.storage_dir,
        embed_model=cfg.get_embed_model(),
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        collection_name=cfg.collection_name,
    )


if __name__ == "__main__":
    main()
