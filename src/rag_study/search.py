from pathlib import Path
from llama_index.core.embeddings import BaseEmbedding
from rag_study.indexing import load_index

class IndexNotFoundError(Exception):
    """Raised when a search is attempted before any index has been built."""

def search_documents(
    query: str,
    top_k: int,
    storage_dir: Path,
    embed_model: BaseEmbedding,
    collection_name: str,
) -> list[dict]:
    index = load_index(storage_dir, embed_model, collection_name)
    if index is None:
        raise IndexNotFoundError(
            "Nenhum índice encontrado. Rode 'uv run rag-ingest' para indexar seus "
            "documentos antes de consultar."
        )
    retriever = index.as_retriever(similarity_top_k=top_k)
    scored_nodes = retriever.retrieve(query)

    results = []
    for scored_node in scored_nodes:
        node = scored_node.node
        results.append(
            {
                "text": node.get_content(),
                "source": node.metadata.get("file_name", "desconhecido"),
                "score": scored_node.score if scored_node.score is not None else 0.0,
            }
        )
    return results