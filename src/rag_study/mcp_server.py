# src/rag_study/mcp_server.py
import sys
from pathlib import Path

from fastmcp import FastMCP
from llama_index.core.embeddings import BaseEmbedding

from rag_study.config import Config
from rag_study.search import IndexNotFoundError, search_documents

mcp = FastMCP("rag-study")

_embed_model_cache: BaseEmbedding | None = None


def _get_embed_model(cfg: Config) -> BaseEmbedding:
    """Loads the embedding model once per server process and reuses it across searches."""
    global _embed_model_cache
    if _embed_model_cache is None:
        _embed_model_cache = cfg.get_embed_model()
    return _embed_model_cache


def format_results(question: str, results: list[dict]) -> str:
    if not results:
        return f"No relevant passages found for: '{question}'."

    parts = []
    for i, result in enumerate(results, start=1):
        parts.append(f"[{i}] Source: {result['source']}\n{result['text']}")
    return "\n\n".join(parts)


def search_docs_core(
    question: str,
    top_k: int,
    storage_dir: Path,
    embed_model: BaseEmbedding,
    collection_name: str,
) -> str:
    try:
        results = search_documents(
            query=question,
            top_k=top_k,
            storage_dir=storage_dir,
            embed_model=embed_model,
            collection_name=collection_name,
        )
    except IndexNotFoundError as exc:
        return str(exc)
    return format_results(question, results)


@mcp.tool()
def search_docs(question: str, top_k: int = 5) -> str:
    """Searches the indexed personal documents for passages relevant to the question."""
    cfg = Config.from_env()
    return search_docs_core(
        question=question,
        top_k=top_k,
        storage_dir=cfg.storage_dir,
        embed_model=_get_embed_model(cfg),
        collection_name=cfg.collection_name,
    )


def main() -> None:
    # Load the embedding model on the main thread BEFORE serving. Besides making the
    # first search instant, this avoids a deadlock on Windows: loading the native
    # DLLs (scipy/torch) for the first time inside FastMCP's worker thread hung the
    # server indefinitely on the first tool call.
    print("[rag-study] loading embedding model...", file=sys.stderr, flush=True)
    _get_embed_model(Config.from_env())
    print("[rag-study] model loaded, server ready.", file=sys.stderr, flush=True)
    mcp.run()


if __name__ == "__main__":
    main()
