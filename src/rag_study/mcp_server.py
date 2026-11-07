# src/rag_study/mcp_server.py
import sys
from pathlib import Path

from fastmcp import FastMCP
from llama_index.core.embeddings import BaseEmbedding
from pydantic import ValidationError

from rag_study.config import Config
from rag_study.constants import errors
from rag_study.schemas import SearchDocsSchema
from rag_study.search import IndexNotFoundError, search_documents

mcp = FastMCP("rag-study")

_embed_model_cache: BaseEmbedding | None = None


def _get_embed_model(cfg: Config) -> BaseEmbedding:
    """Loads the embedding model once per server process and reuses it across searches."""
    global _embed_model_cache
    if _embed_model_cache is None:
        _embed_model_cache = cfg.get_embed_model()
    return _embed_model_cache


def search_docs_core(
    question: str,
    top_k: int,
    storage_dir: Path,
    embed_model: BaseEmbedding,
    collection_name: str,
) -> dict:
    try:
        params = SearchDocsSchema(question=question, top_k=top_k)
    except ValidationError as e:
        return {
            "success": False,
            "message": f"Invalid parameters: {e}",
            "code": errors.VALIDATION_ERROR,
            "errors": e.errors(include_url=False, include_context=False),
        }

    try:
        results = search_documents(
            query=params.question,
            top_k=params.top_k,
            storage_dir=storage_dir,
            embed_model=embed_model,
            collection_name=collection_name,
        )
    except IndexNotFoundError as exc:
        return {
            "success": False,
            "message": str(exc),
            "code": errors.INDEX_NOT_FOUND_ERROR,
        }

    if not results:
        message = f"No relevant passages found for: '{params.question}'."
    else:
        message = f"Found {len(results)} relevant passage(s)."
    return {"success": True, "message": message, "data": results}


@mcp.tool()
def search_docs(question: str, top_k: int = 5) -> dict:
    """
    Searches the indexed personal documents for passages relevant to the question.

    Args:
        question (str): Natural-language question or search text.
        top_k (int, optional): Number of passages to return (1-20). Defaults to 5.

    Returns:
        Dict[str, Union[bool, str, List[Dict[str, Union[str, float]]]]]: Dictionary containing:
            - On success:
                * success (bool): True
                * message (str): Summary of the search
                * data (List[Dict]): Matching passages, each with:
                    - text (str): The passage content
                    - source (str): Source file name
                    - score (float): Similarity score
            - On error:
                * success (bool): False
                * message (str): Error description
                * code (str): Specific error code
    """
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
