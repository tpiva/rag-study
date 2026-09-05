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


def format_results(pergunta: str, results: list[dict]) -> str:
    if not results:
        return f"Nenhum trecho relevante encontrado para: '{pergunta}'."

    partes = []
    for i, result in enumerate(results, start=1):
        partes.append(f"[{i}] Fonte: {result['source']}\n{result['text']}")
    return "\n\n".join(partes)


def buscar_documentos_core(
    pergunta: str,
    top_k: int,
    storage_dir: Path,
    embed_model: BaseEmbedding,
    collection_name: str,
) -> str:
    try:
        results = search_documents(
            query=pergunta,
            top_k=top_k,
            storage_dir=storage_dir,
            embed_model=embed_model,
            collection_name=collection_name,
        )
    except IndexNotFoundError as exc:
        return str(exc)
    return format_results(pergunta, results)


@mcp.tool()
def buscar_documentos(pergunta: str, top_k: int = 5) -> str:
    """Busca trechos relevantes nos documentos pessoais indexados para responder à pergunta."""
    cfg = Config.from_env()
    return buscar_documentos_core(
        pergunta=pergunta,
        top_k=top_k,
        storage_dir=cfg.storage_dir,
        embed_model=_get_embed_model(cfg),
        collection_name=cfg.collection_name,
    )


def main() -> None:
    # Carrega o modelo de embedding no main thread ANTES de servir. Além de deixar a
    # primeira busca instantânea, isso evita um deadlock no Windows: carregar as DLLs
    # nativas (scipy/torch) pela primeira vez dentro da worker thread do FastMCP
    # travava o servidor indefinidamente na primeira chamada da tool.
    print("[rag-study] carregando modelo de embedding...", file=sys.stderr, flush=True)
    _get_embed_model(Config.from_env())
    print("[rag-study] modelo carregado, servidor pronto.", file=sys.stderr, flush=True)
    mcp.run()


if __name__ == "__main__":
    main()
