# src/rag_study/mcp_server.py
from pathlib import Path

from fastmcp import FastMCP
from llama_index.core.embeddings import BaseEmbedding

from rag_study.config import Config
from rag_study.search import IndexNotFoundError, search_documents

mcp = FastMCP("rag-study")


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
        embed_model=cfg.get_embed_model(),
        collection_name=cfg.collection_name,
    )


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
