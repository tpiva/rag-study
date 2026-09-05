# tests/test_mcp_server.py
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.schema import TextNode

from rag_study.indexing import build_index
from rag_study.mcp_server import buscar_documentos_core, format_results

EMBED_MODEL = MockEmbedding(embed_dim=8)


def test_format_results_lists_source_and_snippet_for_each_result():
    results = [
        {"text": "Trecho um.", "source": "arquivo1.pdf", "score": 0.9},
        {"text": "Trecho dois.", "source": "arquivo2.pdf", "score": 0.8},
    ]

    formatted = format_results("minha pergunta", results)

    assert "arquivo1.pdf" in formatted
    assert "Trecho um." in formatted
    assert "arquivo2.pdf" in formatted
    assert "Trecho dois." in formatted


def test_format_results_handles_empty_results():
    formatted = format_results("pergunta sem resposta", [])

    assert "Nenhum trecho relevante" in formatted


def test_buscar_documentos_core_returns_friendly_message_when_no_index(tmp_path):
    resposta = buscar_documentos_core(
        pergunta="qualquer coisa",
        top_k=5,
        storage_dir=tmp_path / "storage",
        embed_model=EMBED_MODEL,
        collection_name="test_mcp_missing",
    )

    assert "rag-ingest" in resposta


def test_buscar_documentos_core_returns_formatted_results(tmp_path):
    storage_dir = tmp_path / "storage"
    nodes = [TextNode(text="Conteudo sobre embeddings locais.", metadata={"file_name": "embeddings.txt"})]
    build_index(nodes, storage_dir, EMBED_MODEL, collection_name="test_mcp_results")

    resposta = buscar_documentos_core(
        pergunta="O que sao embeddings?",
        top_k=3,
        storage_dir=storage_dir,
        embed_model=EMBED_MODEL,
        collection_name="test_mcp_results",
    )

    assert "embeddings.txt" in resposta
    assert "Conteudo sobre embeddings locais." in resposta
