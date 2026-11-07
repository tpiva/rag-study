# tests/test_mcp_server.py
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.schema import TextNode

import rag_study.mcp_server as mcp_server
from rag_study.indexing import build_index
from rag_study.constants import errors
from rag_study.mcp_server import search_docs_core

EMBED_MODEL = MockEmbedding(embed_dim=8)


def test_search_docs_core_returns_error_code_when_no_index(tmp_path):
    response = search_docs_core(
        question="qualquer coisa",
        top_k=5,
        storage_dir=tmp_path / "storage",
        embed_model=EMBED_MODEL,
        collection_name="test_mcp_missing",
    )

    assert response["success"] is False
    assert response["code"] == errors.INDEX_NOT_FOUND_ERROR
    assert "rag-ingest" in response["message"]


def test_search_docs_core_returns_success_with_data(tmp_path):
    storage_dir = tmp_path / "storage"
    nodes = [TextNode(text="Conteudo sobre embeddings locais.", metadata={"file_name": "embeddings.txt"})]
    build_index(nodes, storage_dir, EMBED_MODEL, collection_name="test_mcp_results")

    response = search_docs_core(
        question="O que sao embeddings?",
        top_k=3,
        storage_dir=storage_dir,
        embed_model=EMBED_MODEL,
        collection_name="test_mcp_results",
    )

    assert response["success"] is True
    assert "Found 1 relevant passage" in response["message"]
    assert response["data"][0]["source"] == "embeddings.txt"
    assert response["data"][0]["text"] == "Conteudo sobre embeddings locais."


def test_search_docs_core_rejects_invalid_parameters(tmp_path):
    response = search_docs_core(
        question="   ",
        top_k=0,
        storage_dir=tmp_path / "storage",
        embed_model=EMBED_MODEL,
        collection_name="test_mcp_invalid",
    )

    assert response["success"] is False
    assert response["code"] == errors.VALIDATION_ERROR
    assert len(response["errors"]) == 2


def test_get_embed_model_loads_once_and_reuses_across_calls(monkeypatch):
    monkeypatch.setattr(mcp_server, "_embed_model_cache", None)
    calls = []

    class FakeConfig:
        def get_embed_model(self):
            calls.append(1)
            return EMBED_MODEL

    cfg = FakeConfig()

    first = mcp_server._get_embed_model(cfg)
    second = mcp_server._get_embed_model(cfg)

    assert first is second
    assert len(calls) == 1


def test_main_loads_embed_model_before_serving(monkeypatch):
    order = []
    monkeypatch.setattr(mcp_server, "_get_embed_model", lambda cfg: order.append("load"))
    monkeypatch.setattr(mcp_server.mcp, "run", lambda: order.append("run"))

    mcp_server.main()

    assert order == ["load", "run"]
