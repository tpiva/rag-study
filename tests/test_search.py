import pytest
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.schema import TextNode

from rag_study.indexing import build_index
from rag_study.search import IndexNotFoundError, search_documents

EMBED_MODEL = MockEmbedding(embed_dim=8)


def test_search_documents_raises_when_no_index(tmp_path):
    storage_dir = tmp_path / "storage"

    with pytest.raises(IndexNotFoundError, match="rag-ingest"):
        search_documents(
            query="qualquer pergunta",
            top_k=5,
            storage_dir=storage_dir,
            embed_model=EMBED_MODEL,
            collection_name="test_search_missing",
        )


def test_search_documents_returns_results_with_text_and_source(tmp_path):
    storage_dir = tmp_path / "storage"
    nodes = [
        TextNode(text="Python e uma linguagem de programacao.", metadata={"file_name": "python.txt"}),
        TextNode(text="RAG combina busca com geracao de texto.", metadata={"file_name": "rag.txt"}),
    ]
    build_index(nodes, storage_dir, EMBED_MODEL, collection_name="test_search_results")

    results = search_documents(
        query="O que e RAG?",
        top_k=2,
        storage_dir=storage_dir,
        embed_model=EMBED_MODEL,
        collection_name="test_search_results",
    )

    assert len(results) == 2
    for result in results:
        assert set(result.keys()) == {"text", "source", "score"}
        assert isinstance(result["text"], str) and result["text"]
        assert isinstance(result["source"], str) and result["source"]