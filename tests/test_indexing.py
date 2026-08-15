from llama_index.core.embeddings import MockEmbedding
from llama_index.core.schema import TextNode

from rag_study.indexing import build_index, load_index

EMBED_MODEL = MockEmbedding(embed_dim=8)


def test_build_index_persists_to_storage_dir(tmp_path):
    nodes = [TextNode(text="Primeiro trecho sobre Python.")]
    storage_dir = tmp_path / "storage"

    index = build_index(nodes, storage_dir, EMBED_MODEL, collection_name="test_collection")

    assert index is not None
    assert storage_dir.exists()
    assert any(storage_dir.iterdir())


def test_load_index_returns_none_when_storage_dir_missing(tmp_path):
    missing = tmp_path / "nao-existe"

    assert load_index(missing, EMBED_MODEL, collection_name="test_collection") is None


def test_load_index_returns_index_after_build(tmp_path):
    nodes = [TextNode(text="Trecho indexado para recuperar depois.")]
    storage_dir = tmp_path / "storage"
    build_index(nodes, storage_dir, EMBED_MODEL, collection_name="test_collection")

    index = load_index(storage_dir, EMBED_MODEL, collection_name="test_collection")

    assert index is not None