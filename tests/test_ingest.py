# tests/test_ingest.py
from llama_index.core.embeddings import MockEmbedding

from rag_study.ingest import run_ingestion

EMBED_MODEL = MockEmbedding(embed_dim=8)


def test_run_ingestion_indexes_all_readable_documents(tmp_path, capsys):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.txt").write_text("Documento A com algum conteudo.", encoding="utf-8")
    (docs_dir / "b.txt").write_text("Documento B com outro conteudo.", encoding="utf-8")
    storage_dir = tmp_path / "storage"

    chunk_count = run_ingestion(
        docs_dir=docs_dir,
        storage_dir=storage_dir,
        embed_model=EMBED_MODEL,
        chunk_size=512,
        chunk_overlap=50,
        collection_name="test_ingest_collection",
    )

    assert chunk_count == 2
    assert storage_dir.exists()
    captured = capsys.readouterr()
    assert "2 documento" in captured.out
    assert "2 chunk" in captured.out


def test_run_ingestion_skips_corrupted_files_and_still_indexes_rest(tmp_path, capsys):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "bom.txt").write_text("Documento valido.", encoding="utf-8")
    (docs_dir / "quebrado.pdf").write_bytes(b"nao e um pdf valido")
    storage_dir = tmp_path / "storage"

    chunk_count = run_ingestion(
        docs_dir=docs_dir,
        storage_dir=storage_dir,
        embed_model=EMBED_MODEL,
        chunk_size=512,
        chunk_overlap=50,
        collection_name="test_ingest_skip_collection",
    )

    assert chunk_count == 1
    captured = capsys.readouterr()
    assert "[aviso]" in captured.out


def test_run_ingestion_returns_zero_and_warns_on_empty_folder(tmp_path, capsys):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    storage_dir = tmp_path / "storage"

    chunk_count = run_ingestion(
        docs_dir=docs_dir,
        storage_dir=storage_dir,
        embed_model=EMBED_MODEL,
        chunk_size=512,
        chunk_overlap=50,
        collection_name="test_ingest_empty_collection",
    )

    assert chunk_count == 0
    captured = capsys.readouterr()
    assert "Nenhum documento" in captured.out
