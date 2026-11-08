# tests/test_ingest.py
import logging

from llama_index.core.embeddings import MockEmbedding

from rag_study import ingest
from rag_study.config import Config
from rag_study.ingest import run_ingestion

EMBED_MODEL = MockEmbedding(embed_dim=8)


def test_run_ingestion_indexes_all_readable_documents(tmp_path, caplog):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "a.txt").write_text("Documento A com algum conteudo.", encoding="utf-8")
    (docs_dir / "b.txt").write_text("Documento B com outro conteudo.", encoding="utf-8")
    storage_dir = tmp_path / "storage"

    with caplog.at_level(logging.INFO):
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
    assert any(
        r.levelno == logging.INFO
        and "2 document(s) indexed into 2 chunk(s)" in r.getMessage()
        for r in caplog.records
    )


def test_run_ingestion_skips_corrupted_files_and_still_indexes_rest(tmp_path, caplog):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "bom.txt").write_text("Documento valido.", encoding="utf-8")
    (docs_dir / "quebrado.pdf").write_bytes(b"nao e um pdf valido")
    storage_dir = tmp_path / "storage"

    with caplog.at_level(logging.INFO):
        chunk_count = run_ingestion(
            docs_dir=docs_dir,
            storage_dir=storage_dir,
            embed_model=EMBED_MODEL,
            chunk_size=512,
            chunk_overlap=50,
            collection_name="test_ingest_skip_collection",
        )

    assert chunk_count == 1
    assert any(
        record.levelno == logging.WARNING
        and "quebrado.pdf" in record.getMessage()
        for record in caplog.records
    )


def test_run_ingestion_returns_zero_and_warns_on_empty_folder(tmp_path, caplog):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    storage_dir = tmp_path / "storage"

    with caplog.at_level(logging.INFO):
        chunk_count = run_ingestion(
            docs_dir=docs_dir,
            storage_dir=storage_dir,
            embed_model=EMBED_MODEL,
            chunk_size=512,
            chunk_overlap=50,
            collection_name="test_ingest_empty_collection",
        )

    assert chunk_count == 0
    assert any(
        record.levelno == logging.WARNING
        and "No readable documents" in record.getMessage()
        for record in caplog.records
    )


def test_main_configures_logging(tmp_path, monkeypatch):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    received = []

    monkeypatch.setattr(ingest, "setup_logging", received.append)
    monkeypatch.setenv("RAG_DOCS_DIR", str(docs_dir))
    monkeypatch.setenv("RAG_STORAGE_DIR", str(tmp_path / "storage"))
    monkeypatch.setenv("RAG_LOG_LEVEL", "DEBUG")
    monkeypatch.setattr(Config, "get_embed_model", lambda self: EMBED_MODEL)

    ingest.main()

    assert received == ["DEBUG"]
