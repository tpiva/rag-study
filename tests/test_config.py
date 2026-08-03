from rag_study.config import Config, ConfigError

import pytest


def test_get_docs_dir_raises_when_env_var_missing(monkeypatch):
    monkeypatch.delenv("RAG_DOCS_DIR", raising=False)
    cfg = Config.from_env()
    with pytest.raises(ConfigError, match="RAG_DOCS_DIR"):
        cfg.get_docs_dir()


def test_get_docs_dir_raises_when_path_does_not_exist(monkeypatch, tmp_path):
    missing = tmp_path / "nao-existe"
    monkeypatch.setenv("RAG_DOCS_DIR", str(missing))
    cfg = Config.from_env()
    with pytest.raises(ConfigError, match="não existe"):
        cfg.get_docs_dir()


def test_get_docs_dir_returns_path_when_valid(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_DOCS_DIR", str(tmp_path))
    cfg = Config.from_env()
    assert cfg.get_docs_dir() == tmp_path


def test_storage_dir_defaults_to_storage_folder(monkeypatch):
    monkeypatch.delenv("RAG_STORAGE_DIR", raising=False)
    cfg = Config.from_env()
    assert cfg.storage_dir.name == "storage"


def test_storage_dir_respects_env_override(monkeypatch, tmp_path):
    custom = tmp_path / "custom-storage"
    monkeypatch.setenv("RAG_STORAGE_DIR", str(custom))
    cfg = Config.from_env()
    assert cfg.storage_dir == custom


def test_chunk_size_defaults_and_override(monkeypatch):
    monkeypatch.delenv("RAG_CHUNK_SIZE", raising=False)
    cfg = Config.from_env()
    assert cfg.chunk_size == 512

    monkeypatch.setenv("RAG_CHUNK_SIZE", "256")
    cfg = Config.from_env()
    assert cfg.chunk_size == 256