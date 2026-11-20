from rag_study.config import Config, ConfigError

import pytest

from rag_study.settings_file import save_settings

@pytest.fixture(autouse=True)
def _isolated_config_file(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG_CONFIG_FILE", str(tmp_path / "no-config.json"))


def test_get_docs_dir_raises_when_env_var_missing(monkeypatch):
    monkeypatch.delenv("RAG_DOCS_DIR", raising=False)
    cfg = Config.from_env()
    with pytest.raises(ConfigError, match="RAG_DOCS_DIR"):
        cfg.get_docs_dir()


def test_get_docs_dir_raises_when_path_does_not_exist(monkeypatch, tmp_path):
    missing = tmp_path / "nao-existe"
    monkeypatch.setenv("RAG_DOCS_DIR", str(missing))
    cfg = Config.from_env()
    with pytest.raises(ConfigError, match="does not exist"):
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

def test_storage_dir_defaults_to_storage_folder(monkeypatch):
    monkeypatch.delenv("RAG_LOG_LEVEL", raising=False)
    cfg = Config.from_env()
    assert cfg.log_level == "INFO"

def test_log_level_respects_env_override(monkeypatch):
    monkeypatch.setenv("RAG_LOG_LEVEL", "DEBUG")
    assert Config.from_env().log_level == "DEBUG"

def test_load_env_overrides_file(monkeypatch, tmp_path):
    config_file = tmp_path / "rag-study.config.json"
    save_settings(config_file, {"chunk_size": 256})

    monkeypatch.setenv("RAG_CONFIG_FILE", str(config_file))
    monkeypatch.setenv("RAG_CHUNK_SIZE", "128")

    cfg = Config.load()

    assert cfg.chunk_size == 128
    assert "chunk_size" in cfg.env_locked

def test_load_file_overrides_default(monkeypatch, tmp_path):
    monkeypatch.delenv("RAG_CHUNK_SIZE", raising=False)
    config_file = tmp_path / "rag-study.config.json"
    save_settings(config_file, {"chunk_size": 256})

    monkeypatch.setenv("RAG_CONFIG_FILE", str(config_file))

    cfg = Config.load()

    assert cfg.chunk_size == 256

def test_load_defaults_without_file(monkeypatch):
    monkeypatch.delenv("RAG_CHUNK_SIZE", raising=False)
    monkeypatch.delenv("RAG_DOCS_DIR", raising=False)

    cfg = Config.load()

    assert cfg.chunk_size == 512
    assert cfg.docs_dir_raw is None
    assert cfg.chunk_overlap == 50
    assert cfg.embedding_model == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
def test_get_docs_dir_uses_value_from_file(monkeypatch, tmp_path):
    monkeypatch.delenv("RAG_DOCS_DIR", raising=False)

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    config_file = tmp_path / "rag-study.config.json"
    save_settings(config_file, {"docs_dir": str(docs_dir)})
    monkeypatch.setenv("RAG_CONFIG_FILE", str(config_file))

    cfg = Config.load()

    assert cfg.get_docs_dir() == docs_dir

def test_get_docs_dir_error_mentions_env_and_config_file(monkeypatch):
    monkeypatch.delenv("RAG_DOCS_DIR", raising=False)

    cfg = Config.load()

    with pytest.raises(ConfigError, match="RAG_DOCS_DIR") as exc_info:
        cfg.get_docs_dir()
    assert "rag-study.config.json" in str(exc_info.value)

def test_from_env_ignores_file(monkeypatch, tmp_path):
    monkeypatch.delenv("RAG_CHUNK_SIZE", raising=False)

    config_file = tmp_path / "rag-study.config.json"
    save_settings(config_file, {"chunk_size": 256})
    monkeypatch.setenv("RAG_CONFIG_FILE", str(config_file))

    cfg = Config.from_env()

    assert cfg.chunk_size == 512
