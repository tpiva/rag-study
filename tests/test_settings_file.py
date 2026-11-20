import os

import pytest

from rag_study.config import ConfigError
from rag_study.settings_file import load_settings, save_settings


def test_load_settings_missing_file_returns_empty(tmp_path):
    assert load_settings(tmp_path / "x.json") == {}
def test_save_then_load_roundtrip(tmp_path):
    save_settings(p := tmp_path / "x.json", {"docs_dir": "C:\\Docs\\médicos", "ocr_min_chars": 30})
    assert load_settings(p) == {"docs_dir": "C:\\Docs\\médicos", "ocr_min_chars": 30}
def test_load_settings_invalid_json_raises_config_error_naming_file(tmp_path):
    (p := tmp_path / "x.json").write_text("{nope")
    with pytest.raises(ConfigError, match="x.json"): load_settings(p)
def test_load_settings_rejects_wrong_type(tmp_path):
    (p := tmp_path / "x.json").write_text('{"chunk_size": "grande"}')
    with pytest.raises(ConfigError, match="chunk_size"): load_settings(p)
def test_save_settings_is_atomic(tmp_path, monkeypatch):
    save_settings(p := tmp_path / "x.json", {"chunk_size": 256})
    monkeypatch.setattr(os, "replace", lambda *a: (_ for _ in ()).throw(OSError("boom")))
    with pytest.raises(OSError): save_settings(p, {"chunk_size": 999})
    monkeypatch.undo()
    assert load_settings(p) == {"chunk_size": 256} and list(tmp_path.iterdir()) == [p]