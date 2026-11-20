import json
import logging
import os
from pathlib import Path

from rag_study.config import ConfigError

logger = logging.getLogger(__name__)

_EXPECTED_TYPES: dict[str, type] = {
    "docs_dir": str,
    "embedding_model": str,
    "ocr_min_chars": int,
    "chunk_size": int,
    "chunk_overlap": int,
}

def load_settings(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Could not read config file {path}: {exc}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in config file {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError(f"Config file {path} must contain a JSON object, got {type(data).__name__}")

    for key, value in _EXPECTED_TYPES.items():
        if key in data and not isinstance(data[key], value):
            raise ConfigError(
                f"Config file {path}: key '{key}' must be {value.__name__}, "
                f"got {type(data[key]).__name__}"
            )

    if "active_profile" in data and data["active_profile"] is not None:
        if not isinstance(data["active_profile"], str):
            raise ConfigError(f"Config file {path}: key 'active_profile' must be str or null")

    return data

def save_settings(path: Path, values: dict) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(values, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        os.replace(tmp_path, path)
    except OSError:
        tmp_path.unlink(missing_ok=True)
        raise

