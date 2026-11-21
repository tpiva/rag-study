import pytest

from rag_study.lock import IngestInProgressError, ingest_lock


def test_second_acquisition_fails_with_controlled_error(tmp_path):
    with ingest_lock(tmp_path / "s"):
        with pytest.raises(IngestInProgressError):
            with ingest_lock(tmp_path / "s"): pass
def test_lock_is_released_after_normal_exit_and_after_exception(tmp_path):
    with ingest_lock(tmp_path): pass
    with pytest.raises(RuntimeError):
        with ingest_lock(tmp_path): raise RuntimeError
    with ingest_lock(tmp_path): pass