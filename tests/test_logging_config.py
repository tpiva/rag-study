import logging
import sys

import pytest

from rag_study.logging_config import setup_logging


@pytest.fixture(autouse=True)
def restore_root_logger():
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    yield
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)


def _our_handlers():
    return [h for h in logging.getLogger().handlers if h.name == "rag_study"]


def test_setup_logging_sets_root_level():
    setup_logging("DEBUG")
    assert logging.getLogger().level == logging.DEBUG


def test_setup_logging_accepts_lowercase():
    setup_logging("warning")
    assert logging.getLogger().level == logging.WARNING


def test_setup_logging_invalid_level_falls_back_to_info():
    setup_logging("banana")
    assert logging.getLogger().level == logging.INFO


def test_setup_logging_is_idempotent():
    setup_logging("INFO")
    setup_logging("INFO")
    assert len(_our_handlers()) == 1


def test_setup_logging_writes_to_stderr():
    setup_logging("INFO")
    assert _our_handlers()[0].stream is sys.stderr
