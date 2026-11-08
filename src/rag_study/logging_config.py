import logging
import sys

_HANDLER_NAME = "rag_study"
_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
def setup_logging(level: str = "INFO") -> None:
    resolved = logging.getLevelName(level.upper())

    if not isinstance(resolved, int):
        resolved = logging.INFO

    root = logging.getLogger()

    for handle in list(root.handlers):
        if handle.name == _HANDLER_NAME:
            root.removeHandler(handle)

    handle = logging.StreamHandler(sys.stderr)
    handle.set_name(_HANDLER_NAME)
    handle.setFormatter(logging.Formatter(_FORMAT))
    root.addHandler(handle)
    root.setLevel(resolved)
