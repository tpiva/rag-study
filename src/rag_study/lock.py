import contextlib
from pathlib import Path

from filelock import FileLock, Timeout
from collections.abc import Generator


class IngestInProgressError(Exception):
    pass

@contextlib.contextmanager
def ingest_lock(storage_dir: Path) -> Generator[None, None, None]:
    storage_dir.mkdir(parents=True, exist_ok=True)
        
    file_lock = FileLock(storage_dir / "ingest.lock", timeout=0)
    try:
        file_lock.acquire()
    except Timeout:
        raise IngestInProgressError("Another ingestion is already running ...")
    try:
        yield
    finally:
        file_lock.release()
            
            