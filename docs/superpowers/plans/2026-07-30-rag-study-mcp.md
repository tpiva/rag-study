# RAG Study MCP — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local MCP server that lets Claude (Desktop and Claude Code) semantically search a personal, mostly-PDF document folder, backed by a locally-embedded ChromaDB index — with zero paid Anthropic API calls in the project's code.

**Architecture:** A CLI ingestion script (`ingest.py`) reads documents from an externally configured folder, chunks them, embeds them locally via HuggingFace, and persists them to a local ChromaDB index. A separate MCP server process (`mcp_server.py`) loads that index and exposes a single search tool (`buscar_documentos`) that Claude calls over the MCP protocol. Both scripts share `config.py`, `loader.py`, `chunking.py`, and `indexing.py`.

**Tech Stack:** Python 3.11+, `uv` (package/env manager, project built as an installable package via `uv init --package`), LlamaIndex (`llama-index-core`), `llama-index-embeddings-huggingface` (local embeddings), `llama-index-vector-stores-chroma` + `chromadb` (vector store), `fastmcp` (standalone FastMCP framework), `pytest`.

## Global Constraints

- No code in this project may call the paid Anthropic API — the only Claude integration point is the MCP protocol (`fastmcp` package), consumed by Claude Desktop / Claude Code via the user's existing subscription.
- The project's importable code lives in the `rag_study` package under `src/rag_study/` (an installable package via `uv init --package` — plain `uv init` without `--package` does not support `[project.scripts]` entry points, confirmed by testing).
- Embeddings must run 100% locally (HuggingFace/sentence-transformers) — no external embedding API calls, no documents leaving the machine.
- The documents folder lives outside the project and is configured via the `RAG_DOCS_DIR` environment variable — never hardcode a path or copy documents into the repo.
- Vector storage is ChromaDB, persisted to a local directory (`storage/` by default), gitignored.
- Package/environment management uses `uv` exclusively — no bare `pip install`, no `requirements.txt`.
- Corrupted or text-less documents must be skipped with a console warning during ingestion, never aborting the whole run.
- If the MCP server starts before any index exists, the search tool must return a clear, actionable message instead of crashing.

---

### Task 0: Project scaffolding

**Files:**
- Create: `pyproject.toml` (via `uv init --package`)
- Create: `src/rag_study/__init__.py` (auto-created by `uv init --package`, then emptied)
- Create: `tests/__init__.py`
- Create: `storage/.gitkeep`
- Create: `docs/notes/.gitkeep`

**Interfaces:**
- Produces: an installable `uv` project named `rag-study`, whose code lives in the importable `rag_study` package (`src/rag_study/`), with `pytest` runnable via `uv run pytest` and console scripts runnable via `uv run <script-name>`.

- [ ] **Step 1: Initialize the uv project as an installable package**

Run (from `C:\Users\Thiago\Documents\estudo\rag-study`):

```bash
uv init --name rag-study --package --python 3.11 --no-readme
```

`--package` is required here — without it, `uv init` creates a plain "application" project with no `[build-system]` table, and `[project.scripts]` entries added later (Tasks 5 and 7) never produce a runnable command. Confirmed by testing: with a plain (non-`--package`) `uv init` plus a manually added `[project.scripts]` entry, `uv run <script>` fails with `Failed to spawn: <script> / program not found`, because the project is never built/installed. `--package` also runs `git init` and generates a `.gitignore` automatically — no separate `git init` step is needed later in this task.

This generates `src/rag_study/__init__.py` with a placeholder `main()` and a matching placeholder entry in `pyproject.toml`'s `[project.scripts]`. Clear both — the real entry points come from Tasks 5 and 7:

```bash
echo -n "" > src/rag_study/__init__.py
```

Then open `pyproject.toml` and delete the auto-generated `[project.scripts]` section (it looks like `[project.scripts]` followed by `rag-study = "rag_study:main"`) — remove those two lines. Keep the generated `[project]` and `[build-system]` sections as-is.

- [ ] **Step 2: Add runtime dependencies**

```bash
uv add llama-index-core llama-index-vector-stores-chroma llama-index-embeddings-huggingface chromadb fastmcp pypdf
```

- [ ] **Step 3: Add dev dependencies**

```bash
uv add --dev pytest
```

- [ ] **Step 4: Create the test and data folders**

`src/rag_study/` already exists from Step 1 — this only adds the folders `uv init` doesn't create:

```bash
mkdir -p tests storage docs/notes
touch tests/__init__.py storage/.gitkeep docs/notes/.gitkeep
```

- [ ] **Step 5: Extend the generated `.gitignore`**

`uv init` already wrote a `.gitignore` with the standard Python entries (`__pycache__/`, `.venv`, etc). Append the project-specific ones instead of overwriting it:

```bash
cat >> .gitignore <<'EOF'
storage/*
!storage/.gitkeep
.env
EOF
```

- [ ] **Step 6: Verify the environment installs and imports cleanly**

Run: `uv run python -c "import llama_index.core, chromadb, fastmcp, pypdf, rag_study; print('ok')"`
Expected: prints `ok` with no errors.

- [ ] **Step 7: Commit the scaffold**

`git init` already ran as part of `uv init --package` in Step 1 — just stage and commit:

```bash
git add pyproject.toml .gitignore .python-version src tests storage/.gitkeep docs
git commit -m "chore: scaffold rag-study as an installable uv package"
```

---

### Task 1: Config module

**Files:**
- Create: `src/rag_study/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: environment variables `RAG_DOCS_DIR`, `RAG_STORAGE_DIR`, `RAG_EMBEDDING_MODEL`, `RAG_CHUNK_SIZE`, `RAG_CHUNK_OVERLAP`.
- Produces: `ConfigError(Exception)`, frozen dataclass `Config` with fields `docs_dir_raw: str | None`, `storage_dir: Path`, `embedding_model: str`, `chunk_size: int`, `chunk_overlap: int`, `collection_name: str`; classmethod `Config.from_env() -> Config`; instance methods `get_docs_dir(self) -> Path` and `get_embed_model(self) -> BaseEmbedding` (lazy-imports HuggingFace).

A `Config` is a plain value object built explicitly via `Config.from_env()` — no module-level state, no `importlib.reload()` needed in tests. Each test just builds a fresh `Config.from_env()` after setting the env vars it cares about.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_config.py
import pytest

from rag_study.config import Config, ConfigError


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag_study.config'`

- [ ] **Step 3: Implement `src/rag_study/config.py`**

```python
# src/rag_study/config.py
import os
from dataclasses import dataclass
from pathlib import Path


class ConfigError(Exception):
    """Raised when required project configuration is missing or invalid."""


@dataclass(frozen=True)
class Config:
    docs_dir_raw: str | None
    storage_dir: Path
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    collection_name: str = "rag_study_docs"

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            docs_dir_raw=os.environ.get("RAG_DOCS_DIR"),
            storage_dir=Path(os.environ.get("RAG_STORAGE_DIR", "storage")),
            embedding_model=os.environ.get(
                "RAG_EMBEDDING_MODEL",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            ),
            chunk_size=int(os.environ.get("RAG_CHUNK_SIZE", "512")),
            chunk_overlap=int(os.environ.get("RAG_CHUNK_OVERLAP", "50")),
        )

    def get_docs_dir(self) -> Path:
        if not self.docs_dir_raw:
            raise ConfigError(
                "RAG_DOCS_DIR não está definida. Configure o caminho da pasta "
                "com seus documentos, ex: RAG_DOCS_DIR=C:\\Users\\voce\\Documents\\meus-pdfs"
            )
        path = Path(self.docs_dir_raw)
        if not path.is_dir():
            raise ConfigError(
                f"RAG_DOCS_DIR aponta para um caminho que não existe ou não é uma pasta: {path}"
            )
        return path

    def get_embed_model(self):
        """Lazily imports HuggingFaceEmbedding so building a Config stays fast/cheap."""
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        return HuggingFaceEmbedding(model_name=self.embedding_model)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/rag_study/config.py tests/test_config.py
git commit -m "feat: add config module for docs dir, storage dir and embedding settings"
```

---

### Task 2: Document loader (with per-file error skipping)

**Files:**
- Create: `src/rag_study/loader.py`
- Test: `tests/test_loader.py`

**Interfaces:**
- Consumes: `pathlib.Path` pointing at a folder.
- Produces: `load_documents(docs_dir: Path) -> list[Document]` (uses `llama_index.core.schema.Document`). Prints `[aviso] Pulando '<nome>': ...` to stdout for any file that fails to load, and continues with the rest.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_loader.py
from rag_study.loader import load_documents


def test_load_documents_reads_text_file_content(tmp_path):
    (tmp_path / "notas.txt").write_text("Conteudo de teste sobre RAG.", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert len(documents) == 1
    assert "Conteudo de teste sobre RAG." in documents[0].text


def test_load_documents_reads_multiple_files(tmp_path):
    (tmp_path / "a.txt").write_text("Primeiro documento.", encoding="utf-8")
    (tmp_path / "b.txt").write_text("Segundo documento.", encoding="utf-8")

    documents = load_documents(tmp_path)

    texts = {doc.text for doc in documents}
    assert "Primeiro documento." in texts
    assert "Segundo documento." in texts


def test_load_documents_skips_corrupted_file_with_warning(tmp_path, capsys):
    (tmp_path / "bom.txt").write_text("Documento valido.", encoding="utf-8")
    (tmp_path / "quebrado.pdf").write_bytes(b"isso nao e um pdf valido")

    documents = load_documents(tmp_path)

    assert len(documents) == 1
    assert documents[0].text == "Documento valido."
    captured = capsys.readouterr()
    assert "quebrado.pdf" in captured.out
    assert "[aviso]" in captured.out


def test_load_documents_returns_empty_list_for_empty_folder(tmp_path):
    assert load_documents(tmp_path) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_loader.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag_study.loader'`

- [ ] **Step 3: Implement `src/rag_study/loader.py`**

```python
# src/rag_study/loader.py
from pathlib import Path

from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document


def load_documents(docs_dir: Path) -> list[Document]:
    documents: list[Document] = []
    file_paths = sorted(
        p for p in docs_dir.rglob("*") if p.is_file() and not p.name.startswith(".")
    )
    for file_path in file_paths:
        try:
            loaded = SimpleDirectoryReader(input_files=[str(file_path)]).load_data()
        except Exception as exc:  # noqa: BLE001 - intentional broad catch, see spec
            print(f"[aviso] Pulando '{file_path.name}': não foi possível extrair texto ({exc})")
            continue
        documents.extend(loaded)
    return documents
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_loader.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/rag_study/loader.py tests/test_loader.py
git commit -m "feat: add document loader that skips unreadable files with a warning"
```

---

### Task 3: Chunking

**Files:**
- Create: `src/rag_study/chunking.py`
- Test: `tests/test_chunking.py`

**Interfaces:**
- Consumes: `list[Document]` (from Task 2's `load_documents`), `chunk_size: int`, `chunk_overlap: int` (from Task 1's `Config.chunk_size` / `Config.chunk_overlap`).
- Produces: `split_into_nodes(documents: list[Document], chunk_size: int, chunk_overlap: int) -> list[TextNode]` (`llama_index.core.schema.TextNode`).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_chunking.py
from llama_index.core.schema import Document

from rag_study.chunking import split_into_nodes


def test_split_into_nodes_produces_multiple_chunks_for_long_text():
    long_text = " ".join(f"Esta e a frase numero {i} do documento de teste." for i in range(60))
    doc = Document(text=long_text)

    nodes = split_into_nodes([doc], chunk_size=50, chunk_overlap=10)

    assert len(nodes) > 1
    assert all(node.get_content().strip() for node in nodes)


def test_split_into_nodes_single_chunk_for_short_text():
    doc = Document(text="Um texto curto.")

    nodes = split_into_nodes([doc], chunk_size=512, chunk_overlap=50)

    assert len(nodes) == 1
    assert nodes[0].get_content() == "Um texto curto."


def test_split_into_nodes_preserves_source_metadata():
    doc = Document(text="Texto com metadado.", metadata={"file_name": "arquivo.txt"})

    nodes = split_into_nodes([doc], chunk_size=512, chunk_overlap=50)

    assert nodes[0].metadata.get("file_name") == "arquivo.txt"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_chunking.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag_study.chunking'`

- [ ] **Step 3: Implement `src/rag_study/chunking.py`**

```python
# src/rag_study/chunking.py
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, TextNode


def split_into_nodes(
    documents: list[Document], chunk_size: int, chunk_overlap: int
) -> list[TextNode]:
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.get_nodes_from_documents(documents)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_chunking.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/rag_study/chunking.py tests/test_chunking.py
git commit -m "feat: add sentence-based chunking of loaded documents"
```

---

### Task 4: Embedding + ChromaDB indexing

**Files:**
- Create: `src/rag_study/indexing.py`
- Test: `tests/test_indexing.py`

**Interfaces:**
- Consumes: `list[TextNode]` (from Task 3), `storage_dir: Path` (from `Config.storage_dir`), an `embed_model` (`BaseEmbedding` — real one from `Config.get_embed_model()`, or `MockEmbedding` in tests), `collection_name: str` (from `Config.collection_name`).
- Produces: `build_index(nodes, storage_dir, embed_model, collection_name) -> VectorStoreIndex`, `load_index(storage_dir, embed_model, collection_name) -> VectorStoreIndex | None` (returns `None` when no index exists yet).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_indexing.py
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.schema import TextNode

from rag_study.indexing import build_index, load_index

EMBED_MODEL = MockEmbedding(embed_dim=8)


def test_build_index_persists_to_storage_dir(tmp_path):
    nodes = [TextNode(text="Primeiro trecho sobre Python.")]
    storage_dir = tmp_path / "storage"

    index = build_index(nodes, storage_dir, EMBED_MODEL, collection_name="test_collection")

    assert index is not None
    assert storage_dir.exists()
    assert any(storage_dir.iterdir())


def test_load_index_returns_none_when_storage_dir_missing(tmp_path):
    missing = tmp_path / "nao-existe"

    assert load_index(missing, EMBED_MODEL, collection_name="test_collection") is None


def test_load_index_returns_index_after_build(tmp_path):
    nodes = [TextNode(text="Trecho indexado para recuperar depois.")]
    storage_dir = tmp_path / "storage"
    build_index(nodes, storage_dir, EMBED_MODEL, collection_name="test_collection")

    index = load_index(storage_dir, EMBED_MODEL, collection_name="test_collection")

    assert index is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_indexing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag_study.indexing'`

- [ ] **Step 3: Implement `src/rag_study/indexing.py`**

```python
# src/rag_study/indexing.py
from pathlib import Path

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore


def build_index(
    nodes: list[TextNode],
    storage_dir: Path,
    embed_model: BaseEmbedding,
    collection_name: str,
) -> VectorStoreIndex:
    storage_dir.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(storage_dir))
    collection = chroma_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)


def load_index(
    storage_dir: Path,
    embed_model: BaseEmbedding,
    collection_name: str,
) -> VectorStoreIndex | None:
    if not storage_dir.exists():
        return None
    chroma_client = chromadb.PersistentClient(path=str(storage_dir))
    collection = chroma_client.get_or_create_collection(collection_name)
    if collection.count() == 0:
        return None
    vector_store = ChromaVectorStore(chroma_collection=collection)
    return VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_indexing.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/rag_study/indexing.py tests/test_indexing.py
git commit -m "feat: add ChromaDB-backed index build/load functions"
```

---

### Task 5: Ingestion CLI script

**Files:**
- Create: `src/rag_study/ingest.py`
- Test: `tests/test_ingest.py`
- Modify: `pyproject.toml` (add `[project.scripts]` entry)

**Interfaces:**
- Consumes: `load_documents` (Task 2), `split_into_nodes` (Task 3), `build_index` (Task 4), `Config` (Task 1).
- Produces: `run_ingestion(docs_dir: Path, storage_dir: Path, embed_model, chunk_size: int, chunk_overlap: int, collection_name: str) -> int` (returns number of chunks indexed; prints a human-readable summary), `main() -> None` (CLI entry point wiring `run_ingestion` to `config`).

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_ingest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag_study.ingest'`

- [ ] **Step 3: Implement `src/rag_study/ingest.py`**

```python
# src/rag_study/ingest.py
from pathlib import Path

from llama_index.core.embeddings import BaseEmbedding

from rag_study.chunking import split_into_nodes
from rag_study.config import Config
from rag_study.indexing import build_index
from rag_study.loader import load_documents


def run_ingestion(
    docs_dir: Path,
    storage_dir: Path,
    embed_model: BaseEmbedding,
    chunk_size: int,
    chunk_overlap: int,
    collection_name: str,
) -> int:
    documents = load_documents(docs_dir)
    if not documents:
        print(f"Nenhum documento legível encontrado em '{docs_dir}'.")
        return 0

    nodes = split_into_nodes(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    build_index(nodes, storage_dir, embed_model, collection_name)

    print(
        f"{len(documents)} documento(s) indexado(s) em {len(nodes)} chunk(s). "
        f"Índice salvo em '{storage_dir}'."
    )
    return len(nodes)


def main() -> None:
    cfg = Config.from_env()
    docs_dir = cfg.get_docs_dir()
    run_ingestion(
        docs_dir=docs_dir,
        storage_dir=cfg.storage_dir,
        embed_model=cfg.get_embed_model(),
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        collection_name=cfg.collection_name,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_ingest.py -v`
Expected: 3 passed

- [ ] **Step 5: Register the CLI entry point**

Add to `pyproject.toml` under `[project]` (create `[project.scripts]` if it doesn't exist yet):

```toml
[project.scripts]
rag-ingest = "rag_study.ingest:main"
```

- [ ] **Step 6: Commit**

```bash
git add src/rag_study/ingest.py tests/test_ingest.py pyproject.toml
git commit -m "feat: add ingest CLI wiring loader, chunking and indexing together"
```

---

### Task 6: Search function

**Files:**
- Create: `src/rag_study/search.py`
- Test: `tests/test_search.py`

**Interfaces:**
- Consumes: `load_index` (Task 4), `Config.storage_dir` / `Config.collection_name` (Task 1).
- Produces: `IndexNotFoundError(Exception)`, `search_documents(query: str, top_k: int, storage_dir: Path, embed_model, collection_name: str) -> list[dict]` — each dict has keys `text: str`, `source: str`, `score: float`. Raises `IndexNotFoundError` when no index exists.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_search.py
import pytest
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.schema import TextNode

from rag_study.indexing import build_index
from rag_study.search import IndexNotFoundError, search_documents

EMBED_MODEL = MockEmbedding(embed_dim=8)


def test_search_documents_raises_when_no_index(tmp_path):
    storage_dir = tmp_path / "storage"

    with pytest.raises(IndexNotFoundError, match="rag-ingest"):
        search_documents(
            query="qualquer pergunta",
            top_k=5,
            storage_dir=storage_dir,
            embed_model=EMBED_MODEL,
            collection_name="test_search_missing",
        )


def test_search_documents_returns_results_with_text_and_source(tmp_path):
    storage_dir = tmp_path / "storage"
    nodes = [
        TextNode(text="Python e uma linguagem de programacao.", metadata={"file_name": "python.txt"}),
        TextNode(text="RAG combina busca com geracao de texto.", metadata={"file_name": "rag.txt"}),
    ]
    build_index(nodes, storage_dir, EMBED_MODEL, collection_name="test_search_results")

    results = search_documents(
        query="O que e RAG?",
        top_k=2,
        storage_dir=storage_dir,
        embed_model=EMBED_MODEL,
        collection_name="test_search_results",
    )

    assert len(results) == 2
    for result in results:
        assert set(result.keys()) == {"text", "source", "score"}
        assert isinstance(result["text"], str) and result["text"]
        assert isinstance(result["source"], str) and result["source"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_search.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag_study.search'`

- [ ] **Step 3: Implement `src/rag_study/search.py`**

```python
# src/rag_study/search.py
from pathlib import Path

from llama_index.core.embeddings import BaseEmbedding

from rag_study.indexing import load_index


class IndexNotFoundError(Exception):
    """Raised when a search is attempted before any index has been built."""


def search_documents(
    query: str,
    top_k: int,
    storage_dir: Path,
    embed_model: BaseEmbedding,
    collection_name: str,
) -> list[dict]:
    index = load_index(storage_dir, embed_model, collection_name)
    if index is None:
        raise IndexNotFoundError(
            "Nenhum índice encontrado. Rode 'uv run rag-ingest' para indexar seus "
            "documentos antes de consultar."
        )

    retriever = index.as_retriever(similarity_top_k=top_k)
    scored_nodes = retriever.retrieve(query)

    results = []
    for scored_node in scored_nodes:
        node = scored_node.node
        results.append(
            {
                "text": node.get_content(),
                "source": node.metadata.get("file_name", "desconhecido"),
                "score": scored_node.score if scored_node.score is not None else 0.0,
            }
        )
    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_search.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/rag_study/search.py tests/test_search.py
git commit -m "feat: add semantic search over the persisted index"
```

---

### Task 7: MCP server

**Files:**
- Create: `src/rag_study/mcp_server.py`
- Test: `tests/test_mcp_server.py`
- Modify: `pyproject.toml` (add `[project.scripts]` entry)

**Interfaces:**
- Consumes: `search_documents` / `IndexNotFoundError` (Task 6), `Config` (Task 1).
- Produces: `format_results(pergunta: str, results: list[dict]) -> str`, `buscar_documentos_core(pergunta: str, top_k: int, storage_dir, embed_model, collection_name) -> str` (pure, injectable — used by both the MCP tool and the tests), the `fastmcp.FastMCP`-registered tool `buscar_documentos(pergunta: str, top_k: int = 5) -> str`, `main() -> None`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mcp_server.py
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.schema import TextNode

from rag_study.indexing import build_index
from rag_study.mcp_server import buscar_documentos_core, format_results

EMBED_MODEL = MockEmbedding(embed_dim=8)


def test_format_results_lists_source_and_snippet_for_each_result():
    results = [
        {"text": "Trecho um.", "source": "arquivo1.pdf", "score": 0.9},
        {"text": "Trecho dois.", "source": "arquivo2.pdf", "score": 0.8},
    ]

    formatted = format_results("minha pergunta", results)

    assert "arquivo1.pdf" in formatted
    assert "Trecho um." in formatted
    assert "arquivo2.pdf" in formatted
    assert "Trecho dois." in formatted


def test_format_results_handles_empty_results():
    formatted = format_results("pergunta sem resposta", [])

    assert "Nenhum trecho relevante" in formatted


def test_buscar_documentos_core_returns_friendly_message_when_no_index(tmp_path):
    resposta = buscar_documentos_core(
        pergunta="qualquer coisa",
        top_k=5,
        storage_dir=tmp_path / "storage",
        embed_model=EMBED_MODEL,
        collection_name="test_mcp_missing",
    )

    assert "rag-ingest" in resposta


def test_buscar_documentos_core_returns_formatted_results(tmp_path):
    storage_dir = tmp_path / "storage"
    nodes = [TextNode(text="Conteudo sobre embeddings locais.", metadata={"file_name": "embeddings.txt"})]
    build_index(nodes, storage_dir, EMBED_MODEL, collection_name="test_mcp_results")

    resposta = buscar_documentos_core(
        pergunta="O que sao embeddings?",
        top_k=3,
        storage_dir=storage_dir,
        embed_model=EMBED_MODEL,
        collection_name="test_mcp_results",
    )

    assert "embeddings.txt" in resposta
    assert "Conteudo sobre embeddings locais." in resposta
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_mcp_server.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag_study.mcp_server'`

- [ ] **Step 3: Implement `src/rag_study/mcp_server.py`**

```python
# src/rag_study/mcp_server.py
from pathlib import Path

from fastmcp import FastMCP
from llama_index.core.embeddings import BaseEmbedding

from rag_study.config import Config
from rag_study.search import IndexNotFoundError, search_documents

mcp = FastMCP("rag-study")


def format_results(pergunta: str, results: list[dict]) -> str:
    if not results:
        return f"Nenhum trecho relevante encontrado para: '{pergunta}'."

    partes = []
    for i, result in enumerate(results, start=1):
        partes.append(f"[{i}] Fonte: {result['source']}\n{result['text']}")
    return "\n\n".join(partes)


def buscar_documentos_core(
    pergunta: str,
    top_k: int,
    storage_dir: Path,
    embed_model: BaseEmbedding,
    collection_name: str,
) -> str:
    try:
        results = search_documents(
            query=pergunta,
            top_k=top_k,
            storage_dir=storage_dir,
            embed_model=embed_model,
            collection_name=collection_name,
        )
    except IndexNotFoundError as exc:
        return str(exc)
    return format_results(pergunta, results)


@mcp.tool()
def buscar_documentos(pergunta: str, top_k: int = 5) -> str:
    """Busca trechos relevantes nos documentos pessoais indexados para responder à pergunta."""
    cfg = Config.from_env()
    return buscar_documentos_core(
        pergunta=pergunta,
        top_k=top_k,
        storage_dir=cfg.storage_dir,
        embed_model=cfg.get_embed_model(),
        collection_name=cfg.collection_name,
    )


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_mcp_server.py -v`
Expected: 4 passed

- [ ] **Step 5: Register the CLI entry point**

Add to `pyproject.toml` `[project.scripts]` (alongside `rag-ingest` from Task 5):

```toml
[project.scripts]
rag-ingest = "rag_study.ingest:main"
rag-mcp-server = "rag_study.mcp_server:main"
```

- [ ] **Step 6: Smoke-test the server module imports cleanly**

Run: `uv run python -c "import rag_study.mcp_server; print('mcp server ok')"`
Expected: prints `mcp server ok` with no errors.

- [ ] **Step 7: Commit**

```bash
git add src/rag_study/mcp_server.py tests/test_mcp_server.py pyproject.toml
git commit -m "feat: add MCP server exposing buscar_documentos tool"
```

---

### Task 8: End-to-end verification, README, and Claude client configuration

**Files:**
- Create: `README.md`
- Manual: run the full pipeline against real documents; configure Claude Desktop and Claude Code.

**Interfaces:**
- Consumes: everything built in Tasks 0–7.
- Produces: a documented, runnable project and two working MCP client configurations.

- [ ] **Step 1: Write `README.md`**

```markdown
# rag-study

Servidor MCP que permite consultar, via Claude, uma pasta pessoal de documentos
(majoritariamente PDFs), usando embeddings locais e ChromaDB. Nenhuma chamada à
API paga da Anthropic acontece neste projeto — o Claude Desktop/Claude Code é
quem consome a tool MCP através da sua assinatura.

## Configuração

Defina a pasta com seus documentos antes de rodar qualquer comando:

- Windows (PowerShell): `$env:RAG_DOCS_DIR = "C:\caminho\para\seus\documentos"`
- Bash: `export RAG_DOCS_DIR="/caminho/para/seus/documentos"`

Variáveis opcionais (com valores padrão):

| Variável | Padrão | Descrição |
|---|---|---|
| `RAG_STORAGE_DIR` | `storage` | Onde o índice ChromaDB é persistido |
| `RAG_EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Modelo de embedding local |
| `RAG_CHUNK_SIZE` | `512` | Tamanho de cada chunk |
| `RAG_CHUNK_OVERLAP` | `50` | Sobreposição entre chunks |

## Indexando os documentos

Rode sempre que adicionar/alterar arquivos na pasta:

```bash
uv run rag-ingest
```

## Rodando o servidor MCP manualmente (para testar)

```bash
uv run rag-mcp-server
```

## Configurando no Claude Desktop

Edite `claude_desktop_config.json` e adicione:

\`\`\`json
{
  "mcpServers": {
    "rag-study": {
      "command": "uv",
      "args": ["--directory", "CAMINHO_DO_PROJETO", "run", "rag-mcp-server"],
      "env": {
        "RAG_DOCS_DIR": "CAMINHO_DA_PASTA_DE_DOCUMENTOS"
      }
    }
  }
}
\`\`\`

Reinicie o Claude Desktop após salvar.

## Configurando no Claude Code

Na pasta do projeto:

```bash
claude mcp add rag-study --scope project -- uv --directory "CAMINHO_DO_PROJETO" run rag-mcp-server
```

Ou adicione manualmente ao `.mcp.json` do projeto o mesmo bloco `mcpServers` mostrado acima.

## Rodando os testes

```bash
uv run pytest -v
```
```

- [ ] **Step 2: Run the full test suite**

Run: `uv run pytest -v`
Expected: all tests from Tasks 1–7 pass (25 tests total: 6 + 4 + 3 + 3 + 3 + 2 + 4).

- [ ] **Step 3: Manual end-to-end check with real documents**

1. Set `RAG_DOCS_DIR` to point at a real folder with a few PDFs.
2. Run `uv run rag-ingest` — confirm it prints a summary like `"N documento(s) indexado(s) em M chunk(s)"` and that `storage/` now contains files.
3. Run `uv run rag-mcp-server` in one terminal — confirm it starts without errors (it will block, waiting for an MCP client over stdio).
4. Stop it with Ctrl+C.

- [ ] **Step 4: Configure Claude Desktop**

Follow the README's "Configurando no Claude Desktop" section with the real project path and real `RAG_DOCS_DIR`. Restart Claude Desktop. Ask a question about a document known to be in the indexed folder; confirm Claude calls `buscar_documentos` and answers using a real snippet + source filename.

- [ ] **Step 5: Configure Claude Code**

Follow the README's "Configurando no Claude Code" section. In a new Claude Code session in this project, ask a question about the same document; confirm the tool is called and returns a grounded answer.

- [ ] **Step 6: Commit**

```bash
git add README.md
git commit -m "docs: add README with setup and Claude Desktop/Code MCP configuration"
```

---

### Task 9: Study notes

**Files:**
- Create: `docs/notes/01-extracao-documentos.md`
- Create: `docs/notes/02-chunking.md`
- Create: `docs/notes/03-embeddings-locais.md`
- Create: `docs/notes/04-vector-store-chromadb.md`
- Create: `docs/notes/05-servidor-mcp.md`

**Interfaces:**
- Consumes: nothing (documentation only).
- Produces: five short markdown notes, one per concept learned while building Tasks 2–7.

- [ ] **Step 1: Write `docs/notes/01-extracao-documentos.md`**

```markdown
# Extração de documentos

`SimpleDirectoryReader` (LlamaIndex) sabe ler vários formatos — PDF via `pypdf`
por baixo dos panos, além de `.txt`/`.md`/etc — sem precisar escrever um parser
por tipo de arquivo.

Decisão de projeto: em vez de apontar o reader para a pasta inteira de uma vez
(o que aborta a leitura inteira se um arquivo falhar), `load_documents` itera
arquivo por arquivo e usa `try/except` em volta de cada leitura. Isso segue o
requisito do projeto de "pular documentos corrompidos com um aviso, sem
derrubar a ingestão inteira" — um PDF escaneado sem camada de texto, ou um
arquivo corrompido, não devem impedir os outros documentos de serem indexados.
```

- [ ] **Step 2: Write `docs/notes/02-chunking.md`**

```markdown
# Chunking (divisão em pedaços)

Um documento inteiro é grande demais para caber num único embedding útil —
embeddings capturam melhor o significado de um trecho pequeno e coeso do que
de um texto longo e heterogêneo. Por isso, antes de gerar embeddings, o texto
é dividido em "chunks" menores (nodes, na terminologia do LlamaIndex).

Usamos o `SentenceSplitter`, que tenta quebrar nas fronteiras de frases em vez
de cortar no meio de uma palavra, e mantém uma sobreposição (`chunk_overlap`)
entre chunks vizinhos para não perder contexto que ficaria dividido bem na
fronteira de dois chunks.
```

- [ ] **Step 3: Write `docs/notes/03-embeddings-locais.md`**

```markdown
# Embeddings locais

Um embedding transforma um texto num vetor numérico onde textos com significado
parecido ficam com vetores "próximos" (por similaridade de cosseno, tipicamente).
É isso que permite buscar por significado em vez de só por palavras exatas.

Neste projeto os embeddings são gerados localmente via
`HuggingFaceEmbedding` (biblioteca `sentence-transformers` por baixo), rodando
inteiramente na sua máquina — sem chave de API, sem limite de uso, sem enviar o
conteúdo dos seus documentos para fora. O modelo padrão
(`paraphrase-multilingual-MiniLM-L12-v2`) foi escolhido por ter suporte
multilíngue (incluindo português) num tamanho pequeno o suficiente para rodar
em CPU sem problema.

Nos testes automatizados usamos `MockEmbedding` em vez do modelo real — ele
gera vetores aleatórios determinísticos, o que evita baixar ~470MB de modelo
e deixa os testes rápidos, já que o que estamos testando é a lógica do nosso
código, não a qualidade do modelo de embedding em si.
```

- [ ] **Step 4: Write `docs/notes/04-vector-store-chromadb.md`**

```markdown
# Vector store (ChromaDB)

Depois de gerar os embeddings, precisamos guardá-los em algum lugar que permita
buscar rapidamente "quais vetores são mais parecidos com este aqui" — essa é a
função de um vector store. ChromaDB faz isso localmente, persistindo tudo em
disco (pasta `storage/`), sem precisar rodar um servidor separado.

O LlamaIndex integra com o Chroma através de `ChromaVectorStore` +
`StorageContext`: o índice (`VectorStoreIndex`) guarda a lógica de
alto nível (quais nodes existem, como buscar), enquanto o `ChromaVectorStore`
é quem efetivamente grava/lê os vetores no Chroma. Separar `build_index`
(cria do zero) de `load_index` (recarrega um índice já existente) deixou claro
esse ciclo de vida: a ingestão constrói, o servidor MCP só recarrega.
```

- [ ] **Step 5: Write `docs/notes/05-servidor-mcp.md`**

```markdown
# Servidor MCP

MCP (Model Context Protocol) é um protocolo que permite a um cliente como o
Claude Desktop ou o Claude Code chamar "tools" expostas por um processo externo
— no nosso caso, um processo Python rodando localmente. É isso que permite ao
Claude consultar nossos documentos sem que o código do projeto precise chamar a
API paga da Anthropic: o Claude que já está sendo usado (via assinatura) é
quem inicia a chamada.

Usamos o pacote `fastmcp` (a implementação standalone do padrão FastMCP,
mais madura e com mais recursos que a versão reduzida embutida no SDK
oficial `mcp`): o decorator `@mcp.tool()` registra uma função Python comum
como uma tool MCP, usando a assinatura da função e a docstring para o Claude
entender quando e como chamá-la. Mantivemos a lógica de negócio
(`buscar_documentos_core`) separada da função decorada (`buscar_documentos`)
para poder testar a lógica diretamente, sem precisar simular o protocolo MCP
inteiro nos testes.
```

- [ ] **Step 6: Commit**

```bash
git add docs/notes
git commit -m "docs: add study notes for each RAG concept covered"
```
