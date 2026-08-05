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
            loaded = SimpleDirectoryReader(
                input_files=[str(file_path)], raise_on_error=True
            ).load_data()
        except Exception as exc:
            print(f"[aviso] Pulando '{file_path.name}': não foi possível extrair texto ({exc})")
            continue

        documents.extend(loaded)
    return documents