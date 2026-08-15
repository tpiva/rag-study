from pathlib import Path

import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.schema import TextNode
from llama_index.vector_stores.chroma import ChromaVectorStore

def build_index(
        nodes: list[TextNode],
        storage_dir: Path,
        embed_model: str,
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
        collection_name: str
) -> VectorStoreIndex | None:
    if not storage_dir.exists():
        return None
    chroma_client = chromadb.PersistentClient(path=str(storage_dir))
    collection = chroma_client.get_or_create_collection(collection_name)
    if collection.count() == 0:
        return None
    vector_store = ChromaVectorStore(chroma_collection=collection)
    return VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)