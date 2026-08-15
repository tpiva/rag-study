from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document, TextNode

def split_into_nodes(
        documents: list[Document], chunk_size: int, chunk_overlap: int
) -> list[TextNode]:
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.get_nodes_from_documents(documents)