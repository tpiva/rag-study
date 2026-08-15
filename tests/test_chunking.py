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