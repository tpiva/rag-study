from rag_study import load_documents

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