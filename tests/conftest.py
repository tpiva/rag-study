import os

# nltk's import-security hook (CWE-427 mitigation) treats any module resolved
# from inside the current working directory as suspicious. Since .venv lives
# inside this project's root, packages like `regex` resolve there and get
# false-positive blocked when llama_index triggers nltk's punkt tokenizer.
os.environ.setdefault("NLTK_DISABLE_IMPORT_SECURITY", "1")
