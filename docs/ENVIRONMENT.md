# Environment (SDD 4)

Verified on 2026-10-30: Windows 11, Python 3.11.9, Ollama with `llama3.2` (probe returned valid JSON in ~24 s;
one full QA answer with one retry took ~3 min on CPU). Docker Desktop daemon was not running; Neo4j not yet tested.

1. venv outside OneDrive: `py -3.11 -m venv %USERPROFILE%\.venvs\lawrag` (created at `C:\Users\shoer\.venvs\lawrag`).
2. `pip install -e ".[dev,ui,eval,ingest,neo4j]"`; for embeddings add CPU torch + sentence-transformers.
3. Set `HF_HOME` outside OneDrive. Limit WSL2 memory in `%UserProfile%\.wslconfig` (`[wsl2]` `memory=6GB`).
4. `docker compose up -d` (named volume `lawrag_neo4j_data`; heap 1–1.5 GB, pagecache 512 MB).
5. `lawrag doctor` (add `-c embedding/bge_m3.yaml` to measure bge-m3 load time and RSS).
6. Tesseract (optional, M5): install the Windows binary and add it to PATH.
