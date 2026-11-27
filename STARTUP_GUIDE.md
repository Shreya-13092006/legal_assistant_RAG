# lawrag — Complete Project Startup & Build Guide

This guide walks you through building and running the entire **lawrag** system on Windows (CPU-only).

---

## 1. Prerequisites & System Setup

| Component | Recommendation | Purpose |
|---|---|---|
| **Python** | 3.11 (tested) or 3.12 (`<3.13`) | Core runtime |
| **Git & PowerShell** | Latest | Repository management & scripts |
| **Docker Desktop** | Optional (WSL2 backend) | Required only for Neo4j (in-memory works by default) |
| **Ollama** | Optional | Local offline LLM (`llama3.2`) for zero-cost generation |

> [!IMPORTANT]
> **Windows Path & OneDrive Notice:**
> Create the virtual environment and HuggingFace cache outside OneDrive (e.g. `C:\Users\<user>\.venvs\lawrag`) to avoid SQLite file locking and slow synchronization overhead.

---

## 2. Virtual Environment & Dependencies

### Step 2.1 — Create and Activate Virtual Environment

Open PowerShell:

```powershell
# Create venv outside OneDrive
py -3.11 -m venv "$HOME\.venvs\lawrag"

# Activate venv
& "$HOME\.venvs\lawrag\Scripts\Activate.ps1"
```

### Step 2.2 — Install Dependencies

Install the core package in editable mode with development, UI, evaluation, ingestion, and Neo4j extras:

```powershell
# Core + extras
pip install -e ".[dev,ui,eval,ingest,neo4j]"
```

### Step 2.3 — (Optional) Real Embeddings via Torch CPU

By default, lawrag uses a lightweight lexical hashing embedder as a stand-in for tests. To use real `BAAI/bge-m3` dense embeddings without GPU requirements:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install sentence-transformers
```

### Step 2.4 — Configure Environment Variables

Copy the example environment file:

```powershell
Copy-Item .env.example .env
```

Edit `.env` if using external services:
```ini
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=lawrag-dev-password
OLLAMA_HOST=http://localhost:11434
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
LAWRAG_BUDGET_USD=5
HF_HOME=C:\Users\<user>\.cache\huggingface
```

---

## 3. Verify Health (`lawrag doctor`)

Run the built-in diagnostic tool to test Python version, embedder latency, memory RSS, and LLM probe:

```powershell
# Health check with fake/default provider
lawrag doctor -c providers/fake.yaml

# Health check with real BGE-M3 embeddings
lawrag doctor -c embedding/bge_m3.yaml -c providers/fake.yaml
```

You should see green `OK` marks across all components.

---

## 4. Building the Knowledge Graph

You can build the knowledge graph via **Path A (Fast Seed Build)** or **Path B (Full Ingestion Pipeline from Official PDFs)**.

### Path A: Fast Seed Build (Recommended for Quick Start)

The repository includes a verified seed corpus in `data/seed/` (`ipc.yaml`, `bns.yaml`, `mappings.yaml`, `secondary.yaml`, `sources.yaml`).

```powershell
# 1. Build knowledge graph snapshot into data/processed/graph.json
lawrag load

# 2. (With real BGE-M3 embeddings)
lawrag load -c embedding/bge_m3.yaml

# 3. Run graph integrity audit (checks rules G1-G12)
lawrag audit
```

The audit checks:
- No dangling mappings
- Every verified mapping has supporting evidence
- Strict provision-level hierarchy (`Act -> Chapter -> Section -> Provision -> Chunk`)
- Audit reports are written to `reports/audit/<timestamp>.md`.

---

### Path B: Full Ingestion Pipeline from Raw PDFs (M2 Pipeline)

If processing raw legal PDFs (official Gazette and concordance publications):

```powershell
# 1. Register official source documents into data/raw/
lawrag sources register data/raw/bns_2023.pdf --kind gazette --authority-rank 1 --title "Bharatiya Nyaya Sanhita, 2023" --source-id SRC-BNS-GAZETTE-2023

# 2. Parse Act PDFs into structured trees
lawrag ingest act data/raw/ipc_1860.pdf --code IPC --name "Indian Penal Code" --year 1860 --text-as-of 2024-06-30 --source-id SRC-IPC-OFFICIAL --expected data/expected/ipc.txt

lawrag ingest act data/raw/bns_2023.pdf --code BNS --name "Bharatiya Nyaya Sanhita" --year 2023 --text-as-of 2024-07-01 --source-id SRC-BNS-GAZETTE-2023 --expected data/expected/bns.txt

# 3. Extract concordance comparison tables
lawrag ingest concordance data/raw/concordance_table.pdf --source-id SRC-CONCORDANCE-A --ipc-col 0 --bns-col 1

# 4. Reconcile concordance rows against Act trees into review queue
lawrag review queue

# 5. Validate human-reviewed mappings (after editing review/approved_mappings.csv)
lawrag review validate review/approved_mappings.csv

# 6. Build and load graph from processed trees
lawrag load --source processed
```

---

## 5. Setting Up Neo4j (Optional Backend)

By default, lawrag operates with a high-performance in-memory graph store (`MemoryGraphStore`). To use Neo4j Community with native vector index:

```powershell
# 1. Start Neo4j in Docker
docker compose up -d

# 2. Check container health
docker compose ps

# 3. Load data directly into Neo4j
# Set graph.backend: neo4j in a config overlay, then run:
lawrag load -c configs/providers/fake.yaml
```

Neo4j Browser UI is available at `http://localhost:7474` (login `neo4j` / `lawrag-dev-password`).

---

## 6. Running Interactive Q&A, API, and UI

### 6.1 CLI Question Answering (`lawrag ask`)

Run single questions directly through the CLI:

```powershell
# Test offline with scripted fake LLM (instant, deterministic)
lawrag ask "What replaced IPC 498A?" -c providers/fake.yaml

# Test with local Ollama (Llama 3.2)
lawrag ask "What replaced IPC 498A?" -c providers/ollama_local.yaml

# With debug trace (showing retrieval, RRF ranks, and D1-D7 checks)
lawrag ask "What is the punishment for murder under BNS?" --debug
```

### 6.2 Start the FastAPI Backend (`lawrag serve`)

The backend holds singletons (embedder, BM25, graph store) in memory:

```powershell
lawrag serve --port 8000
```
- API Endpoint: `http://127.0.0.1:8000`
- Interactive OpenAPI Docs: `http://127.0.0.1:8000/docs`

### 6.3 Start the Streamlit UI (`lawrag ui`)

In a second terminal window (with venv activated):

```powershell
lawrag ui --port 8501 --api http://localhost:8000
```
Open `http://localhost:8501` to view:
- Legal Q&A with live section citations
- Side-by-side IPC ↔ BNS provision comparisons
- Layered verification badge breakdown (D1–D7 deterministic checks + Claim Judge verdicts)

---

## 7. Running Tests

The test suite validates units, integration flows, and adversarial edge cases:

```powershell
# Run all fast tests (skips slow model downloads and external services)
pytest -m "not slow and not neo4j and not ollama"

# Run complete test suite with coverage
pytest --cov=lawrag
```

Key test categories:
- `tests/unit/`: UID parsing, section splitting, legal tokenizer, BM25, and decision tables.
- `tests/integration/`: LangGraph QA workflow execution paths and API contract tests.
- `tests/adversarial/`: Deterministic safety checks D1–D7 (invented citations, mismatched punishments, false premises).

---

## 8. Running the Evaluation Framework (E1–E3)

The evaluation suite reproduces the experiments defined in the Software Design Document:

```powershell
# 1. Validate the gold dataset against the knowledge graph
lawrag eval validate data/gold/smoke_v0.jsonl

# 2. Lock the test split hash
lawrag eval lock data/gold/smoke_v0.jsonl

# 3. Execute experiment matrix E1 (Retrieval ablations: Vector vs BM25 vs Hybrid RRF vs Graph-assisted)
lawrag eval run e1_retrieval --split dev -c providers/fake.yaml

# 4. Generate consolidated markdown experiment report
lawrag report e1_retrieval
```

Generated reports and figures will be saved in `reports/e1_retrieval/report.md`.

---

## 9. Summary Command Cheatsheet

```powershell
# Setup
py -3.11 -m venv "$HOME\.venvs\lawrag"; & "$HOME\.venvs\lawrag\Scripts\Activate.ps1"
pip install -e ".[dev,ui,eval,ingest,neo4j]"

# Build & Audit
lawrag load
lawrag audit

# Test
pytest -m "not slow and not neo4j"

# Run
lawrag serve            # Terminal 1 (:8000)
lawrag ui               # Terminal 2 (:8501)
```
