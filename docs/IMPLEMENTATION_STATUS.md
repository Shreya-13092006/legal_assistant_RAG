# Implementation status against the SDD

Date: 2026-10-30. "Built" = code + tests exist and pass. "Needs human/data work" = the code path exists but the
milestone cannot be completed without official documents, a reviewer, or measurements.

| Milestone | Built | Needs human / data work |
|---|---|---|
| M0 Environment | venv (Python 3.11, outside OneDrive), `pyproject.toml`, `docker-compose.yml`, `.env.example`, config + hashing, errors, JSON logging, manifests, LLM client (Ollama/Anthropic/OpenAI-compatible/Fake), `lawrag doctor` | `git init` (not done: committing is your call); start Docker + Neo4j and run the skipped `neo4j` test; install torch + bge-m3 and record RAM/throughput; `requirements.lock` |
| M1 Vertical slice | Seed corpus (10 IPC + 9 BNS sections, 10 mappings covering every type and status), uid module, loader, audit, chunker, embedder + cache, analyzer, ID/vector/BM25 retrievers, graph expansion, RRF, context builder, generator, D1–D7, decision table, retry, LangGraph workflow, services, API (`/health /ask /sections /mappings /compare /search /verify`), Streamlit Ask/Map/Compare/About, 14 smoke questions | **Verify the seed text and mappings against official PDFs**; replace `SEED-DRAFT` reviewers |
| M2 Ingestion | PDF parser, splitter, concordance extractor + `normalise_refs`, reconciler → `review/queue.csv`, review gate (`read_approved`), xref extractor, `lawrag ingest/review/load --source processed` | Download + register official PDFs; tune header regexes to the real print (A-10); 30-section spot check; 50 gold concordance rows; human review pass; expected section lists in `data/expected/` |
| M3 Retrieval + eval | All five retrieval variants as overlays; dataset schema/validation/lock/near-duplicate check; resumable runner with manifests; retrieval metrics; `/search`; embedding benchmark matrix | Write and review the pilot gold set (`qa_v1.jsonl`); run the bge-m3 vs bge-small benchmark |
| M4 Verification + experiments | Claim judge with span post-check, decomposition fallback, D5/D6, partial answers, adversarial suite, eval judge, McNemar/Wilcoxon/bootstrap/Holm/κ, report builder, test-split guard | Freeze `docs/ANALYSIS_PLAN.md`; run E1–E3 on test; human grading. **All results PENDING** |
| M5 Optional | OCR metrics + seeded rasterisation + Tesseract wrapper; LLM / cross-encoder reranker; MCP server (`lawrag mcp`) | Install Tesseract; select pages; run studies only if the M4 gate passes |
| M6 Polish | README, UI error states, citation anchors, diff highlights | Demo script, `lawrag reproduce`, deployment decision (D-6) |

## Deliberate deviations (recorded per SDD 0.1)

1. **In-memory graph backend** in addition to Neo4j, behind the same `GraphStore` interface. Tests and the demo
   run without Docker; audit and snapshot hashing are shared. Neo4j remains the target store (DECIDED S6).
2. **BM25** is a small in-house Okapi implementation instead of `bm25s` (corpus < 20k chunks; deterministic).
3. **Default embedder is `hashing-512`**, a lexical stand-in, so everything runs before torch is installed.
   Use `-c embedding/bge_m3.yaml` for any reported run.
4. Retrievers `id_lookup`, `vector`, `bm25_retriever` live together in `retrieval/retrievers.py`.
5. Premise-correction claims ("section N does not exist") are verified deterministically by D7 rather than
   by the LLM judge, because no context block can cite an absence.
6. The hosted adapter uses the official `anthropic` SDK and does not send `temperature` (current models reject
   sampling parameters); determinism for hosted runs comes from the response cache.
