# Data sources (skeleton — SDD 3.2.1, Appendix A)

| source_id | Document | Publisher | URL | Retrieved | sha256 | Licence / terms | Status |
|---|---|---|---|---|---|---|---|
| SRC-BNS-GAZETTE-2023 | Bharatiya Nyaya Sanhita, 2023 (Gazette) | Govt. of India | VERIFY | — | — | VERIFY (A-13) | not downloaded |
| SRC-IPC-INDIACODE | Indian Penal Code, 1860 as in force on cutoff | India Code | VERIFY | — | — | VERIFY (A-6) | not downloaded |
| SRC-CONCORDANCE-SEED | Official IPC–BNS correspondence table | VERIFY (A-5) | VERIFY | — | — | VERIFY | placeholder |

Register each file with `lawrag sources register <pdf> --kind ... --authority-rank ... --title ...`; this writes
`data/raw/sources.json` and `data/raw/SOURCES.md`. Raw files are never edited; corrections go to `review/overrides.yaml`.
