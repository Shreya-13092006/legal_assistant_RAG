# Gold-set annotation guide (SDD 7.2)

Schema: `src/lawrag/models/eval.py::GoldItem` (one JSON object per line in `data/gold/qa_v1.jsonl`).

* Categories and target counts: SDD 7.2.1. Expected behaviours: `answer`, `partial_ok`, `abstain`,
  `correct_premise`, `state_dispute`, `state_unresolved`.
* Every `answer` / `correct_premise` / `state_*` item lists evidence (source + page; concordance row for mappings).
* `gold_sections` relevance: 2 = primary, 1 = supporting. Use section uids unless the question targets a sub-section.
* Mapping items reference `mapping_id`s that exist in `review/approved_mappings.csv` with the stated type/status.
* Abstain items need `abstain_reason`; false-premise items need `trap_type`.
* Don't copy provision text verbatim into questions. LLM-drafted questions (`drafting_aid: llm_draft`) ≤ 20%.
* `review_status: verified` is required for inclusion. Pilot items → `split: dev`; expansion → `split: test`,
  then `lawrag eval lock` before any M4 tuning.
* `lawrag eval validate` checks uids, mapping consistency, sources, the LLM-draft cap and dev/test near-duplicates.
