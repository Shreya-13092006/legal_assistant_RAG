# Mapping review guide (SDD 3.2.6)

Work from `review/queue.csv` (written by `lawrag review queue`); copy reviewed rows to `review/approved_mappings.csv`.

* **verified** — at least one cited source row states this correspondence and the type is right. Fill
  `reviewed_by`, `reviewed_on` (YYYY-MM-DD), and `change_summary` (your words or a quote from the source).
* **disputed** — sources conflict, or a source contradicts the text comparison. Explain in `dispute_note`.
  The system will describe the disagreement and never pick a side.
* **unresolved** — evidence is insufficient to decide. This is *not* "no equivalent".
* **rejected** — wrong pairing (kept for the audit; never shown in answers).
* **OMITTED / NEW** need a source row that explicitly says deleted / new. A missing row is not evidence.
* **PARTIAL** — mark partial endpoints with a trailing `~` in `ipc_uids` / `bns_uids` (e.g. `IPC:124A~`).
* Bulk approval is allowed for rows where `sources_agree = True` and `text_similarity >= 0.97`; each row still
  carries your identity.
* An LLM may never propose pairs, types or statuses.

`lawrag review validate` rejects rows with a missing reviewer/date/status, unknown uids, unregistered sources,
or `verified` without evidence.
