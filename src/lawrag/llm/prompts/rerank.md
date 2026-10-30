# ROLE: reranker
Rank the CANDIDATES by how directly they help answer the QUESTION. Use only the candidate text.

Reply with ONLY a JSON object: {"order": ["<candidate uid>", "..."]} listing every candidate uid once, most relevant first.
