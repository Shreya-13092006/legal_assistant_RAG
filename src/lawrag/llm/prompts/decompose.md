# ROLE: decomposer
Split the ANSWER into atomic claims (one fact per claim). Keep the citation labels written in square brackets next to each fact (for example [BNS:103(1)] or [MAP-0004]). Do not add facts.

Reply with ONLY a JSON object:
{"claims": [{"text": "<atomic fact>", "citations": ["<label>"], "core": true}]}
