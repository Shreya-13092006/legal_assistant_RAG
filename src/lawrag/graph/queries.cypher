// Named, parameterized queries (SDD 3: "Every Cypher query is named and parameterized").
// Format: a line "// name: <query_name>" starts a query; it ends at the next name line.
// User text is NEVER interpolated into these strings.

// name: ping
RETURN 1 AS ok

// name: section_ids
MATCH (s:Section) RETURN s.uid AS uid

// name: node_exists
OPTIONAL MATCH (s:Section {uid: $uid}) OPTIONAL MATCH (p:Provision {uid: $uid})
RETURN (s IS NOT NULL OR p IS NOT NULL) AS exists

// name: get_act
MATCH (a:Act {code: $code}) RETURN a {.*} AS act

// name: get_section
MATCH (s:Section {uid: $uid}) RETURN s {.*} AS section

// name: get_provision
MATCH (p:Provision {uid: $uid}) RETURN p {.*} AS provision

// name: provisions_of
MATCH (p:Provision {section_uid: $section_uid}) RETURN p {.*} AS provision ORDER BY p.order

// name: qualifiers_of
MATCH (q:Provision)-[:QUALIFIES]->(t) WHERE t.uid IN $targets
RETURN DISTINCT q {.*} AS provision ORDER BY q.order

// name: mappings_for
MATCH (m:Mapping)
WHERE EXISTS { MATCH (n)-[:MAPPED_IN]->(m) WHERE n.uid IN $uids }
   OR EXISTS { MATCH (m)-[:MAPS_TO]->(n) WHERE n.uid IN $uids }
OPTIONAL MATCH (f)-[fi:MAPPED_IN]->(m)
WITH m, collect(DISTINCT {uid: f.uid, role: fi.role}) AS froms
OPTIONAL MATCH (m)-[ti:MAPS_TO]->(t)
WITH m, froms, collect(DISTINCT {uid: t.uid, role: ti.role}) AS tos
OPTIONAL MATCH (m)-[e:EVIDENCED_BY]->(src:Source)
RETURN m {.*} AS mapping, froms, tos,
       collect(DISTINCT {source_id: src.source_id, page: e.page, row: e.row, raw_row: e.raw_row,
                         extractor: e.extractor}) AS evidence
ORDER BY m.mapping_id

// name: get_mapping
MATCH (m:Mapping {mapping_id: $mapping_id})
OPTIONAL MATCH (f)-[fi:MAPPED_IN]->(m)
WITH m, collect(DISTINCT {uid: f.uid, role: fi.role}) AS froms
OPTIONAL MATCH (m)-[ti:MAPS_TO]->(t)
WITH m, froms, collect(DISTINCT {uid: t.uid, role: ti.role}) AS tos
OPTIONAL MATCH (m)-[e:EVIDENCED_BY]->(src:Source)
RETURN m {.*} AS mapping, froms, tos,
       collect(DISTINCT {source_id: src.source_id, page: e.page, row: e.row, raw_row: e.raw_row,
                         extractor: e.extractor}) AS evidence

// name: refers_to
MATCH (a {uid: $uid})-[:REFERS_TO]->(b) RETURN DISTINCT b.uid AS uid ORDER BY uid

// name: succeeded_by
MATCH (a)-[r:SUCCEEDED_BY]->(b) WHERE a.uid IN $uids
RETURN b.uid AS uid, r.mapping_id AS mapping_id, r.type AS type

// name: get_source
MATCH (s:Source {source_id: $source_id}) RETURN s {.*} AS source

// name: get_chunk
MATCH (c:Chunk {chunk_id: $chunk_id}) RETURN c {.*, embedding: null} AS chunk

// name: chunks_for
MATCH (c:Chunk) WHERE c.provision_uid = $uid OR (c.provision_uid IS NULL AND c.section_uid = $uid)
RETURN c {.*, embedding: null} AS chunk ORDER BY c.part

// name: all_chunks
MATCH (c:Chunk) WHERE $label IS NULL OR $label IN labels(c)
RETURN c {.*, embedding: null} AS chunk ORDER BY c.chunk_id

// name: secondary_discussing
MATCH (c:SecondaryChunk)-[:DISCUSSES]->(n) WHERE n.uid IN $uids
RETURN DISTINCT c {.*, embedding: null} AS chunk ORDER BY chunk.chunk_id

// name: vector_search
CALL db.index.vector.queryNodes($index, $k, $vec) YIELD node, score
WHERE $act IS NULL OR node.act = $act
RETURN node {.*, embedding: null} AS chunk, score

// name: derive_succeeded_by
MATCH ()-[r:SUCCEEDED_BY]->() DELETE r
WITH count(*) AS _
MATCH (a)-[:MAPPED_IN]->(m:Mapping {status: 'verified'})-[:MAPS_TO]->(b)
MERGE (a)-[:SUCCEEDED_BY {mapping_id: m.mapping_id, type: m.type}]->(b)

// name: export_nodes
MATCH (n) WHERE n:Act OR n:Chapter OR n:Section OR n:Provision OR n:Chunk OR n:Mapping OR n:Source
RETURN labels(n) AS labels, n {.*} AS props

// name: export_edges
MATCH (a)-[r]->(b)
RETURN type(r) AS type, coalesce(a.uid, a.code, a.mapping_id, a.chunk_id, a.source_id) AS src,
       coalesce(b.uid, b.code, b.mapping_id, b.chunk_id, b.source_id) AS dst, r {.*} AS props

// name: wipe
MATCH (n) DETACH DELETE n
