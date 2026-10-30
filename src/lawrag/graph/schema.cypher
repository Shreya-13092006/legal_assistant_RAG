// Schema DDL (SDD 3.1.5). Statements are separated by ';'. The vector index statement is
// templated by the loader with {index}, {label} and {dim} for the active embedding model.
CREATE CONSTRAINT act_code    IF NOT EXISTS FOR (a:Act)       REQUIRE a.code IS UNIQUE;
CREATE CONSTRAINT chapter_uid IF NOT EXISTS FOR (c:Chapter)   REQUIRE c.uid IS UNIQUE;
CREATE CONSTRAINT section_uid IF NOT EXISTS FOR (s:Section)   REQUIRE s.uid IS UNIQUE;
CREATE CONSTRAINT prov_uid    IF NOT EXISTS FOR (p:Provision) REQUIRE p.uid IS UNIQUE;
CREATE CONSTRAINT chunk_id    IF NOT EXISTS FOR (c:Chunk)     REQUIRE c.chunk_id IS UNIQUE;
CREATE CONSTRAINT mapping_id  IF NOT EXISTS FOR (m:Mapping)   REQUIRE m.mapping_id IS UNIQUE;
CREATE CONSTRAINT source_id   IF NOT EXISTS FOR (s:Source)    REQUIRE s.source_id IS UNIQUE;
CREATE INDEX section_act_num IF NOT EXISTS FOR (s:Section)   ON (s.act, s.number);
CREATE INDEX prov_section    IF NOT EXISTS FOR (p:Provision) ON (p.section_uid);
CREATE INDEX mapping_status  IF NOT EXISTS FOR (m:Mapping)   ON (m.status, m.type);
CREATE FULLTEXT INDEX section_title_ft IF NOT EXISTS FOR (s:Section) ON EACH [s.title];
