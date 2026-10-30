"""Tokenizer, BM25, chunker and embedder cache (SDD 5.1 #8-#10)."""
import numpy as np
import pytest

from lawrag.config import ChunkCfg, EmbeddingCfg
from lawrag.index.bm25 import BM25Index
from lawrag.index.chunker import chunk_tree, count_tokens, split_text
from lawrag.index.embedder import make_embedder
from lawrag.index.tokenizer import LegalTokenizer


@pytest.mark.parametrize("text,expected", [
    ("section 304A", ["304a", "section"]),
    ("376(2)(a)", ["376", "376_2", "376_2_a"]),
    ("120-B", ["120b"]),
    ("not without unless except", ["not", "without", "unless", "except"]),
    ("IPC BNS", ["ipc", "bns"]),
])
def test_tokenizer(text, expected):
    toks = LegalTokenizer()(text)
    for e in expected:
        assert e in toks


def test_bm25_ranking_and_persistence(tmp_path):
    idx = BM25Index().build([("a", "punishment for murder death"), ("b", "cheating property"),
                             ("c", "murder murder culpable homicide")], key="k")
    hits = idx.search("murder", 3)
    assert [h[0] for h in hits][:2] == ["c", "a"]
    assert idx.search("zzz", 3) == []
    idx.save(tmp_path)
    again = BM25Index.load(tmp_path)
    assert again.search("murder", 3) == hits


def test_split_text_caps():
    text = " ".join(f"Sentence number {i} ends here." for i in range(200))
    parts = split_text(text, 50)
    assert all(count_tokens(p) <= 50 for p in parts)
    assert " ".join(parts).split() == text.split()


def test_chunker_seed(seed_data):
    from lawrag.models.legal import ActTree
    acts = {a.code: a for a in seed_data.acts}
    tree = ActTree(act=acts["BNS"], chapters=[c for c in seed_data.chapters if c.act == "BNS"],
                   sections=[s for s in seed_data.sections if s.act == "BNS"],
                   provisions=[p for p in seed_data.provisions if p.section_uid.startswith("BNS")])
    chunks = chunk_tree(tree, ChunkCfg(max_tokens=50))
    provs = {p.uid: p for p in tree.provisions}
    for c in chunks:
        if c.provision_uid:
            assert c.section_uid == provs[c.provision_uid].section_uid  # no boundary crossing
            assert c.embed_text.startswith(f"[BNS s.")
            assert c.chunk_id.startswith(c.provision_uid + "#text#c")
    long = [c for c in chunks if c.n_parts > 1]
    assert long and all(c.provision_uid for c in long)  # parts keep the citation uid
    no_ill = chunk_tree(tree, ChunkCfg(include_illustrations=False))
    assert not any(c.kind == "illustration" for c in no_ill)


def test_embedder_cache(tmp_path):
    emb = make_embedder(EmbeddingCfg(model="hashing-64"), tmp_path)
    v1 = emb.encode(["murder", "cheating"])
    assert emb.misses == 2
    v2 = emb.encode(["murder", "cheating"])
    assert emb.hits == 2 and np.allclose(v1, v2)
    emb2 = make_embedder(EmbeddingCfg(model="hashing-64"), tmp_path)
    emb2.encode(["murder"])
    assert emb2.hits == 1  # persisted
    assert np.allclose(np.linalg.norm(v1, axis=1), 1.0, atol=1e-5)
