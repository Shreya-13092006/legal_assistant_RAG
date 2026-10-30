import random

import pytest

from lawrag.legal import uid as U

VALID = ["IPC:302", "IPC:498A", "IPC:120B", "BNS:103(1)", "BNS:2(1)(a)", "IPC:300:Exc1", "IPC:300:Expl2",
         "IPC:300:Ill(a)", "IPC:376:Prov1", "IPC:302:body", "BNS:106(1):Expl1", "BNS:2(1)(a)(ii)"]
INVALID = ["ipc:302", "IPC302", "XYZ:1", "IPC:", "IPC:302(", "IPC:302:Foo", "BNS:103(1)(A)", "IPC:-1"]


@pytest.mark.parametrize("u", VALID)
def test_valid_roundtrip(u):
    assert U.is_valid(u)
    assert U.format_uid(U.parse_uid(u)) == u


@pytest.mark.parametrize("u", INVALID)
def test_invalid(u):
    assert not U.is_valid(u)


@pytest.mark.parametrize("raw,norm", [("120-B", "120B"), ("498 a", "498A"), (" 302 ", "302"), ("304–A", "304A")])
def test_normalize_number(raw, norm):
    assert U.normalize_number(raw) == norm


def test_property_roundtrip():
    rng = random.Random(13)
    for _ in range(500):
        num = str(rng.randint(1, 511)) + rng.choice(["", "A", "B", "AA"])
        path = tuple([str(rng.randint(1, 9))] if rng.random() < 0.5 else []) + tuple(
            rng.choice("abcdef") for _ in range(rng.randint(0, 1)))
        qual = rng.choice([None, "Exc1", "Expl2", "Prov1", "Ill(b)", "body"])
        if qual == "body":
            path = ()
        u = U.format_uid(U.UidParts(rng.choice(["IPC", "BNS"]), num, path, qual))
        assert U.is_valid(u), u
        assert U.format_uid(U.parse_uid(u)) == u


def test_section_of_and_ancestors():
    assert U.section_of("BNS:2(1)(a)") == "BNS:2"
    assert U.ancestors("BNS:2(1)(a)") == ["BNS:2(1)", "BNS:2"]
    assert U.ancestors("IPC:300:Exc1") == ["IPC:300"]
    assert U.section_of("MAP-0004") == "MAP-0004"
