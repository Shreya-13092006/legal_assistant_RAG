"""API contract and error cases with fake dependencies (SDD 5.1 #23)."""
import pytest
from fastapi.testclient import TestClient

from lawrag.api.app import create_app


@pytest.fixture()
def client(deps):
    with TestClient(create_app(deps)) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["graph"]["backend"] == "memory"


def test_ask(client):
    r = client.post("/ask", json={"question": "What replaced IPC 498A?"})
    body = r.json()
    assert r.status_code == 200 and body["outcome"] == "answered"
    assert body["mapping_facts"][0]["mapping_id"] == "MAP-0004" and body["trace"] is None
    r = client.post("/ask", json={"question": "What replaced IPC 498A?", "options": {"debug": True}})
    assert r.json()["trace"]


def test_ask_validation(client):
    r = client.post("/ask", json={"question": ""})
    assert r.status_code == 422 and r.json()["error"]["code"] == "VALIDATION_ERROR"
    assert client.post("/ask", json={"question": "x" * 1001}).status_code == 422


def test_sections(client):
    r = client.get("/sections/BNS:106")
    assert r.status_code == 200 and {p["uid"] for p in r.json()["provisions"]} >= {"BNS:106(1)", "BNS:106:Expl1"}
    assert client.get("/sections/BNS:999").status_code == 404
    assert client.get("/sections/not-a-uid").json()["error"]["code"] == "MALFORMED_UID"


def test_mappings(client):
    body = client.get("/mappings/IPC:498A").json()
    assert body["direction"] == "ipc_to_bns" and body["mappings"][0]["type"] == "SPLIT"
    body = client.get("/mappings/BNS:100").json()
    assert body["mappings"][0]["mapping_id"] == "MAP-0001"
    body = client.get("/mappings/IPC:377").json()
    assert body["mappings"][0]["status"] == "unresolved"


def test_compare(client):
    body = client.post("/compare", json={"left_uid": "IPC:304A"}).json()
    assert body["pairs"][0]["right"]["uid"] == "BNS:106"
    assert any(s["op"] != "equal" for s in body["pairs"][0]["diff_segments"])
    assert client.post("/compare", json={"left_uid": "IPC:377"}).status_code == 409


def test_search_and_verify(client):
    hits = client.post("/search", json={"query": "punishment for murder", "mode": "hybrid", "k": 5}).json()
    assert hits["candidates"]
    v = client.post("/verify", json={"answer": "See [BNS:999].", "context_uids": ["BNS:103(1)"]}).json()
    checks = {c["check"]: c["passed"] for c in v["deterministic"]["checks"]}
    assert checks["D1"] is False
