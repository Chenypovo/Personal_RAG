import os

import pytest

# The server module builds a real agent at import time unless told to skip it;
# tests inject a fake bundle, so suppress the import-time build (needs no creds).
os.environ["SKIP_APP_BUILD"] = "1"

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from app.agent.agent import AgentResult  # noqa: E402
from app.agent.router import RouteDecision  # noqa: E402
from app.api.server import create_app  # noqa: E402


class _FakeAgent:
    def chat(self, message: str) -> AgentResult:
        return AgentResult(
            answer=f"echo: {message}",
            sources=[{"citation": "Chunk 0, doc.txt"}],
            recalled_memories=[],
            memory_ops=[],
            route=RouteDecision(
                use_docs=True, use_memory=True, write_memory=False, rewritten_query=message
            ),
        )


class _FakeStore:
    def list_active(self):
        return []


class _FakeBundle:
    def __init__(self):
        self.agent = _FakeAgent()
        self.store = _FakeStore()
        self.saved = 0

    def save(self):
        self.saved += 1


def _client(bundle):
    return TestClient(create_app(bundle=bundle))


def test_chat_route_returns_answer_and_persists():
    bundle = _FakeBundle()
    resp = _client(bundle).post("/api/chat", json={"message": "hello"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "echo: hello"
    assert body["route"]["rewritten_query"] == "hello"
    assert bundle.saved == 1  # memory persisted after the turn


def test_memories_route_lists_active_facts():
    resp = _client(_FakeBundle()).get("/api/memories")
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 0
    assert body["memories"] == []
