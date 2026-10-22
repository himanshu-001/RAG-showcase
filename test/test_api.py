import os

os.environ["TEST_MODE"] = "true"

from fastapi.testclient import TestClient
from app.main import app


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["rag_ready"] is True
        assert data["test_mode"] is True


def test_root():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "Gemma 4 & Gemini Showcase" in response.json()["service"]


def test_retrieve():
    with TestClient(app) as client:
        response = client.post("/retrieve", json={"query": "annual leave", "k": 2})
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) > 0


def test_chat_session():
    with TestClient(app) as client:
        client.delete("/sessions/session_test_1")
        # Turn 1
        resp1 = client.post(
            "/chat",
            json={"question": "What is the annual leave?", "session_id": "session_test_1"},
        )
        assert resp1.status_code == 200
        body1 = resp1.json()
        assert "annual leave" in body1["answer"].lower()
        assert len(body1["sources"]) > 0

        # Check session list
        sess_resp = client.get("/sessions")
        assert sess_resp.status_code == 200
        assert "session_test_1" in sess_resp.json()["sessions"]

        # Check session history
        history_resp = client.get("/sessions/session_test_1")
        assert history_resp.status_code == 200
        messages = history_resp.json()["messages"]
        assert len(messages) == 2  # 1 user + 1 assistant


def test_ollama_chat_model():
    with TestClient(app) as client:
        resp = client.post(
            "/chat",
            json={
                "question": "What is the annual leave allowance?",
                "session_id": "session_ollama_test",
                "model": "qwen2.5:0.5b",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_used"] == "qwen2.5:0.5b"
        assert "annual leave" in data["answer"].lower()


def test_sources_lifecycle():
    with TestClient(app) as client:
        # Reset any custom sources first
        reset_resp = client.delete("/sources")
        assert reset_resp.status_code == 200
        assert reset_resp.json()["is_placeholder"] is True

        # Check sources - should show default placeholder
        src_resp = client.get("/sources")
        assert src_resp.status_code == 200
        data = src_resp.json()
        assert data["is_placeholder"] is True
        assert len(data["sources"]) == 1
        assert "company_policy.md" in data["sources"][0]["name"]

        # Add a custom source directly through the engine
        from app.main import rag_engine
        rag_engine.add_source(
            name="products.json",
            content='[{"id": 1, "name": "Laptop", "price": 999}, {"id": 2, "name": "Mouse", "price": 25}]',
            source_type="file",
        )

        # Now GET /sources should indicate custom sources active (not placeholder)
        src_resp2 = client.get("/sources")
        assert src_resp2.status_code == 200
        data2 = src_resp2.json()
        assert data2["is_placeholder"] is False
        assert any(s["name"] == "products.json" for s in data2["sources"])
        # company_policy.md should NOT be in the active sources
        assert not any("company_policy.md" in s["name"] for s in data2["sources"])

        # Reset to placeholder
        reset_resp2 = client.delete("/sources")
        assert reset_resp2.status_code == 200
        assert reset_resp2.json()["is_placeholder"] is True