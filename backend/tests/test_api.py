"""Tests for FastAPI API endpoints.

Uses an inline test app that mirrors app.py's API endpoints without the
static file mounting, which requires a frontend directory absent in test
environments.
"""
import pytest
import httpx
import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel
from typing import List, Optional
from unittest.mock import MagicMock


# ──── Inline test app ─────────────────────────────────────────────────────────
# Mirrors the models and endpoint logic from app.py without the static file
# mount so that tests can run without a built frontend directory.

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class SourceItem(BaseModel):
    label: str
    url: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceItem]
    session_id: str


class CourseStats(BaseModel):
    total_courses: int
    course_titles: List[str]


def create_test_app(rag_system) -> FastAPI:
    """Return a minimal FastAPI app with only the API routes (no static files)."""
    test_app = FastAPI()

    @test_app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = rag_system.session_manager.create_session()
            answer, sources = rag_system.query(request.query, session_id)
            return QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except anthropic.AuthenticationError:
            raise HTTPException(
                status_code=500,
                detail="Anthropic API key is missing or invalid. Check your .env file.",
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @test_app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return test_app


# ──── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def rag(mock_rag_system):
    """Expose the shared mock RAGSystem under a short name."""
    return mock_rag_system


@pytest.fixture
def client(mock_rag_system):
    """TestClient backed by the inline test app with a mocked RAGSystem."""
    app = create_test_app(mock_rag_system)
    return TestClient(app)


def _make_auth_error() -> anthropic.AuthenticationError:
    """Construct a real anthropic.AuthenticationError for use in side_effect."""
    mock_response = httpx.Response(
        401,
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
    )
    return anthropic.AuthenticationError(
        message="Invalid API key",
        response=mock_response,
        body={},
    )


# ──── POST /api/query ──────────────────────────────────────────────────────────

class TestQueryEndpoint:
    """Tests for POST /api/query."""

    def test_happy_path_returns_answer_sources_session(self, client, rag):
        """Valid query returns 200 with answer, sources, and session_id."""
        response = client.post("/api/query", json={"query": "What is RAG?"})

        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "Here is the answer."
        assert body["session_id"] == "session-1"
        assert len(body["sources"]) == 1
        assert body["sources"][0]["label"] == "Intro to RAG - Lesson 1"
        assert body["sources"][0]["url"] == "https://example.com/lesson/1"

    def test_creates_session_when_none_provided(self, client, rag):
        """Omitting session_id causes session_manager.create_session() to be called."""
        client.post("/api/query", json={"query": "What is RAG?"})

        rag.session_manager.create_session.assert_called_once()

    def test_skips_session_creation_when_id_provided(self, client, rag):
        """Providing session_id bypasses create_session() entirely."""
        client.post(
            "/api/query",
            json={"query": "What is RAG?", "session_id": "existing-session"},
        )

        rag.session_manager.create_session.assert_not_called()

    def test_provided_session_id_forwarded_to_query(self, client, rag):
        """The caller-supplied session_id is passed through to rag_system.query()."""
        client.post(
            "/api/query",
            json={"query": "What is RAG?", "session_id": "existing-session"},
        )

        rag.query.assert_called_once_with("What is RAG?", "existing-session")

    def test_session_id_echoed_in_response(self, client, rag):
        """The session_id used for the query is returned in the response body."""
        response = client.post(
            "/api/query",
            json={"query": "What is RAG?", "session_id": "my-session"},
        )

        assert response.json()["session_id"] == "my-session"

    def test_missing_query_field_returns_422(self, client):
        """Omitting the required `query` field returns HTTP 422."""
        response = client.post("/api/query", json={})

        assert response.status_code == 422

    def test_empty_sources_list(self, client, rag):
        """A query with no search hits returns an empty sources list."""
        rag.query.return_value = ("Nothing found.", [])

        response = client.post("/api/query", json={"query": "Unknown topic"})

        assert response.status_code == 200
        assert response.json()["sources"] == []

    def test_authentication_error_returns_500_with_key_message(self, client, rag):
        """An Anthropic auth failure maps to HTTP 500 with a descriptive message."""
        rag.query.side_effect = _make_auth_error()

        response = client.post("/api/query", json={"query": "What is RAG?"})

        assert response.status_code == 500
        assert "API key" in response.json()["detail"]

    def test_generic_exception_returns_500_with_detail(self, client, rag):
        """Unexpected exceptions are caught and surfaced as HTTP 500."""
        rag.query.side_effect = RuntimeError("vector store unavailable")

        response = client.post("/api/query", json={"query": "What is RAG?"})

        assert response.status_code == 500
        assert "vector store unavailable" in response.json()["detail"]


# ──── GET /api/courses ─────────────────────────────────────────────────────────

class TestCoursesEndpoint:
    """Tests for GET /api/courses."""

    def test_happy_path_returns_course_stats(self, client, rag):
        """Returns 200 with total_courses count and course_titles list."""
        response = client.get("/api/courses")

        assert response.status_code == 200
        body = response.json()
        assert body["total_courses"] == 2
        assert body["course_titles"] == ["Intro to RAG", "Advanced ML"]

    def test_delegates_to_get_course_analytics(self, client, rag):
        """Verifies the endpoint calls rag_system.get_course_analytics()."""
        client.get("/api/courses")

        rag.get_course_analytics.assert_called_once()

    def test_empty_catalog(self, client, rag):
        """Returns zeros and an empty list when no courses are loaded."""
        rag.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": [],
        }

        response = client.get("/api/courses")

        assert response.status_code == 200
        body = response.json()
        assert body["total_courses"] == 0
        assert body["course_titles"] == []

    def test_analytics_error_returns_500(self, client, rag):
        """Exceptions from get_course_analytics are surfaced as HTTP 500."""
        rag.get_course_analytics.side_effect = RuntimeError("ChromaDB unavailable")

        response = client.get("/api/courses")

        assert response.status_code == 500
        assert "ChromaDB unavailable" in response.json()["detail"]
