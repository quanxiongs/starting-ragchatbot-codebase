"""Shared fixtures for all tests."""
import sys
import os
import pytest
from unittest.mock import MagicMock

# Make backend modules importable from tests/
_tests_dir = os.path.dirname(__file__)
_backend_dir = os.path.join(_tests_dir, "..")
sys.path.insert(0, _backend_dir)
sys.path.insert(0, _tests_dir)


# ── VectorStore mock ──────────────────────────────────────────────────────────

@pytest.fixture
def mock_vector_store():
    """Return a MagicMock configured to behave like VectorStore."""
    store = MagicMock()
    store.get_lesson_link.return_value = "https://example.com/lesson/1"
    store.get_all_courses_metadata.return_value = []
    store._resolve_course_name.return_value = None
    return store


@pytest.fixture
def mock_vector_store_with_results(mock_vector_store):
    """VectorStore mock pre-loaded with two search results."""
    from vector_store import SearchResults
    mock_vector_store.search.return_value = SearchResults(
        documents=[
            "Lesson 1 content: This is about RAG systems.",
            "Retrieval augmented generation combines search with LLMs.",
        ],
        metadata=[
            {"course_title": "Intro to RAG", "lesson_number": 1, "chunk_index": 0},
            {"course_title": "Intro to RAG", "lesson_number": 1, "chunk_index": 1},
        ],
        distances=[0.1, 0.2],
    )
    mock_vector_store.get_lesson_link.return_value = "https://example.com/lesson/1"
    return mock_vector_store


@pytest.fixture
def mock_vector_store_empty(mock_vector_store):
    """VectorStore mock that returns empty results."""
    from vector_store import SearchResults
    mock_vector_store.search.return_value = SearchResults(
        documents=[], metadata=[], distances=[]
    )
    return mock_vector_store


@pytest.fixture
def mock_vector_store_error(mock_vector_store):
    """VectorStore mock that returns a search error."""
    from vector_store import SearchResults
    mock_vector_store.search.return_value = SearchResults.empty(
        "Search error: collection not found"
    )
    return mock_vector_store


# ── Anthropic client mock ─────────────────────────────────────────────────────

@pytest.fixture
def mock_anthropic_client():
    """Return a mock Anthropic client."""
    return MagicMock()


# ── AIGenerator fixture ───────────────────────────────────────────────────────

@pytest.fixture
def ai_generator(mock_anthropic_client):
    """AIGenerator with mocked Anthropic client (no real API calls)."""
    from ai_generator import AIGenerator
    gen = AIGenerator.__new__(AIGenerator)
    gen.client = mock_anthropic_client
    gen.model = "claude-sonnet-4-20250514"
    gen.base_params = {
        "model": gen.model,
        "temperature": 0,
        "max_tokens": 800,
    }
    return gen


# ── ToolManager fixture ───────────────────────────────────────────────────────

@pytest.fixture
def tool_manager_with_search(mock_vector_store_with_results):
    """ToolManager with a real CourseSearchTool backed by a mocked VectorStore."""
    from search_tools import ToolManager, CourseSearchTool
    manager = ToolManager()
    search_tool = CourseSearchTool(mock_vector_store_with_results)
    manager.register_tool(search_tool)
    return manager, search_tool


@pytest.fixture
def tool_manager_with_both_tools(mock_vector_store_with_results):
    """ToolManager with both CourseSearchTool and CourseOutlineTool for multi-round tests."""
    from search_tools import ToolManager, CourseSearchTool, CourseOutlineTool

    outline_store = MagicMock()
    outline_store._resolve_course_name.return_value = "Intro to RAG"
    outline_store.get_all_courses_metadata.return_value = [{
        "title": "Intro to RAG",
        "course_link": "https://example.com/rag",
        "lessons": [
            {"lesson_number": 1, "lesson_title": "What is RAG?"},
            {"lesson_number": 2, "lesson_title": "Vector Stores"},
        ],
    }]

    manager = ToolManager()
    search_tool = CourseSearchTool(mock_vector_store_with_results)
    outline_tool = CourseOutlineTool(outline_store)
    manager.register_tool(search_tool)
    manager.register_tool(outline_tool)
    return manager, search_tool, outline_tool
