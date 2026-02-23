"""
Integration tests for RAGSystem.query() - tests the full pipeline with all
components mocked (no real ChromaDB, no real Anthropic API calls).

Also includes regression tests for the ChromaDB None-metadata bug using
an in-memory ephemeral ChromaDB (no server required).
"""
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_config():
    """Minimal config object for RAGSystem."""
    config = MagicMock()
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.MAX_RESULTS = 5
    config.MAX_HISTORY = 2
    config.CHROMA_PATH = ":memory:"
    config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    config.ANTHROPIC_API_KEY = "sk-ant-test-key"
    config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
    return config


@pytest.fixture
def rag_system_mocked(mock_config):
    """
    RAGSystem with VectorStore, AIGenerator, and SessionManager all mocked.
    Yields (system, mock_ai_generator, mock_vector_store).
    """
    with (
        patch("rag_system.VectorStore") as MockVS,
        patch("rag_system.AIGenerator") as MockAI,
        patch("rag_system.DocumentProcessor"),
        patch("rag_system.SessionManager") as MockSM,
    ):
        from vector_store import SearchResults

        # Configure mock VectorStore
        mock_vs_instance = MockVS.return_value
        mock_vs_instance.search.return_value = SearchResults(
            documents=["RAG combines retrieval with generation."],
            metadata=[{"course_title": "Intro to RAG", "lesson_number": 1, "chunk_index": 0}],
            distances=[0.1],
        )
        mock_vs_instance.get_lesson_link.return_value = "https://example.com/lesson/1"
        mock_vs_instance.get_existing_course_titles.return_value = []
        mock_vs_instance.get_all_courses_metadata.return_value = []
        mock_vs_instance._resolve_course_name.return_value = None

        # Configure mock AIGenerator
        mock_ai_instance = MockAI.return_value
        mock_ai_instance.generate_response.return_value = "RAG is a technique for grounding LLMs."

        # Configure mock SessionManager
        mock_sm_instance = MockSM.return_value
        mock_sm_instance.get_conversation_history.return_value = None
        mock_sm_instance.create_session.return_value = "session_1"

        from rag_system import RAGSystem
        system = RAGSystem(mock_config)

        yield system, mock_ai_instance, mock_vs_instance


class TestRAGSystemQuery:
    """RAGSystem.query() return values and data flow."""

    def test_query_returns_tuple_of_response_and_sources(self, rag_system_mocked):
        """query() returns a (str, list) tuple."""
        system, _, _ = rag_system_mocked
        result = system.query("what is RAG?")

        assert isinstance(result, tuple)
        assert len(result) == 2
        response, sources = result
        assert isinstance(response, str)
        assert isinstance(sources, list)

    def test_query_returns_ai_generator_response(self, rag_system_mocked):
        """The string in the return tuple is exactly what AIGenerator.generate_response() returned."""
        system, mock_ai, _ = rag_system_mocked
        mock_ai.generate_response.return_value = "This is the AI answer."
        response, _ = system.query("anything")

        assert response == "This is the AI answer."

    def test_query_calls_ai_generator_with_tool_manager(self, rag_system_mocked):
        """AIGenerator.generate_response() is called with tools and tool_manager."""
        system, mock_ai, _ = rag_system_mocked
        system.query("what is RAG?")

        call_kwargs = mock_ai.generate_response.call_args[1]
        assert "tools" in call_kwargs
        assert "tool_manager" in call_kwargs
        assert call_kwargs["tools"] is not None
        assert call_kwargs["tool_manager"] is not None

    def test_sources_are_reset_after_each_query(self, rag_system_mocked):
        """After query() returns, tool_manager sources are cleared for the next query."""
        system, _, _ = rag_system_mocked

        system.tool_manager.get_last_sources = MagicMock(
            return_value=[{"label": "Course A - Lesson 1", "url": "https://example.com"}]
        )
        system.tool_manager.reset_sources = MagicMock()

        system.query("first query")
        system.tool_manager.reset_sources.assert_called_once()

    def test_session_history_retrieved_when_session_id_provided(self, rag_system_mocked):
        """When a session_id is given, conversation history is fetched from SessionManager."""
        system, mock_ai, _ = rag_system_mocked
        system.session_manager.get_conversation_history.return_value = "User: hi\nAssistant: hello"

        system.query("follow-up question", session_id="session_1")

        system.session_manager.get_conversation_history.assert_called_once_with("session_1")
        call_kwargs = mock_ai.generate_response.call_args[1]
        assert call_kwargs["conversation_history"] is not None

    def test_no_session_history_when_no_session_id(self, rag_system_mocked):
        """When session_id is None, conversation_history is not passed to AIGenerator."""
        system, mock_ai, _ = rag_system_mocked
        system.query("standalone question", session_id=None)

        call_kwargs = mock_ai.generate_response.call_args[1]
        assert call_kwargs.get("conversation_history") is None

    def test_session_updated_after_query(self, rag_system_mocked):
        """After a successful query, the exchange is saved to the session."""
        system, mock_ai, _ = rag_system_mocked
        mock_ai.generate_response.return_value = "The answer."

        system.query("my question", session_id="session_1")
        system.session_manager.add_exchange.assert_called_once_with(
            "session_1", "my question", "The answer."
        )

    def test_query_propagates_ai_generator_exception(self, rag_system_mocked):
        """If AIGenerator raises, the exception propagates out of query() (no silent swallowing)."""
        import anthropic
        system, mock_ai, _ = rag_system_mocked
        mock_ai.generate_response.side_effect = anthropic.AuthenticationError(
            message="Invalid key", response=MagicMock(status_code=401), body={}
        )

        with pytest.raises(anthropic.AuthenticationError):
            system.query("what is RAG?")


class TestRAGSystemSourcesFlow:
    """Verify that sources flow correctly from tool execution back to the caller."""

    def test_sources_from_search_tool_are_returned(self, rag_system_mocked):
        """Sources collected by ToolManager are returned as part of query() output."""
        system, _, _ = rag_system_mocked

        expected_sources = [{"label": "Intro to RAG - Lesson 1", "url": "https://example.com"}]
        system.search_tool.last_sources = expected_sources

        _, sources = system.query("what is RAG?")
        assert sources == expected_sources

    def test_sources_empty_when_no_tool_called(self, rag_system_mocked):
        """When AI answers directly without using a tool, sources list is empty."""
        system, _, _ = rag_system_mocked

        system.tool_manager.reset_sources()
        _, sources = system.query("what is 2+2?")
        assert sources == []


class TestVectorStoreNoneMetadataBug:
    """
    Regression test for Bug #1: lesson_number=None crashes ChromaDB.

    These tests use real ephemeral ChromaDB (in-memory, no server required)
    to confirm the exact failure mode and verify the fix.
    """

    def test_adding_chunk_with_none_lesson_number_raises_error(self):
        """
        ChromaDB raises an error when lesson_number=None is in metadata.
        This documents the bug in vector_store.add_course_content().
        """
        import chromadb
        from models import CourseChunk

        client = chromadb.EphemeralClient()
        col = client.create_collection("test_none_bug")

        chunk = CourseChunk(
            content="Content without a lesson",
            course_title="My Course",
            lesson_number=None,  # BUG: None is not a valid ChromaDB metadata value
            chunk_index=0,
        )

        with pytest.raises(Exception):  # TypeError or ValueError depending on chromadb version
            col.add(
                documents=[chunk.content],
                ids=["id_0"],
                metadatas=[{
                    "course_title": chunk.course_title,
                    "lesson_number": chunk.lesson_number,  # None -> error
                    "chunk_index": chunk.chunk_index,
                }],
            )

    def test_adding_chunk_with_sentinel_lesson_number_succeeds(self):
        """
        The fix: replacing None with -1 allows ChromaDB to accept the metadata.
        """
        import chromadb

        client = chromadb.EphemeralClient()
        col = client.create_collection("test_sentinel_fix")

        # After the fix, None is replaced with -1 before calling .add()
        col.add(
            documents=["Content without a lesson"],
            ids=["id_0"],
            metadatas=[{
                "course_title": "My Course",
                "lesson_number": -1,  # FIX: sentinel value instead of None
                "chunk_index": 0,
            }],
        )
        results = col.get(ids=["id_0"])
        assert results["metadatas"][0]["lesson_number"] == -1

    def test_vector_store_add_course_content_succeeds_with_none_lesson(self):
        """
        Regression test (post-fix): VectorStore.add_course_content() must NOT raise
        when a chunk has lesson_number=None. The fix converts None to the sentinel -1
        before passing to ChromaDB, so the add should succeed and the stored value is -1.
        """
        import chromadb
        from models import CourseChunk
        from unittest.mock import patch, MagicMock

        # Use ephemeral client for this test
        ephemeral_client = chromadb.EphemeralClient()

        with patch("vector_store.chromadb.PersistentClient", return_value=ephemeral_client):
            with patch("vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction") as mock_ef:
                mock_ef.return_value = MagicMock()
                from vector_store import VectorStore
                vs = VectorStore.__new__(VectorStore)
                vs.max_results = 5
                vs.client = ephemeral_client
                vs.embedding_function = mock_ef.return_value
                vs.course_catalog = vs.client.get_or_create_collection("test_catalog_fixed")
                vs.course_content = vs.client.get_or_create_collection("test_content_fixed")

        chunk_with_none = CourseChunk(
            content="Some content",
            course_title="Test Course",
            lesson_number=None,  # was a bug; fix converts to -1
            chunk_index=0,
        )

        # After the fix this must not raise
        vs.add_course_content([chunk_with_none])

        # Verify the sentinel value -1 was stored
        stored = vs.course_content.get(ids=["Test_Course_0"])
        assert stored["metadatas"][0]["lesson_number"] == -1
