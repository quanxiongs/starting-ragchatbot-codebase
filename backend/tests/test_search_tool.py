"""
Tests for CourseSearchTool and ToolManager.

These tests use a mocked VectorStore - no real ChromaDB required.
"""
import pytest
from unittest.mock import MagicMock


class TestCourseSearchToolBasicSearch:
    """CourseSearchTool.execute() with normal results."""

    def test_returns_formatted_string_with_headers(self, mock_vector_store_with_results):
        """Result string includes [Course - Lesson N] headers."""
        from search_tools import CourseSearchTool
        tool = CourseSearchTool(mock_vector_store_with_results)
        result = tool.execute(query="what is RAG")

        assert "[Intro to RAG - Lesson 1]" in result
        assert "Lesson 1 content: This is about RAG systems." in result

    def test_returns_all_results_joined_by_blank_line(self, mock_vector_store_with_results):
        """Multiple results are separated by double newlines."""
        from search_tools import CourseSearchTool
        tool = CourseSearchTool(mock_vector_store_with_results)
        result = tool.execute(query="what is RAG")

        # Two results -> one blank-line separator
        assert result.count("\n\n") >= 1

    def test_populates_last_sources_after_search(self, mock_vector_store_with_results):
        """last_sources is populated with label/url dicts after a successful search."""
        from search_tools import CourseSearchTool
        tool = CourseSearchTool(mock_vector_store_with_results)
        tool.execute(query="what is RAG")

        assert len(tool.last_sources) == 2
        first = tool.last_sources[0]
        assert "label" in first
        assert "url" in first
        assert "Intro to RAG" in first["label"]
        assert "Lesson 1" in first["label"]

    def test_sources_include_lesson_link_url(self, mock_vector_store_with_results):
        """URL in last_sources comes from get_lesson_link()."""
        from search_tools import CourseSearchTool
        mock_vector_store_with_results.get_lesson_link.return_value = "https://example.com/lesson/1"
        tool = CourseSearchTool(mock_vector_store_with_results)
        tool.execute(query="what is RAG")

        assert tool.last_sources[0]["url"] == "https://example.com/lesson/1"

    def test_sources_url_empty_when_no_lesson_link(self, mock_vector_store_with_results):
        """URL in last_sources is empty string when get_lesson_link() returns None."""
        from search_tools import CourseSearchTool
        mock_vector_store_with_results.get_lesson_link.return_value = None
        tool = CourseSearchTool(mock_vector_store_with_results)
        tool.execute(query="what is RAG")

        assert tool.last_sources[0]["url"] == ""

    def test_search_called_with_correct_args(self, mock_vector_store_with_results):
        """VectorStore.search() is called with the query and filters passed to execute()."""
        from search_tools import CourseSearchTool
        tool = CourseSearchTool(mock_vector_store_with_results)
        tool.execute(query="what is RAG", course_name="Intro to RAG", lesson_number=1)

        mock_vector_store_with_results.search.assert_called_once_with(
            query="what is RAG",
            course_name="Intro to RAG",
            lesson_number=1,
        )


class TestCourseSearchToolEmptyResults:
    """CourseSearchTool.execute() when VectorStore returns nothing."""

    def test_returns_no_content_found_message(self, mock_vector_store_empty):
        """Returns a human-readable 'no results' string when results are empty."""
        from search_tools import CourseSearchTool
        tool = CourseSearchTool(mock_vector_store_empty)
        result = tool.execute(query="nonexistent topic")

        assert "No relevant content found" in result

    def test_no_content_message_includes_course_filter(self, mock_vector_store_empty):
        """'No results' message mentions the course name filter when provided."""
        from search_tools import CourseSearchTool
        tool = CourseSearchTool(mock_vector_store_empty)
        result = tool.execute(query="topic", course_name="Unknown Course")

        assert "Unknown Course" in result

    def test_no_content_message_includes_lesson_filter(self, mock_vector_store_empty):
        """'No results' message mentions the lesson number filter when provided."""
        from search_tools import CourseSearchTool
        tool = CourseSearchTool(mock_vector_store_empty)
        result = tool.execute(query="topic", lesson_number=99)

        assert "99" in result

    def test_last_sources_empty_after_empty_results(self, mock_vector_store_empty):
        """last_sources remains empty when search returns no documents."""
        from search_tools import CourseSearchTool
        tool = CourseSearchTool(mock_vector_store_empty)
        tool.execute(query="nothing")

        assert tool.last_sources == []


class TestCourseSearchToolErrorHandling:
    """CourseSearchTool.execute() when VectorStore returns an error."""

    def test_returns_error_message_string(self, mock_vector_store_error):
        """When VectorStore returns an error SearchResults, execute() returns the error string."""
        from search_tools import CourseSearchTool
        tool = CourseSearchTool(mock_vector_store_error)
        result = tool.execute(query="what is RAG")

        # Should propagate the error from SearchResults.error
        assert "Search error" in result or "collection not found" in result

    def test_last_sources_empty_on_error(self, mock_vector_store_error):
        """last_sources is not populated when VectorStore returns an error."""
        from search_tools import CourseSearchTool
        tool = CourseSearchTool(mock_vector_store_error)
        tool.execute(query="what is RAG")

        assert tool.last_sources == []


class TestCourseSearchToolGetToolDefinition:
    """Tool definition structure required by Anthropic API."""

    def test_tool_definition_has_correct_name(self, mock_vector_store):
        """Tool name is 'search_course_content' as expected by the AI system prompt."""
        from search_tools import CourseSearchTool
        tool = CourseSearchTool(mock_vector_store)
        defn = tool.get_tool_definition()
        assert defn["name"] == "search_course_content"

    def test_tool_definition_has_input_schema(self, mock_vector_store):
        """Tool definition includes an input_schema with required 'query' field."""
        from search_tools import CourseSearchTool
        tool = CourseSearchTool(mock_vector_store)
        defn = tool.get_tool_definition()
        schema = defn["input_schema"]
        assert schema["type"] == "object"
        assert "query" in schema["required"]
        assert "query" in schema["properties"]


class TestToolManager:
    """ToolManager registration, dispatch, and source management."""

    def test_register_and_execute_tool(self, mock_vector_store_with_results):
        """Registered tool is correctly dispatched by execute_tool()."""
        from search_tools import ToolManager, CourseSearchTool
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store_with_results)
        manager.register_tool(tool)

        result = manager.execute_tool("search_course_content", query="RAG basics")
        assert "[Intro to RAG" in result

    def test_execute_unknown_tool_returns_error_string(self):
        """Dispatching an unregistered tool name returns an error string, not an exception."""
        from search_tools import ToolManager
        manager = ToolManager()
        result = manager.execute_tool("nonexistent_tool", query="test")

        assert "not found" in result.lower()

    def test_get_last_sources_returns_sources_after_search(self, mock_vector_store_with_results):
        """get_last_sources() returns the sources from the most recent search tool execution."""
        from search_tools import ToolManager, CourseSearchTool
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store_with_results)
        manager.register_tool(tool)
        manager.execute_tool("search_course_content", query="RAG")

        sources = manager.get_last_sources()
        assert len(sources) > 0
        assert "label" in sources[0]
        assert "url" in sources[0]

    def test_reset_sources_clears_all_tool_sources(self, mock_vector_store_with_results):
        """reset_sources() empties last_sources on all registered tools."""
        from search_tools import ToolManager, CourseSearchTool
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store_with_results)
        manager.register_tool(tool)
        manager.execute_tool("search_course_content", query="RAG")

        assert len(manager.get_last_sources()) > 0
        manager.reset_sources()
        assert manager.get_last_sources() == []

    def test_get_tool_definitions_returns_list(self, mock_vector_store_with_results):
        """get_tool_definitions() returns a list of dicts suitable for Anthropic tools param."""
        from search_tools import ToolManager, CourseSearchTool
        manager = ToolManager()
        manager.register_tool(CourseSearchTool(mock_vector_store_with_results))

        definitions = manager.get_tool_definitions()
        assert isinstance(definitions, list)
        assert len(definitions) == 1
        assert "name" in definitions[0]
        assert "input_schema" in definitions[0]
