"""
Tests for AIGenerator - all Anthropic API calls are mocked.

These tests ensure the agentic loop (tool use -> tool execution -> final answer)
is correctly wired, and that error paths propagate properly.
"""
import pytest
from unittest.mock import MagicMock
from helpers import make_text_response, make_tool_use_response


class TestDirectResponse:
    """When Claude answers without using any tool."""

    def test_returns_text_when_no_tool_use(self, ai_generator, mock_anthropic_client):
        """Direct text response (stop_reason='end_turn') is returned as a string."""
        mock_anthropic_client.messages.create.return_value = make_text_response(
            "Python is a programming language."
        )
        result = ai_generator.generate_response(query="What is Python?")

        assert result == "Python is a programming language."

    def test_api_called_with_correct_model(self, ai_generator, mock_anthropic_client):
        """The Anthropic API is called with the configured model name."""
        mock_anthropic_client.messages.create.return_value = make_text_response("ok")
        ai_generator.generate_response(query="test")

        call_kwargs = mock_anthropic_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-sonnet-4-20250514"

    def test_api_called_with_user_query(self, ai_generator, mock_anthropic_client):
        """The user query is passed in the messages list."""
        mock_anthropic_client.messages.create.return_value = make_text_response("ok")
        ai_generator.generate_response(query="my specific question")

        call_kwargs = mock_anthropic_client.messages.create.call_args[1]
        messages = call_kwargs["messages"]
        assert any("my specific question" in str(m.get("content", "")) for m in messages)

    def test_api_called_with_system_prompt(self, ai_generator, mock_anthropic_client):
        """The Anthropic API call includes a non-empty system prompt."""
        mock_anthropic_client.messages.create.return_value = make_text_response("ok")
        ai_generator.generate_response(query="test")

        call_kwargs = mock_anthropic_client.messages.create.call_args[1]
        assert "system" in call_kwargs
        assert len(call_kwargs["system"]) > 0

    def test_conversation_history_appended_to_system_prompt(
        self, ai_generator, mock_anthropic_client
    ):
        """Conversation history, when provided, is included in the system prompt."""
        mock_anthropic_client.messages.create.return_value = make_text_response("ok")
        history = "User: hello\nAssistant: hi"
        ai_generator.generate_response(query="test", conversation_history=history)

        call_kwargs = mock_anthropic_client.messages.create.call_args[1]
        assert history in call_kwargs["system"]

    def test_tools_are_passed_when_provided(self, ai_generator, mock_anthropic_client):
        """When tools are provided, they appear in the API params with tool_choice=auto."""
        mock_anthropic_client.messages.create.return_value = make_text_response("ok")
        dummy_tool_def = {
            "name": "search_course_content",
            "description": "Search",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        }
        ai_generator.generate_response(query="test", tools=[dummy_tool_def])

        call_kwargs = mock_anthropic_client.messages.create.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tool_choice"] == {"type": "auto"}


class TestToolUseLoop:
    """When Claude requests a tool and _handle_tool_execution() runs."""

    def test_handle_tool_execution_called_on_tool_use_stop_reason(
        self, ai_generator, mock_anthropic_client, tool_manager_with_search
    ):
        """When stop_reason is 'tool_use', the agentic loop executes the tool."""
        manager, _ = tool_manager_with_search

        tool_response = make_tool_use_response(
            tool_name="search_course_content",
            tool_id="tu_001",
            tool_input={"query": "what is RAG"},
        )
        final_response = make_text_response("RAG combines retrieval with generation.")
        mock_anthropic_client.messages.create.side_effect = [tool_response, final_response]

        result = ai_generator.generate_response(
            query="what is RAG",
            tools=manager.get_tool_definitions(),
            tool_manager=manager,
        )

        # Two API calls: initial + final after tool use
        assert mock_anthropic_client.messages.create.call_count == 2
        assert result == "RAG combines retrieval with generation."

    def test_tool_result_passed_back_to_claude(
        self, ai_generator, mock_anthropic_client, tool_manager_with_search
    ):
        """The tool's string output is sent back to Claude as a tool_result message."""
        manager, _ = tool_manager_with_search

        tool_response = make_tool_use_response(
            tool_name="search_course_content",
            tool_id="tu_002",
            tool_input={"query": "what is RAG"},
        )
        final_response = make_text_response("Here is the answer.")
        mock_anthropic_client.messages.create.side_effect = [tool_response, final_response]

        ai_generator.generate_response(
            query="what is RAG",
            tools=manager.get_tool_definitions(),
            tool_manager=manager,
        )

        # Inspect the second API call's messages
        second_call_kwargs = mock_anthropic_client.messages.create.call_args_list[1][1]
        messages = second_call_kwargs["messages"]

        # Find the tool_result message in the user role
        tool_result_messages = [
            m for m in messages
            if m.get("role") == "user" and isinstance(m.get("content"), list)
        ]
        assert len(tool_result_messages) == 1
        result_content = tool_result_messages[0]["content"]
        assert any(r.get("type") == "tool_result" for r in result_content)
        assert any(r.get("tool_use_id") == "tu_002" for r in result_content)

    def test_final_api_call_has_no_tools(
        self, ai_generator, mock_anthropic_client, tool_manager_with_search
    ):
        """The follow-up API call after tool execution does NOT include tools."""
        manager, _ = tool_manager_with_search

        tool_response = make_tool_use_response(
            tool_name="search_course_content",
            tool_id="tu_003",
            tool_input={"query": "what is RAG"},
        )
        final_response = make_text_response("Answer.")
        mock_anthropic_client.messages.create.side_effect = [tool_response, final_response]

        ai_generator.generate_response(
            query="what is RAG",
            tools=manager.get_tool_definitions(),
            tool_manager=manager,
        )

        second_call_kwargs = mock_anthropic_client.messages.create.call_args_list[1][1]
        assert "tools" not in second_call_kwargs

    def test_correct_tool_is_dispatched_by_name(
        self, ai_generator, mock_anthropic_client, tool_manager_with_search
    ):
        """ToolManager.execute_tool() is called with the exact tool name Claude requested."""
        manager, search_tool = tool_manager_with_search

        tool_response = make_tool_use_response(
            tool_name="search_course_content",
            tool_id="tu_004",
            tool_input={"query": "RAG pipeline"},
        )
        final_response = make_text_response("Answer.")
        mock_anthropic_client.messages.create.side_effect = [tool_response, final_response]

        ai_generator.generate_response(
            query="tell me about RAG pipeline",
            tools=manager.get_tool_definitions(),
            tool_manager=manager,
        )

        # VectorStore.search should have been called with the query Claude sent
        search_tool.store.search.assert_called_once()
        call_kwargs = search_tool.store.search.call_args[1]
        assert call_kwargs["query"] == "RAG pipeline"

    def test_unknown_tool_name_returns_error_string_not_exception(
        self, ai_generator, mock_anthropic_client
    ):
        """If Claude requests a tool that doesn't exist, the error is handled gracefully."""
        from search_tools import ToolManager

        manager = ToolManager()  # empty - no tools registered

        tool_response = make_tool_use_response(
            tool_name="nonexistent_tool",
            tool_id="tu_ghost",
            tool_input={"query": "test"},
        )
        final_response = make_text_response("I could not find anything.")
        mock_anthropic_client.messages.create.side_effect = [tool_response, final_response]

        # Should not raise
        result = ai_generator.generate_response(
            query="test",
            tools=[],
            tool_manager=manager,
        )
        assert isinstance(result, str)


class TestAPIErrorPropagation:
    """Errors from the Anthropic API should propagate as exceptions."""

    def test_authentication_error_propagates(self, ai_generator, mock_anthropic_client):
        """AuthenticationError from Anthropic client propagates up (not swallowed)."""
        import anthropic
        mock_anthropic_client.messages.create.side_effect = anthropic.AuthenticationError(
            message="Invalid API key", response=MagicMock(status_code=401), body={}
        )

        with pytest.raises(anthropic.AuthenticationError):
            ai_generator.generate_response(query="test")

    def test_api_connection_error_propagates(self, ai_generator, mock_anthropic_client):
        """Network errors from Anthropic client propagate up."""
        import anthropic
        mock_anthropic_client.messages.create.side_effect = anthropic.APIConnectionError(
            request=MagicMock()
        )

        with pytest.raises(anthropic.APIConnectionError):
            ai_generator.generate_response(query="test")
