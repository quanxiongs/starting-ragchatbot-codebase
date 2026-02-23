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
    """When Claude requests a tool, the agentic loop runs.

    Flow for a single-tool-round query (2 API calls total):
      call 1: initial → tool_use
      call 2: loop's intermediate call (WITH tools) → end_turn (Claude done, direct_answer returned)
      No synthesis call — Claude's end_turn response is returned directly.
    """

    def test_handle_tool_execution_called_on_tool_use_stop_reason(
        self, ai_generator, mock_anthropic_client, tool_manager_with_search
    ):
        """When stop_reason is 'tool_use', the agentic loop executes the tool and returns text."""
        manager, _ = tool_manager_with_search

        tool_response = make_tool_use_response(
            tool_name="search_course_content",
            tool_id="tu_001",
            tool_input={"query": "what is RAG"},
        )
        # call 2: loop intermediate (end_turn → direct_answer returned, no synthesis call)
        loop_end_response = make_text_response("RAG combines retrieval with generation.")
        mock_anthropic_client.messages.create.side_effect = [
            tool_response, loop_end_response
        ]

        result = ai_generator.generate_response(
            query="what is RAG",
            tools=manager.get_tool_definitions(),
            tool_manager=manager,
        )

        # Two API calls: initial + loop intermediate (no extra synthesis call)
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
        loop_end_response = make_text_response("Here is the answer.")
        mock_anthropic_client.messages.create.side_effect = [
            tool_response, loop_end_response
        ]

        ai_generator.generate_response(
            query="what is RAG",
            tools=manager.get_tool_definitions(),
            tool_manager=manager,
        )

        # Inspect the second API call (loop's intermediate) — it receives the tool_result
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

    def test_synthesis_call_has_no_tools_when_loop_cap_hit(
        self, ai_generator, mock_anthropic_client, tool_manager_with_search
    ):
        """When the 2-round cap is hit, the final synthesis call does NOT include tools."""
        manager, _ = tool_manager_with_search

        # call 1: initial → tool_use; call 2: loop iter 1 → tool_use; call 3: loop iter 2 → tool_use
        # cap exhausted → call 4: synthesis (no tools)
        tool_resp_1 = make_tool_use_response("search_course_content", "tu_c1", {"query": "q1"})
        tool_resp_2 = make_tool_use_response("search_course_content", "tu_c2", {"query": "q2"})
        tool_resp_3 = make_tool_use_response("search_course_content", "tu_c3", {"query": "q3"})
        synthesis_response = make_text_response("Synthesized answer.")
        mock_anthropic_client.messages.create.side_effect = [
            tool_resp_1, tool_resp_2, tool_resp_3, synthesis_response
        ]

        result = ai_generator.generate_response(
            query="what is RAG",
            tools=manager.get_tool_definitions(),
            tool_manager=manager,
        )

        # Call index 3 is the synthesis — must have no tools
        assert mock_anthropic_client.messages.create.call_count == 4
        final_call_kwargs = mock_anthropic_client.messages.create.call_args_list[3][1]
        assert "tools" not in final_call_kwargs
        assert result == "Synthesized answer."

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
        loop_end_response = make_text_response("Answer.")
        mock_anthropic_client.messages.create.side_effect = [
            tool_response, loop_end_response
        ]

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
        loop_end_response = make_text_response("I could not find anything.")
        mock_anthropic_client.messages.create.side_effect = [
            tool_response, loop_end_response
        ]

        # Should not raise — ToolManager returns an error string for unknown tools
        result = ai_generator.generate_response(
            query="test",
            tools=[],
            tool_manager=manager,
        )
        assert isinstance(result, str)


class TestTwoRoundToolLoop:
    """Sequential tool calling: Claude may use up to 2 tool-call rounds.

    When the loop ends with end_turn, the text is returned directly (no synthesis call).
    When the loop cap is hit (both rounds return tool_use), a synthesis call fires.

    Flow for 2 tool rounds ending with end_turn (3 API calls total):
      call 1: initial → tool_use          (generate_response)
      call 2: loop round 1 → tool_use     (_run_tool_loop, iteration 1, WITH tools)
      call 3: loop round 2 → end_turn     (_run_tool_loop, iteration 2, breaks, direct_answer)
      No synthesis call — direct_answer returned.

    Flow for 2-round cap (both tool_use, 4 API calls total):
      call 1: initial → tool_use
      call 2: loop round 1 → tool_use
      call 3: loop round 2 → tool_use     (cap exhausted)
      call 4: final no-tools synthesis
    """

    def _make_two_round_side_effect(self, mock_anthropic_client, final_text="Final answer."):
        """Helper: wire up mock for 2 tool rounds ending with end_turn (3 API calls total)."""
        tool_resp_1 = make_tool_use_response(
            tool_name="search_course_content",
            tool_id="tu_r1",
            tool_input={"query": "RAG overview"},
        )
        tool_resp_2 = make_tool_use_response(
            tool_name="search_course_content",
            tool_id="tu_r2",
            tool_input={"query": "RAG details"},
        )
        loop_end_resp = make_text_response(final_text)  # loop round 2 → end_turn, direct_answer
        mock_anthropic_client.messages.create.side_effect = [
            tool_resp_1, tool_resp_2, loop_end_resp
        ]
        return tool_resp_1, tool_resp_2, loop_end_resp

    def test_two_rounds_make_three_api_calls_total(
        self, ai_generator, mock_anthropic_client, tool_manager_with_search
    ):
        """Two tool-use rounds ending with end_turn = exactly 3 API calls (no synthesis)."""
        manager, _ = tool_manager_with_search
        self._make_two_round_side_effect(mock_anthropic_client)

        ai_generator.generate_response(
            query="Tell me about RAG",
            tools=manager.get_tool_definitions(),
            tool_manager=manager,
        )

        assert mock_anthropic_client.messages.create.call_count == 3

    def test_result_is_text_from_loop_end_turn_response(
        self, ai_generator, mock_anthropic_client, tool_manager_with_search
    ):
        """The returned string is the text from the loop's end_turn response (not a synthesis call)."""
        manager, _ = tool_manager_with_search
        self._make_two_round_side_effect(mock_anthropic_client, final_text="Direct answer.")

        result = ai_generator.generate_response(
            query="Tell me about RAG",
            tools=manager.get_tool_definitions(),
            tool_manager=manager,
        )

        assert result == "Direct answer."

    def test_two_round_early_exit_has_no_synthesis_call(
        self, ai_generator, mock_anthropic_client, tool_manager_with_search
    ):
        """When 2 rounds run and loop exits via end_turn, no synthesis call is made."""
        manager, _ = tool_manager_with_search
        # call 1: initial → tool_use; call 2: loop iter 1 → tool_use; call 3: loop iter 2 → end_turn
        # direct_answer returned — no synthesis call fires
        self._make_two_round_side_effect(mock_anthropic_client, final_text="Direct.")

        result = ai_generator.generate_response(
            query="Tell me about RAG",
            tools=manager.get_tool_definitions(),
            tool_manager=manager,
        )

        # Exactly 3 calls, all with tools (no synthesis call)
        assert mock_anthropic_client.messages.create.call_count == 3
        # Loop iter 2 (call index 2) still has tools — it's not a synthesis call
        loop_iter2_kwargs = mock_anthropic_client.messages.create.call_args_list[2][1]
        assert "tools" in loop_iter2_kwargs
        assert result == "Direct."

    def test_intermediate_round2_call_includes_tools(
        self, ai_generator, mock_anthropic_client, tool_manager_with_search
    ):
        """The second API call inside the loop (call index 1) still includes tools for chaining."""
        manager, _ = tool_manager_with_search
        self._make_two_round_side_effect(mock_anthropic_client)

        ai_generator.generate_response(
            query="Tell me about RAG",
            tools=manager.get_tool_definitions(),
            tool_manager=manager,
        )

        # Call index 1 is the loop's first intermediate call — must still have tools
        round2_call_kwargs = mock_anthropic_client.messages.create.call_args_list[1][1]
        assert "tools" in round2_call_kwargs

    def test_round1_tool_result_present_in_round2_messages(
        self, ai_generator, mock_anthropic_client, tool_manager_with_search
    ):
        """Round 2 API call's messages must contain the tool_result from round 1."""
        manager, _ = tool_manager_with_search
        self._make_two_round_side_effect(mock_anthropic_client)

        ai_generator.generate_response(
            query="Tell me about RAG",
            tools=manager.get_tool_definitions(),
            tool_manager=manager,
        )

        round2_messages = mock_anthropic_client.messages.create.call_args_list[1][1]["messages"]
        tool_result_messages = [
            m for m in round2_messages
            if m.get("role") == "user" and isinstance(m.get("content"), list)
        ]
        assert len(tool_result_messages) >= 1
        result_content = tool_result_messages[0]["content"]
        assert any(r.get("type") == "tool_result" for r in result_content)
        assert any(r.get("tool_use_id") == "tu_r1" for r in result_content)

    def test_loop_capped_at_two_rounds(
        self, ai_generator, mock_anthropic_client, tool_manager_with_search
    ):
        """Even if all loop responses are tool_use, the loop stops after 2 iterations then synthesizes."""
        manager, _ = tool_manager_with_search
        # call 1: initial → tool_use
        # call 2: loop iter 1 → tool_use (continues)
        # call 3: loop iter 2 → tool_use (cap exhausted, direct_answer=None)
        # call 4: synthesis (no tools) — fires because cap was hit
        tool_resp_1 = make_tool_use_response("search_course_content", "tu_1", {"query": "q1"})
        tool_resp_2 = make_tool_use_response("search_course_content", "tu_2", {"query": "q2"})
        tool_resp_3 = make_tool_use_response("search_course_content", "tu_3", {"query": "q3"})
        final_resp = make_text_response("Done.")
        mock_anthropic_client.messages.create.side_effect = [
            tool_resp_1, tool_resp_2, tool_resp_3, final_resp
        ]

        ai_generator.generate_response(
            query="multi-round query",
            tools=manager.get_tool_definitions(),
            tool_manager=manager,
        )

        # initial(1) + loop_iter_1(2) + loop_iter_2(3, cap hit) + synthesis(4)
        assert mock_anthropic_client.messages.create.call_count == 4

    def test_sources_aggregated_across_two_rounds(
        self, ai_generator, mock_anthropic_client, tool_manager_with_both_tools
    ):
        """Sources from both rounds are merged; get_last_sources() returns entries from both tools."""
        manager, search_tool, outline_tool = tool_manager_with_both_tools

        tool_resp_1 = make_tool_use_response(
            "search_course_content", "tu_s1", {"query": "RAG overview"}
        )
        tool_resp_2 = make_tool_use_response(
            "get_course_outline", "tu_o1", {"course_title": "Intro to RAG"}
        )
        loop_end_resp = make_text_response("Combined answer.")
        mock_anthropic_client.messages.create.side_effect = [
            tool_resp_1, tool_resp_2, loop_end_resp
        ]

        ai_generator.generate_response(
            query="RAG overview and outline",
            tools=manager.get_tool_definitions(),
            tool_manager=manager,
        )

        sources = manager.get_last_sources()
        assert len(sources) > 0
        labels = [s["label"] for s in sources]
        # search tool produces lesson-level sources (e.g. "Intro to RAG - Lesson 1")
        assert any("Lesson" in label for label in labels), "Expected search tool source with lesson"
        # outline tool produces course-level sources (e.g. "Intro to RAG" with no lesson suffix)
        assert any(label == "Intro to RAG" for label in labels), "Expected outline tool source"

    def test_tool_error_propagates_as_exception(
        self, ai_generator, mock_anthropic_client, tool_manager_with_search
    ):
        """A RuntimeError raised by a tool propagates out of generate_response (not swallowed)."""
        from unittest.mock import patch

        manager, _ = tool_manager_with_search
        tool_resp = make_tool_use_response("search_course_content", "tu_err", {"query": "q"})
        mock_anthropic_client.messages.create.return_value = tool_resp

        with patch.object(manager, "execute_tool", side_effect=RuntimeError("DB unavailable")):
            with pytest.raises(RuntimeError, match="DB unavailable"):
                ai_generator.generate_response(
                    query="test",
                    tools=manager.get_tool_definitions(),
                    tool_manager=manager,
                )

    def test_claude_stops_after_one_round(
        self, ai_generator, mock_anthropic_client, tool_manager_with_search
    ):
        """If Claude returns end_turn after round 1 inside the loop, total is 2 calls (no synthesis)."""
        manager, _ = tool_manager_with_search
        # call 1: initial → tool_use; call 2: loop iter 1 → end_turn (direct_answer returned)
        tool_resp = make_tool_use_response("search_course_content", "tu_1", {"query": "q"})
        loop_end_text = make_text_response("Early answer.")
        mock_anthropic_client.messages.create.side_effect = [tool_resp, loop_end_text]

        result = ai_generator.generate_response(
            query="simple question",
            tools=manager.get_tool_definitions(),
            tool_manager=manager,
        )

        assert mock_anthropic_client.messages.create.call_count == 2
        assert result == "Early answer."


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
