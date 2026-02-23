"""Shared test helper functions (not pytest fixtures)."""
from unittest.mock import MagicMock


def make_text_response(text: str):
    """Simulate a Claude response that returns text directly (no tool use)."""
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = text

    response = MagicMock()
    response.stop_reason = "end_turn"
    response.content = [text_block]
    return response


def make_tool_use_response(tool_name: str, tool_id: str, tool_input: dict):
    """Simulate a Claude response requesting a tool call."""
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.name = tool_name
    tool_block.id = tool_id
    tool_block.input = tool_input

    response = MagicMock()
    response.stop_reason = "tool_use"
    response.content = [tool_block]
    return response
