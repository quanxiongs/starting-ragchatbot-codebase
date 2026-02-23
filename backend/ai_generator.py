import anthropic
from typing import List, Optional, Dict, Any


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """You are an AI assistant specialized in course materials and educational content with access to tools for searching course content and retrieving course outlines.

Tool Selection:
- **get_course_outline**: Use when the user asks what lessons a course contains, requests a course overview, syllabus, or table of contents. Present results as: course title, course link, and a numbered list of lesson titles.
- **search_course_content**: Use when the user asks detailed questions about specific course topics, concepts, or lesson material.
- **Sequential tool calls**: You may make up to 2 sequential tool calls if the first result is insufficient or the query requires information from two distinct sources. Only use a second call when the first result is clearly incomplete.
- If a tool yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using any tool
- **Course outline questions**: Use get_course_outline, then present the course title, course link (if available), and lesson numbers with titles
- **Course-specific content questions**: Use search_course_content, then synthesize results into a focused answer
- **No meta-commentary**: Provide direct answers only — no reasoning process, tool explanations, or question-type analysis. Do not mention "based on the search results"

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        # Prepare API call parameters efficiently
        api_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": query}],
            "system": system_content,
        }

        # Add tools if available
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = {"type": "auto"}

        # Get response from Claude
        response = self.client.messages.create(**api_params)

        # Run tool loop (up to 2 rounds) if Claude wants to use a tool
        if response.stop_reason == "tool_use" and tool_manager:
            messages, accumulated_sources, direct_answer = self._run_tool_loop(
                first_response=response,
                messages=api_params["messages"],
                system=system_content,
                tools=tools or [],
                tool_manager=tool_manager,
            )
            tool_manager.set_aggregated_sources(accumulated_sources)

            # Loop ended with a direct answer — no extra API call needed
            if direct_answer is not None:
                return direct_answer

            # Loop hit the 2-round cap with tool_use still active — synthesize now
            final_response = self.client.messages.create(
                **self.base_params,
                system=system_content,
                messages=messages,
            )
            return final_response.content[0].text

        # Return direct response
        return response.content[0].text

    def _run_tool_loop(
        self, first_response, messages: List, system: str, tools: List, tool_manager
    ) -> tuple:
        """
        Execute up to 2 rounds of tool calls, preserving full conversation context.

        Each round: append assistant turn → execute tools → append results → call API again with tools.
        Terminates when: (a) Claude responds without tool_use (direct_answer returned), (b) 2 rounds
        completed with tool_use still active (direct_answer=None, caller must synthesize), or
        (c) a tool raises an exception (propagates up).

        Returns:
            (messages, accumulated_sources, direct_answer)
            direct_answer is the response text when Claude ended with end_turn; None when the
            2-round cap was hit and the caller must make a final no-tools synthesis call.
        """
        accumulated_sources = []
        current_response = first_response
        messages = list(messages)  # shallow copy — don't mutate caller's list
        direct_answer = None

        for _ in range(2):
            # Append assistant's tool-use turn
            messages.append({"role": "assistant", "content": current_response.content})

            # Execute all tool calls in this response
            tool_results = []
            for block in current_response.content:
                if block.type != "tool_use":
                    continue

                # Tool exceptions propagate up naturally (same as API errors)
                result = tool_manager.execute_tool(block.name, **block.input)

                # Capture sources immediately before the next round can overwrite them
                for tool in tool_manager.tools.values():
                    if hasattr(tool, "last_sources") and tool.last_sources:
                        accumulated_sources.extend(tool.last_sources)
                        tool.last_sources = []  # clear to avoid double-counting
                        break

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    }
                )

            if tool_results:
                messages.append({"role": "user", "content": tool_results})

            # Ask Claude again — WITH tools so it can chain if rounds remain
            next_response = self.client.messages.create(
                **self.base_params,
                system=system,
                messages=messages,
                tools=tools,
                tool_choice={"type": "auto"},
            )

            if next_response.stop_reason != "tool_use":
                # Claude answered directly — return text; no synthesis call needed
                direct_answer = (
                    next_response.content[0].text if next_response.content else ""
                )
                break

            current_response = next_response

        return messages, accumulated_sources, direct_answer
