from typing import Dict, Any, Optional, Protocol
from abc import ABC, abstractmethod
from vector_store import VectorStore, SearchResults


class Tool(ABC):
    """Abstract base class for all tools"""

    @abstractmethod
    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        pass

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool with given parameters"""
        pass


class CourseSearchTool(Tool):
    """Tool for searching course content with semantic course name matching"""

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources = []  # Track sources from last search

    def get_tool_definition(self) -> Dict[str, Any]:
        """Return Anthropic tool definition for this tool"""
        return {
            "name": "search_course_content",
            "description": "Search course materials with smart course name matching and lesson filtering",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to search for in the course content",
                    },
                    "course_name": {
                        "type": "string",
                        "description": "Course title (partial matches work, e.g. 'MCP', 'Introduction')",
                    },
                    "lesson_number": {
                        "type": "integer",
                        "description": "Specific lesson number to search within (e.g. 1, 2, 3)",
                    },
                },
                "required": ["query"],
            },
        }

    def execute(
        self,
        query: str,
        course_name: Optional[str] = None,
        lesson_number: Optional[int] = None,
    ) -> str:
        """
        Execute the search tool with given parameters.

        Args:
            query: What to search for
            course_name: Optional course filter
            lesson_number: Optional lesson filter

        Returns:
            Formatted search results or error message
        """

        # Use the vector store's unified search interface
        results = self.store.search(
            query=query, course_name=course_name, lesson_number=lesson_number
        )

        # Handle errors
        if results.error:
            return results.error

        # Handle empty results
        if results.is_empty():
            filter_info = ""
            if course_name:
                filter_info += f" in course '{course_name}'"
            if lesson_number:
                filter_info += f" in lesson {lesson_number}"
            return f"No relevant content found{filter_info}."

        # Format and return results
        return self._format_results(results)

    def _format_results(self, results: SearchResults) -> str:
        """Format search results with course and lesson context"""
        formatted = []
        sources = []  # Track sources for the UI

        for doc, meta in zip(results.documents, results.metadata):
            course_title = meta.get("course_title", "unknown")
            lesson_num = meta.get("lesson_number")

            # Build context header
            header = f"[{course_title}"
            if lesson_num is not None and lesson_num != -1:
                header += f" - Lesson {lesson_num}"
            header += "]"

            # Track source for the UI
            label = course_title
            if lesson_num is not None and lesson_num != -1:
                label += f" - Lesson {lesson_num}"

            # Look up lesson link from the catalog
            url = ""
            if lesson_num is not None and lesson_num != -1:
                lesson_link = self.store.get_lesson_link(course_title, lesson_num)
                if lesson_link:
                    url = lesson_link

            sources.append({"label": label, "url": url})

            formatted.append(f"{header}\n{doc}")

        # Store sources for retrieval
        self.last_sources = sources

        return "\n\n".join(formatted)


class CourseOutlineTool(Tool):
    """Tool for retrieving a structured outline of a course's lessons"""

    def __init__(self, vector_store: VectorStore):
        self.store = vector_store
        self.last_sources = []

    def get_tool_definition(self) -> Dict[str, Any]:
        return {
            "name": "get_course_outline",
            "description": "Get the full lesson outline for a course. Use this when the user asks what lessons a course contains, wants a course overview, syllabus, or table of contents.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "course_title": {
                        "type": "string",
                        "description": "The course name or partial title to look up (e.g., 'MCP', 'Introduction to Python')",
                    }
                },
                "required": ["course_title"],
            },
        }

    def execute(self, course_title: str) -> str:
        # Fuzzy-resolve input to exact course title
        resolved_title = self.store._resolve_course_name(course_title)
        if not resolved_title:
            return f"No course found matching '{course_title}'."

        # Find the matching course record
        all_courses = self.store.get_all_courses_metadata()
        course_data = next(
            (c for c in all_courses if c.get("title") == resolved_title), None
        )
        if not course_data:
            return f"Course '{resolved_title}' was found in the catalog but metadata could not be retrieved."

        return self._format_outline(course_data)

    def _format_outline(self, course_data: Dict[str, Any]) -> str:
        title = course_data.get("title", "Unknown Course")
        course_link = course_data.get("course_link", "")
        lessons = course_data.get("lessons", [])

        self.last_sources = [{"label": title, "url": course_link or ""}]

        lines = [f"Course: {title}"]
        if course_link:
            lines.append(f"Link: {course_link}")

        lines.append("\nLessons:")
        for lesson in sorted(lessons, key=lambda l: l.get("lesson_number", 0)):
            num = lesson.get("lesson_number", "?")
            lesson_title = lesson.get("lesson_title", "Untitled")
            lines.append(f"  {num}. {lesson_title}")

        return "\n".join(lines)


class ToolManager:
    """Manages available tools for the AI"""

    def __init__(self):
        self.tools = {}
        self._aggregated_sources: list = []

    def register_tool(self, tool: Tool):
        """Register any tool that implements the Tool interface"""
        tool_def = tool.get_tool_definition()
        tool_name = tool_def.get("name")
        if not tool_name:
            raise ValueError("Tool must have a 'name' in its definition")
        self.tools[tool_name] = tool

    def get_tool_definitions(self) -> list:
        """Get all tool definitions for Anthropic tool calling"""
        return [tool.get_tool_definition() for tool in self.tools.values()]

    def execute_tool(self, tool_name: str, **kwargs) -> str:
        """Execute a tool by name with given parameters"""
        if tool_name not in self.tools:
            return f"Tool '{tool_name}' not found"

        return self.tools[tool_name].execute(**kwargs)

    def set_aggregated_sources(self, sources: list):
        """Store sources aggregated across multiple tool-call rounds"""
        self._aggregated_sources = sources

    def get_last_sources(self) -> list:
        """Get sources from the last search operation.

        Multi-round queries store aggregated sources via set_aggregated_sources().
        Single-round queries fall back to scanning per-tool last_sources.
        """
        if self._aggregated_sources:
            return self._aggregated_sources
        for tool in self.tools.values():
            if hasattr(tool, "last_sources") and tool.last_sources:
                return tool.last_sources
        return []

    def reset_sources(self):
        """Reset sources from all tools that track sources"""
        self._aggregated_sources = []
        for tool in self.tools.values():
            if hasattr(tool, "last_sources"):
                tool.last_sources = []
