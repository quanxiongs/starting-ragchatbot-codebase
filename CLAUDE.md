# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Retrieval-Augmented Generation (RAG) system for course materials. It combines FastAPI backend with a vanilla JavaScript frontend to provide intelligent question-answering about educational content using ChromaDB for vector storage and Anthropic's Claude for AI generation.

## Development Commands

### Running the Application

**IMPORTANT: Always use `uv` to run the server. Do NOT use `pip` directly.**

```bash
# Quick start (recommended)
./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

Access points:
- Web Interface: http://localhost:8000
- API Documentation: http://localhost:8000/docs

### Setup

```bash
# Install dependencies (uses uv, not pip)
uv sync

# Create .env file with:
ANTHROPIC_API_KEY=your_key_here
```

### Running Python Commands

Always prefix Python commands with `uv run`:

```bash
# Run Python scripts
uv run python script.py

# Run any Python command
uv run <command>
```

## Architecture

### Core Component Flow

The system follows a layered architecture where each component has a single responsibility:

1. **RAGSystem** ([rag_system.py](backend/rag_system.py)) - Central orchestrator that coordinates all components
   - Manages document ingestion pipeline
   - Coordinates query processing flow
   - Integrates search tools with AI generator

2. **VectorStore** ([vector_store.py](backend/vector_store.py)) - Dual-collection ChromaDB implementation
   - `course_catalog`: Stores course metadata (titles, instructors, links) for semantic course name matching
   - `course_content`: Stores chunked lesson content for detailed search
   - Resolves fuzzy course names to exact titles using vector similarity

3. **AIGenerator** ([ai_generator.py](backend/ai_generator.py)) - Anthropic Claude integration with tool calling
   - Uses tool-based architecture for search (not direct context injection)
   - Handles agentic loop: tool request → execution → final response
   - System prompt emphasizes concise, educational responses

4. **Tool System** ([search_tools.py](backend/search_tools.py)) - Extensible tool framework
   - `CourseSearchTool`: Semantic search across course content with optional filters
   - `ToolManager`: Registry for tools available to Claude
   - Tools track sources for UI display

5. **DocumentProcessor** ([document_processor.py](backend/document_processor.py)) - Structured document parser
   - Expects specific format: Course metadata (title, link, instructor) followed by "Lesson N:" markers
   - Creates sentence-based chunks with configurable overlap
   - Adds contextual prefixes to chunks for better retrieval

6. **SessionManager** ([session_manager.py](backend/session_manager.py)) - Conversation context tracking
   - Maintains limited conversation history per session
   - Enables multi-turn conversations with context

### Key Architectural Patterns

**Tool-Based RAG vs. Direct Context Injection**
- Claude receives tool definitions, not pre-fetched context
- Claude decides when/how to search based on the query
- Search results are passed back as tool results for synthesis

**Dual Vector Store Strategy**
- Course catalog enables fuzzy matching (e.g., "MCP" → "Introduction to MCP Servers")
- Content store provides fine-grained search with lesson-level filtering
- Separation allows different optimization strategies for metadata vs. content

**Chunk Context Enhancement**
- First chunk of each lesson: `"Lesson {N} content: {chunk}"`
- Subsequent chunks: Include course title and lesson number in prefix
- Ensures retrieved chunks maintain source attribution

## Configuration

All settings in [config.py](backend/config.py):
- `CHUNK_SIZE`: 800 characters per chunk
- `CHUNK_OVERLAP`: 100 characters between chunks
- `MAX_RESULTS`: 5 search results per query
- `MAX_HISTORY`: 2 conversation turns remembered
- `ANTHROPIC_MODEL`: claude-sonnet-4-20250514
- `EMBEDDING_MODEL`: all-MiniLM-L6-v2

## Data Models

See [models.py](backend/models.py) for Pydantic schemas:
- `Course`: Container with title (unique ID), link, instructor, lessons
- `Lesson`: Lesson number, title, optional link
- `CourseChunk`: Content chunk with course/lesson attribution and index

## Document Format

Course documents (in `docs/`) should follow this structure:

```
Course Title: [Title]
Course Link: [URL]
Course Instructor: [Name]

Lesson 0: [Lesson Title]
Lesson Link: [URL]
[Lesson content...]

Lesson 1: [Lesson Title]
Lesson Link: [URL]
[Lesson content...]
```

Documents are auto-loaded on startup and de-duplicated by course title.

## Frontend

Simple vanilla JS in [frontend/](frontend/):
- [index.html](frontend/index.html): Layout
- [script.js](frontend/script.js): API calls, markdown rendering
- [style.css](frontend/style.css): Styling

FastAPI serves static files with no-cache headers for development.

## Dependencies

Python 3.13+ with uv package manager:
- chromadb 1.0.15
- anthropic 0.58.2
- sentence-transformers 5.0.0
- fastapi 0.116.1
- uvicorn 0.35.0
- python-multipart 0.0.20
- python-dotenv 1.1.1
- make sure to use uv to manage all dependencies
- make sure to use uv to manage all dependencies