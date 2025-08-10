# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup

```bash
# Install dependencies using uv (required package manager)
uv sync

# Create .env file with required environment variable
echo "ANTHROPIC_API_KEY=your_api_key_here" > .env
```

### Running the Application

```bash
# Quick start using provided script
chmod +x run.sh
./run.sh

# Manual start (from project root)
cd backend && uv run uvicorn app:app --reload --port 8000
```

The application serves at `http://localhost:8000` with API docs at `http://localhost:8000/docs`.

## Architecture Overview

This is a **RAG (Retrieval-Augmented Generation) system** for course materials with a tool-augmented AI architecture.

### Core Flow

1. **Document Processing**: Course files → structured chunks → vector embeddings
2. **Query Processing**: User query → AI tool selection → semantic search → contextualized response
3. **Session Management**: Conversation history maintained across interactions

### Key Components

**Backend (`/backend/`):**

- `app.py` - FastAPI server with two main endpoints: `/api/query` and `/api/courses`
- `rag_system.py` - **Central orchestrator** coordinating all components
- `document_processor.py` - Parses structured course documents and creates overlapping text chunks
- `vector_store.py` - ChromaDB integration for semantic search using sentence-transformers
- `ai_generator.py` - Anthropic Claude API client with tool-calling capabilities
- `search_tools.py` - Tool definitions for Claude to search course content
- `session_manager.py` - Conversation history tracking for multi-turn dialogues
- `models.py` - Pydantic models: Course, Lesson, CourseChunk

**Frontend (`/frontend/`):** Simple HTML/CSS/JS interface with markdown rendering

**Data (`/docs/`):** Course transcript files with structured format:

```
Course Title: [title]
Course Link: [url]
Course Instructor: [instructor]

Lesson X: [lesson title]
Lesson Link: [lesson url]
[lesson content...]
```

### Tool-Augmented AI Pattern

- Claude decides when to use the `search_course_content` tool based on query type
- Tool performs semantic search with optional course name and lesson number filtering
- Results are synthesized into natural responses with source tracking
- **One search per query maximum** to maintain response quality

### Document Processing Specifics

- **Chunking**: Sentence-based with 800 character limit and 100 character overlap
- **Context Enhancement**: Each chunk prefixed with course and lesson information
- **Metadata Extraction**: Parses structured headers for course/lesson metadata
- **Vector Storage**: ChromaDB with sentence-transformers embeddings (all-MiniLM-L6-v2)

### Session Architecture

- Sessions created automatically for conversation continuity
- History limited to last 2 exchanges (configurable in `config.py`)
- Frontend maintains session ID, backend tracks conversation context

### Configuration

All settings centralized in `backend/config.py`:

- Chunk size/overlap, max results, max history
- Model names (Claude Sonnet 4, sentence-transformers)
- Database paths, API keys via environment variables

This architecture enables natural language querying of course materials with accurate source attribution and conversational context.

- always use uv to run server do not use pip directly
- make sure to use uv to manage all dependencies
- use uv to run Python files
