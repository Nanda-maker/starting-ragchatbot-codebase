import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import List, Dict, Any, Optional
import sys
import os

# Add parent directory to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import Course, Lesson, CourseChunk
from vector_store import SearchResults
from search_tools import CourseSearchTool, ToolManager
from ai_generator import AIGenerator
from config import Config


@pytest.fixture
def sample_courses():
    """Sample course data for testing"""
    course1 = Course(
        title="Introduction to Machine Learning",
        course_link="https://example.com/ml-course",
        instructor="Dr. Smith",
        lessons=[
            Lesson(lesson_number=1, title="What is ML?", lesson_link="https://example.com/ml-lesson1"),
            Lesson(lesson_number=2, title="Supervised Learning", lesson_link="https://example.com/ml-lesson2"),
            Lesson(lesson_number=3, title="Neural Networks", lesson_link="https://example.com/ml-lesson3")
        ]
    )
    
    course2 = Course(
        title="MCP Protocol Deep Dive",
        course_link="https://example.com/mcp-course", 
        instructor="Jane Doe",
        lessons=[
            Lesson(lesson_number=1, title="MCP Basics", lesson_link="https://example.com/mcp-lesson1"),
            Lesson(lesson_number=2, title="Advanced MCP", lesson_link="https://example.com/mcp-lesson2")
        ]
    )
    
    return [course1, course2]


@pytest.fixture
def sample_course_chunks():
    """Sample course chunks for testing"""
    return [
        CourseChunk(
            content="Machine learning is a subset of artificial intelligence that focuses on algorithms that learn from data.",
            course_title="Introduction to Machine Learning",
            lesson_number=1,
            chunk_index=0
        ),
        CourseChunk(
            content="Supervised learning involves training models on labeled data to make predictions on new data.",
            course_title="Introduction to Machine Learning", 
            lesson_number=2,
            chunk_index=1
        ),
        CourseChunk(
            content="Neural networks are computational models inspired by biological neural networks.",
            course_title="Introduction to Machine Learning",
            lesson_number=3,
            chunk_index=2
        ),
        CourseChunk(
            content="MCP (Model Context Protocol) enables seamless communication between AI models and external tools.",
            course_title="MCP Protocol Deep Dive",
            lesson_number=1,
            chunk_index=0
        )
    ]


@pytest.fixture
def mock_search_results_success():
    """Mock successful search results"""
    return SearchResults(
        documents=[
            "Machine learning is a subset of artificial intelligence that focuses on algorithms that learn from data.",
            "Supervised learning involves training models on labeled data to make predictions on new data."
        ],
        metadata=[
            {
                "course_title": "Introduction to Machine Learning",
                "lesson_number": 1,
                "chunk_index": 0
            },
            {
                "course_title": "Introduction to Machine Learning", 
                "lesson_number": 2,
                "chunk_index": 1
            }
        ],
        distances=[0.1, 0.2],
        error=None
    )


@pytest.fixture
def mock_search_results_empty():
    """Mock empty search results"""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[],
        error=None
    )


@pytest.fixture
def mock_search_results_error():
    """Mock search results with error"""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[],
        error="Database connection failed"
    )


@pytest.fixture
def mock_vector_store():
    """Mock VectorStore for testing"""
    mock_store = Mock()
    mock_store.search = Mock()
    mock_store.get_lesson_link = Mock(return_value="https://example.com/lesson-link")
    return mock_store


@pytest.fixture
def course_search_tool(mock_vector_store):
    """CourseSearchTool instance with mocked vector store"""
    return CourseSearchTool(mock_vector_store)


@pytest.fixture
def tool_manager(course_search_tool):
    """ToolManager with registered CourseSearchTool"""
    manager = ToolManager()
    manager.register_tool(course_search_tool)
    return manager


@pytest.fixture
def mock_anthropic_response_no_tools():
    """Mock Anthropic API response without tool usage"""
    mock_response = Mock()
    mock_response.stop_reason = "end_turn"
    mock_response.content = [Mock(text="This is a direct response without tool usage.")]
    return mock_response


@pytest.fixture
def mock_anthropic_response_with_tools():
    """Mock Anthropic API response with tool usage"""
    mock_response = Mock()
    mock_response.stop_reason = "tool_use"
    
    # Mock tool use content block
    mock_tool_block = Mock()
    mock_tool_block.type = "tool_use"
    mock_tool_block.name = "search_course_content"
    mock_tool_block.id = "tool_123"
    mock_tool_block.input = {"query": "machine learning", "course_name": "Introduction to Machine Learning"}
    
    mock_response.content = [mock_tool_block]
    return mock_response


@pytest.fixture
def mock_anthropic_final_response():
    """Mock final Anthropic API response after tool execution"""
    mock_response = Mock()
    mock_response.content = [Mock(text="Based on the search results, machine learning is a subset of AI.")]
    return mock_response


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing"""
    with patch('anthropic.Anthropic') as mock_client_class:
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        yield mock_client


@pytest.fixture
def test_config():
    """Test configuration"""
    config = Config()
    config.ANTHROPIC_API_KEY = "test-api-key"
    config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
    config.MAX_RESULTS = 3
    config.CHUNK_SIZE = 500
    config.CHUNK_OVERLAP = 50
    return config


@pytest.fixture
def ai_generator(test_config, mock_anthropic_client):
    """AIGenerator instance with mocked client"""
    return AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)


# Utility function to create custom search results
def create_search_results(documents: List[str] = None, 
                         metadata: List[Dict] = None,
                         distances: List[float] = None,
                         error: Optional[str] = None) -> SearchResults:
    """Helper function to create SearchResults for testing"""
    return SearchResults(
        documents=documents or [],
        metadata=metadata or [],
        distances=distances or [],
        error=error
    )