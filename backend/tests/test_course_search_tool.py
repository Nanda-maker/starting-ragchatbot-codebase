import pytest
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults
from conftest import create_search_results


class TestCourseSearchTool:
    """Unit tests for CourseSearchTool.execute() method"""
    
    def test_get_tool_definition(self, course_search_tool):
        """Test that tool definition is correctly formatted"""
        definition = course_search_tool.get_tool_definition()
        
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert definition["input_schema"]["type"] == "object"
        assert "query" in definition["input_schema"]["properties"]
        assert definition["input_schema"]["required"] == ["query"]
        
        # Check optional parameters
        properties = definition["input_schema"]["properties"]
        assert "course_name" in properties
        assert "lesson_number" in properties
    
    def test_execute_successful_search_basic_query(self, course_search_tool, mock_search_results_success):
        """Test successful search with basic query only"""
        course_search_tool.store.search.return_value = mock_search_results_success
        
        result = course_search_tool.execute("machine learning")
        
        # Verify search was called correctly
        course_search_tool.store.search.assert_called_once_with(
            query="machine learning",
            course_name=None,
            lesson_number=None
        )
        
        # Verify result formatting
        assert "[Introduction to Machine Learning" in result
        assert "Machine learning is a subset" in result
        assert "Supervised learning involves" in result
        
        # Verify sources were tracked
        assert len(course_search_tool.last_sources) == 2
        assert course_search_tool.last_sources[0]["text"] == "Introduction to Machine Learning - Lesson 1"
    
    def test_execute_successful_search_with_course_filter(self, course_search_tool, mock_search_results_success):
        """Test successful search with course name filter"""
        course_search_tool.store.search.return_value = mock_search_results_success
        
        result = course_search_tool.execute("machine learning", course_name="Introduction to Machine Learning")
        
        # Verify search was called with course filter
        course_search_tool.store.search.assert_called_once_with(
            query="machine learning",
            course_name="Introduction to Machine Learning",
            lesson_number=None
        )
        
        assert "Introduction to Machine Learning" in result
    
    def test_execute_successful_search_with_lesson_filter(self, course_search_tool, mock_search_results_success):
        """Test successful search with lesson number filter"""
        course_search_tool.store.search.return_value = mock_search_results_success
        
        result = course_search_tool.execute("machine learning", lesson_number=1)
        
        # Verify search was called with lesson filter
        course_search_tool.store.search.assert_called_once_with(
            query="machine learning",
            course_name=None,
            lesson_number=1
        )
        
        assert "Lesson 1" in result
    
    def test_execute_successful_search_with_both_filters(self, course_search_tool, mock_search_results_success):
        """Test successful search with both course name and lesson number filters"""
        course_search_tool.store.search.return_value = mock_search_results_success
        
        result = course_search_tool.execute(
            "neural networks", 
            course_name="Introduction to Machine Learning",
            lesson_number=3
        )
        
        # Verify search was called with both filters
        course_search_tool.store.search.assert_called_once_with(
            query="neural networks",
            course_name="Introduction to Machine Learning",
            lesson_number=3
        )
    
    def test_execute_empty_results_no_filters(self, course_search_tool, mock_search_results_empty):
        """Test handling of empty search results without filters"""
        course_search_tool.store.search.return_value = mock_search_results_empty
        
        result = course_search_tool.execute("nonexistent topic")
        
        assert result == "No relevant content found."
        assert len(course_search_tool.last_sources) == 0
    
    def test_execute_empty_results_with_course_filter(self, course_search_tool, mock_search_results_empty):
        """Test handling of empty search results with course filter"""
        course_search_tool.store.search.return_value = mock_search_results_empty
        
        result = course_search_tool.execute("nonexistent topic", course_name="Nonexistent Course")
        
        assert result == "No relevant content found in course 'Nonexistent Course'."
    
    def test_execute_empty_results_with_lesson_filter(self, course_search_tool, mock_search_results_empty):
        """Test handling of empty search results with lesson filter"""
        course_search_tool.store.search.return_value = mock_search_results_empty
        
        result = course_search_tool.execute("nonexistent topic", lesson_number=5)
        
        assert result == "No relevant content found in lesson 5."
    
    def test_execute_empty_results_with_both_filters(self, course_search_tool, mock_search_results_empty):
        """Test handling of empty search results with both filters"""
        course_search_tool.store.search.return_value = mock_search_results_empty
        
        result = course_search_tool.execute(
            "nonexistent topic", 
            course_name="Nonexistent Course",
            lesson_number=5
        )
        
        assert result == "No relevant content found in course 'Nonexistent Course' in lesson 5."
    
    def test_execute_search_error(self, course_search_tool, mock_search_results_error):
        """Test handling of search errors"""
        course_search_tool.store.search.return_value = mock_search_results_error
        
        result = course_search_tool.execute("any query")
        
        assert result == "Database connection failed"
        assert len(course_search_tool.last_sources) == 0
    
    def test_execute_malformed_metadata(self, course_search_tool):
        """Test handling of malformed metadata in search results"""
        malformed_results = create_search_results(
            documents=["Some content"],
            metadata=[{}],  # Missing required fields
            distances=[0.1]
        )
        course_search_tool.store.search.return_value = malformed_results
        
        result = course_search_tool.execute("test query")
        
        # Should handle missing metadata gracefully
        assert "[unknown]" in result
        assert "Some content" in result
    
    def test_execute_missing_lesson_number_in_metadata(self, course_search_tool):
        """Test handling when lesson_number is missing from metadata"""
        results_no_lesson = create_search_results(
            documents=["Course content without lesson info"],
            metadata=[{
                "course_title": "Test Course",
                # lesson_number is missing
                "chunk_index": 0
            }],
            distances=[0.1]
        )
        course_search_tool.store.search.return_value = results_no_lesson
        
        result = course_search_tool.execute("test query")
        
        # Should handle missing lesson number gracefully
        assert "[Test Course]" in result  # No lesson number in header
        assert "Course content without lesson info" in result
        
        # Source should not include lesson number
        assert len(course_search_tool.last_sources) == 1
        assert course_search_tool.last_sources[0]["text"] == "Test Course"
        assert course_search_tool.last_sources[0]["url"] is None
    
    def test_format_results_with_lesson_links(self, course_search_tool, mock_search_results_success):
        """Test that lesson links are correctly retrieved and included in sources"""
        course_search_tool.store.search.return_value = mock_search_results_success
        course_search_tool.store.get_lesson_link.return_value = "https://example.com/lesson-link"
        
        result = course_search_tool.execute("test query")
        
        # Verify lesson link was requested for each result with lesson number
        assert course_search_tool.store.get_lesson_link.call_count == 2
        course_search_tool.store.get_lesson_link.assert_any_call("Introduction to Machine Learning", 1)
        course_search_tool.store.get_lesson_link.assert_any_call("Introduction to Machine Learning", 2)
        
        # Verify sources include URLs
        assert len(course_search_tool.last_sources) == 2
        for source in course_search_tool.last_sources:
            assert source["url"] == "https://example.com/lesson-link"
    
    def test_execute_edge_case_empty_query(self, course_search_tool, mock_search_results_empty):
        """Test behavior with empty query string"""
        course_search_tool.store.search.return_value = mock_search_results_empty
        
        result = course_search_tool.execute("")
        
        course_search_tool.store.search.assert_called_once_with(
            query="",
            course_name=None,
            lesson_number=None
        )
        assert "No relevant content found" in result
    
    def test_execute_with_special_characters_in_query(self, course_search_tool, mock_search_results_success):
        """Test handling of special characters in query"""
        course_search_tool.store.search.return_value = mock_search_results_success
        
        special_query = "What is C++ programming? & how does it work?"
        result = course_search_tool.execute(special_query)
        
        course_search_tool.store.search.assert_called_once_with(
            query=special_query,
            course_name=None,
            lesson_number=None
        )
        
        # Should process normally
        assert "[Introduction to Machine Learning" in result


class TestToolManager:
    """Unit tests for ToolManager class"""
    
    def test_register_tool(self, mock_vector_store):
        """Test tool registration"""
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        
        manager.register_tool(search_tool)
        
        assert "search_course_content" in manager.tools
        assert manager.tools["search_course_content"] == search_tool
    
    def test_get_tool_definitions(self, tool_manager):
        """Test getting tool definitions"""
        definitions = tool_manager.get_tool_definitions()
        
        assert len(definitions) == 1
        assert definitions[0]["name"] == "search_course_content"
    
    def test_execute_tool_success(self, tool_manager, mock_search_results_success):
        """Test successful tool execution"""
        tool_manager.tools["search_course_content"].store.search.return_value = mock_search_results_success
        
        result = tool_manager.execute_tool("search_course_content", query="test")
        
        assert "Introduction to Machine Learning" in result
    
    def test_execute_tool_not_found(self, tool_manager):
        """Test execution of non-existent tool"""
        result = tool_manager.execute_tool("nonexistent_tool", query="test")
        
        assert result == "Tool 'nonexistent_tool' not found"
    
    def test_get_last_sources(self, tool_manager, mock_search_results_success):
        """Test getting sources from last search"""
        tool_manager.tools["search_course_content"].store.search.return_value = mock_search_results_success
        
        # Execute a search first
        tool_manager.execute_tool("search_course_content", query="test")
        
        sources = tool_manager.get_last_sources()
        assert len(sources) == 2
        assert sources[0]["text"] == "Introduction to Machine Learning - Lesson 1"
    
    def test_reset_sources(self, tool_manager, mock_search_results_success):
        """Test resetting sources"""
        tool_manager.tools["search_course_content"].store.search.return_value = mock_search_results_success
        
        # Execute a search first
        tool_manager.execute_tool("search_course_content", query="test")
        assert len(tool_manager.get_last_sources()) == 2
        
        # Reset sources
        tool_manager.reset_sources()
        assert len(tool_manager.get_last_sources()) == 0
    
    def test_register_tool_without_name(self, mock_vector_store):
        """Test registering tool without proper name"""
        manager = ToolManager()
        
        # Create a mock tool with invalid definition
        bad_tool = Mock()
        bad_tool.get_tool_definition.return_value = {"description": "A tool without name"}
        
        with pytest.raises(ValueError, match="Tool must have a 'name' in its definition"):
            manager.register_tool(bad_tool)