import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rag_system import RAGSystem
from config import Config
from vector_store import SearchResults
from conftest import create_search_results


class TestRAGSystemIntegration:
    """Integration tests for the complete RAG system query processing"""
    
    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing"""
        config = Config()
        config.ANTHROPIC_API_KEY = "test-api-key"
        config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
        config.CHROMA_PATH = "./test_chroma_db"
        config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
        config.MAX_RESULTS = 3
        config.CHUNK_SIZE = 500
        config.CHUNK_OVERLAP = 50
        config.MAX_HISTORY = 2
        return config
    
    @pytest.fixture
    def rag_system(self, mock_config):
        """RAG system instance with mocked dependencies"""
        with patch('rag_system.DocumentProcessor'), \
             patch('rag_system.VectorStore') as mock_vector_store, \
             patch('rag_system.AIGenerator') as mock_ai_generator, \
             patch('rag_system.SessionManager') as mock_session_manager:
            
            # Create RAG system
            system = RAGSystem(mock_config)
            
            # Setup mocks
            system.vector_store = mock_vector_store.return_value
            system.ai_generator = mock_ai_generator.return_value
            system.session_manager = mock_session_manager.return_value
            
            return system
    
    def test_query_successful_with_tool_use(self, rag_system, mock_search_results_success):
        """Test successful query processing with tool usage"""
        # Mock vector store search
        rag_system.vector_store.search.return_value = mock_search_results_success
        
        # Mock AI generator response with tool usage
        rag_system.ai_generator.generate_response.return_value = "Machine learning is a subset of AI that focuses on algorithms learning from data."
        
        # Mock session manager
        rag_system.session_manager.get_conversation_history.return_value = None
        
        # Setup tool manager sources
        mock_sources = [
            {"text": "Introduction to Machine Learning - Lesson 1", "url": "https://example.com/lesson1"},
            {"text": "Introduction to Machine Learning - Lesson 2", "url": "https://example.com/lesson2"}
        ]
        rag_system.tool_manager.get_last_sources = Mock(return_value=mock_sources)
        rag_system.tool_manager.reset_sources = Mock()
        
        result, sources = rag_system.query("What is machine learning?")
        
        # Verify AI generator was called with correct parameters
        rag_system.ai_generator.generate_response.assert_called_once()
        call_args = rag_system.ai_generator.generate_response.call_args[1]
        
        assert call_args["query"] == "Answer this question about course materials: What is machine learning?"
        assert call_args["conversation_history"] is None
        assert call_args["tools"] == rag_system.tool_manager.get_tool_definitions()
        assert call_args["tool_manager"] == rag_system.tool_manager
        
        # Verify response and sources
        assert result == "Machine learning is a subset of AI that focuses on algorithms learning from data."
        assert sources == mock_sources
        
        # Verify sources were retrieved and reset
        rag_system.tool_manager.get_last_sources.assert_called_once()
        rag_system.tool_manager.reset_sources.assert_called_once()
    
    def test_query_with_session_history(self, rag_system):
        """Test query processing with existing session history"""
        session_id = "test-session-123"
        mock_history = "User: Previous question\nAssistant: Previous answer"
        
        # Mock session manager
        rag_system.session_manager.get_conversation_history.return_value = mock_history
        rag_system.session_manager.add_exchange = Mock()
        
        # Mock AI generator
        rag_system.ai_generator.generate_response.return_value = "This is the response."
        
        # Mock tool manager
        rag_system.tool_manager.get_last_sources = Mock(return_value=[])
        rag_system.tool_manager.reset_sources = Mock()
        
        result, sources = rag_system.query("Follow up question", session_id=session_id)
        
        # Verify session history was retrieved
        rag_system.session_manager.get_conversation_history.assert_called_once_with(session_id)
        
        # Verify conversation history was passed to AI generator
        call_args = rag_system.ai_generator.generate_response.call_args[1]
        assert call_args["conversation_history"] == mock_history
        
        # Verify exchange was added to session
        rag_system.session_manager.add_exchange.assert_called_once_with(
            session_id, "Follow up question", "This is the response."
        )
    
    def test_query_without_session_id(self, rag_system):
        """Test query processing without session ID (no history)"""
        # Mock AI generator
        rag_system.ai_generator.generate_response.return_value = "Response without history."
        
        # Mock tool manager
        rag_system.tool_manager.get_last_sources = Mock(return_value=[])
        rag_system.tool_manager.reset_sources = Mock()
        
        result, sources = rag_system.query("What is AI?")
        
        # Verify no session history operations
        rag_system.session_manager.get_conversation_history.assert_not_called()
        rag_system.session_manager.add_exchange.assert_not_called()
        
        # Verify AI generator was called without history
        call_args = rag_system.ai_generator.generate_response.call_args[1]
        assert call_args["conversation_history"] is None
    
    def test_query_ai_generator_exception(self, rag_system):
        """Test handling of AI generator exceptions"""
        # Mock AI generator to raise exception
        rag_system.ai_generator.generate_response.side_effect = Exception("API Error")
        
        with pytest.raises(Exception, match="API Error"):
            rag_system.query("test query")
    
    def test_query_tool_manager_exception(self, rag_system):
        """Test handling when tool manager operations fail"""
        # Mock AI generator to succeed
        rag_system.ai_generator.generate_response.return_value = "Response"
        
        # Mock tool manager to fail
        rag_system.tool_manager.get_last_sources.side_effect = Exception("Tool error")
        
        with pytest.raises(Exception, match="Tool error"):
            rag_system.query("test query")
    
    def test_add_course_document_success(self, rag_system, sample_courses, sample_course_chunks):
        """Test successful course document addition"""
        # Mock document processor
        rag_system.document_processor.process_course_document.return_value = (
            sample_courses[0], sample_course_chunks[:3]
        )
        
        # Mock vector store methods
        rag_system.vector_store.add_course_metadata = Mock()
        rag_system.vector_store.add_course_content = Mock()
        
        course, chunk_count = rag_system.add_course_document("/path/to/course.txt")
        
        # Verify processing was called
        rag_system.document_processor.process_course_document.assert_called_once_with("/path/to/course.txt")
        
        # Verify vector store operations
        rag_system.vector_store.add_course_metadata.assert_called_once_with(sample_courses[0])
        rag_system.vector_store.add_course_content.assert_called_once_with(sample_course_chunks[:3])
        
        # Verify return values
        assert course == sample_courses[0]
        assert chunk_count == 3
    
    def test_add_course_document_processing_error(self, rag_system):
        """Test handling of document processing errors"""
        # Mock document processor to raise exception
        rag_system.document_processor.process_course_document.side_effect = Exception("Processing error")
        
        course, chunk_count = rag_system.add_course_document("/invalid/path.txt")
        
        # Should handle error gracefully
        assert course is None
        assert chunk_count == 0
    
    def test_add_course_folder_with_existing_courses(self, rag_system, sample_courses):
        """Test adding course folder with some existing courses"""
        # Mock existing course titles
        rag_system.vector_store.get_existing_course_titles.return_value = [
            "Introduction to Machine Learning"  # First course already exists
        ]
        
        # Mock os.path operations
        with patch('os.path.exists', return_value=True), \
             patch('os.listdir', return_value=["course1.txt", "course2.txt", "invalid.pdf"]), \
             patch('os.path.isfile', return_value=True):
            
            # Mock document processing - only course2 should be processed
            rag_system.document_processor.process_course_document.side_effect = [
                (sample_courses[0], []),  # course1 - already exists
                (sample_courses[1], [Mock(), Mock()])  # course2 - new course
            ]
            
            # Mock vector store methods
            rag_system.vector_store.add_course_metadata = Mock()
            rag_system.vector_store.add_course_content = Mock()
            
            total_courses, total_chunks = rag_system.add_course_folder("/path/to/courses")
            
            # Only new course should be added
            assert total_courses == 1
            assert total_chunks == 2
            
            # Verify only the new course was added to vector store
            rag_system.vector_store.add_course_metadata.assert_called_once_with(sample_courses[1])
    
    def test_add_course_folder_clear_existing(self, rag_system):
        """Test adding course folder with clear_existing=True"""
        # Mock vector store clear operation
        rag_system.vector_store.clear_all_data = Mock()
        rag_system.vector_store.get_existing_course_titles.return_value = []
        
        with patch('os.path.exists', return_value=True), \
             patch('os.listdir', return_value=[]):
            
            rag_system.add_course_folder("/path/to/courses", clear_existing=True)
            
            # Verify data was cleared
            rag_system.vector_store.clear_all_data.assert_called_once()
    
    def test_get_course_analytics(self, rag_system):
        """Test course analytics retrieval"""
        # Mock vector store analytics methods
        rag_system.vector_store.get_course_count.return_value = 5
        rag_system.vector_store.get_existing_course_titles.return_value = [
            "Course 1", "Course 2", "Course 3", "Course 4", "Course 5"
        ]
        
        analytics = rag_system.get_course_analytics()
        
        assert analytics["total_courses"] == 5
        assert len(analytics["course_titles"]) == 5
        assert "Course 1" in analytics["course_titles"]
    
    def test_rag_system_initialization(self, mock_config):
        """Test RAG system component initialization"""
        with patch('rag_system.DocumentProcessor') as mock_doc_processor, \
             patch('rag_system.VectorStore') as mock_vector_store, \
             patch('rag_system.AIGenerator') as mock_ai_generator, \
             patch('rag_system.SessionManager') as mock_session_manager, \
             patch('rag_system.ToolManager') as mock_tool_manager, \
             patch('rag_system.CourseSearchTool') as mock_search_tool:
            
            system = RAGSystem(mock_config)
            
            # Verify all components were initialized with correct parameters
            mock_doc_processor.assert_called_once_with(mock_config.CHUNK_SIZE, mock_config.CHUNK_OVERLAP)
            mock_vector_store.assert_called_once_with(mock_config.CHROMA_PATH, mock_config.EMBEDDING_MODEL, mock_config.MAX_RESULTS)
            mock_ai_generator.assert_called_once_with(mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL)
            mock_session_manager.assert_called_once_with(mock_config.MAX_HISTORY)
            
            # Verify tool manager setup
            mock_tool_manager.assert_called_once()
            mock_search_tool.assert_called_once()
    
    @pytest.mark.integration
    def test_end_to_end_query_flow_with_real_components(self, mock_config):
        """Integration test with real components (mocked external dependencies only)"""
        with patch('anthropic.Anthropic') as mock_anthropic, \
             patch('chromadb.PersistentClient') as mock_chroma, \
             patch('sentence_transformers.SentenceTransformer'):
            
            # Setup mock Anthropic responses
            mock_client = Mock()
            mock_anthropic.return_value = mock_client
            
            # Mock tool use response
            mock_tool_response = Mock()
            mock_tool_response.stop_reason = "tool_use"
            
            tool_block = Mock()
            tool_block.type = "tool_use"
            tool_block.name = "search_course_content"
            tool_block.id = "tool_123"
            tool_block.input = {"query": "machine learning"}
            mock_tool_response.content = [tool_block]
            
            # Mock final response
            mock_final_response = Mock()
            mock_final_response.content = [Mock(text="Machine learning is a subset of AI.")]
            
            mock_client.messages.create.side_effect = [mock_tool_response, mock_final_response]
            
            # Setup mock ChromaDB to return search results
            mock_collection = Mock()
            mock_chroma_client = Mock()
            mock_chroma.return_value = mock_chroma_client
            mock_chroma_client.get_or_create_collection.return_value = mock_collection
            
            # Mock search results
            mock_collection.query.return_value = {
                'documents': [["Machine learning content"]],
                'metadatas': [[{"course_title": "ML Course", "lesson_number": 1}]],
                'distances': [[0.1]]
            }
            
            # Create RAG system with real components
            system = RAGSystem(mock_config)
            
            # Execute query
            result, sources = system.query("What is machine learning?")
            
            # Verify end-to-end flow worked
            assert isinstance(result, str)
            assert isinstance(sources, list)
            
            # Verify Anthropic API was called twice (tool use + final response)
            assert mock_client.messages.create.call_count == 2
    
    def test_query_prompt_formatting(self, rag_system):
        """Test that query prompt is correctly formatted"""
        rag_system.ai_generator.generate_response.return_value = "Response"
        rag_system.tool_manager.get_last_sources = Mock(return_value=[])
        rag_system.tool_manager.reset_sources = Mock()
        
        rag_system.query("What is AI?")
        
        # Verify prompt formatting
        call_args = rag_system.ai_generator.generate_response.call_args[1]
        expected_prompt = "Answer this question about course materials: What is AI?"
        assert call_args["query"] == expected_prompt