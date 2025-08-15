import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ai_generator import AIGenerator
from search_tools import ToolManager, CourseSearchTool


class TestAIGenerator:
    """Unit tests for AIGenerator class focusing on tool calling mechanism"""
    
    def test_init(self, test_config):
        """Test AIGenerator initialization"""
        with patch('anthropic.Anthropic') as mock_anthropic:
            generator = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
            
            mock_anthropic.assert_called_once_with(api_key=test_config.ANTHROPIC_API_KEY)
            assert generator.model == test_config.ANTHROPIC_MODEL
            assert generator.base_params["model"] == test_config.ANTHROPIC_MODEL
            assert generator.base_params["temperature"] == 0
            assert generator.base_params["max_tokens"] == 800
    
    def test_generate_response_without_tools(self, ai_generator, mock_anthropic_response_no_tools):
        """Test basic response generation without tools"""
        ai_generator.client.messages.create.return_value = mock_anthropic_response_no_tools
        
        result = ai_generator.generate_response("What is machine learning?")
        
        # Verify API call parameters
        ai_generator.client.messages.create.assert_called_once()
        call_args = ai_generator.client.messages.create.call_args[1]
        
        assert call_args["model"] == "claude-sonnet-4-20250514"
        assert call_args["temperature"] == 0
        assert call_args["max_tokens"] == 800
        assert call_args["messages"] == [{"role": "user", "content": "What is machine learning?"}]
        assert call_args["system"] == ai_generator.SYSTEM_PROMPT
        assert "tools" not in call_args
        
        assert result == "This is a direct response without tool usage."
    
    def test_generate_response_with_conversation_history(self, ai_generator, mock_anthropic_response_no_tools):
        """Test response generation with conversation history"""
        ai_generator.client.messages.create.return_value = mock_anthropic_response_no_tools
        
        history = "Previous conversation context"
        result = ai_generator.generate_response("Follow up question", conversation_history=history)
        
        call_args = ai_generator.client.messages.create.call_args[1]
        expected_system = f"{ai_generator.SYSTEM_PROMPT}\n\nPrevious conversation:\n{history}"
        assert call_args["system"] == expected_system
    
    def test_generate_response_with_tools_no_tool_use(self, ai_generator, mock_anthropic_response_no_tools, tool_manager):
        """Test response generation with tools available but not used"""
        ai_generator.client.messages.create.return_value = mock_anthropic_response_no_tools
        
        tools = tool_manager.get_tool_definitions()
        result = ai_generator.generate_response(
            "What is 2+2?", 
            tools=tools, 
            tool_manager=tool_manager
        )
        
        # Verify tools were passed to API
        call_args = ai_generator.client.messages.create.call_args[1]
        assert call_args["tools"] == tools
        assert call_args["tool_choice"] == {"type": "auto"}
        
        assert result == "This is a direct response without tool usage."
    
    def test_generate_response_with_tool_use(self, ai_generator, mock_anthropic_response_with_tools, 
                                           mock_anthropic_final_response, tool_manager, mock_search_results_success):
        """Test complete tool calling flow"""
        # Mock the search tool to return results
        tool_manager.tools["search_course_content"].store.search.return_value = mock_search_results_success
        
        # First call returns tool use, second call returns final response
        ai_generator.client.messages.create.side_effect = [
            mock_anthropic_response_with_tools,
            mock_anthropic_final_response
        ]
        
        tools = tool_manager.get_tool_definitions()
        result = ai_generator.generate_response(
            "What is machine learning?",
            tools=tools,
            tool_manager=tool_manager
        )
        
        # Verify first API call (with tools)
        first_call_args = ai_generator.client.messages.create.call_args_list[0][1]
        assert first_call_args["tools"] == tools
        assert first_call_args["tool_choice"] == {"type": "auto"}
        
        # Verify tool was executed
        tool_manager.tools["search_course_content"].store.search.assert_called_once_with(
            query="machine learning",
            course_name="Introduction to Machine Learning",
            lesson_number=None
        )
        
        # Verify second API call (without tools, with tool results)
        second_call_args = ai_generator.client.messages.create.call_args_list[1][1]
        assert "tools" not in second_call_args
        assert len(second_call_args["messages"]) == 3  # Original + assistant + tool result
        
        # Check that tool result was included
        tool_result_message = second_call_args["messages"][2]
        assert tool_result_message["role"] == "user"
        assert tool_result_message["content"][0]["type"] == "tool_result"
        assert tool_result_message["content"][0]["tool_use_id"] == "tool_123"
        
        assert result == "Based on the search results, machine learning is a subset of AI."
    
    def test_handle_tool_execution_with_multiple_tools(self, ai_generator, tool_manager, mock_search_results_success):
        """Test handling multiple tool calls in one response"""
        # Create mock response with multiple tool uses
        mock_response = Mock()
        mock_response.stop_reason = "tool_use"
        
        # Two tool use blocks
        tool_block1 = Mock()
        tool_block1.type = "tool_use"
        tool_block1.name = "search_course_content"
        tool_block1.id = "tool_123"
        tool_block1.input = {"query": "machine learning"}
        
        tool_block2 = Mock()
        tool_block2.type = "tool_use"
        tool_block2.name = "search_course_content" 
        tool_block2.id = "tool_456"
        tool_block2.input = {"query": "neural networks"}
        
        mock_response.content = [tool_block1, tool_block2]
        
        # Mock tool manager to return results
        tool_manager.execute_tool.side_effect = ["Result 1", "Result 2"]
        
        base_params = {
            "messages": [{"role": "user", "content": "test"}],
            "system": "test system"
        }
        
        ai_generator.client.messages.create.return_value = Mock(
            content=[Mock(text="Final response")]
        )
        
        result = ai_generator._handle_tool_execution(mock_response, base_params, tool_manager)
        
        # Verify both tools were executed
        assert tool_manager.execute_tool.call_count == 2
        tool_manager.execute_tool.assert_any_call("search_course_content", query="machine learning")
        tool_manager.execute_tool.assert_any_call("search_course_content", query="neural networks")
        
        # Verify final API call includes both tool results
        final_call_args = ai_generator.client.messages.create.call_args[1]
        tool_results = final_call_args["messages"][2]["content"]
        assert len(tool_results) == 2
        assert tool_results[0]["tool_use_id"] == "tool_123"
        assert tool_results[1]["tool_use_id"] == "tool_456"
        
        assert result == "Final response"
    
    def test_handle_tool_execution_with_mixed_content(self, ai_generator, tool_manager):
        """Test handling response with both text and tool use content"""
        mock_response = Mock()
        mock_response.stop_reason = "tool_use"
        
        # Text block and tool use block
        text_block = Mock()
        text_block.type = "text"
        text_block.text = "Let me search for that information."
        
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = "tool_123"
        tool_block.input = {"query": "test"}
        
        mock_response.content = [text_block, tool_block]
        
        tool_manager.execute_tool.return_value = "Tool result"
        
        base_params = {
            "messages": [{"role": "user", "content": "test"}],
            "system": "test system"
        }
        
        ai_generator.client.messages.create.return_value = Mock(
            content=[Mock(text="Final response")]
        )
        
        result = ai_generator._handle_tool_execution(mock_response, base_params, tool_manager)
        
        # Only tool use blocks should be executed
        tool_manager.execute_tool.assert_called_once_with("search_course_content", query="test")
        
        # Assistant message should include all content
        final_call_args = ai_generator.client.messages.create.call_args[1]
        assistant_message = final_call_args["messages"][1]
        assert assistant_message["role"] == "assistant"
        assert assistant_message["content"] == [text_block, tool_block]
    
    def test_tool_execution_error_handling(self, ai_generator, tool_manager):
        """Test error handling when tool execution fails"""
        mock_response = Mock()
        mock_response.stop_reason = "tool_use"
        
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = "tool_123"
        tool_block.input = {"query": "test"}
        
        mock_response.content = [tool_block]
        
        # Tool execution returns error message
        tool_manager.execute_tool.return_value = "Database connection failed"
        
        base_params = {
            "messages": [{"role": "user", "content": "test"}],
            "system": "test system"
        }
        
        ai_generator.client.messages.create.return_value = Mock(
            content=[Mock(text="I apologize, there was an error")]
        )
        
        result = ai_generator._handle_tool_execution(mock_response, base_params, tool_manager)
        
        # Error should be passed as tool result
        final_call_args = ai_generator.client.messages.create.call_args[1]
        tool_result = final_call_args["messages"][2]["content"][0]
        assert tool_result["content"] == "Database connection failed"
        
        assert result == "I apologize, there was an error"
    
    def test_generate_response_api_error(self, ai_generator):
        """Test handling of Anthropic API errors"""
        ai_generator.client.messages.create.side_effect = Exception("API Error")
        
        with pytest.raises(Exception, match="API Error"):
            ai_generator.generate_response("test query")
    
    def test_system_prompt_content(self, ai_generator):
        """Test that system prompt contains expected instructions"""
        prompt = ai_generator.SYSTEM_PROMPT
        
        # Check for key instructions
        assert "search tool" in prompt.lower()
        assert "one search per query maximum" in prompt.lower()
        assert "course-specific questions" in prompt.lower()
        assert "brief, concise" in prompt.lower()
        assert "no meta-commentary" in prompt.lower()
    
    def test_base_params_configuration(self, test_config):
        """Test that base parameters are correctly configured"""
        with patch('anthropic.Anthropic'):
            generator = AIGenerator(test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL)
            
            assert generator.base_params["model"] == test_config.ANTHROPIC_MODEL
            assert generator.base_params["temperature"] == 0
            assert generator.base_params["max_tokens"] == 800
    
    def test_handle_tool_execution_no_tool_results(self, ai_generator, tool_manager):
        """Test handling when no tool results are generated"""
        mock_response = Mock()
        mock_response.stop_reason = "tool_use"
        mock_response.content = []  # No content blocks
        
        base_params = {
            "messages": [{"role": "user", "content": "test"}],
            "system": "test system"
        }
        
        ai_generator.client.messages.create.return_value = Mock(
            content=[Mock(text="No tools were executed")]
        )
        
        result = ai_generator._handle_tool_execution(mock_response, base_params, tool_manager)
        
        # No tools should be executed
        tool_manager.execute_tool.assert_not_called()
        
        # Should still make final API call
        final_call_args = ai_generator.client.messages.create.call_args[1]
        assert len(final_call_args["messages"]) == 2  # Original + assistant (no tool results)
        
        assert result == "No tools were executed"
    
    def test_conversation_history_formatting(self, ai_generator, mock_anthropic_response_no_tools):
        """Test that conversation history is properly formatted in system prompt"""
        ai_generator.client.messages.create.return_value = mock_anthropic_response_no_tools
        
        # Test with no history
        ai_generator.generate_response("test")
        call_args = ai_generator.client.messages.create.call_args[1]
        assert call_args["system"] == ai_generator.SYSTEM_PROMPT
        
        # Test with history
        ai_generator.client.messages.create.reset_mock()
        history = "User: Previous question\nAssistant: Previous answer"
        ai_generator.generate_response("test", conversation_history=history)
        
        call_args = ai_generator.client.messages.create.call_args[1]
        expected_system = f"{ai_generator.SYSTEM_PROMPT}\n\nPrevious conversation:\n{history}"
        assert call_args["system"] == expected_system