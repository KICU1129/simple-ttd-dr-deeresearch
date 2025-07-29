import pytest
from unittest.mock import MagicMock, patch
from src.ttd_dr.tools import evaluate_quality, get_tavily_mcp_client, EVALUATION_PROMPT_TEMPLATE
from strands import Agent
from mcp import StdioServerParameters, stdio_client

@pytest.fixture
def mock_agent_response():
    """Fixture to mock the Agent's __call__ method for evaluate_quality."""
    with patch('strands.Agent.__call__') as mock_call:
        mock_call.return_value = """
<thinking>
This is a mock thinking process.
</thinking>
<scores>
Helpfulness: 4
Comprehensiveness: 3
</scores>
<feedback>
This is mock feedback for improvement.
</feedback>
"""
        yield mock_call

def test_evaluate_quality_parses_llm_response(mock_agent_response):
    """
    Test that evaluate_quality correctly parses the LLM's structured response.
    """
    query = "Test query"
    text = "Test text to evaluate"
    
    result = evaluate_quality(query, text)
    
    assert result["helpfulness_score"] == 4
    assert result["comprehensiveness_score"] == 3
    assert result["feedback"] == "This is mock feedback for improvement."
    
    # Verify the agent was called with the correct prompt
    expected_prompt = EVALUATION_PROMPT_TEMPLATE.format(query=query, text=text)
    mock_agent_response.assert_called_once_with(expected_prompt)

def test_evaluate_quality_handles_missing_scores_and_feedback():
    """
    Test that evaluate_quality provides default values if parsing fails.
    """
    with patch('strands.Agent.__call__') as mock_call:
        mock_call.return_value = "Malformed response without scores or feedback."
        
        query = "Another query"
        text = "Another text"
        
        result = evaluate_quality(query, text)
        
        assert result["helpfulness_score"] == 1
        assert result["comprehensiveness_score"] == 1
        assert result["feedback"] == "No specific feedback provided."

@patch('src.ttd_dr.tools.MCPClient')
@patch('src.ttd_dr.tools.stdio_client') # Corrected patch path
@patch('src.ttd_dr.tools.StdioServerParameters') # Corrected patch path
def test_get_tavily_mcp_client_initializes_correctly(mock_stdio_server_params_class, mock_stdio_client, mock_mcp_client):
    """
    Test that get_tavily_mcp_client initializes MCPClient with a callable
    that correctly configures the stdio server parameters.
    """
    # When StdioServerParameters is called (as a constructor), it will return this mock
    mock_stdio_server_params_instance = MagicMock()
    mock_stdio_server_params_class.return_value = mock_stdio_server_params_instance

    get_tavily_mcp_client()
    
    mock_mcp_client.assert_called_once()
    callable_arg = mock_mcp_client.call_args[0][0]
    callable_arg() # Execute the lambda

    # Assert that the StdioServerParameters class was called as a constructor
    mock_stdio_server_params_class.assert_called_once_with(
        command="uvx",
        args=["--from", "github.com/tavily-ai/tavily-mcp@latest", "tavily-mcp-server"]
    )
    
    # Assert that stdio_client was called with the instance returned by StdioServerParameters
    mock_stdio_client.assert_called_once_with(mock_stdio_server_params_instance)
