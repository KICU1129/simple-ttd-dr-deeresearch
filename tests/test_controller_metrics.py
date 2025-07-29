import pytest
from unittest.mock import MagicMock, patch
from src.ttd_dr.controller import TTD_DR_Controller
from src.ttd_dr.state import ResearchState

@pytest.fixture
def controller():
    """Fixture to create a TTD_DR_Controller instance."""
    with patch('src.ttd_dr.controller.get_tavily_mcp_client'):
        yield TTD_DR_Controller()

def test_update_metrics_with_token_usage(controller):
    """
    Tests that _update_metrics correctly adds token usage.
    """
    # Arrange
    state = ResearchState(initial_query="test")
    assert state.total_tokens == 0

    agent_result = MagicMock()
    agent_result.metadata = {
        'usage': {
            'input_tokens': 100,
            'output_tokens': 200
        }
    }
    agent_result.tool_calls = []

    # Act
    controller._update_metrics(state, agent_result)

    # Assert
    assert state.total_tokens == 300

def test_update_metrics_with_citations(controller):
    """
    Tests that _update_metrics correctly extracts and adds citations.
    """
    # Arrange
    state = ResearchState(initial_query="test")
    assert state.citations == []

    tool_call = MagicMock()
    tool_call.tool_name = 'tavily-search'
    tool_call.result = [
        {'url': 'https://example.com/page1'},
        {'url': 'https://example.com/page2'}
    ]
    
    agent_result = MagicMock()
    agent_result.metadata = {}
    agent_result.tool_calls = [tool_call]

    # Act
    controller._update_metrics(state, agent_result)

    # Assert
    assert len(state.citations) == 2
    assert 'https://example.com/page1' in state.citations
    assert 'https://example.com/page2' in state.citations

def test_update_metrics_with_no_data(controller):
    """
    Tests that _update_metrics handles agent results with no relevant data.
    """
    # Arrange
    state = ResearchState(initial_query="test")
    state.total_tokens = 50
    state.citations = ['http://existing.com']

    agent_result = MagicMock()
    agent_result.metadata = {}
    agent_result.tool_calls = []

    # Act
    controller._update_metrics(state, agent_result)

    # Assert
    assert state.total_tokens == 50
    assert len(state.citations) == 1

def test_update_metrics_with_malformed_citation_result(controller):
    """
    Tests that _update_metrics handles tool results that are not as expected.
    """
    # Arrange
    state = ResearchState(initial_query="test")
    
    tool_call = MagicMock()
    tool_call.tool_name = 'tavily-search'
    tool_call.result = "this is not a list of dicts" # Malformed result

    agent_result = MagicMock()
    agent_result.metadata = {}
    agent_result.tool_calls = [tool_call]

    # Act
    controller._update_metrics(state, agent_result)

    # Assert
    assert len(state.citations) == 0
