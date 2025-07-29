"""
This module defines custom tools and MCP client setups for the TTD-DR agent.
"""
from strands.tools import tool
from strands.tools.mcp import MCPClient
from mcp import stdio_client, StdioServerParameters

EVALUATION_PROMPT_TEMPLATE = """
You are an expert evaluator. Your task is to assess the quality of a given text based on specific criteria.

**Criteria:**
- **Helpfulness:** Does the text directly address the user's intent? Is it accurate and easy to understand?
- **Comprehensiveness:** Is any key information missing?

**User Query:**
{query}

**Text to Evaluate:**
{text}

**Instructions:**
First, provide your reasoning in a <thinking> block.
Then, provide a numerical score from 1 (worst) to 5 (best) for both Helpfulness and Comprehensiveness.
Finally, provide constructive feedback for improvement in a <feedback> block.

**Output Format:**
<thinking>...</thinking>
<scores>
Helpfulness: [1-5]
Comprehensiveness: [1-5]
</scores>
<feedback>...</feedback>
"""

@tool
def evaluate_quality(query: str, text: str) -> dict:
    """
    Evaluates the quality of a given text against a user query using an LLM-as-a-judge.
    Returns a dictionary with scores and feedback.
    """
    # In a real implementation, this would call an LLM with the prompt template.
    # For now, we return a mock response.
    print(f"--- Evaluating text for query: {query} ---")
    print(text)
    print("--- End of Evaluation ---")
    
    # Mock response for demonstration purposes
    return {
        "helpfulness_score": 4,
        "comprehensiveness_score": 3,
        "feedback": "The text is good but could be more detailed in section X."
    }

def get_tavily_mcp_client() -> MCPClient:
    """
    Initializes and returns an MCPClient for the Tavily search server.
    
    This follows the best practice of using MCPClient to wrap a stdio server
    process, which will be started automatically by `uvx`.
    """
    # As per .clinerules/04_mcp_usage.md and .clinerules/05_strandagent_best_practices.md,
    # we use MCPClient to connect to the Tavily server.
    client = MCPClient(lambda: stdio_client(
        StdioServerParameters(
            command="uvx", 
            args=["--from", "github.com/tavily-ai/tavily-mcp@latest", "tavily-mcp-server"]
        )
    ))
    return client
