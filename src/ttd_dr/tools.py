"""
This module defines custom tools and MCP client setups for the TTD-DR agent.
"""
from strands import Agent
from strands.tools import tool
from strands.tools.mcp import MCPClient
from mcp import stdio_client, StdioServerParameters
import re
import os
from dotenv import load_dotenv
load_dotenv()

EVALUATION_PROMPT_TEMPLATE = """
あなたは専門の評価者です。特定の基準に基づいて、与えられたテキストの品質を評価することがあなたのタスクです。

**基準:**
- **Helpfulness (有用性):** テキストはユーザーの意図に直接対応していますか？正確で理解しやすいですか？
- **Comprehensiveness (網羅性):** 重要な情報が欠落していませんか？

**ユーザーのクエリ:**
{query}

**評価するテキスト:**
{text}

**指示:**
まず、<thinking>ブロックであなたの推論を提供してください。
次に、HelpfulnessとComprehensivenessの両方について、1（最低）から5（最高）までの数値スコアを提供してください。
最後に、<feedback>ブロックで改善のための建設的なフィードバックを提供してください。

**出力形式:**
<thinking>...</thinking>
<scores>
Helpfulness: [1-5]
Comprehensiveness: [1-5]
</scores>
<feedback>...</feedback>
"""

def evaluate_quality(query: str, text: str) -> dict:
    """
    Evaluates the quality of a given text against a user query using an LLM-as-a-judge.
    Returns a dictionary with scores and feedback.
    """
    evaluator_agent = Agent(system_prompt=EVALUATION_PROMPT_TEMPLATE)
    
    # Construct the full prompt for the evaluator agent
    full_prompt = EVALUATION_PROMPT_TEMPLATE.format(query=query, text=text)
    
    # Call the evaluator agent
    agent_result = evaluator_agent(full_prompt)
    response_text = str(agent_result)
    
    # Parse the LLM's response
    helpfulness_match = re.search(r"Helpfulness: (\d)", response_text)
    comprehensiveness_match = re.search(r"Comprehensiveness: (\d)", response_text)
    feedback_match = re.search(r"<feedback>(.*?)</feedback>", response_text, re.DOTALL)
    
    helpfulness_score = int(helpfulness_match.group(1)) if helpfulness_match else 1
    comprehensiveness_score = int(comprehensiveness_match.group(1)) if comprehensiveness_match else 1
    feedback = feedback_match.group(1).strip() if feedback_match else "No specific feedback provided."
    
    return {
        "helpfulness_score": helpfulness_score,
        "comprehensiveness_score": comprehensiveness_score,
        "feedback": feedback
    }

def get_tavily_mcp_client() -> MCPClient:
    """
    Initializes and returns an MCPClient for the Tavily search server.
    
    This follows the best practice of using MCPClient to wrap a stdio server
    process. It loads the TAVILY_API_KEY from the .env file and passes it
    as an environment variable to the server process.
    """
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY not found in .env file.")

    # The command and arguments are based on the user-provided MCP settings.
    # This correctly points to the local Node.js-based Tavily MCP server.
    client = MCPClient(lambda: stdio_client(
        StdioServerParameters(
            command="node",
            args=["C:\\Users\\kisuk\\Documents\\Cline\\MCP\\tavily-mcp\\build\\index.js"],
            env={"TAVILY_API_KEY": tavily_api_key}
        )
    ))
    return client
