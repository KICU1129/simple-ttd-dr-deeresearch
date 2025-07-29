"""
This module defines the core agents for each stage of the TTD-DR workflow.
"""
from strands import Agent
from strands.models.bedrock import BedrockModel

# Define the correct Bedrock model ID to be used by all agents
# As per user request, using a Claude 3.7 Sonnet model.
BEDROCK_MODEL_ID = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
BEDROCK_MODEL = BedrockModel(model_id=BEDROCK_MODEL_ID)

# --- System Prompts ---

INITIAL_DRAFT_PROMPT = """
あなたは専門の研究者です。ユーザーのクエリに基づいて、調査レポートの予備的な高レベルのドラフトを生成してください。このドラフトは出発点として機能し、後で洗練されます。
"""

PLAN_PROMPT = """
あなたは戦略的なプランナーです。ユーザーのクエリに基づいて、構造化された調査計画を作成してください。最終レポートがカバーすべき主要なセクションとトピックを概説してください。
この計画は、潜在的な情報ギャップを特定し、それらを埋めるための戦略を含める必要があります。
"""

QUESTION_PROMPT = """
あなたは好奇心旺盛な研究者です。調査計画と現在のドラフトに基づいて、情報ギャップを埋めるか、既存の主張を検証するためのターゲットを絞った検索クエリを生成してください。
生成するクエリは、`tavily-search`ツールが効果的に利用できる形式である必要があります。
"""

ANSWER_PROMPT = """
あなたは勤勉な調査アシスタントです。あなたの目標は、ユーザーの調査質問に答えることです。
あなたは`tavily-search`ツールにアクセスできます。
ツールを使用して関連情報を検索し、検索結果に基づいて簡潔で正確な回答を合成してください。
"""

REVISE_PROMPT = """
あなたは細心の注意を払う編集者です。最新の質問と回答のペアからの新しい情報と、提供された評価フィードバックに基づいて、以前のドラフトを修正し、その正確性、一貫性、および網羅性を向上させてください。
"""

FINAL_REPORT_PROMPT = """
You are a professional writer. Synthesize all the gathered information, including the research plan and all question-answer pairs, into a final, comprehensive, and well-structured research report.
"""


# --- Agent Definitions ---

def get_initial_draft_agent() -> Agent:
    """Returns the agent responsible for generating the initial draft."""
    return Agent(model=BEDROCK_MODEL, system_prompt=INITIAL_DRAFT_PROMPT)

def get_plan_agent() -> Agent:
    """Returns the agent responsible for creating the research plan."""
    return Agent(model=BEDROCK_MODEL, system_prompt=PLAN_PROMPT)

def get_question_agent() -> Agent:
    """Returns the agent responsible for generating search queries."""
    return Agent(model=BEDROCK_MODEL, system_prompt=QUESTION_PROMPT)

def get_answer_agent(tools: list) -> Agent:
    """
    Returns the agent responsible for synthesizing answers from search results.
    Requires search tools to be passed.
    """
    return Agent(model=BEDROCK_MODEL, system_prompt=ANSWER_PROMPT, tools=tools)

def get_revise_agent() -> Agent:
    """Returns the agent responsible for revising the draft."""
    return Agent(model=BEDROCK_MODEL, system_prompt=REVISE_PROMPT)

def get_final_report_agent() -> Agent:
    """Returns the agent responsible for generating the final report."""
    return Agent(model=BEDROCK_MODEL, system_prompt=FINAL_REPORT_PROMPT)
