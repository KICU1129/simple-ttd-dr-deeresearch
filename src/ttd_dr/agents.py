"""
This module defines the core agents for each stage of the TTD-DR workflow.
"""
from strands import Agent

# --- System Prompts ---

INITIAL_DRAFT_PROMPT = """
You are an expert researcher. Based on the user's query, generate a preliminary, high-level draft for a research report. This draft will serve as a starting point and will be refined later.
"""

PLAN_PROMPT = """
You are a strategic planner. Based on the user's query, create a structured research plan. Outline the key sections and topics that the final report should cover.
"""

QUESTION_PROMPT = """
You are a curious researcher. Based on the research plan and the current draft, generate a targeted search query to fill information gaps or verify existing claims.
"""

ANSWER_PROMPT = """
You are a diligent research assistant. Using the provided search results, synthesize a concise and accurate answer to the given search query.
"""

REVISE_PROMPT = """
You are a meticulous editor. Based on the new information from the latest question-answer pair, revise the previous draft to improve its accuracy, coherence, and comprehensiveness.
"""

FINAL_REPORT_PROMPT = """
You are a professional writer. Synthesize all the gathered information, including the research plan and all question-answer pairs, into a final, comprehensive, and well-structured research report.
"""


# --- Agent Definitions ---

def get_initial_draft_agent() -> Agent:
    """Returns the agent responsible for generating the initial draft."""
    return Agent(system_prompt=INITIAL_DRAFT_PROMPT)

def get_plan_agent() -> Agent:
    """Returns the agent responsible for creating the research plan."""
    return Agent(system_prompt=PLAN_PROMPT)

def get_question_agent() -> Agent:
    """Returns the agent responsible for generating search queries."""
    return Agent(system_prompt=QUESTION_PROMPT)

def get_answer_agent(tools: list) -> Agent:
    """
    Returns the agent responsible for synthesizing answers from search results.
    Requires search tools to be passed.
    """
    return Agent(system_prompt=ANSWER_PROMPT, tools=tools)

def get_revise_agent() -> Agent:
    """Returns the agent responsible for revising the draft."""
    return Agent(system_prompt=REVISE_PROMPT)

def get_final_report_agent() -> Agent:
    """Returns the agent responsible for generating the final report."""
    return Agent(system_prompt=FINAL_REPORT_PROMPT)
