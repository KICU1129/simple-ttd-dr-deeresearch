"""
This module implements the main controller for the TTD-DR workflow.
"""
from .state import ResearchState, QAPair
from . import agents
from .tools import get_tavily_mcp_client

class TTD_DR_Controller:
    """
    Orchestrates the entire Test-Time Diffusion Deep Researcher workflow.
    """
    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        # In a real scenario, we would initialize the MCP client here.
        # self.mcp_client = get_tavily_mcp_client()

    def run(self, query: str) -> str:
        """
        Executes the full TTD-DR process for a given query.

        Args:
            query: The user's research query.

        Returns:
            The final generated research report.
        """
        state = ResearchState(initial_query=query)

        # --- Stage 1: Initial Planning and Drafting ---
        print("--- Stage 1: Planning and Drafting ---")
        plan_agent = agents.get_plan_agent()
        state.plan = plan_agent(f"Query: {query}").split('\n')

        draft_agent = agents.get_initial_draft_agent()
        state.draft = draft_agent(f"Query: {query}")
        print(f"Initial Draft:\n{state.draft}\n")

        # --- Stage 2: Iterative Denoising with Retrieval ---
        print("--- Stage 2: Iterative Search and Refinement ---")
        question_agent = agents.get_question_agent()
        # The answer agent would use real tools in a full implementation
        # mcp_tools = self.mcp_client.list_tools_sync()
        # answer_agent = agents.get_answer_agent(tools=mcp_tools)
        answer_agent = agents.get_answer_agent(tools=[]) # Mock
        revise_agent = agents.get_revise_agent()

        for i in range(self.max_iterations):
            print(f"\n--- Iteration {i+1}/{self.max_iterations} ---")
            
            # 2a: Generate Search Question
            q_context = f"Query: {query}\nPlan: {state.plan}\nDraft: {state.draft}"
            question = question_agent(q_context)
            
            # 2b: Search and Synthesize Answer (mocked)
            # In a real implementation, this would use the search tool.
            answer = answer_agent(f"Question: {question}")
            state.qa_history.append(QAPair(question=question, answer=answer))
            
            # Revise Draft
            r_context = f"Previous Draft:\n{state.draft}\n\nNew Info:\nQ: {question}\nA: {answer}"
            state.draft = revise_agent(r_context)
            print(f"Revised Draft (Iteration {i+1}):\n{state.draft}")

        # --- Stage 3: Final Report Generation ---
        print("\n--- Stage 3: Final Report Generation ---")
        final_report_agent = agents.get_final_report_agent()
        final_context = f"Query: {query}\nPlan: {state.plan}\n\n"
        final_context += "\n".join([f"Q: {qa.question}\nA: {qa.answer}" for qa in state.qa_history])
        state.final_report = final_report_agent(final_context)

        print("\n--- TTD-DR Process Complete ---")
        return state.final_report
