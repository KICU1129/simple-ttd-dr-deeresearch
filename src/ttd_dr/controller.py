"""
This module implements the main controller for the TTD-DR workflow.
"""
import logging
from .state import ResearchState, QAPair
from . import agents
from .tools import get_tavily_mcp_client, evaluate_quality
from .evolution import SelfEvolutionManager

# Set up a logger for this module
logger = logging.getLogger(__name__)

class TTD_DR_Controller:
    """
    Orchestrates the entire Test-Time Diffusion Deep Researcher workflow.
    """
    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.mcp_client = get_tavily_mcp_client()
        self.evolution_manager = SelfEvolutionManager()

    def _update_metrics(self, state: ResearchState, agent_result):
        """Helper to update token counts and citations from an agent result."""
        logger.debug(f"--- METRICS UPDATE ---")
        logger.debug(f"Inspecting agent_result object: {agent_result}")
        logger.debug(f"Type of agent_result: {type(agent_result)}")
        if hasattr(agent_result, '__dict__'):
            logger.debug(f"agent_result.__dict__: {agent_result.__dict__}")
        logger.debug(f"--- END METRICS UPDATE ---")

        # Extract token usage from metadata if available
        if hasattr(agent_result, 'metadata') and 'usage' in agent_result.metadata:
            usage = agent_result.metadata['usage']
            state.total_tokens += usage.get('input_tokens', 0)
            state.total_tokens += usage.get('output_tokens', 0)
            logger.debug(f"Tokens used: Input={usage.get('input_tokens', 0)}, Output={usage.get('output_tokens', 0)}. Total so far: {state.total_tokens}")

        # Extract citations from tool calls if available
        if hasattr(agent_result, 'tool_calls'):
            for tool_call in agent_result.tool_calls:
                if tool_call.tool_name == 'tavily-search' and hasattr(tool_call, 'result'):
                    try:
                        # Assuming the result is a list of dicts with a 'url' key
                        citations = [item['url'] for item in tool_call.result if 'url' in item]
                        if citations:
                            state.citations.extend(citations)
                            logger.info(f"Found {len(citations)} new citations.")
                    except (TypeError, KeyError) as e:
                        logger.warning(f"Could not extract citations from tool result: {e}")


    def run(self, query: str) -> ResearchState:
        """
        Executes the full TTD-DR process for a given query.

        Args:
            query: The user's research query.

        Returns:
            The final state object containing the report and metrics.
        """
        state = ResearchState(initial_query=query)

        with self.mcp_client:
            # --- Stage 1: Initial Planning and Drafting ---
            logger.info("--- Stage 1: Planning and Drafting ---")
            
            logger.info("="*20 + " Running Plan Agent " + "="*20)
            plan_agent = agents.get_plan_agent()
            plan_result = plan_agent(f"Query: {query}")
            self._update_metrics(state, plan_result)
            state.plan = str(plan_result).split('\n')

            logger.info("="*20 + " Running Initial Draft Agent " + "="*20)
            draft_agent = agents.get_initial_draft_agent()
            draft_result = draft_agent(f"Query: {query}")
            self._update_metrics(state, draft_result)
            state.draft = str(draft_result)
            logger.info(f"Initial Draft generated.")
            logger.debug(f"Initial Draft Content:\n{state.draft}\n")

            # --- Stage 2: Iterative Denoising with Retrieval ---
            logger.info("--- Stage 2: Iterative Search and Refinement ---")
            question_agent = agents.get_question_agent()
            
            logger.info("Listing available MCP tools...")
            mcp_tools = self.mcp_client.list_tools_sync()
            try:
                tool_names = [tool.spec.name for tool in mcp_tools]
                logger.info(f"Found tools: {tool_names}")
            except AttributeError:
                logger.warning(f"Could not determine tool names. Found objects: {mcp_tools}")
            
            answer_agent = agents.get_answer_agent(tools=mcp_tools)
            revise_agent = agents.get_revise_agent()

            for i in range(self.max_iterations):
                logger.info(f"\n--- Iteration {i+1}/{self.max_iterations} ---")

                # 2a: Evaluate Quality
                logger.info("="*20 + " Evaluating Quality " + "="*20)
                evaluation_result = evaluate_quality(query=query, text=state.draft)
                logger.info(f"Evaluation Result: {evaluation_result}")

                # 2b: Decide on Self-Evolution
                logger.info("="*20 + " Deciding on Self-Evolution " + "="*20)
                should_continue = self.evolution_manager.decide_evolution(
                    current_iteration=i + 1,
                    max_iterations=self.max_iterations,
                    evaluation_scores={
                        "helpfulness": evaluation_result["helpfulness_score"],
                        "comprehensiveness": evaluation_result["comprehensiveness_score"]
                    },
                    feedback=evaluation_result["feedback"]
                )
                
                if not should_continue:
                    logger.info("Stopping iterations based on self-evolution decision.")
                    break

                # 2c: Generate Search Question
                logger.info("="*20 + " Running Question Agent " + "="*20)
                q_context = f"Query: {query}\nPlan: {state.plan}\nDraft: {state.draft}"
                question_result = question_agent(q_context)
                self._update_metrics(state, question_result)
                question = str(question_result)
                
                # 2d: Search and Synthesize Answer
                logger.info("="*20 + " Running Answer Agent " + "="*20)
                answer_result = answer_agent(f"Question: {question}")
                self._update_metrics(state, answer_result)
                answer = str(answer_result)
                state.qa_history.append(QAPair(question=question, answer=answer))
                
                # 2e: Revise Draft
                logger.info("="*20 + " Running Revise Agent " + "="*20)
                r_context = f"Previous Draft:\n{state.draft}\n\nNew Info:\nQ: {question}\nA: {answer}\n\nEvaluation Feedback:\n{evaluation_result['feedback']}"
                revise_result = revise_agent(r_context)
                self._update_metrics(state, revise_result)
                state.draft = str(revise_result)
                logger.info(f"Revised Draft (Iteration {i+1}) generated.")
                logger.debug(f"Revised Draft Content:\n{state.draft}")

            # --- Stage 3: Final Report Generation ---
            logger.info("\n--- Stage 3: Final Report Generation ---")
            logger.info("="*20 + " Running Final Report Agent " + "="*20)
            final_report_agent = agents.get_final_report_agent()
            final_context = f"Query: {query}\nPlan: {state.plan}\n\n"
            final_context += "\n".join([f"Q: {qa.question}\nA: {qa.answer}" for qa in state.qa_history])
            final_report_result = final_report_agent(final_context)
            self._update_metrics(state, final_report_result)
            state.final_report = str(final_report_result)

        logger.info("\n--- TTD-DR Process Complete ---")
        return state
