"""
This module implements the Self-Evolution mechanism for TTD-DR components.
"""
from typing import List, Callable
from .state import EvolutionVariant
from .tools import evaluate_quality

class SelfEvolutionManager:
    """
    Manages the self-evolution process for a given agent's output.
    """
    def __init__(
        self,
        generation_agent_func: Callable,
        revision_agent_func: Callable,
        num_variants: int = 3,
        num_evolutions: int = 1
    ):
        """
        Initializes the manager.

        Args:
            generation_agent_func: A function that returns a configured generation agent.
            revision_agent_func: A function that returns a configured revision agent.
            num_variants: The number of initial variants to generate.
            num_evolutions: The number of evolution (revision) steps for each variant.
        """
        self.generation_agent_func = generation_agent_func
        self.revision_agent_func = revision_agent_func
        self.num_variants = num_variants
        self.num_evolutions = num_evolutions

    def run(self, query: str, context: str) -> str:
        """
        Runs the full self-evolution process.

        Args:
            query: The initial user query.
            context: The context for the generation (e.g., plan, draft).

        Returns:
            The best-evolved content.
        """
        # 1. Generate initial variants
        generation_agent = self.generation_agent_func()
        variants = [
            EvolutionVariant(content=generation_agent(context))
            for _ in range(self.num_variants)
        ]

        # 2. Evolve each variant
        for _ in range(self.num_evolutions):
            revision_agent = self.revision_agent_func()
            for variant in variants:
                # Evaluate
                eval_result = evaluate_quality(query=query, text=variant.content)
                variant.fitness_score = (eval_result["helpfulness_score"] + eval_result["comprehensiveness_score"]) / 2
                variant.feedback = eval_result["feedback"]

                # Revise
                revision_prompt = f"Based on the following feedback, revise the text.\n\nFeedback: {variant.feedback}\n\nOriginal Text:\n{variant.content}"
                variant.content = revision_agent(revision_prompt)

        # 3. Select the best variant
        best_variant = max(variants, key=lambda v: v.fitness_score)
        
        return best_variant.content
