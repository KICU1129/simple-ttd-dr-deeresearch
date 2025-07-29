"""
This module implements the Self-Evolution mechanism for TTD-DR components.
"""
from typing import Dict, Any

class SelfEvolutionManager:
    """
    Manages the self-evolution process by deciding whether to continue
    iterations based on evaluation scores and feedback.
    """
    def __init__(self):
        """
        Initializes the manager. No specific parameters needed for this simple decision logic.
        """
        pass

    def decide_evolution(
        self,
        current_iteration: int,
        max_iterations: int,
        evaluation_scores: Dict[str, int],
        feedback: str
    ) -> bool:
        """
        Decides whether the self-evolution process should continue for another iteration.

        Args:
            current_iteration: The current iteration number (1-indexed).
            max_iterations: The maximum number of allowed iterations.
            evaluation_scores: A dictionary containing 'helpfulness' and 'comprehensiveness' scores (1-5).
            feedback: The feedback provided by the evaluator agent.

        Returns:
            True if evolution should continue, False otherwise.
        """
        print(f"Self-Evolution Decision for Iteration {current_iteration}:")
        print(f"  Scores: Helpfulness={evaluation_scores['helpfulness']}, Comprehensiveness={evaluation_scores['comprehensiveness']}")
        print(f"  Feedback: {feedback}")

        # Rule 1: Always continue if not at max iterations and scores are below a certain threshold
        # This encourages more iterations if the quality is not yet high.
        if current_iteration < max_iterations:
            if evaluation_scores['helpfulness'] < 4 or evaluation_scores['comprehensiveness'] < 4:
                print("  Decision: Continue (scores are not yet optimal).")
                return True
            else:
                # If scores are good, but there's still specific feedback, consider continuing
                # This is a simple heuristic; more complex logic could parse feedback for actionable items.
                if "more detailed" in feedback.lower() or "missing" in feedback.lower():
                    print("  Decision: Continue (scores are good but feedback suggests further refinement).")
                    return True
                else:
                    print("  Decision: Stop (scores are good and feedback is general).")
                    return False
        
        # Rule 2: Stop if max iterations reached
        print("  Decision: Stop (max iterations reached).")
        return False
