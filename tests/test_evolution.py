import pytest
from unittest.mock import MagicMock, patch
from src.ttd_dr.evolution import SelfEvolutionManager

@pytest.fixture
def evolution_manager():
    """Fixture for SelfEvolutionManager."""
    return SelfEvolutionManager()

def test_decide_evolution_continue_low_scores(evolution_manager):
    """
    Test that evolution continues if scores are low and max iterations not reached.
    """
    current_iteration = 1
    max_iterations = 5
    evaluation_scores = {"helpfulness": 3, "comprehensiveness": 3}
    feedback = "Needs more detail."
    
    should_continue = evolution_manager.decide_evolution(
        current_iteration, max_iterations, evaluation_scores, feedback
    )
    assert should_continue is True

def test_decide_evolution_continue_good_scores_with_specific_feedback(evolution_manager):
    """
    Test that evolution continues if scores are good but feedback suggests refinement.
    """
    current_iteration = 1
    max_iterations = 5
    evaluation_scores = {"helpfulness": 4, "comprehensiveness": 4}
    feedback = "Some information is still missing."
    
    should_continue = evolution_manager.decide_evolution(
        current_iteration, max_iterations, evaluation_scores, feedback
    )
    assert should_continue is True

def test_decide_evolution_stop_good_scores_general_feedback(evolution_manager):
    """
    Test that evolution stops if scores are good and feedback is general.
    """
    current_iteration = 1
    max_iterations = 5
    evaluation_scores = {"helpfulness": 4, "comprehensiveness": 4}
    feedback = "Overall good quality."
    
    should_continue = evolution_manager.decide_evolution(
        current_iteration, max_iterations, evaluation_scores, feedback
    )
    assert should_continue is False

def test_decide_evolution_stop_max_iterations_reached(evolution_manager):
    """
    Test that evolution stops if max iterations are reached, regardless of scores.
    """
    current_iteration = 5
    max_iterations = 5
    evaluation_scores = {"helpfulness": 2, "comprehensiveness": 2}
    feedback = "Still needs a lot of work."
    
    should_continue = evolution_manager.decide_evolution(
        current_iteration, max_iterations, evaluation_scores, feedback
    )
    assert should_continue is False

def test_decide_evolution_edge_case_max_iterations_and_good_scores(evolution_manager):
    """
    Test edge case where max iterations are reached and scores are good.
    """
    current_iteration = 5
    max_iterations = 5
    evaluation_scores = {"helpfulness": 5, "comprehensiveness": 5}
    feedback = "Perfect."
    
    should_continue = evolution_manager.decide_evolution(
        current_iteration, max_iterations, evaluation_scores, feedback
    )
    assert should_continue is False
