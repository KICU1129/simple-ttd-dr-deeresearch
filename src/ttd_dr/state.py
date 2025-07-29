"""
This module defines the Pydantic models for managing the state of the TTD-DR workflow.
"""
from pydantic import BaseModel, Field
from typing import List, Dict

class QAPair(BaseModel):
    """A model to store a question and its corresponding answer."""
    question: str
    answer: str

class EvolutionVariant(BaseModel):
    """
    Represents a single variant within the self-evolution process,
    including its content, score, and feedback.
    """
    content: str
    fitness_score: float = 0.0
    feedback: str = ""

class ResearchState(BaseModel):
    """
    Manages the overall state of the TTD-DR agent throughout its execution.
    """
    initial_query: str
    plan: List[str] = Field(default_factory=list)
    draft: str = ""
    qa_history: List[QAPair] = Field(default_factory=list)
    final_report: str = ""
    
    # A temporary state to hold variants during the self-evolution process.
    # The key could be the stage name, e.g., "plan", "question".
    evolution_variants: Dict[str, List[EvolutionVariant]] = Field(default_factory=dict)
