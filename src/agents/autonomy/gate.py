"""
a6on-i - Agent Autonomy Gate
Enforces the Layer 5 Autonomy Matrix (Confidence vs Risk).
"""

from enum import Enum
from typing import Dict, Any

class AutonomyDecision(Enum):
    AUTO_EXECUTE = "auto-execute"
    AUTO_DRAFT = "auto-draft"
    RECOMMEND = "recommend"
    ESCALATE = "escalate"
    ESCALATE_URGENT = "escalate-urgent"

class AutonomyGate:
    """
    Centralized logic gate for agent action autonomy.
    Follows PRD Line 309 Autonomy Matrix.
    """
    
    @staticmethod
    def get_decision(confidence: float, risk_level: str) -> AutonomyDecision:
        """
        Determines the autonomy level for a proposed action.
        
        Args:
            confidence: 0-1 score of the agent's confidence in the resolution.
            risk_level: 'low', 'medium', or 'high'.
            
        Returns:
            AutonomyDecision enum value.
        """
        risk = risk_level.lower().strip()
        
        # PRD Row 1: >= 95% Confidence
        if confidence >= 0.95:
            if risk == "low": 
                return AutonomyDecision.AUTO_EXECUTE
            if risk == "medium": 
                return AutonomyDecision.AUTO_DRAFT
            return AutonomyDecision.RECOMMEND # High Risk
            
        # PRD Row 2: 80-94% Confidence
        if confidence >= 0.80:
            if risk == "low": 
                return AutonomyDecision.AUTO_DRAFT
            if risk == "medium": 
                return AutonomyDecision.RECOMMEND
            return AutonomyDecision.ESCALATE # High Risk
            
        # PRD Row 3: < 80% Confidence
        if risk == "low": 
            return AutonomyDecision.RECOMMEND
        if risk == "medium": 
            return AutonomyDecision.ESCALATE
        return AutonomyDecision.ESCALATE_URGENT # High Risk
        
    @staticmethod
    def get_action_reasoning(decision: AutonomyDecision, confidence: float, risk_level: str) -> str:
        """Generates a human-readable reasoning for the autonomy decision."""
        mapping = {
            AutonomyDecision.AUTO_EXECUTE: "Confidence is exceptional (>=95%) and risk is low. Auto-executing remediation to optimize trial velocity.",
            AutonomyDecision.AUTO_DRAFT: "High confidence remediation detected. Auto-drafting communication for human review.",
            AutonomyDecision.RECOMMEND: "Action recommended for human verification due to risk/confidence profile.",
            AutonomyDecision.ESCALATE: "Significant risk detected. Escalating to Study Lead for mandatory review.",
            AutonomyDecision.ESCALATE_URGENT: "Low confidence/High risk collision. Urgent escalation to clinical governance required."
        }
        return mapping.get(decision, "Awaiting human-in-the-loop validation.")
