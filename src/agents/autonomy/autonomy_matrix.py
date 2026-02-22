"""
a6on-i - Autonomy Matrix v1.0
Risk-based action classification for agent autonomy.

Implements the Autonomy Matrix:
                    │ Low Risk      │ Medium Risk   │ High Risk
────────────────────┼───────────────┼───────────────┼──────────────
≥95% Confidence     │ AUTO-EXECUTE  │ AUTO-DRAFT    │ RECOMMEND
80-94% Confidence   │ AUTO-DRAFT    │ RECOMMEND     │ ESCALATE
<80% Confidence     │ RECOMMEND     │ ESCALATE      │ ESCALATE+URGENT
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk levels for actions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ActionDecision(Enum):
    """Possible decisions for an action."""
    AUTO_EXECUTE = "auto_execute"    # Execute without human approval
    AUTO_DRAFT = "auto_draft"        # Create draft for quick approval
    RECOMMEND = "recommend"          # Present as recommendation
    ESCALATE = "escalate"            # Require human review
    ESCALATE_URGENT = "escalate_urgent"  # Urgent escalation
    BLOCKED = "blocked"              # Never allow (safety-critical)


@dataclass
class ActionClassification:
    """Classification result for an action."""
    action_type: str
    risk_level: RiskLevel
    confidence: float
    decision: ActionDecision
    reasoning: str
    requires_human: bool
    estimated_impact: str
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'action_type': self.action_type,
            'risk_level': self.risk_level.value,
            'confidence': self.confidence,
            'decision': self.decision.value,
            'reasoning': self.reasoning,
            'requires_human': self.requires_human,
            'estimated_impact': self.estimated_impact,
            'warnings': self.warnings
        }


class AutonomyMatrix:
    """
    Implements risk-based autonomy decisions for agent actions.
    
    Based on two dimensions:
    1. Confidence level (how sure is the agent?)
    2. Risk level (what's the potential impact?)
    
    Usage:
        matrix = get_autonomy_matrix()
        
        classification = matrix.classify_action(
            action_type="send_email",
            confidence=0.92,
            context={"recipient": "site_coordinator"}
        )
        
        if classification.decision == ActionDecision.AUTO_EXECUTE:
            # Execute automatically
        elif classification.requires_human:
            # Request human approval
    """
    
    # Actions that should NEVER be auto-executed (from solution doc)
    NEVER_AUTO_LIST = {
        'sae_causality_assessment',
        'protocol_deviation_classification',
        'regulatory_submission',
        'site_closure',
        'medical_judgment',
        'locked_data_modification',
        'patient_withdrawal',
        'unblinding_request',
        'database_lock_decision',
        'audit_finding_response'
    }
    
    # Risk classification rules
    HIGH_RISK_ACTIONS = {
        'send_external_email',
        'modify_patient_data',
        'close_queries_batch',
        'update_sae_status',
        'create_protocol_deviation',
        'submit_regulatory',
        'close_site'
    }
    
    MEDIUM_RISK_ACTIONS = {
        'send_internal_email',
        'create_task',
        'update_issue_status',
        'assign_query',
        'schedule_visit',
        'update_enrollment'
    }
    
    LOW_RISK_ACTIONS = {
        'generate_report',
        'create_draft',
        'search_data',
        'analyze_metrics',
        'send_notification',
        'log_note',
        'add_comment',
        'draft_query_response',
        'update_priority',
        'flag_for_review',
        'update_dqi_note',
        'create_alert',
        'assign_task',
        'update_status',
        'resolve_query'
    }
    
    def __init__(self):
        """Initialize the autonomy matrix."""
        self._matrix = self._build_matrix()
        logger.info("AutonomyMatrix initialized")
    
    def _build_matrix(self) -> Dict[Tuple[str, str], ActionDecision]:
        """Build the decision matrix."""
        return {
            # High confidence (≥95%)
            ('high', 'low'): ActionDecision.AUTO_EXECUTE,
            ('high', 'medium'): ActionDecision.AUTO_DRAFT,
            ('high', 'high'): ActionDecision.RECOMMEND,
            ('high', 'critical'): ActionDecision.ESCALATE,
            
            # Medium confidence (80-94%)
            ('medium', 'low'): ActionDecision.AUTO_DRAFT,
            ('medium', 'medium'): ActionDecision.RECOMMEND,
            ('medium', 'high'): ActionDecision.ESCALATE,
            ('medium', 'critical'): ActionDecision.ESCALATE_URGENT,
            
            # Low confidence (<80%)
            ('low', 'low'): ActionDecision.RECOMMEND,
            ('low', 'medium'): ActionDecision.ESCALATE,
            ('low', 'high'): ActionDecision.ESCALATE_URGENT,
            ('low', 'critical'): ActionDecision.BLOCKED,
        }
    
    def _get_confidence_tier(self, confidence: float) -> str:
        """Map confidence to tier."""
        if confidence >= 0.95:
            return 'high'
        elif confidence >= 0.80:
            return 'medium'
        return 'low'
    
    def _determine_risk_level(self, action_type: str, context: Optional[Dict] = None) -> RiskLevel:
        """Determine risk level for an action."""
        action_lower = action_type.lower().replace(' ', '_')
        
        # Check explicit lists
        if action_lower in self.NEVER_AUTO_LIST:
            return RiskLevel.CRITICAL
        if action_lower in self.HIGH_RISK_ACTIONS:
            return RiskLevel.HIGH
        if action_lower in self.MEDIUM_RISK_ACTIONS:
            return RiskLevel.MEDIUM
        if action_lower in self.LOW_RISK_ACTIONS:
            return RiskLevel.LOW
        
        # Context-based risk assessment
        if context:
            # External recipients = higher risk
            if context.get('recipient_external', False):
                return RiskLevel.HIGH
            # Batch operations = higher risk
            if context.get('batch_size', 0) > 10:
                return RiskLevel.HIGH
            # Safety-related = higher risk
            if context.get('is_safety_related', False):
                return RiskLevel.HIGH
        
        # Default to medium for unknown actions
        return RiskLevel.MEDIUM
    
    def classify_action(
        self,
        action_type: str,
        confidence: float,
        context: Optional[Dict] = None,
        impact_description: str = ""
    ) -> ActionClassification:
        """
        Classify an action and determine the appropriate decision.
        
        Args:
            action_type: Type of action being requested
            confidence: Agent's confidence level (0-1)
            context: Optional context about the action
            impact_description: Description of potential impact
            
        Returns:
            ActionClassification with decision and reasoning
        """
        # Normalize
        action_lower = action_type.lower().replace(' ', '_')
        confidence = max(0.0, min(1.0, confidence))
        
        # Check NEVER_AUTO list first
        if action_lower in self.NEVER_AUTO_LIST:
            return ActionClassification(
                action_type=action_type,
                risk_level=RiskLevel.CRITICAL,
                confidence=confidence,
                decision=ActionDecision.BLOCKED,
                reasoning=f"'{action_type}' is on the NEVER-AUTO list and requires human decision",
                requires_human=True,
                estimated_impact=impact_description or "Critical - requires human judgment",
                warnings=["This action type always requires human approval"]
            )
        
        # Determine risk and confidence tiers
        risk_level = self._determine_risk_level(action_type, context)
        confidence_tier = self._get_confidence_tier(confidence)
        
        # Look up decision in matrix
        risk_key = risk_level.value if risk_level != RiskLevel.CRITICAL else 'critical'
        matrix_key = (confidence_tier, risk_key)
        decision = self._matrix.get(matrix_key, ActionDecision.ESCALATE)
        
        # Build reasoning
        reasoning = self._build_reasoning(action_type, confidence, risk_level, decision)
        
        # Determine if human required
        requires_human = decision in {
            ActionDecision.RECOMMEND,
            ActionDecision.ESCALATE,
            ActionDecision.ESCALATE_URGENT,
            ActionDecision.BLOCKED
        }
        
        # Collect warnings
        warnings = []
        if confidence < 0.80:
            warnings.append("Low confidence - recommend additional verification")
        if risk_level in {RiskLevel.HIGH, RiskLevel.CRITICAL}:
            warnings.append(f"High-risk action: {risk_level.value}")
        if context and context.get('is_safety_related'):
            warnings.append("Safety-related action - extra caution advised")
        
        return ActionClassification(
            action_type=action_type,
            risk_level=risk_level,
            confidence=confidence,
            decision=decision,
            reasoning=reasoning,
            requires_human=requires_human,
            estimated_impact=impact_description,
            warnings=warnings
        )
    
    def _build_reasoning(
        self,
        action_type: str,
        confidence: float,
        risk_level: RiskLevel,
        decision: ActionDecision
    ) -> str:
        """Build human-readable reasoning for the decision."""
        conf_pct = confidence * 100
        
        decision_explanations = {
            ActionDecision.AUTO_EXECUTE: f"Auto-executing: {conf_pct:.0f}% confidence, {risk_level.value} risk",
            ActionDecision.AUTO_DRAFT: f"Creating draft for quick approval: {conf_pct:.0f}% confidence",
            ActionDecision.RECOMMEND: f"Presenting as recommendation: review before proceeding",
            ActionDecision.ESCALATE: f"Escalating for human review: {risk_level.value} risk or low confidence",
            ActionDecision.ESCALATE_URGENT: f"URGENT escalation required: high risk with low confidence",
            ActionDecision.BLOCKED: f"Action blocked: requires human judgment"
        }
        
        return decision_explanations.get(decision, f"Decision: {decision.value}")
    
    def can_auto_execute(self, action_type: str, confidence: float, context: Optional[Dict] = None) -> bool:
        """Quick check if action can be auto-executed."""
        classification = self.classify_action(action_type, confidence, context)
        return classification.decision == ActionDecision.AUTO_EXECUTE
    
    def get_approval_requirements(self, action_type: str) -> Dict[str, Any]:
        """Get approval requirements for an action type."""
        action_lower = action_type.lower().replace(' ', '_')
        
        if action_lower in self.NEVER_AUTO_LIST:
            return {
                'min_approvers': 2,
                'requires_supervisor': True,
                'timeout_hours': 24,
                'can_auto_approve': False,
                'reason': "Safety-critical action"
            }
        elif action_lower in self.HIGH_RISK_ACTIONS:
            return {
                'min_approvers': 1,
                'requires_supervisor': True,
                'timeout_hours': 4,
                'can_auto_approve': False,
                'reason': "High-risk action"
            }
        elif action_lower in self.MEDIUM_RISK_ACTIONS:
            return {
                'min_approvers': 1,
                'requires_supervisor': False,
                'timeout_hours': 2,
                'can_auto_approve': True,
                'reason': "Standard approval"
            }
        else:
            return {
                'min_approvers': 0,
                'requires_supervisor': False,
                'timeout_hours': 0,
                'can_auto_approve': True,
                'reason': "Low-risk action"
            }


# Singleton accessor
_matrix_instance: Optional[AutonomyMatrix] = None


def get_autonomy_matrix() -> AutonomyMatrix:
    """Get the global AutonomyMatrix instance."""
    global _matrix_instance
    if _matrix_instance is None:
        _matrix_instance = AutonomyMatrix()
    return _matrix_instance


# Export public API
__all__ = [
    'AutonomyMatrix',
    'ActionClassification',
    'ActionDecision',
    'RiskLevel',
    'get_autonomy_matrix'
]
