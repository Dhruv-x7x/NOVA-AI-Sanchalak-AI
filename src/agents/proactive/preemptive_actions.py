"""
a6on-i - Preemptive Actions v1.0
===========================================
Take preemptive actions before issues escalate.
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field

from .pattern_watcher import PatternAlert, get_pattern_watcher

logger = logging.getLogger(__name__)


@dataclass
class PreemptiveAction:
    """A preemptive action to take."""
    action_id: str
    alert_id: str
    action_type: str
    description: str
    target_entities: List[str]
    parameters: Dict[str, Any]
    auto_executable: bool = False
    priority: str = "normal"
    created_at: datetime = field(default_factory=datetime.now)
    executed: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "action_id": self.action_id,
            "alert_id": self.alert_id,
            "action_type": self.action_type,
            "description": self.description,
            "target_entities": self.target_entities,
            "parameters": self.parameters,
            "auto_executable": self.auto_executable,
            "priority": self.priority,
            "created_at": self.created_at.isoformat(),
            "executed": self.executed
        }


class PreemptiveActionEngine:
    """
    Generate and execute preemptive actions based on detected patterns.
    
    Actions:
    - Schedule priority reviews
    - Send early warnings
    - Adjust monitoring thresholds
    - Trigger focused investigations
    """
    
    def __init__(self):
        self._watcher = get_pattern_watcher()
        self._action_generators: Dict[str, Callable] = {}
        self._pending_actions: List[PreemptiveAction] = []
        self._executed_actions: List[PreemptiveAction] = []
        
        # Register default action generators
        self._register_default_generators()
    
    def _register_default_generators(self):
        """Register default action generators."""
        self.register_generator("dqi_degradation", self._gen_dqi_actions)
        self.register_generator("query_spike", self._gen_query_actions)
        self.register_generator("clean_rate_decline", self._gen_clean_rate_actions)
        self.register_generator("site_underperformance", self._gen_site_actions)
        self.register_generator("resolution_slowdown", self._gen_resolution_actions)
    
    def register_generator(self, pattern_type: str, generator: Callable):
        """Register an action generator for a pattern type."""
        self._action_generators[pattern_type] = generator
    
    def generate_actions(self, alert: PatternAlert) -> List[PreemptiveAction]:
        """Generate preemptive actions for an alert."""
        generator = self._action_generators.get(alert.pattern_type)
        
        if not generator:
            logger.debug(f"No action generator for pattern: {alert.pattern_type}")
            return []
        
        actions = generator(alert)
        self._pending_actions.extend(actions)
        return actions
    
    def process_alerts(self) -> List[PreemptiveAction]:
        """Process all active alerts and generate actions."""
        alerts = self._watcher.get_active_alerts()
        all_actions = []
        
        for alert in alerts:
            actions = self.generate_actions(alert)
            all_actions.extend(actions)
        
        return all_actions
    
    # Default action generators
    
    def _gen_dqi_actions(self, alert: PatternAlert) -> List[PreemptiveAction]:
        """Generate actions for DQI degradation."""
        import uuid
        actions = []
        
        actions.append(PreemptiveAction(
            action_id=f"act_{uuid.uuid4().hex[:8]}",
            alert_id=alert.alert_id,
            action_type="schedule_review",
            description="Schedule priority review for affected site",
            target_entities=alert.affected_entities,
            parameters={
                "review_type": "data_quality",
                "urgency": "high" if alert.severity == "critical" else "normal"
            },
            auto_executable=False,
            priority="high"
        ))
        
        actions.append(PreemptiveAction(
            action_id=f"act_{uuid.uuid4().hex[:8]}",
            alert_id=alert.alert_id,
            action_type="send_notification",
            description="Notify site coordinator of DQI trend",
            target_entities=alert.affected_entities,
            parameters={
                "notification_type": "early_warning",
                "message": f"DQI scores trending down: {alert.description}"
            },
            auto_executable=True,
            priority="normal"
        ))
        
        return actions
    
    def _gen_query_actions(self, alert: PatternAlert) -> List[PreemptiveAction]:
        """Generate actions for query spike."""
        import uuid
        
        return [PreemptiveAction(
            action_id=f"act_{uuid.uuid4().hex[:8]}",
            alert_id=alert.alert_id,
            action_type="allocate_resources",
            description="Flag for additional query resolution resources",
            target_entities=[],
            parameters={"resource_type": "query_resolution", "increase_factor": 1.5},
            auto_executable=False,
            priority="high"
        )]
    
    def _gen_clean_rate_actions(self, alert: PatternAlert) -> List[PreemptiveAction]:
        """Generate actions for clean rate decline."""
        import uuid
        
        return [
            PreemptiveAction(
                action_id=f"act_{uuid.uuid4().hex[:8]}",
                alert_id=alert.alert_id,
                action_type="escalate_to_lead",
                description="Escalate clean rate decline to study lead",
                target_entities=[],
                parameters={"escalation_level": "study_lead", "reason": alert.description},
                auto_executable=True,
                priority="critical"
            ),
            PreemptiveAction(
                action_id=f"act_{uuid.uuid4().hex[:8]}",
                alert_id=alert.alert_id,
                action_type="generate_report",
                description="Generate detailed clean rate analysis report",
                target_entities=[],
                parameters={"report_type": "clean_rate_analysis", "period": "7_days"},
                auto_executable=True,
                priority="high"
            )
        ]
    
    def _gen_site_actions(self, alert: PatternAlert) -> List[PreemptiveAction]:
        """Generate actions for site underperformance."""
        import uuid
        
        return [PreemptiveAction(
            action_id=f"act_{uuid.uuid4().hex[:8]}",
            alert_id=alert.alert_id,
            action_type="site_support_outreach",
            description="Initiate support outreach to underperforming site",
            target_entities=alert.affected_entities,
            parameters={"support_type": "quality_improvement", "method": "call"},
            auto_executable=False,
            priority="normal"
        )]
    
    def _gen_resolution_actions(self, alert: PatternAlert) -> List[PreemptiveAction]:
        """Generate actions for resolution slowdown."""
        import uuid
        
        return [PreemptiveAction(
            action_id=f"act_{uuid.uuid4().hex[:8]}",
            alert_id=alert.alert_id,
            action_type="analyze_bottleneck",
            description="Analyze resolution workflow for bottlenecks",
            target_entities=[],
            parameters={"analysis_type": "workflow_efficiency"},
            auto_executable=True,
            priority="low"
        )]
    
    def get_pending_actions(self, priority: Optional[str] = None) -> List[PreemptiveAction]:
        """Get pending actions."""
        actions = [a for a in self._pending_actions if not a.executed]
        if priority:
            actions = [a for a in actions if a.priority == priority]
        return actions
    
    def execute_action(self, action_id: str) -> bool:
        """Mark an action as executed."""
        for action in self._pending_actions:
            if action.action_id == action_id:
                action.executed = True
                self._executed_actions.append(action)
                return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        pending = [a for a in self._pending_actions if not a.executed]
        return {
            "generators_registered": len(self._action_generators),
            "pending_actions": len(pending),
            "executed_actions": len(self._executed_actions),
            "auto_executable_pending": len([a for a in pending if a.auto_executable]),
            "critical_pending": len([a for a in pending if a.priority == "critical"])
        }


# Singleton
_engine: Optional[PreemptiveActionEngine] = None


def get_preemptive_engine() -> PreemptiveActionEngine:
    """Get the global PreemptiveActionEngine instance."""
    global _engine
    if _engine is None:
        _engine = PreemptiveActionEngine()
    return _engine
