"""
a6on-i - Learning Engine v1.0
=========================================
Feedback loop processor for agent learning.
# NOTE: This module previously used SQLite but has been migrated to PostgreSQL-only.
# If you need to use this module, update it to use:
#   from src.database.pg_data_service import get_pg_data_service
#   service = get_pg_data_service()


Tracks outcomes of agent decisions and adjusts confidence over time.
"""

import json
import logging
# SQLite removed - using PostgreSQL
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class LearningOutcome:
    """Outcome of an agent action for learning."""
    outcome_id: str
    action_type: str
    agent_id: str
    recommendation: str
    actual_result: str  # success, partial, failed, rejected
    user_feedback: Optional[str] = None
    feedback_score: float = 0.0  # -1 to 1
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            "outcome_id": self.outcome_id,
            "action_type": self.action_type,
            "agent_id": self.agent_id,
            "recommendation": self.recommendation,
            "actual_result": self.actual_result,
            "user_feedback": self.user_feedback,
            "feedback_score": self.feedback_score,
            "context": self.context,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class AgentPerformance:
    """Performance metrics for an agent."""
    agent_id: str
    total_actions: int = 0
    successful_actions: int = 0
    failed_actions: int = 0
    rejected_actions: int = 0
    average_feedback_score: float = 0.0
    confidence_adjustment: float = 0.0  # -1 to 1
    last_updated: datetime = field(default_factory=datetime.now)
    
    @property
    def success_rate(self) -> float:
        if self.total_actions == 0:
            return 0.0
        return self.successful_actions / self.total_actions


class LearningEngine:
    """
    Feedback loop processor that learns from agent outcomes.
    
    Features:
    - Track action success/failure rates
    - Adjust confidence thresholds per agent
    - Identify patterns in failures
    - Personalize recommendations based on history
    """
    
    _instance = None
    _lock = None
    
    def __new__(cls, db_path: Optional[Path] = None):
        if cls._instance is None:
            import threading
            cls._lock = threading.Lock()
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, db_path: Optional[Path] = None):
        if self._initialized:
            return
        
        if db_path is None:
            db_path = Path(__file__).parent.parent.parent.parent / "data" / "learning" / "agent_learning.db"
        
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._init_db()
        self._performance_cache: Dict[str, AgentPerformance] = {}
        self._initialized = True
        logger.info(f"LearningEngine initialized at {db_path}")
    
    def _init_db(self):
        """Initialize database schema."""
        # DISABLED: SQLite replaced with PostgreSQL
        if False:
             pass
    
    def record_outcome(self, outcome: LearningOutcome) -> bool:
        """
        Record the outcome of an agent action.
        
        Args:
            outcome: The learning outcome to record
            
        Returns:
            True if recorded successfully
        """
        try:
            # DISABLED: SQLite replaced with PostgreSQL
            if False:
                 pass
            
            # Update performance metrics
            self._update_performance(outcome)
            
            logger.debug(f"Recorded outcome: {outcome.outcome_id} ({outcome.actual_result})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to record outcome: {e}")
            return False
    
    def _update_performance(self, outcome: LearningOutcome):
        """Update agent performance based on outcome."""
        agent_id = outcome.agent_id
        
        # DISABLED: SQLite replaced with PostgreSQL
        if False:
             pass
    
    def get_performance(self, agent_id: str) -> AgentPerformance:
        """Get performance metrics for an agent."""
        # DISABLED: SQLite replaced with PostgreSQL
        if False:
             pass
        return AgentPerformance(agent_id=agent_id)
    
    def get_adjusted_confidence(self, agent_id: str, base_confidence: float) -> float:
        """
        Get adjusted confidence based on agent's historical performance.
        
        Args:
            agent_id: The agent ID
            base_confidence: The base confidence level
            
        Returns:
            Adjusted confidence (0-1)
        """
        perf = self.get_performance(agent_id)
        adjusted = base_confidence + perf.confidence_adjustment
        return max(0.0, min(1.0, adjusted))  # Clamp to 0-1
    
    def get_recent_outcomes(self, agent_id: Optional[str] = None, 
                           action_type: Optional[str] = None,
                           limit: int = 50) -> List[LearningOutcome]:
        """Get recent learning outcomes with optional filters."""
        query = "SELECT * FROM learning_outcomes WHERE 1=1"
        params = []
        
        if agent_id:
            query += " AND agent_id = ?"
            params.append(agent_id)
        
        if action_type:
            query += " AND action_type = ?"
            params.append(action_type)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        outcomes = []
        # DISABLED: SQLite replaced with PostgreSQL
        if False:
             pass
        
        return outcomes
    
    def analyze_failure_patterns(self, action_type: str) -> Dict[str, Any]:
        """Analyze patterns in failed actions."""
        outcomes = self.get_recent_outcomes(action_type=action_type, limit=100)
        
        failures = [o for o in outcomes if o.actual_result in ("failed", "rejected")]
        
        analysis = {
            "total_analyzed": len(outcomes),
            "failure_count": len(failures),
            "failure_rate": len(failures) / len(outcomes) if outcomes else 0,
            "common_contexts": defaultdict(int),
            "improvement_suggestions": []
        }
        
        # Analyze failure contexts
        for f in failures:
            for key, value in f.context.items():
                analysis["common_contexts"][f"{key}={value}"] += 1
        
        # Generate suggestions
        if analysis["failure_rate"] > 0.3:
            analysis["improvement_suggestions"].append(
                "High failure rate detected. Consider lowering autonomy level."
            )
        
        return dict(analysis)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get overall learning statistics."""
        # DISABLED: SQLite replaced with PostgreSQL
        if False:
             pass
        total_outcomes = 0
        success_count = 0
        agent_count = 0
        avg_feedback = 0
        
        return {
            "total_outcomes": total_outcomes,
            "success_count": success_count,
            "success_rate": success_count / total_outcomes if total_outcomes else 0,
            "agent_count": agent_count,
            "average_feedback": avg_feedback
        }


# Singleton accessor
_engine_instance: Optional[LearningEngine] = None


def get_learning_engine() -> LearningEngine:
    """Get the global LearningEngine instance."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = LearningEngine()
    return _engine_instance
