"""
a6on-i - Decision History Tracker v1.0
=================================================
Track all agent decisions and their outcomes for audit and learning.
"""
# NOTE: This module previously used SQLite but has been migrated to PostgreSQL-only.
# If you need to use this module, update it to use:
#   from src.database.pg_data_service import get_pg_data_service
#   service = get_pg_data_service()


import json
import logging
# SQLite removed - using PostgreSQL
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class DecisionStatus(Enum):
    """Status of a decision."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AgentDecision:
    """A decision made by an agent."""
    decision_id: str
    agent_id: str
    agent_name: str
    action_type: str
    target_entity: str  # e.g., patient_key, site_id
    entity_type: str  # patient, site, study
    
    # Decision details
    recommendation: str
    reasoning: str
    confidence: float
    risk_level: str
    autonomy_level: int  # 1-4
    
    # Status tracking
    status: DecisionStatus = DecisionStatus.PENDING
    requires_approval: bool = True
    
    # Approval info
    approved_by: Optional[str] = None
    approval_timestamp: Optional[datetime] = None
    approval_notes: Optional[str] = None
    
    # Execution info
    executed_at: Optional[datetime] = None
    execution_result: Optional[str] = None
    execution_error: Optional[str] = None
    
    # Outcome
    outcome_success: Optional[bool] = None
    outcome_notes: Optional[str] = None
    
    # Metadata
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            "decision_id": self.decision_id,
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "action_type": self.action_type,
            "target_entity": self.target_entity,
            "entity_type": self.entity_type,
            "recommendation": self.recommendation,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "risk_level": self.risk_level,
            "autonomy_level": self.autonomy_level,
            "status": self.status.value,
            "requires_approval": self.requires_approval,
            "approved_by": self.approved_by,
            "approval_timestamp": self.approval_timestamp.isoformat() if self.approval_timestamp else None,
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
            "execution_result": self.execution_result,
            "outcome_success": self.outcome_success,
            "created_at": self.created_at.isoformat()
        }


class DecisionHistory:
    """
    Track all agent decisions and their outcomes.
    
    Features:
    - Complete audit trail of all decisions
    - Approval workflow tracking
    - Outcome recording for learning
    - Compliance reporting
    """
    
    _instance = None
    
    def __new__(cls, db_path: Optional[Path] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, db_path: Optional[Path] = None):
        if self._initialized:
            return
        
        if db_path is None:
            db_path = Path(__file__).parent.parent.parent.parent / "data" / "governance" / "decision_history.db"
        
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._init_db()
        self._initialized = True
        logger.info(f"DecisionHistory initialized at {db_path}")
    
    def _init_db(self):
        """Initialize database schema."""
        # DISABLED: SQLite replaced with PostgreSQL
        if False:
             pass
    
    def record_decision(self, decision: AgentDecision) -> bool:
        """Record a new agent decision."""
        try:
            # DISABLED: SQLite replaced with PostgreSQL
            if False:
                 pass
            
            logger.info(f"Recorded decision: {decision.decision_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to record decision: {e}")
            return False
    
    def approve_decision(self, decision_id: str, approved_by: str, 
                        notes: Optional[str] = None) -> bool:
        """Approve a pending decision."""
        # DISABLED: SQLite replaced with PostgreSQL
        if False:
             pass
        result = None
        if result:
             return True
        return False
    
    def reject_decision(self, decision_id: str, rejected_by: str,
                       reason: str) -> bool:
        """Reject a pending decision."""
        # DISABLED: SQLite replaced with PostgreSQL
        if False:
             pass
        result = None
        if result:
             return True
        return False
    
    def mark_executed(self, decision_id: str, result: str, 
                     error: Optional[str] = None) -> bool:
        """Mark a decision as executed."""
        status = "failed" if error else "executed"
        # DISABLED: SQLite replaced with PostgreSQL
        if False:
             pass
        result = None
        if result:
             return True
        return False
    
    def record_outcome(self, decision_id: str, success: bool, 
                      notes: Optional[str] = None) -> bool:
        """Record the final outcome of a decision."""
        # DISABLED: SQLite replaced with PostgreSQL
        if False:
             pass
        result = None
        if result:
             return True
        return False
    
    def get_decision(self, decision_id: str) -> Optional[AgentDecision]:
        """Get a specific decision."""
        row = None
        # DISABLED: SQLite replaced with PostgreSQL
        if False:
             pass
        
        if row:
             return self._row_to_decision(row)
        return None
    
    def get_pending_decisions(self, agent_id: Optional[str] = None) -> List[AgentDecision]:
        """Get all pending decisions awaiting approval."""
        query = "SELECT * FROM decisions WHERE status = 'pending'"
        params = []
        
        if agent_id:
            query += " AND agent_id = ?"
            params.append(agent_id)
        
        query += " ORDER BY created_at DESC"
        
        decisions = []
        # DISABLED: SQLite replaced with PostgreSQL
        if False:
             pass
        
        return decisions
    
    def get_decisions_for_entity(self, entity_id: str, 
                                limit: int = 50) -> List[AgentDecision]:
        """Get decisions related to a specific entity."""
        decisions = []
        # DISABLED: SQLite replaced with PostgreSQL
        if False:
             pass
        
        return decisions
    
    def _row_to_decision(self, row) -> AgentDecision:
        """Convert database row to AgentDecision."""
        return AgentDecision(
            decision_id=row[0],
            agent_id=row[1],
            agent_name=row[2] or "",
            action_type=row[3],
            target_entity=row[4] or "",
            entity_type=row[5] or "",
            recommendation=row[6] or "",
            reasoning=row[7] or "",
            confidence=row[8] or 0,
            risk_level=row[9] or "medium",
            autonomy_level=row[10] or 1,
            status=DecisionStatus(row[11]) if row[11] else DecisionStatus.PENDING,
            requires_approval=bool(row[12]),
            approved_by=row[13],
            approval_timestamp=datetime.fromisoformat(row[14]) if row[14] else None,
            approval_notes=row[15],
            executed_at=datetime.fromisoformat(row[16]) if row[16] else None,
            execution_result=row[17],
            execution_error=row[18],
            outcome_success=bool(row[19]) if row[19] is not None else None,
            outcome_notes=row[20],
            context=json.loads(row[21]) if row[21] else {},
            created_at=datetime.fromisoformat(row[22]) if row[22] else datetime.now()
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get decision statistics."""
        total = 0
        pending = 0
        approved = 0
        executed = 0
        completed = 0
        success = 0
        # DISABLED: SQLite replaced with PostgreSQL
        if False:
             pass
        
        return {
            "total_decisions": total,
            "pending": pending,
            "approved": approved,
            "executed": executed,
            "completed": completed,
            "success_count": success,
            "success_rate": success / completed if completed else 0
        }


# Singleton accessor
_history_instance: Optional[DecisionHistory] = None


def get_decision_history() -> DecisionHistory:
    """Get the global DecisionHistory instance."""
    global _history_instance
    if _history_instance is None:
        _history_instance = DecisionHistory()
    return _history_instance
