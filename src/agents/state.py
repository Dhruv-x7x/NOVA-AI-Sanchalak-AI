"""
Agent State Schema for SANCHALAK AI
Phase 5.2: LangGraph Agent Framework

Defines the shared state that flows between all agents in the orchestration graph.
"""

from typing import Optional, List, Dict, Any, Literal
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class AgentType(str, Enum):
    """Available agent types"""
    SUPERVISOR = "supervisor"
    DIAGNOSTIC = "diagnostic"
    FORECASTER = "forecaster"
    RESOLVER = "resolver"
    EXECUTOR = "executor"
    COMMUNICATOR = "communicator"


class TaskPriority(str, Enum):
    """Task priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TaskStatus(str, Enum):
    """Task execution status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ESCALATED = "escalated"
    AWAITING_APPROVAL = "awaiting_approval"


class ConfidenceLevel(str, Enum):
    """Confidence levels for recommendations"""
    HIGH = "high"        # >= 85%
    MEDIUM = "medium"    # 60-84%
    LOW = "low"          # < 60%


class Message(BaseModel):
    """Single message in conversation"""
    role: Literal["user", "assistant", "system", "tool"] = "user"
    content: str
    agent: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Evidence(BaseModel):
    """Evidence supporting a hypothesis or recommendation"""
    source: str  # e.g., "patient_data", "pattern_library", "rag_knowledge"
    description: str
    confidence: float = 0.0  # 0-1
    data: Dict[str, Any] = Field(default_factory=dict)


class Hypothesis(BaseModel):
    """Diagnostic hypothesis"""
    hypothesis_id: str
    description: str
    confidence: float = 0.0  # 0-1
    evidence: List[Evidence] = Field(default_factory=list)
    confounders: List[str] = Field(default_factory=list)
    verification_steps: List[str] = Field(default_factory=list)
    root_cause: Optional[str] = None


class Recommendation(BaseModel):
    """Action recommendation"""
    recommendation_id: str
    action: str
    priority: TaskPriority = TaskPriority.MEDIUM
    confidence: float = 0.0
    impact: str = ""
    effort_hours: float = 0.0
    responsible_role: str = ""
    rationale: str = ""
    steps: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    requires_approval: bool = True


class Forecast(BaseModel):
    """Prediction with uncertainty"""
    metric: str
    prediction: float
    lower_bound: float
    upper_bound: float
    confidence: float = 0.0
    timeframe: str = ""
    assumptions: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)


class Communication(BaseModel):
    """Draft communication"""
    communication_id: str
    type: Literal["email", "notification", "report", "alert"] = "email"
    recipient: str
    recipient_role: str = ""
    subject: str = ""
    body: str = ""
    priority: TaskPriority = TaskPriority.MEDIUM
    channel: str = "email"
    requires_approval: bool = True
    attachments: List[str] = Field(default_factory=list)


class ExecutionResult(BaseModel):
    """Result of an executed action"""
    action_id: str
    status: TaskStatus = TaskStatus.PENDING
    result: str = ""
    error: Optional[str] = None
    executed_at: Optional[str] = None
    executed_by: str = "system"
    rollback_available: bool = False


class AgentState(BaseModel):
    """
    Shared state that flows between agents in the LangGraph workflow.
    
    This is the central data structure that all agents read from and write to.
    Each agent processes the state and adds its outputs to the appropriate fields.
    """
    
    # === Task Identity ===
    task_id: str = Field(default_factory=lambda: f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    # === Input ===
    user_query: str = ""
    context: Dict[str, Any] = Field(default_factory=dict)
    
    # === Conversation History ===
    messages: List[Message] = Field(default_factory=list)
    
    # === Routing ===
    current_agent: Optional[AgentType] = None
    next_agent: Optional[AgentType] = None
    agent_sequence: List[str] = Field(default_factory=list)
    
    # === Task Decomposition (Supervisor) ===
    subtasks: List[Dict[str, Any]] = Field(default_factory=list)
    task_priority: TaskPriority = TaskPriority.MEDIUM
    task_status: TaskStatus = TaskStatus.PENDING
    
    # === Diagnostic Outputs ===
    hypotheses: List[Hypothesis] = Field(default_factory=list)
    root_causes: List[str] = Field(default_factory=list)
    investigation_notes: str = ""
    
    # === Forecaster Outputs ===
    forecasts: List[Forecast] = Field(default_factory=list)
    timeline_projection: Optional[str] = None
    risk_assessment: str = ""
    
    # === Resolver Outputs ===
    recommendations: List[Recommendation] = Field(default_factory=list)
    action_plan: str = ""
    cascade_impact: str = ""
    
    # === Executor Outputs ===
    pending_actions: List[Dict[str, Any]] = Field(default_factory=list)
    execution_results: List[ExecutionResult] = Field(default_factory=list)
    requires_human_approval: bool = False
    
    # === Communicator Outputs ===
    communications: List[Communication] = Field(default_factory=list)
    notifications_queued: int = 0
    
    # === Final Output ===
    final_response: str = ""
    summary: str = ""
    
    # === Metadata ===
    total_tokens_used: int = 0
    total_latency_ms: float = 0.0
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    # === Governance ===
    audit_trail: List[Dict[str, Any]] = Field(default_factory=list)
    human_overrides: List[Dict[str, Any]] = Field(default_factory=list)
    
    def add_message(self, role: str, content: str, agent: Optional[str] = None):
        """Add a message to the conversation history"""
        self.messages.append(Message(
            role=role,
            content=content,
            agent=agent
        ))
    
    def add_audit_entry(self, action: str, agent: str, details: Dict[str, Any] = None):
        """Add an entry to the audit trail"""
        self.audit_trail.append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "agent": agent,
            "details": details or {}
        })
    
    def get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Convert numeric confidence to level"""
        if confidence >= 0.85:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.60:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """Get a summary of the state for logging"""
        return {
            "task_id": self.task_id,
            "status": self.task_status.value,
            "priority": self.task_priority.value,
            "current_agent": self.current_agent.value if self.current_agent else None,
            "hypotheses_count": len(self.hypotheses),
            "recommendations_count": len(self.recommendations),
            "forecasts_count": len(self.forecasts),
            "communications_count": len(self.communications),
            "errors_count": len(self.errors),
            "tokens_used": self.total_tokens_used
        }


# Type alias for LangGraph
GraphState = AgentState


def create_initial_state(
    user_query: str,
    context: Optional[Dict[str, Any]] = None,
    priority: TaskPriority = TaskPriority.MEDIUM
) -> AgentState:
    """
    Factory function to create an initial agent state.
    
    Args:
        user_query: The user's question or request
        context: Optional context data (patient info, site info, etc.)
        priority: Task priority level
        
    Returns:
        Initialized AgentState
    """
    state = AgentState(
        user_query=user_query,
        context=context or {},
        task_priority=priority
    )
    
    # Add initial user message
    state.add_message("user", user_query)
    
    # Add audit entry
    state.add_audit_entry(
        action="task_created",
        agent="system",
        details={"query": user_query, "priority": priority.value}
    )
    
    return state