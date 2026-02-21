# src/agents/__init__.py
"""
SANCHALAK AI - Agents Module

Contains:
- LLMWrapper: Unified LLM interface (Ollama/Groq)
- Agent State schemas
- Tool Registry
- Orchestrator
- Enhanced Agents
- Tracing, Memory, Autonomy modules
"""

# Core imports
from .llm_wrapper import LLMWrapper, LLMResponse, get_llm
from .state import (
    AgentState,
    Message,
    Hypothesis,
    Forecast,
    Recommendation,
    Communication,
    Evidence,
    TaskPriority,
    TaskStatus,
    AgentType,
    create_initial_state
)
from .tools import ToolRegistry, get_tool_registry

# Orchestrator requires langgraph - make it optional
try:
    from .orchestrator import AgentOrchestrator, get_orchestrator
except ImportError:
    AgentOrchestrator = None
    get_orchestrator = None

# New modules - import directly without going through __init__ chain
try:
    from .tracing.agent_tracer import get_agent_tracer, AgentTracer, TracingContext
except ImportError:
    get_agent_tracer = None
    AgentTracer = None
    TracingContext = None

try:
    from .memory.conversation_memory import get_conversation_memory, ConversationMemory
except ImportError:
    get_conversation_memory = None
    ConversationMemory = None

try:
    from .autonomy.autonomy_matrix import get_autonomy_matrix, AutonomyMatrix, ActionDecision, RiskLevel
except ImportError:
    get_autonomy_matrix = None
    AutonomyMatrix = None
    ActionDecision = None
    RiskLevel = None

# Try to import enhanced agents (may not exist yet)
try:
    from .supervisor_enhanced import EnhancedSupervisorAgent
except ImportError:
    EnhancedSupervisorAgent = None

try:
    from .diagnostic_enhanced import (
        EnhancedDiagnosticAgent,
        get_diagnostic_agent,
        InvestigationType,
        EvidenceStrength,
        EvidenceChain,
        DiagnosticHypothesis,
        InvestigationResult
    )
    # Rename to avoid conflict with state.Evidence
    from .diagnostic_enhanced import Evidence as DiagnosticEvidence
except ImportError:
    EnhancedDiagnosticAgent = None
    get_diagnostic_agent = None
    InvestigationType = None
    EvidenceStrength = None
    EvidenceChain = None
    DiagnosticHypothesis = None
    InvestigationResult = None
    DiagnosticEvidence = None

__all__ = [
    # LLM
    'LLMWrapper',
    'LLMResponse', 
    'get_llm',
    
    # State
    'AgentState',
    'Message',
    'Hypothesis',
    'Forecast',
    'Recommendation',
    'Communication',
    'Evidence',
    'TaskPriority',
    'TaskStatus',
    'AgentType',
    'create_initial_state',
    
    # Tools
    'ToolRegistry',
    'get_tool_registry',
    
    # Orchestrator
    'AgentOrchestrator',
    'get_orchestrator',
    
    # Agents
    'EnhancedSupervisorAgent',
    'EnhancedDiagnosticAgent',
    'get_diagnostic_agent',
    
    # Diagnostic types
    'InvestigationType',
    'EvidenceStrength',
    'DiagnosticEvidence',
    'EvidenceChain',
    'DiagnosticHypothesis',
    'InvestigationResult'
]