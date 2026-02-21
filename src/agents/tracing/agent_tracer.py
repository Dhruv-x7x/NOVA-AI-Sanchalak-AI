"""
SANCHALAK AI - Agent Tracer v1.0
Comprehensive observability for agent decision chains.

Logs every Thought → Action → Observation in the ReAct loop.
Provides execution timeline, durations, and JSON export.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum
from pathlib import Path
import threading
import uuid

logger = logging.getLogger(__name__)


class TraceStepType(Enum):
    """Types of trace steps."""
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ROUTING = "routing"
    ERROR = "error"
    SYNTHESIS = "synthesis"


@dataclass
class TraceStep:
    """A single step in the agent trace."""
    step_id: str
    step_type: TraceStepType
    agent_name: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_step_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'step_id': self.step_id,
            'step_type': self.step_type.value,
            'agent_name': self.agent_name,
            'content': self.content[:500] if self.content else '',  # Truncate for display
            'timestamp': self.timestamp.isoformat(),
            'duration_ms': self.duration_ms,
            'metadata': self.metadata,
            'parent_step_id': self.parent_step_id
        }


@dataclass
class AgentTrace:
    """Complete trace for an agent execution."""
    trace_id: str
    query: str
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    steps: List[TraceStep] = field(default_factory=list)
    agent_sequence: List[str] = field(default_factory=list)
    total_tool_calls: int = 0
    total_llm_calls: int = 0
    status: str = "running"
    final_response: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    
    @property
    def total_duration_ms(self) -> float:
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return (datetime.now() - self.started_at).total_seconds() * 1000
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'trace_id': self.trace_id,
            'query': self.query[:200] if self.query else '',
            'started_at': self.started_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'total_duration_ms': self.total_duration_ms,
            'steps': [s.to_dict() for s in self.steps],
            'agent_sequence': self.agent_sequence,
            'total_tool_calls': self.total_tool_calls,
            'total_llm_calls': self.total_llm_calls,
            'status': self.status,
            'errors': self.errors
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a concise summary of the trace."""
        step_counts = {}
        for step in self.steps:
            step_type = step.step_type.value
            step_counts[step_type] = step_counts.get(step_type, 0) + 1
        
        return {
            'trace_id': self.trace_id,
            'query': self.query[:100] + '...' if len(self.query) > 100 else self.query,
            'duration_ms': self.total_duration_ms,
            'agents_used': self.agent_sequence,
            'step_counts': step_counts,
            'tool_calls': self.total_tool_calls,
            'llm_calls': self.total_llm_calls,
            'status': self.status,
            'has_errors': len(self.errors) > 0
        }


class AgentTracer:
    """
    Singleton tracer for agent orchestration observability.
    
    Usage:
        tracer = get_agent_tracer()
        trace_id = tracer.start_trace("user query")
        tracer.log_thought(trace_id, "Supervisor", "Analyzing query...")
        tracer.log_action(trace_id, "Supervisor", "Routing to Diagnostic")
        tracer.log_tool_call(trace_id, "Diagnostic", "get_patient", {"patient_key": "123"})
        tracer.log_tool_result(trace_id, "Diagnostic", "get_patient", {"dqi": 85})
        tracer.end_trace(trace_id, "Final response here")
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._traces: Dict[str, AgentTrace] = {}
        self._current_trace_id: Optional[str] = None
        self._max_traces = 100  # Keep last 100 traces
        self._initialized = True
        logger.info("AgentTracer initialized")
    
    def start_trace(self, query: str) -> str:
        """Start a new trace for a query."""
        trace_id = f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        self._traces[trace_id] = AgentTrace(
            trace_id=trace_id,
            query=query
        )
        self._current_trace_id = trace_id
        
        # Prune old traces
        if len(self._traces) > self._max_traces:
            oldest = sorted(self._traces.keys())[0]
            del self._traces[oldest]
        
        logger.info(f"Started trace: {trace_id}")
        return trace_id
    
    def end_trace(self, trace_id: str, final_response: Optional[str] = None, status: str = "completed"):
        """End a trace."""
        if trace_id in self._traces:
            trace = self._traces[trace_id]
            trace.completed_at = datetime.now()
            trace.status = status
            trace.final_response = final_response
            logger.info(f"Ended trace: {trace_id} ({trace.total_duration_ms:.0f}ms)")
    
    def _add_step(
        self,
        trace_id: str,
        step_type: TraceStepType,
        agent_name: str,
        content: str,
        metadata: Optional[Dict] = None,
        duration_ms: float = 0.0
    ):
        """Add a step to a trace."""
        if trace_id not in self._traces:
            logger.warning(f"Trace not found: {trace_id}")
            return
        
        step = TraceStep(
            step_id=f"step_{len(self._traces[trace_id].steps) + 1}",
            step_type=step_type,
            agent_name=agent_name,
            content=content,
            metadata=metadata or {},
            duration_ms=duration_ms
        )
        
        self._traces[trace_id].steps.append(step)
        
        # Track agent sequence
        if agent_name not in self._traces[trace_id].agent_sequence:
            self._traces[trace_id].agent_sequence.append(agent_name)
    
    def log_thought(self, trace_id: str, agent_name: str, thought: str):
        """Log an agent thought (reasoning)."""
        self._add_step(trace_id, TraceStepType.THOUGHT, agent_name, thought)
        logger.debug(f"[{agent_name}] THOUGHT: {thought[:100]}...")
    
    def log_action(self, trace_id: str, agent_name: str, action: str, metadata: Optional[Dict] = None):
        """Log an agent action decision."""
        self._add_step(trace_id, TraceStepType.ACTION, agent_name, action, metadata)
        logger.debug(f"[{agent_name}] ACTION: {action}")
    
    def log_observation(self, trace_id: str, agent_name: str, observation: str, metadata: Optional[Dict] = None):
        """Log an observation from action result."""
        self._add_step(trace_id, TraceStepType.OBSERVATION, agent_name, observation, metadata)
        logger.debug(f"[{agent_name}] OBSERVATION: {observation[:100]}...")
    
    def log_tool_call(
        self,
        trace_id: str,
        agent_name: str,
        tool_name: str,
        parameters: Dict[str, Any],
        duration_ms: float = 0.0
    ):
        """Log a tool call."""
        content = f"Calling {tool_name} with {json.dumps(parameters, default=str)[:200]}"
        self._add_step(
            trace_id,
            TraceStepType.TOOL_CALL,
            agent_name,
            content,
            {'tool_name': tool_name, 'parameters': parameters},
            duration_ms
        )
        self._traces[trace_id].total_tool_calls += 1
        logger.debug(f"[{agent_name}] TOOL_CALL: {tool_name}")
    
    def log_tool_result(
        self,
        trace_id: str,
        agent_name: str,
        tool_name: str,
        result: Any,
        success: bool = True,
        duration_ms: float = 0.0
    ):
        """Log a tool result."""
        result_str = json.dumps(result, default=str)[:500] if result else "No result"
        content = f"Result from {tool_name}: {'SUCCESS' if success else 'FAILED'}"
        self._add_step(
            trace_id,
            TraceStepType.TOOL_RESULT,
            agent_name,
            content,
            {'tool_name': tool_name, 'result': result_str, 'success': success},
            duration_ms
        )
        logger.debug(f"[{agent_name}] TOOL_RESULT: {tool_name} - {'SUCCESS' if success else 'FAILED'}")
    
    def log_routing(self, trace_id: str, from_agent: str, to_agent: str, reason: str):
        """Log agent routing decision."""
        content = f"Routing from {from_agent} to {to_agent}: {reason}"
        self._add_step(
            trace_id,
            TraceStepType.ROUTING,
            from_agent,
            content,
            {'from_agent': from_agent, 'to_agent': to_agent}
        )
        logger.debug(f"[{from_agent}] ROUTING → {to_agent}")
    
    def log_llm_call(self, trace_id: str, agent_name: str, prompt_tokens: int = 0, completion_tokens: int = 0):
        """Track LLM API calls."""
        if trace_id in self._traces:
            self._traces[trace_id].total_llm_calls += 1
    
    def log_error(self, trace_id: str, agent_name: str, error: str):
        """Log an error."""
        self._add_step(trace_id, TraceStepType.ERROR, agent_name, error)
        if trace_id in self._traces:
            self._traces[trace_id].errors.append(f"[{agent_name}] {error}")
        logger.error(f"[{agent_name}] ERROR: {error}")
    
    def log_synthesis(self, trace_id: str, agent_name: str, synthesis: str):
        """Log final synthesis."""
        self._add_step(trace_id, TraceStepType.SYNTHESIS, agent_name, synthesis)
        logger.debug(f"[{agent_name}] SYNTHESIS: {synthesis[:100]}...")
    
    def get_trace(self, trace_id: str) -> Optional[AgentTrace]:
        """Get a trace by ID."""
        return self._traces.get(trace_id)
    
    def get_current_trace(self) -> Optional[AgentTrace]:
        """Get the current active trace."""
        if self._current_trace_id:
            return self._traces.get(self._current_trace_id)
        return None
    
    def get_recent_traces(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get summaries of recent traces."""
        sorted_traces = sorted(
            self._traces.values(),
            key=lambda t: t.started_at,
            reverse=True
        )[:limit]
        return [t.get_summary() for t in sorted_traces]
    
    def export_trace(self, trace_id: str, path: Optional[Path] = None) -> str:
        """Export trace to JSON."""
        if trace_id not in self._traces:
            return "{}"
        
        trace_json = json.dumps(self._traces[trace_id].to_dict(), indent=2, default=str)
        
        if path:
            path.write_text(trace_json)
            logger.info(f"Exported trace to {path}")
        
        return trace_json
    
    def get_trace_timeline(self, trace_id: str) -> List[Dict[str, Any]]:
        """Get trace as a timeline for visualization."""
        if trace_id not in self._traces:
            return []
        
        trace = self._traces[trace_id]
        timeline = []
        
        for i, step in enumerate(trace.steps):
            icon = {
                TraceStepType.THOUGHT: "💭",
                TraceStepType.ACTION: "⚡",
                TraceStepType.OBSERVATION: "👁️",
                TraceStepType.TOOL_CALL: "🔧",
                TraceStepType.TOOL_RESULT: "📤",
                TraceStepType.ROUTING: "➡️",
                TraceStepType.ERROR: "❌",
                TraceStepType.SYNTHESIS: "✨"
            }.get(step.step_type, "•")
            
            timeline.append({
                'index': i + 1,
                'icon': icon,
                'type': step.step_type.value.upper(),
                'agent': step.agent_name,
                'content': step.content[:200] + ('...' if len(step.content) > 200 else ''),
                'duration_ms': step.duration_ms,
                'timestamp': step.timestamp.strftime('%H:%M:%S.%f')[:-3]
            })
        
        return timeline
    
    def clear_traces(self):
        """Clear all traces."""
        self._traces.clear()
        self._current_trace_id = None
        logger.info("Cleared all traces")


# Singleton accessor
_tracer_instance: Optional[AgentTracer] = None


def get_agent_tracer() -> AgentTracer:
    """Get the global AgentTracer instance."""
    global _tracer_instance
    if _tracer_instance is None:
        _tracer_instance = AgentTracer()
    return _tracer_instance


# Convenience context manager for tracing
class TracingContext:
    """Context manager for automatic trace start/end."""
    
    def __init__(self, query: str):
        self.query = query
        self.trace_id = None
        self.tracer = get_agent_tracer()
    
    def __enter__(self) -> str:
        self.trace_id = self.tracer.start_trace(self.query)
        return self.trace_id
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.tracer.log_error(self.trace_id, "System", str(exc_val))
            self.tracer.end_trace(self.trace_id, status="failed")
        else:
            self.tracer.end_trace(self.trace_id, status="completed")


# Export public API
__all__ = [
    'AgentTracer',
    'AgentTrace',
    'TraceStep',
    'TraceStepType',
    'get_agent_tracer',
    'TracingContext'
]
