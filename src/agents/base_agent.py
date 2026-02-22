"""
Base Agent Class for a6on-i
Phase 5.2: LangGraph Agent Framework

Provides the foundation for all specialized agents.
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from abc import ABC, abstractmethod
import logging
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.state import AgentState, AgentType, TaskStatus, Message
from src.agents.tools import ToolRegistry, get_tool_registry, ToolResult
from src.agents.llm_wrapper import LLMWrapper, get_llm, LLMResponse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Base class for all agents in the a6on-i system.
    
    Each agent:
    1. Has a specific role and system prompt
    2. Can use tools from the registry
    3. Processes state and returns updated state
    4. Logs all actions for audit trail
    """
    
    def __init__(
        self,
        agent_type: AgentType,
        llm: Optional[LLMWrapper] = None,
        tools: Optional[ToolRegistry] = None
    ):
        """
        Initialize the base agent.
        
        Args:
            agent_type: The type of agent
            llm: LLM wrapper instance (uses default if None)
            tools: Tool registry instance (uses default if None)
        """
        self.agent_type = agent_type
        self.llm = llm or get_llm()
        self.tools = tools or get_tool_registry()
        self.name = agent_type.value.capitalize()
        
        logger.info(f"Initialized {self.name} Agent")
    
    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Return the system prompt for this agent"""
        pass
    
    @property
    def allowed_tools(self) -> List[str]:
        """Return list of tool names this agent can use"""
        return []  # Override in subclasses
    
    def _build_context(self, state: AgentState) -> str:
        """Build context string from state for LLM"""
        context_parts = [
            f"Task ID: {state.task_id}",
            f"Priority: {state.task_priority.value}",
            f"Status: {state.task_status.value}",
        ]
        
        # Add relevant context from state
        if state.context:
            context_parts.append(f"Context: {state.context}")
        
        if state.hypotheses:
            context_parts.append(f"Hypotheses: {len(state.hypotheses)} generated")
        
        if state.recommendations:
            context_parts.append(f"Recommendations: {len(state.recommendations)} available")
        
        # Add conversation history (last 5 messages)
        if state.messages:
            recent = state.messages[-5:]
            context_parts.append("Recent conversation:")
            for msg in recent:
                context_parts.append(f"  [{msg.role}]: {msg.content[:100]}...")
        
        return "\n".join(context_parts)
    
    def _get_available_tools_prompt(self) -> str:
        """Get prompt describing available tools"""
        if not self.allowed_tools:
            return ""
        
        tools_desc = ["Available tools:"]
        for tool_name in self.allowed_tools:
            tool = self.tools.get(tool_name)
            if tool:
                tools_desc.append(f"- {tool.name}: {tool.description}")
        
        return "\n".join(tools_desc)
    
    def use_tool(self, tool_name: str, **kwargs) -> ToolResult:
        """
        Use a tool from the registry.
        
        Args:
            tool_name: Name of the tool to use
            **kwargs: Parameters for the tool
            
        Returns:
            ToolResult with success status and data
        """
        if tool_name not in self.allowed_tools:
            return ToolResult(
                success=False,
                error=f"Tool '{tool_name}' not allowed for {self.name} agent"
            )
        
        result = self.tools.execute(tool_name, **kwargs)
        logger.debug(f"{self.name} used tool '{tool_name}': success={result.success}")
        return result
    
    def generate(
        self,
        prompt: str,
        state: AgentState,
        include_tools: bool = True
    ) -> LLMResponse:
        """
        Generate a response using the LLM.
        
        Args:
            prompt: The prompt to send
            state: Current agent state for context
            include_tools: Whether to include tool descriptions
            
        Returns:
            LLMResponse with content and metadata
        """
        # Build full prompt with context
        context = self._build_context(state)
        tools_prompt = self._get_available_tools_prompt() if include_tools else ""
        
        full_prompt = f"""Context:
{context}

{tools_prompt}

User Query: {state.user_query}

Task: {prompt}"""
        
        # Generate response
        response = self.llm.generate(
            prompt=full_prompt,
            system_prompt=self.system_prompt
        )
        
        # Update state tokens
        state.total_tokens_used += response.total_tokens
        state.total_latency_ms += response.latency_ms
        
        return response
    
    def add_to_audit(self, state: AgentState, action: str, details: Dict[str, Any] = None):
        """Add an entry to the audit trail"""
        state.add_audit_entry(
            action=action,
            agent=self.agent_type.value,
            details=details
        )
    
    @abstractmethod
    def process(self, state: AgentState) -> AgentState:
        """
        Process the state and return updated state.
        
        This is the main method that each agent implements.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated agent state
        """
        pass
    
    def __call__(self, state: AgentState) -> Dict[str, Any]:
        """Make the agent callable for LangGraph - returns dict for state updates"""
        logger.info(f"🤖 {self.name} Agent processing...")
        
        # Handle input - may be dict from LangGraph
        if isinstance(state, dict):
            state = AgentState(**state)
        
        # Update current agent in state
        state.current_agent = self.agent_type
        state.agent_sequence.append(self.agent_type.value)
        
        # Add audit entry
        self.add_to_audit(state, "agent_started")
        
        try:
            # Process the state
            result_state = self.process(state)
            
            # Add completion audit
            self.add_to_audit(result_state, "agent_completed")
            
            logger.info(f"✅ {self.name} Agent completed")
            
            # Return dict for LangGraph StateGraph compatibility
            return result_state.model_dump()
            
        except Exception as e:
            logger.error(f"❌ {self.name} Agent error: {e}")
            state.errors.append(f"{self.name}: {str(e)}")
            self.add_to_audit(state, "agent_error", {"error": str(e)})
            return state.model_dump()
    
    def __repr__(self):
        return f"{self.name}Agent(tools={len(self.allowed_tools)})"
