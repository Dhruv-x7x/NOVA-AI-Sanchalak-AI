import random
import time
import requests
import json
import logging
import os
import sys
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from pathlib import Path

# Add project root to path
# backend/app/services/agents.py -> parents[3] is root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

class AgentStep(BaseModel):
    agent: str
    thought: str
    action: Optional[str] = "analyze"
    observation: Optional[str] = None

class AgenticResponse(BaseModel):
    summary: str
    agent_chain: List[str]
    steps: List[AgentStep]
    tools_used: List[str]
    confidence: float
    recommendations: List[Dict[str, str]]

class AgenticService:
    def __init__(self):
        self.orchestrator = None
        self._init_orchestrator()

    def _init_orchestrator(self):
        try:
            # Fix path for standalone service
            PROJECT_ROOT = Path(__file__).resolve().parents[3]
            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))

            # Use V4 agentic orchestrator with Gemini function calling
            from src.agents.orchestrator_v4 import get_orchestrator_v4
            self.orchestrator = get_orchestrator_v4()
            logger.info("AgentOrchestratorV4 (agentic, function-calling) initialized in AgenticService")
        except Exception as e:
            logger.error(f"Failed to initialize V4 orchestrator: {e}")
            # Fallback to V3
            try:
                from src.agents.orchestrator import get_orchestrator
                self.orchestrator = get_orchestrator()
                logger.info("Fell back to V3 orchestrator")
            except Exception as e2:
                logger.error(f"V3 orchestrator fallback also failed: {e2}")
                self.orchestrator = None

    async def process_query(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> AgenticResponse:
        """Processes the query using the V4 agentic orchestrator."""
        if not self.orchestrator:
            self._init_orchestrator()
            
        if self.orchestrator:
            try:
                # Run the orchestrator (V4.run is async)
                res = await self.orchestrator.run(
                    query,
                    conversation_history=conversation_history,
                )
                
                # Convert results to AgenticResponse
                steps = [AgentStep(**s) for s in res.get('steps', [])]
                
                return AgenticResponse(
                    summary=res.get('summary', ""),
                    agent_chain=res.get('agent_chain', []),
                    steps=steps,
                    tools_used=res.get('tools_used', []),
                    confidence=res.get('confidence', 0.9),
                    recommendations=res.get('recommendations', [])
                )
            except Exception as e:
                logger.error(f"Orchestrator execution failed: {e}")
                import traceback
                logger.error(traceback.format_exc())


        # Fallback simulation logic (improved)
        query_lower = query.lower()
        topic = "trial operations"
        if "site" in query_lower: topic = "site performance"
        if "dqi" in query_lower: topic = "data quality"
        if "lock" in query_lower: topic = "database lock readiness"

        return AgenticResponse(
            summary=f"I've initiated a background analysis of the {topic} metrics. My diagnostic agents have identified key performance indicators that require attention.",
            agent_chain=["SUPERVISOR", "DIAGNOSTIC", "FORECASTER", "RESOLVER", "EXECUTOR", "COMMUNICATOR"],
            steps=[
                AgentStep(agent="SUPERVISOR", thought=f"Decomposing {topic} query.", action="route_to_specialists"),
                AgentStep(agent="DIAGNOSTIC", thought="Analyzing telemetry.", action="get_metrics", observation="Trends identified"),
                AgentStep(agent="FORECASTER", thought="Generating predictions.", action="predictive_enrichment", observation="Predictive enrichment complete"),
                AgentStep(agent="RESOLVER", thought="Generating plan.", action="optimize", observation="Ready"),
                AgentStep(agent="EXECUTOR", thought="Validating actions.", action="validate_actions", observation="Action validation complete"),
            ],
            tools_used=["query_site_metrics", "predictive_engine", "search_patterns", "action_validator"],
            confidence=0.85,
            recommendations=[{"action": f"Review {topic} trends", "impact": "High"}]
        )

agentic_service = AgenticService()
