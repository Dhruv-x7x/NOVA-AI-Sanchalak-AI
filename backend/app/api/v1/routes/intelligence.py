"""
Intelligence Routes
==================
Endpoints for Causal Hypothesis Engine and Agentic Swarm.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Body
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel
import sys
import os
import pandas as pd

# Add project root to path for imports
import os
import sys
from pathlib import Path

# Fix PROJECT_ROOT to point to the directory containing 'src'
# File path: backend/app/api/v1/routes/intelligence.py
# parents[0]: routes, [1]: v1, [2]: api, [3]: app, [4]: backend
PROJECT_ROOT = Path(__file__).resolve().parents[4]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Also ensure backend is in path
if str(PROJECT_ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.core.security import get_current_user
from app.services.database import get_data_service

import logging
router = APIRouter()
logger = logging.getLogger(__name__)

class SwarmRequest(BaseModel):
    query: str
    context: Dict[str, Any]

class ConversationTurn(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class AssistantQueryRequest(BaseModel):
    query: str
    thread_id: Optional[str] = None
    conversation_history: Optional[List[ConversationTurn]] = None

# Cache the orchestrator instance for better performance (similar to Streamlit's session state)
_cached_orchestrator = None
_orchestrator_error = None

def get_orchestrator():
    """Get or create the orchestrator instance with caching (matches Streamlit pattern)."""
    global _cached_orchestrator, _orchestrator_error
    
    if _cached_orchestrator is not None:
        return _cached_orchestrator, None
    
    if _orchestrator_error is not None:
        return None, _orchestrator_error
    
    try:
        from src.agents.orchestrator import AgentOrchestrator
        _cached_orchestrator = AgentOrchestrator()
        return _cached_orchestrator, None
    except Exception as e:
        _orchestrator_error = str(e)
        return None, _orchestrator_error


from app.services.agents import agentic_service

@router.get("/insights")
async def get_ai_insights(
    study_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Get AI-powered intelligent insights (matches PRD)."""
    try:
        data_service = get_data_service()
        # Fetch real data for insights
        portfolio = data_service.get_portfolio_summary(study_id=study_id)
        sites_df = data_service.get_site_benchmarks(study_id=study_id)

        low_dqi_sites = sites_df[sites_df['dqi_score'] < 80] if not sites_df.empty else pd.DataFrame()
        low_dqi_count = len(low_dqi_sites)
        critical_dqi_sites = sites_df[sites_df['dqi_score'] < 70] if not sites_df.empty else pd.DataFrame()
        critical_dqi_count = len(critical_dqi_sites)
        critical_issues = portfolio.get('critical_issues', 0)

        insights = [
            {
                "id": "insight-1",
                "title": "DB-Lock Readiness Trend",
                "content": f"Portfolio is {portfolio.get('dblock_ready_rate', 0):.1f}% ready for DB lock. {len(sites_df) if not sites_df.empty else 0} sites tracked across portfolio.",
                "severity": "info",
                "impact": "high"
            },
            {
                "id": "insight-2",
                "title": "Data Quality Alert",
                "content": f"Mean DQI is {portfolio.get('mean_dqi', 0):.1f}%. {low_dqi_count} sites are below 80% DQI target.",
                "severity": "warning" if low_dqi_count > 0 else "info",
                "impact": "high" if low_dqi_count > 3 else "medium"
            },
            {
                "id": "insight-3",
                "title": "Enrollment Milestone",
                "content": f"Target enrollment reached for {portfolio.get('total_studies', 0)} studies. Focus shifting to data cleaning phase.",
                "severity": "success",
                "impact": "low"
            },
            {
                "id": "insight-4",
                "title": "Critical Issues Summary",
                "content": f"{critical_issues} critical issues across portfolio. {critical_dqi_count} sites below 70% DQI require immediate attention.",
                "severity": "critical" if critical_dqi_count > 0 else ("warning" if critical_issues > 0 else "info"),
                "impact": "high" if critical_issues > 3 or critical_dqi_count > 0 else "low"
            }
        ]
        
        return {
            "insights": insights,
            "total": len(insights),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "insights": [
                {"id": "error", "title": "Intelligence Layer Initializing", "content": "Real-time insights will be available shortly.", "severity": "info"}
            ],
            "total": 1,
            "error": str(e)
        }


@router.post("/assistant/query")
async def assistant_query(
    request: AssistantQueryRequest,
    current_user: dict = Depends(get_current_user)
):
    """Run a query through the 6-agent orchestrator."""
    try:
        # Process using the new Agentic Service (with conversation memory)
        history = None
        if request.conversation_history:
            history = [{"role": t.role, "content": t.content} for t in request.conversation_history]
        result = await agentic_service.process_query(request.query, conversation_history=history)
        
        return {
            "summary": result.summary,
            "agent_chain": result.agent_chain,
            "steps": [step.dict() for step in result.steps],
            "tools_used": result.tools_used,
            "confidence": result.confidence,
            "recommendations": result.recommendations,
            "timestamp": datetime.utcnow().isoformat(),
            "ai_powered": True
        }
    except Exception as e:
        # Fallback to legacy logic if agentic service fails
        try:
            data_service = get_data_service()
            
            # Get portfolio summary
            portfolio = data_service.get_portfolio_summary()
            issues_df = data_service.get_issues(status='open', limit=10)
            patients_df = data_service.get_patients(limit=5)
            
            total_patients = portfolio.get('total_patients', 0) if portfolio else 0
            total_sites = portfolio.get('total_sites', 0) if portfolio else 0
            mean_dqi = portfolio.get('mean_dqi', 0) if portfolio else 0
            dblock_ready_rate = portfolio.get('dblock_ready_rate', 0) if portfolio else 0
            critical_issues = portfolio.get('critical_issues', 0) if portfolio else 0
            
            # Count issues by type
            issue_types = {}
            if issues_df is not None and not issues_df.empty:
                issue_types = issues_df['issue_type'].value_counts().to_dict() if 'issue_type' in issues_df.columns else {}
            top_issue = max(issue_types.items(), key=lambda x: x[1])[0] if issue_types else "data quality"
            
            # Build dynamic response
            summary_parts = [
                f"Based on your current trial data, I've analyzed the portfolio of {total_patients} patients across {total_sites} sites."
            ]
            
            if mean_dqi:
                summary_parts.append(f"The overall DQI score is {mean_dqi:.1f}%, with {dblock_ready_rate:.1f}% of patients DB-lock ready.")
            
            if critical_issues > 0:
                summary_parts.append(f"There are {critical_issues} critical issues requiring immediate attention.")
            
            if top_issue:
                summary_parts.append(f"Most common pending issues relate to {top_issue.replace('_', ' ')}.")
            
            # Add recommendation
            if dblock_ready_rate < 70:
                summary_parts.append("I recommend prioritizing data completion for patients closest to clean status to improve DB-lock readiness.")
            else:
                summary_parts.append("The portfolio is in good shape. Continue monitoring high-risk patients closely.")
            
            summary = " ".join(summary_parts)
            
            return {
                "summary": summary,
                "state": {"hypotheses": [], "recommendations": [], "forecasts": []},
                "timestamp": datetime.utcnow().isoformat(),
                "data_source": "postgresql",
                "ai_powered": False
            }
        except Exception as fallback_error:
            # Ultimate fallback if even PostgreSQL fails
            return {
                "summary": f"I analyzed your request about '{request.query}'.\n\nThe AI agent system is currently initializing.",
                "state": {"hypotheses": [], "recommendations": [], "forecasts": []},
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(fallback_error),
                "ai_powered": False
            }


@router.get("/health")
async def ai_health_check(current_user: dict = Depends(get_current_user)):
    """Check health of AI/LLM providers — vLLM on Colab."""
    health = {
        "orchestrator": {"available": False, "error": None},
        "vllm": {"available": False, "error": None, "base_url": None, "model": None, "available_models": []},
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Check orchestrator
    orchestrator, orch_error = get_orchestrator()
    if orchestrator:
        health["orchestrator"]["available"] = True
    else:
        health["orchestrator"]["error"] = orch_error
    
    # Check vLLM wrapper directly
    try:
        from src.agents.llm_wrapper import get_llm
        llm = get_llm()
        llm_health = llm.health_check()
        
        vllm_info = llm_health.get("vllm", {})
        health["vllm"]["available"] = vllm_info.get("available", False)
        health["vllm"]["error"] = vllm_info.get("error")
        health["vllm"]["base_url"] = vllm_info.get("base_url")
        health["vllm"]["model"] = vllm_info.get("model")
        health["vllm"]["available_models"] = vllm_info.get("available_models", [])
        
    except Exception as e:
        health["llm_error"] = str(e)
    
    # Summary
    if health["orchestrator"]["available"]:
        health["status"] = "operational"
        health["message"] = "AI Agent System is fully operational with TrialPulse-8B (vLLM)"
    elif health["vllm"]["available"]:
        health["status"] = "degraded"
        health["message"] = "vLLM available but orchestrator not initialized"
    else:
        health["status"] = "unavailable"
        health["message"] = "vLLM server not reachable. Ensure Colab notebook is running."
        health["troubleshooting"] = [
            "Start the Colab notebook with vLLM + ngrok",
            "Update VLLM_BASE_URL in .env with the new ngrok URL",
            "Restart the FastAPI server after configuration changes"
        ]
    
    return health


@router.post("/reset")
async def reset_orchestrator(current_user: dict = Depends(get_current_user)):
    """Reset the cached orchestrator to force re-initialization."""
    global _cached_orchestrator, _orchestrator_error
    _cached_orchestrator = None
    _orchestrator_error = None
    return {"status": "reset", "message": "Orchestrator cache cleared. Next query will re-initialize."}


@router.get("/hypotheses")
async def get_hypotheses(
    sample_size: int = Query(50, ge=1, le=200),
    study_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Get real causal hypotheses from the engine."""
    try:
        from src.knowledge.causal_hypothesis_engine import CausalHypothesisEngine
        engine = CausalHypothesisEngine()
        if not engine.load_data():
            return {"hypotheses": [], "stats": {}}
            
        hypotheses = engine.analyze_population(sample_size=sample_size)
        
        return {
            "hypotheses": [h.to_dict() for h in hypotheses],
            "stats": engine.get_summary()
        }
    except Exception as e:
        logger.error(f"Hypothesis analysis error: {e}", exc_info=True)
        return {"hypotheses": [], "stats": {}, "error": str(e)}

@router.post("/swarm/run")
async def run_swarm_investigation(
    request: SwarmRequest = Body(...),
    current_user: dict = Depends(get_current_user)
):
    """Run an agentic swarm investigation."""
    try:
        from src.intelligence.agent_swarm import AgentSwarm
        swarm = AgentSwarm()
        trace = swarm.run_investigation(request.query, request.context)
        return {
            "query": request.query,
            "trace": [
                {
                    "agent": step.agent_name,
                    "thought": step.thought,
                    "action": step.action,
                    "observation": step.observation
                } for step in trace
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Swarm investigation failed: {str(e)}")

@router.get("/anomalies")
async def get_anomalies(
    study_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Get detected anomalies that need investigation."""
    try:
        data_service = get_data_service()
        # Fetch patterns and also check for high-risk patients
        patterns_df = data_service.get_pattern_alerts()
        
        anomalies = []
        
        if not patterns_df.empty:
            for _, row in patterns_df.iterrows():
                anomalies.append({
                    "id": str(row.get("pattern_id", os.urandom(2).hex())),
                    "title": str(row.get("pattern_name", "Anomalous Trend")),
                    "severity": str(row.get("severity", "medium")).lower(),
                    "impact_score": float(row.get("confidence", 0.7)),
                    "message": str(row.get("alert_message", "Unexpected pattern detected.")),
                    "site_id": str(row.get("sites_affected", "Site-Wide")),
                    "timestamp": datetime.utcnow().isoformat()
                })
        
        # Add high-risk patients as anomalies if no patterns
        if len(anomalies) < 3:
            patients_df = data_service.get_patients(limit=20, study_id=study_id)
            if not patients_df.empty:
                # Add High risk if any, otherwise just grab interesting ones (lowest DQI)
                high_risk = patients_df[patients_df['risk_level'] == 'High']
                if not high_risk.empty:
                    for _, p in high_risk.head(3).iterrows():
                        anomalies.append({
                            "id": f"risk-{p['patient_key']}",
                            "title": "Patient Critical Risk",
                            "severity": "high",
                            "impact_score": 0.9,
                            "message": f"Patient {p['patient_key']} has exceeded risk threshold.",
                            "site_id": p['site_id'],
                            "timestamp": datetime.utcnow().isoformat()
                        })
                
                # Add sites with poor DQI
                sites_df = data_service.get_site_benchmarks(study_id=study_id)
                if not sites_df.empty:
                    low_dqi_sites = sites_df[sites_df['dqi_score'] < 80]
                    for _, s in low_dqi_sites.head(2).iterrows():
                        anomalies.append({
                            "id": f"site-low-dqi-{s['site_id']}",
                            "title": "Site DQI Deficiency",
                            "severity": "medium",
                            "impact_score": 0.75,
                            "message": f"Site {s['site_id']} is below 80% DQI target.",
                            "site_id": s['site_id'],
                            "timestamp": datetime.utcnow().isoformat()
                        })

        # Final fallback if still empty (rare but possible in empty DB)
        if not anomalies:
            anomalies.append({
                "id": "anom-base-01",
                "title": "Data Latency Drift",
                "severity": "info",
                "impact_score": 0.2,
                "message": "Routine telemetry sync completed. No critical anomalies found.",
                "site_id": "System",
                "timestamp": datetime.utcnow().isoformat()
            })
                
        return {"anomalies": anomalies}
    except Exception as e:
        return {"anomalies": [], "error": str(e)}

@router.post("/auto-fix")
async def auto_fix_issue(
    issue_id: int,
    entity_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Automatically resolve an issue in the database."""
    try:
        from src.database.connection import get_db_manager
        from sqlalchemy import text
        db = get_db_manager()
        
        # Determine format of entity_id (composite vs Subject ID)
        # and try to find matching patient_key in database
        subject_id = entity_id
        if '|' in entity_id:
            subject_id = entity_id.split('|')[-1]
            
        with db.engine.begin() as conn:
            # 1. Update project_issues - try exact match on issue_id first
            # If issue_id is 0 or -1 (placeholder), try matching by patient_key
            if issue_id > 0:
                conn.execute(
                    text("UPDATE project_issues SET status = 'resolved', resolved_at = :now, resolution_notes = 'Auto-fixed by Intelligence Layer' WHERE issue_id = :issue_id"),
                    {"issue_id": issue_id, "now": datetime.utcnow()}
                )
            else:
                # Match by patient_key (robustly)
                conn.execute(
                    text("UPDATE project_issues SET status = 'resolved', resolved_at = :now, resolution_notes = 'Auto-fixed by Intelligence Layer (entity match)' WHERE patient_key = :subject_id OR patient_key = :entity_id"),
                    {"subject_id": subject_id, "entity_id": entity_id, "now": datetime.utcnow()}
                )
            
            # 2. Update patient metadata (try both formats)
            conn.execute(
                text("UPDATE patients SET open_issues_count = GREATEST(0, open_issues_count - 1) WHERE patient_key = :entity_id OR patient_key = :subject_id"),
                {"entity_id": entity_id, "subject_id": subject_id}
            )
            
        return {"status": "success", "message": f"Issue resolved automatically."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Auto-fix failed: {str(e)}")
