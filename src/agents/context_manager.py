"""
SANCHALAK AI - LLM Context Manager
=======================================
Build rich context for LLM interactions including:
- User role and permissions
- User's recent queries
- Current focus (study/site/patient)
- Relevant recent alerts
- Previously discussed topics

Per riyaz.md Section 26: LLM Context Enhancement
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class UserContext:
    """Represents a user's current context state."""
    user_id: str
    role: str
    permissions: List[str] = field(default_factory=list)
    current_study: Optional[str] = None
    current_site: Optional[str] = None
    current_patient: Optional[str] = None
    recent_queries: List[Dict[str, Any]] = field(default_factory=list)
    active_alerts: List[Dict[str, Any]] = field(default_factory=list)
    discussed_topics: List[str] = field(default_factory=list)
    session_start: str = field(default_factory=lambda: datetime.now().isoformat())
    last_activity: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ContextWindow:
    """Sliding window of context for LLM."""
    max_tokens: int = 4000
    max_queries: int = 10
    max_alerts: int = 5
    max_topics: int = 10
    include_role_context: bool = True
    include_data_context: bool = True
    include_alert_context: bool = True


class ContextManager:
    """
    Manages rich context for LLM interactions.
    
    Builds comprehensive context including:
    1. User role and permissions
    2. User's recent queries
    3. Current focus (study/site/patient)
    4. Relevant recent alerts
    5. Previously discussed topics
    """
    
    # Role-based context templates
    ROLE_CONTEXTS = {
        "study_lead": {
            "focus_areas": ["study progress", "timeline", "enrollment", "DQI trends"],
            "key_metrics": ["overall_dqi", "dblock_ready_rate", "enrollment_rate"],
            "permissions": ["view_all_sites", "approve_actions", "generate_reports", "manage_study"]
        },
        "cra": {
            "focus_areas": ["site monitoring", "SDV completion", "query resolution", "site issues"],
            "key_metrics": ["sdv_rate", "query_age", "site_performance"],
            "permissions": ["view_assigned_sites", "execute_monitoring_actions", "raise_queries"]
        },
        "data_manager": {
            "focus_areas": ["data quality", "query management", "coding", "reconciliation"],
            "key_metrics": ["query_volume", "coding_backlog", "reconciliation_status"],
            "permissions": ["manage_queries", "approve_coding", "run_edit_checks"]
        },
        "medical_coder": {
            "focus_areas": ["MedDRA coding", "WHODrug coding", "coding accuracy"],
            "key_metrics": ["uncoded_terms", "coding_accuracy", "autocoding_rate"],
            "permissions": ["code_terms", "approve_autocoding", "view_coding_queue"]
        },
        "safety_officer": {
            "focus_areas": ["SAE management", "safety signals", "expedited reporting"],
            "key_metrics": ["sae_pending", "reporting_compliance", "signal_detection"],
            "permissions": ["manage_sae", "medical_review", "regulatory_reporting"]
        },
        "site_coordinator": {
            "focus_areas": ["data entry", "query response", "visit scheduling"],
            "key_metrics": ["pending_queries", "missing_data", "upcoming_visits"],
            "permissions": ["enter_data", "respond_queries", "schedule_visits"]
        },
        "executive": {
            "focus_areas": ["portfolio overview", "risk assessment", "resource allocation"],
            "key_metrics": ["portfolio_dqi", "at_risk_studies", "resource_utilization"],
            "permissions": ["view_portfolio", "approve_resources", "strategic_decisions"]
        }
    }
    
    def __init__(self, db_service=None):
        """Initialize context manager with optional database service."""
        self.db = db_service
        self._user_contexts: Dict[str, UserContext] = {}
        self._context_cache: Dict[str, Dict] = {}
        self._cache_ttl = 300  # 5 minutes
    
    def get_or_create_user_context(self, user_id: str, role: str = "data_manager") -> UserContext:
        """Get existing or create new user context."""
        if user_id not in self._user_contexts:
            role_info = self.ROLE_CONTEXTS.get(role, self.ROLE_CONTEXTS["data_manager"])
            self._user_contexts[user_id] = UserContext(
                user_id=user_id,
                role=role,
                permissions=role_info.get("permissions", [])
            )
        return self._user_contexts[user_id]
    
    def update_focus(
        self, 
        user_id: str, 
        study_id: Optional[str] = None,
        site_id: Optional[str] = None,
        patient_id: Optional[str] = None
    ) -> None:
        """Update user's current focus (study/site/patient)."""
        ctx = self.get_or_create_user_context(user_id)
        if study_id:
            ctx.current_study = study_id
        if site_id:
            ctx.current_site = site_id
        if patient_id:
            ctx.current_patient = patient_id
        ctx.last_activity = datetime.now().isoformat()
    
    def add_query_to_history(self, user_id: str, query: str, response_summary: str) -> None:
        """Add a query to user's history for context."""
        ctx = self.get_or_create_user_context(user_id)
        ctx.recent_queries.append({
            "query": query,
            "response_summary": response_summary[:200],  # Truncate
            "timestamp": datetime.now().isoformat()
        })
        # Keep only recent queries
        ctx.recent_queries = ctx.recent_queries[-10:]
        
        # Extract topics
        topics = self._extract_topics(query)
        for topic in topics:
            if topic not in ctx.discussed_topics:
                ctx.discussed_topics.append(topic)
        ctx.discussed_topics = ctx.discussed_topics[-10:]
    
    def add_alert(self, user_id: str, alert: Dict[str, Any]) -> None:
        """Add an active alert to user's context."""
        ctx = self.get_or_create_user_context(user_id)
        ctx.active_alerts.append({
            **alert,
            "added_at": datetime.now().isoformat()
        })
        # Keep only recent alerts
        ctx.active_alerts = ctx.active_alerts[-5:]
    
    def build_context(
        self, 
        user_id: str, 
        query: str,
        window: Optional[ContextWindow] = None
    ) -> str:
        """
        Build rich context string for LLM.
        
        Args:
            user_id: User identifier
            query: Current query being processed
            window: Context window configuration
            
        Returns:
            Formatted context string for LLM
        """
        window = window or ContextWindow()
        ctx = self.get_or_create_user_context(user_id)
        
        sections = []
        
        # 1. Role and permissions context
        if window.include_role_context:
            role_ctx = self._build_role_context(ctx)
            sections.append(role_ctx)
        
        # 2. Current focus context
        focus_ctx = self._build_focus_context(ctx)
        if focus_ctx:
            sections.append(focus_ctx)
        
        # 3. Data context (current state)
        if window.include_data_context:
            data_ctx = self._build_data_context(ctx)
            if data_ctx:
                sections.append(data_ctx)
        
        # 4. Alert context
        if window.include_alert_context and ctx.active_alerts:
            alert_ctx = self._build_alert_context(ctx, window.max_alerts)
            sections.append(alert_ctx)
        
        # 5. Conversation history context
        if ctx.recent_queries:
            history_ctx = self._build_history_context(ctx, window.max_queries)
            sections.append(history_ctx)
        
        # 6. Topic continuity context
        if ctx.discussed_topics:
            topic_ctx = self._build_topic_context(ctx, window.max_topics)
            sections.append(topic_ctx)
        
        # Combine all sections
        full_context = "\n\n".join(sections)
        
        # Truncate if too long (rough token estimate)
        if len(full_context) > window.max_tokens * 4:  # ~4 chars per token
            full_context = full_context[:window.max_tokens * 4] + "\n[Context truncated]"
        
        return full_context
    
    def _build_role_context(self, ctx: UserContext) -> str:
        """Build role-based context section."""
        role_info = self.ROLE_CONTEXTS.get(ctx.role, {})
        focus_areas = role_info.get("focus_areas", [])
        
        return f"""## User Context
- **Role**: {ctx.role.replace('_', ' ').title()}
- **Focus Areas**: {', '.join(focus_areas)}
- **Permissions**: {', '.join(ctx.permissions[:5])}"""
    
    def _build_focus_context(self, ctx: UserContext) -> Optional[str]:
        """Build current focus context section."""
        focus_parts = []
        
        if ctx.current_study:
            focus_parts.append(f"Study: {ctx.current_study}")
        if ctx.current_site:
            focus_parts.append(f"Site: {ctx.current_site}")
        if ctx.current_patient:
            focus_parts.append(f"Patient: {ctx.current_patient}")
        
        if focus_parts:
            return f"""## Current Focus
{chr(10).join('- ' + p for p in focus_parts)}"""
        return None
    
    def _build_data_context(self, ctx: UserContext) -> Optional[str]:
        """Build data context with current metrics."""
        if not ctx.current_study:
            return None
        
        # Try to get live metrics
        metrics = self._get_study_metrics(ctx.current_study)
        if not metrics:
            return None
        
        return f"""## Current Study Metrics ({ctx.current_study})
- DQI Score: {metrics.get('dqi_score', 'N/A')}
- DB Lock Ready: {metrics.get('dblock_ready_pct', 'N/A')}%
- Open Queries: {metrics.get('open_queries', 'N/A')}
- Signature Gaps: {metrics.get('signature_gaps', 'N/A')}
- Last Updated: {metrics.get('last_updated', 'N/A')}"""
    
    def _build_alert_context(self, ctx: UserContext, max_alerts: int) -> str:
        """Build active alerts context section."""
        alerts = ctx.active_alerts[:max_alerts]
        alert_lines = []
        
        for alert in alerts:
            severity = alert.get('severity', 'INFO')
            message = alert.get('message', 'Unknown alert')
            alert_lines.append(f"- [{severity}] {message}")
        
        return f"""## Active Alerts
{chr(10).join(alert_lines)}"""
    
    def _build_history_context(self, ctx: UserContext, max_queries: int) -> str:
        """Build conversation history context section."""
        recent = ctx.recent_queries[-max_queries:]
        history_lines = []
        
        for q in recent:
            query_text = q['query'][:100] + "..." if len(q['query']) > 100 else q['query']
            history_lines.append(f"- Q: {query_text}")
        
        return f"""## Recent Conversation Topics
{chr(10).join(history_lines)}"""
    
    def _build_topic_context(self, ctx: UserContext, max_topics: int) -> str:
        """Build discussed topics context section."""
        topics = ctx.discussed_topics[-max_topics:]
        return f"""## Previously Discussed Topics
{', '.join(topics)}"""
    
    def _extract_topics(self, query: str) -> List[str]:
        """Extract key topics from a query."""
        # Simple keyword extraction
        topic_keywords = {
            "dqi": "DQI scores",
            "query": "queries",
            "signature": "signatures",
            "sdv": "SDV completion",
            "site": "site performance",
            "patient": "patient data",
            "timeline": "timelines",
            "enrollment": "enrollment",
            "sae": "safety events",
            "coding": "medical coding",
            "report": "reports",
            "risk": "risk assessment"
        }
        
        topics = []
        query_lower = query.lower()
        for keyword, topic in topic_keywords.items():
            if keyword in query_lower and topic not in topics:
                topics.append(topic)
        
        return topics[:3]  # Max 3 topics per query
    
    def _get_study_metrics(self, study_id: str) -> Optional[Dict[str, Any]]:
        """Get current metrics for a study."""
        # Check cache first
        cache_key = f"study_metrics_{study_id}"
        if cache_key in self._context_cache:
            cached = self._context_cache[cache_key]
            if datetime.fromisoformat(cached['cached_at']) > datetime.now() - timedelta(seconds=self._cache_ttl):
                return cached['data']
        
        # Try to get from database
        try:
            if self.db:
                metrics = self.db.get_study_summary(study_id)
                self._context_cache[cache_key] = {
                    'data': metrics,
                    'cached_at': datetime.now().isoformat()
                }
                return metrics
        except Exception as e:
            logger.warning(f"Failed to get study metrics: {e}")
        
        return None
    
    def clear_user_context(self, user_id: str) -> bool:
        """Clear context for a user (e.g., on logout)."""
        if user_id in self._user_contexts:
            del self._user_contexts[user_id]
            return True
        return False
    
    def get_context_summary(self, user_id: str) -> Dict[str, Any]:
        """Get summary of user's current context state."""
        if user_id not in self._user_contexts:
            return {"error": "No context found for user"}
        
        ctx = self._user_contexts[user_id]
        return {
            "user_id": ctx.user_id,
            "role": ctx.role,
            "current_study": ctx.current_study,
            "current_site": ctx.current_site,
            "current_patient": ctx.current_patient,
            "recent_queries_count": len(ctx.recent_queries),
            "active_alerts_count": len(ctx.active_alerts),
            "discussed_topics": ctx.discussed_topics,
            "session_start": ctx.session_start,
            "last_activity": ctx.last_activity
        }


# Singleton instance
_context_manager: Optional[ContextManager] = None


def get_context_manager() -> ContextManager:
    """Get singleton context manager instance."""
    global _context_manager
    if _context_manager is None:
        _context_manager = ContextManager()
    return _context_manager
