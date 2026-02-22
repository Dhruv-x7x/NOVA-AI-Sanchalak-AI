"""
a6on-i - Phase 10.4: Trust Metrics System v1.0
# NOTE: This module previously used SQLite but has been migrated to PostgreSQL-only.
# If you need to use this module, update it to use:
#   from src.database.pg_data_service import get_pg_data_service
#   service = get_pg_data_service()


Tracks user trust in AI system through:
- Override rate monitoring
- Suggestion acceptance tracking
- User satisfaction scoring
- Adoption metrics

Author: a6on-i Team
Date: 2026-01-02
"""

# SQLite removed - using PostgreSQL
import sqlite3
import hashlib
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
from pathlib import Path
import statistics


# =============================================================================
# ENUMS
# =============================================================================

class MetricType(Enum):
    """Types of trust metrics tracked."""
    OVERRIDE_RATE = "override_rate"
    ACCEPTANCE_RATE = "acceptance_rate"
    SATISFACTION_SCORE = "satisfaction_score"
    ADOPTION_RATE = "adoption_rate"
    ENGAGEMENT_SCORE = "engagement_score"
    TIME_TO_DECISION = "time_to_decision"
    TRUST_SCORE = "trust_score"


class InteractionType(Enum):
    """Types of user-AI interactions."""
    RECOMMENDATION = "recommendation"
    PREDICTION = "prediction"
    ALERT = "alert"
    REPORT = "report"
    QUERY_RESPONSE = "query_response"
    ACTION_SUGGESTION = "action_suggestion"
    DIAGNOSIS = "diagnosis"
    FORECAST = "forecast"


class InteractionOutcome(Enum):
    """Outcomes of user-AI interactions."""
    ACCEPTED = "accepted"
    ACCEPTED_MODIFIED = "accepted_modified"
    REJECTED = "rejected"
    OVERRIDDEN = "overridden"
    IGNORED = "ignored"
    DEFERRED = "deferred"
    AUTO_EXECUTED = "auto_executed"


class SatisfactionLevel(Enum):
    """User satisfaction levels."""
    VERY_SATISFIED = 5
    SATISFIED = 4
    NEUTRAL = 3
    DISSATISFIED = 2
    VERY_DISSATISFIED = 1


class FeatureCategory(Enum):
    """Feature categories for adoption tracking."""
    DASHBOARD = "dashboard"
    AI_ASSISTANT = "ai_assistant"
    REPORTS = "reports"
    CASCADE_EXPLORER = "cascade_explorer"
    RESOLUTION_GENOME = "resolution_genome"
    INVESTIGATION_ROOMS = "investigation_rooms"
    ALERTS = "alerts"
    NATURAL_LANGUAGE = "natural_language"


class TrustLevel(Enum):
    """Overall trust level classification."""
    VERY_HIGH = "very_high"      # > 85%
    HIGH = "high"                # 70-85%
    MODERATE = "moderate"        # 55-70%
    LOW = "low"                  # 40-55%
    VERY_LOW = "very_low"        # < 40%


class TrendDirection(Enum):
    """Trend direction for metrics."""
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Interaction:
    """Record of a user-AI interaction."""
    interaction_id: str
    timestamp: datetime
    user_id: str
    user_role: str
    agent_name: str
    interaction_type: InteractionType
    entity_type: Optional[str] = None
    entity_id: Optional[str] = None
    ai_confidence: float = 0.0
    ai_suggestion: Optional[str] = None
    outcome: Optional[InteractionOutcome] = None
    outcome_timestamp: Optional[datetime] = None
    time_to_decision_seconds: Optional[float] = None
    user_modification: Optional[str] = None
    override_reason: Optional[str] = None
    satisfaction_score: Optional[int] = None
    satisfaction_feedback: Optional[str] = None
    context: Dict = field(default_factory=dict)


@dataclass
class SatisfactionFeedback:
    """User satisfaction feedback record."""
    feedback_id: str
    timestamp: datetime
    user_id: str
    user_role: str
    interaction_id: Optional[str] = None
    feature_category: Optional[FeatureCategory] = None
    agent_name: Optional[str] = None
    satisfaction_level: SatisfactionLevel = SatisfactionLevel.NEUTRAL
    rating: int = 3  # 1-5 scale
    feedback_text: Optional[str] = None
    would_recommend: Optional[bool] = None  # NPS-style
    tags: List[str] = field(default_factory=list)


@dataclass
class FeatureUsage:
    """Feature usage record for adoption tracking."""
    usage_id: str
    timestamp: datetime
    user_id: str
    user_role: str
    feature_category: FeatureCategory
    feature_name: str
    action: str  # view, click, use, complete
    duration_seconds: Optional[float] = None
    success: bool = True
    context: Dict = field(default_factory=dict)


@dataclass
class TrustMetrics:
    """Aggregated trust metrics for a period."""
    period_start: datetime
    period_end: datetime
    
    # Override metrics
    total_interactions: int = 0
    overrides: int = 0
    override_rate: float = 0.0
    override_rate_trend: TrendDirection = TrendDirection.STABLE
    
    # Acceptance metrics
    accepted: int = 0
    accepted_modified: int = 0
    rejected: int = 0
    acceptance_rate: float = 0.0
    acceptance_with_modification_rate: float = 0.0
    
    # Satisfaction metrics
    satisfaction_count: int = 0
    avg_satisfaction: float = 0.0
    nps_score: float = 0.0  # Net Promoter Score
    satisfaction_trend: TrendDirection = TrendDirection.STABLE
    
    # Adoption metrics
    active_users: int = 0
    total_users: int = 0
    adoption_rate: float = 0.0
    feature_usage_count: int = 0
    avg_session_duration: float = 0.0
    
    # Overall trust
    trust_score: float = 0.0
    trust_level: TrustLevel = TrustLevel.MODERATE
    
    # Breakdown by agent
    by_agent: Dict[str, Dict] = field(default_factory=dict)
    
    # Breakdown by role
    by_role: Dict[str, Dict] = field(default_factory=dict)
    
    # Breakdown by feature
    by_feature: Dict[str, Dict] = field(default_factory=dict)


@dataclass
class TrustTrend:
    """Trust trend over time."""
    metric_type: MetricType
    data_points: List[Tuple[datetime, float]] = field(default_factory=list)
    trend_direction: TrendDirection = TrendDirection.STABLE
    change_percent: float = 0.0
    forecast_next_period: Optional[float] = None


@dataclass
class TrustAlert:
    """Alert for trust metric anomalies."""
    alert_id: str
    timestamp: datetime
    metric_type: MetricType
    severity: str  # low, medium, high, critical
    current_value: float
    threshold: float
    message: str
    recommendation: str
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None


# =============================================================================
# TRUST METRICS SYSTEM
# =============================================================================

class TrustMetricsSystem:
    """
    Comprehensive trust metrics tracking system.
    
    Tracks:
    - Override rates (how often humans override AI)
    - Suggestion acceptance (AI recommendation adoption)
    - User satisfaction (ratings and feedback)
    - Adoption tracking (feature usage and engagement)
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize the trust metrics system."""
        if db_path is None:
            db_dir = Path("data/governance")
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(db_dir / "trust_metrics.db")
        
        self.db_path = db_path
        self._init_database()
        
        # Thresholds for alerts
        self.thresholds = {
            'override_rate_high': 0.30,
            'override_rate_critical': 0.50,
            'acceptance_rate_low': 0.60,
            'acceptance_rate_critical': 0.40,
            'satisfaction_low': 3.0,
            'satisfaction_critical': 2.5,
            'adoption_low': 0.50,
            'nps_low': 0,
        }
    
    def _init_database(self):
        """Initialize SQLite database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Interactions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interactions (
                interaction_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                user_id TEXT NOT NULL,
                user_role TEXT NOT NULL,
                agent_name TEXT NOT NULL,
                interaction_type TEXT NOT NULL,
                entity_type TEXT,
                entity_id TEXT,
                ai_confidence REAL DEFAULT 0,
                ai_suggestion TEXT,
                outcome TEXT,
                outcome_timestamp TEXT,
                time_to_decision_seconds REAL,
                user_modification TEXT,
                override_reason TEXT,
                satisfaction_score INTEGER,
                satisfaction_feedback TEXT,
                context TEXT
            )
        ''')
        
        # Satisfaction feedback table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS satisfaction_feedback (
                feedback_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                user_id TEXT NOT NULL,
                user_role TEXT NOT NULL,
                interaction_id TEXT,
                feature_category TEXT,
                agent_name TEXT,
                satisfaction_level INTEGER NOT NULL,
                rating INTEGER NOT NULL,
                feedback_text TEXT,
                would_recommend INTEGER,
                tags TEXT
            )
        ''')
        
        # Feature usage table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feature_usage (
                usage_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                user_id TEXT NOT NULL,
                user_role TEXT NOT NULL,
                feature_category TEXT NOT NULL,
                feature_name TEXT NOT NULL,
                action TEXT NOT NULL,
                duration_seconds REAL,
                success INTEGER DEFAULT 1,
                context TEXT
            )
        ''')
        
        # Trust alerts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trust_alerts (
                alert_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                metric_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                current_value REAL NOT NULL,
                threshold REAL NOT NULL,
                message TEXT NOT NULL,
                recommendation TEXT NOT NULL,
                acknowledged INTEGER DEFAULT 0,
                acknowledged_by TEXT,
                acknowledged_at TEXT
            )
        ''')
        
        # Daily metrics snapshot table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_metrics (
                date TEXT PRIMARY KEY,
                total_interactions INTEGER DEFAULT 0,
                overrides INTEGER DEFAULT 0,
                override_rate REAL DEFAULT 0,
                accepted INTEGER DEFAULT 0,
                rejected INTEGER DEFAULT 0,
                acceptance_rate REAL DEFAULT 0,
                avg_satisfaction REAL DEFAULT 0,
                satisfaction_count INTEGER DEFAULT 0,
                active_users INTEGER DEFAULT 0,
                feature_usage_count INTEGER DEFAULT 0,
                trust_score REAL DEFAULT 0,
                metrics_json TEXT
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_interactions_user ON interactions(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_interactions_agent ON interactions(agent_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_interactions_timestamp ON interactions(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_interactions_outcome ON interactions(outcome)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_user ON satisfaction_feedback(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_usage_user ON feature_usage(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_usage_feature ON feature_usage(feature_category)')
        
        conn.commit()
        conn.close()
    
    # =========================================================================
    # INTERACTION TRACKING
    # =========================================================================
    
    def record_interaction(
        self,
        user_id: str,
        user_role: str,
        agent_name: str,
        interaction_type: InteractionType,
        ai_confidence: float = 0.0,
        ai_suggestion: Optional[str] = None,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        context: Optional[Dict] = None
    ) -> str:
        """Record a new user-AI interaction."""
        interaction_id = f"INT-{datetime.now().strftime('%Y%m%d%H%M%S')}-{hashlib.md5(f'{user_id}{agent_name}{datetime.now().isoformat()}'.encode()).hexdigest()[:8]}"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO interactions (
                interaction_id, timestamp, user_id, user_role, agent_name,
                interaction_type, entity_type, entity_id, ai_confidence,
                ai_suggestion, context
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            interaction_id,
            datetime.now().isoformat(),
            user_id,
            user_role,
            agent_name,
            interaction_type.value,
            entity_type,
            entity_id,
            ai_confidence,
            ai_suggestion,
            json.dumps(context or {})
        ))
        
        conn.commit()
        conn.close()
        
        return interaction_id
    
    def record_outcome(
        self,
        interaction_id: str,
        outcome: InteractionOutcome,
        time_to_decision_seconds: Optional[float] = None,
        user_modification: Optional[str] = None,
        override_reason: Optional[str] = None,
        satisfaction_score: Optional[int] = None,
        satisfaction_feedback: Optional[str] = None
    ) -> bool:
        """Record the outcome of an interaction."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE interactions SET
                outcome = ?,
                outcome_timestamp = ?,
                time_to_decision_seconds = ?,
                user_modification = ?,
                override_reason = ?,
                satisfaction_score = ?,
                satisfaction_feedback = ?
            WHERE interaction_id = ?
        ''', (
            outcome.value,
            datetime.now().isoformat(),
            time_to_decision_seconds,
            user_modification,
            override_reason,
            satisfaction_score,
            satisfaction_feedback,
            interaction_id
        ))
        
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        return success
    
    def get_interaction(self, interaction_id: str) -> Optional[Interaction]:
        """Get interaction by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM interactions WHERE interaction_id = ?', (interaction_id,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return self._row_to_interaction(row)
    
    def _row_to_interaction(self, row) -> Interaction:
        """Convert database row to Interaction object."""
        return Interaction(
            interaction_id=row[0],
            timestamp=datetime.fromisoformat(row[1]),
            user_id=row[2],
            user_role=row[3],
            agent_name=row[4],
            interaction_type=InteractionType(row[5]),
            entity_type=row[6],
            entity_id=row[7],
            ai_confidence=row[8] or 0.0,
            ai_suggestion=row[9],
            outcome=InteractionOutcome(row[10]) if row[10] else None,
            outcome_timestamp=datetime.fromisoformat(row[11]) if row[11] else None,
            time_to_decision_seconds=row[12],
            user_modification=row[13],
            override_reason=row[14],
            satisfaction_score=row[15],
            satisfaction_feedback=row[16],
            context=json.loads(row[17]) if row[17] else {}
        )
    
    # =========================================================================
    # OVERRIDE TRACKING
    # =========================================================================
    
    def get_override_rate(
        self,
        days: int = 30,
        agent_name: Optional[str] = None,
        user_role: Optional[str] = None
    ) -> Dict:
        """Calculate override rate for a period."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        start_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        query = '''
            SELECT 
                outcome,
                COUNT(*) as count
            FROM interactions
            WHERE timestamp >= ?
            AND outcome IS NOT NULL
        '''
        params = [start_date]
        
        if agent_name:
            query += ' AND agent_name = ?'
            params.append(agent_name)
        
        if user_role:
            query += ' AND user_role = ?'
            params.append(user_role)
        
        query += ' GROUP BY outcome'
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        outcomes = {row[0]: row[1] for row in rows}
        total = sum(outcomes.values())
        overrides = outcomes.get('overridden', 0)
        
        override_rate = overrides / total if total > 0 else 0.0
        
        return {
            'total_interactions': total,
            'overrides': overrides,
            'override_rate': override_rate,
            'override_rate_percent': f"{override_rate * 100:.1f}%",
            'by_outcome': outcomes,
            'period_days': days
        }
    
    def get_override_reasons(self, days: int = 30) -> Dict[str, int]:
        """Get breakdown of override reasons."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        start_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        cursor.execute('''
            SELECT override_reason, COUNT(*) as count
            FROM interactions
            WHERE timestamp >= ?
            AND outcome = 'overridden'
            AND override_reason IS NOT NULL
            GROUP BY override_reason
            ORDER BY count DESC
        ''', (start_date,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return {row[0]: row[1] for row in rows}
    
    # =========================================================================
    # ACCEPTANCE TRACKING
    # =========================================================================
    
    def get_acceptance_rate(
        self,
        days: int = 30,
        agent_name: Optional[str] = None,
        interaction_type: Optional[InteractionType] = None
    ) -> Dict:
        """Calculate acceptance rate for AI suggestions."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        start_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        query = '''
            SELECT 
                outcome,
                COUNT(*) as count,
                AVG(time_to_decision_seconds) as avg_decision_time
            FROM interactions
            WHERE timestamp >= ?
            AND outcome IS NOT NULL
        '''
        params = [start_date]
        
        if agent_name:
            query += ' AND agent_name = ?'
            params.append(agent_name)
        
        if interaction_type:
            query += ' AND interaction_type = ?'
            params.append(interaction_type.value)
        
        query += ' GROUP BY outcome'
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        outcomes = {}
        total = 0
        total_decision_time = 0
        decision_time_count = 0
        
        for row in rows:
            outcome, count, avg_time = row
            outcomes[outcome] = count
            total += count
            if avg_time:
                total_decision_time += avg_time * count
                decision_time_count += count
        
        accepted = outcomes.get('accepted', 0)
        accepted_modified = outcomes.get('accepted_modified', 0)
        rejected = outcomes.get('rejected', 0)
        auto_executed = outcomes.get('auto_executed', 0)
        
        acceptance_rate = (accepted + accepted_modified + auto_executed) / total if total > 0 else 0.0
        pure_acceptance_rate = accepted / total if total > 0 else 0.0
        modification_rate = accepted_modified / total if total > 0 else 0.0
        rejection_rate = rejected / total if total > 0 else 0.0
        avg_decision_time = total_decision_time / decision_time_count if decision_time_count > 0 else 0.0
        
        return {
            'total_interactions': total,
            'accepted': accepted,
            'accepted_modified': accepted_modified,
            'rejected': rejected,
            'auto_executed': auto_executed,
            'acceptance_rate': acceptance_rate,
            'acceptance_rate_percent': f"{acceptance_rate * 100:.1f}%",
            'pure_acceptance_rate': pure_acceptance_rate,
            'modification_rate': modification_rate,
            'rejection_rate': rejection_rate,
            'avg_decision_time_seconds': avg_decision_time,
            'period_days': days
        }
    
    def get_acceptance_by_agent(self, days: int = 30) -> Dict[str, Dict]:
        """Get acceptance rates broken down by agent."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        start_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        cursor.execute('''
            SELECT 
                agent_name,
                outcome,
                COUNT(*) as count
            FROM interactions
            WHERE timestamp >= ?
            AND outcome IS NOT NULL
            GROUP BY agent_name, outcome
        ''', (start_date,))
        
        rows = cursor.fetchall()
        conn.close()
        
        agents = {}
        for row in rows:
            agent, outcome, count = row
            if agent not in agents:
                agents[agent] = {'total': 0, 'accepted': 0, 'rejected': 0, 'overridden': 0}
            agents[agent]['total'] += count
            if outcome in ['accepted', 'accepted_modified', 'auto_executed']:
                agents[agent]['accepted'] += count
            elif outcome == 'rejected':
                agents[agent]['rejected'] += count
            elif outcome == 'overridden':
                agents[agent]['overridden'] += count
        
        for agent in agents:
            total = agents[agent]['total']
            agents[agent]['acceptance_rate'] = agents[agent]['accepted'] / total if total > 0 else 0.0
            agents[agent]['rejection_rate'] = agents[agent]['rejected'] / total if total > 0 else 0.0
            agents[agent]['override_rate'] = agents[agent]['overridden'] / total if total > 0 else 0.0
        
        return agents
    
    # =========================================================================
    # SATISFACTION TRACKING
    # =========================================================================
    
    def record_satisfaction(
        self,
        user_id: str,
        user_role: str,
        satisfaction_level: SatisfactionLevel,
        rating: int = 3,
        feedback_text: Optional[str] = None,
        would_recommend: Optional[bool] = None,
        interaction_id: Optional[str] = None,
        feature_category: Optional[FeatureCategory] = None,
        agent_name: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """Record user satisfaction feedback."""
        feedback_id = f"FB-{datetime.now().strftime('%Y%m%d%H%M%S')}-{hashlib.md5(f'{user_id}{datetime.now().isoformat()}'.encode()).hexdigest()[:8]}"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO satisfaction_feedback (
                feedback_id, timestamp, user_id, user_role, interaction_id,
                feature_category, agent_name, satisfaction_level, rating,
                feedback_text, would_recommend, tags
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            feedback_id,
            datetime.now().isoformat(),
            user_id,
            user_role,
            interaction_id,
            feature_category.value if feature_category else None,
            agent_name,
            satisfaction_level.value,
            rating,
            feedback_text,
            1 if would_recommend else 0 if would_recommend is False else None,
            json.dumps(tags or [])
        ))
        
        conn.commit()
        conn.close()
        
        return feedback_id
    
    def get_satisfaction_metrics(
        self,
        days: int = 30,
        agent_name: Optional[str] = None,
        feature_category: Optional[FeatureCategory] = None
    ) -> Dict:
        """Get satisfaction metrics for a period."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        start_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        query = '''
            SELECT 
                rating,
                would_recommend,
                satisfaction_level
            FROM satisfaction_feedback
            WHERE timestamp >= ?
        '''
        params = [start_date]
        
        if agent_name:
            query += ' AND agent_name = ?'
            params.append(agent_name)
        
        if feature_category:
            query += ' AND feature_category = ?'
            params.append(feature_category.value)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return {
                'count': 0,
                'avg_rating': 0.0,
                'avg_rating_formatted': '0.0/5',
                'nps_score': 0.0,
                'promoters': 0,
                'detractors': 0,
                'satisfaction_distribution': {},
                'period_days': days
            }
        
        ratings = [row[0] for row in rows if row[0] is not None]
        recommendations = [row[1] for row in rows if row[1] is not None]
        levels = [row[2] for row in rows if row[2] is not None]
        
        avg_rating = statistics.mean(ratings) if ratings else 0.0
        
        promoters = sum(1 for r in recommendations if r == 1)
        detractors = sum(1 for r in recommendations if r == 0)
        total_nps = len(recommendations)
        nps_score = ((promoters - detractors) / total_nps * 100) if total_nps > 0 else 0.0
        
        level_counts = {}
        for level in levels:
            level_counts[level] = level_counts.get(level, 0) + 1
        
        return {
            'count': len(rows),
            'avg_rating': round(avg_rating, 2),
            'avg_rating_formatted': f"{avg_rating:.1f}/5",
            'nps_score': round(nps_score, 1),
            'promoters': promoters,
            'detractors': detractors,
            'satisfaction_distribution': level_counts,
            'period_days': days
        }
    
    def get_recent_feedback(self, limit: int = 10) -> List[SatisfactionFeedback]:
        """Get recent satisfaction feedback."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM satisfaction_feedback
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_feedback(row) for row in rows]
    
    def _row_to_feedback(self, row) -> SatisfactionFeedback:
        """Convert database row to SatisfactionFeedback object."""
        return SatisfactionFeedback(
            feedback_id=row[0],
            timestamp=datetime.fromisoformat(row[1]),
            user_id=row[2],
            user_role=row[3],
            interaction_id=row[4],
            feature_category=FeatureCategory(row[5]) if row[5] else None,
            agent_name=row[6],
            satisfaction_level=SatisfactionLevel(row[7]),
            rating=row[8],
            feedback_text=row[9],
            would_recommend=bool(row[10]) if row[10] is not None else None,
            tags=json.loads(row[11]) if row[11] else []
        )
    
    # =========================================================================
    # ADOPTION TRACKING
    # =========================================================================
    
    def record_feature_usage(
        self,
        user_id: str,
        user_role: str,
        feature_category: FeatureCategory,
        feature_name: str,
        action: str = "use",
        duration_seconds: Optional[float] = None,
        success: bool = True,
        context: Optional[Dict] = None
    ) -> str:
        """Record feature usage for adoption tracking."""
        usage_id = f"USG-{datetime.now().strftime('%Y%m%d%H%M%S')}-{hashlib.md5(f'{user_id}{feature_name}{datetime.now().isoformat()}'.encode()).hexdigest()[:8]}"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO feature_usage (
                usage_id, timestamp, user_id, user_role, feature_category,
                feature_name, action, duration_seconds, success, context
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            usage_id,
            datetime.now().isoformat(),
            user_id,
            user_role,
            feature_category.value,
            feature_name,
            action,
            duration_seconds,
            1 if success else 0,
            json.dumps(context or {})
        ))
        
        conn.commit()
        conn.close()
        
        return usage_id
    
    def get_adoption_metrics(
        self,
        days: int = 30,
        total_users: int = 100
    ) -> Dict:
        """Get feature adoption metrics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        start_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        cursor.execute('''
            SELECT COUNT(DISTINCT user_id)
            FROM feature_usage
            WHERE timestamp >= ?
        ''', (start_date,))
        active_users = cursor.fetchone()[0]
        
        cursor.execute('''
            SELECT 
                feature_category,
                COUNT(*) as usage_count,
                COUNT(DISTINCT user_id) as unique_users,
                AVG(duration_seconds) as avg_duration
            FROM feature_usage
            WHERE timestamp >= ?
            GROUP BY feature_category
        ''', (start_date,))
        feature_rows = cursor.fetchall()
        
        cursor.execute('''
            SELECT COUNT(*), AVG(duration_seconds)
            FROM feature_usage
            WHERE timestamp >= ?
        ''', (start_date,))
        total_usage, avg_duration = cursor.fetchone()
        
        conn.close()
        
        adoption_rate = active_users / total_users if total_users > 0 else 0.0
        
        by_feature = {}
        for row in feature_rows:
            category, count, users, avg_dur = row
            by_feature[category] = {
                'usage_count': count,
                'unique_users': users,
                'avg_duration_seconds': round(avg_dur, 1) if avg_dur else 0.0,
                'adoption_rate': users / total_users if total_users > 0 else 0.0
            }
        
        return {
            'active_users': active_users,
            'total_users': total_users,
            'adoption_rate': adoption_rate,
            'adoption_rate_percent': f"{adoption_rate * 100:.1f}%",
            'total_usage_count': total_usage or 0,
            'avg_session_duration': round(avg_duration, 1) if avg_duration else 0.0,
            'by_feature': by_feature,
            'period_days': days
        }
    
    def get_feature_usage_trend(
        self,
        feature_category: FeatureCategory,
        days: int = 30
    ) -> List[Dict]:
        """Get daily usage trend for a feature."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        start_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        cursor.execute('''
            SELECT 
                DATE(timestamp) as date,
                COUNT(*) as usage_count,
                COUNT(DISTINCT user_id) as unique_users
            FROM feature_usage
            WHERE timestamp >= ?
            AND feature_category = ?
            GROUP BY DATE(timestamp)
            ORDER BY date
        ''', (start_date, feature_category.value))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                'date': row[0],
                'usage_count': row[1],
                'unique_users': row[2]
            }
            for row in rows
        ]
    
    # =========================================================================
    # TRUST SCORE CALCULATION
    # =========================================================================
    
    def calculate_trust_metrics(
        self,
        days: int = 30,
        total_users: int = 100
    ) -> TrustMetrics:
        """Calculate comprehensive trust metrics for a period."""
        period_start = datetime.now() - timedelta(days=days)
        period_end = datetime.now()
        
        override_data = self.get_override_rate(days=days)
        acceptance_data = self.get_acceptance_rate(days=days)
        satisfaction_data = self.get_satisfaction_metrics(days=days)
        adoption_data = self.get_adoption_metrics(days=days, total_users=total_users)
        
        by_agent = self.get_acceptance_by_agent(days=days)
        
        acceptance_score = acceptance_data['acceptance_rate'] * 100
        satisfaction_score = (satisfaction_data['avg_rating'] / 5) * 100 if satisfaction_data['avg_rating'] > 0 else 50
        adoption_score = adoption_data['adoption_rate'] * 100
        override_penalty = override_data['override_rate'] * 50
        
        trust_score = (
            acceptance_score * 0.35 +
            satisfaction_score * 0.25 +
            adoption_score * 0.25 +
            (100 - override_penalty) * 0.15
        )
        
        trust_score = max(0, min(100, trust_score))
        
        if trust_score >= 85:
            trust_level = TrustLevel.VERY_HIGH
        elif trust_score >= 70:
            trust_level = TrustLevel.HIGH
        elif trust_score >= 55:
            trust_level = TrustLevel.MODERATE
        elif trust_score >= 40:
            trust_level = TrustLevel.LOW
        else:
            trust_level = TrustLevel.VERY_LOW
        
        override_trend = self._calculate_trend('override_rate', days)
        satisfaction_trend = self._calculate_trend('satisfaction', days)
        
        return TrustMetrics(
            period_start=period_start,
            period_end=period_end,
            total_interactions=override_data['total_interactions'],
            overrides=override_data['overrides'],
            override_rate=override_data['override_rate'],
            override_rate_trend=override_trend,
            accepted=acceptance_data['accepted'],
            accepted_modified=acceptance_data['accepted_modified'],
            rejected=acceptance_data['rejected'],
            acceptance_rate=acceptance_data['acceptance_rate'],
            acceptance_with_modification_rate=acceptance_data['modification_rate'],
            satisfaction_count=satisfaction_data['count'],
            avg_satisfaction=satisfaction_data['avg_rating'],
            nps_score=satisfaction_data['nps_score'],
            satisfaction_trend=satisfaction_trend,
            active_users=adoption_data['active_users'],
            total_users=adoption_data['total_users'],
            adoption_rate=adoption_data['adoption_rate'],
            feature_usage_count=adoption_data['total_usage_count'],
            avg_session_duration=adoption_data['avg_session_duration'],
            trust_score=round(trust_score, 1),
            trust_level=trust_level,
            by_agent=by_agent,
            by_feature=adoption_data['by_feature']
        )
    
    def _calculate_trend(self, metric_name: str, days: int) -> TrendDirection:
        """Calculate trend direction for a metric."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        current_start = (datetime.now() - timedelta(days=days)).isoformat()
        previous_start = (datetime.now() - timedelta(days=days * 2)).isoformat()
        previous_end = (datetime.now() - timedelta(days=days)).isoformat()
        
        if metric_name == 'override_rate':
            cursor.execute('''
                SELECT 
                    SUM(CASE WHEN outcome = 'overridden' THEN 1 ELSE 0 END) * 1.0 / 
                    NULLIF(COUNT(*), 0)
                FROM interactions
                WHERE timestamp >= ? AND outcome IS NOT NULL
            ''', (current_start,))
            current = cursor.fetchone()[0] or 0
            
            cursor.execute('''
                SELECT 
                    SUM(CASE WHEN outcome = 'overridden' THEN 1 ELSE 0 END) * 1.0 / 
                    NULLIF(COUNT(*), 0)
                FROM interactions
                WHERE timestamp >= ? AND timestamp < ? AND outcome IS NOT NULL
            ''', (previous_start, previous_end))
            previous = cursor.fetchone()[0] or 0
            
        elif metric_name == 'satisfaction':
            cursor.execute('''
                SELECT AVG(rating) FROM satisfaction_feedback WHERE timestamp >= ?
            ''', (current_start,))
            current = cursor.fetchone()[0] or 0
            
            cursor.execute('''
                SELECT AVG(rating) FROM satisfaction_feedback 
                WHERE timestamp >= ? AND timestamp < ?
            ''', (previous_start, previous_end))
            previous = cursor.fetchone()[0] or 0
        
        else:
            conn.close()
            return TrendDirection.STABLE
        
        conn.close()
        
        if previous == 0:
            return TrendDirection.STABLE
        
        change = (current - previous) / previous
        
        if metric_name == 'override_rate':
            if change < -0.1:
                return TrendDirection.IMPROVING
            elif change > 0.1:
                return TrendDirection.DECLINING
        else:
            if change > 0.1:
                return TrendDirection.IMPROVING
            elif change < -0.1:
                return TrendDirection.DECLINING
        
        return TrendDirection.STABLE
    
    # =========================================================================
    # ALERTS
    # =========================================================================
    
    def check_trust_alerts(self, days: int = 30) -> List[TrustAlert]:
        """Check for trust metric anomalies and create alerts."""
        alerts = []
        
        metrics = self.calculate_trust_metrics(days=days)
        
        if metrics.override_rate > self.thresholds['override_rate_critical']:
            alerts.append(self._create_alert(
                MetricType.OVERRIDE_RATE,
                'critical',
                metrics.override_rate,
                self.thresholds['override_rate_critical'],
                f"Critical: Override rate is {metrics.override_rate:.1%}, exceeding {self.thresholds['override_rate_critical']:.0%} threshold",
                "Review AI recommendations quality. Consider retraining models or adjusting confidence thresholds."
            ))
        elif metrics.override_rate > self.thresholds['override_rate_high']:
            alerts.append(self._create_alert(
                MetricType.OVERRIDE_RATE,
                'high',
                metrics.override_rate,
                self.thresholds['override_rate_high'],
                f"High override rate: {metrics.override_rate:.1%}",
                "Analyze override reasons and improve AI accuracy for commonly overridden decisions."
            ))
        
        if metrics.acceptance_rate < self.thresholds['acceptance_rate_critical']:
            alerts.append(self._create_alert(
                MetricType.ACCEPTANCE_RATE,
                'critical',
                metrics.acceptance_rate,
                self.thresholds['acceptance_rate_critical'],
                f"Critical: Acceptance rate is {metrics.acceptance_rate:.1%}, below {self.thresholds['acceptance_rate_critical']:.0%}",
                "Urgent review of AI recommendation quality needed. Consider user feedback sessions."
            ))
        elif metrics.acceptance_rate < self.thresholds['acceptance_rate_low']:
            alerts.append(self._create_alert(
                MetricType.ACCEPTANCE_RATE,
                'medium',
                metrics.acceptance_rate,
                self.thresholds['acceptance_rate_low'],
                f"Low acceptance rate: {metrics.acceptance_rate:.1%}",
                "Review recommendation relevance and presentation. Gather user feedback."
            ))
        
        if metrics.avg_satisfaction > 0 and metrics.avg_satisfaction < self.thresholds['satisfaction_critical']:
            alerts.append(self._create_alert(
                MetricType.SATISFACTION_SCORE,
                'critical',
                metrics.avg_satisfaction,
                self.thresholds['satisfaction_critical'],
                f"Critical: Satisfaction score is {metrics.avg_satisfaction:.1f}/5",
                "Immediate action needed. Conduct user interviews and identify pain points."
            ))
        elif metrics.avg_satisfaction > 0 and metrics.avg_satisfaction < self.thresholds['satisfaction_low']:
            alerts.append(self._create_alert(
                MetricType.SATISFACTION_SCORE,
                'medium',
                metrics.avg_satisfaction,
                self.thresholds['satisfaction_low'],
                f"Low satisfaction: {metrics.avg_satisfaction:.1f}/5",
                "Review recent feedback and implement improvements."
            ))
        
        if metrics.nps_score < self.thresholds['nps_low']:
            alerts.append(self._create_alert(
                MetricType.TRUST_SCORE,
                'high',
                metrics.nps_score,
                self.thresholds['nps_low'],
                f"Negative NPS: {metrics.nps_score:.0f}",
                "More detractors than promoters. Focus on converting detractors."
            ))
        
        if metrics.adoption_rate < self.thresholds['adoption_low']:
            alerts.append(self._create_alert(
                MetricType.ADOPTION_RATE,
                'medium',
                metrics.adoption_rate,
                self.thresholds['adoption_low'],
                f"Low adoption: {metrics.adoption_rate:.1%}",
                "Increase training and awareness. Simplify onboarding."
            ))
        
        for alert in alerts:
            self._save_alert(alert)
        
        return alerts
    
    def _create_alert(
        self,
        metric_type: MetricType,
        severity: str,
        current_value: float,
        threshold: float,
        message: str,
        recommendation: str
    ) -> TrustAlert:
        """Create a trust alert."""
        alert_id = f"TA-{datetime.now().strftime('%Y%m%d%H%M%S')}-{hashlib.md5(f'{metric_type.value}{datetime.now().isoformat()}'.encode()).hexdigest()[:6]}"
        
        return TrustAlert(
            alert_id=alert_id,
            timestamp=datetime.now(),
            metric_type=metric_type,
            severity=severity,
            current_value=current_value,
            threshold=threshold,
            message=message,
            recommendation=recommendation
        )
    
    def _save_alert(self, alert: TrustAlert):
        """Save alert to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO trust_alerts (
                alert_id, timestamp, metric_type, severity, current_value,
                threshold, message, recommendation, acknowledged,
                acknowledged_by, acknowledged_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            alert.alert_id,
            alert.timestamp.isoformat(),
            alert.metric_type.value,
            alert.severity,
            alert.current_value,
            alert.threshold,
            alert.message,
            alert.recommendation,
            1 if alert.acknowledged else 0,
            alert.acknowledged_by,
            alert.acknowledged_at.isoformat() if alert.acknowledged_at else None
        ))
        
        conn.commit()
        conn.close()
    
    def get_active_alerts(self) -> List[TrustAlert]:
        """Get unacknowledged trust alerts."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM trust_alerts
            WHERE acknowledged = 0
            ORDER BY 
                CASE severity 
                    WHEN 'critical' THEN 1 
                    WHEN 'high' THEN 2 
                    WHEN 'medium' THEN 3 
                    ELSE 4 
                END,
                timestamp DESC
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_alert(row) for row in rows]
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge a trust alert."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE trust_alerts SET
                acknowledged = 1,
                acknowledged_by = ?,
                acknowledged_at = ?
            WHERE alert_id = ?
        ''', (acknowledged_by, datetime.now().isoformat(), alert_id))
        
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        return success
    
    def _row_to_alert(self, row) -> TrustAlert:
        """Convert database row to TrustAlert object."""
        return TrustAlert(
            alert_id=row[0],
            timestamp=datetime.fromisoformat(row[1]),
            metric_type=MetricType(row[2]),
            severity=row[3],
            current_value=row[4],
            threshold=row[5],
            message=row[6],
            recommendation=row[7],
            acknowledged=bool(row[8]),
            acknowledged_by=row[9],
            acknowledged_at=datetime.fromisoformat(row[10]) if row[10] else None
        )
    
    # =========================================================================
    # STATISTICS & EXPORT
    # =========================================================================
    
    def get_statistics(self) -> Dict:
        """Get overall trust metrics statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM interactions')
        total_interactions = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM satisfaction_feedback')
        total_feedback = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM feature_usage')
        total_usage = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM trust_alerts WHERE acknowledged = 0')
        active_alerts = cursor.fetchone()[0]
        
        conn.close()
        
        metrics = self.calculate_trust_metrics(days=30)
        
        return {
            'total_interactions': total_interactions,
            'total_feedback': total_feedback,
            'total_feature_usage': total_usage,
            'active_alerts': active_alerts,
            'trust_score': metrics.trust_score,
            'trust_level': metrics.trust_level.value,
            'override_rate': metrics.override_rate,
            'acceptance_rate': metrics.acceptance_rate,
            'avg_satisfaction': metrics.avg_satisfaction,
            'adoption_rate': metrics.adoption_rate
        }
    
    def save_daily_snapshot(self) -> bool:
        """Save daily metrics snapshot."""
        date = datetime.now().strftime('%Y-%m-%d')
        metrics = self.calculate_trust_metrics(days=1)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO daily_metrics (
                date, total_interactions, overrides, override_rate,
                accepted, rejected, acceptance_rate, avg_satisfaction,
                satisfaction_count, active_users, feature_usage_count,
                trust_score, metrics_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            date,
            metrics.total_interactions,
            metrics.overrides,
            metrics.override_rate,
            metrics.accepted,
            metrics.rejected,
            metrics.acceptance_rate,
            metrics.avg_satisfaction,
            metrics.satisfaction_count,
            metrics.active_users,
            metrics.feature_usage_count,
            metrics.trust_score,
            json.dumps(asdict(metrics), default=str)
        ))
        
        conn.commit()
        conn.close()
        
        return True
    
    def export_metrics(
        self,
        start_date: datetime,
        end_date: datetime,
        output_path: str
    ) -> str:
        """Export metrics for a date range."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM daily_metrics
            WHERE date >= ? AND date <= ?
            ORDER BY date
        ''', (start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')))
        
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        
        conn.close()
        
        data = {
            'export_date': datetime.now().isoformat(),
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'daily_metrics': [dict(zip(columns, row)) for row in rows]
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        return output_path


# =============================================================================
# SINGLETON & CONVENIENCE FUNCTIONS
# =============================================================================

_trust_metrics_system: Optional[TrustMetricsSystem] = None


def get_trust_metrics_system(db_path: Optional[str] = None) -> TrustMetricsSystem:
    """Get or create the TrustMetricsSystem singleton."""
    global _trust_metrics_system
    if _trust_metrics_system is None:
        _trust_metrics_system = TrustMetricsSystem(db_path=db_path)
    return _trust_metrics_system


def reset_trust_metrics_system():
    """Reset the singleton (for testing)."""
    global _trust_metrics_system
    _trust_metrics_system = None


def record_interaction(
    user_id: str,
    user_role: str,
    agent_name: str,
    interaction_type: InteractionType,
    **kwargs
) -> str:
    """Convenience function to record an interaction."""
    return get_trust_metrics_system().record_interaction(
        user_id=user_id,
        user_role=user_role,
        agent_name=agent_name,
        interaction_type=interaction_type,
        **kwargs
    )


def record_outcome(interaction_id: str, outcome: InteractionOutcome, **kwargs) -> bool:
    """Convenience function to record an outcome."""
    return get_trust_metrics_system().record_outcome(
        interaction_id=interaction_id,
        outcome=outcome,
        **kwargs
    )


def record_satisfaction(
    user_id: str,
    user_role: str,
    satisfaction_level: SatisfactionLevel,
    rating: int = 3,
    **kwargs
) -> str:
    """Convenience function to record satisfaction."""
    return get_trust_metrics_system().record_satisfaction(
        user_id=user_id,
        user_role=user_role,
        satisfaction_level=satisfaction_level,
        rating=rating,
        **kwargs
    )


def record_feature_usage(
    user_id: str,
    user_role: str,
    feature_category: FeatureCategory,
    feature_name: str,
    **kwargs
) -> str:
    """Convenience function to record feature usage."""
    return get_trust_metrics_system().record_feature_usage(
        user_id=user_id,
        user_role=user_role,
        feature_category=feature_category,
        feature_name=feature_name,
        **kwargs
    )


def get_trust_metrics(days: int = 30) -> TrustMetrics:
    """Convenience function to get trust metrics."""
    return get_trust_metrics_system().calculate_trust_metrics(days=days)


def get_trust_alerts() -> List[TrustAlert]:
    """Convenience function to get active trust alerts."""
    return get_trust_metrics_system().get_active_alerts()


def get_trust_stats() -> Dict:
    """Convenience function to get trust statistics."""
    return get_trust_metrics_system().get_statistics()
