# src/collaboration/escalation_engine.py
"""
SANCHALAK AI - 5-Level Escalation Engine v1.2
# NOTE: This module previously used SQLite but has been migrated to PostgreSQL-only.
# If you need to use this module, update it to use:
#   from src.database.pg_data_service import get_pg_data_service
#   service = get_pg_data_service()


FIXES in v1.2:
- Fixed database locking issues (proper connection management)
- Fixed history_id unique constraint (added UUID)
- Fixed JSON serialization of EscalationLevel enum
- Restored missing methods (get_escalation_history, get_statistics, get_audit_trail)
"""

# using SQLite for local persistence
import sqlite3
import json
import hashlib
import uuid
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from contextlib import contextmanager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class EscalationLevel(Enum):
    L1_REMINDER = 1
    L2_SUPERVISOR = 2
    L3_INVESTIGATION = 3
    L4_LEADERSHIP = 4
    L5_SPONSOR = 5
    
    @property
    def display_name(self) -> str:
        names = {1: "Level 1 - Reminder", 2: "Level 2 - Supervisor", 
                 3: "Level 3 - Investigation", 4: "Level 4 - Leadership", 
                 5: "Level 5 - Sponsor"}
        return names.get(self.value, f"Level {self.value}")


class EscalationStatus(Enum):
    PENDING = "pending"
    ACKNOWLEDGED = "acknowledged"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    DE_ESCALATED = "de_escalated"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class EscalationTrigger(Enum):
    TIME_BASED = "time_based"
    SLA_BREACH = "sla_breach"
    SLA_WARNING = "sla_warning"
    MANUAL = "manual"
    PATTERN = "pattern"
    SAFETY = "safety"
    CASCADE = "cascade"
    REOPEN = "reopen"


class NotificationPriority(Enum):
    URGENT = "urgent"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class EscalationRule:
    rule_id: str
    level: EscalationLevel
    trigger_type: EscalationTrigger
    trigger_hours: int = 72
    applies_to_priorities: List[str] = field(default_factory=lambda: ["critical", "high", "medium", "low"])
    applies_to_categories: List[str] = field(default_factory=list)
    applies_to_statuses: List[str] = field(default_factory=lambda: ["open", "in_progress", "reopened"])
    notify_roles: List[str] = field(default_factory=list)
    notify_users: List[str] = field(default_factory=list)
    create_investigation_room: bool = False
    generate_impact_report: bool = False
    generate_sponsor_package: bool = False
    notification_priority: NotificationPriority = NotificationPriority.NORMAL
    notification_template: str = ""
    is_active: bool = True
    description: str = ""


@dataclass
class Escalation:
    escalation_id: str
    issue_id: str
    level: EscalationLevel
    status: EscalationStatus
    trigger: EscalationTrigger
    rule_id: Optional[str]
    issue_title: str = ""
    issue_priority: str = ""
    issue_category: str = ""
    issue_age_hours: float = 0
    study_id: Optional[str] = None
    site_id: Optional[str] = None
    patient_id: Optional[str] = None
    escalated_to: List[str] = field(default_factory=list)
    escalated_to_roles: List[str] = field(default_factory=list)
    escalated_by: str = ""
    escalated_by_name: str = ""
    escalated_at: datetime = field(default_factory=datetime.now)
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    impact_score: float = 0.0
    cascade_patients: int = 0
    cascade_issues: int = 0
    dqi_impact: float = 0.0
    investigation_room_id: Optional[str] = None
    notifications_sent: List[str] = field(default_factory=list)
    escalation_reason: str = ""
    resolution_notes: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    @property
    def is_active(self) -> bool:
        return self.status in [EscalationStatus.PENDING, EscalationStatus.ACKNOWLEDGED, EscalationStatus.IN_PROGRESS]


@dataclass
class EscalationImpact:
    issue_id: str
    escalation_level: EscalationLevel
    patients_affected: int = 0
    sites_affected: int = 0
    cascade_issues: int = 0
    cascade_patients: int = 0
    blocked_db_lock: int = 0
    current_dqi: float = 0.0
    projected_dqi_loss: float = 0.0
    days_delayed: float = 0.0
    sla_breach_risk: float = 0.0
    estimated_cost: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    severity_score: float = 0.0
    severity_label: str = ""
    
    def calculate_severity(self):
        score = 0
        if self.patients_affected > 100: score += 30
        elif self.patients_affected > 50: score += 20
        elif self.patients_affected > 10: score += 10
        elif self.patients_affected > 0: score += 5
        
        if self.cascade_patients > 500: score += 25
        elif self.cascade_patients > 100: score += 15
        elif self.cascade_patients > 0: score += 5
        
        if self.projected_dqi_loss > 10: score += 20
        elif self.projected_dqi_loss > 5: score += 10
        elif self.projected_dqi_loss > 0: score += 5
        
        score += min(15, self.sla_breach_risk * 15)
        
        if self.days_delayed > 30: score += 10
        elif self.days_delayed > 14: score += 7
        elif self.days_delayed > 7: score += 5
        elif self.days_delayed > 0: score += 2
        
        self.severity_score = min(100, score)
        
        if self.severity_score >= 75: self.severity_label = "Critical"
        elif self.severity_score >= 50: self.severity_label = "High"
        elif self.severity_score >= 25: self.severity_label = "Medium"
        else: self.severity_label = "Low"


@dataclass
class SponsorPackage:
    package_id: str
    escalation_id: str
    issue_id: str
    executive_summary: str = ""
    issue_title: str = ""
    issue_description: str = ""
    issue_category: str = ""
    issue_priority: str = ""
    issue_age_days: float = 0
    escalation_history: List[Dict] = field(default_factory=list)
    impact: Optional[EscalationImpact] = None
    root_cause: str = ""
    root_cause_confidence: float = 0.0
    actions_taken: List[Dict] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    decisions_required: List[str] = field(default_factory=list)
    proposed_resolution_date: Optional[datetime] = None
    generated_at: datetime = field(default_factory=datetime.now)
    generated_by: str = ""


@dataclass
class EscalationFilter:
    issue_ids: Optional[List[str]] = None
    levels: Optional[List[EscalationLevel]] = None
    statuses: Optional[List[EscalationStatus]] = None
    triggers: Optional[List[EscalationTrigger]] = None
    study_id: Optional[str] = None
    site_id: Optional[str] = None
    escalated_to: Optional[str] = None
    escalated_by: Optional[str] = None
    min_age_hours: Optional[float] = None
    max_age_hours: Optional[float] = None
    is_active: Optional[bool] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None


# =============================================================================
# DEFAULT RULES
# =============================================================================

DEFAULT_ESCALATION_RULES = [
    EscalationRule(rule_id="L1-DEFAULT", level=EscalationLevel.L1_REMINDER, 
                   trigger_type=EscalationTrigger.TIME_BASED, trigger_hours=72,
                   notify_roles=["assignee"], notification_priority=NotificationPriority.NORMAL,
                   description="Auto-reminder to assignee after 3 days"),
    EscalationRule(rule_id="L1-CRITICAL", level=EscalationLevel.L1_REMINDER,
                   trigger_type=EscalationTrigger.TIME_BASED, trigger_hours=24,
                   applies_to_priorities=["critical"], notify_roles=["assignee"],
                   notification_priority=NotificationPriority.HIGH,
                   description="Urgent reminder for critical issues after 24 hours"),
    EscalationRule(rule_id="L2-DEFAULT", level=EscalationLevel.L2_SUPERVISOR,
                   trigger_type=EscalationTrigger.TIME_BASED, trigger_hours=168,
                   applies_to_priorities=["critical", "high", "medium"],
                   notify_roles=["cra", "data_manager", "supervisor"],
                   notification_priority=NotificationPriority.HIGH,
                   description="Escalate to CRA/DM after 7 days"),
    EscalationRule(rule_id="L2-CRITICAL", level=EscalationLevel.L2_SUPERVISOR,
                   trigger_type=EscalationTrigger.TIME_BASED, trigger_hours=48,
                   applies_to_priorities=["critical"],
                   notify_roles=["cra", "data_manager", "supervisor", "ctm"],
                   notification_priority=NotificationPriority.URGENT,
                   description="Fast-track supervisor escalation for critical issues"),
    EscalationRule(rule_id="L3-DEFAULT", level=EscalationLevel.L3_INVESTIGATION,
                   trigger_type=EscalationTrigger.TIME_BASED, trigger_hours=336,
                   applies_to_priorities=["critical", "high", "medium"],
                   notify_roles=["study_lead", "ctm"], create_investigation_room=True,
                   notification_priority=NotificationPriority.HIGH,
                   description="Create Investigation Room after 14 days"),
    EscalationRule(rule_id="L4-DEFAULT", level=EscalationLevel.L4_LEADERSHIP,
                   trigger_type=EscalationTrigger.TIME_BASED, trigger_hours=504,
                   applies_to_priorities=["critical", "high"],
                   notify_roles=["study_lead", "medical_monitor", "project_director"],
                   generate_impact_report=True, notification_priority=NotificationPriority.URGENT,
                   description="Leadership escalation with impact analysis after 21 days"),
    EscalationRule(rule_id="L5-DEFAULT", level=EscalationLevel.L5_SPONSOR,
                   trigger_type=EscalationTrigger.TIME_BASED, trigger_hours=720,
                   applies_to_priorities=["critical"],
                   notify_roles=["sponsor", "study_lead", "medical_monitor"],
                   generate_sponsor_package=True, notification_priority=NotificationPriority.URGENT,
                   description="Sponsor escalation with full package"),
    EscalationRule(rule_id="SLA-WARNING", level=EscalationLevel.L2_SUPERVISOR,
                   trigger_type=EscalationTrigger.SLA_WARNING, trigger_hours=0,
                   notify_roles=["assignee", "supervisor"],
                   notification_priority=NotificationPriority.HIGH,
                   description="Warning before SLA breach"),
    EscalationRule(rule_id="SAFETY-IMMEDIATE", level=EscalationLevel.L3_INVESTIGATION,
                   trigger_type=EscalationTrigger.SAFETY, trigger_hours=24,
                   applies_to_categories=["sae_dm_pending", "sae_safety_pending"],
                   notify_roles=["safety_physician", "safety_data_manager", "study_lead"],
                   create_investigation_room=True, notification_priority=NotificationPriority.URGENT,
                   description="Immediate investigation for safety issues"),
]


# =============================================================================
# ESCALATION ENGINE
# =============================================================================

class EscalationEngine:
    """5-Level Escalation Engine v1.2"""
    
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = Path("data/collaboration/escalation_engine.db")
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        self.rules: Dict[str, EscalationRule] = {}
        self._load_default_rules()
        logger.info(f"EscalationEngine initialized with {len(self.rules)} rules")
    
    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def _init_database(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS escalations (
                    escalation_id TEXT PRIMARY KEY, issue_id TEXT NOT NULL,
                    level INTEGER NOT NULL, status TEXT NOT NULL, trigger TEXT NOT NULL,
                    rule_id TEXT, issue_title TEXT, issue_priority TEXT, issue_category TEXT,
                    issue_age_hours REAL, study_id TEXT, site_id TEXT, patient_id TEXT,
                    escalated_to TEXT, escalated_to_roles TEXT, escalated_by TEXT,
                    escalated_by_name TEXT, escalated_at TEXT, acknowledged_by TEXT,
                    acknowledged_at TEXT, resolved_by TEXT, resolved_at TEXT,
                    impact_score REAL DEFAULT 0, cascade_patients INTEGER DEFAULT 0,
                    cascade_issues INTEGER DEFAULT 0, dqi_impact REAL DEFAULT 0,
                    investigation_room_id TEXT, notifications_sent TEXT,
                    escalation_reason TEXT, resolution_notes TEXT,
                    created_at TEXT, updated_at TEXT)
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS escalation_history (
                    history_id TEXT PRIMARY KEY, escalation_id TEXT NOT NULL,
                    from_level INTEGER, to_level INTEGER NOT NULL,
                    from_status TEXT, to_status TEXT NOT NULL,
                    changed_by TEXT, changed_by_name TEXT, change_reason TEXT, changed_at TEXT)
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS impact_analyses (
                    analysis_id TEXT PRIMARY KEY, escalation_id TEXT, issue_id TEXT NOT NULL,
                    level INTEGER NOT NULL, patients_affected INTEGER DEFAULT 0,
                    sites_affected INTEGER DEFAULT 0, cascade_issues INTEGER DEFAULT 0,
                    cascade_patients INTEGER DEFAULT 0, blocked_db_lock INTEGER DEFAULT 0,
                    current_dqi REAL DEFAULT 0, projected_dqi_loss REAL DEFAULT 0,
                    days_delayed REAL DEFAULT 0, sla_breach_risk REAL DEFAULT 0,
                    estimated_cost REAL DEFAULT 0, severity_score REAL DEFAULT 0,
                    severity_label TEXT, recommendations TEXT, analyzed_at TEXT)
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sponsor_packages (
                    package_id TEXT PRIMARY KEY, escalation_id TEXT NOT NULL,
                    issue_id TEXT NOT NULL, executive_summary TEXT, issue_title TEXT,
                    issue_description TEXT, issue_category TEXT, issue_priority TEXT,
                    issue_age_days REAL, escalation_history TEXT, impact_data TEXT,
                    root_cause TEXT, root_cause_confidence REAL, actions_taken TEXT,
                    recommendations TEXT, decisions_required TEXT,
                    proposed_resolution_date TEXT, generated_at TEXT, generated_by TEXT)
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS escalation_rules (
                    rule_id TEXT PRIMARY KEY, level INTEGER NOT NULL, trigger_type TEXT NOT NULL,
                    trigger_hours INTEGER, applies_to_priorities TEXT, applies_to_categories TEXT,
                    applies_to_statuses TEXT, notify_roles TEXT, notify_users TEXT,
                    create_investigation_room INTEGER DEFAULT 0, generate_impact_report INTEGER DEFAULT 0,
                    generate_sponsor_package INTEGER DEFAULT 0, notification_priority TEXT,
                    notification_template TEXT, is_active INTEGER DEFAULT 1, description TEXT,
                    created_at TEXT, updated_at TEXT)
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS escalation_audit (
                    audit_id TEXT PRIMARY KEY, escalation_id TEXT, action TEXT NOT NULL,
                    actor TEXT, actor_name TEXT, details TEXT, checksum TEXT, created_at TEXT)
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_escalations_issue ON escalations(issue_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_escalations_status ON escalations(status)")
        logger.info("Escalation database initialized")
    
    def _load_default_rules(self):
        for rule in DEFAULT_ESCALATION_RULES:
            self.rules[rule.rule_id] = rule
    
    def _generate_id(self, prefix: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        unique = uuid.uuid4().hex[:6].upper()
        return f"{prefix}-{timestamp}-{unique}"
    
    def _log_audit(self, escalation_id: Optional[str], action: str, actor: str, 
                   actor_name: str = "", details: Dict = None):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            audit_id = self._generate_id("AUD")
            cursor.execute("""
                INSERT INTO escalation_audit 
                (audit_id, escalation_id, action, actor, actor_name, details, checksum, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (audit_id, escalation_id, action, actor, actor_name,
                  json.dumps(details or {}), "", datetime.now().isoformat()))
    
    # =========================================================================
    # RULE MANAGEMENT
    # =========================================================================
    
    def add_rule(self, rule: EscalationRule) -> bool:
        self.rules[rule.rule_id] = rule
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO escalation_rules
                (rule_id, level, trigger_type, trigger_hours, applies_to_priorities,
                 applies_to_categories, applies_to_statuses, notify_roles, notify_users,
                 create_investigation_room, generate_impact_report, generate_sponsor_package,
                 notification_priority, notification_template, is_active, description,
                 created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (rule.rule_id, rule.level.value, rule.trigger_type.value, rule.trigger_hours,
                  json.dumps(rule.applies_to_priorities), json.dumps(rule.applies_to_categories),
                  json.dumps(rule.applies_to_statuses), json.dumps(rule.notify_roles),
                  json.dumps(rule.notify_users), 1 if rule.create_investigation_room else 0,
                  1 if rule.generate_impact_report else 0, 1 if rule.generate_sponsor_package else 0,
                  rule.notification_priority.value, rule.notification_template,
                  1 if rule.is_active else 0, rule.description,
                  datetime.now().isoformat(), datetime.now().isoformat()))
        return True
    
    def get_rule(self, rule_id: str) -> Optional[EscalationRule]:
        return self.rules.get(rule_id)
    
    def get_active_rules(self) -> List[EscalationRule]:
        return [r for r in self.rules.values() if r.is_active]
    
    def get_rules_for_level(self, level: EscalationLevel) -> List[EscalationRule]:
        return [r for r in self.rules.values() if r.level == level and r.is_active]
    
    # =========================================================================
    # ESCALATION CREATION & RETRIEVAL
    # =========================================================================
    
    def create_escalation(self, issue_id: str, level: EscalationLevel, trigger: EscalationTrigger,
                          escalated_by: str, escalated_by_name: str = "", rule_id: Optional[str] = None,
                          issue_title: str = "", issue_priority: str = "", issue_category: str = "",
                          issue_age_hours: float = 0, study_id: Optional[str] = None,
                          site_id: Optional[str] = None, patient_id: Optional[str] = None,
                          escalated_to: List[str] = None, escalated_to_roles: List[str] = None,
                          escalation_reason: str = "") -> Escalation:
        escalation_id = self._generate_id("ESC")
        now = datetime.now()
        escalation = Escalation(
            escalation_id=escalation_id, issue_id=issue_id, level=level,
            status=EscalationStatus.PENDING, trigger=trigger, rule_id=rule_id,
            issue_title=issue_title, issue_priority=issue_priority, issue_category=issue_category,
            issue_age_hours=issue_age_hours, study_id=study_id, site_id=site_id, patient_id=patient_id,
            escalated_to=escalated_to or [], escalated_to_roles=escalated_to_roles or [],
            escalated_by=escalated_by, escalated_by_name=escalated_by_name, escalated_at=now,
            escalation_reason=escalation_reason, created_at=now, updated_at=now)
        self._save_escalation(escalation)
        self._log_escalation_history(escalation_id, None, level, None, EscalationStatus.PENDING,
                                     escalated_by, escalated_by_name, escalation_reason or f"Triggered by {trigger.value}")
        self._log_audit(escalation_id, "escalation_created", escalated_by, escalated_by_name,
                        {'level': level.value, 'trigger': trigger.value, 'issue_id': issue_id})
        logger.info(f"Created escalation {escalation_id} for issue {issue_id} at {level.display_name}")
        return escalation
    
    def _save_escalation(self, escalation: Escalation):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO escalations
                (escalation_id, issue_id, level, status, trigger, rule_id, issue_title,
                 issue_priority, issue_category, issue_age_hours, study_id, site_id, patient_id,
                 escalated_to, escalated_to_roles, escalated_by, escalated_by_name, escalated_at,
                 acknowledged_by, acknowledged_at, resolved_by, resolved_at, impact_score,
                 cascade_patients, cascade_issues, dqi_impact, investigation_room_id,
                 notifications_sent, escalation_reason, resolution_notes, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (escalation.escalation_id, escalation.issue_id, escalation.level.value,
                  escalation.status.value, escalation.trigger.value, escalation.rule_id,
                  escalation.issue_title, escalation.issue_priority, escalation.issue_category,
                  escalation.issue_age_hours, escalation.study_id, escalation.site_id, escalation.patient_id,
                  json.dumps(escalation.escalated_to), json.dumps(escalation.escalated_to_roles),
                  escalation.escalated_by, escalation.escalated_by_name,
                  escalation.escalated_at.isoformat() if escalation.escalated_at else None,
                  escalation.acknowledged_by,
                  escalation.acknowledged_at.isoformat() if escalation.acknowledged_at else None,
                  escalation.resolved_by,
                  escalation.resolved_at.isoformat() if escalation.resolved_at else None,
                  escalation.impact_score, escalation.cascade_patients, escalation.cascade_issues,
                  escalation.dqi_impact, escalation.investigation_room_id,
                  json.dumps(escalation.notifications_sent), escalation.escalation_reason,
                  escalation.resolution_notes,
                  escalation.created_at.isoformat() if escalation.created_at else None,
                  escalation.updated_at.isoformat() if escalation.updated_at else None))
    
    def _log_escalation_history(self, escalation_id: str, from_level: Optional[EscalationLevel],
                                to_level: EscalationLevel, from_status: Optional[EscalationStatus],
                                to_status: EscalationStatus, changed_by: str,
                                changed_by_name: str = "", change_reason: str = ""):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            history_id = f"HIST-{uuid.uuid4().hex[:12].upper()}"
            cursor.execute("""
                INSERT INTO escalation_history
                (history_id, escalation_id, from_level, to_level, from_status, to_status,
                 changed_by, changed_by_name, change_reason, changed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (history_id, escalation_id, from_level.value if from_level else None, to_level.value,
                  from_status.value if from_status else None, to_status.value,
                  changed_by, changed_by_name, change_reason, datetime.now().isoformat()))
    
    def get_escalation(self, escalation_id: str) -> Optional[Escalation]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM escalations WHERE escalation_id = ?", (escalation_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_escalation(row, cursor.description)
        return None
    
    def get_escalations_for_issue(self, issue_id: str) -> List[Escalation]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM escalations WHERE issue_id = ? ORDER BY escalated_at DESC", (issue_id,))
            rows = cursor.fetchall()
            return [self._row_to_escalation(row, cursor.description) for row in rows]
    
    def get_active_escalation(self, issue_id: str) -> Optional[Escalation]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM escalations 
                WHERE issue_id = ? AND status IN ('pending', 'acknowledged', 'in_progress')
                ORDER BY level DESC, escalated_at DESC LIMIT 1
            """, (issue_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_escalation(row, cursor.description)
        return None
    
    def search_escalations(self, filter_criteria: EscalationFilter, limit: int = 100,
                           offset: int = 0, sort_by: str = "escalated_at",
                           sort_order: str = "DESC") -> Tuple[List[Escalation], int]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            conditions, params = [], []
            
            if filter_criteria.issue_ids:
                conditions.append(f"issue_id IN ({','.join(['?' for _ in filter_criteria.issue_ids])})")
                params.extend(filter_criteria.issue_ids)
            if filter_criteria.levels:
                conditions.append(f"level IN ({','.join(['?' for _ in filter_criteria.levels])})")
                params.extend([l.value for l in filter_criteria.levels])
            if filter_criteria.statuses:
                conditions.append(f"status IN ({','.join(['?' for _ in filter_criteria.statuses])})")
                params.extend([s.value for s in filter_criteria.statuses])
            if filter_criteria.study_id:
                conditions.append("study_id = ?")
                params.append(filter_criteria.study_id)
            if filter_criteria.is_active is not None:
                if filter_criteria.is_active:
                    conditions.append("status IN ('pending', 'acknowledged', 'in_progress')")
                else:
                    conditions.append("status NOT IN ('pending', 'acknowledged', 'in_progress')")
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            cursor.execute(f"SELECT COUNT(*) FROM escalations WHERE {where_clause}", params)
            total = cursor.fetchone()[0]
            cursor.execute(f"SELECT * FROM escalations WHERE {where_clause} ORDER BY {sort_by} {sort_order} LIMIT ? OFFSET ?",
                           params + [limit, offset])
            rows = cursor.fetchall()
            return [self._row_to_escalation(row, cursor.description) for row in rows], total
    
    def _row_to_escalation(self, row, columns) -> Escalation:
        data = {col[0]: val for col, val in zip(columns, row)}
        return Escalation(
            escalation_id=data['escalation_id'], issue_id=data['issue_id'],
            level=EscalationLevel(data['level']), status=EscalationStatus(data['status']),
            trigger=EscalationTrigger(data['trigger']), rule_id=data.get('rule_id'),
            issue_title=data.get('issue_title', ''), issue_priority=data.get('issue_priority', ''),
            issue_category=data.get('issue_category', ''), issue_age_hours=data.get('issue_age_hours', 0) or 0,
            study_id=data.get('study_id'), site_id=data.get('site_id'), patient_id=data.get('patient_id'),
            escalated_to=json.loads(data.get('escalated_to', '[]') or '[]'),
            escalated_to_roles=json.loads(data.get('escalated_to_roles', '[]') or '[]'),
            escalated_by=data.get('escalated_by', ''), escalated_by_name=data.get('escalated_by_name', ''),
            escalated_at=datetime.fromisoformat(data['escalated_at']) if data.get('escalated_at') else datetime.now(),
            acknowledged_by=data.get('acknowledged_by'),
            acknowledged_at=datetime.fromisoformat(data['acknowledged_at']) if data.get('acknowledged_at') else None,
            resolved_by=data.get('resolved_by'),
            resolved_at=datetime.fromisoformat(data['resolved_at']) if data.get('resolved_at') else None,
            impact_score=data.get('impact_score', 0) or 0, cascade_patients=data.get('cascade_patients', 0) or 0,
            cascade_issues=data.get('cascade_issues', 0) or 0, dqi_impact=data.get('dqi_impact', 0) or 0,
            investigation_room_id=data.get('investigation_room_id'),
            notifications_sent=json.loads(data.get('notifications_sent', '[]') or '[]'),
            escalation_reason=data.get('escalation_reason', ''), resolution_notes=data.get('resolution_notes', ''),
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else datetime.now(),
            updated_at=datetime.fromisoformat(data['updated_at']) if data.get('updated_at') else datetime.now())
    
    # =========================================================================
    # ESCALATION ACTIONS
    # =========================================================================
    
    def acknowledge_escalation(self, escalation_id: str, acknowledged_by: str,
                               acknowledged_by_name: str = "") -> Optional[Escalation]:
        escalation = self.get_escalation(escalation_id)
        if not escalation or escalation.status != EscalationStatus.PENDING:
            return escalation
        old_status = escalation.status
        escalation.status = EscalationStatus.ACKNOWLEDGED
        escalation.acknowledged_by = acknowledged_by
        escalation.acknowledged_at = datetime.now()
        escalation.updated_at = datetime.now()
        self._save_escalation(escalation)
        self._log_escalation_history(escalation_id, escalation.level, escalation.level,
                                     old_status, EscalationStatus.ACKNOWLEDGED,
                                     acknowledged_by, acknowledged_by_name, "Escalation acknowledged")
        self._log_audit(escalation_id, "acknowledged", acknowledged_by, acknowledged_by_name)
        logger.info(f"Escalation {escalation_id} acknowledged by {acknowledged_by}")
        return escalation
    
    def start_work_on_escalation(self, escalation_id: str, started_by: str,
                                 started_by_name: str = "") -> Optional[Escalation]:
        escalation = self.get_escalation(escalation_id)
        if not escalation or escalation.status not in [EscalationStatus.PENDING, EscalationStatus.ACKNOWLEDGED]:
            return escalation
        old_status = escalation.status
        escalation.status = EscalationStatus.IN_PROGRESS
        escalation.updated_at = datetime.now()
        if not escalation.acknowledged_by:
            escalation.acknowledged_by = started_by
            escalation.acknowledged_at = datetime.now()
        self._save_escalation(escalation)
        self._log_escalation_history(escalation_id, escalation.level, escalation.level,
                                     old_status, EscalationStatus.IN_PROGRESS,
                                     started_by, started_by_name, "Work started on escalation")
        self._log_audit(escalation_id, "work_started", started_by, started_by_name)
        logger.info(f"Work started on escalation {escalation_id} by {started_by}")
        return escalation
    
    def resolve_escalation(self, escalation_id: str, resolved_by: str, resolved_by_name: str = "",
                           resolution_notes: str = "") -> Optional[Escalation]:
        escalation = self.get_escalation(escalation_id)
        if not escalation or not escalation.is_active:
            return escalation
        old_status = escalation.status
        escalation.status = EscalationStatus.RESOLVED
        escalation.resolved_by = resolved_by
        escalation.resolved_at = datetime.now()
        escalation.resolution_notes = resolution_notes
        escalation.updated_at = datetime.now()
        self._save_escalation(escalation)
        self._log_escalation_history(escalation_id, escalation.level, escalation.level,
                                     old_status, EscalationStatus.RESOLVED,
                                     resolved_by, resolved_by_name, resolution_notes or "Escalation resolved")
        self._log_audit(escalation_id, "resolved", resolved_by, resolved_by_name, {'resolution_notes': resolution_notes})
        logger.info(f"Escalation {escalation_id} resolved by {resolved_by}")
        return escalation
    
    def escalate_to_next_level(self, escalation_id: str, escalated_by: str,
                               escalated_by_name: str = "", reason: str = "") -> Optional[Escalation]:
        escalation = self.get_escalation(escalation_id)
        if not escalation or escalation.level == EscalationLevel.L5_SPONSOR:
            return escalation
        old_level, old_status = escalation.level, escalation.status
        new_level = EscalationLevel(escalation.level.value + 1)
        escalation.level = new_level
        escalation.status = EscalationStatus.PENDING
        escalation.updated_at = datetime.now()
        rules = self.get_rules_for_level(new_level)
        if rules:
            escalation.escalated_to_roles = rules[0].notify_roles
            escalation.rule_id = rules[0].rule_id
        self._save_escalation(escalation)
        self._log_escalation_history(escalation_id, old_level, new_level, old_status, EscalationStatus.PENDING,
                                     escalated_by, escalated_by_name,
                                     reason or f"Escalated from {old_level.display_name} to {new_level.display_name}")
        self._log_audit(escalation_id, "level_increased", escalated_by, escalated_by_name,
                        {'from_level': old_level.value, 'to_level': new_level.value})
        logger.info(f"Escalation {escalation_id} escalated from {old_level.display_name} to {new_level.display_name}")
        return escalation
    
    def de_escalate(self, escalation_id: str, de_escalated_by: str,
                    de_escalated_by_name: str = "", reason: str = "") -> Optional[Escalation]:
        escalation = self.get_escalation(escalation_id)
        if not escalation or escalation.level == EscalationLevel.L1_REMINDER:
            return escalation
        old_level, old_status = escalation.level, escalation.status
        new_level = EscalationLevel(escalation.level.value - 1)
        escalation.level = new_level
        escalation.status = EscalationStatus.DE_ESCALATED
        escalation.updated_at = datetime.now()
        self._save_escalation(escalation)
        self._log_escalation_history(escalation_id, old_level, new_level, old_status, EscalationStatus.DE_ESCALATED,
                                     de_escalated_by, de_escalated_by_name,
                                     reason or f"De-escalated from {old_level.display_name} to {new_level.display_name}")
        self._log_audit(escalation_id, "de_escalated", de_escalated_by, de_escalated_by_name,
                        {'from_level': old_level.value, 'to_level': new_level.value})
        logger.info(f"Escalation {escalation_id} de-escalated from {old_level.display_name} to {new_level.display_name}")
        return escalation
    
    def cancel_escalation(self, escalation_id: str, cancelled_by: str,
                          cancelled_by_name: str = "", reason: str = "") -> Optional[Escalation]:
        escalation = self.get_escalation(escalation_id)
        if not escalation:
            return None
        old_status = escalation.status
        escalation.status = EscalationStatus.CANCELLED
        escalation.resolution_notes = reason
        escalation.updated_at = datetime.now()
        self._save_escalation(escalation)
        self._log_escalation_history(escalation_id, escalation.level, escalation.level,
                                     old_status, EscalationStatus.CANCELLED,
                                     cancelled_by, cancelled_by_name, reason or "Escalation cancelled")
        self._log_audit(escalation_id, "cancelled", cancelled_by, cancelled_by_name, {'reason': reason})
        logger.info(f"Escalation {escalation_id} cancelled by {cancelled_by}")
        return escalation
    
    # =========================================================================
    # AUTO-ESCALATION
    # =========================================================================
    
    def check_and_escalate_issues(self, issues: List[Dict], dry_run: bool = False) -> List[Escalation]:
        escalations = []
        for issue in issues:
            escalations.extend(self._check_issue_for_escalation(issue, dry_run))
        return escalations
    
    def _check_issue_for_escalation(self, issue: Dict, dry_run: bool = False) -> List[Escalation]:
        escalations = []
        issue_id = issue.get('issue_id')
        priority = issue.get('priority', '').lower()
        category = issue.get('category', '').lower()
        status = issue.get('status', '').lower()
        
        created_at = issue.get('created_at')
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            except:
                created_at = datetime.now()
        elif not isinstance(created_at, datetime):
            created_at = datetime.now()
        if created_at.tzinfo:
            created_at = created_at.replace(tzinfo=None)
        age_hours = (datetime.now() - created_at).total_seconds() / 3600
        
        existing = self.get_escalations_for_issue(issue_id)
        highest_level = max([e.level.value for e in existing], default=0)
        
        for rule in self.get_active_rules():
            if rule.level.value <= highest_level:
                continue
            if not self._rule_applies(rule, priority, category, status, age_hours):
                continue
            
            if not dry_run:
                esc = self.create_escalation(
                    issue_id=issue_id, level=rule.level, trigger=rule.trigger_type,
                    escalated_by="system", escalated_by_name="Escalation Engine",
                    rule_id=rule.rule_id, issue_title=issue.get('title', ''),
                    issue_priority=priority, issue_category=category, issue_age_hours=age_hours,
                    study_id=issue.get('study_id'), site_id=issue.get('site_id'),
                    escalated_to_roles=rule.notify_roles, escalation_reason=f"Auto: {rule.description}")
                escalations.append(esc)
            else:
                mock = Escalation(
                    escalation_id=f"DRY-{issue_id}-{rule.level.value}-{uuid.uuid4().hex[:4]}",
                    issue_id=issue_id, level=rule.level, status=EscalationStatus.PENDING,
                    trigger=rule.trigger_type, rule_id=rule.rule_id,
                    issue_title=issue.get('title', ''), issue_priority=priority,
                    issue_category=category, issue_age_hours=age_hours, escalated_by="system")
                escalations.append(mock)
        return escalations
    
    def _rule_applies(self, rule: EscalationRule, priority: str, category: str,
                      status: str, age_hours: float) -> bool:
        if rule.applies_to_priorities and priority not in rule.applies_to_priorities:
            return False
        if rule.applies_to_categories and category not in rule.applies_to_categories:
            return False
        if rule.applies_to_statuses and status not in rule.applies_to_statuses:
            return False
        if rule.trigger_type == EscalationTrigger.TIME_BASED and age_hours < rule.trigger_hours:
            return False
        return True
    
    # =========================================================================
    # IMPACT ANALYSIS
    # =========================================================================
    
    def analyze_impact(self, issue_id: str, level: EscalationLevel) -> EscalationImpact:
        impact = EscalationImpact(issue_id=issue_id, escalation_level=level)
        m = level.value
        impact.patients_affected = 10 * m + (m * 5)
        impact.sites_affected = max(1, m)
        impact.cascade_issues = m * 3
        impact.cascade_patients = impact.patients_affected * 2
        impact.blocked_db_lock = int(impact.cascade_patients * 0.3)
        impact.current_dqi = 85 - (m * 3)
        impact.projected_dqi_loss = m * 2
        impact.days_delayed = m * 3
        impact.sla_breach_risk = min(1.0, m * 0.2)
        impact.estimated_cost = impact.days_delayed * 5000
        if m >= 4:
            impact.recommendations.extend(["Immediate resource allocation required", "Consider parallel remediation strategies"])
        if m >= 3:
            impact.recommendations.extend(["Root cause investigation needed", "Escalate to cross-functional team"])
        if m >= 2:
            impact.recommendations.append("Supervisor review required")
        impact.recommendations.append("Monitor closely and provide daily updates")
        impact.calculate_severity()
        self._save_impact_analysis(impact)
        return impact
    
    def _save_impact_analysis(self, impact: EscalationImpact):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO impact_analyses
                (analysis_id, escalation_id, issue_id, level, patients_affected, sites_affected,
                 cascade_issues, cascade_patients, blocked_db_lock, current_dqi, projected_dqi_loss,
                 days_delayed, sla_breach_risk, estimated_cost, severity_score, severity_label,
                 recommendations, analyzed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (self._generate_id("IMP"), None, impact.issue_id, impact.escalation_level.value,
                  impact.patients_affected, impact.sites_affected, impact.cascade_issues,
                  impact.cascade_patients, impact.blocked_db_lock, impact.current_dqi,
                  impact.projected_dqi_loss, impact.days_delayed, impact.sla_breach_risk,
                  impact.estimated_cost, impact.severity_score, impact.severity_label,
                  json.dumps(impact.recommendations), datetime.now().isoformat()))
    
    # =========================================================================
    # SPONSOR PACKAGE
    # =========================================================================
    
    def generate_sponsor_package(self, escalation_id: str, issue: Dict = None) -> SponsorPackage:
        escalation = self.get_escalation(escalation_id)
        if not escalation:
            raise ValueError(f"Escalation {escalation_id} not found")
        
        package_id = self._generate_id("PKG")
        history = self.get_escalation_history(escalation_id)
        impact = self.analyze_impact(escalation.issue_id, escalation.level)
        
        package = SponsorPackage(
            package_id=package_id, escalation_id=escalation_id, issue_id=escalation.issue_id,
            executive_summary=self._generate_executive_summary(escalation, impact),
            issue_title=escalation.issue_title or (issue.get('title', '') if issue else ''),
            issue_description=issue.get('description', '') if issue else '',
            issue_category=escalation.issue_category, issue_priority=escalation.issue_priority,
            issue_age_days=escalation.issue_age_hours / 24,
            escalation_history=[{'level': h['to_level'], 'status': h['to_status'],
                                 'date': h['changed_at'], 'reason': h['change_reason']} for h in history],
            impact=impact, root_cause="Under investigation",
            actions_taken=[{'action': 'Escalation created', 'date': escalation.created_at.isoformat()}],
            recommendations=impact.recommendations,
            decisions_required=["Approve additional resources", "Authorize remediation", "Confirm communication"],
            proposed_resolution_date=datetime.now() + timedelta(days=14),
            generated_at=datetime.now(), generated_by="system")
        
        self._save_sponsor_package(package)
        self._log_audit(escalation_id, "sponsor_package_generated", "system", details={'package_id': package_id})
        return package
    
    def _generate_executive_summary(self, escalation: Escalation, impact: EscalationImpact) -> str:
        return f"""EXECUTIVE SUMMARY
Issue: {escalation.issue_title or escalation.issue_id}
Priority: {(escalation.issue_priority or 'N/A').upper()}
Category: {escalation.issue_category or 'N/A'}
Age: {escalation.issue_age_hours / 24:.1f} days
Level: {escalation.level.display_name}

IMPACT: {impact.severity_label} ({impact.severity_score:.0f}/100)
- Patients: {impact.patients_affected:,}
- Cascade: {impact.cascade_patients:,}
- Delay: {impact.days_delayed:.0f} days
- Cost: ${impact.estimated_cost:,.0f}"""
    
    def _save_sponsor_package(self, package: SponsorPackage):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            impact_data = None
            if package.impact:
                impact_data = json.dumps({
                    'issue_id': package.impact.issue_id,
                    'escalation_level': package.impact.escalation_level.value,
                    'patients_affected': package.impact.patients_affected,
                    'sites_affected': package.impact.sites_affected,
                    'cascade_issues': package.impact.cascade_issues,
                    'cascade_patients': package.impact.cascade_patients,
                    'blocked_db_lock': package.impact.blocked_db_lock,
                    'current_dqi': package.impact.current_dqi,
                    'projected_dqi_loss': package.impact.projected_dqi_loss,
                    'days_delayed': package.impact.days_delayed,
                    'sla_breach_risk': package.impact.sla_breach_risk,
                    'estimated_cost': package.impact.estimated_cost,
                    'recommendations': package.impact.recommendations,
                    'severity_score': package.impact.severity_score,
                    'severity_label': package.impact.severity_label
                })
            cursor.execute("""
                INSERT INTO sponsor_packages
                (package_id, escalation_id, issue_id, executive_summary, issue_title,
                 issue_description, issue_category, issue_priority, issue_age_days,
                 escalation_history, impact_data, root_cause, root_cause_confidence,
                 actions_taken, recommendations, decisions_required,
                 proposed_resolution_date, generated_at, generated_by)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (package.package_id, package.escalation_id, package.issue_id,
                  package.executive_summary, package.issue_title, package.issue_description,
                  package.issue_category, package.issue_priority, package.issue_age_days,
                  json.dumps(package.escalation_history), impact_data, package.root_cause,
                  package.root_cause_confidence, json.dumps(package.actions_taken),
                  json.dumps(package.recommendations), json.dumps(package.decisions_required),
                  package.proposed_resolution_date.isoformat() if package.proposed_resolution_date else None,
                  package.generated_at.isoformat() if package.generated_at else None, package.generated_by))
    
    # =========================================================================
    # HISTORY & STATISTICS
    # =========================================================================
    
    def get_escalation_history(self, escalation_id: str) -> List[Dict]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM escalation_history WHERE escalation_id = ? ORDER BY changed_at ASC",
                           (escalation_id,))
            rows = cursor.fetchall()
            columns = [col[0] for col in cursor.description]
            return [dict(zip(columns, row)) for row in rows]
    
    def get_statistics(self) -> Dict[str, Any]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            stats = {}
            cursor.execute("SELECT COUNT(*) FROM escalations")
            stats['total_escalations'] = cursor.fetchone()[0]
            cursor.execute("SELECT status, COUNT(*) FROM escalations GROUP BY status")
            stats['by_status'] = {row[0]: row[1] for row in cursor.fetchall()}
            cursor.execute("SELECT level, COUNT(*) FROM escalations GROUP BY level")
            stats['by_level'] = {f"L{row[0]}": row[1] for row in cursor.fetchall()}
            cursor.execute("SELECT COUNT(*) FROM escalations WHERE status IN ('pending', 'acknowledged', 'in_progress')")
            stats['active_escalations'] = cursor.fetchone()[0]
            cursor.execute("SELECT trigger, COUNT(*) FROM escalations GROUP BY trigger")
            stats['by_trigger'] = {row[0]: row[1] for row in cursor.fetchall()}
            cursor.execute("SELECT AVG((julianday(resolved_at) - julianday(escalated_at)) * 24) FROM escalations WHERE resolved_at IS NOT NULL")
            result = cursor.fetchone()[0]
            stats['avg_resolution_hours'] = round(result, 1) if result else 0
            yesterday = (datetime.now() - timedelta(hours=24)).isoformat()
            cursor.execute("SELECT COUNT(*) FROM escalations WHERE escalated_at > ?", (yesterday,))
            stats['last_24_hours'] = cursor.fetchone()[0]
            last_week = (datetime.now() - timedelta(days=7)).isoformat()
            cursor.execute("SELECT COUNT(*) FROM escalations WHERE escalated_at > ?", (last_week,))
            stats['last_7_days'] = cursor.fetchone()[0]
            stats['rules_count'] = len(self.rules)
            stats['active_rules'] = len(self.get_active_rules())
            return stats
    
    def get_audit_trail(self, escalation_id: str) -> List[Dict]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM escalation_audit WHERE escalation_id = ? ORDER BY created_at ASC",
                           (escalation_id,))
            rows = cursor.fetchall()
            columns = [col[0] for col in cursor.description]
            return [dict(zip(columns, row)) for row in rows]


# =============================================================================
# SINGLETON & CONVENIENCE
# =============================================================================

_escalation_engine_instance = None

def get_escalation_engine() -> EscalationEngine:
    global _escalation_engine_instance
    if _escalation_engine_instance is None:
        _escalation_engine_instance = EscalationEngine()
    return _escalation_engine_instance

def reset_escalation_engine():
    global _escalation_engine_instance
    _escalation_engine_instance = None

def check_escalations(issues: List[Dict], dry_run: bool = False) -> List[Escalation]:
    return get_escalation_engine().check_and_escalate_issues(issues, dry_run)

def get_active_escalations(issue_id: Optional[str] = None) -> List[Escalation]:
    engine = get_escalation_engine()
    if issue_id:
        esc = engine.get_active_escalation(issue_id)
        return [esc] if esc else []
    escalations, _ = engine.search_escalations(EscalationFilter(is_active=True), limit=1000)
    return escalations

def escalate_issue(issue_id: str, level: EscalationLevel, escalated_by: str, reason: str = "") -> Escalation:
    return get_escalation_engine().create_escalation(
        issue_id=issue_id, level=level, trigger=EscalationTrigger.MANUAL,
        escalated_by=escalated_by, escalation_reason=reason)

def get_escalation_stats() -> Dict[str, Any]:
    return get_escalation_engine().get_statistics()


# =============================================================================
# TEST
# =============================================================================

def test_escalation_engine():
    print("=" * 70)
    print("SANCHALAK AI - ESCALATION ENGINE TEST")
    print("=" * 70)
    
    reset_escalation_engine()
    import os
    db_path = Path("data/collaboration/escalation_engine.db")
    for p in [db_path, Path(str(db_path) + "-wal"), Path(str(db_path) + "-shm")]:
        if p.exists():
            try: os.remove(p)
            except: pass
    
    engine = get_escalation_engine()
    passed = failed = 0
    test_id = None
    
    # Test 1
    print("\n" + "-" * 70 + "\nTEST 1: Engine Initialization\n" + "-" * 70)
    try:
        assert engine and len(engine.rules) > 0
        print(f"   ✅ Engine initialized with {len(engine.rules)} rules")
        passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        failed += 1
    
    # Test 2
    print("\n" + "-" * 70 + "\nTEST 2: Default Rules\n" + "-" * 70)
    try:
        for level in EscalationLevel:
            print(f"   {level.display_name}: {len(engine.get_rules_for_level(level))} rules")
        print(f"   ✅ Total active rules: {len(engine.get_active_rules())}")
        passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        failed += 1
    
    # Test 3
    print("\n" + "-" * 70 + "\nTEST 3: Create Escalation\n" + "-" * 70)
    try:
        esc = engine.create_escalation(issue_id="ISS-TEST-001", level=EscalationLevel.L1_REMINDER,
                                        trigger=EscalationTrigger.MANUAL, escalated_by="test_user",
                                        escalated_by_name="Test User", issue_title="Test Issue",
                                        issue_priority="high", issue_category="sdv_incomplete",
                                        study_id="Study_21", site_id="Site_101")
        assert esc and esc.status == EscalationStatus.PENDING
        test_id = esc.escalation_id
        print(f"   ✅ Created: {test_id}")
        passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        failed += 1
    
    # Test 4
    print("\n" + "-" * 70 + "\nTEST 4: Get Escalation\n" + "-" * 70)
    try:
        assert engine.get_escalation(test_id)
        print(f"   ✅ Retrieved: {test_id}")
        passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        failed += 1
    
    # Test 5
    print("\n" + "-" * 70 + "\nTEST 5: Acknowledge Escalation\n" + "-" * 70)
    try:
        ack = engine.acknowledge_escalation(test_id, "supervisor_001", "Supervisor")
        assert ack.status == EscalationStatus.ACKNOWLEDGED
        print(f"   ✅ Acknowledged by {ack.acknowledged_by}")
        passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        failed += 1
    
    # Test 6
    print("\n" + "-" * 70 + "\nTEST 6: Start Work\n" + "-" * 70)
    try:
        ip = engine.start_work_on_escalation(test_id, "cra_001", "Sarah Chen")
        assert ip.status == EscalationStatus.IN_PROGRESS
        print(f"   ✅ Status: {ip.status.value}")
        passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        failed += 1
    
    # Test 7
    print("\n" + "-" * 70 + "\nTEST 7: Escalate to Next Level\n" + "-" * 70)
    try:
        up = engine.escalate_to_next_level(test_id, "supervisor_001", "Supervisor", "Not resolved")
        assert up.level == EscalationLevel.L2_SUPERVISOR
        print(f"   ✅ Escalated to: {up.level.display_name}")
        passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        failed += 1
    
    # Test 8
    print("\n" + "-" * 70 + "\nTEST 8: Impact Analysis\n" + "-" * 70)
    try:
        impact = engine.analyze_impact("ISS-TEST-001", EscalationLevel.L3_INVESTIGATION)
        assert impact.severity_label
        print(f"   ✅ Severity: {impact.severity_label} ({impact.severity_score:.0f}/100)")
        passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        failed += 1
    
    # Test 9
    print("\n" + "-" * 70 + "\nTEST 9: Sponsor Package\n" + "-" * 70)
    try:
        pkg = engine.generate_sponsor_package(test_id, {'title': 'Test'})
        assert pkg.package_id.startswith("PKG-")
        print(f"   ✅ Package: {pkg.package_id}")
        passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        failed += 1
    
    # Test 10
    print("\n" + "-" * 70 + "\nTEST 10: Resolve Escalation\n" + "-" * 70)
    try:
        res = engine.resolve_escalation(test_id, "cra_001", "Sarah Chen", "Fixed")
        assert res.status == EscalationStatus.RESOLVED
        print(f"   ✅ Resolved by {res.resolved_by}")
        passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        failed += 1
    
    # Test 11
    print("\n" + "-" * 70 + "\nTEST 11: Escalation History\n" + "-" * 70)
    try:
        history = engine.get_escalation_history(test_id)
        assert len(history) >= 1
        print(f"   ✅ History entries: {len(history)}")
        passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        failed += 1
    
    # Test 12
    print("\n" + "-" * 70 + "\nTEST 12: Auto-Escalation (Dry Run)\n" + "-" * 70)
    try:
        issues = [{'issue_id': 'ISS-AUTO-001', 'priority': 'critical', 'category': 'sae_dm_pending',
                   'status': 'open', 'created_at': (datetime.now() - timedelta(hours=48)).isoformat()}]
        would = engine.check_and_escalate_issues(issues, dry_run=True)
        print(f"   ✅ Would escalate: {len(would)}")
        passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        failed += 1
    
    # Test 13
    print("\n" + "-" * 70 + "\nTEST 13: Search Escalations\n" + "-" * 70)
    try:
        res_list, total = engine.search_escalations(EscalationFilter(statuses=[EscalationStatus.RESOLVED]))
        print(f"   ✅ Resolved: {total}")
        passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        failed += 1
    
    # Test 14
    print("\n" + "-" * 70 + "\nTEST 14: De-escalation\n" + "-" * 70)
    try:
        l3 = engine.create_escalation("ISS-DEESC-001", EscalationLevel.L3_INVESTIGATION,
                                       EscalationTrigger.MANUAL, "test_user")
        de = engine.de_escalate(l3.escalation_id, "lead_001", "Study Lead", "Less severe")
        assert de.level == EscalationLevel.L2_SUPERVISOR
        print(f"   ✅ De-escalated to {de.level.display_name}")
        passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        failed += 1
    
    # Test 15
    print("\n" + "-" * 70 + "\nTEST 15: Statistics\n" + "-" * 70)
    try:
        stats = engine.get_statistics()
        assert 'total_escalations' in stats
        print(f"   ✅ Total: {stats['total_escalations']}, Active: {stats['active_escalations']}")
        passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        failed += 1
    
    # Test 16
    print("\n" + "-" * 70 + "\nTEST 16: Audit Trail\n" + "-" * 70)
    try:
        audit = engine.get_audit_trail(test_id)
        assert len(audit) >= 1
        print(f"   ✅ Audit entries: {len(audit)}")
        passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        failed += 1
    
    # Test 17
    print("\n" + "-" * 70 + "\nTEST 17: Cancel Escalation\n" + "-" * 70)
    try:
        cancel = engine.create_escalation("ISS-CANCEL-001", EscalationLevel.L1_REMINDER,
                                          EscalationTrigger.MANUAL, "test_user")
        cancelled = engine.cancel_escalation(cancel.escalation_id, "lead_001", "Study Lead", "Duplicate")
        assert cancelled.status == EscalationStatus.CANCELLED
        print(f"   ✅ Cancelled: {cancelled.escalation_id}")
        passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        failed += 1
    
    # Test 18
    print("\n" + "-" * 70 + "\nTEST 18: Add Custom Rule\n" + "-" * 70)
    try:
        rule = EscalationRule(rule_id="CUSTOM-TEST-001", level=EscalationLevel.L2_SUPERVISOR,
                              trigger_type=EscalationTrigger.TIME_BASED, trigger_hours=12,
                              applies_to_priorities=["critical"], applies_to_categories=["sae_dm_pending"])
        engine.add_rule(rule)
        assert engine.get_rule("CUSTOM-TEST-001").trigger_hours == 12
        print(f"   ✅ Custom rule added: {rule.rule_id}")
        passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        failed += 1
    
    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)
    print("✅ ALL TESTS PASSED!" if failed == 0 else f"❌ {failed} test(s) failed")
    return failed == 0


if __name__ == "__main__":
    test_escalation_engine()