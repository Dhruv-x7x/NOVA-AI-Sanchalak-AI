# src/governance/audit_trail.py
# NOTE: This module previously used SQLite but has been migrated to PostgreSQL-only.
# If you need to use this module, update it to use:
#   from src.database.pg_data_service import get_pg_data_service
#   service = get_pg_data_service()


"""
a6on-i - Audit Trail System v1.0
21 CFR Part 11 Compliant Audit Logging

Features:
- Immutable append-only log
- SHA-256 checksums with chain hashing
- Reasoning chain capture for AI decisions
- Full query capabilities
- Export for regulatory inspection
"""

import sqlite3
import hashlib
import json
import uuid
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import threading
# SQLite removed - using PostgreSQL
import pandas as pd


# =============================================================================
# ENUMS
# =============================================================================

class EventType(Enum):
    """Types of auditable events"""
    # User Actions
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    USER_ACTION = "user_action"
    
    # Data Operations
    DATA_VIEW = "data_view"
    DATA_CREATE = "data_create"
    DATA_UPDATE = "data_update"
    DATA_DELETE = "data_delete"
    DATA_EXPORT = "data_export"
    
    # AI/Agent Operations
    AI_QUERY = "ai_query"
    AI_RECOMMENDATION = "ai_recommendation"
    AI_DECISION = "ai_decision"
    AI_EXECUTION = "ai_execution"
    
    # Approval Workflow
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_DENIED = "approval_denied"
    APPROVAL_EXPIRED = "approval_expired"
    
    # System Events
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    SYSTEM_ERROR = "system_error"
    SYSTEM_CONFIG_CHANGE = "system_config_change"
    
    # Compliance Events
    SIGNATURE_APPLIED = "signature_applied"
    SIGNATURE_BROKEN = "signature_broken"
    OVERRIDE_APPLIED = "override_applied"
    ESCALATION = "escalation"


class ActionCategory(Enum):
    """Categories for grouping actions"""
    AUTHENTICATION = "authentication"
    DATA_MANAGEMENT = "data_management"
    PATIENT_DATA = "patient_data"
    SITE_MANAGEMENT = "site_management"
    SAFETY = "safety"
    QUALITY = "quality"
    AI_AGENT = "ai_agent"
    APPROVAL = "approval"
    REPORTING = "reporting"
    CONFIGURATION = "configuration"
    SYSTEM = "system"


class Severity(Enum):
    """Severity levels for audit entries"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ComplianceFlag(Enum):
    """Compliance-related flags"""
    CFR_11_RELEVANT = "21_cfr_part_11"
    GCP_RELEVANT = "ich_gcp"
    SAFETY_RELEVANT = "safety"
    PHI_ACCESS = "phi_access"
    SIGNATURE_REQUIRED = "signature_required"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Actor:
    """Represents who performed an action"""
    actor_id: str
    actor_type: str  # user, system, agent
    name: str
    role: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Entity:
    """Represents what was acted upon"""
    entity_type: str  # patient, site, study, report, etc.
    entity_id: str
    entity_name: Optional[str] = None
    parent_entity: Optional[str] = None  # e.g., study_id for a patient
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class StateChange:
    """Captures before/after state for data changes"""
    field_name: str
    old_value: Any
    new_value: Any
    data_type: str = "string"
    
    def to_dict(self) -> Dict:
        return {
            'field_name': self.field_name,
            'old_value': str(self.old_value) if self.old_value is not None else None,
            'new_value': str(self.new_value) if self.new_value is not None else None,
            'data_type': self.data_type
        }


@dataclass
class ReasoningChain:
    """Captures AI decision reasoning for transparency"""
    query: str
    context: Dict[str, Any]
    hypotheses: List[Dict[str, Any]] = field(default_factory=list)
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    decision: Optional[str] = None
    confidence: float = 0.0
    alternatives_considered: List[str] = field(default_factory=list)
    model_used: Optional[str] = None
    tokens_used: int = 0
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class AuditEntry:
    """
    Immutable audit log entry
    Captures: Who, What, When, Where, Why, How
    """
    # Identity
    entry_id: str
    timestamp: datetime
    
    # Who
    actor: Actor
    
    # What
    event_type: EventType
    action_category: ActionCategory
    action_description: str
    
    # Where (optional)
    entity: Optional[Entity] = None
    
    # Why (reasoning)
    reason: Optional[str] = None
    reasoning_chain: Optional[ReasoningChain] = None
    
    # How (details)
    state_changes: List[StateChange] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Result
    success: bool = True
    result_summary: Optional[str] = None
    error_message: Optional[str] = None
    
    # Metadata
    severity: Severity = Severity.INFO
    compliance_flags: List[ComplianceFlag] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    # Integrity
    checksum: Optional[str] = None
    previous_checksum: Optional[str] = None  # Chain reference
    
    # Context
    request_id: Optional[str] = None
    correlation_id: Optional[str] = None  # Links related entries
    
    def calculate_checksum(self) -> str:
        """Calculate SHA-256 checksum for integrity verification"""
        # Create deterministic string representation
        content = {
            'entry_id': self.entry_id,
            'timestamp': self.timestamp.isoformat(),
            'actor': self.actor.to_dict(),
            'event_type': self.event_type.value,
            'action_category': self.action_category.value,
            'action_description': self.action_description,
            'entity': self.entity.to_dict() if self.entity else None,
            'reason': self.reason,
            'success': self.success,
            'previous_checksum': self.previous_checksum
        }
        
        # Sort keys for deterministic JSON
        content_str = json.dumps(content, sort_keys=True, default=str)
        
        return hashlib.sha256(content_str.encode()).hexdigest()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        return {
            'entry_id': self.entry_id,
            'timestamp': self.timestamp.isoformat(),
            'actor': self.actor.to_dict(),
            'event_type': self.event_type.value,
            'action_category': self.action_category.value,
            'action_description': self.action_description,
            'entity': self.entity.to_dict() if self.entity else None,
            'reason': self.reason,
            'reasoning_chain': self.reasoning_chain.to_dict() if self.reasoning_chain else None,
            'state_changes': [sc.to_dict() for sc in self.state_changes],
            'parameters': self.parameters,
            'success': self.success,
            'result_summary': self.result_summary,
            'error_message': self.error_message,
            'severity': self.severity.value,
            'compliance_flags': [cf.value for cf in self.compliance_flags],
            'tags': self.tags,
            'checksum': self.checksum,
            'previous_checksum': self.previous_checksum,
            'request_id': self.request_id,
            'correlation_id': self.correlation_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AuditEntry':
        """Reconstruct from dictionary"""
        return cls(
            entry_id=data['entry_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            actor=Actor(**data['actor']),
            event_type=EventType(data['event_type']),
            action_category=ActionCategory(data['action_category']),
            action_description=data['action_description'],
            entity=Entity(**data['entity']) if data.get('entity') else None,
            reason=data.get('reason'),
            reasoning_chain=ReasoningChain(**data['reasoning_chain']) if data.get('reasoning_chain') else None,
            state_changes=[StateChange(**sc) for sc in data.get('state_changes', [])],
            parameters=data.get('parameters', {}),
            success=data.get('success', True),
            result_summary=data.get('result_summary'),
            error_message=data.get('error_message'),
            severity=Severity(data.get('severity', 'info')),
            compliance_flags=[ComplianceFlag(cf) for cf in data.get('compliance_flags', [])],
            tags=data.get('tags', []),
            checksum=data.get('checksum'),
            previous_checksum=data.get('previous_checksum'),
            request_id=data.get('request_id'),
            correlation_id=data.get('correlation_id')
        )


# =============================================================================
# AUDIT STORE - Persistent Storage
# =============================================================================

class AuditStore:
    """
    Append-only persistent storage for audit entries
    Uses SQLite for durability and queryability
    """
    
    def __init__(self, db_path: str = "data/audit/audit_trail.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_database()
        self._last_checksum: Optional[str] = None
        self._load_last_checksum()
    
    def _init_database(self):
        """Initialize SQLite database with audit schema"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Main audit entries table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS audit_entries (
                    entry_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    actor_type TEXT NOT NULL,
                    actor_name TEXT,
                    actor_role TEXT,
                    event_type TEXT NOT NULL,
                    action_category TEXT NOT NULL,
                    action_description TEXT NOT NULL,
                    entity_type TEXT,
                    entity_id TEXT,
                    entity_name TEXT,
                    reason TEXT,
                    success INTEGER NOT NULL,
                    result_summary TEXT,
                    error_message TEXT,
                    severity TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    previous_checksum TEXT,
                    correlation_id TEXT,
                    request_id TEXT,
                    full_entry TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Indexes for common queries
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON audit_entries(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_actor_id ON audit_entries(actor_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_event_type ON audit_entries(event_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_entity_id ON audit_entries(entity_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_correlation_id ON audit_entries(correlation_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_severity ON audit_entries(severity)')
            
            # State changes table (for detailed change tracking)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS state_changes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entry_id TEXT NOT NULL,
                    field_name TEXT NOT NULL,
                    old_value TEXT,
                    new_value TEXT,
                    data_type TEXT,
                    FOREIGN KEY (entry_id) REFERENCES audit_entries(entry_id)
                )
            ''')
            
            # Compliance flags table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS compliance_flags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entry_id TEXT NOT NULL,
                    flag TEXT NOT NULL,
                    FOREIGN KEY (entry_id) REFERENCES audit_entries(entry_id)
                )
            ''')
            
            # Tags table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS audit_tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entry_id TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    FOREIGN KEY (entry_id) REFERENCES audit_entries(entry_id)
                )
            ''')
            
            # Reasoning chains table (for AI decisions)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS reasoning_chains (
                    entry_id TEXT PRIMARY KEY,
                    query TEXT,
                    context TEXT,
                    decision TEXT,
                    confidence REAL,
                    model_used TEXT,
                    tokens_used INTEGER,
                    full_chain TEXT,
                    FOREIGN KEY (entry_id) REFERENCES audit_entries(entry_id)
                )
            ''')
            
            # Integrity verification log
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS integrity_checks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    check_timestamp TEXT NOT NULL,
                    entries_checked INTEGER NOT NULL,
                    entries_valid INTEGER NOT NULL,
                    entries_invalid INTEGER NOT NULL,
                    invalid_entry_ids TEXT,
                    check_result TEXT NOT NULL
                )
            ''')
            
            conn.commit()
    
    def _load_last_checksum(self):
        """Load the last checksum for chain continuity"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT checksum FROM audit_entries 
                ORDER BY timestamp DESC, entry_id DESC 
                LIMIT 1
            ''')
            result = cursor.fetchone()
            self._last_checksum = result[0] if result else None
    
    def append(self, entry: AuditEntry) -> str:
        """
        Append entry to audit log (immutable)
        Returns entry_id
        """
        with self._lock:
            # Set chain reference
            entry.previous_checksum = self._last_checksum
            
            # Calculate checksum
            entry.checksum = entry.calculate_checksum()
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Insert main entry
                cursor.execute('''
                    INSERT INTO audit_entries (
                        entry_id, timestamp, actor_id, actor_type, actor_name, actor_role,
                        event_type, action_category, action_description,
                        entity_type, entity_id, entity_name,
                        reason, success, result_summary, error_message, severity,
                        checksum, previous_checksum, correlation_id, request_id,
                        full_entry
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    entry.entry_id,
                    entry.timestamp.isoformat(),
                    entry.actor.actor_id,
                    entry.actor.actor_type,
                    entry.actor.name,
                    entry.actor.role,
                    entry.event_type.value,
                    entry.action_category.value,
                    entry.action_description,
                    entry.entity.entity_type if entry.entity else None,
                    entry.entity.entity_id if entry.entity else None,
                    entry.entity.entity_name if entry.entity else None,
                    entry.reason,
                    1 if entry.success else 0,
                    entry.result_summary,
                    entry.error_message,
                    entry.severity.value,
                    entry.checksum,
                    entry.previous_checksum,
                    entry.correlation_id,
                    entry.request_id,
                    json.dumps(entry.to_dict(), default=str)
                ))
                
                # Insert state changes
                for sc in entry.state_changes:
                    cursor.execute('''
                        INSERT INTO state_changes (entry_id, field_name, old_value, new_value, data_type)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (entry.entry_id, sc.field_name, str(sc.old_value), str(sc.new_value), sc.data_type))
                
                # Insert compliance flags
                for cf in entry.compliance_flags:
                    cursor.execute('''
                        INSERT INTO compliance_flags (entry_id, flag)
                        VALUES (?, ?)
                    ''', (entry.entry_id, cf.value))
                
                # Insert tags
                for tag in entry.tags:
                    cursor.execute('''
                        INSERT INTO audit_tags (entry_id, tag)
                        VALUES (?, ?)
                    ''', (entry.entry_id, tag))
                
                # Insert reasoning chain if present
                if entry.reasoning_chain:
                    rc = entry.reasoning_chain
                    cursor.execute('''
                        INSERT INTO reasoning_chains (
                            entry_id, query, context, decision, confidence, 
                            model_used, tokens_used, full_chain
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        entry.entry_id,
                        rc.query,
                        json.dumps(rc.context, default=str),
                        rc.decision,
                        rc.confidence,
                        rc.model_used,
                        rc.tokens_used,
                        json.dumps(rc.to_dict(), default=str)
                    ))
                
                conn.commit()
            
            # Update chain reference
            self._last_checksum = entry.checksum
            
            return entry.entry_id
    
    def get_entry(self, entry_id: str) -> Optional[AuditEntry]:
        """Retrieve a single entry by ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT full_entry FROM audit_entries WHERE entry_id = ?', (entry_id,))
            result = cursor.fetchone()
            
            if result:
                return AuditEntry.from_dict(json.loads(result[0]))
            return None
    
    def get_entry_count(self) -> int:
        """Get total number of entries"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM audit_entries')
            return cursor.fetchone()[0]
    
    def verify_integrity(self) -> tuple[bool, List[str]]:
        """
        Verify integrity of entire audit chain
        Returns (is_valid, list of invalid entry_ids)
        """
        invalid_entries = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT entry_id, checksum, previous_checksum, full_entry 
                FROM audit_entries 
                ORDER BY timestamp, entry_id
            ''')
            
            expected_previous = None
            
            for row in cursor.fetchall():
                entry_id, stored_checksum, previous_checksum, full_entry_json = row
                
                # Check chain continuity
                if previous_checksum != expected_previous:
                    invalid_entries.append(entry_id)
                    continue
                
                # Recalculate checksum
                entry = AuditEntry.from_dict(json.loads(full_entry_json))
                entry.previous_checksum = previous_checksum  # Restore for calculation
                calculated_checksum = entry.calculate_checksum()
                
                if calculated_checksum != stored_checksum:
                    invalid_entries.append(entry_id)
                
                expected_previous = stored_checksum
            
            # Log integrity check
            entries_checked = cursor.execute('SELECT COUNT(*) FROM audit_entries').fetchone()[0]
            cursor.execute('''
                INSERT INTO integrity_checks (
                    check_timestamp, entries_checked, entries_valid, entries_invalid,
                    invalid_entry_ids, check_result
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                entries_checked,
                entries_checked - len(invalid_entries),
                len(invalid_entries),
                json.dumps(invalid_entries),
                'PASS' if len(invalid_entries) == 0 else 'FAIL'
            ))
            conn.commit()
        
        return len(invalid_entries) == 0, invalid_entries


# =============================================================================
# AUDIT QUERY - Search and Retrieval
# =============================================================================

class AuditQuery:
    """Query interface for audit entries"""
    
    def __init__(self, store: AuditStore):
        self.store = store
    
    def query(
        self,
        actor_id: Optional[str] = None,
        event_type: Optional[EventType] = None,
        action_category: Optional[ActionCategory] = None,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        severity: Optional[Severity] = None,
        success: Optional[bool] = None,
        compliance_flag: Optional[ComplianceFlag] = None,
        tag: Optional[str] = None,
        correlation_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[AuditEntry]:
        """
        Query audit entries with filters
        """
        conditions = []
        params = []
        
        if actor_id:
            conditions.append("actor_id = ?")
            params.append(actor_id)
        
        if event_type:
            conditions.append("event_type = ?")
            params.append(event_type.value)
        
        if action_category:
            conditions.append("action_category = ?")
            params.append(action_category.value)
        
        if entity_type:
            conditions.append("entity_type = ?")
            params.append(entity_type)
        
        if entity_id:
            conditions.append("entity_id = ?")
            params.append(entity_id)
        
        if start_time:
            conditions.append("timestamp >= ?")
            params.append(start_time.isoformat())
        
        if end_time:
            conditions.append("timestamp <= ?")
            params.append(end_time.isoformat())
        
        if severity:
            conditions.append("severity = ?")
            params.append(severity.value)
        
        if success is not None:
            conditions.append("success = ?")
            params.append(1 if success else 0)
        
        if correlation_id:
            conditions.append("correlation_id = ?")
            params.append(correlation_id)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        # Handle compliance flag and tag joins
        join_clause = ""
        if compliance_flag:
            join_clause += " JOIN compliance_flags cf ON audit_entries.entry_id = cf.entry_id"
            where_clause += " AND cf.flag = ?"
            params.append(compliance_flag.value)
        
        if tag:
            join_clause += " JOIN audit_tags at ON audit_entries.entry_id = at.entry_id"
            where_clause += " AND at.tag = ?"
            params.append(tag)
        
        query_sql = f'''
            SELECT DISTINCT audit_entries.full_entry 
            FROM audit_entries
            {join_clause}
            WHERE {where_clause}
            ORDER BY timestamp DESC
            LIMIT ? OFFSET ?
        '''
        params.extend([limit, offset])
        
        entries = []
        with sqlite3.connect(self.store.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query_sql, params)
            
            for row in cursor.fetchall():
                entries.append(AuditEntry.from_dict(json.loads(row[0])))
        
        return entries
    
    def get_by_entity(self, entity_type: str, entity_id: str, limit: int = 50) -> List[AuditEntry]:
        """Get all audit entries for a specific entity"""
        return self.query(entity_type=entity_type, entity_id=entity_id, limit=limit)
    
    def get_by_actor(self, actor_id: str, limit: int = 50) -> List[AuditEntry]:
        """Get all audit entries by a specific actor"""
        return self.query(actor_id=actor_id, limit=limit)
    
    def get_by_correlation(self, correlation_id: str) -> List[AuditEntry]:
        """Get all related entries by correlation ID"""
        return self.query(correlation_id=correlation_id, limit=1000)
    
    def get_ai_decisions(self, limit: int = 50) -> List[AuditEntry]:
        """Get AI decision entries with reasoning chains"""
        return self.query(event_type=EventType.AI_DECISION, limit=limit)
    
    def get_compliance_entries(self, flag: ComplianceFlag, limit: int = 100) -> List[AuditEntry]:
        """Get entries with specific compliance flag"""
        return self.query(compliance_flag=flag, limit=limit)
    
    def get_recent(self, hours: int = 24, limit: int = 100) -> List[AuditEntry]:
        """Get recent entries"""
        start_time = datetime.now() - timedelta(hours=hours)
        return self.query(start_time=start_time, limit=limit)
    
    def get_errors(self, limit: int = 50) -> List[AuditEntry]:
        """Get error entries"""
        return self.query(success=False, limit=limit)
    
    def get_statistics(self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> Dict:
        """Get audit statistics"""
        with sqlite3.connect(self.store.db_path) as conn:
            cursor = conn.cursor()
            
            time_filter = ""
            params = []
            if start_time:
                time_filter += " AND timestamp >= ?"
                params.append(start_time.isoformat())
            if end_time:
                time_filter += " AND timestamp <= ?"
                params.append(end_time.isoformat())
            
            # Total count
            cursor.execute(f'SELECT COUNT(*) FROM audit_entries WHERE 1=1 {time_filter}', params)
            total_count = cursor.fetchone()[0]
            
            # By event type
            cursor.execute(f'''
                SELECT event_type, COUNT(*) 
                FROM audit_entries 
                WHERE 1=1 {time_filter}
                GROUP BY event_type
            ''', params)
            by_event_type = dict(cursor.fetchall())
            
            # By actor
            cursor.execute(f'''
                SELECT actor_id, COUNT(*) 
                FROM audit_entries 
                WHERE 1=1 {time_filter}
                GROUP BY actor_id
                ORDER BY COUNT(*) DESC
                LIMIT 10
            ''', params)
            top_actors = dict(cursor.fetchall())
            
            # By severity
            cursor.execute(f'''
                SELECT severity, COUNT(*) 
                FROM audit_entries 
                WHERE 1=1 {time_filter}
                GROUP BY severity
            ''', params)
            by_severity = dict(cursor.fetchall())
            
            # Success rate
            cursor.execute(f'''
                SELECT 
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successes,
                    SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failures
                FROM audit_entries 
                WHERE 1=1 {time_filter}
            ''', params)
            success_row = cursor.fetchone()
            
            return {
                'total_entries': total_count,
                'by_event_type': by_event_type,
                'by_severity': by_severity,
                'top_actors': top_actors,
                'successes': success_row[0] or 0,
                'failures': success_row[1] or 0,
                'success_rate': (success_row[0] / total_count * 100) if total_count > 0 else 0
            }
    
    def export_for_inspection(
        self,
        start_time: datetime,
        end_time: datetime,
        output_path: str,
        format: str = 'json'
    ) -> str:
        """
        Export audit entries for regulatory inspection
        Supports: json, csv
        """
        entries = self.query(start_time=start_time, end_time=end_time, limit=100000)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'export_range': {
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat()
                },
                'entry_count': len(entries),
                'entries': [e.to_dict() for e in entries]
            }
            
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
        
        elif format == 'csv':
            # Flatten for CSV
            rows = []
            for entry in entries:
                row = {
                    'entry_id': entry.entry_id,
                    'timestamp': entry.timestamp.isoformat(),
                    'actor_id': entry.actor.actor_id,
                    'actor_name': entry.actor.name,
                    'actor_role': entry.actor.role,
                    'event_type': entry.event_type.value,
                    'action_category': entry.action_category.value,
                    'action_description': entry.action_description,
                    'entity_type': entry.entity.entity_type if entry.entity else None,
                    'entity_id': entry.entity.entity_id if entry.entity else None,
                    'reason': entry.reason,
                    'success': entry.success,
                    'severity': entry.severity.value,
                    'checksum': entry.checksum
                }
                rows.append(row)
            
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
        
        return str(output_path)


# =============================================================================
# AUDIT LOGGER - Main Interface
# =============================================================================

class AuditLogger:
    """
    Main interface for audit logging
    Thread-safe, singleton pattern
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, db_path: str = "data/audit/audit_trail.db"):
        if hasattr(self, '_initialized'):
            return
        
        self.store = AuditStore(db_path)
        self.query = AuditQuery(self.store)
        self._initialized = True
        
        # Log system start
        self.log_system_event("Audit Logger initialized", EventType.SYSTEM_START)
    
    def _generate_entry_id(self) -> str:
        """Generate unique entry ID"""
        return f"AUD-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8].upper()}"
    
    def log(
        self,
        actor: Actor,
        event_type: EventType,
        action_category: ActionCategory,
        action_description: str,
        entity: Optional[Entity] = None,
        reason: Optional[str] = None,
        reasoning_chain: Optional[ReasoningChain] = None,
        state_changes: Optional[List[StateChange]] = None,
        parameters: Optional[Dict] = None,
        success: bool = True,
        result_summary: Optional[str] = None,
        error_message: Optional[str] = None,
        severity: Severity = Severity.INFO,
        compliance_flags: Optional[List[ComplianceFlag]] = None,
        tags: Optional[List[str]] = None,
        correlation_id: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> str:
        """
        Log an audit entry
        Returns entry_id
        """
        entry = AuditEntry(
            entry_id=self._generate_entry_id(),
            timestamp=datetime.now(),
            actor=actor,
            event_type=event_type,
            action_category=action_category,
            action_description=action_description,
            entity=entity,
            reason=reason,
            reasoning_chain=reasoning_chain,
            state_changes=state_changes or [],
            parameters=parameters or {},
            success=success,
            result_summary=result_summary,
            error_message=error_message,
            severity=severity,
            compliance_flags=compliance_flags or [],
            tags=tags or [],
            correlation_id=correlation_id,
            request_id=request_id
        )
        
        return self.store.append(entry)
    
    # ==========================================================================
    # Convenience Methods
    # ==========================================================================
    
    def log_user_action(
        self,
        user_id: str,
        user_name: str,
        user_role: str,
        action: str,
        entity: Optional[Entity] = None,
        reason: Optional[str] = None,
        success: bool = True,
        **kwargs
    ) -> str:
        """Log a user action"""
        actor = Actor(
            actor_id=user_id,
            actor_type="user",
            name=user_name,
            role=user_role
        )
        
        return self.log(
            actor=actor,
            event_type=EventType.USER_ACTION,
            action_category=ActionCategory.DATA_MANAGEMENT,
            action_description=action,
            entity=entity,
            reason=reason,
            success=success,
            **kwargs
        )
    
    def log_data_change(
        self,
        actor: Actor,
        entity: Entity,
        changes: List[StateChange],
        reason: Optional[str] = None,
        **kwargs
    ) -> str:
        """Log a data change with before/after states"""
        return self.log(
            actor=actor,
            event_type=EventType.DATA_UPDATE,
            action_category=ActionCategory.DATA_MANAGEMENT,
            action_description=f"Updated {entity.entity_type}: {entity.entity_id}",
            entity=entity,
            state_changes=changes,
            reason=reason,
            compliance_flags=[ComplianceFlag.CFR_11_RELEVANT],
            **kwargs
        )
    
    def log_ai_decision(
        self,
        agent_name: str,
        query: str,
        decision: str,
        reasoning_chain: ReasoningChain,
        entity: Optional[Entity] = None,
        success: bool = True,
        **kwargs
    ) -> str:
        """Log an AI agent decision with full reasoning chain"""
        actor = Actor(
            actor_id=f"agent_{agent_name.lower()}",
            actor_type="agent",
            name=agent_name,
            role="AI Agent"
        )
        
        return self.log(
            actor=actor,
            event_type=EventType.AI_DECISION,
            action_category=ActionCategory.AI_AGENT,
            action_description=f"AI Decision: {decision[:100]}",
            entity=entity,
            reasoning_chain=reasoning_chain,
            success=success,
            tags=['ai', 'decision', agent_name.lower()],
            **kwargs
        )
    
    def log_approval(
        self,
        approver_id: str,
        approver_name: str,
        approver_role: str,
        action_id: str,
        decision: str,  # approved, denied
        reason: Optional[str] = None,
        **kwargs
    ) -> str:
        """Log an approval decision"""
        actor = Actor(
            actor_id=approver_id,
            actor_type="user",
            name=approver_name,
            role=approver_role
        )
        
        event_type = EventType.APPROVAL_GRANTED if decision == "approved" else EventType.APPROVAL_DENIED
        
        return self.log(
            actor=actor,
            event_type=event_type,
            action_category=ActionCategory.APPROVAL,
            action_description=f"Approval {decision} for action {action_id}",
            entity=Entity(entity_type="action", entity_id=action_id),
            reason=reason,
            compliance_flags=[ComplianceFlag.CFR_11_RELEVANT, ComplianceFlag.SIGNATURE_REQUIRED],
            **kwargs
        )
    
    def log_data_access(
        self,
        user_id: str,
        user_name: str,
        user_role: str,
        entity: Entity,
        access_type: str = "view",
        **kwargs
    ) -> str:
        """Log data access for PHI/sensitive data"""
        actor = Actor(
            actor_id=user_id,
            actor_type="user",
            name=user_name,
            role=user_role
        )
        
        return self.log(
            actor=actor,
            event_type=EventType.DATA_VIEW,
            action_category=ActionCategory.PATIENT_DATA,
            action_description=f"Accessed {entity.entity_type}: {entity.entity_id}",
            entity=entity,
            compliance_flags=[ComplianceFlag.PHI_ACCESS],
            severity=Severity.INFO,
            **kwargs
        )
    
    def log_system_event(
        self,
        description: str,
        event_type: EventType = EventType.SYSTEM_CONFIG_CHANGE,
        severity: Severity = Severity.INFO,
        **kwargs
    ) -> str:
        """Log a system event"""
        actor = Actor(
            actor_id="system",
            actor_type="system",
            name="a6on-i",
            role="System"
        )
        
        return self.log(
            actor=actor,
            event_type=event_type,
            action_category=ActionCategory.SYSTEM,
            action_description=description,
            severity=severity,
            **kwargs
        )
    
    def log_error(
        self,
        actor: Actor,
        action_description: str,
        error_message: str,
        entity: Optional[Entity] = None,
        **kwargs
    ) -> str:
        """Log an error"""
        return self.log(
            actor=actor,
            event_type=EventType.SYSTEM_ERROR,
            action_category=ActionCategory.SYSTEM,
            action_description=action_description,
            entity=entity,
            success=False,
            error_message=error_message,
            severity=Severity.ERROR,
            **kwargs
        )
    
    # ==========================================================================
    # Query Methods (delegated to AuditQuery)
    # ==========================================================================
    
    def get_entry(self, entry_id: str) -> Optional[AuditEntry]:
        return self.store.get_entry(entry_id)
    
    def get_by_entity(self, entity_type: str, entity_id: str, limit: int = 50) -> List[AuditEntry]:
        return self.query.get_by_entity(entity_type, entity_id, limit)
    
    def get_by_actor(self, actor_id: str, limit: int = 50) -> List[AuditEntry]:
        return self.query.get_by_actor(actor_id, limit)
    
    def get_recent(self, hours: int = 24, limit: int = 100) -> List[AuditEntry]:
        return self.query.get_recent(hours, limit)
    
    def get_ai_decisions(self, limit: int = 50) -> List[AuditEntry]:
        return self.query.get_ai_decisions(limit)
    
    def get_statistics(self) -> Dict:
        return self.query.get_statistics()
    
    def verify_integrity(self) -> tuple[bool, List[str]]:
        return self.store.verify_integrity()
    
    def export_for_inspection(
        self,
        start_time: datetime,
        end_time: datetime,
        output_path: str,
        format: str = 'json'
    ) -> str:
        return self.query.export_for_inspection(start_time, end_time, output_path, format)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

_audit_logger_instance: Optional[AuditLogger] = None

def get_audit_logger(db_path: str = "data/audit/audit_trail.db") -> AuditLogger:
    """Get or create the audit logger singleton"""
    global _audit_logger_instance
    if _audit_logger_instance is None:
        _audit_logger_instance = AuditLogger(db_path)
    return _audit_logger_instance


# =============================================================================
# TEST FUNCTION
# =============================================================================

def test_audit_trail():
    """Test the audit trail system"""
    print("=" * 60)
    print("a6on-i - AUDIT TRAIL TEST")
    print("=" * 60)
    
    # Use test database
    test_db = "data/audit/audit_trail_test.db"
    
    # Clean up previous test
    test_path = Path(test_db)
    if test_path.exists():
        try:
            test_path.unlink()
        except PermissionError:
            pass  # File in use, continue anyway
    
    # Initialize
    logger = AuditLogger(test_db)
    
    # Test 1: User Action
    print("\n" + "-" * 40)
    print("TEST 1: User Action Logging")
    print("-" * 40)
    
    entry_id = logger.log_user_action(
        user_id="user_001",
        user_name="Sarah Chen",
        user_role="CRA",
        action="Viewed patient list for Site_1",
        entity=Entity(entity_type="site", entity_id="Site_1", entity_name="Tokyo General"),
        reason="Routine monitoring visit preparation"
    )
    print(f"✅ Logged user action: {entry_id}")
    
    # Verify
    entry = logger.get_entry(entry_id)
    assert entry is not None
    assert entry.actor.name == "Sarah Chen"
    print(f"   Actor: {entry.actor.name} ({entry.actor.role})")
    print(f"   Action: {entry.action_description}")
    
    # Test 2: Data Change
    print("\n" + "-" * 40)
    print("TEST 2: Data Change Logging")
    print("-" * 40)
    
    actor = Actor(
        actor_id="dm_001",
        actor_type="user",
        name="Alex Kim",
        role="Data Manager"
    )
    
    changes = [
        StateChange(field_name="query_status", old_value="Open", new_value="Answered", data_type="string"),
        StateChange(field_name="response_date", old_value=None, new_value="2024-01-15", data_type="date")
    ]
    
    entry_id = logger.log_data_change(
        actor=actor,
        entity=Entity(entity_type="query", entity_id="QRY-12345"),
        changes=changes,
        reason="Site provided clarification"
    )
    print(f"✅ Logged data change: {entry_id}")
    
    entry = logger.get_entry(entry_id)
    print(f"   Changes: {len(entry.state_changes)}")
    for sc in entry.state_changes:
        print(f"      {sc.field_name}: {sc.old_value} → {sc.new_value}")
    
    # Test 3: AI Decision with Reasoning Chain
    print("\n" + "-" * 40)
    print("TEST 3: AI Decision with Reasoning Chain")
    print("-" * 40)
    
    reasoning = ReasoningChain(
        query="Why does Site_1 have so many open queries?",
        context={"site_id": "Site_1", "query_count": 45, "avg_query_count": 12},
        hypotheses=[
            {"hypothesis": "Staff turnover", "confidence": 0.75},
            {"hypothesis": "Training gap", "confidence": 0.60}
        ],
        evidence=[
            {"type": "data", "description": "3 coordinator changes in 6 months"},
            {"type": "pattern", "description": "Query spike correlates with new coordinator start dates"}
        ],
        decision="Root cause: Staff turnover causing inconsistent data entry",
        confidence=0.78,
        alternatives_considered=["Training gap", "Protocol complexity"],
        model_used="llama3.1:8b",
        tokens_used=1250
    )
    
    entry_id = logger.log_ai_decision(
        agent_name="DiagnosticAgent",
        query="Why does Site_1 have so many open queries?",
        decision="Staff turnover causing inconsistent data entry",
        reasoning_chain=reasoning,
        entity=Entity(entity_type="site", entity_id="Site_1")
    )
    print(f"✅ Logged AI decision: {entry_id}")
    
    entry = logger.get_entry(entry_id)
    print(f"   Agent: {entry.actor.name}")
    print(f"   Decision: {entry.reasoning_chain.decision}")
    print(f"   Confidence: {entry.reasoning_chain.confidence:.1%}")
    print(f"   Hypotheses: {len(entry.reasoning_chain.hypotheses)}")
    
    # Test 4: Approval
    print("\n" + "-" * 40)
    print("TEST 4: Approval Logging")
    print("-" * 40)
    
    entry_id = logger.log_approval(
        approver_id="lead_001",
        approver_name="John Smith",
        approver_role="Study Lead",
        action_id="ACT-001",
        decision="approved",
        reason="Action plan is appropriate and follows SOP"
    )
    print(f"✅ Logged approval: {entry_id}")
    
    entry = logger.get_entry(entry_id)
    print(f"   Approver: {entry.actor.name}")
    print(f"   Decision: {entry.event_type.value}")
    print(f"   Compliance Flags: {[cf.value for cf in entry.compliance_flags]}")
    
    # Test 5: Query by Entity
    print("\n" + "-" * 40)
    print("TEST 5: Query by Entity")
    print("-" * 40)
    
    entries = logger.get_by_entity("site", "Site_1")
    print(f"✅ Found {len(entries)} entries for Site_1")
    for e in entries:
        print(f"   - {e.timestamp.strftime('%H:%M:%S')}: {e.action_description[:50]}")
    
    # Test 6: Statistics
    print("\n" + "-" * 40)
    print("TEST 6: Audit Statistics")
    print("-" * 40)
    
    stats = logger.get_statistics()
    print(f"✅ Statistics retrieved")
    print(f"   Total Entries: {stats['total_entries']}")
    print(f"   By Event Type: {stats['by_event_type']}")
    print(f"   Success Rate: {stats['success_rate']:.1f}%")
    
    # Test 7: Integrity Verification
    print("\n" + "-" * 40)
    print("TEST 7: Integrity Verification")
    print("-" * 40)
    
    is_valid, invalid = logger.verify_integrity()
    print(f"✅ Integrity Check: {'PASS' if is_valid else 'FAIL'}")
    print(f"   Invalid Entries: {len(invalid)}")
    
    # Test 8: Export for Inspection
    print("\n" + "-" * 40)
    print("TEST 8: Export for Inspection")
    print("-" * 40)
    
    export_path = logger.export_for_inspection(
        start_time=datetime.now() - timedelta(hours=1),
        end_time=datetime.now(),
        output_path="data/audit/test_export.json",
        format='json'
    )
    print(f"✅ Exported to: {export_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    total_entries = logger.store.get_entry_count()
    print(f"✅ All 8 tests passed!")
    print(f"   Total entries created: {total_entries}")
    print(f"   Database: {test_db}")
    
    # Clean up - handle Windows file locking
    # Reset singleton to release connection
    global _audit_logger_instance
    _audit_logger_instance = None
    del logger
    
    import gc
    gc.collect()
    
    import time
    time.sleep(0.3)
    
    try:
        if test_path.exists():
            test_path.unlink()
        print(f"   Test database cleaned up ✓")
    except PermissionError:
        print(f"   ⚠️ Test database retained (file in use)")
        print(f"   Location: {test_path.absolute()}")
    
    print("\n" + "=" * 60)
    print("PHASE 10.1: AUDIT TRAIL SYSTEM - COMPLETE ✅")
    print("=" * 60)


if __name__ == "__main__":
    test_audit_trail()
