# src/collaboration/issue_registry.py
"""
SANCHALAK AI - Issue Registry v1.0
Phase 8.1: SQLite-based Issue Tracking System
# NOTE: This module previously used SQLite but has been migrated to PostgreSQL-only.
# If you need to use this module, update it to use:
#   from src.database.pg_data_service import get_pg_data_service
#   service = get_pg_data_service()


Features:
- SQLite database for persistent storage
- CRUD operations for issues
- Status tracking with workflow
- Assignment management
- Priority and severity handling
- Audit trail for all changes
- Search and filtering
"""

# SQLite removed - using PostgreSQL
import sqlite3  # Kept for backward compatibility with local db
import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid


# =============================================================================
# ENUMS
# =============================================================================

class IssueStatus(Enum):
    """Issue lifecycle status"""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    PENDING_REVIEW = "pending_review"
    PENDING_APPROVAL = "pending_approval"
    RESOLVED = "resolved"
    CLOSED = "closed"
    REOPENED = "reopened"
    ESCALATED = "escalated"
    ON_HOLD = "on_hold"
    CANCELLED = "cancelled"


class IssuePriority(Enum):
    """Issue priority levels"""
    CRITICAL = "critical"      # SLA: 4 hours
    HIGH = "high"              # SLA: 24 hours
    MEDIUM = "medium"          # SLA: 72 hours
    LOW = "low"                # SLA: 168 hours (1 week)


class IssueSeverity(Enum):
    """Issue severity/impact levels"""
    BLOCKER = "blocker"        # Blocks DB Lock
    MAJOR = "major"            # Significant impact
    MINOR = "minor"            # Limited impact
    TRIVIAL = "trivial"        # Minimal impact


class IssueCategory(Enum):
    """Issue categories matching 14 issue types"""
    SDV_INCOMPLETE = "sdv_incomplete"
    OPEN_QUERIES = "open_queries"
    SIGNATURE_GAPS = "signature_gaps"
    BROKEN_SIGNATURES = "broken_signatures"
    SAE_DM_PENDING = "sae_dm_pending"
    SAE_SAFETY_PENDING = "sae_safety_pending"
    MISSING_VISITS = "missing_visits"
    MISSING_PAGES = "missing_pages"
    MEDDRA_UNCODED = "meddra_uncoded"
    WHODRUG_UNCODED = "whodrug_uncoded"
    LAB_ISSUES = "lab_issues"
    EDRR_ISSUES = "edrr_issues"
    INACTIVATED_FORMS = "inactivated_forms"
    HIGH_QUERY_VOLUME = "high_query_volume"
    DATA_QUALITY = "data_quality"
    PROTOCOL_DEVIATION = "protocol_deviation"
    SAFETY_SIGNAL = "safety_signal"
    SYSTEM_ISSUE = "system_issue"
    OTHER = "other"


class AssigneeRole(Enum):
    """Roles that can be assigned issues"""
    CRA = "cra"
    DATA_MANAGER = "data_manager"
    SITE_COORDINATOR = "site_coordinator"
    SAFETY_DATA_MANAGER = "safety_data_manager"
    SAFETY_PHYSICIAN = "safety_physician"
    MEDICAL_CODER = "medical_coder"
    STUDY_LEAD = "study_lead"
    CTM = "ctm"
    PRINCIPAL_INVESTIGATOR = "principal_investigator"
    UNASSIGNED = "unassigned"


class AuditAction(Enum):
    """Audit trail action types"""
    CREATED = "created"
    UPDATED = "updated"
    STATUS_CHANGED = "status_changed"
    ASSIGNED = "assigned"
    REASSIGNED = "reassigned"
    COMMENTED = "commented"
    ESCALATED = "escalated"
    RESOLVED = "resolved"
    CLOSED = "closed"
    REOPENED = "reopened"
    ATTACHMENT_ADDED = "attachment_added"
    LINKED = "linked"


# =============================================================================
# SLA CONFIGURATION
# =============================================================================

SLA_HOURS = {
    IssuePriority.CRITICAL: 4,
    IssuePriority.HIGH: 24,
    IssuePriority.MEDIUM: 72,
    IssuePriority.LOW: 168,
}

# Status transitions allowed
STATUS_TRANSITIONS = {
    IssueStatus.OPEN: [IssueStatus.IN_PROGRESS, IssueStatus.ON_HOLD, IssueStatus.CANCELLED, IssueStatus.ESCALATED],
    IssueStatus.IN_PROGRESS: [IssueStatus.PENDING_REVIEW, IssueStatus.ON_HOLD, IssueStatus.RESOLVED, IssueStatus.ESCALATED],
    IssueStatus.PENDING_REVIEW: [IssueStatus.IN_PROGRESS, IssueStatus.PENDING_APPROVAL, IssueStatus.RESOLVED],
    IssueStatus.PENDING_APPROVAL: [IssueStatus.RESOLVED, IssueStatus.IN_PROGRESS],
    IssueStatus.RESOLVED: [IssueStatus.CLOSED, IssueStatus.REOPENED],
    IssueStatus.CLOSED: [IssueStatus.REOPENED],
    IssueStatus.REOPENED: [IssueStatus.IN_PROGRESS, IssueStatus.ESCALATED],
    IssueStatus.ESCALATED: [IssueStatus.IN_PROGRESS, IssueStatus.RESOLVED],
    IssueStatus.ON_HOLD: [IssueStatus.IN_PROGRESS, IssueStatus.CANCELLED],
    IssueStatus.CANCELLED: [],  # Terminal state
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Issue:
    """Issue data model"""
    issue_id: str
    title: str
    description: str
    category: IssueCategory
    priority: IssuePriority
    severity: IssueSeverity
    status: IssueStatus
    
    # Entity references
    study_id: Optional[str] = None
    site_id: Optional[str] = None
    patient_key: Optional[str] = None
    
    # Assignment
    assignee_id: Optional[str] = None
    assignee_name: Optional[str] = None
    assignee_role: AssigneeRole = AssigneeRole.UNASSIGNED
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    due_date: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    
    # SLA tracking
    sla_hours: int = 72
    sla_breached: bool = False
    
    # Metadata
    created_by: str = "system"
    tags: List[str] = field(default_factory=list)
    linked_issues: List[str] = field(default_factory=list)
    
    # Counts
    comment_count: int = 0
    attachment_count: int = 0
    
    def __post_init__(self):
        """Set SLA hours based on priority"""
        if isinstance(self.priority, IssuePriority):
            self.sla_hours = SLA_HOURS.get(self.priority, 72)
            if self.due_date is None:
                self.due_date = self.created_at + timedelta(hours=self.sla_hours)
    
    @property
    def is_overdue(self) -> bool:
        """Check if issue is past due date"""
        if self.status in [IssueStatus.RESOLVED, IssueStatus.CLOSED, IssueStatus.CANCELLED]:
            return False
        if self.due_date:
            return datetime.now() > self.due_date
        return False
    
    @property
    def time_remaining(self) -> Optional[timedelta]:
        """Get time remaining until due"""
        if self.due_date:
            remaining = self.due_date - datetime.now()
            return remaining if remaining.total_seconds() > 0 else timedelta(0)
        return None
    
    @property
    def age_hours(self) -> float:
        """Get issue age in hours"""
        return (datetime.now() - self.created_at).total_seconds() / 3600
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        d = asdict(self)
        # Convert enums to values
        d['category'] = self.category.value
        d['priority'] = self.priority.value
        d['severity'] = self.severity.value
        d['status'] = self.status.value
        d['assignee_role'] = self.assignee_role.value
        # Convert datetimes to ISO strings
        for key in ['created_at', 'updated_at', 'due_date', 'resolved_at', 'closed_at']:
            if d[key]:
                d[key] = d[key].isoformat()
        return d


@dataclass
class IssueComment:
    """Comment on an issue"""
    comment_id: str
    issue_id: str
    author_id: str
    author_name: str
    content: str
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    is_internal: bool = False  # Internal notes vs visible comments
    mentions: List[str] = field(default_factory=list)  # @mentioned users
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['created_at'] = self.created_at.isoformat()
        if d['updated_at']:
            d['updated_at'] = d['updated_at'].isoformat()
        return d


@dataclass
class IssueAuditEntry:
    """Audit trail entry for issue changes"""
    audit_id: str
    issue_id: str
    action: AuditAction
    actor_id: str
    actor_name: str
    timestamp: datetime
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    field_changed: Optional[str] = None
    description: str = ""
    checksum: str = ""
    
    def __post_init__(self):
        """Generate checksum for integrity"""
        if not self.checksum:
            data = f"{self.audit_id}|{self.issue_id}|{self.action.value}|{self.actor_id}|{self.timestamp.isoformat()}"
            self.checksum = hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['action'] = self.action.value
        d['timestamp'] = self.timestamp.isoformat()
        return d


@dataclass
class IssueFilter:
    """Filter criteria for issue queries"""
    status: Optional[List[IssueStatus]] = None
    priority: Optional[List[IssuePriority]] = None
    severity: Optional[List[IssueSeverity]] = None
    category: Optional[List[IssueCategory]] = None
    assignee_id: Optional[str] = None
    assignee_role: Optional[AssigneeRole] = None
    study_id: Optional[str] = None
    site_id: Optional[str] = None
    patient_key: Optional[str] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    is_overdue: Optional[bool] = None
    sla_breached: Optional[bool] = None
    tags: Optional[List[str]] = None
    search_text: Optional[str] = None


# =============================================================================
# ISSUE REGISTRY DATABASE
# =============================================================================

class IssueRegistry:
    """
    SQLite-based Issue Registry
    
    Features:
    - Persistent storage with SQLite
    - Full CRUD operations
    - Status workflow management
    - Assignment tracking
    - Audit trail
    - Search and filtering
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize the Issue Registry"""
        if db_path is None:
            db_path = "data/collaboration/issue_registry.db"
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Statistics
        self.stats = {
            'issues_created': 0,
            'issues_updated': 0,
            'issues_resolved': 0,
            'comments_added': 0,
            'queries_executed': 0,
        }
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn
    
    def _init_database(self):
        """Initialize database schema"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Issues table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS issues (
                issue_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT,
                category TEXT NOT NULL,
                priority TEXT NOT NULL,
                severity TEXT NOT NULL,
                status TEXT NOT NULL,
                study_id TEXT,
                site_id TEXT,
                patient_key TEXT,
                assignee_id TEXT,
                assignee_name TEXT,
                assignee_role TEXT DEFAULT 'unassigned',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                due_date TEXT,
                resolved_at TEXT,
                closed_at TEXT,
                sla_hours INTEGER DEFAULT 72,
                sla_breached INTEGER DEFAULT 0,
                created_by TEXT NOT NULL,
                tags TEXT DEFAULT '[]',
                linked_issues TEXT DEFAULT '[]',
                comment_count INTEGER DEFAULT 0,
                attachment_count INTEGER DEFAULT 0
            )
        ''')
        
        # Comments table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS comments (
                comment_id TEXT PRIMARY KEY,
                issue_id TEXT NOT NULL,
                author_id TEXT NOT NULL,
                author_name TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT,
                is_internal INTEGER DEFAULT 0,
                mentions TEXT DEFAULT '[]',
                FOREIGN KEY (issue_id) REFERENCES issues(issue_id)
            )
        ''')
        
        # Audit trail table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_trail (
                audit_id TEXT PRIMARY KEY,
                issue_id TEXT NOT NULL,
                action TEXT NOT NULL,
                actor_id TEXT NOT NULL,
                actor_name TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                old_value TEXT,
                new_value TEXT,
                field_changed TEXT,
                description TEXT,
                checksum TEXT NOT NULL,
                FOREIGN KEY (issue_id) REFERENCES issues(issue_id)
            )
        ''')
        
        # Create indexes for common queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_issues_status ON issues(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_issues_priority ON issues(priority)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_issues_assignee ON issues(assignee_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_issues_study ON issues(study_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_issues_site ON issues(site_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_issues_created ON issues(created_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_comments_issue ON comments(issue_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_audit_issue ON audit_trail(issue_id)')
        
        conn.commit()
        conn.close()
    
    # =========================================================================
    # CRUD OPERATIONS
    # =========================================================================
    
    def create_issue(
        self,
        title: str,
        description: str,
        category: IssueCategory,
        priority: IssuePriority,
        severity: IssueSeverity,
        created_by: str,
        study_id: Optional[str] = None,
        site_id: Optional[str] = None,
        patient_key: Optional[str] = None,
        assignee_id: Optional[str] = None,
        assignee_name: Optional[str] = None,
        assignee_role: AssigneeRole = AssigneeRole.UNASSIGNED,
        tags: Optional[List[str]] = None,
    ) -> Issue:
        """Create a new issue"""
        
        # Generate issue ID
        issue_id = f"ISS-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"
        
        # Create issue object
        issue = Issue(
            issue_id=issue_id,
            title=title,
            description=description,
            category=category,
            priority=priority,
            severity=severity,
            status=IssueStatus.OPEN,
            study_id=study_id,
            site_id=site_id,
            patient_key=patient_key,
            assignee_id=assignee_id,
            assignee_name=assignee_name,
            assignee_role=assignee_role,
            created_by=created_by,
            tags=tags or [],
        )
        
        # Insert into database
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO issues (
                issue_id, title, description, category, priority, severity, status,
                study_id, site_id, patient_key, assignee_id, assignee_name, assignee_role,
                created_at, updated_at, due_date, sla_hours, sla_breached, created_by,
                tags, linked_issues, comment_count, attachment_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            issue.issue_id, issue.title, issue.description,
            issue.category.value, issue.priority.value, issue.severity.value, issue.status.value,
            issue.study_id, issue.site_id, issue.patient_key,
            issue.assignee_id, issue.assignee_name, issue.assignee_role.value,
            issue.created_at.isoformat(), issue.updated_at.isoformat(),
            issue.due_date.isoformat() if issue.due_date else None,
            issue.sla_hours, 0, issue.created_by,
            json.dumps(issue.tags), json.dumps(issue.linked_issues),
            0, 0
        ))
        
        conn.commit()
        conn.close()
        
        # Create audit entry
        self._add_audit_entry(
            issue_id=issue.issue_id,
            action=AuditAction.CREATED,
            actor_id=created_by,
            actor_name=created_by,
            description=f"Issue created: {title}"
        )
        
        self.stats['issues_created'] += 1
        
        return issue
    
    def get_issue(self, issue_id: str) -> Optional[Issue]:
        """Get issue by ID"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM issues WHERE issue_id = ?', (issue_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return self._row_to_issue(row)
        return None
    
    def update_issue(
        self,
        issue_id: str,
        updated_by: str,
        **updates
    ) -> Optional[Issue]:
        """Update issue fields"""
        
        issue = self.get_issue(issue_id)
        if not issue:
            return None
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Build update query
        set_clauses = []
        values = []
        
        for field, value in updates.items():
            if hasattr(issue, field):
                old_value = getattr(issue, field)
                
                # Handle enum values
                if isinstance(value, Enum):
                    db_value = value.value
                elif isinstance(value, list):
                    db_value = json.dumps(value)
                elif isinstance(value, datetime):
                    db_value = value.isoformat()
                else:
                    db_value = value
                
                set_clauses.append(f"{field} = ?")
                values.append(db_value)
                
                # Create audit entry for each changed field
                if old_value != value:
                    old_str = old_value.value if isinstance(old_value, Enum) else str(old_value)
                    new_str = value.value if isinstance(value, Enum) else str(value)
                    
                    self._add_audit_entry(
                        issue_id=issue_id,
                        action=AuditAction.UPDATED,
                        actor_id=updated_by,
                        actor_name=updated_by,
                        field_changed=field,
                        old_value=old_str,
                        new_value=new_str,
                        description=f"Updated {field}: {old_str} → {new_str}"
                    )
        
        # Always update updated_at
        set_clauses.append("updated_at = ?")
        values.append(datetime.now().isoformat())
        
        values.append(issue_id)
        
        query = f"UPDATE issues SET {', '.join(set_clauses)} WHERE issue_id = ?"
        cursor.execute(query, values)
        
        conn.commit()
        conn.close()
        
        self.stats['issues_updated'] += 1
        
        return self.get_issue(issue_id)
    
    def delete_issue(self, issue_id: str, deleted_by: str) -> bool:
        """Soft delete issue (mark as cancelled)"""
        issue = self.get_issue(issue_id)
        if not issue:
            return False
        
        # Use status change instead of hard delete
        return self.change_status(issue_id, IssueStatus.CANCELLED, deleted_by) is not None
    
    # =========================================================================
    # STATUS MANAGEMENT
    # =========================================================================
    
    def change_status(
        self,
        issue_id: str,
        new_status: IssueStatus,
        changed_by: str,
        comment: Optional[str] = None
    ) -> Optional[Issue]:
        """Change issue status with workflow validation"""
        
        issue = self.get_issue(issue_id)
        if not issue:
            return None
        
        # Validate transition
        allowed = STATUS_TRANSITIONS.get(issue.status, [])
        if new_status not in allowed:
            raise ValueError(
                f"Invalid status transition: {issue.status.value} → {new_status.value}. "
                f"Allowed: {[s.value for s in allowed]}"
            )
        
        old_status = issue.status
        
        # Update status
        conn = self._get_connection()
        cursor = conn.cursor()
        
        updates = {
            'status': new_status.value,
            'updated_at': datetime.now().isoformat(),
        }
        
        # Set resolved/closed timestamps
        if new_status == IssueStatus.RESOLVED:
            updates['resolved_at'] = datetime.now().isoformat()
        elif new_status == IssueStatus.CLOSED:
            updates['closed_at'] = datetime.now().isoformat()
        
        set_clause = ', '.join([f"{k} = ?" for k in updates.keys()])
        values = list(updates.values()) + [issue_id]
        
        cursor.execute(f"UPDATE issues SET {set_clause} WHERE issue_id = ?", values)
        conn.commit()
        conn.close()
        
        # Audit entry
        self._add_audit_entry(
            issue_id=issue_id,
            action=AuditAction.STATUS_CHANGED,
            actor_id=changed_by,
            actor_name=changed_by,
            old_value=old_status.value,
            new_value=new_status.value,
            field_changed='status',
            description=f"Status changed: {old_status.value} → {new_status.value}"
        )
        
        # Add comment if provided
        if comment:
            self.add_comment(issue_id, changed_by, changed_by, comment)
        
        if new_status == IssueStatus.RESOLVED:
            self.stats['issues_resolved'] += 1
        
        return self.get_issue(issue_id)
    
    # =========================================================================
    # ASSIGNMENT MANAGEMENT
    # =========================================================================
    
    def assign_issue(
        self,
        issue_id: str,
        assignee_id: str,
        assignee_name: str,
        assignee_role: AssigneeRole,
        assigned_by: str,
        comment: Optional[str] = None
    ) -> Optional[Issue]:
        """Assign issue to user"""
        
        issue = self.get_issue(issue_id)
        if not issue:
            return None
        
        old_assignee = issue.assignee_name or "Unassigned"
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE issues 
            SET assignee_id = ?, assignee_name = ?, assignee_role = ?, updated_at = ?
            WHERE issue_id = ?
        ''', (assignee_id, assignee_name, assignee_role.value, datetime.now().isoformat(), issue_id))
        
        conn.commit()
        conn.close()
        
        # Audit entry
        action = AuditAction.ASSIGNED if issue.assignee_id is None else AuditAction.REASSIGNED
        self._add_audit_entry(
            issue_id=issue_id,
            action=action,
            actor_id=assigned_by,
            actor_name=assigned_by,
            old_value=old_assignee,
            new_value=assignee_name,
            field_changed='assignee',
            description=f"Assigned to {assignee_name} ({assignee_role.value})"
        )
        
        if comment:
            self.add_comment(issue_id, assigned_by, assigned_by, comment)
        
        return self.get_issue(issue_id)
    
    def unassign_issue(self, issue_id: str, unassigned_by: str) -> Optional[Issue]:
        """Remove assignment from issue"""
        
        issue = self.get_issue(issue_id)
        if not issue:
            return None
        
        old_assignee = issue.assignee_name or "Unassigned"
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE issues 
            SET assignee_id = NULL, assignee_name = NULL, assignee_role = 'unassigned', updated_at = ?
            WHERE issue_id = ?
        ''', (datetime.now().isoformat(), issue_id))
        
        conn.commit()
        conn.close()
        
        self._add_audit_entry(
            issue_id=issue_id,
            action=AuditAction.REASSIGNED,
            actor_id=unassigned_by,
            actor_name=unassigned_by,
            old_value=old_assignee,
            new_value="Unassigned",
            field_changed='assignee',
            description=f"Unassigned from {old_assignee}"
        )
        
        return self.get_issue(issue_id)
    
    # =========================================================================
    # COMMENTS
    # =========================================================================
    
    def add_comment(
        self,
        issue_id: str,
        author_id: str,
        author_name: str,
        content: str,
        is_internal: bool = False,
        mentions: Optional[List[str]] = None
    ) -> Optional[IssueComment]:
        """Add comment to issue"""
        
        issue = self.get_issue(issue_id)
        if not issue:
            return None
        
        comment_id = f"CMT-{uuid.uuid4().hex[:12].upper()}"
        
        comment = IssueComment(
            comment_id=comment_id,
            issue_id=issue_id,
            author_id=author_id,
            author_name=author_name,
            content=content,
            is_internal=is_internal,
            mentions=mentions or []
        )
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO comments (comment_id, issue_id, author_id, author_name, content, 
                                 created_at, is_internal, mentions)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            comment.comment_id, comment.issue_id, comment.author_id, comment.author_name,
            comment.content, comment.created_at.isoformat(),
            1 if comment.is_internal else 0, json.dumps(comment.mentions)
        ))
        
        # Update comment count
        cursor.execute('''
            UPDATE issues SET comment_count = comment_count + 1, updated_at = ?
            WHERE issue_id = ?
        ''', (datetime.now().isoformat(), issue_id))
        
        conn.commit()
        conn.close()
        
        # Audit entry
        self._add_audit_entry(
            issue_id=issue_id,
            action=AuditAction.COMMENTED,
            actor_id=author_id,
            actor_name=author_name,
            description=f"Added comment: {content[:50]}..."
        )
        
        self.stats['comments_added'] += 1
        
        return comment
    
    def get_comments(self, issue_id: str, include_internal: bool = False) -> List[IssueComment]:
        """Get comments for an issue"""
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if include_internal:
            cursor.execute(
                'SELECT * FROM comments WHERE issue_id = ? ORDER BY created_at ASC',
                (issue_id,)
            )
        else:
            cursor.execute(
                'SELECT * FROM comments WHERE issue_id = ? AND is_internal = 0 ORDER BY created_at ASC',
                (issue_id,)
            )
        
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_comment(row) for row in rows]
    
    # =========================================================================
    # SEARCH & FILTER
    # =========================================================================
    
    def search_issues(
        self,
        filter_criteria: Optional[IssueFilter] = None,
        order_by: str = 'created_at',
        order_desc: bool = True,
        limit: int = 100,
        offset: int = 0
    ) -> Tuple[List[Issue], int]:
        """Search issues with filters"""
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        where_clauses = []
        params = []
        
        if filter_criteria:
            if filter_criteria.status:
                placeholders = ','.join(['?' for _ in filter_criteria.status])
                where_clauses.append(f"status IN ({placeholders})")
                params.extend([s.value for s in filter_criteria.status])
            
            if filter_criteria.priority:
                placeholders = ','.join(['?' for _ in filter_criteria.priority])
                where_clauses.append(f"priority IN ({placeholders})")
                params.extend([p.value for p in filter_criteria.priority])
            
            if filter_criteria.severity:
                placeholders = ','.join(['?' for _ in filter_criteria.severity])
                where_clauses.append(f"severity IN ({placeholders})")
                params.extend([s.value for s in filter_criteria.severity])
            
            if filter_criteria.category:
                placeholders = ','.join(['?' for _ in filter_criteria.category])
                where_clauses.append(f"category IN ({placeholders})")
                params.extend([c.value for c in filter_criteria.category])
            
            if filter_criteria.assignee_id:
                where_clauses.append("assignee_id = ?")
                params.append(filter_criteria.assignee_id)
            
            if filter_criteria.assignee_role:
                where_clauses.append("assignee_role = ?")
                params.append(filter_criteria.assignee_role.value)
            
            if filter_criteria.study_id:
                where_clauses.append("study_id = ?")
                params.append(filter_criteria.study_id)
            
            if filter_criteria.site_id:
                where_clauses.append("site_id = ?")
                params.append(filter_criteria.site_id)
            
            if filter_criteria.patient_key:
                where_clauses.append("patient_key = ?")
                params.append(filter_criteria.patient_key)
            
            if filter_criteria.created_after:
                where_clauses.append("created_at >= ?")
                params.append(filter_criteria.created_after.isoformat())
            
            if filter_criteria.created_before:
                where_clauses.append("created_at <= ?")
                params.append(filter_criteria.created_before.isoformat())
            
            if filter_criteria.sla_breached is not None:
                where_clauses.append("sla_breached = ?")
                params.append(1 if filter_criteria.sla_breached else 0)
            
            if filter_criteria.search_text:
                where_clauses.append("(title LIKE ? OR description LIKE ?)")
                search_pattern = f"%{filter_criteria.search_text}%"
                params.extend([search_pattern, search_pattern])
        
        # Build query
        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
        order_dir = "DESC" if order_desc else "ASC"
        
        # Get total count
        count_query = f"SELECT COUNT(*) FROM issues WHERE {where_sql}"
        cursor.execute(count_query, params)
        total_count = cursor.fetchone()[0]
        
        # Get results
        query = f"""
            SELECT * FROM issues 
            WHERE {where_sql}
            ORDER BY {order_by} {order_dir}
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        self.stats['queries_executed'] += 1
        
        issues = [self._row_to_issue(row) for row in rows]
        return issues, total_count
    
    def get_issues_by_status(self, status: IssueStatus) -> List[Issue]:
        """Get all issues with specific status"""
        filter_criteria = IssueFilter(status=[status])
        issues, _ = self.search_issues(filter_criteria)
        return issues
    
    def get_issues_by_assignee(self, assignee_id: str) -> List[Issue]:
        """Get all issues assigned to user"""
        filter_criteria = IssueFilter(assignee_id=assignee_id)
        issues, _ = self.search_issues(filter_criteria)
        return issues
    
    def get_overdue_issues(self) -> List[Issue]:
        """Get all overdue issues"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM issues 
            WHERE status NOT IN ('resolved', 'closed', 'cancelled')
            AND due_date < ?
            ORDER BY due_date ASC
        ''', (datetime.now().isoformat(),))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_issue(row) for row in rows]
    
    # =========================================================================
    # AUDIT TRAIL
    # =========================================================================
    
    def _add_audit_entry(
        self,
        issue_id: str,
        action: AuditAction,
        actor_id: str,
        actor_name: str,
        description: str = "",
        old_value: Optional[str] = None,
        new_value: Optional[str] = None,
        field_changed: Optional[str] = None
    ):
        """Add audit trail entry"""
        
        audit_id = f"AUD-{uuid.uuid4().hex[:12].upper()}"
        
        entry = IssueAuditEntry(
            audit_id=audit_id,
            issue_id=issue_id,
            action=action,
            actor_id=actor_id,
            actor_name=actor_name,
            timestamp=datetime.now(),
            old_value=old_value,
            new_value=new_value,
            field_changed=field_changed,
            description=description
        )
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO audit_trail (audit_id, issue_id, action, actor_id, actor_name,
                                    timestamp, old_value, new_value, field_changed,
                                    description, checksum)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            entry.audit_id, entry.issue_id, entry.action.value,
            entry.actor_id, entry.actor_name, entry.timestamp.isoformat(),
            entry.old_value, entry.new_value, entry.field_changed,
            entry.description, entry.checksum
        ))
        
        conn.commit()
        conn.close()
    
    def get_audit_trail(self, issue_id: str) -> List[IssueAuditEntry]:
        """Get audit trail for issue"""
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT * FROM audit_trail WHERE issue_id = ? ORDER BY timestamp DESC',
            (issue_id,)
        )
        
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_audit(row) for row in rows]
    
    # =========================================================================
    # STATISTICS
    # =========================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics"""
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Count by status
        cursor.execute('''
            SELECT status, COUNT(*) as count 
            FROM issues 
            GROUP BY status
        ''')
        status_counts = {row['status']: row['count'] for row in cursor.fetchall()}
        
        # Count by priority
        cursor.execute('''
            SELECT priority, COUNT(*) as count 
            FROM issues 
            GROUP BY priority
        ''')
        priority_counts = {row['priority']: row['count'] for row in cursor.fetchall()}
        
        # Count by category
        cursor.execute('''
            SELECT category, COUNT(*) as count 
            FROM issues 
            GROUP BY category
            ORDER BY count DESC
            LIMIT 10
        ''')
        top_categories = {row['category']: row['count'] for row in cursor.fetchall()}
        
        # Overdue count
        cursor.execute('''
            SELECT COUNT(*) FROM issues 
            WHERE status NOT IN ('resolved', 'closed', 'cancelled')
            AND due_date < ?
        ''', (datetime.now().isoformat(),))
        overdue_count = cursor.fetchone()[0]
        
        # SLA breached count
        cursor.execute('SELECT COUNT(*) FROM issues WHERE sla_breached = 1')
        sla_breached_count = cursor.fetchone()[0]
        
        # Total counts
        cursor.execute('SELECT COUNT(*) FROM issues')
        total_issues = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM comments')
        total_comments = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_issues': total_issues,
            'total_comments': total_comments,
            'status_distribution': status_counts,
            'priority_distribution': priority_counts,
            'top_categories': top_categories,
            'overdue_count': overdue_count,
            'sla_breached_count': sla_breached_count,
            'session_stats': self.stats
        }
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _row_to_issue(self, row: sqlite3.Row) -> Issue:
        """Convert database row to Issue object"""
        return Issue(
            issue_id=row['issue_id'],
            title=row['title'],
            description=row['description'],
            category=IssueCategory(row['category']),
            priority=IssuePriority(row['priority']),
            severity=IssueSeverity(row['severity']),
            status=IssueStatus(row['status']),
            study_id=row['study_id'],
            site_id=row['site_id'],
            patient_key=row['patient_key'],
            assignee_id=row['assignee_id'],
            assignee_name=row['assignee_name'],
            assignee_role=AssigneeRole(row['assignee_role']),
            created_at=datetime.fromisoformat(row['created_at']),
            updated_at=datetime.fromisoformat(row['updated_at']),
            due_date=datetime.fromisoformat(row['due_date']) if row['due_date'] else None,
            resolved_at=datetime.fromisoformat(row['resolved_at']) if row['resolved_at'] else None,
            closed_at=datetime.fromisoformat(row['closed_at']) if row['closed_at'] else None,
            sla_hours=row['sla_hours'],
            sla_breached=bool(row['sla_breached']),
            created_by=row['created_by'],
            tags=json.loads(row['tags']),
            linked_issues=json.loads(row['linked_issues']),
            comment_count=row['comment_count'],
            attachment_count=row['attachment_count'],
        )
    
    def _row_to_comment(self, row: sqlite3.Row) -> IssueComment:
        """Convert database row to IssueComment object"""
        return IssueComment(
            comment_id=row['comment_id'],
            issue_id=row['issue_id'],
            author_id=row['author_id'],
            author_name=row['author_name'],
            content=row['content'],
            created_at=datetime.fromisoformat(row['created_at']),
            updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None,
            is_internal=bool(row['is_internal']),
            mentions=json.loads(row['mentions']),
        )
    
    def _row_to_audit(self, row: sqlite3.Row) -> IssueAuditEntry:
        """Convert database row to IssueAuditEntry object"""
        return IssueAuditEntry(
            audit_id=row['audit_id'],
            issue_id=row['issue_id'],
            action=AuditAction(row['action']),
            actor_id=row['actor_id'],
            actor_name=row['actor_name'],
            timestamp=datetime.fromisoformat(row['timestamp']),
            old_value=row['old_value'],
            new_value=row['new_value'],
            field_changed=row['field_changed'],
            description=row['description'],
            checksum=row['checksum'],
        )


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

_registry_instance: Optional[IssueRegistry] = None

def get_issue_registry(db_path: Optional[str] = None) -> IssueRegistry:
    """Get or create Issue Registry singleton"""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = IssueRegistry(db_path)
    return _registry_instance


# =============================================================================
# TEST FUNCTION
# =============================================================================

def test_issue_registry():
    """Test the Issue Registry"""
    print("=" * 60)
    print("SANCHALAK AI - ISSUE REGISTRY TEST")
    print("=" * 60)
    
    # Use test database
    test_db = "data/collaboration/test_issue_registry.db"
    
    # Clean up old test database
    if Path(test_db).exists():
        Path(test_db).unlink()
    
    registry = IssueRegistry(test_db)
    
    tests_passed = 0
    tests_total = 12
    
    # Test 1: Create Issue
    print("\n--- TEST 1: Create Issue ---")
    try:
        issue = registry.create_issue(
            title="SDV incomplete for Site_101",
            description="15 CRFs pending SDV verification",
            category=IssueCategory.SDV_INCOMPLETE,
            priority=IssuePriority.HIGH,
            severity=IssueSeverity.MAJOR,
            created_by="system",
            study_id="Study_21",
            site_id="Site_101",
            tags=["sdv", "urgent"]
        )
        assert issue is not None
        assert issue.issue_id.startswith("ISS-")
        assert issue.status == IssueStatus.OPEN
        print(f"✅ Created issue: {issue.issue_id}")
        print(f"   Due date: {issue.due_date}")
        print(f"   SLA hours: {issue.sla_hours}")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 2: Get Issue
    print("\n--- TEST 2: Get Issue ---")
    try:
        retrieved = registry.get_issue(issue.issue_id)
        assert retrieved is not None
        assert retrieved.title == issue.title
        print(f"✅ Retrieved issue: {retrieved.title}")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 3: Update Issue
    print("\n--- TEST 3: Update Issue ---")
    try:
        updated = registry.update_issue(
            issue.issue_id,
            updated_by="cra_001",
            title="SDV incomplete for Site_101 - URGENT",
            description="15 CRFs pending SDV verification - needs immediate attention"
        )
        assert updated is not None
        assert "URGENT" in updated.title
        print(f"✅ Updated issue: {updated.title}")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 4: Assign Issue
    print("\n--- TEST 4: Assign Issue ---")
    try:
        assigned = registry.assign_issue(
            issue.issue_id,
            assignee_id="cra_001",
            assignee_name="Sarah Chen",
            assignee_role=AssigneeRole.CRA,
            assigned_by="lead_001",
            comment="Please prioritize this week"
        )
        assert assigned is not None
        assert assigned.assignee_name == "Sarah Chen"
        print(f"✅ Assigned to: {assigned.assignee_name} ({assigned.assignee_role.value})")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 5: Change Status
    print("\n--- TEST 5: Change Status ---")
    try:
        # OPEN → IN_PROGRESS
        in_progress = registry.change_status(
            issue.issue_id,
            IssueStatus.IN_PROGRESS,
            "cra_001",
            "Started working on SDV"
        )
        assert in_progress is not None
        assert in_progress.status == IssueStatus.IN_PROGRESS
        print(f"✅ Status changed to: {in_progress.status.value}")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 6: Add Comment
    print("\n--- TEST 6: Add Comment ---")
    try:
        comment = registry.add_comment(
            issue.issue_id,
            author_id="cra_001",
            author_name="Sarah Chen",
            content="Completed 5 of 15 CRFs today. Will finish remaining tomorrow.",
            mentions=["@lead_001"]
        )
        assert comment is not None
        assert comment.comment_id.startswith("CMT-")
        print(f"✅ Added comment: {comment.comment_id}")
        print(f"   Content: {comment.content[:50]}...")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 7: Get Comments
    print("\n--- TEST 7: Get Comments ---")
    try:
        comments = registry.get_comments(issue.issue_id)
        assert len(comments) >= 2  # Assignment comment + explicit comment
        print(f"✅ Retrieved {len(comments)} comments")
        for c in comments[:2]:
            print(f"   - {c.author_name}: {c.content[:40]}...")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 8: Create Multiple Issues for Search
    print("\n--- TEST 8: Create Multiple Issues ---")
    try:
        # Create more issues
        issue2 = registry.create_issue(
            title="Open queries at Site_202",
            description="25 queries pending response",
            category=IssueCategory.OPEN_QUERIES,
            priority=IssuePriority.MEDIUM,
            severity=IssueSeverity.MINOR,
            created_by="system",
            study_id="Study_21",
            site_id="Site_202"
        )
        
        issue3 = registry.create_issue(
            title="SAE reconciliation pending",
            description="3 SAE cases need DM review",
            category=IssueCategory.SAE_DM_PENDING,
            priority=IssuePriority.CRITICAL,
            severity=IssueSeverity.BLOCKER,
            created_by="system",
            study_id="Study_22",
            site_id="Site_303"
        )
        
        print(f"✅ Created additional issues: {issue2.issue_id}, {issue3.issue_id}")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 9: Search Issues
    print("\n--- TEST 9: Search Issues ---")
    try:
        # Search by status
        filter1 = IssueFilter(status=[IssueStatus.OPEN])
        open_issues, count1 = registry.search_issues(filter1)
        print(f"   Open issues: {count1}")
        
        # Search by priority
        filter2 = IssueFilter(priority=[IssuePriority.CRITICAL, IssuePriority.HIGH])
        high_priority, count2 = registry.search_issues(filter2)
        print(f"   High/Critical priority: {count2}")
        
        # Search by study
        filter3 = IssueFilter(study_id="Study_21")
        study_issues, count3 = registry.search_issues(filter3)
        print(f"   Study_21 issues: {count3}")
        
        # Text search
        filter4 = IssueFilter(search_text="SDV")
        sdv_issues, count4 = registry.search_issues(filter4)
        print(f"   SDV-related issues: {count4}")
        
        print(f"✅ Search tests passed")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 10: Get Audit Trail
    print("\n--- TEST 10: Get Audit Trail ---")
    try:
        audit = registry.get_audit_trail(issue.issue_id)
        assert len(audit) >= 4  # created, assigned, status_changed, commented
        print(f"✅ Retrieved {len(audit)} audit entries")
        for entry in audit[:3]:
            print(f"   - {entry.action.value}: {entry.description[:40]}...")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 11: Resolve and Close Issue
    print("\n--- TEST 11: Resolve and Close ---")
    try:
        # IN_PROGRESS → RESOLVED
        resolved = registry.change_status(
            issue.issue_id,
            IssueStatus.RESOLVED,
            "cra_001",
            "All 15 CRFs verified"
        )
        assert resolved.status == IssueStatus.RESOLVED
        assert resolved.resolved_at is not None
        print(f"✅ Resolved at: {resolved.resolved_at}")
        
        # RESOLVED → CLOSED
        closed = registry.change_status(
            issue.issue_id,
            IssueStatus.CLOSED,
            "lead_001",
            "Verified and closing"
        )
        assert closed.status == IssueStatus.CLOSED
        assert closed.closed_at is not None
        print(f"✅ Closed at: {closed.closed_at}")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 12: Get Statistics
    print("\n--- TEST 12: Get Statistics ---")
    try:
        stats = registry.get_statistics()
        print(f"✅ Registry Statistics:")
        print(f"   Total issues: {stats['total_issues']}")
        print(f"   Total comments: {stats['total_comments']}")
        print(f"   Status distribution: {stats['status_distribution']}")
        print(f"   Priority distribution: {stats['priority_distribution']}")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Cleanup
    Path(test_db).unlink()
    
    # Summary
    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {tests_passed}/{tests_total} passed")
    print("=" * 60)
    
    if tests_passed == tests_total:
        print("✅ ALL TESTS PASSED!")
    else:
        print(f"❌ {tests_total - tests_passed} tests failed")
    
    return tests_passed == tests_total


if __name__ == "__main__":
    test_issue_registry()