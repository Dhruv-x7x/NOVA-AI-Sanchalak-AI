"""
a6on-i - Phase 8.5: Team Workspaces v1.0
# NOTE: This module previously used SQLite but has been migrated to PostgreSQL-only.
# If you need to use this module, update it to use:
#   from src.database.pg_data_service import get_pg_data_service
#   service = get_pg_data_service()


Team collaboration workspaces with:
- Workspace creation and management
- Member management with roles
- Goal tracking with progress
- Activity feed with real-time updates
- Shared resources and announcements

Author: a6on-i Team
Date: 2026-01-02
"""

# using SQLite for local persistence
import sqlite3
import json
import hashlib
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class WorkspaceType(Enum):
    """Types of workspaces."""
    STUDY = "study"
    REGION = "region"
    FUNCTIONAL = "functional"
    PROJECT = "project"
    COMMITTEE = "committee"
    AD_HOC = "ad_hoc"


class WorkspaceStatus(Enum):
    """Workspace status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ARCHIVED = "archived"


class MemberRole(Enum):
    """Workspace member roles."""
    OWNER = "owner"
    ADMIN = "admin"
    MODERATOR = "moderator"
    MEMBER = "member"
    GUEST = "guest"


class GoalStatus(Enum):
    """Goal status."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    AT_RISK = "at_risk"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    OVERDUE = "overdue"


class GoalPriority(Enum):
    """Goal priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ActivityType(Enum):
    """Activity types for feed."""
    WORKSPACE_CREATED = "workspace_created"
    WORKSPACE_UPDATED = "workspace_updated"
    MEMBER_JOINED = "member_joined"
    MEMBER_LEFT = "member_left"
    MEMBER_ROLE_CHANGED = "member_role_changed"
    GOAL_CREATED = "goal_created"
    GOAL_UPDATED = "goal_updated"
    GOAL_COMPLETED = "goal_completed"
    GOAL_PROGRESS = "goal_progress"
    ANNOUNCEMENT_POSTED = "announcement_posted"
    RESOURCE_SHARED = "resource_shared"
    DISCUSSION_STARTED = "discussion_started"
    COMMENT_ADDED = "comment_added"
    MILESTONE_REACHED = "milestone_reached"
    METRICS_UPDATED = "metrics_updated"


class ResourceType(Enum):
    """Shared resource types."""
    DOCUMENT = "document"
    LINK = "link"
    REPORT = "report"
    TEMPLATE = "template"
    DASHBOARD = "dashboard"
    DATA = "data"


# =============================================================================
# DATA CLASSES - All required fields before optional fields
# =============================================================================

@dataclass
class Workspace:
    """Workspace data class."""
    # Required fields first
    workspace_id: str
    name: str
    description: str
    workspace_type: WorkspaceType
    status: WorkspaceStatus
    created_by: str
    created_by_name: str
    created_at: datetime
    
    # Optional fields with defaults
    study_id: Optional[str] = None
    region: Optional[str] = None
    is_private: bool = False
    allow_guest_access: bool = True
    auto_archive_days: Optional[int] = None
    icon: str = "👥"
    color: str = "#3498db"
    tags: List[str] = field(default_factory=list)
    member_count: int = 0
    goal_count: int = 0
    active_goal_count: int = 0
    updated_at: Optional[datetime] = None
    last_activity_at: Optional[datetime] = None
    archived_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'workspace_id': self.workspace_id,
            'name': self.name,
            'description': self.description,
            'workspace_type': self.workspace_type.value,
            'status': self.status.value,
            'created_by': self.created_by,
            'created_by_name': self.created_by_name,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'study_id': self.study_id,
            'region': self.region,
            'is_private': self.is_private,
            'allow_guest_access': self.allow_guest_access,
            'auto_archive_days': self.auto_archive_days,
            'icon': self.icon,
            'color': self.color,
            'tags': self.tags,
            'member_count': self.member_count,
            'goal_count': self.goal_count,
            'active_goal_count': self.active_goal_count,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'last_activity_at': self.last_activity_at.isoformat() if self.last_activity_at else None
        }


@dataclass
class WorkspaceMember:
    """Workspace member data class."""
    # Required fields first
    member_id: str
    workspace_id: str
    user_id: str
    user_name: str
    role: MemberRole
    
    # Optional fields with defaults
    user_email: Optional[str] = None
    title: Optional[str] = None
    department: Optional[str] = None
    joined_at: datetime = field(default_factory=datetime.now)
    last_active_at: Optional[datetime] = None
    notification_preferences: Dict = field(default_factory=lambda: {
        'email': True,
        'in_app': True,
        'digest': 'daily'
    })
    is_active: bool = True
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'member_id': self.member_id,
            'workspace_id': self.workspace_id,
            'user_id': self.user_id,
            'user_name': self.user_name,
            'user_email': self.user_email,
            'role': self.role.value,
            'title': self.title,
            'department': self.department,
            'joined_at': self.joined_at.isoformat() if self.joined_at else None,
            'last_active_at': self.last_active_at.isoformat() if self.last_active_at else None,
            'notification_preferences': self.notification_preferences,
            'is_active': self.is_active
        }


@dataclass
class WorkspaceGoal:
    """Workspace goal data class."""
    # Required fields first
    goal_id: str
    workspace_id: str
    title: str
    description: str
    priority: GoalPriority
    status: GoalStatus
    target_value: float
    current_value: float
    unit: str
    start_date: datetime
    due_date: datetime
    created_by: str
    created_by_name: str
    
    # Optional fields with defaults
    completed_at: Optional[datetime] = None
    owner_id: Optional[str] = None
    owner_name: Optional[str] = None
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    milestones: List[Dict] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    
    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage."""
        if self.target_value == 0:
            return 0.0
        return min(100.0, (self.current_value / self.target_value) * 100)
    
    @property
    def is_overdue(self) -> bool:
        """Check if goal is overdue."""
        if self.status == GoalStatus.COMPLETED:
            return False
        return datetime.now() > self.due_date
    
    @property
    def days_remaining(self) -> int:
        """Days remaining until due date."""
        delta = self.due_date - datetime.now()
        return delta.days
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'goal_id': self.goal_id,
            'workspace_id': self.workspace_id,
            'title': self.title,
            'description': self.description,
            'priority': self.priority.value,
            'status': self.status.value,
            'target_value': self.target_value,
            'current_value': self.current_value,
            'unit': self.unit,
            'progress_percent': self.progress_percent,
            'start_date': self.start_date.isoformat() if self.start_date else None,
            'due_date': self.due_date.isoformat() if self.due_date else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'is_overdue': self.is_overdue,
            'days_remaining': self.days_remaining,
            'created_by': self.created_by,
            'created_by_name': self.created_by_name,
            'owner_id': self.owner_id,
            'owner_name': self.owner_name,
            'category': self.category,
            'tags': self.tags,
            'milestones': self.milestones
        }


@dataclass
class ActivityFeedItem:
    """Activity feed item data class."""
    # Required fields first
    activity_id: str
    workspace_id: str
    activity_type: ActivityType
    actor_id: str
    actor_name: str
    title: str
    
    # Optional fields with defaults
    description: str = ""
    entity_type: Optional[str] = None
    entity_id: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    is_pinned: bool = False
    is_highlighted: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'activity_id': self.activity_id,
            'workspace_id': self.workspace_id,
            'activity_type': self.activity_type.value,
            'actor_id': self.actor_id,
            'actor_name': self.actor_name,
            'title': self.title,
            'description': self.description,
            'entity_type': self.entity_type,
            'entity_id': self.entity_id,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'is_pinned': self.is_pinned,
            'is_highlighted': self.is_highlighted
        }


@dataclass
class Announcement:
    """Workspace announcement data class."""
    # Required fields first
    announcement_id: str
    workspace_id: str
    title: str
    content: str
    author_id: str
    author_name: str
    
    # Optional fields with defaults
    is_pinned: bool = False
    priority: str = "normal"
    expires_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    reactions: Dict[str, List[str]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'announcement_id': self.announcement_id,
            'workspace_id': self.workspace_id,
            'title': self.title,
            'content': self.content,
            'author_id': self.author_id,
            'author_name': self.author_name,
            'is_pinned': self.is_pinned,
            'priority': self.priority,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'reactions': self.reactions
        }


@dataclass
class SharedResource:
    """Shared resource data class."""
    # Required fields first
    resource_id: str
    workspace_id: str
    name: str
    description: str
    resource_type: ResourceType
    shared_by: str
    shared_by_name: str
    
    # Optional fields with defaults
    url: Optional[str] = None
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    mime_type: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    download_count: int = 0
    last_accessed_at: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'resource_id': self.resource_id,
            'workspace_id': self.workspace_id,
            'name': self.name,
            'description': self.description,
            'resource_type': self.resource_type.value,
            'url': self.url,
            'file_path': self.file_path,
            'file_size': self.file_size,
            'shared_by': self.shared_by,
            'shared_by_name': self.shared_by_name,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'download_count': self.download_count,
            'tags': self.tags
        }


@dataclass
class WorkspaceMetrics:
    """Workspace metrics snapshot."""
    # Required fields first
    metrics_id: str
    workspace_id: str
    
    # Optional fields with defaults
    total_patients: int = 0
    mean_dqi: float = 0.0
    clean_rate: float = 0.0
    dblock_ready_rate: float = 0.0
    open_issues: int = 0
    dqi_trend: str = "stable"
    clean_rate_trend: str = "stable"
    custom_metrics: Dict = field(default_factory=dict)
    captured_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'metrics_id': self.metrics_id,
            'workspace_id': self.workspace_id,
            'total_patients': self.total_patients,
            'mean_dqi': self.mean_dqi,
            'clean_rate': self.clean_rate,
            'dblock_ready_rate': self.dblock_ready_rate,
            'open_issues': self.open_issues,
            'dqi_trend': self.dqi_trend,
            'clean_rate_trend': self.clean_rate_trend,
            'custom_metrics': self.custom_metrics,
            'captured_at': self.captured_at.isoformat() if self.captured_at else None
        }


# =============================================================================
# TEAM WORKSPACES MANAGER
# =============================================================================

class TeamWorkspacesManager:
    """
    Team Workspaces Manager for collaborative team spaces.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize the Team Workspaces Manager."""
        if db_path is None:
            db_dir = Path("data/collaboration")
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(db_dir / "team_workspaces.db")
        
        self.db_path = db_path
        self._init_database()
        logger.info(f"TeamWorkspacesManager initialized with database: {db_path}")
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        return conn
    
    def _init_database(self):
        """Initialize database schema."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Workspaces table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS workspaces (
                workspace_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                workspace_type TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                created_by TEXT NOT NULL,
                created_by_name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                study_id TEXT,
                region TEXT,
                is_private INTEGER DEFAULT 0,
                allow_guest_access INTEGER DEFAULT 1,
                auto_archive_days INTEGER,
                icon TEXT DEFAULT '👥',
                color TEXT DEFAULT '#3498db',
                tags TEXT DEFAULT '[]',
                updated_at TEXT,
                last_activity_at TEXT,
                archived_at TEXT
            )
        """)
        
        # Members table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS workspace_members (
                member_id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                user_name TEXT NOT NULL,
                user_email TEXT,
                role TEXT NOT NULL DEFAULT 'member',
                title TEXT,
                department TEXT,
                joined_at TEXT NOT NULL,
                last_active_at TEXT,
                notification_preferences TEXT DEFAULT '{}',
                is_active INTEGER DEFAULT 1,
                FOREIGN KEY (workspace_id) REFERENCES workspaces(workspace_id),
                UNIQUE(workspace_id, user_id)
            )
        """)
        
        # Goals table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS workspace_goals (
                goal_id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                priority TEXT NOT NULL DEFAULT 'medium',
                status TEXT NOT NULL DEFAULT 'not_started',
                target_value REAL NOT NULL,
                current_value REAL DEFAULT 0,
                unit TEXT NOT NULL,
                start_date TEXT NOT NULL,
                due_date TEXT NOT NULL,
                completed_at TEXT,
                created_by TEXT NOT NULL,
                created_by_name TEXT NOT NULL,
                owner_id TEXT,
                owner_name TEXT,
                category TEXT DEFAULT 'general',
                tags TEXT DEFAULT '[]',
                milestones TEXT DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT,
                FOREIGN KEY (workspace_id) REFERENCES workspaces(workspace_id)
            )
        """)
        
        # Activity feed table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS activity_feed (
                activity_id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                activity_type TEXT NOT NULL,
                actor_id TEXT NOT NULL,
                actor_name TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                entity_type TEXT,
                entity_id TEXT,
                metadata TEXT DEFAULT '{}',
                created_at TEXT NOT NULL,
                is_pinned INTEGER DEFAULT 0,
                is_highlighted INTEGER DEFAULT 0,
                FOREIGN KEY (workspace_id) REFERENCES workspaces(workspace_id)
            )
        """)
        
        # Announcements table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS announcements (
                announcement_id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                author_id TEXT NOT NULL,
                author_name TEXT NOT NULL,
                is_pinned INTEGER DEFAULT 0,
                priority TEXT DEFAULT 'normal',
                expires_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT,
                reactions TEXT DEFAULT '{}',
                FOREIGN KEY (workspace_id) REFERENCES workspaces(workspace_id)
            )
        """)
        
        # Shared resources table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS shared_resources (
                resource_id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                resource_type TEXT NOT NULL,
                url TEXT,
                file_path TEXT,
                file_size INTEGER,
                mime_type TEXT,
                shared_by TEXT NOT NULL,
                shared_by_name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT,
                download_count INTEGER DEFAULT 0,
                last_accessed_at TEXT,
                tags TEXT DEFAULT '[]',
                FOREIGN KEY (workspace_id) REFERENCES workspaces(workspace_id)
            )
        """)
        
        # Workspace metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS workspace_metrics (
                metrics_id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                total_patients INTEGER DEFAULT 0,
                mean_dqi REAL DEFAULT 0,
                clean_rate REAL DEFAULT 0,
                dblock_ready_rate REAL DEFAULT 0,
                open_issues INTEGER DEFAULT 0,
                dqi_trend TEXT DEFAULT 'stable',
                clean_rate_trend TEXT DEFAULT 'stable',
                custom_metrics TEXT DEFAULT '{}',
                captured_at TEXT NOT NULL,
                FOREIGN KEY (workspace_id) REFERENCES workspaces(workspace_id)
            )
        """)
        
        # Audit trail table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS workspace_audit (
                audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                workspace_id TEXT NOT NULL,
                action TEXT NOT NULL,
                actor_id TEXT NOT NULL,
                actor_name TEXT,
                details TEXT,
                created_at TEXT NOT NULL,
                checksum TEXT
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_workspaces_status ON workspaces(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_workspaces_type ON workspaces(workspace_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_members_workspace ON workspace_members(workspace_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_members_user ON workspace_members(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_goals_workspace ON workspace_goals(workspace_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_goals_status ON workspace_goals(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_activity_workspace ON activity_feed(workspace_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_activity_created ON activity_feed(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_announcements_workspace ON announcements(workspace_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_resources_workspace ON shared_resources(workspace_id)")
        
        conn.commit()
        conn.close()
    
    def _generate_id(self, prefix: str) -> str:
        """Generate unique ID."""
        import time
        import random
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        # Add microseconds and random component for uniqueness
        micro = datetime.now().microsecond
        random_part = hashlib.md5(f"{timestamp}{micro}{random.randint(0, 999999)}{time.time_ns()}".encode()).hexdigest()[:8].upper()
        return f"{prefix}-{timestamp}-{random_part}"
    
    def _log_audit(self, conn: sqlite3.Connection, workspace_id: str, action: str,
                   actor_id: str, actor_name: str = None, details: Dict = None):
        """Log audit entry."""
        now = datetime.now().isoformat()
        details_json = json.dumps(details) if details else "{}"
        checksum = hashlib.sha256(f"{workspace_id}{action}{actor_id}{now}{details_json}".encode()).hexdigest()
        
        conn.execute("""
            INSERT INTO workspace_audit (workspace_id, action, actor_id, actor_name, details, created_at, checksum)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (workspace_id, action, actor_id, actor_name, details_json, now, checksum))
    
    def _add_activity(self, conn: sqlite3.Connection, workspace_id: str, 
                      activity_type: ActivityType, actor_id: str, actor_name: str,
                      title: str, description: str = None, entity_type: str = None,
                      entity_id: str = None, metadata: Dict = None,
                      is_pinned: bool = False, is_highlighted: bool = False) -> str:
        """Add activity to feed."""
        activity_id = self._generate_id("ACT")
        now = datetime.now().isoformat()
        
        conn.execute("""
            INSERT INTO activity_feed 
            (activity_id, workspace_id, activity_type, actor_id, actor_name, title, description,
             entity_type, entity_id, metadata, created_at, is_pinned, is_highlighted)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (activity_id, workspace_id, activity_type.value, actor_id, actor_name,
              title, description, entity_type, entity_id, 
              json.dumps(metadata) if metadata else "{}", now, 
              1 if is_pinned else 0, 1 if is_highlighted else 0))
        
        # Update workspace last activity
        conn.execute("""
            UPDATE workspaces SET last_activity_at = ? WHERE workspace_id = ?
        """, (now, workspace_id))
        
        return activity_id
    
    # =========================================================================
    # WORKSPACE MANAGEMENT
    # =========================================================================
    
    def create_workspace(self, name: str, description: str, 
                         workspace_type: WorkspaceType,
                         created_by: str, created_by_name: str,
                         study_id: str = None, region: str = None,
                         is_private: bool = False, allow_guest_access: bool = True,
                         icon: str = "👥", color: str = "#3498db",
                         tags: List[str] = None) -> Workspace:
        """Create a new workspace."""
        workspace_id = self._generate_id("WS")
        now = datetime.now()
        
        workspace = Workspace(
            workspace_id=workspace_id,
            name=name,
            description=description,
            workspace_type=workspace_type,
            status=WorkspaceStatus.ACTIVE,
            created_by=created_by,
            created_by_name=created_by_name,
            created_at=now,
            study_id=study_id,
            region=region,
            is_private=is_private,
            allow_guest_access=allow_guest_access,
            icon=icon,
            color=color,
            tags=tags or [],
            updated_at=now,
            last_activity_at=now
        )
        
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO workspaces 
                (workspace_id, name, description, workspace_type, status, created_by, created_by_name,
                 created_at, study_id, region, is_private, allow_guest_access, icon, color, tags,
                 updated_at, last_activity_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (workspace.workspace_id, workspace.name, workspace.description,
                  workspace.workspace_type.value, workspace.status.value,
                  workspace.created_by, workspace.created_by_name,
                  workspace.created_at.isoformat(), workspace.study_id, workspace.region,
                  1 if workspace.is_private else 0, 1 if workspace.allow_guest_access else 0,
                  workspace.icon, workspace.color, json.dumps(workspace.tags),
                  workspace.updated_at.isoformat(), workspace.last_activity_at.isoformat()))
            
            # Add creator as owner
            member_id = self._generate_id("MEM")
            cursor.execute("""
                INSERT INTO workspace_members
                (member_id, workspace_id, user_id, user_name, role, joined_at, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (member_id, workspace_id, created_by, created_by_name,
                  MemberRole.OWNER.value, now.isoformat(), 1))
            
            # Add activity
            self._add_activity(conn, workspace_id, ActivityType.WORKSPACE_CREATED,
                              created_by, created_by_name, 
                              f"Created workspace: {name}",
                              f"{created_by_name} created the workspace",
                              is_highlighted=True)
            
            # Log audit
            self._log_audit(conn, workspace_id, "workspace_created", created_by, 
                           created_by_name, {'name': name, 'type': workspace_type.value})
            
            conn.commit()
            workspace.member_count = 1
            logger.info(f"Created workspace: {workspace_id} - {name}")
            return workspace
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error creating workspace: {e}")
            raise
        finally:
            conn.close()
    
    def get_workspace(self, workspace_id: str) -> Optional[Workspace]:
        """Get workspace by ID."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM workspaces WHERE workspace_id = ?", (workspace_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            # Get counts
            cursor.execute("""
                SELECT COUNT(*) FROM workspace_members 
                WHERE workspace_id = ? AND is_active = 1
            """, (workspace_id,))
            member_count = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT COUNT(*), SUM(CASE WHEN status NOT IN ('completed', 'cancelled') THEN 1 ELSE 0 END)
                FROM workspace_goals WHERE workspace_id = ?
            """, (workspace_id,))
            goal_row = cursor.fetchone()
            goal_count = goal_row[0] or 0
            active_goal_count = goal_row[1] or 0
            
            return Workspace(
                workspace_id=row['workspace_id'],
                name=row['name'],
                description=row['description'],
                workspace_type=WorkspaceType(row['workspace_type']),
                status=WorkspaceStatus(row['status']),
                created_by=row['created_by'],
                created_by_name=row['created_by_name'],
                created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
                study_id=row['study_id'],
                region=row['region'],
                is_private=bool(row['is_private']),
                allow_guest_access=bool(row['allow_guest_access']),
                auto_archive_days=row['auto_archive_days'],
                icon=row['icon'],
                color=row['color'],
                tags=json.loads(row['tags']) if row['tags'] else [],
                member_count=member_count,
                goal_count=goal_count,
                active_goal_count=active_goal_count,
                updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None,
                last_activity_at=datetime.fromisoformat(row['last_activity_at']) if row['last_activity_at'] else None,
                archived_at=datetime.fromisoformat(row['archived_at']) if row['archived_at'] else None
            )
        finally:
            conn.close()
    
    def update_workspace(self, workspace_id: str, updated_by: str,
                         updated_by_name: str = None, **kwargs) -> Optional[Workspace]:
        """Update workspace properties."""
        conn = self._get_connection()
        try:
            allowed_fields = ['name', 'description', 'is_private', 'allow_guest_access',
                            'auto_archive_days', 'icon', 'color', 'tags']
            updates = []
            values = []
            
            for field in allowed_fields:
                if field in kwargs:
                    updates.append(f"{field} = ?")
                    value = kwargs[field]
                    if field == 'tags':
                        value = json.dumps(value)
                    elif field in ['is_private', 'allow_guest_access']:
                        value = 1 if value else 0
                    values.append(value)
            
            if not updates:
                return self.get_workspace(workspace_id)
            
            updates.append("updated_at = ?")
            values.append(datetime.now().isoformat())
            values.append(workspace_id)
            
            conn.execute(f"""
                UPDATE workspaces SET {', '.join(updates)} WHERE workspace_id = ?
            """, values)
            
            self._add_activity(conn, workspace_id, ActivityType.WORKSPACE_UPDATED,
                              updated_by, updated_by_name or updated_by,
                              "Workspace settings updated",
                              f"Updated: {', '.join(kwargs.keys())}")
            
            self._log_audit(conn, workspace_id, "workspace_updated", updated_by,
                           updated_by_name, {'changes': list(kwargs.keys())})
            
            conn.commit()
            return self.get_workspace(workspace_id)
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error updating workspace: {e}")
            raise
        finally:
            conn.close()
    
    def archive_workspace(self, workspace_id: str, archived_by: str,
                          archived_by_name: str = None) -> bool:
        """Archive a workspace."""
        conn = self._get_connection()
        try:
            now = datetime.now().isoformat()
            conn.execute("""
                UPDATE workspaces 
                SET status = ?, archived_at = ?, updated_at = ?
                WHERE workspace_id = ?
            """, (WorkspaceStatus.ARCHIVED.value, now, now, workspace_id))
            
            self._log_audit(conn, workspace_id, "workspace_archived", archived_by, archived_by_name)
            conn.commit()
            return True
        except Exception as e:
            conn.rollback()
            logger.error(f"Error archiving workspace: {e}")
            return False
        finally:
            conn.close()
    
    def list_workspaces(self, status: WorkspaceStatus = None, 
                        workspace_type: WorkspaceType = None,
                        study_id: str = None, region: str = None,
                        user_id: str = None, limit: int = 50,
                        offset: int = 0) -> Tuple[List[Workspace], int]:
        """List workspaces with filters."""
        conn = self._get_connection()
        try:
            conditions = []
            params = []
            
            if status:
                conditions.append("w.status = ?")
                params.append(status.value)
            
            if workspace_type:
                conditions.append("w.workspace_type = ?")
                params.append(workspace_type.value)
            
            if study_id:
                conditions.append("w.study_id = ?")
                params.append(study_id)
            
            if region:
                conditions.append("w.region = ?")
                params.append(region)
            
            if user_id:
                conditions.append("""
                    EXISTS (SELECT 1 FROM workspace_members m 
                            WHERE m.workspace_id = w.workspace_id 
                            AND m.user_id = ? AND m.is_active = 1)
                """)
                params.append(user_id)
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            cursor = conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM workspaces w WHERE {where_clause}", params)
            total = cursor.fetchone()[0]
            
            cursor.execute(f"""
                SELECT w.*, 
                       (SELECT COUNT(*) FROM workspace_members WHERE workspace_id = w.workspace_id AND is_active = 1) as member_count,
                       (SELECT COUNT(*) FROM workspace_goals WHERE workspace_id = w.workspace_id) as goal_count
                FROM workspaces w
                WHERE {where_clause}
                ORDER BY w.last_activity_at DESC NULLS LAST
                LIMIT ? OFFSET ?
            """, params + [limit, offset])
            
            workspaces = []
            for row in cursor.fetchall():
                workspaces.append(Workspace(
                    workspace_id=row['workspace_id'],
                    name=row['name'],
                    description=row['description'],
                    workspace_type=WorkspaceType(row['workspace_type']),
                    status=WorkspaceStatus(row['status']),
                    created_by=row['created_by'],
                    created_by_name=row['created_by_name'],
                    created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None,
                    study_id=row['study_id'],
                    region=row['region'],
                    is_private=bool(row['is_private']),
                    allow_guest_access=bool(row['allow_guest_access']),
                    icon=row['icon'],
                    color=row['color'],
                    tags=json.loads(row['tags']) if row['tags'] else [],
                    member_count=row['member_count'],
                    goal_count=row['goal_count'],
                    last_activity_at=datetime.fromisoformat(row['last_activity_at']) if row['last_activity_at'] else None
                ))
            
            return workspaces, total
            
        finally:
            conn.close()
    
    # =========================================================================
    # MEMBER MANAGEMENT
    # =========================================================================
    
    def add_member(self, workspace_id: str, user_id: str, user_name: str,
                   role: MemberRole = MemberRole.MEMBER,
                   added_by: str = None, added_by_name: str = None,
                   user_email: str = None, title: str = None,
                   department: str = None) -> WorkspaceMember:
        """Add a member to workspace."""
        member_id = self._generate_id("MEM")
        now = datetime.now()
        
        member = WorkspaceMember(
            member_id=member_id,
            workspace_id=workspace_id,
            user_id=user_id,
            user_name=user_name,
            role=role,
            user_email=user_email,
            title=title,
            department=department,
            joined_at=now,
            is_active=True
        )
        
        conn = self._get_connection()
        try:
            conn.execute("""
                INSERT INTO workspace_members
                (member_id, workspace_id, user_id, user_name, user_email, role, 
                 title, department, joined_at, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (member.member_id, member.workspace_id, member.user_id,
                  member.user_name, member.user_email, member.role.value,
                  member.title, member.department, member.joined_at.isoformat(), 1))
            
            self._add_activity(conn, workspace_id, ActivityType.MEMBER_JOINED,
                              user_id, user_name, f"{user_name} joined the workspace",
                              entity_type="member", entity_id=member_id)
            
            self._log_audit(conn, workspace_id, "member_added", added_by or "system",
                           added_by_name, {'user_id': user_id, 'role': role.value})
            
            conn.commit()
            logger.info(f"Added member {user_id} to workspace {workspace_id}")
            return member
            
        except sqlite3.IntegrityError:
            conn.rollback()
            conn.execute("""
                UPDATE workspace_members 
                SET is_active = 1, role = ?, joined_at = ?
                WHERE workspace_id = ? AND user_id = ?
            """, (role.value, now.isoformat(), workspace_id, user_id))
            conn.commit()
            return self.get_member(workspace_id, user_id)
        except Exception as e:
            conn.rollback()
            logger.error(f"Error adding member: {e}")
            raise
        finally:
            conn.close()
    
    def get_member(self, workspace_id: str, user_id: str) -> Optional[WorkspaceMember]:
        """Get a specific member."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM workspace_members 
                WHERE workspace_id = ? AND user_id = ? AND is_active = 1
            """, (workspace_id, user_id))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            return WorkspaceMember(
                member_id=row['member_id'],
                workspace_id=row['workspace_id'],
                user_id=row['user_id'],
                user_name=row['user_name'],
                role=MemberRole(row['role']),
                user_email=row['user_email'],
                title=row['title'],
                department=row['department'],
                joined_at=datetime.fromisoformat(row['joined_at']) if row['joined_at'] else None,
                last_active_at=datetime.fromisoformat(row['last_active_at']) if row['last_active_at'] else None,
                notification_preferences=json.loads(row['notification_preferences']) if row['notification_preferences'] else {},
                is_active=bool(row['is_active'])
            )
        finally:
            conn.close()
    
    def get_members(self, workspace_id: str, role: MemberRole = None,
                    include_inactive: bool = False) -> List[WorkspaceMember]:
        """Get all members of a workspace."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            conditions = ["workspace_id = ?"]
            params = [workspace_id]
            
            if not include_inactive:
                conditions.append("is_active = 1")
            
            if role:
                conditions.append("role = ?")
                params.append(role.value)
            
            cursor.execute(f"""
                SELECT * FROM workspace_members 
                WHERE {' AND '.join(conditions)}
                ORDER BY 
                    CASE role 
                        WHEN 'owner' THEN 1 
                        WHEN 'admin' THEN 2 
                        WHEN 'moderator' THEN 3 
                        WHEN 'member' THEN 4 
                        ELSE 5 
                    END,
                    user_name
            """, params)
            
            members = []
            for row in cursor.fetchall():
                members.append(WorkspaceMember(
                    member_id=row['member_id'],
                    workspace_id=row['workspace_id'],
                    user_id=row['user_id'],
                    user_name=row['user_name'],
                    role=MemberRole(row['role']),
                    user_email=row['user_email'],
                    title=row['title'],
                    department=row['department'],
                    joined_at=datetime.fromisoformat(row['joined_at']) if row['joined_at'] else None,
                    last_active_at=datetime.fromisoformat(row['last_active_at']) if row['last_active_at'] else None,
                    is_active=bool(row['is_active'])
                ))
            
            return members
        finally:
            conn.close()
    
    def update_member_role(self, workspace_id: str, user_id: str, 
                           new_role: MemberRole, updated_by: str,
                           updated_by_name: str = None) -> bool:
        """Update member role."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT role FROM workspace_members 
                WHERE workspace_id = ? AND user_id = ? AND is_active = 1
            """, (workspace_id, user_id))
            row = cursor.fetchone()
            
            if not row:
                return False
            
            old_role = row['role']
            
            conn.execute("""
                UPDATE workspace_members SET role = ? 
                WHERE workspace_id = ? AND user_id = ?
            """, (new_role.value, workspace_id, user_id))
            
            cursor.execute("SELECT user_name FROM workspace_members WHERE workspace_id = ? AND user_id = ?",
                          (workspace_id, user_id))
            user_name = cursor.fetchone()['user_name']
            
            self._add_activity(conn, workspace_id, ActivityType.MEMBER_ROLE_CHANGED,
                              updated_by, updated_by_name or updated_by,
                              f"{user_name}'s role changed to {new_role.value}",
                              entity_type="member", entity_id=user_id,
                              metadata={'old_role': old_role, 'new_role': new_role.value})
            
            self._log_audit(conn, workspace_id, "member_role_changed", updated_by,
                           updated_by_name, {'user_id': user_id, 'old_role': old_role,
                                             'new_role': new_role.value})
            
            conn.commit()
            return True
        except Exception as e:
            conn.rollback()
            logger.error(f"Error updating member role: {e}")
            return False
        finally:
            conn.close()
    
    def remove_member(self, workspace_id: str, user_id: str, 
                      removed_by: str, removed_by_name: str = None) -> bool:
        """Remove a member from workspace (soft delete)."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT user_name, role FROM workspace_members 
                WHERE workspace_id = ? AND user_id = ? AND is_active = 1
            """, (workspace_id, user_id))
            row = cursor.fetchone()
            
            if not row:
                return False
            
            user_name = row['user_name']
            
            if row['role'] == MemberRole.OWNER.value:
                logger.warning("Cannot remove workspace owner")
                return False
            
            conn.execute("""
                UPDATE workspace_members SET is_active = 0 
                WHERE workspace_id = ? AND user_id = ?
            """, (workspace_id, user_id))
            
            self._add_activity(conn, workspace_id, ActivityType.MEMBER_LEFT,
                              removed_by, removed_by_name or removed_by,
                              f"{user_name} left the workspace",
                              entity_type="member", entity_id=user_id)
            
            self._log_audit(conn, workspace_id, "member_removed", removed_by,
                           removed_by_name, {'user_id': user_id, 'user_name': user_name})
            
            conn.commit()
            return True
        except Exception as e:
            conn.rollback()
            logger.error(f"Error removing member: {e}")
            return False
        finally:
            conn.close()
    
    # =========================================================================
    # GOAL TRACKING
    # =========================================================================
    
    def create_goal(self, workspace_id: str, title: str, description: str,
                    target_value: float, unit: str, due_date: datetime,
                    created_by: str, created_by_name: str,
                    priority: GoalPriority = GoalPriority.MEDIUM,
                    start_date: datetime = None,
                    owner_id: str = None, owner_name: str = None,
                    category: str = "general", tags: List[str] = None,
                    milestones: List[Dict] = None) -> WorkspaceGoal:
        """Create a new goal."""
        goal_id = self._generate_id("GOAL")
        now = datetime.now()
        
        goal = WorkspaceGoal(
            goal_id=goal_id,
            workspace_id=workspace_id,
            title=title,
            description=description,
            priority=priority,
            status=GoalStatus.NOT_STARTED,
            target_value=target_value,
            current_value=0,
            unit=unit,
            start_date=start_date or now,
            due_date=due_date,
            created_by=created_by,
            created_by_name=created_by_name,
            owner_id=owner_id,
            owner_name=owner_name,
            category=category,
            tags=tags or [],
            milestones=milestones or [],
            created_at=now
        )
        
        conn = self._get_connection()
        try:
            conn.execute("""
                INSERT INTO workspace_goals
                (goal_id, workspace_id, title, description, priority, status,
                 target_value, current_value, unit, start_date, due_date,
                 created_by, created_by_name, owner_id, owner_name, category,
                 tags, milestones, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (goal.goal_id, goal.workspace_id, goal.title, goal.description,
                  goal.priority.value, goal.status.value, goal.target_value,
                  goal.current_value, goal.unit, goal.start_date.isoformat(),
                  goal.due_date.isoformat(), goal.created_by, goal.created_by_name,
                  goal.owner_id, goal.owner_name, goal.category,
                  json.dumps(goal.tags), json.dumps(goal.milestones),
                  goal.created_at.isoformat()))
            
            self._add_activity(conn, workspace_id, ActivityType.GOAL_CREATED,
                              created_by, created_by_name,
                              f"New goal: {title}",
                              f"Target: {target_value} {unit} by {due_date.strftime('%Y-%m-%d')}",
                              entity_type="goal", entity_id=goal_id)
            
            self._log_audit(conn, workspace_id, "goal_created", created_by,
                           created_by_name, {'goal_id': goal_id, 'title': title})
            
            conn.commit()
            logger.info(f"Created goal: {goal_id} - {title}")
            return goal
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error creating goal: {e}")
            raise
        finally:
            conn.close()
    
    def get_goal(self, goal_id: str) -> Optional[WorkspaceGoal]:
        """Get goal by ID."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM workspace_goals WHERE goal_id = ?", (goal_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            return WorkspaceGoal(
                goal_id=row['goal_id'],
                workspace_id=row['workspace_id'],
                title=row['title'],
                description=row['description'],
                priority=GoalPriority(row['priority']),
                status=GoalStatus(row['status']),
                target_value=row['target_value'],
                current_value=row['current_value'],
                unit=row['unit'],
                start_date=datetime.fromisoformat(row['start_date']) if row['start_date'] else datetime.now(),
                due_date=datetime.fromisoformat(row['due_date']) if row['due_date'] else datetime.now(),
                created_by=row['created_by'],
                created_by_name=row['created_by_name'],
                completed_at=datetime.fromisoformat(row['completed_at']) if row['completed_at'] else None,
                owner_id=row['owner_id'],
                owner_name=row['owner_name'],
                category=row['category'],
                tags=json.loads(row['tags']) if row['tags'] else [],
                milestones=json.loads(row['milestones']) if row['milestones'] else [],
                created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else datetime.now(),
                updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None
            )
        finally:
            conn.close()
    
    def get_goals(self, workspace_id: str, status: GoalStatus = None,
                  priority: GoalPriority = None, category: str = None,
                  include_completed: bool = True) -> List[WorkspaceGoal]:
        """Get goals for a workspace."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            conditions = ["workspace_id = ?"]
            params = [workspace_id]
            
            if status:
                conditions.append("status = ?")
                params.append(status.value)
            elif not include_completed:
                conditions.append("status NOT IN ('completed', 'cancelled')")
            
            if priority:
                conditions.append("priority = ?")
                params.append(priority.value)
            
            if category:
                conditions.append("category = ?")
                params.append(category)
            
            cursor.execute(f"""
                SELECT * FROM workspace_goals
                WHERE {' AND '.join(conditions)}
                ORDER BY 
                    CASE priority WHEN 'critical' THEN 1 WHEN 'high' THEN 2 
                         WHEN 'medium' THEN 3 ELSE 4 END,
                    due_date
            """, params)
            
            goals = []
            for row in cursor.fetchall():
                goals.append(WorkspaceGoal(
                    goal_id=row['goal_id'],
                    workspace_id=row['workspace_id'],
                    title=row['title'],
                    description=row['description'],
                    priority=GoalPriority(row['priority']),
                    status=GoalStatus(row['status']),
                    target_value=row['target_value'],
                    current_value=row['current_value'],
                    unit=row['unit'],
                    start_date=datetime.fromisoformat(row['start_date']) if row['start_date'] else datetime.now(),
                    due_date=datetime.fromisoformat(row['due_date']) if row['due_date'] else datetime.now(),
                    created_by=row['created_by'],
                    created_by_name=row['created_by_name'],
                    completed_at=datetime.fromisoformat(row['completed_at']) if row['completed_at'] else None,
                    owner_id=row['owner_id'],
                    owner_name=row['owner_name'],
                    category=row['category'],
                    tags=json.loads(row['tags']) if row['tags'] else [],
                    milestones=json.loads(row['milestones']) if row['milestones'] else []
                ))
            
            return goals
        finally:
            conn.close()
    
    def update_goal_progress(self, goal_id: str, current_value: float,
                             updated_by: str, updated_by_name: str = None,
                             notes: str = None) -> Optional[WorkspaceGoal]:
        """Update goal progress."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM workspace_goals WHERE goal_id = ?", (goal_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            workspace_id = row['workspace_id']
            old_value = row['current_value']
            target_value = row['target_value']
            title = row['title']
            unit = row['unit']
            old_status = row['status']
            
            progress_percent = (current_value / target_value * 100) if target_value > 0 else 0
            
            if current_value >= target_value:
                new_status = GoalStatus.COMPLETED
                completed_at = datetime.now().isoformat()
            elif datetime.fromisoformat(row['due_date']) < datetime.now():
                new_status = GoalStatus.OVERDUE
                completed_at = None
            elif progress_percent > 0:
                new_status = GoalStatus.IN_PROGRESS
                completed_at = None
            else:
                new_status = GoalStatus(old_status)
                completed_at = None
            
            now = datetime.now().isoformat()
            conn.execute("""
                UPDATE workspace_goals 
                SET current_value = ?, status = ?, updated_at = ?, completed_at = COALESCE(?, completed_at)
                WHERE goal_id = ?
            """, (current_value, new_status.value, now, completed_at, goal_id))
            
            if new_status == GoalStatus.COMPLETED and old_status != GoalStatus.COMPLETED.value:
                self._add_activity(conn, workspace_id, ActivityType.GOAL_COMPLETED,
                                  updated_by, updated_by_name or updated_by,
                                  f"🎉 Goal completed: {title}",
                                  f"Achieved {current_value} {unit}",
                                  entity_type="goal", entity_id=goal_id,
                                  is_highlighted=True)
            else:
                self._add_activity(conn, workspace_id, ActivityType.GOAL_PROGRESS,
                                  updated_by, updated_by_name or updated_by,
                                  f"Progress update: {title}",
                                  f"{old_value} → {current_value} {unit} ({progress_percent:.1f}%)",
                                  entity_type="goal", entity_id=goal_id,
                                  metadata={'old_value': old_value, 'new_value': current_value,
                                           'progress_percent': progress_percent})
            
            self._log_audit(conn, workspace_id, "goal_progress_updated", updated_by,
                           updated_by_name, {'goal_id': goal_id, 'old_value': old_value,
                                             'new_value': current_value})
            
            conn.commit()
            return self.get_goal(goal_id)
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error updating goal progress: {e}")
            raise
        finally:
            conn.close()
    
    def update_goal_status(self, goal_id: str, new_status: GoalStatus,
                           updated_by: str, updated_by_name: str = None) -> bool:
        """Update goal status."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT workspace_id, title, status FROM workspace_goals WHERE goal_id = ?", 
                          (goal_id,))
            row = cursor.fetchone()
            
            if not row:
                return False
            
            now = datetime.now().isoformat()
            completed_at = now if new_status == GoalStatus.COMPLETED else None
            
            conn.execute("""
                UPDATE workspace_goals 
                SET status = ?, updated_at = ?, completed_at = COALESCE(?, completed_at)
                WHERE goal_id = ?
            """, (new_status.value, now, completed_at, goal_id))
            
            activity_type = ActivityType.GOAL_COMPLETED if new_status == GoalStatus.COMPLETED else ActivityType.GOAL_UPDATED
            self._add_activity(conn, row['workspace_id'], activity_type,
                              updated_by, updated_by_name or updated_by,
                              f"Goal status changed: {row['title']}",
                              f"Status: {row['status']} → {new_status.value}",
                              entity_type="goal", entity_id=goal_id,
                              is_highlighted=new_status == GoalStatus.COMPLETED)
            
            conn.commit()
            return True
        except Exception as e:
            conn.rollback()
            logger.error(f"Error updating goal status: {e}")
            return False
        finally:
            conn.close()
    
    # =========================================================================
    # ACTIVITY FEED
    # =========================================================================
    
    def get_activity_feed(self, workspace_id: str, limit: int = 50,
                          offset: int = 0, activity_type: ActivityType = None,
                          include_pinned_first: bool = True) -> List[ActivityFeedItem]:
        """Get activity feed for workspace."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            conditions = ["workspace_id = ?"]
            params = [workspace_id]
            
            if activity_type:
                conditions.append("activity_type = ?")
                params.append(activity_type.value)
            
            order_clause = "ORDER BY is_pinned DESC, created_at DESC" if include_pinned_first else "ORDER BY created_at DESC"
            
            cursor.execute(f"""
                SELECT * FROM activity_feed
                WHERE {' AND '.join(conditions)}
                {order_clause}
                LIMIT ? OFFSET ?
            """, params + [limit, offset])
            
            activities = []
            for row in cursor.fetchall():
                activities.append(ActivityFeedItem(
                    activity_id=row['activity_id'],
                    workspace_id=row['workspace_id'],
                    activity_type=ActivityType(row['activity_type']),
                    actor_id=row['actor_id'],
                    actor_name=row['actor_name'],
                    title=row['title'],
                    description=row['description'] or "",
                    entity_type=row['entity_type'],
                    entity_id=row['entity_id'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else {},
                    created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else datetime.now(),
                    is_pinned=bool(row['is_pinned']),
                    is_highlighted=bool(row['is_highlighted'])
                ))
            
            return activities
        finally:
            conn.close()
    
    def pin_activity(self, activity_id: str, pinned_by: str) -> bool:
        """Pin an activity to top of feed."""
        conn = self._get_connection()
        try:
            conn.execute("UPDATE activity_feed SET is_pinned = 1 WHERE activity_id = ?", (activity_id,))
            conn.commit()
            return True
        except Exception as e:
            conn.rollback()
            logger.error(f"Error pinning activity: {e}")
            return False
        finally:
            conn.close()
    
    # =========================================================================
    # ANNOUNCEMENTS
    # =========================================================================
    
    def post_announcement(self, workspace_id: str, title: str, content: str,
                          author_id: str, author_name: str,
                          is_pinned: bool = False, priority: str = "normal",
                          expires_at: datetime = None) -> Announcement:
        """Post an announcement."""
        announcement_id = self._generate_id("ANN")
        now = datetime.now()
        
        announcement = Announcement(
            announcement_id=announcement_id,
            workspace_id=workspace_id,
            title=title,
            content=content,
            author_id=author_id,
            author_name=author_name,
            is_pinned=is_pinned,
            priority=priority,
            expires_at=expires_at,
            created_at=now
        )
        
        conn = self._get_connection()
        try:
            conn.execute("""
                INSERT INTO announcements
                (announcement_id, workspace_id, title, content, author_id, author_name,
                 is_pinned, priority, expires_at, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (announcement.announcement_id, announcement.workspace_id,
                  announcement.title, announcement.content, announcement.author_id,
                  announcement.author_name, 1 if announcement.is_pinned else 0,
                  announcement.priority,
                  announcement.expires_at.isoformat() if announcement.expires_at else None,
                  announcement.created_at.isoformat()))
            
            self._add_activity(conn, workspace_id, ActivityType.ANNOUNCEMENT_POSTED,
                              author_id, author_name,
                              f"📢 {title}",
                              content[:100] + "..." if len(content) > 100 else content,
                              entity_type="announcement", entity_id=announcement_id,
                              is_highlighted=priority in ['urgent', 'high'])
            
            self._log_audit(conn, workspace_id, "announcement_posted", author_id,
                           author_name, {'announcement_id': announcement_id, 'title': title})
            
            conn.commit()
            return announcement
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error posting announcement: {e}")
            raise
        finally:
            conn.close()
    
    def get_announcements(self, workspace_id: str, include_expired: bool = False,
                          limit: int = 20) -> List[Announcement]:
        """Get announcements for workspace."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            conditions = ["workspace_id = ?"]
            params = [workspace_id]
            
            if not include_expired:
                conditions.append("(expires_at IS NULL OR expires_at > ?)")
                params.append(datetime.now().isoformat())
            
            cursor.execute(f"""
                SELECT * FROM announcements
                WHERE {' AND '.join(conditions)}
                ORDER BY is_pinned DESC, 
                         CASE priority WHEN 'urgent' THEN 1 WHEN 'high' THEN 2 ELSE 3 END,
                         created_at DESC
                LIMIT ?
            """, params + [limit])
            
            announcements = []
            for row in cursor.fetchall():
                announcements.append(Announcement(
                    announcement_id=row['announcement_id'],
                    workspace_id=row['workspace_id'],
                    title=row['title'],
                    content=row['content'],
                    author_id=row['author_id'],
                    author_name=row['author_name'],
                    is_pinned=bool(row['is_pinned']),
                    priority=row['priority'],
                    expires_at=datetime.fromisoformat(row['expires_at']) if row['expires_at'] else None,
                    created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else datetime.now(),
                    reactions=json.loads(row['reactions']) if row['reactions'] else {}
                ))
            
            return announcements
        finally:
            conn.close()
    
    # =========================================================================
    # SHARED RESOURCES
    # =========================================================================
    
    def share_resource(self, workspace_id: str, name: str, description: str,
                       resource_type: ResourceType, shared_by: str, shared_by_name: str,
                       url: str = None, file_path: str = None,
                       tags: List[str] = None) -> SharedResource:
        """Share a resource in workspace."""
        resource_id = self._generate_id("RES")
        now = datetime.now()
        
        resource = SharedResource(
            resource_id=resource_id,
            workspace_id=workspace_id,
            name=name,
            description=description,
            resource_type=resource_type,
            shared_by=shared_by,
            shared_by_name=shared_by_name,
            url=url,
            file_path=file_path,
            created_at=now,
            tags=tags or []
        )
        
        conn = self._get_connection()
        try:
            conn.execute("""
                INSERT INTO shared_resources
                (resource_id, workspace_id, name, description, resource_type,
                 url, file_path, shared_by, shared_by_name, created_at, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (resource.resource_id, resource.workspace_id, resource.name,
                  resource.description, resource.resource_type.value,
                  resource.url, resource.file_path, resource.shared_by,
                  resource.shared_by_name, resource.created_at.isoformat(),
                  json.dumps(resource.tags)))
            
            self._add_activity(conn, workspace_id, ActivityType.RESOURCE_SHARED,
                              shared_by, shared_by_name,
                              f"Shared: {name}",
                              description,
                              entity_type="resource", entity_id=resource_id)
            
            conn.commit()
            return resource
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error sharing resource: {e}")
            raise
        finally:
            conn.close()
    
    def get_resources(self, workspace_id: str, resource_type: ResourceType = None,
                      limit: int = 50) -> List[SharedResource]:
        """Get shared resources for workspace."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            conditions = ["workspace_id = ?"]
            params = [workspace_id]
            
            if resource_type:
                conditions.append("resource_type = ?")
                params.append(resource_type.value)
            
            cursor.execute(f"""
                SELECT * FROM shared_resources
                WHERE {' AND '.join(conditions)}
                ORDER BY created_at DESC
                LIMIT ?
            """, params + [limit])
            
            resources = []
            for row in cursor.fetchall():
                resources.append(SharedResource(
                    resource_id=row['resource_id'],
                    workspace_id=row['workspace_id'],
                    name=row['name'],
                    description=row['description'],
                    resource_type=ResourceType(row['resource_type']),
                    shared_by=row['shared_by'],
                    shared_by_name=row['shared_by_name'],
                    url=row['url'],
                    file_path=row['file_path'],
                    created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else datetime.now(),
                    download_count=row['download_count'],
                    tags=json.loads(row['tags']) if row['tags'] else []
                ))
            
            return resources
        finally:
            conn.close()
    
    # =========================================================================
    # METRICS
    # =========================================================================
    
    def update_metrics(self, workspace_id: str, total_patients: int = None,
                       mean_dqi: float = None, clean_rate: float = None,
                       dblock_ready_rate: float = None, open_issues: int = None,
                       custom_metrics: Dict = None) -> WorkspaceMetrics:
        """Update workspace metrics snapshot."""
        metrics_id = self._generate_id("MET")
        now = datetime.now()
        
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT mean_dqi, clean_rate FROM workspace_metrics
                WHERE workspace_id = ?
                ORDER BY captured_at DESC LIMIT 1
            """, (workspace_id,))
            prev = cursor.fetchone()
            
            dqi_trend = "stable"
            clean_rate_trend = "stable"
            
            if prev and mean_dqi is not None:
                if mean_dqi > prev['mean_dqi'] + 1:
                    dqi_trend = "improving"
                elif mean_dqi < prev['mean_dqi'] - 1:
                    dqi_trend = "declining"
            
            if prev and clean_rate is not None:
                if clean_rate > prev['clean_rate'] + 1:
                    clean_rate_trend = "improving"
                elif clean_rate < prev['clean_rate'] - 1:
                    clean_rate_trend = "declining"
            
            metrics = WorkspaceMetrics(
                metrics_id=metrics_id,
                workspace_id=workspace_id,
                total_patients=total_patients or 0,
                mean_dqi=mean_dqi or 0.0,
                clean_rate=clean_rate or 0.0,
                dblock_ready_rate=dblock_ready_rate or 0.0,
                open_issues=open_issues or 0,
                dqi_trend=dqi_trend,
                clean_rate_trend=clean_rate_trend,
                custom_metrics=custom_metrics or {},
                captured_at=now
            )
            
            conn.execute("""
                INSERT INTO workspace_metrics
                (metrics_id, workspace_id, total_patients, mean_dqi, clean_rate,
                 dblock_ready_rate, open_issues, dqi_trend, clean_rate_trend,
                 custom_metrics, captured_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (metrics.metrics_id, metrics.workspace_id, metrics.total_patients,
                  metrics.mean_dqi, metrics.clean_rate, metrics.dblock_ready_rate,
                  metrics.open_issues, metrics.dqi_trend, metrics.clean_rate_trend,
                  json.dumps(metrics.custom_metrics), metrics.captured_at.isoformat()))
            
            if dqi_trend != "stable" or clean_rate_trend != "stable":
                self._add_activity(conn, workspace_id, ActivityType.METRICS_UPDATED,
                                  "system", "System",
                                  "Metrics updated",
                                  f"DQI: {mean_dqi:.1f} ({dqi_trend}), Clean: {clean_rate:.1f}% ({clean_rate_trend})")
            
            conn.commit()
            return metrics
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error updating metrics: {e}")
            raise
        finally:
            conn.close()
    
    def get_latest_metrics(self, workspace_id: str) -> Optional[WorkspaceMetrics]:
        """Get latest metrics for workspace."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM workspace_metrics
                WHERE workspace_id = ?
                ORDER BY captured_at DESC LIMIT 1
            """, (workspace_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            return WorkspaceMetrics(
                metrics_id=row['metrics_id'],
                workspace_id=row['workspace_id'],
                total_patients=row['total_patients'],
                mean_dqi=row['mean_dqi'],
                clean_rate=row['clean_rate'],
                dblock_ready_rate=row['dblock_ready_rate'],
                open_issues=row['open_issues'],
                dqi_trend=row['dqi_trend'],
                clean_rate_trend=row['clean_rate_trend'],
                custom_metrics=json.loads(row['custom_metrics']) if row['custom_metrics'] else {},
                captured_at=datetime.fromisoformat(row['captured_at']) if row['captured_at'] else datetime.now()
            )
        finally:
            conn.close()
    
    def get_metrics_history(self, workspace_id: str, days: int = 30) -> List[WorkspaceMetrics]:
        """Get metrics history for workspace."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            since = (datetime.now() - timedelta(days=days)).isoformat()
            
            cursor.execute("""
                SELECT * FROM workspace_metrics
                WHERE workspace_id = ? AND captured_at >= ?
                ORDER BY captured_at ASC
            """, (workspace_id, since))
            
            metrics_list = []
            for row in cursor.fetchall():
                metrics_list.append(WorkspaceMetrics(
                    metrics_id=row['metrics_id'],
                    workspace_id=row['workspace_id'],
                    total_patients=row['total_patients'],
                    mean_dqi=row['mean_dqi'],
                    clean_rate=row['clean_rate'],
                    dblock_ready_rate=row['dblock_ready_rate'],
                    open_issues=row['open_issues'],
                    dqi_trend=row['dqi_trend'],
                    clean_rate_trend=row['clean_rate_trend'],
                                        custom_metrics=json.loads(row['custom_metrics']) if row['custom_metrics'] else {},
                    captured_at=datetime.fromisoformat(row['captured_at']) if row['captured_at'] else datetime.now()
                ))
            
            return metrics_list
        finally:
            conn.close()
    
    # =========================================================================
    # STATISTICS
    # =========================================================================
    
    def get_statistics(self) -> Dict:
        """Get overall statistics."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT status, COUNT(*) as count FROM workspaces GROUP BY status
            """)
            workspace_by_status = {row['status']: row['count'] for row in cursor.fetchall()}
            
            cursor.execute("""
                SELECT workspace_type, COUNT(*) as count FROM workspaces 
                WHERE status = 'active' GROUP BY workspace_type
            """)
            workspace_by_type = {row['workspace_type']: row['count'] for row in cursor.fetchall()}
            
            cursor.execute("SELECT COUNT(DISTINCT user_id) FROM workspace_members WHERE is_active = 1")
            total_members = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT status, COUNT(*) as count FROM workspace_goals GROUP BY status
            """)
            goals_by_status = {row['status']: row['count'] for row in cursor.fetchall()}
            
            week_ago = (datetime.now() - timedelta(days=7)).isoformat()
            cursor.execute("""
                SELECT COUNT(*) FROM activity_feed WHERE created_at >= ?
            """, (week_ago,))
            recent_activities = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT COUNT(*) FROM announcements 
                WHERE expires_at IS NULL OR expires_at > ?
            """, (datetime.now().isoformat(),))
            active_announcements = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM shared_resources")
            total_resources = cursor.fetchone()[0]
            
            return {
                'workspaces': {
                    'by_status': workspace_by_status,
                    'by_type': workspace_by_type,
                    'total_active': workspace_by_status.get('active', 0)
                },
                'members': {
                    'total_unique': total_members
                },
                'goals': {
                    'by_status': goals_by_status,
                    'total': sum(goals_by_status.values()) if goals_by_status else 0,
                    'active': goals_by_status.get('not_started', 0) + 
                              goals_by_status.get('in_progress', 0) +
                              goals_by_status.get('at_risk', 0)
                },
                'activities': {
                    'last_7_days': recent_activities
                },
                'announcements': {
                    'active': active_announcements
                },
                'resources': {
                    'total': total_resources
                }
            }
        finally:
            conn.close()
    
    def get_workspace_summary(self, workspace_id: str) -> Optional[Dict]:
        """Get comprehensive workspace summary."""
        workspace = self.get_workspace(workspace_id)
        if not workspace:
            return None
        
        members = self.get_members(workspace_id)
        goals = self.get_goals(workspace_id, include_completed=False)
        activities = self.get_activity_feed(workspace_id, limit=10)
        announcements = self.get_announcements(workspace_id, limit=5)
        resources = self.get_resources(workspace_id, limit=10)
        metrics = self.get_latest_metrics(workspace_id)
        
        goals_summary = {
            'total': len(goals),
            'completed': len([g for g in goals if g.status == GoalStatus.COMPLETED]),
            'in_progress': len([g for g in goals if g.status == GoalStatus.IN_PROGRESS]),
            'at_risk': len([g for g in goals if g.status == GoalStatus.AT_RISK]),
            'overdue': len([g for g in goals if g.is_overdue])
        }
        
        member_roles = {}
        for m in members:
            role = m.role.value
            member_roles[role] = member_roles.get(role, 0) + 1
        
        return {
            'workspace': workspace.to_dict(),
            'members': {
                'total': len(members),
                'by_role': member_roles,
                'list': [m.to_dict() for m in members[:10]]
            },
            'goals': {
                'summary': goals_summary,
                'list': [g.to_dict() for g in goals[:5]]
            },
            'activities': [a.to_dict() for a in activities],
            'announcements': [a.to_dict() for a in announcements],
            'resources': [r.to_dict() for r in resources],
            'metrics': metrics.to_dict() if metrics else None
        }
    
    # =========================================================================
    # AUDIT TRAIL
    # =========================================================================
    
    def get_audit_trail(self, workspace_id: str, limit: int = 100) -> List[Dict]:
        """Get audit trail for workspace."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM workspace_audit
                WHERE workspace_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (workspace_id, limit))
            
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_team_workspaces_manager: Optional[TeamWorkspacesManager] = None


def get_team_workspaces_manager(db_path: str = None) -> TeamWorkspacesManager:
    """Get singleton instance of TeamWorkspacesManager."""
    global _team_workspaces_manager
    if _team_workspaces_manager is None:
        _team_workspaces_manager = TeamWorkspacesManager(db_path)
    return _team_workspaces_manager


def reset_team_workspaces_manager():
    """Reset singleton instance (for testing)."""
    global _team_workspaces_manager
    _team_workspaces_manager = None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_workspace(name: str, description: str, workspace_type: WorkspaceType,
                     created_by: str, created_by_name: str, **kwargs) -> Workspace:
    """Convenience function to create a workspace."""
    return get_team_workspaces_manager().create_workspace(
        name, description, workspace_type, created_by, created_by_name, **kwargs
    )


def get_workspace(workspace_id: str) -> Optional[Workspace]:
    """Convenience function to get a workspace."""
    return get_team_workspaces_manager().get_workspace(workspace_id)


def list_user_workspaces(user_id: str) -> List[Workspace]:
    """Get all workspaces for a user."""
    workspaces, _ = get_team_workspaces_manager().list_workspaces(
        user_id=user_id, status=WorkspaceStatus.ACTIVE
    )
    return workspaces


def get_workspace_stats() -> Dict:
    """Get workspace statistics."""
    return get_team_workspaces_manager().get_statistics()
