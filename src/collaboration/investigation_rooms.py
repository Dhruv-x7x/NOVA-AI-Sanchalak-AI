# src/collaboration/investigation_rooms.py
"""
SANCHALAK AI - Investigation Rooms v1.0
Phase 8.2: Collaborative Investigation System
# NOTE: This module previously used SQLite but has been migrated to PostgreSQL-only.
# If you need to use this module, update it to use:
#   from src.database.pg_data_service import get_pg_data_service
#   service = get_pg_data_service()


Features:
- Room creation workflow (auto or manual)
- Context auto-population from 9 data sources
- Timeline reconstruction
- Threaded discussions with @mentions
- Evidence pinning with strength levels
- Root cause voting
- AI-generated running summary
- Resolution tracking with audit
"""

# using SQLite for local persistence
import sqlite3
import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
import pandas as pd

# Import from Issue Registry
from .issue_registry import (
    IssueRegistry, Issue, IssueStatus, IssuePriority, 
    IssueSeverity, IssueCategory, get_issue_registry
)


# =============================================================================
# ENUMS
# =============================================================================

class RoomStatus(Enum):
    """Investigation room lifecycle status"""
    DRAFT = "draft"                    # Room created, not started
    ACTIVE = "active"                  # Investigation in progress
    PENDING_REVIEW = "pending_review"  # Findings ready for review
    REVIEW_COMPLETE = "review_complete"  # Review done, pending closure
    CLOSED = "closed"                  # Investigation complete
    ARCHIVED = "archived"              # Historical reference


class RoomType(Enum):
    """Types of investigation rooms"""
    ISSUE = "issue"                    # Single issue investigation
    PATTERN = "pattern"                # Pattern-based (multiple related issues)
    SITE = "site"                      # Site-level investigation
    STUDY = "study"                    # Study-level investigation
    SAFETY = "safety"                  # Safety signal investigation
    AUDIT = "audit"                    # Audit/inspection prep
    AD_HOC = "ad_hoc"                  # Manual/ad-hoc investigation


class ParticipantRole(Enum):
    """Roles within investigation room"""
    OWNER = "owner"                    # Room creator, full control
    LEAD = "lead"                      # Investigation lead
    CONTRIBUTOR = "contributor"        # Can add evidence, comments
    REVIEWER = "reviewer"              # Can review and approve
    OBSERVER = "observer"              # Read-only access


class EvidenceType(Enum):
    """Types of evidence that can be pinned"""
    DATA_POINT = "data_point"          # Specific data from source
    DOCUMENT = "document"              # Attached document
    SCREENSHOT = "screenshot"          # Visual evidence
    QUERY = "query"                    # Query/response chain
    TIMELINE_EVENT = "timeline_event"  # Event in timeline
    PATTERN_MATCH = "pattern_match"    # Pattern library match
    HYPOTHESIS = "hypothesis"          # Diagnostic hypothesis
    EXTERNAL = "external"              # External reference


class EvidenceStrength(Enum):
    """Evidence strength levels"""
    STRONG = "strong"                  # High confidence, verified
    MODERATE = "moderate"              # Medium confidence
    WEAK = "weak"                      # Low confidence, needs verification
    INCONCLUSIVE = "inconclusive"      # Cannot determine
    CONTRADICTORY = "contradictory"    # Contradicts other evidence


class ThreadType(Enum):
    """Types of discussion threads"""
    GENERAL = "general"                # General discussion
    HYPOTHESIS = "hypothesis"          # Hypothesis discussion
    EVIDENCE = "evidence"              # Evidence-related
    ACTION = "action"                  # Action items
    DECISION = "decision"              # Decision points
    QUESTION = "question"              # Questions/clarifications


class VoteType(Enum):
    """Types of votes"""
    ROOT_CAUSE = "root_cause"          # Vote on root cause
    HYPOTHESIS = "hypothesis"          # Vote on hypothesis validity
    ACTION = "action"                  # Vote on proposed action
    CLOSURE = "closure"                # Vote to close investigation


class ResolutionType(Enum):
    """Types of resolution"""
    ROOT_CAUSE_IDENTIFIED = "root_cause_identified"
    CORRECTIVE_ACTION = "corrective_action"
    PREVENTIVE_ACTION = "preventive_action"
    NO_ACTION_REQUIRED = "no_action_required"
    ESCALATED = "escalated"
    INCONCLUSIVE = "inconclusive"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class RoomParticipant:
    """Participant in investigation room"""
    participant_id: str
    room_id: str
    user_id: str
    user_name: str
    role: ParticipantRole
    joined_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    notifications_enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['role'] = self.role.value
        d['joined_at'] = self.joined_at.isoformat()
        d['last_active'] = self.last_active.isoformat()
        return d


@dataclass
class RoomContext:
    """Auto-populated context for investigation"""
    context_id: str
    room_id: str
    source: str                        # Data source name
    entity_type: str                   # patient, site, study
    entity_id: str                     # Specific entity ID
    data_snapshot: Dict[str, Any]      # Snapshot of relevant data
    retrieved_at: datetime = field(default_factory=datetime.now)
    is_stale: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['retrieved_at'] = self.retrieved_at.isoformat()
        return d


@dataclass
class TimelineEvent:
    """Event in investigation timeline"""
    event_id: str
    room_id: str
    event_type: str                    # issue_created, status_changed, etc.
    event_time: datetime
    title: str
    description: str
    source: str                        # Where event came from
    entity_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_key_event: bool = False         # Highlighted in timeline
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['event_time'] = self.event_time.isoformat()
        return d


@dataclass
class Evidence:
    """Pinned evidence in investigation"""
    evidence_id: str
    room_id: str
    evidence_type: EvidenceType
    title: str
    description: str
    source: str                        # Where evidence came from
    source_id: Optional[str] = None    # Reference to source record
    data: Dict[str, Any] = field(default_factory=dict)
    strength: EvidenceStrength = EvidenceStrength.MODERATE
    pinned_by: str = ""
    pinned_by_name: str = ""
    pinned_at: datetime = field(default_factory=datetime.now)
    is_supporting: bool = True         # Supporting vs contradicting
    linked_hypothesis: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['evidence_type'] = self.evidence_type.value
        d['strength'] = self.strength.value
        d['pinned_at'] = self.pinned_at.isoformat()
        return d


@dataclass
class DiscussionThread:
    """Discussion thread in room"""
    thread_id: str
    room_id: str
    thread_type: ThreadType
    title: str
    created_by: str
    created_by_name: str
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    is_resolved: bool = False
    is_pinned: bool = False
    message_count: int = 0
    participants: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['thread_type'] = self.thread_type.value
        d['created_at'] = self.created_at.isoformat()
        d['updated_at'] = self.updated_at.isoformat()
        return d


@dataclass
class ThreadMessage:
    """Message in discussion thread"""
    message_id: str
    thread_id: str
    room_id: str
    author_id: str
    author_name: str
    content: str
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    is_edited: bool = False
    reply_to: Optional[str] = None     # Parent message ID for replies
    mentions: List[str] = field(default_factory=list)
    reactions: Dict[str, List[str]] = field(default_factory=dict)  # emoji: [user_ids]
    attachments: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['created_at'] = self.created_at.isoformat()
        if d['updated_at']:
            d['updated_at'] = d['updated_at'].isoformat()
        return d


@dataclass
class RootCauseVote:
    """Vote on root cause hypothesis"""
    vote_id: str
    room_id: str
    vote_type: VoteType
    hypothesis_id: str
    hypothesis_text: str
    voter_id: str
    voter_name: str
    vote: str                          # agree, disagree, abstain
    confidence: float = 0.0            # 0-1 confidence in vote
    comment: str = ""
    voted_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['vote_type'] = self.vote_type.value
        d['voted_at'] = self.voted_at.isoformat()
        return d


@dataclass
class Resolution:
    """Investigation resolution"""
    resolution_id: str
    room_id: str
    resolution_type: ResolutionType
    root_cause: str
    root_cause_confidence: float       # 0-1
    findings_summary: str
    recommendations: List[str]
    action_items: List[Dict[str, Any]]
    lessons_learned: List[str]
    resolved_by: str
    resolved_by_name: str
    resolved_at: datetime = field(default_factory=datetime.now)
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    linked_issues: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['resolution_type'] = self.resolution_type.value
        d['resolved_at'] = self.resolved_at.isoformat()
        if d['approved_at']:
            d['approved_at'] = d['approved_at'].isoformat()
        return d


@dataclass
class InvestigationRoom:
    """Investigation room model"""
    room_id: str
    title: str
    description: str
    room_type: RoomType
    status: RoomStatus
    
    # Related entities
    issue_id: Optional[str] = None
    study_id: Optional[str] = None
    site_id: Optional[str] = None
    patient_key: Optional[str] = None
    
    # Ownership
    created_by: str = ""
    created_by_name: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    # Priority
    priority: IssuePriority = IssuePriority.MEDIUM
    
    # Counts
    participant_count: int = 0
    evidence_count: int = 0
    thread_count: int = 0
    message_count: int = 0
    
    # Resolution
    resolution_id: Optional[str] = None
    closed_at: Optional[datetime] = None
    
    # AI Summary
    ai_summary: str = ""
    ai_summary_updated: Optional[datetime] = None
    
    # Tags
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['room_type'] = self.room_type.value
        d['status'] = self.status.value
        d['priority'] = self.priority.value
        d['created_at'] = self.created_at.isoformat()
        d['updated_at'] = self.updated_at.isoformat()
        if d['closed_at']:
            d['closed_at'] = d['closed_at'].isoformat()
        if d['ai_summary_updated']:
            d['ai_summary_updated'] = d['ai_summary_updated'].isoformat()
        return d


# =============================================================================
# INVESTIGATION ROOMS MANAGER
# =============================================================================

class InvestigationRoomsManager:
    """
    Investigation Rooms Manager
    
    Features:
    - Room lifecycle management
    - Context auto-population
    - Timeline reconstruction
    - Threaded discussions
    - Evidence management
    - Voting system
    - Resolution tracking
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize Investigation Rooms Manager"""
        if db_path is None:
            db_path = "data/collaboration/investigation_rooms.db"
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Data loader for context
        self.data_loader = InvestigationDataLoader()
        
        # Issue registry reference
        self.issue_registry = get_issue_registry()
        
        # Statistics
        self.stats = {
            'rooms_created': 0,
            'evidence_pinned': 0,
            'threads_created': 0,
            'messages_posted': 0,
            'resolutions_completed': 0,
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
        
        # Investigation Rooms table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS rooms (
                room_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                description TEXT,
                room_type TEXT NOT NULL,
                status TEXT NOT NULL,
                issue_id TEXT,
                study_id TEXT,
                site_id TEXT,
                patient_key TEXT,
                created_by TEXT NOT NULL,
                created_by_name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                priority TEXT DEFAULT 'medium',
                participant_count INTEGER DEFAULT 0,
                evidence_count INTEGER DEFAULT 0,
                thread_count INTEGER DEFAULT 0,
                message_count INTEGER DEFAULT 0,
                resolution_id TEXT,
                closed_at TEXT,
                ai_summary TEXT DEFAULT '',
                ai_summary_updated TEXT,
                tags TEXT DEFAULT '[]'
            )
        ''')
        
        # Participants table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS participants (
                participant_id TEXT PRIMARY KEY,
                room_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                user_name TEXT NOT NULL,
                role TEXT NOT NULL,
                joined_at TEXT NOT NULL,
                last_active TEXT NOT NULL,
                is_active INTEGER DEFAULT 1,
                notifications_enabled INTEGER DEFAULT 1,
                FOREIGN KEY (room_id) REFERENCES rooms(room_id)
            )
        ''')
        
        # Context table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS room_context (
                context_id TEXT PRIMARY KEY,
                room_id TEXT NOT NULL,
                source TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                data_snapshot TEXT NOT NULL,
                retrieved_at TEXT NOT NULL,
                is_stale INTEGER DEFAULT 0,
                FOREIGN KEY (room_id) REFERENCES rooms(room_id)
            )
        ''')
        
        # Timeline events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS timeline_events (
                event_id TEXT PRIMARY KEY,
                room_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                event_time TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                source TEXT,
                entity_id TEXT,
                metadata TEXT DEFAULT '{}',
                is_key_event INTEGER DEFAULT 0,
                FOREIGN KEY (room_id) REFERENCES rooms(room_id)
            )
        ''')
        
        # Evidence table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS evidence (
                evidence_id TEXT PRIMARY KEY,
                room_id TEXT NOT NULL,
                evidence_type TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                source TEXT,
                source_id TEXT,
                data TEXT DEFAULT '{}',
                strength TEXT DEFAULT 'moderate',
                pinned_by TEXT NOT NULL,
                pinned_by_name TEXT NOT NULL,
                pinned_at TEXT NOT NULL,
                is_supporting INTEGER DEFAULT 1,
                linked_hypothesis TEXT,
                tags TEXT DEFAULT '[]',
                FOREIGN KEY (room_id) REFERENCES rooms(room_id)
            )
        ''')
        
        # Discussion threads table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS threads (
                thread_id TEXT PRIMARY KEY,
                room_id TEXT NOT NULL,
                thread_type TEXT NOT NULL,
                title TEXT NOT NULL,
                created_by TEXT NOT NULL,
                created_by_name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                is_resolved INTEGER DEFAULT 0,
                is_pinned INTEGER DEFAULT 0,
                message_count INTEGER DEFAULT 0,
                participants TEXT DEFAULT '[]',
                FOREIGN KEY (room_id) REFERENCES rooms(room_id)
            )
        ''')
        
        # Thread messages table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                message_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                room_id TEXT NOT NULL,
                author_id TEXT NOT NULL,
                author_name TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT,
                is_edited INTEGER DEFAULT 0,
                reply_to TEXT,
                mentions TEXT DEFAULT '[]',
                reactions TEXT DEFAULT '{}',
                attachments TEXT DEFAULT '[]',
                FOREIGN KEY (thread_id) REFERENCES threads(thread_id),
                FOREIGN KEY (room_id) REFERENCES rooms(room_id)
            )
        ''')
        
        # Votes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS votes (
                vote_id TEXT PRIMARY KEY,
                room_id TEXT NOT NULL,
                vote_type TEXT NOT NULL,
                hypothesis_id TEXT NOT NULL,
                hypothesis_text TEXT NOT NULL,
                voter_id TEXT NOT NULL,
                voter_name TEXT NOT NULL,
                vote TEXT NOT NULL,
                confidence REAL DEFAULT 0.0,
                comment TEXT,
                voted_at TEXT NOT NULL,
                FOREIGN KEY (room_id) REFERENCES rooms(room_id)
            )
        ''')
        
        # Resolutions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS resolutions (
                resolution_id TEXT PRIMARY KEY,
                room_id TEXT NOT NULL,
                resolution_type TEXT NOT NULL,
                root_cause TEXT NOT NULL,
                root_cause_confidence REAL NOT NULL,
                findings_summary TEXT NOT NULL,
                recommendations TEXT DEFAULT '[]',
                action_items TEXT DEFAULT '[]',
                lessons_learned TEXT DEFAULT '[]',
                resolved_by TEXT NOT NULL,
                resolved_by_name TEXT NOT NULL,
                resolved_at TEXT NOT NULL,
                approved_by TEXT,
                approved_at TEXT,
                linked_issues TEXT DEFAULT '[]',
                FOREIGN KEY (room_id) REFERENCES rooms(room_id)
            )
        ''')
        
        # Room audit trail
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS room_audit (
                audit_id TEXT PRIMARY KEY,
                room_id TEXT NOT NULL,
                action TEXT NOT NULL,
                actor_id TEXT NOT NULL,
                actor_name TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                details TEXT DEFAULT '{}',
                FOREIGN KEY (room_id) REFERENCES rooms(room_id)
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_rooms_status ON rooms(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_rooms_issue ON rooms(issue_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_rooms_study ON rooms(study_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_participants_room ON participants(room_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_participants_user ON participants(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_evidence_room ON evidence(room_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_threads_room ON threads(room_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timeline_room ON timeline_events(room_id)')
        
        conn.commit()
        conn.close()
    
    # =========================================================================
    # ROOM MANAGEMENT
    # =========================================================================
    
    def create_room(
        self,
        title: str,
        description: str,
        room_type: RoomType,
        created_by: str,
        created_by_name: str,
        issue_id: Optional[str] = None,
        study_id: Optional[str] = None,
        site_id: Optional[str] = None,
        patient_key: Optional[str] = None,
        priority: IssuePriority = IssuePriority.MEDIUM,
        tags: Optional[List[str]] = None,
        auto_populate_context: bool = True,
        participants: Optional[List[Dict[str, Any]]] = None,
    ) -> InvestigationRoom:
        """Create a new investigation room"""
        
        # Generate room ID
        room_id = f"ROOM-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"
        
        # Create room object
        room = InvestigationRoom(
            room_id=room_id,
            title=title,
            description=description,
            room_type=room_type,
            status=RoomStatus.DRAFT,
            issue_id=issue_id,
            study_id=study_id,
            site_id=site_id,
            patient_key=patient_key,
            created_by=created_by,
            created_by_name=created_by_name,
            priority=priority,
            tags=tags or [],
        )
        
        # Insert into database
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO rooms (
                room_id, title, description, room_type, status,
                issue_id, study_id, site_id, patient_key,
                created_by, created_by_name, created_at, updated_at,
                priority, tags
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            room.room_id, room.title, room.description,
            room.room_type.value, room.status.value,
            room.issue_id, room.study_id, room.site_id, room.patient_key,
            room.created_by, room.created_by_name,
            room.created_at.isoformat(), room.updated_at.isoformat(),
            room.priority.value, json.dumps(room.tags)
        ))
        
        conn.commit()
        conn.close()
        
        # Add creator as owner
        self.add_participant(room_id, created_by, created_by_name, ParticipantRole.OWNER)
        
        # Add additional participants
        if participants:
            for p in participants:
                self.add_participant(
                    room_id,
                    p['user_id'],
                    p['user_name'],
                    ParticipantRole(p.get('role', 'contributor'))
                )
        
        # Auto-populate context
        if auto_populate_context:
            self._populate_context(room_id, issue_id, study_id, site_id, patient_key)
        
        # Create audit entry
        self._add_room_audit(room_id, "room_created", created_by, created_by_name, {
            'title': title,
            'room_type': room_type.value
        })
        
        self.stats['rooms_created'] += 1
        
        return self.get_room(room_id)
    
    def create_room_from_issue(
        self,
        issue_id: str,
        created_by: str,
        created_by_name: str,
        additional_context: Optional[str] = None
    ) -> Optional[InvestigationRoom]:
        """Create investigation room from existing issue"""
        
        # Get issue
        issue = self.issue_registry.get_issue(issue_id)
        if not issue:
            return None
        
        # Determine room type
        if issue.category in [IssueCategory.SAE_DM_PENDING, IssueCategory.SAE_SAFETY_PENDING, 
                              IssueCategory.SAFETY_SIGNAL]:
            room_type = RoomType.SAFETY
        elif issue.category == IssueCategory.PROTOCOL_DEVIATION:
            room_type = RoomType.AUDIT
        else:
            room_type = RoomType.ISSUE
        
        # Create room
        description = f"Investigation for: {issue.description}"
        if additional_context:
            description += f"\n\nAdditional Context: {additional_context}"
        
        room = self.create_room(
            title=f"Investigation: {issue.title}",
            description=description,
            room_type=room_type,
            created_by=created_by,
            created_by_name=created_by_name,
            issue_id=issue_id,
            study_id=issue.study_id,
            site_id=issue.site_id,
            patient_key=issue.patient_key,
            priority=issue.priority,
            tags=issue.tags + ['from_issue'],
            auto_populate_context=True
        )
        
        # Add issue assignee as participant if assigned
        if issue.assignee_id and issue.assignee_id != created_by:
            self.add_participant(
                room.room_id,
                issue.assignee_id,
                issue.assignee_name or issue.assignee_id,
                ParticipantRole.CONTRIBUTOR
            )
        
        # Reconstruct timeline from issue history
        self._reconstruct_timeline_from_issue(room.room_id, issue)
        
        return room
    
    def get_room(self, room_id: str) -> Optional[InvestigationRoom]:
        """Get room by ID"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM rooms WHERE room_id = ?', (room_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return self._row_to_room(row)
        return None
    
    def update_room(
        self,
        room_id: str,
        updated_by: str,
        updated_by_name: str,
        **updates
    ) -> Optional[InvestigationRoom]:
        """Update room fields"""
        
        room = self.get_room(room_id)
        if not room:
            return None
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        set_clauses = []
        values = []
        
        for field, value in updates.items():
            if hasattr(room, field):
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
        
        set_clauses.append("updated_at = ?")
        values.append(datetime.now().isoformat())
        values.append(room_id)
        
        query = f"UPDATE rooms SET {', '.join(set_clauses)} WHERE room_id = ?"
        cursor.execute(query, values)
        
        conn.commit()
        conn.close()
        
        self._add_room_audit(room_id, "room_updated", updated_by, updated_by_name, updates)
        
        return self.get_room(room_id)
    
    def change_room_status(
        self,
        room_id: str,
        new_status: RoomStatus,
        changed_by: str,
        changed_by_name: str,
        comment: Optional[str] = None
    ) -> Optional[InvestigationRoom]:
        """Change room status"""
        
        room = self.get_room(room_id)
        if not room:
            return None
        
        old_status = room.status
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        updates = {
            'status': new_status.value,
            'updated_at': datetime.now().isoformat()
        }
        
        if new_status == RoomStatus.CLOSED:
            updates['closed_at'] = datetime.now().isoformat()
        
        set_clause = ', '.join([f"{k} = ?" for k in updates.keys()])
        values = list(updates.values()) + [room_id]
        
        cursor.execute(f"UPDATE rooms SET {set_clause} WHERE room_id = ?", values)
        conn.commit()
        conn.close()
        
        self._add_room_audit(room_id, "status_changed", changed_by, changed_by_name, {
            'old_status': old_status.value,
            'new_status': new_status.value,
            'comment': comment
        })
        
        return self.get_room(room_id)
    
    def activate_room(self, room_id: str, activated_by: str, activated_by_name: str) -> Optional[InvestigationRoom]:
        """Activate a draft room to start investigation"""
        return self.change_room_status(room_id, RoomStatus.ACTIVE, activated_by, activated_by_name)
    
    def close_room(
        self,
        room_id: str,
        closed_by: str,
        closed_by_name: str,
        resolution_id: Optional[str] = None
    ) -> Optional[InvestigationRoom]:
        """Close an investigation room"""
        
        if resolution_id:
            self.update_room(room_id, closed_by, closed_by_name, resolution_id=resolution_id)
        
        return self.change_room_status(room_id, RoomStatus.CLOSED, closed_by, closed_by_name)
    
    # =========================================================================
    # PARTICIPANTS
    # =========================================================================
    
    def add_participant(
        self,
        room_id: str,
        user_id: str,
        user_name: str,
        role: ParticipantRole
    ) -> Optional[RoomParticipant]:
        """Add participant to room"""
        
        participant_id = f"PART-{uuid.uuid4().hex[:12].upper()}"
        
        participant = RoomParticipant(
            participant_id=participant_id,
            room_id=room_id,
            user_id=user_id,
            user_name=user_name,
            role=role
        )
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Check if already participant
        cursor.execute(
            'SELECT participant_id FROM participants WHERE room_id = ? AND user_id = ?',
            (room_id, user_id)
        )
        if cursor.fetchone():
            conn.close()
            return None  # Already a participant
        
        cursor.execute('''
            INSERT INTO participants (
                participant_id, room_id, user_id, user_name, role,
                joined_at, last_active, is_active, notifications_enabled
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            participant.participant_id, participant.room_id,
            participant.user_id, participant.user_name, participant.role.value,
            participant.joined_at.isoformat(), participant.last_active.isoformat(),
            1, 1
        ))
        
        # Update participant count
        cursor.execute('''
            UPDATE rooms SET participant_count = participant_count + 1, updated_at = ?
            WHERE room_id = ?
        ''', (datetime.now().isoformat(), room_id))
        
        conn.commit()
        conn.close()
        
        return participant
    
    def get_participants(self, room_id: str) -> List[RoomParticipant]:
        """Get all participants in room"""
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT * FROM participants WHERE room_id = ? AND is_active = 1',
            (room_id,)
        )
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_participant(row) for row in rows]
    
    def remove_participant(self, room_id: str, user_id: str) -> bool:
        """Remove participant from room (soft delete)"""
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE participants SET is_active = 0 
            WHERE room_id = ? AND user_id = ?
        ''', (room_id, user_id))
        
        cursor.execute('''
            UPDATE rooms SET participant_count = participant_count - 1, updated_at = ?
            WHERE room_id = ?
        ''', (datetime.now().isoformat(), room_id))
        
        conn.commit()
        affected = cursor.rowcount > 0
        conn.close()
        
        return affected
    
    # =========================================================================
    # CONTEXT AUTO-POPULATION
    # =========================================================================
    
    def _populate_context(
        self,
        room_id: str,
        issue_id: Optional[str],
        study_id: Optional[str],
        site_id: Optional[str],
        patient_key: Optional[str]
    ):
        """Auto-populate context from data sources"""
        
        contexts = []
        
        # Get issue context
        if issue_id:
            issue_context = self.data_loader.get_issue_context(issue_id)
            if issue_context:
                contexts.append(RoomContext(
                    context_id=f"CTX-{uuid.uuid4().hex[:12].upper()}",
                    room_id=room_id,
                    source="issue_registry",
                    entity_type="issue",
                    entity_id=issue_id,
                    data_snapshot=issue_context
                ))
        
        # Get patient context
        if patient_key:
            patient_context = self.data_loader.get_patient_context(patient_key)
            if patient_context:
                contexts.append(RoomContext(
                    context_id=f"CTX-{uuid.uuid4().hex[:12].upper()}",
                    room_id=room_id,
                    source="unified_patient_record",
                    entity_type="patient",
                    entity_id=patient_key,
                    data_snapshot=patient_context
                ))
        
        # Get site context
        if site_id:
            site_context = self.data_loader.get_site_context(site_id)
            if site_context:
                contexts.append(RoomContext(
                    context_id=f"CTX-{uuid.uuid4().hex[:12].upper()}",
                    room_id=room_id,
                    source="site_benchmarks",
                    entity_type="site",
                    entity_id=site_id,
                    data_snapshot=site_context
                ))
        
        # Get study context
        if study_id:
            study_context = self.data_loader.get_study_context(study_id)
            if study_context:
                contexts.append(RoomContext(
                    context_id=f"CTX-{uuid.uuid4().hex[:12].upper()}",
                    room_id=room_id,
                    source="study_metrics",
                    entity_type="study",
                    entity_id=study_id,
                    data_snapshot=study_context
                ))
        
        # Save contexts
        conn = self._get_connection()
        cursor = conn.cursor()
        
        for ctx in contexts:
            cursor.execute('''
                INSERT INTO room_context (
                    context_id, room_id, source, entity_type, entity_id,
                    data_snapshot, retrieved_at, is_stale
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                ctx.context_id, ctx.room_id, ctx.source, ctx.entity_type,
                ctx.entity_id, json.dumps(ctx.data_snapshot),
                ctx.retrieved_at.isoformat(), 0
            ))
        
        conn.commit()
        conn.close()
    
    def get_room_context(self, room_id: str) -> List[RoomContext]:
        """Get all context for room"""
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM room_context WHERE room_id = ?', (room_id,))
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_context(row) for row in rows]
    
    def refresh_context(self, room_id: str) -> int:
        """Refresh stale context data"""
        
        contexts = self.get_room_context(room_id)
        refreshed = 0
        
        for ctx in contexts:
            # Check if stale (older than 1 hour)
            age = datetime.now() - ctx.retrieved_at
            if age.total_seconds() > 3600:
                # Refresh based on entity type
                if ctx.entity_type == "patient":
                    new_data = self.data_loader.get_patient_context(ctx.entity_id)
                elif ctx.entity_type == "site":
                    new_data = self.data_loader.get_site_context(ctx.entity_id)
                elif ctx.entity_type == "study":
                    new_data = self.data_loader.get_study_context(ctx.entity_id)
                elif ctx.entity_type == "issue":
                    new_data = self.data_loader.get_issue_context(ctx.entity_id)
                else:
                    continue
                
                if new_data:
                    conn = self._get_connection()
                    cursor = conn.cursor()
                    cursor.execute('''
                        UPDATE room_context 
                        SET data_snapshot = ?, retrieved_at = ?, is_stale = 0
                        WHERE context_id = ?
                    ''', (json.dumps(new_data), datetime.now().isoformat(), ctx.context_id))
                    conn.commit()
                    conn.close()
                    refreshed += 1
        
        return refreshed
    
    # =========================================================================
    # TIMELINE
    # =========================================================================
    
    def _reconstruct_timeline_from_issue(self, room_id: str, issue: Issue):
        """Reconstruct timeline from issue audit trail"""
        
        # Get audit trail
        audit_entries = self.issue_registry.get_audit_trail(issue.issue_id)
        
        events = []
        
        # Issue creation event
        events.append(TimelineEvent(
            event_id=f"EVT-{uuid.uuid4().hex[:12].upper()}",
            room_id=room_id,
            event_type="issue_created",
            event_time=issue.created_at,
            title="Issue Created",
            description=f"Issue '{issue.title}' was created",
            source="issue_registry",
            entity_id=issue.issue_id,
            is_key_event=True
        ))
        
        # Add audit events
        for entry in audit_entries:
            events.append(TimelineEvent(
                event_id=f"EVT-{uuid.uuid4().hex[:12].upper()}",
                room_id=room_id,
                event_type=entry.action.value,
                event_time=entry.timestamp,
                title=entry.action.value.replace('_', ' ').title(),
                description=entry.description,
                source="issue_registry",
                entity_id=issue.issue_id,
                metadata={'actor': entry.actor_name, 'old_value': entry.old_value, 'new_value': entry.new_value}
            ))
        
        # Save events
        conn = self._get_connection()
        cursor = conn.cursor()
        
        for event in events:
            cursor.execute('''
                INSERT INTO timeline_events (
                    event_id, room_id, event_type, event_time, title,
                    description, source, entity_id, metadata, is_key_event
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                event.event_id, event.room_id, event.event_type,
                event.event_time.isoformat(), event.title, event.description,
                event.source, event.entity_id, json.dumps(event.metadata),
                1 if event.is_key_event else 0
            ))
        
        conn.commit()
        conn.close()
    
    def add_timeline_event(
        self,
        room_id: str,
        event_type: str,
        title: str,
        description: str,
        event_time: Optional[datetime] = None,
        source: str = "manual",
        entity_id: Optional[str] = None,
        is_key_event: bool = False,
        metadata: Optional[Dict] = None
    ) -> TimelineEvent:
        """Add event to timeline"""
        
        event = TimelineEvent(
            event_id=f"EVT-{uuid.uuid4().hex[:12].upper()}",
            room_id=room_id,
            event_type=event_type,
            event_time=event_time or datetime.now(),
            title=title,
            description=description,
            source=source,
            entity_id=entity_id,
            is_key_event=is_key_event,
            metadata=metadata or {}
        )
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO timeline_events (
                event_id, room_id, event_type, event_time, title,
                description, source, entity_id, metadata, is_key_event
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            event.event_id, event.room_id, event.event_type,
            event.event_time.isoformat(), event.title, event.description,
            event.source, event.entity_id, json.dumps(event.metadata),
            1 if event.is_key_event else 0
        ))
        
        conn.commit()
        conn.close()
        
        return event
    
    def get_timeline(self, room_id: str, key_events_only: bool = False) -> List[TimelineEvent]:
        """Get timeline events for room"""
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if key_events_only:
            cursor.execute(
                'SELECT * FROM timeline_events WHERE room_id = ? AND is_key_event = 1 ORDER BY event_time ASC',
                (room_id,)
            )
        else:
            cursor.execute(
                'SELECT * FROM timeline_events WHERE room_id = ? ORDER BY event_time ASC',
                (room_id,)
            )
        
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_timeline_event(row) for row in rows]
    
    # =========================================================================
    # EVIDENCE
    # =========================================================================
    
    def pin_evidence(
        self,
        room_id: str,
        evidence_type: EvidenceType,
        title: str,
        description: str,
        pinned_by: str,
        pinned_by_name: str,
        source: str = "manual",
        source_id: Optional[str] = None,
        data: Optional[Dict] = None,
        strength: EvidenceStrength = EvidenceStrength.MODERATE,
        is_supporting: bool = True,
        linked_hypothesis: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Evidence:
        """Pin evidence to investigation"""
        
        evidence = Evidence(
            evidence_id=f"EVI-{uuid.uuid4().hex[:12].upper()}",
            room_id=room_id,
            evidence_type=evidence_type,
            title=title,
            description=description,
            source=source,
            source_id=source_id,
            data=data or {},
            strength=strength,
            pinned_by=pinned_by,
            pinned_by_name=pinned_by_name,
            is_supporting=is_supporting,
            linked_hypothesis=linked_hypothesis,
            tags=tags or []
        )
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO evidence (
                evidence_id, room_id, evidence_type, title, description,
                source, source_id, data, strength, pinned_by, pinned_by_name,
                pinned_at, is_supporting, linked_hypothesis, tags
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            evidence.evidence_id, evidence.room_id, evidence.evidence_type.value,
            evidence.title, evidence.description, evidence.source, evidence.source_id,
            json.dumps(evidence.data), evidence.strength.value,
            evidence.pinned_by, evidence.pinned_by_name,
            evidence.pinned_at.isoformat(), 1 if evidence.is_supporting else 0,
            evidence.linked_hypothesis, json.dumps(evidence.tags)
        ))
        
        # Update evidence count
        cursor.execute('''
            UPDATE rooms SET evidence_count = evidence_count + 1, updated_at = ?
            WHERE room_id = ?
        ''', (datetime.now().isoformat(), room_id))
        
        conn.commit()
        conn.close()
        
        # Add timeline event
        self.add_timeline_event(
            room_id=room_id,
            event_type="evidence_pinned",
            title=f"Evidence Pinned: {title}",
            description=f"{pinned_by_name} pinned {strength.value} evidence",
            source="investigation",
            metadata={'evidence_id': evidence.evidence_id, 'strength': strength.value}
        )
        
        self.stats['evidence_pinned'] += 1
        
        return evidence
    
    def get_evidence(self, room_id: str, strength_filter: Optional[List[EvidenceStrength]] = None) -> List[Evidence]:
        """Get all evidence for room"""
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if strength_filter:
            placeholders = ','.join(['?' for _ in strength_filter])
            cursor.execute(
                f'SELECT * FROM evidence WHERE room_id = ? AND strength IN ({placeholders}) ORDER BY pinned_at DESC',
                [room_id] + [s.value for s in strength_filter]
            )
        else:
            cursor.execute(
                'SELECT * FROM evidence WHERE room_id = ? ORDER BY pinned_at DESC',
                (room_id,)
            )
        
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_evidence(row) for row in rows]
    
    def update_evidence_strength(
        self,
        evidence_id: str,
        new_strength: EvidenceStrength,
        updated_by: str,
        updated_by_name: str
    ) -> bool:
        """Update evidence strength"""
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            'UPDATE evidence SET strength = ? WHERE evidence_id = ?',
            (new_strength.value, evidence_id)
        )
        
        conn.commit()
        affected = cursor.rowcount > 0
        conn.close()
        
        return affected
    
    def remove_evidence(self, evidence_id: str, room_id: str) -> bool:
        """Remove evidence from investigation"""
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM evidence WHERE evidence_id = ?', (evidence_id,))
        
        cursor.execute('''
            UPDATE rooms SET evidence_count = evidence_count - 1, updated_at = ?
            WHERE room_id = ?
        ''', (datetime.now().isoformat(), room_id))
        
        conn.commit()
        affected = cursor.rowcount > 0
        conn.close()
        
        return affected
    
    # =========================================================================
    # DISCUSSION THREADS
    # =========================================================================
    
    def create_thread(
        self,
        room_id: str,
        thread_type: ThreadType,
        title: str,
        created_by: str,
        created_by_name: str,
        initial_message: Optional[str] = None
    ) -> DiscussionThread:
        """Create discussion thread"""
        
        thread = DiscussionThread(
            thread_id=f"THR-{uuid.uuid4().hex[:12].upper()}",
            room_id=room_id,
            thread_type=thread_type,
            title=title,
            created_by=created_by,
            created_by_name=created_by_name,
            participants=[created_by]
        )
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO threads (
                thread_id, room_id, thread_type, title,
                created_by, created_by_name, created_at, updated_at,
                is_resolved, is_pinned, message_count, participants
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            thread.thread_id, thread.room_id, thread.thread_type.value,
            thread.title, thread.created_by, thread.created_by_name,
            thread.created_at.isoformat(), thread.updated_at.isoformat(),
            0, 0, 0, json.dumps(thread.participants)
        ))
        
        # Update thread count
        cursor.execute('''
            UPDATE rooms SET thread_count = thread_count + 1, updated_at = ?
            WHERE room_id = ?
        ''', (datetime.now().isoformat(), room_id))
        
        conn.commit()
        conn.close()
        
        # Add initial message if provided
        if initial_message:
            self.post_message(thread.thread_id, room_id, created_by, created_by_name, initial_message)
        
        self.stats['threads_created'] += 1
        
        return thread
    
    def get_threads(self, room_id: str, include_resolved: bool = True) -> List[DiscussionThread]:
        """Get all threads in room"""
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if include_resolved:
            cursor.execute(
                'SELECT * FROM threads WHERE room_id = ? ORDER BY is_pinned DESC, updated_at DESC',
                (room_id,)
            )
        else:
            cursor.execute(
                'SELECT * FROM threads WHERE room_id = ? AND is_resolved = 0 ORDER BY is_pinned DESC, updated_at DESC',
                (room_id,)
            )
        
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_thread(row) for row in rows]
    
    def post_message(
        self,
        thread_id: str,
        room_id: str,
        author_id: str,
        author_name: str,
        content: str,
        reply_to: Optional[str] = None,
        mentions: Optional[List[str]] = None
    ) -> ThreadMessage:
        """Post message to thread"""
        
        message = ThreadMessage(
            message_id=f"MSG-{uuid.uuid4().hex[:12].upper()}",
            thread_id=thread_id,
            room_id=room_id,
            author_id=author_id,
            author_name=author_name,
            content=content,
            reply_to=reply_to,
            mentions=mentions or []
        )
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO messages (
                message_id, thread_id, room_id, author_id, author_name,
                content, created_at, is_edited, reply_to, mentions, reactions, attachments
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            message.message_id, message.thread_id, message.room_id,
            message.author_id, message.author_name, message.content,
            message.created_at.isoformat(), 0, message.reply_to,
            json.dumps(message.mentions), json.dumps(message.reactions),
            json.dumps(message.attachments)
        ))
        
        # Update counts
        cursor.execute('''
            UPDATE threads SET message_count = message_count + 1, updated_at = ?
            WHERE thread_id = ?
        ''', (datetime.now().isoformat(), thread_id))
        
        cursor.execute('''
            UPDATE rooms SET message_count = message_count + 1, updated_at = ?
            WHERE room_id = ?
        ''', (datetime.now().isoformat(), room_id))
        
        # Add author to thread participants
        cursor.execute('SELECT participants FROM threads WHERE thread_id = ?', (thread_id,))
        row = cursor.fetchone()
        if row:
            participants = json.loads(row['participants'])
            if author_id not in participants:
                participants.append(author_id)
                cursor.execute(
                    'UPDATE threads SET participants = ? WHERE thread_id = ?',
                    (json.dumps(participants), thread_id)
                )
        
        conn.commit()
        conn.close()
        
        self.stats['messages_posted'] += 1
        
        return message
    
    def get_messages(self, thread_id: str, limit: int = 100, offset: int = 0) -> List[ThreadMessage]:
        """Get messages in thread"""
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT * FROM messages WHERE thread_id = ? ORDER BY created_at ASC LIMIT ? OFFSET ?',
            (thread_id, limit, offset)
        )
        
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_message(row) for row in rows]
    
    def resolve_thread(self, thread_id: str) -> bool:
        """Mark thread as resolved"""
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            'UPDATE threads SET is_resolved = 1, updated_at = ? WHERE thread_id = ?',
            (datetime.now().isoformat(), thread_id)
        )
        
        conn.commit()
        affected = cursor.rowcount > 0
        conn.close()
        
        return affected
    
    def pin_thread(self, thread_id: str, pinned: bool = True) -> bool:
        """Pin/unpin thread"""
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            'UPDATE threads SET is_pinned = ?, updated_at = ? WHERE thread_id = ?',
            (1 if pinned else 0, datetime.now().isoformat(), thread_id)
        )
        
        conn.commit()
        affected = cursor.rowcount > 0
        conn.close()
        
        return affected
    
    # =========================================================================
    # VOTING
    # =========================================================================
    
    def cast_vote(
        self,
        room_id: str,
        vote_type: VoteType,
        hypothesis_id: str,
        hypothesis_text: str,
        voter_id: str,
        voter_name: str,
        vote: str,  # agree, disagree, abstain
        confidence: float = 0.0,
        comment: str = ""
    ) -> RootCauseVote:
        """Cast vote on hypothesis"""
        
        vote_obj = RootCauseVote(
            vote_id=f"VOT-{uuid.uuid4().hex[:12].upper()}",
            room_id=room_id,
            vote_type=vote_type,
            hypothesis_id=hypothesis_id,
            hypothesis_text=hypothesis_text,
            voter_id=voter_id,
            voter_name=voter_name,
            vote=vote,
            confidence=confidence,
            comment=comment
        )
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Check for existing vote
        cursor.execute(
            'SELECT vote_id FROM votes WHERE room_id = ? AND hypothesis_id = ? AND voter_id = ?',
            (room_id, hypothesis_id, voter_id)
        )
        existing = cursor.fetchone()
        
        if existing:
            # Update existing vote
            cursor.execute('''
                UPDATE votes SET vote = ?, confidence = ?, comment = ?, voted_at = ?
                WHERE vote_id = ?
            ''', (vote, confidence, comment, datetime.now().isoformat(), existing['vote_id']))
        else:
            # Insert new vote
            cursor.execute('''
                INSERT INTO votes (
                    vote_id, room_id, vote_type, hypothesis_id, hypothesis_text,
                    voter_id, voter_name, vote, confidence, comment, voted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                vote_obj.vote_id, vote_obj.room_id, vote_obj.vote_type.value,
                vote_obj.hypothesis_id, vote_obj.hypothesis_text,
                vote_obj.voter_id, vote_obj.voter_name, vote_obj.vote,
                vote_obj.confidence, vote_obj.comment, vote_obj.voted_at.isoformat()
            ))
        
        conn.commit()
        conn.close()
        
        return vote_obj
    
    def get_votes(self, room_id: str, hypothesis_id: Optional[str] = None) -> List[RootCauseVote]:
        """Get votes for room or specific hypothesis"""
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if hypothesis_id:
            cursor.execute(
                'SELECT * FROM votes WHERE room_id = ? AND hypothesis_id = ?',
                (room_id, hypothesis_id)
            )
        else:
            cursor.execute('SELECT * FROM votes WHERE room_id = ?', (room_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_vote(row) for row in rows]
    
    def get_vote_summary(self, room_id: str, hypothesis_id: str) -> Dict[str, Any]:
        """Get vote summary for hypothesis"""
        
        votes = self.get_votes(room_id, hypothesis_id)
        
        agree = sum(1 for v in votes if v.vote == 'agree')
        disagree = sum(1 for v in votes if v.vote == 'disagree')
        abstain = sum(1 for v in votes if v.vote == 'abstain')
        avg_confidence = sum(v.confidence for v in votes) / len(votes) if votes else 0
        
        return {
            'hypothesis_id': hypothesis_id,
            'total_votes': len(votes),
            'agree': agree,
            'disagree': disagree,
            'abstain': abstain,
            'agree_percentage': (agree / len(votes) * 100) if votes else 0,
            'average_confidence': avg_confidence,
            'consensus': 'agree' if agree > disagree else ('disagree' if disagree > agree else 'split')
        }
    
    # =========================================================================
    # RESOLUTION
    # =========================================================================
    
    def create_resolution(
        self,
        room_id: str,
        resolution_type: ResolutionType,
        root_cause: str,
        root_cause_confidence: float,
        findings_summary: str,
        resolved_by: str,
        resolved_by_name: str,
        recommendations: Optional[List[str]] = None,
        action_items: Optional[List[Dict]] = None,
        lessons_learned: Optional[List[str]] = None,
        linked_issues: Optional[List[str]] = None
    ) -> Resolution:
        """Create investigation resolution"""
        
        resolution = Resolution(
            resolution_id=f"RES-{uuid.uuid4().hex[:12].upper()}",
            room_id=room_id,
            resolution_type=resolution_type,
            root_cause=root_cause,
            root_cause_confidence=root_cause_confidence,
            findings_summary=findings_summary,
            recommendations=recommendations or [],
            action_items=action_items or [],
            lessons_learned=lessons_learned or [],
            resolved_by=resolved_by,
            resolved_by_name=resolved_by_name,
            linked_issues=linked_issues or []
        )
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO resolutions (
                resolution_id, room_id, resolution_type, root_cause, root_cause_confidence,
                findings_summary, recommendations, action_items, lessons_learned,
                resolved_by, resolved_by_name, resolved_at, linked_issues
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            resolution.resolution_id, resolution.room_id, resolution.resolution_type.value,
            resolution.root_cause, resolution.root_cause_confidence,
            resolution.findings_summary, json.dumps(resolution.recommendations),
            json.dumps(resolution.action_items), json.dumps(resolution.lessons_learned),
            resolution.resolved_by, resolution.resolved_by_name,
            resolution.resolved_at.isoformat(), json.dumps(resolution.linked_issues)
        ))
        
        # Update room
        cursor.execute('''
            UPDATE rooms SET resolution_id = ?, updated_at = ?
            WHERE room_id = ?
        ''', (resolution.resolution_id, datetime.now().isoformat(), room_id))
        
        conn.commit()
        conn.close()
        
        # Add timeline event
        self.add_timeline_event(
            room_id=room_id,
            event_type="resolution_created",
            title="Resolution Created",
            description=f"Root cause identified: {root_cause[:50]}...",
            is_key_event=True,
            metadata={'resolution_id': resolution.resolution_id, 'confidence': root_cause_confidence}
        )
        
        self.stats['resolutions_completed'] += 1
        
        return resolution
    
    def get_resolution(self, room_id: str) -> Optional[Resolution]:
        """Get resolution for room"""
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM resolutions WHERE room_id = ?', (room_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return self._row_to_resolution(row)
        return None
    
    def approve_resolution(
        self,
        resolution_id: str,
        approved_by: str
    ) -> bool:
        """Approve resolution"""
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE resolutions SET approved_by = ?, approved_at = ?
            WHERE resolution_id = ?
        ''', (approved_by, datetime.now().isoformat(), resolution_id))
        
        conn.commit()
        affected = cursor.rowcount > 0
        conn.close()
        
        return affected
    
    # =========================================================================
    # AI SUMMARY
    # =========================================================================
    
    def generate_ai_summary(self, room_id: str) -> str:
        """Generate AI summary of investigation (placeholder for LLM integration)"""
        
        room = self.get_room(room_id)
        if not room:
            return ""
        
        # Get all data
        evidence = self.get_evidence(room_id)
        threads = self.get_threads(room_id)
        timeline = self.get_timeline(room_id, key_events_only=True)
        votes = self.get_votes(room_id)
        
        # Build summary (simplified - would use LLM in production)
        summary_parts = []
        
        summary_parts.append(f"## Investigation Summary: {room.title}")
        summary_parts.append(f"\n**Status:** {room.status.value}")
        summary_parts.append(f"**Type:** {room.room_type.value}")
        summary_parts.append(f"**Priority:** {room.priority.value}")
        
        if evidence:
            strong_evidence = [e for e in evidence if e.strength == EvidenceStrength.STRONG]
            summary_parts.append(f"\n### Evidence ({len(evidence)} items)")
            summary_parts.append(f"- Strong evidence: {len(strong_evidence)}")
            summary_parts.append(f"- Supporting: {sum(1 for e in evidence if e.is_supporting)}")
            summary_parts.append(f"- Contradicting: {sum(1 for e in evidence if not e.is_supporting)}")
        
        if timeline:
            summary_parts.append(f"\n### Key Events ({len(timeline)} events)")
            for event in timeline[:5]:
                summary_parts.append(f"- {event.event_time.strftime('%Y-%m-%d')}: {event.title}")
        
        if threads:
            summary_parts.append(f"\n### Discussion ({len(threads)} threads, {room.message_count} messages)")
            active_threads = [t for t in threads if not t.is_resolved]
            summary_parts.append(f"- Active threads: {len(active_threads)}")
        
        if votes:
            summary_parts.append(f"\n### Voting ({len(votes)} votes cast)")
        
        summary = "\n".join(summary_parts)
        
        # Save summary
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE rooms SET ai_summary = ?, ai_summary_updated = ?, updated_at = ?
            WHERE room_id = ?
        ''', (summary, datetime.now().isoformat(), datetime.now().isoformat(), room_id))
        conn.commit()
        conn.close()
        
        return summary
    
    # =========================================================================
    # SEARCH & LIST
    # =========================================================================
    
    def list_rooms(
        self,
        status: Optional[List[RoomStatus]] = None,
        room_type: Optional[RoomType] = None,
        study_id: Optional[str] = None,
        participant_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> Tuple[List[InvestigationRoom], int]:
        """List investigation rooms with filters"""
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        where_clauses = []
        params = []
        
        if status:
            placeholders = ','.join(['?' for _ in status])
            where_clauses.append(f"r.status IN ({placeholders})")
            params.extend([s.value for s in status])
        
        if room_type:
            where_clauses.append("r.room_type = ?")
            params.append(room_type.value)
        
        if study_id:
            where_clauses.append("r.study_id = ?")
            params.append(study_id)
        
        if participant_id:
            where_clauses.append('''
                EXISTS (SELECT 1 FROM participants p 
                        WHERE p.room_id = r.room_id AND p.user_id = ? AND p.is_active = 1)
            ''')
            params.append(participant_id)
        
        where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
        
        # Get count
        cursor.execute(f"SELECT COUNT(*) FROM rooms r WHERE {where_sql}", params)
        total = cursor.fetchone()[0]
        
        # Get rooms
        query = f"""
            SELECT r.* FROM rooms r
            WHERE {where_sql}
            ORDER BY r.updated_at DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_room(row) for row in rows], total
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get investigation rooms statistics"""
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Room counts by status
        cursor.execute('SELECT status, COUNT(*) FROM rooms GROUP BY status')
        status_counts = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Room counts by type
        cursor.execute('SELECT room_type, COUNT(*) FROM rooms GROUP BY room_type')
        type_counts = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Total counts
        cursor.execute('SELECT COUNT(*) FROM rooms')
        total_rooms = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM evidence')
        total_evidence = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM threads')
        total_threads = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM messages')
        total_messages = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM resolutions')
        total_resolutions = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_rooms': total_rooms,
            'total_evidence': total_evidence,
            'total_threads': total_threads,
            'total_messages': total_messages,
            'total_resolutions': total_resolutions,
            'status_distribution': status_counts,
            'type_distribution': type_counts,
            'session_stats': self.stats
        }
    
    # =========================================================================
    # AUDIT
    # =========================================================================
    
    def _add_room_audit(
        self,
        room_id: str,
        action: str,
        actor_id: str,
        actor_name: str,
        details: Dict[str, Any]
    ):
        """Add room audit entry"""
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        audit_id = f"RAUD-{uuid.uuid4().hex[:12].upper()}"
        
        cursor.execute('''
            INSERT INTO room_audit (audit_id, room_id, action, actor_id, actor_name, timestamp, details)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            audit_id, room_id, action, actor_id, actor_name,
            datetime.now().isoformat(), json.dumps(details)
        ))
        
        conn.commit()
        conn.close()
    
    def get_room_audit(self, room_id: str) -> List[Dict[str, Any]]:
        """Get audit trail for room"""
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT * FROM room_audit WHERE room_id = ? ORDER BY timestamp DESC',
            (room_id,)
        )
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _row_to_room(self, row: sqlite3.Row) -> InvestigationRoom:
        """Convert row to InvestigationRoom"""
        return InvestigationRoom(
            room_id=row['room_id'],
            title=row['title'],
            description=row['description'],
            room_type=RoomType(row['room_type']),
            status=RoomStatus(row['status']),
            issue_id=row['issue_id'],
            study_id=row['study_id'],
            site_id=row['site_id'],
            patient_key=row['patient_key'],
            created_by=row['created_by'],
            created_by_name=row['created_by_name'],
            created_at=datetime.fromisoformat(row['created_at']),
            updated_at=datetime.fromisoformat(row['updated_at']),
            priority=IssuePriority(row['priority']),
            participant_count=row['participant_count'],
            evidence_count=row['evidence_count'],
            thread_count=row['thread_count'],
            message_count=row['message_count'],
            resolution_id=row['resolution_id'],
            closed_at=datetime.fromisoformat(row['closed_at']) if row['closed_at'] else None,
            ai_summary=row['ai_summary'] or '',
            ai_summary_updated=datetime.fromisoformat(row['ai_summary_updated']) if row['ai_summary_updated'] else None,
            tags=json.loads(row['tags'])
        )
    
    def _row_to_participant(self, row: sqlite3.Row) -> RoomParticipant:
        """Convert row to RoomParticipant"""
        return RoomParticipant(
            participant_id=row['participant_id'],
            room_id=row['room_id'],
            user_id=row['user_id'],
            user_name=row['user_name'],
            role=ParticipantRole(row['role']),
            joined_at=datetime.fromisoformat(row['joined_at']),
            last_active=datetime.fromisoformat(row['last_active']),
            is_active=bool(row['is_active']),
            notifications_enabled=bool(row['notifications_enabled'])
        )
    
    def _row_to_context(self, row: sqlite3.Row) -> RoomContext:
        """Convert row to RoomContext"""
        return RoomContext(
            context_id=row['context_id'],
            room_id=row['room_id'],
            source=row['source'],
            entity_type=row['entity_type'],
            entity_id=row['entity_id'],
            data_snapshot=json.loads(row['data_snapshot']),
            retrieved_at=datetime.fromisoformat(row['retrieved_at']),
            is_stale=bool(row['is_stale'])
        )
    
    def _row_to_timeline_event(self, row: sqlite3.Row) -> TimelineEvent:
        """Convert row to TimelineEvent"""
        return TimelineEvent(
            event_id=row['event_id'],
            room_id=row['room_id'],
            event_type=row['event_type'],
            event_time=datetime.fromisoformat(row['event_time']),
            title=row['title'],
            description=row['description'],
            source=row['source'],
            entity_id=row['entity_id'],
            metadata=json.loads(row['metadata']),
            is_key_event=bool(row['is_key_event'])
        )
    
    def _row_to_evidence(self, row: sqlite3.Row) -> Evidence:
        """Convert row to Evidence"""
        return Evidence(
            evidence_id=row['evidence_id'],
            room_id=row['room_id'],
            evidence_type=EvidenceType(row['evidence_type']),
            title=row['title'],
            description=row['description'],
            source=row['source'],
            source_id=row['source_id'],
            data=json.loads(row['data']),
            strength=EvidenceStrength(row['strength']),
            pinned_by=row['pinned_by'],
            pinned_by_name=row['pinned_by_name'],
            pinned_at=datetime.fromisoformat(row['pinned_at']),
            is_supporting=bool(row['is_supporting']),
            linked_hypothesis=row['linked_hypothesis'],
            tags=json.loads(row['tags'])
        )
    
    def _row_to_thread(self, row: sqlite3.Row) -> DiscussionThread:
        """Convert row to DiscussionThread"""
        return DiscussionThread(
            thread_id=row['thread_id'],
            room_id=row['room_id'],
            thread_type=ThreadType(row['thread_type']),
            title=row['title'],
            created_by=row['created_by'],
            created_by_name=row['created_by_name'],
            created_at=datetime.fromisoformat(row['created_at']),
            updated_at=datetime.fromisoformat(row['updated_at']),
            is_resolved=bool(row['is_resolved']),
            is_pinned=bool(row['is_pinned']),
            message_count=row['message_count'],
            participants=json.loads(row['participants'])
        )
    
    def _row_to_message(self, row: sqlite3.Row) -> ThreadMessage:
        """Convert row to ThreadMessage"""
        return ThreadMessage(
            message_id=row['message_id'],
            thread_id=row['thread_id'],
            room_id=row['room_id'],
            author_id=row['author_id'],
            author_name=row['author_name'],
            content=row['content'],
            created_at=datetime.fromisoformat(row['created_at']),
            updated_at=datetime.fromisoformat(row['updated_at']) if row['updated_at'] else None,
            is_edited=bool(row['is_edited']),
            reply_to=row['reply_to'],
            mentions=json.loads(row['mentions']),
            reactions=json.loads(row['reactions']),
            attachments=json.loads(row['attachments'])
        )
    
    def _row_to_vote(self, row: sqlite3.Row) -> RootCauseVote:
        """Convert row to RootCauseVote"""
        return RootCauseVote(
            vote_id=row['vote_id'],
            room_id=row['room_id'],
            vote_type=VoteType(row['vote_type']),
            hypothesis_id=row['hypothesis_id'],
            hypothesis_text=row['hypothesis_text'],
            voter_id=row['voter_id'],
            voter_name=row['voter_name'],
            vote=row['vote'],
            confidence=row['confidence'],
            comment=row['comment'],
            voted_at=datetime.fromisoformat(row['voted_at'])
        )
    
    def _row_to_resolution(self, row: sqlite3.Row) -> Resolution:
        """Convert row to Resolution"""
        return Resolution(
            resolution_id=row['resolution_id'],
            room_id=row['room_id'],
            resolution_type=ResolutionType(row['resolution_type']),
            root_cause=row['root_cause'],
            root_cause_confidence=row['root_cause_confidence'],
            findings_summary=row['findings_summary'],
            recommendations=json.loads(row['recommendations']),
            action_items=json.loads(row['action_items']),
            lessons_learned=json.loads(row['lessons_learned']),
            resolved_by=row['resolved_by'],
            resolved_by_name=row['resolved_by_name'],
            resolved_at=datetime.fromisoformat(row['resolved_at']),
            approved_by=row['approved_by'],
            approved_at=datetime.fromisoformat(row['approved_at']) if row['approved_at'] else None,
            linked_issues=json.loads(row['linked_issues'])
        )


# =============================================================================
# DATA LOADER FOR CONTEXT
# =============================================================================

class InvestigationDataLoader:
    """Load context data from analytics pipeline for investigations"""
    
    def __init__(self):
        """Initialize data loader"""
        self.data_dir = Path("data/processed")
        self._cache = {}
        self._cache_time = {}
        self._cache_ttl = 300  # 5 minutes
    
    def _load_parquet(self, filepath: Path) -> Optional[pd.DataFrame]:
        """Load parquet file with caching"""
        key = str(filepath)
        now = datetime.now()
        
        # Check cache
        if key in self._cache:
            if (now - self._cache_time[key]).total_seconds() < self._cache_ttl:
                return self._cache[key]
        
        # Load file
        if filepath.exists():
            try:
                df = pd.read_parquet(filepath)
                self._cache[key] = df
                self._cache_time[key] = now
                return df
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                return None
        return None
    
    def get_patient_context(self, patient_key: str) -> Optional[Dict[str, Any]]:
        """Get patient context from UPR and related data"""
        
        context = {}
        
        # Load UPR
        upr = self._load_parquet(self.data_dir / "upr" / "unified_patient_record.parquet")
        if upr is not None and 'patient_key' in upr.columns:
            patient = upr[upr['patient_key'] == patient_key]
            if not patient.empty:
                row = patient.iloc[0]
                context['patient'] = {
                    'patient_key': patient_key,
                    'study_id': row.get('study_id', ''),
                    'site_id': row.get('site_id', ''),
                    'subject_status': row.get('subject_status', ''),
                    'region': row.get('region', ''),
                    'country': row.get('country', ''),
                }
        
        # Load issues
        issues = self._load_parquet(self.data_dir / "analytics" / "patient_issues.parquet")
        if issues is not None and 'patient_key' in issues.columns:
            patient_issues = issues[issues['patient_key'] == patient_key]
            if not patient_issues.empty:
                row = patient_issues.iloc[0]
                issue_cols = [c for c in issues.columns if c.startswith('issue_')]
                context['issues'] = {col: bool(row.get(col, False)) for col in issue_cols}
        
        # Load DQI
        dqi = self._load_parquet(self.data_dir / "analytics" / "patient_dqi_enhanced.parquet")
        if dqi is not None and 'patient_key' in dqi.columns:
            patient_dqi = dqi[dqi['patient_key'] == patient_key]
            if not patient_dqi.empty:
                row = patient_dqi.iloc[0]
                context['dqi'] = {
                    'dqi_score': float(row.get('dqi_score', row.get('enhanced_dqi', 0))),
                    'dqi_band': row.get('dqi_band', ''),
                    'primary_issue': row.get('primary_issue', ''),
                }
        
        # Load clean status
        clean = self._load_parquet(self.data_dir / "analytics" / "patient_clean_status.parquet")
        if clean is not None and 'patient_key' in clean.columns:
            patient_clean = clean[clean['patient_key'] == patient_key]
            if not patient_clean.empty:
                row = patient_clean.iloc[0]
                context['clean_status'] = {
                    'tier1_clean': bool(row.get('tier1_clean', False)),
                    'tier2_clean': bool(row.get('tier2_clean', False)),
                    'blocking_reasons': row.get('blocking_reasons', []),
                }
        
        return context if context else None
    
    def get_site_context(self, site_id: str) -> Optional[Dict[str, Any]]:
        """Get site context from benchmarks and aggregated data"""
        
        context = {}
        
        # Load site benchmarks
        benchmarks = self._load_parquet(self.data_dir / "analytics" / "site_benchmarks.parquet")
        if benchmarks is not None and 'site_id' in benchmarks.columns:
            site = benchmarks[benchmarks['site_id'] == site_id]
            if not site.empty:
                row = site.iloc[0]
                context['benchmark'] = {
                    'site_id': site_id,
                    'study_id': row.get('study_id', ''),
                    'patient_count': int(row.get('patient_count', 0)),
                    'mean_dqi': float(row.get('mean_dqi', 0)),
                    'tier2_clean_rate': float(row.get('tier2_clean_rate', 0)),
                    'performance_tier': row.get('performance_tier', ''),
                    'composite_score': float(row.get('composite_score', 0)),
                }
        
        # Get site-level issue counts from UPR
        upr = self._load_parquet(self.data_dir / "upr" / "unified_patient_record.parquet")
        if upr is not None and 'site_id' in upr.columns:
            site_patients = upr[upr['site_id'] == site_id]
            context['patients'] = {
                'total': len(site_patients),
                'by_status': site_patients['subject_status'].value_counts().to_dict() if 'subject_status' in site_patients.columns else {}
            }
        
        return context if context else None
    
    def get_study_context(self, study_id: str) -> Optional[Dict[str, Any]]:
        """Get study context from aggregated metrics"""
        
        context = {}
        
        # Load UPR for study summary
        upr = self._load_parquet(self.data_dir / "upr" / "unified_patient_record.parquet")
        if upr is not None and 'study_id' in upr.columns:
            study = upr[upr['study_id'] == study_id]
            if not study.empty:
                context['summary'] = {
                    'study_id': study_id,
                    'patient_count': len(study),
                    'site_count': study['site_id'].nunique() if 'site_id' in study.columns else 0,
                    'status_distribution': study['subject_status'].value_counts().to_dict() if 'subject_status' in study.columns else {}
                }
        
        # Load DQI for study metrics
        dqi = self._load_parquet(self.data_dir / "analytics" / "patient_dqi_enhanced.parquet")
        if dqi is not None and 'study_id' in dqi.columns:
            study_dqi = dqi[dqi['study_id'] == study_id]
            if not study_dqi.empty:
                dqi_col = 'dqi_score' if 'dqi_score' in study_dqi.columns else 'enhanced_dqi'
                if dqi_col in study_dqi.columns:
                    context['dqi'] = {
                        'mean_dqi': float(study_dqi[dqi_col].mean()),
                        'median_dqi': float(study_dqi[dqi_col].median()),
                        'min_dqi': float(study_dqi[dqi_col].min()),
                        'max_dqi': float(study_dqi[dqi_col].max()),
                    }
        
        return context if context else None
    
    def get_issue_context(self, issue_id: str) -> Optional[Dict[str, Any]]:
        """Get issue context from issue registry"""
        
        try:
            registry = get_issue_registry()
            issue = registry.get_issue(issue_id)
            if issue:
                return {
                    'issue': issue.to_dict(),
                    'comments': [c.to_dict() for c in registry.get_comments(issue_id)],
                    'audit': [{'action': a.action.value, 'timestamp': a.timestamp.isoformat(), 
                              'description': a.description} for a in registry.get_audit_trail(issue_id)[:10]]
                }
        except Exception as e:
            print(f"Error getting issue context: {e}")
        
        return None


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

_rooms_manager_instance: Optional[InvestigationRoomsManager] = None

def get_investigation_rooms_manager(db_path: Optional[str] = None) -> InvestigationRoomsManager:
    """Get or create Investigation Rooms Manager singleton"""
    global _rooms_manager_instance
    if _rooms_manager_instance is None:
        _rooms_manager_instance = InvestigationRoomsManager(db_path)
    return _rooms_manager_instance


# =============================================================================
# TEST FUNCTION
# =============================================================================

def test_investigation_rooms():
    """Test the Investigation Rooms system"""
    print("=" * 70)
    print("SANCHALAK AI - INVESTIGATION ROOMS TEST")
    print("=" * 70)
    
    # Use test database
    test_db = "data/collaboration/test_investigation_rooms.db"
    
    # Clean up old test database
    if Path(test_db).exists():
        Path(test_db).unlink()
    
    manager = InvestigationRoomsManager(test_db)
    
    tests_passed = 0
    tests_total = 15
    
    # Test 1: Create Room
    print("\n--- TEST 1: Create Room ---")
    try:
        room = manager.create_room(
            title="Investigation: SDV Backlog at Site_101",
            description="Investigating root cause of SDV backlog affecting 25 patients",
            room_type=RoomType.SITE,
            created_by="lead_001",
            created_by_name="John Smith",
            study_id="Study_21",
            site_id="Site_101",
            priority=IssuePriority.HIGH,
            tags=["sdv", "urgent", "site_investigation"],
            auto_populate_context=False  # Skip for test
        )
        assert room is not None
        assert room.room_id.startswith("ROOM-")
        assert room.status == RoomStatus.DRAFT
        print(f"✅ Created room: {room.room_id}")
        print(f"   Title: {room.title}")
        print(f"   Status: {room.status.value}")
        print(f"   Participants: {room.participant_count}")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Get Room
    print("\n--- TEST 2: Get Room ---")
    try:
        retrieved = manager.get_room(room.room_id)
        assert retrieved is not None
        assert retrieved.title == room.title
        print(f"✅ Retrieved room: {retrieved.title}")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 3: Add Participants
    print("\n--- TEST 3: Add Participants ---")
    try:
        p1 = manager.add_participant(room.room_id, "cra_001", "Sarah Chen", ParticipantRole.LEAD)
        p2 = manager.add_participant(room.room_id, "dm_001", "Alex Kim", ParticipantRole.CONTRIBUTOR)
        p3 = manager.add_participant(room.room_id, "safety_001", "Dr. Garcia", ParticipantRole.REVIEWER)
        
        participants = manager.get_participants(room.room_id)
        assert len(participants) == 4  # Owner + 3 added
        print(f"✅ Added 3 participants, total: {len(participants)}")
        for p in participants:
            print(f"   - {p.user_name} ({p.role.value})")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 4: Activate Room
    print("\n--- TEST 4: Activate Room ---")
    try:
        activated = manager.activate_room(room.room_id, "lead_001", "John Smith")
        assert activated is not None
        assert activated.status == RoomStatus.ACTIVE
        print(f"✅ Room activated: {activated.status.value}")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 5: Add Timeline Events
    print("\n--- TEST 5: Add Timeline Events ---")
    try:
        event1 = manager.add_timeline_event(
            room_id=room.room_id,
            event_type="investigation_started",
            title="Investigation Started",
            description="Team began investigating SDV backlog",
            is_key_event=True
        )
        
        event2 = manager.add_timeline_event(
            room_id=room.room_id,
            event_type="data_reviewed",
            title="Initial Data Review",
            description="Reviewed patient records and SDV status",
            source="manual"
        )
        
        timeline = manager.get_timeline(room.room_id)
        assert len(timeline) >= 2
        print(f"✅ Added timeline events, total: {len(timeline)}")
        for e in timeline[:3]:
            print(f"   - {e.title}: {e.description[:40]}...")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 6: Pin Evidence
    print("\n--- TEST 6: Pin Evidence ---")
    try:
        evidence1 = manager.pin_evidence(
            room_id=room.room_id,
            evidence_type=EvidenceType.DATA_POINT,
            title="CRA Visit Log Gap",
            description="CRA visits dropped from 2/week to 1/month in Q4",
            pinned_by="cra_001",
            pinned_by_name="Sarah Chen",
            source="visit_tracker",
            strength=EvidenceStrength.STRONG,
            is_supporting=True,
            tags=["visit_frequency", "staffing"]
        )
        
        evidence2 = manager.pin_evidence(
            room_id=room.room_id,
            evidence_type=EvidenceType.PATTERN_MATCH,
            title="Coordinator Overload Pattern",
            description="Pattern library match: Site has >50 patients with 1 coordinator",
            pinned_by="dm_001",
            pinned_by_name="Alex Kim",
            source="pattern_library",
            strength=EvidenceStrength.MODERATE,
            is_supporting=True,
            linked_hypothesis="HYP-001"
        )
        
        evidence3 = manager.pin_evidence(
            room_id=room.room_id,
            evidence_type=EvidenceType.DATA_POINT,
            title="Recent Staff Change",
            description="New coordinator started 2 weeks ago, learning curve",
            pinned_by="cra_001",
            pinned_by_name="Sarah Chen",
            source="site_management",
            strength=EvidenceStrength.WEAK,
            is_supporting=False  # Contradicting evidence
        )
        
        all_evidence = manager.get_evidence(room.room_id)
        assert len(all_evidence) == 3
        
        strong = manager.get_evidence(room.room_id, [EvidenceStrength.STRONG])
        assert len(strong) == 1
        
        print(f"✅ Pinned 3 evidence items")
        print(f"   Strong: {len(strong)}, Total: {len(all_evidence)}")
        for e in all_evidence:
            support = "✓" if e.is_supporting else "✗"
            print(f"   [{support}] {e.title} ({e.strength.value})")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 7: Create Discussion Threads
    print("\n--- TEST 7: Create Discussion Threads ---")
    try:
        thread1 = manager.create_thread(
            room_id=room.room_id,
            thread_type=ThreadType.HYPOTHESIS,
            title="Root Cause: CRA Resource Constraint",
            created_by="cra_001",
            created_by_name="Sarah Chen",
            initial_message="Based on the visit log data, I believe the root cause is insufficient CRA coverage."
        )
        
        thread2 = manager.create_thread(
            room_id=room.room_id,
            thread_type=ThreadType.ACTION,
            title="Proposed Actions",
            created_by="lead_001",
            created_by_name="John Smith"
        )
        
        threads = manager.get_threads(room.room_id)
        assert len(threads) == 2
        print(f"✅ Created 2 threads")
        for t in threads:
            print(f"   - [{t.thread_type.value}] {t.title}")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 8: Post Messages
    print("\n--- TEST 8: Post Messages ---")
    try:
        msg1 = manager.post_message(
            thread_id=thread1.thread_id,
            room_id=room.room_id,
            author_id="dm_001",
            author_name="Alex Kim",
            content="I agree with this hypothesis. The data supports reduced CRA presence.",
            mentions=["@cra_001", "@lead_001"]
        )
        
        msg2 = manager.post_message(
            thread_id=thread1.thread_id,
            room_id=room.room_id,
            author_id="safety_001",
            author_name="Dr. Garcia",
            content="We should also consider the coordinator overload pattern.",
            reply_to=msg1.message_id
        )
        
        msg3 = manager.post_message(
            thread_id=thread1.thread_id,
            room_id=room.room_id,
            author_id="cra_001",
            author_name="Sarah Chen",
            content="Good point @safety_001. I'll add that as supporting evidence.",
            mentions=["@safety_001"]
        )
        
        messages = manager.get_messages(thread1.thread_id)
        assert len(messages) >= 3  # Initial + 3 replies
        print(f"✅ Posted 3 messages, total in thread: {len(messages)}")
        for m in messages[:4]:
            print(f"   - {m.author_name}: {m.content[:40]}...")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 9: Cast Votes
    print("\n--- TEST 9: Cast Votes ---")
    try:
        vote1 = manager.cast_vote(
            room_id=room.room_id,
            vote_type=VoteType.ROOT_CAUSE,
            hypothesis_id="HYP-CRA-RESOURCE",
            hypothesis_text="CRA Resource Constraint is the primary root cause",
            voter_id="cra_001",
            voter_name="Sarah Chen",
            vote="agree",
            confidence=0.85,
            comment="Strong evidence supports this"
        )
        
        vote2 = manager.cast_vote(
            room_id=room.room_id,
            vote_type=VoteType.ROOT_CAUSE,
            hypothesis_id="HYP-CRA-RESOURCE",
            hypothesis_text="CRA Resource Constraint is the primary root cause",
            voter_id="dm_001",
            voter_name="Alex Kim",
            vote="agree",
            confidence=0.75
        )
        
        vote3 = manager.cast_vote(
            room_id=room.room_id,
            vote_type=VoteType.ROOT_CAUSE,
            hypothesis_id="HYP-CRA-RESOURCE",
            hypothesis_text="CRA Resource Constraint is the primary root cause",
            voter_id="safety_001",
            voter_name="Dr. Garcia",
            vote="abstain",
            confidence=0.5,
            comment="Need more data on coordinator workload"
        )
        
        summary = manager.get_vote_summary(room.room_id, "HYP-CRA-RESOURCE")
        print(f"✅ Cast 3 votes")
        print(f"   Total: {summary['total_votes']}")
        print(f"   Agree: {summary['agree']} ({summary['agree_percentage']:.0f}%)")
        print(f"   Disagree: {summary['disagree']}")
        print(f"   Abstain: {summary['abstain']}")
        print(f"   Avg Confidence: {summary['average_confidence']:.2f}")
        print(f"   Consensus: {summary['consensus']}")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 10: Create Resolution
    print("\n--- TEST 10: Create Resolution ---")
    try:
        resolution = manager.create_resolution(
            room_id=room.room_id,
            resolution_type=ResolutionType.ROOT_CAUSE_IDENTIFIED,
            root_cause="CRA Resource Constraint leading to reduced site visits and SDV backlog",
            root_cause_confidence=0.82,
            findings_summary="""
            Investigation identified that the primary root cause of the SDV backlog 
            is insufficient CRA coverage. CRA visits dropped from 2/week to 1/month 
            in Q4 2025, correlating with the SDV backlog increase.
            """,
            resolved_by="lead_001",
            resolved_by_name="John Smith",
            recommendations=[
                "Increase CRA coverage to minimum 2 visits per week",
                "Consider temporary CRA support for backlog clearance",
                "Implement remote SDV where applicable"
            ],
            action_items=[
                {"title": "Assign additional CRA", "owner": "CTM", "due_days": 7},
                {"title": "Schedule backlog clearance visits", "owner": "CRA", "due_days": 14},
                {"title": "Review SDV progress weekly", "owner": "Study Lead", "due_days": 30}
            ],
            lessons_learned=[
                "Early detection of visit frequency drops needed",
                "Workload monitoring should trigger alerts at 1.5x threshold"
            ]
        )
        
        assert resolution is not None
        assert resolution.resolution_id.startswith("RES-")
        print(f"✅ Created resolution: {resolution.resolution_id}")
        print(f"   Type: {resolution.resolution_type.value}")
        print(f"   Confidence: {resolution.root_cause_confidence:.0%}")
        print(f"   Recommendations: {len(resolution.recommendations)}")
        print(f"   Action Items: {len(resolution.action_items)}")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 11: Approve Resolution
    print("\n--- TEST 11: Approve Resolution ---")
    try:
        approved = manager.approve_resolution(resolution.resolution_id, "sponsor_001")
        assert approved
        
        updated_resolution = manager.get_resolution(room.room_id)
        assert updated_resolution.approved_by == "sponsor_001"
        print(f"✅ Resolution approved by: {updated_resolution.approved_by}")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 12: Generate AI Summary
    print("\n--- TEST 12: Generate AI Summary ---")
    try:
        summary = manager.generate_ai_summary(room.room_id)
        assert len(summary) > 0
        print(f"✅ Generated AI summary ({len(summary)} chars)")
        print(f"   Preview: {summary[:200]}...")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 13: Close Room
    print("\n--- TEST 13: Close Room ---")
    try:
        closed = manager.close_room(
            room.room_id,
            closed_by="lead_001",
            closed_by_name="John Smith",
            resolution_id=resolution.resolution_id
        )
        assert closed is not None
        assert closed.status == RoomStatus.CLOSED
        assert closed.closed_at is not None
        print(f"✅ Room closed at: {closed.closed_at}")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 14: Get Room Audit Trail
    print("\n--- TEST 14: Get Room Audit Trail ---")
    try:
        audit = manager.get_room_audit(room.room_id)
        assert len(audit) >= 3  # Created, activated, closed
        print(f"✅ Retrieved {len(audit)} audit entries")
        for entry in audit[:5]:
            print(f"   - {entry['action']}: {entry['actor_name']}")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Test 15: Get Statistics
    print("\n--- TEST 15: Get Statistics ---")
    try:
        stats = manager.get_statistics()
        print(f"✅ Investigation Rooms Statistics:")
        print(f"   Total Rooms: {stats['total_rooms']}")
        print(f"   Total Evidence: {stats['total_evidence']}")
        print(f"   Total Threads: {stats['total_threads']}")
        print(f"   Total Messages: {stats['total_messages']}")
        print(f"   Total Resolutions: {stats['total_resolutions']}")
        print(f"   Status Distribution: {stats['status_distribution']}")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Cleanup
    Path(test_db).unlink()
    
    # Summary
    print("\n" + "=" * 70)
    print(f"TEST RESULTS: {tests_passed}/{tests_total} passed")
    print("=" * 70)
    
    if tests_passed == tests_total:
        print("✅ ALL TESTS PASSED!")
    else:
        print(f"❌ {tests_total - tests_passed} tests failed")
    
    return tests_passed == tests_total


if __name__ == "__main__":
    test_investigation_rooms()