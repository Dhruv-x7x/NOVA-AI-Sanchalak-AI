"""
SANCHALAK AI - Action Persistence Layer
============================================
PostgreSQL-backed service for persisting all user actions.

Features:
- Persist every user action to PostgreSQL
- Action tracking with audit trail
- Undo/redo support for reversible actions
- Action analytics and reporting
- Integration with notification service
"""

import logging
import hashlib
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any, Callable
from sqlalchemy import text
import pandas as pd

from src.database.connection import get_db_manager

logger = logging.getLogger(__name__)


# ============================================================
# ENUMS
# ============================================================

class ActionType(Enum):
    """Types of user actions"""
    # Issue actions
    ISSUE_CREATE = "issue_create"
    ISSUE_UPDATE = "issue_update"
    ISSUE_ASSIGN = "issue_assign"
    ISSUE_RESOLVE = "issue_resolve"
    ISSUE_CLOSE = "issue_close"
    ISSUE_REOPEN = "issue_reopen"
    ISSUE_COMMENT = "issue_comment"
    
    # Escalation actions
    ESCALATION_CREATE = "escalation_create"
    ESCALATION_ACKNOWLEDGE = "escalation_acknowledge"
    ESCALATION_RESOLVE = "escalation_resolve"
    ESCALATION_TRANSFER = "escalation_transfer"
    
    # Room actions
    ROOM_CREATE = "room_create"
    ROOM_JOIN = "room_join"
    ROOM_LEAVE = "room_leave"
    ROOM_CLOSE = "room_close"
    ROOM_ADD_EVIDENCE = "room_add_evidence"
    ROOM_POST_MESSAGE = "room_post_message"
    
    # Workspace actions
    WORKSPACE_CREATE = "workspace_create"
    WORKSPACE_JOIN = "workspace_join"
    WORKSPACE_LEAVE = "workspace_leave"
    WORKSPACE_GOAL_CREATE = "workspace_goal_create"
    WORKSPACE_GOAL_UPDATE = "workspace_goal_update"
    
    # Query actions
    QUERY_CREATE = "query_create"
    QUERY_RESPOND = "query_respond"
    QUERY_CLOSE = "query_close"
    
    # Signature actions
    SIGNATURE_CREATE = "signature_create"
    SIGNATURE_REVOKE = "signature_revoke"
    
    # Patient actions
    PATIENT_UPDATE = "patient_update"
    PATIENT_STATUS_CHANGE = "patient_status_change"
    PATIENT_LOCK_READY = "patient_lock_ready"
    
    # Report actions
    REPORT_GENERATE = "report_generate"
    REPORT_EXPORT = "report_export"
    
    # General actions
    VIEW = "view"
    SEARCH = "search"
    FILTER = "filter"
    EXPORT = "export"
    LOGIN = "login"
    LOGOUT = "logout"


class ActionStatus(Enum):
    """Status of an action"""
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class UserAction:
    """Represents a user action"""
    action_id: str
    action_type: ActionType
    user_id: str
    user_name: str = ""
    user_role: str = ""
    
    # Target
    target_type: str = ""  # issue, escalation, room, patient, etc.
    target_id: str = ""
    
    # Context
    study_id: Optional[str] = None
    site_id: Optional[str] = None
    patient_key: Optional[str] = None
    
    # Details
    description: str = ""
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Status
    status: ActionStatus = ActionStatus.COMPLETED
    error_message: Optional[str] = None
    
    # Tracking
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    # Reversibility
    is_reversible: bool = False
    reversal_action_id: Optional[str] = None
    
    # Session
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'action_id': self.action_id,
            'action_type': self.action_type.value,
            'user_id': self.user_id,
            'user_name': self.user_name,
            'user_role': self.user_role,
            'target_type': self.target_type,
            'target_id': self.target_id,
            'study_id': self.study_id,
            'site_id': self.site_id,
            'patient_key': self.patient_key,
            'description': self.description,
            'old_value': self.old_value,
            'new_value': self.new_value,
            'metadata': self.metadata,
            'status': self.status.value,
            'error_message': self.error_message,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'is_reversible': self.is_reversible,
            'reversal_action_id': self.reversal_action_id,
            'session_id': self.session_id,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserAction':
        """Create from dictionary."""
        return cls(
            action_id=data.get('action_id', ''),
            action_type=ActionType(data.get('action_type', 'view')),
            user_id=data.get('user_id', ''),
            user_name=data.get('user_name', ''),
            user_role=data.get('user_role', ''),
            target_type=data.get('target_type', ''),
            target_id=data.get('target_id', ''),
            study_id=data.get('study_id'),
            site_id=data.get('site_id'),
            patient_key=data.get('patient_key'),
            description=data.get('description', ''),
            old_value=data.get('old_value'),
            new_value=data.get('new_value'),
            metadata=data.get('metadata', {}),
            status=ActionStatus(data.get('status', 'completed')),
            error_message=data.get('error_message'),
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else datetime.now(),
            completed_at=datetime.fromisoformat(data['completed_at']) if data.get('completed_at') else None,
            is_reversible=data.get('is_reversible', False),
            reversal_action_id=data.get('reversal_action_id'),
            session_id=data.get('session_id'),
            ip_address=data.get('ip_address'),
            user_agent=data.get('user_agent'),
        )


# ============================================================
# ACTION PERSISTENCE SERVICE
# ============================================================

class ActionPersistenceService:
    """PostgreSQL-backed action persistence service."""
    
    _instance: Optional['ActionPersistenceService'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._db_manager = None
        self._action_handlers: Dict[ActionType, List[Callable]] = {}
        self._initialize()
        self._initialized = True
    
    def _initialize(self):
        """Initialize database connection and tables."""
        try:
            self._db_manager = get_db_manager()
            self._ensure_tables()
            logger.info("ActionPersistenceService initialized with PostgreSQL")
        except Exception as e:
            logger.error(f"ActionPersistenceService initialization failed: {e}")
            raise
    
    def _ensure_tables(self):
        """Ensure action tables exist."""
        create_actions_table = text("""
            CREATE TABLE IF NOT EXISTS user_actions (
                action_id VARCHAR(100) PRIMARY KEY,
                action_type VARCHAR(50) NOT NULL,
                user_id VARCHAR(100) NOT NULL,
                user_name VARCHAR(200),
                user_role VARCHAR(50),
                target_type VARCHAR(50),
                target_id VARCHAR(100),
                study_id VARCHAR(50),
                site_id VARCHAR(50),
                patient_key VARCHAR(50),
                description TEXT,
                old_value TEXT,
                new_value TEXT,
                metadata JSONB,
                status VARCHAR(30) NOT NULL DEFAULT 'completed',
                error_message TEXT,
                created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                completed_at TIMESTAMP,
                is_reversible BOOLEAN DEFAULT FALSE,
                reversal_action_id VARCHAR(100),
                session_id VARCHAR(100),
                ip_address VARCHAR(50),
                user_agent TEXT
            )
        """)
        
        create_action_metrics_table = text("""
            CREATE TABLE IF NOT EXISTS action_metrics (
                id SERIAL PRIMARY KEY,
                date DATE NOT NULL,
                action_type VARCHAR(50) NOT NULL,
                user_id VARCHAR(100),
                action_count INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                avg_duration_ms FLOAT,
                UNIQUE (date, action_type, user_id)
            )
        """)
        
        create_indexes = [
            text("CREATE INDEX IF NOT EXISTS idx_actions_user ON user_actions(user_id)"),
            text("CREATE INDEX IF NOT EXISTS idx_actions_type ON user_actions(action_type)"),
            text("CREATE INDEX IF NOT EXISTS idx_actions_target ON user_actions(target_type, target_id)"),
            text("CREATE INDEX IF NOT EXISTS idx_actions_created ON user_actions(created_at DESC)"),
            text("CREATE INDEX IF NOT EXISTS idx_actions_study ON user_actions(study_id)"),
            text("CREATE INDEX IF NOT EXISTS idx_actions_site ON user_actions(site_id)"),
        ]
        
        try:
            if self._db_manager and self._db_manager.engine:
                with self._db_manager.engine.connect() as conn:
                    conn.execute(create_actions_table)
                    conn.execute(create_action_metrics_table)
                    for idx in create_indexes:
                        try:
                            conn.execute(idx)
                        except Exception:
                            pass  # Index may already exist
                    conn.commit()
                    logger.info("Action tables verified")
        except Exception as e:
            logger.warning(f"Could not create action tables: {e}")
    
    def _generate_id(self) -> str:
        """Generate unique action ID."""
        import uuid
        return f"ACT-{uuid.uuid4().hex[:12].upper()}"
    
    def record_action(
        self,
        action_type: ActionType,
        user_id: str,
        target_type: str = "",
        target_id: str = "",
        description: str = "",
        old_value: Optional[str] = None,
        new_value: Optional[str] = None,
        user_name: str = "",
        user_role: str = "",
        study_id: Optional[str] = None,
        site_id: Optional[str] = None,
        patient_key: Optional[str] = None,
        metadata: Optional[Dict] = None,
        is_reversible: bool = False,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> Optional[UserAction]:
        """Record a user action."""
        try:
            action = UserAction(
                action_id=self._generate_id(),
                action_type=action_type,
                user_id=user_id,
                user_name=user_name,
                user_role=user_role,
                target_type=target_type,
                target_id=target_id,
                study_id=study_id,
                site_id=site_id,
                patient_key=patient_key,
                description=description,
                old_value=old_value,
                new_value=new_value,
                metadata=metadata or {},
                status=ActionStatus.COMPLETED,
                created_at=datetime.now(),
                completed_at=datetime.now(),
                is_reversible=is_reversible,
                session_id=session_id,
                ip_address=ip_address,
                user_agent=user_agent,
            )
            
            # Persist to database
            self._save_action(action)
            
            # Notify handlers
            self._notify_handlers(action)
            
            logger.debug(f"Action recorded: {action.action_id} - {action_type.value}")
            return action
            
        except Exception as e:
            logger.error(f"Error recording action: {e}")
            return None
    
    def _save_action(self, action: UserAction) -> None:
        """Save action to database."""
        if not self._db_manager or not self._db_manager.engine:
            return
        
        insert_query = text("""
            INSERT INTO user_actions (
                action_id, action_type, user_id, user_name, user_role,
                target_type, target_id, study_id, site_id, patient_key,
                description, old_value, new_value, metadata, status,
                created_at, completed_at, is_reversible, session_id,
                ip_address, user_agent
            ) VALUES (
                :action_id, :action_type, :user_id, :user_name, :user_role,
                :target_type, :target_id, :study_id, :site_id, :patient_key,
                :description, :old_value, :new_value, :metadata::jsonb, :status,
                :created_at, :completed_at, :is_reversible, :session_id,
                :ip_address, :user_agent
            )
        """)
        
        try:
            with self._db_manager.engine.connect() as conn:
                conn.execute(insert_query, {
                    'action_id': action.action_id,
                    'action_type': action.action_type.value,
                    'user_id': action.user_id,
                    'user_name': action.user_name,
                    'user_role': action.user_role,
                    'target_type': action.target_type,
                    'target_id': action.target_id,
                    'study_id': action.study_id,
                    'site_id': action.site_id,
                    'patient_key': action.patient_key,
                    'description': action.description,
                    'old_value': action.old_value,
                    'new_value': action.new_value,
                    'metadata': json.dumps(action.metadata),
                    'status': action.status.value,
                    'created_at': action.created_at,
                    'completed_at': action.completed_at,
                    'is_reversible': action.is_reversible,
                    'session_id': action.session_id,
                    'ip_address': action.ip_address,
                    'user_agent': action.user_agent,
                })
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving action: {e}")
    
    def register_handler(self, action_type: ActionType, handler: Callable) -> None:
        """Register a handler for an action type."""
        if action_type not in self._action_handlers:
            self._action_handlers[action_type] = []
        self._action_handlers[action_type].append(handler)
    
    def _notify_handlers(self, action: UserAction) -> None:
        """Notify registered handlers of an action."""
        handlers = self._action_handlers.get(action.action_type, [])
        for handler in handlers:
            try:
                handler(action)
            except Exception as e:
                logger.error(f"Handler error for {action.action_type}: {e}")
    
    def get_user_actions(
        self,
        user_id: str,
        action_types: Optional[List[ActionType]] = None,
        limit: int = 100,
        since: Optional[datetime] = None,
    ) -> List[UserAction]:
        """Get actions by user."""
        if not self._db_manager or not self._db_manager.engine:
            return []
        
        try:
            query = "SELECT * FROM user_actions WHERE user_id = :user_id"
            params: Dict[str, Any] = {'user_id': user_id}
            
            if action_types:
                type_values = [at.value for at in action_types]
                query += f" AND action_type IN ({','.join([f':at{i}' for i in range(len(type_values))])})"
                for i, v in enumerate(type_values):
                    params[f'at{i}'] = v
            
            if since:
                query += " AND created_at >= :since"
                params['since'] = since
            
            query += " ORDER BY created_at DESC LIMIT :limit"
            params['limit'] = limit
            
            with self._db_manager.engine.connect() as conn:
                df = pd.read_sql(text(query), conn, params=params)
            
            return self._df_to_actions(df)
            
        except Exception as e:
            logger.error(f"Error getting user actions: {e}")
            return []
    
    def get_target_actions(
        self,
        target_type: str,
        target_id: str,
        limit: int = 100,
    ) -> List[UserAction]:
        """Get actions for a specific target."""
        if not self._db_manager or not self._db_manager.engine:
            return []
        
        try:
            query = text("""
                SELECT * FROM user_actions 
                WHERE target_type = :target_type AND target_id = :target_id
                ORDER BY created_at DESC LIMIT :limit
            """)
            
            with self._db_manager.engine.connect() as conn:
                df = pd.read_sql(query, conn, params={
                    'target_type': target_type,
                    'target_id': target_id,
                    'limit': limit,
                })
            
            return self._df_to_actions(df)
            
        except Exception as e:
            logger.error(f"Error getting target actions: {e}")
            return []
    
    def get_recent_actions(
        self,
        limit: int = 100,
        action_types: Optional[List[ActionType]] = None,
        study_id: Optional[str] = None,
        site_id: Optional[str] = None,
    ) -> List[UserAction]:
        """Get recent actions with optional filters."""
        if not self._db_manager or not self._db_manager.engine:
            return []
        
        try:
            query = "SELECT * FROM user_actions WHERE 1=1"
            params: Dict[str, Any] = {}
            
            if action_types:
                type_values = [at.value for at in action_types]
                query += f" AND action_type IN ({','.join([f':at{i}' for i in range(len(type_values))])})"
                for i, v in enumerate(type_values):
                    params[f'at{i}'] = v
            
            if study_id:
                query += " AND study_id = :study_id"
                params['study_id'] = study_id
            
            if site_id:
                query += " AND site_id = :site_id"
                params['site_id'] = site_id
            
            query += " ORDER BY created_at DESC LIMIT :limit"
            params['limit'] = limit
            
            with self._db_manager.engine.connect() as conn:
                df = pd.read_sql(text(query), conn, params=params)
            
            return self._df_to_actions(df)
            
        except Exception as e:
            logger.error(f"Error getting recent actions: {e}")
            return []
    
    def _df_to_actions(self, df: pd.DataFrame) -> List[UserAction]:
        """Convert DataFrame to list of UserAction objects."""
        actions = []
        for _, row in df.iterrows():
            try:
                action = UserAction(
                    action_id=row['action_id'],
                    action_type=ActionType(row.get('action_type', 'view')),
                    user_id=row['user_id'],
                    user_name=row.get('user_name', ''),
                    user_role=row.get('user_role', ''),
                    target_type=row.get('target_type', ''),
                    target_id=row.get('target_id', ''),
                    study_id=row.get('study_id'),
                    site_id=row.get('site_id'),
                    patient_key=row.get('patient_key'),
                    description=row.get('description', ''),
                    old_value=row.get('old_value'),
                    new_value=row.get('new_value'),
                    metadata=row.get('metadata', {}) or {},
                    status=ActionStatus(row.get('status', 'completed')),
                    error_message=row.get('error_message'),
                    created_at=row.get('created_at', datetime.now()),
                    completed_at=row.get('completed_at'),
                    is_reversible=row.get('is_reversible', False),
                    reversal_action_id=row.get('reversal_action_id'),
                    session_id=row.get('session_id'),
                    ip_address=row.get('ip_address'),
                    user_agent=row.get('user_agent'),
                )
                actions.append(action)
            except Exception as e:
                logger.warning(f"Error parsing action: {e}")
        
        return actions
    
    def get_action_stats(
        self,
        days: int = 7,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get action statistics."""
        if not self._db_manager or not self._db_manager.engine:
            return {}
        
        try:
            since = datetime.now() - timedelta(days=days)
            
            query = """
                SELECT 
                    action_type,
                    COUNT(*) as count,
                    COUNT(DISTINCT user_id) as unique_users
                FROM user_actions
                WHERE created_at >= :since
            """
            params: Dict[str, Any] = {'since': since}
            
            if user_id:
                query += " AND user_id = :user_id"
                params['user_id'] = user_id
            
            query += " GROUP BY action_type ORDER BY count DESC"
            
            with self._db_manager.engine.connect() as conn:
                df = pd.read_sql(text(query), conn, params=params)
            
            # Total count
            total_query = text("""
                SELECT COUNT(*) as total FROM user_actions WHERE created_at >= :since
            """)
            
            with self._db_manager.engine.connect() as conn:
                total_df = pd.read_sql(total_query, conn, params={'since': since})
            
            total = int(total_df['total'].iloc[0]) if not total_df.empty else 0
            
            by_type = {}
            for _, row in df.iterrows():
                by_type[row['action_type']] = {
                    'count': int(row['count']),
                    'unique_users': int(row['unique_users']),
                }
            
            return {
                'total_actions': total,
                'days': days,
                'by_type': by_type,
            }
            
        except Exception as e:
            logger.error(f"Error getting action stats: {e}")
            return {}


# ============================================================
# SINGLETON ACCESSORS
# ============================================================

_action_service: Optional[ActionPersistenceService] = None


def get_action_service() -> ActionPersistenceService:
    """Get singleton action persistence service."""
    global _action_service
    if _action_service is None:
        _action_service = ActionPersistenceService()
    return _action_service


def reset_action_service() -> None:
    """Reset the singleton (for testing)."""
    global _action_service
    _action_service = None
    ActionPersistenceService._instance = None


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def record_action(
    action_type: ActionType,
    user_id: str,
    **kwargs
) -> Optional[UserAction]:
    """Record a user action (convenience function)."""
    service = get_action_service()
    return service.record_action(action_type=action_type, user_id=user_id, **kwargs)


def get_user_activity(user_id: str, **kwargs) -> List[UserAction]:
    """Get user activity (convenience function)."""
    service = get_action_service()
    return service.get_user_actions(user_id, **kwargs)
