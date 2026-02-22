"""
a6on-i - Notification Service
========================================
PostgreSQL-backed notification queue system.

Features:
- Real-time notification queue with PostgreSQL persistence
- Priority-based delivery (urgent, high, normal, low)
- Multi-channel support (in-app, email, SMS, push)
- User preferences management
- Read/unread tracking
- Batch notifications support
"""

import logging
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any
from sqlalchemy import text
import pandas as pd

from src.database.connection import get_db_manager

logger = logging.getLogger(__name__)


# ============================================================
# ENUMS
# ============================================================

class NotificationType(Enum):
    """Types of notifications"""
    ALERT = "alert"
    MENTION = "mention"
    ESCALATION = "escalation"
    TASK = "task"
    ISSUE = "issue"
    SAFETY = "safety"
    SYSTEM = "system"
    DIGEST = "digest"
    REPORT = "report"
    COLLABORATION = "collaboration"


class NotificationPriority(Enum):
    """Notification priority levels"""
    URGENT = "urgent"      # Immediate delivery
    HIGH = "high"          # Within 15 minutes
    NORMAL = "normal"      # Within 1 hour
    LOW = "low"            # Batched, daily digest


class NotificationChannel(Enum):
    """Delivery channels"""
    IN_APP = "in_app"
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"
    SLACK = "slack"
    TEAMS = "teams"


class NotificationStatus(Enum):
    """Notification lifecycle status"""
    PENDING = "pending"
    QUEUED = "queued"
    SENT = "sent"
    DELIVERED = "delivered"
    READ = "read"
    ARCHIVED = "archived"
    FAILED = "failed"


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class Notification:
    """Individual notification"""
    notification_id: str
    user_id: str
    title: str
    message: str
    notification_type: NotificationType
    priority: NotificationPriority
    channel: NotificationChannel = NotificationChannel.IN_APP
    status: NotificationStatus = NotificationStatus.PENDING
    
    # Source information
    source_type: str = ""  # issue, escalation, room, workspace
    source_id: str = ""
    source_url: str = ""
    
    # Context
    study_id: Optional[str] = None
    site_id: Optional[str] = None
    patient_key: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Tracking
    created_at: datetime = field(default_factory=datetime.now)
    sent_at: Optional[datetime] = None
    read_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'notification_id': self.notification_id,
            'user_id': self.user_id,
            'title': self.title,
            'message': self.message,
            'notification_type': self.notification_type.value,
            'priority': self.priority.value,
            'channel': self.channel.value,
            'status': self.status.value,
            'source_type': self.source_type,
            'source_id': self.source_id,
            'source_url': self.source_url,
            'study_id': self.study_id,
            'site_id': self.site_id,
            'patient_key': self.patient_key,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'sent_at': self.sent_at.isoformat() if self.sent_at else None,
            'read_at': self.read_at.isoformat() if self.read_at else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Notification':
        """Create from dictionary."""
        return cls(
            notification_id=data.get('notification_id', ''),
            user_id=data.get('user_id', ''),
            title=data.get('title', ''),
            message=data.get('message', ''),
            notification_type=NotificationType(data.get('notification_type', 'system')),
            priority=NotificationPriority(data.get('priority', 'normal')),
            channel=NotificationChannel(data.get('channel', 'in_app')),
            status=NotificationStatus(data.get('status', 'pending')),
            source_type=data.get('source_type', ''),
            source_id=data.get('source_id', ''),
            source_url=data.get('source_url', ''),
            study_id=data.get('study_id'),
            site_id=data.get('site_id'),
            patient_key=data.get('patient_key'),
            metadata=data.get('metadata', {}),
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else datetime.now(),
            sent_at=datetime.fromisoformat(data['sent_at']) if data.get('sent_at') else None,
            read_at=datetime.fromisoformat(data['read_at']) if data.get('read_at') else None,
        )


# ============================================================
# NOTIFICATION QUEUE
# ============================================================

class NotificationQueue:
    """Queue for managing pending notifications."""
    
    def __init__(self):
        self._queue: List[Notification] = []
    
    def enqueue(self, notification: Notification) -> None:
        """Add notification to queue."""
        self._queue.append(notification)
    
    def dequeue(self) -> Optional[Notification]:
        """Remove and return highest priority notification."""
        if not self._queue:
            return None
        
        # Sort by priority (urgent first)
        priority_order = {
            NotificationPriority.URGENT: 0,
            NotificationPriority.HIGH: 1,
            NotificationPriority.NORMAL: 2,
            NotificationPriority.LOW: 3,
        }
        self._queue.sort(key=lambda n: priority_order.get(n.priority, 99))
        return self._queue.pop(0)
    
    def peek(self) -> Optional[Notification]:
        """View next notification without removing."""
        if not self._queue:
            return None
        return self._queue[0]
    
    def size(self) -> int:
        """Get queue size."""
        return len(self._queue)
    
    def clear(self) -> None:
        """Clear the queue."""
        self._queue.clear()


# ============================================================
# NOTIFICATION SERVICE
# ============================================================

class NotificationService:
    """PostgreSQL-backed notification service."""
    
    _instance: Optional['NotificationService'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._db_manager = None
        self._queue = NotificationQueue()
        self._initialize()
        self._initialized = True
    
    def _initialize(self):
        """Initialize database connection and tables."""
        try:
            self._db_manager = get_db_manager()
            self._ensure_tables()
            logger.info("NotificationService initialized with PostgreSQL")
        except Exception as e:
            logger.error(f"NotificationService initialization failed: {e}")
            raise
    
    def _ensure_tables(self):
        """Ensure notification tables exist."""
        create_notifications_table = text("""
            CREATE TABLE IF NOT EXISTS notifications (
                notification_id VARCHAR(100) PRIMARY KEY,
                user_id VARCHAR(100) NOT NULL,
                title VARCHAR(500) NOT NULL,
                message TEXT,
                notification_type VARCHAR(50) NOT NULL,
                priority VARCHAR(20) NOT NULL DEFAULT 'normal',
                channel VARCHAR(30) NOT NULL DEFAULT 'in_app',
                status VARCHAR(30) NOT NULL DEFAULT 'pending',
                source_type VARCHAR(50),
                source_id VARCHAR(100),
                source_url VARCHAR(500),
                study_id VARCHAR(50),
                site_id VARCHAR(50),
                patient_key VARCHAR(50),
                metadata JSONB,
                created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                sent_at TIMESTAMP,
                read_at TIMESTAMP
            )
        """)
        
        create_user_preferences_table = text("""
            CREATE TABLE IF NOT EXISTS notification_preferences (
                user_id VARCHAR(100) PRIMARY KEY,
                email_enabled BOOLEAN DEFAULT TRUE,
                sms_enabled BOOLEAN DEFAULT FALSE,
                push_enabled BOOLEAN DEFAULT TRUE,
                in_app_enabled BOOLEAN DEFAULT TRUE,
                quiet_hours_start TIME,
                quiet_hours_end TIME,
                digest_frequency VARCHAR(20) DEFAULT 'daily',
                updated_at TIMESTAMP DEFAULT NOW()
            )
        """)
        
        create_indexes = [
            text("CREATE INDEX IF NOT EXISTS idx_notifications_user ON notifications(user_id)"),
            text("CREATE INDEX IF NOT EXISTS idx_notifications_status ON notifications(status)"),
            text("CREATE INDEX IF NOT EXISTS idx_notifications_created ON notifications(created_at DESC)"),
            text("CREATE INDEX IF NOT EXISTS idx_notifications_priority ON notifications(priority)"),
        ]
        
        try:
            if self._db_manager and self._db_manager.engine:
                with self._db_manager.engine.connect() as conn:
                    conn.execute(create_notifications_table)
                    conn.execute(create_user_preferences_table)
                    for idx in create_indexes:
                        try:
                            conn.execute(idx)
                        except Exception:
                            pass  # Index may already exist
                    conn.commit()
                    logger.info("Notification tables verified")
        except Exception as e:
            logger.warning(f"Could not create notification tables: {e}")
    
    def _generate_id(self) -> str:
        """Generate unique notification ID."""
        import uuid
        return f"NOTIF-{uuid.uuid4().hex[:12].upper()}"
    
    def send_notification(
        self,
        user_id: str,
        title: str,
        message: str,
        notification_type: NotificationType = NotificationType.SYSTEM,
        priority: NotificationPriority = NotificationPriority.NORMAL,
        channel: NotificationChannel = NotificationChannel.IN_APP,
        source_type: str = "",
        source_id: str = "",
        source_url: str = "",
        study_id: Optional[str] = None,
        site_id: Optional[str] = None,
        patient_key: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> Optional[Notification]:
        """Send a notification to a user."""
        try:
            notification = Notification(
                notification_id=self._generate_id(),
                user_id=user_id,
                title=title,
                message=message,
                notification_type=notification_type,
                priority=priority,
                channel=channel,
                status=NotificationStatus.PENDING,
                source_type=source_type,
                source_id=source_id,
                source_url=source_url,
                study_id=study_id,
                site_id=site_id,
                patient_key=patient_key,
                metadata=metadata or {},
                created_at=datetime.now(),
            )
            
            # Persist to database
            self._save_notification(notification)
            
            # Mark as sent
            notification.status = NotificationStatus.SENT
            notification.sent_at = datetime.now()
            self._update_notification_status(notification.notification_id, NotificationStatus.SENT)
            
            logger.info(f"Notification sent: {notification.notification_id} to {user_id}")
            return notification
            
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
            return None
    
    def _save_notification(self, notification: Notification) -> None:
        """Save notification to database."""
        if not self._db_manager or not self._db_manager.engine:
            return
            
        insert_query = text("""
            INSERT INTO notifications (
                notification_id, user_id, title, message, notification_type,
                priority, channel, status, source_type, source_id, source_url,
                study_id, site_id, patient_key, metadata, created_at
            ) VALUES (
                :notification_id, :user_id, :title, :message, :notification_type,
                :priority, :channel, :status, :source_type, :source_id, :source_url,
                :study_id, :site_id, :patient_key, :metadata::jsonb, :created_at
            )
        """)
        
        try:
            import json
            with self._db_manager.engine.connect() as conn:
                conn.execute(insert_query, {
                    'notification_id': notification.notification_id,
                    'user_id': notification.user_id,
                    'title': notification.title,
                    'message': notification.message,
                    'notification_type': notification.notification_type.value,
                    'priority': notification.priority.value,
                    'channel': notification.channel.value,
                    'status': notification.status.value,
                    'source_type': notification.source_type,
                    'source_id': notification.source_id,
                    'source_url': notification.source_url,
                    'study_id': notification.study_id,
                    'site_id': notification.site_id,
                    'patient_key': notification.patient_key,
                    'metadata': json.dumps(notification.metadata),
                    'created_at': notification.created_at,
                })
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving notification: {e}")
    
    def _update_notification_status(
        self, 
        notification_id: str, 
        status: NotificationStatus,
        read_at: Optional[datetime] = None
    ) -> None:
        """Update notification status."""
        if not self._db_manager or not self._db_manager.engine:
            return
            
        if status == NotificationStatus.SENT:
            update_query = text("""
                UPDATE notifications 
                SET status = :status, sent_at = NOW()
                WHERE notification_id = :notification_id
            """)
        elif status == NotificationStatus.READ:
            update_query = text("""
                UPDATE notifications 
                SET status = :status, read_at = :read_at
                WHERE notification_id = :notification_id
            """)
        else:
            update_query = text("""
                UPDATE notifications 
                SET status = :status
                WHERE notification_id = :notification_id
            """)
        
        try:
            with self._db_manager.engine.connect() as conn:
                conn.execute(update_query, {
                    'notification_id': notification_id,
                    'status': status.value,
                    'read_at': read_at or datetime.now(),
                })
                conn.commit()
        except Exception as e:
            logger.error(f"Error updating notification status: {e}")
    
    def get_user_notifications(
        self,
        user_id: str,
        status: Optional[NotificationStatus] = None,
        limit: int = 50,
        include_read: bool = True,
    ) -> List[Notification]:
        """Get notifications for a user."""
        if not self._db_manager or not self._db_manager.engine:
            return []
        
        try:
            query = """
                SELECT * FROM notifications
                WHERE user_id = :user_id
            """
            
            params: Dict[str, Any] = {'user_id': user_id}
            
            if status:
                query += " AND status = :status"
                params['status'] = status.value
            
            if not include_read:
                query += " AND status != 'read'"
            
            query += " ORDER BY created_at DESC LIMIT :limit"
            params['limit'] = limit
            
            with self._db_manager.engine.connect() as conn:
                df = pd.read_sql(text(query), conn, params=params)
            
            notifications = []
            for _, row in df.iterrows():
                try:
                    notif = Notification(
                        notification_id=row['notification_id'],
                        user_id=row['user_id'],
                        title=row['title'],
                        message=row.get('message', ''),
                        notification_type=NotificationType(row.get('notification_type', 'system')),
                        priority=NotificationPriority(row.get('priority', 'normal')),
                        channel=NotificationChannel(row.get('channel', 'in_app')),
                        status=NotificationStatus(row.get('status', 'pending')),
                        source_type=row.get('source_type', ''),
                        source_id=row.get('source_id', ''),
                        source_url=row.get('source_url', ''),
                        study_id=row.get('study_id'),
                        site_id=row.get('site_id'),
                        patient_key=row.get('patient_key'),
                        metadata=row.get('metadata', {}) or {},
                        created_at=row.get('created_at', datetime.now()),
                        sent_at=row.get('sent_at'),
                        read_at=row.get('read_at'),
                    )
                    notifications.append(notif)
                except Exception as e:
                    logger.warning(f"Error parsing notification: {e}")
            
            return notifications
            
        except Exception as e:
            logger.error(f"Error getting user notifications: {e}")
            return []
    
    def mark_as_read(self, notification_id: str) -> bool:
        """Mark a notification as read."""
        try:
            self._update_notification_status(
                notification_id, 
                NotificationStatus.READ,
                read_at=datetime.now()
            )
            return True
        except Exception as e:
            logger.error(f"Error marking notification as read: {e}")
            return False
    
    def mark_all_as_read(self, user_id: str) -> int:
        """Mark all notifications as read for a user."""
        if not self._db_manager or not self._db_manager.engine:
            return 0
        
        try:
            update_query = text("""
                UPDATE notifications 
                SET status = 'read', read_at = NOW()
                WHERE user_id = :user_id AND status != 'read'
            """)
            
            with self._db_manager.engine.connect() as conn:
                result = conn.execute(update_query, {'user_id': user_id})
                conn.commit()
                return result.rowcount
                
        except Exception as e:
            logger.error(f"Error marking all as read: {e}")
            return 0
    
    def get_unread_count(self, user_id: str) -> int:
        """Get count of unread notifications for a user."""
        if not self._db_manager or not self._db_manager.engine:
            return 0
        
        try:
            query = text("""
                SELECT COUNT(*) as cnt FROM notifications
                WHERE user_id = :user_id AND status NOT IN ('read', 'archived')
            """)
            
            with self._db_manager.engine.connect() as conn:
                result = pd.read_sql(query, conn, params={'user_id': user_id})
                return int(result['cnt'].iloc[0]) if not result.empty else 0
                
        except Exception as e:
            logger.error(f"Error getting unread count: {e}")
            return 0
    
    def delete_notification(self, notification_id: str) -> bool:
        """Delete a notification."""
        if not self._db_manager or not self._db_manager.engine:
            return False
        
        try:
            delete_query = text("""
                DELETE FROM notifications WHERE notification_id = :notification_id
            """)
            
            with self._db_manager.engine.connect() as conn:
                conn.execute(delete_query, {'notification_id': notification_id})
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Error deleting notification: {e}")
            return False
    
    def get_notification_stats(self, user_id: str) -> Dict[str, Any]:
        """Get notification statistics for a user."""
        if not self._db_manager or not self._db_manager.engine:
            return {}
        
        try:
            stats_query = text("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN status NOT IN ('read', 'archived') THEN 1 ELSE 0 END) as unread,
                    SUM(CASE WHEN priority = 'urgent' AND status NOT IN ('read', 'archived') THEN 1 ELSE 0 END) as urgent_unread,
                    SUM(CASE WHEN priority = 'high' AND status NOT IN ('read', 'archived') THEN 1 ELSE 0 END) as high_unread
                FROM notifications
                WHERE user_id = :user_id
            """)
            
            with self._db_manager.engine.connect() as conn:
                result = pd.read_sql(stats_query, conn, params={'user_id': user_id})
            
            if result.empty:
                return {'total': 0, 'unread': 0, 'urgent_unread': 0, 'high_unread': 0}
            
            row = result.iloc[0]
            return {
                'total': int(row['total']),
                'unread': int(row['unread']),
                'urgent_unread': int(row['urgent_unread']),
                'high_unread': int(row['high_unread']),
            }
            
        except Exception as e:
            logger.error(f"Error getting notification stats: {e}")
            return {}


# ============================================================
# SINGLETON ACCESSORS
# ============================================================

_notification_service: Optional[NotificationService] = None


def get_notification_service() -> NotificationService:
    """Get singleton notification service."""
    global _notification_service
    if _notification_service is None:
        _notification_service = NotificationService()
    return _notification_service


def reset_notification_service() -> None:
    """Reset the singleton (for testing)."""
    global _notification_service
    _notification_service = None
    NotificationService._instance = None


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def send_notification(
    user_id: str,
    title: str,
    message: str,
    notification_type: NotificationType = NotificationType.SYSTEM,
    priority: NotificationPriority = NotificationPriority.NORMAL,
    **kwargs
) -> Optional[Notification]:
    """Send a notification (convenience function)."""
    service = get_notification_service()
    return service.send_notification(
        user_id=user_id,
        title=title,
        message=message,
        notification_type=notification_type,
        priority=priority,
        **kwargs
    )


def get_user_notifications(user_id: str, **kwargs) -> List[Notification]:
    """Get user notifications (convenience function)."""
    service = get_notification_service()
    return service.get_user_notifications(user_id, **kwargs)


def mark_as_read(notification_id: str) -> bool:
    """Mark notification as read (convenience function)."""
    service = get_notification_service()
    return service.mark_as_read(notification_id)
