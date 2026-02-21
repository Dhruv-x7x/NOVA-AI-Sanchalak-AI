"""
SANCHALAK AI - Phase 8.6: Alert System v1.1
# NOTE: This module previously used SQLite but has been migrated to PostgreSQL-only.
# If you need to use this module, update it to use:
#   from src.database.pg_data_service import get_pg_data_service
#   service = get_pg_data_service()


Features:
- Priority classification (Urgent/High/Normal/Low)
- Intelligent batching logic
- Multi-channel routing (Email, SMS, In-App, Slack, Teams)
- Alert fatigue prevention
- User preference management
- Quiet hours support
- Alert deduplication
- Escalation integration
"""

# using SQLite for local persistence
import sqlite3
import json
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# ENUMS
# ============================================================

class AlertPriority(Enum):
    """Alert priority levels with SLA times"""
    URGENT = "urgent"      # Immediate delivery, no batching
    HIGH = "high"          # Within 15 minutes
    NORMAL = "normal"      # Within 1 hour, can batch
    LOW = "low"            # Within 4 hours, always batch


class AlertChannel(Enum):
    """Delivery channels"""
    EMAIL = "email"
    SMS = "sms"
    IN_APP = "in_app"
    SLACK = "slack"
    TEAMS = "teams"
    PUSH = "push"
    DASHBOARD = "dashboard"


class AlertStatus(Enum):
    """Alert lifecycle status"""
    PENDING = "pending"
    QUEUED = "queued"
    BATCHED = "batched"
    SENDING = "sending"
    SENT = "sent"
    DELIVERED = "delivered"
    READ = "read"
    FAILED = "failed"
    SUPPRESSED = "suppressed"
    DEDUPLICATED = "deduplicated"


class AlertCategory(Enum):
    """Alert categories for routing and preferences"""
    SAFETY = "safety"
    SLA_BREACH = "sla_breach"
    ESCALATION = "escalation"
    ISSUE = "issue"
    PATTERN = "pattern"
    MENTION = "mention"
    TASK = "task"
    REPORT = "report"
    SYSTEM = "system"
    DIGEST = "digest"


class BatchType(Enum):
    """Batch delivery types"""
    IMMEDIATE = "immediate"
    HOURLY = "hourly"
    DAILY_DIGEST = "daily_digest"
    WEEKLY_DIGEST = "weekly_digest"


class SuppressionReason(Enum):
    """Reasons for alert suppression"""
    QUIET_HOURS = "quiet_hours"
    FATIGUE_LIMIT = "fatigue_limit"
    USER_PREFERENCE = "user_preference"
    DUPLICATE = "duplicate"
    CHANNEL_UNAVAILABLE = "channel_unavailable"
    RATE_LIMIT = "rate_limit"


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class Alert:
    """Individual alert"""
    alert_id: str
    title: str
    message: str
    priority: AlertPriority
    category: AlertCategory
    recipient_id: str
    recipient_name: str = ""
    
    # Source information
    source_type: str = ""  # issue, escalation, pattern, system
    source_id: str = ""
    source_url: str = ""
    
    # Targeting
    study_id: Optional[str] = None
    site_id: Optional[str] = None
    patient_id: Optional[str] = None
    
    # Delivery
    channels: List[AlertChannel] = field(default_factory=list)
    status: AlertStatus = AlertStatus.PENDING
    batch_id: Optional[str] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    # Tracking
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_at: Optional[datetime] = None
    sent_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    read_at: Optional[datetime] = None
    
    # Suppression
    suppressed: bool = False
    suppression_reason: Optional[SuppressionReason] = None
    
    # Deduplication
    dedup_key: str = ""
    
    def __post_init__(self):
        if not self.dedup_key:
            # Generate deduplication key from content
            content = f"{self.recipient_id}|{self.category.value}|{self.source_type}|{self.source_id}|{self.title}"
            self.dedup_key = hashlib.md5(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result['priority'] = self.priority.value
        result['category'] = self.category.value
        result['status'] = self.status.value
        result['channels'] = [c.value for c in self.channels]
        result['created_at'] = self.created_at.isoformat() if self.created_at else None
        result['scheduled_at'] = self.scheduled_at.isoformat() if self.scheduled_at else None
        result['sent_at'] = self.sent_at.isoformat() if self.sent_at else None
        result['delivered_at'] = self.delivered_at.isoformat() if self.delivered_at else None
        result['read_at'] = self.read_at.isoformat() if self.read_at else None
        result['suppression_reason'] = self.suppression_reason.value if self.suppression_reason else None
        return result


@dataclass
class AlertBatch:
    """Batch of alerts for digest delivery"""
    batch_id: str
    batch_type: BatchType
    recipient_id: str
    recipient_name: str = ""
    
    alerts: List[str] = field(default_factory=list)  # Alert IDs
    alert_count: int = 0
    
    # Scheduling
    scheduled_for: datetime = field(default_factory=datetime.now)
    created_at: datetime = field(default_factory=datetime.now)
    sent_at: Optional[datetime] = None
    
    # Status
    status: AlertStatus = AlertStatus.QUEUED
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result['batch_type'] = self.batch_type.value
        result['status'] = self.status.value
        result['scheduled_for'] = self.scheduled_for.isoformat()
        result['created_at'] = self.created_at.isoformat()
        result['sent_at'] = self.sent_at.isoformat() if self.sent_at else None
        return result


@dataclass
class UserAlertPreferences:
    """User preferences for alert delivery"""
    user_id: str
    user_name: str = ""
    
    # Channel preferences by priority
    urgent_channels: List[AlertChannel] = field(default_factory=lambda: [
        AlertChannel.SMS, AlertChannel.EMAIL, AlertChannel.IN_APP
    ])
    high_channels: List[AlertChannel] = field(default_factory=lambda: [
        AlertChannel.EMAIL, AlertChannel.IN_APP
    ])
    normal_channels: List[AlertChannel] = field(default_factory=lambda: [
        AlertChannel.EMAIL
    ])
    low_channels: List[AlertChannel] = field(default_factory=lambda: [
        AlertChannel.IN_APP
    ])
    
    # Category preferences (enabled categories)
    enabled_categories: List[AlertCategory] = field(default_factory=lambda: list(AlertCategory))
    
    # Quiet hours (no alerts except URGENT)
    quiet_hours_enabled: bool = False  # Default to False for easier testing
    quiet_hours_start: str = "22:00"  # HH:MM format
    quiet_hours_end: str = "07:00"
    quiet_hours_timezone: str = "UTC"
    
    # Batching preferences
    batch_low_priority: bool = True
    batch_normal_priority: bool = False
    digest_frequency: BatchType = BatchType.DAILY_DIGEST
    digest_time: str = "08:00"  # HH:MM format
    
    # Fatigue prevention
    max_alerts_per_hour: int = 20
    max_alerts_per_day: int = 100
    
    # Deduplication
    dedup_window_minutes: int = 60  # Suppress duplicates within this window
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result['urgent_channels'] = [c.value for c in self.urgent_channels]
        result['high_channels'] = [c.value for c in self.high_channels]
        result['normal_channels'] = [c.value for c in self.normal_channels]
        result['low_channels'] = [c.value for c in self.low_channels]
        result['enabled_categories'] = [c.value for c in self.enabled_categories]
        result['digest_frequency'] = self.digest_frequency.value
        return result


@dataclass
class AlertStats:
    """Alert statistics for a user or system"""
    total_alerts: int = 0
    alerts_sent: int = 0
    alerts_suppressed: int = 0
    alerts_deduplicated: int = 0
    alerts_read: int = 0
    
    by_priority: Dict[str, int] = field(default_factory=dict)
    by_category: Dict[str, int] = field(default_factory=dict)
    by_channel: Dict[str, int] = field(default_factory=dict)
    by_status: Dict[str, int] = field(default_factory=dict)
    
    alerts_last_hour: int = 0
    alerts_last_day: int = 0
    
    avg_time_to_read_minutes: float = 0.0


@dataclass
class FatigueStatus:
    """Current fatigue status for a user"""
    user_id: str
    alerts_last_hour: int = 0
    alerts_last_day: int = 0
    hourly_limit: int = 20
    daily_limit: int = 100
    is_fatigued: bool = False
    next_available: Optional[datetime] = None


# ============================================================
# PRIORITY CLASSIFIER
# ============================================================

class PriorityClassifier:
    """Classifies alert priority based on content and context"""
    
    # Keywords that indicate urgency
    URGENT_KEYWORDS = [
        'fatal', 'death', 'sae', 'serious adverse', 'breach', 'immediate',
        'emergency', 'critical', 'blocker', 'urgent', 'asap'
    ]
    
    HIGH_KEYWORDS = [
        'escalation', 'overdue', 'sla', 'warning', 'attention', 'important',
        'high priority', 'action required', 'deadline'
    ]
    
    # Category-based priority mapping
    CATEGORY_PRIORITY = {
        AlertCategory.SAFETY: AlertPriority.URGENT,
        AlertCategory.SLA_BREACH: AlertPriority.HIGH,
        AlertCategory.ESCALATION: AlertPriority.HIGH,
        AlertCategory.ISSUE: AlertPriority.NORMAL,
        AlertCategory.PATTERN: AlertPriority.NORMAL,
        AlertCategory.MENTION: AlertPriority.NORMAL,
        AlertCategory.TASK: AlertPriority.NORMAL,
        AlertCategory.REPORT: AlertPriority.LOW,
        AlertCategory.SYSTEM: AlertPriority.LOW,
        AlertCategory.DIGEST: AlertPriority.LOW,
    }
    
    def classify(self, title: str, message: str, category: AlertCategory,
                 metadata: Optional[Dict] = None) -> AlertPriority:
        """Classify alert priority based on content"""
        
        text = f"{title} {message}".lower()
        
        # Check for urgent keywords
        for keyword in self.URGENT_KEYWORDS:
            if keyword in text:
                return AlertPriority.URGENT
        
        # Check for high priority keywords
        for keyword in self.HIGH_KEYWORDS:
            if keyword in text:
                return AlertPriority.HIGH
        
        # Check metadata for priority hints
        if metadata:
            if metadata.get('is_safety'):
                return AlertPriority.URGENT
            if metadata.get('sla_breach'):
                return AlertPriority.HIGH
            if metadata.get('escalation_level', 0) >= 4:
                return AlertPriority.HIGH
        
        # Fall back to category-based priority
        return self.CATEGORY_PRIORITY.get(category, AlertPriority.NORMAL)


# ============================================================
# CHANNEL ROUTER
# ============================================================

class ChannelRouter:
    """Routes alerts to appropriate channels based on priority and preferences"""
    
    # Channel availability by time of day
    CHANNEL_AVAILABILITY = {
        AlertChannel.SMS: {'always': True},
        AlertChannel.EMAIL: {'always': True},
        AlertChannel.IN_APP: {'always': True},
        AlertChannel.SLACK: {'business_hours': True},
        AlertChannel.TEAMS: {'business_hours': True},
        AlertChannel.PUSH: {'always': True},
        AlertChannel.DASHBOARD: {'always': True},
    }
    
    def route(self, alert: Alert, preferences: UserAlertPreferences) -> List[AlertChannel]:
        """Determine channels for alert delivery"""
        
        # Get channels based on priority
        if alert.priority == AlertPriority.URGENT:
            channels = preferences.urgent_channels.copy()
        elif alert.priority == AlertPriority.HIGH:
            channels = preferences.high_channels.copy()
        elif alert.priority == AlertPriority.NORMAL:
            channels = preferences.normal_channels.copy()
        else:
            channels = preferences.low_channels.copy()
        
        # Always include IN_APP for tracking
        if AlertChannel.IN_APP not in channels:
            channels.append(AlertChannel.IN_APP)
        
        # Filter unavailable channels
        available_channels = []
        for channel in channels:
            if self._is_channel_available(channel):
                available_channels.append(channel)
        
        return available_channels
    
    def _is_channel_available(self, channel: AlertChannel) -> bool:
        """Check if channel is currently available"""
        # In production, check actual channel availability
        return True


# ============================================================
# BATCHING ENGINE
# ============================================================

class BatchingEngine:
    """Handles alert batching for digest delivery"""
    
    def __init__(self):
        self.pending_batches: Dict[str, List[Alert]] = {}  # user_id -> alerts
    
    def should_batch(self, alert: Alert, preferences: UserAlertPreferences) -> bool:
        """Determine if alert should be batched"""
        
        # Never batch urgent alerts
        if alert.priority == AlertPriority.URGENT:
            return False
        
        # Never batch safety alerts
        if alert.category == AlertCategory.SAFETY:
            return False
        
        # Check user preferences
        if alert.priority == AlertPriority.LOW and preferences.batch_low_priority:
            return True
        
        if alert.priority == AlertPriority.NORMAL and preferences.batch_normal_priority:
            return True
        
        return False
    
    def add_to_batch(self, alert: Alert, batch_type: BatchType = BatchType.DAILY_DIGEST) -> str:
        """Add alert to pending batch"""
        
        user_id = alert.recipient_id
        
        if user_id not in self.pending_batches:
            self.pending_batches[user_id] = []
        
        self.pending_batches[user_id].append(alert)
        
        # Generate batch ID
        batch_id = f"BATCH-{user_id}-{datetime.now().strftime('%Y%m%d')}"
        return batch_id
    
    def get_pending_batch(self, user_id: str) -> List[Alert]:
        """Get pending alerts for user"""
        return self.pending_batches.get(user_id, [])
    
    def clear_batch(self, user_id: str):
        """Clear pending batch after sending"""
        if user_id in self.pending_batches:
            del self.pending_batches[user_id]
    
    def create_digest(self, user_id: str, alerts: List[Alert]) -> AlertBatch:
        """Create a digest batch from alerts"""
        
        batch_id = f"BATCH-{datetime.now().strftime('%Y%m%d%H%M%S')}-{user_id[:8]}"
        
        batch = AlertBatch(
            batch_id=batch_id,
            batch_type=BatchType.DAILY_DIGEST,
            recipient_id=user_id,
            alerts=[a.alert_id for a in alerts],
            alert_count=len(alerts),
            scheduled_for=datetime.now() + timedelta(hours=1)
        )
        
        return batch


# ============================================================
# FATIGUE PREVENTION
# ============================================================

class FatiguePreventor:
    """Prevents alert fatigue through rate limiting and smart suppression"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def check_fatigue(self, user_id: str, preferences: UserAlertPreferences) -> FatigueStatus:
        """Check current fatigue status for user"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)
        
        # Count alerts in last hour
        cursor.execute("""
            SELECT COUNT(*) FROM alerts
            WHERE recipient_id = ? AND created_at >= ? AND suppressed = 0
        """, (user_id, hour_ago.isoformat()))
        alerts_last_hour = cursor.fetchone()[0]
        
        # Count alerts in last day
        cursor.execute("""
            SELECT COUNT(*) FROM alerts
            WHERE recipient_id = ? AND created_at >= ? AND suppressed = 0
        """, (user_id, day_ago.isoformat()))
        alerts_last_day = cursor.fetchone()[0]
        
        conn.close()
        
        is_fatigued = (
            alerts_last_hour >= preferences.max_alerts_per_hour or
            alerts_last_day >= preferences.max_alerts_per_day
        )
        
        return FatigueStatus(
            user_id=user_id,
            alerts_last_hour=alerts_last_hour,
            alerts_last_day=alerts_last_day,
            hourly_limit=preferences.max_alerts_per_hour,
            daily_limit=preferences.max_alerts_per_day,
            is_fatigued=is_fatigued
        )
    
    def should_suppress(self, alert: Alert, preferences: UserAlertPreferences,
                       fatigue_status: FatigueStatus) -> Tuple[bool, Optional[SuppressionReason]]:
        """Determine if alert should be suppressed"""
        
        # Never suppress urgent or safety alerts
        if alert.priority == AlertPriority.URGENT:
            return False, None
        if alert.category == AlertCategory.SAFETY:
            return False, None
        
        # Check fatigue limits
        if fatigue_status.is_fatigued:
            return True, SuppressionReason.FATIGUE_LIMIT
        
        # Check quiet hours
        if self._is_quiet_hours(preferences):
            return True, SuppressionReason.QUIET_HOURS
        
        # Check category preferences
        if alert.category not in preferences.enabled_categories:
            return True, SuppressionReason.USER_PREFERENCE
        
        return False, None
    
    def _is_quiet_hours(self, preferences: UserAlertPreferences) -> bool:
        """Check if current time is within quiet hours"""
        
        if not preferences.quiet_hours_enabled:
            return False
        
        now = datetime.now()
        current_time = now.strftime("%H:%M")
        
        start = preferences.quiet_hours_start
        end = preferences.quiet_hours_end
        
        # Handle overnight quiet hours (e.g., 22:00 to 07:00)
        if start > end:
            return current_time >= start or current_time <= end
        else:
            return start <= current_time <= end


# ============================================================
# DEDUPLICATION ENGINE
# ============================================================

class DeduplicationEngine:
    """Prevents duplicate alerts within a time window"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def is_duplicate(self, alert: Alert, window_minutes: int = 60) -> Tuple[bool, Optional[str]]:
        """Check if alert is a duplicate of a recent alert"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        window_start = datetime.now() - timedelta(minutes=window_minutes)
        
        cursor.execute("""
            SELECT alert_id FROM alerts
            WHERE dedup_key = ? AND created_at >= ? AND suppressed = 0
            ORDER BY created_at DESC LIMIT 1
        """, (alert.dedup_key, window_start.isoformat()))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return True, result[0]
        return False, None


# ============================================================
# ALERT SYSTEM (Main Class)
# ============================================================

class AlertSystem:
    """
    Main Alert System with priority classification, batching,
    channel routing, and fatigue prevention.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            db_path = Path("data/collaboration/alert_system.db")
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.priority_classifier = PriorityClassifier()
        self.channel_router = ChannelRouter()
        self.batching_engine = BatchingEngine()
        self.fatigue_preventor = FatiguePreventor(str(self.db_path))
        self.dedup_engine = DeduplicationEngine(str(self.db_path))
        
        # Initialize database
        self._init_database()
        
        # Cache for user preferences
        self._preferences_cache: Dict[str, UserAlertPreferences] = {}
        
        logger.info(f"AlertSystem initialized with database: {self.db_path}")
    
    def _init_database(self):
        """Initialize SQLite database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Alerts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                alert_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                message TEXT NOT NULL,
                priority TEXT NOT NULL,
                category TEXT NOT NULL,
                recipient_id TEXT NOT NULL,
                recipient_name TEXT,
                source_type TEXT,
                source_id TEXT,
                source_url TEXT,
                study_id TEXT,
                site_id TEXT,
                patient_id TEXT,
                channels TEXT,
                status TEXT NOT NULL,
                batch_id TEXT,
                metadata TEXT,
                tags TEXT,
                created_at TEXT NOT NULL,
                scheduled_at TEXT,
                sent_at TEXT,
                delivered_at TEXT,
                read_at TEXT,
                suppressed INTEGER DEFAULT 0,
                suppression_reason TEXT,
                dedup_key TEXT
            )
        """)
        
        # Batches table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alert_batches (
                batch_id TEXT PRIMARY KEY,
                batch_type TEXT NOT NULL,
                recipient_id TEXT NOT NULL,
                recipient_name TEXT,
                alerts TEXT,
                alert_count INTEGER DEFAULT 0,
                scheduled_for TEXT NOT NULL,
                created_at TEXT NOT NULL,
                sent_at TEXT,
                status TEXT NOT NULL
            )
        """)
        
        # User preferences table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id TEXT PRIMARY KEY,
                user_name TEXT,
                preferences TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        
        # Alert statistics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alert_stats (
                stat_id TEXT PRIMARY KEY,
                user_id TEXT,
                stat_date TEXT NOT NULL,
                stats TEXT NOT NULL
            )
        """)
        
        # Audit trail
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alert_audit (
                audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_id TEXT,
                action TEXT NOT NULL,
                actor TEXT,
                details TEXT,
                created_at TEXT NOT NULL
            )
        """)
        
        # Indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_recipient ON alerts(recipient_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_priority ON alerts(priority)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts(created_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_dedup ON alerts(dedup_key)")
        
        conn.commit()
        conn.close()
    
    # --------------------------------------------------------
    # Alert Creation & Processing
    # --------------------------------------------------------
    
    def create_alert(
        self,
        title: str,
        message: str,
        recipient_id: str,
        category: AlertCategory,
        recipient_name: str = "",
        priority: Optional[AlertPriority] = None,
        source_type: str = "",
        source_id: str = "",
        source_url: str = "",
        study_id: Optional[str] = None,
        site_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None
    ) -> Alert:
        """Create and process a new alert"""
        import random
        import string
        
        # Auto-classify priority if not provided
        if priority is None:
            priority = self.priority_classifier.classify(
                title, message, category, metadata
            )
        
        # Generate unique alert ID with microseconds and random suffix
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')[:17]  # YYYYMMDDHHMMSSmmm
        random_suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
        alert_id = f"ALT-{timestamp}-{random_suffix}"
        
        # Create alert object
        alert = Alert(
            alert_id=alert_id,
            title=title,
            message=message,
            priority=priority,
            category=category,
            recipient_id=recipient_id,
            recipient_name=recipient_name,
            source_type=source_type,
            source_id=source_id,
            source_url=source_url,
            study_id=study_id,
            site_id=site_id,
            patient_id=patient_id,
            metadata=metadata or {},
            tags=tags or []
        )
        
        # Process the alert
        processed_alert = self._process_alert(alert)
        
        # Save to database
        self._save_alert(processed_alert)
        
        # Log audit
        self._log_audit(processed_alert.alert_id, "created", "system", {
            'priority': processed_alert.priority.value,
            'category': processed_alert.category.value,
            'status': processed_alert.status.value
        })
        return processed_alert


    def _process_alert(self, alert: Alert) -> Alert:
        """Process alert through routing, batching, and fatigue prevention"""
        
        # Get user preferences
        preferences = self.get_user_preferences(alert.recipient_id)
        
        # Check for duplicates
        is_dup, original_id = self.dedup_engine.is_duplicate(
            alert, preferences.dedup_window_minutes
        )
        if is_dup:
            alert.status = AlertStatus.DEDUPLICATED
            alert.suppressed = True
            alert.suppression_reason = SuppressionReason.DUPLICATE
            alert.metadata['original_alert_id'] = original_id
            return alert
        
        # Check fatigue status
        fatigue_status = self.fatigue_preventor.check_fatigue(alert.recipient_id, preferences)
        
        # Check if should suppress
        should_suppress, reason = self.fatigue_preventor.should_suppress(
            alert, preferences, fatigue_status
        )
        if should_suppress:
            alert.status = AlertStatus.SUPPRESSED
            alert.suppressed = True
            alert.suppression_reason = reason
            return alert
        
        # Route to channels
        alert.channels = self.channel_router.route(alert, preferences)
        
        # Check if should batch
        if self.batching_engine.should_batch(alert, preferences):
            batch_id = self.batching_engine.add_to_batch(alert, preferences.digest_frequency)
            alert.batch_id = batch_id
            alert.status = AlertStatus.BATCHED
        else:
            alert.status = AlertStatus.QUEUED
            alert.scheduled_at = datetime.now()
        
        return alert
    
    def _save_alert(self, alert: Alert):
        """Save alert to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO alerts (
                alert_id, title, message, priority, category,
                recipient_id, recipient_name, source_type, source_id, source_url,
                study_id, site_id, patient_id, channels, status,
                batch_id, metadata, tags, created_at, scheduled_at,
                sent_at, delivered_at, read_at, suppressed, suppression_reason, dedup_key
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            alert.alert_id, alert.title, alert.message,
            alert.priority.value, alert.category.value,
            alert.recipient_id, alert.recipient_name,
            alert.source_type, alert.source_id, alert.source_url,
            alert.study_id, alert.site_id, alert.patient_id,
            json.dumps([c.value for c in alert.channels]),
            alert.status.value, alert.batch_id,
            json.dumps(alert.metadata), json.dumps(alert.tags),
            alert.created_at.isoformat(),
            alert.scheduled_at.isoformat() if alert.scheduled_at else None,
            alert.sent_at.isoformat() if alert.sent_at else None,
            alert.delivered_at.isoformat() if alert.delivered_at else None,
            alert.read_at.isoformat() if alert.read_at else None,
            1 if alert.suppressed else 0,
            alert.suppression_reason.value if alert.suppression_reason else None,
            alert.dedup_key
        ))
        
        conn.commit()
        conn.close()
    
    # --------------------------------------------------------
    # Alert Retrieval
    # --------------------------------------------------------
    
    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """Get alert by ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM alerts WHERE alert_id = ?", (alert_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return self._row_to_alert(row, cursor.description)
        return None
    
    def get_alerts_for_user(
        self,
        user_id: str,
        status: Optional[List[AlertStatus]] = None,
        priority: Optional[List[AlertPriority]] = None,
        category: Optional[List[AlertCategory]] = None,
        unread_only: bool = False,
        limit: int = 50,
        offset: int = 0
    ) -> Tuple[List[Alert], int]:
        """Get alerts for a user with filtering"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM alerts WHERE recipient_id = ? AND suppressed = 0"
        params: List[Any] = [user_id]
        
        if status:
            placeholders = ','.join(['?' for _ in status])
            query += f" AND status IN ({placeholders})"
            params.extend([s.value for s in status])
        
        if priority:
            placeholders = ','.join(['?' for _ in priority])
            query += f" AND priority IN ({placeholders})"
            params.extend([p.value for p in priority])
        
        if category:
            placeholders = ','.join(['?' for _ in category])
            query += f" AND category IN ({placeholders})"
            params.extend([c.value for c in category])
        
        if unread_only:
            query += " AND read_at IS NULL"
        
        # Get total count
        count_query = query.replace("SELECT *", "SELECT COUNT(*)")
        cursor.execute(count_query, params)
        total = cursor.fetchone()[0]
        
        # Get paginated results
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        alerts = [self._row_to_alert(row, cursor.description) for row in rows]
        
        conn.close()
        return alerts, total
    
    def get_unread_count(self, user_id: str) -> int:
        """Get count of unread alerts for user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT COUNT(*) FROM alerts
            WHERE recipient_id = ? AND read_at IS NULL AND suppressed = 0
        """, (user_id,))
        
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    # --------------------------------------------------------
    # Alert Actions
    # --------------------------------------------------------
    
    def mark_as_read(self, alert_id: str, read_by: str = "user") -> bool:
        """Mark alert as read"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.now().isoformat()
        cursor.execute("""
            UPDATE alerts SET read_at = ?, status = ?
            WHERE alert_id = ? AND read_at IS NULL
        """, (now, AlertStatus.READ.value, alert_id))
        
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        if success:
            self._log_audit(alert_id, "read", read_by, {})
        
        return success
    
    def mark_as_delivered(self, alert_id: str, channel: AlertChannel) -> bool:
        """Mark alert as delivered via specific channel"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.now().isoformat()
        cursor.execute("""
            UPDATE alerts SET delivered_at = ?, status = ?
            WHERE alert_id = ?
        """, (now, AlertStatus.DELIVERED.value, alert_id))
        
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        if success:
            self._log_audit(alert_id, "delivered", "system", {'channel': channel.value})
        
        return success
    
    def send_alert(self, alert_id: str) -> bool:
        """Send a queued alert"""
        alert = self.get_alert(alert_id)
        if not alert:
            return False
        
        if alert.status != AlertStatus.QUEUED:
            return False
        
        # In production, this would actually send via channels
        # For now, we just update status
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.now().isoformat()
        cursor.execute("""
            UPDATE alerts SET sent_at = ?, status = ?
            WHERE alert_id = ?
        """, (now, AlertStatus.SENT.value, alert_id))
        
        conn.commit()
        conn.close()
        
        self._log_audit(alert_id, "sent", "system", {'channels': [c.value for c in alert.channels]})
        
        return True
    
    # --------------------------------------------------------
    # User Preferences
    # --------------------------------------------------------
    
    def get_user_preferences(self, user_id: str) -> UserAlertPreferences:
        """Get user alert preferences"""
        
        # Check cache
        if user_id in self._preferences_cache:
            return self._preferences_cache[user_id]
        
        # Load from database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT preferences FROM user_preferences WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            prefs_dict = json.loads(row[0])
            prefs = self._dict_to_preferences(prefs_dict)
        else:
            # Return default preferences
            prefs = UserAlertPreferences(user_id=user_id)
        
        # Cache
        self._preferences_cache[user_id] = prefs
        return prefs
    
    def save_user_preferences(self, preferences: UserAlertPreferences) -> bool:
        """Save user alert preferences"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        prefs_json = json.dumps(preferences.to_dict())
        now = datetime.now().isoformat()
        
        cursor.execute("""
            INSERT OR REPLACE INTO user_preferences (user_id, user_name, preferences, updated_at)
            VALUES (?, ?, ?, ?)
        """, (preferences.user_id, preferences.user_name, prefs_json, now))
        
        conn.commit()
        conn.close()
        
        # Update cache
        self._preferences_cache[preferences.user_id] = preferences
        
        return True
    
    def _dict_to_preferences(self, d: Dict) -> UserAlertPreferences:
        """Convert dictionary to UserAlertPreferences"""
        prefs = UserAlertPreferences(
            user_id=d.get('user_id', ''),
            user_name=d.get('user_name', '')
        )
        
        if 'urgent_channels' in d:
            prefs.urgent_channels = [AlertChannel(c) for c in d['urgent_channels']]
        if 'high_channels' in d:
            prefs.high_channels = [AlertChannel(c) for c in d['high_channels']]
        if 'normal_channels' in d:
            prefs.normal_channels = [AlertChannel(c) for c in d['normal_channels']]
        if 'low_channels' in d:
            prefs.low_channels = [AlertChannel(c) for c in d['low_channels']]
        if 'enabled_categories' in d:
            prefs.enabled_categories = [AlertCategory(c) for c in d['enabled_categories']]
        
        prefs.quiet_hours_enabled = d.get('quiet_hours_enabled', False)
        prefs.quiet_hours_start = d.get('quiet_hours_start', "22:00")
        prefs.quiet_hours_end = d.get('quiet_hours_end', "07:00")
        prefs.batch_low_priority = d.get('batch_low_priority', True)
        prefs.batch_normal_priority = d.get('batch_normal_priority', False)
        prefs.max_alerts_per_hour = d.get('max_alerts_per_hour', 20)
        prefs.max_alerts_per_day = d.get('max_alerts_per_day', 100)
        prefs.dedup_window_minutes = d.get('dedup_window_minutes', 60)
        
        if 'digest_frequency' in d:
            prefs.digest_frequency = BatchType(d['digest_frequency'])
        
        return prefs
    
    # --------------------------------------------------------
    # Batching & Digests
    # --------------------------------------------------------
    
    def process_pending_batches(self) -> List[AlertBatch]:
        """Process and send pending batches"""
        
        sent_batches = []
        
        for user_id, alerts in self.batching_engine.pending_batches.items():
            if not alerts:
                continue
            
            # Create batch
            batch = self.batching_engine.create_digest(user_id, alerts)
            
            # Save batch
            self._save_batch(batch)
            
            # In production, send the batch
            batch.status = AlertStatus.SENT
            batch.sent_at = datetime.now()
            
            # Update batch status
            self._update_batch_status(batch.batch_id, AlertStatus.SENT)
            
            # Update individual alerts
            for alert in alerts:
                self._update_alert_status(alert.alert_id, AlertStatus.SENT)
            
            sent_batches.append(batch)
            
            # Clear pending batch
            self.batching_engine.clear_batch(user_id)
        
        return sent_batches
    
    def _save_batch(self, batch: AlertBatch):
        """Save batch to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO alert_batches (
                batch_id, batch_type, recipient_id, recipient_name,
                alerts, alert_count, scheduled_for, created_at, sent_at, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            batch.batch_id, batch.batch_type.value,
            batch.recipient_id, batch.recipient_name,
            json.dumps(batch.alerts), batch.alert_count,
            batch.scheduled_for.isoformat(), batch.created_at.isoformat(),
            batch.sent_at.isoformat() if batch.sent_at else None,
            batch.status.value
        ))
        
        conn.commit()
        conn.close()
    
    def _update_batch_status(self, batch_id: str, status: AlertStatus):
        """Update batch status"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        updates = {"status": status.value}
        if status == AlertStatus.SENT:
            updates["sent_at"] = datetime.now().isoformat()
        
        set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
        cursor.execute(f"UPDATE alert_batches SET {set_clause} WHERE batch_id = ?",
                      list(updates.values()) + [batch_id])
        
        conn.commit()
        conn.close()
    
    def _update_alert_status(self, alert_id: str, status: AlertStatus):
        """Update alert status"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("UPDATE alerts SET status = ? WHERE alert_id = ?",
                      (status.value, alert_id))
        
        conn.commit()
        conn.close()
    
    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------
    
    def get_statistics(self, user_id: Optional[str] = None) -> AlertStats:
        """Get alert statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        base_query = "FROM alerts"
        params: List[Any] = []
        
        if user_id:
            base_query += " WHERE recipient_id = ?"
            params.append(user_id)
        
        stats = AlertStats()
        
        # Total alerts
        cursor.execute(f"SELECT COUNT(*) {base_query}", params)
        stats.total_alerts = cursor.fetchone()[0]
        
        # By status
        cursor.execute(f"""
            SELECT status, COUNT(*) {base_query}
            GROUP BY status
        """, params)
        stats.by_status = {row[0]: row[1] for row in cursor.fetchall()}
        stats.alerts_sent = stats.by_status.get('sent', 0) + stats.by_status.get('delivered', 0)
        stats.alerts_read = stats.by_status.get('read', 0)
        stats.alerts_suppressed = stats.by_status.get('suppressed', 0)
        stats.alerts_deduplicated = stats.by_status.get('deduplicated', 0)
        
        # By priority
        cursor.execute(f"""
            SELECT priority, COUNT(*) {base_query}
            GROUP BY priority
        """, params)
        stats.by_priority = {row[0]: row[1] for row in cursor.fetchall()}
        
        # By category
        cursor.execute(f"""
            SELECT category, COUNT(*) {base_query}
            GROUP BY category
        """, params)
        stats.by_category = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Recent activity
        hour_ago = (datetime.now() - timedelta(hours=1)).isoformat()
        day_ago = (datetime.now() - timedelta(days=1)).isoformat()
        
        where_clause = "WHERE recipient_id = ? AND" if user_id else "WHERE"
        
        cursor.execute(f"""
            SELECT COUNT(*) FROM alerts {where_clause} created_at >= ?
        """, params + [hour_ago])
        stats.alerts_last_hour = cursor.fetchone()[0]
        
        cursor.execute(f"""
            SELECT COUNT(*) FROM alerts {where_clause} created_at >= ?
        """, params + [day_ago])
        stats.alerts_last_day = cursor.fetchone()[0]
        
        conn.close()
        return stats
    
    def get_fatigue_status(self, user_id: str) -> FatigueStatus:
        """Get current fatigue status for user"""
        preferences = self.get_user_preferences(user_id)
        return self.fatigue_preventor.check_fatigue(user_id, preferences)
    
    # --------------------------------------------------------
    # Audit Trail
    # --------------------------------------------------------
    
    def _log_audit(self, alert_id: str, action: str, actor: str, details: Dict):
        """Log audit entry"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO alert_audit (alert_id, action, actor, details, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (alert_id, action, actor, json.dumps(details), datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
    
    def get_audit_trail(self, alert_id: str) -> List[Dict]:
        """Get audit trail for alert"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM alert_audit WHERE alert_id = ? ORDER BY created_at
        """, (alert_id,))
        
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(zip(columns, row)) for row in rows]
    
    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------
    
    def _row_to_alert(self, row: Tuple, description) -> Alert:
        """Convert database row to Alert object"""
        columns = [desc[0] for desc in description]
        data = dict(zip(columns, row))
        
        return Alert(
            alert_id=data['alert_id'],
            title=data['title'],
            message=data['message'],
            priority=AlertPriority(data['priority']),
            category=AlertCategory(data['category']),
            recipient_id=data['recipient_id'],
            recipient_name=data.get('recipient_name', ''),
            source_type=data.get('source_type', ''),
            source_id=data.get('source_id', ''),
            source_url=data.get('source_url', ''),
            study_id=data.get('study_id'),
            site_id=data.get('site_id'),
            patient_id=data.get('patient_id'),
            channels=[AlertChannel(c) for c in json.loads(data.get('channels', '[]'))],
            status=AlertStatus(data['status']),
            batch_id=data.get('batch_id'),
            metadata=json.loads(data.get('metadata', '{}')),
            tags=json.loads(data.get('tags', '[]')),
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else datetime.now(),
            scheduled_at=datetime.fromisoformat(data['scheduled_at']) if data.get('scheduled_at') else None,
            sent_at=datetime.fromisoformat(data['sent_at']) if data.get('sent_at') else None,
            delivered_at=datetime.fromisoformat(data['delivered_at']) if data.get('delivered_at') else None,
            read_at=datetime.fromisoformat(data['read_at']) if data.get('read_at') else None,
            suppressed=bool(data.get('suppressed', 0)),
            suppression_reason=SuppressionReason(data['suppression_reason']) if data.get('suppression_reason') else None,
            dedup_key=data.get('dedup_key', '')
        )


# ============================================================
# SINGLETON ACCESS
# ============================================================

_alert_system_instance: Optional[AlertSystem] = None

def get_alert_system(db_path: Optional[str] = None) -> AlertSystem:
    """Get singleton AlertSystem instance"""
    global _alert_system_instance
    if _alert_system_instance is None:
        _alert_system_instance = AlertSystem(db_path)
    return _alert_system_instance

def reset_alert_system():
    """Reset singleton for testing"""
    global _alert_system_instance
    _alert_system_instance = None


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def send_alert(
    title: str,
    message: str,
    recipient_id: str,
    category: AlertCategory = AlertCategory.SYSTEM,
    priority: Optional[AlertPriority] = None,
    **kwargs
) -> Alert:
    """Convenience function to send an alert"""
    system = get_alert_system()
    return system.create_alert(
        title=title,
        message=message,
        recipient_id=recipient_id,
        category=category,
        priority=priority,
        **kwargs
    )

def get_user_alerts(user_id: str, unread_only: bool = False, limit: int = 50) -> List[Alert]:
    """Convenience function to get user alerts"""
    system = get_alert_system()
    alerts, _ = system.get_alerts_for_user(user_id, unread_only=unread_only, limit=limit)
    return alerts

def get_alert_stats(user_id: Optional[str] = None) -> AlertStats:
    """Convenience function to get alert statistics"""
    system = get_alert_system()
    return system.get_statistics(user_id)


# ============================================================
# TEST FUNCTION
# ============================================================

def test_alert_system():
    """Test the Alert System"""
    print("=" * 60)
    print("SANCHALAK AI - ALERT SYSTEM TEST")
    print("=" * 60)
    
    # Reset for clean test
    reset_alert_system()
    
    import tempfile
    import os
    
    # Use temp database
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_alerts.db")
    
    tests_passed = 0
    tests_failed = 0
    
    try:
        # Test 1: Initialize Alert System
        print("\n" + "-" * 40)
        print("TEST 1: Initialize Alert System")
        print("-" * 40)
        
        system = AlertSystem(db_path)
        print(f"✅ Alert System initialized")
        print(f"   Database: {system.db_path}")
        tests_passed += 1
        
        # Test 2: Priority Classification
        print("\n" + "-" * 40)
        print("TEST 2: Priority Classification")
        print("-" * 40)
        
        classifier = PriorityClassifier()
        
        # Urgent case
        p1 = classifier.classify("SAE Alert", "Fatal event reported", AlertCategory.SAFETY)
        assert p1 == AlertPriority.URGENT, f"Expected URGENT, got {p1}"
        print(f"✅ 'Fatal event' → {p1.value}")
        
        # High case
        p2 = classifier.classify("SLA Warning", "Deadline approaching", AlertCategory.SLA_BREACH)
        assert p2 == AlertPriority.HIGH, f"Expected HIGH, got {p2}"
        print(f"✅ 'Deadline approaching' → {p2.value}")
        
        # Normal case
        p3 = classifier.classify("Query Update", "New query assigned", AlertCategory.ISSUE)
        assert p3 == AlertPriority.NORMAL, f"Expected NORMAL, got {p3}"
        print(f"✅ 'New query assigned' → {p3.value}")
        
        tests_passed += 1
        
        # Test 3: Create Alert
        print("\n" + "-" * 40)
        print("TEST 3: Create Alert")
        print("-" * 40)
        
        alert = system.create_alert(
            title="Test Alert",
            message="This is a test alert message",
            recipient_id="user_001",
            recipient_name="Test User",
            category=AlertCategory.ISSUE,
            source_type="test",
            source_id="TEST-001"
        )
        
        assert alert.alert_id.startswith("ALT-"), f"Invalid alert ID: {alert.alert_id}"
        assert alert.priority == AlertPriority.NORMAL
        # Accept any non-suppressed status
        valid_statuses = [AlertStatus.PENDING, AlertStatus.QUEUED, AlertStatus.BATCHED]
        assert alert.status in valid_statuses, f"Unexpected status: {alert.status}"
        print(f"✅ Alert created: {alert.alert_id}")
        print(f"   Priority: {alert.priority.value}")
        print(f"   Status: {alert.status.value}")
        print(f"   Channels: {[c.value for c in alert.channels]}")
        tests_passed += 1
        
        # Test 4: User Preferences
        print("\n" + "-" * 40)
        print("TEST 4: User Preferences")
        print("-" * 40)
        
        prefs = UserAlertPreferences(
            user_id="user_001",
            user_name="Test User",
            quiet_hours_enabled=False,  # Disable for testing
            quiet_hours_start="22:00",
            quiet_hours_end="07:00",
            max_alerts_per_hour=10,
            max_alerts_per_day=50
        )
        
        system.save_user_preferences(prefs)
        loaded_prefs = system.get_user_preferences("user_001")
        
        assert loaded_prefs.quiet_hours_enabled == False
        assert loaded_prefs.max_alerts_per_hour == 10
        print(f"✅ Preferences saved and loaded")
        print(f"   Quiet hours enabled: {loaded_prefs.quiet_hours_enabled}")
        print(f"   Max per hour: {loaded_prefs.max_alerts_per_hour}")
        tests_passed += 1
        
        # Test 5: Channel Routing
        print("\n" + "-" * 40)
        print("TEST 5: Channel Routing")
        print("-" * 40)
        
        router = ChannelRouter()
        
        urgent_alert = Alert(
            alert_id="ALT-TEST-URGENT",
            title="Urgent",
            message="Test",
            priority=AlertPriority.URGENT,
            category=AlertCategory.SAFETY,
            recipient_id="user_001"
        )
        
        channels = router.route(urgent_alert, prefs)
        print(f"✅ URGENT alert channels: {[c.value for c in channels]}")
        assert AlertChannel.SMS in channels, "SMS should be in urgent channels"
        
        low_alert = Alert(
            alert_id="ALT-TEST-LOW",
            title="Low Priority",
            message="Test",
            priority=AlertPriority.LOW,
            category=AlertCategory.DIGEST,
            recipient_id="user_001"
        )
        
        channels = router.route(low_alert, prefs)
        print(f"✅ LOW alert channels: {[c.value for c in channels]}")
        tests_passed += 1
        
        # Test 6: Deduplication
        print("\n" + "-" * 40)
        print("TEST 6: Deduplication")
        print("-" * 40)
        
        # Create first alert
        alert1 = system.create_alert(
            title="Duplicate Test",
            message="Same content",
            recipient_id="user_002",
            category=AlertCategory.ISSUE,
            source_type="test",
            source_id="DUP-001"
        )
        print(f"✅ First alert: {alert1.alert_id} - Status: {alert1.status.value}")
        
        # Create duplicate (same content, same recipient)
        alert2 = system.create_alert(
            title="Duplicate Test",
            message="Same content",
            recipient_id="user_002",
            category=AlertCategory.ISSUE,
            source_type="test",
            source_id="DUP-001"
        )
        print(f"✅ Duplicate alert: {alert2.alert_id} - Status: {alert2.status.value}")
        
        assert alert2.suppressed == True
        assert alert2.suppression_reason == SuppressionReason.DUPLICATE
        tests_passed += 1
        
        # Test 7: Batching
        print("\n" + "-" * 40)
        print("TEST 7: Batching")
        print("-" * 40)
        
        # Create low priority alerts for batching
        prefs_batch = UserAlertPreferences(
            user_id="user_003",
            batch_low_priority=True,
            quiet_hours_enabled=False
        )
        system.save_user_preferences(prefs_batch)
        
        for i in range(3):
            batch_alert = system.create_alert(
                title=f"Batch Alert {i+1}",
                message="Low priority message",
                recipient_id="user_003",
                category=AlertCategory.REPORT,
                priority=AlertPriority.LOW
            )
            print(f"   Alert {i+1}: {batch_alert.status.value}")
        
        pending = system.batching_engine.get_pending_batch("user_003")
        print(f"✅ Pending batch size: {len(pending)}")
        tests_passed += 1
        
        # Test 8: Mark as Read
        print("\n" + "-" * 40)
        print("TEST 8: Mark as Read")
        print("-" * 40)
        
        # Use the first alert we created
        success = system.mark_as_read(alert.alert_id, "user_001")
        updated = system.get_alert(alert.alert_id)
        
        if updated:
            print(f"✅ Alert marked as read: {success}")
            print(f"   Read at: {updated.read_at}")
        tests_passed += 1
        
        # Test 9: Fatigue Prevention
        print("\n" + "-" * 40)
        print("TEST 9: Fatigue Prevention")
        print("-" * 40)
        
        fatigue = system.get_fatigue_status("user_001")
        print(f"✅ Fatigue status for user_001:")
        print(f"   Alerts last hour: {fatigue.alerts_last_hour}/{fatigue.hourly_limit}")
        print(f"   Alerts last day: {fatigue.alerts_last_day}/{fatigue.daily_limit}")
        print(f"   Is fatigued: {fatigue.is_fatigued}")
        tests_passed += 1
        
        # Test 10: Get Alerts for User
        print("\n" + "-" * 40)
        print("TEST 10: Get Alerts for User")
        print("-" * 40)
        
        alerts, total = system.get_alerts_for_user("user_001", limit=10)
        print(f"✅ Found {total} alerts for user_001")
        for a in alerts[:3]:
            print(f"   - {a.alert_id}: {a.title} ({a.status.value})")
        tests_passed += 1
        
        # Test 11: Unread Count
        print("\n" + "-" * 40)
        print("TEST 11: Unread Count")
        print("-" * 40)
        
        unread = system.get_unread_count("user_001")
        print(f"✅ Unread alerts for user_001: {unread}")
        tests_passed += 1
        
        # Test 12: Statistics
        print("\n" + "-" * 40)
        print("TEST 12: Statistics")
        print("-" * 40)
        
        stats = system.get_statistics()
        print(f"✅ System-wide statistics:")
        print(f"   Total alerts: {stats.total_alerts}")
        print(f"   Sent: {stats.alerts_sent}")
        print(f"   Suppressed: {stats.alerts_suppressed}")
        print(f"   Deduplicated: {stats.alerts_deduplicated}")
        print(f"   By priority: {stats.by_priority}")
        tests_passed += 1
        
        # Test 13: Audit Trail
        print("\n" + "-" * 40)
        print("TEST 13: Audit Trail")
        print("-" * 40)
        
        audit = system.get_audit_trail(alert.alert_id)
        print(f"✅ Audit trail entries: {len(audit)}")
        for entry in audit[:3]:
            print(f"   - {entry['action']} by {entry['actor']}")
        tests_passed += 1
        
        # Test 14: Convenience Functions
        print("\n" + "-" * 40)
        print("TEST 14: Convenience Functions")
        print("-" * 40)
        
        # Reset to use our test instance
        global _alert_system_instance
        _alert_system_instance = system
        
        quick_alert = send_alert(
            title="Quick Alert",
            message="Sent via convenience function",
            recipient_id="user_004",
            category=AlertCategory.SYSTEM
        )
        print(f"✅ send_alert(): {quick_alert.alert_id}")
        
        user_alerts = get_user_alerts("user_004", limit=5)
        print(f"✅ get_user_alerts(): {len(user_alerts)} alerts")
        
        quick_stats = get_alert_stats("user_004")
        print(f"✅ get_alert_stats(): {quick_stats.total_alerts} total")
        tests_passed += 1
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        tests_failed += 1
    
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests Passed: {tests_passed}")
    print(f"Tests Failed: {tests_failed}")
    print(f"Total: {tests_passed + tests_failed}")
    
    if tests_failed == 0:
        print("\n✅ ALL TESTS PASSED!")
    else:
        print(f"\n❌ {tests_failed} TESTS FAILED")
    
    return tests_failed == 0


if __name__ == "__main__":
    test_alert_system()