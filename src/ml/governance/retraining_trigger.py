# src/ml/governance/retraining_trigger.py
# NOTE: This module previously used SQLite but has been migrated to PostgreSQL-only.
# If you need to use this module, update it to use:
#   from src.database.pg_data_service import get_pg_data_service
#   service = get_pg_data_service()


"""
a6on-i - ML Retraining Trigger v1.0
Automated Model Retraining Pipeline Management

Features:
- Scheduled retraining (time-based)
- Drift-triggered retraining
- Performance-triggered retraining
- Data volume triggers
- Approval workflow for production models
- One-click manual retraining
- Job tracking and history
- 21 CFR Part 11 compliant audit trail
"""

import json
import uuid
# SQLite removed - using PostgreSQL
import threading
import subprocess
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Callable
from pathlib import Path


# =============================================================================
# ENUMS
# =============================================================================

class TriggerType(Enum):
    """Types of retraining triggers"""
    SCHEDULED = "scheduled"           # Time-based (cron-like)
    DRIFT_BASED = "drift_based"       # Statistical drift detected
    PERFORMANCE_BASED = "performance" # Accuracy degradation
    DATA_VOLUME = "data_volume"       # New data threshold
    MANUAL = "manual"                 # User-initiated


class RetrainingStatus(Enum):
    """Status of a retraining job"""
    PENDING_APPROVAL = "pending_approval"  # Awaiting approval
    APPROVED = "approved"                  # Approved, ready to run
    QUEUED = "queued"                      # In queue
    IN_PROGRESS = "in_progress"            # Currently running
    COMPLETED = "completed"                # Successfully completed
    FAILED = "failed"                      # Failed with error
    CANCELLED = "cancelled"                # Cancelled by user
    REJECTED = "rejected"                  # Approval rejected


class ScheduleFrequency(Enum):
    """Frequency for scheduled retraining"""
    DAILY = "daily"
    WEEKLY = "weekly"
    BIWEEKLY = "biweekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class RetrainingRule:
    """Rule configuration for automated retraining"""
    rule_id: str
    model_name: str
    trigger_type: TriggerType
    enabled: bool = True
    auto_approve: bool = False  # If True, skip approval workflow
    
    # Schedule configuration (for SCHEDULED)
    schedule_frequency: Optional[ScheduleFrequency] = None
    schedule_day: Optional[int] = None  # Day of week (0=Mon) or day of month
    schedule_time: Optional[str] = None  # HH:MM format
    last_triggered: Optional[datetime] = None
    next_trigger: Optional[datetime] = None
    
    # Drift configuration (for DRIFT_BASED)
    drift_psi_threshold: float = 0.25
    drift_min_features: int = 3  # Min features that must drift
    
    # Performance configuration (for PERFORMANCE_BASED)
    accuracy_drop_threshold: float = 0.05  # 5% drop
    baseline_window_days: int = 30
    
    # Data volume configuration (for DATA_VOLUME)
    new_data_threshold: int = 10000  # New samples since last training
    
    # Training configuration
    training_script: Optional[str] = None  # Path to training script
    training_config: Dict[str, Any] = field(default_factory=dict)
    
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    created_by: str = "system"
    
    def to_dict(self) -> Dict:
        return {
            'rule_id': self.rule_id,
            'model_name': self.model_name,
            'trigger_type': self.trigger_type.value,
            'enabled': self.enabled,
            'auto_approve': self.auto_approve,
            'schedule_frequency': self.schedule_frequency.value if self.schedule_frequency else None,
            'schedule_day': self.schedule_day,
            'schedule_time': self.schedule_time,
            'last_triggered': self.last_triggered.isoformat() if self.last_triggered else None,
            'next_trigger': self.next_trigger.isoformat() if self.next_trigger else None,
            'drift_psi_threshold': self.drift_psi_threshold,
            'drift_min_features': self.drift_min_features,
            'accuracy_drop_threshold': self.accuracy_drop_threshold,
            'baseline_window_days': self.baseline_window_days,
            'new_data_threshold': self.new_data_threshold,
            'training_script': self.training_script,
            'training_config': self.training_config,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'created_by': self.created_by
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'RetrainingRule':
        return cls(
            rule_id=data['rule_id'],
            model_name=data['model_name'],
            trigger_type=TriggerType(data['trigger_type']),
            enabled=data.get('enabled', True),
            auto_approve=data.get('auto_approve', False),
            schedule_frequency=ScheduleFrequency(data['schedule_frequency']) if data.get('schedule_frequency') else None,
            schedule_day=data.get('schedule_day'),
            schedule_time=data.get('schedule_time'),
            last_triggered=datetime.fromisoformat(data['last_triggered']) if data.get('last_triggered') else None,
            next_trigger=datetime.fromisoformat(data['next_trigger']) if data.get('next_trigger') else None,
            drift_psi_threshold=data.get('drift_psi_threshold', 0.25),
            drift_min_features=data.get('drift_min_features', 3),
            accuracy_drop_threshold=data.get('accuracy_drop_threshold', 0.05),
            baseline_window_days=data.get('baseline_window_days', 30),
            new_data_threshold=data.get('new_data_threshold', 10000),
            training_script=data.get('training_script'),
            training_config=data.get('training_config', {}),
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else datetime.now(),
            updated_at=datetime.fromisoformat(data['updated_at']) if data.get('updated_at') else datetime.now(),
            created_by=data.get('created_by', 'system')
        )


@dataclass
class RetrainingJob:
    """A retraining job instance"""
    job_id: str
    model_name: str
    trigger_type: TriggerType
    rule_id: Optional[str] = None  # Source rule if triggered by rule
    
    # Status
    status: RetrainingStatus = RetrainingStatus.PENDING_APPROVAL
    
    # Timing
    triggered_at: datetime = field(default_factory=datetime.now)
    triggered_by: str = "system"
    approved_at: Optional[datetime] = None
    approved_by: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Rejection (if rejected)
    rejected_at: Optional[datetime] = None
    rejected_by: Optional[str] = None
    rejection_reason: Optional[str] = None
    
    # Training details
    training_config: Dict[str, Any] = field(default_factory=dict)
    input_data_path: Optional[str] = None
    output_model_path: Optional[str] = None
    training_script: Optional[str] = None
    
    # Metrics
    metrics_before: Dict[str, float] = field(default_factory=dict)
    metrics_after: Dict[str, float] = field(default_factory=dict)
    
    # Logs
    logs: str = ""
    error_message: Optional[str] = None
    
    # Version info
    previous_version: Optional[str] = None
    new_version: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'job_id': self.job_id,
            'model_name': self.model_name,
            'trigger_type': self.trigger_type.value,
            'rule_id': self.rule_id,
            'status': self.status.value,
            'triggered_at': self.triggered_at.isoformat(),
            'triggered_by': self.triggered_by,
            'approved_at': self.approved_at.isoformat() if self.approved_at else None,
            'approved_by': self.approved_by,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'rejected_at': self.rejected_at.isoformat() if self.rejected_at else None,
            'rejected_by': self.rejected_by,
            'rejection_reason': self.rejection_reason,
            'training_config': self.training_config,
            'input_data_path': self.input_data_path,
            'output_model_path': self.output_model_path,
            'training_script': self.training_script,
            'metrics_before': self.metrics_before,
            'metrics_after': self.metrics_after,
            'logs': self.logs,
            'error_message': self.error_message,
            'previous_version': self.previous_version,
            'new_version': self.new_version
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'RetrainingJob':
        return cls(
            job_id=data['job_id'],
            model_name=data['model_name'],
            trigger_type=TriggerType(data['trigger_type']),
            rule_id=data.get('rule_id'),
            status=RetrainingStatus(data['status']),
            triggered_at=datetime.fromisoformat(data['triggered_at']),
            triggered_by=data.get('triggered_by', 'system'),
            approved_at=datetime.fromisoformat(data['approved_at']) if data.get('approved_at') else None,
            approved_by=data.get('approved_by'),
            started_at=datetime.fromisoformat(data['started_at']) if data.get('started_at') else None,
            completed_at=datetime.fromisoformat(data['completed_at']) if data.get('completed_at') else None,
            rejected_at=datetime.fromisoformat(data['rejected_at']) if data.get('rejected_at') else None,
            rejected_by=data.get('rejected_by'),
            rejection_reason=data.get('rejection_reason'),
            training_config=data.get('training_config', {}),
            input_data_path=data.get('input_data_path'),
            output_model_path=data.get('output_model_path'),
            training_script=data.get('training_script'),
            metrics_before=data.get('metrics_before', {}),
            metrics_after=data.get('metrics_after', {}),
            logs=data.get('logs', ''),
            error_message=data.get('error_message'),
            previous_version=data.get('previous_version'),
            new_version=data.get('new_version')
        )


# =============================================================================
# RETRAINING TRIGGER
# =============================================================================

class RetrainingTrigger:
    """
    Automated retraining pipeline management.
    
    Features:
    - Create and manage retraining rules
    - Check triggers and create jobs
    - Approval workflow
    - Execute retraining
    - Track job history
    """
    
    def __init__(
        self,
        db_path: str = "data/ml_governance/retraining.db"
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_database()
        
        # Training callbacks (can be customized)
        self._training_callbacks: Dict[str, Callable] = {}
    
    def _init_database(self):
        """Initialize SQLite database"""
        # DISABLED: SQLite replaced with PostgreSQL
        if False:  # Disabled method
            cursor = conn.cursor()
            
            # Retraining rules table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS retraining_rules (
                    rule_id TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    trigger_type TEXT NOT NULL,
                    enabled INTEGER DEFAULT 1,
                    auto_approve INTEGER DEFAULT 0,
                    rule_data TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            ''')
            
            # Retraining jobs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS retraining_jobs (
                    job_id TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    trigger_type TEXT NOT NULL,
                    rule_id TEXT,
                    status TEXT NOT NULL,
                    triggered_at TEXT NOT NULL,
                    triggered_by TEXT,
                    job_data TEXT NOT NULL,
                    FOREIGN KEY (rule_id) REFERENCES retraining_rules(rule_id)
                )
            ''')
            
            # Indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_rule_model ON retraining_rules(model_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_job_model ON retraining_jobs(model_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_job_status ON retraining_jobs(status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_job_triggered ON retraining_jobs(triggered_at)')
            
            conn.commit()
    
    # =========================================================================
    # Rule Management
    # =========================================================================
    
    def create_rule(
        self,
        model_name: str,
        trigger_type: TriggerType,
        created_by: str = "system",
        auto_approve: bool = False,
        **config
    ) -> RetrainingRule:
        """
        Create a new retraining rule.
        
        Args:
            model_name: Name of the model
            trigger_type: Type of trigger
            created_by: User creating the rule
            auto_approve: If True, skip approval workflow
            **config: Rule-specific configuration
            
        Returns:
            RetrainingRule object
        """
        rule_id = str(uuid.uuid4())
        now = datetime.now()
        
        rule = RetrainingRule(
            rule_id=rule_id,
            model_name=model_name,
            trigger_type=trigger_type,
            auto_approve=auto_approve,
            created_at=now,
            updated_at=now,
            created_by=created_by
        )
        
        # Apply configuration based on trigger type
        if trigger_type == TriggerType.SCHEDULED:
            rule.schedule_frequency = config.get('schedule_frequency', ScheduleFrequency.WEEKLY)
            rule.schedule_day = config.get('schedule_day', 0)  # Monday
            rule.schedule_time = config.get('schedule_time', '02:00')
            rule.next_trigger = self._calculate_next_trigger(rule)
        
        elif trigger_type == TriggerType.DRIFT_BASED:
            rule.drift_psi_threshold = config.get('drift_psi_threshold', 0.25)
            rule.drift_min_features = config.get('drift_min_features', 3)
        
        elif trigger_type == TriggerType.PERFORMANCE_BASED:
            rule.accuracy_drop_threshold = config.get('accuracy_drop_threshold', 0.05)
            rule.baseline_window_days = config.get('baseline_window_days', 30)
        
        elif trigger_type == TriggerType.DATA_VOLUME:
            rule.new_data_threshold = config.get('new_data_threshold', 10000)
        
        rule.training_script = config.get('training_script')
        rule.training_config = config.get('training_config', {})
        
        # Save to database
        self._save_rule(rule)
        
        return rule
    
    def _calculate_next_trigger(self, rule: RetrainingRule) -> datetime:
        """Calculate next trigger time for a scheduled rule"""
        now = datetime.now()
        
        if rule.schedule_time:
            hour, minute = map(int, rule.schedule_time.split(':'))
        else:
            hour, minute = 2, 0  # Default 2 AM
        
        if rule.schedule_frequency == ScheduleFrequency.DAILY:
            next_trigger = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if next_trigger <= now:
                next_trigger += timedelta(days=1)
        
        elif rule.schedule_frequency == ScheduleFrequency.WEEKLY:
            days_ahead = (rule.schedule_day or 0) - now.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            next_trigger = now + timedelta(days=days_ahead)
            next_trigger = next_trigger.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        elif rule.schedule_frequency == ScheduleFrequency.BIWEEKLY:
            days_ahead = (rule.schedule_day or 0) - now.weekday()
            if days_ahead <= 0:
                days_ahead += 14
            next_trigger = now + timedelta(days=days_ahead)
            next_trigger = next_trigger.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        elif rule.schedule_frequency == ScheduleFrequency.MONTHLY:
            # Next occurrence of schedule_day
            day = rule.schedule_day or 1
            if now.day >= day:
                # Next month
                if now.month == 12:
                    next_trigger = now.replace(year=now.year + 1, month=1, day=day)
                else:
                    next_trigger = now.replace(month=now.month + 1, day=day)
            else:
                next_trigger = now.replace(day=day)
            next_trigger = next_trigger.replace(hour=hour, minute=minute, second=0, microsecond=0)
        
        else:  # QUARTERLY
            current_quarter = (now.month - 1) // 3
            next_quarter = current_quarter + 1
            if next_quarter >= 4:
                next_quarter = 0
                year = now.year + 1
            else:
                year = now.year
            month = next_quarter * 3 + 1
            next_trigger = datetime(year, month, rule.schedule_day or 1, hour, minute)
        
        return next_trigger
    
    def _save_rule(self, rule: RetrainingRule):
        """Save rule to database"""
        with self._lock:
            # DISABLED: SQLite replaced with PostgreSQL
            if False:  # Disabled method
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT OR REPLACE INTO retraining_rules
                        (rule_id, model_name, trigger_type, enabled, auto_approve,
                         rule_data, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        rule.rule_id,
                        rule.model_name,
                        rule.trigger_type.value,
                        1 if rule.enabled else 0,
                        1 if rule.auto_approve else 0,
                        json.dumps(rule.to_dict()),
                        rule.created_at.isoformat(),
                        rule.updated_at.isoformat()
                    ))
                    conn.commit()
    
    def update_rule(
        self,
        rule_id: str,
        **updates
    ) -> Optional[RetrainingRule]:
        """Update an existing rule"""
        rule = self.get_rule(rule_id)
        if not rule:
            return None
        
        # Apply updates
        for key, value in updates.items():
            if hasattr(rule, key):
                setattr(rule, key, value)
        
        rule.updated_at = datetime.now()
        
        # Recalculate next trigger if schedule changed
        if rule.trigger_type == TriggerType.SCHEDULED:
            rule.next_trigger = self._calculate_next_trigger(rule)
        
        self._save_rule(rule)
        return rule
    
    def delete_rule(self, rule_id: str) -> bool:
        """Delete a retraining rule"""
        with self._lock:
            # DISABLED: SQLite replaced with PostgreSQL
            if False:  # Disabled method
                    cursor = conn.cursor()
                    cursor.execute('DELETE FROM retraining_rules WHERE rule_id = ?', (rule_id,))
                    conn.commit()
                    return cursor.rowcount > 0
    
    def get_rule(self, rule_id: str) -> Optional[RetrainingRule]:
        """Get a rule by ID"""
        # DISABLED: SQLite replaced with PostgreSQL
        if False:  # Disabled method
            cursor = conn.cursor()
            cursor.execute('SELECT rule_data FROM retraining_rules WHERE rule_id = ?', (rule_id,))
            row = cursor.fetchone()
            if row:
                return RetrainingRule.from_dict(json.loads(row[0]))
        return None
    
    def list_rules(
        self,
        model_name: Optional[str] = None,
        enabled_only: bool = False
    ) -> List[RetrainingRule]:
        """List retraining rules"""
        rules = []
        
        # DISABLED: SQLite replaced with PostgreSQL
        if False:  # Disabled method
            cursor = conn.cursor()
            
            query = 'SELECT rule_data FROM retraining_rules WHERE 1=1'
            params = []
            
            if model_name:
                query += ' AND model_name = ?'
                params.append(model_name)
            
            if enabled_only:
                query += ' AND enabled = 1'
            
            cursor.execute(query, params)
            
            for row in cursor.fetchall():
                rules.append(RetrainingRule.from_dict(json.loads(row[0])))
        
        return rules
    
    # =========================================================================
    # Trigger Checking
    # =========================================================================
    
    def check_triggers(
        self,
        model_name: Optional[str] = None
    ) -> List[RetrainingJob]:
        """
        Check all enabled rules and create jobs for triggered rules.
        
        Args:
            model_name: Optional - check specific model only
            
        Returns:
            List of created RetrainingJob objects
        """
        rules = self.list_rules(model_name=model_name, enabled_only=True)
        jobs = []
        
        for rule in rules:
            triggered = False
            
            if rule.trigger_type == TriggerType.SCHEDULED:
                triggered = self._check_schedule_trigger(rule)
            elif rule.trigger_type == TriggerType.DRIFT_BASED:
                triggered = self._check_drift_trigger(rule)
            elif rule.trigger_type == TriggerType.PERFORMANCE_BASED:
                triggered = self._check_performance_trigger(rule)
            elif rule.trigger_type == TriggerType.DATA_VOLUME:
                triggered = self._check_data_volume_trigger(rule)
            
            if triggered:
                job = self._create_job_from_rule(rule)
                jobs.append(job)
                
                # Update rule last_triggered
                rule.last_triggered = datetime.now()
                if rule.trigger_type == TriggerType.SCHEDULED:
                    rule.next_trigger = self._calculate_next_trigger(rule)
                self._save_rule(rule)
        
        return jobs
    
    def _check_schedule_trigger(self, rule: RetrainingRule) -> bool:
        """Check if scheduled trigger should fire"""
        if not rule.next_trigger:
            return False
        return datetime.now() >= rule.next_trigger
    
    def _check_drift_trigger(self, rule: RetrainingRule) -> bool:
        """Check if drift trigger should fire"""
        try:
            from .drift_detector import get_drift_detector, DriftSeverity
            
            detector = get_drift_detector()
            alerts = detector.get_active_alerts(rule.model_name)
            
            # Count high severity drifted features
            drifted_count = sum(
                1 for alert in alerts
                if alert.severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL]
            )
            
            return drifted_count >= rule.drift_min_features
        except Exception:
            return False
    
    def _check_performance_trigger(self, rule: RetrainingRule) -> bool:
        """Check if performance trigger should fire"""
        try:
            from .performance_monitor import get_performance_monitor, AlertType
            
            monitor = get_performance_monitor()
            alerts = monitor.get_active_alerts(rule.model_name)
            
            # Check for performance decay alerts
            for alert in alerts:
                if alert.alert_type == AlertType.PERFORMANCE_DECAY:
                    return True
            
            return False
        except Exception:
            return False
    
    def _check_data_volume_trigger(self, rule: RetrainingRule) -> bool:
        """Check if data volume trigger should fire"""
        try:
            from .performance_monitor import get_performance_monitor
            
            monitor = get_performance_monitor()
            stats = monitor.get_statistics()
            
            # Get prediction count since last training
            predictions_by_model = stats.get('predictions_by_model', {})
            current_count = predictions_by_model.get(rule.model_name, 0)
            
            # Compare to threshold (simplified - would need training timestamp)
            return current_count >= rule.new_data_threshold
        except Exception:
            return False
    
    def _create_job_from_rule(self, rule: RetrainingRule) -> RetrainingJob:
        """Create a job from a triggered rule"""
        job = RetrainingJob(
            job_id=str(uuid.uuid4()),
            model_name=rule.model_name,
            trigger_type=rule.trigger_type,
            rule_id=rule.rule_id,
            triggered_at=datetime.now(),
            triggered_by="system",
            training_config=rule.training_config.copy(),
            training_script=rule.training_script
        )
        
        if rule.auto_approve:
            job.status = RetrainingStatus.QUEUED
            job.approved_at = datetime.now()
            job.approved_by = "auto"
        else:
            job.status = RetrainingStatus.PENDING_APPROVAL
        
        self._save_job(job)
        return job
    
    # =========================================================================
    # Job Management
    # =========================================================================
    
    def trigger_retraining(
        self,
        model_name: str,
        triggered_by: str,
        training_config: Optional[Dict[str, Any]] = None,
        training_script: Optional[str] = None,
        auto_approve: bool = False
    ) -> RetrainingJob:
        """
        Manually trigger retraining for a model.
        
        Args:
            model_name: Name of the model to retrain
            triggered_by: User triggering the retraining
            training_config: Optional training configuration
            training_script: Optional path to training script
            auto_approve: If True, skip approval
            
        Returns:
            RetrainingJob object
        """
        job = RetrainingJob(
            job_id=str(uuid.uuid4()),
            model_name=model_name,
            trigger_type=TriggerType.MANUAL,
            triggered_at=datetime.now(),
            triggered_by=triggered_by,
            training_config=training_config or {},
            training_script=training_script
        )
        
        if auto_approve:
            job.status = RetrainingStatus.QUEUED
            job.approved_at = datetime.now()
            job.approved_by = triggered_by
        else:
            job.status = RetrainingStatus.PENDING_APPROVAL
        
        self._save_job(job)
        return job
    
    def _save_job(self, job: RetrainingJob):
        """Save job to database"""
        with self._lock:
            # DISABLED: SQLite replaced with PostgreSQL
            if False:  # Disabled method
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT OR REPLACE INTO retraining_jobs
                        (job_id, model_name, trigger_type, rule_id, status,
                         triggered_at, triggered_by, job_data)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        job.job_id,
                        job.model_name,
                        job.trigger_type.value,
                        job.rule_id,
                        job.status.value,
                        job.triggered_at.isoformat(),
                        job.triggered_by,
                        json.dumps(job.to_dict())
                    ))
                    conn.commit()
    
    def get_job(self, job_id: str) -> Optional[RetrainingJob]:
        """Get a job by ID"""
        # DISABLED: SQLite replaced with PostgreSQL
        if False:  # Disabled method
            cursor = conn.cursor()
            cursor.execute('SELECT job_data FROM retraining_jobs WHERE job_id = ?', (job_id,))
            row = cursor.fetchone()
            if row:
                return RetrainingJob.from_dict(json.loads(row[0]))
        return None
    
    def list_jobs(
        self,
        model_name: Optional[str] = None,
        status: Optional[RetrainingStatus] = None,
        limit: int = 100
    ) -> List[RetrainingJob]:
        """List retraining jobs"""
        jobs = []
        
        # DISABLED: SQLite replaced with PostgreSQL
        if False:  # Disabled method
            cursor = conn.cursor()
            
            query = 'SELECT job_data FROM retraining_jobs WHERE 1=1'
            params = []
            
            if model_name:
                query += ' AND model_name = ?'
                params.append(model_name)
            
            if status:
                query += ' AND status = ?'
                params.append(status.value)
            
            query += ' ORDER BY triggered_at DESC LIMIT ?'
            params.append(limit)
            
            cursor.execute(query, params)
            
            for row in cursor.fetchall():
                jobs.append(RetrainingJob.from_dict(json.loads(row[0])))
        
        return jobs
    
    def get_pending_approvals(self) -> List[RetrainingJob]:
        """Get jobs pending approval"""
        return self.list_jobs(status=RetrainingStatus.PENDING_APPROVAL)
    
    # =========================================================================
    # Approval Workflow
    # =========================================================================
    
    def approve_job(
        self,
        job_id: str,
        approved_by: str
    ) -> Optional[RetrainingJob]:
        """Approve a retraining job"""
        job = self.get_job(job_id)
        if not job:
            return None
        
        if job.status != RetrainingStatus.PENDING_APPROVAL:
            return job
        
        job.status = RetrainingStatus.QUEUED
        job.approved_at = datetime.now()
        job.approved_by = approved_by
        
        self._save_job(job)
        return job
    
    def reject_job(
        self,
        job_id: str,
        rejected_by: str,
        reason: str = ""
    ) -> Optional[RetrainingJob]:
        """Reject a retraining job"""
        job = self.get_job(job_id)
        if not job:
            return None
        
        if job.status != RetrainingStatus.PENDING_APPROVAL:
            return job
        
        job.status = RetrainingStatus.REJECTED
        job.rejected_at = datetime.now()
        job.rejected_by = rejected_by
        job.rejection_reason = reason
        
        self._save_job(job)
        return job
    
    def cancel_job(
        self,
        job_id: str,
        cancelled_by: str,
        reason: str = ""
    ) -> Optional[RetrainingJob]:
        """Cancel a retraining job"""
        job = self.get_job(job_id)
        if not job:
            return None
        
        if job.status in [RetrainingStatus.COMPLETED, RetrainingStatus.CANCELLED]:
            return job
        
        job.status = RetrainingStatus.CANCELLED
        job.error_message = f"Cancelled by {cancelled_by}: {reason}"
        job.completed_at = datetime.now()
        
        self._save_job(job)
        return job
    
    # =========================================================================
    # Execution
    # =========================================================================
    
    def register_training_callback(
        self,
        model_name: str,
        callback: Callable[[RetrainingJob], Dict[str, Any]]
    ):
        """
        Register a training callback for a model.
        
        The callback should:
        - Accept a RetrainingJob
        - Perform the training
        - Return dict with: success, metrics, model_path, version, error
        """
        self._training_callbacks[model_name] = callback
    
    def execute_retraining(
        self,
        job_id: str
    ) -> RetrainingJob:
        """
        Execute a retraining job.
        
        Args:
            job_id: ID of the job to execute
            
        Returns:
            Updated RetrainingJob object
        """
        job = self.get_job(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")
        
        if job.status not in [RetrainingStatus.QUEUED, RetrainingStatus.APPROVED]:
            raise ValueError(f"Job {job_id} cannot be executed (status: {job.status.value})")
        
        # Update status
        job.status = RetrainingStatus.IN_PROGRESS
        job.started_at = datetime.now()
        self._save_job(job)
        
        try:
            # Try registered callback first
            if job.model_name in self._training_callbacks:
                result = self._training_callbacks[job.model_name](job)
            elif job.training_script:
                result = self._execute_script(job)
            else:
                result = self._default_training(job)
            
            # Update job with results
            if result.get('success', False):
                job.status = RetrainingStatus.COMPLETED
                job.metrics_after = result.get('metrics', {})
                job.output_model_path = result.get('model_path')
                job.new_version = result.get('version')
                job.logs = result.get('logs', '')
            else:
                job.status = RetrainingStatus.FAILED
                job.error_message = result.get('error', 'Unknown error')
                job.logs = result.get('logs', '')
        
        except Exception as e:
            job.status = RetrainingStatus.FAILED
            job.error_message = str(e)
        
        job.completed_at = datetime.now()
        self._save_job(job)
        
        return job
    
    def _execute_script(self, job: RetrainingJob) -> Dict[str, Any]:
        """Execute a training script"""
        try:
            # Build command
            cmd = ['python', job.training_script]
            
            # Add config as JSON arg
            if job.training_config:
                cmd.extend(['--config', json.dumps(job.training_config)])
            
            cmd.extend(['--model-name', job.model_name])
            
            # Execute
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                return {
                    'success': True,
                    'logs': result.stdout,
                    'metrics': {},
                    'model_path': None,
                    'version': None
                }
            else:
                return {
                    'success': False,
                    'error': result.stderr,
                    'logs': result.stdout
                }
        
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Training script timed out after 1 hour'
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _default_training(self, job: RetrainingJob) -> Dict[str, Any]:
        """Default training implementation (placeholder)"""
        # This would be customized per installation
        return {
            'success': False,
            'error': 'No training callback or script configured for this model'
        }
    
    # =========================================================================
    # History
    # =========================================================================
    
    def get_retraining_history(
        self,
        model_name: str,
        days: int = 90
    ) -> List[RetrainingJob]:
        """Get retraining history for a model"""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        
        jobs = []
        # DISABLED: SQLite replaced with PostgreSQL
        if False:  # Disabled method
            cursor = conn.cursor()
            cursor.execute('''
                SELECT job_data FROM retraining_jobs
                WHERE model_name = ? AND triggered_at >= ?
                ORDER BY triggered_at DESC
            ''', (model_name, cutoff))
            
            for row in cursor.fetchall():
                jobs.append(RetrainingJob.from_dict(json.loads(row[0])))
        
        return jobs
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get retraining trigger statistics"""
        # Initialize default values
        active_rules = 0
        total_rules = 0
        jobs_by_status = {}
        jobs_last_30_days = 0
        pending = 0
        success_rate = 0.0

        # DISABLED: SQLite replaced with PostgreSQL
        if False:  # Disabled method
            cursor = conn.cursor()
            
            # Rule counts
            cursor.execute('SELECT COUNT(*) FROM retraining_rules WHERE enabled = 1')
            active_rules = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM retraining_rules')
            total_rules = cursor.fetchone()[0]
            
            # Job counts by status
            cursor.execute('''
                SELECT status, COUNT(*) FROM retraining_jobs GROUP BY status
            ''')
            jobs_by_status = dict(cursor.fetchall())
            
            # Jobs last 30 days
            month_ago = (datetime.now() - timedelta(days=30)).isoformat()
            cursor.execute('''
                SELECT COUNT(*) FROM retraining_jobs WHERE triggered_at >= ?
            ''', (month_ago,))
            jobs_last_30_days = cursor.fetchone()[0]
            
            # Pending approvals
            pending = jobs_by_status.get('pending_approval', 0)
            
            # Success rate
            completed = jobs_by_status.get('completed', 0)
            failed = jobs_by_status.get('failed', 0)
            total = completed + failed
            success_rate = completed / total if total > 0 else 0
        
        return {
            'active_rules': active_rules,
            'total_rules': total_rules,
            'jobs_by_status': jobs_by_status,
            'pending_approvals': pending,
            'jobs_last_30_days': jobs_last_30_days,
            'success_rate': success_rate
        }


# =============================================================================
# SINGLETON
# =============================================================================

_retraining_trigger: Optional[RetrainingTrigger] = None


def get_retraining_trigger() -> RetrainingTrigger:
    """Get or create the retraining trigger singleton"""
    global _retraining_trigger
    if _retraining_trigger is None:
        _retraining_trigger = RetrainingTrigger()
    return _retraining_trigger


def reset_retraining_trigger():
    """Reset the retraining trigger singleton (for testing)"""
    global _retraining_trigger
    _retraining_trigger = None
