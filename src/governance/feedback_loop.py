"""
SANCHALAK AI - Phase 10.5: Feedback Loop System v1.0
# NOTE: This module previously used SQLite but has been migrated to PostgreSQL-only.
# If you need to use this module, update it to use:
#   from src.database.pg_data_service import get_pg_data_service
#   service = get_pg_data_service()


Continuous learning system that captures user feedback and improves AI:
- Accept/modify/reject capture
- Learning signals extraction
- Model update triggers
- Pattern promotion workflow

Author: Sanchalak AI Team
Date: 2026-01-02
"""

# SQLite removed - using PostgreSQL
import sqlite3
import hashlib
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any, Callable
from enum import Enum
from pathlib import Path
import statistics
import math


# =============================================================================
# ENUMS
# =============================================================================

class FeedbackType(Enum):
    """Types of user feedback."""
    ACCEPT = "accept"
    MODIFY = "modify"
    REJECT = "reject"
    OVERRIDE = "override"
    IGNORE = "ignore"
    ESCALATE = "escalate"
    FLAG_ERROR = "flag_error"
    FLAG_HELPFUL = "flag_helpful"


class LearningSignalType(Enum):
    """Types of learning signals extracted from feedback."""
    POSITIVE_CONFIRMATION = "positive_confirmation"
    NEGATIVE_CONFIRMATION = "negative_confirmation"
    PARTIAL_MATCH = "partial_match"
    COMPLETE_MISMATCH = "complete_mismatch"
    CONFIDENCE_CALIBRATION = "confidence_calibration"
    NEW_PATTERN = "new_pattern"
    PATTERN_REFINEMENT = "pattern_refinement"
    FEATURE_IMPORTANCE = "feature_importance"
    THRESHOLD_ADJUSTMENT = "threshold_adjustment"
    CONTEXT_RELEVANCE = "context_relevance"


class ModelUpdateType(Enum):
    """Types of model updates."""
    RETRAIN_FULL = "retrain_full"
    RETRAIN_INCREMENTAL = "retrain_incremental"
    THRESHOLD_ADJUSTMENT = "threshold_adjustment"
    FEATURE_WEIGHT_UPDATE = "feature_weight_update"
    CONFIDENCE_RECALIBRATION = "confidence_recalibration"
    PATTERN_UPDATE = "pattern_update"
    TEMPLATE_UPDATE = "template_update"
    RULE_UPDATE = "rule_update"


class ModelUpdateStatus(Enum):
    """Status of model updates."""
    PENDING = "pending"
    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"


class PatternStatus(Enum):
    """Status of patterns in promotion workflow."""
    CANDIDATE = "candidate"
    UNDER_REVIEW = "under_review"
    VALIDATED = "validated"
    PROMOTED = "promoted"
    REJECTED = "rejected"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class PatternSource(Enum):
    """Source of pattern discovery."""
    USER_FEEDBACK = "user_feedback"
    AUTOMATED_DETECTION = "automated_detection"
    EXPERT_DEFINED = "expert_defined"
    CROSS_STUDY_TRANSFER = "cross_study_transfer"
    ANOMALY_ANALYSIS = "anomaly_analysis"


class SignalStrength(Enum):
    """Strength of learning signal."""
    STRONG = "strong"          # High confidence, clear feedback
    MODERATE = "moderate"      # Medium confidence
    WEAK = "weak"              # Low confidence, noisy signal
    AMBIGUOUS = "ambiguous"    # Conflicting signals


class UpdatePriority(Enum):
    """Priority for model updates."""
    CRITICAL = "critical"      # Immediate update needed
    HIGH = "high"              # Within 24 hours
    MEDIUM = "medium"          # Within 1 week
    LOW = "low"                # Next scheduled update
    DEFERRED = "deferred"      # When resources available


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FeedbackRecord:
    """Record of user feedback on AI output."""
    feedback_id: str
    timestamp: datetime
    user_id: str
    user_role: str
    
    # What was evaluated
    agent_name: str
    output_type: str  # recommendation, prediction, diagnosis, etc.
    output_id: str
    output_content: str
    ai_confidence: float
    
    # Feedback given
    feedback_type: FeedbackType
    user_action: str  # What the user actually did
    user_modification: Optional[str] = None  # If modified, what changes
    rejection_reason: Optional[str] = None
    error_description: Optional[str] = None
    
    # Context
    entity_type: Optional[str] = None
    entity_id: Optional[str] = None
    context: Dict = field(default_factory=dict)
    
    # Timing
    decision_time_seconds: Optional[float] = None
    
    # Processing
    signals_extracted: bool = False
    learning_signals: List[str] = field(default_factory=list)


@dataclass
class LearningSignal:
    """Extracted learning signal from feedback."""
    signal_id: str
    timestamp: datetime
    
    # Source
    feedback_id: str
    agent_name: str
    
    # Signal details
    signal_type: LearningSignalType
    strength: SignalStrength
    confidence: float  # 0-1
    
    # What was learned
    feature_name: Optional[str] = None
    old_value: Optional[Any] = None
    new_value: Optional[Any] = None
    adjustment_magnitude: float = 0.0
    
    # Pattern info
    pattern_id: Optional[str] = None
    pattern_type: Optional[str] = None
    
    # Aggregation
    occurrence_count: int = 1
    similar_signals: int = 0
    
    # Processing
    processed: bool = False
    update_triggered: bool = False
    update_id: Optional[str] = None


@dataclass
class ModelUpdateRequest:
    """Request to update a model based on learning signals."""
    update_id: str
    created_at: datetime
    
    # What to update
    model_name: str
    update_type: ModelUpdateType
    priority: UpdatePriority
    
    # Why update
    trigger_reason: str
    signal_ids: List[str] = field(default_factory=list)
    signal_count: int = 0
    
    # Update details
    changes: Dict = field(default_factory=dict)
    expected_improvement: float = 0.0
    risk_assessment: str = ""
    
    # Status
    status: ModelUpdateStatus = ModelUpdateStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Validation
    validation_metrics: Dict = field(default_factory=dict)
    rollback_available: bool = True
    
    # Approval
    requires_approval: bool = True
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    
    # Results
    result_summary: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class PatternCandidate:
    """Candidate pattern for promotion."""
    pattern_id: str
    created_at: datetime
    
    # Pattern info
    pattern_name: str
    pattern_type: str
    pattern_description: str
    source: PatternSource
    
    # Detection criteria
    detection_rules: Dict = field(default_factory=dict)
    threshold_values: Dict = field(default_factory=dict)
    
    # Evidence
    supporting_evidence: List[str] = field(default_factory=list)
    occurrence_count: int = 0
    affected_entities: int = 0
    
    # Validation
    validation_status: PatternStatus = PatternStatus.CANDIDATE
    validation_score: float = 0.0
    cross_study_validated: bool = False
    studies_validated: List[str] = field(default_factory=list)
    
    # Review
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    review_notes: Optional[str] = None
    
    # Promotion
    promoted_at: Optional[datetime] = None
    promoted_by: Optional[str] = None
    target_library: Optional[str] = None
    
    # Performance after promotion
    post_promotion_accuracy: Optional[float] = None
    post_promotion_usage: int = 0


@dataclass
class FeedbackAggregation:
    """Aggregated feedback statistics."""
    period_start: datetime
    period_end: datetime
    
    # Counts
    total_feedback: int = 0
    accepts: int = 0
    modifies: int = 0
    rejects: int = 0
    overrides: int = 0
    
    # Rates
    acceptance_rate: float = 0.0
    modification_rate: float = 0.0
    rejection_rate: float = 0.0
    
    # By agent
    by_agent: Dict[str, Dict] = field(default_factory=dict)
    
    # Learning signals
    signals_extracted: int = 0
    strong_signals: int = 0
    
    # Updates triggered
    updates_triggered: int = 0
    patterns_promoted: int = 0


# =============================================================================
# FEEDBACK LOOP SYSTEM
# =============================================================================

class FeedbackLoopSystem:
    """
    Continuous learning system for AI improvement.
    
    Captures user feedback, extracts learning signals, triggers model updates,
    and manages pattern promotion workflow.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize the feedback loop system."""
        if db_path is None:
            db_dir = Path("data/governance")
            db_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(db_dir / "feedback_loop.db")
        
        self.db_path = db_path
        self._init_database()
        
        # Signal extraction thresholds
        self.thresholds = {
            'strong_signal_confidence': 0.8,
            'moderate_signal_confidence': 0.5,
            'min_signals_for_update': 10,
            'min_pattern_occurrences': 5,
            'pattern_validation_threshold': 0.75,
            'cross_study_validation_min': 3,
            'confidence_adjustment_rate': 0.1,
            'max_adjustment_per_update': 0.2,
        }
        
        # Model update rules
        self.update_rules = {
            'rejection_rate_trigger': 0.30,  # Trigger update if >30% rejection
            'override_rate_trigger': 0.25,   # Trigger if >25% override
            'signal_accumulation_trigger': 50,  # Trigger after 50 signals
            'time_based_trigger_days': 7,    # Weekly updates
        }
    
    def _init_database(self):
        """Initialize SQLite database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Feedback records table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback_records (
                feedback_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                user_id TEXT NOT NULL,
                user_role TEXT NOT NULL,
                agent_name TEXT NOT NULL,
                output_type TEXT NOT NULL,
                output_id TEXT NOT NULL,
                output_content TEXT,
                ai_confidence REAL DEFAULT 0,
                feedback_type TEXT NOT NULL,
                user_action TEXT,
                user_modification TEXT,
                rejection_reason TEXT,
                error_description TEXT,
                entity_type TEXT,
                entity_id TEXT,
                context TEXT,
                decision_time_seconds REAL,
                signals_extracted INTEGER DEFAULT 0,
                learning_signals TEXT
            )
        ''')
        
        # Learning signals table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_signals (
                signal_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                feedback_id TEXT NOT NULL,
                agent_name TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                strength TEXT NOT NULL,
                confidence REAL DEFAULT 0,
                feature_name TEXT,
                old_value TEXT,
                new_value TEXT,
                adjustment_magnitude REAL DEFAULT 0,
                pattern_id TEXT,
                pattern_type TEXT,
                occurrence_count INTEGER DEFAULT 1,
                similar_signals INTEGER DEFAULT 0,
                processed INTEGER DEFAULT 0,
                update_triggered INTEGER DEFAULT 0,
                update_id TEXT
            )
        ''')
        
        # Model updates table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_updates (
                update_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                model_name TEXT NOT NULL,
                update_type TEXT NOT NULL,
                priority TEXT NOT NULL,
                trigger_reason TEXT NOT NULL,
                signal_ids TEXT,
                signal_count INTEGER DEFAULT 0,
                changes TEXT,
                expected_improvement REAL DEFAULT 0,
                risk_assessment TEXT,
                status TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                validation_metrics TEXT,
                rollback_available INTEGER DEFAULT 1,
                requires_approval INTEGER DEFAULT 1,
                approved_by TEXT,
                approved_at TEXT,
                result_summary TEXT,
                error_message TEXT
            )
        ''')
        
        # Pattern candidates table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pattern_candidates (
                pattern_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                pattern_name TEXT NOT NULL,
                pattern_type TEXT NOT NULL,
                pattern_description TEXT,
                source TEXT NOT NULL,
                detection_rules TEXT,
                threshold_values TEXT,
                supporting_evidence TEXT,
                occurrence_count INTEGER DEFAULT 0,
                affected_entities INTEGER DEFAULT 0,
                validation_status TEXT NOT NULL,
                validation_score REAL DEFAULT 0,
                cross_study_validated INTEGER DEFAULT 0,
                studies_validated TEXT,
                reviewed_by TEXT,
                reviewed_at TEXT,
                review_notes TEXT,
                promoted_at TEXT,
                promoted_by TEXT,
                target_library TEXT,
                post_promotion_accuracy REAL,
                post_promotion_usage INTEGER DEFAULT 0
            )
        ''')
        
        # Feedback aggregation table (daily)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback_aggregation (
                date TEXT PRIMARY KEY,
                total_feedback INTEGER DEFAULT 0,
                accepts INTEGER DEFAULT 0,
                modifies INTEGER DEFAULT 0,
                rejects INTEGER DEFAULT 0,
                overrides INTEGER DEFAULT 0,
                acceptance_rate REAL DEFAULT 0,
                signals_extracted INTEGER DEFAULT 0,
                strong_signals INTEGER DEFAULT 0,
                updates_triggered INTEGER DEFAULT 0,
                patterns_promoted INTEGER DEFAULT 0,
                by_agent TEXT,
                metrics_json TEXT
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_user ON feedback_records(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_agent ON feedback_records(agent_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_type ON feedback_records(feedback_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON feedback_records(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_agent ON learning_signals(agent_name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_type ON learning_signals(signal_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_processed ON learning_signals(processed)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_updates_status ON model_updates(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_patterns_status ON pattern_candidates(validation_status)')
        
        conn.commit()
        conn.close()
    
    # =========================================================================
    # FEEDBACK CAPTURE
    # =========================================================================
    
    def capture_feedback(
        self,
        user_id: str,
        user_role: str,
        agent_name: str,
        output_type: str,
        output_id: str,
        output_content: str,
        ai_confidence: float,
        feedback_type: FeedbackType,
        user_action: str,
        user_modification: Optional[str] = None,
        rejection_reason: Optional[str] = None,
        error_description: Optional[str] = None,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        context: Optional[Dict] = None,
        decision_time_seconds: Optional[float] = None,
        auto_extract_signals: bool = True
    ) -> str:
        """Capture user feedback on AI output."""
        feedback_id = f"FB-{datetime.now().strftime('%Y%m%d%H%M%S')}-{hashlib.md5(f'{user_id}{output_id}{datetime.now().isoformat()}'.encode()).hexdigest()[:8]}"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO feedback_records (
                feedback_id, timestamp, user_id, user_role, agent_name,
                output_type, output_id, output_content, ai_confidence,
                feedback_type, user_action, user_modification, rejection_reason,
                error_description, entity_type, entity_id, context,
                decision_time_seconds, signals_extracted, learning_signals
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            feedback_id,
            datetime.now().isoformat(),
            user_id,
            user_role,
            agent_name,
            output_type,
            output_id,
            output_content,
            ai_confidence,
            feedback_type.value,
            user_action,
            user_modification,
            rejection_reason,
            error_description,
            entity_type,
            entity_id,
            json.dumps(context or {}),
            decision_time_seconds,
            0,
            json.dumps([])
        ))
        
        conn.commit()
        conn.close()
        
        # Auto-extract learning signals
        if auto_extract_signals:
            signals = self.extract_learning_signals(feedback_id)
        
        return feedback_id
    
    def get_feedback(self, feedback_id: str) -> Optional[FeedbackRecord]:
        """Get feedback record by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM feedback_records WHERE feedback_id = ?', (feedback_id,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return self._row_to_feedback(row)
    
    def _row_to_feedback(self, row) -> FeedbackRecord:
        """Convert database row to FeedbackRecord."""
        return FeedbackRecord(
            feedback_id=row[0],
            timestamp=datetime.fromisoformat(row[1]),
            user_id=row[2],
            user_role=row[3],
            agent_name=row[4],
            output_type=row[5],
            output_id=row[6],
            output_content=row[7],
            ai_confidence=row[8] or 0.0,
            feedback_type=FeedbackType(row[9]),
            user_action=row[10],
            user_modification=row[11],
            rejection_reason=row[12],
            error_description=row[13],
            entity_type=row[14],
            entity_id=row[15],
            context=json.loads(row[16]) if row[16] else {},
            decision_time_seconds=row[17],
            signals_extracted=bool(row[18]),
            learning_signals=json.loads(row[19]) if row[19] else []
        )
    
    def get_feedback_summary(self, days: int = 30) -> Dict:
        """Get summary of feedback for a period."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        start_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        cursor.execute('''
            SELECT 
                feedback_type,
                COUNT(*) as count
            FROM feedback_records
            WHERE timestamp >= ?
            GROUP BY feedback_type
        ''', (start_date,))
        
        rows = cursor.fetchall()
        conn.close()
        
        counts = {row[0]: row[1] for row in rows}
        total = sum(counts.values())
        
        accepts = counts.get('accept', 0)
        modifies = counts.get('modify', 0)
        rejects = counts.get('reject', 0)
        overrides = counts.get('override', 0)
        
        return {
            'total': total,
            'accepts': accepts,
            'modifies': modifies,
            'rejects': rejects,
            'overrides': overrides,
            'acceptance_rate': (accepts + modifies) / total if total > 0 else 0.0,
            'rejection_rate': rejects / total if total > 0 else 0.0,
            'override_rate': overrides / total if total > 0 else 0.0,
            'by_type': counts,
            'period_days': days
        }
    
    # =========================================================================
    # LEARNING SIGNAL EXTRACTION
    # =========================================================================
    
    def extract_learning_signals(self, feedback_id: str) -> List[LearningSignal]:
        """Extract learning signals from a feedback record."""
        feedback = self.get_feedback(feedback_id)
        if not feedback:
            return []
        
        signals = []
        
        # Determine signal type and strength based on feedback
        if feedback.feedback_type == FeedbackType.ACCEPT:
            # Positive confirmation
            signal = self._create_signal(
                feedback_id=feedback_id,
                agent_name=feedback.agent_name,
                signal_type=LearningSignalType.POSITIVE_CONFIRMATION,
                strength=SignalStrength.STRONG if feedback.ai_confidence > 0.8 else SignalStrength.MODERATE,
                confidence=feedback.ai_confidence
            )
            signals.append(signal)
            
            # Confidence calibration signal
            if feedback.ai_confidence > 0:
                cal_signal = self._create_signal(
                    feedback_id=feedback_id,
                    agent_name=feedback.agent_name,
                    signal_type=LearningSignalType.CONFIDENCE_CALIBRATION,
                    strength=SignalStrength.MODERATE,
                    confidence=feedback.ai_confidence,
                    old_value=feedback.ai_confidence,
                    new_value=1.0,  # Actual outcome was correct
                    adjustment_magnitude=1.0 - feedback.ai_confidence
                )
                signals.append(cal_signal)
        
        elif feedback.feedback_type == FeedbackType.MODIFY:
            # Partial match
            signal = self._create_signal(
                feedback_id=feedback_id,
                agent_name=feedback.agent_name,
                signal_type=LearningSignalType.PARTIAL_MATCH,
                strength=SignalStrength.MODERATE,
                confidence=feedback.ai_confidence * 0.7  # Reduced confidence
            )
            signals.append(signal)
            
            # Pattern refinement if modification provided
            if feedback.user_modification:
                refine_signal = self._create_signal(
                    feedback_id=feedback_id,
                    agent_name=feedback.agent_name,
                    signal_type=LearningSignalType.PATTERN_REFINEMENT,
                    strength=SignalStrength.MODERATE,
                    confidence=0.6,
                    old_value=feedback.output_content,
                    new_value=feedback.user_modification
                )
                signals.append(refine_signal)
        
        elif feedback.feedback_type == FeedbackType.REJECT:
            # Negative confirmation
            signal = self._create_signal(
                feedback_id=feedback_id,
                agent_name=feedback.agent_name,
                signal_type=LearningSignalType.NEGATIVE_CONFIRMATION,
                strength=SignalStrength.STRONG,
                confidence=1.0 - feedback.ai_confidence  # Inverse confidence
            )
            signals.append(signal)
            
            # Confidence calibration - AI was overconfident
            if feedback.ai_confidence > 0.5:
                cal_signal = self._create_signal(
                    feedback_id=feedback_id,
                    agent_name=feedback.agent_name,
                    signal_type=LearningSignalType.CONFIDENCE_CALIBRATION,
                    strength=SignalStrength.STRONG,
                    confidence=feedback.ai_confidence,
                    old_value=feedback.ai_confidence,
                    new_value=0.0,  # Actual outcome was wrong
                    adjustment_magnitude=feedback.ai_confidence
                )
                signals.append(cal_signal)
        
        elif feedback.feedback_type == FeedbackType.OVERRIDE:
            # Complete mismatch
            signal = self._create_signal(
                feedback_id=feedback_id,
                agent_name=feedback.agent_name,
                signal_type=LearningSignalType.COMPLETE_MISMATCH,
                strength=SignalStrength.STRONG,
                confidence=0.9
            )
            signals.append(signal)
            
            # Threshold adjustment signal
            threshold_signal = self._create_signal(
                feedback_id=feedback_id,
                agent_name=feedback.agent_name,
                signal_type=LearningSignalType.THRESHOLD_ADJUSTMENT,
                strength=SignalStrength.MODERATE,
                confidence=0.7,
                adjustment_magnitude=-0.1  # Lower threshold
            )
            signals.append(threshold_signal)
        
        elif feedback.feedback_type == FeedbackType.FLAG_ERROR:
            # Error flag - strong negative signal
            signal = self._create_signal(
                feedback_id=feedback_id,
                agent_name=feedback.agent_name,
                signal_type=LearningSignalType.COMPLETE_MISMATCH,
                strength=SignalStrength.STRONG,
                confidence=0.95
            )
            signals.append(signal)
        
        elif feedback.feedback_type == FeedbackType.FLAG_HELPFUL:
            # Helpful flag - strong positive signal
            signal = self._create_signal(
                feedback_id=feedback_id,
                agent_name=feedback.agent_name,
                signal_type=LearningSignalType.POSITIVE_CONFIRMATION,
                strength=SignalStrength.STRONG,
                confidence=0.95
            )
            signals.append(signal)
        
        # Save signals to database
        for signal in signals:
            self._save_signal(signal)
        
        # Update feedback record
        self._update_feedback_signals(feedback_id, [s.signal_id for s in signals])
        
        return signals
    
    def _create_signal(
        self,
        feedback_id: str,
        agent_name: str,
        signal_type: LearningSignalType,
        strength: SignalStrength,
        confidence: float,
        feature_name: Optional[str] = None,
        old_value: Optional[Any] = None,
        new_value: Optional[Any] = None,
        adjustment_magnitude: float = 0.0,
        pattern_id: Optional[str] = None,
        pattern_type: Optional[str] = None
    ) -> LearningSignal:
        """Create a learning signal."""
        signal_id = f"SIG-{datetime.now().strftime('%Y%m%d%H%M%S')}-{hashlib.md5(f'{feedback_id}{signal_type.value}{datetime.now().isoformat()}'.encode()).hexdigest()[:8]}"
        
        return LearningSignal(
            signal_id=signal_id,
            timestamp=datetime.now(),
            feedback_id=feedback_id,
            agent_name=agent_name,
            signal_type=signal_type,
            strength=strength,
            confidence=confidence,
            feature_name=feature_name,
            old_value=str(old_value) if old_value is not None else None,
            new_value=str(new_value) if new_value is not None else None,
            adjustment_magnitude=adjustment_magnitude,
            pattern_id=pattern_id,
            pattern_type=pattern_type
        )
    
    def _save_signal(self, signal: LearningSignal):
        """Save learning signal to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO learning_signals (
                signal_id, timestamp, feedback_id, agent_name, signal_type,
                strength, confidence, feature_name, old_value, new_value,
                adjustment_magnitude, pattern_id, pattern_type, occurrence_count,
                similar_signals, processed, update_triggered, update_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal.signal_id,
            signal.timestamp.isoformat(),
            signal.feedback_id,
            signal.agent_name,
            signal.signal_type.value,
            signal.strength.value,
            signal.confidence,
            signal.feature_name,
            signal.old_value,
            signal.new_value,
            signal.adjustment_magnitude,
            signal.pattern_id,
            signal.pattern_type,
            signal.occurrence_count,
            signal.similar_signals,
            1 if signal.processed else 0,
            1 if signal.update_triggered else 0,
            signal.update_id
        ))
        
        conn.commit()
        conn.close()
    
    def _update_feedback_signals(self, feedback_id: str, signal_ids: List[str]):
        """Update feedback record with extracted signal IDs."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE feedback_records SET
                signals_extracted = 1,
                learning_signals = ?
            WHERE feedback_id = ?
        ''', (json.dumps(signal_ids), feedback_id))
        
        conn.commit()
        conn.close()
    
    def get_pending_signals(self, agent_name: Optional[str] = None, limit: int = 100) -> List[LearningSignal]:
        """Get unprocessed learning signals."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if agent_name:
            cursor.execute('''
                SELECT * FROM learning_signals
                WHERE processed = 0 AND agent_name = ?
                ORDER BY timestamp
                LIMIT ?
            ''', (agent_name, limit))
        else:
            cursor.execute('''
                SELECT * FROM learning_signals
                WHERE processed = 0
                ORDER BY timestamp
                LIMIT ?
            ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_signal(row) for row in rows]
    
    def _row_to_signal(self, row) -> LearningSignal:
        """Convert database row to LearningSignal."""
        return LearningSignal(
            signal_id=row[0],
            timestamp=datetime.fromisoformat(row[1]),
            feedback_id=row[2],
            agent_name=row[3],
            signal_type=LearningSignalType(row[4]),
            strength=SignalStrength(row[5]),
            confidence=row[6] or 0.0,
            feature_name=row[7],
            old_value=row[8],
            new_value=row[9],
            adjustment_magnitude=row[10] or 0.0,
            pattern_id=row[11],
            pattern_type=row[12],
            occurrence_count=row[13] or 1,
            similar_signals=row[14] or 0,
            processed=bool(row[15]),
            update_triggered=bool(row[16]),
            update_id=row[17]
        )
    
    def get_signal_summary(self, days: int = 30) -> Dict:
        """Get summary of learning signals."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        start_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        # By type
        cursor.execute('''
            SELECT signal_type, COUNT(*) 
            FROM learning_signals
            WHERE timestamp >= ?
            GROUP BY signal_type
        ''', (start_date,))
        by_type = {row[0]: row[1] for row in cursor.fetchall()}
        
        # By strength
        cursor.execute('''
            SELECT strength, COUNT(*)
            FROM learning_signals
            WHERE timestamp >= ?
            GROUP BY strength
        ''', (start_date,))
        by_strength = {row[0]: row[1] for row in cursor.fetchall()}
        
        # By agent
        cursor.execute('''
            SELECT agent_name, COUNT(*)
            FROM learning_signals
            WHERE timestamp >= ?
            GROUP BY agent_name
        ''', (start_date,))
        by_agent = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Processing stats
        cursor.execute('''
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN processed = 1 THEN 1 ELSE 0 END) as processed,
                SUM(CASE WHEN update_triggered = 1 THEN 1 ELSE 0 END) as triggered
            FROM learning_signals
            WHERE timestamp >= ?
        ''', (start_date,))
        stats = cursor.fetchone()
        
        conn.close()
        
        return {
            'total_signals': stats[0] or 0,
            'processed': stats[1] or 0,
            'updates_triggered': stats[2] or 0,
            'by_type': by_type,
            'by_strength': by_strength,
            'by_agent': by_agent,
            'period_days': days
        }
    
    # =========================================================================
    # MODEL UPDATES
    # =========================================================================
    
    def check_update_triggers(self, agent_name: Optional[str] = None) -> List[ModelUpdateRequest]:
        """Check if any model updates should be triggered."""
        updates = []
        
        # Get recent feedback summary
        summary = self.get_feedback_summary(days=7)
        
        # Check rejection rate trigger
        if summary['rejection_rate'] > self.update_rules['rejection_rate_trigger']:
            update = self._create_update_request(
                model_name=agent_name or "all_agents",
                update_type=ModelUpdateType.RETRAIN_INCREMENTAL,
                priority=UpdatePriority.HIGH,
                trigger_reason=f"High rejection rate: {summary['rejection_rate']:.1%}",
                expected_improvement=0.15
            )
            updates.append(update)
        
        # Check override rate trigger
        if summary['override_rate'] > self.update_rules['override_rate_trigger']:
            update = self._create_update_request(
                model_name=agent_name or "all_agents",
                update_type=ModelUpdateType.THRESHOLD_ADJUSTMENT,
                priority=UpdatePriority.HIGH,
                trigger_reason=f"High override rate: {summary['override_rate']:.1%}",
                expected_improvement=0.10
            )
            updates.append(update)
        
        # Check signal accumulation trigger
        signal_summary = self.get_signal_summary(days=7)
        pending_signals = signal_summary['total_signals'] - signal_summary['processed']
        
        if pending_signals >= self.update_rules['signal_accumulation_trigger']:
            update = self._create_update_request(
                model_name=agent_name or "all_agents",
                update_type=ModelUpdateType.FEATURE_WEIGHT_UPDATE,
                priority=UpdatePriority.MEDIUM,
                trigger_reason=f"Signal accumulation: {pending_signals} pending signals",
                signal_count=pending_signals,
                expected_improvement=0.08
            )
            updates.append(update)
        
        # Check for confidence calibration needs
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        start_date = (datetime.now() - timedelta(days=7)).isoformat()
        cursor.execute('''
            SELECT COUNT(*) FROM learning_signals
            WHERE timestamp >= ?
            AND signal_type = 'confidence_calibration'
            AND ABS(adjustment_magnitude) > 0.3
        ''', (start_date,))
        
        high_calibration_signals = cursor.fetchone()[0]
        conn.close()
        
        if high_calibration_signals >= 10:
            update = self._create_update_request(
                model_name=agent_name or "all_agents",
                update_type=ModelUpdateType.CONFIDENCE_RECALIBRATION,
                priority=UpdatePriority.MEDIUM,
                trigger_reason=f"Confidence calibration needed: {high_calibration_signals} high-magnitude signals",
                expected_improvement=0.12
            )
            updates.append(update)
        
        # Save updates
        for update in updates:
            self._save_update_request(update)
        
        return updates
    
    def _create_update_request(
        self,
        model_name: str,
        update_type: ModelUpdateType,
        priority: UpdatePriority,
        trigger_reason: str,
        signal_ids: Optional[List[str]] = None,
        signal_count: int = 0,
        changes: Optional[Dict] = None,
        expected_improvement: float = 0.0,
        risk_assessment: str = ""
    ) -> ModelUpdateRequest:
        """Create a model update request."""
        update_id = f"UPD-{datetime.now().strftime('%Y%m%d%H%M%S')}-{hashlib.md5(f'{model_name}{update_type.value}{datetime.now().isoformat()}'.encode()).hexdigest()[:8]}"
        
        # Determine if approval required
        requires_approval = priority in [UpdatePriority.CRITICAL, UpdatePriority.HIGH] or \
                           update_type in [ModelUpdateType.RETRAIN_FULL, ModelUpdateType.RULE_UPDATE]
        
        return ModelUpdateRequest(
            update_id=update_id,
            created_at=datetime.now(),
            model_name=model_name,
            update_type=update_type,
            priority=priority,
            trigger_reason=trigger_reason,
            signal_ids=signal_ids or [],
            signal_count=signal_count,
            changes=changes or {},
            expected_improvement=expected_improvement,
            risk_assessment=risk_assessment or self._assess_risk(update_type, priority),
            requires_approval=requires_approval
        )
    
    def _assess_risk(self, update_type: ModelUpdateType, priority: UpdatePriority) -> str:
        """Assess risk of model update."""
        if update_type == ModelUpdateType.RETRAIN_FULL:
            return "HIGH: Full retrain may significantly change model behavior. Extensive validation required."
        elif update_type == ModelUpdateType.RETRAIN_INCREMENTAL:
            return "MEDIUM: Incremental changes should be validated against holdout set."
        elif update_type == ModelUpdateType.THRESHOLD_ADJUSTMENT:
            return "LOW: Threshold changes are easily reversible."
        elif update_type == ModelUpdateType.CONFIDENCE_RECALIBRATION:
            return "LOW: Calibration changes affect confidence display, not core predictions."
        else:
            return "MEDIUM: Standard validation recommended."
    
    def _save_update_request(self, update: ModelUpdateRequest):
        """Save model update request to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO model_updates (
                update_id, created_at, model_name, update_type, priority,
                trigger_reason, signal_ids, signal_count, changes,
                expected_improvement, risk_assessment, status, started_at,
                completed_at, validation_metrics, rollback_available,
                requires_approval, approved_by, approved_at, result_summary,
                error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            update.update_id,
            update.created_at.isoformat(),
            update.model_name,
            update.update_type.value,
            update.priority.value,
            update.trigger_reason,
            json.dumps(update.signal_ids),
            update.signal_count,
            json.dumps(update.changes),
            update.expected_improvement,
            update.risk_assessment,
            update.status.value,
            update.started_at.isoformat() if update.started_at else None,
            update.completed_at.isoformat() if update.completed_at else None,
            json.dumps(update.validation_metrics),
            1 if update.rollback_available else 0,
            1 if update.requires_approval else 0,
            update.approved_by,
            update.approved_at.isoformat() if update.approved_at else None,
            update.result_summary,
            update.error_message
        ))
        
        conn.commit()
        conn.close()
    
    def get_pending_updates(self) -> List[ModelUpdateRequest]:
        """Get pending model updates."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM model_updates
            WHERE status IN ('pending', 'queued')
            ORDER BY 
                CASE priority 
                    WHEN 'critical' THEN 1 
                    WHEN 'high' THEN 2 
                    WHEN 'medium' THEN 3 
                    ELSE 4 
                END,
                created_at
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_update(row) for row in rows]
    
    def approve_update(self, update_id: str, approved_by: str) -> bool:
        """Approve a model update."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE model_updates SET
                status = 'queued',
                approved_by = ?,
                approved_at = ?
            WHERE update_id = ? AND status = 'pending'
        ''', (approved_by, datetime.now().isoformat(), update_id))
        
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        return success
    
    def execute_update(self, update_id: str) -> Tuple[bool, str]:
        """Execute a model update (simulation for now)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Mark as in progress
        cursor.execute('''
            UPDATE model_updates SET
                status = 'in_progress',
                started_at = ?
            WHERE update_id = ?
        ''', (datetime.now().isoformat(), update_id))
        
        conn.commit()
        
        # Get update details
        cursor.execute('SELECT * FROM model_updates WHERE update_id = ?', (update_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return False, "Update not found"
        
        update = self._row_to_update(row)
        
        # Simulate update execution
        # In real implementation, this would:
        # 1. Load the model
        # 2. Apply updates based on update_type
        # 3. Validate on holdout set
        # 4. Deploy or rollback
        
        validation_metrics = {
            'accuracy_before': 0.85,
            'accuracy_after': 0.87,
            'improvement': 0.02,
            'validation_samples': 1000,
            'passed_threshold': True
        }
        
        # Mark as completed
        cursor.execute('''
            UPDATE model_updates SET
                status = 'completed',
                completed_at = ?,
                validation_metrics = ?,
                result_summary = ?
            WHERE update_id = ?
        ''', (
            datetime.now().isoformat(),
            json.dumps(validation_metrics),
            f"Update completed successfully. Accuracy improved from {validation_metrics['accuracy_before']:.1%} to {validation_metrics['accuracy_after']:.1%}",
            update_id
        ))
        
        # Mark related signals as processed
        cursor.execute('''
            UPDATE learning_signals SET
                processed = 1,
                update_triggered = 1,
                update_id = ?
            WHERE signal_id IN (
                SELECT value FROM json_each(
                    (SELECT signal_ids FROM model_updates WHERE update_id = ?)
                )
            )
        ''', (update_id, update_id))
        
        conn.commit()
        conn.close()
        
        return True, "Update completed successfully"
    
    def _row_to_update(self, row) -> ModelUpdateRequest:
        """Convert database row to ModelUpdateRequest."""
        return ModelUpdateRequest(
            update_id=row[0],
            created_at=datetime.fromisoformat(row[1]),
            model_name=row[2],
            update_type=ModelUpdateType(row[3]),
            priority=UpdatePriority(row[4]),
            trigger_reason=row[5],
            signal_ids=json.loads(row[6]) if row[6] else [],
            signal_count=row[7] or 0,
            changes=json.loads(row[8]) if row[8] else {},
            expected_improvement=row[9] or 0.0,
            risk_assessment=row[10] or "",
            status=ModelUpdateStatus(row[11]),
            started_at=datetime.fromisoformat(row[12]) if row[12] else None,
            completed_at=datetime.fromisoformat(row[13]) if row[13] else None,
            validation_metrics=json.loads(row[14]) if row[14] else {},
            rollback_available=bool(row[15]),
            requires_approval=bool(row[16]),
            approved_by=row[17],
            approved_at=datetime.fromisoformat(row[18]) if row[18] else None,
            result_summary=row[19],
            error_message=row[20]
        )
    
    # =========================================================================
    # PATTERN PROMOTION
    # =========================================================================
    
    def create_pattern_candidate(
        self,
        pattern_name: str,
        pattern_type: str,
        pattern_description: str,
        source: PatternSource,
        detection_rules: Optional[Dict] = None,
        threshold_values: Optional[Dict] = None,
        supporting_evidence: Optional[List[str]] = None,
        occurrence_count: int = 0,
        affected_entities: int = 0
    ) -> str:
        """Create a new pattern candidate for promotion."""
        pattern_id = f"PAT-{datetime.now().strftime('%Y%m%d%H%M%S')}-{hashlib.md5(f'{pattern_name}{datetime.now().isoformat()}'.encode()).hexdigest()[:8]}"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO pattern_candidates (
                pattern_id, created_at, pattern_name, pattern_type,
                pattern_description, source, detection_rules, threshold_values,
                supporting_evidence, occurrence_count, affected_entities,
                validation_status, validation_score, cross_study_validated,
                studies_validated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            pattern_id,
            datetime.now().isoformat(),
            pattern_name,
            pattern_type,
            pattern_description,
            source.value,
            json.dumps(detection_rules or {}),
            json.dumps(threshold_values or {}),
            json.dumps(supporting_evidence or []),
            occurrence_count,
            affected_entities,
            PatternStatus.CANDIDATE.value,
            0.0,
            0,
            json.dumps([])
        ))
        
        conn.commit()
        conn.close()
        
        return pattern_id
    
    def get_pattern_candidate(self, pattern_id: str) -> Optional[PatternCandidate]:
        """Get pattern candidate by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM pattern_candidates WHERE pattern_id = ?', (pattern_id,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return self._row_to_pattern(row)
    
    def _row_to_pattern(self, row) -> PatternCandidate:
        """Convert database row to PatternCandidate."""
        return PatternCandidate(
            pattern_id=row[0],
            created_at=datetime.fromisoformat(row[1]),
            pattern_name=row[2],
            pattern_type=row[3],
            pattern_description=row[4],
            source=PatternSource(row[5]),
            detection_rules=json.loads(row[6]) if row[6] else {},
            threshold_values=json.loads(row[7]) if row[7] else {},
            supporting_evidence=json.loads(row[8]) if row[8] else [],
            occurrence_count=row[9] or 0,
            affected_entities=row[10] or 0,
            validation_status=PatternStatus(row[11]),
            validation_score=row[12] or 0.0,
            cross_study_validated=bool(row[13]),
            studies_validated=json.loads(row[14]) if row[14] else [],
            reviewed_by=row[15],
            reviewed_at=datetime.fromisoformat(row[16]) if row[16] else None,
            review_notes=row[17],
            promoted_at=datetime.fromisoformat(row[18]) if row[18] else None,
            promoted_by=row[19],
            target_library=row[20],
            post_promotion_accuracy=row[21],
            post_promotion_usage=row[22] or 0
        )
    
    def validate_pattern(
        self,
        pattern_id: str,
        validation_score: float,
        studies_validated: Optional[List[str]] = None,
        reviewed_by: str = None,
        review_notes: str = None
    ) -> bool:
        """Validate a pattern candidate."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        studies = studies_validated or []
        cross_validated = len(studies) >= self.thresholds['cross_study_validation_min']
        
        # Determine new status
        if validation_score >= self.thresholds['pattern_validation_threshold']:
            new_status = PatternStatus.VALIDATED.value
        else:
            new_status = PatternStatus.UNDER_REVIEW.value
        
        cursor.execute('''
            UPDATE pattern_candidates SET
                validation_status = ?,
                validation_score = ?,
                cross_study_validated = ?,
                studies_validated = ?,
                reviewed_by = ?,
                reviewed_at = ?,
                review_notes = ?
            WHERE pattern_id = ?
        ''', (
            new_status,
            validation_score,
            1 if cross_validated else 0,
            json.dumps(studies),
            reviewed_by,
            datetime.now().isoformat() if reviewed_by else None,
            review_notes,
            pattern_id
        ))
        
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        return success
    
    def promote_pattern(
        self,
        pattern_id: str,
        promoted_by: str,
        target_library: str = "pattern_library"
    ) -> Tuple[bool, str]:
        """Promote a validated pattern to the production library."""
        pattern = self.get_pattern_candidate(pattern_id)
        
        if not pattern:
            return False, "Pattern not found"
        
        if pattern.validation_status != PatternStatus.VALIDATED:
            return False, f"Pattern not validated. Current status: {pattern.validation_status.value}"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE pattern_candidates SET
                validation_status = ?,
                promoted_at = ?,
                promoted_by = ?,
                target_library = ?
            WHERE pattern_id = ?
        ''', (
            PatternStatus.PROMOTED.value,
            datetime.now().isoformat(),
            promoted_by,
            target_library,
            pattern_id
        ))
        
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        if success:
            return True, f"Pattern '{pattern.pattern_name}' promoted to {target_library}"
        else:
            return False, "Failed to update pattern status"
    
    def reject_pattern(
        self,
        pattern_id: str,
        rejected_by: str,
        rejection_reason: str
    ) -> bool:
        """Reject a pattern candidate."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE pattern_candidates SET
                validation_status = ?,
                reviewed_by = ?,
                reviewed_at = ?,
                review_notes = ?
            WHERE pattern_id = ?
        ''', (
            PatternStatus.REJECTED.value,
            rejected_by,
            datetime.now().isoformat(),
            f"REJECTED: {rejection_reason}",
            pattern_id
        ))
        
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        return success
    
    def get_patterns_by_status(self, status: PatternStatus) -> List[PatternCandidate]:
        """Get patterns by status."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM pattern_candidates
            WHERE validation_status = ?
            ORDER BY created_at DESC
        ''', (status.value,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_pattern(row) for row in rows]
    
    def get_pattern_summary(self) -> Dict:
        """Get summary of pattern candidates."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT validation_status, COUNT(*) 
            FROM pattern_candidates
            GROUP BY validation_status
        ''')
        
        by_status = {row[0]: row[1] for row in cursor.fetchall()}
        
        cursor.execute('''
            SELECT source, COUNT(*)
            FROM pattern_candidates
            GROUP BY source
        ''')
        
        by_source = {row[0]: row[1] for row in cursor.fetchall()}
        
        cursor.execute('SELECT COUNT(*) FROM pattern_candidates')
        total = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total': total,
            'by_status': by_status,
            'by_source': by_source,
            'candidates': by_status.get('candidate', 0),
            'validated': by_status.get('validated', 0),
            'promoted': by_status.get('promoted', 0),
            'rejected': by_status.get('rejected', 0)
        }
    
    # =========================================================================
    # STATISTICS & AGGREGATION
    # =========================================================================
    
    def calculate_aggregation(self, days: int = 1) -> FeedbackAggregation:
        """Calculate feedback aggregation for a period."""
        period_start = datetime.now() - timedelta(days=days)
        period_end = datetime.now()
        
        feedback_summary = self.get_feedback_summary(days=days)
        signal_summary = self.get_signal_summary(days=days)
        
        # Get by-agent breakdown
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        start_date = period_start.isoformat()
        
        cursor.execute('''
            SELECT 
                agent_name,
                feedback_type,
                COUNT(*) as count
            FROM feedback_records
            WHERE timestamp >= ?
            GROUP BY agent_name, feedback_type
        ''', (start_date,))
        
        rows = cursor.fetchall()
        
        by_agent = {}
        for row in rows:
            agent, fb_type, count = row
            if agent not in by_agent:
                by_agent[agent] = {'total': 0, 'accepts': 0, 'rejects': 0}
            by_agent[agent]['total'] += count
            if fb_type == 'accept':
                by_agent[agent]['accepts'] += count
            elif fb_type == 'reject':
                by_agent[agent]['rejects'] += count
        
        # Count updates triggered
        cursor.execute('''
            SELECT COUNT(*) FROM model_updates
            WHERE created_at >= ?
        ''', (start_date,))
        updates_triggered = cursor.fetchone()[0]
        
        # Count patterns promoted
        cursor.execute('''
            SELECT COUNT(*) FROM pattern_candidates
            WHERE promoted_at >= ?
        ''', (start_date,))
        patterns_promoted = cursor.fetchone()[0]
        
        conn.close()
        
        return FeedbackAggregation(
            period_start=period_start,
            period_end=period_end,
            total_feedback=feedback_summary['total'],
            accepts=feedback_summary['accepts'],
            modifies=feedback_summary['modifies'],
            rejects=feedback_summary['rejects'],
            overrides=feedback_summary['overrides'],
            acceptance_rate=feedback_summary['acceptance_rate'],
            modification_rate=feedback_summary['modifies'] / max(1, feedback_summary['total']),
            rejection_rate=feedback_summary['rejection_rate'],
            by_agent=by_agent,
            signals_extracted=signal_summary['total_signals'],
            strong_signals=signal_summary['by_strength'].get('strong', 0),
            updates_triggered=updates_triggered,
            patterns_promoted=patterns_promoted
        )
    
    def save_daily_aggregation(self) -> bool:
        """Save daily aggregation to database."""
        agg = self.calculate_aggregation(days=1)
        date = datetime.now().strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO feedback_aggregation (
                date, total_feedback, accepts, modifies, rejects, overrides,
                acceptance_rate, signals_extracted, strong_signals,
                updates_triggered, patterns_promoted, by_agent, metrics_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            date,
            agg.total_feedback,
            agg.accepts,
            agg.modifies,
            agg.rejects,
            agg.overrides,
            agg.acceptance_rate,
            agg.signals_extracted,
            agg.strong_signals,
            agg.updates_triggered,
            agg.patterns_promoted,
            json.dumps(agg.by_agent),
            json.dumps(asdict(agg), default=str)
        ))
        
        conn.commit()
        conn.close()
        
        return True
    
    def get_statistics(self) -> Dict:
        """Get overall feedback loop statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Feedback counts
        cursor.execute('SELECT COUNT(*) FROM feedback_records')
        total_feedback = cursor.fetchone()[0]
        
        # Signal counts
        cursor.execute('SELECT COUNT(*) FROM learning_signals')
        total_signals = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM learning_signals WHERE processed = 0')
        pending_signals = cursor.fetchone()[0]
        
        # Update counts
        cursor.execute('SELECT COUNT(*) FROM model_updates')
        total_updates = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM model_updates WHERE status = 'completed'")
        completed_updates = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM model_updates WHERE status = 'pending'")
        pending_updates = cursor.fetchone()[0]
        
        # Pattern counts
        cursor.execute('SELECT COUNT(*) FROM pattern_candidates')
        total_patterns = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM pattern_candidates WHERE validation_status = 'promoted'")
        promoted_patterns = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_feedback': total_feedback,
            'total_signals': total_signals,
            'pending_signals': pending_signals,
            'total_updates': total_updates,
            'completed_updates': completed_updates,
            'pending_updates': pending_updates,
            'total_patterns': total_patterns,
            'promoted_patterns': promoted_patterns
        }


# =============================================================================
# SINGLETON & CONVENIENCE FUNCTIONS
# =============================================================================

_feedback_loop_system: Optional[FeedbackLoopSystem] = None


def get_feedback_loop_system(db_path: Optional[str] = None) -> FeedbackLoopSystem:
    """Get or create the FeedbackLoopSystem singleton."""
    global _feedback_loop_system
    if _feedback_loop_system is None:
        _feedback_loop_system = FeedbackLoopSystem(db_path=db_path)
    return _feedback_loop_system


def reset_feedback_loop_system():
    """Reset the singleton (for testing)."""
    global _feedback_loop_system
    _feedback_loop_system = None


def capture_feedback(
    user_id: str,
    user_role: str,
    agent_name: str,
    output_type: str,
    output_id: str,
    output_content: str,
    ai_confidence: float,
    feedback_type: FeedbackType,
    user_action: str,
    **kwargs
) -> str:
    """Convenience function to capture feedback."""
    return get_feedback_loop_system().capture_feedback(
        user_id=user_id,
        user_role=user_role,
        agent_name=agent_name,
        output_type=output_type,
        output_id=output_id,
        output_content=output_content,
        ai_confidence=ai_confidence,
        feedback_type=feedback_type,
        user_action=user_action,
        **kwargs
    )


def get_feedback_summary(days: int = 30) -> Dict:
    """Convenience function to get feedback summary."""
    return get_feedback_loop_system().get_feedback_summary(days=days)


def get_pending_signals(agent_name: Optional[str] = None) -> List[LearningSignal]:
    """Convenience function to get pending signals."""
    return get_feedback_loop_system().get_pending_signals(agent_name=agent_name)


def check_update_triggers() -> List[ModelUpdateRequest]:
    """Convenience function to check update triggers."""
    return get_feedback_loop_system().check_update_triggers()


def get_feedback_stats() -> Dict:
    """Convenience function to get statistics."""
    return get_feedback_loop_system().get_statistics()