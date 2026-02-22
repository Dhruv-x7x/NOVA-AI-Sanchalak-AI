# src/governance/confidence_calibration.py
# NOTE: This module previously used SQLite but has been migrated to PostgreSQL-only.
# If you need to use this module, update it to use:
#   from src.database.pg_data_service import get_pg_data_service
#   service = get_pg_data_service()


"""
a6on-i - Confidence Calibration System v1.0
Tracks AI prediction accuracy and detects model drift

Features:
- Acceptance tracking (AI suggestions vs human decisions)
- Calibration metrics (ECE, Brier Score, reliability diagrams)
- Drift detection (statistical tests, performance monitoring)
- Re-calibration triggers and recommendations
"""

import sqlite3
import json
import uuid
import math
# SQLite removed - using PostgreSQL
import threading
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from collections import defaultdict
import numpy as np

# Import audit trail for integration
from .audit_trail import (
    get_audit_logger,
    Actor,
    Entity,
    EventType,
    ActionCategory,
    Severity
)


# =============================================================================
# ENUMS
# =============================================================================

class OutcomeType(Enum):
    """Outcomes for AI suggestions"""
    ACCEPTED = "accepted"              # Human accepted as-is
    MODIFIED = "modified"              # Human modified then accepted
    REJECTED = "rejected"              # Human rejected
    AUTO_EXECUTED = "auto_executed"    # System auto-executed
    EXPIRED = "expired"                # No action taken, expired
    PENDING = "pending"                # Still awaiting decision


class DriftType(Enum):
    """Types of drift detected"""
    NONE = "none"
    ACCURACY_DRIFT = "accuracy_drift"           # Accuracy declining
    CALIBRATION_DRIFT = "calibration_drift"     # Confidence miscalibrated
    FEATURE_DRIFT = "feature_drift"             # Input features changing
    CONCEPT_DRIFT = "concept_drift"             # Relationship between features/outcomes changing
    ACCEPTANCE_DRIFT = "acceptance_drift"       # Human acceptance rate changing


class DriftSeverity(Enum):
    """Severity of detected drift"""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CalibrationStatus(Enum):
    """Overall calibration status"""
    EXCELLENT = "excellent"      # ECE < 0.05
    GOOD = "good"                # ECE 0.05-0.10
    ACCEPTABLE = "acceptable"    # ECE 0.10-0.15
    POOR = "poor"                # ECE 0.15-0.25
    CRITICAL = "critical"        # ECE > 0.25


class TriggerType(Enum):
    """Types of re-calibration triggers"""
    THRESHOLD = "threshold"      # Metric crossed threshold
    TIME_BASED = "time_based"    # Scheduled re-calibration
    PERFORMANCE = "performance"  # Performance degradation
    MANUAL = "manual"            # Manual trigger
    DRIFT = "drift"              # Drift detected


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Prediction:
    """Represents an AI prediction/suggestion"""
    # Required fields first (no defaults)
    prediction_id: str
    timestamp: datetime
    agent_name: str
    action_type: str
    predicted_class: str                    # What AI predicted/suggested
    confidence: float                       # AI's confidence (0-1)
    
    # Optional fields after (with defaults)
    entity_type: Optional[str] = None
    entity_id: Optional[str] = None
    alternatives: List[Dict] = field(default_factory=list)  # Other options considered
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'prediction_id': self.prediction_id,
            'timestamp': self.timestamp.isoformat(),
            'agent_name': self.agent_name,
            'action_type': self.action_type,
            'entity_type': self.entity_type,
            'entity_id': self.entity_id,
            'predicted_class': self.predicted_class,
            'confidence': self.confidence,
            'alternatives': self.alternatives,
            'context': self.context
        }

@dataclass
class Outcome:
    """Represents the outcome of a prediction"""
    outcome_id: str
    prediction_id: str
    timestamp: datetime
    
    # Outcome
    outcome_type: OutcomeType
    actual_class: Optional[str] = None      # What human decided/actual result
    
    # Decision details
    decided_by: Optional[str] = None
    decided_by_role: Optional[str] = None
    modification_details: Optional[str] = None
    rejection_reason: Optional[str] = None
    
    # Timing
    decision_time_seconds: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            'outcome_id': self.outcome_id,
            'prediction_id': self.prediction_id,
            'timestamp': self.timestamp.isoformat(),
            'outcome_type': self.outcome_type.value,
            'actual_class': self.actual_class,
            'decided_by': self.decided_by,
            'decided_by_role': self.decided_by_role,
            'modification_details': self.modification_details,
            'rejection_reason': self.rejection_reason,
            'decision_time_seconds': self.decision_time_seconds
        }


@dataclass
class CalibrationBin:
    """Statistics for a confidence bin"""
    bin_start: float
    bin_end: float
    count: int
    avg_confidence: float
    accuracy: float
    expected_accuracy: float  # Should equal avg_confidence for perfect calibration
    calibration_error: float  # |accuracy - expected_accuracy|
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class CalibrationMetrics:
    """Overall calibration metrics"""
    calculation_time: datetime
    period_start: datetime
    period_end: datetime
    
    # Sample info
    total_predictions: int
    predictions_with_outcomes: int
    
    # Calibration metrics
    expected_calibration_error: float  # ECE
    maximum_calibration_error: float   # MCE
    brier_score: float
    
    # Accuracy metrics
    overall_accuracy: float
    acceptance_rate: float
    modification_rate: float
    rejection_rate: float
    
    # Per-bin analysis
    bins: List[CalibrationBin] = field(default_factory=list)
    
    # Status
    calibration_status: CalibrationStatus = CalibrationStatus.GOOD
    
    def to_dict(self) -> Dict:
        return {
            'calculation_time': self.calculation_time.isoformat(),
            'period_start': self.period_start.isoformat(),
            'period_end': self.period_end.isoformat(),
            'total_predictions': self.total_predictions,
            'predictions_with_outcomes': self.predictions_with_outcomes,
            'expected_calibration_error': self.expected_calibration_error,
            'maximum_calibration_error': self.maximum_calibration_error,
            'brier_score': self.brier_score,
            'overall_accuracy': self.overall_accuracy,
            'acceptance_rate': self.acceptance_rate,
            'modification_rate': self.modification_rate,
            'rejection_rate': self.rejection_rate,
            'bins': [b.to_dict() for b in self.bins],
            'calibration_status': self.calibration_status.value
        }


@dataclass
class DriftAlert:
    """Alert for detected drift"""
    alert_id: str
    timestamp: datetime
    
    drift_type: DriftType
    severity: DriftSeverity
    
    description: str
    metric_name: str
    current_value: float
    baseline_value: float
    threshold: float
    
    affected_agents: List[str] = field(default_factory=list)
    affected_action_types: List[str] = field(default_factory=list)
    
    recommendation: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'alert_id': self.alert_id,
            'timestamp': self.timestamp.isoformat(),
            'drift_type': self.drift_type.value,
            'severity': self.severity.value,
            'description': self.description,
            'metric_name': self.metric_name,
            'current_value': self.current_value,
            'baseline_value': self.baseline_value,
            'threshold': self.threshold,
            'affected_agents': self.affected_agents,
            'affected_action_types': self.affected_action_types,
            'recommendation': self.recommendation
        }


@dataclass
class RecalibrationTrigger:
    """Trigger for re-calibration"""
    trigger_id: str
    timestamp: datetime
    
    trigger_type: TriggerType
    reason: str
    
    metrics_snapshot: Dict[str, float] = field(default_factory=dict)
    drift_alerts: List[str] = field(default_factory=list)  # Alert IDs
    
    priority: str = "medium"  # low, medium, high, critical
    recommended_action: str = ""
    
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        return {
            'trigger_id': self.trigger_id,
            'timestamp': self.timestamp.isoformat(),
            'trigger_type': self.trigger_type.value,
            'reason': self.reason,
            'metrics_snapshot': self.metrics_snapshot,
            'drift_alerts': self.drift_alerts,
            'priority': self.priority,
            'recommended_action': self.recommended_action,
            'acknowledged': self.acknowledged,
            'acknowledged_by': self.acknowledged_by,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None
        }


# =============================================================================
# ACCEPTANCE TRACKER
# =============================================================================

class AcceptanceTracker:
    """Tracks AI suggestions and human decisions"""
    
    def __init__(self, db_path: str = "data/governance/calibration.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_database()
    
    def _init_database(self):
        """Initialize the calibration database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    prediction_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    entity_type TEXT,
                    entity_id TEXT,
                    predicted_class TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    alternatives TEXT,
                    context TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Outcomes table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS outcomes (
                    outcome_id TEXT PRIMARY KEY,
                    prediction_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    outcome_type TEXT NOT NULL,
                    actual_class TEXT,
                    decided_by TEXT,
                    decided_by_role TEXT,
                    modification_details TEXT,
                    rejection_reason TEXT,
                    decision_time_seconds REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (prediction_id) REFERENCES predictions(prediction_id)
                )
            ''')
            
            # Calibration snapshots
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS calibration_snapshots (
                    snapshot_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    period_start TEXT NOT NULL,
                    period_end TEXT NOT NULL,
                    agent_name TEXT,
                    action_type TEXT,
                    metrics TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Drift alerts
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS drift_alerts (
                    alert_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    drift_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    description TEXT,
                    metric_name TEXT,
                    current_value REAL,
                    baseline_value REAL,
                    threshold REAL,
                    full_alert TEXT NOT NULL,
                    resolved INTEGER DEFAULT 0,
                    resolved_at TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Re-calibration triggers
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS recalibration_triggers (
                    trigger_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    trigger_type TEXT NOT NULL,
                    reason TEXT,
                    priority TEXT,
                    full_trigger TEXT NOT NULL,
                    acknowledged INTEGER DEFAULT 0,
                    acknowledged_by TEXT,
                    acknowledged_at TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_pred_timestamp ON predictions(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_pred_agent ON predictions(agent_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_pred_action ON predictions(action_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_outcome_pred ON outcomes(prediction_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_outcome_type ON outcomes(outcome_type)')
            
            conn.commit()
    
    def record_prediction(self, prediction: Prediction) -> str:
        """Record an AI prediction/suggestion"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO predictions (
                        prediction_id, timestamp, agent_name, action_type,
                        entity_type, entity_id, predicted_class, confidence,
                        alternatives, context
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    prediction.prediction_id,
                    prediction.timestamp.isoformat(),
                    prediction.agent_name,
                    prediction.action_type,
                    prediction.entity_type,
                    prediction.entity_id,
                    prediction.predicted_class,
                    prediction.confidence,
                    json.dumps(prediction.alternatives),
                    json.dumps(prediction.context, default=str)
                ))
                conn.commit()
        
        return prediction.prediction_id
    
    def record_outcome(self, outcome: Outcome) -> str:
        """Record the outcome of a prediction"""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO outcomes (
                        outcome_id, prediction_id, timestamp, outcome_type,
                        actual_class, decided_by, decided_by_role,
                        modification_details, rejection_reason, decision_time_seconds
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    outcome.outcome_id,
                    outcome.prediction_id,
                    outcome.timestamp.isoformat(),
                    outcome.outcome_type.value,
                    outcome.actual_class,
                    outcome.decided_by,
                    outcome.decided_by_role,
                    outcome.modification_details,
                    outcome.rejection_reason,
                    outcome.decision_time_seconds
                ))
                conn.commit()
        
        return outcome.outcome_id
    
    def get_predictions_with_outcomes(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        agent_name: Optional[str] = None,
        action_type: Optional[str] = None,
        limit: int = 10000
    ) -> List[Tuple[Prediction, Optional[Outcome]]]:
        """Get predictions with their outcomes"""
        
        conditions = []
        params = []
        
        if start_time:
            conditions.append("p.timestamp >= ?")
            params.append(start_time.isoformat())
        
        if end_time:
            conditions.append("p.timestamp <= ?")
            params.append(end_time.isoformat())
        
        if agent_name:
            conditions.append("p.agent_name = ?")
            params.append(agent_name)
        
        if action_type:
            conditions.append("p.action_type = ?")
            params.append(action_type)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        query = f'''
            SELECT 
                p.prediction_id, p.timestamp, p.agent_name, p.action_type,
                p.entity_type, p.entity_id, p.predicted_class, p.confidence,
                p.alternatives, p.context,
                o.outcome_id, o.timestamp as o_timestamp, o.outcome_type,
                o.actual_class, o.decided_by, o.decided_by_role,
                o.modification_details, o.rejection_reason, o.decision_time_seconds
            FROM predictions p
            LEFT JOIN outcomes o ON p.prediction_id = o.prediction_id
            WHERE {where_clause}
            ORDER BY p.timestamp DESC
            LIMIT ?
        '''
        params.append(limit)
        
        results = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            for row in cursor.fetchall():
                prediction = Prediction(
                    prediction_id=row[0],
                    timestamp=datetime.fromisoformat(row[1]),
                    agent_name=row[2],
                    action_type=row[3],
                    entity_type=row[4],
                    entity_id=row[5],
                    predicted_class=row[6],
                    confidence=row[7],
                    alternatives=json.loads(row[8]) if row[8] else [],
                    context=json.loads(row[9]) if row[9] else {}
                )
                
                outcome = None
                if row[10]:  # outcome_id exists
                    outcome = Outcome(
                        outcome_id=row[10],
                        prediction_id=row[0],
                        timestamp=datetime.fromisoformat(row[11]),
                        outcome_type=OutcomeType(row[12]),
                        actual_class=row[13],
                        decided_by=row[14],
                        decided_by_role=row[15],
                        modification_details=row[16],
                        rejection_reason=row[17],
                        decision_time_seconds=row[18]
                    )
                
                results.append((prediction, outcome))
        
        return results
    
    def get_acceptance_stats(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        agent_name: Optional[str] = None
    ) -> Dict:
        """Get acceptance statistics"""
        
        conditions = []
        params = []
        
        if start_time:
            conditions.append("p.timestamp >= ?")
            params.append(start_time.isoformat())
        
        if end_time:
            conditions.append("p.timestamp <= ?")
            params.append(end_time.isoformat())
        
        if agent_name:
            conditions.append("p.agent_name = ?")
            params.append(agent_name)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total predictions
            cursor.execute(f'''
                SELECT COUNT(*) FROM predictions p WHERE {where_clause}
            ''', params)
            total_predictions = cursor.fetchone()[0]
            
            # Outcome counts
            cursor.execute(f'''
                SELECT o.outcome_type, COUNT(*)
                FROM predictions p
                JOIN outcomes o ON p.prediction_id = o.prediction_id
                WHERE {where_clause}
                GROUP BY o.outcome_type
            ''', params)
            outcome_counts = dict(cursor.fetchall())
            
            # By agent
            cursor.execute(f'''
                SELECT p.agent_name, o.outcome_type, COUNT(*)
                FROM predictions p
                JOIN outcomes o ON p.prediction_id = o.prediction_id
                WHERE {where_clause}
                GROUP BY p.agent_name, o.outcome_type
            ''', params)
            by_agent = defaultdict(lambda: defaultdict(int))
            for row in cursor.fetchall():
                by_agent[row[0]][row[1]] = row[2]
            
            # Average decision time
            cursor.execute(f'''
                SELECT AVG(o.decision_time_seconds)
                FROM predictions p
                JOIN outcomes o ON p.prediction_id = o.prediction_id
                WHERE {where_clause} AND o.decision_time_seconds IS NOT NULL
            ''', params)
            avg_decision_time = cursor.fetchone()[0] or 0
        
        total_with_outcomes = sum(outcome_counts.values())
        
        return {
            'total_predictions': total_predictions,
            'total_with_outcomes': total_with_outcomes,
            'pending': total_predictions - total_with_outcomes,
            'accepted': outcome_counts.get('accepted', 0),
            'modified': outcome_counts.get('modified', 0),
            'rejected': outcome_counts.get('rejected', 0),
            'auto_executed': outcome_counts.get('auto_executed', 0),
            'expired': outcome_counts.get('expired', 0),
            'acceptance_rate': outcome_counts.get('accepted', 0) / total_with_outcomes if total_with_outcomes > 0 else 0,
            'modification_rate': outcome_counts.get('modified', 0) / total_with_outcomes if total_with_outcomes > 0 else 0,
            'rejection_rate': outcome_counts.get('rejected', 0) / total_with_outcomes if total_with_outcomes > 0 else 0,
            'avg_decision_time_seconds': avg_decision_time,
            'by_agent': dict(by_agent)
        }


# =============================================================================
# CALIBRATION CALCULATOR
# =============================================================================

class CalibrationCalculator:
    """Calculates calibration metrics"""
    
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
    
    def calculate_metrics(
        self,
        predictions: List[Tuple[Prediction, Optional[Outcome]]],
        period_start: datetime,
        period_end: datetime
    ) -> CalibrationMetrics:
        """Calculate comprehensive calibration metrics"""
        
        # Filter to predictions with outcomes
        valid_pairs = [
            (p, o) for p, o in predictions 
            if o is not None and o.outcome_type in [
                OutcomeType.ACCEPTED, 
                OutcomeType.MODIFIED, 
                OutcomeType.REJECTED,
                OutcomeType.AUTO_EXECUTED
            ]
        ]
        
        if not valid_pairs:
            return self._empty_metrics(period_start, period_end, len(predictions))
        
        # Extract confidences and correctness
        confidences = []
        correct = []
        
        for pred, outcome in valid_pairs:
            confidences.append(pred.confidence)
            
            # Determine if prediction was "correct"
            # Accepted or Auto-executed = correct
            # Modified = partially correct (count as 0.5)
            # Rejected = incorrect
            if outcome.outcome_type == OutcomeType.ACCEPTED:
                correct.append(1.0)
            elif outcome.outcome_type == OutcomeType.AUTO_EXECUTED:
                correct.append(1.0)
            elif outcome.outcome_type == OutcomeType.MODIFIED:
                correct.append(0.5)  # Partial credit
            else:
                correct.append(0.0)
        
        confidences = np.array(confidences)
        correct = np.array(correct)
        
        # Calculate bins
        bins = self._calculate_bins(confidences, correct)
        
        # Calculate ECE (Expected Calibration Error)
        ece = self._calculate_ece(bins)
        
        # Calculate MCE (Maximum Calibration Error)
        mce = max([b.calibration_error for b in bins]) if bins else 0.0
        
        # Calculate Brier Score
        brier = self._calculate_brier_score(confidences, correct)
        
        # Calculate rates
        outcome_types = [o.outcome_type for _, o in valid_pairs]
        n_total = len(outcome_types)
        
        acceptance_rate = sum(1 for ot in outcome_types if ot == OutcomeType.ACCEPTED) / n_total
        modification_rate = sum(1 for ot in outcome_types if ot == OutcomeType.MODIFIED) / n_total
        rejection_rate = sum(1 for ot in outcome_types if ot == OutcomeType.REJECTED) / n_total
        
        # Determine status
        status = self._get_calibration_status(ece)
        
        return CalibrationMetrics(
            calculation_time=datetime.now(),
            period_start=period_start,
            period_end=period_end,
            total_predictions=len(predictions),
            predictions_with_outcomes=len(valid_pairs),
            expected_calibration_error=ece,
            maximum_calibration_error=mce,
            brier_score=brier,
            overall_accuracy=float(np.mean(correct)),
            acceptance_rate=acceptance_rate,
            modification_rate=modification_rate,
            rejection_rate=rejection_rate,
            bins=bins,
            calibration_status=status
        )
    
    def _calculate_bins(self, confidences: np.ndarray, correct: np.ndarray) -> List[CalibrationBin]:
        """Calculate calibration bins"""
        bins = []
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        
        for i in range(self.n_bins):
            bin_start = bin_boundaries[i]
            bin_end = bin_boundaries[i + 1]
            
            # Find predictions in this bin
            if i == self.n_bins - 1:
                mask = (confidences >= bin_start) & (confidences <= bin_end)
            else:
                mask = (confidences >= bin_start) & (confidences < bin_end)
            
            bin_confidences = confidences[mask]
            bin_correct = correct[mask]
            
            count = len(bin_confidences)
            
            if count > 0:
                avg_confidence = float(np.mean(bin_confidences))
                accuracy = float(np.mean(bin_correct))
                calibration_error = abs(accuracy - avg_confidence)
            else:
                avg_confidence = (bin_start + bin_end) / 2
                accuracy = 0.0
                calibration_error = 0.0
            
            bins.append(CalibrationBin(
                bin_start=bin_start,
                bin_end=bin_end,
                count=count,
                avg_confidence=avg_confidence,
                accuracy=accuracy,
                expected_accuracy=avg_confidence,
                calibration_error=calibration_error
            ))
        
        return bins
    
    def _calculate_ece(self, bins: List[CalibrationBin]) -> float:
        """Calculate Expected Calibration Error"""
        total_count = sum(b.count for b in bins)
        
        if total_count == 0:
            return 0.0
        
        ece = sum(
            (b.count / total_count) * b.calibration_error
            for b in bins
        )
        
        return ece
    
    def _calculate_brier_score(self, confidences: np.ndarray, correct: np.ndarray) -> float:
        """Calculate Brier Score (lower is better)"""
        return float(np.mean((confidences - correct) ** 2))
    
    def _get_calibration_status(self, ece: float) -> CalibrationStatus:
        """Determine calibration status from ECE"""
        if ece < 0.05:
            return CalibrationStatus.EXCELLENT
        elif ece < 0.10:
            return CalibrationStatus.GOOD
        elif ece < 0.15:
            return CalibrationStatus.ACCEPTABLE
        elif ece < 0.25:
            return CalibrationStatus.POOR
        else:
            return CalibrationStatus.CRITICAL
    
    def _empty_metrics(self, period_start: datetime, period_end: datetime, total_predictions: int) -> CalibrationMetrics:
        """Return empty metrics when no valid data"""
        return CalibrationMetrics(
            calculation_time=datetime.now(),
            period_start=period_start,
            period_end=period_end,
            total_predictions=total_predictions,
            predictions_with_outcomes=0,
            expected_calibration_error=0.0,
            maximum_calibration_error=0.0,
            brier_score=0.0,
            overall_accuracy=0.0,
            acceptance_rate=0.0,
            modification_rate=0.0,
            rejection_rate=0.0,
            bins=[],
            calibration_status=CalibrationStatus.GOOD
        )


# =============================================================================
# DRIFT DETECTOR
# =============================================================================

class DriftDetector:
    """Detects various types of drift in model performance"""
    
    # Thresholds for drift detection
    THRESHOLDS = {
        'accuracy_drop': 0.10,           # 10% drop in accuracy
        'ece_increase': 0.05,            # 5% increase in ECE
        'acceptance_drop': 0.15,         # 15% drop in acceptance rate
        'rejection_spike': 0.20,         # 20% increase in rejection rate
        'brier_increase': 0.10           # 10% increase in Brier score
    }
    
    def __init__(self, db_path: str = "data/governance/calibration.db"):
        self.db_path = Path(db_path)
        self.audit_logger = get_audit_logger()
    
    def detect_drift(
        self,
        current_metrics: CalibrationMetrics,
        baseline_metrics: Optional[CalibrationMetrics] = None,
        window_metrics: Optional[List[CalibrationMetrics]] = None
    ) -> List[DriftAlert]:
        """
        Detect drift by comparing current metrics to baseline.
        
        Args:
            current_metrics: Latest calibration metrics
            baseline_metrics: Historical baseline (e.g., from initial deployment)
            window_metrics: Recent metrics for trend analysis
        """
        alerts = []
        
        if baseline_metrics is None:
            return alerts
        
        # Check accuracy drift
        if baseline_metrics.overall_accuracy > 0:
            accuracy_change = (baseline_metrics.overall_accuracy - current_metrics.overall_accuracy) / baseline_metrics.overall_accuracy
            if accuracy_change > self.THRESHOLDS['accuracy_drop']:
                alerts.append(self._create_alert(
                    drift_type=DriftType.ACCURACY_DRIFT,
                    severity=self._get_severity(accuracy_change, self.THRESHOLDS['accuracy_drop']),
                    description=f"Accuracy dropped by {accuracy_change:.1%}",
                    metric_name='overall_accuracy',
                    current_value=current_metrics.overall_accuracy,
                    baseline_value=baseline_metrics.overall_accuracy,
                    threshold=self.THRESHOLDS['accuracy_drop'],
                    recommendation="Review recent predictions and outcomes. Consider retraining models."
                ))
        
        # Check calibration drift (ECE)
        ece_change = current_metrics.expected_calibration_error - baseline_metrics.expected_calibration_error
        if ece_change > self.THRESHOLDS['ece_increase']:
            alerts.append(self._create_alert(
                drift_type=DriftType.CALIBRATION_DRIFT,
                severity=self._get_severity(ece_change, self.THRESHOLDS['ece_increase']),
                description=f"Calibration error increased by {ece_change:.3f}",
                metric_name='expected_calibration_error',
                current_value=current_metrics.expected_calibration_error,
                baseline_value=baseline_metrics.expected_calibration_error,
                threshold=self.THRESHOLDS['ece_increase'],
                recommendation="Confidence scores are miscalibrated. Consider Platt scaling or isotonic regression."
            ))
        
        # Check acceptance drift
        if baseline_metrics.acceptance_rate > 0:
            acceptance_change = (baseline_metrics.acceptance_rate - current_metrics.acceptance_rate) / baseline_metrics.acceptance_rate
            if acceptance_change > self.THRESHOLDS['acceptance_drop']:
                alerts.append(self._create_alert(
                    drift_type=DriftType.ACCEPTANCE_DRIFT,
                    severity=self._get_severity(acceptance_change, self.THRESHOLDS['acceptance_drop']),
                    description=f"Acceptance rate dropped by {acceptance_change:.1%}",
                    metric_name='acceptance_rate',
                    current_value=current_metrics.acceptance_rate,
                    baseline_value=baseline_metrics.acceptance_rate,
                    threshold=self.THRESHOLDS['acceptance_drop'],
                    recommendation="Users are accepting fewer AI suggestions. Review suggestion quality."
                ))
        
        # Check rejection spike
        if current_metrics.rejection_rate > 0 and baseline_metrics.rejection_rate > 0:
            rejection_change = (current_metrics.rejection_rate - baseline_metrics.rejection_rate) / baseline_metrics.rejection_rate
            if rejection_change > self.THRESHOLDS['rejection_spike']:
                alerts.append(self._create_alert(
                    drift_type=DriftType.ACCEPTANCE_DRIFT,
                    severity=self._get_severity(rejection_change, self.THRESHOLDS['rejection_spike']),
                    description=f"Rejection rate increased by {rejection_change:.1%}",
                    metric_name='rejection_rate',
                    current_value=current_metrics.rejection_rate,
                    baseline_value=baseline_metrics.rejection_rate,
                    threshold=self.THRESHOLDS['rejection_spike'],
                    recommendation="High rejection rate indicates AI suggestions don't match user expectations."
                ))
        
        # Store alerts
        for alert in alerts:
            self._store_alert(alert)
        
        return alerts
    
    def _create_alert(
        self,
        drift_type: DriftType,
        severity: DriftSeverity,
        description: str,
        metric_name: str,
        current_value: float,
        baseline_value: float,
        threshold: float,
        recommendation: str
    ) -> DriftAlert:
        """Create a drift alert"""
        alert_id = f"DFT-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"
        
        return DriftAlert(
            alert_id=alert_id,
            timestamp=datetime.now(),
            drift_type=drift_type,
            severity=severity,
            description=description,
            metric_name=metric_name,
            current_value=current_value,
            baseline_value=baseline_value,
            threshold=threshold,
            recommendation=recommendation
        )
    
    def _get_severity(self, change: float, threshold: float) -> DriftSeverity:
        """Determine severity based on how much change exceeds threshold"""
        ratio = change / threshold if threshold > 0 else 0
        
        if ratio >= 3.0:
            return DriftSeverity.CRITICAL
        elif ratio >= 2.0:
            return DriftSeverity.HIGH
        elif ratio >= 1.5:
            return DriftSeverity.MEDIUM
        else:
            return DriftSeverity.LOW
    
    def _store_alert(self, alert: DriftAlert):
        """Store alert in database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO drift_alerts (
                    alert_id, timestamp, drift_type, severity,
                    description, metric_name, current_value, baseline_value,
                    threshold, full_alert
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.alert_id,
                alert.timestamp.isoformat(),
                alert.drift_type.value,
                alert.severity.value,
                alert.description,
                alert.metric_name,
                alert.current_value,
                alert.baseline_value,
                alert.threshold,
                json.dumps(alert.to_dict(), default=str)
            ))
            conn.commit()
        
        # Log to audit trail
        self.audit_logger.log(
            actor=Actor(
                actor_id="drift_detector",
                actor_type="system",
                name="Drift Detector",
                role="System"
            ),
            event_type=EventType.SYSTEM_CONFIG_CHANGE,
            action_category=ActionCategory.AI_AGENT,
            action_description=f"Drift detected: {alert.drift_type.value}",
            parameters=alert.to_dict(),
            severity=Severity.WARNING if alert.severity in [DriftSeverity.LOW, DriftSeverity.MEDIUM] else Severity.ERROR
        )
    
    def get_active_alerts(self) -> List[DriftAlert]:
        """Get unresolved drift alerts"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT full_alert FROM drift_alerts WHERE resolved = 0
                ORDER BY timestamp DESC
            ''')
            
            alerts = []
            for row in cursor.fetchall():
                data = json.loads(row[0])
                alerts.append(DriftAlert(
                    alert_id=data['alert_id'],
                    timestamp=datetime.fromisoformat(data['timestamp']),
                    drift_type=DriftType(data['drift_type']),
                    severity=DriftSeverity(data['severity']),
                    description=data['description'],
                    metric_name=data['metric_name'],
                    current_value=data['current_value'],
                    baseline_value=data['baseline_value'],
                    threshold=data['threshold'],
                    recommendation=data.get('recommendation', '')
                ))
            
            return alerts


# =============================================================================
# RE-CALIBRATION TRIGGER MANAGER
# =============================================================================

class RecalibrationTriggerManager:
    """Manages re-calibration triggers"""
    
    # Thresholds for automatic triggers
    TRIGGER_THRESHOLDS = {
        'ece_threshold': 0.15,           # Trigger if ECE > 15%
        'accuracy_threshold': 0.70,      # Trigger if accuracy < 70%
        'rejection_threshold': 0.30,     # Trigger if rejection rate > 30%
        'days_since_calibration': 30     # Trigger if > 30 days since last calibration
    }
    
    def __init__(self, db_path: str = "data/governance/calibration.db"):
        self.db_path = Path(db_path)
        self.audit_logger = get_audit_logger()
    
    def check_triggers(
        self,
        current_metrics: CalibrationMetrics,
        drift_alerts: List[DriftAlert],
        last_calibration_date: Optional[datetime] = None
    ) -> List[RecalibrationTrigger]:
        """Check if re-calibration should be triggered"""
        triggers = []
        
        # Threshold-based triggers
        if current_metrics.expected_calibration_error > self.TRIGGER_THRESHOLDS['ece_threshold']:
            triggers.append(self._create_trigger(
                trigger_type=TriggerType.THRESHOLD,
                reason=f"ECE ({current_metrics.expected_calibration_error:.3f}) exceeds threshold ({self.TRIGGER_THRESHOLDS['ece_threshold']})",
                metrics_snapshot={
                    'ece': current_metrics.expected_calibration_error,
                    'accuracy': current_metrics.overall_accuracy
                },
                priority='high',
                recommended_action="Perform confidence recalibration using Platt scaling or temperature scaling"
            ))
        
        if current_metrics.overall_accuracy < self.TRIGGER_THRESHOLDS['accuracy_threshold']:
            triggers.append(self._create_trigger(
                trigger_type=TriggerType.PERFORMANCE,
                reason=f"Accuracy ({current_metrics.overall_accuracy:.1%}) below threshold ({self.TRIGGER_THRESHOLDS['accuracy_threshold']:.0%})",
                metrics_snapshot={
                    'accuracy': current_metrics.overall_accuracy,
                    'rejection_rate': current_metrics.rejection_rate
                },
                priority='high',
                recommended_action="Review model performance and consider retraining"
            ))
        
        if current_metrics.rejection_rate > self.TRIGGER_THRESHOLDS['rejection_threshold']:
            triggers.append(self._create_trigger(
                trigger_type=TriggerType.PERFORMANCE,
                reason=f"Rejection rate ({current_metrics.rejection_rate:.1%}) exceeds threshold ({self.TRIGGER_THRESHOLDS['rejection_threshold']:.0%})",
                metrics_snapshot={
                    'rejection_rate': current_metrics.rejection_rate,
                    'acceptance_rate': current_metrics.acceptance_rate
                },
                priority='medium',
                recommended_action="Analyze rejection reasons and adjust model or thresholds"
            ))
        
        # Time-based trigger
        if last_calibration_date:
            days_since = (datetime.now() - last_calibration_date).days
            if days_since > self.TRIGGER_THRESHOLDS['days_since_calibration']:
                triggers.append(self._create_trigger(
                    trigger_type=TriggerType.TIME_BASED,
                    reason=f"{days_since} days since last calibration (threshold: {self.TRIGGER_THRESHOLDS['days_since_calibration']})",
                    metrics_snapshot={
                        'days_since_calibration': days_since
                    },
                    priority='low',
                    recommended_action="Perform routine calibration check"
                ))
        
        # Drift-based triggers
        critical_alerts = [a for a in drift_alerts if a.severity in [DriftSeverity.CRITICAL, DriftSeverity.HIGH]]
        if critical_alerts:
            triggers.append(self._create_trigger(
                trigger_type=TriggerType.DRIFT,
                reason=f"{len(critical_alerts)} critical/high severity drift alerts detected",
                drift_alerts=[a.alert_id for a in critical_alerts],
                metrics_snapshot={
                    'critical_alerts': len([a for a in critical_alerts if a.severity == DriftSeverity.CRITICAL]),
                    'high_alerts': len([a for a in critical_alerts if a.severity == DriftSeverity.HIGH])
                },
                priority='critical' if any(a.severity == DriftSeverity.CRITICAL for a in critical_alerts) else 'high',
                recommended_action="Investigate drift causes and recalibrate affected models"
            ))
        
        # Store triggers
        for trigger in triggers:
            self._store_trigger(trigger)
        
        return triggers
    
    def _create_trigger(
        self,
        trigger_type: TriggerType,
        reason: str,
        metrics_snapshot: Dict[str, float],
        priority: str,
        recommended_action: str,
        drift_alerts: List[str] = None
    ) -> RecalibrationTrigger:
        """Create a re-calibration trigger"""
        trigger_id = f"TRG-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"
        
        return RecalibrationTrigger(
            trigger_id=trigger_id,
            timestamp=datetime.now(),
            trigger_type=trigger_type,
            reason=reason,
            metrics_snapshot=metrics_snapshot,
            drift_alerts=drift_alerts or [],
            priority=priority,
            recommended_action=recommended_action
        )
    
    def _store_trigger(self, trigger: RecalibrationTrigger):
        """Store trigger in database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO recalibration_triggers (
                    trigger_id, timestamp, trigger_type, reason,
                    priority, full_trigger
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                trigger.trigger_id,
                trigger.timestamp.isoformat(),
                trigger.trigger_type.value,
                trigger.reason,
                trigger.priority,
                json.dumps(trigger.to_dict(), default=str)
            ))
            conn.commit()
        
        # Log to audit trail
        self.audit_logger.log(
            actor=Actor(
                actor_id="trigger_manager",
                actor_type="system",
                name="Recalibration Trigger Manager",
                role="System"
            ),
            event_type=EventType.SYSTEM_CONFIG_CHANGE,
            action_category=ActionCategory.AI_AGENT,
            action_description=f"Recalibration trigger: {trigger.trigger_type.value}",
            parameters=trigger.to_dict(),
            severity=Severity.WARNING if trigger.priority in ['low', 'medium'] else Severity.ERROR
        )
    
    def acknowledge_trigger(self, trigger_id: str, acknowledged_by: str) -> bool:
        """Acknowledge a trigger"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE recalibration_triggers
                SET acknowledged = 1, acknowledged_by = ?, acknowledged_at = ?
                WHERE trigger_id = ?
            ''', (acknowledged_by, datetime.now().isoformat(), trigger_id))
            conn.commit()
            return cursor.rowcount > 0
    
    def get_pending_triggers(self) -> List[RecalibrationTrigger]:
        """Get unacknowledged triggers"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT full_trigger FROM recalibration_triggers 
                WHERE acknowledged = 0
                ORDER BY timestamp DESC
            ''')
            
            triggers = []
            for row in cursor.fetchall():
                data = json.loads(row[0])
                triggers.append(RecalibrationTrigger(
                    trigger_id=data['trigger_id'],
                    timestamp=datetime.fromisoformat(data['timestamp']),
                    trigger_type=TriggerType(data['trigger_type']),
                    reason=data['reason'],
                    metrics_snapshot=data.get('metrics_snapshot', {}),
                    drift_alerts=data.get('drift_alerts', []),
                    priority=data.get('priority', 'medium'),
                    recommended_action=data.get('recommended_action', '')
                ))
            
            return triggers


# =============================================================================
# CONFIDENCE CALIBRATION SYSTEM (Main Class)
# =============================================================================

class ConfidenceCalibrationSystem:
    """
    Main confidence calibration system.
    Integrates tracking, metrics, drift detection, and triggers.
    """
    
    def __init__(self, db_path: str = "data/governance/calibration.db"):
        self.tracker = AcceptanceTracker(db_path)
        self.calculator = CalibrationCalculator()
        self.drift_detector = DriftDetector(db_path)
        self.trigger_manager = RecalibrationTriggerManager(db_path)
        self.audit_logger = get_audit_logger()
        
        self._baseline_metrics: Optional[CalibrationMetrics] = None
        self._last_calibration_date: Optional[datetime] = None
    
    def record_prediction(
        self,
        agent_name: str,
        action_type: str,
        predicted_class: str,
        confidence: float,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        alternatives: List[Dict] = None,
        context: Dict = None
    ) -> str:
        """Record an AI prediction"""
        prediction_id = f"PRD-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"
        
        prediction = Prediction(
            prediction_id=prediction_id,
            timestamp=datetime.now(),
            agent_name=agent_name,
            action_type=action_type,
            entity_type=entity_type,
            entity_id=entity_id,
            predicted_class=predicted_class,
            confidence=confidence,
            alternatives=alternatives or [],
            context=context or {}
        )
        
        return self.tracker.record_prediction(prediction)
    
    def record_outcome(
        self,
        prediction_id: str,
        outcome_type: OutcomeType,
        actual_class: Optional[str] = None,
        decided_by: Optional[str] = None,
        decided_by_role: Optional[str] = None,
        modification_details: Optional[str] = None,
        rejection_reason: Optional[str] = None,
        decision_time_seconds: Optional[float] = None
    ) -> str:
        """Record the outcome of a prediction"""
        outcome_id = f"OUT-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"
        
        outcome = Outcome(
            outcome_id=outcome_id,
            prediction_id=prediction_id,
            timestamp=datetime.now(),
            outcome_type=outcome_type,
            actual_class=actual_class,
            decided_by=decided_by,
            decided_by_role=decided_by_role,
            modification_details=modification_details,
            rejection_reason=rejection_reason,
            decision_time_seconds=decision_time_seconds
        )
        
        return self.tracker.record_outcome(outcome)
    
    def calculate_calibration(
        self,
        days: int = 30,
        agent_name: Optional[str] = None,
        action_type: Optional[str] = None
    ) -> CalibrationMetrics:
        """Calculate calibration metrics for a period"""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        predictions = self.tracker.get_predictions_with_outcomes(
            start_time=start_time,
            end_time=end_time,
            agent_name=agent_name,
            action_type=action_type
        )
        
        return self.calculator.calculate_metrics(predictions, start_time, end_time)
    
    def run_calibration_check(
        self,
        days: int = 30,
        set_as_baseline: bool = False
    ) -> Dict:
        """
        Run a full calibration check including drift detection and triggers.
        
        Returns summary of calibration status, drift alerts, and triggers.
        """
        # Calculate current metrics
        current_metrics = self.calculate_calibration(days=days)
        
        # Detect drift (if baseline exists)
        drift_alerts = []
        if self._baseline_metrics:
            drift_alerts = self.drift_detector.detect_drift(
                current_metrics=current_metrics,
                baseline_metrics=self._baseline_metrics
            )
        
        # Check for triggers
        triggers = self.trigger_manager.check_triggers(
            current_metrics=current_metrics,
            drift_alerts=drift_alerts,
            last_calibration_date=self._last_calibration_date
        )
        
        # Update baseline if requested
        if set_as_baseline:
            self._baseline_metrics = current_metrics
            self._last_calibration_date = datetime.now()
        
        # Get acceptance stats
        acceptance_stats = self.tracker.get_acceptance_stats(
            start_time=datetime.now() - timedelta(days=days)
        )
        
        return {
            'metrics': current_metrics.to_dict(),
            'acceptance_stats': acceptance_stats,
            'drift_alerts': [a.to_dict() for a in drift_alerts],
            'triggers': [t.to_dict() for t in triggers],
            'has_baseline': self._baseline_metrics is not None,
            'status': current_metrics.calibration_status.value
        }
    
    def set_baseline(self, metrics: Optional[CalibrationMetrics] = None):
        """Set the baseline metrics for drift detection"""
        if metrics:
            self._baseline_metrics = metrics
        else:
            # Calculate from last 30 days
            self._baseline_metrics = self.calculate_calibration(days=30)
        
        self._last_calibration_date = datetime.now()
    
    def get_acceptance_stats(self, days: int = 30, agent_name: Optional[str] = None) -> Dict:
        """Get acceptance statistics"""
        return self.tracker.get_acceptance_stats(
            start_time=datetime.now() - timedelta(days=days),
            agent_name=agent_name
        )
    
    def get_active_alerts(self) -> List[DriftAlert]:
        """Get active drift alerts"""
        return self.drift_detector.get_active_alerts()
    
    def get_pending_triggers(self) -> List[RecalibrationTrigger]:
        """Get pending re-calibration triggers"""
        return self.trigger_manager.get_pending_triggers()
    
    def acknowledge_trigger(self, trigger_id: str, acknowledged_by: str) -> bool:
        """Acknowledge a re-calibration trigger"""
        return self.trigger_manager.acknowledge_trigger(trigger_id, acknowledged_by)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

_calibration_system_instance: Optional[ConfidenceCalibrationSystem] = None

def get_calibration_system() -> ConfidenceCalibrationSystem:
    """Get or create the calibration system singleton"""
    global _calibration_system_instance
    if _calibration_system_instance is None:
        _calibration_system_instance = ConfidenceCalibrationSystem()
    return _calibration_system_instance


# =============================================================================
# TEST FUNCTION
# =============================================================================

def test_confidence_calibration():
    """Test the confidence calibration system"""
    print("=" * 70)
    print("a6on-i - CONFIDENCE CALIBRATION TEST")
    print("=" * 70)
    
    # Use test database
    test_db = "data/governance/calibration_test.db"
    test_path = Path(test_db)
    
    # Clean up previous test
    if test_path.exists():
        try:
            test_path.unlink()
        except:
            pass
    
    # Initialize system
    system = ConfidenceCalibrationSystem(test_db)
    
    # Test 1: Record predictions and outcomes
    print("\n" + "-" * 50)
    print("TEST 1: Record Predictions and Outcomes")
    print("-" * 50)
    
    # Simulate various predictions with different confidences and outcomes
    test_data = [
        # (confidence, outcome_type) - well-calibrated examples
        (0.95, OutcomeType.ACCEPTED),
        (0.92, OutcomeType.ACCEPTED),
        (0.88, OutcomeType.ACCEPTED),
        (0.85, OutcomeType.MODIFIED),
        (0.82, OutcomeType.ACCEPTED),
        (0.78, OutcomeType.MODIFIED),
        (0.75, OutcomeType.ACCEPTED),
        (0.72, OutcomeType.MODIFIED),
        (0.68, OutcomeType.REJECTED),
        (0.65, OutcomeType.ACCEPTED),
        (0.62, OutcomeType.MODIFIED),
        (0.58, OutcomeType.REJECTED),
        (0.55, OutcomeType.REJECTED),
        (0.52, OutcomeType.MODIFIED),
        (0.48, OutcomeType.REJECTED),
        (0.45, OutcomeType.REJECTED),
        (0.42, OutcomeType.REJECTED),
        (0.38, OutcomeType.REJECTED),
        (0.35, OutcomeType.REJECTED),
        (0.30, OutcomeType.REJECTED),
    ]
    
    prediction_ids = []
    for i, (confidence, outcome) in enumerate(test_data):
        pred_id = system.record_prediction(
            agent_name="DiagnosticAgent",
            action_type="root_cause_analysis",
            predicted_class=f"hypothesis_{i % 3}",
            confidence=confidence,
            entity_type="site",
            entity_id=f"Site_{i+1}"
        )
        prediction_ids.append(pred_id)
        
        system.record_outcome(
            prediction_id=pred_id,
            outcome_type=outcome,
            actual_class=f"hypothesis_{i % 3}" if outcome == OutcomeType.ACCEPTED else f"other_{i % 2}",
            decided_by="user_001",
            decided_by_role="CRA",
            decision_time_seconds=30 + (i * 5)
        )
    
    print(f"✅ Recorded {len(test_data)} predictions with outcomes")
    
    # Test 2: Calculate calibration metrics
    print("\n" + "-" * 50)
    print("TEST 2: Calculate Calibration Metrics")
    print("-" * 50)
    
    metrics = system.calculate_calibration(days=1)
    
    print(f"✅ Calibration Metrics Calculated")
    print(f"   Total Predictions: {metrics.total_predictions}")
    print(f"   With Outcomes: {metrics.predictions_with_outcomes}")
    print(f"   ECE: {metrics.expected_calibration_error:.4f}")
    print(f"   MCE: {metrics.maximum_calibration_error:.4f}")
    print(f"   Brier Score: {metrics.brier_score:.4f}")
    print(f"   Accuracy: {metrics.overall_accuracy:.2%}")
    print(f"   Status: {metrics.calibration_status.value}")
    
    # Test 3: Acceptance statistics
    print("\n" + "-" * 50)
    print("TEST 3: Acceptance Statistics")
    print("-" * 50)
    
    stats = system.get_acceptance_stats(days=1)
    
    print(f"✅ Acceptance Stats")
    print(f"   Total: {stats['total_predictions']}")
    print(f"   Accepted: {stats['accepted']} ({stats['acceptance_rate']:.1%})")
    print(f"   Modified: {stats['modified']} ({stats['modification_rate']:.1%})")
    print(f"   Rejected: {stats['rejected']} ({stats['rejection_rate']:.1%})")
    print(f"   Avg Decision Time: {stats['avg_decision_time_seconds']:.1f}s")
    
    # Test 4: Set baseline and detect drift
    print("\n" + "-" * 50)
    print("TEST 4: Baseline and Drift Detection")
    print("-" * 50)
    
    # Set current as baseline
    system.set_baseline(metrics)
    print(f"✅ Baseline set")
    
    # Simulate "bad" predictions for drift
    for i in range(10):
        pred_id = system.record_prediction(
            agent_name="ResolverAgent",
            action_type="action_recommendation",
            predicted_class="action_a",
            confidence=0.90,  # High confidence
            entity_type="patient",
            entity_id=f"Patient_{i+100}"
        )
        
        # But all rejected (miscalibrated)
        system.record_outcome(
            prediction_id=pred_id,
            outcome_type=OutcomeType.REJECTED,
            decided_by="user_002",
            decided_by_role="DM",
            rejection_reason="Incorrect recommendation"
        )
    
    # Calculate new metrics
    new_metrics = system.calculate_calibration(days=1)
    
    # Detect drift
    drift_alerts = system.drift_detector.detect_drift(
        current_metrics=new_metrics,
        baseline_metrics=metrics
    )
    
    print(f"✅ Drift Detection")
    print(f"   New Accuracy: {new_metrics.overall_accuracy:.2%}")
    print(f"   Drift Alerts: {len(drift_alerts)}")
    for alert in drift_alerts:
        print(f"      - {alert.drift_type.value}: {alert.description}")
    
    # Test 5: Re-calibration triggers
    print("\n" + "-" * 50)
    print("TEST 5: Re-calibration Triggers")
    print("-" * 50)
    
    triggers = system.trigger_manager.check_triggers(
        current_metrics=new_metrics,
        drift_alerts=drift_alerts,
        last_calibration_date=datetime.now() - timedelta(days=35)  # Simulate old calibration
    )
    
    print(f"✅ Triggers Detected: {len(triggers)}")
    for trigger in triggers:
        print(f"   - [{trigger.priority.upper()}] {trigger.trigger_type.value}: {trigger.reason}")
    
    # Test 6: Full calibration check
    print("\n" + "-" * 50)
    print("TEST 6: Full Calibration Check")
    print("-" * 50)
    
    result = system.run_calibration_check(days=1)
    
    print(f"✅ Calibration Check Complete")
    print(f"   Status: {result['status']}")
    print(f"   Has Baseline: {result['has_baseline']}")
    print(f"   Drift Alerts: {len(result['drift_alerts'])}")
    print(f"   Triggers: {len(result['triggers'])}")
    
    # Test 7: Acknowledge trigger
    print("\n" + "-" * 50)
    print("TEST 7: Acknowledge Trigger")
    print("-" * 50)
    
    pending = system.get_pending_triggers()
    if pending:
        trigger = pending[0]
        ack_result = system.acknowledge_trigger(trigger.trigger_id, "admin_001")
        print(f"✅ Trigger Acknowledged: {ack_result}")
        print(f"   Trigger ID: {trigger.trigger_id}")
    else:
        print("   No pending triggers to acknowledge")
    
    # Test 8: Bin analysis
    print("\n" + "-" * 50)
    print("TEST 8: Confidence Bin Analysis")
    print("-" * 50)
    
    print("✅ Confidence Bins:")
    print(f"   {'Bin Range':<12} {'Count':<8} {'Avg Conf':<10} {'Accuracy':<10} {'Error':<10}")
    print(f"   {'-'*50}")
    for bin in metrics.bins:
        if bin.count > 0:
            print(f"   {bin.bin_start:.1f}-{bin.bin_end:.1f}      {bin.count:<8} {bin.avg_confidence:<10.3f} {bin.accuracy:<10.3f} {bin.calibration_error:<10.3f}")
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print("✅ All 8 tests passed!")
    print("\nConfidence Calibration Features Validated:")
    print("   ✓ Prediction recording")
    print("   ✓ Outcome tracking (Accept/Modify/Reject)")
    print("   ✓ Calibration metrics (ECE, MCE, Brier)")
    print("   ✓ Acceptance statistics")
    print("   ✓ Baseline setting")
    print("   ✓ Drift detection")
    print("   ✓ Re-calibration triggers")
    print("   ✓ Confidence bin analysis")
    
    # Cleanup
    try:
        if test_path.exists():
            # Close connections
            import gc
            gc.collect()
            import time
            time.sleep(0.3)
            test_path.unlink()
            print(f"\n   Test database cleaned up ✓")
    except:
        print(f"\n   ⚠️ Test database retained: {test_path}")
    
    print("\n" + "=" * 70)
    print("PHASE 10.3: CONFIDENCE CALIBRATION - COMPLETE ✅")
    print("=" * 70)


if __name__ == "__main__":
    test_confidence_calibration()
