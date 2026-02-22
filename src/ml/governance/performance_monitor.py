# src/ml/governance/performance_monitor.py
# NOTE: This module previously used SQLite but has been migrated to PostgreSQL-only.
# If you need to use this module, update it to use:
#   from src.database.pg_data_service import get_pg_data_service
#   service = get_pg_data_service()


"""
a6on-i - ML Performance Monitor v1.0
Real-time Performance Tracking for ML Models

Features:
- Prediction logging with ground truth tracking
- Rolling window metrics calculation
- Accuracy, F1, AUC tracking over time
- Threshold-based alerting
- Performance trending and decay detection
- 21 CFR Part 11 compliant logging
"""

import json
import uuid
# SQLite removed - using PostgreSQL
import threading
import numpy as np
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple, Union
from pathlib import Path
from collections import defaultdict

# Try sklearn imports for metrics
try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, brier_score_loss,
        mean_absolute_error, mean_squared_error, r2_score,
        confusion_matrix
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# =============================================================================
# ENUMS
# =============================================================================

class MetricType(Enum):
    """Types of performance metrics"""
    # Classification
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"
    AUC_ROC = "auc_roc"
    AUC_PR = "auc_pr"
    BRIER_SCORE = "brier_score"
    LOG_LOSS = "log_loss"
    
    # Regression
    MAE = "mae"
    RMSE = "rmse"
    R2 = "r2"
    MAPE = "mape"
    
    # Ranking
    NDCG_5 = "ndcg_5"
    NDCG_10 = "ndcg_10"
    MAP = "map"
    MRR = "mrr"
    
    # Volume
    PREDICTION_COUNT = "prediction_count"
    LABELED_COUNT = "labeled_count"
    LABEL_RATE = "label_rate"


class TimeWindow(Enum):
    """Time windows for metrics aggregation"""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


class AlertSeverity(Enum):
    """Severity levels for performance alerts"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of performance alerts"""
    THRESHOLD_BREACH = "threshold_breach"
    PERFORMANCE_DECAY = "performance_decay"
    DATA_VOLUME_LOW = "data_volume_low"
    LABEL_RATE_LOW = "label_rate_low"
    DRIFT_DETECTED = "drift_detected"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class PredictionLog:
    """Log entry for a single prediction"""
    log_id: str
    timestamp: datetime
    model_name: str
    model_version: str
    
    # Prediction info
    prediction_id: str        # Unique ID for this prediction
    features_hash: str        # Hash of input features
    predicted_class: str      # Predicted class/value
    confidence: float         # Prediction confidence (0-1)
    probabilities: Dict[str, float] = field(default_factory=dict)  # Class probabilities
    
    # Ground truth (may be added later)
    actual_class: Optional[str] = None
    labeled_at: Optional[datetime] = None
    labeled_by: Optional[str] = None
    
    # Metadata
    latency_ms: Optional[float] = None
    request_source: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'log_id': self.log_id,
            'timestamp': self.timestamp.isoformat(),
            'model_name': self.model_name,
            'model_version': self.model_version,
            'prediction_id': self.prediction_id,
            'features_hash': self.features_hash,
            'predicted_class': self.predicted_class,
            'confidence': self.confidence,
            'probabilities': self.probabilities,
            'actual_class': self.actual_class,
            'labeled_at': self.labeled_at.isoformat() if self.labeled_at else None,
            'labeled_by': self.labeled_by,
            'latency_ms': self.latency_ms,
            'request_source': self.request_source
        }


@dataclass
class PerformanceMetrics:
    """Aggregated performance metrics for a time window"""
    metrics_id: str
    timestamp: datetime
    model_name: str
    model_version: str
    window: TimeWindow
    window_start: datetime
    window_end: datetime
    
    # Volume
    prediction_count: int = 0
    labeled_count: int = 0
    label_rate: float = 0.0
    
    # Metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # Per-class metrics
    per_class_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Confusion matrix (as dict for JSON serialization)
    confusion_matrix: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict:
        return {
            'metrics_id': self.metrics_id,
            'timestamp': self.timestamp.isoformat(),
            'model_name': self.model_name,
            'model_version': self.model_version,
            'window': self.window.value,
            'window_start': self.window_start.isoformat(),
            'window_end': self.window_end.isoformat(),
            'prediction_count': self.prediction_count,
            'labeled_count': self.labeled_count,
            'label_rate': self.label_rate,
            'metrics': self.metrics,
            'per_class_metrics': self.per_class_metrics,
            'confusion_matrix': self.confusion_matrix
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PerformanceMetrics':
        return cls(
            metrics_id=data['metrics_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            model_name=data['model_name'],
            model_version=data['model_version'],
            window=TimeWindow(data['window']),
            window_start=datetime.fromisoformat(data['window_start']),
            window_end=datetime.fromisoformat(data['window_end']),
            prediction_count=data['prediction_count'],
            labeled_count=data['labeled_count'],
            label_rate=data['label_rate'],
            metrics=data['metrics'],
            per_class_metrics=data.get('per_class_metrics', {}),
            confusion_matrix=data.get('confusion_matrix')
        )


@dataclass
class MetricThreshold:
    """Threshold configuration for a metric"""
    threshold_id: str
    model_name: str
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    comparison: str = "less_than"  # less_than, greater_than
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            'threshold_id': self.threshold_id,
            'model_name': self.model_name,
            'metric_name': self.metric_name,
            'warning_threshold': self.warning_threshold,
            'critical_threshold': self.critical_threshold,
            'comparison': self.comparison,
            'enabled': self.enabled,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class PerformanceAlert:
    """Performance alert"""
    alert_id: str
    timestamp: datetime
    model_name: str
    model_version: str
    
    alert_type: AlertType
    severity: AlertSeverity
    
    metric_name: str
    current_value: float
    threshold_value: float
    baseline_value: Optional[float] = None
    
    description: str = ""
    recommendation: str = ""
    
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'alert_id': self.alert_id,
            'timestamp': self.timestamp.isoformat(),
            'model_name': self.model_name,
            'model_version': self.model_version,
            'alert_type': self.alert_type.value,
            'severity': self.severity.value,
            'metric_name': self.metric_name,
            'current_value': self.current_value,
            'threshold_value': self.threshold_value,
            'baseline_value': self.baseline_value,
            'description': self.description,
            'recommendation': self.recommendation,
            'resolved': self.resolved,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'resolved_by': self.resolved_by
        }


# =============================================================================
# PERFORMANCE MONITOR
# =============================================================================

class PerformanceMonitor:
    """
    Real-time performance monitoring for ML models.
    
    Features:
    - Log predictions and ground truth
    - Calculate rolling window metrics
    - Threshold-based alerting
    - Performance trending
    - Decay detection
    """
    
    # Default thresholds (metric_name -> (warning, critical, comparison))
    DEFAULT_THRESHOLDS = {
        'accuracy': (0.85, 0.75, 'less_than'),
        'f1': (0.80, 0.70, 'less_than'),
        'auc_roc': (0.85, 0.75, 'less_than'),
        'precision': (0.80, 0.70, 'less_than'),
        'recall': (0.80, 0.70, 'less_than'),
        'label_rate': (0.10, 0.05, 'less_than'),
    }
    
    def __init__(self, db_path: str = "data/ml_governance/performance.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database"""
        # DISABLED: SQLite replaced with PostgreSQL
        if False:  # Disabled method
            cursor = conn.cursor()
            
            # Prediction logs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS prediction_logs (
                    log_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    prediction_id TEXT NOT NULL UNIQUE,
                    features_hash TEXT,
                    predicted_class TEXT NOT NULL,
                    confidence REAL,
                    probabilities TEXT,
                    actual_class TEXT,
                    labeled_at TEXT,
                    labeled_by TEXT,
                    latency_ms REAL,
                    request_source TEXT
                )
            ''')
            
            # Performance metrics snapshots
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    metrics_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    window TEXT NOT NULL,
                    window_start TEXT NOT NULL,
                    window_end TEXT NOT NULL,
                    metrics_data TEXT NOT NULL
                )
            ''')
            
            # Metric thresholds
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metric_thresholds (
                    threshold_id TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    warning_threshold REAL NOT NULL,
                    critical_threshold REAL NOT NULL,
                    comparison TEXT NOT NULL,
                    enabled INTEGER DEFAULT 1,
                    created_at TEXT NOT NULL,
                    UNIQUE(model_name, metric_name)
                )
            ''')
            
            # Performance alerts
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_alerts (
                    alert_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    alert_data TEXT NOT NULL,
                    resolved INTEGER DEFAULT 0,
                    resolved_at TEXT,
                    resolved_by TEXT
                )
            ''')
            
            # Indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_pred_model ON prediction_logs(model_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_pred_time ON prediction_logs(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_pred_id ON prediction_logs(prediction_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_model ON performance_metrics(model_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_window ON performance_metrics(window_start)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alert_model ON performance_alerts(model_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alert_resolved ON performance_alerts(resolved)')
            
            conn.commit()
    
    # =========================================================================
    # Prediction Logging
    # =========================================================================
    
    def log_prediction(
        self,
        model_name: str,
        model_version: str,
        prediction_id: str,
        predicted_class: str,
        confidence: float,
        features_hash: Optional[str] = None,
        probabilities: Optional[Dict[str, float]] = None,
        latency_ms: Optional[float] = None,
        request_source: Optional[str] = None
    ) -> str:
        """
        Log a model prediction.
        
        Args:
            model_name: Name of the model
            model_version: Version of the model
            prediction_id: Unique ID for this prediction
            predicted_class: The predicted class/value
            confidence: Confidence score (0-1)
            features_hash: Optional hash of input features
            probabilities: Optional class probabilities
            latency_ms: Optional inference latency
            request_source: Optional source of the request
            
        Returns:
            Log ID
        """
        log_id = str(uuid.uuid4())
        now = datetime.now()
        
        with self._lock:
            # DISABLED: SQLite replaced with PostgreSQL
            if False:  # Disabled method
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO prediction_logs 
                        (log_id, timestamp, model_name, model_version, prediction_id,
                         features_hash, predicted_class, confidence, probabilities,
                         latency_ms, request_source)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        log_id,
                        now.isoformat(),
                        model_name,
                        model_version,
                        prediction_id,
                        features_hash,
                        predicted_class,
                        confidence,
                        json.dumps(probabilities) if probabilities else None,
                        latency_ms,
                        request_source
                    ))
                    conn.commit()
        
        return log_id
    
    def log_ground_truth(
        self,
        prediction_id: str,
        actual_class: str,
        labeled_by: Optional[str] = None
    ) -> bool:
        """
        Record ground truth for a prediction.
        
        Args:
            prediction_id: ID of the prediction
            actual_class: The actual/correct class
            labeled_by: Who provided the label
            
        Returns:
            True if successful
        """
        now = datetime.now()
        
        with self._lock:
            # DISABLED: SQLite replaced with PostgreSQL
            if False:  # Disabled method
                    cursor = conn.cursor()
                    cursor.execute('''
                        UPDATE prediction_logs
                        SET actual_class = ?, labeled_at = ?, labeled_by = ?
                        WHERE prediction_id = ?
                    ''', (actual_class, now.isoformat(), labeled_by, prediction_id))
                    conn.commit()
                    return cursor.rowcount > 0
    
    def log_batch_predictions(
        self,
        model_name: str,
        model_version: str,
        predictions: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Log multiple predictions at once.
        
        Args:
            model_name: Name of the model
            model_version: Version of the model
            predictions: List of prediction dicts with keys:
                        prediction_id, predicted_class, confidence, etc.
                        
        Returns:
            List of log IDs
        """
        log_ids = []
        now = datetime.now()
        
        with self._lock:
            # DISABLED: SQLite replaced with PostgreSQL
            if False:  # Disabled method
                    cursor = conn.cursor()
                
                    for pred in predictions:
                        log_id = str(uuid.uuid4())
                        cursor.execute('''
                            INSERT INTO prediction_logs 
                            (log_id, timestamp, model_name, model_version, prediction_id,
                             features_hash, predicted_class, confidence, probabilities,
                             latency_ms, request_source, actual_class)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            log_id,
                            now.isoformat(),
                            model_name,
                            model_version,
                            pred['prediction_id'],
                            pred.get('features_hash'),
                            pred['predicted_class'],
                            pred.get('confidence', 0.0),
                            json.dumps(pred.get('probabilities')) if pred.get('probabilities') else None,
                            pred.get('latency_ms'),
                            pred.get('request_source'),
                            pred.get('actual_class')  # Allow setting ground truth in batch
                        ))
                        log_ids.append(log_id)
                
                    conn.commit()
        
        return log_ids
    
    # =========================================================================
    # Metrics Calculation
    # =========================================================================
    
    def calculate_metrics(
        self,
        model_name: str,
        model_version: Optional[str] = None,
        window: TimeWindow = TimeWindow.DAILY,
        window_end: Optional[datetime] = None
    ) -> PerformanceMetrics:
        """
        Calculate performance metrics for a time window.
        
        Args:
            model_name: Name of the model
            model_version: Specific version or None for all versions
            window: Time window for aggregation
            window_end: End of window (default: now)
            
        Returns:
            PerformanceMetrics object
        """
        if window_end is None:
            window_end = datetime.now()
        
        # Calculate window start
        if window == TimeWindow.HOURLY:
            window_start = window_end - timedelta(hours=1)
        elif window == TimeWindow.DAILY:
            window_start = window_end - timedelta(days=1)
        elif window == TimeWindow.WEEKLY:
            window_start = window_end - timedelta(weeks=1)
        else:  # MONTHLY
            window_start = window_end - timedelta(days=30)
        
        # Get predictions with labels
        predictions = self._get_labeled_predictions(
            model_name, model_version, window_start, window_end
        )
        
        # Get all predictions (for volume)
        all_predictions = self._get_all_predictions(
            model_name, model_version, window_start, window_end
        )
        
        prediction_count = len(all_predictions)
        labeled_count = len(predictions)
        label_rate = labeled_count / prediction_count if prediction_count > 0 else 0.0
        
        # Calculate metrics
        metrics = {}
        per_class_metrics = {}
        conf_matrix = None
        
        if labeled_count > 0:
            y_true = [p['actual_class'] for p in predictions]
            y_pred = [p['predicted_class'] for p in predictions]
            
            # Get unique classes
            classes = sorted(set(y_true) | set(y_pred))
            
            if SKLEARN_AVAILABLE:
                # Basic metrics
                metrics['accuracy'] = accuracy_score(y_true, y_pred)
                
                # Handle multi-class metrics
                if len(classes) == 2:
                    # Binary classification
                    metrics['precision'] = precision_score(y_true, y_pred, pos_label=classes[1], zero_division=0)
                    metrics['recall'] = recall_score(y_true, y_pred, pos_label=classes[1], zero_division=0)
                    metrics['f1'] = f1_score(y_true, y_pred, pos_label=classes[1], zero_division=0)
                    
                    # ROC AUC (if probabilities available)
                    if all('confidence' in p for p in predictions):
                        confidences = [p['confidence'] for p in predictions]
                        y_true_binary = [1 if y == classes[1] else 0 for y in y_true]
                        try:
                            metrics['auc_roc'] = roc_auc_score(y_true_binary, confidences)
                        except ValueError:
                            pass
                else:
                    # Multi-class classification
                    metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                    metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                    metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                    
                    # Per-class metrics
                    for cls in classes:
                        y_true_cls = [1 if y == cls else 0 for y in y_true]
                        y_pred_cls = [1 if y == cls else 0 for y in y_pred]
                        
                        per_class_metrics[str(cls)] = {
                            'precision': precision_score(y_true_cls, y_pred_cls, zero_division=0),
                            'recall': recall_score(y_true_cls, y_pred_cls, zero_division=0),
                            'f1': f1_score(y_true_cls, y_pred_cls, zero_division=0),
                            'support': sum(y_true_cls)
                        }
                
                # Confusion matrix
                cm = confusion_matrix(y_true, y_pred, labels=classes)
                conf_matrix = {
                    'labels': [str(c) for c in classes],
                    'matrix': cm.tolist()
                }
            else:
                # Fallback: manual calculation
                correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
                metrics['accuracy'] = correct / len(y_true)
        
        # Add volume metrics
        metrics['prediction_count'] = prediction_count
        metrics['labeled_count'] = labeled_count
        metrics['label_rate'] = label_rate
        
        # Create metrics object
        perf_metrics = PerformanceMetrics(
            metrics_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            model_name=model_name,
            model_version=model_version or "all",
            window=window,
            window_start=window_start,
            window_end=window_end,
            prediction_count=prediction_count,
            labeled_count=labeled_count,
            label_rate=label_rate,
            metrics=metrics,
            per_class_metrics=per_class_metrics,
            confusion_matrix=conf_matrix
        )
        
        # Save snapshot
        self._save_metrics_snapshot(perf_metrics)
        
        return perf_metrics
    
    def _get_labeled_predictions(
        self,
        model_name: str,
        model_version: Optional[str],
        start: datetime,
        end: datetime
    ) -> List[Dict]:
        """Get predictions with ground truth labels"""
        predictions = []
        
        # DISABLED: SQLite replaced with PostgreSQL
        if False:  # Disabled method
            cursor = conn.cursor()
            
            query = '''
                SELECT prediction_id, predicted_class, confidence, actual_class,
                       probabilities
                FROM prediction_logs
                WHERE model_name = ? AND timestamp >= ? AND timestamp <= ?
                AND actual_class IS NOT NULL
            '''
            params = [model_name, start.isoformat(), end.isoformat()]
            
            if model_version:
                query += ' AND model_version = ?'
                params.append(model_version)
            
            cursor.execute(query, params)
            
            for row in cursor.fetchall():
                predictions.append({
                    'prediction_id': row[0],
                    'predicted_class': row[1],
                    'confidence': row[2],
                    'actual_class': row[3],
                    'probabilities': json.loads(row[4]) if row[4] else None
                })
        
        return predictions
    
    def _get_all_predictions(
        self,
        model_name: str,
        model_version: Optional[str],
        start: datetime,
        end: datetime
    ) -> List[Dict]:
        """Get all predictions (with or without labels)"""
        predictions = []
        
        # DISABLED: SQLite replaced with PostgreSQL
        if False:  # Disabled method
            cursor = conn.cursor()
            
            query = '''
                SELECT prediction_id, predicted_class, confidence, actual_class
                FROM prediction_logs
                WHERE model_name = ? AND timestamp >= ? AND timestamp <= ?
            '''
            params = [model_name, start.isoformat(), end.isoformat()]
            
            if model_version:
                query += ' AND model_version = ?'
                params.append(model_version)
            
            cursor.execute(query, params)
            
            for row in cursor.fetchall():
                predictions.append({
                    'prediction_id': row[0],
                    'predicted_class': row[1],
                    'confidence': row[2],
                    'actual_class': row[3]
                })
        
        return predictions
    
    def _save_metrics_snapshot(self, metrics: PerformanceMetrics):
        """Save metrics snapshot to database"""
        with self._lock:
            # DISABLED: SQLite replaced with PostgreSQL
            if False:  # Disabled method
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO performance_metrics
                        (metrics_id, timestamp, model_name, model_version,
                         window, window_start, window_end, metrics_data)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        metrics.metrics_id,
                        metrics.timestamp.isoformat(),
                        metrics.model_name,
                        metrics.model_version,
                        metrics.window.value,
                        metrics.window_start.isoformat(),
                        metrics.window_end.isoformat(),
                        json.dumps(metrics.to_dict())
                    ))
                    conn.commit()
    
    # =========================================================================
    # Metrics History
    # =========================================================================
    
    def get_metrics_history(
        self,
        model_name: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        window: Optional[TimeWindow] = None
    ) -> List[PerformanceMetrics]:
        """Get historical metrics snapshots"""
        metrics_list = []
        
        # DISABLED: SQLite replaced with PostgreSQL
        if False:  # Disabled method
            cursor = conn.cursor()
            
            query = 'SELECT metrics_data FROM performance_metrics WHERE model_name = ?'
            params = [model_name]
            
            if start:
                query += ' AND window_start >= ?'
                params.append(start.isoformat())
            
            if end:
                query += ' AND window_end <= ?'
                params.append(end.isoformat())
            
            if window:
                query += ' AND window = ?'
                params.append(window.value)
            
            query += ' ORDER BY window_start DESC'
            
            cursor.execute(query, params)
            
            for row in cursor.fetchall():
                data = json.loads(row[0])
                metrics_list.append(PerformanceMetrics.from_dict(data))
        
        return metrics_list
    
    def get_performance_trend(
        self,
        model_name: str,
        metric_name: str,
        days: int = 30
    ) -> List[Tuple[datetime, float]]:
        """Get trend data for a specific metric"""
        start = datetime.now() - timedelta(days=days)
        history = self.get_metrics_history(model_name, start=start, window=TimeWindow.DAILY)
        
        trend = []
        for metrics in history:
            value = metrics.metrics.get(metric_name)
            if value is not None:
                trend.append((metrics.window_start, value))
        
        # Sort by date
        trend.sort(key=lambda x: x[0])
        return trend
    
    # =========================================================================
    # Thresholds and Alerts
    # =========================================================================
    
    def set_threshold(
        self,
        model_name: str,
        metric_name: str,
        warning_threshold: float,
        critical_threshold: float,
        comparison: str = "less_than"
    ) -> MetricThreshold:
        """Set threshold for a metric"""
        threshold_id = str(uuid.uuid4())
        
        threshold = MetricThreshold(
            threshold_id=threshold_id,
            model_name=model_name,
            metric_name=metric_name,
            warning_threshold=warning_threshold,
            critical_threshold=critical_threshold,
            comparison=comparison
        )
        
        with self._lock:
            # DISABLED: SQLite replaced with PostgreSQL
            if False:  # Disabled method
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT OR REPLACE INTO metric_thresholds
                        (threshold_id, model_name, metric_name, warning_threshold,
                         critical_threshold, comparison, enabled, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, 1, ?)
                    ''', (
                        threshold_id,
                        model_name,
                        metric_name,
                        warning_threshold,
                        critical_threshold,
                        comparison,
                        datetime.now().isoformat()
                    ))
                    conn.commit()
        
        return threshold
    
    def get_thresholds(self, model_name: str) -> List[MetricThreshold]:
        """Get all thresholds for a model"""
        thresholds = []
        
        # DISABLED: SQLite replaced with PostgreSQL
        if False:  # Disabled method
            cursor = conn.cursor()
            cursor.execute('''
                SELECT threshold_id, model_name, metric_name, warning_threshold,
                       critical_threshold, comparison, enabled, created_at
                FROM metric_thresholds WHERE model_name = ?
            ''', (model_name,))
            
            for row in cursor.fetchall():
                thresholds.append(MetricThreshold(
                    threshold_id=row[0],
                    model_name=row[1],
                    metric_name=row[2],
                    warning_threshold=row[3],
                    critical_threshold=row[4],
                    comparison=row[5],
                    enabled=bool(row[6]),
                    created_at=datetime.fromisoformat(row[7])
                ))
        
        return thresholds
    
    def check_thresholds(
        self,
        model_name: str,
        model_version: Optional[str] = None
    ) -> List[PerformanceAlert]:
        """Check current metrics against thresholds and create alerts"""
        # Get current metrics
        metrics = self.calculate_metrics(model_name, model_version)
        
        # Get thresholds
        thresholds = self.get_thresholds(model_name)
        
        # If no custom thresholds, use defaults
        if not thresholds:
            for metric_name, (warn, crit, comp) in self.DEFAULT_THRESHOLDS.items():
                thresholds.append(MetricThreshold(
                    threshold_id='default',
                    model_name=model_name,
                    metric_name=metric_name,
                    warning_threshold=warn,
                    critical_threshold=crit,
                    comparison=comp
                ))
        
        alerts = []
        
        for threshold in thresholds:
            if not threshold.enabled:
                continue
            
            value = metrics.metrics.get(threshold.metric_name)
            if value is None:
                continue
            
            # Check threshold
            breached = False
            severity = None
            
            if threshold.comparison == 'less_than':
                if value < threshold.critical_threshold:
                    breached = True
                    severity = AlertSeverity.CRITICAL
                elif value < threshold.warning_threshold:
                    breached = True
                    severity = AlertSeverity.WARNING
            else:  # greater_than
                if value > threshold.critical_threshold:
                    breached = True
                    severity = AlertSeverity.CRITICAL
                elif value > threshold.warning_threshold:
                    breached = True
                    severity = AlertSeverity.WARNING
            
            if breached:
                alert = PerformanceAlert(
                    alert_id=str(uuid.uuid4()),
                    timestamp=datetime.now(),
                    model_name=model_name,
                    model_version=model_version or "all",
                    alert_type=AlertType.THRESHOLD_BREACH,
                    severity=severity,
                    metric_name=threshold.metric_name,
                    current_value=value,
                    threshold_value=threshold.warning_threshold if severity == AlertSeverity.WARNING else threshold.critical_threshold,
                    description=f"{threshold.metric_name} = {value:.4f} is {threshold.comparison.replace('_', ' ')} threshold",
                    recommendation=self._get_alert_recommendation(threshold.metric_name, severity)
                )
                
                alerts.append(alert)
                self._save_alert(alert)
        
        return alerts
    
    def _get_alert_recommendation(self, metric_name: str, severity: AlertSeverity) -> str:
        """Get recommendation based on metric and severity"""
        recommendations = {
            'accuracy': "Review recent predictions for errors. Consider retraining with more recent data.",
            'f1': "Check for class imbalance issues. Review precision/recall trade-offs.",
            'auc_roc': "Model discrimination is degrading. Investigate feature drift or concept drift.",
            'precision': "Too many false positives. Consider adjusting classification threshold.",
            'recall': "Too many false negatives. Consider adjusting classification threshold.",
            'label_rate': "Not enough ground truth labels for reliable monitoring. Increase labeling efforts."
        }
        
        rec = recommendations.get(metric_name, "Investigate the metric degradation.")
        
        if severity == AlertSeverity.CRITICAL:
            rec = "URGENT: " + rec + " Consider model rollback if available."
        
        return rec
    
    def _save_alert(self, alert: PerformanceAlert):
        """Save alert to database"""
        with self._lock:
            # DISABLED: SQLite replaced with PostgreSQL
            if False:  # Disabled method
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO performance_alerts
                        (alert_id, timestamp, model_name, model_version,
                         alert_type, severity, alert_data)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        alert.alert_id,
                        alert.timestamp.isoformat(),
                        alert.model_name,
                        alert.model_version,
                        alert.alert_type.value,
                        alert.severity.value,
                        json.dumps(alert.to_dict())
                    ))
                    conn.commit()
    
    def get_active_alerts(
        self,
        model_name: Optional[str] = None
    ) -> List[PerformanceAlert]:
        """Get unresolved alerts"""
        alerts = []
        
        # DISABLED: SQLite replaced with PostgreSQL
        if False:  # Disabled method
            cursor = conn.cursor()
            
            if model_name:
                cursor.execute('''
                    SELECT alert_data FROM performance_alerts
                    WHERE model_name = ? AND resolved = 0
                    ORDER BY timestamp DESC
                ''', (model_name,))
            else:
                cursor.execute('''
                    SELECT alert_data FROM performance_alerts
                    WHERE resolved = 0
                    ORDER BY timestamp DESC
                ''')
            
            for row in cursor.fetchall():
                data = json.loads(row[0])
                alert = PerformanceAlert(
                    alert_id=data['alert_id'],
                    timestamp=datetime.fromisoformat(data['timestamp']),
                    model_name=data['model_name'],
                    model_version=data['model_version'],
                    alert_type=AlertType(data['alert_type']),
                    severity=AlertSeverity(data['severity']),
                    metric_name=data['metric_name'],
                    current_value=data['current_value'],
                    threshold_value=data['threshold_value'],
                    baseline_value=data.get('baseline_value'),
                    description=data['description'],
                    recommendation=data['recommendation']
                )
                alerts.append(alert)
        
        return alerts
    
    def resolve_alert(self, alert_id: str, resolved_by: str) -> bool:
        """Mark an alert as resolved"""
        with self._lock:
            # DISABLED: SQLite replaced with PostgreSQL
            if False:  # Disabled method
                    cursor = conn.cursor()
                    cursor.execute('''
                        UPDATE performance_alerts
                        SET resolved = 1, resolved_at = ?, resolved_by = ?
                        WHERE alert_id = ?
                    ''', (datetime.now().isoformat(), resolved_by, alert_id))
                    conn.commit()
                    return cursor.rowcount > 0
    
    # =========================================================================
    # Performance Decay Detection
    # =========================================================================
    
    def detect_performance_decay(
        self,
        model_name: str,
        metric_name: str = 'accuracy',
        baseline_window_days: int = 30,
        current_window_days: int = 7,
        decay_threshold: float = 0.05
    ) -> Optional[PerformanceAlert]:
        """
        Detect performance decay by comparing current window to baseline.
        
        Args:
            model_name: Name of the model
            metric_name: Metric to check
            baseline_window_days: Days for baseline window
            current_window_days: Days for current window
            decay_threshold: Relative decay threshold (0.05 = 5%)
            
        Returns:
            PerformanceAlert if decay detected, None otherwise
        """
        now = datetime.now()
        
        # Get baseline metrics
        baseline_start = now - timedelta(days=baseline_window_days)
        baseline_end = now - timedelta(days=current_window_days)
        
        baseline_history = self.get_metrics_history(
            model_name,
            start=baseline_start,
            end=baseline_end,
            window=TimeWindow.DAILY
        )
        
        if not baseline_history:
            return None
        
        baseline_values = [
            m.metrics.get(metric_name) for m in baseline_history
            if m.metrics.get(metric_name) is not None
        ]
        
        if not baseline_values:
            return None
        
        baseline_avg = np.mean(baseline_values)
        
        # Get current metrics
        current_metrics = self.calculate_metrics(
            model_name,
            window=TimeWindow.WEEKLY if current_window_days >= 7 else TimeWindow.DAILY
        )
        
        current_value = current_metrics.metrics.get(metric_name)
        if current_value is None:
            return None
        
        # Calculate decay
        decay = (baseline_avg - current_value) / baseline_avg if baseline_avg > 0 else 0
        
        if decay >= decay_threshold:
            severity = AlertSeverity.CRITICAL if decay >= 0.10 else AlertSeverity.WARNING
            
            alert = PerformanceAlert(
                alert_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                model_name=model_name,
                model_version=current_metrics.model_version,
                alert_type=AlertType.PERFORMANCE_DECAY,
                severity=severity,
                metric_name=metric_name,
                current_value=current_value,
                threshold_value=decay_threshold,
                baseline_value=baseline_avg,
                description=f"{metric_name} decayed by {decay*100:.1f}% from baseline {baseline_avg:.4f} to {current_value:.4f}",
                recommendation="Model performance is degrading. Consider retraining or investigating data drift."
            )
            
            self._save_alert(alert)
            return alert
        
        return None
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get monitor statistics"""
        # Initialize default values to avoid UnboundLocalError
        total_predictions = 0
        total_labeled = 0
        by_model = {}
        active_alerts = 0
        total_snapshots = 0
        
        # DISABLED: SQLite replaced with PostgreSQL
        if False:  # Disabled method
            cursor = conn.cursor()
            
            # Prediction counts
            cursor.execute('SELECT COUNT(*) FROM prediction_logs')
            total_predictions = cursor.fetchone()[0]
            
            # Labeled counts
            cursor.execute('SELECT COUNT(*) FROM prediction_logs WHERE actual_class IS NOT NULL')
            total_labeled = cursor.fetchone()[0]
            
            # By model
            cursor.execute('''
                SELECT model_name, COUNT(*) FROM prediction_logs GROUP BY model_name
            ''')
            by_model = dict(cursor.fetchall())
            
            # Active alerts
            cursor.execute('SELECT COUNT(*) FROM performance_alerts WHERE resolved = 0')
            active_alerts = cursor.fetchone()[0]
            
            # Metrics snapshots
            cursor.execute('SELECT COUNT(*) FROM performance_metrics')
            total_snapshots = cursor.fetchone()[0]
        
        return {
            'total_predictions': total_predictions,
            'total_labeled': total_labeled,
            'label_rate': total_labeled / total_predictions if total_predictions > 0 else 0,
            'predictions_by_model': by_model,
            'active_alerts': active_alerts,
            'total_metrics_snapshots': total_snapshots
        }


# =============================================================================
# SINGLETON
# =============================================================================

_performance_monitor: Optional[PerformanceMonitor] = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get or create the performance monitor singleton"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


def reset_performance_monitor():
    """Reset the performance monitor singleton (for testing)"""
    global _performance_monitor
    _performance_monitor = None
