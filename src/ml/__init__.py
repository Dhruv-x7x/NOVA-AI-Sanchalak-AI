"""
a6on-i - ML Module
Phase 3: Intelligence Engines

Components:
- 3.1 Data Preparation
- 3.2 Risk Classifier
- 3.3 Issue Detector
- 3.4 Pattern Library
- 3.5 Resolution Genome
- 3.6 Model Loader

Layer 8: Model Governance
- 8.1 Drift Detector
- 8.2 Model Registry
- 8.3 Performance Monitor
- 8.4 Retraining Trigger
- 8.5 ML Audit Trail
"""

from .data_preparation import MLDataPreparator
from .risk_classifier import RiskClassifier

# ML Governance (Layer 8)
try:
    from .governance import (
        # Drift Detector
        DriftDetector, DriftType, DriftSeverity, DriftReport, DriftAlert,
        get_drift_detector,
        # Model Registry
        ModelRegistry, ModelStatus, ModelVersion, ModelMetrics,
        get_model_registry,
        # Performance Monitor
        PerformanceMonitor, PerformanceMetrics, PerformanceAlert,
        get_performance_monitor,
        # Retraining Trigger
        RetrainingTrigger, RetrainingJob, TriggerType, RetrainingStatus,
        get_retraining_trigger,
        # ML Audit Trail
        MLAuditLogger, MLAuditEntry, MLEventType, MLActor,
        get_ml_audit_logger,
        # Status
        get_governance_status,
        get_governance_statistics,
    )
    _governance_available = True
except ImportError as e:
    _governance_available = False

__all__ = [
    # Core ML
    'MLDataPreparator', 
    'RiskClassifier',
    # Governance (if available)
    'DriftDetector', 'DriftType', 'DriftSeverity', 'DriftReport', 'DriftAlert',
    'ModelRegistry', 'ModelStatus', 'ModelVersion', 'ModelMetrics',
    'PerformanceMonitor', 'PerformanceMetrics', 'PerformanceAlert',
    'RetrainingTrigger', 'RetrainingJob', 'TriggerType', 'RetrainingStatus',
    'MLAuditLogger', 'MLAuditEntry', 'MLEventType', 'MLActor',
    'get_drift_detector', 'get_model_registry', 'get_performance_monitor',
    'get_retraining_trigger', 'get_ml_audit_logger',
    'get_governance_status', 'get_governance_statistics',
]
