# src/ml/governance/__init__.py

"""
SANCHALAK AI - ML Governance Module
Model Drift Detection & Compliance System

Layer 8: ML Model Governance

Components:
- 8.1 Drift Detector - Statistical drift detection (PSI, KS, Chi-Square)
- 8.2 Model Registry - Version control and artifact management
- 8.3 Performance Monitor - Real-time accuracy tracking
- 8.4 Retraining Trigger - Automated retraining pipeline
- 8.5 ML Audit Trail - 21 CFR Part 11 compliance

21 CFR Part 11 Compliance Features:
- Immutable audit logs with SHA-256 checksums
- Electronic signatures for model approvals
- Chain integrity verification
- Complete traceability for regulatory audits
"""

# =============================================================================
# DRIFT DETECTOR (8.1)
# =============================================================================

try:
    from .drift_detector import (
        # Enums
        DriftType,
        DriftSeverity,
        FeatureType,
        # Data classes
        FeatureBaseline,
        FeatureDriftResult,
        PerformanceDriftResult,
        DriftReport,
        DriftAlert,
        # Main class
        DriftDetector,
        # Singleton getter
        get_drift_detector,
        reset_drift_detector,
    )
    _drift_detector_available = True
except ImportError as e:
    _drift_detector_available = False
    print(f"Warning: Drift detector not available: {e}")

# =============================================================================
# MODEL REGISTRY (8.2)
# =============================================================================

try:
    from .model_registry import (
        # Enums
        ModelStatus,
        ModelType,
        ArtifactType,
        # Data classes
        ModelMetrics,
        ModelArtifact,
        TrainingConfig,
        PromotionRecord,
        ModelVersion,
        # Main class
        ModelRegistry,
        # Singleton getter
        get_model_registry,
        reset_model_registry,
    )
    _model_registry_available = True
except ImportError as e:
    _model_registry_available = False
    print(f"Warning: Model registry not available: {e}")

# =============================================================================
# PERFORMANCE MONITOR (8.3)
# =============================================================================

try:
    from .performance_monitor import (
        # Enums
        MetricType,
        TimeWindow,
        AlertSeverity,
        AlertType,
        # Data classes
        PredictionLog,
        PerformanceMetrics,
        MetricThreshold,
        PerformanceAlert,
        # Main class
        PerformanceMonitor,
        # Singleton getter
        get_performance_monitor,
        reset_performance_monitor,
    )
    _performance_monitor_available = True
except ImportError as e:
    _performance_monitor_available = False
    print(f"Warning: Performance monitor not available: {e}")

# =============================================================================
# RETRAINING TRIGGER (8.4)
# =============================================================================

try:
    from .retraining_trigger import (
        # Enums
        TriggerType,
        RetrainingStatus,
        ScheduleFrequency,
        # Data classes
        RetrainingRule,
        RetrainingJob,
        # Main class
        RetrainingTrigger,
        # Singleton getter
        get_retraining_trigger,
        reset_retraining_trigger,
    )
    _retraining_trigger_available = True
except ImportError as e:
    _retraining_trigger_available = False
    print(f"Warning: Retraining trigger not available: {e}")

# =============================================================================
# ML AUDIT TRAIL (8.5)
# =============================================================================

try:
    from .ml_audit_trail import (
        # Enums
        MLEventType,
        ActorType,
        SignatureType,
        # Data classes
        MLActor,
        ElectronicSignature,
        MLAuditEntry,
        # Main class
        MLAuditLogger,
        # Singleton getter
        get_ml_audit_logger,
        reset_ml_audit_logger,
    )
    _ml_audit_trail_available = True
except ImportError as e:
    _ml_audit_trail_available = False
    print(f"Warning: ML audit trail not available: {e}")


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Drift Detector
    'DriftType', 'DriftSeverity', 'FeatureType',
    'FeatureBaseline', 'FeatureDriftResult', 'PerformanceDriftResult', 
    'DriftReport', 'DriftAlert',
    'DriftDetector', 'get_drift_detector', 'reset_drift_detector',
    
    # Model Registry
    'ModelStatus', 'ModelType', 'ArtifactType',
    'ModelMetrics', 'ModelArtifact', 'TrainingConfig', 
    'PromotionRecord', 'ModelVersion',
    'ModelRegistry', 'get_model_registry', 'reset_model_registry',
    
    # Performance Monitor
    'MetricType', 'TimeWindow', 'AlertSeverity', 'AlertType',
    'PredictionLog', 'PerformanceMetrics', 'MetricThreshold', 'PerformanceAlert',
    'PerformanceMonitor', 'get_performance_monitor', 'reset_performance_monitor',
    
    # Retraining Trigger
    'TriggerType', 'RetrainingStatus', 'ScheduleFrequency',
    'RetrainingRule', 'RetrainingJob',
    'RetrainingTrigger', 'get_retraining_trigger', 'reset_retraining_trigger',
    
    # ML Audit Trail
    'MLEventType', 'ActorType', 'SignatureType',
    'MLActor', 'ElectronicSignature', 'MLAuditEntry',
    'MLAuditLogger', 'get_ml_audit_logger', 'reset_ml_audit_logger',
]


# =============================================================================
# MODULE STATUS
# =============================================================================

def get_governance_status() -> dict:
    """Get status of all governance module components"""
    return {
        'drift_detector': _drift_detector_available,
        'model_registry': _model_registry_available,
        'performance_monitor': _performance_monitor_available,
        'retraining_trigger': _retraining_trigger_available,
        'ml_audit_trail': _ml_audit_trail_available,
        'all_available': all([
            _drift_detector_available,
            _model_registry_available,
            _performance_monitor_available,
            _retraining_trigger_available,
            _ml_audit_trail_available
        ])
    }


def get_governance_statistics() -> dict:
    """Get combined statistics from all governance components"""
    stats = {}
    
    if _drift_detector_available:
        try:
            stats['drift_detector'] = get_drift_detector().get_statistics()
        except Exception as e:
            stats['drift_detector'] = {'error': str(e)}
    
    if _model_registry_available:
        try:
            stats['model_registry'] = get_model_registry().get_statistics()
        except Exception as e:
            stats['model_registry'] = {'error': str(e)}
    
    if _performance_monitor_available:
        try:
            stats['performance_monitor'] = get_performance_monitor().get_statistics()
        except Exception as e:
            stats['performance_monitor'] = {'error': str(e)}
    
    if _retraining_trigger_available:
        try:
            stats['retraining_trigger'] = get_retraining_trigger().get_statistics()
        except Exception as e:
            stats['retraining_trigger'] = {'error': str(e)}
    
    if _ml_audit_trail_available:
        try:
            stats['ml_audit_trail'] = get_ml_audit_logger().get_statistics()
        except Exception as e:
            stats['ml_audit_trail'] = {'error': str(e)}
    
    return stats
