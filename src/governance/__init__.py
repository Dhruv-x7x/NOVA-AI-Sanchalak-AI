"""
a6on-i - Governance Module

Phase 10: Governance & Compliance

10.1 Audit Trail System
10.2 Governance Rules Engine
10.3 Confidence Calibration
10.4 Trust Metrics
10.5 Feedback Loop
"""

# Phase 10.1 - Audit Trail
try:
    from .audit_trail import (
        AuditLogger,
        AuditEntry,
        Actor,
        Entity,
        ReasoningChain,
        StateChange,
        EventType,
        ActionCategory,
        get_audit_logger,
    )
    _audit_trail_available = True
except ImportError as e:
    _audit_trail_available = False

# Phase 10.2 - Governance Rules
try:
    from .rules_engine import (
        GovernanceRulesEngine,
        ActionContext,
        GovernanceDecision,
        ApprovalRequest,
        OverrideRequest,
        RiskLevel,
        ConfidenceLevel,
        ActionDecision,
        ApprovalLevel,
        ApprovalStatus,
        ProhibitedActionType,
        get_governance_engine,
    )
    _rules_engine_available = True
except ImportError as e:
    _rules_engine_available = False

# Phase 10.3 - Confidence Calibration
try:
    from .confidence_calibration import (
        ConfidenceCalibrationSystem,
        Prediction,
        Outcome,
        CalibrationMetrics,
        CalibrationBin,
        DriftAlert,
        RecalibrationTrigger,
        OutcomeType,
        DriftType,
        DriftSeverity,
        CalibrationStatus,
        TriggerType,
        get_calibration_system,
    )
    _calibration_available = True
except ImportError as e:
    _calibration_available = False

# Phase 10.4 - Trust Metrics
try:
    from .trust_metrics import (
        TrustMetricsSystem,
        Interaction,
        SatisfactionFeedback,
        FeatureUsage,
        TrustMetrics,
        TrustTrend,
        TrustAlert,
        MetricType,
        InteractionType,
        InteractionOutcome,
        SatisfactionLevel,
        FeatureCategory,
        TrustLevel,
        TrendDirection,
        get_trust_metrics_system,
        reset_trust_metrics_system,
        record_interaction,
        record_outcome,
        record_satisfaction,
        record_feature_usage,
        get_trust_metrics,
        get_trust_alerts,
        get_trust_stats,
    )
    _trust_metrics_available = True
except ImportError as e:
    _trust_metrics_available = False

# Phase 10.5 - Feedback Loop
try:
    from .feedback_loop import (
        FeedbackLoopSystem,
        FeedbackRecord,
        LearningSignal,
        ModelUpdateRequest,
        PatternCandidate,
        FeedbackAggregation,
        FeedbackType,
        LearningSignalType,
        SignalStrength,
        ModelUpdateType,
        ModelUpdateStatus,
        UpdatePriority,
        PatternStatus,
        PatternSource,
        get_feedback_loop_system,
        reset_feedback_loop_system,
        capture_feedback,
        get_feedback_summary,
        get_pending_signals,
        check_update_triggers,
        get_feedback_stats,
    )
    _feedback_loop_available = True
except ImportError as e:
    _feedback_loop_available = False
