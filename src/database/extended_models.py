"""
a6on-i - Extended Database Models
============================================
Additional models for migrated functionality from Parquet/SQLite.

New Tables:
- AnomalyDetection - ML anomaly detection results
- PatternMatch - Pattern library matches
- ConfidenceCalibration - Model confidence calibration data
- TrustMetric - Trust scores and metrics
- ConversationHistory - Agent conversation memory
- AgentDecision - Agent decision history
- PipelineRun - Orchestration pipeline state
- PerformanceMetric - System performance tracking
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import Column, String, Float, DateTime, ForeignKey, Text, Boolean, Integer, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import JSONB
import uuid

from .models import Base


# =============================================================================
# ML RESULTS & ANALYTICS
# =============================================================================

class AnomalyDetection(Base):
    """ML-detected anomalies for patients and sites."""
    __tablename__ = "anomaly_detections"
    
    anomaly_id: Mapped[str] = mapped_column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Entity
    entity_type: Mapped[str] = mapped_column(String(20), nullable=False, index=True)  # 'patient' or 'site'
    entity_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    
    # Anomaly details
    anomaly_type: Mapped[str] = mapped_column(String(50), nullable=False)
    anomaly_score: Mapped[float] = mapped_column(Float, nullable=False)
    severity: Mapped[str] = mapped_column(String(20), default='low')
    
    # Features
    feature_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    expected_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    actual_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Context
    description: Mapped[str] = mapped_column(Text, nullable=False)
    recommendations: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Model info
    model_version: Mapped[str] = mapped_column(String(20), default='1.0')
    
    # Timestamps
    detected_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    reviewed: Mapped[bool] = mapped_column(Boolean, default=False)


class PatternMatch(Base):
    """Pattern library matches and alerts."""
    __tablename__ = "pattern_matches"
    
    match_id: Mapped[str] = mapped_column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Pattern
    pattern_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    pattern_name: Mapped[str] = mapped_column(String(100), nullable=False)
    pattern_category: Mapped[str] = mapped_column(String(50), nullable=False)
    
    # Match details
    patient_key: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, index=True)
    site_id: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, index=True)
    
    # Confidence
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    match_strength: Mapped[str] = mapped_column(String(20), default='medium')
    
    # Alert
    is_alert: Mapped[bool] = mapped_column(Boolean, default=False)
    alert_level: Mapped[str] = mapped_column(String(20), default='info')
    alert_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Context
    matched_features: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    
    # Timestamps
    matched_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    acknowledged_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


# =============================================================================
# GOVERNANCE & CALIBRATION
# =============================================================================

class ConfidenceCalibration(Base):
    """Model confidence calibration data."""
    __tablename__ = "confidence_calibrations"
    
    calibration_id: Mapped[str] = mapped_column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Model
    model_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    model_version: Mapped[str] = mapped_column(String(20), nullable=False)
    
    # Prediction
    prediction_id: Mapped[str] = mapped_column(String(50), nullable=False)
    predicted_class: Mapped[str] = mapped_column(String(50), nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    
    # Actual outcome
    actual_class: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    was_correct: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    
    # Calibration bin
    confidence_bin: Mapped[str] = mapped_column(String(20), nullable=False)  # e.g., '0.8-0.9'
    
    # Timestamps
    predicted_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    verified_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class TrustMetric(Base):
    """Trust scores and metrics for AI/ML components."""
    __tablename__ = "trust_metrics"
    
    metric_id: Mapped[str] = mapped_column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Component
    component_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    component_type: Mapped[str] = mapped_column(String(50), nullable=False)  # 'model', 'agent', 'pipeline'
    
    # Trust scores
    overall_trust_score: Mapped[float] = mapped_column(Float, nullable=False)
    accuracy_score: Mapped[float] = mapped_column(Float, default=0.0)
    reliability_score: Mapped[float] = mapped_column(Float, default=0.0)
    consistency_score: Mapped[float] = mapped_column(Float, default=0.0)
    
    # Statistics
    total_predictions: Mapped[int] = mapped_column(Integer, default=0)
    correct_predictions: Mapped[int] = mapped_column(Integer, default=0)
    failed_predictions: Mapped[int] = mapped_column(Integer, default=0)
    
    # Period
    period_start: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    period_end: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    
    # Timestamps
    calculated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


# =============================================================================
# AGENT MEMORY & DECISIONS
# =============================================================================

class ConversationHistory(Base):
    """Agent conversation memory."""
    __tablename__ = "conversation_history"
    
    conversation_id: Mapped[str] = mapped_column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Session
    session_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    
    # Message
    role: Mapped[str] = mapped_column(String(20), nullable=False)  # 'user', 'assistant', 'system'
    content: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Context
    intent: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    entities: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class AgentDecision(Base):
    """Agent decision history and reasoning."""
    __tablename__ = "agent_decisions"
    
    decision_id: Mapped[str] = mapped_column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Agent
    agent_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    agent_type: Mapped[str] = mapped_column(String(50), nullable=False)
    
    # Decision
    decision_type: Mapped[str] = mapped_column(String(50), nullable=False)
    decision: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Context
    context: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    reasoning: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Confidence
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    
    # Outcome
    outcome: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    success: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    
    # Timestamps
    decided_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    executed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


# =============================================================================
# ORCHESTRATION & PIPELINE
# =============================================================================

class PipelineRun(Base):
    """Pipeline orchestration run history."""
    __tablename__ = "pipeline_runs"
    
    run_id: Mapped[str] = mapped_column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Pipeline
    pipeline_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    pipeline_version: Mapped[str] = mapped_column(String(20), default='1.0')
    
    # Status
    status: Mapped[str] = mapped_column(String(30), nullable=False, index=True)  # 'running', 'completed', 'failed'
    
    # Execution
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Results
    records_processed: Mapped[int] = mapped_column(Integer, default=0)
    records_failed: Mapped[int] = mapped_column(Integer, default=0)
    
    # Logs
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    execution_log: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    
    # Trigger
    triggered_by: Mapped[str] = mapped_column(String(100), default='manual')
    trigger_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class PerformanceMetric(Base):
    """System performance tracking."""
    __tablename__ = "performance_metrics"
    
    metric_id: Mapped[str] = mapped_column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Component
    component_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    metric_name: Mapped[str] = mapped_column(String(100), nullable=False)
    
    # Value
    metric_value: Mapped[float] = mapped_column(Float, nullable=False)
    metric_unit: Mapped[str] = mapped_column(String(50), default='ms')
    
    # Context
    tags: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    
    # Timestamps
    recorded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
