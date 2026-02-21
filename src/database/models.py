"""
SANCHALAK AI - Database Models
===================================
Complete production-grade SQLAlchemy ORM models for clinical trial data.

Models:
- Study, ClinicalSite, Patient (Core entities)
- Visit, LabResult, AdverseEvent (Clinical data)
- ProjectIssue, Query, ResolutionAction (Data management)
- Signature, AuditLog (Compliance - 21 CFR Part 11)
- User, Role, UserRoleAssignment (Authentication)
- MLModelVersion, DriftReport (ML Governance)
"""

from datetime import datetime
from typing import List, Optional
from decimal import Decimal

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, ForeignKey, Text, Boolean, 
    Numeric, Date, Index, UniqueConstraint, Table, Enum as SQLEnum, JSON
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID, INET, JSONB
import uuid

from .enums import (
    PatientStatus, CleanStatusTier, RiskLevel,
    SiteStatus, Region,
    StudyPhase, StudyStatus,
    VisitStatus, VisitType,
    IssueStatus, IssuePriority, IssueSeverity, IssueCategory,
    QueryStatus, QueryType,
    AdverseEventSeverity, AdverseEventCausality, AdverseEventOutcome, SAEClassification,
    SignatureType, SignatureMeaning,
    AuditAction, EntityType,
    UserRole, UserStatus,
    ModelType, ModelStatus, DriftSeverity,
)


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


# =============================================================================
# ASSOCIATION TABLES (Many-to-Many)
# =============================================================================

study_sites = Table(
    'study_sites',
    Base.metadata,
    Column('study_id', String(50), ForeignKey('studies.study_id'), primary_key=True),
    Column('site_id', String(50), ForeignKey('clinical_sites.site_id'), primary_key=True),
    Column('activation_date', DateTime, nullable=True),
    Column('status', String(20), default='active'),
)

user_role_assignments = Table(
    'user_role_assignments',
    Base.metadata,
    Column('user_id', String(50), ForeignKey('users.user_id'), primary_key=True),
    Column('role_id', String(50), ForeignKey('roles.role_id'), primary_key=True),
    Column('assigned_at', DateTime, default=datetime.utcnow),
    Column('assigned_by', String(50)),
)


# =============================================================================
# CORE ENTITIES
# =============================================================================

class Study(Base):
    """Clinical trial/study information."""
    __tablename__ = "studies"
    
    study_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    protocol_number: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)
    
    # Study metadata
    phase: Mapped[str] = mapped_column(String(20), default=StudyPhase.PHASE_III.value)
    status: Mapped[str] = mapped_column(String(30), default=StudyStatus.ACTIVE.value, index=True)
    therapeutic_area: Mapped[str] = mapped_column(String(100), nullable=True)
    indication: Mapped[str] = mapped_column(String(200), nullable=True)
    sponsor: Mapped[str] = mapped_column(String(100), nullable=True)
    
    # Dates
    start_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    end_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    target_enrollment: Mapped[int] = mapped_column(Integer, default=0)
    current_enrollment: Mapped[int] = mapped_column(Integer, default=0)
    
    # Relationships
    sites: Mapped[List["ClinicalSite"]] = relationship(
        secondary=study_sites, back_populates="studies"
    )
    patients: Mapped[List["Patient"]] = relationship(back_populates="study")
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ClinicalSite(Base):
    """Clinical trial site/facility."""
    __tablename__ = "clinical_sites"
    
    site_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    
    # Location
    country: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    region: Mapped[str] = mapped_column(String(30), default=Region.NORTH_AMERICA.value, index=True)
    city: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    address: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Status
    status: Mapped[str] = mapped_column(String(30), default=SiteStatus.ACTIVE.value, index=True)
    activation_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Metrics (computed/cached)
    performance_score: Mapped[float] = mapped_column(Float, default=85.0)
    risk_level: Mapped[str] = mapped_column(String(20), default=RiskLevel.LOW.value)
    dqi_score: Mapped[float] = mapped_column(Float, default=90.0)
    enrollment_rate: Mapped[float] = mapped_column(Float, default=0.0)
    query_resolution_days: Mapped[float] = mapped_column(Float, default=5.0)
    
    # Personnel
    principal_investigator: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    pi_email: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    coordinator_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    coordinator_email: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    # Relationships
    studies: Mapped[List["Study"]] = relationship(
        secondary=study_sites, back_populates="sites"
    )
    patients: Mapped[List["Patient"]] = relationship(back_populates="site")
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Patient(Base):
    """Patient/subject record with UPR (Unified Patient Record) features."""
    __tablename__ = "patients"
    
    patient_key: Mapped[str] = mapped_column(String(50), primary_key=True)
    
    # Study/Site assignment
    study_id: Mapped[str] = mapped_column(String(50), ForeignKey("studies.study_id"), nullable=False, index=True)
    site_id: Mapped[str] = mapped_column(String(50), ForeignKey("clinical_sites.site_id"), nullable=False, index=True)
    
    # Patient status
    status: Mapped[str] = mapped_column(String(30), default=PatientStatus.ACTIVE.value, index=True)
    enrollment_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    randomization_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completion_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Demographics (de-identified)
    age_at_enrollment: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    gender: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    
    # Clean status
    clean_status_tier: Mapped[str] = mapped_column(String(20), default=CleanStatusTier.TIER_0.value)
    is_db_lock_ready: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Risk metrics
    risk_level: Mapped[str] = mapped_column(String(20), default=RiskLevel.LOW.value)
    risk_score: Mapped[float] = mapped_column(Float, default=0.0)
    
    # Data quality metrics
    dqi_score: Mapped[float] = mapped_column(Float, default=100.0)
    has_issues: Mapped[bool] = mapped_column(Boolean, default=False)
    open_issues_count: Mapped[int] = mapped_column(Integer, default=0)
    open_queries_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # visit tracking
    total_visits_planned: Mapped[int] = mapped_column(Integer, default=0)
    total_visits_completed: Mapped[int] = mapped_column(Integer, default=0)
    visit_compliance_pct: Mapped[float] = mapped_column(Float, default=100.0)
    
    # Derived Metrics (Percentages)
    pct_missing_visits: Mapped[float] = mapped_column(Float, default=0.0)
    pct_missing_pages: Mapped[float] = mapped_column(Float, default=0.0)
    pct_verified_forms: Mapped[float] = mapped_column(Float, default=0.0)
    is_clean_patient: Mapped[bool] = mapped_column(Boolean, default=False)

    # Reconciliation & Extended Status
    lab_discrepancy_count: Mapped[int] = mapped_column(Integer, default=0)
    frozen_forms_count: Mapped[int] = mapped_column(Integer, default=0)
    inactivated_folders_count: Mapped[int] = mapped_column(Integer, default=0)
    sdtm_ready: Mapped[bool] = mapped_column(Boolean, default=False)

    
    # Documentation
    all_signatures_complete: Mapped[bool] = mapped_column(Boolean, default=False)
    consent_valid: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Safety
    has_sae: Mapped[bool] = mapped_column(Boolean, default=False)
    sae_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # UPR Additional Features (264 total - key ones stored, others computed)
    days_since_last_activity: Mapped[int] = mapped_column(Integer, default=0)
    data_entry_lag_days: Mapped[float] = mapped_column(Float, default=0.0)
    avg_query_age_days: Mapped[float] = mapped_column(Float, default=0.0)
    
    # Relationships
    study: Mapped["Study"] = relationship(back_populates="patients")
    site: Mapped["ClinicalSite"] = relationship(back_populates="patients")
    visits: Mapped[List["Visit"]] = relationship(back_populates="patient", cascade="all, delete-orphan")
    lab_results: Mapped[List["LabResult"]] = relationship(back_populates="patient", cascade="all, delete-orphan")
    adverse_events: Mapped[List["AdverseEvent"]] = relationship(back_populates="patient", cascade="all, delete-orphan")
    issues: Mapped[List["ProjectIssue"]] = relationship(back_populates="patient", cascade="all, delete-orphan")
    queries: Mapped[List["Query"]] = relationship(back_populates="patient", cascade="all, delete-orphan")
    signatures: Mapped[List["Signature"]] = relationship(back_populates="patient", cascade="all, delete-orphan")
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        Index('idx_patient_study_site', 'study_id', 'site_id'),
        Index('idx_patient_status_risk', 'status', 'risk_level'),
    )


# =============================================================================
# CLINICAL DATA
# =============================================================================

class Visit(Base):
    """Patient visit records."""
    __tablename__ = "visits"
    
    visit_id: Mapped[str] = mapped_column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_key: Mapped[str] = mapped_column(String(50), ForeignKey("patients.patient_key"), nullable=False, index=True)
    
    # Visit details
    visit_number: Mapped[int] = mapped_column(Integer, nullable=False)
    visit_name: Mapped[str] = mapped_column(String(100), nullable=False)
    visit_type: Mapped[str] = mapped_column(String(30), default=VisitType.TREATMENT.value)
    
    # Scheduling
    scheduled_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    actual_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    window_start: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    window_end: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Status
    status: Mapped[str] = mapped_column(String(30), default=VisitStatus.SCHEDULED.value, index=True)
    is_in_window: Mapped[bool] = mapped_column(Boolean, default=True)
    deviation_days: Mapped[int] = mapped_column(Integer, default=0)
    
    # Data completeness
    data_entry_complete: Mapped[bool] = mapped_column(Boolean, default=False)
    sdv_complete: Mapped[bool] = mapped_column(Boolean, default=False)
    queries_resolved: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Relationships
    patient: Mapped["Patient"] = relationship(back_populates="visits")
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class LabResult(Base):
    """Laboratory test results."""
    __tablename__ = "lab_results"
    
    lab_id: Mapped[str] = mapped_column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_key: Mapped[str] = mapped_column(String(50), ForeignKey("patients.patient_key"), nullable=False, index=True)
    visit_id: Mapped[Optional[str]] = mapped_column(String(50), ForeignKey("visits.visit_id"), nullable=True)
    
    # Test details
    test_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    test_code: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    category: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)  # hematology, chemistry, etc.
    
    # Results
    result_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    result_text: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)
    unit: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    
    # Reference ranges
    lower_limit: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    upper_limit: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    is_abnormal: Mapped[bool] = mapped_column(Boolean, default=False)
    is_clinically_significant: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Dates
    collection_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    result_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Relationships
    patient: Mapped["Patient"] = relationship(back_populates="lab_results")
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class AdverseEvent(Base):
    """Adverse event and safety records."""
    __tablename__ = "adverse_events"
    
    ae_id: Mapped[str] = mapped_column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_key: Mapped[str] = mapped_column(String(50), ForeignKey("patients.patient_key"), nullable=False, index=True)
    
    # Event details
    ae_term: Mapped[str] = mapped_column(String(200), nullable=False)
    meddra_code: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, index=True)
    meddra_pt: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)  # Preferred Term
    meddra_soc: Mapped[Optional[str]] = mapped_column(String(200), nullable=True)  # System Organ Class
    
    # Classification
    severity: Mapped[str] = mapped_column(String(30), default=AdverseEventSeverity.MILD.value)
    causality: Mapped[str] = mapped_column(String(30), default=AdverseEventCausality.NOT_RELATED.value)
    outcome: Mapped[str] = mapped_column(String(30), default=AdverseEventOutcome.RECOVERED.value)
    
    # SAE Classification
    is_sae: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    sae_classification: Mapped[str] = mapped_column(String(30), default=SAEClassification.NOT_SAE.value)
    
    # Dates
    onset_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    resolution_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    reported_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Status
    is_ongoing: Mapped[bool] = mapped_column(Boolean, default=True)
    action_taken: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    # Relationships
    patient: Mapped["Patient"] = relationship(back_populates="adverse_events")
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# =============================================================================
# DATA MANAGEMENT
# =============================================================================

class ProjectIssue(Base):
    """Data quality issues and problems."""
    __tablename__ = "project_issues"
    
    issue_id: Mapped[str] = mapped_column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_key: Mapped[Optional[str]] = mapped_column(String(50), ForeignKey("patients.patient_key"), nullable=True, index=True)
    site_id: Mapped[Optional[str]] = mapped_column(String(50), ForeignKey("clinical_sites.site_id"), nullable=True, index=True)
    study_id: Mapped[Optional[str]] = mapped_column(String(50), ForeignKey("studies.study_id"), nullable=True, index=True)
    
    # Issue classification
    category: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    issue_type: Mapped[str] = mapped_column(String(50), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Priority/Severity
    priority: Mapped[str] = mapped_column(String(20), default=IssuePriority.MEDIUM.value, index=True)
    severity: Mapped[str] = mapped_column(String(20), default=IssueSeverity.MINOR.value)
    
    # Status
    status: Mapped[str] = mapped_column(String(20), default=IssueStatus.OPEN.value, index=True)
    
    # Assignment
    assigned_to: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    assigned_role: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    
    # Cascade impact
    blocking_count: Mapped[int] = mapped_column(Integer, default=0)  # How many items this blocks
    cascade_impact_score: Mapped[float] = mapped_column(Float, default=0.0)
    
    # Resolution
    resolution_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    resolution_template_id: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    
    # Dates
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    due_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Relationships
    patient: Mapped[Optional["Patient"]] = relationship(back_populates="issues")
    actions: Mapped[List["ResolutionAction"]] = relationship(back_populates="issue", cascade="all, delete-orphan")
    
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Query(Base):
    """Data queries (clarification requests)."""
    __tablename__ = "queries"
    
    query_id: Mapped[str] = mapped_column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    patient_key: Mapped[str] = mapped_column(String(50), ForeignKey("patients.patient_key"), nullable=False, index=True)
    visit_id: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    
    # Query details
    field_name: Mapped[str] = mapped_column(String(100), nullable=False)
    form_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    query_type: Mapped[str] = mapped_column(String(30), default=QueryType.MANUAL.value)
    
    # Response
    response_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Status
    status: Mapped[str] = mapped_column(String(20), default=QueryStatus.OPEN.value, index=True)
    
    # Dates
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    answered_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    closed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Age tracking
    age_days: Mapped[int] = mapped_column(Integer, default=0)
    
    # Relationships
    patient: Mapped["Patient"] = relationship(back_populates="queries")
    
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ResolutionAction(Base):
    """Actions taken to resolve issues."""
    __tablename__ = "resolution_actions"
    
    action_id: Mapped[str] = mapped_column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    issue_id: Mapped[str] = mapped_column(String(50), ForeignKey("project_issues.issue_id"), nullable=False, index=True)
    
    # Action details
    action_type: Mapped[str] = mapped_column(String(50), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Assignment
    assigned_to: Mapped[str] = mapped_column(String(100), nullable=False)
    assigned_role: Mapped[str] = mapped_column(String(50), default="data_manager")
    
    # Status
    status: Mapped[str] = mapped_column(String(20), default="pending")
    
    # Dates
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    due_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Outcome
    outcome_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    success: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    
    # Relationships
    issue: Mapped["ProjectIssue"] = relationship(back_populates="actions")


# =============================================================================
# COMPLIANCE (21 CFR Part 11)
# =============================================================================

class Signature(Base):
    """Electronic signatures for compliance."""
    __tablename__ = "signatures"
    
    signature_id: Mapped[str] = mapped_column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # What is being signed
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    entity_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    patient_key: Mapped[Optional[str]] = mapped_column(String(50), ForeignKey("patients.patient_key"), nullable=True)
    
    # Signature details
    signature_type: Mapped[str] = mapped_column(String(30), default=SignatureType.APPROVAL.value)
    meaning: Mapped[str] = mapped_column(String(50), default=SignatureMeaning.DATA_APPROVAL.value)
    
    # Signer information
    user_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    user_name: Mapped[str] = mapped_column(String(100), nullable=False)
    user_role: Mapped[str] = mapped_column(String(50), nullable=False)
    
    # Signature timestamp and verification
    signed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    ip_address: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    
    # Integrity
    data_hash: Mapped[str] = mapped_column(String(128), nullable=False)  # SHA-256 of signed data
    is_valid: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Comments
    comments: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Relationships
    patient: Mapped[Optional["Patient"]] = relationship(back_populates="signatures")


class AuditLog(Base):
    """Immutable audit trail for compliance."""
    __tablename__ = "audit_logs"
    
    log_id: Mapped[str] = mapped_column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Timestamp
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    
    # User
    user_id: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    user_name: Mapped[str] = mapped_column(String(100), nullable=False)
    user_role: Mapped[str] = mapped_column(String(50), nullable=False)
    
    # Action
    action: Mapped[str] = mapped_column(String(30), nullable=False, index=True)
    
    # Entity
    entity_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    entity_id: Mapped[str] = mapped_column(String(100), nullable=False)
    
    # Change details
    field_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    old_value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    new_value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Reason for change
    reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Session info
    ip_address: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    user_agent: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    session_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    # Integrity chain
    checksum: Mapped[str] = mapped_column(String(128), nullable=False)  # SHA-256 of this entry
    previous_checksum: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)  # Chain to previous
    
    # Indexes for compliance queries
    __table_args__ = (
        Index('idx_audit_user_time', 'user_id', 'timestamp'),
        Index('idx_audit_entity', 'entity_type', 'entity_id'),
    )


# =============================================================================
# AUTHENTICATION
# =============================================================================

class User(Base):
    """User accounts."""
    __tablename__ = "users"
    
    user_id: Mapped[str] = mapped_column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Credentials
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    
    # Profile
    first_name: Mapped[str] = mapped_column(String(50), nullable=False)
    last_name: Mapped[str] = mapped_column(String(50), nullable=False)
    
    # Status
    status: Mapped[str] = mapped_column(String(30), default=UserStatus.ACTIVE.value, index=True)
    
    # Security
    failed_login_attempts: Mapped[int] = mapped_column(Integer, default=0)
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    password_changed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    mfa_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    mfa_secret: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    # Dates
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    roles: Mapped[List["Role"]] = relationship(
        secondary=user_role_assignments, back_populates="users"
    )


class Role(Base):
    """User roles for RBAC."""
    __tablename__ = "roles"
    
    role_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    name: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    description: Mapped[str] = mapped_column(String(200), nullable=True)
    
    # Permissions (JSON object)
    permissions: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    
    # Hierarchy
    level: Mapped[int] = mapped_column(Integer, default=0)  # Higher = more access
    
    # Relationships
    users: Mapped[List["User"]] = relationship(
        secondary=user_role_assignments, back_populates="roles"
    )
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


# =============================================================================
# ML GOVERNANCE
# =============================================================================

class MLModelVersion(Base):
    """ML model version registry."""
    __tablename__ = "ml_model_versions"
    
    version_id: Mapped[str] = mapped_column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Model identity
    model_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    model_type: Mapped[str] = mapped_column(String(50), nullable=False)
    version: Mapped[str] = mapped_column(String(20), nullable=False)
    
    # Status
    status: Mapped[str] = mapped_column(String(30), default=ModelStatus.TRAINING.value, index=True)
    
    # Artifacts
    artifact_path: Mapped[str] = mapped_column(String(500), nullable=False)
    
    # Metrics
    training_metrics: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    validation_metrics: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    
    # Training info
    trained_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    training_data_hash: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    training_samples: Mapped[int] = mapped_column(Integer, default=0)
    
    # Deployment
    deployed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    deprecated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Governance
    feature_baselines: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Relationships
    drift_reports: Mapped[List["DriftReport"]] = relationship(back_populates="model_version")
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint('model_name', 'version', name='uq_model_version'),
    )


class DriftReport(Base):
    """Model drift detection reports."""
    __tablename__ = "drift_reports"
    
    report_id: Mapped[str] = mapped_column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    version_id: Mapped[str] = mapped_column(String(50), ForeignKey("ml_model_versions.version_id"), nullable=False, index=True)
    
    # Analysis period
    analysis_start: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    analysis_end: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    
    # Results
    severity: Mapped[str] = mapped_column(String(20), default=DriftSeverity.NONE.value)
    psi_score: Mapped[float] = mapped_column(Float, default=0.0)  # Population Stability Index
    ks_statistic: Mapped[float] = mapped_column(Float, default=0.0)  # Kolmogorov-Smirnov
    
    # Feature-level drift (JSON)
    feature_drift_details: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    
    # Performance drift
    baseline_accuracy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    current_accuracy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Recommendations
    recommendations: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    retrain_recommended: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Relationships
    model_version: Mapped["MLModelVersion"] = relationship(back_populates="drift_reports")
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


# =============================================================================
# CRA ACTIVITY & LOGGING
# =============================================================================

class CRAActivityLog(Base):
    """Historical log of CRA monitoring activities."""
    __tablename__ = "cra_activity_logs"
    
    log_id: Mapped[str] = mapped_column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    site_id: Mapped[str] = mapped_column(String(50), ForeignKey("clinical_sites.site_id"), nullable=False, index=True)
    cra_name: Mapped[str] = mapped_column(String(100), nullable=False)
    
    activity_type: Mapped[str] = mapped_column(String(50), nullable=False) # PSV, SIV, RMV, COV, Follow-up
    visit_date: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    status: Mapped[str] = mapped_column(String(30), default="completed") # planned, completed, cancelled
    follow_up_letter_sent: Mapped[bool] = mapped_column(Boolean, default=False)
    follow_up_letter_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    metrics_at_visit: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True) # Snapshot of site metrics at that time
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class DQIWeightConfiguration(Base):
    """User-defined weights for Data Quality Index calculation."""
    __tablename__ = "dqi_weight_configs"
    
    config_id: Mapped[str] = mapped_column(String(50), primary_key=True, default="default")
    study_id: Mapped[Optional[str]] = mapped_column(String(50), ForeignKey("studies.study_id"), nullable=True)
    
    # Weight factors (sum to 1.0 or used as multipliers)
    safety_weight: Mapped[float] = mapped_column(Float, default=0.4)
    query_weight: Mapped[float] = mapped_column(Float, default=0.2)
    visit_weight: Mapped[float] = mapped_column(Float, default=0.2)
    lab_weight: Mapped[float] = mapped_column(Float, default=0.1)
    integrity_weight: Mapped[float] = mapped_column(Float, default=0.1)
    
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    updated_by: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class CollaborationRoom(Base):
    """Real-time investigation and collaboration rooms."""
    __tablename__ = "collaboration_rooms"
    
    room_id: Mapped[str] = mapped_column(String(50), primary_key=True, default=lambda: f"ROOM-{uuid.uuid4().hex[:8].upper()}")
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    room_type: Mapped[str] = mapped_column(String(30), default="investigation") # investigation, workspace, site_chat
    
    # Status and priority
    status: Mapped[str] = mapped_column(String(20), default="active")
    priority: Mapped[str] = mapped_column(String(20), default="medium")
    
    # Link to entities
    related_entity_type: Mapped[Optional[str]] = mapped_column(String(30), nullable=True) # site, study, patient, issue
    related_entity_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    # Escalation
    escalation_level: Mapped[int] = mapped_column(Integer, default=0)
    
    # Dates
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    closed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Relationships
    messages: Mapped[List["RoomMessage"]] = relationship(back_populates="room", cascade="all, delete-orphan")
    participants: Mapped[List["RoomParticipant"]] = relationship(back_populates="room", cascade="all, delete-orphan")


class RoomMessage(Base):
    """Messages within a collaboration room."""
    __tablename__ = "room_messages"
    
    message_id: Mapped[str] = mapped_column(String(50), primary_key=True, default=lambda: str(uuid.uuid4()))
    room_id: Mapped[str] = mapped_column(String(50), ForeignKey("collaboration_rooms.room_id"), nullable=False, index=True)
    user_id: Mapped[str] = mapped_column(String(50), ForeignKey("users.user_id"), nullable=False)
    
    content: Mapped[str] = mapped_column(Text, nullable=False)
    message_type: Mapped[str] = mapped_column(String(20), default="text") # text, system, attachment
    
    # Metadata for @tagging and threading
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True) 
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    room: Mapped["CollaborationRoom"] = relationship(back_populates="messages")


class RoomParticipant(Base):
    """Participants in a collaboration room."""
    __tablename__ = "room_participants"
    
    room_id: Mapped[str] = mapped_column(String(50), ForeignKey("collaboration_rooms.room_id"), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(50), ForeignKey("users.user_id"), primary_key=True)
    
    role: Mapped[str] = mapped_column(String(30), default="member") # owner, member, viewer
    last_read_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    room: Mapped["CollaborationRoom"] = relationship(back_populates="participants")

