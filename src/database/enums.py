"""
a6on-i - Database Enums
==================================
SQLAlchemy enum types for consistent database values.
"""

from enum import Enum


# =============================================================================
# PATIENT ENUMS
# =============================================================================

class PatientStatus(str, Enum):
    """Patient lifecycle status."""
    SCREENING = "screening"
    ENROLLED = "enrolled"
    ACTIVE = "active"
    COMPLETED = "completed"
    WITHDRAWN = "withdrawn"
    LOST_TO_FOLLOWUP = "lost_to_followup"
    SCREEN_FAILURE = "screen_failure"


class CleanStatusTier(str, Enum):
    """Clean patient status tiers."""
    TIER_0 = "tier_0"  # Not clean - has blockers
    TIER_1 = "tier_1"  # Clinical clean - no hard blocks
    TIER_2 = "tier_2"  # Operational clean - no soft blocks
    DB_LOCK_READY = "db_lock_ready"  # Ready for database lock


class RiskLevel(str, Enum):
    """Risk classification levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# =============================================================================
# SITE ENUMS
# =============================================================================

class SiteStatus(str, Enum):
    """Clinical site status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    CLOSED = "closed"
    PENDING_ACTIVATION = "pending_activation"


class Region(str, Enum):
    """Geographic regions."""
    NORTH_AMERICA = "north_america"
    EUROPE = "europe"
    ASIA_PACIFIC = "asia_pacific"
    LATIN_AMERICA = "latin_america"
    MIDDLE_EAST_AFRICA = "middle_east_africa"


# =============================================================================
# STUDY ENUMS
# =============================================================================

class StudyPhase(str, Enum):
    """Clinical trial phases."""
    PHASE_I = "phase_1"
    PHASE_II = "phase_2"
    PHASE_III = "phase_3"
    PHASE_IV = "phase_4"


class StudyStatus(str, Enum):
    """Study lifecycle status."""
    PLANNING = "planning"
    STARTUP = "startup"
    ENROLLING = "enrolling"
    ACTIVE = "active"
    CLOSED_TO_ENROLLMENT = "closed_to_enrollment"
    COMPLETED = "completed"
    TERMINATED = "terminated"
    SUSPENDED = "suspended"


# =============================================================================
# VISIT ENUMS
# =============================================================================

class VisitStatus(str, Enum):
    """Visit completion status."""
    SCHEDULED = "scheduled"
    COMPLETED = "completed"
    MISSED = "missed"
    CANCELLED = "cancelled"
    PARTIALLY_COMPLETED = "partially_completed"


class VisitType(str, Enum):
    """Types of patient visits."""
    SCREENING = "screening"
    BASELINE = "baseline"
    TREATMENT = "treatment"
    FOLLOW_UP = "follow_up"
    UNSCHEDULED = "unscheduled"
    EARLY_TERMINATION = "early_termination"
    END_OF_STUDY = "end_of_study"


# =============================================================================
# ISSUE ENUMS
# =============================================================================

class IssueStatus(str, Enum):
    """Issue lifecycle status."""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    PENDING_REVIEW = "pending_review"
    RESOLVED = "resolved"
    CLOSED = "closed"
    WONT_FIX = "wont_fix"


class IssuePriority(str, Enum):
    """Issue priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IssueSeverity(str, Enum):
    """Issue severity classification."""
    MINOR = "minor"
    MAJOR = "major"
    BLOCKER = "blocker"


class IssueCategory(str, Enum):
    """Issue type categories (14 types from ML model)."""
    MISSING_VISITS = "missing_visits"
    OVERDUE_QUERIES = "overdue_queries"
    SIGNATURE_GAPS = "signature_gaps"
    PROTOCOL_DEVIATION = "protocol_deviation"
    SAE_PENDING = "sae_pending"
    LAB_DISCREPANCY = "lab_discrepancy"
    CONSENT_ISSUE = "consent_issue"
    SDV_INCOMPLETE = "sdv_incomplete"
    CODING_REQUIRED = "coding_required"
    DATA_ENTRY_ERROR = "data_entry_error"
    ELIGIBILITY_ISSUE = "eligibility_issue"
    MEDICATION_ISSUE = "medication_issue"
    DOCUMENTATION_MISSING = "documentation_missing"
    COMPLIANCE_ISSUE = "compliance_issue"


# =============================================================================
# QUERY ENUMS
# =============================================================================

class QueryStatus(str, Enum):
    """Data query status."""
    OPEN = "open"
    ANSWERED = "answered"
    CLOSED = "closed"
    CANCELLED = "cancelled"


class QueryType(str, Enum):
    """Types of data queries."""
    MANUAL = "manual"
    AUTO_GENERATED = "auto_generated"
    SYSTEM = "system"


# =============================================================================
# SAFETY ENUMS
# =============================================================================

class AdverseEventSeverity(str, Enum):
    """Adverse event severity grades."""
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    LIFE_THREATENING = "life_threatening"
    DEATH = "death"


class AdverseEventCausality(str, Enum):
    """Relationship to study drug."""
    NOT_RELATED = "not_related"
    UNLIKELY = "unlikely"
    POSSIBLE = "possible"
    PROBABLE = "probable"
    DEFINITE = "definite"


class AdverseEventOutcome(str, Enum):
    """Adverse event outcome."""
    RECOVERED = "recovered"
    RECOVERING = "recovering"
    NOT_RECOVERED = "not_recovered"
    RECOVERED_WITH_SEQUELAE = "recovered_with_sequelae"
    FATAL = "fatal"
    UNKNOWN = "unknown"


class SAEClassification(str, Enum):
    """Serious Adverse Event classification."""
    NOT_SAE = "not_sae"
    HOSPITALIZATION = "hospitalization"
    DISABILITY = "disability"
    LIFE_THREATENING = "life_threatening"
    DEATH = "death"
    CONGENITAL_ANOMALY = "congenital_anomaly"
    MEDICALLY_SIGNIFICANT = "medically_significant"


# =============================================================================
# SIGNATURE ENUMS (21 CFR Part 11)
# =============================================================================

class SignatureType(str, Enum):
    """Electronic signature types."""
    APPROVAL = "approval"
    REVIEW = "review"
    ACKNOWLEDGEMENT = "acknowledgement"
    ATTESTATION = "attestation"
    CORRECTION = "correction"


class SignatureMeaning(str, Enum):
    """Signature intent/meaning."""
    DATA_ENTRY = "data_entry"
    DATA_VERIFICATION = "data_verification"
    DATA_APPROVAL = "data_approval"
    MEDICAL_REVIEW = "medical_review"
    QUERY_RESPONSE = "query_response"
    ISSUE_RESOLUTION = "issue_resolution"


# =============================================================================
# AUDIT ENUMS
# =============================================================================

class AuditAction(str, Enum):
    """Audit trail action types."""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    LOGIN = "login"
    LOGOUT = "logout"
    EXPORT = "export"
    SIGN = "sign"
    APPROVE = "approve"
    REJECT = "reject"


class EntityType(str, Enum):
    """Auditable entity types."""
    PATIENT = "patient"
    SITE = "site"
    STUDY = "study"
    VISIT = "visit"
    ISSUE = "issue"
    QUERY = "query"
    LAB_RESULT = "lab_result"
    ADVERSE_EVENT = "adverse_event"
    SIGNATURE = "signature"
    USER = "user"
    REPORT = "report"
    ML_MODEL = "ml_model"


# =============================================================================
# USER ENUMS
# =============================================================================

class UserRole(str, Enum):
    """User role types."""
    SYSTEM_ADMIN = "system_admin"
    STUDY_LEAD = "study_lead"
    CRA = "cra"  # Clinical Research Associate
    DATA_MANAGER = "data_manager"
    SAFETY_OFFICER = "safety_officer"
    SITE_COORDINATOR = "site_coordinator"
    MEDICAL_CODER = "medical_coder"
    AUDITOR = "auditor"
    VIEWER = "viewer"


class UserStatus(str, Enum):
    """User account status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    LOCKED = "locked"
    PENDING_VERIFICATION = "pending_verification"


# =============================================================================
# ML MODEL ENUMS
# =============================================================================

class ModelType(str, Enum):
    """ML model types."""
    RISK_CLASSIFIER = "risk_classifier"
    ISSUE_DETECTOR = "issue_detector"
    SITE_RANKER = "site_ranker"
    RESOLUTION_PREDICTOR = "resolution_predictor"
    ANOMALY_DETECTOR = "anomaly_detector"


class ModelStatus(str, Enum):
    """ML model deployment status."""
    TRAINING = "training"
    VALIDATING = "validating"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    FAILED = "failed"


class DriftSeverity(str, Enum):
    """Model drift severity levels."""
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
