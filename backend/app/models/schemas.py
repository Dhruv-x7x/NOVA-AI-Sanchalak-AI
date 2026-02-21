"""
Pydantic Models/Schemas for API
================================
Request and response models for the Sanchalak AI API.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# =============================================================================
# ENUMS
# =============================================================================

class UserRole(str, Enum):
    LEAD = "lead"
    DM = "dm"
    CRA = "cra"
    CODER = "coder"
    SAFETY = "safety"
    EXECUTIVE = "executive"


class IssueStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"


class IssuePriority(str, Enum):
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class ReportType(str, Enum):
    CRA_MONITORING = "cra_monitoring"
    SITE_PERFORMANCE = "site_performance"
    EXECUTIVE_BRIEF = "executive_brief"
    DB_LOCK_READINESS = "db_lock_readiness"
    QUERY_SUMMARY = "query_summary"
    SAFETY_NARRATIVE = "safety_narrative"
    PATIENT_RISK = "patient_risk"
    REGIONAL_SUMMARY = "regional_summary"
    CODING_STATUS = "coding_status"
    ENROLLMENT_TRACKER = "enrollment_tracker"


# =============================================================================
# AUTH SCHEMAS
# =============================================================================

class LoginRequest(BaseModel):
    username: Optional[str] = Field(None, min_length=2, max_length=50)
    email: Optional[str] = Field(None)
    password: str = Field(..., min_length=4)


class UserCreateRequest(BaseModel):
    username: str = Field(..., min_length=2, max_length=50)
    email: str
    password: str = Field(..., min_length=4)
    full_name: Optional[str] = None
    role: Optional[str] = "lead"


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: "UserResponse"


class RefreshRequest(BaseModel):
    refresh_token: str


class UserResponse(BaseModel):
    user_id: str
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    role: str
    permissions: List[str] = []


# =============================================================================
# PATIENT SCHEMAS
# =============================================================================

class PatientSummary(BaseModel):
    patient_key: str
    site_id: str
    study_id: Optional[str] = None
    status: Optional[str] = None
    risk_level: Optional[str] = None
    dqi_score: Optional[float] = None
    clean_status_tier: Optional[str] = None
    is_db_lock_ready: Optional[bool] = None


class PatientDetail(PatientSummary):
    enrollment_date: Optional[datetime] = None
    last_visit_date: Optional[datetime] = None
    next_visit_date: Optional[datetime] = None
    open_queries_count: Optional[int] = 0
    open_issues_count: Optional[int] = 0
    visit_compliance_pct: Optional[float] = None


class PatientListResponse(BaseModel):
    patients: List[Dict[str, Any]]
    items: List[Dict[str, Any]] = [] # For test compatibility
    data: List[Dict[str, Any]] = [] # For test compatibility
    total: int
    page: int = 1
    page_size: int = 50


class PatientSearchRequest(BaseModel):
    query: str
    limit: int = 20


# =============================================================================
# SITE SCHEMAS
# =============================================================================

class SiteSummary(BaseModel):
    site_id: str
    name: Optional[str] = None
    country: Optional[str] = None
    region: Optional[str] = None
    status: Optional[str] = None
    performance_score: Optional[float] = None
    dqi_score: Optional[float] = None
    patient_count: Optional[int] = 0


class SiteBenchmark(BaseModel):
    site_id: str
    patient_count: int
    dqi_score: float
    visit_compliance: float
    query_rate: float
    enrollment_rate: float
    cascade_mean: float
    avg_risk_score: float


class SiteListResponse(BaseModel):
    sites: List[Dict[str, Any]]
    items: List[Dict[str, Any]] = [] # For test compatibility
    data: List[Dict[str, Any]] = [] # For test compatibility
    total: int


# =============================================================================
# STUDY SCHEMAS
# =============================================================================

class StudySummary(BaseModel):
    study_id: str
    name: Optional[str] = None
    protocol_number: Optional[str] = None
    phase: Optional[str] = None
    status: Optional[str] = None
    therapeutic_area: Optional[str] = None
    sponsor: Optional[str] = None
    target_enrollment: Optional[int] = None
    current_enrollment: Optional[int] = None


class StudyListResponse(BaseModel):
    studies: List[Dict[str, Any]]
    total: int


# =============================================================================
# ANALYTICS SCHEMAS
# =============================================================================

class PortfolioSummary(BaseModel):
    total_patients: int
    total_sites: int
    total_studies: int
    total_issues: int
    mean_dqi: float
    dblock_ready_count: int
    dblock_ready_rate: float
    tier1_clean_count: int = 0
    tier1_clean_rate: float = 0.0
    tier2_clean_count: int
    tier2_clean_rate: float
    critical_issues: int
    high_issues: int


class DQIDistribution(BaseModel):
    dqi_band: str
    count: int
    percentage: float = 0.0


class RegionalMetric(BaseModel):
    region: str
    site_count: int
    avg_dqi: float
    avg_performance: float
    patient_count: int = 0


class TrendData(BaseModel):
    date: str
    value: float
    metric: str


# =============================================================================
# DIGITAL TWIN SCHEMAS
# =============================================================================

class PredictRequest(BaseModel):
    scenario_id: str
    entity_type: str
    entity_id: str
    params: Optional[Dict[str, Any]] = None


# =============================================================================
# ISSUE SCHEMAS
# =============================================================================

class IssueSummary(BaseModel):
    issue_id: Optional[int] = None
    patient_key: Optional[str] = None
    site_id: Optional[str] = None
    issue_type: Optional[str] = None
    priority: Optional[str] = None
    status: Optional[str] = None
    description: Optional[str] = None
    created_at: Optional[datetime] = None


class IssueCreateRequest(BaseModel):
    patient_key: str
    site_id: str
    issue_type: str
    priority: IssuePriority = IssuePriority.MEDIUM
    description: str


class IssueUpdateRequest(BaseModel):
    status: Optional[IssueStatus] = None
    priority: Optional[IssuePriority] = None
    resolution_notes: Optional[str] = None
    assigned_to: Optional[str] = None


class IssueListResponse(BaseModel):
    issues: List[Dict[str, Any]]
    total: int


# =============================================================================
# REPORT SCHEMAS
# =============================================================================

class ReportRequest(BaseModel):
    report_type: str
    site_id: Optional[str] = None
    study_id: Optional[str] = None
    view_level: str = "portfolio" # portfolio, study, site, country
    date_range_days: int = 30
    format: str = "xlsx"  # Default to xlsx


class ReportResponse(BaseModel):
    report_type: str
    generated_at: datetime
    content: str
    format: str
    metadata: Dict[str, Any] = {}


# =============================================================================
# ML GOVERNANCE SCHEMAS
# =============================================================================

class MLModelSummary(BaseModel):
    version_id: Optional[int] = None
    ml_model_name: str = ""
    ml_model_type: Optional[str] = None
    version: Optional[str] = None
    status: Optional[str] = None
    trained_at: Optional[datetime] = None
    deployed_at: Optional[datetime] = None
    training_samples: Optional[int] = None
    
    model_config = {"protected_namespaces": ()}


class MLModelListResponse(BaseModel):
    models: List[Dict[str, Any]]
    total: int


class MLModelApproveRequest(BaseModel):
    ml_model_id: int
    approved_by: str
    notes: Optional[str] = None
    
    model_config = {"protected_namespaces": ()}


# =============================================================================
# CASCADE ANALYSIS SCHEMAS
# =============================================================================

class CascadeImpact(BaseModel):
    patient_key: str
    site_id: str
    cascade_impact_score: float
    blocking_issues: int
    open_queries_count: int
    dqi_score: float


class CascadeAnalysisResponse(BaseModel):
    impacts: List[Dict[str, Any]]
    total: int
    high_risk_count: int


# =============================================================================
# PATTERN ALERT SCHEMAS
# =============================================================================

class PatternAlert(BaseModel):
    pattern_id: str
    pattern_name: str
    severity: str
    match_count: int
    sites_affected: int
    last_detected: Optional[datetime] = None
    status: str
    alert_message: Optional[str] = None


class PatternAlertListResponse(BaseModel):
    alerts: List[Dict[str, Any]]
    total: int


# Update forward references
TokenResponse.model_rebuild()
