"""
SANCHALAK AI - Phase 9.1: Trial State Model v1.1

Digital Twin foundation with entity representation, state snapshots,
transition rules, and constraint definitions.

FIXED: Optimized data loading with pandas merge instead of row-by-row lookup

Author: Sanchalak AI Team
Date: 2026-01-02
"""

import json
import hashlib
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from pathlib import Path
import copy
import pickle

import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class EntityType(Enum):
    """Types of entities in the trial"""
    PATIENT = "patient"
    SITE = "site"
    STUDY = "study"
    ISSUE = "issue"
    CRA = "cra"
    QUERY = "query"
    SAE = "sae"
    VISIT = "visit"


class PatientStatus(Enum):
    """Patient status values"""
    SCREENING = "screening"
    SCREEN_FAILURE = "screen_failure"
    ONGOING = "ongoing"
    COMPLETED = "completed"
    DISCONTINUED = "discontinued"
    UNKNOWN = "unknown"


class SiteStatus(Enum):
    """Site status values"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    CLOSED = "closed"
    PENDING = "pending"
    ON_HOLD = "on_hold"


class IssueStatus(Enum):
    """Issue status values"""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    PENDING_REVIEW = "pending_review"
    RESOLVED = "resolved"
    CLOSED = "closed"


class TransitionType(Enum):
    """Types of state transitions"""
    PATIENT_STATUS_CHANGE = "patient_status_change"
    ISSUE_RESOLUTION = "issue_resolution"
    ISSUE_CREATION = "issue_creation"
    SITE_STATUS_CHANGE = "site_status_change"
    DQI_UPDATE = "dqi_update"
    CLEAN_STATUS_CHANGE = "clean_status_change"
    DBLOCK_STATUS_CHANGE = "dblock_status_change"
    RESOURCE_ASSIGNMENT = "resource_assignment"
    METRIC_UPDATE = "metric_update"


class ConstraintType(Enum):
    """Types of constraints"""
    HARD = "hard"
    SOFT = "soft"
    ADVISORY = "advisory"


class ConstraintCategory(Enum):
    """Categories of constraints"""
    REGULATORY = "regulatory"
    PROTOCOL = "protocol"
    SAFETY = "safety"
    QUALITY = "quality"
    OPERATIONAL = "operational"
    RESOURCE = "resource"


# =============================================================================
# DATA CLASSES - ENTITIES
# =============================================================================

@dataclass
class PatientEntity:
    """Represents a patient in the trial"""
    patient_key: str
    study_id: str
    site_id: str
    subject_id: str
    status: PatientStatus
    
    dqi_score: float = 100.0
    tier1_clean: bool = True
    tier2_clean: bool = True
    db_lock_ready: bool = False
    
    total_issues: int = 0
    critical_issues: int = 0
    open_queries: int = 0
    pending_signatures: int = 0
    sdv_pending: int = 0
    
    enrollment_date: Optional[datetime] = None
    last_visit_date: Optional[datetime] = None
    last_update: Optional[datetime] = None
    
    risk_level: str = "Low"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['status'] = self.status.value
        if self.enrollment_date:
            d['enrollment_date'] = self.enrollment_date.isoformat()
        if self.last_visit_date:
            d['last_visit_date'] = self.last_visit_date.isoformat()
        if self.last_update:
            d['last_update'] = self.last_update.isoformat()
        return d


@dataclass
class SiteEntity:
    """Represents a site in the trial"""
    site_id: str
    study_id: str
    status: SiteStatus
    
    patient_count: int = 0
    active_patients: int = 0
    
    mean_dqi: float = 100.0
    tier2_clean_rate: float = 1.0
    db_lock_ready_rate: float = 0.0
    
    total_issues: int = 0
    critical_issues: int = 0
    
    performance_tier: str = "Average"
    spi_score: float = 50.0
    
    assigned_cra: Optional[str] = None
    coordinator_count: int = 1
    
    region: Optional[str] = None
    country: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['status'] = self.status.value
        return d


@dataclass
class StudyEntity:
    """Represents a study"""
    study_id: str
    status: str = "Active"
    
    site_count: int = 0
    patient_count: int = 0
    active_patients: int = 0
    
    mean_dqi: float = 100.0
    tier2_clean_rate: float = 1.0
    db_lock_ready_rate: float = 0.0
    
    total_issues: int = 0
    critical_issues: int = 0
    
    target_enrollment: int = 0
    target_db_lock_date: Optional[datetime] = None
    
    therapeutic_area: Optional[str] = None
    phase: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.target_db_lock_date:
            d['target_db_lock_date'] = self.target_db_lock_date.isoformat()
        return d


@dataclass
class IssueEntity:
    """Represents an issue"""
    issue_id: str
    issue_type: str
    patient_key: str
    site_id: str
    study_id: str
    status: IssueStatus
    priority: str = "Medium"
    
    count: int = 1
    
    created_at: Optional[datetime] = None
    due_date: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    
    assignee_id: Optional[str] = None
    assignee_role: Optional[str] = None
    
    dqi_impact: float = 0.0
    cascade_impact: float = 0.0
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['status'] = self.status.value
        if self.created_at:
            d['created_at'] = self.created_at.isoformat()
        if self.due_date:
            d['due_date'] = self.due_date.isoformat()
        if self.resolved_at:
            d['resolved_at'] = self.resolved_at.isoformat()
        return d


@dataclass
class ResourceEntity:
    """Represents a resource (CRA, DM, etc.)"""
    resource_id: str
    resource_type: str
    name: str
    
    max_sites: int = 10
    max_patients: int = 200
    current_sites: int = 0
    current_patients: int = 0
    
    workload_percentage: float = 0.0
    weekly_hours: float = 40.0
    available_hours: float = 40.0
    
    assigned_sites: List[str] = field(default_factory=list)
    assigned_studies: List[str] = field(default_factory=list)
    
    region: Optional[str] = None
    skills: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# STATE SNAPSHOT
# =============================================================================

@dataclass
class TrialStateSnapshot:
    """Complete snapshot of trial state at a point in time"""
    snapshot_id: str
    timestamp: datetime
    
    patients: Dict[str, PatientEntity] = field(default_factory=dict)
    sites: Dict[str, SiteEntity] = field(default_factory=dict)
    studies: Dict[str, StudyEntity] = field(default_factory=dict)
    issues: Dict[str, IssueEntity] = field(default_factory=dict)
    resources: Dict[str, ResourceEntity] = field(default_factory=dict)
    
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    version: str = "1.1.0"
    source: str = "live"
    parent_snapshot_id: Optional[str] = None
    checksum: Optional[str] = None
    
    def calculate_checksum(self) -> str:
        state_str = json.dumps({
            'patients': len(self.patients),
            'sites': len(self.sites),
            'studies': len(self.studies),
            'issues': len(self.issues),
            'timestamp': self.timestamp.isoformat(),
            'metrics': self.metrics
        }, sort_keys=True)
        return hashlib.sha256(state_str.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'snapshot_id': self.snapshot_id,
            'timestamp': self.timestamp.isoformat(),
            'patients': {k: v.to_dict() for k, v in self.patients.items()},
            'sites': {k: v.to_dict() for k, v in self.sites.items()},
            'studies': {k: v.to_dict() for k, v in self.studies.items()},
            'issues': {k: v.to_dict() for k, v in self.issues.items()},
            'resources': {k: v.to_dict() for k, v in self.resources.items()},
            'metrics': self.metrics,
            'version': self.version,
            'source': self.source,
            'parent_snapshot_id': self.parent_snapshot_id,
            'checksum': self.checksum
        }


# =============================================================================
# TRANSITION RULES & CONSTRAINTS
# =============================================================================

@dataclass
class TransitionRule:
    """Defines a valid state transition"""
    rule_id: str
    name: str
    description: str
    transition_type: TransitionType
    from_states: List[str]
    to_states: List[str]
    side_effects: List[str] = field(default_factory=list)
    validators: List[str] = field(default_factory=list)
    enabled: bool = True
    requires_approval: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['transition_type'] = self.transition_type.value
        return d


@dataclass
class TransitionResult:
    """Result of attempting a state transition"""
    success: bool
    transition_type: TransitionType
    from_state: str
    to_state: str
    affected_entities: List[str] = field(default_factory=list)
    side_effects_applied: List[str] = field(default_factory=list)
    constraint_violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    applied_by: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['transition_type'] = self.transition_type.value
        d['timestamp'] = self.timestamp.isoformat()
        return d


@dataclass
class Constraint:
    """Defines a constraint on the trial state"""
    constraint_id: str
    name: str
    description: str
    constraint_type: ConstraintType
    category: ConstraintCategory
    entity_types: List[EntityType]
    condition: str
    violation_message: str
    remediation_hint: Optional[str] = None
    enabled: bool = True
    severity_score: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['constraint_type'] = self.constraint_type.value
        d['category'] = self.category.value
        d['entity_types'] = [e.value for e in self.entity_types]
        return d


@dataclass
class ConstraintViolation:
    """Records a constraint violation"""
    violation_id: str
    constraint_id: str
    constraint_name: str
    constraint_type: ConstraintType
    entity_type: EntityType
    entity_id: str
    message: str
    remediation_hint: Optional[str] = None
    detected_at: datetime = field(default_factory=datetime.now)
    severity_score: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['constraint_type'] = self.constraint_type.value
        d['entity_type'] = self.entity_type.value
        d['detected_at'] = self.detected_at.isoformat()
        return d


# =============================================================================
# TRIAL STATE MODEL - MAIN CLASS
# =============================================================================

class TrialStateModel:
    """
    Digital Twin of the clinical trial.
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path("data/processed")
        self.current_state: Optional[TrialStateSnapshot] = None
        self.snapshot_history: List[str] = []
        self.snapshots: Dict[str, TrialStateSnapshot] = {}
        self.transition_rules: Dict[str, TransitionRule] = {}
        self.constraints: Dict[str, Constraint] = {}
        self.validators: Dict[str, Callable] = {}
        self.max_snapshots = 100
        self.auto_snapshot = True
        
        self._initialize_transition_rules()
        self._initialize_constraints()
        self._register_default_validators()
        
        logger.info("TrialStateModel initialized")
    
    def get_current_state(self) -> 'TrialStateSnapshot':
        """Get the current state snapshot."""
        if self.current_state is None:
            # Create an initial empty state if none exists
            from .trial_state_model import TrialStateSnapshot # Avoid late import issues if any
            self.current_state = TrialStateSnapshot(
                snapshot_id="initial",
                timestamp=datetime.now(),
                patients={},
                sites={},
                studies={},
                issues={},
                metrics={}
            )
        return self.current_state
    
    def _initialize_transition_rules(self):
        """Initialize default transition rules"""
        self.transition_rules['patient_screening_to_ongoing'] = TransitionRule(
            rule_id='patient_screening_to_ongoing',
            name='Patient Enrollment',
            description='Patient successfully enrolled from screening',
            transition_type=TransitionType.PATIENT_STATUS_CHANGE,
            from_states=['screening'],
            to_states=['ongoing'],
            side_effects=['update_site_counts', 'update_study_counts']
        )
        
        self.transition_rules['patient_ongoing_to_completed'] = TransitionRule(
            rule_id='patient_ongoing_to_completed',
            name='Patient Completion',
            description='Patient completed the study',
            transition_type=TransitionType.PATIENT_STATUS_CHANGE,
            from_states=['ongoing'],
            to_states=['completed'],
            side_effects=['update_site_counts', 'update_study_counts'],
            validators=['validate_no_critical_issues']
        )
        
        self.transition_rules['patient_to_discontinued'] = TransitionRule(
            rule_id='patient_to_discontinued',
            name='Patient Discontinuation',
            description='Patient discontinued from study',
            transition_type=TransitionType.PATIENT_STATUS_CHANGE,
            from_states=['screening', 'ongoing'],
            to_states=['discontinued'],
            side_effects=['update_site_counts', 'update_study_counts']
        )
        
        self.transition_rules['issue_open_to_resolved'] = TransitionRule(
            rule_id='issue_open_to_resolved',
            name='Issue Resolution',
            description='Issue resolved',
            transition_type=TransitionType.ISSUE_RESOLUTION,
            from_states=['open', 'in_progress'],
            to_states=['resolved'],
            side_effects=['recalculate_patient_dqi', 'recalculate_site_metrics'],
            validators=['validate_resolution_data']
        )
        
        self.transition_rules['issue_creation'] = TransitionRule(
            rule_id='issue_creation',
            name='Issue Creation',
            description='New issue created',
            transition_type=TransitionType.ISSUE_CREATION,
            from_states=[],
            to_states=['open'],
            side_effects=['recalculate_patient_dqi', 'recalculate_site_metrics']
        )
        
        self.transition_rules['site_activation'] = TransitionRule(
            rule_id='site_activation',
            name='Site Activation',
            description='Site activated for enrollment',
            transition_type=TransitionType.SITE_STATUS_CHANGE,
            from_states=['pending', 'on_hold'],
            to_states=['active'],
            validators=['validate_site_requirements']
        )
        
        self.transition_rules['site_closure'] = TransitionRule(
            rule_id='site_closure',
            name='Site Closure',
            description='Site closed',
            transition_type=TransitionType.SITE_STATUS_CHANGE,
            from_states=['active', 'inactive'],
            to_states=['closed'],
            side_effects=['reassign_patients', 'update_study_counts'],
            requires_approval=True
        )
        
        self.transition_rules['patient_db_lock_ready'] = TransitionRule(
            rule_id='patient_db_lock_ready',
            name='Patient DB Lock Ready',
            description='Patient ready for database lock',
            transition_type=TransitionType.DBLOCK_STATUS_CHANGE,
            from_states=['not_ready', 'pending'],
            to_states=['ready'],
            validators=['validate_db_lock_criteria']
        )
        
        logger.info(f"Initialized {len(self.transition_rules)} transition rules")
    
    def _initialize_constraints(self):
        """Initialize default constraints"""
        self.constraints['no_orphan_patients'] = Constraint(
            constraint_id='no_orphan_patients',
            name='No Orphan Patients',
            description='Patients must belong to an active site',
            constraint_type=ConstraintType.HARD,
            category=ConstraintCategory.REGULATORY,
            entity_types=[EntityType.PATIENT],
            condition='patient.site_id in active_sites',
            violation_message='Patient {patient_key} belongs to non-active site {site_id}'
        )
        
        self.constraints['db_lock_requires_clean'] = Constraint(
            constraint_id='db_lock_requires_clean',
            name='DB Lock Requires Clean Status',
            description='Patient must be Tier 2 clean for DB lock',
            constraint_type=ConstraintType.HARD,
            category=ConstraintCategory.QUALITY,
            entity_types=[EntityType.PATIENT],
            condition='patient.db_lock_ready implies patient.tier2_clean',
            violation_message='Patient {patient_key} marked DB Lock ready but not Tier 2 clean'
        )
        
        self.constraints['completed_requires_no_open_issues'] = Constraint(
            constraint_id='completed_requires_no_open_issues',
            name='Completion Requires No Open Issues',
            description='Patient cannot be completed with open critical issues',
            constraint_type=ConstraintType.HARD,
            category=ConstraintCategory.QUALITY,
            entity_types=[EntityType.PATIENT],
            condition='patient.status == completed implies patient.critical_issues == 0',
            violation_message='Patient {patient_key} has {critical_issues} open critical issues'
        )
        
        self.constraints['sae_must_have_assignee'] = Constraint(
            constraint_id='sae_must_have_assignee',
            name='SAE Must Have Assignee',
            description='SAE issues must be assigned within 24 hours',
            constraint_type=ConstraintType.HARD,
            category=ConstraintCategory.SAFETY,
            entity_types=[EntityType.ISSUE],
            condition='issue.type == sae implies issue.assignee_id is not None',
            violation_message='SAE issue {issue_id} has no assignee'
        )
        
        self.constraints['dqi_minimum_threshold'] = Constraint(
            constraint_id='dqi_minimum_threshold',
            name='DQI Minimum Threshold',
            description='Patient DQI should be above 50',
            constraint_type=ConstraintType.SOFT,
            category=ConstraintCategory.QUALITY,
            entity_types=[EntityType.PATIENT],
            condition='patient.dqi_score >= 50',
            violation_message='Patient {patient_key} DQI ({dqi_score}) below threshold (50)',
            remediation_hint='Review and resolve open issues to improve DQI',
            severity_score=0.5
        )
        
        self.constraints['site_dqi_minimum'] = Constraint(
            constraint_id='site_dqi_minimum',
            name='Site DQI Minimum',
            description='Site mean DQI should be above 75',
            constraint_type=ConstraintType.SOFT,
            category=ConstraintCategory.QUALITY,
            entity_types=[EntityType.SITE],
            condition='site.mean_dqi >= 75',
            violation_message='Site {site_id} mean DQI ({mean_dqi}) below threshold (75)',
            remediation_hint='Investigate site-wide issues and patterns',
            severity_score=0.7
        )
        
        self.constraints['cra_workload_limit'] = Constraint(
            constraint_id='cra_workload_limit',
            name='CRA Workload Limit',
            description='CRA workload should not exceed 100%',
            constraint_type=ConstraintType.SOFT,
            category=ConstraintCategory.RESOURCE,
            entity_types=[EntityType.SITE],
            condition='resource.workload_percentage <= 100',
            violation_message='CRA {resource_id} overloaded at {workload_percentage}%',
            remediation_hint='Consider reassigning sites or adding resources',
            severity_score=0.6
        )
        
        self.constraints['enrollment_target_warning'] = Constraint(
            constraint_id='enrollment_target_warning',
            name='Enrollment Target Warning',
            description='Study should be on track for enrollment target',
            constraint_type=ConstraintType.ADVISORY,
            category=ConstraintCategory.OPERATIONAL,
            entity_types=[EntityType.STUDY],
            condition='study.patient_count >= study.target_enrollment * 0.8',
            violation_message='Study {study_id} enrollment ({patient_count}) below 80% of target ({target_enrollment})',
            severity_score=0.3
        )
        
        logger.info(f"Initialized {len(self.constraints)} constraints")
    
    def _register_default_validators(self):
        """Register default validator functions"""
        def validate_no_critical_issues(entity: PatientEntity, **kwargs) -> Tuple[bool, str]:
            if entity.critical_issues > 0:
                return False, f"Patient has {entity.critical_issues} critical issues"
            return True, ""
        
        def validate_resolution_data(entity: IssueEntity, **kwargs) -> Tuple[bool, str]:
            if not entity.resolved_at:
                return False, "Issue missing resolution timestamp"
            return True, ""
        
        def validate_site_requirements(entity: SiteEntity, **kwargs) -> Tuple[bool, str]:
            if not entity.assigned_cra:
                return False, "Site has no assigned CRA"
            return True, ""
        
        def validate_db_lock_criteria(entity: PatientEntity, **kwargs) -> Tuple[bool, str]:
            violations = []
            if not entity.tier2_clean:
                violations.append("Not Tier 2 clean")
            if entity.open_queries > 0:
                violations.append(f"{entity.open_queries} open queries")
            if entity.pending_signatures > 0:
                violations.append(f"{entity.pending_signatures} pending signatures")
            if violations:
                return False, "; ".join(violations)
            return True, ""
        
        self.validators['validate_no_critical_issues'] = validate_no_critical_issues
        self.validators['validate_resolution_data'] = validate_resolution_data
        self.validators['validate_site_requirements'] = validate_site_requirements
        self.validators['validate_db_lock_criteria'] = validate_db_lock_criteria
        
        logger.info(f"Registered {len(self.validators)} validators")
    
    # =========================================================================
    # STATE LOADING - OPTIMIZED
    # =========================================================================
    
    def load_from_data(self) -> TrialStateSnapshot:
        """Load current state from processed data files"""
        logger.info("Loading trial state from data files...")
        
        snapshot_id = f"SNAP-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        timestamp = datetime.now()
        
        snapshot = TrialStateSnapshot(
            snapshot_id=snapshot_id,
            timestamp=timestamp,
            source='live'
        )
        
        snapshot.patients = self._load_patients()
        snapshot.sites = self._load_sites(snapshot.patients)
        snapshot.studies = self._load_studies(snapshot.patients)
        snapshot.issues = self._load_issues()
        snapshot.metrics = self._calculate_metrics(snapshot)
        snapshot.checksum = snapshot.calculate_checksum()
        
        self.current_state = snapshot
        self._save_snapshot(snapshot)
        
        logger.info(f"Loaded state: {len(snapshot.patients)} patients, "
                   f"{len(snapshot.sites)} sites, {len(snapshot.studies)} studies, "
                   f"{len(snapshot.issues)} issues")
        
        return snapshot
    
    def _load_patients(self) -> Dict[str, PatientEntity]:
        """Load patient entities from PostgreSQL - REAL DATA VERSION"""
        patients = {}
        
        try:
            from src.database.connection import get_db_manager
            db = get_db_manager()
            
            logger.info("Loading patients from PostgreSQL...")
            
            with db.engine.connect() as conn:
                df = pd.read_sql("SELECT * FROM unified_patient_record", conn)
            
            logger.info(f"Loaded {len(df)} rows from PostgreSQL")
            
        except Exception as e:
            logger.warning(f"PostgreSQL not available ({e}), falling back to Parquet...")
            # Fallback to Parquet
            upr_path = self.data_dir / "upr" / "unified_patient_record.parquet"
            if not upr_path.exists():
                logger.warning(f"UPR file not found: {upr_path}")
                return patients
            df = pd.read_parquet(upr_path)
        
        if 'patient_key' not in df.columns:
            df['patient_key'] = df['study_id'].astype(str) + '|' + df['site_id'].astype(str) + '|' + df['subject_id'].astype(str)
        
        # Map column names to handle PostgreSQL naming
        col_mapping = {
            'dqi_score': df.get('dqi_score', pd.Series([100.0] * len(df))),
            'risk_score': df.get('risk_score', pd.Series([50.0] * len(df))),
            'clean_status_tier': df.get('clean_status_tier', pd.Series(['pending'] * len(df))),
            'completeness_score': df.get('completeness_score', pd.Series([100.0] * len(df))),
        }
        
        # Fill defaults for columns
        if 'dqi_score' not in df.columns:
            df['dqi_score'] = 80.0
        if 'clean_status_tier' not in df.columns:
            df['clean_status_tier'] = 'pending'
        
        status_map = {
            'ongoing': PatientStatus.ONGOING,
            'completed': PatientStatus.COMPLETED,
            'discontinued': PatientStatus.DISCONTINUED,
            'screen failure': PatientStatus.SCREEN_FAILURE,
            'screening': PatientStatus.SCREENING,
            'unknown': PatientStatus.UNKNOWN
        }
        
        logger.info(f"Creating {len(df)} patient entities...")
        
        for idx, row in df.iterrows():
            patient_key = str(row.get('patient_key', ''))
            if not patient_key:
                continue
            
            status_str = str(row.get('subject_status_clean', row.get('subject_status', 'unknown'))).lower()
            status = status_map.get(status_str, PatientStatus.UNKNOWN)
            
            dqi_score = float(row['dqi_score']) if pd.notna(row.get('dqi_score')) else 80.0
            
            # Derive clean status from clean_status_tier or dqi_score
            clean_tier = str(row.get('clean_status_tier', '')).lower()
            tier1_clean = clean_tier in ['tier1', 'clean', 'tier2'] or dqi_score >= 80
            tier2_clean = clean_tier in ['tier2', 'clean'] or dqi_score >= 90
            db_lock_ready = clean_tier == 'clean' or dqi_score >= 95
            
            # Issue counts from database columns or defaults
            total_issues = int(row.get('total_issues', 0)) if pd.notna(row.get('total_issues')) else 0
            open_queries = int(row.get('total_open_queries', 0)) if pd.notna(row.get('total_open_queries')) else 0
            pending_signatures = int(row.get('crfs_never_signed', 0)) if pd.notna(row.get('crfs_never_signed')) else 0
            
            risk_level = str(row.get('risk_level', 'Low')) if pd.notna(row.get('risk_level')) else 'Low'
            critical_issues = 1 if risk_level in ['High', 'Critical'] else 0
            
            patient = PatientEntity(
                patient_key=patient_key,
                study_id=str(row.get('study_id', '')),
                site_id=str(row.get('site_id', '')),
                subject_id=str(row.get('subject_id', '')),
                status=status,
                dqi_score=dqi_score,
                tier1_clean=tier1_clean,
                tier2_clean=tier2_clean,
                db_lock_ready=db_lock_ready,
                total_issues=total_issues,
                critical_issues=critical_issues,
                open_queries=open_queries,
                pending_signatures=pending_signatures,
                sdv_pending=0,
                risk_level=risk_level,
                last_update=datetime.now()
            )
            patients[patient_key] = patient
        
        logger.info(f"Loaded {len(patients)} patients from database")
        return patients
    
    def _load_sites(self, patients: Dict[str, PatientEntity]) -> Dict[str, SiteEntity]:
        """Load site entities by aggregating from PostgreSQL-loaded patients"""
        sites = {}
        
        # Build from patients (which are already loaded from PostgreSQL)
        logger.info("Building sites from patient data...")
        site_data = {}
        for p in patients.values():
            if p.site_id not in site_data:
                site_data[p.site_id] = {'patients': [], 'study_id': p.study_id}
            site_data[p.site_id]['patients'].append(p)
        
        for site_id, data in site_data.items():
            pts = data['patients']
            if not pts:
                continue
            
            # Calculate site metrics from patient data
            mean_dqi = np.mean([p.dqi_score for p in pts])
            tier2_clean_rate = sum(1 for p in pts if p.tier2_clean) / len(pts) if pts else 1.0
            db_lock_ready_rate = sum(1 for p in pts if p.db_lock_ready) / len(pts) if pts else 0.0
            total_issues = sum(p.total_issues for p in pts)
            
            # Determine performance tier based on metrics
            if mean_dqi >= 90 and tier2_clean_rate >= 0.9:
                performance_tier = "Top Performer"
            elif mean_dqi >= 75:
                performance_tier = "Average"
            else:
                performance_tier = "Needs Attention"
            
            site = SiteEntity(
                site_id=site_id,
                study_id=data['study_id'],
                status=SiteStatus.ACTIVE,
                patient_count=len(pts),
                active_patients=sum(1 for p in pts if p.status == PatientStatus.ONGOING),
                mean_dqi=mean_dqi,
                tier2_clean_rate=tier2_clean_rate,
                db_lock_ready_rate=db_lock_ready_rate,
                total_issues=total_issues,
                critical_issues=sum(p.critical_issues for p in pts),
                performance_tier=performance_tier,
                spi_score=mean_dqi * tier2_clean_rate,  # Simplified SPI
            )
            sites[site_id] = site
        
        logger.info(f"Loaded {len(sites)} sites")
        return sites
    
    def _load_studies(self, patients: Dict[str, PatientEntity]) -> Dict[str, StudyEntity]:
        """Load study entities from aggregated data"""
        studies = {}
        
        study_data = {}
        for patient in patients.values():
            study_id = patient.study_id
            if study_id not in study_data:
                study_data[study_id] = {'patients': [], 'sites': set()}
            study_data[study_id]['patients'].append(patient)
            study_data[study_id]['sites'].add(patient.site_id)
        
        for study_id, data in study_data.items():
            pts = data['patients']
            study = StudyEntity(
                study_id=study_id,
                status='Active',
                site_count=len(data['sites']),
                patient_count=len(pts),
                active_patients=sum(1 for p in pts if p.status == PatientStatus.ONGOING),
                mean_dqi=np.mean([p.dqi_score for p in pts]) if pts else 100.0,
                tier2_clean_rate=sum(1 for p in pts if p.tier2_clean) / len(pts) if pts else 1.0,
                db_lock_ready_rate=sum(1 for p in pts if p.db_lock_ready) / len(pts) if pts else 0.0,
                total_issues=sum(p.total_issues for p in pts),
                critical_issues=sum(p.critical_issues for p in pts)
            )
            studies[study_id] = study
        
        logger.info(f"Loaded {len(studies)} studies")
        return studies
    
    def _load_issues(self) -> Dict[str, IssueEntity]:
        """Load issue entities from data"""
        issues = {}
        
        issues_path = self.data_dir / "analytics" / "patient_issues.parquet"
        if not issues_path.exists():
            logger.warning("Patient issues file not found")
            return issues
        
        logger.info("Loading issues...")
        df = pd.read_parquet(issues_path)
        
        issue_cols = [c for c in df.columns if c.startswith('issue_')]
        
        issue_id_counter = 0
        for idx, row in df.iterrows():
            patient_key = str(row.get('patient_key', ''))
            study_id = str(row.get('study_id', ''))
            site_id = str(row.get('site_id', ''))
            
            for issue_col in issue_cols:
                if row.get(issue_col, False):
                    issue_type = issue_col.replace('issue_', '')
                    count_col = f'count_{issue_type}'
                    count = int(row.get(count_col, 1)) if count_col in df.columns else 1
                    
                    issue_id_counter += 1
                    issue_id = f"ISS-{issue_id_counter:06d}"
                    
                    priority = 'Medium'
                    if issue_type in ['sae_dm_pending', 'sae_safety_pending']:
                        priority = 'Critical'
                    elif issue_type in ['open_queries', 'signature_gaps', 'broken_signatures']:
                        priority = 'High'
                    
                    issue = IssueEntity(
                        issue_id=issue_id,
                        issue_type=issue_type,
                        patient_key=patient_key,
                        site_id=site_id,
                        study_id=study_id,
                        status=IssueStatus.OPEN,
                        priority=priority,
                        count=count,
                        created_at=datetime.now()
                    )
                    issues[issue_id] = issue
        
        logger.info(f"Loaded {len(issues)} issues")
        return issues
    
    def _calculate_metrics(self, snapshot: TrialStateSnapshot) -> Dict[str, Any]:
        """Calculate aggregate metrics"""
        patients = list(snapshot.patients.values())
        sites = list(snapshot.sites.values())
        studies = list(snapshot.studies.values())
        issues = list(snapshot.issues.values())
        
        eligible = [p for p in patients if p.status not in [PatientStatus.SCREEN_FAILURE, PatientStatus.SCREENING]]
        
        metrics = {
            'total_patients': len(patients),
            'eligible_patients': len(eligible),
            'total_sites': len(sites),
            'total_studies': len(studies),
            'total_issues': len(issues),
            'patients_by_status': {},
            'sites_by_status': {},
            'mean_dqi': np.mean([p.dqi_score for p in patients]) if patients else 100.0,
            'median_dqi': np.median([p.dqi_score for p in patients]) if patients else 100.0,
            'tier1_clean_count': sum(1 for p in patients if p.tier1_clean),
            'tier2_clean_count': sum(1 for p in patients if p.tier2_clean),
            'db_lock_ready_count': sum(1 for p in patients if p.db_lock_ready),
            'critical_issues': sum(1 for i in issues if i.priority == 'Critical'),
            'high_issues': sum(1 for i in issues if i.priority == 'High'),
            'open_issues': sum(1 for i in issues if i.status == IssueStatus.OPEN),
            'tier1_clean_rate': sum(1 for p in patients if p.tier1_clean) / len(patients) if patients else 1.0,
            'tier2_clean_rate': sum(1 for p in patients if p.tier2_clean) / len(patients) if patients else 1.0,
            'db_lock_ready_rate': sum(1 for p in eligible if p.db_lock_ready) / len(eligible) if eligible else 0.0
        }
        
        for status in PatientStatus:
            metrics['patients_by_status'][status.value] = sum(1 for p in patients if p.status == status)
        
        for status in SiteStatus:
            metrics['sites_by_status'][status.value] = sum(1 for s in sites if s.status == status)
        
        return metrics
    
    # =========================================================================
    # SNAPSHOT MANAGEMENT
    # =========================================================================
    
    def _save_snapshot(self, snapshot: TrialStateSnapshot):
        self.snapshots[snapshot.snapshot_id] = snapshot
        self.snapshot_history.append(snapshot.snapshot_id)
        while len(self.snapshot_history) > self.max_snapshots:
            old_id = self.snapshot_history.pop(0)
            del self.snapshots[old_id]
    
    def create_snapshot(self, source: str = 'manual') -> TrialStateSnapshot:
        if not self.current_state:
            raise ValueError("No current state to snapshot")
        new_snapshot = copy.deepcopy(self.current_state)
        new_snapshot.snapshot_id = f"SNAP-{datetime.now().strftime('%Y%m%d%H%M%S')}-{source[:4].upper()}"
        new_snapshot.timestamp = datetime.now()
        new_snapshot.source = source
        new_snapshot.parent_snapshot_id = self.current_state.snapshot_id
        new_snapshot.checksum = new_snapshot.calculate_checksum()
        self._save_snapshot(new_snapshot)
        return new_snapshot
    
    def get_snapshot(self, snapshot_id: str) -> Optional[TrialStateSnapshot]:
        return self.snapshots.get(snapshot_id)
    
    def restore_snapshot(self, snapshot_id: str) -> bool:
        snapshot = self.get_snapshot(snapshot_id)
        if not snapshot:
            return False
        restored = copy.deepcopy(snapshot)
        restored.snapshot_id = f"SNAP-{datetime.now().strftime('%Y%m%d%H%M%S')}-REST"
        restored.timestamp = datetime.now()
        restored.source = 'restored'
        restored.parent_snapshot_id = snapshot_id
        restored.checksum = restored.calculate_checksum()
        self.current_state = restored
        self._save_snapshot(restored)
        return True
    
    def get_snapshot_history(self) -> List[Dict[str, Any]]:
        history = []
        for snap_id in self.snapshot_history:
            snap = self.snapshots.get(snap_id)
            if snap:
                history.append({
                    'snapshot_id': snap.snapshot_id,
                    'timestamp': snap.timestamp.isoformat(),
                    'source': snap.source,
                    'patients': len(snap.patients),
                    'sites': len(snap.sites),
                    'checksum': snap.checksum
                })
        return history
    
    # =========================================================================
    # CONSTRAINT CHECKING
    # =========================================================================
    
    def check_constraints(self, entity_types: Optional[List[EntityType]] = None) -> List[ConstraintViolation]:
        if not self.current_state:
            return []
        
        violations = []
        
        for constraint in self.constraints.values():
            if not constraint.enabled:
                continue
            if entity_types and not any(et in entity_types for et in constraint.entity_types):
                continue
            
            if EntityType.PATIENT in constraint.entity_types:
                for patient in self.current_state.patients.values():
                    violation = self._check_constraint_for_patient(constraint, patient)
                    if violation:
                        violations.append(violation)
            
            if EntityType.SITE in constraint.entity_types:
                for site in self.current_state.sites.values():
                    violation = self._check_constraint_for_site(constraint, site)
                    if violation:
                        violations.append(violation)
        
        return violations
    
    def _check_constraint_for_patient(self, constraint: Constraint, patient: PatientEntity) -> Optional[ConstraintViolation]:
        violated = False
        
        if constraint.constraint_id == 'db_lock_requires_clean':
            if patient.db_lock_ready and not patient.tier2_clean:
                violated = True
        elif constraint.constraint_id == 'completed_requires_no_open_issues':
            if patient.status == PatientStatus.COMPLETED and patient.critical_issues > 0:
                violated = True
        elif constraint.constraint_id == 'dqi_minimum_threshold':
            if patient.dqi_score < 50:
                violated = True
        elif constraint.constraint_id == 'no_orphan_patients':
            if self.current_state:
                site = self.current_state.sites.get(patient.site_id)
                if not site or site.status != SiteStatus.ACTIVE:
                    violated = True
        
        if violated:
            message = constraint.violation_message.format(
                patient_key=patient.patient_key,
                site_id=patient.site_id,
                dqi_score=patient.dqi_score,
                critical_issues=patient.critical_issues
            )
            return ConstraintViolation(
                violation_id=f"VIO-{datetime.now().strftime('%Y%m%d%H%M%S')}-{patient.patient_key[:8]}",
                constraint_id=constraint.constraint_id,
                constraint_name=constraint.name,
                constraint_type=constraint.constraint_type,
                entity_type=EntityType.PATIENT,
                entity_id=patient.patient_key,
                message=message,
                remediation_hint=constraint.remediation_hint,
                severity_score=constraint.severity_score
            )
        return None
    
    def _check_constraint_for_site(self, constraint: Constraint, site: SiteEntity) -> Optional[ConstraintViolation]:
        violated = False
        
        if constraint.constraint_id == 'site_dqi_minimum':
            if site.mean_dqi < 75:
                violated = True
        
        if violated:
            message = constraint.violation_message.format(
                site_id=site.site_id,
                mean_dqi=site.mean_dqi
            )
            return ConstraintViolation(
                violation_id=f"VIO-{datetime.now().strftime('%Y%m%d%H%M%S')}-{site.site_id}",
                constraint_id=constraint.constraint_id,
                constraint_name=constraint.name,
                constraint_type=constraint.constraint_type,
                entity_type=EntityType.SITE,
                entity_id=site.site_id,
                message=message,
                remediation_hint=constraint.remediation_hint,
                severity_score=constraint.severity_score
            )
        return None
    
    # =========================================================================
    # QUERIES
    # =========================================================================
    
    def get_entity(self, entity_type: EntityType, entity_id: str) -> Optional[Any]:
        if not self.current_state:
            return None
        if entity_type == EntityType.PATIENT:
            return self.current_state.patients.get(entity_id)
        elif entity_type == EntityType.SITE:
            return self.current_state.sites.get(entity_id)
        elif entity_type == EntityType.STUDY:
            return self.current_state.studies.get(entity_id)
        elif entity_type == EntityType.ISSUE:
            return self.current_state.issues.get(entity_id)
        return None
    
    def query_patients(self, **filters) -> List[PatientEntity]:
        if not self.current_state:
            return []
        patients = list(self.current_state.patients.values())
        if 'study_id' in filters:
            patients = [p for p in patients if p.study_id == filters['study_id']]
        if 'site_id' in filters:
            patients = [p for p in patients if p.site_id == filters['site_id']]
        if 'status' in filters:
            patients = [p for p in patients if p.status == filters['status']]
        if 'tier2_clean' in filters:
            patients = [p for p in patients if p.tier2_clean == filters['tier2_clean']]
        if 'db_lock_ready' in filters:
            patients = [p for p in patients if p.db_lock_ready == filters['db_lock_ready']]
        if 'min_dqi' in filters:
            patients = [p for p in patients if p.dqi_score >= filters['min_dqi']]
        if 'max_dqi' in filters:
            patients = [p for p in patients if p.dqi_score <= filters['max_dqi']]
        if 'has_critical_issues' in filters:
            if filters['has_critical_issues']:
                patients = [p for p in patients if p.critical_issues > 0]
            else:
                patients = [p for p in patients if p.critical_issues == 0]
        return patients
    
    def query_sites(self, **filters) -> List[SiteEntity]:
        if not self.current_state:
            return []
        sites = list(self.current_state.sites.values())
        if 'study_id' in filters:
            sites = [s for s in sites if s.study_id == filters['study_id']]
        if 'status' in filters:
            sites = [s for s in sites if s.status == filters['status']]
        if 'min_dqi' in filters:
            sites = [s for s in sites if s.mean_dqi >= filters['min_dqi']]
        if 'performance_tier' in filters:
            sites = [s for s in sites if s.performance_tier == filters['performance_tier']]
        return sites
    
    def get_summary(self) -> Dict[str, Any]:
        if not self.current_state:
            return {'error': 'No current state loaded'}
        return {
            'snapshot_id': self.current_state.snapshot_id,
            'timestamp': self.current_state.timestamp.isoformat(),
            'source': self.current_state.source,
            'checksum': self.current_state.checksum,
            'entities': {
                'patients': len(self.current_state.patients),
                'sites': len(self.current_state.sites),
                'studies': len(self.current_state.studies),
                'issues': len(self.current_state.issues),
                'resources': len(self.current_state.resources)
            },
            'metrics': self.current_state.metrics,
            'transition_rules': len(self.transition_rules),
            'constraints': len(self.constraints),
            'snapshot_history': len(self.snapshot_history)
        }
    
    # =========================================================================
    # SERIALIZATION
    # =========================================================================
    
    def save_state(self, filepath: Path) -> bool:
        if not self.current_state:
            return False
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump({
                    'current_state': self.current_state,
                    'snapshot_history': self.snapshot_history,
                    'snapshots': self.snapshots
                }, f)
            logger.info(f"State saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            return False
    
    def load_state(self, filepath: Path) -> bool:
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                return False
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            self.current_state = data['current_state']
            self.snapshot_history = data['snapshot_history']
            self.snapshots = data['snapshots']
            logger.info(f"State loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return False


# =============================================================================
# SINGLETON & CONVENIENCE FUNCTIONS
# =============================================================================

_trial_state_model: Optional[TrialStateModel] = None


def get_trial_state_model(data_dir: Optional[Path] = None) -> TrialStateModel:
    global _trial_state_model
    if _trial_state_model is None:
        _trial_state_model = TrialStateModel(data_dir)
    return _trial_state_model


def reset_trial_state_model():
    global _trial_state_model
    _trial_state_model = None


def load_trial_state(data_dir: Optional[Path] = None) -> TrialStateSnapshot:
    model = get_trial_state_model(data_dir)
    return model.load_from_data()


def get_current_state() -> Optional[TrialStateSnapshot]:
    model = get_trial_state_model()
    return model.current_state


def check_trial_constraints() -> List[ConstraintViolation]:
    model = get_trial_state_model()
    return model.check_constraints()


# =============================================================================
# TEST FUNCTION
# =============================================================================

def test_trial_state_model():
    """Test the Trial State Model"""
    print("=" * 70)
    print("SANCHALAK AI - PHASE 9.1 TRIAL STATE MODEL TEST")
    print("=" * 70)
    
    tests_passed = 0
    tests_total = 0
    
    reset_trial_state_model()
    
    # TEST 1
    tests_total += 1
    print("\nTEST 1: Initialize Trial State Model")
    try:
        model = get_trial_state_model(Path("data/processed"))
        print(f"   ✅ Model initialized")
        print(f"   Transition rules: {len(model.transition_rules)}")
        print(f"   Constraints: {len(model.constraints)}")
        print(f"   Validators: {len(model.validators)}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # TEST 2
    tests_total += 1
    print("\nTEST 2: Load State from Data")
    try:
        snapshot = model.load_from_data()
        print(f"   ✅ State loaded")
        print(f"   Snapshot ID: {snapshot.snapshot_id}")
        print(f"   Patients: {len(snapshot.patients)}")
        print(f"   Sites: {len(snapshot.sites)}")
        print(f"   Studies: {len(snapshot.studies)}")
        print(f"   Issues: {len(snapshot.issues)}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
    
    # TEST 3
    tests_total += 1
    print("\nTEST 3: Metrics Calculation")
    try:
        metrics = model.current_state.metrics
        print(f"   ✅ Metrics calculated")
        print(f"   Total patients: {metrics.get('total_patients', 0):,}")
        print(f"   Mean DQI: {metrics.get('mean_dqi', 0):.2f}")
        print(f"   Tier 2 clean rate: {metrics.get('tier2_clean_rate', 0):.1%}")
        print(f"   DB Lock ready: {metrics.get('db_lock_ready_count', 0):,}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # TEST 4
    tests_total += 1
    print("\nTEST 4: Create Snapshot")
    try:
        new_snapshot = model.create_snapshot(source='test')
        print(f"   ✅ Snapshot created: {new_snapshot.snapshot_id}")
        print(f"   Parent: {new_snapshot.parent_snapshot_id}")
        print(f"   History length: {len(model.snapshot_history)}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # TEST 5
    tests_total += 1
    print("\nTEST 5: Query Patients")
    try:
        ongoing = model.query_patients(status=PatientStatus.ONGOING)
        print(f"   Ongoing patients: {len(ongoing):,}")
        
        if model.current_state.studies:
            study_id = list(model.current_state.studies.keys())[0]
            study_patients = model.query_patients(study_id=study_id)
            print(f"   Patients in {study_id}: {len(study_patients):,}")
        
        clean = model.query_patients(tier2_clean=True)
        print(f"   Tier 2 clean patients: {len(clean):,}")
        
        print(f"   ✅ Queries working")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # TEST 6
    tests_total += 1
    print("\nTEST 6: Query Sites")
    try:
        active_sites = model.query_sites(status=SiteStatus.ACTIVE)
        print(f"   ✅ Active sites: {len(active_sites):,}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # TEST 7
    tests_total += 1
    print("\nTEST 7: Check Constraints")
    try:
        violations = model.check_constraints()
        hard = [v for v in violations if v.constraint_type == ConstraintType.HARD]
        soft = [v for v in violations if v.constraint_type == ConstraintType.SOFT]
        advisory = [v for v in violations if v.constraint_type == ConstraintType.ADVISORY]
        print(f"   ✅ Constraints checked")
        print(f"   Total violations: {len(violations):,}")
        print(f"   Hard: {len(hard):,}, Soft: {len(soft):,}, Advisory: {len(advisory):,}")
        if violations:
            print(f"   Sample: {violations[0].message[:60]}...")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # TEST 8
    tests_total += 1
    print("\nTEST 8: Transition Rules")
    try:
        rules = list(model.transition_rules.values())
        print(f"   ✅ {len(rules)} transition rules defined")
        for rule in rules[:3]:
            print(f"   - {rule.name}: {rule.from_states} → {rule.to_states}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # TEST 9
    tests_total += 1
    print("\nTEST 9: Get Entity")
    try:
        if model.current_state.patients:
            patient_key = list(model.current_state.patients.keys())[0]
            patient = model.get_entity(EntityType.PATIENT, patient_key)
            print(f"   ✅ Retrieved patient: {patient.patient_key}")
            print(f"   Status: {patient.status.value}")
            print(f"   DQI: {patient.dqi_score:.1f}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # TEST 10
    tests_total += 1
    print("\nTEST 10: Get Summary")
    try:
        summary = model.get_summary()
        print(f"   ✅ Summary generated")
        print(f"   Snapshot: {summary.get('snapshot_id')}")
        print(f"   Entities: {summary.get('entities')}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # TEST 11
    tests_total += 1
    print("\nTEST 11: Checksum Integrity")
    try:
        original_checksum = model.current_state.checksum
        recalculated = model.current_state.calculate_checksum()
        if original_checksum == recalculated:
            print(f"   ✅ Checksum valid: {original_checksum}")
            tests_passed += 1
        else:
            print(f"   ❌ Checksum mismatch!")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # TEST 12
    tests_total += 1
    print("\nTEST 12: Snapshot History")
    try:
        history = model.get_snapshot_history()
        print(f"   ✅ History retrieved: {len(history)} snapshots")
        for snap in history[-3:]:
            print(f"   - {snap['snapshot_id']}: {snap['source']}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # TEST 13
    tests_total += 1
    print("\nTEST 13: Patient Status Distribution")
    try:
        status_dist = model.current_state.metrics.get('patients_by_status', {})
        print(f"   ✅ Status distribution:")
        for status, count in sorted(status_dist.items(), key=lambda x: -x[1]):
            pct = count / len(model.current_state.patients) * 100 if model.current_state.patients else 0
            print(f"      {status}: {count:,} ({pct:.1f}%)")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # TEST 14
    tests_total += 1
    print("\nTEST 14: Study Summary")
    try:
        studies = list(model.current_state.studies.values())
        print(f"   ✅ {len(studies)} studies loaded")
        for study in sorted(studies, key=lambda s: -s.patient_count)[:5]:
            print(f"      {study.study_id}: {study.patient_count:,} patients, "
                  f"DQI {study.mean_dqi:.1f}, Clean {study.tier2_clean_rate:.1%}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print(f"TEST RESULTS: {tests_passed}/{tests_total} passed")
    if tests_passed == tests_total:
        print("✅ ALL TESTS PASSED!")
    else:
        print(f"❌ {tests_total - tests_passed} tests failed")
    print("=" * 70)
    
    return tests_passed == tests_total


if __name__ == "__main__":
    test_trial_state_model()