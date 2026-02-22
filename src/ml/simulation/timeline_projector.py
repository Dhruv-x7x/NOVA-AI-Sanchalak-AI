"""
a6on-i - Phase 9.4: Timeline Projector v1.0

Projects trial timelines with trajectory analysis, best/worst case
scenarios, critical path identification, and milestone tracking.

Features:
- Current trajectory projection
- Best/worst case scenario modeling
- Critical path analysis (CPM)
- Milestone tracking with probability
- DB Lock date projection
- Gantt-style timeline visualization data

Author: a6on-i Team
Date: 2026-01-02
"""

import json
import hashlib
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any, Set
from enum import Enum
from pathlib import Path
import numpy as np

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


# =============================================================================
# ENUMS
# =============================================================================

class MilestoneType(Enum):
    """Types of trial milestones."""
    ENROLLMENT_START = "enrollment_start"
    ENROLLMENT_COMPLETE = "enrollment_complete"
    LAST_PATIENT_FIRST_VISIT = "lpfv"
    LAST_PATIENT_LAST_VISIT = "lplv"
    DATA_CLEANING_COMPLETE = "data_cleaning_complete"
    QUERY_RESOLUTION_COMPLETE = "query_resolution_complete"
    SDV_COMPLETE = "sdv_complete"
    CODING_COMPLETE = "coding_complete"
    SAE_RECONCILIATION_COMPLETE = "sae_reconciliation_complete"
    DATABASE_SOFT_LOCK = "db_soft_lock"
    DATABASE_HARD_LOCK = "db_hard_lock"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    CSR_COMPLETE = "csr_complete"
    STUDY_CLOSE = "study_close"


class MilestoneStatus(Enum):
    """Status of a milestone."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    AT_RISK = "at_risk"
    DELAYED = "delayed"
    COMPLETED = "completed"
    BLOCKED = "blocked"


class TrajectoryType(Enum):
    """Types of trajectory projections."""
    CURRENT = "current"
    BEST_CASE = "best_case"
    WORST_CASE = "worst_case"
    OPTIMISTIC = "optimistic"
    PESSIMISTIC = "pessimistic"
    PLANNED = "planned"


class RiskLevel(Enum):
    """Risk levels for timeline."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# =============================================================================
# CONFIGURATION
# =============================================================================

# Default milestone dependencies (what must complete before each milestone)
MILESTONE_DEPENDENCIES = {
    MilestoneType.ENROLLMENT_COMPLETE: [MilestoneType.ENROLLMENT_START],
    MilestoneType.LAST_PATIENT_FIRST_VISIT: [MilestoneType.ENROLLMENT_START],
    MilestoneType.LAST_PATIENT_LAST_VISIT: [MilestoneType.LAST_PATIENT_FIRST_VISIT],
    MilestoneType.QUERY_RESOLUTION_COMPLETE: [MilestoneType.LAST_PATIENT_LAST_VISIT],
    MilestoneType.SDV_COMPLETE: [MilestoneType.LAST_PATIENT_LAST_VISIT],
    MilestoneType.CODING_COMPLETE: [MilestoneType.LAST_PATIENT_LAST_VISIT],
    MilestoneType.SAE_RECONCILIATION_COMPLETE: [MilestoneType.LAST_PATIENT_LAST_VISIT],
    MilestoneType.DATA_CLEANING_COMPLETE: [
        MilestoneType.QUERY_RESOLUTION_COMPLETE,
        MilestoneType.SDV_COMPLETE,
        MilestoneType.CODING_COMPLETE,
    ],
    MilestoneType.DATABASE_SOFT_LOCK: [
        MilestoneType.DATA_CLEANING_COMPLETE,
        MilestoneType.SAE_RECONCILIATION_COMPLETE,
    ],
    MilestoneType.DATABASE_HARD_LOCK: [MilestoneType.DATABASE_SOFT_LOCK],
    MilestoneType.STATISTICAL_ANALYSIS: [MilestoneType.DATABASE_HARD_LOCK],
    MilestoneType.CSR_COMPLETE: [MilestoneType.STATISTICAL_ANALYSIS],
    MilestoneType.STUDY_CLOSE: [MilestoneType.CSR_COMPLETE],
}

# Default durations for milestones (in days)
DEFAULT_MILESTONE_DURATIONS = {
    MilestoneType.ENROLLMENT_START: 0,
    MilestoneType.ENROLLMENT_COMPLETE: 180,
    MilestoneType.LAST_PATIENT_FIRST_VISIT: 30,
    MilestoneType.LAST_PATIENT_LAST_VISIT: 365,
    MilestoneType.QUERY_RESOLUTION_COMPLETE: 30,
    MilestoneType.SDV_COMPLETE: 45,
    MilestoneType.CODING_COMPLETE: 14,
    MilestoneType.SAE_RECONCILIATION_COMPLETE: 21,
    MilestoneType.DATA_CLEANING_COMPLETE: 14,
    MilestoneType.DATABASE_SOFT_LOCK: 7,
    MilestoneType.DATABASE_HARD_LOCK: 14,
    MilestoneType.STATISTICAL_ANALYSIS: 60,
    MilestoneType.CSR_COMPLETE: 90,
    MilestoneType.STUDY_CLOSE: 30,
}

# Issue type to resolution rate (issues per day with current resources)
ISSUE_RESOLUTION_RATES = {
    'sdv_incomplete': 150,  # CRAs can SDV 150 CRFs/day
    'open_queries': 200,    # 200 queries resolved/day
    'signature_gaps': 100,  # 100 signatures/day
    'sae_dm_pending': 50,   # 50 SAE reconciliations/day
    'meddra_uncoded': 500,  # 500 terms coded/day
    'whodrug_uncoded': 500,
    'broken_signatures': 80,
    'missing_visits': 20,
    'missing_pages': 100,
    'lab_issues': 50,
    'edrr_issues': 30,
    'inactivated_forms': 100,
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Milestone:
    """Represents a trial milestone."""
    milestone_id: str
    milestone_type: MilestoneType
    name: str
    description: str = ""
    planned_date: Optional[datetime] = None
    projected_date: Optional[datetime] = None
    actual_date: Optional[datetime] = None
    status: MilestoneStatus = MilestoneStatus.NOT_STARTED
    progress_percent: float = 0.0
    dependencies: List[str] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)
    owner: str = ""
    study_id: Optional[str] = None
    
    @property
    def is_complete(self) -> bool:
        return self.status == MilestoneStatus.COMPLETED
    
    @property
    def is_on_track(self) -> bool:
        if self.is_complete:
            return True
        if not self.planned_date or not self.projected_date:
            return True
        return self.projected_date <= self.planned_date
    
    @property
    def days_variance(self) -> Optional[int]:
        """Days ahead (negative) or behind (positive) schedule."""
        if not self.planned_date or not self.projected_date:
            return None
        return (self.projected_date - self.planned_date).days
    
    @property
    def days_remaining(self) -> Optional[int]:
        """Days until projected completion."""
        if self.is_complete:
            return 0
        if not self.projected_date:
            return None
        return max(0, (self.projected_date - datetime.now()).days)
    
    def to_dict(self) -> Dict:
        return {
            'milestone_id': self.milestone_id,
            'milestone_type': self.milestone_type.value,
            'name': self.name,
            'description': self.description,
            'planned_date': self.planned_date.isoformat() if self.planned_date else None,
            'projected_date': self.projected_date.isoformat() if self.projected_date else None,
            'actual_date': self.actual_date.isoformat() if self.actual_date else None,
            'status': self.status.value,
            'progress_percent': self.progress_percent,
            'is_on_track': self.is_on_track,
            'days_variance': self.days_variance,
            'days_remaining': self.days_remaining,
            'dependencies': self.dependencies,
            'blockers': self.blockers,
            'owner': self.owner,
        }


@dataclass
class TrajectoryPoint:
    """A point on a trajectory timeline."""
    date: datetime
    metric_value: float
    metric_name: str
    trajectory_type: TrajectoryType
    confidence: float = 1.0
    
    def to_dict(self) -> Dict:
        return {
            'date': self.date.isoformat(),
            'metric_value': self.metric_value,
            'metric_name': self.metric_name,
            'trajectory_type': self.trajectory_type.value,
            'confidence': self.confidence,
        }


@dataclass
class Trajectory:
    """A complete trajectory projection."""
    trajectory_id: str
    trajectory_type: TrajectoryType
    metric_name: str
    start_date: datetime
    end_date: datetime
    points: List[TrajectoryPoint]
    start_value: float
    end_value: float
    daily_rate: float
    confidence_band: Tuple[float, float]  # (lower, upper) multipliers
    
    def get_value_at_date(self, target_date: datetime) -> float:
        """Interpolate value at a specific date."""
        if target_date <= self.start_date:
            return self.start_value
        if target_date >= self.end_date:
            return self.end_value
        
        days_elapsed = (target_date - self.start_date).days
        return self.start_value + (self.daily_rate * days_elapsed)
    
    def to_dict(self) -> Dict:
        return {
            'trajectory_id': self.trajectory_id,
            'trajectory_type': self.trajectory_type.value,
            'metric_name': self.metric_name,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'points': [p.to_dict() for p in self.points],
            'start_value': self.start_value,
            'end_value': self.end_value,
            'daily_rate': self.daily_rate,
            'confidence_band': self.confidence_band,
        }


@dataclass
class CriticalPathNode:
    """A node in the critical path."""
    node_id: str
    name: str
    duration_days: int
    earliest_start: int  # Days from project start
    earliest_finish: int
    latest_start: int
    latest_finish: int
    slack: int  # Float time
    is_critical: bool
    dependencies: List[str]
    
    def to_dict(self) -> Dict:
        return {
            'node_id': self.node_id,
            'name': self.name,
            'duration_days': self.duration_days,
            'earliest_start': self.earliest_start,
            'earliest_finish': self.earliest_finish,
            'latest_start': self.latest_start,
            'latest_finish': self.latest_finish,
            'slack': self.slack,
            'is_critical': self.is_critical,
            'dependencies': self.dependencies,
        }


@dataclass
class CriticalPath:
    """Critical path analysis result."""
    path_id: str
    created_at: datetime
    nodes: List[CriticalPathNode]
    critical_nodes: List[str]
    total_duration_days: int
    project_end_date: datetime
    bottlenecks: List[str]
    risk_areas: List[Dict]
    
    @property
    def critical_path_names(self) -> List[str]:
        return [n.name for n in self.nodes if n.is_critical]
    
    def to_dict(self) -> Dict:
        return {
            'path_id': self.path_id,
            'created_at': self.created_at.isoformat(),
            'nodes': [n.to_dict() for n in self.nodes],
            'critical_nodes': self.critical_nodes,
            'critical_path_names': self.critical_path_names,
            'total_duration_days': self.total_duration_days,
            'project_end_date': self.project_end_date.isoformat(),
            'bottlenecks': self.bottlenecks,
            'risk_areas': self.risk_areas,
        }


@dataclass
class TimelineProjection:
    """Complete timeline projection."""
    projection_id: str
    created_at: datetime
    study_id: Optional[str]
    
    # Key dates
    current_date: datetime
    db_lock_target: Optional[datetime]
    db_lock_projected: datetime
    db_lock_best_case: datetime
    db_lock_worst_case: datetime
    
    # Probability
    probability_on_target: float
    probability_within_30_days: float
    
    # Trajectories
    trajectories: Dict[str, Trajectory]
    
    # Milestones
    milestones: List[Milestone]
    milestones_on_track: int
    milestones_at_risk: int
    milestones_delayed: int
    
    @property
    def probability_on_time(self) -> float:
        """Alias for probability_on_target."""
        return self.probability_on_target
    # Critical path
    critical_path: Optional[CriticalPath] = None
    
    # Risks and recommendations
    risk_level: RiskLevel = RiskLevel.LOW
    risk_factors: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    milestones_complete: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'projection_id': self.projection_id,
            'created_at': self.created_at.isoformat(),
            'study_id': self.study_id,
            'current_date': self.current_date.isoformat(),
            'db_lock_target': self.db_lock_target.isoformat() if self.db_lock_target else None,
            'db_lock_projected': self.db_lock_projected.isoformat(),
            'db_lock_best_case': self.db_lock_best_case.isoformat(),
            'db_lock_worst_case': self.db_lock_worst_case.isoformat(),
            'probability_on_target': self.probability_on_target,
            'probability_within_30_days': self.probability_within_30_days,
            'trajectories': {k: v.to_dict() for k, v in self.trajectories.items()},
            'milestones': [m.to_dict() for m in self.milestones],
            'milestones_on_track': self.milestones_on_track,
            'milestones_at_risk': self.milestones_at_risk,
            'milestones_delayed': self.milestones_delayed,
            'critical_path': self.critical_path.to_dict() if self.critical_path else None,
            'risk_level': self.risk_level.value,
            'risk_factors': self.risk_factors,
            'recommendations': self.recommendations,
        }
    
    @property
    def summary(self) -> str:
        """Generate text summary."""
        variance = (self.db_lock_projected - self.db_lock_target).days if self.db_lock_target else 0
        status = "ON TRACK" if variance <= 0 else f"{variance} DAYS DELAYED"
        
        return (
            f"Timeline Projection {self.projection_id}\n"
            f"{'=' * 50}\n"
            f"Status: {status}\n"
            f"DB Lock Target: {self.db_lock_target.strftime('%Y-%m-%d') if self.db_lock_target else 'Not Set'}\n"
            f"DB Lock Projected: {self.db_lock_projected.strftime('%Y-%m-%d')}\n"
            f"Best Case: {self.db_lock_best_case.strftime('%Y-%m-%d')}\n"
            f"Worst Case: {self.db_lock_worst_case.strftime('%Y-%m-%d')}\n"
            f"P(On Target): {self.probability_on_target:.0%}\n"
            f"Risk Level: {self.risk_level.value.upper()}\n"
            f"Milestones: {self.milestones_on_track} on track, {self.milestones_at_risk} at risk, {self.milestones_delayed} delayed\n"
        )


# =============================================================================
# DATA LOADER
# =============================================================================

class TimelineDataLoader:
    """Loads data for timeline projection from PostgreSQL."""
    
    def __init__(self):
        # Import here to avoid circular dependencies
        from src.database.pg_data_service import get_pg_data_service
        self.data_service = get_pg_data_service()
        self._cache = {}
    
    def load_patient_issues(self) -> Optional[pd.DataFrame]:
        """Load patient issues data."""
        if 'patient_issues' in self._cache:
            return self._cache['patient_issues']
        
        df = self.data_service.get_issues()
        if not df.empty:
            # Normalize columns to match expected format
            # ensure 'study' or 'study_id' exists
            if 'study_id' in df.columns and 'study' not in df.columns:
                df['study'] = df['study_id']
                
            self._cache['patient_issues'] = df
            return df
        return None
    
    def load_upr(self) -> Optional[pd.DataFrame]:
        """Load unified patient record (Patients)."""
        if 'upr' in self._cache:
            return self._cache['upr']
        
        df = self.data_service.get_patients()
        if not df.empty:
             if 'study_id' in df.columns and 'study' not in df.columns:
                df['study'] = df['study_id']
             self._cache['upr'] = df
             return df
        return None
    
    def load_clean_status(self) -> Optional[pd.DataFrame]:
        """Load clean status data."""
        if 'clean_status' in self._cache:
            return self._cache['clean_status']
        
        df = self.data_service.get_patient_clean_status()
        if not df.empty:
            if 'study_id' in df.columns and 'study' not in df.columns:
                df['study'] = df['study_id']
            self._cache['clean_status'] = df
            return df
        return None
    
    def load_dblock_status(self) -> Optional[pd.DataFrame]:
        """Load DB lock status data."""
        if 'dblock_status' in self._cache:
            return self._cache['dblock_status']
        
        df = self.data_service.get_patient_dblock_status()
        if not df.empty:
            if 'study_id' in df.columns and 'study' not in df.columns:
                df['study'] = df['study_id']
            self._cache['dblock_status'] = df
            return df
        return None
    
    def get_issue_counts(self, study_id: Optional[str] = None) -> Dict[str, int]:
        """Get current issue counts."""
        issues_df = self.load_patient_issues()
        if issues_df is None:
            return {}
        
        # DEBUG: Print columns to investigate 'count_type' mystery
        print(f"DEBUG: issues_df columns: {list(issues_df.columns)}")

        if study_id:
            study_col = 'study_id' if 'study_id' in issues_df.columns else 'study'
            if study_col in issues_df.columns:
                issues_df = issues_df[issues_df[study_col] == study_id]
        
        counts = {}
        
        # Priority 1: Use 'count_' columns (SQL source standard)
        count_cols = [c for c in issues_df.columns if c.startswith('count_')]
        if count_cols:
            for col in count_cols:
                issue_type = col.replace('count_', '')
                try:
                    count = int(issues_df[col].sum())
                    if count > 0:
                        counts[issue_type] = count
                except:
                    pass
            return counts

        # Priority 2: Use 'issue_' columns (Legacy Parquet or boolean flags)
        issue_cols = [c for c in issues_df.columns 
                      if c.startswith('issue_') and c != 'issue_id']
        for col in issue_cols:
            issue_type = col.replace('issue_', '')
            try:
                # Handle possible string/boolean variations
                if issues_df[col].dtype == object:
                     # If string, count non-null/non-empty
                     count = issues_df[col].notna().sum()
                else:
                     # If boolean/int, sum
                     count = int(issues_df[col].sum())
                
                if count > 0:
                    counts[issue_type] = count
            except:
                pass
        
        return counts
    
    def get_clean_rates(self, study_id: Optional[str] = None) -> Dict[str, float]:
        """Get current clean rates."""
        clean_df = self.load_clean_status()
        if clean_df is None:
            return {'tier1_clean': 0.5, 'tier2_clean': 0.4}
        
        if study_id:
            study_col = 'study_id' if 'study_id' in clean_df.columns else 'study'
            if study_col in clean_df.columns:
                clean_df = clean_df[clean_df[study_col] == study_id]
        
        tier1_col = 'tier1_clean' if 'tier1_clean' in clean_df.columns else None
        tier2_col = 'tier2_clean' if 'tier2_clean' in clean_df.columns else None
        
        return {
            'tier1_clean': float(clean_df[tier1_col].mean()) if tier1_col else 0.5,
            'tier2_clean': float(clean_df[tier2_col].mean()) if tier2_col else 0.4,
        }
    
    def get_db_lock_rates(self, study_id: Optional[str] = None) -> Dict[str, float]:
        """Get DB lock readiness rates."""
        dblock_df = self.load_dblock_status()
        if dblock_df is None:
            return {'ready_rate': 0.3, 'eligible_rate': 0.6}
        
        if study_id:
            study_col = 'study_id' if 'study_id' in dblock_df.columns else 'study'
            if study_col in dblock_df.columns:
                dblock_df = dblock_df[dblock_df[study_col] == study_id]
        
        # Find ready column
        ready_col = None
        for col in ['db_lock_tier1_ready', 'dblock_tier1_ready', 'dblock_ready']:
            if col in dblock_df.columns:
                ready_col = col
                break
        
        # Find eligible column
        eligible_col = None
        for col in ['db_lock_eligible', 'dblock_eligible']:
            if col in dblock_df.columns:
                eligible_col = col
                break
        
        ready_rate = float(dblock_df[ready_col].mean()) if ready_col else 0.3
        eligible_rate = float(dblock_df[eligible_col].mean()) if eligible_col else 0.6
        
        return {
            'ready_rate': ready_rate,
            'eligible_rate': eligible_rate,
        }
    
    def get_patient_count(self, study_id: Optional[str] = None) -> int:
        """Get total patient count."""
        upr = self.load_upr()
        if upr is None:
            return 50000
        
        if study_id:
            study_col = 'study_id' if 'study_id' in upr.columns else 'study'
            if study_col in upr.columns:
                return len(upr[upr[study_col] == study_id])
        
        return len(upr)


# =============================================================================
# TIMELINE PROJECTOR
# =============================================================================

class TimelineProjector:
    """
    Projects trial timelines with trajectory analysis.
    
    Provides:
    - Current trajectory projection
    - Best/worst case scenarios
    - Critical path analysis
    - Milestone tracking
    """
    
    def __init__(self):
        self.data_loader = TimelineDataLoader()
        self.milestones: Dict[str, Milestone] = {}
        self.projections: Dict[str, TimelineProjection] = {}
        
        # Configuration
        self.resolution_rates = ISSUE_RESOLUTION_RATES.copy()
        self.best_case_multiplier = 1.3  # 30% faster
        self.worst_case_multiplier = 0.6  # 40% slower
    
    def _generate_id(self, prefix: str) -> str:
        """Generate unique ID."""
        return f"{prefix}-{datetime.now().strftime('%Y%m%d%H%M%S')}-{np.random.randint(1000, 9999)}"
    
    def create_milestone(self,
                         milestone_type: MilestoneType,
                         planned_date: Optional[datetime] = None,
                         study_id: Optional[str] = None,
                         owner: str = "") -> Milestone:
        """Create a milestone."""
        milestone_id = self._generate_id("MS")
        
        # Get dependencies
        deps = MILESTONE_DEPENDENCIES.get(milestone_type, [])
        dep_ids = [f"MS-{d.value}" for d in deps]
        
        milestone = Milestone(
            milestone_id=milestone_id,
            milestone_type=milestone_type,
            name=milestone_type.value.replace('_', ' ').title(),
            planned_date=planned_date,
            dependencies=dep_ids,
            owner=owner,
            study_id=study_id,
        )
        
        self.milestones[milestone_id] = milestone
        return milestone
    
    def initialize_default_milestones(self,
                                       study_start: datetime,
                                       db_lock_target: datetime,
                                       study_id: Optional[str] = None) -> List[Milestone]:
        """Initialize default milestone set."""
        milestones = []
        
        # Calculate intermediate dates
        total_days = (db_lock_target - study_start).days
        
        # Milestone planned dates (relative to study start)
        milestone_schedule = {
            MilestoneType.ENROLLMENT_START: 0,
            MilestoneType.ENROLLMENT_COMPLETE: int(total_days * 0.3),
            MilestoneType.LAST_PATIENT_FIRST_VISIT: int(total_days * 0.35),
            MilestoneType.LAST_PATIENT_LAST_VISIT: int(total_days * 0.7),
            MilestoneType.QUERY_RESOLUTION_COMPLETE: int(total_days * 0.85),
            MilestoneType.SDV_COMPLETE: int(total_days * 0.85),
            MilestoneType.CODING_COMPLETE: int(total_days * 0.82),
            MilestoneType.SAE_RECONCILIATION_COMPLETE: int(total_days * 0.88),
            MilestoneType.DATA_CLEANING_COMPLETE: int(total_days * 0.90),
            MilestoneType.DATABASE_SOFT_LOCK: int(total_days * 0.95),
            MilestoneType.DATABASE_HARD_LOCK: total_days,
        }
        
        for ms_type, days_offset in milestone_schedule.items():
            planned = study_start + timedelta(days=days_offset)
            ms = self.create_milestone(
                milestone_type=ms_type,
                planned_date=planned,
                study_id=study_id,
            )
            milestones.append(ms)
        
        return milestones
    
    def calculate_days_to_resolve_issues(self,
                                          issue_counts: Dict[str, int],
                                          rate_multiplier: float = 1.0) -> Dict[str, int]:
        """Calculate days needed to resolve each issue type."""
        days_needed = {}
        
        for issue_type, count in issue_counts.items():
            rate = self.resolution_rates.get(issue_type, 50) * rate_multiplier
            days = int(np.ceil(count / rate)) if rate > 0 else 999
            days_needed[issue_type] = days
        
        return days_needed
    
    def project_trajectory(self,
                           metric_name: str,
                           start_value: float,
                           target_value: float,
                           daily_rate: float,
                           trajectory_type: TrajectoryType = TrajectoryType.CURRENT,
                           max_days: int = 365) -> Trajectory:
        """Project a single metric trajectory."""
        trajectory_id = self._generate_id("TRAJ")
        
        # Calculate end date
        if daily_rate == 0:
            days_to_target = max_days
        else:
            days_to_target = int(np.ceil(abs(target_value - start_value) / abs(daily_rate)))
        days_to_target = min(days_to_target, max_days)
        
        start_date = datetime.now()
        end_date = start_date + timedelta(days=days_to_target)
        
        # Generate points (weekly)
        points = []
        for day in range(0, days_to_target + 1, 7):
            point_date = start_date + timedelta(days=day)
            value = start_value + (daily_rate * day)
            
            # Apply confidence decay over time
            confidence = max(0.5, 1.0 - (day / max_days) * 0.5)
            
            points.append(TrajectoryPoint(
                date=point_date,
                metric_value=value,
                metric_name=metric_name,
                trajectory_type=trajectory_type,
                confidence=confidence,
            ))
        
        # Confidence band
        if trajectory_type == TrajectoryType.BEST_CASE:
            confidence_band = (0.9, 1.0)
        elif trajectory_type == TrajectoryType.WORST_CASE:
            confidence_band = (1.0, 1.2)
        else:
            confidence_band = (0.85, 1.15)
        
        return Trajectory(
            trajectory_id=trajectory_id,
            trajectory_type=trajectory_type,
            metric_name=metric_name,
            start_date=start_date,
            end_date=end_date,
            points=points,
            start_value=start_value,
            end_value=target_value,
            daily_rate=daily_rate,
            confidence_band=confidence_band,
        )
    
    def calculate_critical_path(self,
                                 milestones: List[Milestone],
                                 project_start: datetime) -> CriticalPath:
        """
        Calculate critical path using CPM (Critical Path Method).
        """
        path_id = self._generate_id("CP")
        
        # Build nodes from milestones
        nodes: Dict[str, CriticalPathNode] = {}
        
        # Calculate dynamic durations based on current issues
        issue_counts = self.data_loader.get_issue_counts()
        days_needed = self.calculate_days_to_resolve_issues(issue_counts)
        
        # Mapping issue types to milestones
        dynamic_durations = DEFAULT_MILESTONE_DURATIONS.copy()
        
        if 'open_queries' in days_needed:
             dynamic_durations[MilestoneType.QUERY_RESOLUTION_COMPLETE] = max(
                 DEFAULT_MILESTONE_DURATIONS[MilestoneType.QUERY_RESOLUTION_COMPLETE],
                 days_needed['open_queries']
             )
        
        coding_days = 0
        if 'meddra_uncoded' in days_needed:
             coding_days += days_needed['meddra_uncoded']
        if 'whodrug_uncoded' in days_needed:
             coding_days += days_needed['whodrug_uncoded']
        
        if coding_days > 0:
             dynamic_durations[MilestoneType.CODING_COMPLETE] = max(
                 DEFAULT_MILESTONE_DURATIONS[MilestoneType.CODING_COMPLETE], 
                 coding_days
             )

        # Check for blocking issues affecting milestones
        if 'protocol_deviation' in days_needed and days_needed['protocol_deviation'] > 30:
             # Major impact on data cleaning
             dynamic_durations[MilestoneType.DATA_CLEANING_COMPLETE] += 30

        for ms in milestones:
            duration = dynamic_durations.get(ms.milestone_type, 14)
            nodes[ms.milestone_id] = CriticalPathNode(
                node_id=ms.milestone_id,
                name=ms.name,
                duration_days=duration,
                earliest_start=0,
                earliest_finish=0,
                latest_start=0,
                latest_finish=0,
                slack=0,
                is_critical=False,
                dependencies=ms.dependencies,
            )
        
        # Forward pass - calculate earliest times
        for node_id, node in nodes.items():
            if not node.dependencies:
                node.earliest_start = 0
            else:
                # Find max earliest finish of dependencies
                max_ef = 0
                for dep_id in node.dependencies:
                    if dep_id in nodes:
                        max_ef = max(max_ef, nodes[dep_id].earliest_finish)
                node.earliest_start = max_ef
            node.earliest_finish = node.earliest_start + node.duration_days
        
        # Find project duration
        project_duration = max(n.earliest_finish for n in nodes.values()) if nodes else 0
        
        # Backward pass - calculate latest times
        for node in nodes.values():
            node.latest_finish = project_duration
            node.latest_start = node.latest_finish - node.duration_days
        
        # Recalculate latest times considering successors
        for node_id in reversed(list(nodes.keys())):
            node = nodes[node_id]
            # Find all nodes that depend on this one
            successors = [n for n in nodes.values() if node_id in n.dependencies]
            if successors:
                min_ls = min(s.latest_start for s in successors)
                node.latest_finish = min_ls
                node.latest_start = node.latest_finish - node.duration_days
        
        # Calculate slack and identify critical path
        critical_nodes = []
        for node in nodes.values():
            node.slack = node.latest_start - node.earliest_start
            node.is_critical = node.slack == 0
            if node.is_critical:
                critical_nodes.append(node.node_id)
        

        

        return self._identify_bottlenecks(
            nodes, critical_nodes, path_id, project_start, project_duration
        )

    def _identify_bottlenecks(self, 
                              nodes: Dict[str, CriticalPathNode],
                              critical_nodes: List[str],
                              path_id: str,
                              project_start: datetime,
                              project_duration: int) -> CriticalPath:
        """Identify bottlenecks and risks in the path."""
        bottlenecks = []
        risk_areas = []
        
        # Analyze critical nodes for resource constraints
        for node_id in critical_nodes:
            node = nodes[node_id]
            # Simple heuristic: long duration critical tasks are bottlenecks
            if node.duration_days > 45:
                bottlenecks.append(f"Long duration task: {node.name} ({node.duration_days} days)")
            
            # Check for resource contention (placeholder logic)
            if "Database Lock" in node.name:
                 bottlenecks.append(f"Critical process: {node.name}")
        
        # Check specifically for issue resolution bottlenecks if issue count is high
        try:
             # Look for high volume issues based on resolution node duraration
             for node in nodes.values():
                 if "Resolution" in node.name and node.duration_days > 30 and node.is_critical:
                     bottlenecks.append(f"High volume of issues delaying {node.name}")
             
        except:
             pass

        # Identify near-critical paths (low slack)
        for node in nodes.values():
            if node.slack < 7 and not node.is_critical:
                risk_areas.append({
                    'node': node.name,
                    'slack_days': node.slack,
                    'risk': 'Near-critical path, low buffer'
                })
        
        return CriticalPath(
            path_id=path_id,
            created_at=datetime.now(),
            nodes=list(nodes.values()),
            critical_nodes=critical_nodes,
            total_duration_days=project_duration,
            project_end_date=project_start + timedelta(days=project_duration),
            bottlenecks=bottlenecks,
            risk_areas=risk_areas,
        )
    
    def project_db_lock_date(self,
                              study_id: Optional[str] = None,
                              scenario: str = 'current') -> Tuple[datetime, float]:
        """
        Project database lock date.
        
        Args:
            study_id: Optional study filter
            scenario: 'current', 'best_case', or 'worst_case'
            
        Returns:
            (projected_date, confidence)
        """
        # Get current issue counts
        issue_counts = self.data_loader.get_issue_counts(study_id)
        
        if not issue_counts:
            # Default if no data
            return datetime.now() + timedelta(days=90), 0.5
        
        # Calculate rate multiplier based on scenario
        if scenario == 'best_case':
            rate_mult = self.best_case_multiplier
        elif scenario == 'worst_case':
            rate_mult = self.worst_case_multiplier
        else:
            rate_mult = 1.0
        
        # Calculate days to resolve each issue type
        days_needed = self.calculate_days_to_resolve_issues(issue_counts, rate_mult)
        
        # The bottleneck determines the timeline
        max_days = max(days_needed.values()) if days_needed else 30
        
        # Add buffer for DB lock activities
        db_lock_buffer = 14  # 2 weeks for soft lock + hard lock
        total_days = max_days + db_lock_buffer
        
        projected_date = datetime.now() + timedelta(days=total_days)
        
        # Confidence based on scenario
        confidence = 0.7 if scenario == 'current' else 0.5
        
        return projected_date, confidence
    
    def project_timeline(self,
                          study_id: Optional[str] = None,
                          db_lock_target: Optional[datetime] = None,
                          study_start: Optional[datetime] = None) -> TimelineProjection:
        """
        Create complete timeline projection.
        
        Args:
            study_id: Optional study filter
            db_lock_target: Target DB lock date
            study_start: Study start date (defaults to 1 year ago)
            
        Returns:
            TimelineProjection with full analysis
        """
        projection_id = self._generate_id("PROJ")
        now = datetime.now()
        
        # Default dates
        if study_start is None:
            study_start = now - timedelta(days=365)
        if db_lock_target is None:
            db_lock_target = now + timedelta(days=90)
        
        # Project DB lock dates for each scenario
        db_lock_current, conf_current = self.project_db_lock_date(study_id, 'current')
        db_lock_best, conf_best = self.project_db_lock_date(study_id, 'best_case')
        db_lock_worst, conf_worst = self.project_db_lock_date(study_id, 'worst_case')
        
        # Calculate probabilities
        days_to_target = (db_lock_target - now).days
        days_to_projected = (db_lock_current - now).days
        
        # Simple probability model
        if days_to_projected <= days_to_target:
            prob_on_target = 0.85
        else:
            delay = days_to_projected - days_to_target
            prob_on_target = max(0.1, 0.85 - (delay / 100))
        
        prob_within_30 = min(1.0, prob_on_target + 0.1)
        
        # Create trajectories
        trajectories = {}
        
        # Issue resolution trajectory
        issue_counts = self.data_loader.get_issue_counts(study_id)
        total_issues = sum(issue_counts.values())
        daily_resolution = sum(self.resolution_rates.get(k, 50) for k in issue_counts.keys())
        
        if total_issues > 0:
            trajectories['issue_resolution'] = self.project_trajectory(
                metric_name='issues_remaining',
                start_value=total_issues,
                target_value=0,
                daily_rate=-daily_resolution,
                trajectory_type=TrajectoryType.CURRENT,
            )
        
        # Clean rate trajectory
        clean_rates = self.data_loader.get_clean_rates(study_id)
        current_clean = clean_rates.get('tier2_clean', 0.5)
        daily_clean_rate = (1.0 - current_clean) / max(1, days_to_projected) if days_to_projected > 0 else 0.01
        
        trajectories['clean_rate'] = self.project_trajectory(
            metric_name='tier2_clean_rate',
            start_value=current_clean,
            target_value=1.0,
            daily_rate=daily_clean_rate,
            trajectory_type=TrajectoryType.CURRENT,
        )
        
        # Initialize milestones
        milestones = self.initialize_default_milestones(study_start, db_lock_target, study_id)
        
        # Update milestone statuses based on current progress
        for ms in milestones:
            if ms.planned_date and ms.planned_date < now:
                # Should be complete by now
                if ms.milestone_type in [MilestoneType.ENROLLMENT_START, 
                                          MilestoneType.ENROLLMENT_COMPLETE,
                                          MilestoneType.LAST_PATIENT_FIRST_VISIT]:
                    ms.status = MilestoneStatus.COMPLETED
                    ms.progress_percent = 100.0
                    ms.actual_date = ms.planned_date
                else:
                    # Check if at risk
                    days_past = (now - ms.planned_date).days
                    if days_past > 14:
                        ms.status = MilestoneStatus.DELAYED
                    else:
                        ms.status = MilestoneStatus.AT_RISK
                    ms.progress_percent = min(95, 50 + days_past * 2)
            else:
                ms.status = MilestoneStatus.IN_PROGRESS if ms.progress_percent > 0 else MilestoneStatus.NOT_STARTED
            
            # Set projected date
            if not ms.is_complete:
                ms.projected_date = ms.planned_date
        
        # Count milestone statuses
        on_track = sum(1 for m in milestones if m.is_on_track or m.status == MilestoneStatus.COMPLETED)
        at_risk = sum(1 for m in milestones if m.status == MilestoneStatus.AT_RISK)
        delayed = sum(1 for m in milestones if m.status == MilestoneStatus.DELAYED)
        
        # Calculate critical path
        critical_path = self.calculate_critical_path(milestones, study_start)
        
        # Determine risk level
        if delayed > 2 or prob_on_target < 0.3:
            risk_level = RiskLevel.CRITICAL
        elif delayed > 0 or at_risk > 3 or prob_on_target < 0.5:
            risk_level = RiskLevel.HIGH
        elif at_risk > 1 or prob_on_target < 0.7:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        # Generate risk factors
        risk_factors = []
        if delayed > 0:
            risk_factors.append(f"{delayed} milestone(s) are delayed")
        if at_risk > 0:
            risk_factors.append(f"{at_risk} milestone(s) are at risk")
        if total_issues > 10000:
            risk_factors.append(f"High issue backlog: {total_issues:,} issues")
        if prob_on_target < 0.5:
            risk_factors.append(f"Low probability of meeting target: {prob_on_target:.0%}")
        if critical_path.bottlenecks:
            risk_factors.append(f"Bottlenecks identified: {', '.join(critical_path.bottlenecks[:3])}")
        
        # Generate recommendations
        recommendations = []
        if delayed > 0:
            recommendations.append("Immediately address delayed milestones with additional resources")
        if total_issues > 5000:
            recommendations.append("Consider adding resources to accelerate issue resolution")
        if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            recommendations.append("Schedule urgent review meeting with stakeholders")
        if prob_on_target < 0.7:
            recommendations.append(f"Adjust target date or increase resources to improve probability")
        recommendations.append("Monitor critical path activities closely")
        
        projection = TimelineProjection(
            projection_id=projection_id,
            created_at=now,
            study_id=study_id,
            current_date=now,
            db_lock_target=db_lock_target,
            db_lock_projected=db_lock_current,
            db_lock_best_case=db_lock_best,
            db_lock_worst_case=db_lock_worst,
            probability_on_target=prob_on_target,
            probability_within_30_days=prob_within_30,
            trajectories=trajectories,
            milestones=milestones,
            milestones_on_track=on_track,
            milestones_at_risk=at_risk,
            milestones_delayed=delayed,
            critical_path=critical_path,
            risk_level=risk_level,
            risk_factors=risk_factors,
            recommendations=recommendations,
        )
        
        self.projections[projection_id] = projection
        return projection
    
    def update_milestone(self,
                          milestone_id: str,
                          status: Optional[MilestoneStatus] = None,
                          progress_percent: Optional[float] = None,
                          actual_date: Optional[datetime] = None,
                          projected_date: Optional[datetime] = None) -> Optional[Milestone]:
        """Update a milestone."""
        ms = self.milestones.get(milestone_id)
        if not ms:
            return None
        
        if status:
            ms.status = status
        if progress_percent is not None:
            ms.progress_percent = progress_percent
        if actual_date:
            ms.actual_date = actual_date
        if projected_date:
            ms.projected_date = projected_date
        
        return ms
    
    def get_gantt_data(self, projection_id: str) -> List[Dict]:
        """Get data for Gantt chart visualization."""
        projection = self.projections.get(projection_id)
        if not projection:
            return []
        
        gantt_data = []
        for ms in projection.milestones:
            start = ms.planned_date or projection.current_date
            end = ms.projected_date or ms.planned_date or projection.current_date + timedelta(days=30)
            
            gantt_data.append({
                'task': ms.name,
                'start': start.isoformat(),
                'end': end.isoformat(),
                'progress': ms.progress_percent,
                'status': ms.status.value,
                'is_critical': ms.milestone_id in (projection.critical_path.critical_nodes if projection.critical_path else []),
                'dependencies': ms.dependencies,
            })
        
        return gantt_data
    
    def compare_scenarios(self,
                           study_id: Optional[str] = None,
                           db_lock_target: Optional[datetime] = None) -> Dict[str, Any]:
        """Compare best/current/worst case scenarios."""
        if db_lock_target is None:
            db_lock_target = datetime.now() + timedelta(days=90)
        
        current, _ = self.project_db_lock_date(study_id, 'current')
        best, _ = self.project_db_lock_date(study_id, 'best_case')
        worst, _ = self.project_db_lock_date(study_id, 'worst_case')
        
        target_days = (db_lock_target - datetime.now()).days
        
        return {
            'target_date': db_lock_target.isoformat(),
            'target_days': target_days,
            'scenarios': {
                'best_case': {
                    'date': best.isoformat(),
                    'days': (best - datetime.now()).days,
                    'variance': (best - db_lock_target).days,
                    'on_track': best <= db_lock_target,
                },
                'current': {
                    'date': current.isoformat(),
                    'days': (current - datetime.now()).days,
                    'variance': (current - db_lock_target).days,
                    'on_track': current <= db_lock_target,
                },
                'worst_case': {
                    'date': worst.isoformat(),
                    'days': (worst - datetime.now()).days,
                    'variance': (worst - db_lock_target).days,
                    'on_track': worst <= db_lock_target,
                },
            },
            'range_days': (worst - best).days,
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get projector statistics."""
        return {
            'milestones': {
                'total': len(self.milestones),
                'completed': sum(1 for m in self.milestones.values() if m.is_complete),
                'in_progress': sum(1 for m in self.milestones.values() 
                                   if m.status == MilestoneStatus.IN_PROGRESS),
                'at_risk': sum(1 for m in self.milestones.values() 
                               if m.status == MilestoneStatus.AT_RISK),
                'delayed': sum(1 for m in self.milestones.values() 
                               if m.status == MilestoneStatus.DELAYED),
            },
            'projections': {
                'total': len(self.projections),
            },
            'resolution_rates': self.resolution_rates,
        }


# =============================================================================
# SINGLETON AND CONVENIENCE FUNCTIONS
# =============================================================================

_projector_instance: Optional[TimelineProjector] = None


def get_timeline_projector() -> TimelineProjector:
    """Get singleton TimelineProjector instance."""
    global _projector_instance
    if _projector_instance is None:
        _projector_instance = TimelineProjector()
    return _projector_instance


def reset_timeline_projector():
    """Reset the projector instance (for testing)."""
    global _projector_instance
    _projector_instance = None


def project_timeline(study_id: Optional[str] = None,
                      db_lock_target: Optional[datetime] = None) -> TimelineProjection:
    """Convenience function to project timeline."""
    projector = get_timeline_projector()
    return projector.project_timeline(study_id, db_lock_target)


def get_db_lock_projection(study_id: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function to get DB lock projection."""
    projector = get_timeline_projector()
    return projector.compare_scenarios(study_id)


def get_critical_path(study_id: Optional[str] = None) -> Optional[CriticalPath]:
    """Convenience function to get critical path."""
    projector = get_timeline_projector()
    projection = projector.project_timeline(study_id)
    return projection.critical_path


# =============================================================================
# TEST FUNCTION
# =============================================================================

def test_timeline_projector():
    """Test the Timeline Projector."""
    print("=" * 70)
    print("a6on-i - TIMELINE PROJECTOR TEST")
    print("=" * 70)
    
    tests_passed = 0
    tests_failed = 0
    
    # Reset for clean test
    reset_timeline_projector()
    
    # Test 1: Initialize projector
    print("\n" + "-" * 70)
    print("TEST 1: Initialize Projector")
    print("-" * 70)
    try:
        projector = get_timeline_projector()
        print(f"   Resolution rates configured: {len(projector.resolution_rates)}")
        print(f"   Best case multiplier: {projector.best_case_multiplier}")
        print(f"   Worst case multiplier: {projector.worst_case_multiplier}")
        print("   ✅ PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        tests_failed += 1
    
    # Test 2: Load issue data
    print("\n" + "-" * 70)
    print("TEST 2: Load Issue Data")
    print("-" * 70)
    try:
        issue_counts = projector.data_loader.get_issue_counts()
        print(f"   Issue types: {len(issue_counts)}")
        total = sum(issue_counts.values())
        print(f"   Total issues: {total:,}")
        if issue_counts:
            top_3 = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            for issue, count in top_3:
                print(f"     - {issue}: {count:,}")
        print("   ✅ PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        tests_failed += 1
    
    # Test 3: Calculate days to resolve
    print("\n" + "-" * 70)
    print("TEST 3: Calculate Days to Resolve Issues")
    print("-" * 70)
    try:
        days_needed = projector.calculate_days_to_resolve_issues(issue_counts)
        print(f"   Issue types calculated: {len(days_needed)}")
        if days_needed:
            max_issue = max(days_needed.items(), key=lambda x: x[1])
            print(f"   Bottleneck: {max_issue[0]} ({max_issue[1]} days)")
            total_days = max(days_needed.values())
            print(f"   Max days needed: {total_days}")
        print("   ✅ PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        tests_failed += 1
    
    # Test 4: Project DB lock date
    print("\n" + "-" * 70)
    print("TEST 4: Project DB Lock Date")
    print("-" * 70)
    try:
        current_date, conf = projector.project_db_lock_date(scenario='current')
        best_date, _ = projector.project_db_lock_date(scenario='best_case')
        worst_date, _ = projector.project_db_lock_date(scenario='worst_case')
        
        print(f"   Current projection: {current_date.strftime('%Y-%m-%d')}")
        print(f"   Best case: {best_date.strftime('%Y-%m-%d')}")
        print(f"   Worst case: {worst_date.strftime('%Y-%m-%d')}")
        print(f"   Range: {(worst_date - best_date).days} days")
        print(f"   Confidence: {conf:.0%}")
        print("   ✅ PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        tests_failed += 1
    
    # Test 5: Create milestones
    print("\n" + "-" * 70)
    print("TEST 5: Create Milestones")
    print("-" * 70)
    try:
        study_start = datetime.now() - timedelta(days=180)
        db_lock_target = datetime.now() + timedelta(days=90)
        
        milestones = projector.initialize_default_milestones(study_start, db_lock_target)
        print(f"   Milestones created: {len(milestones)}")
        for ms in milestones[:3]:
            print(f"     - {ms.name}: {ms.status.value}")
        print("   ✅ PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        tests_failed += 1
    
    # Test 6: Project trajectory
    print("\n" + "-" * 70)
    print("TEST 6: Project Trajectory")
    print("-" * 70)
    try:
        trajectory = projector.project_trajectory(
            metric_name='issues_remaining',
            start_value=50000,
            target_value=0,
            daily_rate=-500,
            trajectory_type=TrajectoryType.CURRENT,
        )
        print(f"   Trajectory ID: {trajectory.trajectory_id}")
        print(f"   Start: {trajectory.start_value:,.0f}")
        print(f"   End: {trajectory.end_value:,.0f}")
        print(f"   Daily rate: {trajectory.daily_rate:,.0f}")
        print(f"   Duration: {(trajectory.end_date - trajectory.start_date).days} days")
        print(f"   Points: {len(trajectory.points)}")
        print("   ✅ PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        tests_failed += 1
    
    # Test 7: Calculate critical path
    print("\n" + "-" * 70)
    print("TEST 7: Calculate Critical Path")
    print("-" * 70)
    try:
        critical_path = projector.calculate_critical_path(milestones, study_start)
        print(f"   Path ID: {critical_path.path_id}")
        print(f"   Total nodes: {len(critical_path.nodes)}")
        print(f"   Critical nodes: {len(critical_path.critical_nodes)}")
        print(f"   Total duration: {critical_path.total_duration_days} days")
        print(f"   Bottlenecks: {len(critical_path.bottlenecks)}")
        if critical_path.critical_path_names:
            print(f"   Critical path: {' → '.join(critical_path.critical_path_names[:4])}...")
        print("   ✅ PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        tests_failed += 1
    
    # Test 8: Full timeline projection
    print("\n" + "-" * 70)
    print("TEST 8: Full Timeline Projection")
    print("-" * 70)
    try:
        projection = projector.project_timeline(
            db_lock_target=datetime.now() + timedelta(days=90)
        )
        print(f"   Projection ID: {projection.projection_id}")
        print(f"   DB Lock Target: {projection.db_lock_target.strftime('%Y-%m-%d')}")
        print(f"   DB Lock Projected: {projection.db_lock_projected.strftime('%Y-%m-%d')}")
        print(f"   P(On Target): {projection.probability_on_target:.0%}")
        print(f"   Risk Level: {projection.risk_level.value}")
        print(f"   Milestones: {len(projection.milestones)}")
        print(f"     - On Track: {projection.milestones_on_track}")
        print(f"     - At Risk: {projection.milestones_at_risk}")
        print(f"     - Delayed: {projection.milestones_delayed}")
        print(f"   Trajectories: {len(projection.trajectories)}")
        print("   ✅ PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        tests_failed += 1
    
    # Test 9: Scenario comparison
    print("\n" + "-" * 70)
    print("TEST 9: Scenario Comparison")
    print("-" * 70)
    try:
        comparison = projector.compare_scenarios(
            db_lock_target=datetime.now() + timedelta(days=90)
        )
        print(f"   Target: {comparison['target_date'][:10]}")
        for scenario, data in comparison['scenarios'].items():
            status = "✅" if data['on_track'] else "❌"
            print(f"   {scenario}: {data['date'][:10]} ({data['variance']:+d} days) {status}")
        print(f"   Range: {comparison['range_days']} days")
        print("   ✅ PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        tests_failed += 1
    
    # Test 10: Get Gantt data
    print("\n" + "-" * 70)
    print("TEST 10: Get Gantt Data")
    print("-" * 70)
    try:
        gantt_data = projector.get_gantt_data(projection.projection_id)
        print(f"   Gantt items: {len(gantt_data)}")
        if gantt_data:
            print(f"   Sample: {gantt_data[0]['task']}")
            print(f"     Start: {gantt_data[0]['start'][:10]}")
            print(f"     End: {gantt_data[0]['end'][:10]}")
        print("   ✅ PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        tests_failed += 1
    
    # Test 11: Risk factors and recommendations
    print("\n" + "-" * 70)
    print("TEST 11: Risk Factors and Recommendations")
    print("-" * 70)
    try:
        print(f"   Risk factors: {len(projection.risk_factors)}")
        for rf in projection.risk_factors[:3]:
            print(f"     - {rf}")
        print(f"   Recommendations: {len(projection.recommendations)}")
        for rec in projection.recommendations[:3]:
            print(f"     - {rec[:60]}...")
        print("   ✅ PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        tests_failed += 1
    
    # Test 12: Convenience functions
    print("\n" + "-" * 70)
    print("TEST 12: Convenience Functions")
    print("-" * 70)
    try:
        reset_timeline_projector()
        proj = project_timeline()
        db_proj = get_db_lock_projection()
        cp = get_critical_path()
        
        print(f"   project_timeline: {proj.projection_id}")
        print(f"   get_db_lock_projection: {len(db_proj['scenarios'])} scenarios")
        print(f"   get_critical_path: {len(cp.nodes) if cp else 0} nodes")
        print("   ✅ PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        tests_failed += 1
    
    # Test 13: Statistics
    print("\n" + "-" * 70)
    print("TEST 13: Statistics")
    print("-" * 70)
    try:
        stats = get_timeline_projector().get_statistics()
        print(f"   Milestones total: {stats['milestones']['total']}")
        print(f"   Projections total: {stats['projections']['total']}")
        print(f"   Resolution rates: {len(stats['resolution_rates'])}")
        print("   ✅ PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        tests_failed += 1
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests Passed: {tests_passed}")
    print(f"Tests Failed: {tests_failed}")
    print(f"Total: {tests_passed + tests_failed}")
    
    if tests_failed == 0:
        print("\n✅ ALL TESTS PASSED!")
    else:
        print(f"\n❌ {tests_failed} TEST(S) FAILED")
    
    return tests_passed, tests_failed


if __name__ == "__main__":
    test_timeline_projector()
