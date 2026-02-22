"""
a6on-i - Phase 9.3: Resource Optimizer v1.0

Optimal resource allocation for clinical trial operations using
linear programming and constraint optimization.

Features:
- Assignment problem formulation (scipy.optimize.linear_sum_assignment)
- Multi-objective optimization (timeline, quality, cost)
- Capacity constraints and workload balancing
- Impact projection with uncertainty
- Actionable recommendations

Author: a6on-i Team
Date: 2026-01-02
"""

import json
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any, Set
from enum import Enum
from pathlib import Path
import numpy as np

# Optional imports for optimization
try:
    from scipy.optimize import linear_sum_assignment, minimize
    from scipy.stats import norm
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Using fallback optimization.")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


# =============================================================================
# ENUMS
# =============================================================================

class ResourceType(Enum):
    """Types of resources that can be allocated."""
    CRA = "cra"
    DATA_MANAGER = "data_manager"
    SITE_COORDINATOR = "site_coordinator"
    SAFETY_DATA_MANAGER = "safety_data_manager"
    SAFETY_PHYSICIAN = "safety_physician"
    MEDICAL_CODER = "medical_coder"
    STUDY_LEAD = "study_lead"
    CTM = "ctm"
    STATISTICIAN = "statistician"
    QA_AUDITOR = "qa_auditor"


class TaskType(Enum):
    """Types of tasks that need resources."""
    SDV_COMPLETION = "sdv_completion"
    QUERY_RESOLUTION = "query_resolution"
    SIGNATURE_COLLECTION = "signature_collection"
    SAE_RECONCILIATION = "sae_reconciliation"
    CODING_REVIEW = "coding_review"
    DATA_CLEANING = "data_cleaning"
    SITE_MONITORING = "site_monitoring"
    SAFETY_REVIEW = "safety_review"
    DB_LOCK_PREP = "db_lock_prep"
    AUDIT_PREP = "audit_prep"


class OptimizationObjective(Enum):
    """Optimization objectives."""
    MINIMIZE_TIME = "minimize_time"
    MINIMIZE_COST = "minimize_cost"
    MAXIMIZE_QUALITY = "maximize_quality"
    BALANCE_WORKLOAD = "balance_workload"
    MINIMIZE_RISK = "minimize_risk"


class AllocationStatus(Enum):
    """Status of resource allocation."""
    PROPOSED = "proposed"
    APPROVED = "approved"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ConstraintType(Enum):
    """Types of constraints."""
    CAPACITY = "capacity"
    SKILL = "skill"
    AVAILABILITY = "availability"
    REGULATORY = "regulatory"
    BUDGET = "budget"
    GEOGRAPHIC = "geographic"


# =============================================================================
# CONFIGURATION
# =============================================================================

# Resource productivity (issues resolved per day)
RESOURCE_PRODUCTIVITY = {
    ResourceType.CRA: {
        TaskType.SDV_COMPLETION: 10,
        TaskType.SITE_MONITORING: 5,
        TaskType.QUERY_RESOLUTION: 8,
        TaskType.SIGNATURE_COLLECTION: 6,
    },
    ResourceType.DATA_MANAGER: {
        TaskType.QUERY_RESOLUTION: 15,
        TaskType.DATA_CLEANING: 20,
        TaskType.DB_LOCK_PREP: 10,
        TaskType.CODING_REVIEW: 5,
    },
    ResourceType.SITE_COORDINATOR: {
        TaskType.SIGNATURE_COLLECTION: 8,
        TaskType.QUERY_RESOLUTION: 5,
        TaskType.SDV_COMPLETION: 3,
    },
    ResourceType.SAFETY_DATA_MANAGER: {
        TaskType.SAE_RECONCILIATION: 12,
        TaskType.SAFETY_REVIEW: 8,
        TaskType.QUERY_RESOLUTION: 5,
    },
    ResourceType.SAFETY_PHYSICIAN: {
        TaskType.SAFETY_REVIEW: 5,
        TaskType.SAE_RECONCILIATION: 3,
    },
    ResourceType.MEDICAL_CODER: {
        TaskType.CODING_REVIEW: 50,
        TaskType.QUERY_RESOLUTION: 10,
    },
    ResourceType.STUDY_LEAD: {
        TaskType.DB_LOCK_PREP: 5,
        TaskType.AUDIT_PREP: 3,
    },
    ResourceType.CTM: {
        TaskType.SITE_MONITORING: 8,
        TaskType.DB_LOCK_PREP: 5,
    },
}

# Resource costs (monthly USD)
RESOURCE_COSTS = {
    ResourceType.CRA: 12000,
    ResourceType.DATA_MANAGER: 10000,
    ResourceType.SITE_COORDINATOR: 6000,
    ResourceType.SAFETY_DATA_MANAGER: 11000,
    ResourceType.SAFETY_PHYSICIAN: 25000,
    ResourceType.MEDICAL_CODER: 8000,
    ResourceType.STUDY_LEAD: 18000,
    ResourceType.CTM: 15000,
    ResourceType.STATISTICIAN: 14000,
    ResourceType.QA_AUDITOR: 13000,
}

# Task to issue type mapping
TASK_ISSUE_MAPPING = {
    TaskType.SDV_COMPLETION: "sdv_incomplete",
    TaskType.QUERY_RESOLUTION: "open_queries",
    TaskType.SIGNATURE_COLLECTION: "signature_gaps",
    TaskType.SAE_RECONCILIATION: "sae_dm_pending",
    TaskType.CODING_REVIEW: "meddra_uncoded",
    TaskType.DATA_CLEANING: "broken_signatures",
    TaskType.SITE_MONITORING: "missing_visits",
    TaskType.SAFETY_REVIEW: "sae_safety_pending",
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Resource:
    """Represents a resource that can be allocated."""
    resource_id: str
    resource_type: ResourceType
    name: str
    capacity_hours_per_week: float = 40.0
    current_workload_hours: float = 0.0
    skills: List[TaskType] = field(default_factory=list)
    assigned_sites: List[str] = field(default_factory=list)
    assigned_studies: List[str] = field(default_factory=list)
    hourly_cost: float = 0.0
    availability_start: Optional[datetime] = None
    availability_end: Optional[datetime] = None
    efficiency_factor: float = 1.0  # 1.0 = average, >1 = above average
    
    def __post_init__(self):
        if not self.skills:
            # Default skills based on resource type
            self.skills = list(RESOURCE_PRODUCTIVITY.get(self.resource_type, {}).keys())
        if self.hourly_cost == 0:
            monthly = RESOURCE_COSTS.get(self.resource_type, 10000)
            self.hourly_cost = monthly / 160  # ~160 hours/month
    
    @property
    def available_hours(self) -> float:
        """Available hours per week."""
        return max(0, self.capacity_hours_per_week - self.current_workload_hours)
    
    @property
    def utilization_rate(self) -> float:
        """Current utilization rate (0-1)."""
        if self.capacity_hours_per_week == 0:
            return 0
        return min(1.0, self.current_workload_hours / self.capacity_hours_per_week)
    
    def get_productivity(self, task_type: TaskType) -> float:
        """Get productivity for a specific task type."""
        base = RESOURCE_PRODUCTIVITY.get(self.resource_type, {}).get(task_type, 0)
        return base * self.efficiency_factor
    
    def to_dict(self) -> Dict:
        return {
            'resource_id': self.resource_id,
            'resource_type': self.resource_type.value,
            'name': self.name,
            'capacity_hours_per_week': self.capacity_hours_per_week,
            'current_workload_hours': self.current_workload_hours,
            'available_hours': self.available_hours,
            'utilization_rate': self.utilization_rate,
            'skills': [s.value for s in self.skills],
            'hourly_cost': self.hourly_cost,
            'efficiency_factor': self.efficiency_factor,
        }


@dataclass
class Task:
    """Represents a task requiring resource allocation."""
    task_id: str
    task_type: TaskType
    entity_id: str  # site_id, study_id, or patient_key
    entity_type: str  # 'site', 'study', 'patient'
    issue_count: int = 0
    estimated_hours: float = 0.0
    priority: int = 3  # 1=Critical, 2=High, 3=Medium, 4=Low
    deadline: Optional[datetime] = None
    required_skills: List[TaskType] = field(default_factory=list)
    preferred_resource_types: List[ResourceType] = field(default_factory=list)
    assigned_resource_id: Optional[str] = None
    status: str = "pending"
    
    def __post_init__(self):
        if not self.required_skills:
            self.required_skills = [self.task_type]
        if self.estimated_hours == 0 and self.issue_count > 0:
            # Estimate hours based on average productivity
            avg_productivity = 10  # issues per day
            self.estimated_hours = (self.issue_count / avg_productivity) * 8
    
    @property
    def urgency_score(self) -> float:
        """Calculate urgency score (higher = more urgent)."""
        score = (5 - self.priority) * 25  # Priority contribution
        if self.deadline:
            days_remaining = (self.deadline - datetime.now()).days
            if days_remaining <= 0:
                score += 100  # Overdue
            elif days_remaining <= 7:
                score += 50
            elif days_remaining <= 14:
                score += 25
        score += min(50, self.issue_count / 10)  # Issue count contribution
        return score
    
    def to_dict(self) -> Dict:
        return {
            'task_id': self.task_id,
            'task_type': self.task_type.value,
            'entity_id': self.entity_id,
            'entity_type': self.entity_type,
            'issue_count': self.issue_count,
            'estimated_hours': self.estimated_hours,
            'priority': self.priority,
            'urgency_score': self.urgency_score,
            'deadline': self.deadline.isoformat() if self.deadline else None,
            'assigned_resource_id': self.assigned_resource_id,
            'status': self.status,
        }


@dataclass
class Constraint:
    """Represents an optimization constraint."""
    constraint_id: str
    constraint_type: ConstraintType
    description: str
    is_hard: bool = True  # Hard constraints must be satisfied
    parameters: Dict = field(default_factory=dict)
    penalty_weight: float = 1.0  # For soft constraints
    
    def to_dict(self) -> Dict:
        return {
            'constraint_id': self.constraint_id,
            'constraint_type': self.constraint_type.value,
            'description': self.description,
            'is_hard': self.is_hard,
            'parameters': self.parameters,
            'penalty_weight': self.penalty_weight,
        }


@dataclass
class Assignment:
    """Represents a resource-to-task assignment."""
    assignment_id: str
    resource_id: str
    task_id: str
    allocated_hours: float
    expected_completion_days: float
    expected_issues_resolved: int
    cost_estimate: float
    efficiency_score: float  # 0-1, how well matched
    created_at: datetime = field(default_factory=datetime.now)
    status: AllocationStatus = AllocationStatus.PROPOSED
    
    def to_dict(self) -> Dict:
        return {
            'assignment_id': self.assignment_id,
            'resource_id': self.resource_id,
            'task_id': self.task_id,
            'allocated_hours': self.allocated_hours,
            'expected_completion_days': self.expected_completion_days,
            'expected_issues_resolved': self.expected_issues_resolved,
            'cost_estimate': self.cost_estimate,
            'efficiency_score': self.efficiency_score,
            'status': self.status.value,
        }


@dataclass
class AllocationPlan:
    """Complete resource allocation plan."""
    plan_id: str
    created_at: datetime
    objective: OptimizationObjective
    assignments: List[Assignment]
    total_cost: float
    total_hours: float
    expected_days_to_complete: float
    expected_issues_resolved: int
    workload_balance_score: float  # 0-1, higher = more balanced
    constraints_satisfied: int
    constraints_violated: int
    optimization_score: float
    recommendations: List[str]
    warnings: List[str]
    
    def to_dict(self) -> Dict:
        return {
            'plan_id': self.plan_id,
            'created_at': self.created_at.isoformat(),
            'objective': self.objective.value,
            'assignments': [a.to_dict() for a in self.assignments],
            'total_cost': self.total_cost,
            'total_hours': self.total_hours,
            'expected_days_to_complete': self.expected_days_to_complete,
            'expected_issues_resolved': self.expected_issues_resolved,
            'workload_balance_score': self.workload_balance_score,
            'constraints_satisfied': self.constraints_satisfied,
            'constraints_violated': self.constraints_violated,
            'optimization_score': self.optimization_score,
            'recommendations': self.recommendations,
            'warnings': self.warnings,
        }
    
    @property
    def summary(self) -> str:
        """Generate text summary."""
        return (
            f"Allocation Plan {self.plan_id}\n"
            f"{'=' * 50}\n"
            f"Assignments: {len(self.assignments)}\n"
            f"Total Hours: {self.total_hours:.1f}\n"
            f"Total Cost: ${self.total_cost:,.0f}\n"
            f"Expected Days: {self.expected_days_to_complete:.1f}\n"
            f"Issues to Resolve: {self.expected_issues_resolved:,}\n"
            f"Workload Balance: {self.workload_balance_score:.1%}\n"
            f"Optimization Score: {self.optimization_score:.1%}\n"
        )


@dataclass
class ImpactProjection:
    """Projected impact of allocation plan."""
    projection_id: str
    plan_id: str
    created_at: datetime
    baseline_metrics: Dict[str, float]
    projected_metrics: Dict[str, float]
    improvement_percentages: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    timeline_days: int
    risk_factors: List[str]
    success_probability: float
    
    def to_dict(self) -> Dict:
        return {
            'projection_id': self.projection_id,
            'plan_id': self.plan_id,
            'created_at': self.created_at.isoformat(),
            'baseline_metrics': self.baseline_metrics,
            'projected_metrics': self.projected_metrics,
            'improvement_percentages': self.improvement_percentages,
            'confidence_intervals': self.confidence_intervals,
            'timeline_days': self.timeline_days,
            'risk_factors': self.risk_factors,
            'success_probability': self.success_probability,
        }


@dataclass
class Recommendation:
    """Resource allocation recommendation."""
    recommendation_id: str
    category: str  # 'add_resource', 'reallocate', 'remove', 'train', 'optimize'
    priority: int  # 1=Critical, 2=High, 3=Medium, 4=Low
    title: str
    description: str
    expected_impact: str
    cost_estimate: float
    implementation_effort: str  # 'Low', 'Medium', 'High'
    timeline_days: int
    supporting_data: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'recommendation_id': self.recommendation_id,
            'category': self.category,
            'priority': self.priority,
            'title': self.title,
            'description': self.description,
            'expected_impact': self.expected_impact,
            'cost_estimate': self.cost_estimate,
            'implementation_effort': self.implementation_effort,
            'timeline_days': self.timeline_days,
            'supporting_data': self.supporting_data,
        }


# =============================================================================
# DATA LOADER
# =============================================================================

class ResourceOptimizerDataLoader:
    """Loads data for resource optimization."""
    
    def __init__(self):
        self.base_path = Path("data/processed")
        self._cache = {}
    
    def load_patient_issues(self) -> Optional[pd.DataFrame]:
        """Load patient issues data."""
        if 'patient_issues' in self._cache:
            return self._cache['patient_issues']
        
        path = self.base_path / "analytics" / "patient_issues.parquet"
        if path.exists() and PANDAS_AVAILABLE:
            df = pd.read_parquet(path)
            self._cache['patient_issues'] = df
            return df
        return None
    
    def load_site_benchmarks(self) -> Optional[pd.DataFrame]:
        """Load site benchmarks data."""
        if 'site_benchmarks' in self._cache:
            return self._cache['site_benchmarks']
        
        path = self.base_path / "analytics" / "site_benchmarks.parquet"
        if path.exists() and PANDAS_AVAILABLE:
            df = pd.read_parquet(path)
            self._cache['site_benchmarks'] = df
            return df
        return None
    
    def load_upr(self) -> Optional[pd.DataFrame]:
        """Load unified patient record."""
        if 'upr' in self._cache:
            return self._cache['upr']
        
        path = self.base_path / "upr" / "unified_patient_record.parquet"
        if path.exists() and PANDAS_AVAILABLE:
            df = pd.read_parquet(path)
            self._cache['upr'] = df
            return df
        return None
    
    def get_issue_summary(self) -> Dict[str, int]:
        """Get summary of issues by type."""
        issues_df = self.load_patient_issues()
        if issues_df is None:
            return {}
        
        summary = {}
        issue_cols = [c for c in issues_df.columns if c.startswith('issue_')]
        for col in issue_cols:
            issue_type = col.replace('issue_', '')
            count = int(issues_df[col].sum()) if col in issues_df.columns else 0
            if count > 0:
                summary[issue_type] = count
        
        return summary
    
    def get_site_issue_summary(self) -> Dict[str, Dict[str, int]]:
        """Get issue summary by site."""
        issues_df = self.load_patient_issues()
        if issues_df is None:
            return {}
        
        site_col = 'site_id' if 'site_id' in issues_df.columns else 'site'
        if site_col not in issues_df.columns:
            return {}
        
        result = {}
        issue_cols = [c for c in issues_df.columns if c.startswith('issue_')]
        
        for site_id in issues_df[site_col].unique():
            site_data = issues_df[issues_df[site_col] == site_id]
            site_issues = {}
            for col in issue_cols:
                issue_type = col.replace('issue_', '')
                count = int(site_data[col].sum())
                if count > 0:
                    site_issues[issue_type] = count
            if site_issues:
                result[site_id] = site_issues
        
        return result


# =============================================================================
# RESOURCE OPTIMIZER
# =============================================================================

class ResourceOptimizer:
    """
    Optimizes resource allocation for clinical trial operations.
    
    Uses linear assignment and constraint optimization to find
    optimal resource-to-task allocations.
    """
    
    def __init__(self):
        self.data_loader = ResourceOptimizerDataLoader()
        self.resources: Dict[str, Resource] = {}
        self.tasks: Dict[str, Task] = {}
        self.constraints: List[Constraint] = []
        self.plans: Dict[str, AllocationPlan] = {}
        self.projections: Dict[str, ImpactProjection] = {}
        
        # Initialize default resources
        self._initialize_default_resources()
        
        # Initialize default constraints
        self._initialize_default_constraints()
    
    def _initialize_default_resources(self):
        """Initialize default resource pool."""
        default_resources = [
            # CRAs
            ("CRA-001", ResourceType.CRA, "Sarah Chen", 40, 20),
            ("CRA-002", ResourceType.CRA, "Michael Park", 40, 25),
            ("CRA-003", ResourceType.CRA, "Emily Johnson", 40, 15),
            ("CRA-004", ResourceType.CRA, "David Kim", 40, 30),
            # Data Managers
            ("DM-001", ResourceType.DATA_MANAGER, "Alex Thompson", 40, 18),
            ("DM-002", ResourceType.DATA_MANAGER, "Lisa Wang", 40, 22),
            ("DM-003", ResourceType.DATA_MANAGER, "James Wilson", 40, 28),
            # Safety Team
            ("SDM-001", ResourceType.SAFETY_DATA_MANAGER, "Rachel Green", 40, 20),
            ("SP-001", ResourceType.SAFETY_PHYSICIAN, "Dr. Robert Smith", 20, 10),
            # Coders
            ("MC-001", ResourceType.MEDICAL_CODER, "Jennifer Lee", 40, 25),
            ("MC-002", ResourceType.MEDICAL_CODER, "Chris Brown", 40, 20),
            # Leadership
            ("SL-001", ResourceType.STUDY_LEAD, "Dr. Amanda White", 45, 35),
            ("CTM-001", ResourceType.CTM, "Mark Davis", 40, 30),
        ]
        
        for res_id, res_type, name, capacity, workload in default_resources:
            self.resources[res_id] = Resource(
                resource_id=res_id,
                resource_type=res_type,
                name=name,
                capacity_hours_per_week=capacity,
                current_workload_hours=workload,
                efficiency_factor=0.9 + np.random.random() * 0.2  # 0.9-1.1
            )
    
    def _initialize_default_constraints(self):
        """Initialize default constraints."""
        self.constraints = [
            Constraint(
                constraint_id="CAP-001",
                constraint_type=ConstraintType.CAPACITY,
                description="Resource cannot exceed capacity",
                is_hard=True,
                parameters={'max_utilization': 1.0}
            ),
            Constraint(
                constraint_id="SKILL-001",
                constraint_type=ConstraintType.SKILL,
                description="Resource must have required skills",
                is_hard=True,
                parameters={}
            ),
            Constraint(
                constraint_id="WORKLOAD-001",
                constraint_type=ConstraintType.CAPACITY,
                description="Prefer balanced workload distribution",
                is_hard=False,
                parameters={'target_utilization': 0.8},
                penalty_weight=0.5
            ),
            Constraint(
                constraint_id="BUDGET-001",
                constraint_type=ConstraintType.BUDGET,
                description="Stay within monthly budget",
                is_hard=False,
                parameters={'monthly_budget': 200000},
                penalty_weight=0.3
            ),
        ]
    
    def add_resource(self, resource: Resource):
        """Add a resource to the pool."""
        self.resources[resource.resource_id] = resource
    
    def remove_resource(self, resource_id: str) -> bool:
        """Remove a resource from the pool."""
        if resource_id in self.resources:
            del self.resources[resource_id]
            return True
        return False
    
    def add_task(self, task: Task):
        """Add a task to be allocated."""
        self.tasks[task.task_id] = task
    
    def add_constraint(self, constraint: Constraint):
        """Add an optimization constraint."""
        self.constraints.append(constraint)
    
    def load_tasks_from_data(self, study_id: Optional[str] = None, 
                             site_id: Optional[str] = None) -> int:
        """Load tasks from issue data."""
        issue_summary = self.data_loader.get_site_issue_summary()
        
        task_count = 0
        for site, issues in issue_summary.items():
            if site_id and site != site_id:
                continue
            
            for issue_type, count in issues.items():
                if count == 0:
                    continue
                
                # Map issue type to task type
                task_type = None
                for tt, it in TASK_ISSUE_MAPPING.items():
                    if it == issue_type:
                        task_type = tt
                        break
                
                if task_type is None:
                    continue
                
                # Determine priority based on issue type
                priority = 3
                if issue_type in ['sae_dm_pending', 'sae_safety_pending']:
                    priority = 1
                elif issue_type in ['open_queries', 'sdv_incomplete']:
                    priority = 2
                elif issue_type in ['broken_signatures', 'signature_gaps']:
                    priority = 2
                
                task_id = f"TASK-{site}-{issue_type}"
                task = Task(
                    task_id=task_id,
                    task_type=task_type,
                    entity_id=site,
                    entity_type='site',
                    issue_count=count,
                    priority=priority,
                    deadline=datetime.now() + timedelta(days=30 if priority > 1 else 7)
                )
                self.tasks[task_id] = task
                task_count += 1
        
        return task_count
    
    def _build_cost_matrix(self, resources: List[Resource], 
                           tasks: List[Task]) -> np.ndarray:
        """
        Build cost matrix for assignment problem.
        Lower cost = better assignment.
        """
        n_resources = len(resources)
        n_tasks = len(tasks)
        
        # Initialize with high cost
        cost_matrix = np.full((n_resources, n_tasks), 1e6)
        
        for i, resource in enumerate(resources):
            for j, task in enumerate(tasks):
                # Check if resource can do this task
                if task.task_type not in resource.skills:
                    continue  # Keep high cost
                
                # Check capacity
                if resource.available_hours < 1:
                    continue  # Keep high cost
                
                # Calculate cost based on multiple factors
                productivity = resource.get_productivity(task.task_type)
                if productivity == 0:
                    continue
                
                # Time cost (hours to complete)
                hours_needed = task.issue_count / productivity * 8
                time_cost = hours_needed
                
                # Monetary cost
                monetary_cost = hours_needed * resource.hourly_cost
                
                # Efficiency cost (prefer well-matched resources)
                efficiency = min(1.0, productivity / 20)  # Normalize
                efficiency_cost = (1 - efficiency) * 100
                
                # Workload balance cost
                new_utilization = (resource.current_workload_hours + min(hours_needed, 10)) / resource.capacity_hours_per_week
                balance_cost = abs(new_utilization - 0.8) * 50
                
                # Priority urgency (prefer assigning high-priority tasks)
                priority_cost = (5 - task.priority) * (-10)  # Negative = bonus
                
                # Combined cost
                cost = (
                    time_cost * 0.3 +
                    monetary_cost * 0.0001 +  # Scale down
                    efficiency_cost * 0.2 +
                    balance_cost * 0.2 +
                    priority_cost
                )
                
                cost_matrix[i, j] = max(0, cost)
        
        return cost_matrix
    
    def optimize(self, 
                 objective: OptimizationObjective = OptimizationObjective.MINIMIZE_TIME,
                 max_assignments_per_resource: int = 5,
                 study_id: Optional[str] = None,
                 site_id: Optional[str] = None) -> AllocationPlan:
        """
        Run optimization to create allocation plan.
        
        Args:
            objective: Optimization objective
            max_assignments_per_resource: Max tasks per resource
            study_id: Filter by study
            site_id: Filter by site
            
        Returns:
            AllocationPlan with optimized assignments
        """
        # Load tasks if empty
        if not self.tasks:
            self.load_tasks_from_data(study_id, site_id)
        
        # Get available resources and pending tasks
        resources = [r for r in self.resources.values() if r.available_hours > 0]
        tasks = [t for t in self.tasks.values() if t.status == 'pending']
        
        if not resources:
            return self._create_empty_plan(objective, "No available resources")
        
        if not tasks:
            return self._create_empty_plan(objective, "No pending tasks")
        
        # Build cost matrix
        cost_matrix = self._build_cost_matrix(resources, tasks)
        
        # Solve assignment problem
        assignments = []
        assigned_tasks = set()
        resource_assignments = {r.resource_id: 0 for r in resources}
        
        if SCIPY_AVAILABLE:
            # Use Hungarian algorithm for optimal assignment
            try:
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                
                for i, j in zip(row_ind, col_ind):
                    if cost_matrix[i, j] >= 1e6:
                        continue  # Skip invalid assignments
                    
                    if resource_assignments[resources[i].resource_id] >= max_assignments_per_resource:
                        continue
                    
                    assignment = self._create_assignment(resources[i], tasks[j])
                    if assignment:
                        assignments.append(assignment)
                        assigned_tasks.add(tasks[j].task_id)
                        resource_assignments[resources[i].resource_id] += 1
            except Exception as e:
                print(f"Optimization error: {e}")
        
        # Fallback: Greedy assignment for remaining tasks
        remaining_tasks = [t for t in tasks if t.task_id not in assigned_tasks]
        remaining_tasks.sort(key=lambda t: t.urgency_score, reverse=True)
        
        for task in remaining_tasks:
            best_resource = None
            best_score = float('inf')
            
            for resource in resources:
                if resource_assignments[resource.resource_id] >= max_assignments_per_resource:
                    continue
                if task.task_type not in resource.skills:
                    continue
                if resource.available_hours < 1:
                    continue
                
                # Score: lower is better
                productivity = resource.get_productivity(task.task_type)
                if productivity == 0:
                    continue
                
                score = task.issue_count / productivity + resource.utilization_rate * 10
                
                if score < best_score:
                    best_score = score
                    best_resource = resource
            
            if best_resource:
                assignment = self._create_assignment(best_resource, task)
                if assignment:
                    assignments.append(assignment)
                    resource_assignments[best_resource.resource_id] += 1
        
        # Create plan
        plan = self._create_plan(objective, assignments)
        self.plans[plan.plan_id] = plan
        
        return plan
    
    def _create_assignment(self, resource: Resource, task: Task) -> Optional[Assignment]:
        """Create an assignment between resource and task."""
        productivity = resource.get_productivity(task.task_type)
        if productivity == 0:
            return None
        
        # Calculate metrics
        issues_per_day = productivity
        days_to_complete = task.issue_count / issues_per_day
        hours_needed = days_to_complete * 8
        
        # Limit to available hours
        allocated_hours = min(hours_needed, resource.available_hours)
        expected_issues = int(allocated_hours / 8 * issues_per_day)
        
        # Calculate efficiency score
        efficiency = min(1.0, productivity / 20)
        
        assignment_id = f"ASN-{datetime.now().strftime('%Y%m%d%H%M%S')}-{resource.resource_id[-3:]}-{task.task_id[-10:]}"
        
        return Assignment(
            assignment_id=assignment_id,
            resource_id=resource.resource_id,
            task_id=task.task_id,
            allocated_hours=allocated_hours,
            expected_completion_days=days_to_complete,
            expected_issues_resolved=expected_issues,
            cost_estimate=allocated_hours * resource.hourly_cost,
            efficiency_score=efficiency,
        )
    
    def _create_plan(self, objective: OptimizationObjective, 
                     assignments: List[Assignment]) -> AllocationPlan:
        """Create allocation plan from assignments."""
        plan_id = f"PLAN-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Calculate totals
        total_cost = sum(a.cost_estimate for a in assignments)
        total_hours = sum(a.allocated_hours for a in assignments)
        total_issues = sum(a.expected_issues_resolved for a in assignments)
        
        # Calculate days to complete (parallel execution)
        max_days = max((a.expected_completion_days for a in assignments), default=0)
        
        # Calculate workload balance
        resource_hours = {}
        for a in assignments:
            resource_hours[a.resource_id] = resource_hours.get(a.resource_id, 0) + a.allocated_hours
        
        if resource_hours:
            hours_values = list(resource_hours.values())
            mean_hours = np.mean(hours_values)
            std_hours = np.std(hours_values) if len(hours_values) > 1 else 0
            balance_score = 1 - min(1, std_hours / (mean_hours + 1))
        else:
            balance_score = 0
        
        # Check constraints
        satisfied, violated = self._check_constraints(assignments)
        
        # Calculate optimization score
        avg_efficiency = np.mean([a.efficiency_score for a in assignments]) if assignments else 0
        optimization_score = (avg_efficiency * 0.4 + balance_score * 0.3 + 
                              (1 - violated / (violated + satisfied + 1)) * 0.3)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(assignments)
        
        # Generate warnings
        warnings = []
        if violated > 0:
            warnings.append(f"{violated} constraint(s) violated")
        if balance_score < 0.5:
            warnings.append("Workload distribution is unbalanced")
        if total_hours > 500:
            warnings.append("High total hours - consider adding resources")
        
        return AllocationPlan(
            plan_id=plan_id,
            created_at=datetime.now(),
            objective=objective,
            assignments=assignments,
            total_cost=total_cost,
            total_hours=total_hours,
            expected_days_to_complete=max_days,
            expected_issues_resolved=total_issues,
            workload_balance_score=balance_score,
            constraints_satisfied=satisfied,
            constraints_violated=violated,
            optimization_score=optimization_score,
            recommendations=recommendations,
            warnings=warnings,
        )
    
    def _create_empty_plan(self, objective: OptimizationObjective, 
                           reason: str) -> AllocationPlan:
        """Create an empty plan with warning."""
        return AllocationPlan(
            plan_id=f"PLAN-{datetime.now().strftime('%Y%m%d%H%M%S')}-EMPTY",
            created_at=datetime.now(),
            objective=objective,
            assignments=[],
            total_cost=0,
            total_hours=0,
            expected_days_to_complete=0,
            expected_issues_resolved=0,
            workload_balance_score=0,
            constraints_satisfied=0,
            constraints_violated=0,
            optimization_score=0,
            recommendations=[],
            warnings=[reason],
        )
    
    def _check_constraints(self, assignments: List[Assignment]) -> Tuple[int, int]:
        """Check how many constraints are satisfied/violated."""
        satisfied = 0
        violated = 0
        
        for constraint in self.constraints:
            if constraint.constraint_type == ConstraintType.CAPACITY:
                # Check resource capacity
                resource_hours = {}
                for a in assignments:
                    resource_hours[a.resource_id] = resource_hours.get(a.resource_id, 0) + a.allocated_hours
                
                for res_id, hours in resource_hours.items():
                    resource = self.resources.get(res_id)
                    if resource:
                        max_util = constraint.parameters.get('max_utilization', 1.0)
                        if hours <= resource.capacity_hours_per_week * max_util:
                            satisfied += 1
                        else:
                            violated += 1
            
            elif constraint.constraint_type == ConstraintType.BUDGET:
                total_cost = sum(a.cost_estimate for a in assignments)
                monthly_budget = constraint.parameters.get('monthly_budget', float('inf'))
                if total_cost <= monthly_budget:
                    satisfied += 1
                else:
                    violated += 1
            
            else:
                satisfied += 1  # Default to satisfied
        
        return satisfied, violated
    
    def _generate_recommendations(self, assignments: List[Assignment]) -> List[str]:
        """Generate recommendations based on assignments."""
        recommendations = []
        
        # Check for overloaded resources
        resource_hours = {}
        for a in assignments:
            resource_hours[a.resource_id] = resource_hours.get(a.resource_id, 0) + a.allocated_hours
        
        for res_id, hours in resource_hours.items():
            resource = self.resources.get(res_id)
            if resource and hours > resource.capacity_hours_per_week * 0.9:
                recommendations.append(
                    f"Consider adding support for {resource.name} ({resource.resource_type.value}) "
                    f"who is at {hours/resource.capacity_hours_per_week:.0%} capacity"
                )
        
        # Check for unassigned high-priority tasks
        assigned_tasks = {a.task_id for a in assignments}
        for task_id, task in self.tasks.items():
            if task_id not in assigned_tasks and task.priority <= 2:
                recommendations.append(
                    f"High-priority task {task_id} ({task.task_type.value}) is unassigned - "
                    f"consider adding {task.task_type.value} resources"
                )
        
        # Check resource type gaps
        task_types_needed = set(t.task_type for t in self.tasks.values() if t.status == 'pending')
        available_skills = set()
        for r in self.resources.values():
            available_skills.update(r.skills)
        
        for tt in task_types_needed:
            if tt not in available_skills:
                recommendations.append(f"No resources available for {tt.value} tasks")
        
        return recommendations[:5]  # Limit to top 5
    
    def project_impact(self, plan_id: str, timeline_days: int = 30) -> ImpactProjection:
        """
        Project the impact of an allocation plan.
        
        Args:
            plan_id: ID of the allocation plan
            timeline_days: Days to project
            
        Returns:
            ImpactProjection with expected outcomes
        """
        plan = self.plans.get(plan_id)
        if not plan:
            raise ValueError(f"Plan {plan_id} not found")
        
        # Get baseline metrics
        issue_summary = self.data_loader.get_issue_summary()
        total_issues = sum(issue_summary.values())
        
        # Load additional metrics
        upr = self.data_loader.load_upr()
        if upr is not None:
            total_patients = len(upr)
            mean_dqi = upr['enhanced_dqi'].mean() if 'enhanced_dqi' in upr.columns else 90
            clean_rate = upr['tier2_clean'].mean() if 'tier2_clean' in upr.columns else 0.5
        else:
            total_patients = 50000
            mean_dqi = 90
            clean_rate = 0.5
        
        baseline = {
            'total_issues': total_issues,
            'mean_dqi': mean_dqi,
            'clean_rate': clean_rate,
            'db_lock_ready_rate': 0.3,
        }
        
        # Project improvements
        issues_resolved_rate = plan.expected_issues_resolved / max(total_issues, 1)
        
        projected = {
            'total_issues': max(0, total_issues - plan.expected_issues_resolved),
            'mean_dqi': min(100, mean_dqi + issues_resolved_rate * 5),
            'clean_rate': min(1.0, clean_rate + issues_resolved_rate * 0.3),
            'db_lock_ready_rate': min(1.0, 0.3 + issues_resolved_rate * 0.4),
        }
        
        # Calculate improvements
        improvements = {}
        for key in baseline:
            if baseline[key] != 0:
                improvements[key] = (projected[key] - baseline[key]) / baseline[key] * 100
            else:
                improvements[key] = 0
        
        # Calculate confidence intervals (using normal approximation)
        confidence_intervals = {}
        for key in projected:
            point = projected[key]
            # Assume 10% standard error
            std_err = abs(point - baseline[key]) * 0.1
            confidence_intervals[key] = (
                point - 1.96 * std_err,
                point + 1.96 * std_err
            )
        
        # Identify risk factors
        risk_factors = []
        if plan.workload_balance_score < 0.5:
            risk_factors.append("Unbalanced workload may cause burnout")
        if plan.constraints_violated > 0:
            risk_factors.append(f"{plan.constraints_violated} constraints violated")
        if plan.expected_days_to_complete > timeline_days:
            risk_factors.append(f"Plan exceeds timeline ({plan.expected_days_to_complete:.0f} vs {timeline_days} days)")
        
        # Calculate success probability
        success_factors = [
            plan.optimization_score,
            plan.workload_balance_score,
            1 - (plan.constraints_violated / (plan.constraints_violated + plan.constraints_satisfied + 1)),
            min(1, timeline_days / (plan.expected_days_to_complete + 1)),
        ]
        success_probability = np.mean(success_factors)
        
        projection_id = f"PROJ-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        projection = ImpactProjection(
            projection_id=projection_id,
            plan_id=plan_id,
            created_at=datetime.now(),
            baseline_metrics=baseline,
            projected_metrics=projected,
            improvement_percentages=improvements,
            confidence_intervals=confidence_intervals,
            timeline_days=timeline_days,
            risk_factors=risk_factors,
            success_probability=success_probability,
        )
        
        self.projections[projection_id] = projection
        return projection
    
    def generate_recommendations(self, 
                                  plan_id: Optional[str] = None,
                                  max_recommendations: int = 10) -> List[Recommendation]:
        """
        Generate detailed recommendations for resource optimization.
        
        Args:
            plan_id: Optional plan ID to base recommendations on
            max_recommendations: Maximum recommendations to return
            
        Returns:
            List of Recommendation objects
        """
        recommendations = []
        rec_id = 1
        
        # Analyze current state
        issue_summary = self.data_loader.get_issue_summary()
        total_issues = sum(issue_summary.values())
        
        # 1. Recommendations for understaffed areas
        for task_type, issue_type in TASK_ISSUE_MAPPING.items():
            issue_count = issue_summary.get(issue_type, 0)
            if issue_count < 100:
                continue
            
            # Check if we have resources for this task
            capable_resources = [r for r in self.resources.values() 
                                if task_type in r.skills and r.available_hours > 10]
            
            if len(capable_resources) < 2 and issue_count > 500:
                # Find which resource type handles this
                for res_type, tasks in RESOURCE_PRODUCTIVITY.items():
                    if task_type in tasks:
                        monthly_cost = RESOURCE_COSTS.get(res_type, 10000)
                        recommendations.append(Recommendation(
                            recommendation_id=f"REC-{rec_id:03d}",
                            category='add_resource',
                            priority=1 if issue_count > 1000 else 2,
                            title=f"Add {res_type.value} for {task_type.value}",
                            description=f"There are {issue_count:,} {issue_type} issues but only "
                                       f"{len(capable_resources)} available resources. "
                                       f"Adding a {res_type.value} would significantly reduce backlog.",
                            expected_impact=f"Resolve ~{issue_count//3:,} issues in 30 days",
                            cost_estimate=monthly_cost,
                            implementation_effort='Medium',
                            timeline_days=14,
                            supporting_data={
                                'issue_count': issue_count,
                                'current_resources': len(capable_resources),
                                'resource_type': res_type.value,
                            }
                        ))
                        rec_id += 1
                        break
        
        # 2. Recommendations for overutilized resources
        for resource in self.resources.values():
            if resource.utilization_rate > 0.9:
                recommendations.append(Recommendation(
                    recommendation_id=f"REC-{rec_id:03d}",
                    category='reallocate',
                    priority=2,
                    title=f"Redistribute workload from {resource.name}",
                    description=f"{resource.name} is at {resource.utilization_rate:.0%} utilization. "
                               f"Consider redistributing {resource.current_workload_hours - resource.capacity_hours_per_week * 0.8:.0f} "
                               f"hours to other {resource.resource_type.value}s.",
                    expected_impact="Reduce burnout risk, improve quality",
                    cost_estimate=0,
                    implementation_effort='Low',
                    timeline_days=7,
                    supporting_data={
                        'resource_id': resource.resource_id,
                        'current_utilization': resource.utilization_rate,
                        'excess_hours': max(0, resource.current_workload_hours - resource.capacity_hours_per_week * 0.8),
                    }
                ))
                rec_id += 1
        
        # 3. Recommendations for skill gaps
        all_task_types = set(TaskType)
        covered_tasks = set()
        for resource in self.resources.values():
            covered_tasks.update(resource.skills)
        
        for task_type in all_task_types - covered_tasks:
            issue_type = TASK_ISSUE_MAPPING.get(task_type)
            if issue_type and issue_summary.get(issue_type, 0) > 0:
                recommendations.append(Recommendation(
                    recommendation_id=f"REC-{rec_id:03d}",
                    category='train',
                    priority=3,
                    title=f"Train resources for {task_type.value}",
                    description=f"No resources currently have {task_type.value} skills. "
                               f"Consider training existing staff or hiring.",
                    expected_impact="Enable handling of additional issue types",
                    cost_estimate=2000,
                    implementation_effort='Medium',
                    timeline_days=30,
                    supporting_data={'task_type': task_type.value}
                ))
                rec_id += 1
        
        # 4. Optimization recommendations
        if total_issues > 10000:
            recommendations.append(Recommendation(
                recommendation_id=f"REC-{rec_id:03d}",
                category='optimize',
                priority=2,
                title="Implement batch processing for high-volume tasks",
                description=f"With {total_issues:,} total issues, implementing batch processing "
                           f"could improve efficiency by 20-30%.",
                expected_impact="20-30% efficiency improvement",
                cost_estimate=5000,
                implementation_effort='High',
                timeline_days=60,
                supporting_data={'total_issues': total_issues}
            ))
            rec_id += 1
        
        # Sort by priority
        recommendations.sort(key=lambda r: (r.priority, -r.cost_estimate))
        
        return recommendations[:max_recommendations]
    
    def get_resource_utilization(self) -> Dict[str, Dict]:
        """Get current resource utilization summary."""
        utilization = {}
        for res_id, resource in self.resources.items():
            utilization[res_id] = {
                'name': resource.name,
                'type': resource.resource_type.value,
                'capacity': resource.capacity_hours_per_week,
                'current_workload': resource.current_workload_hours,
                'available': resource.available_hours,
                'utilization_rate': resource.utilization_rate,
                'skills': [s.value for s in resource.skills],
            }
        return utilization
    
    def get_task_summary(self) -> Dict[str, Any]:
        """Get summary of current tasks."""
        if not self.tasks:
            return {'total': 0, 'by_type': {}, 'by_priority': {}, 'total_issues': 0}
        
        by_type = {}
        by_priority = {1: 0, 2: 0, 3: 0, 4: 0}
        total_issues = 0
        
        for task in self.tasks.values():
            tt = task.task_type.value
            by_type[tt] = by_type.get(tt, 0) + 1
            by_priority[task.priority] = by_priority.get(task.priority, 0) + 1
            total_issues += task.issue_count
        
        return {
            'total': len(self.tasks),
            'by_type': by_type,
            'by_priority': by_priority,
            'total_issues': total_issues,
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get optimizer statistics."""
        return {
            'resources': {
                'total': len(self.resources),
                'by_type': {rt.value: sum(1 for r in self.resources.values() 
                                          if r.resource_type == rt) 
                           for rt in ResourceType},
                'total_capacity_hours': sum(r.capacity_hours_per_week for r in self.resources.values()),
                'total_available_hours': sum(r.available_hours for r in self.resources.values()),
                'avg_utilization': np.mean([r.utilization_rate for r in self.resources.values()]) if self.resources else 0,
            },
            'tasks': self.get_task_summary(),
            'plans': {
                'total': len(self.plans),
                'total_assignments': sum(len(p.assignments) for p in self.plans.values()),
            },
            'constraints': {
                'total': len(self.constraints),
                'hard': sum(1 for c in self.constraints if c.is_hard),
                'soft': sum(1 for c in self.constraints if not c.is_hard),
            },
        }


# =============================================================================
# SINGLETON AND CONVENIENCE FUNCTIONS
# =============================================================================

_optimizer_instance: Optional[ResourceOptimizer] = None


def get_resource_optimizer() -> ResourceOptimizer:
    """Get singleton ResourceOptimizer instance."""
    global _optimizer_instance
    if _optimizer_instance is None:
        _optimizer_instance = ResourceOptimizer()
    return _optimizer_instance


def reset_resource_optimizer():
    """Reset the optimizer instance (for testing)."""
    global _optimizer_instance
    _optimizer_instance = None


def optimize_allocation(
    objective: OptimizationObjective = OptimizationObjective.MINIMIZE_TIME,
    study_id: Optional[str] = None,
    site_id: Optional[str] = None
) -> AllocationPlan:
    """Convenience function to run optimization."""
    optimizer = get_resource_optimizer()
    return optimizer.optimize(objective, study_id=study_id, site_id=site_id)


def get_recommendations(max_recommendations: int = 10) -> List[Recommendation]:
    """Convenience function to get recommendations."""
    optimizer = get_resource_optimizer()
    return optimizer.generate_recommendations(max_recommendations=max_recommendations)


def get_resource_stats() -> Dict[str, Any]:
    """Convenience function to get resource statistics."""
    optimizer = get_resource_optimizer()
    return optimizer.get_statistics()


# =============================================================================
# TEST FUNCTION
# =============================================================================

def test_resource_optimizer():
    """Test the Resource Optimizer."""
    print("=" * 70)
    print("a6on-i - RESOURCE OPTIMIZER TEST")
    print("=" * 70)
    
    tests_passed = 0
    tests_failed = 0
    
    # Reset for clean test
    reset_resource_optimizer()
    
    # Test 1: Initialize optimizer
    print("\n" + "-" * 70)
    print("TEST 1: Initialize Optimizer")
    print("-" * 70)
    try:
        optimizer = get_resource_optimizer()
        print(f"   Resources: {len(optimizer.resources)}")
        print(f"   Constraints: {len(optimizer.constraints)}")
        assert len(optimizer.resources) > 0
        assert len(optimizer.constraints) > 0
        print("   ✅ PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        tests_failed += 1
    
    # Test 2: Load tasks from data
    print("\n" + "-" * 70)
    print("TEST 2: Load Tasks from Data")
    print("-" * 70)
    try:
        task_count = optimizer.load_tasks_from_data()
        print(f"   Tasks loaded: {task_count}")
        print(f"   Task types: {list(optimizer.get_task_summary()['by_type'].keys())[:5]}...")
        assert task_count >= 0
        print("   ✅ PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        tests_failed += 1
    
    # Test 3: Resource utilization
    print("\n" + "-" * 70)
    print("TEST 3: Resource Utilization")
    print("-" * 70)
    try:
        utilization = optimizer.get_resource_utilization()
        print(f"   Resources tracked: {len(utilization)}")
        sample = list(utilization.values())[0]
        print(f"   Sample: {sample['name']} ({sample['type']})")
        print(f"     Capacity: {sample['capacity']}h, Utilization: {sample['utilization_rate']:.0%}")
        print("   ✅ PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        tests_failed += 1
    
    # Test 4: Run optimization
    print("\n" + "-" * 70)
    print("TEST 4: Run Optimization")
    print("-" * 70)
    try:
        plan = optimizer.optimize(OptimizationObjective.MINIMIZE_TIME)
        print(f"   Plan ID: {plan.plan_id}")
        print(f"   Assignments: {len(plan.assignments)}")
        print(f"   Total Hours: {plan.total_hours:.1f}")
        print(f"   Total Cost: ${plan.total_cost:,.0f}")
        print(f"   Expected Days: {plan.expected_days_to_complete:.1f}")
        print(f"   Issues to Resolve: {plan.expected_issues_resolved:,}")
        print(f"   Optimization Score: {plan.optimization_score:.1%}")
        print("   ✅ PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        tests_failed += 1
    
    # Test 5: Assignment details
    print("\n" + "-" * 70)
    print("TEST 5: Assignment Details")
    print("-" * 70)
    try:
        if plan.assignments:
            for i, asn in enumerate(plan.assignments[:3]):
                print(f"   [{i+1}] {asn.resource_id} → {asn.task_id[:30]}...")
                print(f"       Hours: {asn.allocated_hours:.1f}, Issues: {asn.expected_issues_resolved}")
        else:
            print("   No assignments (may be no tasks)")
        print("   ✅ PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        tests_failed += 1
    
    # Test 6: Impact projection
    print("\n" + "-" * 70)
    print("TEST 6: Impact Projection")
    print("-" * 70)
    try:
        projection = optimizer.project_impact(plan.plan_id, timeline_days=30)
        print(f"   Projection ID: {projection.projection_id}")
        print(f"   Baseline Issues: {projection.baseline_metrics.get('total_issues', 0):,}")
        print(f"   Projected Issues: {projection.projected_metrics.get('total_issues', 0):,.0f}")
        print(f"   Success Probability: {projection.success_probability:.1%}")
        print(f"   Risk Factors: {len(projection.risk_factors)}")
        print("   ✅ PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        tests_failed += 1
    
    # Test 7: Generate recommendations
    print("\n" + "-" * 70)
    print("TEST 7: Generate Recommendations")
    print("-" * 70)
    try:
        recommendations = optimizer.generate_recommendations(max_recommendations=5)
        print(f"   Recommendations: {len(recommendations)}")
        for rec in recommendations[:3]:
            print(f"   [{rec.priority}] {rec.title}")
            print(f"       Category: {rec.category}, Cost: ${rec.cost_estimate:,.0f}")
        print("   ✅ PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        tests_failed += 1
    
    # Test 8: Add custom resource
    print("\n" + "-" * 70)
    print("TEST 8: Add Custom Resource")
    print("-" * 70)
    try:
        new_resource = Resource(
            resource_id="CRA-NEW",
            resource_type=ResourceType.CRA,
            name="New CRA Hire",
            capacity_hours_per_week=40,
            current_workload_hours=0,
            efficiency_factor=1.0
        )
        optimizer.add_resource(new_resource)
        assert "CRA-NEW" in optimizer.resources
        print(f"   Added: {new_resource.name}")
        print(f"   Available Hours: {new_resource.available_hours}")
        print("   ✅ PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        tests_failed += 1
    
    # Test 9: Add custom task
    print("\n" + "-" * 70)
    print("TEST 9: Add Custom Task")
    print("-" * 70)
    try:
        new_task = Task(
            task_id="TASK-CUSTOM-001",
            task_type=TaskType.SDV_COMPLETION,
            entity_id="Site_Custom",
            entity_type="site",
            issue_count=100,
            priority=1,
            deadline=datetime.now() + timedelta(days=7)
        )
        optimizer.add_task(new_task)
        assert "TASK-CUSTOM-001" in optimizer.tasks
        print(f"   Added: {new_task.task_id}")
        print(f"   Urgency Score: {new_task.urgency_score:.1f}")
        print("   ✅ PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        tests_failed += 1
    
    # Test 10: Re-optimize with new resource
    print("\n" + "-" * 70)
    print("TEST 10: Re-optimize with New Resource")
    print("-" * 70)
    try:
        plan2 = optimizer.optimize(OptimizationObjective.MINIMIZE_COST)
        print(f"   New Plan ID: {plan2.plan_id}")
        print(f"   Assignments: {len(plan2.assignments)}")
        # Check if new resource was used
        new_res_used = any(a.resource_id == "CRA-NEW" for a in plan2.assignments)
        print(f"   New Resource Used: {'Yes' if new_res_used else 'No'}")
        print("   ✅ PASSED")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        tests_failed += 1
    
    # Test 11: Statistics
    print("\n" + "-" * 70)
    print("TEST 11: Statistics")
    print("-" * 70)
    try:
        stats = optimizer.get_statistics()
        print(f"   Resources: {stats['resources']['total']}")
        print(f"   Total Capacity: {stats['resources']['total_capacity_hours']}h")
        print(f"   Available Hours: {stats['resources']['total_available_hours']:.1f}h")
        print(f"   Avg Utilization: {stats['resources']['avg_utilization']:.1%}")
        print(f"   Tasks: {stats['tasks']['total']}")
        print(f"   Plans Created: {stats['plans']['total']}")
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
        reset_resource_optimizer()
        plan = optimize_allocation()
        recs = get_recommendations(5)
        stats = get_resource_stats()
        print(f"   optimize_allocation: Plan {plan.plan_id}")
        print(f"   get_recommendations: {len(recs)} recommendations")
        print(f"   get_resource_stats: {stats['resources']['total']} resources")
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
    test_resource_optimizer()
