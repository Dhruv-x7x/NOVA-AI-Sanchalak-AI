"""
SANCHALAK AI - Phase 9.2: Scenario Simulator v1.0
# NOTE: This module previously used SQLite but has been migrated to PostgreSQL-only.
# If you need to use this module, update it to use:
#   from src.database.pg_data_service import get_pg_data_service
#   service = get_pg_data_service()


What-if scenario simulation with Monte Carlo analysis for:
- Site closure impact
- Resource addition (CRA, DM, etc.)
- Deadline probability
- Process changes
- Enrollment scenarios

Author: Sanchalak AI Team
Date: 2026-01-02
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Callable
import json
import hashlib
from pathlib import Path
import logging
# SQLite removed - using PostgreSQL
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class ScenarioType(Enum):
    """Types of scenarios that can be simulated."""
    CLOSE_SITE = "close_site"
    ADD_RESOURCE = "add_resource"
    REMOVE_RESOURCE = "remove_resource"
    PROCESS_CHANGE = "process_change"
    DEADLINE_CHECK = "deadline_check"
    ENROLLMENT_CHANGE = "enrollment_change"
    ISSUE_RESOLUTION = "issue_resolution"
    QUALITY_IMPROVEMENT = "quality_improvement"
    SITE_ACTIVATION = "site_activation"
    PROTOCOL_AMENDMENT = "protocol_amendment"
    CUSTOM = "custom"


class ResourceType(Enum):
    """Types of resources that can be added/removed."""
    CRA = "cra"
    DATA_MANAGER = "data_manager"
    SITE_COORDINATOR = "site_coordinator"
    SAFETY_PHYSICIAN = "safety_physician"
    MEDICAL_CODER = "medical_coder"
    STUDY_LEAD = "study_lead"
    CTM = "ctm"
    STATISTICIAN = "statistician"


class SimulationMethod(Enum):
    """Simulation methods available."""
    MONTE_CARLO = "monte_carlo"
    DETERMINISTIC = "deterministic"
    SENSITIVITY = "sensitivity"
    BOOTSTRAP = "bootstrap"


class OutcomeMetric(Enum):
    """Metrics that can be projected."""
    DB_LOCK_DATE = "db_lock_date"
    CLEAN_RATE = "clean_rate"
    DQI_SCORE = "dqi_score"
    ISSUE_COUNT = "issue_count"
    ENROLLMENT = "enrollment"
    COST = "cost"
    TIMELINE_DAYS = "timeline_days"
    QUALITY_SCORE = "quality_score"


class ConfidenceLevel(Enum):
    """Confidence levels for intervals."""
    CI_50 = 0.50
    CI_80 = 0.80
    CI_90 = 0.90
    CI_95 = 0.95
    CI_99 = 0.99


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ScenarioParameter:
    """A parameter that can be varied in a scenario."""
    name: str
    base_value: float
    min_value: float
    max_value: float
    distribution: str = "normal"  # normal, uniform, triangular, beta
    std_dev: Optional[float] = None
    description: str = ""
    
    def sample(self, n: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        """Sample values from the distribution."""
        rng = np.random.default_rng(random_state)
        
        if self.distribution == "normal":
            std = self.std_dev or (self.max_value - self.min_value) / 4
            samples = rng.normal(self.base_value, std, n)
        elif self.distribution == "uniform":
            samples = rng.uniform(self.min_value, self.max_value, n)
        elif self.distribution == "triangular":
            samples = rng.triangular(self.min_value, self.base_value, self.max_value, n)
        elif self.distribution == "beta":
            # Scale to 0-1, sample, scale back
            alpha, beta_param = 2, 2  # Default symmetric
            samples = rng.beta(alpha, beta_param, n)
            samples = self.min_value + samples * (self.max_value - self.min_value)
        else:
            samples = np.full(n, self.base_value)
        
        # Clip to bounds
        return np.clip(samples, self.min_value, self.max_value)


@dataclass
class UncertaintyBand:
    """Uncertainty band for a metric."""
    point_estimate: float
    ci_50_lower: float
    ci_50_upper: float
    ci_80_lower: float
    ci_80_upper: float
    ci_95_lower: float
    ci_95_upper: float
    std_error: float
    samples: Optional[np.ndarray] = None
    
    @classmethod
    def from_samples(cls, samples: np.ndarray) -> 'UncertaintyBand':
        """Create uncertainty band from Monte Carlo samples."""
        return cls(
            point_estimate=float(np.median(samples)),
            ci_50_lower=float(np.percentile(samples, 25)),
            ci_50_upper=float(np.percentile(samples, 75)),
            ci_80_lower=float(np.percentile(samples, 10)),
            ci_80_upper=float(np.percentile(samples, 90)),
            ci_95_lower=float(np.percentile(samples, 2.5)),
            ci_95_upper=float(np.percentile(samples, 97.5)),
            std_error=float(np.std(samples)),
            samples=samples
        )
    
    def to_dict(self) -> Dict:
        return {
            'point_estimate': self.point_estimate,
            'ci_50': [self.ci_50_lower, self.ci_50_upper],
            'ci_80': [self.ci_80_lower, self.ci_80_upper],
            'ci_95': [self.ci_95_lower, self.ci_95_upper],
            'std_error': self.std_error
        }


@dataclass
class ScenarioOutcome:
    """Outcome of a single scenario simulation."""
    metric: OutcomeMetric
    baseline_value: float
    projected_value: UncertaintyBand
    change_absolute: float
    change_percent: float
    probability_improvement: float
    probability_target_met: Optional[float] = None
    target_value: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            'metric': self.metric.value,
            'baseline_value': self.baseline_value,
            'projected_value': self.projected_value.to_dict(),
            'change_absolute': self.change_absolute,
            'change_percent': self.change_percent,
            'probability_improvement': self.probability_improvement,
            'probability_target_met': self.probability_target_met,
            'target_value': self.target_value
        }


@dataclass
class ScenarioResult:
    """Complete result of a scenario simulation."""
    scenario_id: str
    scenario_type: ScenarioType
    title: str
    description: str
    parameters: Dict[str, Any]
    outcomes: List[ScenarioOutcome]
    simulation_method: SimulationMethod
    n_iterations: int
    execution_time_ms: float
    created_at: datetime
    recommendations: List[str]
    risks: List[str]
    confidence_level: str
    summary: str
    
    def to_dict(self) -> Dict:
        return {
            'scenario_id': self.scenario_id,
            'scenario_type': self.scenario_type.value,
            'title': self.title,
            'description': self.description,
            'parameters': self.parameters,
            'outcomes': [o.to_dict() for o in self.outcomes],
            'simulation_method': self.simulation_method.value,
            'n_iterations': self.n_iterations,
            'execution_time_ms': self.execution_time_ms,
            'created_at': self.created_at.isoformat(),
            'recommendations': self.recommendations,
            'risks': self.risks,
            'confidence_level': self.confidence_level,
            'summary': self.summary
        }


@dataclass
class ComparisonResult:
    """Result of comparing multiple scenarios."""
    comparison_id: str
    scenarios: List[ScenarioResult]
    best_scenario: str
    ranking: List[Tuple[str, float]]  # (scenario_id, score)
    comparison_summary: str
    decision_matrix: Dict[str, Dict[str, float]]
    created_at: datetime


# =============================================================================
# MONTE CARLO ENGINE
# =============================================================================

class MonteCarloEngine:
    """
    Monte Carlo simulation engine for scenario analysis.
    """
    
    def __init__(
        self,
        n_iterations: int = 10000,
        random_seed: Optional[int] = None,
        convergence_threshold: float = 0.01
    ):
        self.n_iterations = n_iterations
        self.random_seed = random_seed
        self.convergence_threshold = convergence_threshold
        self.rng = np.random.default_rng(random_seed)
    
    def simulate(
        self,
        model_func: Callable,
        parameters: Dict[str, ScenarioParameter],
        n_iterations: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Run Monte Carlo simulation.
        
        Args:
            model_func: Function that takes parameter dict and returns outcome dict
            parameters: Dictionary of scenario parameters
            n_iterations: Override default iterations
            
        Returns:
            Dictionary of outcome arrays
        """
        n = n_iterations or self.n_iterations
        
        # Sample all parameters
        sampled_params = {}
        for name, param in parameters.items():
            sampled_params[name] = param.sample(n, self.random_seed)
        
        # Run simulations
        outcomes = []
        for i in range(n):
            param_values = {name: samples[i] for name, samples in sampled_params.items()}
            outcome = model_func(param_values)
            outcomes.append(outcome)
        
        # Aggregate outcomes
        if not outcomes:
            return {}
        
        result = {}
        for key in outcomes[0].keys():
            result[key] = np.array([o[key] for o in outcomes])
        
        return result
    
    def simulate_timeline(
        self,
        initial_state: Dict,
        transition_func: Callable,
        n_steps: int,
        parameters: Dict[str, ScenarioParameter],
        n_iterations: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Simulate timeline evolution with uncertainty.
        
        Args:
            initial_state: Starting state
            transition_func: Function(state, params, step) -> new_state
            n_steps: Number of time steps
            parameters: Scenario parameters
            n_iterations: Override default iterations
            
        Returns:
            Dictionary of trajectory arrays (n_iterations x n_steps)
        """
        n = n_iterations or self.n_iterations
        
        # Sample parameters
        sampled_params = {}
        for name, param in parameters.items():
            sampled_params[name] = param.sample(n, self.random_seed)
        
        # Initialize trajectories
        trajectories = {key: np.zeros((n, n_steps)) for key in initial_state.keys()}
        
        # Run simulations
        for i in range(n):
            state = initial_state.copy()
            param_values = {name: samples[i] for name, samples in sampled_params.items()}
            
            for step in range(n_steps):
                state = transition_func(state, param_values, step)
                for key in state.keys():
                    trajectories[key][i, step] = state[key]
        
        return trajectories
    
    def check_convergence(self, samples: np.ndarray) -> Tuple[bool, float]:
        """Check if simulation has converged."""
        if len(samples) < 100:
            return False, 1.0
        
        # Compare first half to full sample
        half_mean = np.mean(samples[:len(samples)//2])
        full_mean = np.mean(samples)
        
        relative_diff = abs(half_mean - full_mean) / (abs(full_mean) + 1e-10)
        converged = relative_diff < self.convergence_threshold
        
        return converged, relative_diff


# =============================================================================
# SCENARIO SIMULATOR
# =============================================================================

class ScenarioSimulator:
    """
    Main scenario simulator for what-if analysis.
    """
    
    def __init__(self, data_path: Optional[Path] = None):
        """Initialize with optional data path."""
        self.data_path = data_path or Path("data/processed")
        self.monte_carlo = MonteCarloEngine(n_iterations=5000)
        self.trial_state = None
        self.scenarios_run: List[ScenarioResult] = []
        
        # Load baseline data
        self._load_baseline_data()
        
        # Default parameters for different scenario types
        self._init_default_parameters()
    
    def _load_baseline_data(self):
        """Load baseline trial state data."""
        try:
            # Load UPR
            upr_path = self.data_path / "upr" / "unified_patient_record.parquet"
            if upr_path.exists():
                self.upr = pd.read_parquet(upr_path)
            else:
                self.upr = pd.DataFrame()
            
            # Load issues
            issues_path = self.data_path / "analytics" / "patient_issues.parquet"
            if issues_path.exists():
                self.issues = pd.read_parquet(issues_path)
            else:
                self.issues = pd.DataFrame()
            
            # Load DQI
            dqi_path = self.data_path / "analytics" / "patient_dqi_enhanced.parquet"
            if dqi_path.exists():
                self.dqi = pd.read_parquet(dqi_path)
            else:
                self.dqi = pd.DataFrame()
            
            # Load clean status
            clean_path = self.data_path / "analytics" / "patient_clean_status.parquet"
            if clean_path.exists():
                self.clean_status = pd.read_parquet(clean_path)
            else:
                self.clean_status = pd.DataFrame()
            
            # Load DB lock status
            dblock_path = self.data_path / "analytics" / "patient_dblock_status.parquet"
            if dblock_path.exists():
                self.dblock = pd.read_parquet(dblock_path)
            else:
                self.dblock = pd.DataFrame()
            
            # Calculate baseline metrics
            self._calculate_baseline_metrics()
            
            logger.info(f"Loaded baseline data: {len(self.upr)} patients")
            
        except Exception as e:
            logger.error(f"Error loading baseline data: {e}")
            self.upr = pd.DataFrame()
            self.baseline_metrics = {}
    
    def _calculate_baseline_metrics(self):
        """Calculate baseline metrics from loaded data."""
        self.baseline_metrics = {}
        
        if len(self.upr) > 0:
            self.baseline_metrics['total_patients'] = len(self.upr)
            self.baseline_metrics['total_sites'] = self.upr['site_id'].nunique() if 'site_id' in self.upr.columns else 0
            self.baseline_metrics['total_studies'] = self.upr['study_id'].nunique() if 'study_id' in self.upr.columns else 0
        
        if len(self.dqi) > 0:
            dqi_col = 'enhanced_dqi' if 'enhanced_dqi' in self.dqi.columns else 'dqi_score'
            if dqi_col in self.dqi.columns:
                self.baseline_metrics['mean_dqi'] = float(self.dqi[dqi_col].mean())
        
        if len(self.clean_status) > 0:
            if 'tier2_clean' in self.clean_status.columns:
                self.baseline_metrics['clean_rate'] = float(self.clean_status['tier2_clean'].mean())
        
        if len(self.issues) > 0:
            # Count total issues
            issue_cols = [c for c in self.issues.columns if c.startswith('issue_') or c.startswith('count_')]
            if issue_cols:
                count_cols = [c for c in issue_cols if c.startswith('count_')]
                if count_cols:
                    self.baseline_metrics['total_issues'] = int(self.issues[count_cols].sum().sum())
                else:
                    bool_cols = [c for c in issue_cols if c.startswith('issue_')]
                    self.baseline_metrics['total_issues'] = int(self.issues[bool_cols].sum().sum())
        
        if len(self.dblock) > 0:
            ready_col = None
            for col in ['db_lock_tier1_ready', 'dblock_tier1_ready', 'dblock_ready']:
                if col in self.dblock.columns:
                    ready_col = col
                    break
            if ready_col:
                self.baseline_metrics['dblock_ready_count'] = int(self.dblock[ready_col].sum())
                self.baseline_metrics['dblock_ready_rate'] = float(self.dblock[ready_col].mean())
    
    def _init_default_parameters(self):
        """Initialize default parameters for scenario types."""
        self.default_parameters = {
            ScenarioType.CLOSE_SITE: {
                'patient_transfer_rate': ScenarioParameter(
                    'patient_transfer_rate', 0.85, 0.70, 0.95,
                    distribution='triangular',
                    description='Rate of patients successfully transferred'
                ),
                'dropout_rate': ScenarioParameter(
                    'dropout_rate', 0.10, 0.05, 0.20,
                    distribution='triangular',
                    description='Rate of patients lost to dropout'
                ),
                'transfer_delay_days': ScenarioParameter(
                    'transfer_delay_days', 14, 7, 30,
                    distribution='triangular',
                    description='Days to complete patient transfer'
                )
            },
            ScenarioType.ADD_RESOURCE: {
                'productivity_multiplier': ScenarioParameter(
                    'productivity_multiplier', 1.0, 0.7, 1.3,
                    distribution='normal', std_dev=0.15,
                    description='Resource productivity relative to average'
                ),
                'ramp_up_days': ScenarioParameter(
                    'ramp_up_days', 30, 14, 60,
                    distribution='triangular',
                    description='Days for new resource to reach full productivity'
                ),
                'resolution_rate_per_day': ScenarioParameter(
                    'resolution_rate_per_day', 5, 3, 10,
                    distribution='normal', std_dev=2,
                    description='Issues resolved per resource per day'
                )
            },
            ScenarioType.DEADLINE_CHECK: {
                'resolution_rate_daily': ScenarioParameter(
                    'resolution_rate_daily', 100, 50, 200,
                    distribution='normal', std_dev=30,
                    description='Total issues resolved per day'
                ),
                'new_issues_daily': ScenarioParameter(
                    'new_issues_daily', 20, 5, 50,
                    distribution='normal', std_dev=10,
                    description='New issues created per day'
                ),
                'clean_conversion_rate': ScenarioParameter(
                    'clean_conversion_rate', 0.02, 0.01, 0.05,
                    distribution='triangular',
                    description='Rate of patients becoming clean per day'
                )
            },
            ScenarioType.PROCESS_CHANGE: {
                'improvement_factor': ScenarioParameter(
                    'improvement_factor', 0.15, 0.05, 0.30,
                    distribution='triangular',
                    description='Expected improvement from process change'
                ),
                'adoption_rate': ScenarioParameter(
                    'adoption_rate', 0.80, 0.50, 1.0,
                    distribution='triangular',
                    description='Rate of adoption across sites'
                ),
                'implementation_delay_days': ScenarioParameter(
                    'implementation_delay_days', 14, 7, 30,
                    distribution='triangular',
                    description='Days to implement process change'
                )
            }
        }
    
    def simulate_close_site(
        self,
        site_id: str,
        target_sites: Optional[List[str]] = None,
        n_iterations: int = 5000
    ) -> ScenarioResult:
        """
        Simulate impact of closing a site.
        
        Args:
            site_id: Site to close
            target_sites: Sites to transfer patients to (None = distribute evenly)
            n_iterations: Monte Carlo iterations
            
        Returns:
            ScenarioResult with impact analysis
        """
        start_time = datetime.now()
        
        # Get site data
        if 'site_id' not in self.upr.columns:
            site_col = 'site' if 'site' in self.upr.columns else None
        else:
            site_col = 'site_id'
        
        if site_col is None:
            return self._create_error_result("No site column found", ScenarioType.CLOSE_SITE)
        
        site_patients = self.upr[self.upr[site_col] == site_id]
        n_patients = len(site_patients)
        
        if n_patients == 0:
            return self._create_error_result(f"Site {site_id} not found", ScenarioType.CLOSE_SITE)
        
        # Get parameters
        params = self.default_parameters[ScenarioType.CLOSE_SITE]
        
        # Define model function
        def close_site_model(param_values: Dict) -> Dict:
            transfer_rate = param_values['patient_transfer_rate']
            dropout_rate = param_values['dropout_rate']
            transfer_delay = param_values['transfer_delay_days']
            
            transferred = int(n_patients * transfer_rate)
            dropped = int(n_patients * dropout_rate)
            lost = n_patients - transferred - dropped
            
            # Impact on timeline
            timeline_impact_days = transfer_delay * (1 + dropout_rate)
            
            # Impact on clean rate (transferred patients reset some progress)
            clean_rate_impact = -0.02 * (1 - transfer_rate)
            
            return {
                'patients_transferred': transferred,
                'patients_dropped': dropped,
                'patients_lost': lost,
                'timeline_delay_days': timeline_impact_days,
                'clean_rate_change': clean_rate_impact,
                'cost_estimate': n_patients * 5000 * (1 - transfer_rate)  # $5K per lost patient
            }
        
        # Run Monte Carlo
        results = self.monte_carlo.simulate(close_site_model, params, n_iterations)
        
        # Create outcomes
        outcomes = []
        
        # Timeline impact
        timeline_band = UncertaintyBand.from_samples(results['timeline_delay_days'])
        outcomes.append(ScenarioOutcome(
            metric=OutcomeMetric.TIMELINE_DAYS,
            baseline_value=0,
            projected_value=timeline_band,
            change_absolute=timeline_band.point_estimate,
            change_percent=0,  # N/A for delay
            probability_improvement=float(np.mean(results['timeline_delay_days'] < 14))
        ))
        
        # Patients lost
        lost_band = UncertaintyBand.from_samples(results['patients_lost'])
        outcomes.append(ScenarioOutcome(
            metric=OutcomeMetric.ENROLLMENT,
            baseline_value=n_patients,
            projected_value=lost_band,
            change_absolute=-lost_band.point_estimate,
            change_percent=-100 * lost_band.point_estimate / n_patients if n_patients > 0 else 0,
            probability_improvement=float(np.mean(results['patients_lost'] < n_patients * 0.05))
        ))
        
        # Cost
        cost_band = UncertaintyBand.from_samples(results['cost_estimate'])
        outcomes.append(ScenarioOutcome(
            metric=OutcomeMetric.COST,
            baseline_value=0,
            projected_value=cost_band,
            change_absolute=cost_band.point_estimate,
            change_percent=0,
            probability_improvement=float(np.mean(results['cost_estimate'] < 50000))
        ))
        
        # Generate recommendations
        recommendations = []
        risks = []
        
        avg_lost = np.mean(results['patients_lost'])
        if avg_lost > n_patients * 0.1:
            recommendations.append(f"Consider phased closure to minimize {int(avg_lost):.0f} expected patient losses")
            risks.append(f"High dropout risk: {100*avg_lost/n_patients:.1f}% patients may be lost")
        
        if np.mean(results['timeline_delay_days']) > 21:
            recommendations.append("Plan for significant timeline impact; consider adding resources to compensate")
            risks.append(f"Timeline delay of {np.mean(results['timeline_delay_days']):.0f} days expected")
        
        recommendations.append(f"Identify {len(target_sites) if target_sites else 'nearby'} sites for patient transfer")
        
        # Create result
        exec_time = (datetime.now() - start_time).total_seconds() * 1000
        
        result = ScenarioResult(
            scenario_id=f"SIM-{datetime.now().strftime('%Y%m%d%H%M%S')}-{hashlib.md5(site_id.encode()).hexdigest()[:6]}",
            scenario_type=ScenarioType.CLOSE_SITE,
            title=f"Close Site {site_id}",
            description=f"Impact analysis of closing site {site_id} with {n_patients} patients",
            parameters={'site_id': site_id, 'patient_count': n_patients, 'target_sites': target_sites},
            outcomes=outcomes,
            simulation_method=SimulationMethod.MONTE_CARLO,
            n_iterations=n_iterations,
            execution_time_ms=exec_time,
            created_at=datetime.now(),
            recommendations=recommendations,
            risks=risks,
            confidence_level="HIGH" if n_iterations >= 5000 else "MEDIUM",
            summary=f"Closing {site_id} affects {n_patients} patients. "
                    f"Expected: {int(np.mean(results['patients_transferred'])):.0f} transferred, "
                    f"{int(np.mean(results['patients_lost'])):.0f} lost, "
                    f"{np.mean(results['timeline_delay_days']):.0f} days delay."
        )
        
        self.scenarios_run.append(result)
        return result
    
    def simulate_add_resource(
        self,
        resource_type: ResourceType,
        count: int = 1,
        study_id: Optional[str] = None,
        site_id: Optional[str] = None,
        n_iterations: int = 5000
    ) -> ScenarioResult:
        """
        Simulate impact of adding resources.
        
        Args:
            resource_type: Type of resource to add
            count: Number of resources to add
            study_id: Limit to specific study (optional)
            site_id: Limit to specific site (optional)
            n_iterations: Monte Carlo iterations
            
        Returns:
            ScenarioResult with impact analysis
        """
        start_time = datetime.now()
        
        # Get current issue count
        if len(self.issues) > 0:
            count_cols = [c for c in self.issues.columns if c.startswith('count_')]
            if count_cols:
                total_issues = int(self.issues[count_cols].sum().sum())
            else:
                issue_cols = [c for c in self.issues.columns if c.startswith('issue_')]
                total_issues = int(self.issues[issue_cols].sum().sum())
        else:
            total_issues = self.baseline_metrics.get('total_issues', 50000)
        
        # Resource productivity mapping
        productivity_map = {
            ResourceType.CRA: 8,  # Issues per day
            ResourceType.DATA_MANAGER: 15,
            ResourceType.SITE_COORDINATOR: 5,
            ResourceType.SAFETY_PHYSICIAN: 3,
            ResourceType.MEDICAL_CODER: 50,
            ResourceType.STUDY_LEAD: 2,
            ResourceType.CTM: 5,
            ResourceType.STATISTICIAN: 1
        }
        
        base_productivity = productivity_map.get(resource_type, 5)
        
        # Cost mapping (monthly)
        cost_map = {
            ResourceType.CRA: 12000,
            ResourceType.DATA_MANAGER: 10000,
            ResourceType.SITE_COORDINATOR: 6000,
            ResourceType.SAFETY_PHYSICIAN: 25000,
            ResourceType.MEDICAL_CODER: 8000,
            ResourceType.STUDY_LEAD: 18000,
            ResourceType.CTM: 15000,
            ResourceType.STATISTICIAN: 14000
        }
        
        monthly_cost = cost_map.get(resource_type, 10000) * count
        
        # Get parameters
        params = self.default_parameters[ScenarioType.ADD_RESOURCE]
        
        # Define model function
        def add_resource_model(param_values: Dict) -> Dict:
            productivity = param_values['productivity_multiplier'] * base_productivity * count
            ramp_up = param_values['ramp_up_days']
            resolution_rate = param_values['resolution_rate_per_day'] * count
            
            # Calculate impact over 90 days
            days_simulated = 90
            effective_days = days_simulated - ramp_up * 0.5  # Average ramp-up effect
            
            issues_resolved = productivity * effective_days
            days_saved = issues_resolved / (total_issues / 365) if total_issues > 0 else 0
            
            # Timeline improvement
            current_days_to_complete = total_issues / 100  # Assume 100 issues/day baseline
            new_days_to_complete = total_issues / (100 + resolution_rate)
            timeline_improvement = current_days_to_complete - new_days_to_complete
            
            # Cost over 6 months
            cost_6m = monthly_cost * 6
            
            return {
                'issues_resolved_90d': issues_resolved,
                'days_saved': days_saved,
                'timeline_improvement': timeline_improvement,
                'cost_6m': cost_6m,
                'roi': days_saved * 10000 / cost_6m if cost_6m > 0 else 0,
                'clean_rate_improvement': min(0.10, issues_resolved / total_issues) if total_issues > 0 else 0
            }
        
        # Run Monte Carlo
        results = self.monte_carlo.simulate(add_resource_model, params, n_iterations)
        
        # Create outcomes
        outcomes = []
        
        # Timeline improvement
        timeline_band = UncertaintyBand.from_samples(results['timeline_improvement'])
        outcomes.append(ScenarioOutcome(
            metric=OutcomeMetric.TIMELINE_DAYS,
            baseline_value=0,
            projected_value=timeline_band,
            change_absolute=-timeline_band.point_estimate,  # Negative = improvement
            change_percent=-100 * timeline_band.point_estimate / 365 if timeline_band.point_estimate > 0 else 0,
            probability_improvement=float(np.mean(results['timeline_improvement'] > 0))
        ))
        
        # Issues resolved
        issues_band = UncertaintyBand.from_samples(results['issues_resolved_90d'])
        outcomes.append(ScenarioOutcome(
            metric=OutcomeMetric.ISSUE_COUNT,
            baseline_value=total_issues,
            projected_value=issues_band,
            change_absolute=-issues_band.point_estimate,
            change_percent=-100 * issues_band.point_estimate / total_issues if total_issues > 0 else 0,
            probability_improvement=float(np.mean(results['issues_resolved_90d'] > 100))
        ))
        
        # Cost
        cost_band = UncertaintyBand.from_samples(results['cost_6m'])
        outcomes.append(ScenarioOutcome(
            metric=OutcomeMetric.COST,
            baseline_value=0,
            projected_value=cost_band,
            change_absolute=cost_band.point_estimate,
            change_percent=0,
            probability_improvement=1.0  # Cost is known
        ))
        
        # Generate recommendations
        recommendations = []
        risks = []
        
        avg_roi = np.mean(results['roi'])
        if avg_roi > 1.5:
            recommendations.append(f"Strong ROI of {avg_roi:.1f}x - recommend proceeding")
        elif avg_roi > 1.0:
            recommendations.append(f"Moderate ROI of {avg_roi:.1f}x - proceed with monitoring")
        else:
            recommendations.append(f"Low ROI of {avg_roi:.1f}x - consider alternatives")
            risks.append("May not justify investment")
        
        if np.mean(results['timeline_improvement']) > 30:
            recommendations.append(f"Significant timeline acceleration: {np.mean(results['timeline_improvement']):.0f} days")
        
        risks.append(f"6-month investment: ${monthly_cost * 6:,.0f}")
        
        # Create result
        exec_time = (datetime.now() - start_time).total_seconds() * 1000
        
        result = ScenarioResult(
            scenario_id=f"SIM-{datetime.now().strftime('%Y%m%d%H%M%S')}-{hashlib.md5(resource_type.value.encode()).hexdigest()[:6]}",
            scenario_type=ScenarioType.ADD_RESOURCE,
            title=f"Add {count} {resource_type.value.replace('_', ' ').title()}",
            description=f"Impact of adding {count} {resource_type.value} to the trial",
            parameters={
                'resource_type': resource_type.value,
                'count': count,
                'study_id': study_id,
                'site_id': site_id,
                'base_productivity': base_productivity,
                'monthly_cost': monthly_cost
            },
            outcomes=outcomes,
            simulation_method=SimulationMethod.MONTE_CARLO,
            n_iterations=n_iterations,
            execution_time_ms=exec_time,
            created_at=datetime.now(),
            recommendations=recommendations,
            risks=risks,
            confidence_level="HIGH" if n_iterations >= 5000 else "MEDIUM",
            summary=f"Adding {count} {resource_type.value}: "
                    f"~{int(np.mean(results['issues_resolved_90d'])):.0f} issues resolved in 90d, "
                    f"{np.mean(results['timeline_improvement']):.0f} days saved, "
                    f"ROI: {avg_roi:.1f}x"
        )
        
        self.scenarios_run.append(result)
        return result
    
    def simulate_deadline(
        self,
        target_date: datetime,
        target_clean_rate: float = 0.95,
        target_dblock_rate: float = 0.90,
        n_iterations: int = 5000
    ) -> ScenarioResult:
        """
        Simulate probability of meeting a deadline.
        
        Args:
            target_date: Target deadline
            target_clean_rate: Target clean patient rate
            target_dblock_rate: Target DB lock ready rate
            n_iterations: Monte Carlo iterations
            
        Returns:
            ScenarioResult with probability analysis
        """
        start_time = datetime.now()
        
        # Current state
        current_clean_rate = self.baseline_metrics.get('clean_rate', 0.55)
        current_dblock_rate = self.baseline_metrics.get('dblock_ready_rate', 0.28)
        total_issues = self.baseline_metrics.get('total_issues', 50000)
        total_patients = self.baseline_metrics.get('total_patients', 58000)
        
        days_until_deadline = (target_date - datetime.now()).days
        
        if days_until_deadline <= 0:
            return self._create_error_result("Target date is in the past", ScenarioType.DEADLINE_CHECK)
        
        # Get parameters
        params = self.default_parameters[ScenarioType.DEADLINE_CHECK]
        
        # Define timeline simulation
        def deadline_model(param_values: Dict) -> Dict:
            resolution_rate = param_values['resolution_rate_daily']
            new_issues_rate = param_values['new_issues_daily']
            clean_conversion = param_values['clean_conversion_rate']
            
            # Simulate day by day
            issues = total_issues
            clean_rate = current_clean_rate
            dblock_rate = current_dblock_rate
            
            for day in range(days_until_deadline):
                # Resolve issues
                resolved = min(issues, resolution_rate * (1 + np.random.normal(0, 0.1)))
                new_issues = new_issues_rate * (1 + np.random.normal(0, 0.2))
                issues = max(0, issues - resolved + new_issues)
                
                # Improve clean rate
                clean_improvement = clean_conversion * (1 - clean_rate)
                clean_rate = min(1.0, clean_rate + clean_improvement)
                
                # DB lock follows clean rate with lag
                dblock_rate = min(clean_rate, dblock_rate + clean_conversion * 0.5)
            
            return {
                'final_issues': issues,
                'final_clean_rate': clean_rate,
                'final_dblock_rate': dblock_rate,
                'clean_target_met': 1 if clean_rate >= target_clean_rate else 0,
                'dblock_target_met': 1 if dblock_rate >= target_dblock_rate else 0,
                'both_targets_met': 1 if (clean_rate >= target_clean_rate and dblock_rate >= target_dblock_rate) else 0,
                'days_to_clean_target': self._estimate_days_to_target(
                    current_clean_rate, target_clean_rate, clean_conversion
                ),
                'days_to_dblock_target': self._estimate_days_to_target(
                    current_dblock_rate, target_dblock_rate, clean_conversion * 0.5
                )
            }
        
        # Run Monte Carlo
        results = self.monte_carlo.simulate(deadline_model, params, n_iterations)
        
        # Calculate probabilities
        prob_clean_met = float(np.mean(results['clean_target_met']))
        prob_dblock_met = float(np.mean(results['dblock_target_met']))
        prob_both_met = float(np.mean(results['both_targets_met']))
        
        # Create outcomes
        outcomes = []
        
        # Clean rate outcome
        clean_band = UncertaintyBand.from_samples(results['final_clean_rate'])
        outcomes.append(ScenarioOutcome(
            metric=OutcomeMetric.CLEAN_RATE,
            baseline_value=current_clean_rate,
            projected_value=clean_band,
            change_absolute=clean_band.point_estimate - current_clean_rate,
            change_percent=100 * (clean_band.point_estimate - current_clean_rate) / current_clean_rate if current_clean_rate > 0 else 0,
            probability_improvement=float(np.mean(results['final_clean_rate'] > current_clean_rate)),
            probability_target_met=prob_clean_met,
            target_value=target_clean_rate
        ))
        
        # DB Lock rate outcome
        dblock_band = UncertaintyBand.from_samples(results['final_dblock_rate'])
        outcomes.append(ScenarioOutcome(
            metric=OutcomeMetric.DB_LOCK_DATE,
            baseline_value=current_dblock_rate,
            projected_value=dblock_band,
            change_absolute=dblock_band.point_estimate - current_dblock_rate,
            change_percent=100 * (dblock_band.point_estimate - current_dblock_rate) / current_dblock_rate if current_dblock_rate > 0 else 0,
            probability_improvement=float(np.mean(results['final_dblock_rate'] > current_dblock_rate)),
            probability_target_met=prob_dblock_met,
            target_value=target_dblock_rate
        ))
        
        # Issues remaining
        issues_band = UncertaintyBand.from_samples(results['final_issues'])
        outcomes.append(ScenarioOutcome(
            metric=OutcomeMetric.ISSUE_COUNT,
            baseline_value=total_issues,
            projected_value=issues_band,
            change_absolute=issues_band.point_estimate - total_issues,
            change_percent=100 * (issues_band.point_estimate - total_issues) / total_issues if total_issues > 0 else 0,
            probability_improvement=float(np.mean(results['final_issues'] < total_issues * 0.5))
        ))
        
        # Generate recommendations
        recommendations = []
        risks = []
        
        if prob_both_met >= 0.80:
            recommendations.append(f"High probability ({100*prob_both_met:.0f}%) of meeting deadline - maintain current pace")
        elif prob_both_met >= 0.50:
            recommendations.append(f"Moderate probability ({100*prob_both_met:.0f}%) - consider adding resources")
            risks.append("Timeline at risk without intervention")
        else:
            recommendations.append(f"Low probability ({100*prob_both_met:.0f}%) - significant intervention needed")
            risks.append("High risk of missing deadline")
            recommendations.append("Consider: adding resources, process improvements, or deadline extension")
        
        if prob_clean_met > prob_dblock_met:
            risks.append(f"DB Lock readiness ({100*prob_dblock_met:.0f}%) lagging behind clean rate ({100*prob_clean_met:.0f}%)")
        
        # Days buffer
        days_to_clean = np.mean(results['days_to_clean_target'])
        if days_to_clean < days_until_deadline * 0.8:
            recommendations.append(f"Buffer of ~{int(days_until_deadline - days_to_clean)} days to target")
        else:
            risks.append("Minimal timeline buffer")
        
        # Create result
        exec_time = (datetime.now() - start_time).total_seconds() * 1000
        
        result = ScenarioResult(
            scenario_id=f"SIM-{datetime.now().strftime('%Y%m%d%H%M%S')}-deadline",
            scenario_type=ScenarioType.DEADLINE_CHECK,
            title=f"Deadline Check: {target_date.strftime('%Y-%m-%d')}",
            description=f"Probability of meeting {target_date.strftime('%Y-%m-%d')} deadline with {target_clean_rate*100:.0f}% clean, {target_dblock_rate*100:.0f}% DB lock ready",
            parameters={
                'target_date': target_date.isoformat(),
                'days_until_deadline': days_until_deadline,
                'target_clean_rate': target_clean_rate,
                'target_dblock_rate': target_dblock_rate,
                'current_clean_rate': current_clean_rate,
                'current_dblock_rate': current_dblock_rate
            },
            outcomes=outcomes,
            simulation_method=SimulationMethod.MONTE_CARLO,
            n_iterations=n_iterations,
            execution_time_ms=exec_time,
            created_at=datetime.now(),
            recommendations=recommendations,
            risks=risks,
            confidence_level="HIGH" if n_iterations >= 5000 else "MEDIUM",
            summary=f"Deadline {target_date.strftime('%Y-%m-%d')} ({days_until_deadline} days): "
                    f"P(success)={100*prob_both_met:.0f}%, "
                    f"P(clean≥{100*target_clean_rate:.0f}%)={100*prob_clean_met:.0f}%, "
                    f"P(DBLock≥{100*target_dblock_rate:.0f}%)={100*prob_dblock_met:.0f}%"
        )
        
        self.scenarios_run.append(result)
        return result
    
    def _estimate_days_to_target(self, current: float, target: float, daily_rate: float) -> float:
        """Estimate days to reach target given daily improvement rate."""
        if current >= target:
            return 0
        if daily_rate <= 0:
            return float('inf')
        
        # Simple linear approximation
        gap = target - current
        days = gap / daily_rate
        return min(days, 365 * 2)  # Cap at 2 years
    
    def simulate_process_change(
        self,
        process_name: str,
        expected_improvement: float = 0.15,
        affected_issue_types: Optional[List[str]] = None,
        n_iterations: int = 5000
    ) -> ScenarioResult:
        """
        Simulate impact of a process change.
        
        Args:
            process_name: Name of the process change
            expected_improvement: Expected improvement factor (e.g., 0.15 = 15%)
            affected_issue_types: Issue types affected (None = all)
            n_iterations: Monte Carlo iterations
            
        Returns:
            ScenarioResult with impact analysis
        """
        start_time = datetime.now()
        
        total_issues = self.baseline_metrics.get('total_issues', 50000)
        
        # Get parameters
        params = self.default_parameters[ScenarioType.PROCESS_CHANGE].copy()
        params['improvement_factor'] = ScenarioParameter(
            'improvement_factor', expected_improvement,
            expected_improvement * 0.5, expected_improvement * 1.5,
            distribution='triangular'
        )
        
        # Define model
        def process_change_model(param_values: Dict) -> Dict:
            improvement = param_values['improvement_factor']
            adoption = param_values['adoption_rate']
            delay = param_values['implementation_delay_days']
            
            # Effective improvement
            effective_improvement = improvement * adoption
            
            # Issues reduced
            issues_reduced = total_issues * effective_improvement
            
            # Time saved (issues / daily resolution rate)
            days_saved = issues_reduced / 100
            
            # Net benefit after implementation delay
            net_days_saved = days_saved - delay * (1 - adoption)
            
            return {
                'effective_improvement': effective_improvement,
                'issues_reduced': issues_reduced,
                'days_saved': days_saved,
                'implementation_delay': delay,
                'net_days_saved': net_days_saved,
                'clean_rate_improvement': effective_improvement * 0.5
            }
        
        # Run Monte Carlo
        results = self.monte_carlo.simulate(process_change_model, params, n_iterations)
        
        # Create outcomes
        outcomes = []
        
        # Quality improvement
        quality_band = UncertaintyBand.from_samples(results['effective_improvement'] * 100)
        outcomes.append(ScenarioOutcome(
            metric=OutcomeMetric.QUALITY_SCORE,
            baseline_value=0,
            projected_value=quality_band,
            change_absolute=quality_band.point_estimate,
            change_percent=quality_band.point_estimate,
            probability_improvement=float(np.mean(results['effective_improvement'] > 0.05))
        ))
        
        # Timeline improvement
        timeline_band = UncertaintyBand.from_samples(results['net_days_saved'])
        outcomes.append(ScenarioOutcome(
            metric=OutcomeMetric.TIMELINE_DAYS,
            baseline_value=0,
            projected_value=timeline_band,
            change_absolute=-timeline_band.point_estimate,
            change_percent=0,
            probability_improvement=float(np.mean(results['net_days_saved'] > 0))
        ))
        
        # Issues reduced
        issues_band = UncertaintyBand.from_samples(results['issues_reduced'])
        outcomes.append(ScenarioOutcome(
            metric=OutcomeMetric.ISSUE_COUNT,
            baseline_value=total_issues,
            projected_value=issues_band,
            change_absolute=-issues_band.point_estimate,
            change_percent=-100 * issues_band.point_estimate / total_issues if total_issues > 0 else 0,
            probability_improvement=float(np.mean(results['issues_reduced'] > 1000))
        ))
        
        # Recommendations
        recommendations = []
        risks = []
        
        avg_improvement = np.mean(results['effective_improvement'])
        if avg_improvement > 0.10:
            recommendations.append(f"Strong expected improvement ({100*avg_improvement:.0f}%) - proceed with implementation")
        else:
            recommendations.append(f"Modest improvement expected ({100*avg_improvement:.0f}%) - ensure change is worth effort")
        
        avg_adoption = np.mean([params['adoption_rate'].base_value])
        if avg_adoption < 0.8:
            risks.append(f"Adoption risk: expecting only {100*avg_adoption:.0f}% adoption")
            recommendations.append("Develop strong change management plan")
        
        if np.mean(results['implementation_delay']) > 21:
            risks.append(f"Implementation delay of ~{np.mean(results['implementation_delay']):.0f} days")
        
        # Create result
        exec_time = (datetime.now() - start_time).total_seconds() * 1000
        
        result = ScenarioResult(
            scenario_id=f"SIM-{datetime.now().strftime('%Y%m%d%H%M%S')}-process",
            scenario_type=ScenarioType.PROCESS_CHANGE,
            title=f"Process Change: {process_name}",
            description=f"Impact of implementing '{process_name}' with expected {100*expected_improvement:.0f}% improvement",
            parameters={
                'process_name': process_name,
                'expected_improvement': expected_improvement,
                'affected_issue_types': affected_issue_types
            },
            outcomes=outcomes,
            simulation_method=SimulationMethod.MONTE_CARLO,
            n_iterations=n_iterations,
            execution_time_ms=exec_time,
            created_at=datetime.now(),
            recommendations=recommendations,
            risks=risks,
            confidence_level="HIGH" if n_iterations >= 5000 else "MEDIUM",
            summary=f"Process change '{process_name}': "
                    f"{100*avg_improvement:.0f}% effective improvement, "
                    f"~{int(np.mean(results['issues_reduced']))} issues reduced, "
                    f"{np.mean(results['net_days_saved']):.0f} net days saved"
        )
        
        self.scenarios_run.append(result)
        return result
    
    def compare_scenarios(
        self,
        scenario_ids: List[str],
        weights: Optional[Dict[str, float]] = None
    ) -> ComparisonResult:
        """
        Compare multiple scenarios.
        
        Args:
            scenario_ids: List of scenario IDs to compare
            weights: Optional weights for metrics (default: equal)
            
        Returns:
            ComparisonResult with ranking
        """
        # Get scenarios
        scenarios = [s for s in self.scenarios_run if s.scenario_id in scenario_ids]
        
        if len(scenarios) < 2:
            raise ValueError("Need at least 2 scenarios to compare")
        
        # Default weights
        if weights is None:
            weights = {
                'timeline': 0.30,
                'quality': 0.25,
                'cost': 0.20,
                'risk': 0.15,
                'probability': 0.10
            }
        
        # Score each scenario
        scores = {}
        decision_matrix = {}
        
        for scenario in scenarios:
            # Calculate composite score
            score = 0
            decision_matrix[scenario.scenario_id] = {}
            
            for outcome in scenario.outcomes:
                metric_score = 0
                
                if outcome.metric == OutcomeMetric.TIMELINE_DAYS:
                    # Lower is better
                    metric_score = max(0, 100 - abs(outcome.change_absolute))
                    decision_matrix[scenario.scenario_id]['timeline'] = metric_score
                    score += weights.get('timeline', 0.25) * metric_score
                
                elif outcome.metric in [OutcomeMetric.CLEAN_RATE, OutcomeMetric.DQI_SCORE, OutcomeMetric.QUALITY_SCORE]:
                    # Higher is better
                    metric_score = 50 + outcome.change_percent
                    decision_matrix[scenario.scenario_id]['quality'] = metric_score
                    score += weights.get('quality', 0.25) * metric_score
                
                elif outcome.metric == OutcomeMetric.COST:
                    # Lower is better
                    metric_score = max(0, 100 - outcome.projected_value.point_estimate / 10000)
                    decision_matrix[scenario.scenario_id]['cost'] = metric_score
                    score += weights.get('cost', 0.20) * metric_score
            
            # Risk penalty
            risk_penalty = len(scenario.risks) * 10
            decision_matrix[scenario.scenario_id]['risk'] = 100 - risk_penalty
            score -= weights.get('risk', 0.15) * risk_penalty
            
            scores[scenario.scenario_id] = score
        
        # Rank scenarios
        ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_scenario = ranking[0][0]
        
        # Generate summary
        best = next(s for s in scenarios if s.scenario_id == best_scenario)
        summary = f"Best scenario: {best.title} (score: {scores[best_scenario]:.1f}). "
        summary += f"Compared {len(scenarios)} scenarios. "
        if len(ranking) > 1:
            summary += f"Runner-up: {ranking[1][0]} (score: {ranking[1][1]:.1f})."
        
        return ComparisonResult(
            comparison_id=f"CMP-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            scenarios=scenarios,
            best_scenario=best_scenario,
            ranking=ranking,
            comparison_summary=summary,
            decision_matrix=decision_matrix,
            created_at=datetime.now()
        )
    
    def run_sensitivity_analysis(
        self,
        scenario_type: ScenarioType,
        base_params: Dict,
        vary_param: str,
        values: List[float],
        n_iterations: int = 1000
    ) -> Dict[str, List[ScenarioResult]]:
        """
        Run sensitivity analysis by varying a single parameter.
        
        Args:
            scenario_type: Type of scenario to run
            base_params: Base parameters
            vary_param: Parameter to vary
            values: Values to test
            n_iterations: Iterations per value
            
        Returns:
            Dictionary mapping values to results
        """
        results = {}
        
        for value in values:
            params = base_params.copy()
            params[vary_param] = value
            
            if scenario_type == ScenarioType.ADD_RESOURCE:
                result = self.simulate_add_resource(
                    resource_type=params.get('resource_type', ResourceType.CRA),
                    count=params.get('count', 1),
                    n_iterations=n_iterations
                )
            elif scenario_type == ScenarioType.PROCESS_CHANGE:
                result = self.simulate_process_change(
                    process_name=params.get('process_name', 'Test'),
                    expected_improvement=params.get('expected_improvement', 0.15),
                    n_iterations=n_iterations
                )
            else:
                continue
            
            results[str(value)] = result
        
        return results
    
    def _create_error_result(self, error_message: str, scenario_type: ScenarioType) -> ScenarioResult:
        """Create an error result."""
        return ScenarioResult(
            scenario_id=f"ERR-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            scenario_type=scenario_type,
            title="Error",
            description=error_message,
            parameters={},
            outcomes=[],
            simulation_method=SimulationMethod.DETERMINISTIC,
            n_iterations=0,
            execution_time_ms=0,
            created_at=datetime.now(),
            recommendations=[],
            risks=[error_message],
            confidence_level="NONE",
            summary=f"Error: {error_message}"
        )
    
    def get_summary(self) -> Dict:
        """Get simulator summary."""
        return {
            'baseline_metrics': self.baseline_metrics,
            'scenarios_run': len(self.scenarios_run),
            'data_loaded': {
                'patients': len(self.upr) if hasattr(self, 'upr') else 0,
                'issues': len(self.issues) if hasattr(self, 'issues') else 0
            }
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_simulator_instance: Optional[ScenarioSimulator] = None


def get_scenario_simulator() -> ScenarioSimulator:
    """Get singleton ScenarioSimulator instance."""
    global _simulator_instance
    if _simulator_instance is None:
        _simulator_instance = ScenarioSimulator()
    return _simulator_instance


def reset_scenario_simulator():
    """Reset simulator instance."""
    global _simulator_instance
    _simulator_instance = None


def simulate_close_site(site_id: str, **kwargs) -> ScenarioResult:
    """Quick function to simulate closing a site."""
    return get_scenario_simulator().simulate_close_site(site_id, **kwargs)


def simulate_add_resource(resource_type: ResourceType, count: int = 1, **kwargs) -> ScenarioResult:
    """Quick function to simulate adding resources."""
    return get_scenario_simulator().simulate_add_resource(resource_type, count, **kwargs)


def simulate_deadline(target_date: datetime, **kwargs) -> ScenarioResult:
    """Quick function to simulate deadline probability."""
    return get_scenario_simulator().simulate_deadline(target_date, **kwargs)


def simulate_process_change(process_name: str, expected_improvement: float = 0.15, **kwargs) -> ScenarioResult:
    """Quick function to simulate process change."""
    return get_scenario_simulator().simulate_process_change(process_name, expected_improvement, **kwargs)


# =============================================================================
# TEST FUNCTION
# =============================================================================

def test_scenario_simulator():
    """Test the Scenario Simulator."""
    print("=" * 70)
    print("SANCHALAK AI - SCENARIO SIMULATOR TEST")
    print("=" * 70)
    
    tests_passed = 0
    tests_failed = 0
    
    # Reset for clean test
    reset_scenario_simulator()
    
    # Test 1: Initialize
    print("\nTEST 1: Initialize Scenario Simulator")
    try:
        simulator = get_scenario_simulator()
        assert simulator is not None
        summary = simulator.get_summary()
        print(f"  ✅ Initialized with {summary['data_loaded']['patients']} patients")
        print(f"  ✅ Baseline metrics: {len(summary['baseline_metrics'])} metrics")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        tests_failed += 1
    
    # Test 2: Monte Carlo Engine
    print("\nTEST 2: Monte Carlo Engine")
    try:
        mc = MonteCarloEngine(n_iterations=1000, random_seed=42)
        
        params = {
            'rate': ScenarioParameter('rate', 0.5, 0.3, 0.7, distribution='normal', std_dev=0.1)
        }
        
        def simple_model(p):
            return {'output': p['rate'] * 100}
        
        results = mc.simulate(simple_model, params, 1000)
        assert 'output' in results
        assert len(results['output']) == 1000
        
        band = UncertaintyBand.from_samples(results['output'])
        assert 40 < band.point_estimate < 60
        
        print(f"  ✅ Simulated 1,000 iterations")
        print(f"  ✅ Point estimate: {band.point_estimate:.1f}")
        print(f"  ✅ 95% CI: [{band.ci_95_lower:.1f}, {band.ci_95_upper:.1f}]")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        tests_failed += 1
    
    # Test 3: Close Site Simulation
    print("\nTEST 3: Simulate Close Site")
    try:
        # Get a real site ID
        if len(simulator.upr) > 0:
            site_col = 'site_id' if 'site_id' in simulator.upr.columns else 'site'
            if site_col in simulator.upr.columns:
                site_id = simulator.upr[site_col].iloc[0]
            else:
                site_id = "Site_1"
        else:
            site_id = "Site_1"
        
        result = simulator.simulate_close_site(site_id, n_iterations=1000)
        
        assert result is not None
        assert result.scenario_type == ScenarioType.CLOSE_SITE
        assert len(result.outcomes) >= 2
        
        print(f"  ✅ Simulated closing site: {site_id}")
        print(f"  ✅ Summary: {result.summary[:100]}...")
        print(f"  ✅ Recommendations: {len(result.recommendations)}")
        print(f"  ✅ Risks: {len(result.risks)}")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        tests_failed += 1
    
    # Test 4: Add Resource Simulation
    print("\nTEST 4: Simulate Add Resource")
    try:
        result = simulator.simulate_add_resource(
            resource_type=ResourceType.CRA,
            count=2,
            n_iterations=1000
        )
        
        assert result is not None
        assert result.scenario_type == ScenarioType.ADD_RESOURCE
        assert 'resource_type' in result.parameters
        
        # Find timeline outcome
        timeline_outcome = next((o for o in result.outcomes if o.metric == OutcomeMetric.TIMELINE_DAYS), None)
        
        print(f"  ✅ Simulated adding 2 CRAs")
        print(f"  ✅ Summary: {result.summary}")
        if timeline_outcome:
            print(f"  ✅ Timeline improvement: {timeline_outcome.projected_value.point_estimate:.1f} days")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        tests_failed += 1
    
    # Test 5: Deadline Simulation
    print("\nTEST 5: Simulate Deadline Check")
    try:
        target_date = datetime.now() + timedelta(days=90)
        
        result = simulator.simulate_deadline(
            target_date=target_date,
            target_clean_rate=0.95,
            target_dblock_rate=0.90,
            n_iterations=1000
        )
        
        assert result is not None
        assert result.scenario_type == ScenarioType.DEADLINE_CHECK
        
        # Find clean rate outcome
        clean_outcome = next((o for o in result.outcomes if o.metric == OutcomeMetric.CLEAN_RATE), None)
        
        print(f"  ✅ Simulated deadline: {target_date.strftime('%Y-%m-%d')}")
        print(f"  ✅ Summary: {result.summary}")
        if clean_outcome and clean_outcome.probability_target_met is not None:
            print(f"  ✅ P(clean target met): {100*clean_outcome.probability_target_met:.0f}%")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        tests_failed += 1
    
    # Test 6: Process Change Simulation
    print("\nTEST 6: Simulate Process Change")
    try:
        result = simulator.simulate_process_change(
            process_name="Enhanced SDV Process",
            expected_improvement=0.20,
            n_iterations=1000
        )
        
        assert result is not None
        assert result.scenario_type == ScenarioType.PROCESS_CHANGE
        
        print(f"  ✅ Simulated process change")
        print(f"  ✅ Summary: {result.summary}")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        tests_failed += 1
    
    # Test 7: Scenario Comparison
    print("\nTEST 7: Compare Scenarios")
    try:
        # Run another scenario for comparison
        result2 = simulator.simulate_add_resource(
            resource_type=ResourceType.DATA_MANAGER,
            count=1,
            n_iterations=500
        )
        
        # Get scenario IDs
        scenario_ids = [s.scenario_id for s in simulator.scenarios_run[-2:]]
        
        comparison = simulator.compare_scenarios(scenario_ids)
        
        assert comparison is not None
        assert comparison.best_scenario in scenario_ids
        
        print(f"  ✅ Compared {len(comparison.scenarios)} scenarios")
        print(f"  ✅ Best scenario: {comparison.best_scenario}")
        print(f"  ✅ Summary: {comparison.comparison_summary}")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        tests_failed += 1
    
    # Test 8: Uncertainty Band
    print("\nTEST 8: Uncertainty Band Calculations")
    try:
        samples = np.random.normal(50, 10, 10000)
        band = UncertaintyBand.from_samples(samples)
        
        assert 45 < band.point_estimate < 55
        assert band.ci_95_lower < band.point_estimate < band.ci_95_upper
        assert band.std_error > 0
        
        band_dict = band.to_dict()
        assert 'point_estimate' in band_dict
        assert 'ci_95' in band_dict
        
        print(f"  ✅ Point estimate: {band.point_estimate:.2f}")
        print(f"  ✅ 95% CI: [{band.ci_95_lower:.2f}, {band.ci_95_upper:.2f}]")
        print(f"  ✅ Standard error: {band.std_error:.2f}")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        tests_failed += 1
    
    # Test 9: Scenario Parameter Sampling
    print("\nTEST 9: Scenario Parameter Sampling")
    try:
        param = ScenarioParameter(
            name='test_param',
            base_value=0.5,
            min_value=0.2,
            max_value=0.8,
            distribution='triangular'
        )
        
        samples = param.sample(1000, random_state=42)
        
        assert len(samples) == 1000
        assert all(0.2 <= s <= 0.8 for s in samples)
        assert 0.4 < np.mean(samples) < 0.6  # Should be near base value
        
        print(f"  ✅ Sampled 1,000 values")
        print(f"  ✅ Mean: {np.mean(samples):.3f}")
        print(f"  ✅ Range: [{np.min(samples):.3f}, {np.max(samples):.3f}]")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        tests_failed += 1
    
    # Test 10: Convergence Check
    print("\nTEST 10: Convergence Check")
    try:
        mc = MonteCarloEngine(convergence_threshold=0.02)
        
        # Stable samples
        stable_samples = np.random.normal(50, 5, 10000)
        converged, diff = mc.check_convergence(stable_samples)
        
        assert converged
        assert diff < 0.02
        
        print(f"  ✅ Converged: {converged}")
        print(f"  ✅ Relative difference: {diff:.4f}")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        tests_failed += 1
    
    # Test 11: Scenarios Run History
    print("\nTEST 11: Scenarios Run History")
    try:
        history = simulator.scenarios_run
        assert len(history) >= 4  # We ran at least 4 scenarios
        
        # Check each scenario has required fields
        for scenario in history[:3]:
            assert scenario.scenario_id is not None
            assert scenario.created_at is not None
            assert scenario.summary is not None
        
        print(f"  ✅ Scenarios in history: {len(history)}")
        print(f"  ✅ Latest: {history[-1].title}")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        tests_failed += 1
    
    # Test 12: Convenience Functions
    print("\nTEST 12: Convenience Functions")
    try:
        # Test quick functions
        result1 = simulate_add_resource(ResourceType.MEDICAL_CODER, 1, n_iterations=100)
        assert result1 is not None
        
        result2 = simulate_process_change("Quick Test", 0.10, n_iterations=100)
        assert result2 is not None
        
        print(f"  ✅ simulate_add_resource: {result1.summary[:50]}...")
        print(f"  ✅ simulate_process_change: {result2.summary[:50]}...")
        tests_passed += 1
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        tests_failed += 1
    
    # Summary
    print("\n" + "=" * 70)
    print(f"TESTS PASSED: {tests_passed}/{tests_passed + tests_failed}")
    if tests_failed == 0:
        print("✅ ALL TESTS PASSED!")
    else:
        print(f"❌ {tests_failed} TESTS FAILED")
    print("=" * 70)
    
    return tests_passed, tests_failed


if __name__ == "__main__":
    test_scenario_simulator()