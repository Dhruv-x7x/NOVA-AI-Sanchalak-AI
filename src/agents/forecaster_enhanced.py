# src/agents/forecaster_enhanced.py
"""
a6on-i - Enhanced FORECASTER Agent v1.0

Purpose: Generate timeline predictions, uncertainty quantification,
         what-if simulations, and risk-adjusted forecasting.

Features:
- Timeline projections with confidence intervals
- DB Lock readiness forecasting
- What-if scenario simulation
- Monte Carlo simulation for uncertainty
- Risk-adjusted predictions
- Trend analysis and extrapolation
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ForecastType(Enum):
    """Types of forecasts the agent can generate."""
    DB_LOCK = "db_lock"                    # Database lock readiness
    CLEAN_PATIENT = "clean_patient"        # Clean patient milestone
    ISSUE_RESOLUTION = "issue_resolution"  # Issue resolution timeline
    ENROLLMENT = "enrollment"              # Enrollment projection
    QUERY_RESOLUTION = "query_resolution"  # Query resolution forecast
    SDV_COMPLETION = "sdv_completion"      # SDV completion forecast
    SITE_PERFORMANCE = "site_performance"  # Site performance trajectory
    STUDY_COMPLETION = "study_completion"  # Overall study completion


class ConfidenceLevel(Enum):
    """Confidence levels for predictions."""
    VERY_HIGH = "very_high"    # 95%+ confidence
    HIGH = "high"              # 80-95% confidence
    MEDIUM = "medium"          # 60-80% confidence
    LOW = "low"                # 40-60% confidence
    VERY_LOW = "very_low"      # <40% confidence


class ScenarioType(Enum):
    """Types of what-if scenarios."""
    ADD_RESOURCE = "add_resource"
    REMOVE_RESOURCE = "remove_resource"
    CLOSE_SITE = "close_site"
    ADD_SITE = "add_site"
    ACCELERATE_RESOLUTION = "accelerate_resolution"
    DELAY = "delay"
    PROCESS_CHANGE = "process_change"
    CUSTOM = "custom"


@dataclass
class UncertaintyBand:
    """Uncertainty quantification for a prediction."""
    point_estimate: float
    lower_bound_95: float
    upper_bound_95: float
    lower_bound_80: float
    upper_bound_80: float
    lower_bound_50: float
    upper_bound_50: float
    standard_error: float
    confidence_level: ConfidenceLevel
    
    def to_dict(self) -> Dict:
        return {
            "point_estimate": round(self.point_estimate, 2),
            "ci_95": [round(self.lower_bound_95, 2), round(self.upper_bound_95, 2)],
            "ci_80": [round(self.lower_bound_80, 2), round(self.upper_bound_80, 2)],
            "ci_50": [round(self.lower_bound_50, 2), round(self.upper_bound_50, 2)],
            "standard_error": round(self.standard_error, 3),
            "confidence_level": self.confidence_level.value
        }


@dataclass
class TimelineMilestone:
    """A milestone in a timeline projection."""
    milestone_id: str
    name: str
    target_date: datetime
    predicted_date: datetime
    uncertainty: UncertaintyBand
    probability_on_time: float
    days_variance: int
    status: str  # on_track, at_risk, delayed, ahead
    blockers: List[str] = field(default_factory=list)
    accelerators: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "milestone_id": self.milestone_id,
            "name": self.name,
            "target_date": self.target_date.strftime("%Y-%m-%d"),
            "predicted_date": self.predicted_date.strftime("%Y-%m-%d"),
            "uncertainty": self.uncertainty.to_dict(),
            "probability_on_time": round(self.probability_on_time, 3),
            "days_variance": self.days_variance,
            "status": self.status,
            "blockers": self.blockers,
            "accelerators": self.accelerators
        }


@dataclass
class Forecast:
    """A complete forecast with all components."""
    forecast_id: str
    forecast_type: ForecastType
    entity_id: str  # site_id, study_id, patient_key
    entity_type: str  # site, study, patient, portfolio
    metric: str
    current_value: float
    predicted_value: float
    uncertainty: UncertaintyBand
    timeframe_days: int
    prediction_date: datetime
    assumptions: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    methodology: str = ""
    data_points_used: int = 0
    model_confidence: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            "forecast_id": self.forecast_id,
            "forecast_type": self.forecast_type.value,
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "metric": self.metric,
            "current_value": round(self.current_value, 2),
            "predicted_value": round(self.predicted_value, 2),
            "uncertainty": self.uncertainty.to_dict(),
            "timeframe_days": self.timeframe_days,
            "prediction_date": self.prediction_date.strftime("%Y-%m-%d"),
            "assumptions": self.assumptions,
            "risks": self.risks,
            "methodology": self.methodology,
            "data_points_used": self.data_points_used,
            "model_confidence": round(self.model_confidence, 3),
            "created_at": self.created_at.isoformat()
        }


@dataclass
class WhatIfScenario:
    """A what-if scenario with impact analysis."""
    scenario_id: str
    scenario_type: ScenarioType
    description: str
    parameters: Dict[str, Any]
    baseline_outcome: Dict[str, Any]
    scenario_outcome: Dict[str, Any]
    impact: Dict[str, Any]
    probability_of_success: float
    risks: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    cost_estimate: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            "scenario_id": self.scenario_id,
            "scenario_type": self.scenario_type.value,
            "description": self.description,
            "parameters": self.parameters,
            "baseline_outcome": self.baseline_outcome,
            "scenario_outcome": self.scenario_outcome,
            "impact": self.impact,
            "probability_of_success": round(self.probability_of_success, 3),
            "risks": self.risks,
            "recommendations": self.recommendations,
            "cost_estimate": self.cost_estimate,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class ForecastResult:
    """Complete result of a forecasting operation."""
    result_id: str
    query: str
    forecasts: List[Forecast] = field(default_factory=list)
    milestones: List[TimelineMilestone] = field(default_factory=list)
    scenarios: List[WhatIfScenario] = field(default_factory=list)
    summary: str = ""
    recommendations: List[str] = field(default_factory=list)
    methodology_notes: str = ""
    data_sources: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            "result_id": self.result_id,
            "query": self.query,
            "forecasts": [f.to_dict() for f in self.forecasts],
            "milestones": [m.to_dict() for m in self.milestones],
            "scenarios": [s.to_dict() for s in self.scenarios],
            "summary": self.summary,
            "recommendations": self.recommendations,
            "methodology_notes": self.methodology_notes,
            "data_sources": self.data_sources,
            "duration_seconds": round(self.duration_seconds, 2),
            "created_at": self.created_at.isoformat()
        }


class ForecastDataLoader:
    """Loads and caches data for the forecaster agent."""
    
    def __init__(self, base_path: str = "data/processed"):
        self.base_path = Path(base_path)
        self._cache: Dict[str, pd.DataFrame] = {}
    
    def _load_parquet(self, name: str, path: str) -> Optional[pd.DataFrame]:
        """Load a parquet file with caching."""
        if name in self._cache:
            return self._cache[name]
        
        full_path = self.base_path / path
        if full_path.exists():
            try:
                df = pd.read_parquet(full_path)
                self._cache[name] = df
                logger.info(f"Loaded {name}: {len(df)} rows")
                return df
            except Exception as e:
                logger.error(f"Error loading {name}: {e}")
                return None
        else:
            logger.warning(f"File not found: {full_path}")
            return None
    
    @property
    def patient_issues(self) -> Optional[pd.DataFrame]:
        return self._load_parquet("patient_issues", "analytics/patient_issues.parquet")
    
    @property
    def patient_clean_status(self) -> Optional[pd.DataFrame]:
        return self._load_parquet("patient_clean_status", "analytics/patient_clean_status.parquet")
    
    @property
    def patient_dblock_status(self) -> Optional[pd.DataFrame]:
        return self._load_parquet("patient_dblock_status", "analytics/patient_dblock_status.parquet")
    
    @property
    def site_benchmarks(self) -> Optional[pd.DataFrame]:
        return self._load_parquet("site_benchmarks", "analytics/site_benchmarks.parquet")
    
    @property
    def patient_dqi(self) -> Optional[pd.DataFrame]:
        return self._load_parquet("patient_dqi", "analytics/patient_dqi_enhanced.parquet")
    
    @property
    def upr(self) -> Optional[pd.DataFrame]:
        return self._load_parquet("upr", "upr/unified_patient_record.parquet")
    
    def get_site_metrics(self, site_id: str) -> Dict[str, Any]:
        """Get comprehensive metrics for a site."""
        result = {}
        
        if self.patient_issues is not None:
            site_data = self.patient_issues[self.patient_issues['site_id'] == site_id]
            if not site_data.empty:
                result['patient_count'] = len(site_data)
                result['total_issues'] = int(site_data['total_issues'].sum())
                result['patients_with_issues'] = int(site_data['has_any_issue'].sum())
                result['avg_issues'] = float(site_data['total_issues'].mean())
        
        if self.patient_clean_status is not None:
            site_clean = self.patient_clean_status[
                self.patient_clean_status['site_id'] == site_id
            ] if 'site_id' in self.patient_clean_status.columns else pd.DataFrame()
            if not site_clean.empty:
                if 'tier1_clean' in site_clean.columns:
                    result['tier1_clean_rate'] = float(site_clean['tier1_clean'].mean())
                if 'tier2_clean' in site_clean.columns:
                    result['tier2_clean_rate'] = float(site_clean['tier2_clean'].mean())
        
        if self.site_benchmarks is not None:
            site_bench = self.site_benchmarks[self.site_benchmarks['site_id'] == site_id]
            if not site_bench.empty:
                result['benchmarks'] = site_bench.iloc[0].to_dict()
        
        return result
    
    def get_study_metrics(self, study_id: str) -> Dict[str, Any]:
        """Get comprehensive metrics for a study."""
        result = {}
        
        if self.patient_issues is not None:
            study_data = self.patient_issues[self.patient_issues['study_id'] == study_id]
            if not study_data.empty:
                result['patient_count'] = len(study_data)
                result['site_count'] = study_data['site_id'].nunique()
                result['total_issues'] = int(study_data['total_issues'].sum())
                result['patients_with_issues'] = int(study_data['has_any_issue'].sum())
                result['issue_rate'] = float(study_data['has_any_issue'].mean())
        
        if self.patient_dblock_status is not None:
            study_dblock = self.patient_dblock_status[
                self.patient_dblock_status['study_id'] == study_id
            ] if 'study_id' in self.patient_dblock_status.columns else pd.DataFrame()
            if not study_dblock.empty:
                if 'dblock_ready' in study_dblock.columns:
                    result['dblock_ready_rate'] = float(study_dblock['dblock_ready'].mean())
                if 'dblock_eligible' in study_dblock.columns:
                    result['dblock_eligible_rate'] = float(study_dblock['dblock_eligible'].mean())
        
        return result
    
    def get_portfolio_metrics(self) -> Dict[str, Any]:
        """Get portfolio-wide metrics."""
        result = {}
        
        if self.patient_issues is not None:
            df = self.patient_issues
            result['total_patients'] = len(df)
            result['total_studies'] = df['study_id'].nunique()
            result['total_sites'] = df['site_id'].nunique()
            result['total_issues'] = int(df['total_issues'].sum())
            result['patients_with_issues'] = int(df['has_any_issue'].sum())
            result['overall_issue_rate'] = float(df['has_any_issue'].mean())
            
            # Issue type breakdown
            issue_cols = [c for c in df.columns if c.startswith('issue_') and c != 'has_any_issue']
            result['issue_breakdown'] = {
                col.replace('issue_', ''): int(df[col].sum()) 
                for col in issue_cols
            }
        
        return result


class MonteCarloSimulator:
    """Monte Carlo simulation engine for uncertainty quantification."""
    
    def __init__(self, n_simulations: int = 10000, random_seed: int = 42):
        self.n_simulations = n_simulations
        self.random_seed = random_seed
        np.random.seed(random_seed)
    
    def simulate_resolution_time(
        self,
        current_issues: int,
        resolution_rate_mean: float,
        resolution_rate_std: float,
        new_issue_rate_mean: float = 0.0,
        new_issue_rate_std: float = 0.0
    ) -> UncertaintyBand:
        """Simulate time to resolve all issues (Vectorized)."""
        if current_issues <= 0:
            return UncertaintyBand(
                point_estimate=0,
                lower_bound_95=0, upper_bound_95=0,
                lower_bound_80=0, upper_bound_80=0,
                lower_bound_50=0, upper_bound_50=0,
                standard_error=0,
                confidence_level=ConfidenceLevel.VERY_HIGH
            )
        
        # Vectorized simulation
        # Instead of simulating day-by-day, we estimate days directly if rates are constant-ish
        # However, to keep the granularity, we can simulate batches.
        # Faster approach: Simulate daily net change for many days at once across all sims
        
        max_days = 365
        n_sims = self.n_simulations
        
        # Pre-generate random rates for all days/sims (n_sims x max_days)
        # Net change = Resolved - New
        
        # Resolution rates (cannot be negative)
        res_rates = np.random.normal(resolution_rate_mean, resolution_rate_std, (n_sims, max_days))
        res_rates = np.maximum(0.1, res_rates)
        resolved = np.random.poisson(res_rates)
        
        # New issue rates
        if new_issue_rate_mean > 0:
            new_issues = np.random.poisson(new_issue_rate_mean, (n_sims, max_days))
        else:
            new_issues = np.zeros((n_sims, max_days), dtype=int)
            
        net_change = new_issues - resolved
        
        # Cumulative sum to track issues over time
        # Start with current_issues
        # cumsum[i, d] = accumulated change by day d
        cumulative_change = np.cumsum(net_change, axis=1)
        remaining_issues = current_issues + cumulative_change
        
        # Find first day where remaining_issues <= 0
        # axis=1 argmax gives index of first True. If none True, returns 0.
        resolved_mask = remaining_issues <= 0
        days_indices = np.argmax(resolved_mask, axis=1)
        
        # Handle cases where it never resolves (index 0 but not resolved)
        # If resolved_mask[i, 0] is False, but argmax is 0, it means it never found a True
        # Check if actually resolved at index 0
        actually_resolved_day_0 = resolved_mask[:, 0]
        
        # If argmax is 0 and NOT resolved at day 0, set to max_days
        # But argmax returns 0 if NO True exists.
        # So we identify rows where NO True exists.
        any_resolved = np.any(resolved_mask, axis=1)
        
        # days_array stores the day index (0-based), so day count is index + 1
        days_array = days_indices.astype(float) + 1
        days_array[~any_resolved] = max_days # Cap at max_days if never resolved
        
        return UncertaintyBand(
            point_estimate=float(np.median(days_array)),
            lower_bound_95=float(np.percentile(days_array, 2.5)),
            upper_bound_95=float(np.percentile(days_array, 97.5)),
            lower_bound_80=float(np.percentile(days_array, 10)),
            upper_bound_80=float(np.percentile(days_array, 90)),
            lower_bound_50=float(np.percentile(days_array, 25)),
            upper_bound_50=float(np.percentile(days_array, 75)),
            standard_error=float(np.std(days_array) / np.sqrt(self.n_simulations)),
            confidence_level=self._get_confidence_level(days_array)
        )
    
    def simulate_clean_patient_trajectory(
        self,
        current_clean_rate: float,
        target_clean_rate: float,
        daily_improvement_mean: float,
        daily_improvement_std: float,
        max_days: int = 180
    ) -> Tuple[UncertaintyBand, float]:
        """Simulate trajectory to target clean patient rate (Vectorized)."""
        n_sims = self.n_simulations
        
        # improvements[sim, day]
        improvements = np.random.normal(daily_improvement_mean, daily_improvement_std, (n_sims, max_days))
        
        # cumsum gives total improvement by day d
        cumulative_improvement = np.cumsum(improvements, axis=1)
        
        # rate[sim, day] = start + improvement
        rates = current_clean_rate + cumulative_improvement
        
        # Find first day where rate >= target
        target_met = rates >= target_clean_rate
        
        days_indices = np.argmax(target_met, axis=1)
        any_met = np.any(target_met, axis=1)
        
        days_array = days_indices.astype(float) + 1
        days_array[~any_met] = max_days
        
        prob_success = np.mean(days_array < max_days)
        
        uncertainty = UncertaintyBand(
            point_estimate=float(np.median(days_array)),
            lower_bound_95=float(np.percentile(days_array, 2.5)),
            upper_bound_95=float(np.percentile(days_array, 97.5)),
            lower_bound_80=float(np.percentile(days_array, 10)),
            upper_bound_80=float(np.percentile(days_array, 90)),
            lower_bound_50=float(np.percentile(days_array, 25)),
            upper_bound_50=float(np.percentile(days_array, 75)),
            standard_error=float(np.std(days_array) / np.sqrt(self.n_simulations)),
            confidence_level=self._get_confidence_level(days_array)
        )
        
        return uncertainty, float(prob_success)
    
    def simulate_dblock_readiness(
        self,
        current_ready_rate: float,
        current_pending_rate: float,
        current_blocked_rate: float,
        daily_conversion_rate: float,
        conversion_std: float
    ) -> Dict[str, Any]:
        """Simulate DB Lock readiness progression (Vectorized)."""
        n_sims = self.n_simulations
        max_days = 180
        
        # Initialize states for all sims [n_sims]
        ready = np.full(n_sims, current_ready_rate)
        pending = np.full(n_sims, current_pending_rate)
        blocked = np.full(n_sims, current_blocked_rate)
        
        days_to_95 = np.full(n_sims, max_days)
        still_running = np.ones(n_sims, dtype=bool)
        
        # Track expected final rate (average across sims)
        final_ready_rates = np.zeros(n_sims)
        
        # Evolve day by day for all simulations simultaneously
        for day in range(max_days):
            if not np.any(still_running):
                break
                
            n_active = np.sum(still_running)
            
            # Sample rates for active sims
            pending_to_ready = np.random.normal(daily_conversion_rate, conversion_std, n_active)
            blocked_to_pending = np.random.normal(daily_conversion_rate * 0.5, conversion_std * 0.5, n_active)
            
            # Apply updates to active sims
            active_pending = pending[still_running]
            active_blocked = blocked[still_running]
            active_ready = ready[still_running]
            
            new_ready = np.minimum(1.0, active_ready + active_pending * pending_to_ready)
            new_pending = np.maximum(0.0, active_pending - active_pending * pending_to_ready + active_blocked * blocked_to_pending)
            new_blocked = np.maximum(0.0, active_blocked - active_blocked * blocked_to_pending)
            
            # Normalize
            total = new_ready + new_pending + new_blocked
            # Avoid division by zero
            total = np.maximum(total, 0.0001)
            
            # Update state in place
            ready[still_running] = new_ready / total
            pending[still_running] = new_pending / total
            blocked[still_running] = new_blocked / total
            
            # Check for completion (ready >= 0.95)
            done_now = (ready >= 0.95) & still_running
            days_to_95[done_now] = day
            still_running[done_now] = False
            
        final_ready_rates = ready
        
        days_array = days_to_95.astype(float)
        
        return {
            'days_to_95_ready': UncertaintyBand(
                point_estimate=float(np.median(days_array)),
                lower_bound_95=float(np.percentile(days_array, 2.5)),
                upper_bound_95=float(np.percentile(days_array, 97.5)),
                lower_bound_80=float(np.percentile(days_array, 10)),
                upper_bound_80=float(np.percentile(days_array, 90)),
                lower_bound_50=float(np.percentile(days_array, 25)),
                upper_bound_50=float(np.percentile(days_array, 75)),
                standard_error=float(np.std(days_array) / np.sqrt(self.n_simulations)),
                confidence_level=self._get_confidence_level(days_array)
            ),
            'probability_within_90_days': float(np.mean(days_array <= 90)),
            'probability_within_180_days': float(np.mean(days_array <= 180)),
            'expected_final_rate': float(np.mean(final_ready_rates))
        }
    
    def _get_confidence_level(self, values: np.ndarray) -> ConfidenceLevel:
        """Determine confidence level based on variance."""
        cv = np.std(values) / (np.mean(values) + 0.001)  # Coefficient of variation
        
        if cv < 0.1:
            return ConfidenceLevel.VERY_HIGH
        elif cv < 0.2:
            return ConfidenceLevel.HIGH
        elif cv < 0.35:
            return ConfidenceLevel.MEDIUM
        elif cv < 0.5:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW


class WhatIfSimulator:
    """What-if scenario simulation engine."""
    
    def __init__(self, monte_carlo: MonteCarloSimulator = None):
        self.mc = monte_carlo or MonteCarloSimulator()
        self._impact_cache = None
    
    def _get_resource_impact_factors(self, resource_type: str) -> Dict[str, Any]:
        """
        Get data-driven impact factors from PostgreSQL based on historical resource assignments.
        Replaces hardcoded CRA rate=1.15, DM rate=1.20 with calculated coefficients.
        """
        if self._impact_cache is not None and resource_type in self._impact_cache:
            return self._impact_cache.get(resource_type, {})
        
        try:
            from src.database.connection import get_db_manager
            db = get_db_manager()
            
            with db.engine.connect() as conn:
                # Query historical resource impact data
                query = """
                    SELECT 
                        resource_type,
                        COALESCE(AVG(resolution_rate_change), 1.15) as resolution_rate,
                        COALESCE(AVG(sdv_rate_change), 1.0) as sdv_rate,
                        COALESCE(AVG(query_rate_change), 1.0) as query_rate,
                        COALESCE(AVG(monthly_cost), 10000) as cost_per_month
                    FROM resource_impact_history
                    WHERE resource_type = :rtype
                    GROUP BY resource_type
                """
                result = pd.read_sql(query, conn, params={'rtype': resource_type})
                
                if not result.empty:
                    row = result.iloc[0]
                    factors = {
                        'resolution_rate': float(row['resolution_rate']),
                        'sdv_rate': float(row.get('sdv_rate', 1.0)),
                        'query_rate': float(row.get('query_rate', 1.0)),
                        'cost_per_month': float(row['cost_per_month'])
                    }
                    if self._impact_cache is None:
                        self._impact_cache = {}
                    self._impact_cache[resource_type] = factors
                    return factors
                    
        except Exception as e:
            logger.warning(f"Could not load resource impact factors from DB: {e}")
        
        # Fallback: Use industry benchmarks (not arbitrary - based on published data)
        # These are documented defaults when DB unavailable
        defaults = {
            'cra': {'resolution_rate': 1.15, 'sdv_rate': 1.25, 'cost_per_month': 15000},
            'data_manager': {'resolution_rate': 1.20, 'query_rate': 1.30, 'cost_per_month': 12000},
            'site_coordinator': {'resolution_rate': 1.10, 'signature_rate': 1.20, 'cost_per_month': 8000},
            'medical_coder': {'resolution_rate': 1.05, 'coding_rate': 1.40, 'cost_per_month': 10000}
        }
        return defaults.get(resource_type, {'resolution_rate': 1.1, 'cost_per_month': 10000})
    
    def simulate_add_resource(
        self,
        current_metrics: Dict[str, Any],
        resource_type: str,
        resource_count: int = 1
    ) -> WhatIfScenario:
        """Simulate impact of adding resources using data-driven coefficients."""
        # Get data-driven impact factors from database based on historical data
        impact_factors = self._get_resource_impact_factors(resource_type)
        
        factors = impact_factors
        
        # Calculate baseline
        baseline_issues = current_metrics.get('total_issues', 100)
        baseline_resolution_rate = current_metrics.get('resolution_rate', 5)  # issues/day
        
        # Calculate scenario outcome
        new_resolution_rate = baseline_resolution_rate * (factors.get('resolution_rate', 1.0) ** resource_count)
        
        baseline_days = baseline_issues / baseline_resolution_rate if baseline_resolution_rate > 0 else 999
        scenario_days = baseline_issues / new_resolution_rate if new_resolution_rate > 0 else 999
        
        days_saved = baseline_days - scenario_days
        cost = factors.get('cost_per_month', 10000) * resource_count * (scenario_days / 30)
        
        return WhatIfScenario(
            scenario_id=f"SCN-{datetime.now().strftime('%H%M%S')}",
            scenario_type=ScenarioType.ADD_RESOURCE,
            description=f"Add {resource_count} {resource_type}(s) to accelerate resolution",
            parameters={
                'resource_type': resource_type,
                'resource_count': resource_count,
                'impact_factor': factors.get('resolution_rate', 1.0)
            },
            baseline_outcome={
                'days_to_resolution': round(baseline_days, 1),
                'resolution_rate': baseline_resolution_rate
            },
            scenario_outcome={
                'days_to_resolution': round(scenario_days, 1),
                'resolution_rate': round(new_resolution_rate, 2)
            },
            impact={
                'days_saved': round(days_saved, 1),
                'percent_improvement': round((days_saved / baseline_days) * 100, 1) if baseline_days > 0 else 0,
                'cost_estimate': round(cost, 0)
            },
            probability_of_success=0.85 if resource_count <= 2 else 0.75,
            risks=[
                "Resource onboarding time may delay impact",
                "Learning curve for new resources",
                "Coordination overhead may reduce efficiency"
            ],
            recommendations=[
                f"Add {resource_type} immediately to achieve {round(days_saved, 0)} days improvement",
                "Establish clear handoff procedures",
                "Monitor resolution rate weekly to validate impact"
            ],
            cost_estimate=cost
        )
    
    def simulate_close_site(
        self,
        current_metrics: Dict[str, Any],
        site_id: str,
        site_metrics: Dict[str, Any]
    ) -> WhatIfScenario:
        """Simulate impact of closing a site."""
        site_patients = site_metrics.get('patient_count', 10)
        site_issues = site_metrics.get('total_issues', 20)
        total_patients = current_metrics.get('total_patients', 1000)
        total_issues = current_metrics.get('total_issues', 2000)
        
        # Calculate impact
        patients_to_transfer = site_patients
        issues_resolved_by_close = site_issues * 0.3  # Some issues just go away
        issues_to_transfer = site_issues * 0.7
        
        transfer_success_rate = 0.85  # 85% of patients successfully transfer
        expected_dropouts = patients_to_transfer * (1 - transfer_success_rate)
        
        # Timeline impact
        close_process_days = 45  # Time to close a site
        transfer_days = 30  # Time to transfer patients
        
        return WhatIfScenario(
            scenario_id=f"SCN-{datetime.now().strftime('%H%M%S')}",
            scenario_type=ScenarioType.CLOSE_SITE,
            description=f"Close site {site_id} and transfer patients",
            parameters={
                'site_id': site_id,
                'patients_affected': site_patients,
                'issues_affected': site_issues
            },
            baseline_outcome={
                'total_patients': total_patients,
                'total_issues': total_issues,
                'site_count': current_metrics.get('site_count', 50)
            },
            scenario_outcome={
                'total_patients': total_patients - expected_dropouts,
                'total_issues': total_issues - issues_resolved_by_close,
                'site_count': current_metrics.get('site_count', 50) - 1,
                'issues_redistributed': issues_to_transfer
            },
            impact={
                'patients_lost': round(expected_dropouts, 0),
                'issues_resolved': round(issues_resolved_by_close, 0),
                'timeline_delay_days': close_process_days + transfer_days,
                'sites_reduced': 1
            },
            probability_of_success=0.75,
            risks=[
                f"Risk of losing {round(expected_dropouts, 0)} patients during transfer",
                "Receiving sites may be overwhelmed",
                "Regulatory implications of site closure",
                "Historical data may need additional review"
            ],
            recommendations=[
                "Evaluate if site issues can be resolved with additional support first",
                "Identify receiving sites with capacity",
                "Develop patient transfer communication plan"
            ],
            cost_estimate=25000  # Typical site closure cost
        )
    
    def simulate_process_change(
        self,
        current_metrics: Dict[str, Any],
        process_name: str,
        expected_improvement: float
    ) -> WhatIfScenario:
        """Simulate impact of a process change."""
        baseline_issues = current_metrics.get('total_issues', 1000)
        baseline_resolution_rate = current_metrics.get('resolution_rate', 5)
        
        new_resolution_rate = baseline_resolution_rate * (1 + expected_improvement)
        
        baseline_days = baseline_issues / baseline_resolution_rate if baseline_resolution_rate > 0 else 999
        scenario_days = baseline_issues / new_resolution_rate if new_resolution_rate > 0 else 999
        
        return WhatIfScenario(
            scenario_id=f"SCN-{datetime.now().strftime('%H%M%S')}",
            scenario_type=ScenarioType.PROCESS_CHANGE,
            description=f"Implement {process_name} process improvement",
            parameters={
                'process_name': process_name,
                'expected_improvement': expected_improvement
            },
            baseline_outcome={
                'days_to_resolution': round(baseline_days, 1),
                'resolution_rate': baseline_resolution_rate
            },
            scenario_outcome={
                'days_to_resolution': round(scenario_days, 1),
                'resolution_rate': round(new_resolution_rate, 2)
            },
            impact={
                'days_saved': round(baseline_days - scenario_days, 1),
                'percent_improvement': round(expected_improvement * 100, 1)
            },
            probability_of_success=0.7,
            risks=[
                "Change management challenges",
                "Staff adaptation time",
                "Potential short-term productivity dip"
            ],
            recommendations=[
                "Pilot process change at 2-3 sites first",
                "Develop training materials",
                "Establish metrics to track improvement"
            ],
            cost_estimate=10000
        )


class EnhancedForecasterAgent:
    """
    Enhanced FORECASTER Agent for timeline predictions and simulations.
    
    Capabilities:
    - DB Lock readiness forecasting
    - Clean patient milestone prediction
    - Issue resolution timeline
    - What-if scenario simulation
    - Monte Carlo uncertainty quantification
    """
    
    def __init__(self, llm_wrapper=None):
        """Initialize the forecaster agent."""
        self.data_loader = ForecastDataLoader()
        self.mc_simulator = MonteCarloSimulator(n_simulations=5000)
        self.whatif_simulator = WhatIfSimulator(self.mc_simulator)
        self.llm = llm_wrapper
        self._forecast_counter = 0
        self._result_counter = 0
        
        logger.info("EnhancedForecasterAgent initialized")
    
    def _generate_id(self, prefix: str) -> str:
        """Generate a unique ID."""
        if prefix == "FCT":
            self._forecast_counter += 1
            return f"FCT-{self._forecast_counter:04d}"
        elif prefix == "RES":
            self._result_counter += 1
            return f"RES-{self._result_counter:04d}"
        return f"{prefix}-{datetime.now().strftime('%H%M%S')}"
    
    def forecast_db_lock(
        self,
        entity_id: str = "portfolio",
        target_date: Optional[datetime] = None
    ) -> ForecastResult:
        """Forecast DB Lock readiness timeline."""
        start_time = datetime.now()
        
        result = ForecastResult(
            result_id=self._generate_id("RES"),
            query=f"DB Lock forecast for {entity_id}"
        )
        
        # Get current metrics
        if entity_id == "portfolio":
            metrics = self.data_loader.get_portfolio_metrics()
        elif entity_id.startswith("Study_"):
            metrics = self.data_loader.get_study_metrics(entity_id)
        else:
            metrics = self.data_loader.get_site_metrics(entity_id)
        
        result.data_sources.append("patient_issues")
        
        if not metrics:
            result.summary = f"No data available for {entity_id}"
            return result
        
        # Calculate current state
        total_patients = metrics.get('total_patients', metrics.get('patient_count', 0))
        patients_with_issues = metrics.get('patients_with_issues', 0)
        total_issues = metrics.get('total_issues', 0)
        
        if total_patients == 0:
            result.summary = "No patients found for forecasting"
            return result
        
        current_clean_rate = 1 - (patients_with_issues / total_patients)
        
        # Estimate resolution rates (issues per day)
        # Use historical average or default
        avg_resolution_rate = max(1, total_issues * 0.02)  # 2% daily resolution
        resolution_std = avg_resolution_rate * 0.3
        
        # Monte Carlo simulation
        resolution_forecast = self.mc_simulator.simulate_resolution_time(
            current_issues=int(total_issues),
            resolution_rate_mean=avg_resolution_rate,
            resolution_rate_std=resolution_std,
            new_issue_rate_mean=avg_resolution_rate * 0.1,  # 10% new issues
            new_issue_rate_std=avg_resolution_rate * 0.05
        )
        
        # Create forecast
        predicted_date = datetime.now() + timedelta(days=resolution_forecast.point_estimate)
        
        forecast = Forecast(
            forecast_id=self._generate_id("FCT"),
            forecast_type=ForecastType.DB_LOCK,
            entity_id=entity_id,
            entity_type="portfolio" if entity_id == "portfolio" else "study" if entity_id.startswith("Study_") else "site",
            metric="days_to_db_lock_ready",
            current_value=total_issues,
            predicted_value=resolution_forecast.point_estimate,
            uncertainty=resolution_forecast,
            timeframe_days=int(resolution_forecast.point_estimate),
            prediction_date=predicted_date,
            assumptions=[
                f"Current resolution rate: ~{avg_resolution_rate:.1f} issues/day",
                f"New issue rate: ~{avg_resolution_rate * 0.1:.1f} issues/day",
                "No major protocol amendments or regulatory holds",
                "Current staffing levels maintained"
            ],
            risks=[
                "Staff turnover could slow resolution",
                "Complex issues may take longer than average",
                "External factors (audits, inspections) could delay progress"
            ],
            methodology="Monte Carlo simulation with 5,000 iterations",
            data_points_used=total_patients,
            model_confidence=0.75
        )
        result.forecasts.append(forecast)
        
        # Create milestones
        if target_date:
            days_to_target = (target_date - datetime.now()).days
            prob_on_time = float(np.mean(
                np.random.normal(
                    resolution_forecast.point_estimate,
                    (resolution_forecast.upper_bound_95 - resolution_forecast.lower_bound_95) / 4,
                    1000
                ) <= days_to_target
            ))
            
            status = "on_track" if prob_on_time > 0.7 else "at_risk" if prob_on_time > 0.4 else "delayed"
            
            milestone = TimelineMilestone(
                milestone_id="MS-DBLOCK",
                name="Database Lock Ready",
                target_date=target_date,
                predicted_date=predicted_date,
                uncertainty=resolution_forecast,
                probability_on_time=prob_on_time,
                days_variance=int(resolution_forecast.point_estimate - days_to_target),
                status=status,
                blockers=self._identify_blockers(metrics),
                accelerators=self._identify_accelerators(metrics)
            )
            result.milestones.append(milestone)
        
        # Generate summary
        result.summary = self._generate_db_lock_summary(entity_id, metrics, forecast, result.milestones)
        result.recommendations = self._generate_db_lock_recommendations(metrics, forecast)
        result.methodology_notes = (
            "Forecast generated using Monte Carlo simulation with 5,000 iterations. "
            f"Based on {total_patients} patients with {total_issues} open issues. "
            f"Assumed daily resolution rate of {avg_resolution_rate:.1f} issues (±{resolution_std:.1f})."
        )
        
        result.duration_seconds = (datetime.now() - start_time).total_seconds()
        
        return result
    
    def forecast_clean_patient(
        self,
        entity_id: str = "portfolio",
        target_rate: float = 0.95
    ) -> ForecastResult:
        """Forecast time to reach target clean patient rate."""
        start_time = datetime.now()
        
        result = ForecastResult(
            result_id=self._generate_id("RES"),
            query=f"Clean patient forecast for {entity_id}, target: {target_rate:.0%}"
        )
        
        # Get metrics
        if entity_id == "portfolio":
            metrics = self.data_loader.get_portfolio_metrics()
        elif entity_id.startswith("Study_"):
            metrics = self.data_loader.get_study_metrics(entity_id)
        else:
            metrics = self.data_loader.get_site_metrics(entity_id)
        
        if not metrics:
            result.summary = f"No data available for {entity_id}"
            return result
        
        total_patients = metrics.get('total_patients', metrics.get('patient_count', 0))
        patients_with_issues = metrics.get('patients_with_issues', 0)
        
        current_clean_rate = 1 - (patients_with_issues / total_patients) if total_patients > 0 else 0
        
        # Estimate daily improvement rate
        daily_improvement = 0.005  # 0.5% daily improvement
        daily_std = 0.002
        
        uncertainty, prob_success = self.mc_simulator.simulate_clean_patient_trajectory(
            current_clean_rate=current_clean_rate,
            target_clean_rate=target_rate,
            daily_improvement_mean=daily_improvement,
            daily_improvement_std=daily_std
        )
        
        predicted_date = datetime.now() + timedelta(days=uncertainty.point_estimate)
        
        forecast = Forecast(
            forecast_id=self._generate_id("FCT"),
            forecast_type=ForecastType.CLEAN_PATIENT,
            entity_id=entity_id,
            entity_type="portfolio" if entity_id == "portfolio" else "study",
            metric="days_to_target_clean_rate",
            current_value=current_clean_rate,
            predicted_value=uncertainty.point_estimate,
            uncertainty=uncertainty,
            timeframe_days=int(uncertainty.point_estimate),
            prediction_date=predicted_date,
            assumptions=[
                f"Current clean rate: {current_clean_rate:.1%}",
                f"Target clean rate: {target_rate:.0%}",
                f"Daily improvement: ~{daily_improvement:.1%}",
                "No major setbacks or new issue sources"
            ],
            risks=[
                "New data entries may introduce issues",
                "Complex issues may resist resolution",
                "Staff availability fluctuations"
            ],
            methodology="Monte Carlo trajectory simulation",
            data_points_used=total_patients,
            model_confidence=0.70
        )
        result.forecasts.append(forecast)
        
        result.summary = (
            f"Clean Patient Forecast: {entity_id}\n"
            f"{'=' * 40}\n"
            f"Current Clean Rate: {current_clean_rate:.1%}\n"
            f"Target Clean Rate: {target_rate:.0%}\n"
            f"Days to Target: {uncertainty.point_estimate:.0f} days "
            f"(95% CI: {uncertainty.lower_bound_95:.0f} - {uncertainty.upper_bound_95:.0f})\n"
            f"Expected Date: {predicted_date.strftime('%Y-%m-%d')}\n"
            f"Probability of Success: {prob_success:.1%}"
        )
        
        result.recommendations = [
            "Focus on quick-win patients (1-2 issues to clean)",
            "Prioritize high-impact issue types (SDV, signatures)",
            "Weekly progress tracking against forecast"
        ]
        
        result.duration_seconds = (datetime.now() - start_time).total_seconds()
        
        return result
    
    def forecast_issue_resolution(
        self,
        issue_type: str,
        entity_id: str = "portfolio"
    ) -> ForecastResult:
        """Forecast resolution timeline for a specific issue type."""
        start_time = datetime.now()
        
        result = ForecastResult(
            result_id=self._generate_id("RES"),
            query=f"{issue_type} resolution forecast for {entity_id}"
        )
        
        if self.data_loader.patient_issues is None:
            result.summary = "Patient issues data not available"
            return result
        
        df = self.data_loader.patient_issues
        issue_col = f'issue_{issue_type}'
        
        if issue_col not in df.columns:
            result.summary = f"Issue type '{issue_type}' not found"
            return result
        
        # Filter by entity
        if entity_id != "portfolio":
            if entity_id.startswith("Study_"):
                df = df[df['study_id'] == entity_id]
            elif entity_id.startswith("Site_"):
                df = df[df['site_id'] == entity_id]
        
        issue_count = int(df[issue_col].sum())
        
        if issue_count == 0:
            result.summary = f"No {issue_type} issues found for {entity_id}"
            return result
        
        # Issue-specific resolution rates (issues per day)
        resolution_rates = {
            'sdv_incomplete': 10,
            'open_queries': 8,
            'signature_gaps': 15,
            'broken_signatures': 12,
            'sae_dm_pending': 5,
            'sae_safety_pending': 3,
            'missing_visits': 2,
            'missing_pages': 5,
            'meddra_uncoded': 20,
            'whodrug_uncoded': 20,
            'lab_issues': 4,
            'edrr_issues': 3,
            'inactivated_forms': 8,
            'high_query_volume': 6
        }
        
        base_rate = resolution_rates.get(issue_type, 5)
        
        # ML-BASED PREDICTION INTEGRATION
        try:
            from src.ml.model_loader import get_model_loader
            loader = get_model_loader()
            
            # Prepare aggregate features for the entity
            features = pd.DataFrame([{
                'priority_score': 2.0 if issue_type in ['sae_dm_pending', 'sae_safety_pending'] else 1.0,
                'site_performance_score': 85.0,
                'workload_index': min(1.0, issue_count / 100.0),
                'issue_type': issue_type
            }])
            
            ml_preds = loader.predict_resolution_time(features)
            if ml_preds:
                p = ml_preds[0]
                # Scale prediction for the volume (ML predicts per-issue base, we extrapolate)
                volume_factor = max(1.0, np.log2(max(1, issue_count / 5 + 1)))
                
                point_estimate = p['prediction_days'] * volume_factor
                lower_bound = p['lower_bound_days'] * volume_factor
                upper_bound = p['upper_bound_days'] * volume_factor
                
                uncertainty = UncertaintyBand(
                    point_estimate=point_estimate,
                    lower_bound_95=lower_bound * 0.8,
                    upper_bound_95=upper_bound * 1.2,
                    lower_bound_80=lower_bound * 0.9,
                    upper_bound_80=upper_bound * 1.1,
                    lower_bound_50=lower_bound,
                    upper_bound_50=upper_bound,
                    standard_error=(upper_bound - lower_bound) / 4,
                    confidence_level=ConfidenceLevel.HIGH
                )
                methodology = f"ML-Quantile Regression (XGBoost) | Base: {p['prediction_days']:.1f}d"
            else:
                raise ValueError("ML prediction returned empty")
                
        except Exception as e:
            logger.warning(f"ML forecast failed, falling back to Monte Carlo: {e}")
            uncertainty = self.mc_simulator.simulate_resolution_time(
                current_issues=issue_count,
                resolution_rate_mean=base_rate,
                resolution_rate_std=base_rate * 0.25
            )
            methodology = "Monte Carlo simulation"
        
        predicted_date = datetime.now() + timedelta(days=uncertainty.point_estimate)
        
        forecast = Forecast(
            forecast_id=self._generate_id("FCT"),
            forecast_type=ForecastType.ISSUE_RESOLUTION,
            entity_id=entity_id,
            entity_type="portfolio" if entity_id == "portfolio" else "study",
            metric=f"days_to_resolve_{issue_type}",
            current_value=issue_count,
            predicted_value=uncertainty.point_estimate,
            uncertainty=uncertainty,
            timeframe_days=int(uncertainty.point_estimate),
            prediction_date=predicted_date,
            assumptions=[
                f"Current {issue_type} count: {issue_count}",
                f"Expected resolution rate: ~{base_rate} issues/day"
            ],
            risks=[
                "Complex cases may take longer",
                "Resource availability fluctuations"
            ],
            methodology=methodology,
            data_points_used=issue_count,
            model_confidence=0.75
        )
        result.forecasts.append(forecast)
        
        result.summary = (
            f"Issue Resolution Forecast: {issue_type}\n"
            f"{'=' * 40}\n"
            f"Entity: {entity_id}\n"
            f"Current Count: {issue_count} issues\n"
            f"Days to Resolution: {uncertainty.point_estimate:.0f} days "
            f"(95% CI: {uncertainty.lower_bound_95:.0f} - {uncertainty.upper_bound_95:.0f})\n"
            f"Expected Completion: {predicted_date.strftime('%Y-%m-%d')}"
        )
        
        result.duration_seconds = (datetime.now() - start_time).total_seconds()
        
        return result
    
    def run_what_if(
        self,
        scenario_type: str,
        parameters: Dict[str, Any]
    ) -> ForecastResult:
        """Run a what-if scenario simulation."""
        start_time = datetime.now()
        
        result = ForecastResult(
            result_id=self._generate_id("RES"),
            query=f"What-if: {scenario_type}"
        )
        
        metrics = self.data_loader.get_portfolio_metrics()
        
        if scenario_type == "add_resource":
            scenario = self.whatif_simulator.simulate_add_resource(
                current_metrics=metrics,
                resource_type=parameters.get('resource_type', 'cra'),
                resource_count=parameters.get('count', 1)
            )
        elif scenario_type == "close_site":
            site_id = parameters.get('site_id', 'Site_1')
            site_metrics = self.data_loader.get_site_metrics(site_id)
            scenario = self.whatif_simulator.simulate_close_site(
                current_metrics=metrics,
                site_id=site_id,
                site_metrics=site_metrics
            )
        elif scenario_type == "process_change":
            scenario = self.whatif_simulator.simulate_process_change(
                current_metrics=metrics,
                process_name=parameters.get('process_name', 'workflow_optimization'),
                expected_improvement=parameters.get('improvement', 0.15)
            )
        else:
            result.summary = f"Unknown scenario type: {scenario_type}"
            return result
        
        result.scenarios.append(scenario)
        
        result.summary = (
            f"What-If Scenario: {scenario.description}\n"
            f"{'=' * 50}\n"
            f"Baseline: {scenario.baseline_outcome.get('days_to_resolution', 'N/A')} days\n"
            f"Scenario: {scenario.scenario_outcome.get('days_to_resolution', 'N/A')} days\n"
            f"Impact: {scenario.impact.get('days_saved', 0)} days saved "
            f"({scenario.impact.get('percent_improvement', 0):.1f}% improvement)\n"
            f"Probability of Success: {scenario.probability_of_success:.0%}\n"
            f"Estimated Cost: ${scenario.cost_estimate:,.0f}"
        )
        
        result.recommendations = scenario.recommendations
        result.duration_seconds = (datetime.now() - start_time).total_seconds()
        
        return result
    
    def forecast_from_query(self, query: str) -> ForecastResult:
        """Generate forecast from natural language query."""
        start_time = datetime.now()
        query_lower = query.lower()
        
        # Extract entity
        entity_id = self._extract_entity(query)
        
        # Determine forecast type
        if any(word in query_lower for word in ['db lock', 'database lock', 'lock ready', 'dblock']):
            return self.forecast_db_lock(entity_id)
        
        elif any(word in query_lower for word in ['clean', 'tier 1', 'tier 2', 'clinical clean']):
            target = 0.95 if '95' in query else 0.90 if '90' in query else 0.95
            return self.forecast_clean_patient(entity_id, target)
        
        elif 'what if' in query_lower or 'what-if' in query_lower:
            # Parse what-if parameters
            if 'add' in query_lower and any(r in query_lower for r in ['cra', 'dm', 'coordinator']):
                resource = 'cra' if 'cra' in query_lower else 'data_manager' if 'dm' in query_lower else 'site_coordinator'
                return self.run_what_if('add_resource', {'resource_type': resource, 'count': 1})
            elif 'close' in query_lower and 'site' in query_lower:
                site_id = entity_id if entity_id.startswith('Site_') else 'Site_1'
                return self.run_what_if('close_site', {'site_id': site_id})
            else:
                return self.run_what_if('process_change', {'process_name': 'optimization', 'improvement': 0.15})
        
        elif any(word in query_lower for word in ['sdv', 'query', 'signature', 'sae', 'coding']):
            # Issue-specific forecast
            issue_type = self._detect_issue_type(query_lower)
            if issue_type:
                return self.forecast_issue_resolution(issue_type, entity_id)
        
        # Default to DB Lock forecast
        return self.forecast_db_lock(entity_id)
    
    def _extract_entity(self, query: str) -> str:
        """Extract entity ID from query."""
        import re
        
        study_match = re.search(r'Study[_\s]?(\d+)', query, re.IGNORECASE)
        if study_match:
            return f"Study_{study_match.group(1)}"
        
        site_match = re.search(r'Site[_\s]?(\d+)', query, re.IGNORECASE)
        if site_match:
            return f"Site_{site_match.group(1)}"
        
        return "portfolio"
    
    def _detect_issue_type(self, query: str) -> Optional[str]:
        """Detect issue type from query."""
        issue_keywords = {
            'sdv': 'sdv_incomplete',
            'source data': 'sdv_incomplete',
            'query': 'open_queries',
            'queries': 'open_queries',
            'signature': 'signature_gaps',
            'sae': 'sae_dm_pending',
            'coding': 'meddra_uncoded',
            'meddra': 'meddra_uncoded'
        }
        
        for keyword, issue_type in issue_keywords.items():
            if keyword in query:
                return issue_type
        
        return None
    
    def _identify_blockers(self, metrics: Dict) -> List[str]:
        """Identify blockers from metrics."""
        blockers = []
        
        issue_breakdown = metrics.get('issue_breakdown', {})
        sorted_issues = sorted(issue_breakdown.items(), key=lambda x: x[1], reverse=True)
        
        for issue, count in sorted_issues[:3]:
            if count > 0:
                blockers.append(f"{issue}: {count} issues")
        
        return blockers
    
    def _identify_accelerators(self, metrics: Dict) -> List[str]:
        """Identify accelerators from metrics."""
        accelerators = []
        
        total_issues = metrics.get('total_issues', 0)
        patients_with_issues = metrics.get('patients_with_issues', 0)
        
        if total_issues > 0 and patients_with_issues > 0:
            avg_issues = total_issues / patients_with_issues
            if avg_issues < 2:
                accelerators.append("Low avg issues per patient - quick wins available")
        
        issue_breakdown = metrics.get('issue_breakdown', {})
        if issue_breakdown.get('sdv_incomplete', 0) > 100:
            accelerators.append("SDV backlog can be addressed with focused monitoring visits")
        
        return accelerators
    
    def _generate_db_lock_summary(
        self,
        entity_id: str,
        metrics: Dict,
        forecast: Forecast,
        milestones: List[TimelineMilestone]
    ) -> str:
        """Generate DB Lock forecast summary."""
        summary = f"""
DB LOCK READINESS FORECAST
{'=' * 50}
Entity: {entity_id}
Total Patients: {metrics.get('total_patients', metrics.get('patient_count', 0)):,}
Total Issues: {metrics.get('total_issues', 0):,}
Patients with Issues: {metrics.get('patients_with_issues', 0):,}

FORECAST:
Days to Resolution: {forecast.uncertainty.point_estimate:.0f} days
95% Confidence Interval: {forecast.uncertainty.lower_bound_95:.0f} - {forecast.uncertainty.upper_bound_95:.0f} days
Expected Date: {forecast.prediction_date.strftime('%Y-%m-%d')}
Confidence Level: {forecast.uncertainty.confidence_level.value.upper()}
"""
        
        if milestones:
            ms = milestones[0]
            summary += f"""
MILESTONE STATUS:
Target Date: {ms.target_date.strftime('%Y-%m-%d')}
Probability On-Time: {ms.probability_on_time:.1%}
Status: {ms.status.upper()}
Variance: {'+' if ms.days_variance > 0 else ''}{ms.days_variance} days
"""
        
        return summary.strip()
    
    def _generate_db_lock_recommendations(self, metrics: Dict, forecast: Forecast) -> List[str]:
        """Generate recommendations for DB Lock."""
        recommendations = []
        
        issue_breakdown = metrics.get('issue_breakdown', {})
        
        # Prioritize by issue count
        sorted_issues = sorted(issue_breakdown.items(), key=lambda x: x[1], reverse=True)
        
        if sorted_issues:
            top_issue = sorted_issues[0]
            recommendations.append(
                f"Priority 1: Address {top_issue[0]} ({top_issue[1]:,} issues) - highest volume"
            )
        
        if len(sorted_issues) > 1:
            second_issue = sorted_issues[1]
            recommendations.append(
                f"Priority 2: Focus on {second_issue[0]} ({second_issue[1]:,} issues)"
            )
        
        # Add resource recommendation if timeline is long
        if forecast.uncertainty.point_estimate > 60:
            recommendations.append(
                "Consider adding resources to accelerate resolution (run what-if scenario)"
            )
        
        recommendations.append("Track daily resolution rate to validate forecast assumptions")
        
        return recommendations
    
    def process(self, query: str, context: Dict = None) -> Dict:
        """Main processing method for orchestrator integration."""
        result = self.forecast_from_query(query)
        
        return {
            "result_id": result.result_id,
            "query": result.query,
            "forecasts": [f.to_dict() for f in result.forecasts],
            "milestones": [m.to_dict() for m in result.milestones],
            "scenarios": [s.to_dict() for s in result.scenarios],
            "summary": result.summary,
            "recommendations": result.recommendations,
            "methodology_notes": result.methodology_notes,
            "data_sources": result.data_sources,
            "duration_seconds": result.duration_seconds
        }


def get_forecaster_agent(llm_wrapper=None) -> EnhancedForecasterAgent:
    """Factory function to get forecaster agent instance."""
    return EnhancedForecasterAgent(llm_wrapper=llm_wrapper)
