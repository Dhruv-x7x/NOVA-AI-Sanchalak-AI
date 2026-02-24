"""
a6on-i - Monte Carlo Simulation Engine
=====================================================
Implements 10,000+ run Monte Carlo simulations with statistical distributions
for timeline prediction, risk quantification, and scenario analysis.

As specified in SOLUTION.md L357-360:
- Monte Carlo Simulation: 10,000 runs with uncertainty
- Probability distributions for timelines
- Risk quantification
"""

import numpy as np
from scipy import stats
from sqlalchemy import text
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class Distribution:
    """Statistical distribution result from Monte Carlo simulation."""
    percentiles: Dict[int, float]  # P10, P25, P50, P75, P90
    mean: float
    std: float
    min_value: float
    max_value: float
    confidence_interval: Tuple[float, float]  # 95% CI
    n_simulations: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "percentiles": self.percentiles,
            "mean": round(self.mean, 2),
            "std": round(self.std, 2),
            "min": round(self.min_value, 2),
            "max": round(self.max_value, 2),
            "confidence_interval": {
                "lower": round(self.confidence_interval[0], 2),
                "upper": round(self.confidence_interval[1], 2)
            },
            "n_simulations": self.n_simulations
        }


@dataclass
class ImpactAnalysis:
    """Impact analysis result from Monte Carlo simulation."""
    scenario_name: str
    success_probability: float
    timeline_distribution: Distribution
    cost_distribution: Optional[Distribution] = None
    risk_score: float = 0.0
    affected_entities: int = 0
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario_name,
            "success_probability": round(self.success_probability * 100, 1),
            "timeline": self.timeline_distribution.to_dict(),
            "cost": self.cost_distribution.to_dict() if self.cost_distribution else None,
            "risk_score": round(self.risk_score, 2),
            "affected_entities": self.affected_entities,
            "recommendations": self.recommendations
        }


@dataclass
class TrialState:
    """Current state of a clinical trial for simulation."""
    study_id: str
    total_patients: int = 0
    tier1_clean_patients: int = 0
    tier2_clean_patients: int = 0
    db_lock_ready: int = 0
    open_queries: int = 0
    pending_signatures: int = 0
    pending_sdv: int = 0
    active_sites: int = 0
    avg_dqi: float = 0.0
    
    # Daily rates - will be loaded from database in _load_real_rates()
    query_resolution_rate: float = 0.0  # Will be populated from DB
    signature_completion_rate: float = 0.0  # Will be populated from DB
    sdv_completion_rate: float = 0.0  # Will be populated from DB
    
    # Variability (standard deviations)
    query_rate_std: float = 0.0
    signature_rate_std: float = 0.0
    sdv_rate_std: float = 0.0
    
    _rates_loaded: bool = False
    
    def _load_real_rates(self):
        """Load real daily completion rates derived from queries, visits, and patients tables.
        
        Rates represent portfolio-wide daily throughput (all sites combined).
        Derivation:
          - Query rate: from queries table (resolved / avg_age) scaled by active sites
          - SDV rate: from visits table (completed SDVs / date span) scaled by active sites
          - Signature rate: from visits data_entry_complete over same span
        """
        if self._rates_loaded:
            return

        try:
            from src.database.connection import get_db_manager
            db = get_db_manager()

            with db.engine.connect() as conn:
                # ── Count active sites for parallelism scaling ─────────
                site_r = conn.execute(text("""
                    SELECT COUNT(DISTINCT site_id) FROM clinical_sites
                """))
                active_sites = max(1, int(site_r.scalar() or 1))

                # ── Query resolution rate ──────────────────────────────
                # Queries answered per day: (answered count / avg_age_days) × site_parallelism
                qr = conn.execute(text("""
                    SELECT
                        COUNT(*) AS resolved,
                        COALESCE(AVG(q.age_days), 1) AS avg_age,
                        COALESCE(STDDEV(q.age_days), 3) AS std_age
                    FROM queries q
                    WHERE q.status IN ('answered', 'closed')
                """))
                qrow = qr.fetchone()
                resolved_count = float(qrow[0]) if qrow and qrow[0] else 0
                avg_age = max(1.0, float(qrow[1]) if qrow and qrow[1] else 1.0)
                std_age = float(qrow[2]) if qrow and qrow[2] else 3.0

                if resolved_count > 0:
                    # Per-site daily resolution rate, then scale by sites
                    per_site_qrate = resolved_count / avg_age
                    # Scale up: each site resolves queries independently
                    self.query_resolution_rate = round(per_site_qrate * (active_sites / max(1, resolved_count / max(1, per_site_qrate))), 1)
                    self.query_resolution_rate = max(5.0, self.query_resolution_rate)
                    self.query_rate_std = round(std_age / avg_age * self.query_resolution_rate, 1)
                else:
                    self.query_resolution_rate = 12.0
                    self.query_rate_std = 3.0

                # ── SDV completion rate ────────────────────────────────
                # SDVs completed over date span, scaled proportionally
                vr = conn.execute(text("""
                    SELECT
                        COUNT(*) AS total_visits,
                        SUM(CASE WHEN v.sdv_complete THEN 1 ELSE 0 END) AS sdv_done,
                        SUM(CASE WHEN v.data_entry_complete THEN 1 ELSE 0 END) AS de_done,
                        EXTRACT(DAY FROM MAX(v.actual_date) - MIN(v.actual_date)) AS date_span
                    FROM visits v
                    WHERE v.actual_date IS NOT NULL
                """))
                vrow = vr.fetchone()
                total_visits = float(vrow[0]) if vrow and vrow[0] else 0
                sdv_done = float(vrow[1]) if vrow and vrow[1] else 0
                de_done = float(vrow[2]) if vrow and vrow[2] else 0
                date_span = max(1.0, float(vrow[3]) if vrow and vrow[3] else 1.0)

                if sdv_done > 0:
                    self.sdv_completion_rate = max(3.0, round(sdv_done / date_span * active_sites, 1))
                    self.sdv_rate_std = round(self.sdv_completion_rate * 0.25, 1)
                else:
                    self.sdv_completion_rate = 15.0
                    self.sdv_rate_std = 4.0

                # ── Signature / data-entry completion rate ─────────────
                if de_done > 0:
                    self.signature_completion_rate = max(2.0, round(de_done / date_span * active_sites, 1))
                    self.signature_rate_std = round(self.signature_completion_rate * 0.3, 1)
                else:
                    self.signature_completion_rate = 8.0
                    self.signature_rate_std = 2.5

                logger.info(
                    f"Loaded real rates from DB ({active_sites} sites): "
                    f"query={self.query_resolution_rate}/day, "
                    f"sig={self.signature_completion_rate}/day, "
                    f"sdv={self.sdv_completion_rate}/day"
                )

        except Exception as e:
            logger.warning(f"Could not derive rates from DB: {e}. Using industry defaults.")
            self.query_resolution_rate = 12.0
            self.query_rate_std = 3.0
            self.signature_completion_rate = 8.0
            self.signature_rate_std = 2.5
            self.sdv_completion_rate = 15.0
            self.sdv_rate_std = 4.0

        self._rates_loaded = True
    
    @classmethod
    def from_data(cls, data: Dict[str, Any]) -> "TrialState":
        """Create TrialState from data dictionary."""
        state = cls(
            study_id=data.get("study_id", "unknown"),
            total_patients=data.get("total_patients", 0),
            tier1_clean_patients=data.get("tier1_clean", 0),
            tier2_clean_patients=data.get("tier2_clean", 0),
            db_lock_ready=data.get("db_lock_ready", 0),
            open_queries=data.get("open_queries", 0),
            pending_signatures=data.get("pending_signatures", 0),
            pending_sdv=data.get("pending_sdv", 0),
            active_sites=data.get("active_sites", 0),
            avg_dqi=data.get("avg_dqi", 75.0)
        )
        # Load real rates from database
        state._load_real_rates()
        return state


class MonteCarloEngine:
    """
    Monte Carlo Simulation Engine for clinical trial timeline prediction.
    
    Runs 10,000+ simulations with:
    - Query resolution rate (normal distribution)
    - Signature completion (beta distribution)  
    - Issue resolution (exponential distribution)
    
    Returns percentile timelines (P10, P25, P50, P75, P90)
    """
    
    def __init__(self, n_simulations: int = 10000):
        self.n_simulations = n_simulations
        self._rng = np.random.default_rng(seed=42)
        logger.info(f"MonteCarloEngine initialized with {n_simulations} simulations")
    
    def simulate_dblock_timeline(
        self,
        current_state: TrialState,
        target_date: Optional[datetime] = None
    ) -> Distribution:
        """
        Run Monte Carlo simulation for DB Lock timeline.
        
        Simulates:
        - Query/SDV/Signature rates: Log-normal distribution (CV capped at 40%)
        - Issue resolution delays: Exponential(mean=2 days)
        - Weekend/holiday adjustment: ×1.4
        
        Returns:
            Distribution with P10, P25, P50, P75, P90 timeline estimates (days)
        """
        logger.info(f"Running {self.n_simulations} Monte Carlo simulations for DB Lock timeline")
        
        # Current work remaining
        remaining_queries = max(current_state.open_queries, 0)
        remaining_signatures = max(current_state.pending_signatures, 0)
        remaining_sdv = max(current_state.pending_sdv, 0)
        
        # If all work is done, return zero timeline
        if remaining_queries == 0 and remaining_signatures == 0 and remaining_sdv == 0:
            return Distribution(
                percentiles={10: 0, 25: 0, 50: 0, 75: 0, 90: 0},
                mean=0,
                std=0,
                min_value=0,
                max_value=0,
                confidence_interval=(0, 0),
                n_simulations=self.n_simulations
            )
        
        # Store simulation results (days to completion)
        completion_days = []
        
        for _ in range(self.n_simulations):
            # Sample daily rates from distributions
            # Use log-normal for rates to avoid nonsensical near-zero draws
            # Convert (mean, std) → log-normal params: mu = ln(mean), sigma capped at CV ≤ 40%
            def _sample_rate(mean_rate, std_rate):
                """Sample a positive daily rate from log-normal distribution."""
                if mean_rate <= 0:
                    return 1.0
                cv = min(0.4, std_rate / mean_rate)  # cap coefficient of variation at 40%
                sigma = np.sqrt(np.log(1 + cv ** 2))
                mu = np.log(mean_rate) - 0.5 * sigma ** 2
                return max(1.0, float(self._rng.lognormal(mu, sigma)))

            # Query resolution rate
            query_rate = _sample_rate(
                current_state.query_resolution_rate,
                current_state.query_rate_std
            )
            
            # Signature completion rate
            sig_rate = _sample_rate(
                current_state.signature_completion_rate,
                current_state.signature_rate_std
            )
            
            # SDV completion rate
            sdv_rate = _sample_rate(
                current_state.sdv_completion_rate,
                current_state.sdv_rate_std
            )
            
            # Calculate days for each work stream
            days_queries = remaining_queries / query_rate if remaining_queries > 0 else 0
            days_signatures = remaining_signatures / sig_rate if remaining_signatures > 0 else 0
            days_sdv = remaining_sdv / sdv_rate if remaining_sdv > 0 else 0
            
            # Add randomness for unexpected issues (exponential)
            issue_delay = self._rng.exponential(scale=2.0)  # Mean 2 days delay
            
            # Total days is max of parallel work streams plus issue delay
            total_days = max(days_queries, days_signatures, days_sdv) + issue_delay
            
            # Add weekend/holiday adjustment (~30% non-working days)
            total_days *= 1.4
            
            completion_days.append(total_days)
        
        # Convert to numpy array for statistics
        completion_days = np.array(completion_days)
        
        # Calculate percentiles
        percentiles = {
            10: np.percentile(completion_days, 10),
            25: np.percentile(completion_days, 25),
            50: np.percentile(completion_days, 50),
            75: np.percentile(completion_days, 75),
            90: np.percentile(completion_days, 90)
        }
        
        # Round percentiles
        percentiles = {k: round(v, 1) for k, v in percentiles.items()}
        
        # 95% confidence interval
        ci_lower = np.percentile(completion_days, 2.5)
        ci_upper = np.percentile(completion_days, 97.5)
        
        return Distribution(
            percentiles=percentiles,
            mean=float(np.mean(completion_days)),
            std=float(np.std(completion_days)),
            min_value=float(np.min(completion_days)),
            max_value=float(np.max(completion_days)),
            confidence_interval=(float(ci_lower), float(ci_upper)),
            n_simulations=self.n_simulations
        )
    
    def simulate_site_closure_impact(
        self,
        site_id: str,
        site_patient_count: int = 0,
        avg_transfer_success_rate: float = 0.85,
        site_performance_score: float = 50.0
    ) -> ImpactAnalysis:
        """
        Simulate downstream effects of closing a site.
        
        Simulates:
        - Subject transfer success rates
        - Timeline delay distribution
        - Cost impact ranges
        
        Args:
            site_id: Site to close
            site_patient_count: Number of patients at site
            avg_transfer_success_rate: Historical transfer success rate
            site_performance_score: Site's current performance (0-100)
            
        Returns:
            ImpactAnalysis with success probability and timeline impact
        """
        logger.info(f"Simulating site closure impact for {site_id}")
        
        if site_patient_count == 0:
            site_patient_count = self._rng.integers(5, 50)  # Estimate
        
        transfer_results = []
        timeline_delays = []
        cost_impacts = []
        
        for _ in range(self.n_simulations):
            # Transfer success: Beta distribution around expected rate
            alpha = avg_transfer_success_rate * 20
            beta = (1 - avg_transfer_success_rate) * 20
            transfer_rate = self._rng.beta(max(1, alpha), max(1, beta))
            
            successful_transfers = int(site_patient_count * transfer_rate)
            failed_transfers = site_patient_count - successful_transfers
            
            # Timeline delay per failed transfer
            delay_per_failure = self._rng.gamma(shape=3, scale=5)  # Mean ~15 days
            total_delay = failed_transfers * delay_per_failure / max(1, site_patient_count)
            
            # Cost impact (transfer + delay costs)
            cost_per_transfer = self._rng.normal(5000, 1000)  # $5000 +/- $1000
            cost_per_delay_day = self._rng.normal(10000, 2000)  # $10k/day +/- $2k
            total_cost = (site_patient_count * cost_per_transfer + 
                         total_delay * cost_per_delay_day)
            
            transfer_results.append(transfer_rate)
            timeline_delays.append(total_delay)
            cost_impacts.append(total_cost)
        
        transfer_results = np.array(transfer_results)
        timeline_delays = np.array(timeline_delays)
        cost_impacts = np.array(cost_impacts)
        
        # Timeline distribution
        timeline_dist = Distribution(
            percentiles={
                10: round(np.percentile(timeline_delays, 10), 1),
                25: round(np.percentile(timeline_delays, 25), 1),
                50: round(np.percentile(timeline_delays, 50), 1),
                75: round(np.percentile(timeline_delays, 75), 1),
                90: round(np.percentile(timeline_delays, 90), 1)
            },
            mean=float(np.mean(timeline_delays)),
            std=float(np.std(timeline_delays)),
            min_value=float(np.min(timeline_delays)),
            max_value=float(np.max(timeline_delays)),
            confidence_interval=(
                float(np.percentile(timeline_delays, 2.5)),
                float(np.percentile(timeline_delays, 97.5))
            ),
            n_simulations=self.n_simulations
        )
        
        # Cost distribution
        cost_dist = Distribution(
            percentiles={
                10: round(np.percentile(cost_impacts, 10), 0),
                25: round(np.percentile(cost_impacts, 25), 0),
                50: round(np.percentile(cost_impacts, 50), 0),
                75: round(np.percentile(cost_impacts, 75), 0),
                90: round(np.percentile(cost_impacts, 90), 0)
            },
            mean=float(np.mean(cost_impacts)),
            std=float(np.std(cost_impacts)),
            min_value=float(np.min(cost_impacts)),
            max_value=float(np.max(cost_impacts)),
            confidence_interval=(
                float(np.percentile(cost_impacts, 2.5)),
                float(np.percentile(cost_impacts, 97.5))
            ),
            n_simulations=self.n_simulations
        )
        
        # Calculate risk score (higher = more risk)
        avg_delay = float(np.mean(timeline_delays))
        risk_score = min(100, (avg_delay * 2) + ((1 - np.mean(transfer_results)) * 50))
        
        # Generate recommendations
        recommendations = []
        if np.mean(transfer_results) < 0.80:
            recommendations.append("Consider phased closure to improve transfer rates")
        if avg_delay > 20:
            recommendations.append("Significant timeline risk - ensure backup sites ready")
        if np.mean(cost_impacts) > 200000:
            recommendations.append("High cost impact - evaluate if closure is cost-effective")
        if site_performance_score < 40:
            recommendations.append("Low performance justifies closure despite impact")
        
        return ImpactAnalysis(
            scenario_name=f"Site Closure: {site_id}",
            success_probability=float(np.mean(transfer_results)),
            timeline_distribution=timeline_dist,
            cost_distribution=cost_dist,
            risk_score=risk_score,
            affected_entities=site_patient_count,
            recommendations=recommendations
        )
    
    def simulate_deadline_probability(
        self,
        current_state: TrialState,
        target_date: datetime
    ) -> Dict[str, Any]:
        """
        Calculate probability of meeting a deadline.
        
        Args:
            current_state: Current trial state
            target_date: Target deadline
            
        Returns:
            Dictionary with probability and analysis
        """
        # Get timeline distribution
        timeline_dist = self.simulate_dblock_timeline(current_state)
        
        # Days until target
        days_until_target = (target_date - datetime.now()).days
        
        # Simulate meeting deadline
        completion_days = []
        
        for _ in range(self.n_simulations):
            # Use similar logic to simulate_dblock_timeline
            remaining_work = (current_state.open_queries + 
                            current_state.pending_signatures + 
                            current_state.pending_sdv)
            
            if remaining_work == 0:
                completion_days.append(0)
                continue
            
            # Sample rates
            daily_rate = max(1, self._rng.normal(
                current_state.query_resolution_rate + 
                current_state.signature_completion_rate,
                current_state.query_rate_std + current_state.signature_rate_std
            ))
            
            days = remaining_work / daily_rate
            days *= 1.3  # Weekend adjustment
            days += self._rng.exponential(scale=1.5)  # Random delays
            
            completion_days.append(days)
        
        completion_days = np.array(completion_days)
        
        # Probability of meeting deadline
        prob_on_time = np.mean(completion_days <= days_until_target)
        
        # Determine confidence assessment
        if prob_on_time >= 0.90:
            assessment = "HIGH CONFIDENCE - Very likely to meet deadline"
        elif prob_on_time >= 0.70:
            assessment = "MODERATE CONFIDENCE - Good chance, but monitor closely"
        elif prob_on_time >= 0.50:
            assessment = "LOW CONFIDENCE - Risk of missing deadline"
        else:
            assessment = "CRITICAL RISK - Deadline likely to be missed"
        
        return {
            "target_date": target_date.isoformat(),
            "days_until_target": days_until_target,
            "probability_on_time": round(prob_on_time * 100, 1),
            "assessment": assessment,
            "timeline_distribution": timeline_dist.to_dict(),
            "recommended_actions": self._get_deadline_recommendations(prob_on_time)
        }
    
    def _get_deadline_recommendations(self, prob: float) -> List[str]:
        """Get recommendations based on deadline probability."""
        if prob >= 0.90:
            return ["Continue current pace", "Monitor for any emerging risks"]
        elif prob >= 0.70:
            return [
                "Increase daily resolution rate by 20%",
                "Prioritize high-cascade queries",
                "Consider temporary resource allocation"
            ]
        elif prob >= 0.50:
            return [
                "URGENT: Escalate to leadership",
                "Add 2+ CRAs to support sites",
                "Focus on tier-1 clean requirements only",
                "Re-negotiate deadline if possible"
            ]
        else:
            return [
                "CRITICAL: Immediate intervention required",
                "Request deadline extension",
                "Emergency resource deployment",
                "Scope reduction - focus on essential patients only",
                "Consider partial DB Lock approach"
            ]
    
    def run_batch_scenarios(
        self,
        current_state: TrialState,
        scenarios: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Run multiple what-if scenarios in batch.
        
        Args:
            current_state: Current trial state
            scenarios: List of scenario configurations
            
        Returns:
            Comparison of all scenario outcomes
        """
        results = []
        
        for scenario in scenarios:
            scenario_type = scenario.get("type", "timeline")
            
            if scenario_type == "timeline":
                result = self.simulate_dblock_timeline(current_state)
                results.append({
                    "scenario": scenario.get("name", "Timeline Projection"),
                    "type": scenario_type,
                    "result": result.to_dict()
                })
            
            elif scenario_type == "site_closure":
                result = self.simulate_site_closure_impact(
                    site_id=scenario.get("site_id", "unknown"),
                    site_patient_count=scenario.get("patient_count", 20)
                )
                results.append({
                    "scenario": scenario.get("name", f"Close {scenario.get('site_id')}"),
                    "type": scenario_type,
                    "result": result.to_dict()
                })
        
        return {
            "n_simulations_per_scenario": self.n_simulations,
            "scenarios_analyzed": len(results),
            "results": results
        }


# Singleton instance
_engine: Optional[MonteCarloEngine] = None


def get_monte_carlo_engine(n_simulations: int = 10000) -> MonteCarloEngine:
    """Get the Monte Carlo engine singleton."""
    global _engine
    if _engine is None or _engine.n_simulations != n_simulations:
        _engine = MonteCarloEngine(n_simulations)
    return _engine
