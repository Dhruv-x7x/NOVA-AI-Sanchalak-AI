"""
Simulation Routes (v4.0 - Real Data-Driven Monte Carlo)
========================================================
Monte Carlo simulation endpoints grounded in real trailpulse_test database data.
All simulations pull live metrics from PostgreSQL and use proper statistical modeling.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Body, Request
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from pydantic import BaseModel
import sys
import os
import logging
import traceback

# Setup Logging
logger = logging.getLogger(__name__)

try:
    import numpy as np
except ImportError:
    np = None
    logger.error("Numpy not found. Monte Carlo simulations will be disabled.")

# Setup Paths for resilient imports
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
# parents: [0]routes, [1]v1, [2]api, [3]app, [4]backend, [5]D:\mk-5\mk-5\mk-5
PROJECT_ROOT = os.path.abspath(os.path.join(FILE_DIR, "../../../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Also ensure backend is in path for 'app' imports
BACKEND_ROOT = os.path.join(PROJECT_ROOT, "backend")
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

try:
    from app.core.security import get_current_user
    from app.services.database import get_data_service
except ImportError:
    from backend.app.core.security import get_current_user
    from backend.app.services.database import get_data_service

router = APIRouter()

class SimulationRequest(BaseModel):
    scenario_type: str
    parameters: Dict[str, Any]
    iterations: int = 1000
    seed: Optional[int] = None


# =============================================================================
# HELPER: Fetch live baseline from database
# =============================================================================

def _get_live_baseline(study_id: Optional[str] = None) -> Dict[str, Any]:
    """Pull real-time baseline metrics from the trailpulse_test database."""
    try:
        ds = get_data_service()
        summary = ds.get_portfolio_summary(study_id=study_id)
        issue_stats = ds.get_issue_summary_stats(study_id=study_id)

        total_patients = summary.get('total_patients', 0) or 0
        total_sites = summary.get('total_sites', 0) or 0
        mean_dqi = summary.get('mean_dqi', 0) or 0
        dblock_ready_rate = summary.get('dblock_ready_rate', 0) or 0
        dblock_ready_count = summary.get('dblock_ready_count', 0) or 0
        tier1_clean_rate = summary.get('tier1_clean_rate', 0) or 0
        tier2_clean_rate = summary.get('tier2_clean_rate', 0) or 0
        total_issues = summary.get('total_issues', 0) or 0
        critical_issues = summary.get('critical_issues', 0) or 0
        high_issues = summary.get('high_issues', 0) or 0
        open_count = issue_stats.get('open_count', total_issues) or total_issues

        return {
            "total_patients": total_patients,
            "total_sites": total_sites,
            "mean_dqi": mean_dqi,
            "dblock_ready_rate": dblock_ready_rate,
            "dblock_ready_count": dblock_ready_count,
            "tier1_clean_rate": tier1_clean_rate,
            "tier2_clean_rate": tier2_clean_rate,
            "total_issues": total_issues,
            "open_issues": open_count,
            "critical_issues": critical_issues,
            "high_issues": high_issues,
        }
    except Exception as e:
        logger.warning(f"Could not fetch live baseline: {e}")
        return {
            "total_patients": 0, "total_sites": 0, "mean_dqi": 0,
            "dblock_ready_rate": 0, "dblock_ready_count": 0,
            "tier1_clean_rate": 0, "tier2_clean_rate": 0,
            "total_issues": 0, "open_issues": 0,
            "critical_issues": 0, "high_issues": 0,
        }


def _get_site_level_data(study_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Pull per-site metrics for site-level variability modeling."""
    try:
        ds = get_data_service()
        benchmarks = ds.get_site_benchmarks(study_id=study_id)
        if benchmarks is not None and len(benchmarks) > 0:
            return benchmarks.to_dict('records')
        return []
    except Exception:
        return []


def _build_acceleration_scenarios(
    study_id: Optional[str],
    baseline: Dict[str, Any],
    gap: float,
    daily_capacity_mean: float,
    dblock_rate: float,
) -> List[Dict[str, str]]:
    """Build acceleration scenarios from real site-level data and baseline."""
    sites = _get_site_level_data(study_id)
    open_issues = baseline.get("open_issues", 0)
    tier1_clean_rate = baseline.get("tier1_clean_rate", 100)
    total_sites = max(1, baseline.get("total_sites", 1))

    scenarios: List[Dict[str, str]] = []

    # 1. Find sites with lowest DQI and suggest targeted intervention
    if sites:
        scored_sites = [
            s for s in sites if (s.get("dqi_score") or 0) > 0
        ]
        if scored_sites:
            scored_sites.sort(key=lambda s: s.get("dqi_score", 0))
            worst = scored_sites[:max(1, len(scored_sites) // 5)]  # bottom 20%
            worst_ids = ", ".join(
                str(s.get("site_id", "?")) for s in worst[:5]
            )
            avg_worst_dqi = sum(s.get("dqi_score", 0) for s in worst) / len(worst)
            # Estimate days saved: gap patients / boosted capacity
            boosted_capacity = daily_capacity_mean * 1.3 if daily_capacity_mean > 0 else 1
            days_saved_targeted = max(1, int(gap / boosted_capacity - gap / max(1, daily_capacity_mean)))
            scenarios.append({
                "name": f"Targeted Intervention on {len(worst)} low-DQI sites ({worst_ids})",
                "impact": f"Saves ~{days_saved_targeted} days (avg DQI {avg_worst_dqi:.0f} → boost capacity 30%)",
            })

    # 2. Issue resolution acceleration
    if open_issues > 500:
        resolvable_per_day = max(1, open_issues // (total_sites * 3))
        days_to_clear = open_issues // max(1, resolvable_per_day)
        auto_days = max(1, days_to_clear // 2)
        scenarios.append({
            "name": "Automated Issue Triage",
            "impact": f"Saves ~{auto_days} days by accelerating resolution of {open_issues:,} open issues",
        })
    elif open_issues > 0:
        resolvable_per_day = max(1, open_issues // (total_sites * 3))
        days_to_clear = open_issues // max(1, resolvable_per_day)
        sprint_days = max(1, int(days_to_clear * 0.35))
        scenarios.append({
            "name": "Issue Resolution Sprint",
            "impact": f"Saves ~{sprint_days} days by resolving {open_issues:,} open issues faster",
        })

    # 3. Data cleaning acceleration (if tier1 clean rate is low)
    if tier1_clean_rate < 80:
        cleaning_gap = 80 - tier1_clean_rate
        days_saved_cleaning = max(1, int(cleaning_gap * 0.5))
        scenarios.append({
            "name": "Data Cleaning Acceleration",
            "impact": f"Saves ~{days_saved_cleaning} days (Tier-1 clean rate {tier1_clean_rate:.0f}% → 80%)",
        })

    # 4. Fallback: if we still have fewer than 2 scenarios, add a lock-rate based one
    if len(scenarios) < 2:
        lock_gap = 100 - dblock_rate
        if lock_gap > 5:
            sprint_days = max(1, int(lock_gap * 0.3))
            scenarios.append({
                "name": "PI Signature Sprint",
                "impact": f"Saves ~{sprint_days} days ({lock_gap:.0f}% gap to DB-lock)",
            })

    return scenarios


# =============================================================================
# 1. TACTICAL WHAT-IF (POST /simulate)
# =============================================================================

@router.post("/what-if")
@router.post("/simulate")
async def run_tactical_scenario(
    request: Request,
    scenario_type: Optional[str] = Query(None),
    entity_id: Optional[str] = Query(None),
    action: str = Query("simulate"),
    current_user: dict = Depends(get_current_user)
):
    """POST handler for tactical scenarios (Site Closure, Resource Shifting)."""
    try:
        from app.services.digital_twin import digital_twin_service
        
        s_type = scenario_type
        e_id = entity_id
        
        if not s_type or not e_id:
            try:
                body = await request.json()
                s_type = s_type or body.get('scenario_type') or body.get('type')
                e_id = e_id or body.get('entity_id') or body.get('id')
            except:
                pass
        
        s_type = s_type or 'site_closure'
        e_id = e_id or 'US-001'
        
        res = digital_twin_service.run_what_if(s_type, e_id, action)
        
        # Service now returns data-driven results — no overrides needed
        
        return {
            "simulation_id": f"SIM-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            "scenario": res.get('scenario', s_type.title()),
            "impact_analysis": res.get('impact_analysis', {}),
            "alternatives": res.get('alternatives', []),
            "recommendation": res.get('recommendation', "Consult study lead."),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# 2. MONTE CARLO PROJECTIONS (GET)
# =============================================================================

@router.get("/db-lock-projection")
async def get_monte_carlo_db_lock(
    target_ready: Optional[int] = None,
    current_ready: Optional[int] = None,
    study_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Monte Carlo DB lock projection using real database metrics."""
    try:
        baseline = _get_live_baseline(study_id)
        total = baseline["total_patients"]
        dblock_count = baseline["dblock_ready_count"]
        dblock_rate = baseline["dblock_ready_rate"]
        
        t_ready = target_ready if (target_ready and target_ready > 100) else total
        c_ready = current_ready if (current_ready and current_ready > 100) else dblock_count
        
        if t_ready == 0:
            t_ready = 1
        
        gap = max(0, t_ready - c_ready)
        
        if gap == 0:
            return {
                "current_status": {"ready": int(c_ready), "target": int(t_ready), "percent": 100.0},
                "projection": {
                    "percentile_10": "Locked", "percentile_25": "Locked",
                    "percentile_50": "Locked", "percentile_75": "Locked",
                    "percentile_90": "Locked",
                    "acceleration_scenarios": []
                }
            }
        
        # Use real clean rate to inform daily processing capacity
        clean_rate = baseline["tier1_clean_rate"] / 100.0 if baseline["tier1_clean_rate"] > 0 else 0.5
        
        daily_capacity_mean = max(10, gap * 0.015 * clean_rate)
        daily_capacity_std = daily_capacity_mean * 0.3
        
        if np is None:
             return {
                "current_status": {"ready": c_ready, "target": t_ready, "percent": round((c_ready / t_ready) * 100, 1) if t_ready > 0 else 0},
                "projection": {
                    "percentile_10": "N/A", "percentile_25": "N/A", "percentile_50": "N/A", "percentile_75": "N/A", "percentile_90": "N/A",
                    "acceleration_scenarios": []
                }
            }

        if np is None:
             return {
                "current_status": {"ready": c_ready, "target": t_ready, "percent": round((c_ready / t_ready) * 100, 1) if t_ready > 0 else 0},
                "projection": {
                    "percentile_10": "N/A", "percentile_25": "N/A", "percentile_50": "N/A", "percentile_75": "N/A", "percentile_90": "N/A",
                    "acceleration_scenarios": []
                }
            }
        
        rng = np.random.default_rng()
        days_samples = []
        for _ in range(10000):
            remaining = gap
            days = 0
            while remaining > 0 and days < 365:
                daily = max(1, rng.normal(daily_capacity_mean, daily_capacity_std))
                remaining -= daily
                days += 1
            days_samples.append(days)
        
        days_dist = np.array(days_samples)
        days_dist.sort()
        base_date = datetime.now()
        
        return {
            "current_status": {
                "ready": int(c_ready), "target": int(t_ready),
                "percent": round((c_ready / t_ready) * 100, 1) if t_ready > 0 else 0
            },
            "projection": {
                "percentile_10": (base_date + timedelta(days=float(np.percentile(days_dist, 10)))).strftime("%B %d"),
                "percentile_25": (base_date + timedelta(days=float(np.percentile(days_dist, 25)))).strftime("%B %d"),
                "percentile_50": (base_date + timedelta(days=float(np.percentile(days_dist, 50)))).strftime("%B %d"),
                "percentile_75": (base_date + timedelta(days=float(np.percentile(days_dist, 75)))).strftime("%B %d"),
                "percentile_90": (base_date + timedelta(days=float(np.percentile(days_dist, 90)))).strftime("%B %d"),
                "acceleration_scenarios": _build_acceleration_scenarios(study_id, baseline, gap, daily_capacity_mean, dblock_rate)
            }
        }
    except Exception as e:
        logger.error(f"Error in db-lock-projection: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projections")
async def get_dashboard_projections(
    metric: str = Query(...),
    horizon_days: int = Query(90),
    study_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Dynamic projections grounded in real database state."""
    baseline = _get_live_baseline(study_id)
    
    if metric == "db_lock":
        current_val = baseline["dblock_ready_rate"]
        daily_growth = max(0.05, (100 - current_val) * 0.003)
    elif metric == "enrollment":
        current_val = baseline["total_patients"]
        daily_growth = max(1, current_val * 0.0008)
    elif metric == "dqi":
        current_val = baseline["mean_dqi"]
        daily_growth = max(0.001, (100 - current_val) * 0.002)
    else:
        current_val = baseline["mean_dqi"]
        daily_growth = 0.005
    
    rng = np.random.default_rng()
    projections = []
    val = current_val
    for day in range(horizon_days + 1):
        date = (datetime.now() + timedelta(days=day)).strftime('%Y-%m-%d')
        noise = rng.normal(0, daily_growth * 0.15)
        val = val + daily_growth + noise
        if metric != "enrollment":
            val = min(100, val)
        projections.append({
            "date": date,
            "value": round(val, 2),
            "lower_bound": round(val - abs(daily_growth * 2), 2),
            "upper_bound": round(min(100 if metric != "enrollment" else val * 1.01, val + abs(daily_growth * 2)), 2)
        })
    
    return {
        "metric": metric,
        "projections": projections,
        "summary": {
            "current": round(current_val, 2),
            "projected": projections[-1]['value'],
            "confidence": 0.95
        }
    }


# =============================================================================
# 3. MONTE CARLO ENGINE (POST /run) - Real Data-Driven
# =============================================================================

def _sim_enrollment_projection(baseline: Dict, sites: List[Dict], rng, iters: int, params: Optional[Dict] = None) -> Dict:
    """
    Enrollment Projection: Simulates days to reach enrollment target.
    Uses real patient count and site-level enrollment rates.
    Output: projected days to target enrollment (20% above current).
    """
    params = params or {}
    total_patients = baseline["total_patients"]
    total_sites = max(1, baseline["total_sites"])
    
    # Use frontend params if available
    target_enrollment = params.get('target_enrollment', None)
    current_enrollment = params.get('current_enrollment', None)
    enrollment_rate_override = params.get('enrollment_rate', None)
    variance_factor = params.get('variance', 0.25)
    
    current = current_enrollment if current_enrollment else total_patients
    target = target_enrollment if target_enrollment else int(total_patients * 1.20)
    gap = target - current
    
    if gap <= 0:
        return {
            "p10": 0, "p50": 0, "p90": 0, "mean_expected": 0,
            "unit": "days",
            "label": "Days to Target Enrollment",
            "description": f"Already at or above target ({current:,} patients enrolled)",
            "projected_completion_date": datetime.now().strftime('%Y-%m-%d'),
            "current_enrolled": current,
            "target_enrolled": target,
        }
    
    # Derive per-site daily enrollment rate from actual data
    site_enrollment_rates = []
    if sites:
        for s in sites:
            pc = s.get('patientCount', s.get('patient_count', 0)) or 0
            if pc > 0:
                site_enrollment_rates.append(pc / 365.0)
    
    if not site_enrollment_rates:
        site_enrollment_rates = [total_patients / total_sites / 365.0] * total_sites
    
    mean_daily_rate = sum(site_enrollment_rates)
    # Override with frontend-provided rate if available
    if enrollment_rate_override:
        mean_daily_rate = float(enrollment_rate_override) * total_sites
    
    std_daily_rate = mean_daily_rate * max(0.15, float(variance_factor))
    
    days_samples = []
    for _ in range(iters):
        remaining = gap
        days = 0
        # Per-iteration rate perturbation for real variability
        iter_rate = max(0.5, rng.normal(mean_daily_rate, std_daily_rate * 0.5))
        iter_std = iter_rate * rng.uniform(0.15, 0.45)
        while remaining > 0 and days < 730:
            daily = max(0, rng.normal(iter_rate, iter_std))
            remaining -= daily
            days += 1
        days_samples.append(days)
    
    days_arr = np.array(days_samples)
    days_arr.sort()
    p50 = float(np.median(days_arr))
    
    return {
        "p10": round(float(np.percentile(days_arr, 10)), 1),
        "p50": round(p50, 1),
        "p90": round(float(np.percentile(days_arr, 90)), 1),
        "mean_expected": round(float(np.mean(days_arr)), 1),
        "unit": "days",
        "label": "Days to Target Enrollment",
        "description": f"Projected days to enroll {gap:,} additional patients (target: {target:,})",
        "projected_completion_date": (datetime.now() + timedelta(days=p50)).strftime('%Y-%m-%d'),
        "current_enrolled": current,
        "target_enrolled": target,
    }


def _sim_db_lock_readiness(baseline: Dict, sites: List[Dict], rng, iters: int, params: Optional[Dict] = None) -> Dict:
    """
    DB Lock Readiness: Simulates projected DB lock ready % after 90 days.
    Uses real current lock rate, clean rates, and issue counts as friction.
    """
    params = params or {}
    current_rate = baseline["dblock_ready_rate"]
    clean_rate = baseline["tier1_clean_rate"]
    open_issues = baseline["open_issues"]
    total_patients = max(1, baseline["total_patients"])
    
    # Use frontend params if available
    resolution_rate = params.get('resolution_rate', 0.6)
    new_issues_rate = params.get('new_issues_rate', 0.10)
    
    issue_density = open_issues / total_patients if total_patients > 0 else 0
    clean_factor = (clean_rate / 100.0) if clean_rate > 0 else 0.3
    
    final_rates = []
    for _ in range(iters):
        rate = current_rate
        iss_dens = issue_density
        # Per-iteration variability in resolution effectiveness
        iter_resolution = rng.uniform(resolution_rate * 0.6, resolution_rate * 1.4)
        iter_new_issues = rng.uniform(new_issues_rate * 0.5, new_issues_rate * 1.5)
        for day in range(90):
            gap = 100.0 - rate
            if gap <= 0:
                break
            base_improvement = gap * 0.008 * clean_factor * iter_resolution / 0.6
            issue_friction = iss_dens * rng.uniform(0.1, 0.6)
            noise = rng.normal(0, base_improvement * 0.4)
            daily_change = base_improvement - issue_friction + noise
            rate = min(100.0, max(rate - 0.3, rate + daily_change))
            # Issue density changes over time
            resolution_effect = rng.uniform(0.002, 0.008)
            new_issue_effect = rng.uniform(0, iter_new_issues * 0.01)
            iss_dens = max(0, iss_dens * (1 - resolution_effect) + new_issue_effect)
        final_rates.append(rate)
    
    final_arr = np.array(final_rates)
    final_arr.sort()
    
    return {
        "p10": round(float(np.percentile(final_arr, 10)), 1),
        "p50": round(float(np.percentile(final_arr, 50)), 1),
        "p90": round(float(np.percentile(final_arr, 90)), 1),
        "mean_expected": round(float(np.mean(final_arr)), 1),
        "unit": "%",
        "label": "Projected DB Lock Ready Rate (90 days)",
        "description": f"Current: {current_rate:.1f}% lock ready | Clean rate: {clean_rate:.1f}% | {open_issues:,} open issues as friction",
        "current_rate": round(current_rate, 1),
        "probability_95_target": round(float(np.mean(final_arr >= 95.0)) * 100, 1),
    }


def _sim_risk_mitigation(baseline: Dict, sites: List[Dict], rng, iters: int, params: Optional[Dict] = None) -> Dict:
    """
    Risk Mitigation Impact: Simulates the % of open issues resolvable
    within 90 days with targeted interventions.
    """
    params = params or {}
    open_issues = max(1, baseline["open_issues"])
    critical = baseline["critical_issues"]
    high = baseline["high_issues"]
    other_issues = max(0, open_issues - critical - high)
    
    # Frontend params influence resolution probability ranges
    expected_improvement = params.get('expected_improvement', 0.2)
    improvement_scale = max(0.5, min(2.0, expected_improvement / 0.2))
    
    resolution_pcts = []
    for _ in range(iters):
        # Per-iteration variability in intervention effectiveness
        intervention_boost = rng.uniform(0.8, 1.2) * improvement_scale
        crit_prob = rng.uniform(0.55, 0.95) * min(1.0, intervention_boost)
        high_prob = rng.uniform(0.35, 0.85) * min(1.0, intervention_boost)
        other_prob = rng.uniform(0.20, 0.65) * min(1.0, intervention_boost)
        
        critical_resolved = rng.binomial(critical, min(0.99, crit_prob)) if critical > 0 else 0
        high_resolved = rng.binomial(high, min(0.99, high_prob)) if high > 0 else 0
        other_resolved = rng.binomial(other_issues, min(0.99, other_prob)) if other_issues > 0 else 0
        total_resolved = critical_resolved + high_resolved + other_resolved
        pct = (total_resolved / open_issues) * 100
        resolution_pcts.append(pct)
    
    pct_arr = np.array(resolution_pcts)
    pct_arr.sort()
    mean_resolved = int(open_issues * float(np.mean(pct_arr)) / 100)
    
    return {
        "p10": round(float(np.percentile(pct_arr, 10)), 1),
        "p50": round(float(np.percentile(pct_arr, 50)), 1),
        "p90": round(float(np.percentile(pct_arr, 90)), 1),
        "mean_expected": round(float(np.mean(pct_arr)), 1),
        "unit": "%",
        "label": "Projected Issue Resolution Rate (90 days)",
        "description": f"{open_issues:,} open issues ({critical:,} critical, {high:,} high) | Expected ~{mean_resolved:,} resolved",
        "total_open": open_issues,
        "expected_resolved": mean_resolved,
        "expected_remaining": open_issues - mean_resolved,
    }


def _sim_timeline_acceleration(baseline: Dict, sites: List[Dict], rng, iters: int, params: Optional[Dict] = None) -> Dict:
    """
    Timeline Acceleration: Simulates days saved through targeted interventions
    at bottleneck sites. Uses real site-level variability.
    """
    params = params or {}
    total_sites = max(1, baseline["total_sites"])
    mean_dqi = baseline["mean_dqi"]
    
    # Frontend params
    target_accel_days = params.get('target_acceleration_days', 30)
    resource_increase = params.get('resource_increase', 1.25)
    
    site_dqis = []
    if sites:
        for s in sites:
            dqi = s.get('dqi_score', 0) or 0
            if dqi > 0:
                site_dqis.append(dqi)
    
    if site_dqis and len(site_dqis) > 1:
        dqi_std = np.std(site_dqis)
        improvement_potential = min(1.0, dqi_std / 20.0)
    else:
        improvement_potential = 0.3
    
    gap_to_lock = max(1, 100 - baseline["dblock_ready_rate"])
    base_timeline_days = gap_to_lock * 2.5
    resource_multiplier = max(0.8, min(2.0, resource_increase))
    
    savings_samples = []
    for _ in range(iters):
        # Per-iteration resource effectiveness varies widely
        iter_resource = rng.uniform(resource_multiplier * 0.7, resource_multiplier * 1.3)
        intervention_effect = rng.beta(2 + improvement_potential * 5, 3) * improvement_potential
        days_saved = base_timeline_days * intervention_effect * iter_resource * rng.uniform(0.6, 1.4)
        savings_samples.append(max(0, days_saved))
    
    savings_arr = np.array(savings_samples)
    savings_arr.sort()
    
    return {
        "p10": round(float(np.percentile(savings_arr, 10)), 1),
        "p50": round(float(np.percentile(savings_arr, 50)), 1),
        "p90": round(float(np.percentile(savings_arr, 90)), 1),
        "mean_expected": round(float(np.mean(savings_arr)), 1),
        "unit": "days saved",
        "label": "Projected Days Saved via Interventions",
        "description": f"{total_sites} sites | {gap_to_lock:.0f}% gap to lock | DQI variance-driven improvement potential: {improvement_potential*100:.0f}%",
        "base_timeline_days": round(base_timeline_days, 0),
        "improvement_potential": round(improvement_potential * 100, 1),
    }


def _sim_resource_optimization(baseline: Dict, sites: List[Dict], rng, iters: int, params: Optional[Dict] = None) -> Dict:
    """
    Resource Optimization: Simulates DQI improvement achievable through
    optimal CRA resource allocation across sites.
    """
    params = params or {}
    mean_dqi = baseline["mean_dqi"]
    total_sites = max(1, baseline["total_sites"])
    
    # Frontend params
    available_cras = params.get('available_cras', 10)
    num_sites = params.get('num_sites', total_sites)
    cra_ratio = available_cras / max(1, num_sites)
    
    site_dqis = []
    if sites:
        for s in sites:
            dqi = s.get('dqi_score', 0) or 0
            if dqi > 0:
                site_dqis.append(dqi)
    
    if site_dqis and len(site_dqis) > 1:
        dqi_std = np.std(site_dqis)
        dqi_min = min(site_dqis)
        improvement_room = mean_dqi - dqi_min
    else:
        dqi_std = mean_dqi * 0.05
        improvement_room = mean_dqi * 0.1
    
    # Scale max improvement by CRA ratio (more CRAs per site = more improvement)
    resource_effect = min(2.0, cra_ratio * 1.5)
    max_improvement = min(8.0, improvement_room * 0.4 * resource_effect)
    
    improvement_samples = []
    for _ in range(iters):
        # Per-iteration CRA effectiveness varies
        iter_effectiveness = rng.uniform(0.6, 1.4)
        effect = rng.beta(2.5, 2) * max_improvement * iter_effectiveness
        noise = rng.normal(0, max_improvement * 0.15)
        improvement_samples.append(max(0, effect + noise))
    
    imp_arr = np.array(improvement_samples)
    imp_arr.sort()
    projected_dqi = round(mean_dqi + float(np.mean(imp_arr)), 1)
    
    return {
        "p10": round(float(np.percentile(imp_arr, 10)), 2),
        "p50": round(float(np.percentile(imp_arr, 50)), 2),
        "p90": round(float(np.percentile(imp_arr, 90)), 2),
        "mean_expected": round(float(np.mean(imp_arr)), 2),
        "unit": "DQI points",
        "label": "Projected DQI Improvement via Resource Optimization",
        "description": f"Current mean DQI: {mean_dqi:.1f} | Projected: {projected_dqi} | {total_sites} sites (std dev: {dqi_std:.1f})",
        "current_dqi": round(mean_dqi, 1),
        "projected_dqi": projected_dqi,
        "sites_analyzed": total_sites,
    }


def _sim_data_quality_forecast(baseline: Dict, sites: List[Dict], rng, iters: int, params: Optional[Dict] = None) -> Dict:
    """
    Data Quality Forecast: Simulates the portfolio DQI trajectory over 90 days.
    Models natural DQI improvement from ongoing data cleaning activities.
    """
    params = params or {}
    current_dqi = baseline["mean_dqi"]
    open_issues = baseline["open_issues"]
    total_patients = max(1, baseline["total_patients"])
    
    issue_drag = min(0.5, (open_issues / total_patients) * 0.1)
    gap_to_perfect = 100 - current_dqi
    
    final_dqis = []
    for _ in range(iters):
        dqi = current_dqi
        # Per-iteration cleaning effectiveness varies
        iter_cleaning_rate = rng.uniform(0.003, 0.008)
        iter_issue_drag = issue_drag * rng.uniform(0.5, 1.5)
        for day in range(90):
            gap = 100 - dqi
            if gap <= 0.01:
                break
            daily_improvement = gap * iter_cleaning_rate * (1 - iter_issue_drag)
            noise = rng.normal(0, daily_improvement * 0.5)
            dqi = min(100.0, dqi + daily_improvement + noise)
            # Random setback events (data re-review, new findings)
            if rng.random() < 0.08:
                dqi = max(current_dqi * 0.96, dqi - rng.uniform(0.05, 0.3))
        final_dqis.append(dqi)
    
    dqi_arr = np.array(final_dqis)
    dqi_arr.sort()
    
    return {
        "p10": round(float(np.percentile(dqi_arr, 10)), 1),
        "p50": round(float(np.percentile(dqi_arr, 50)), 1),
        "p90": round(float(np.percentile(dqi_arr, 90)), 1),
        "mean_expected": round(float(np.mean(dqi_arr)), 1),
        "unit": "DQI score",
        "label": "Projected DQI Score (90-day forecast)",
        "description": f"Current DQI: {current_dqi:.1f} | Gap to perfect: {gap_to_perfect:.1f} pts | Issue drag: {issue_drag:.2f}",
        "current_dqi": round(current_dqi, 1),
        "improvement_range": f"{round(float(np.percentile(dqi_arr, 10)) - current_dqi, 1)} to {round(float(np.percentile(dqi_arr, 90)) - current_dqi, 1)} points",
    }


def _sim_query_backlog(baseline: Dict, sites: List[Dict], rng, iters: int, params: Optional[Dict] = None) -> Dict:
    """
    Query Backlog Resolution: Simulates days to resolve the open issue backlog.
    Uses real open issue counts with stochastic resolution rates.
    """
    params = params or {}
    open_issues = max(1, baseline["open_issues"])
    total_sites = max(1, baseline["total_sites"])
    
    capacity_per_site = 3.0
    mean_daily_resolution = total_sites * capacity_per_site
    std_daily = mean_daily_resolution * 0.35
    new_issue_rate = mean_daily_resolution * 0.10
    
    days_samples = []
    for _ in range(iters):
        remaining = open_issues
        days = 0
        # Per-iteration team capacity varies
        iter_capacity = max(1, rng.normal(mean_daily_resolution, std_daily * 0.4))
        iter_std = iter_capacity * rng.uniform(0.2, 0.5)
        iter_new_rate = new_issue_rate * rng.uniform(0.5, 1.5)
        while remaining > 10 and days < 365:
            resolved = max(0, rng.normal(iter_capacity, iter_std))
            new_issues = max(0, rng.poisson(iter_new_rate))
            remaining = remaining - resolved + new_issues
            remaining = max(0, remaining)
            days += 1
        days_samples.append(days)
    
    days_arr = np.array(days_samples)
    days_arr.sort()
    
    return {
        "p10": round(float(np.percentile(days_arr, 10)), 1),
        "p50": round(float(np.percentile(days_arr, 50)), 1),
        "p90": round(float(np.percentile(days_arr, 90)), 1),
        "mean_expected": round(float(np.mean(days_arr)), 1),
        "unit": "days",
        "label": "Days to Clear Issue Backlog",
        "description": f"{open_issues:,} open issues across {total_sites} sites | ~{int(mean_daily_resolution)} resolved/day, ~{int(new_issue_rate)} new/day",
        "current_backlog": open_issues,
        "daily_resolution_rate": round(mean_daily_resolution, 0),
        "projected_completion_date": (datetime.now() + timedelta(days=float(np.median(days_arr)))).strftime('%Y-%m-%d'),
    }


def _sim_site_risk_analysis(baseline: Dict, sites: List[Dict], rng, iters: int, params: Optional[Dict] = None) -> Dict:
    """
    Site Risk Analysis: Simulates expected number of sites that will fall
    below DQI threshold (< 70) within the next 90 days.
    """
    params = params or {}
    total_sites = max(1, baseline["total_sites"])
    
    site_dqis = []
    if sites:
        for s in sites:
            dqi = s.get('dqi_score', 0) or 0
            if dqi > 0:
                site_dqis.append(dqi)
    
    if not site_dqis:
        site_dqis = [baseline["mean_dqi"]] * total_sites
    
    current_at_risk = sum(1 for d in site_dqis if d < 70)
    
    at_risk_samples = []
    for _ in range(iters):
        at_risk_count = 0
        # Per-iteration overall volatility
        iter_volatility = rng.uniform(2.0, 5.0)
        shock_prob = rng.uniform(0.02, 0.10)
        for dqi in site_dqis:
            drift = rng.normal(0, iter_volatility)
            if rng.random() < shock_prob:
                drift -= rng.uniform(3, 10)
            projected = dqi + drift
            if projected < 70:
                at_risk_count += 1
        at_risk_samples.append(at_risk_count)
    
    risk_arr = np.array(at_risk_samples)
    risk_arr.sort()
    at_risk_pct = round(float(np.mean(risk_arr)) / len(site_dqis) * 100, 1)
    
    return {
        "p10": round(float(np.percentile(risk_arr, 10)), 0),
        "p50": round(float(np.percentile(risk_arr, 50)), 0),
        "p90": round(float(np.percentile(risk_arr, 90)), 0),
        "mean_expected": round(float(np.mean(risk_arr)), 1),
        "unit": "sites",
        "label": "Projected At-Risk Sites (DQI < 70)",
        "description": f"Currently {current_at_risk} of {len(site_dqis)} sites below DQI 70 | Projected {at_risk_pct}% at risk in 90 days",
        "current_at_risk": current_at_risk,
        "total_sites": len(site_dqis),
        "risk_threshold": 70,
    }


@router.post("/run")
async def run_probabilistic_simulation(
    request: SimulationRequest,
    study_id: Optional[str] = Query(None),
    current_user: dict = Depends(get_current_user)
):
    """
    Monte Carlo Simulation Engine - Real data-driven.
    Pulls live metrics from trailpulse_test database and runs proper stochastic simulations.
    """
    try:
        if np is None:
            raise HTTPException(status_code=500, detail="Numpy not installed on server. Simulation unavailable.")
            
        # Use explicit seed only if provided; otherwise truly random each run
        if request.seed is not None:
            rng = np.random.default_rng(request.seed)
        else:
            rng = np.random.default_rng()
        iters = min(request.iterations, 10000)
        scenario = request.scenario_type
        params = request.parameters or {}
        
        # Pull real data from the database
        baseline = _get_live_baseline(study_id)
        sites = _get_site_level_data(study_id)
        
        sim_map = {
            "enrollment_projection": _sim_enrollment_projection,
            "enrollment": _sim_enrollment_projection,
            "db_lock_readiness": _sim_db_lock_readiness,
            "db_lock": _sim_db_lock_readiness,
            "risk_mitigation": _sim_risk_mitigation,
            "timeline_acceleration": _sim_timeline_acceleration,
            "resource_optimization": _sim_resource_optimization,
            "data_quality_forecast": _sim_data_quality_forecast,
            "dqi_forecast": _sim_data_quality_forecast,
            "query_backlog": _sim_query_backlog,
            "query_backlog_resolution": _sim_query_backlog,
            "site_risk_analysis": _sim_site_risk_analysis,
            "site_risk": _sim_site_risk_analysis,
        }
        
        sim_fn = sim_map.get(scenario)
        if not sim_fn:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown scenario: {scenario}. Available: {[k for k in sim_map if '_' not in k or k.count('_') <= 1]}"
            )
        
        sim_results = sim_fn(baseline, sites, rng, iters, params)
        
        # Map to standardized response format (frontend expects p10_days, p50_days, etc.)
        results = {
            "p10_days": sim_results["p10"],
            "p50_days": sim_results["p50"],
            "p90_days": sim_results["p90"],
            "mean_expected": sim_results["mean_expected"],
            "unit": sim_results.get("unit", ""),
            "label": sim_results.get("label", scenario),
            "description": sim_results.get("description", ""),
        }
        
        # Forward scenario-specific fields
        for key in [
            "projected_completion_date", "current_rate", "probability_95_target",
            "current_dqi", "projected_dqi", "total_open", "expected_resolved",
            "expected_remaining", "current_backlog", "current_at_risk",
            "total_sites", "improvement_range", "base_timeline_days",
            "daily_resolution_rate", "current_enrolled", "target_enrolled",
            "sites_analyzed", "improvement_potential", "risk_threshold",
        ]:
            if key in sim_results:
                results[key] = sim_results[key]
        
        return {
            "simulation_type": scenario,
            "results": results,
            "iterations": iters,
            "data_source": "trailpulse_test (live)",
            "timestamp": datetime.utcnow().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Simulation error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Simulation error: {str(e)}")


# =============================================================================
# 4. CURRENT STATE (LIVE BASELINE)
# =============================================================================

@router.get("/current-state")
async def get_real_time_state(study_id: Optional[str] = None, current_user: dict = Depends(get_current_user)):
    try:
        baseline = _get_live_baseline(study_id)
        lock_rate = baseline["dblock_ready_rate"]
        days_to_lock = max(1, int((100 - lock_rate) * 2.5))
        
        return {
            "baseline": {
                "total_patients": baseline["total_patients"],
                "mean_dqi": round(baseline["mean_dqi"], 1),
                "db_lock_ready_rate": round(lock_rate, 1),
                "open_issues": baseline["open_issues"],
                "total_sites": baseline["total_sites"],
                "tier1_clean_rate": round(baseline["tier1_clean_rate"], 1),
                "tier2_clean_rate": round(baseline["tier2_clean_rate"], 1),
            },
            "projections": {
                "days_to_db_lock": days_to_lock,
                "expected_dblock_date": (datetime.now() + timedelta(days=days_to_lock)).strftime('%Y-%m-%d')
            }
        }
    except Exception:
        return {
            "baseline": {
                "total_patients": 0, "mean_dqi": 0,
                "db_lock_ready_rate": 0, "open_issues": 0,
                "total_sites": 0,
            },
            "projections": {"days_to_db_lock": 0, "expected_dblock_date": "N/A"}
        }


@router.get("/scenarios")
async def list_scenario_templates(current_user: dict = Depends(get_current_user)):
    return {
        "scenarios": [
            {"id": "enrollment_projection", "name": "Enrollment Projection", "description": "Days to reach target enrollment", "unit": "days"},
            {"id": "db_lock_readiness", "name": "DB Lock Readiness", "description": "Projected lock-ready rate after 90 days", "unit": "%"},
            {"id": "risk_mitigation", "name": "Risk Mitigation Impact", "description": "Projected issue resolution rate", "unit": "%"},
            {"id": "timeline_acceleration", "name": "Timeline Acceleration", "description": "Days saved through targeted interventions", "unit": "days saved"},
            {"id": "resource_optimization", "name": "Resource Optimization", "description": "DQI improvement from optimal CRA allocation", "unit": "DQI points"},
            {"id": "data_quality_forecast", "name": "Data Quality Forecast", "description": "90-day DQI trajectory projection", "unit": "DQI score"},
            {"id": "query_backlog", "name": "Query Backlog Resolution", "description": "Days to clear issue backlog", "unit": "days"},
            {"id": "site_risk_analysis", "name": "Site Risk Analysis", "description": "Projected at-risk sites below DQI threshold", "unit": "sites"},
        ]
    }


@router.get("/what-if-analysis")
async def get_strategic_impact(
    intervention: str,
    magnitude: float = 1.0,
    study_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    baseline = _get_live_baseline(study_id)
    shift = (magnitude - 1.0)
    open_issues = baseline["open_issues"]
    mean_dqi = baseline["mean_dqi"]
    dblock_ready_rate = baseline["dblock_ready_rate"]
    tier1_clean_rate = baseline["tier1_clean_rate"]

    # Data-driven improvement estimates based on real gaps
    dqi_gap = 100 - mean_dqi
    lock_gap = 100 - dblock_ready_rate

    dqi_improvement = round(dqi_gap * shift * 0.05, 2)
    db_lock_improvement = round(lock_gap * shift * 0.08, 2)
    issues_reduced = int(open_issues * shift * 0.20) if open_issues > 0 else 0
    days_saved = int(lock_gap * shift * 0.3)

    # Confidence decreases with higher magnitude (more aggressive = less certain)
    confidence = round(max(0.70, 0.95 - (magnitude - 1.0) * 0.15), 2)

    # Dynamic recommendations based on which area has the most room for improvement
    recommendations = []
    improvement_areas = [
        (dqi_gap, "Optimize CRA workload distribution to close the DQI gap"),
        (lock_gap, "Accelerate DB-lock readiness across lagging sites"),
        (open_issues / max(1, baseline["total_patients"]) * 100,
         "Prioritize critical-path issue resolution"),
    ]
    if tier1_clean_rate < 80:
        improvement_areas.append(
            (80 - tier1_clean_rate, "Improve Tier-1 data cleaning processes")
        )
    if baseline["critical_issues"] > 0:
        improvement_areas.append(
            (baseline["critical_issues"], "Address critical issues immediately to unblock lock")
        )
    # Sort descending by gap/room and pick top 2-3
    improvement_areas.sort(key=lambda x: x[0], reverse=True)
    recommendations = [area[1] for area in improvement_areas[:3]]

    return {
        "intervention": intervention,
        "magnitude": magnitude,
        "improvement": {
            "dqi_improvement": dqi_improvement,
            "db_lock_improvement": db_lock_improvement,
            "issues_reduced": issues_reduced,
            "days_saved": days_saved
        },
        "confidence": confidence,
        "recommendations": recommendations,
        "data_source": "trailpulse_test (live)"
    }
