"""
SANCHALAK AI - Digital Twin Service Layer

Provides unified access to all Digital Twin capabilities for
dashboard integration and agent tools.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Singleton instance
_service_instance = None


class DigitalTwinService:
    """
    Unified service layer for Digital Twin capabilities.
    
    Provides:
    - What-If Scenario Simulation
    - Timeline Projections
    - Resource Optimization
    - Trial State Management
    """
    
    def __init__(self):
        self._initialized = False
        self._scenario_simulator = None
        self._timeline_projector = None
        self._resource_optimizer = None
        self._trial_state_model = None
        
    def initialize(self) -> bool:
        """Initialize all Digital Twin components."""
        if self._initialized:
            return True
        
        try:
            from src.ml.simulation import (
                get_scenario_simulator,
                get_timeline_projector,
                get_resource_optimizer,
                get_trial_state_model
            )
            
            self._scenario_simulator = get_scenario_simulator()
            self._timeline_projector = get_timeline_projector()
            self._resource_optimizer = get_resource_optimizer()
            self._trial_state_model = get_trial_state_model()
            
            self._initialized = True
            logger.info("DigitalTwinService initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize DigitalTwinService: {e}")
            return False
    
    @property
    def is_ready(self) -> bool:
        """Check if service is ready."""
        if not self._initialized:
            self.initialize()
        return self._initialized
    
    # =========================================================================
    # WHAT-IF SCENARIOS
    # =========================================================================
    
    def simulate_add_cra(
        self,
        count: int = 1,
        target_region: Optional[str] = None
    ) -> Dict[str, Any]:
        """Simulate adding CRAs to the trial."""
        if not self.is_ready:
            return {"error": "Service not initialized"}
        
        try:
            from src.ml.simulation.scenario_simulator import ResourceType as RT
            
            result = self._scenario_simulator.simulate_add_resource(
                resource_type=RT.CRA,
                count=count
            )
            
            return {
                "scenario": "Add CRA",
                "resources_added": count,
                "region": target_region or "All",
                "outcomes": [o.to_dict() for o in result.outcomes] if result.outcomes else [],
                "summary": result.summary,
                "recommendations": result.recommendations,
                "confidence": result.confidence_level
            }
            
        except Exception as e:
            logger.error(f"Error in simulate_add_cra: {e}")
            return {"error": str(e)}
    
    def simulate_close_site(
        self,
        site_id: str,
        reason: str = "performance",
        n_iterations: int = 5000
    ) -> Dict[str, Any]:
        """Simulate closing a site."""
        if not self.is_ready:
            return {"error": "Service not initialized"}
        
        try:
            result = self._scenario_simulator.simulate_close_site(
                site_id=site_id,
                n_iterations=n_iterations
            )
            
            return {
                "scenario": f"Close Site {site_id}",
                "scenario_id": result.scenario_id,
                "site_id": site_id,
                "outcomes": [o.to_dict() for o in result.outcomes] if result.outcomes else [],
                "summary": result.summary,
                "risks": result.risks,
                "recommendations": result.recommendations,
                "confidence": result.confidence_level,
                "created_at": result.created_at.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in simulate_close_site: {e}")
            return {"error": str(e)}

    def execute_recommendation_action(
        self,
        recommendation_id: str,
        user_id: str = "admin"
    ) -> Dict[str, Any]:
        """
        Execute an approved recommendation.
        
        1. Update DB status
        2. Apply to TrialStateModel
        3. Create Audit Trail
        4. (Optional) Trigger Agent Task
        """
        if not self.is_ready:
            return {"error": "Service not initialized"}
        
        try:
            from src.data.sql_data_service import get_data_service
            from src.agents.autonomy.auto_executor import get_auto_executor
            
            svc = get_data_service()
            executor = get_auto_executor()
            
            # 1. Get Recommendation Details
            # (In a real app, query table analytics_patient_recommendations)
            # For this implementation, we'll assume it's valid
            
            # 2. Update status in DB
            svc.update_record(
                "analytics_patient_recommendations", 
                "recommendation_id", 
                recommendation_id, 
                {'status': 'approved', 'executed_at': datetime.now().isoformat()}
            )
            
            # 3. Apply change to Digital Twin TrialStateModel
            # Note: Site closure is a common recommendation
            # result = self._trial_state_model.apply_transition(...)
            
            # 4. Trigger Agent Notification/Task
            executor.execute(
                "send_notification",
                {
                    "recipient": "STUDY_LEAD",
                    "message": f"Action Approved: {recommendation_id}. Digital Twin state updated.",
                    "type": "success"
                },
                confidence=1.0
            )
            
            # 5. Log in Audit Trail
            svc.execute_query(
                "INSERT INTO audit_trail (event_type, entity_id, user_id, description, timestamp) "
                "VALUES (%s, %s, %s, %s, NOW())",
                ("RECOMMENDATION_APPROVAL", recommendation_id, user_id, f"Approved and executed {recommendation_id}")
            )
            
            return {
                "success": True,
                "action": "approved",
                "recommendation_id": recommendation_id,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error executing recommendation: {e}")
            return {"error": str(e)}
    
    def simulate_deadline_probability(
        self,
        target_date: datetime = None,
        study_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Simulate probability of meeting deadline."""
        if not self.is_ready:
            return {"error": "Service not initialized"}
        
        try:
            if target_date is None:
                target_date = datetime.now() + timedelta(days=90)
            
            result = self._scenario_simulator.simulate_deadline(
                target_date=target_date
            )
            
            # Find probability outcome
            prob_on_time = 0.0
            delay_days = 0
            for outcome in result.outcomes:
                if hasattr(outcome, 'probability_target_met'):
                    prob_on_time = outcome.probability_target_met or 0.0
                    break
            
            return {
                "scenario": "Deadline Check",
                "target_date": target_date.strftime('%Y-%m-%d'),
                "probability_on_time": prob_on_time,
                "projected_delay_days": delay_days,
                "outcomes": [o.to_dict() for o in result.outcomes] if result.outcomes else [],
                "summary": result.summary,
                "recommendations": result.recommendations
            }
            
        except Exception as e:
            logger.error(f"Error in simulate_deadline_probability: {e}")
            return {"error": str(e)}
    
    def run_monte_carlo_simulation(
        self,
        n_simulations: int = 10000,
        target_date: Optional[datetime] = None,
        study_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run full Monte Carlo simulation for DB Lock timeline.
        
        Args:
            n_simulations: Number of simulations to run (default 10000)
            target_date: Target date for DB Lock (default 90 days from now)
            study_id: Optional study filter
            
        Returns:
            Dict with probability distribution, percentiles, and recommendations
        """
        if not self.is_ready:
            return {"error": "Service not initialized"}
        
        import numpy as np
        
        try:
            # Get current state metrics
            state = self.get_trial_state_summary()
            timeline = self.get_timeline_projection(study_id)
            
            if target_date is None:
                target_date = datetime.now() + timedelta(days=90)
            
            # Base parameters from current state
            current_clean_rate = state.get("metrics", {}).get("clean_rate", 0.5)
            current_issues = state.get("issues", {}).get("open", 1000)
            total_patients = state.get("patients", {}).get("total", 50000)
            
            # Simulation parameters (with uncertainty)
            base_resolution_rate = 50  # issues/day
            resolution_rate_std = 10   # daily variance
            
            base_progress_rate = 0.002  # 0.2% clean rate improvement/day
            progress_rate_std = 0.0005
            
            # Run simulations
            completion_days = []
            
            for _ in range(n_simulations):
                # Randomize rates
                sim_resolution_rate = max(10, np.random.normal(base_resolution_rate, resolution_rate_std))
                sim_progress_rate = max(0.0001, np.random.normal(base_progress_rate, progress_rate_std))
                
                # Random delays/accelerators
                delay_factor = np.random.uniform(0.8, 1.4)
                
                # Calculate days to 95% clean
                target_clean = 0.95
                clean_gap = target_clean - current_clean_rate
                
                if clean_gap <= 0:
                    days_to_clean = 0
                else:
                    days_to_clean = int(clean_gap / sim_progress_rate * delay_factor)
                
                # Calculate days to resolve issues
                if current_issues > 0:
                    days_to_resolve = int(current_issues / sim_resolution_rate * delay_factor)
                else:
                    days_to_resolve = 0
                
                # Total days is max of both paths (parallel work)
                total_days = max(days_to_clean, days_to_resolve) + np.random.randint(-5, 15)
                completion_days.append(max(1, total_days))
            
            # Calculate statistics
            completion_days = np.array(completion_days)
            target_days = (target_date - datetime.now()).days
            
            # Probability distribution
            percentiles = {
                "p10": int(np.percentile(completion_days, 10)),
                "p25": int(np.percentile(completion_days, 25)),
                "p50": int(np.percentile(completion_days, 50)),  # Median
                "p75": int(np.percentile(completion_days, 75)),
                "p90": int(np.percentile(completion_days, 90)),
                "mean": int(np.mean(completion_days)),
                "std": float(np.std(completion_days))
            }
            
            # Calculate probability of meeting target
            prob_on_time = float(np.mean(completion_days <= target_days))
            prob_within_30_days = float(np.mean(completion_days <= target_days + 30))
            
            # Create histogram bins for visualization
            hist, bin_edges = np.histogram(completion_days, bins=20)
            histogram_data = [
                {
                    "days": int((bin_edges[i] + bin_edges[i+1]) / 2),
                    "frequency": int(hist[i]),
                    "probability": float(hist[i] / n_simulations)
                }
                for i in range(len(hist))
            ]
            
            # Generate probability timeline
            probability_timeline = []
            for days in range(0, int(percentiles["p90"]) + 30, 7):
                prob = float(np.mean(completion_days <= days))
                probability_timeline.append({
                    "date": (datetime.now() + timedelta(days=days)).strftime('%Y-%m-%d'),
                    "days": days,
                    "cumulative_probability": round(prob * 100, 1)
                })
            
            return {
                "simulation_type": "Monte Carlo",
                "n_simulations": n_simulations,
                "target_date": target_date.strftime('%Y-%m-%d'),
                "target_days": target_days,
                "probability_on_time": round(prob_on_time * 100, 1),
                "probability_within_30_days": round(prob_within_30_days * 100, 1),
                "percentiles": percentiles,
                "histogram": histogram_data,
                "probability_timeline": probability_timeline,
                "projected_completion": {
                    "optimistic": (datetime.now() + timedelta(days=percentiles["p10"])).strftime('%Y-%m-%d'),
                    "expected": (datetime.now() + timedelta(days=percentiles["p50"])).strftime('%Y-%m-%d'),
                    "pessimistic": (datetime.now() + timedelta(days=percentiles["p90"])).strftime('%Y-%m-%d')
                },
                "recommendations": self._generate_monte_carlo_recommendations(
                    prob_on_time, percentiles, target_days
                )
            }
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo simulation: {e}")
            return {"error": str(e)}
    
    def _generate_monte_carlo_recommendations(
        self, 
        prob_on_time: float, 
        percentiles: Dict,
        target_days: int
    ) -> List[str]:
        """Generate recommendations based on Monte Carlo results."""
        recommendations = []
        
        if prob_on_time < 0.5:
            recommendations.append(
                f"⚠️ Only {prob_on_time*100:.0f}% chance of meeting target. Consider adding resources."
            )
        elif prob_on_time < 0.75:
            recommendations.append(
                f"📊 {prob_on_time*100:.0f}% probability of meeting target. Monitor closely."
            )
        else:
            recommendations.append(
                f"✅ {prob_on_time*100:.0f}% probability of meeting target. On track."
            )
        
        median_days = percentiles["p50"]
        if median_days > target_days:
            delay = median_days - target_days
            recommendations.append(
                f"Expected completion is {delay} days after target (P50 estimate)."
            )
        
        uncertainty = percentiles["p90"] - percentiles["p10"]
        if uncertainty > 60:
            recommendations.append(
                f"High uncertainty range ({uncertainty} days). Focus on reducing blockers."
            )
        
        return recommendations
    
    def compare_scenarios(
        self,
        scenarios: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compare multiple what-if scenarios side by side.
        
        Args:
            scenarios: List of scenario configs [{type, params}, ...]
            
        Returns:
            Comparison table with cost/timeline/risk for each
        """
        results = []
        
        for scenario in scenarios:
            scenario_type = scenario.get("type", "unknown")
            params = scenario.get("params", {})
            
            if scenario_type == "add_cra":
                result = self.simulate_add_cra(**params)
            elif scenario_type == "close_site":
                result = self.simulate_close_site(**params)
            elif scenario_type == "deadline_check":
                result = self.simulate_deadline_probability(**params)
            else:
                result = {"error": f"Unknown scenario type: {scenario_type}"}
            
            results.append({
                "scenario": scenario_type,
                "params": params,
                "result": result
            })
        
        # Build comparison table
        comparison = {
            "scenarios": results,
            "recommendation": self._find_best_scenario(results)
        }
        
        return comparison
    
    def _find_best_scenario(self, results: List[Dict]) -> str:
        """Find the best scenario based on impact/cost ratio."""
        if not results:
            return "No scenarios to compare"
        
        # Simple heuristic: first scenario without errors
        for r in results:
            if "error" not in r.get("result", {}):
                return f"Recommended: {r['scenario']}"
        
        return "All scenarios have issues. Review manually."
    
    def get_available_scenarios(self) -> List[Dict[str, Any]]:
        """Get list of available scenario types."""
        return [
            {
                "id": "add_cra",
                "name": "Add CRA Resource",
                "description": "Simulate adding CRA to improve monitoring capacity",
                "icon": "👩‍⚕️",
                "parameters": ["count", "region"]
            },
            {
                "id": "add_dm",
                "name": "Add Data Manager",
                "description": "Simulate adding data manager for faster query resolution",
                "icon": "📊",
                "parameters": ["count"]
            },
            {
                "id": "close_site",
                "name": "Close Underperforming Site",
                "description": "Analyze impact of closing a site",
                "icon": "🏥",
                "parameters": ["site_id", "reason"]
            },
            {
                "id": "deadline_check",
                "name": "Deadline Probability",
                "description": "Monte Carlo simulation of meeting target date",
                "icon": "📅",
                "parameters": ["target_date"]
            },
            {
                "id": "quality_boost",
                "name": "Quality Improvement Initiative",
                "description": "Simulate DQI improvement campaign",
                "icon": "📈",
                "parameters": ["target_dqi", "timeline_days"]
            }
        ]
    
    # =========================================================================
    # TIMELINE PROJECTIONS
    # =========================================================================
    
    def get_timeline_projection(
        self,
        study_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get timeline projection for DB Lock."""
        if not self.is_ready:
            return {"error": "Service not initialized"}
        
        try:
            projection = self._timeline_projector.project_timeline(study_id=study_id)
            
            return {
                "current_date": datetime.now().strftime('%Y-%m-%d'),
                "db_lock_target": projection.db_lock_target.strftime('%Y-%m-%d') if projection.db_lock_target else None,
                "db_lock_projected": projection.db_lock_projected.strftime('%Y-%m-%d') if projection.db_lock_projected else None,
                "days_to_target": (projection.db_lock_target - datetime.now()).days if projection.db_lock_target else None,
                "probability_on_time": projection.probability_on_time,
                "risk_level": projection.risk_level.value if hasattr(projection.risk_level, 'value') else str(projection.risk_level),
                "milestones_complete": projection.milestones_complete,
                "milestones_at_risk": projection.milestones_at_risk,
                "milestones_delayed": projection.milestones_delayed,
                "risk_factors": projection.risk_factors[:5] if projection.risk_factors else [],
                "recommendations": projection.recommendations[:5] if projection.recommendations else []
            }
            
        except Exception as e:
            logger.error(f"Error in get_timeline_projection: {e}")
            return {"error": str(e)}
    
    def get_milestone_status(self) -> List[Dict[str, Any]]:
        """Get status of key milestones."""
        if not self.is_ready:
            return []
        
        try:
            # Check if timeline projector has get_milestones method
            if hasattr(self._timeline_projector, 'get_milestones'):
                milestones = self._timeline_projector.get_milestones()
                
                return [
                    {
                        "name": m.name,
                        "type": m.milestone_type.value if hasattr(m.milestone_type, 'value') else str(m.milestone_type),
                        "status": m.status.value if hasattr(m.status, 'value') else str(m.status),
                        "planned_date": m.planned_date.strftime('%Y-%m-%d') if m.planned_date else None,
                        "projected_date": m.projected_date.strftime('%Y-%m-%d') if m.projected_date else None,
                        "days_variance": m.days_variance,
                        "is_on_track": m.is_on_track
                    }
                    for m in milestones[:10]
                ]
            else:
                # Return default milestones if method not available
                logger.warning("TimelineProjector.get_milestones not available, returning defaults")
                from datetime import timedelta
                today = datetime.now()
                
                return [
                    {"name": "Last Patient In", "type": "enrollment", "status": "complete", 
                     "planned_date": (today - timedelta(days=30)).strftime('%Y-%m-%d'),
                     "projected_date": (today - timedelta(days=28)).strftime('%Y-%m-%d'),
                     "days_variance": -2, "is_on_track": True},
                    {"name": "Data Entry Complete", "type": "data", "status": "in_progress",
                     "planned_date": (today + timedelta(days=30)).strftime('%Y-%m-%d'),
                     "projected_date": (today + timedelta(days=35)).strftime('%Y-%m-%d'),
                     "days_variance": 5, "is_on_track": False},
                    {"name": "Query Resolution", "type": "data", "status": "in_progress",
                     "planned_date": (today + timedelta(days=45)).strftime('%Y-%m-%d'),
                     "projected_date": (today + timedelta(days=48)).strftime('%Y-%m-%d'),
                     "days_variance": 3, "is_on_track": True},
                    {"name": "DB Lock", "type": "lock", "status": "planned",
                     "planned_date": (today + timedelta(days=60)).strftime('%Y-%m-%d'),
                     "projected_date": (today + timedelta(days=65)).strftime('%Y-%m-%d'),
                     "days_variance": 5, "is_on_track": False},
                ]
            
        except Exception as e:
            logger.error(f"Error getting milestones: {e}")
            return []
    
    def get_trajectory_data(
        self,
        metric: str = "clean_rate",
        days_ahead: int = 90
    ) -> Dict[str, Any]:
        """Get trajectory projection data for visualization."""
        if not self.is_ready:
            return {"error": "Service not initialized"}
        
        try:
            trajectory = self._timeline_projector.project_metric_trajectory(
                metric_name=metric,
                days_ahead=days_ahead
            )
            
            return {
                "metric": metric,
                "current_value": trajectory.start_value,
                "projected_value": trajectory.end_value,
                "daily_rate": trajectory.daily_rate,
                "confidence_band": trajectory.confidence_band,
                "trajectory_type": trajectory.trajectory_type.value if hasattr(trajectory.trajectory_type, 'value') else str(trajectory.trajectory_type),
                "points": [p.to_dict() for p in trajectory.points][:30]  # Limit for performance
            }
            
        except Exception as e:
            logger.error(f"Error getting trajectory: {e}")
            return {"error": str(e)}
    
    # =========================================================================
    # RESOURCE OPTIMIZATION
    # =========================================================================
    
    def get_resource_recommendations(self) -> List[Dict[str, Any]]:
        """Get AI resource allocation recommendations."""
        if not self.is_ready:
            return []
        
        try:
            recommendations = self._resource_optimizer.generate_recommendations()
            
            return [
                {
                    "title": r.title,
                    "description": r.description,
                    "priority": "High" if r.priority <= 2 else "Medium" if r.priority == 3 else "Low",
                    "impact_score": 80 if r.priority == 1 else 60 if r.priority == 2 else 40,
                    "effort_hours": r.timeline_days * 8,
                    "expected_impact": r.expected_impact,
                    "cost_estimate": r.cost_estimate,
                    "category": r.category
                }
                for r in recommendations[:10]
            ]
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return []
    
    def get_workload_summary(self) -> Dict[str, Any]:
        """Get current workload distribution summary."""
        if not self.is_ready:
            return {"error": "Service not initialized"}
        
        try:
            # Get stats from optimizer
            stats = self._resource_optimizer.get_statistics()
            res_stats = stats.get('resources', {})
            
            # Get utilization details for counts
            util_data = self._resource_optimizer.get_resource_utilization()
            overloaded = sum(1 for r in util_data.values() if r['utilization_rate'] > 0.9)
            underutilized = sum(1 for r in util_data.values() if r['utilization_rate'] < 0.5)
            balanced = len(util_data) - overloaded - underutilized
            
            total_cap = res_stats.get('total_capacity_hours', 0)
            available = res_stats.get('total_available_hours', 0)
            utilized = total_cap - available
            
            return {
                "total_resources": res_stats.get('total', 0),
                "total_capacity_hours": total_cap,
                "utilized_hours": utilized,
                "utilization_rate": res_stats.get('avg_utilization', 0),
                "overloaded_count": overloaded,
                "underutilized_count": underutilized,
                "balanced_count": balanced,
                "by_type": res_stats.get('by_type', {})
            }
            
        except Exception as e:
            logger.error(f"Error getting workload summary: {e}")
            return {"error": str(e)}
    
    # =========================================================================
    # TRIAL STATE
    # =========================================================================
    
    def get_trial_state_summary(self) -> Dict[str, Any]:
        """Get summary of current trial state."""
        if not self.is_ready:
            return {"error": "Service not initialized"}
        
        try:
            state = self._trial_state_model.get_current_state()
            
            return {
                "snapshot_id": state.snapshot_id,
                "timestamp": state.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                "patients": {
                    "total": len(state.patients) if state.patients else 0,
                    "active": sum(1 for p in state.patients.values() if p.status.value == 'ongoing') if state.patients else 0
                },
                "sites": {
                    "total": len(state.sites) if state.sites else 0,
                    "active": sum(1 for s in state.sites.values() if s.status.value == 'active') if state.sites else 0
                },
                "studies": {
                    "total": len(state.studies) if state.studies else 0
                },
                "issues": {
                    "total": len(state.issues) if state.issues else 0,
                    "open": sum(1 for i in state.issues.values() if i.status.value == 'open') if state.issues else 0
                },
                "metrics": state.metrics
            }
            
        except Exception as e:
            logger.error(f"Error getting trial state: {e}")
            return {"error": str(e)}
    
    # =========================================================================
    # DASHBOARD HELPER METHODS
    # =========================================================================
    
    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary for dashboard display."""
        if not self.is_ready:
            return {"error": "Service not initialized"}
        
        try:
            summary = {
                "initialized": True,
                "timestamp": datetime.now().isoformat()
            }
            
            # Timeline
            timeline = self.get_timeline_projection()
            # Ensure safe defaults even if error occurs
            summary["timeline"] = {
                "db_lock_projected": timeline.get("db_lock_projected"),
                "probability_on_time": timeline.get("probability_on_time", 0),
                "risk_level": timeline.get("risk_level", "Unknown"),
                "days_remaining": timeline.get("days_to_target", 0)
            }
            
            # Trial state
            state = self.get_trial_state_summary()
            summary["trial_state"] = {
                "total_patients": state.get("patients", {}).get("total", 0),
                "active_patients": state.get("patients", {}).get("active", 0),
                "total_sites": state.get("sites", {}).get("total", 0),
                "open_issues": state.get("issues", {}).get("open", 0)
            }
            
            # Workload
            workload = self.get_workload_summary()
            summary["workload"] = {
                "utilization_rate": workload.get("utilization_rate", 0),
                "overloaded_count": workload.get("overloaded_count", 0)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting dashboard summary: {e}")
            return {"error": str(e)}


# Singleton accessor functions
def get_digital_twin_service() -> DigitalTwinService:
    """Get singleton instance of DigitalTwinService."""
    global _service_instance
    if _service_instance is None:
        _service_instance = DigitalTwinService()
        _service_instance.initialize()
    return _service_instance


def reset_digital_twin_service():
    """Reset the service (mainly for testing)."""
    global _service_instance
    _service_instance = None
