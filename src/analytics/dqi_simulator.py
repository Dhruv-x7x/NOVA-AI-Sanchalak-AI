"""
a6on-i - DQI Improvement Simulator
Interactive "Fix X → DQI improves by Y" simulation engine.

Allows sites and users to understand the impact of fixing specific issues
on their Data Quality Index score.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# DQI Component Weights (from solution doc)
DQI_WEIGHTS = {
    'safety_score': 0.25,
    'query_score': 0.20,
    'completeness_score': 0.15,
    'coding_score': 0.12,
    'lab_score': 0.10,
    'sdv_score': 0.08,
    'signature_score': 0.05,
    'edrr_score': 0.05
}

# Issue to DQI Component Mapping
ISSUE_TO_COMPONENT = {
    'issue_sae_dm_pending': 'safety_score',
    'issue_sae_safety_pending': 'safety_score',
    'issue_open_queries': 'query_score',
    'issue_missing_visits': 'completeness_score',
    'issue_missing_pages': 'completeness_score',
    'issue_meddra_uncoded': 'coding_score',
    'issue_whodrug_uncoded': 'coding_score',
    'issue_lab_issues': 'lab_score',
    'issue_sdv_incomplete': 'sdv_score',
    'issue_signature_gaps': 'signature_score',
    'issue_broken_signatures': 'signature_score',
    'issue_edrr_issues': 'edrr_score',
    'issue_inactivated_forms': 'completeness_score'
}

# Base penalty per issue type
ISSUE_PENALTIES = {
    'issue_sae_dm_pending': 8.0,
    'issue_sae_safety_pending': 10.0,
    'issue_open_queries': 5.0,
    'issue_missing_visits': 6.0,
    'issue_missing_pages': 4.0,
    'issue_meddra_uncoded': 3.0,
    'issue_whodrug_uncoded': 3.0,
    'issue_lab_issues': 4.0,
    'issue_sdv_incomplete': 5.0,
    'issue_signature_gaps': 6.0,
    'issue_broken_signatures': 4.0,
    'issue_edrr_issues': 3.0,
    'issue_inactivated_forms': 2.0
}


@dataclass
class FixSimulation:
    """Result of simulating a fix"""
    issue_type: str
    issue_display_name: str
    current_count: int
    fixed_count: int
    dqi_improvement: float
    new_dqi: float
    component_affected: str
    effort_estimate_minutes: int
    priority: str


@dataclass
class SimulationResult:
    """Complete simulation result"""
    patient_key: Optional[str]
    site_id: Optional[str]
    current_dqi: float
    projected_dqi: float
    total_improvement: float
    fixes_applied: List[FixSimulation]
    improvement_percentage: float
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'patient_key': self.patient_key,
            'site_id': self.site_id,
            'current_dqi': self.current_dqi,
            'projected_dqi': self.projected_dqi,
            'total_improvement': self.total_improvement,
            'improvement_percentage': self.improvement_percentage,
            'fixes_count': len(self.fixes_applied),
            'recommendations': self.recommendations,
            'timestamp': self.timestamp.isoformat()
        }


class DQIImprovementSimulator:
    """
    Simulates DQI improvements based on fixing specific issues.
    
    Usage:
        simulator = get_dqi_simulator()
        
        # Single issue simulation
        result = simulator.simulate_fix("patient_key", "issue_open_queries")
        
        # Multiple issues
        result = simulator.simulate_batch_fixes("patient_key", ["issue_open_queries", "issue_sdv_incomplete"])
        
        # Get recommendations for a site
        recs = simulator.get_fix_recommendations("Site_001", top_n=5)
    """
    
    def __init__(self):
        self._data_cache = {}
        logger.info("DQIImprovementSimulator initialized")
    
    def _get_patient_data(self, patient_key: str = None, site_id: str = None) -> Dict[str, Any]:
        """Get patient or site data for simulation"""
        try:
            from src.data.sql_data_service import get_data_service
            service = get_data_service()
            
            if patient_key:
                patients = service.get_patients()
                if patients is not None and not patients.empty:
                    patient = patients[patients.get('patient_key', patients.get('patient_id', 'None')) == patient_key]
                    if not patient.empty:
                        return patient.iloc[0].to_dict()
            
            if site_id:
                patients = service.get_patients()
                if patients is not None and not patients.empty:
                    site_patients = patients[patients['site_id'] == site_id]
                    if not site_patients.empty:
                        return {
                            'site_id': site_id,
                            'patient_count': len(site_patients),
                            'avg_dqi': site_patients['dqi_score'].mean() if 'dqi_score' in site_patients.columns else 75.0,
                            'patients': site_patients.to_dict('records')[:50]
                        }
            
            return {}
        except Exception as e:
            logger.warning(f"Could not get patient data: {e}")
            return {}
    
    def _calculate_current_dqi(self, data: Dict[str, Any]) -> float:
        """Calculate current DQI from data"""
        if 'dqi_score' in data:
            return float(data['dqi_score'])
        if 'avg_dqi' in data:
            return float(data['avg_dqi'])
        return 75.0  # Default
    
    def _count_issues(self, data: Dict[str, Any], issue_type: str) -> int:
        """Count issues of a specific type"""
        # Check for issue flag
        if issue_type in data:
            val = data[issue_type]
            if isinstance(val, bool):
                return 1 if val else 0
            return int(val) if val else 0
        
        # Check for count columns
        count_col = issue_type.replace('issue_', 'count_')
        if count_col in data:
            return int(data[count_col]) if data[count_col] else 0
        
        return 0
    
    def _get_effort_estimate(self, issue_type: str, count: int = 1) -> int:
        """Estimate effort in minutes to resolve issue"""
        base_effort = {
            'issue_sae_dm_pending': 45,
            'issue_sae_safety_pending': 60,
            'issue_open_queries': 15,
            'issue_missing_visits': 30,
            'issue_missing_pages': 20,
            'issue_meddra_uncoded': 10,
            'issue_whodrug_uncoded': 10,
            'issue_lab_issues': 25,
            'issue_sdv_incomplete': 40,
            'issue_signature_gaps': 15,
            'issue_broken_signatures': 10,
            'issue_edrr_issues': 20,
            'issue_inactivated_forms': 10
        }
        return base_effort.get(issue_type, 20) * count
    
    def _get_display_name(self, issue_type: str) -> str:
        """Get human-readable name for issue type"""
        names = {
            'issue_sae_dm_pending': 'SAE DM Reconciliation',
            'issue_sae_safety_pending': 'SAE Safety Review',
            'issue_open_queries': 'Open Queries',
            'issue_missing_visits': 'Missing Visits',
            'issue_missing_pages': 'Missing CRF Pages',
            'issue_meddra_uncoded': 'MedDRA Coding Required',
            'issue_whodrug_uncoded': 'WHODrug Coding Required',
            'issue_lab_issues': 'Lab Issues',
            'issue_sdv_incomplete': 'SDV Incomplete',
            'issue_signature_gaps': 'PI Signatures Needed',
            'issue_broken_signatures': 'Broken Signatures',
            'issue_edrr_issues': 'EDRR Reconciliation',
            'issue_inactivated_forms': 'Inactivated Forms Review'
        }
        return names.get(issue_type, issue_type.replace('issue_', '').replace('_', ' ').title())
    
    def _get_priority(self, issue_type: str) -> str:
        """Determine priority based on issue type"""
        high_priority = {'issue_sae_dm_pending', 'issue_sae_safety_pending', 'issue_signature_gaps', 'issue_open_queries'}
        medium_priority = {'issue_sdv_incomplete', 'issue_missing_visits', 'issue_missing_pages', 'issue_lab_issues'}
        
        if issue_type in high_priority:
            return 'HIGH'
        elif issue_type in medium_priority:
            return 'MEDIUM'
        return 'LOW'
    
    def simulate_fix(self, patient_key: str, issue_type: str) -> FixSimulation:
        """
        Simulate fixing a single issue type for a patient.
        
        Returns:
            FixSimulation with DQI improvement details
        """
        data = self._get_patient_data(patient_key=patient_key)
        current_dqi = self._calculate_current_dqi(data)
        current_count = self._count_issues(data, issue_type)
        
        # Calculate improvement
        component = ISSUE_TO_COMPONENT.get(issue_type, 'completeness_score')
        weight = DQI_WEIGHTS.get(component, 0.10)
        penalty = ISSUE_PENALTIES.get(issue_type, 3.0)
        
        # Improvement = penalty * weight * count (simplified)
        improvement = penalty * weight * max(1, current_count)
        new_dqi = min(100.0, current_dqi + improvement)
        
        return FixSimulation(
            issue_type=issue_type,
            issue_display_name=self._get_display_name(issue_type),
            current_count=max(1, current_count),
            fixed_count=max(1, current_count),
            dqi_improvement=round(improvement, 2),
            new_dqi=round(new_dqi, 2),
            component_affected=component,
            effort_estimate_minutes=self._get_effort_estimate(issue_type, current_count),
            priority=self._get_priority(issue_type)
        )
    
    def simulate_batch_fixes(self, patient_key: str = None, site_id: str = None,
                            issue_types: List[str] = None) -> SimulationResult:
        """
        Simulate fixing multiple issues.
        
        Args:
            patient_key: Patient to simulate
            site_id: Site to simulate (for site-level)
            issue_types: List of issue types to fix
            
        Returns:
            SimulationResult with total improvement
        """
        if patient_key:
            data = self._get_patient_data(patient_key=patient_key)
        elif site_id:
            data = self._get_patient_data(site_id=site_id)
        else:
            data = {}
        
        current_dqi = self._calculate_current_dqi(data)
        
        if not issue_types:
            # Detect all available issues
            issue_types = [k for k in ISSUE_TO_COMPONENT.keys() if self._count_issues(data, k) > 0]
        
        fixes = []
        total_improvement = 0.0
        
        for issue_type in issue_types:
            component = ISSUE_TO_COMPONENT.get(issue_type, 'completeness_score')
            weight = DQI_WEIGHTS.get(component, 0.10)
            penalty = ISSUE_PENALTIES.get(issue_type, 3.0)
            count = max(1, self._count_issues(data, issue_type))
            
            improvement = penalty * weight * count
            total_improvement += improvement
            
            fixes.append(FixSimulation(
                issue_type=issue_type,
                issue_display_name=self._get_display_name(issue_type),
                current_count=count,
                fixed_count=count,
                dqi_improvement=round(improvement, 2),
                new_dqi=0,  # Will be set overall
                component_affected=component,
                effort_estimate_minutes=self._get_effort_estimate(issue_type, count),
                priority=self._get_priority(issue_type)
            ))
        
        projected_dqi = min(100.0, current_dqi + total_improvement)
        improvement_pct = ((projected_dqi - current_dqi) / current_dqi * 100) if current_dqi > 0 else 0
        
        # Update fixes with new DQI (cumulative)
        running_dqi = current_dqi
        for fix in fixes:
            running_dqi = min(100.0, running_dqi + fix.dqi_improvement)
            fix.new_dqi = round(running_dqi, 2)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(fixes)
        
        return SimulationResult(
            patient_key=patient_key,
            site_id=site_id,
            current_dqi=round(current_dqi, 2),
            projected_dqi=round(projected_dqi, 2),
            total_improvement=round(total_improvement, 2),
            fixes_applied=fixes,
            improvement_percentage=round(improvement_pct, 1),
            recommendations=recommendations
        )
    
    def _generate_recommendations(self, fixes: List[FixSimulation]) -> List[str]:
        """Generate actionable recommendations based on fixes"""
        recs = []
        
        # Sort by impact
        sorted_fixes = sorted(fixes, key=lambda f: f.dqi_improvement, reverse=True)
        
        for fix in sorted_fixes[:5]:
            recs.append(
                f"Fix {fix.issue_display_name}: +{fix.dqi_improvement:.1f} DQI points "
                f"(~{fix.effort_estimate_minutes} min effort)"
            )
        
        return recs
    
    def get_fix_recommendations(self, site_id: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Get top N fix recommendations for a site sorted by DQI impact.
        
        Returns:
            List of recommendations with impact details
        """
        # Simulate all possible fixes
        result = self.simulate_batch_fixes(site_id=site_id)
        
        # Sort by DQI improvement
        sorted_fixes = sorted(result.fixes_applied, key=lambda f: f.dqi_improvement, reverse=True)
        
        recommendations = []
        for fix in sorted_fixes[:top_n]:
            recommendations.append({
                'issue_type': fix.issue_type,
                'display_name': fix.issue_display_name,
                'dqi_improvement': fix.dqi_improvement,
                'new_dqi': fix.new_dqi,
                'effort_minutes': fix.effort_estimate_minutes,
                'priority': fix.priority,
                'component': fix.component_affected,
                'impact_description': f"Fix {fix.current_count} {fix.issue_display_name} → +{fix.dqi_improvement:.1f} DQI"
            })
        
        return recommendations
    
    def get_available_issue_types(self) -> List[Dict[str, str]]:
        """Get list of all issue types that can be simulated"""
        return [
            {
                'issue_type': k,
                'display_name': self._get_display_name(k),
                'component': ISSUE_TO_COMPONENT.get(k, 'other'),
                'priority': self._get_priority(k)
            }
            for k in ISSUE_TO_COMPONENT.keys()
        ]
    
    def get_dqi_breakdown(self, current_dqi: float = 75.0) -> Dict[str, Any]:
        """Get DQI component breakdown for visualization"""
        return {
            'current_dqi': current_dqi,
            'components': [
                {'name': 'Safety', 'weight': DQI_WEIGHTS['safety_score'] * 100, 'key': 'safety_score'},
                {'name': 'Queries', 'weight': DQI_WEIGHTS['query_score'] * 100, 'key': 'query_score'},
                {'name': 'Completeness', 'weight': DQI_WEIGHTS['completeness_score'] * 100, 'key': 'completeness_score'},
                {'name': 'Coding', 'weight': DQI_WEIGHTS['coding_score'] * 100, 'key': 'coding_score'},
                {'name': 'Lab', 'weight': DQI_WEIGHTS['lab_score'] * 100, 'key': 'lab_score'},
                {'name': 'SDV', 'weight': DQI_WEIGHTS['sdv_score'] * 100, 'key': 'sdv_score'},
                {'name': 'Signatures', 'weight': DQI_WEIGHTS['signature_score'] * 100, 'key': 'signature_score'},
                {'name': 'EDRR', 'weight': DQI_WEIGHTS['edrr_score'] * 100, 'key': 'edrr_score'},
            ],
            'max_score': 100
        }


# Singleton instance
_simulator: Optional[DQIImprovementSimulator] = None


def get_dqi_simulator() -> DQIImprovementSimulator:
    """Get the global DQI Simulator instance."""
    global _simulator
    if _simulator is None:
        _simulator = DQIImprovementSimulator()
    return _simulator


__all__ = ['DQIImprovementSimulator', 'get_dqi_simulator', 'SimulationResult', 'FixSimulation']
