"""
Causal Hypothesis Engine Service for SANCHALAK AI
Provides a singleton interface to the Causal Hypothesis Engine for dashboard and agents.

Version: 1.0 - Initial Integration
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import pandas as pd

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

# Singleton instance
_engine_instance: Optional['CausalHypothesisService'] = None


class CausalHypothesisService:
    """
    Singleton service for Causal Hypothesis Engine access.
    Provides unified interface for dashboard and agents.
    """
    
    def __init__(self):
        self._engine = None
        self._initialized = False
        self._cached_hypotheses: List[Dict] = []
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl = 300  # 5 minutes
        
        # Data paths
        self.data_dir = PROJECT_ROOT / "data" / "processed"
    
    def initialize(self) -> 'CausalHypothesisService':
        """Initialize the service with the Causal Hypothesis Engine."""
        if self._initialized:
            return self
        
        try:
            from src.knowledge.causal_hypothesis_engine import CausalHypothesisEngine
            self._engine = CausalHypothesisEngine()
            if self._engine.load_data():
                logger.info("Causal Hypothesis Engine initialized successfully")
                self._initialized = True
            else:
                logger.warning("Causal Hypothesis Engine data loading failed")
        except Exception as e:
            logger.warning(f"Could not initialize Causal Hypothesis Engine: {e}")
            self._engine = None
        
        return self
    
    @property
    def is_ready(self) -> bool:
        """Check if service is ready."""
        return self._initialized and self._engine is not None
    
    # =========================================================================
    # HYPOTHESIS GENERATION - For Agents
    # =========================================================================
    
    def generate_patient_hypothesis(
        self, 
        patient_key: str,
        issue_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate causal hypotheses for a specific patient.
        Used by Diagnostic Agent for root cause analysis.
        
        Args:
            patient_key: Patient identifier
            issue_type: Optional filter for specific issue type
        
        Returns:
            Dict with hypotheses and analysis summary
        """
        if not self.is_ready:
            return {"error": "Causal Hypothesis Engine not initialized"}
        
        try:
            # Find patient in data
            if self._engine.patient_data is None:
                return {"error": "Patient data not loaded"}
            
            patient_rows = self._engine.patient_data[
                self._engine.patient_data['patient_key'] == patient_key
            ]
            
            if patient_rows.empty:
                return {"error": f"Patient not found: {patient_key}"}
            
            patient_row = patient_rows.iloc[0]
            
            # Generate hypotheses
            if issue_type:
                hypotheses = []
                for i in range(len(self._engine.templates.get(issue_type, []))):
                    h = self._engine.generate_hypothesis(issue_type, patient_row, i)
                    if h:
                        hypotheses.append(h.to_dict())
            else:
                raw_hypotheses = self._engine.generate_all_hypotheses_for_patient(patient_row)
                hypotheses = [h.to_dict() for h in raw_hypotheses]
            
            return {
                "patient_key": patient_key,
                "hypotheses_count": len(hypotheses),
                "hypotheses": hypotheses[:10],  # Limit to top 10
                "top_hypothesis": hypotheses[0] if hypotheses else None,
                "summary": {
                    "total_hypotheses": len(hypotheses),
                    "high_confidence": len([h for h in hypotheses if h['confidence'] >= 0.7]),
                    "critical_priority": len([h for h in hypotheses if h['priority'] == 'Critical'])
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating patient hypothesis: {e}")
            return {"error": str(e)}
    
    def analyze_issue_root_causes(
        self, 
        issue_type: str,
        sample_size: int = 50
    ) -> Dict[str, Any]:
        """
        Analyze root causes across population for a specific issue type.
        Used by agents for trend analysis.
        
        Args:
            issue_type: Type of issue to analyze
            sample_size: Number of patients to sample
        
        Returns:
            Dict with aggregated root cause analysis
        """
        if not self.is_ready:
            return {"error": "Causal Hypothesis Engine not initialized"}
        
        try:
            if self._engine.patient_data is None:
                return {"error": "Patient data not loaded"}
            
            # Get patients with this issue
            patients_with_issue = self._engine._get_patients_with_issues()
            if issue_type in self._engine.issue_columns:
                col = self._engine.issue_columns[issue_type]
                patients_with_issue = patients_with_issue[patients_with_issue[col] > 0]
            
            if patients_with_issue.empty:
                return {"issue_type": issue_type, "message": "No patients found with this issue"}
            
            # Sample patients
            if len(patients_with_issue) > sample_size:
                patients_to_analyze = patients_with_issue.sample(n=sample_size)
            else:
                patients_to_analyze = patients_with_issue
            
            # Generate hypotheses
            hypotheses = []
            root_cause_counts = {}
            confidence_sum = 0
            
            for _, patient_row in patients_to_analyze.iterrows():
                for i in range(len(self._engine.templates.get(issue_type, []))):
                    h = self._engine.generate_hypothesis(issue_type, patient_row, i)
                    if h:
                        hypotheses.append(h.to_dict())
                        root_cause = h.root_cause
                        root_cause_counts[root_cause] = root_cause_counts.get(root_cause, 0) + 1
                        confidence_sum += h.confidence
            
            # Calculate statistics
            avg_confidence = confidence_sum / len(hypotheses) if hypotheses else 0
            sorted_causes = sorted(root_cause_counts.items(), key=lambda x: x[1], reverse=True)
            
            return {
                "issue_type": issue_type,
                "patients_analyzed": len(patients_to_analyze),
                "hypotheses_generated": len(hypotheses),
                "average_confidence": round(avg_confidence, 3),
                "top_root_causes": [
                    {"root_cause": cause, "count": count, "percentage": round(count/len(hypotheses)*100, 1)}
                    for cause, count in sorted_causes[:5]
                ],
                "recommendations": self._get_aggregated_recommendations(hypotheses[:20])
            }
            
        except Exception as e:
            logger.error(f"Error analyzing issue root causes: {e}")
            return {"error": str(e)}
    
    def _get_aggregated_recommendations(self, hypotheses: List[Dict]) -> List[str]:
        """Aggregate recommendations from multiple hypotheses."""
        recommendation_counts = {}
        for h in hypotheses:
            for rec in h.get('recommendations', []):
                recommendation_counts[rec] = recommendation_counts.get(rec, 0) + 1
        
        sorted_recs = sorted(recommendation_counts.items(), key=lambda x: x[1], reverse=True)
        return [rec for rec, count in sorted_recs[:5]]
    
    def get_site_hypothesis(self, site_id: str) -> Dict[str, Any]:
        """
        Generate aggregated hypotheses for a site.
        Used by Diagnostic Agent for site-level analysis.
        """
        if not self.is_ready:
            return {"error": "Causal Hypothesis Engine not initialized"}
        
        try:
            if self._engine.patient_data is None:
                return {"error": "Patient data not loaded"}
            
            # Get patients at this site
            site_patients = self._engine.patient_data[
                self._engine.patient_data['site_id'] == site_id
            ]
            
            if site_patients.empty:
                return {"error": f"Site not found: {site_id}"}
            
            # Analyze patients with issues
            all_hypotheses = []
            issue_type_summary = {}
            
            for _, patient_row in site_patients.head(50).iterrows():  # Limit for performance
                hypotheses = self._engine.generate_all_hypotheses_for_patient(patient_row)
                for h in hypotheses:
                    h_dict = h.to_dict()
                    all_hypotheses.append(h_dict)
                    issue = h.issue_type
                    if issue not in issue_type_summary:
                        issue_type_summary[issue] = {
                            "count": 0,
                            "total_confidence": 0,
                            "root_causes": {}
                        }
                    issue_type_summary[issue]["count"] += 1
                    issue_type_summary[issue]["total_confidence"] += h.confidence
                    rc = h.root_cause
                    issue_type_summary[issue]["root_causes"][rc] = \
                        issue_type_summary[issue]["root_causes"].get(rc, 0) + 1
            
            # Build summary
            summary = []
            for issue, data in sorted(issue_type_summary.items(), key=lambda x: x[1]["count"], reverse=True):
                avg_conf = data["total_confidence"] / data["count"] if data["count"] > 0 else 0
                top_cause = max(data["root_causes"].items(), key=lambda x: x[1]) if data["root_causes"] else ("Unknown", 0)
                summary.append({
                    "issue_type": issue,
                    "hypothesis_count": data["count"],
                    "avg_confidence": round(avg_conf, 3),
                    "top_root_cause": top_cause[0]
                })
            
            return {
                "site_id": site_id,
                "patients_analyzed": min(len(site_patients), 50),
                "total_hypotheses": len(all_hypotheses),
                "issue_summary": summary[:10],
                "top_hypotheses": sorted(all_hypotheses, key=lambda h: h['confidence'], reverse=True)[:5]
            }
            
        except Exception as e:
            logger.error(f"Error analyzing site: {e}")
            return {"error": str(e)}
    
    # =========================================================================
    # DASHBOARD METHODS
    # =========================================================================
    
    def get_population_analysis(self, sample_size: int = 100) -> Dict[str, Any]:
        """
        Run population-level analysis for dashboard display.
        Caches results to avoid repeated computation.
        """
        if not self.is_ready:
            return {"error": "Causal Hypothesis Engine not initialized"}
        
        # Check cache
        if (self._cached_hypotheses and self._cache_timestamp and 
            (datetime.now() - self._cache_timestamp).seconds < self._cache_ttl):
            return {
                "from_cache": True,
                "hypotheses": self._cached_hypotheses[:50],
                "summary": self._engine.get_summary()
            }
        
        try:
            # Run analysis
            raw_hypotheses = self._engine.analyze_population(sample_size=sample_size)
            self._cached_hypotheses = [h.to_dict() for h in raw_hypotheses]
            self._cache_timestamp = datetime.now()
            
            return {
                "from_cache": False,
                "hypotheses_count": len(self._cached_hypotheses),
                "hypotheses": self._cached_hypotheses[:50],
                "summary": self._engine.get_summary()
            }
            
        except Exception as e:
            logger.error(f"Error in population analysis: {e}")
            return {"error": str(e)}
    
    def get_hypothesis_details(self, hypothesis_id: str) -> Optional[Dict[str, Any]]:
        """Get details for a specific hypothesis from cache."""
        for h in self._cached_hypotheses:
            if h.get('hypothesis_id') == hypothesis_id:
                return h
        return None
    
    def get_root_cause_distribution(self) -> Dict[str, int]:
        """Get distribution of root causes from cached analysis."""
        if not self._cached_hypotheses:
            analysis = self.get_population_analysis()
            if "error" in analysis:
                return {}
        
        distribution = {}
        for h in self._cached_hypotheses:
            cause = h.get('root_cause', 'Unknown')
            distribution[cause] = distribution.get(cause, 0) + 1
        
        return dict(sorted(distribution.items(), key=lambda x: x[1], reverse=True))
    
    def get_confidence_distribution(self) -> Dict[str, int]:
        """Get distribution of confidence levels."""
        if not self._cached_hypotheses:
            analysis = self.get_population_analysis()
            if "error" in analysis:
                return {}
        
        high = len([h for h in self._cached_hypotheses if h.get('confidence', 0) >= 0.7])
        medium = len([h for h in self._cached_hypotheses if 0.4 <= h.get('confidence', 0) < 0.7])
        low = len([h for h in self._cached_hypotheses if h.get('confidence', 0) < 0.4])
        
        return {
            "High (≥70%)": high,
            "Medium (40-70%)": medium,
            "Low (<40%)": low
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get overall engine statistics."""
        if not self.is_ready:
            return {"initialized": False}
        
        stats = {
            "initialized": True,
            "engine_loaded": self._engine is not None,
            "templates_available": len(self._engine.templates) if self._engine else 0,
            "total_templates": sum(len(v) for v in self._engine.templates.values()) if self._engine else 0,
            "cached_hypotheses": len(self._cached_hypotheses),
        }
        
        if self._engine:
            stats.update(self._engine.get_summary())
        
        return stats
    
    def format_hypothesis_report(self, hypothesis: Dict) -> str:
        """Format a hypothesis as a readable report."""
        if not hypothesis:
            return "No hypothesis provided"
        
        lines = [
            f"═══════════════════════════════════════════════════════════",
            f"HYPOTHESIS: {hypothesis.get('hypothesis_id', 'N/A')}",
            f"═══════════════════════════════════════════════════════════",
            f"",
            f"Issue Type: {hypothesis.get('issue_type', 'N/A')}",
            f"Patient: {hypothesis.get('entity_id', 'N/A')}",
            f"Priority: {hypothesis.get('priority', 'N/A')}",
            f"Confidence: {hypothesis.get('confidence', 0)*100:.1f}%",
            f"",
            f"ROOT CAUSE: {hypothesis.get('root_cause', 'N/A')}",
            f"Description: {hypothesis.get('description', 'N/A')}",
            f"",
            f"MECHANISM:",
            f"  {hypothesis.get('mechanism', 'N/A')}",
            f"",
            f"EVIDENCE ({len(hypothesis.get('evidence_chain', {}).get('evidences', []))} items):"
        ]
        
        for e in hypothesis.get('evidence_chain', {}).get('evidences', [])[:5]:
            lines.append(f"  • {e.get('description', 'N/A')}: {e.get('value', 'N/A')}")
        
        lines.extend([
            f"",
            f"RECOMMENDATIONS:"
        ])
        for rec in hypothesis.get('recommendations', []):
            lines.append(f"  ➤ {rec}")
        
        lines.extend([
            f"",
            f"VERIFICATION STEPS:"
        ])
        for step in hypothesis.get('verification_steps', [])[:3]:
            lines.append(f"  ✓ {step}")
        
        lines.append(f"═══════════════════════════════════════════════════════════")
        
        return "\n".join(lines)


def get_causal_hypothesis_service() -> CausalHypothesisService:
    """Get or create the Causal Hypothesis Engine service singleton."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = CausalHypothesisService()
        _engine_instance.initialize()
    return _engine_instance


def reset_causal_hypothesis_service():
    """Reset the singleton for testing purposes."""
    global _engine_instance
    _engine_instance = None
