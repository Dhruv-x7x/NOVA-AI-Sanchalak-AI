"""
Cross-Study Pattern Matcher Service for SANCHALAK AI
Provides a singleton interface to the Pattern Matcher for dashboard and agents.

Version: 1.0 - Initial Integration
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

# Singleton instance
_matcher_instance: Optional['CrossStudyPatternService'] = None


class CrossStudyPatternService:
    """
    Singleton service for Cross-Study Pattern Matcher access.
    Provides unified interface for dashboard and agents.
    """
    
    def __init__(self):
        self._matcher = None
        self._initialized = False
        self._patterns_loaded = False
        self._matches_computed = False
        
        # Data paths
        self.data_dir = PROJECT_ROOT / "data" / "processed"
    
    def initialize(self) -> 'CrossStudyPatternService':
        """Initialize the service with the Cross-Study Pattern Matcher."""
        if self._initialized:
            return self
        
        try:
            from src.knowledge.cross_study_pattern_matcher import CrossStudyPatternMatcher
            self._matcher = CrossStudyPatternMatcher(data_dir=str(self.data_dir))
            
            # Load patterns
            pattern_count = self._matcher.load_existing_patterns()
            if pattern_count > 0:
                self._patterns_loaded = True
                logger.info(f"Pattern Matcher loaded {pattern_count} patterns")
            
            self._initialized = True
            logger.info("Cross-Study Pattern Matcher Service initialized")
            
        except Exception as e:
            logger.warning(f"Could not initialize Pattern Matcher: {e}")
            self._matcher = None
        
        return self
    
    @property
    def is_ready(self) -> bool:
        """Check if service is ready."""
        return self._initialized and self._matcher is not None
    
    def ensure_matches_computed(self):
        """Ensure cross-study matches are computed."""
        if not self._matches_computed and self.is_ready and self._patterns_loaded:
            try:
                # Generate embeddings if sentence_transformers available
                self._matcher.generate_pattern_embeddings()
                # Find matches
                self._matcher.find_cross_study_matches(similarity_threshold=0.7)
                self._matches_computed = True
            except Exception as e:
                logger.warning(f"Could not compute matches: {e}")
    
    # =========================================================================
    # AGENT TOOLS - Pattern Discovery
    # =========================================================================
    
    def find_similar_patterns(
        self, 
        study_id: str,
        pattern_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Find patterns in other studies similar to patterns in the given study.
        Used by agents for cross-study insights.
        """
        if not self.is_ready:
            return {"error": "Pattern Matcher not initialized"}
        
        self.ensure_matches_computed()
        
        try:
            # Get patterns for this study
            if study_id not in self._matcher.study_patterns:
                return {"error": f"Study not found: {study_id}"}
            
            study_patterns = self._matcher.study_patterns[study_id]
            
            # Filter by type if specified
            if pattern_type:
                study_patterns = [p for p in study_patterns if p.pattern_type == pattern_type]
            
            if not study_patterns:
                return {"study_id": study_id, "message": "No patterns found"}
            
            # Find matches where this study is source or target
            matches = []
            for m in self._matcher.pattern_matches:
                if m.source_study_id == study_id or m.target_study_id == study_id:
                    matches.append({
                        "source_study": m.source_study_id,
                        "target_study": m.target_study_id,
                        "source_pattern": m.source_pattern_id,
                        "target_pattern": m.target_pattern_id,
                        "similarity": round(m.similarity_score, 3),
                        "match_type": m.match_type,
                        "transferable": m.transferable,
                        "confidence": round(m.confidence, 3)
                    })
            
            # Sort by similarity
            matches.sort(key=lambda x: x['similarity'], reverse=True)
            
            return {
                "study_id": study_id,
                "patterns_in_study": len(study_patterns),
                "cross_study_matches": len(matches),
                "matches": matches[:10],
                "pattern_types": list(set(p.pattern_type for p in study_patterns))
            }
            
        except Exception as e:
            logger.error(f"Error finding similar patterns: {e}")
            return {"error": str(e)}
    
    def get_transfer_recommendations(
        self, 
        study_id: Optional[str] = None,
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Get recommendations for transferring successful patterns.
        Used by agents for resolution recommendations.
        """
        if not self.is_ready:
            return {"error": "Pattern Matcher not initialized"}
        
        self.ensure_matches_computed()
        
        try:
            recommendations = self._matcher.generate_transfer_recommendations()
            
            # Filter by study if specified
            if study_id:
                recommendations = [r for r in recommendations if r['target_study'] == study_id]
            
            # Sort by priority
            priority_order = {'Critical': 0, 'High': 1, 'Medium': 2, 'Low': 3}
            recommendations.sort(key=lambda x: (priority_order.get(x['priority'], 4), -x['confidence']))
            
            return {
                "total_recommendations": len(recommendations),
                "study_filter": study_id,
                "recommendations": recommendations[:limit]
            }
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return {"error": str(e)}
    
    def get_pattern_details(self, pattern_id: str) -> Dict[str, Any]:
        """Get details for a specific pattern."""
        if not self.is_ready:
            return {"error": "Pattern Matcher not initialized"}
        
        try:
            # Find pattern
            for study_id, patterns in self._matcher.study_patterns.items():
                for p in patterns:
                    if p.pattern_id == pattern_id:
                        return {
                            "pattern_id": p.pattern_id,
                            "study_id": p.study_id,
                            "pattern_type": p.pattern_type,
                            "description": p.description,
                            "issue_types": p.issue_types,
                            "affected_sites": p.affected_sites,
                            "affected_patients": p.affected_patients,
                            "severity": p.severity,
                            "confidence": round(p.confidence, 3),
                            "resolution_rate": round(p.resolution_rate, 3),
                            "features": p.features
                        }
            
            return {"error": f"Pattern not found: {pattern_id}"}
            
        except Exception as e:
            logger.error(f"Error getting pattern details: {e}")
            return {"error": str(e)}
    
    def search_patterns_by_issue(
        self, 
        issue_type: str,
        min_confidence: float = 0.5
    ) -> Dict[str, Any]:
        """
        Search for patterns related to a specific issue type.
        Used by agents for root cause investigation.
        """
        if not self.is_ready:
            return {"error": "Pattern Matcher not initialized"}
        
        try:
            matching_patterns = []
            
            for study_id, patterns in self._matcher.study_patterns.items():
                for p in patterns:
                    if issue_type in p.issue_types and p.confidence >= min_confidence:
                        matching_patterns.append({
                            "pattern_id": p.pattern_id,
                            "study_id": p.study_id,
                            "pattern_type": p.pattern_type,
                            "description": p.description,
                            "affected_patients": p.affected_patients,
                            "severity": p.severity,
                            "confidence": round(p.confidence, 3)
                        })
            
            # Sort by confidence
            matching_patterns.sort(key=lambda x: x['confidence'], reverse=True)
            
            return {
                "issue_type": issue_type,
                "patterns_found": len(matching_patterns),
                "patterns": matching_patterns[:20]
            }
            
        except Exception as e:
            logger.error(f"Error searching patterns: {e}")
            return {"error": str(e)}
    
    # =========================================================================
    # DASHBOARD METHODS
    # =========================================================================
    
    def get_cross_study_report(self) -> Dict[str, Any]:
        """Get comprehensive cross-study report for dashboard."""
        if not self.is_ready:
            return {"error": "Pattern Matcher not initialized"}
        
        self.ensure_matches_computed()
        
        try:
            report = self._matcher.generate_cross_study_report()
            return report
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return {"error": str(e)}
    
    def get_study_pattern_summary(self, study_id: str) -> Dict[str, Any]:
        """Get pattern summary for a specific study."""
        if not self.is_ready:
            return {"error": "Pattern Matcher not initialized"}
        
        try:
            return self._matcher.get_study_pattern_summary(study_id)
        except Exception as e:
            logger.error(f"Error getting study summary: {e}")
            return {"error": str(e)}
    
    def get_pattern_type_definitions(self) -> Dict[str, Any]:
        """Get all pattern type definitions."""
        if not self.is_ready:
            return {"error": "Pattern Matcher not initialized"}
        
        return {
            "pattern_types": self._matcher.pattern_types,
            "count": len(self._matcher.pattern_types)
        }
    
    def validate_pattern(
        self,
        pattern_id: str,
        status: str,
        validated_by: str,
        effectiveness_score: float,
        notes: str = ""
    ) -> Dict[str, Any]:
        """Validate a pattern and record effectiveness."""
        if not self.is_ready:
            return {"error": "Pattern Matcher not initialized"}
        
        try:
            validation = self._matcher.validate_pattern(
                pattern_id=pattern_id,
                status=status,
                validated_by=validated_by,
                effectiveness_score=effectiveness_score,
                notes=notes
            )
            
            return {
                "status": "validated",
                "pattern_id": validation.pattern_id,
                "validation_status": validation.validation_status,
                "effectiveness_score": validation.effectiveness_score
            }
        except Exception as e:
            logger.error(f"Error validating pattern: {e}")
            return {"error": str(e)}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get overall pattern matcher statistics."""
        if not self.is_ready:
            return {"initialized": False}
        
        all_patterns = []
        for patterns in self._matcher.study_patterns.values():
            all_patterns.extend(patterns)
        
        return {
            "initialized": True,
            "patterns_loaded": self._patterns_loaded,
            "matches_computed": self._matches_computed,
            "total_studies": len(self._matcher.study_patterns),
            "total_patterns": len(all_patterns),
            "total_matches": len(self._matcher.pattern_matches),
            "total_validations": len(self._matcher.validations),
            "pattern_types_defined": len(self._matcher.pattern_types),
            "transferable_matches": len([m for m in self._matcher.pattern_matches if m.transferable])
        }
    
    def get_all_studies(self) -> List[str]:
        """Get list of all studies with patterns."""
        if not self.is_ready:
            return []
        return list(self._matcher.study_patterns.keys())
    
    def get_match_distribution(self) -> Dict[str, int]:
        """Get distribution of match types."""
        if not self.is_ready or not self._matches_computed:
            return {}
        
        distribution = {}
        for m in self._matcher.pattern_matches:
            distribution[m.match_type] = distribution.get(m.match_type, 0) + 1
        
        return distribution


def get_cross_study_pattern_service() -> CrossStudyPatternService:
    """Get or create the Cross-Study Pattern Matcher service singleton."""
    global _matcher_instance
    if _matcher_instance is None:
        _matcher_instance = CrossStudyPatternService()
        _matcher_instance.initialize()
    return _matcher_instance


def reset_cross_study_pattern_service():
    """Reset the singleton for testing purposes."""
    global _matcher_instance
    _matcher_instance = None
