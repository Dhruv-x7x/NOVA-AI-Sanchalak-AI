# src/agents/diagnostic_enhanced.py
"""
SANCHALAK AI - Enhanced DIAGNOSTIC Agent v1.1

Purpose: Investigate data quality issues, generate hypotheses with evidence chains,
         and provide root cause analysis with confidence scoring.

Features:
- Multi-source data investigation
- Evidence chain builder
- Hypothesis generation with confidence intervals
- Pattern matching integration
- Causal pathway analysis
- Investigation trail logging

v1.1 Changes:
- Fixed pattern_matches column handling
- Added column existence checks
- Improved error handling for missing columns
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import json
import pandas as pd
import numpy as np
from pathlib import Path

# PostgreSQL integration
from src.database.pg_data_service import get_data_service
from src.database.pg_writer import get_pg_writer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InvestigationType(Enum):
    """Types of investigations the diagnostic agent can perform."""
    PATIENT = "patient"
    SITE = "site"
    STUDY = "study"
    ISSUE_TYPE = "issue_type"
    PATTERN = "pattern"
    ANOMALY = "anomaly"
    TREND = "trend"
    COMPARISON = "comparison"


class EvidenceStrength(Enum):
    """Strength of evidence for hypothesis support."""
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    INCONCLUSIVE = "inconclusive"


@dataclass
class Evidence:
    """A piece of evidence supporting or refuting a hypothesis."""
    evidence_id: str
    source: str
    description: str
    strength: EvidenceStrength
    data_points: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    supports_hypothesis: bool = True
    
    def to_dict(self) -> Dict:
        return {
            "evidence_id": self.evidence_id,
            "source": self.source,
            "description": self.description,
            "strength": self.strength.value,
            "data_points": self.data_points,
            "timestamp": self.timestamp.isoformat(),
            "supports_hypothesis": self.supports_hypothesis
        }


@dataclass
class EvidenceChain:
    """A chain of evidence leading to a conclusion."""
    chain_id: str
    hypothesis_id: str
    evidence_list: List[Evidence] = field(default_factory=list)
    causal_pathway: List[str] = field(default_factory=list)
    overall_strength: EvidenceStrength = EvidenceStrength.INCONCLUSIVE
    confidence_score: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    
    def add_evidence(self, evidence: Evidence):
        self.evidence_list.append(evidence)
        self._recalculate_strength()
    
    def _recalculate_strength(self):
        """Recalculate overall strength based on evidence."""
        if not self.evidence_list:
            self.overall_strength = EvidenceStrength.INCONCLUSIVE
            self.confidence_score = 0.0
            return
        
        strength_scores = {
            EvidenceStrength.STRONG: 1.0,
            EvidenceStrength.MODERATE: 0.7,
            EvidenceStrength.WEAK: 0.4,
            EvidenceStrength.INCONCLUSIVE: 0.1
        }
        
        supporting = [e for e in self.evidence_list if e.supports_hypothesis]
        opposing = [e for e in self.evidence_list if not e.supports_hypothesis]
        
        if not supporting:
            self.overall_strength = EvidenceStrength.INCONCLUSIVE
            self.confidence_score = 0.0
            return
        
        support_score = sum(strength_scores[e.strength] for e in supporting)
        oppose_score = sum(strength_scores[e.strength] for e in opposing)
        
        total_evidence = len(self.evidence_list)
        net_score = (support_score - oppose_score) / max(total_evidence, 1)
        
        self.confidence_score = min(max((net_score + 1) / 2, 0.0), 1.0)
        
        margin = 0.15 / np.sqrt(max(total_evidence, 1))
        self.confidence_interval = (
            max(0.0, self.confidence_score - margin),
            min(1.0, self.confidence_score + margin)
        )
        
        if self.confidence_score >= 0.8:
            self.overall_strength = EvidenceStrength.STRONG
        elif self.confidence_score >= 0.6:
            self.overall_strength = EvidenceStrength.MODERATE
        elif self.confidence_score >= 0.4:
            self.overall_strength = EvidenceStrength.WEAK
        else:
            self.overall_strength = EvidenceStrength.INCONCLUSIVE
    
    def to_dict(self) -> Dict:
        return {
            "chain_id": self.chain_id,
            "hypothesis_id": self.hypothesis_id,
            "evidence_count": len(self.evidence_list),
            "evidence_list": [e.to_dict() for e in self.evidence_list],
            "causal_pathway": self.causal_pathway,
            "overall_strength": self.overall_strength.value,
            "confidence_score": round(self.confidence_score, 3),
            "confidence_interval": [round(x, 3) for x in self.confidence_interval]
        }


@dataclass
class DiagnosticHypothesis:
    """A hypothesis generated by the diagnostic agent."""
    hypothesis_id: str
    title: str
    description: str
    root_cause: str
    investigation_type: InvestigationType
    entity_id: str
    evidence_chain: EvidenceChain
    confounders: List[str] = field(default_factory=list)
    verification_steps: List[str] = field(default_factory=list)
    alternative_hypotheses: List[str] = field(default_factory=list)
    actionable: bool = True
    priority: str = "Medium"
    created_at: datetime = field(default_factory=datetime.now)
    
    @property
    def confidence(self) -> float:
        return self.evidence_chain.confidence_score
    
    @property
    def confidence_interval(self) -> Tuple[float, float]:
        return self.evidence_chain.confidence_interval
    
    def to_dict(self) -> Dict:
        return {
            "hypothesis_id": self.hypothesis_id,
            "title": self.title,
            "description": self.description,
            "root_cause": self.root_cause,
            "investigation_type": self.investigation_type.value,
            "entity_id": self.entity_id,
            "evidence_chain": self.evidence_chain.to_dict(),
            "confounders": self.confounders,
            "verification_steps": self.verification_steps,
            "alternative_hypotheses": self.alternative_hypotheses,
            "confidence": round(self.confidence, 3),
            "confidence_interval": [round(x, 3) for x in self.confidence_interval],
            "actionable": self.actionable,
            "priority": self.priority,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class InvestigationResult:
    """Complete result of a diagnostic investigation."""
    investigation_id: str
    query: str
    investigation_type: InvestigationType
    entity_id: str
    hypotheses: List[DiagnosticHypothesis] = field(default_factory=list)
    data_sources_consulted: List[str] = field(default_factory=list)
    patterns_matched: List[Dict] = field(default_factory=list)
    anomalies_detected: List[Dict] = field(default_factory=list)
    investigation_trail: List[Dict] = field(default_factory=list)
    summary: str = ""
    recommendations: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    
    def add_trail_entry(self, step: str, details: Dict):
        self.investigation_trail.append({
            "step": step,
            "details": details,
            "timestamp": datetime.now().isoformat()
        })
    
    def to_dict(self) -> Dict:
        return {
            "investigation_id": self.investigation_id,
            "query": self.query,
            "investigation_type": self.investigation_type.value,
            "entity_id": self.entity_id,
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "hypothesis_count": len(self.hypotheses),
            "data_sources_consulted": self.data_sources_consulted,
            "patterns_matched": self.patterns_matched,
            "anomalies_detected": self.anomalies_detected,
            "investigation_trail": self.investigation_trail,
            "summary": self.summary,
            "recommendations": self.recommendations,
            "duration_seconds": round(self.duration_seconds, 2),
            "created_at": self.created_at.isoformat()
        }


class DiagnosticDataLoader:
    """Loads and caches data for the diagnostic agent."""
    
    def __init__(self, base_path: str = "data/processed"):
        self.base_path = Path(base_path)
        self._cache: Dict[str, pd.DataFrame] = {}
        self._json_cache: Dict[str, Any] = {}
        self._column_cache: Dict[str, List[str]] = {}
    
    def _load_parquet(self, name: str, path: str) -> Optional[pd.DataFrame]:
        """Load a parquet file with caching."""
        if name in self._cache:
            return self._cache[name]
        
        full_path = self.base_path / path
        if full_path.exists():
            try:
                df = pd.read_parquet(full_path)
                self._cache[name] = df
                self._column_cache[name] = df.columns.tolist()
                logger.info(f"Loaded {name}: {len(df)} rows")
                return df
            except Exception as e:
                logger.error(f"Error loading {name}: {e}")
                return None
        else:
            logger.warning(f"File not found: {full_path}")
            return None
    
    def _load_json(self, name: str, path: str) -> Optional[Any]:
        """Load a JSON file with caching."""
        if name in self._json_cache:
            return self._json_cache[name]
        
        full_path = self.base_path / path
        if full_path.exists():
            try:
                with open(full_path, 'r') as f:
                    data = json.load(f)
                self._json_cache[name] = data
                logger.info(f"Loaded {name}")
                return data
            except Exception as e:
                logger.error(f"Error loading {name}: {e}")
                return None
        else:
            logger.warning(f"File not found: {full_path}")
            return None
    
    def get_columns(self, name: str) -> List[str]:
        """Get column names for a loaded dataframe."""
        return self._column_cache.get(name, [])
    
    def _safe_filter(self, df: pd.DataFrame, column: str, value: Any) -> pd.DataFrame:
        """Safely filter a dataframe, returning empty df if column doesn't exist."""
        if column in df.columns:
            return df[df[column] == value]
        else:
            logger.warning(f"Column '{column}' not found in dataframe. Available: {df.columns.tolist()[:10]}...")
            return pd.DataFrame()
    
    @property
    def patient_issues(self) -> Optional[pd.DataFrame]:
        return self._load_parquet("patient_issues", "analytics/patient_issues.parquet")
    
    @property
    def patient_cascade(self) -> Optional[pd.DataFrame]:
        return self._load_parquet("patient_cascade", "analytics/patient_cascade_analysis.parquet")
    
    @property
    def site_benchmarks(self) -> Optional[pd.DataFrame]:
        return self._load_parquet("site_benchmarks", "analytics/site_benchmarks.parquet")
    
    @property
    def patient_anomalies(self) -> Optional[pd.DataFrame]:
        return self._load_parquet("patient_anomalies", "analytics/patient_anomalies.parquet")
    
    @property
    def pattern_matches(self) -> Optional[pd.DataFrame]:
        return self._load_parquet("pattern_matches", "analytics/pattern_library/pattern_matches.parquet")
    
    @property
    def causal_hypotheses(self) -> Optional[List[Dict]]:
        return self._load_json("causal_hypotheses", "analytics/causal_hypotheses/causal_hypotheses.json")
    
    @property
    def upr(self) -> Optional[pd.DataFrame]:
        return self._load_parquet("upr", "upr/unified_patient_record.parquet")
    
    def get_patient_data(self, patient_key: str) -> Dict[str, Any]:
        """Get all data for a specific patient."""
        result = {}
        
        # Patient issues
        if self.patient_issues is not None:
            patient_issues = self._safe_filter(self.patient_issues, 'patient_key', patient_key)
            if not patient_issues.empty:
                result['issues'] = patient_issues.iloc[0].to_dict()
        
        # Cascade analysis
        if self.patient_cascade is not None:
            cascade = self._safe_filter(self.patient_cascade, 'patient_key', patient_key)
            if not cascade.empty:
                result['cascade'] = cascade.iloc[0].to_dict()
        
        # Anomalies
        if self.patient_anomalies is not None:
            anomalies = self._safe_filter(self.patient_anomalies, 'patient_key', patient_key)
            if not anomalies.empty:
                result['anomalies'] = anomalies.iloc[0].to_dict()
        
        # UPR base data
        if self.upr is not None:
            upr_data = self._safe_filter(self.upr, 'patient_key', patient_key)
            if not upr_data.empty:
                result['upr'] = upr_data.iloc[0].to_dict()
        
        return result
    
    def get_site_data(self, site_id: str) -> Dict[str, Any]:
        """Get all data for a specific site."""
        result = {}
        
        # Site benchmarks
        if self.site_benchmarks is not None:
            site_bench = self._safe_filter(self.site_benchmarks, 'site_id', site_id)
            if not site_bench.empty:
                result['benchmarks'] = site_bench.iloc[0].to_dict()
        
        # Aggregate patient issues for site
        if self.patient_issues is not None:
            site_issues = self._safe_filter(self.patient_issues, 'site_id', site_id)
            if not site_issues.empty:
                result['patient_count'] = len(site_issues)
                
                # Check for required columns
                has_any_issue_col = 'has_any_issue' if 'has_any_issue' in site_issues.columns else None
                total_issues_col = 'total_issues' if 'total_issues' in site_issues.columns else None
                priority_col = 'priority_tier' if 'priority_tier' in site_issues.columns else None
                
                result['issues_summary'] = {
                    'total_patients_with_issues': int(site_issues[has_any_issue_col].sum()) if has_any_issue_col else 0,
                    'total_issues': int(site_issues[total_issues_col].sum()) if total_issues_col else 0,
                    'avg_issues_per_patient': float(site_issues[total_issues_col].mean()) if total_issues_col else 0.0,
                    'priority_distribution': site_issues[priority_col].value_counts().to_dict() if priority_col else {}
                }
                
                # Count by issue type
                issue_cols = [c for c in site_issues.columns if c.startswith('issue_') and c != 'has_any_issue']
                issue_counts = {}
                for col in issue_cols:
                    issue_name = col.replace('issue_', '')
                    issue_counts[issue_name] = int(site_issues[col].sum())
                result['issue_type_counts'] = issue_counts
        
        # Pattern matches for site - check available columns first
        if self.pattern_matches is not None:
            pm = self.pattern_matches
            pm_columns = pm.columns.tolist()
            
            # Try different possible column names for site identifier
            site_col = None
            for possible_col in ['site_id', 'entity_id', 'site', 'location_id']:
                if possible_col in pm_columns:
                    site_col = possible_col
                    break
            
            if site_col:
                site_patterns = self._safe_filter(pm, site_col, site_id)
                if not site_patterns.empty:
                    result['patterns'] = site_patterns.to_dict('records')
            else:
                # If no site column, check if patterns can be matched via patient_key
                logger.debug(f"Pattern matches columns: {pm_columns}")
                result['patterns'] = []
        
        return result
    
    def get_study_data(self, study_id: str) -> Dict[str, Any]:
        """Get aggregated data for a study."""
        result = {}
        
        if self.patient_issues is not None:
            study_issues = self._safe_filter(self.patient_issues, 'study_id', study_id)
            if not study_issues.empty:
                result['patient_count'] = len(study_issues)
                result['site_count'] = study_issues['site_id'].nunique() if 'site_id' in study_issues.columns else 0
                
                has_any_issue_col = 'has_any_issue' if 'has_any_issue' in study_issues.columns else None
                total_issues_col = 'total_issues' if 'total_issues' in study_issues.columns else None
                priority_col = 'priority_tier' if 'priority_tier' in study_issues.columns else None
                
                result['issues_summary'] = {
                    'total_patients_with_issues': int(study_issues[has_any_issue_col].sum()) if has_any_issue_col else 0,
                    'total_issues': int(study_issues[total_issues_col].sum()) if total_issues_col else 0,
                    'avg_issues_per_patient': float(study_issues[total_issues_col].mean()) if total_issues_col else 0.0,
                    'priority_distribution': study_issues[priority_col].value_counts().to_dict() if priority_col else {}
                }
        
        return result
    
    def get_pattern_matches_columns(self) -> List[str]:
        """Get columns available in pattern_matches for debugging."""
        if self.pattern_matches is not None:
            return self.pattern_matches.columns.tolist()
        return []


class EnhancedDiagnosticAgent:
    """
    Enhanced DIAGNOSTIC Agent for root cause analysis and hypothesis generation.
    """
    
    ROOT_CAUSE_TEMPLATES = {
        'sdv_incomplete': [
            {
                'cause': 'CRA Resource Constraint',
                'description': 'Insufficient CRA time allocated for SDV activities',
                'confounders': ['site visit frequency', 'travel restrictions', 'CRA workload'],
                'verification': ['Check CRA assignment', 'Review visit logs', 'Compare with peer sites']
            },
            {
                'cause': 'Source Document Unavailability',
                'description': 'Source documents not accessible during monitoring visits',
                'confounders': ['site organization', 'document storage', 'staff availability'],
                'verification': ['Check document availability logs', 'Review site feedback']
            },
            {
                'cause': 'High Patient Volume',
                'description': 'Site enrollment exceeded SDV capacity',
                'confounders': ['enrollment rate', 'study phase', 'protocol complexity'],
                'verification': ['Compare enrollment vs SDV rates', 'Check SDV backlog trend']
            }
        ],
        'open_queries': [
            {
                'cause': 'Site Response Delay',
                'description': 'Site not responding to queries in a timely manner',
                'confounders': ['site workload', 'query complexity', 'staff turnover'],
                'verification': ['Check query age distribution', 'Review site response times']
            },
            {
                'cause': 'Query Complexity',
                'description': 'Queries require additional clarification or medical review',
                'confounders': ['query type', 'protocol requirements', 'data complexity'],
                'verification': ['Analyze query types', 'Check escalation patterns']
            },
            {
                'cause': 'Data Entry Errors',
                'description': 'Systematic data entry issues creating repeated queries',
                'confounders': ['training adequacy', 'form design', 'workload'],
                'verification': ['Pattern analysis of query types', 'Check training records']
            }
        ],
        'signature_gaps': [
            {
                'cause': 'PI Absence',
                'description': 'Principal Investigator unavailable for signatures',
                'confounders': ['PI schedule', 'vacation', 'conference attendance'],
                'verification': ['Check PI availability', 'Review signature patterns']
            },
            {
                'cause': 'Batch Signing Pattern',
                'description': 'PI accumulates forms for batch signing sessions',
                'confounders': ['PI workflow', 'site processes', 'reminder systems'],
                'verification': ['Analyze signature timing patterns', 'Check batch sizes']
            }
        ],
        'sae_dm_pending': [
            {
                'cause': 'Cross-System Mismatch',
                'description': 'Discrepancies between EDC and safety database entries',
                'confounders': ['data entry timing', 'system sync', 'manual processes'],
                'verification': ['Compare system records', 'Check reconciliation status']
            },
            {
                'cause': 'Data Manager Workload',
                'description': 'DM team overloaded with reconciliation tasks',
                'confounders': ['team size', 'SAE volume', 'competing priorities'],
                'verification': ['Review DM task queue', 'Check resolution times']
            }
        ],
        'missing_visits': [
            {
                'cause': 'Patient Non-Compliance',
                'description': 'Patients missing scheduled visits',
                'confounders': ['visit schedule', 'patient engagement', 'travel distance'],
                'verification': ['Check patient visit history', 'Review compliance patterns']
            },
            {
                'cause': 'Scheduling Issues',
                'description': 'Site scheduling challenges or capacity constraints',
                'confounders': ['site capacity', 'staff availability', 'equipment access'],
                'verification': ['Review scheduling logs', 'Check site capacity']
            }
        ],
        'broken_signatures': [
            {
                'cause': 'Post-Signature Data Changes',
                'description': 'Data modified after PI signature, breaking the signature',
                'confounders': ['query resolution', 'data corrections', 'audit trail'],
                'verification': ['Review audit trail', 'Check modification patterns']
            }
        ],
        'sae_safety_pending': [
            {
                'cause': 'Medical Review Backlog',
                'description': 'Safety physician review queue overwhelmed',
                'confounders': ['case complexity', 'reviewer availability', 'case volume'],
                'verification': ['Check review queue', 'Analyze case complexity']
            }
        ],
        'inactivated_forms': [
            {
                'cause': 'Protocol Amendment',
                'description': 'Forms inactivated due to protocol changes',
                'confounders': ['amendment timing', 'form relevance', 'migration issues'],
                'verification': ['Check amendment history', 'Review inactivation reasons']
            }
        ],
        'lab_issues': [
            {
                'cause': 'Missing Lab Ranges',
                'description': 'Laboratory reference ranges not configured',
                'confounders': ['lab setup', 'data migration', 'lab changes'],
                'verification': ['Check lab configuration', 'Review setup status']
            }
        ],
        'edrr_issues': [
            {
                'cause': 'Third-Party Data Mismatch',
                'description': 'Discrepancies with external data sources',
                'confounders': ['data timing', 'format differences', 'ID matching'],
                'verification': ['Compare source records', 'Check reconciliation rules']
            }
        ],
        'meddra_uncoded': [
            {
                'cause': 'Coding Backlog',
                'description': 'Medical coder workload exceeds capacity',
                'confounders': ['term volume', 'coder availability', 'term complexity'],
                'verification': ['Check coding queue', 'Review coder assignment']
            },
            {
                'cause': 'Ambiguous Terms',
                'description': 'Verbatim terms requiring manual review or clarification',
                'confounders': ['term quality', 'source language', 'medical complexity'],
                'verification': ['Analyze uncoded terms', 'Check query status']
            }
        ],
        'whodrug_uncoded': [
            {
                'cause': 'Non-Standard Drug Names',
                'description': 'Drug names not matching WHODrug dictionary',
                'confounders': ['local drug names', 'spelling variations', 'combination drugs'],
                'verification': ['Review uncoded terms', 'Check dictionary version']
            }
        ],
        'high_query_volume': [
            {
                'cause': 'Training Deficiency',
                'description': 'Site staff inadequately trained on data entry',
                'confounders': ['staff turnover', 'training recency', 'protocol complexity'],
                'verification': ['Review training records', 'Check error patterns']
            },
            {
                'cause': 'Protocol Complexity',
                'description': 'Complex protocol requirements causing data entry errors',
                'confounders': ['form design', 'edit checks', 'guidance clarity'],
                'verification': ['Analyze query types', 'Review protocol sections']
            }
        ]
    }
    
    def __init__(self, llm_wrapper=None):
        """Initialize the diagnostic agent."""
        self.data_loader = DiagnosticDataLoader()
        self.llm = llm_wrapper
        self._hypothesis_counter = 0
        self._evidence_counter = 0
        self._chain_counter = 0
        self._investigation_counter = 0
        
        logger.info("EnhancedDiagnosticAgent initialized")
    
    def _generate_id(self, prefix: str) -> str:
        """Generate a unique ID."""
        if prefix == "HYP":
            self._hypothesis_counter += 1
            return f"HYP-{self._hypothesis_counter:04d}"
        elif prefix == "EVD":
            self._evidence_counter += 1
            return f"EVD-{self._evidence_counter:04d}"
        elif prefix == "CHN":
            self._chain_counter += 1
            return f"CHN-{self._chain_counter:04d}"
        elif prefix == "INV":
            self._investigation_counter += 1
            return f"INV-{self._investigation_counter:04d}"
        return f"{prefix}-{datetime.now().strftime('%H%M%S')}"
    
    def _determine_investigation_type(self, query: str, entity_id: str = None) -> Tuple[InvestigationType, str]:
        """Determine the type of investigation based on query and entity."""
        query_lower = query.lower()
        
        if entity_id:
            if entity_id.startswith("Study_"):
                return InvestigationType.STUDY, entity_id
            elif entity_id.startswith("Site_"):
                return InvestigationType.SITE, entity_id
            elif "|" in entity_id:
                return InvestigationType.PATIENT, entity_id
        
        if any(word in query_lower for word in ['patient', 'subject']):
            return InvestigationType.PATIENT, "unknown"
        elif any(word in query_lower for word in ['site', 'center']):
            return InvestigationType.SITE, "unknown"
        elif any(word in query_lower for word in ['study', 'trial', 'protocol']):
            return InvestigationType.STUDY, "unknown"
        elif any(word in query_lower for word in ['pattern', 'recurring', 'systematic']):
            return InvestigationType.PATTERN, "unknown"
        elif any(word in query_lower for word in ['anomaly', 'unusual', 'outlier']):
            return InvestigationType.ANOMALY, "unknown"
        elif any(word in query_lower for word in ['trend', 'over time', 'increasing', 'decreasing']):
            return InvestigationType.TREND, "unknown"
        elif any(word in query_lower for word in ['compare', 'versus', 'difference']):
            return InvestigationType.COMPARISON, "unknown"
        
        issue_keywords = {
            'sdv': 'sdv_incomplete',
            'query': 'open_queries',
            'queries': 'open_queries',
            'signature': 'signature_gaps',
            'sae': 'sae_dm_pending',
            'safety': 'sae_safety_pending',
            'missing visit': 'missing_visits',
            'missing page': 'missing_pages',
            'lab': 'lab_issues',
            'coding': 'meddra_uncoded',
            'meddra': 'meddra_uncoded',
            'whodrug': 'whodrug_uncoded'
        }
        
        for keyword, issue_type in issue_keywords.items():
            if keyword in query_lower:
                return InvestigationType.ISSUE_TYPE, issue_type
        
        return InvestigationType.SITE, "unknown"
    
    def _extract_entity_from_query(self, query: str) -> Optional[str]:
        """Extract entity ID from query text."""
        import re



        
        study_match = re.search(r'Study[_\s]?(\d+)', query, re.IGNORECASE)
        if study_match:
            return f"Study_{study_match.group(1)}"
        
        site_match = re.search(r'Site[_\s]?(\d+)', query, re.IGNORECASE)
        if site_match:
            return f"Site_{site_match.group(1)}"
        
        patient_match = re.search(r'(Study_\d+\|Site_\d+\|Subject_\d+)', query)
        if patient_match:
            return patient_match.group(1)
        
        return None
    
    def _build_evidence_from_data(self, data: Dict, source: str) -> List[Evidence]:
        """Build evidence objects from data."""
        evidence_list = []
        
        if source == "patient_issues" and 'issues' in data:
            issues = data['issues']
            
            issue_cols = [k for k in issues.keys() if k.startswith('issue_') and k != 'has_any_issue']
            for col in issue_cols:
                if issues.get(col, 0) == 1:
                    issue_type = col.replace('issue_', '')
                    count_col = f'count_{issue_type}'
                    severity_col = f'severity_{issue_type}'
                    
                    evidence = Evidence(
                        evidence_id=self._generate_id("EVD"),
                        source="patient_issues",
                        description=f"Patient has {issue_type} issue",
                        strength=EvidenceStrength.STRONG if issues.get(count_col, 0) > 5 else EvidenceStrength.MODERATE,
                        data_points={
                            'issue_type': issue_type,
                            'count': issues.get(count_col, 1),
                            'severity': issues.get(severity_col, 0),
                            'priority_tier': issues.get('priority_tier', 'Unknown')
                        },
                        supports_hypothesis=True
                    )
                    evidence_list.append(evidence)
        
        elif source == "site_data" and 'issues_summary' in data:
            summary = data['issues_summary']
            
            if summary.get('avg_issues_per_patient', 0) > 2:
                evidence = Evidence(
                    evidence_id=self._generate_id("EVD"),
                    source="site_analysis",
                    description=f"Site has high average issues per patient: {summary.get('avg_issues_per_patient', 0):.2f}",
                    strength=EvidenceStrength.STRONG,
                    data_points=summary,
                    supports_hypothesis=True
                )
                evidence_list.append(evidence)
            
            if 'issue_type_counts' in data:
                for issue_type, count in data.get('issue_type_counts', {}).items():
                    if count > 10:
                        evidence = Evidence(
                            evidence_id=self._generate_id("EVD"),
                            source="site_analysis",
                            description=f"Site has {count} patients with {issue_type}",
                            strength=EvidenceStrength.MODERATE,
                            data_points={'issue_type': issue_type, 'count': count},
                            supports_hypothesis=True
                        )
                        evidence_list.append(evidence)
        
        elif source == "patterns" and 'patterns' in data:
            for pattern in data.get('patterns', []):
                evidence = Evidence(
                    evidence_id=self._generate_id("EVD"),
                    source="pattern_library",
                    description=f"Pattern matched: {pattern.get('pattern_id', 'Unknown')}",
                    strength=EvidenceStrength.STRONG,
                    data_points=pattern,
                    supports_hypothesis=True
                )
                evidence_list.append(evidence)
        
        elif source == "benchmarks" and 'benchmarks' in data:
            bench = data['benchmarks']
            
            if bench.get('performance_tier', '') in ['Needs Improvement', 'Below Average']:
                evidence = Evidence(
                    evidence_id=self._generate_id("EVD"),
                    source="benchmarks",
                    description=f"Site performance tier: {bench.get('performance_tier')}",
                    strength=EvidenceStrength.MODERATE,
                    data_points={
                        'composite_score': bench.get('composite_score'),
                        'percentile': bench.get('percentile'),
                        'tier': bench.get('performance_tier')
                    },
                    supports_hypothesis=True
                )
                evidence_list.append(evidence)
        
        return evidence_list
    
    def _generate_hypotheses_for_issue(
        self, 
        issue_type: str, 
        entity_id: str,
        evidence_list: List[Evidence],
        context: Dict
    ) -> List[DiagnosticHypothesis]:
        """Generate hypotheses for a specific issue type."""
        hypotheses = []
        
        templates = self.ROOT_CAUSE_TEMPLATES.get(issue_type, [])
        
        for template in templates:
            chain = EvidenceChain(
                chain_id=self._generate_id("CHN"),
                hypothesis_id="",
                causal_pathway=[template['cause'], issue_type, "Data Quality Impact"]
            )
            
            for evidence in evidence_list:
                if evidence.data_points.get('issue_type') == issue_type:
                    chain.add_evidence(evidence)
            
            if context:
                context_evidence = Evidence(
                    evidence_id=self._generate_id("EVD"),
                    source="context_analysis",
                    description=f"Contextual factors supporting {template['cause']}",
                    strength=EvidenceStrength.WEAK,
                    data_points=context,
                    supports_hypothesis=True
                )
                chain.add_evidence(context_evidence)
            
            hyp_id = self._generate_id("HYP")
            chain.hypothesis_id = hyp_id
            
            priority = "Medium"
            if issue_type in ['sae_dm_pending', 'sae_safety_pending']:
                priority = "Critical"
            elif issue_type in ['open_queries', 'signature_gaps'] and chain.confidence_score > 0.7:
                priority = "High"
            elif chain.confidence_score < 0.4:
                priority = "Low"
            
            hypothesis = DiagnosticHypothesis(
                hypothesis_id=hyp_id,
                title=template['cause'],
                description=template['description'],
                root_cause=template['cause'],
                investigation_type=InvestigationType.ISSUE_TYPE,
                entity_id=entity_id,
                evidence_chain=chain,
                confounders=template.get('confounders', []),
                verification_steps=template.get('verification', []),
                alternative_hypotheses=[t['cause'] for t in templates if t['cause'] != template['cause']],
                actionable=True,
                priority=priority
            )
            
            hypotheses.append(hypothesis)
        
        return hypotheses
    
    def investigate_patient(self, patient_key: str, query: str = "") -> InvestigationResult:
        """Investigate issues for a specific patient."""
        start_time = datetime.now()
        
        investigation = InvestigationResult(
            investigation_id=self._generate_id("INV"),
            query=query or f"Investigate patient {patient_key}",
            investigation_type=InvestigationType.PATIENT,
            entity_id=patient_key
        )
        
        investigation.add_trail_entry("data_collection", {"source": "patient_data", "patient_key": patient_key})
        patient_data = self.data_loader.get_patient_data(patient_key)
        
        if not patient_data:
            investigation.summary = f"No data found for patient {patient_key}"
            return investigation
        
        investigation.data_sources_consulted = list(patient_data.keys())
        
        evidence_list = self._build_evidence_from_data(patient_data, "patient_issues")
        investigation.add_trail_entry("evidence_building", {"evidence_count": len(evidence_list)})
        
        if 'issues' in patient_data:
            issues = patient_data['issues']
            issue_types = [
                k.replace('issue_', '') 
                for k in issues.keys() 
                if k.startswith('issue_') and k != 'has_any_issue' and issues.get(k) == 1
            ]
            
            for issue_type in issue_types:
                hyps = self._generate_hypotheses_for_issue(
                    issue_type, patient_key, evidence_list, patient_data.get('cascade', {})
                )
                investigation.hypotheses.extend(hyps)
        
        investigation.add_trail_entry("hypothesis_generation", {"hypothesis_count": len(investigation.hypotheses)})
        
        investigation.summary = self._generate_patient_summary(patient_key, patient_data, investigation.hypotheses)
        investigation.recommendations = self._generate_recommendations(investigation.hypotheses)
        investigation.duration_seconds = (datetime.now() - start_time).total_seconds()
        
        return investigation
    
    def investigate_site(self, site_id: str, query: str = "") -> InvestigationResult:
        """Investigate issues for a specific site."""
        start_time = datetime.now()
        
        investigation = InvestigationResult(
            investigation_id=self._generate_id("INV"),
            query=query or f"Investigate site {site_id}",
            investigation_type=InvestigationType.SITE,
            entity_id=site_id
        )
        
        investigation.add_trail_entry("data_collection", {"source": "site_data", "site_id": site_id})
        site_data = self.data_loader.get_site_data(site_id)
        
        if not site_data:
            investigation.summary = f"No data found for site {site_id}"
            investigation.duration_seconds = (datetime.now() - start_time).total_seconds()
            return investigation
        
        investigation.data_sources_consulted = list(site_data.keys())
        
        evidence_list = []
        evidence_list.extend(self._build_evidence_from_data(site_data, "site_data"))
        evidence_list.extend(self._build_evidence_from_data(site_data, "patterns"))
        evidence_list.extend(self._build_evidence_from_data(site_data, "benchmarks"))
        
        investigation.add_trail_entry("evidence_building", {"evidence_count": len(evidence_list)})
        
        if 'patterns' in site_data:
            investigation.patterns_matched = site_data['patterns']
        
        if 'issue_type_counts' in site_data:
            for issue_type, count in site_data['issue_type_counts'].items():
                if count > 0:
                    hyps = self._generate_hypotheses_for_issue(
                        issue_type, site_id, evidence_list, 
                        {'patient_count': site_data.get('patient_count', 0), 'issue_count': count}
                    )
                    investigation.hypotheses.extend(hyps)
        
        investigation.add_trail_entry("hypothesis_generation", {"hypothesis_count": len(investigation.hypotheses)})
        
        investigation.summary = self._generate_site_summary(site_id, site_data, investigation.hypotheses)
        investigation.recommendations = self._generate_recommendations(investigation.hypotheses)
        investigation.duration_seconds = (datetime.now() - start_time).total_seconds()
        
        return investigation
    
    def investigate_query(self, query: str) -> InvestigationResult:
        """Main entry point: investigate based on natural language query."""
        start_time = datetime.now()
        
        entity_id = self._extract_entity_from_query(query)
        inv_type, determined_entity = self._determine_investigation_type(query, entity_id)
        
        if entity_id:
            determined_entity = entity_id
        
        investigation = InvestigationResult(
            investigation_id=self._generate_id("INV"),
            query=query,
            investigation_type=inv_type,
            entity_id=determined_entity
        )
        
        investigation.add_trail_entry("query_analysis", {
            "investigation_type": inv_type.value,
            "entity_id": determined_entity,
            "extracted_entity": entity_id
        })
        
        if inv_type == InvestigationType.PATIENT and determined_entity != "unknown":
            result = self.investigate_patient(determined_entity, query)
            investigation.hypotheses = result.hypotheses
            investigation.data_sources_consulted = result.data_sources_consulted
            investigation.patterns_matched = result.patterns_matched
            investigation.investigation_trail.extend(result.investigation_trail)
            investigation.summary = result.summary
            investigation.recommendations = result.recommendations
        
        elif inv_type == InvestigationType.SITE and determined_entity != "unknown":
            result = self.investigate_site(determined_entity, query)
            investigation.hypotheses = result.hypotheses
            investigation.data_sources_consulted = result.data_sources_consulted
            investigation.patterns_matched = result.patterns_matched
            investigation.investigation_trail.extend(result.investigation_trail)
            investigation.summary = result.summary
            investigation.recommendations = result.recommendations
        
        elif inv_type == InvestigationType.ISSUE_TYPE:
            result = self._investigate_issue_type(determined_entity, query)
            investigation.hypotheses = result.hypotheses
            investigation.data_sources_consulted = result.data_sources_consulted
            investigation.summary = result.summary
            investigation.recommendations = result.recommendations
        
        else:
            investigation.summary = self._generate_general_summary(query)
            investigation.recommendations = [
                "Specify a site (e.g., 'Site_101') for detailed investigation",
                "Specify an issue type (e.g., 'open queries') for focused analysis",
                "Use 'Why does Site_X have issues?' for root cause analysis"
            ]
        
        investigation.duration_seconds = (datetime.now() - start_time).total_seconds()
        
        return investigation
    
    def _investigate_issue_type(self, issue_type: str, query: str) -> InvestigationResult:
        """Investigate a specific issue type across all entities."""
        investigation = InvestigationResult(
            investigation_id=self._generate_id("INV"),
            query=query,
            investigation_type=InvestigationType.ISSUE_TYPE,
            entity_id=issue_type
        )
        
        if self.data_loader.patient_issues is not None:
            df = self.data_loader.patient_issues
            issue_col = f'issue_{issue_type}'
            
            if issue_col in df.columns:
                site_counts = df.groupby('site_id')[issue_col].sum().sort_values(ascending=False)
                top_sites = site_counts.head(10)
                
                investigation.data_sources_consulted.append("patient_issues")
                
                for site_id, count in top_sites.items():
                    if count > 0:
                        evidence = Evidence(
                            evidence_id=self._generate_id("EVD"),
                            source="patient_issues",
                            description=f"Site {site_id} has {int(count)} patients with {issue_type}",
                            strength=EvidenceStrength.STRONG if count > 20 else EvidenceStrength.MODERATE,
                            data_points={'site_id': site_id, 'count': int(count), 'issue_type': issue_type},
                            supports_hypothesis=True
                        )
                        
                        hyps = self._generate_hypotheses_for_issue(
                            issue_type, site_id, [evidence], {'issue_count': int(count)}
                        )
                        investigation.hypotheses.extend(hyps[:1])
                
                total_affected = df[issue_col].sum()
                investigation.summary = (
                    f"Issue Type: {issue_type}\n"
                    f"Total Affected Patients: {int(total_affected)}\n"
                    f"Top Sites: {', '.join([f'{s} ({int(c)})' for s, c in top_sites.head(5).items()])}\n"
                    f"Hypotheses Generated: {len(investigation.hypotheses)}"
                )
        
        investigation.recommendations = self._generate_recommendations(investigation.hypotheses)
        
        return investigation
    
    def _generate_patient_summary(
        self, patient_key: str, data: Dict, hypotheses: List[DiagnosticHypothesis]
    ) -> str:
        """Generate a summary for patient investigation."""
        issues = data.get('issues', {})
        issue_count = issues.get('total_issues', 0)
        priority = issues.get('priority_tier', 'Unknown')
        
        issue_types = [
            k.replace('issue_', '') 
            for k in issues.keys() 
            if k.startswith('issue_') and k != 'has_any_issue' and issues.get(k) == 1
        ]
        
        top_hypotheses = sorted(hypotheses, key=lambda h: h.confidence, reverse=True)[:3]
        
        summary = f"""
PATIENT INVESTIGATION SUMMARY
=============================
Patient: {patient_key}
Total Issues: {issue_count}
Priority: {priority}
Issue Types: {', '.join(issue_types) if issue_types else 'None'}

TOP HYPOTHESES:
"""
        for i, hyp in enumerate(top_hypotheses, 1):
            summary += f"""
{i}. {hyp.title}
   Confidence: {hyp.confidence:.1%} (CI: {hyp.confidence_interval[0]:.1%} - {hyp.confidence_interval[1]:.1%})
   Root Cause: {hyp.description}
   Priority: {hyp.priority}
"""
        
        return summary.strip()
    
    def _generate_site_summary(
        self, site_id: str, data: Dict, hypotheses: List[DiagnosticHypothesis]
    ) -> str:
        """Generate a summary for site investigation."""
        patient_count = data.get('patient_count', 0)
        issues_summary = data.get('issues_summary', {})
        
        top_hypotheses = sorted(hypotheses, key=lambda h: h.confidence, reverse=True)[:5]
        
        issue_counts = data.get('issue_type_counts', {})
        top_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        summary = f"""
SITE INVESTIGATION SUMMARY
==========================
Site: {site_id}
Total Patients: {patient_count}
Patients with Issues: {issues_summary.get('total_patients_with_issues', 0)}
Total Issues: {issues_summary.get('total_issues', 0)}
Avg Issues/Patient: {issues_summary.get('avg_issues_per_patient', 0):.2f}

TOP ISSUE TYPES:
"""
        for issue, count in top_issues:
            summary += f"  - {issue}: {count} patients\n"
        
        summary += "\nTOP ROOT CAUSE HYPOTHESES:\n"
        for i, hyp in enumerate(top_hypotheses, 1):
            summary += f"""
{i}. {hyp.title}
   Issue: {hyp.entity_id}
   Confidence: {hyp.confidence:.1%}
   Description: {hyp.description}
"""
        
        return summary.strip()
    
    def _generate_general_summary(self, query: str) -> str:
        """Generate a general summary when no specific entity is identified."""
        summary = f"""
GENERAL INVESTIGATION
=====================
Query: {query}

Unable to identify a specific entity (patient, site, or study) from the query.

SUGGESTIONS:
1. Specify a site: "Why does Site_101 have so many open queries?"
2. Specify a patient: "Investigate patient Study_21|Site_200|Subject_1001"
3. Specify an issue type: "What's causing the SDV backlog?"
4. Ask about trends: "Why are queries increasing across APAC region?"

AVAILABLE DATA SOURCES:
- Patient Issues (57,997 patients)
- Site Benchmarks (3,416 sites)
- Pattern Library (13 patterns)
- Causal Hypotheses (23 templates)
"""
        return summary.strip()
    
    def _generate_recommendations(self, hypotheses: List[DiagnosticHypothesis]) -> List[str]:
        """Generate recommendations based on hypotheses."""
        recommendations = []
        
        sorted_hyps = sorted(
            hypotheses, 
            key=lambda h: (
                {'Critical': 4, 'High': 3, 'Medium': 2, 'Low': 1}.get(h.priority, 0),
                h.confidence
            ),
            reverse=True
        )
        
        seen_causes = set()
        for hyp in sorted_hyps[:5]:
            if hyp.root_cause not in seen_causes:
                seen_causes.add(hyp.root_cause)
                
                if "Resource" in hyp.root_cause or "Workload" in hyp.root_cause:
                    recommendations.append(
                        f"Review {hyp.root_cause}: Consider resource reallocation or additional support"
                    )
                elif "Training" in hyp.root_cause:
                    recommendations.append(
                        f"Address {hyp.root_cause}: Schedule refresher training for site staff"
                    )
                elif "Process" in hyp.root_cause or "Delay" in hyp.root_cause:
                    recommendations.append(
                        f"Optimize {hyp.root_cause}: Review and streamline current processes"
                    )
                elif "PI" in hyp.root_cause or "Investigator" in hyp.root_cause:
                    recommendations.append(
                        f"Address {hyp.root_cause}: Engage with PI to establish regular review schedule"
                    )
                else:
                    recommendations.append(
                        f"Investigate {hyp.root_cause}: {hyp.verification_steps[0] if hyp.verification_steps else 'Review underlying data'}"
                    )
        
        if not recommendations:
            recommendations.append("No actionable recommendations at this time - more data needed")
        
        return recommendations
    
    def process(self, query: str, context: Dict = None) -> Dict:
        """Main processing method for integration with orchestrator."""
        result = self.investigate_query(query)
        
        return {
            "investigation_id": result.investigation_id,
            "query": result.query,
            "investigation_type": result.investigation_type.value,
            "entity_id": result.entity_id,
            "hypotheses": [h.to_dict() for h in result.hypotheses],
            "hypothesis_count": len(result.hypotheses),
            "data_sources": result.data_sources_consulted,
            "patterns_matched": result.patterns_matched,
            "summary": result.summary,
            "recommendations": result.recommendations,
            "investigation_trail": result.investigation_trail,
            "duration_seconds": result.duration_seconds
        }


def get_diagnostic_agent(llm_wrapper=None) -> EnhancedDiagnosticAgent:
    """Factory function to get diagnostic agent instance."""
    return EnhancedDiagnosticAgent(llm_wrapper=llm_wrapper)