"""
SANCHALAK AI - Phase 3.6: Pattern Library v1.2 (FIXED)
=======================================================

FIXES:
1. Load from correct source file (patient_clean_status.parquet or patient_dqi_enhanced.parquet)
2. Merge with issue data if needed
3. Use correct column names from actual data
4. Relaxed detection thresholds
5. Better logging for debugging

Author: Sanchalak AI Team
Version: 1.2
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
import logging
import warnings

# PostgreSQL data access
try:
    from src.database.pg_data_service import get_data_service
    from src.database.pg_writer import get_pg_writer
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False


warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# HELPER FUNCTION FOR JSON SERIALIZATION
# =============================================================================

def json_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [json_serializable(item) for item in obj]
    return obj


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class PatternCategory(Enum):
    SITE_PERFORMANCE = "site_performance"
    RESOURCE_CAPACITY = "resource_capacity"
    DATA_QUALITY = "data_quality"
    SAFETY = "safety"
    TIMELINE = "timeline"
    COMPLIANCE = "compliance"


class PatternSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertAction(Enum):
    ESCALATE = "escalate"
    INVESTIGATE = "investigate"
    MONITOR = "monitor"
    NOTIFY = "notify"
    AUTO_RESOLVE = "auto_resolve"


@dataclass
class PatternDefinition:
    pattern_id: str
    name: str
    category: PatternCategory
    description: str
    detection_rules: Dict[str, Any]
    severity: PatternSeverity
    recommended_action: AlertAction
    responsible_role: str
    sla_days: int
    resolution_template: str
    cross_study_validated: bool = False
    validation_count: int = 0
    success_rate: float = 0.0
    avg_resolution_days: float = 0.0


@dataclass
class PatternMatch:
    match_id: str
    pattern_id: str
    entity_type: str
    entity_id: str
    study_id: str
    confidence: float
    evidence: Dict[str, Any]
    detected_at: str
    severity: PatternSeverity
    recommended_action: AlertAction
    responsible_role: str
    sla_deadline: str
    status: str = "open"


@dataclass
class Alert:
    alert_id: str
    pattern_id: str
    pattern_name: str
    entity_type: str
    entity_id: str
    study_id: str
    severity: PatternSeverity
    title: str
    message: str
    evidence_summary: str
    recommended_action: AlertAction
    responsible_role: str
    sla_deadline: str
    created_at: str
    status: str = "new"


# =============================================================================
# PATTERN LIBRARY CLASS
# =============================================================================

class PatternLibrary:
    """
    Pattern Library Engine v1.2 - Fixed column mappings
    """
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path("data/processed")
        self.output_dir = self.data_dir / "analytics" / "pattern_library"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.patterns: Dict[str, PatternDefinition] = {}
        self.matches: List[PatternMatch] = []
        self.alerts: List[Alert] = []
        
        # Column mapping for flexibility
        self.column_map = {}
        
        self._initialize_patterns()
        logger.info(f"Pattern Library initialized with {len(self.patterns)} patterns")
    
    def _initialize_patterns(self):
        """Initialize all known patterns"""
        
        # SITE PERFORMANCE
        self._add_pattern(PatternDefinition(
            pattern_id="PAT-SP-001",
            name="Underperforming Site",
            category=PatternCategory.SITE_PERFORMANCE,
            description="Site with high issue count compared to peers",
            detection_rules={"type": "composite"},
            severity=PatternSeverity.HIGH,
            recommended_action=AlertAction.ESCALATE,
            responsible_role="CTM",
            sla_days=7,
            resolution_template="Review site training needs and resource allocation."
        ))
        
        self._add_pattern(PatternDefinition(
            pattern_id="PAT-SP-002",
            name="High Issue Density Site",
            category=PatternCategory.SITE_PERFORMANCE,
            description="Site with above-average issues per patient",
            detection_rules={"type": "threshold"},
            severity=PatternSeverity.MEDIUM,
            recommended_action=AlertAction.INVESTIGATE,
            responsible_role="CRA",
            sla_days=14,
            resolution_template="Analyze issue patterns and provide targeted support."
        ))
        
        # RESOURCE CAPACITY
        self._add_pattern(PatternDefinition(
            pattern_id="PAT-RC-001",
            name="Coordinator Overload",
            category=PatternCategory.RESOURCE_CAPACITY,
            description="Site managing more patients than optimal threshold (>25)",
            detection_rules={"type": "threshold", "value": 25},
            severity=PatternSeverity.HIGH,
            recommended_action=AlertAction.ESCALATE,
            responsible_role="CTM",
            sla_days=7,
            resolution_template="Evaluate need for additional coordinator support."
        ))
        
        self._add_pattern(PatternDefinition(
            pattern_id="PAT-RC-002",
            name="Signature Backlog",
            category=PatternCategory.RESOURCE_CAPACITY,
            description="Site with high signature gap count indicating PI availability issue",
            detection_rules={"type": "threshold"},
            severity=PatternSeverity.HIGH,
            recommended_action=AlertAction.ESCALATE,
            responsible_role="CRA",
            sla_days=3,
            resolution_template="Contact site to arrange PI signature session."
        ))
        
        # DATA QUALITY
        self._add_pattern(PatternDefinition(
            pattern_id="PAT-DQ-001",
            name="Query Overload",
            category=PatternCategory.DATA_QUALITY,
            description="Site with excessive open queries per patient",
            detection_rules={"type": "threshold"},
            severity=PatternSeverity.HIGH,
            recommended_action=AlertAction.ESCALATE,
            responsible_role="Data Manager",
            sla_days=5,
            resolution_template="Schedule focused query resolution session with site."
        ))
        
        self._add_pattern(PatternDefinition(
            pattern_id="PAT-DQ-002",
            name="SDV Backlog",
            category=PatternCategory.DATA_QUALITY,
            description="Site with high SDV incomplete count",
            detection_rules={"type": "threshold"},
            severity=PatternSeverity.MEDIUM,
            recommended_action=AlertAction.INVESTIGATE,
            responsible_role="CRA",
            sla_days=14,
            resolution_template="Review SDV strategy and prioritization."
        ))
        
        self._add_pattern(PatternDefinition(
            pattern_id="PAT-DQ-003",
            name="Missing Data Pattern",
            category=PatternCategory.DATA_QUALITY,
            description="Site with missing visits or pages",
            detection_rules={"type": "threshold"},
            severity=PatternSeverity.MEDIUM,
            recommended_action=AlertAction.INVESTIGATE,
            responsible_role="CRA",
            sla_days=7,
            resolution_template="Investigate missing data root cause."
        ))
        
        self._add_pattern(PatternDefinition(
            pattern_id="PAT-DQ-004",
            name="Coding Backlog",
            category=PatternCategory.DATA_QUALITY,
            description="Site with uncoded MedDRA or WHODrug terms",
            detection_rules={"type": "threshold"},
            severity=PatternSeverity.LOW,
            recommended_action=AlertAction.NOTIFY,
            responsible_role="Medical Coder",
            sla_days=14,
            resolution_template="Prioritize coding queue for this site."
        ))
        
        # SAFETY
        self._add_pattern(PatternDefinition(
            pattern_id="PAT-SF-001",
            name="SAE Pending - DM",
            category=PatternCategory.SAFETY,
            description="Site with pending SAE data management issues",
            detection_rules={"type": "threshold"},
            severity=PatternSeverity.CRITICAL,
            recommended_action=AlertAction.ESCALATE,
            responsible_role="Safety Data Manager",
            sla_days=1,
            resolution_template="URGENT: Reconcile SAE discrepancies immediately."
        ))
        
        self._add_pattern(PatternDefinition(
            pattern_id="PAT-SF-002",
            name="SAE Pending - Safety",
            category=PatternCategory.SAFETY,
            description="Site with pending safety review items",
            detection_rules={"type": "threshold"},
            severity=PatternSeverity.CRITICAL,
            recommended_action=AlertAction.ESCALATE,
            responsible_role="Safety Physician",
            sla_days=1,
            resolution_template="URGENT: Complete SAE safety assessment."
        ))
        
        # TIMELINE
        self._add_pattern(PatternDefinition(
            pattern_id="PAT-TL-001",
            name="High Priority Site",
            category=PatternCategory.TIMELINE,
            description="Site with critical priority patients requiring immediate attention",
            detection_rules={"type": "threshold"},
            severity=PatternSeverity.HIGH,
            recommended_action=AlertAction.ESCALATE,
            responsible_role="Study Lead",
            sla_days=3,
            resolution_template="Prioritize resources for critical patients."
        ))
        
        # COMPLIANCE
        self._add_pattern(PatternDefinition(
            pattern_id="PAT-CP-001",
            name="Form Inactivation Issues",
            category=PatternCategory.COMPLIANCE,
            description="Site with high number of inactivated forms",
            detection_rules={"type": "threshold"},
            severity=PatternSeverity.MEDIUM,
            recommended_action=AlertAction.INVESTIGATE,
            responsible_role="Data Manager",
            sla_days=7,
            resolution_template="Review inactivation reasons and document properly."
        ))
        
        self._add_pattern(PatternDefinition(
            pattern_id="PAT-CP-002",
            name="EDRR Reconciliation Issues",
            category=PatternCategory.COMPLIANCE,
            description="Site with third-party data reconciliation issues",
            detection_rules={"type": "threshold"},
            severity=PatternSeverity.MEDIUM,
            recommended_action=AlertAction.INVESTIGATE,
            responsible_role="Data Manager",
            sla_days=7,
            resolution_template="Investigate EDRR discrepancies with vendor."
        ))
        
        logger.info(f"Initialized {len(self.patterns)} patterns across {len(PatternCategory)} categories")
    
    def _add_pattern(self, pattern: PatternDefinition):
        self.patterns[pattern.pattern_id] = pattern
    
    def load_data(self) -> pd.DataFrame:
        """Load and merge data from multiple sources"""
        logger.info("Loading data from multiple sources...")
        
        # Primary source - patient issues (has issue counts)
        issues_path = self.data_dir / "analytics" / "patient_issues.parquet"
        if issues_path.exists():
            df = pd.read_parquet(issues_path)
            logger.info(f"Loaded patient_issues: {len(df)} rows, {len(df.columns)} cols")
        else:
            raise FileNotFoundError(f"Required file not found: {issues_path}")
        
        # Log available columns
        logger.info(f"Available columns: {list(df.columns)}")
        
        return df
    
    def _aggregate_to_site(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate patient data to site level using ACTUAL columns"""
        logger.info("Aggregating patient data to site level...")
        
        # Define aggregation based on ACTUAL columns in patient_issues.parquet
        agg_dict = {
            'patient_key': 'count',
            'total_issues': 'sum',
            'total_severity': 'sum',
            'priority_score': 'mean',
        }
        
        # Add count columns that exist
        count_cols = [col for col in df.columns if col.startswith('count_')]
        for col in count_cols:
            agg_dict[col] = 'sum'
        
        # Check for priority tier
        if 'priority_tier' in df.columns:
            # Count critical patients
            df['is_critical'] = df['priority_tier'] == 'critical'
            df['is_high'] = df['priority_tier'] == 'high'
            agg_dict['is_critical'] = 'sum'
            agg_dict['is_high'] = 'sum'
        
        # Perform aggregation
        site_df = df.groupby(['study_id', 'site_id']).agg(agg_dict).reset_index()
        site_df = site_df.rename(columns={'patient_key': 'patient_count'})
        
        # Calculate derived metrics
        site_df['issues_per_patient'] = site_df['total_issues'] / site_df['patient_count'].clip(lower=1)
        site_df['avg_severity'] = site_df['total_severity'] / site_df['patient_count'].clip(lower=1)
        
        # Calculate per-patient rates for each issue type
        for col in count_cols:
            issue_name = col.replace('count_', '')
            site_df[f'{issue_name}_per_patient'] = site_df[col] / site_df['patient_count'].clip(lower=1)
        
        logger.info(f"Aggregated to {len(site_df)} sites with {len(site_df.columns)} columns")
        logger.info(f"Site columns: {list(site_df.columns)}")
        
        return site_df
    
    def _aggregate_to_study(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate patient data to study level"""
        logger.info("Aggregating patient data to study level...")
        
        agg_dict = {
            'patient_key': 'count',
            'total_issues': 'sum',
        }
        
        # Add count columns
        count_cols = [col for col in df.columns if col.startswith('count_')]
        for col in count_cols:
            agg_dict[col] = 'sum'
        
        if 'is_critical' in df.columns:
            agg_dict['is_critical'] = 'sum'
        
        study_df = df.groupby('study_id').agg(agg_dict).reset_index()
        study_df = study_df.rename(columns={'patient_key': 'patient_count'})
        
        # Calculate rates
        study_df['issues_per_patient'] = study_df['total_issues'] / study_df['patient_count'].clip(lower=1)
        
        logger.info(f"Aggregated to {len(study_df)} studies")
        return study_df
    
    def detect_patterns(self, df: pd.DataFrame) -> Tuple[List[PatternMatch], List[Alert]]:
        """Detect patterns in the data"""
        logger.info("Starting pattern detection...")
        
        self.matches = []
        self.alerts = []
        
        # Aggregate data
        site_df = self._aggregate_to_site(df)
        study_df = self._aggregate_to_study(df)
        
        # Calculate study-level benchmarks for comparison
        study_benchmarks = {}
        for _, study in study_df.iterrows():
            study_id = study['study_id']
            study_benchmarks[study_id] = {
                'issues_per_patient': float(study['issues_per_patient']),
                'patient_count': int(study['patient_count'])
            }
        
        # Detect site-level patterns
        self._detect_site_patterns(site_df, study_benchmarks)
        
        # Detect study-level patterns
        self._detect_study_patterns(study_df)
        
        # Generate alerts
        self._generate_alerts()
        
        logger.info(f"Detected {len(self.matches)} pattern matches, generated {len(self.alerts)} alerts")
        
        return self.matches, self.alerts
    
    def _detect_site_patterns(self, site_df: pd.DataFrame, study_benchmarks: Dict):
        """Detect all site-level patterns"""
        logger.info("Detecting site-level patterns...")
        
        detection_counts = {pid: 0 for pid in self.patterns.keys()}
        
        for _, site in site_df.iterrows():
            site_id = str(site['site_id'])
            study_id = str(site['study_id'])
            patient_count = int(site['patient_count'])
            
            # Get study benchmark for comparison
            study_bench = study_benchmarks.get(study_id, {'issues_per_patient': 0})
            study_avg_issues = study_bench.get('issues_per_patient', 0)
            
            # PAT-RC-001: Coordinator Overload (>25 patients)
            if patient_count > 25:
                self._add_match(
                    self.patterns['PAT-RC-001'], 'site', site_id, study_id,
                    confidence=min(0.95, 0.6 + (patient_count - 25) * 0.01),
                    evidence={'patient_count': patient_count, 'threshold': 25}
                )
                detection_counts['PAT-RC-001'] += 1
            
            # PAT-SP-001: Underperforming Site (high issues vs study avg)
            issues_pp = float(site.get('issues_per_patient', 0))
            if issues_pp > study_avg_issues * 1.5 and issues_pp > 2:
                self._add_match(
                    self.patterns['PAT-SP-001'], 'site', site_id, study_id,
                    confidence=min(0.9, 0.5 + (issues_pp / max(study_avg_issues, 0.1) - 1.5) * 0.2),
                    evidence={
                        'issues_per_patient': round(issues_pp, 2),
                        'study_average': round(study_avg_issues, 2),
                        'ratio': f"{issues_pp / max(study_avg_issues, 0.1):.1f}x"
                    }
                )
                detection_counts['PAT-SP-001'] += 1
            
            # PAT-SP-002: High Issue Density (>3 issues per patient)
            if issues_pp > 3 and patient_count >= 3:
                self._add_match(
                    self.patterns['PAT-SP-002'], 'site', site_id, study_id,
                    confidence=min(0.85, 0.5 + issues_pp * 0.1),
                    evidence={'issues_per_patient': round(issues_pp, 2), 'patient_count': patient_count}
                )
                detection_counts['PAT-SP-002'] += 1
            
            # PAT-RC-002: Signature Backlog
            sig_gaps = int(site.get('count_signature_gaps', 0))
            broken_sigs = int(site.get('count_broken_signatures', 0))
            total_sig_issues = sig_gaps + broken_sigs
            if total_sig_issues > 10:
                self._add_match(
                    self.patterns['PAT-RC-002'], 'site', site_id, study_id,
                    confidence=min(0.9, 0.5 + total_sig_issues * 0.02),
                    evidence={'signature_gaps': sig_gaps, 'broken_signatures': broken_sigs}
                )
                detection_counts['PAT-RC-002'] += 1
            
            # PAT-DQ-001: Query Overload
            open_queries = int(site.get('count_open_queries', 0))
            queries_pp = open_queries / max(patient_count, 1)
            if queries_pp > 2 or open_queries > 20:
                self._add_match(
                    self.patterns['PAT-DQ-001'], 'site', site_id, study_id,
                    confidence=min(0.85, 0.5 + queries_pp * 0.15),
                    evidence={'open_queries': open_queries, 'per_patient': round(queries_pp, 2)}
                )
                detection_counts['PAT-DQ-001'] += 1
            
            # PAT-DQ-002: SDV Backlog
            sdv_incomplete = int(site.get('count_sdv_incomplete', 0))
            if sdv_incomplete > 20:
                self._add_match(
                    self.patterns['PAT-DQ-002'], 'site', site_id, study_id,
                    confidence=min(0.85, 0.5 + sdv_incomplete * 0.01),
                    evidence={'sdv_incomplete': sdv_incomplete}
                )
                detection_counts['PAT-DQ-002'] += 1
            
            # PAT-DQ-003: Missing Data Pattern
            missing_visits = int(site.get('count_missing_visits', 0))
            missing_pages = int(site.get('count_missing_pages', 0))
            total_missing = missing_visits + missing_pages
            if total_missing > 5:
                self._add_match(
                    self.patterns['PAT-DQ-003'], 'site', site_id, study_id,
                    confidence=min(0.8, 0.5 + total_missing * 0.03),
                    evidence={'missing_visits': missing_visits, 'missing_pages': missing_pages}
                )
                detection_counts['PAT-DQ-003'] += 1
            
            # PAT-DQ-004: Coding Backlog
            meddra_uncoded = int(site.get('count_meddra_uncoded', 0))
            whodrug_uncoded = int(site.get('count_whodrug_uncoded', 0))
            total_uncoded = meddra_uncoded + whodrug_uncoded
            if total_uncoded > 5:
                self._add_match(
                    self.patterns['PAT-DQ-004'], 'site', site_id, study_id,
                    confidence=min(0.8, 0.5 + total_uncoded * 0.05),
                    evidence={'meddra_uncoded': meddra_uncoded, 'whodrug_uncoded': whodrug_uncoded}
                )
                detection_counts['PAT-DQ-004'] += 1
            
            # PAT-SF-001: SAE Pending - DM
            sae_dm_pending = int(site.get('count_sae_dm_pending', 0))
            if sae_dm_pending > 0:
                self._add_match(
                    self.patterns['PAT-SF-001'], 'site', site_id, study_id,
                    confidence=min(0.95, 0.7 + sae_dm_pending * 0.1),
                    evidence={'sae_dm_pending': sae_dm_pending}
                )
                detection_counts['PAT-SF-001'] += 1
            
            # PAT-SF-002: SAE Pending - Safety
            sae_safety_pending = int(site.get('count_sae_safety_pending', 0))
            if sae_safety_pending > 0:
                self._add_match(
                    self.patterns['PAT-SF-002'], 'site', site_id, study_id,
                    confidence=min(0.95, 0.7 + sae_safety_pending * 0.1),
                    evidence={'sae_safety_pending': sae_safety_pending}
                )
                detection_counts['PAT-SF-002'] += 1
            
            # PAT-TL-001: High Priority Site (has critical patients)
            critical_count = int(site.get('is_critical', 0))
            high_count = int(site.get('is_high', 0))
            if critical_count > 0 or high_count > 3:
                self._add_match(
                    self.patterns['PAT-TL-001'], 'site', site_id, study_id,
                    confidence=min(0.9, 0.6 + critical_count * 0.1 + high_count * 0.05),
                    evidence={'critical_patients': critical_count, 'high_priority_patients': high_count}
                )
                detection_counts['PAT-TL-001'] += 1
            
            # PAT-CP-001: Form Inactivation Issues
            inactivated = int(site.get('count_inactivated_forms', 0))
            if inactivated > 10:
                self._add_match(
                    self.patterns['PAT-CP-001'], 'site', site_id, study_id,
                    confidence=min(0.8, 0.5 + inactivated * 0.02),
                    evidence={'inactivated_forms': inactivated}
                )
                detection_counts['PAT-CP-001'] += 1
            
            # PAT-CP-002: EDRR Issues
            edrr_issues = int(site.get('count_edrr_issues', 0))
            if edrr_issues > 0:
                self._add_match(
                    self.patterns['PAT-CP-002'], 'site', site_id, study_id,
                    confidence=min(0.85, 0.6 + edrr_issues * 0.1),
                    evidence={'edrr_issues': edrr_issues}
                )
                detection_counts['PAT-CP-002'] += 1
        
        # Log detection counts
        logger.info("Pattern detection counts:")
        for pid, count in detection_counts.items():
            if count > 0:
                pattern = self.patterns[pid]
                logger.info(f"  {pid} ({pattern.name}): {count} matches")
    
    def _detect_study_patterns(self, study_df: pd.DataFrame):
        """Detect study-level patterns"""
        logger.info("Detecting study-level patterns...")
        # Study-level patterns can be added here
        pass
    
    def _add_match(self, pattern: PatternDefinition, entity_type: str,
                   entity_id: str, study_id: str, confidence: float,
                   evidence: Dict[str, Any]):
        """Add a pattern match"""
        match_id = f"M-{pattern.pattern_id}-{entity_id}-{len(self.matches):04d}"
        detected_at = datetime.now().isoformat()
        sla_deadline = (datetime.now() + timedelta(days=pattern.sla_days)).isoformat()
        
        match = PatternMatch(
            match_id=match_id,
            pattern_id=pattern.pattern_id,
            entity_type=entity_type,
            entity_id=str(entity_id),
            study_id=str(study_id),
            confidence=round(float(confidence), 3),
            evidence=evidence,
            detected_at=detected_at,
            severity=pattern.severity,
            recommended_action=pattern.recommended_action,
            responsible_role=pattern.responsible_role,
            sla_deadline=sla_deadline
        )
        
        self.matches.append(match)
    
    def _generate_alerts(self):
        """Generate alerts from pattern matches"""
        logger.info("Generating alerts from pattern matches...")
        
        for match in self.matches:
            pattern = self.patterns.get(match.pattern_id)
            if not pattern:
                continue
            
            alert_id = f"A-{match.match_id}"
            
            evidence_lines = [f"• {k}: {v}" for k, v in match.evidence.items()]
            evidence_summary = "\n".join(evidence_lines)
            
            alert = Alert(
                alert_id=alert_id,
                pattern_id=match.pattern_id,
                pattern_name=pattern.name,
                entity_type=match.entity_type,
                entity_id=match.entity_id,
                study_id=match.study_id,
                severity=match.severity,
                title=f"{pattern.name} - {match.entity_id}",
                message=pattern.description,
                evidence_summary=evidence_summary,
                recommended_action=match.recommended_action,
                responsible_role=match.responsible_role,
                sla_deadline=match.sla_deadline,
                created_at=match.detected_at
            )
            
            self.alerts.append(alert)
    
    def validate_across_studies(self) -> Dict[str, Any]:
        """Validate patterns across multiple studies"""
        logger.info("Validating patterns across studies...")
        
        validation_results = {}
        
        for pattern_id, pattern in self.patterns.items():
            matches = [m for m in self.matches if m.pattern_id == pattern_id]
            
            if not matches:
                validation_results[pattern_id] = {
                    'pattern_name': pattern.name,
                    'occurrence_count': 0,
                    'studies_affected': [],
                    'cross_study_validated': False
                }
                continue
            
            studies_affected = list(set(m.study_id for m in matches))
            avg_confidence = float(np.mean([m.confidence for m in matches]))
            
            is_validated = len(studies_affected) >= 2 and avg_confidence >= 0.7
            
            validation_results[pattern_id] = {
                'pattern_name': pattern.name,
                'occurrence_count': int(len(matches)),
                'studies_affected': studies_affected,
                'avg_confidence': round(avg_confidence, 3),
                'cross_study_validated': bool(is_validated)
            }
            
            pattern.cross_study_validated = bool(is_validated)
            pattern.validation_count = int(len(matches))
        
        return validation_results
    
    def get_pattern_summary(self) -> pd.DataFrame:
        """Get summary of all patterns and their matches"""
        summary_data = []
        
        for pattern_id, pattern in self.patterns.items():
            matches = [m for m in self.matches if m.pattern_id == pattern_id]
            avg_conf = float(np.mean([m.confidence for m in matches])) if matches else 0.0
            
            summary_data.append({
                'pattern_id': pattern_id,
                'pattern_name': pattern.name,
                'category': pattern.category.value,
                'severity': pattern.severity.value,
                'responsible_role': pattern.responsible_role,
                'sla_days': int(pattern.sla_days),
                'match_count': int(len(matches)),
                'avg_confidence': round(avg_conf, 3),
                'cross_study_validated': bool(pattern.cross_study_validated)
            })
        
        return pd.DataFrame(summary_data)
    
    def save_outputs(self):
        """Save all outputs to files"""
        logger.info("Saving Pattern Library outputs...")
        
        # 1. Pattern definitions
        patterns_data = []
        for pid, p in self.patterns.items():
            patterns_data.append({
                'pattern_id': pid,
                'name': p.name,
                'category': p.category.value,
                'description': p.description,
                'severity': p.severity.value,
                'recommended_action': p.recommended_action.value,
                'responsible_role': p.responsible_role,
                'sla_days': int(p.sla_days),
                'resolution_template': p.resolution_template,
                'cross_study_validated': bool(p.cross_study_validated),
                'validation_count': int(p.validation_count)
            })
        
        pd.DataFrame(patterns_data).to_csv(self.output_dir / "pattern_definitions.csv", index=False)
        
        # 2. Pattern matches
        if self.matches:
            matches_data = []
            for m in self.matches:
                matches_data.append({
                    'match_id': m.match_id,
                    'pattern_id': m.pattern_id,
                    'entity_type': m.entity_type,
                    'entity_id': m.entity_id,
                    'study_id': m.study_id,
                    'confidence': float(m.confidence),
                    'evidence': json.dumps(m.evidence),
                    'detected_at': m.detected_at,
                    'severity': m.severity.value,
                    'recommended_action': m.recommended_action.value,
                    'responsible_role': m.responsible_role,
                    'sla_deadline': m.sla_deadline,
                    'status': m.status
                })
            
            matches_df = pd.DataFrame(matches_data)
            matches_df.to_parquet(self.output_dir / "pattern_matches.parquet")
            matches_df.to_csv(self.output_dir / "pattern_matches.csv", index=False)
        
        # 3. Alerts
        if self.alerts:
            alerts_data = []
            for a in self.alerts:
                alerts_data.append({
                    'alert_id': a.alert_id,
                    'pattern_id': a.pattern_id,
                    'pattern_name': a.pattern_name,
                    'entity_type': a.entity_type,
                    'entity_id': a.entity_id,
                    'study_id': a.study_id,
                    'severity': a.severity.value,
                    'title': a.title,
                    'message': a.message,
                    'evidence_summary': a.evidence_summary,
                    'recommended_action': a.recommended_action.value,
                    'responsible_role': a.responsible_role,
                    'sla_deadline': a.sla_deadline,
                    'created_at': a.created_at,
                    'status': a.status
                })
            
            alerts_df = pd.DataFrame(alerts_data)
            alerts_df.to_parquet(self.output_dir / "alerts.parquet")
            alerts_df.to_csv(self.output_dir / "alerts.csv", index=False)
            
            # High priority alerts
            high_priority = alerts_df[alerts_df['severity'].isin(['critical', 'high'])]
            high_priority.to_csv(self.output_dir / "alerts_high_priority.csv", index=False)
        
        # 4. Pattern summary
        summary_df = self.get_pattern_summary()
        summary_df.to_csv(self.output_dir / "pattern_summary.csv", index=False)
        
        # 5. Validation results
        validation = self.validate_across_studies()
        validation_safe = json_serializable(validation)
        with open(self.output_dir / "cross_study_validation.json", 'w') as f:
            json.dump(validation_safe, f, indent=2)
        
        # 6. Summary JSON
        alerts_by_severity = {}
        for sev in PatternSeverity:
            alerts_by_severity[sev.value] = sum(1 for a in self.alerts if a.severity == sev)
        
        alerts_by_role = {}
        for alert in self.alerts:
            role = alert.responsible_role
            alerts_by_role[role] = alerts_by_role.get(role, 0) + 1
        
        summary = {
            'generated_at': datetime.now().isoformat(),
            'version': '1.2',
            'total_patterns': len(self.patterns),
            'patterns_by_category': {cat.value: sum(1 for p in self.patterns.values() if p.category == cat) for cat in PatternCategory},
            'total_matches': len(self.matches),
            'total_alerts': len(self.alerts),
            'alerts_by_severity': alerts_by_severity,
            'alerts_by_role': alerts_by_role,
            'cross_study_validated_patterns': sum(1 for p in self.patterns.values() if p.cross_study_validated)
        }
        
        summary_safe = json_serializable(summary)
        with open(self.output_dir / "pattern_library_summary.json", 'w') as f:
            json.dump(summary_safe, f, indent=2)
        
        logger.info(f"Outputs saved to {self.output_dir}")
        
        return summary


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main execution function"""
    print("=" * 70)
    print("SANCHALAK AI - PHASE 3.6: PATTERN LIBRARY v1.2")
    print("=" * 70)
    
    start_time = datetime.now()
    
    data_dir = Path("data/processed")
    
    # Initialize Pattern Library
    print("\n🔧 Initializing Pattern Library...")
    library = PatternLibrary(data_dir=data_dir)
    
    # Load data
    print("\n📂 Loading patient data...")
    try:
        df = library.load_data()
        print(f"   Loaded {len(df):,} patients with {len(df.columns)} columns")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return None, None
    
    # Detect patterns
    print("\n🔍 Detecting patterns...")
    matches, alerts = library.detect_patterns(df)
    
    # Validate across studies
    print("\n✅ Validating patterns across studies...")
    validation = library.validate_across_studies()
    
    # Save outputs
    print("\n💾 Saving outputs...")
    summary = library.save_outputs()
    
    duration = (datetime.now() - start_time).total_seconds()
    
    # Print results
    print("\n" + "=" * 70)
    print("PATTERN LIBRARY RESULTS")
    print("=" * 70)
    
    print(f"\n📊 PATTERNS DEFINED: {summary['total_patterns']}")
    print("\n   By Category:")
    for cat, count in summary['patterns_by_category'].items():
        print(f"      {cat}: {count}")
    
    print(f"\n🔍 PATTERN MATCHES: {summary['total_matches']}")
    
    # Show pattern breakdown
    print("\n   By Pattern:")
    pattern_summary = library.get_pattern_summary()
    for _, row in pattern_summary.iterrows():
        if row['match_count'] > 0:
            print(f"      {row['pattern_id']} ({row['pattern_name']}): {row['match_count']} matches")
    
    print(f"\n🚨 ALERTS GENERATED: {summary['total_alerts']}")
    print("\n   By Severity:")
    for severity, count in summary['alerts_by_severity'].items():
        emoji = {'critical': '🔴', 'high': '🟠', 'medium': '🟡', 'low': '🟢', 'info': 'ℹ️'}.get(severity, '⚪')
        print(f"      {emoji} {severity}: {count}")
    
    print("\n   By Role:")
    for role, count in summary['alerts_by_role'].items():
        print(f"      {role}: {count}")
    
    validated_count = sum(1 for p in library.patterns.values() if p.cross_study_validated)
    print(f"\n✅ Cross-Study Validated: {validated_count} patterns")
    
    print(f"\n⏱️  Duration: {duration:.2f} seconds")
    
    print("\n" + "=" * 70)
    print("📁 OUTPUT FILES:")
    print("=" * 70)
    output_dir = data_dir / "analytics" / "pattern_library"
    print(f"\n   {output_dir}/")
    for f in sorted(output_dir.glob("*")):
        size = f.stat().st_size
        size_str = f"{size:,}" if size < 1024 else f"{size/1024:.1f}KB"
        print(f"   ├── {f.name} ({size_str})")
    
    print("\n" + "=" * 70)
    print("✅ PHASE 3.6 COMPLETE - Pattern Library Ready")
    print("=" * 70)
    
    return library, summary


if __name__ == "__main__":
    main()