"""
SANCHALAK AI - Resolution Genome Engine v1.2
=====================================================
Every resolution becomes reusable knowledge.

Version: 1.2 - Fixed column detection, dynamic issue mapping
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import pickle
import logging
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import hashlib

# Embedding imports
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class ResolutionGenomeConfig:
    """Configuration for Resolution Genome."""
    embedding_model: str = 'all-MiniLM-L6-v2'
    exact_match_threshold: float = 0.98
    similar_match_threshold: float = 0.85
    type_match_threshold: float = 0.70
    min_samples_for_confidence: int = 5
    success_rate_weight: float = 0.6
    recency_weight: float = 0.2
    similarity_weight: float = 0.2
    top_k_matches: int = 3
    recency_decay: float = 0.95


# =============================================================================
# FLEXIBLE ISSUE TYPE MAPPING
# =============================================================================

# Maps various possible column names to canonical issue types
ISSUE_COLUMN_PATTERNS = {
    # Pattern: canonical_issue_type
    'sdv': 'sdv_incomplete',
    'query': 'open_queries',
    'queries': 'open_queries',
    'signature_gap': 'signature_gaps',
    'signature_gaps': 'signature_gaps',
    'unsigned': 'signature_gaps',
    'broken_sig': 'broken_signatures',
    'broken_signature': 'broken_signatures',
    'sae_dm': 'sae_dm_pending',
    'sae_safety': 'sae_safety_pending',
    'missing_visit': 'missing_visits',
    'missing_page': 'missing_pages',
    'lab': 'lab_issues',
    'edrr': 'edrr_issues',
    'inactivat': 'inactivated_forms',
    'meddra': 'meddra_uncoded',
    'whodrug': 'whodrug_uncoded',
    'high_query': 'high_query_volume',
    # Temporal patterns
    'holiday': 'holiday_drift',
    'month_end': 'end_of_month_rush',
    'quarter_end': 'quarter_end_closeout',
    'enrollment_surge': 'enrollment_surge',
    'turnover': 'site_staff_turnover',
    'pi_change': 'site_staff_turnover',
    'amendment': 'protocol_amendment',
}

ISSUE_PRIORITY_ORDER = {
    'sae_safety_pending': 1,
    'sae_dm_pending': 2,
    'missing_visits': 3,
    'missing_pages': 4,
    'open_queries': 5,
    'signature_gaps': 6,
    'broken_signatures': 7,
    'sdv_incomplete': 8,
    'lab_issues': 9,
    'edrr_issues': 10,
    'meddra_uncoded': 11,
    'whodrug_uncoded': 12,
    'inactivated_forms': 13,
    'high_query_volume': 14,
    # Temporal patterns (medium-low priority as they're proactive)
    'holiday_drift': 15,
    'end_of_month_rush': 16,
    'quarter_end_closeout': 17,
    'enrollment_surge': 18,
    'site_staff_turnover': 19,
    'protocol_amendment': 20,
}


def detect_issue_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    Dynamically detect issue columns and map them to canonical types.
    
    Returns:
        Dict mapping column_name -> canonical_issue_type
    """
    issue_mapping = {}
    
    # Look for boolean columns that might indicate issues
    for col in df.columns:
        col_lower = col.lower()
        
        # Skip non-issue columns
        if col_lower in ['patient_key', 'study_id', 'site_id', 'subject_id', 
                         'priority_tier', 'issue_count', 'primary_issue']:
            continue
        
        # Check if column is boolean or can indicate presence of issue
        is_boolean = df[col].dtype == bool
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        
        # Try to match column name to issue type
        for pattern, issue_type in ISSUE_COLUMN_PATTERNS.items():
            if pattern in col_lower:
                # Verify this column actually has True values or positive counts
                if is_boolean:
                    if df[col].any():
                        issue_mapping[col] = issue_type
                        break
                elif is_numeric:
                    if (df[col] > 0).any():
                        issue_mapping[col] = issue_type
                        break
    
    return issue_mapping


# =============================================================================
# RESOLUTION TEMPLATES
# =============================================================================

@dataclass
class ResolutionTemplate:
    """A single resolution template."""
    template_id: str
    issue_type: str
    issue_subtype: str
    title: str
    description: str
    steps: List[str]
    responsible_role: str
    estimated_effort_hours: float
    success_rate: float = 0.0
    times_used: int = 0
    times_successful: int = 0
    avg_resolution_days: float = 0.0
    tags: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())


RESOLUTION_TEMPLATES: List[Dict] = [
    # SDV INCOMPLETE
    {
        "template_id": "SDV-001",
        "issue_type": "sdv_incomplete",
        "issue_subtype": "pending_sdv",
        "title": "Complete Source Data Verification",
        "description": "CRFs requiring SDV have not been verified against source documents.",
        "steps": [
            "1. Access EDC system and navigate to SDV queue",
            "2. Review list of CRFs pending SDV for the subject",
            "3. Obtain source documents (medical records, lab reports)",
            "4. Compare each data point in CRF against source",
            "5. Mark discrepancies and raise queries if needed",
            "6. Complete SDV checkbox for verified pages",
            "7. Document verification in monitoring report"
        ],
        "responsible_role": "CRA",
        "estimated_effort_hours": 0.25,
        "tags": ["sdv", "monitoring", "source_verification"],
        "prerequisites": ["Site visit scheduled", "Source access available"]
    },
    {
        "template_id": "SDV-002",
        "issue_type": "sdv_incomplete",
        "issue_subtype": "remote_sdv",
        "title": "Remote Source Data Verification",
        "description": "Perform SDV remotely using certified copies of source documents.",
        "steps": [
            "1. Request certified copies from site coordinator",
            "2. Receive and log documents in TMF",
            "3. Perform remote SDV comparison",
            "4. Document any discrepancies",
            "5. Mark SDV complete in EDC",
            "6. Archive certified copies"
        ],
        "responsible_role": "CRA",
        "estimated_effort_hours": 0.5,
        "tags": ["sdv", "remote", "certified_copies"],
        "prerequisites": ["Remote monitoring approved"]
    },
    
    # OPEN QUERIES
    {
        "template_id": "QRY-001",
        "issue_type": "open_queries",
        "issue_subtype": "data_clarification",
        "title": "Resolve Data Clarification Query",
        "description": "Query raised for data clarification requires site response.",
        "steps": [
            "1. Review query text and understand the question",
            "2. Check source documents for correct information",
            "3. Enter response in EDC query interface",
            "4. Correct data if needed",
            "5. Close query with explanation"
        ],
        "responsible_role": "Site",
        "estimated_effort_hours": 0.1,
        "tags": ["query", "data_clarification", "site_response"],
        "prerequisites": []
    },
    {
        "template_id": "QRY-002",
        "issue_type": "open_queries",
        "issue_subtype": "missing_data",
        "title": "Resolve Missing Data Query",
        "description": "Query raised for missing data entry.",
        "steps": [
            "1. Identify missing data field",
            "2. Obtain information from source documents",
            "3. Enter missing data in EDC",
            "4. Respond to query with explanation",
            "5. Close query"
        ],
        "responsible_role": "Site",
        "estimated_effort_hours": 0.15,
        "tags": ["query", "missing_data", "data_entry"],
        "prerequisites": ["Source data available"]
    },
    {
        "template_id": "QRY-003",
        "issue_type": "open_queries",
        "issue_subtype": "range_check",
        "title": "Resolve Range Check Query",
        "description": "Query raised due to value outside expected range.",
        "steps": [
            "1. Review the flagged value",
            "2. Verify against source document",
            "3. If correct, provide clinical justification",
            "4. If incorrect, correct the value",
            "5. Close query with response"
        ],
        "responsible_role": "Site",
        "estimated_effort_hours": 0.1,
        "tags": ["query", "range_check", "edit_check"],
        "prerequisites": []
    },
    {
        "template_id": "QRY-004",
        "issue_type": "open_queries",
        "issue_subtype": "batch_resolution",
        "title": "Batch Query Resolution",
        "description": "Multiple similar queries can be resolved together.",
        "steps": [
            "1. Filter queries by type/form",
            "2. Group similar queries together",
            "3. Prepare batch response template",
            "4. Resolve queries in sequence",
            "5. Verify all closed"
        ],
        "responsible_role": "Data Manager",
        "estimated_effort_hours": 0.5,
        "tags": ["query", "batch", "efficiency"],
        "prerequisites": ["Multiple similar queries exist"]
    },
    
    # SIGNATURE GAPS
    {
        "template_id": "SIG-001",
        "issue_type": "signature_gaps",
        "issue_subtype": "investigator_signature",
        "title": "Obtain Investigator Signature",
        "description": "CRFs require Principal Investigator signature.",
        "steps": [
            "1. Generate list of unsigned CRFs",
            "2. Contact PI or designee",
            "3. Schedule signing session",
            "4. PI reviews and signs CRFs in EDC",
            "5. Verify signature timestamps"
        ],
        "responsible_role": "Site",
        "estimated_effort_hours": 0.1,
        "tags": ["signature", "investigator", "compliance"],
        "prerequisites": ["PI available", "CRFs complete"]
    },
    {
        "template_id": "SIG-002",
        "issue_type": "signature_gaps",
        "issue_subtype": "batch_signing",
        "title": "Batch Signature Session",
        "description": "Schedule dedicated session for multiple signatures.",
        "steps": [
            "1. Compile list of all pending signatures",
            "2. Prepare summary for PI review",
            "3. Schedule 30-minute signing block",
            "4. Assist PI during signing session",
            "5. Confirm all signatures complete"
        ],
        "responsible_role": "Site",
        "estimated_effort_hours": 0.5,
        "tags": ["signature", "batch", "efficiency"],
        "prerequisites": ["PI availability confirmed"]
    },
    
    # BROKEN SIGNATURES
    {
        "template_id": "BSIG-001",
        "issue_type": "broken_signatures",
        "issue_subtype": "data_correction",
        "title": "Re-sign After Data Correction",
        "description": "Signature invalidated due to data correction.",
        "steps": [
            "1. Identify which correction broke signature",
            "2. Notify PI of required re-signature",
            "3. PI reviews correction",
            "4. PI re-signs the form",
            "5. Verify signature restored"
        ],
        "responsible_role": "Site",
        "estimated_effort_hours": 0.1,
        "tags": ["signature", "broken", "re-sign"],
        "prerequisites": ["Correction already made"]
    },
    
    # SAE-DM PENDING
    {
        "template_id": "SAE-DM-001",
        "issue_type": "sae_dm_pending",
        "issue_subtype": "reconciliation",
        "title": "SAE Data Management Reconciliation",
        "description": "Reconcile SAE data between safety database and EDC.",
        "steps": [
            "1. Export SAE records from safety database",
            "2. Export SAE records from EDC",
            "3. Compare key fields (dates, terms, outcomes)",
            "4. Identify discrepancies",
            "5. Raise queries for mismatches",
            "6. Update reconciliation log"
        ],
        "responsible_role": "Safety Data Manager",
        "estimated_effort_hours": 0.5,
        "tags": ["sae", "reconciliation", "safety"],
        "prerequisites": ["Access to both databases"]
    },
    {
        "template_id": "SAE-DM-002",
        "issue_type": "sae_dm_pending",
        "issue_subtype": "missing_fields",
        "title": "Complete Missing SAE Fields",
        "description": "SAE record has missing required fields.",
        "steps": [
            "1. Identify missing fields",
            "2. Contact site for information",
            "3. Update SAE record",
            "4. Verify completeness",
            "5. Mark as reconciled"
        ],
        "responsible_role": "Safety Data Manager",
        "estimated_effort_hours": 0.3,
        "tags": ["sae", "missing_data", "completeness"],
        "prerequisites": []
    },
    
    # SAE-SAFETY PENDING
    {
        "template_id": "SAE-SAF-001",
        "issue_type": "sae_safety_pending",
        "issue_subtype": "medical_review",
        "title": "SAE Medical Review",
        "description": "SAE requires medical review and causality assessment.",
        "steps": [
            "1. Review SAE narrative and supporting documents",
            "2. Assess relationship to study drug",
            "3. Determine expectedness",
            "4. Complete causality assessment",
            "5. Sign off medical review"
        ],
        "responsible_role": "Safety Physician",
        "estimated_effort_hours": 0.5,
        "tags": ["sae", "medical_review", "causality"],
        "prerequisites": ["Complete SAE narrative available"]
    },
    
    # MISSING VISITS
    {
        "template_id": "VIS-001",
        "issue_type": "missing_visits",
        "issue_subtype": "scheduled_visit",
        "title": "Complete Missing Visit Data",
        "description": "Scheduled visit occurred but data not entered.",
        "steps": [
            "1. Confirm visit occurred (check source)",
            "2. Obtain visit date and assessments",
            "3. Enter visit data in EDC",
            "4. Complete all required forms",
            "5. Raise queries if data unavailable"
        ],
        "responsible_role": "Site",
        "estimated_effort_hours": 0.5,
        "tags": ["visit", "missing", "data_entry"],
        "prerequisites": ["Visit actually occurred"]
    },
    {
        "template_id": "VIS-002",
        "issue_type": "missing_visits",
        "issue_subtype": "missed_visit",
        "title": "Document Missed Visit",
        "description": "Subject missed a scheduled visit.",
        "steps": [
            "1. Confirm visit was missed",
            "2. Document reason for missed visit",
            "3. Enter 'Not Done' with reason in EDC",
            "4. Assess protocol deviation if applicable",
            "5. Plan make-up visit if protocol allows"
        ],
        "responsible_role": "Site",
        "estimated_effort_hours": 0.2,
        "tags": ["visit", "missed", "documentation"],
        "prerequisites": []
    },
    
    # MISSING PAGES
    {
        "template_id": "PG-001",
        "issue_type": "missing_pages",
        "issue_subtype": "form_not_started",
        "title": "Complete Missing CRF Page",
        "description": "Required CRF page has not been started.",
        "steps": [
            "1. Identify which page is missing",
            "2. Verify if assessment was performed",
            "3. Enter data from source documents",
            "4. Complete all required fields",
            "5. Save and verify page status"
        ],
        "responsible_role": "Site",
        "estimated_effort_hours": 0.3,
        "tags": ["page", "missing", "crf"],
        "prerequisites": ["Source data available"]
    },
    {
        "template_id": "PG-002",
        "issue_type": "missing_pages",
        "issue_subtype": "not_applicable",
        "title": "Mark Page as Not Applicable",
        "description": "Page not required for this subject.",
        "steps": [
            "1. Confirm page is truly not applicable",
            "2. Enter 'Not Applicable' in EDC",
            "3. Provide reason/justification",
            "4. Save page"
        ],
        "responsible_role": "Site",
        "estimated_effort_hours": 0.1,
        "tags": ["page", "not_applicable"],
        "prerequisites": []
    },
    
    # LAB ISSUES
    {
        "template_id": "LAB-001",
        "issue_type": "lab_issues",
        "issue_subtype": "missing_ranges",
        "title": "Obtain Missing Lab Ranges",
        "description": "Laboratory normal ranges not provided.",
        "steps": [
            "1. Contact local laboratory",
            "2. Request current reference ranges",
            "3. Enter ranges in EDC lab panel",
            "4. Verify units match",
            "5. Confirm entry"
        ],
        "responsible_role": "Site",
        "estimated_effort_hours": 0.3,
        "tags": ["lab", "ranges", "reference"],
        "prerequisites": ["Lab contact available"]
    },
    {
        "template_id": "LAB-002",
        "issue_type": "lab_issues",
        "issue_subtype": "missing_results",
        "title": "Enter Missing Lab Results",
        "description": "Lab results not entered in EDC.",
        "steps": [
            "1. Obtain lab report from source",
            "2. Enter results in EDC lab form",
            "3. Flag abnormal values",
            "4. Provide clinical significance if abnormal",
            "5. Save and verify"
        ],
        "responsible_role": "Site",
        "estimated_effort_hours": 0.2,
        "tags": ["lab", "results", "data_entry"],
        "prerequisites": ["Lab report available"]
    },
    
    # EDRR ISSUES
    {
        "template_id": "EDRR-001",
        "issue_type": "edrr_issues",
        "issue_subtype": "third_party_mismatch",
        "title": "Resolve Third-Party Data Mismatch",
        "description": "Discrepancy between EDC and external data source.",
        "steps": [
            "1. Review EDRR discrepancy report",
            "2. Identify source of mismatch",
            "3. Correct data in appropriate system",
            "4. Re-run reconciliation",
            "5. Confirm resolution"
        ],
        "responsible_role": "Data Manager",
        "estimated_effort_hours": 0.3,
        "tags": ["edrr", "reconciliation", "third_party"],
        "prerequisites": ["Access to external system"]
    },
    
    # INACTIVATED FORMS
    {
        "template_id": "INACT-001",
        "issue_type": "inactivated_forms",
        "issue_subtype": "review_required",
        "title": "Review Inactivated Form",
        "description": "Form was inactivated and requires review.",
        "steps": [
            "1. Review reason for inactivation",
            "2. Verify inactivation was appropriate",
            "3. Document review in audit trail",
            "4. Reactivate if inactivation was error",
            "5. Update tracking log"
        ],
        "responsible_role": "Data Manager",
        "estimated_effort_hours": 0.2,
        "tags": ["inactivated", "review", "audit"],
        "prerequisites": []
    },
    
    # MEDDRA UNCODED
    {
        "template_id": "MED-001",
        "issue_type": "meddra_uncoded",
        "issue_subtype": "adverse_event",
        "title": "Code Adverse Event to MedDRA",
        "description": "Adverse event term requires MedDRA coding.",
        "steps": [
            "1. Review verbatim AE term",
            "2. Search MedDRA dictionary",
            "3. Select appropriate LLT/PT",
            "4. Verify SOC assignment",
            "5. Approve coding"
        ],
        "responsible_role": "Medical Coder",
        "estimated_effort_hours": 0.05,
        "tags": ["coding", "meddra", "adverse_event"],
        "prerequisites": []
    },
    {
        "template_id": "MED-002",
        "issue_type": "meddra_uncoded",
        "issue_subtype": "medical_history",
        "title": "Code Medical History to MedDRA",
        "description": "Medical history term requires MedDRA coding.",
        "steps": [
            "1. Review verbatim term",
            "2. Search MedDRA dictionary",
            "3. Select appropriate code",
            "4. Apply coding",
            "5. Verify"
        ],
        "responsible_role": "Medical Coder",
        "estimated_effort_hours": 0.05,
        "tags": ["coding", "meddra", "medical_history"],
        "prerequisites": []
    },
    
    # WHODRUG UNCODED
    {
        "template_id": "WHO-001",
        "issue_type": "whodrug_uncoded",
        "issue_subtype": "concomitant_med",
        "title": "Code Medication to WHODrug",
        "description": "Concomitant medication requires WHODrug coding.",
        "steps": [
            "1. Review verbatim medication name",
            "2. Search WHODrug dictionary",
            "3. Select appropriate drug code (ATC)",
            "4. Verify formulation/route match",
            "5. Apply coding"
        ],
        "responsible_role": "Medical Coder",
        "estimated_effort_hours": 0.05,
        "tags": ["coding", "whodrug", "medication"],
        "prerequisites": []
    },
    
    # HIGH QUERY VOLUME
    {
        "template_id": "HQV-001",
        "issue_type": "high_query_volume",
        "issue_subtype": "site_training",
        "title": "Site Re-training on Data Entry",
        "description": "High query volume indicates training need.",
        "steps": [
            "1. Analyze query patterns",
            "2. Identify common errors",
            "3. Prepare training materials",
            "4. Schedule training session",
            "5. Conduct training",
            "6. Monitor improvement"
        ],
        "responsible_role": "CRA",
        "estimated_effort_hours": 2.0,
        "tags": ["training", "quality", "prevention"],
        "prerequisites": ["Pattern analysis complete"]
    },
    
    # ============ TEMPORAL PATTERNS (NEW) ============
    
    # HOLIDAY DRIFT
    {
        "template_id": "TEMP-001",
        "issue_type": "holiday_drift",
        "issue_subtype": "pre_holiday_backlog",
        "title": "Pre-Holiday Backlog Prevention",
        "description": "Address issue backlog building up before major holidays.",
        "steps": [
            "1. Identify upcoming holiday periods (Thanksgiving, Christmas, etc.)",
            "2. Generate priority task list 2 weeks before holiday",
            "3. Schedule dedicated resolution sessions",
            "4. Assign additional resources to high-priority items",
            "5. Clear signature backlogs before PI unavailability",
            "6. Document any items that will carry over"
        ],
        "responsible_role": "Study Lead",
        "estimated_effort_hours": 4.0,
        "tags": ["temporal", "holiday", "backlog", "prevention"],
        "prerequisites": ["Holiday calendar identified"]
    },
    {
        "template_id": "TEMP-002",
        "issue_type": "holiday_drift",
        "issue_subtype": "post_holiday_catchup",
        "title": "Post-Holiday Catch-up Sprint",
        "description": "Rapidly address backlog accumulated during holiday period.",
        "steps": [
            "1. Run full issue inventory on first day back",
            "2. Identify new SAEs and safety issues (highest priority)",
            "3. Prioritize by age and clinical significance",
            "4. Schedule catch-up signing sessions with PI",
            "5. Batch resolve similar query types",
            "6. Update metrics and report to sponsor"
        ],
        "responsible_role": "Data Manager",
        "estimated_effort_hours": 6.0,
        "tags": ["temporal", "holiday", "catchup", "sprint"],
        "prerequisites": ["Holiday period ended"]
    },
    
    # END-OF-MONTH RUSH
    {
        "template_id": "TEMP-003",
        "issue_type": "end_of_month_rush",
        "issue_subtype": "milestone_push",
        "title": "End-of-Month Milestone Push",
        "description": "Accelerated resolution to meet monthly reporting deadlines.",
        "steps": [
            "1. Review monthly targets vs current status (day 25)",
            "2. Generate gap-to-target list by site",
            "3. Prioritize issues blocking milestone completion",
            "4. Coordinate with sites for extended availability",
            "5. Run daily stand-ups last week of month",
            "6. Track completion rate hourly on final day"
        ],
        "responsible_role": "Study Lead",
        "estimated_effort_hours": 3.0,
        "tags": ["temporal", "month_end", "milestone", "deadline"],
        "prerequisites": ["Monthly targets defined"]
    },
    {
        "template_id": "TEMP-004",
        "issue_type": "end_of_month_rush",
        "issue_subtype": "query_blitz",
        "title": "Query Resolution Blitz",
        "description": "Focused effort to clear query backlog before month-end.",
        "steps": [
            "1. Export all open queries sorted by age",
            "2. Assign queries to team members by form type",
            "3. Set 2-hour resolution targets",
            "4. Conduct hourly check-ins",
            "5. Escalate blockers immediately",
            "6. Report final counts at end of day"
        ],
        "responsible_role": "Data Manager",
        "estimated_effort_hours": 4.0,
        "tags": ["temporal", "month_end", "query", "blitz"],
        "prerequisites": ["Team availability confirmed"]
    },
    
    # QUARTER-END CLOSEOUT
    {
        "template_id": "TEMP-005",
        "issue_type": "quarter_end_closeout",
        "issue_subtype": "regulatory_deadline",
        "title": "Quarter-End Regulatory Closeout",
        "description": "Comprehensive review to meet quarterly regulatory deadlines.",
        "steps": [
            "1. Generate full regulatory compliance report",
            "2. Identify all overdue safety reports",
            "3. Clear SAE reconciliation discrepancies",
            "4. Complete audit trail review",
            "5. Verify e-signature chain integrity",
            "6. Prepare executive summary for sponsor"
        ],
        "responsible_role": "Safety Data Manager",
        "estimated_effort_hours": 8.0,
        "tags": ["temporal", "quarter_end", "regulatory", "compliance"],
        "prerequisites": ["Access to safety database"]
    },
    
    # ENROLLMENT SURGE
    {
        "template_id": "ENR-001",
        "issue_type": "enrollment_surge",
        "issue_subtype": "capacity_planning",
        "title": "Enrollment Surge Capacity Planning",
        "description": "Prepare for increased data volume during enrollment spikes.",
        "steps": [
            "1. Forecast enrollment trajectory from screening logs",
            "2. Project data entry volume increase",
            "3. Pre-assign CRF review responsibilities",
            "4. Schedule additional monitoring visits",
            "5. Prepare site for increased query volume",
            "6. Brief sponsor on capacity plan"
        ],
        "responsible_role": "CRA",
        "estimated_effort_hours": 2.0,
        "tags": ["enrollment", "capacity", "planning"],
        "prerequisites": ["Enrollment forecast available"]
    },
    
    # SITE TURNOVER
    {
        "template_id": "TURN-001",
        "issue_type": "site_staff_turnover",
        "issue_subtype": "coordinator_change",
        "title": "Site Coordinator Transition",
        "description": "Manage data quality during site staff transitions.",
        "steps": [
            "1. Identify outgoing coordinator's pending items",
            "2. Complete knowledge transfer session",
            "3. Re-train new coordinator on EDC system",
            "4. Verify access credentials and delegation logs",
            "5. Monitor error rates during first 30 days",
            "6. Schedule additional support visits if needed"
        ],
        "responsible_role": "CRA",
        "estimated_effort_hours": 4.0,
        "tags": ["turnover", "transition", "training"],
        "prerequisites": ["New coordinator identified"]
    },
    {
        "template_id": "TURN-002",
        "issue_type": "site_staff_turnover",
        "issue_subtype": "pi_change",
        "title": "Principal Investigator Transition",
        "description": "Critical transition when PI changes at a site.",
        "steps": [
            "1. Document all pending PI signature requirements",
            "2. Coordinate IRB/EC notification and approval",
            "3. Update delegation logs in TMF",
            "4. Transfer EDC signature authority",
            "5. Brief new PI on study status and open issues",
            "6. Clear signature backlog from previous PI"
        ],
        "responsible_role": "CRA",
        "estimated_effort_hours": 8.0,
        "tags": ["turnover", "pi_change", "critical", "regulatory"],
        "prerequisites": ["IRB approval for PI change"]
    },
    
    # PROTOCOL AMENDMENT
    {
        "template_id": "AMEND-001",
        "issue_type": "protocol_amendment",
        "issue_subtype": "data_impact",
        "title": "Protocol Amendment Data Impact Assessment",
        "description": "Assess and address data implications of protocol changes.",
        "steps": [
            "1. Review amendment impact on CRF structure",
            "2. Identify retrospective data collection needs",
            "3. Update edit checks and validations",
            "4. Re-train sites on new procedures",
            "5. Generate deviation reports for transition period",
            "6. Validate new data collection completeness"
        ],
        "responsible_role": "Data Manager",
        "estimated_effort_hours": 6.0,
        "tags": ["amendment", "protocol", "data_impact"],
        "prerequisites": ["Amendment approved"]
    },
]


# =============================================================================
# RESOLUTION GENOME ENGINE
# =============================================================================

class ResolutionGenome:
    """Resolution Genome: Every resolution becomes reusable knowledge."""
    
    def __init__(self, config: Optional[ResolutionGenomeConfig] = None):
        self.config = config or ResolutionGenomeConfig()
        self.templates: Dict[str, ResolutionTemplate] = {}
        self.templates_by_type: Dict[str, List[ResolutionTemplate]] = defaultdict(list)
        self.embeddings: Dict[str, np.ndarray] = {}
        self.embedding_model = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.resolution_history: List[Dict] = []
        self.is_initialized: bool = False
        
    def initialize(self) -> 'ResolutionGenome':
        """Initialize the genome with templates and embeddings."""
        logger.info("=" * 60)
        logger.info("INITIALIZING RESOLUTION GENOME")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        self._load_templates()
        self._initialize_embeddings()
        self._build_embedding_index()
        self.is_initialized = True
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"\nInitialization complete in {duration:.2f} seconds")
        logger.info(f"Templates loaded: {len(self.templates)}")
        
        return self
    
    def _load_templates(self):
        logger.info("\n1. Loading resolution templates...")
        
        for template_data in RESOLUTION_TEMPLATES:
            template = ResolutionTemplate(**template_data)
            self.templates[template.template_id] = template
            self.templates_by_type[template.issue_type].append(template)
            
        logger.info(f"   Loaded {len(self.templates)} templates")
        for issue_type, templates in sorted(self.templates_by_type.items()):
            logger.info(f"   - {issue_type}: {len(templates)} templates")
    
    def _initialize_embeddings(self):
        logger.info("\n2. Initializing embedding model...")
        
        if HAS_SENTENCE_TRANSFORMERS:
            try:
                self.embedding_model = SentenceTransformer(self.config.embedding_model)
                logger.info(f"   Using SentenceTransformer: {self.config.embedding_model}")
            except Exception as e:
                logger.warning(f"   Failed to load SentenceTransformer: {e}")
                self.embedding_model = None
        else:
            logger.info("   sentence-transformers not installed, using TF-IDF")
            
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000, ngram_range=(1, 2), stop_words='english'
        )
    
    def _build_embedding_index(self):
        logger.info("\n3. Building embedding index...")
        
        template_texts = []
        template_ids = []
        
        for template_id, template in self.templates.items():
            text = self._template_to_text(template)
            template_texts.append(text)
            template_ids.append(template_id)
        
        if self.embedding_model is not None:
            embeddings = self.embedding_model.encode(template_texts, show_progress_bar=False)
            for i, template_id in enumerate(template_ids):
                self.embeddings[template_id] = embeddings[i]
            logger.info(f"   Generated {len(embeddings)} semantic embeddings")
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(template_texts)
        self._tfidf_template_ids = template_ids
        logger.info(f"   Built TF-IDF matrix: {self.tfidf_matrix.shape}")
    
    def _template_to_text(self, template: ResolutionTemplate) -> str:
        parts = [
            template.title, template.description,
            template.issue_type.replace('_', ' '),
            template.issue_subtype.replace('_', ' '),
            ' '.join(template.steps), ' '.join(template.tags),
            template.responsible_role
        ]
        return ' '.join(parts)
    
    def get_best_template_for_issue(self, issue_type: str) -> Optional[Dict]:
        """Get the best resolution template for an issue type."""
        templates = self.templates_by_type.get(issue_type, [])
        
        if not templates:
            for key, tmpl_list in self.templates_by_type.items():
                if issue_type in key or key in issue_type:
                    templates = tmpl_list
                    break
        
        if not templates:
            return None
        
        best = max(templates, key=lambda t: (t.success_rate, -t.estimated_effort_hours))
        
        return {
            'template_id': best.template_id,
            'title': best.title,
            'description': best.description,
            'steps': best.steps,
            'responsible_role': best.responsible_role,
            'estimated_effort_hours': best.estimated_effort_hours,
            'issue_type': best.issue_type,
            'issue_subtype': best.issue_subtype,
            'tags': best.tags,
            'success_rate': best.success_rate,
            'times_used': best.times_used,
        }
    
    def assign_resolutions_to_patients(
        self, 
        issues_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        THE RESOLUTION ASSIGNMENT LOOP
        """
        logger.info("\n" + "=" * 60)
        logger.info("RESOLUTION ASSIGNMENT LOOP")
        logger.info("=" * 60)
        
        # Dynamically detect issue columns
        issue_mapping = detect_issue_columns(issues_df)
        logger.info(f"Detected {len(issue_mapping)} issue columns:")
        for col, issue_type in issue_mapping.items():
            count = issues_df[col].sum() if issues_df[col].dtype == bool else (issues_df[col] > 0).sum()
            logger.info(f"   {col} -> {issue_type} ({count:,} patients)")
        
        if not issue_mapping:
            logger.warning("No issue columns detected! Checking for alternative patterns...")
            
            # Try alternative detection: look for any column with positive values
            # that might indicate issues
            alt_cols = []
            for col in issues_df.columns:
                if any(x in col.lower() for x in ['count_', 'issue', 'pending', 'missing', 'open', 'uncoded']):
                    if pd.api.types.is_numeric_dtype(issues_df[col]):
                        if (issues_df[col] > 0).any():
                            alt_cols.append(col)
            
            if alt_cols:
                logger.info(f"Found {len(alt_cols)} alternative issue indicator columns")
                for col in alt_cols:
                    # Try to map to issue type
                    for pattern, issue_type in ISSUE_COLUMN_PATTERNS.items():
                        if pattern in col.lower():
                            issue_mapping[col] = issue_type
                            break
                    else:
                        # Create generic mapping
                        issue_mapping[col] = col.replace('count_', '').replace('_count', '')
        
        if not issue_mapping:
            logger.error("Still no issue columns found!")
            logger.info("Available columns: " + ", ".join(sorted(issues_df.columns)[:20]))
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        # Find patients with at least one issue
        has_any_issue = pd.Series(False, index=issues_df.index)
        for col in issue_mapping.keys():
            if issues_df[col].dtype == bool:
                has_any_issue |= issues_df[col]
            else:
                has_any_issue |= (issues_df[col] > 0)
        
        patients_with_issues = issues_df[has_any_issue].copy()
        logger.info(f"Patients with issues: {len(patients_with_issues):,}")
        
        if len(patients_with_issues) == 0:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        
        # Generate patient-level recommendations
        logger.info("\nGenerating patient-level recommendations...")
        
        patient_recommendations = []
        
        for idx, patient in patients_with_issues.iterrows():
            patient_key = patient.get('patient_key', str(idx))
            study_id = patient.get('study_id', 'Unknown')
            site_id = patient.get('site_id', 'Unknown')
            subject_id = patient.get('subject_id', 'Unknown')
            priority_tier = patient.get('priority_tier', 'Medium')
            
            # Collect all issues for this patient
            patient_issues = []
            for col, issue_type in issue_mapping.items():
                has_issue = patient.get(col, False)
                if isinstance(has_issue, bool) and has_issue:
                    issue_count = 1
                elif isinstance(has_issue, (int, float)) and has_issue > 0:
                    issue_count = int(has_issue)
                else:
                    continue
                
                priority = ISSUE_PRIORITY_ORDER.get(issue_type, 99)
                patient_issues.append({
                    'issue_type': issue_type,
                    'priority': priority,
                    'count': issue_count,
                    'source_column': col
                })
            
            if not patient_issues:
                continue
            
            # Sort by priority
            patient_issues.sort(key=lambda x: x['priority'])
            
            # Get resolution for each issue
            for i, issue in enumerate(patient_issues):
                template = self.get_best_template_for_issue(issue['issue_type'])
                
                if template:
                    patient_recommendations.append({
                        'patient_key': patient_key,
                        'study_id': study_id,
                        'site_id': site_id,
                        'subject_id': subject_id,
                        'priority_tier': priority_tier,
                        'issue_rank': i + 1,
                        'issue_type': issue['issue_type'],
                        'issue_count': issue['count'],
                        'is_primary_issue': i == 0,
                        'template_id': template['template_id'],
                        'recommended_action': template['title'],
                        'action_description': template['description'],
                        'responsible_role': template['responsible_role'],
                        'estimated_effort_hours': template['estimated_effort_hours'],
                        'confidence': 0.85 if i == 0 else 0.75,
                        'steps': '|'.join(template['steps'][:3]),
                    })
                else:
                    # No template found, create generic recommendation
                    patient_recommendations.append({
                        'patient_key': patient_key,
                        'study_id': study_id,
                        'site_id': site_id,
                        'subject_id': subject_id,
                        'priority_tier': priority_tier,
                        'issue_rank': i + 1,
                        'issue_type': issue['issue_type'],
                        'issue_count': issue['count'],
                        'is_primary_issue': i == 0,
                        'template_id': 'GENERIC-001',
                        'recommended_action': f"Resolve {issue['issue_type'].replace('_', ' ')} issue",
                        'action_description': f"Review and resolve {issue['issue_type']} for this patient",
                        'responsible_role': 'Data Manager',
                        'estimated_effort_hours': 0.5,
                        'confidence': 0.5,
                        'steps': '1. Review issue|2. Take corrective action|3. Verify resolution',
                    })
        
        patient_recs_df = pd.DataFrame(patient_recommendations)
        logger.info(f"Generated {len(patient_recs_df):,} patient-issue recommendations")
        
        # Generate role-based queues
        logger.info("\nGenerating role-based task queues...")
        
        if len(patient_recs_df) > 0:
            role_queue = patient_recs_df.groupby('responsible_role').agg({
                'patient_key': 'count',
                'estimated_effort_hours': 'sum',
            }).reset_index()
            
            role_queue.columns = ['responsible_role', 'task_count', 'total_effort_hours']
            role_queue['avg_effort_per_task'] = role_queue['total_effort_hours'] / role_queue['task_count']
            role_queue = role_queue.sort_values('task_count', ascending=False)
            
            logger.info(f"Role queues generated for {len(role_queue)} roles:")
            for _, row in role_queue.iterrows():
                logger.info(f"   {row['responsible_role']}: {row['task_count']:,} tasks, "
                           f"{row['total_effort_hours']:.1f} hours")
        else:
            role_queue = pd.DataFrame()
        
        # Generate issue type summary
        logger.info("\nGenerating issue type summary...")
        
        if len(patient_recs_df) > 0:
            issue_summary = patient_recs_df.groupby('issue_type').agg({
                'patient_key': 'nunique',
                'template_id': 'first',
                'recommended_action': 'first',
                'responsible_role': 'first',
                'estimated_effort_hours': 'mean',
                'confidence': 'mean',
            }).reset_index()
            
            issue_summary.columns = [
                'issue_type', 'patient_count', 'primary_template_id', 
                'primary_action', 'responsible_role', 'avg_effort_hours', 'avg_confidence'
            ]
            issue_summary = issue_summary.sort_values('patient_count', ascending=False)
        else:
            issue_summary = pd.DataFrame()
        
        return patient_recs_df, role_queue, issue_summary
    
    def learn_from_outcome(
        self,
        issue_id: str,
        resolution_template_id: str,
        outcome: str,  # "success", "partial", "failure"
        resolution_time_days: float,
        feedback_notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update pattern success rates based on actual outcomes.
        This enables live learning from resolved issues.
        
        Per SOLUTION.md L179: "LEARNING: Success rates update daily based on outcomes"
        
        Args:
            issue_id: The issue that was resolved
            resolution_template_id: The template that was applied
            outcome: "success", "partial", or "failure"
            resolution_time_days: Actual time to resolution
            feedback_notes: Optional notes from user
            
        Returns:
            Dict with updated statistics
        """
        # Find the template
        if resolution_template_id not in self.templates:
            logger.warning(f"Template not found: {resolution_template_id}")
            return {"error": f"Template not found: {resolution_template_id}"}
        
        template = self.templates[resolution_template_id]
        
        # Update usage counts
        template.times_used += 1
        
        # Update success count based on outcome
        outcome_weights = {
            "success": 1.0,
            "partial": 0.5,
            "failure": 0.0
        }
        
        if outcome in outcome_weights:
            template.times_successful += outcome_weights[outcome]
        else:
            logger.warning(f"Unknown outcome: {outcome}, treating as partial")
            template.times_successful += 0.5
        
        # Recalculate success rate
        if template.times_used > 0:
            template.success_rate = template.times_successful / template.times_used
        
        # Update average resolution time (exponential moving average)
        if template.avg_resolution_days == 0:
            template.avg_resolution_days = resolution_time_days
        else:
            # EMA with alpha=0.1 for smooth updates
            alpha = 0.1
            template.avg_resolution_days = (
                alpha * resolution_time_days + 
                (1 - alpha) * template.avg_resolution_days
            )
        
        # Update timestamp
        template.updated_at = datetime.now().isoformat()
        
        # Record in resolution history
        history_entry = {
            "issue_id": issue_id,
            "template_id": resolution_template_id,
            "outcome": outcome,
            "resolution_time_days": resolution_time_days,
            "timestamp": datetime.now().isoformat(),
            "feedback_notes": feedback_notes,
            "new_success_rate": template.success_rate,
            "new_avg_resolution_days": template.avg_resolution_days
        }
        self.resolution_history.append(history_entry)
        
        # Persist to PostgreSQL if available
        try:
            self._persist_learning_to_db(history_entry, template)
        except Exception as e:
            logger.warning(f"Failed to persist learning to DB: {e}")
        
        logger.info(
            f"Learned from outcome: {resolution_template_id} - {outcome} "
            f"(success_rate: {template.success_rate:.2%}, "
            f"avg_days: {template.avg_resolution_days:.1f})"
        )
        
        return {
            "template_id": resolution_template_id,
            "outcome": outcome,
            "times_used": template.times_used,
            "times_successful": template.times_successful,
            "success_rate": template.success_rate,
            "avg_resolution_days": template.avg_resolution_days,
            "history_size": len(self.resolution_history)
        }
    
    def _persist_learning_to_db(
        self, 
        history_entry: Dict[str, Any], 
        template: ResolutionTemplate
    ) -> None:
        """Persist learning data to PostgreSQL for durability."""
        try:
            from src.database.pg_writer import get_pg_writer
            import pandas as pd
            
            pg_writer = get_pg_writer()
            
            # Write history entry
            history_df = pd.DataFrame([history_entry])
            pg_writer.write(history_df, 'resolution_learning_history', if_exists='append')
            
            # Update template stats in DB
            template_stats = {
                "template_id": template.template_id,
                "issue_type": template.issue_type,
                "times_used": template.times_used,
                "times_successful": template.times_successful,
                "success_rate": template.success_rate,
                "avg_resolution_days": template.avg_resolution_days,
                "updated_at": template.updated_at
            }
            
            stats_df = pd.DataFrame([template_stats])
            pg_writer.upsert(
                stats_df, 
                'resolution_template_stats', 
                conflict_columns=['template_id']
            )
            
        except ImportError:
            logger.debug("PostgreSQL writer not available, using local storage only")
        except Exception as e:
            logger.warning(f"DB persistence failed: {e}")
    
    def batch_learn_from_outcomes(
        self, 
        outcomes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Process multiple outcome learnings in batch.
        Useful for daily batch updates from resolved issues.
        
        Args:
            outcomes: List of dicts with keys:
                - issue_id, resolution_template_id, outcome, resolution_time_days
                
        Returns:
            Summary of batch learning
        """
        results = {
            "processed": 0,
            "success": 0,
            "partial": 0,
            "failure": 0,
            "errors": 0,
            "templates_updated": set()
        }
        
        for outcome_data in outcomes:
            try:
                result = self.learn_from_outcome(
                    issue_id=outcome_data.get("issue_id", "unknown"),
                    resolution_template_id=outcome_data["resolution_template_id"],
                    outcome=outcome_data["outcome"],
                    resolution_time_days=outcome_data.get("resolution_time_days", 0),
                    feedback_notes=outcome_data.get("feedback_notes")
                )
                
                if "error" not in result:
                    results["processed"] += 1
                    results[outcome_data["outcome"]] = results.get(outcome_data["outcome"], 0) + 1
                    results["templates_updated"].add(outcome_data["resolution_template_id"])
                else:
                    results["errors"] += 1
                    
            except Exception as e:
                logger.error(f"Error processing outcome: {e}")
                results["errors"] += 1
        
        results["templates_updated"] = list(results["templates_updated"])
        
        logger.info(
            f"Batch learning complete: {results['processed']} processed, "
            f"{results['errors']} errors, {len(results['templates_updated'])} templates updated"
        )
        
        return results
    
    def get_template_performance(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Get performance statistics for a specific template."""
        if template_id not in self.templates:
            return None
        
        template = self.templates[template_id]
        
        # Calculate recent performance from history
        recent_outcomes = [
            h for h in self.resolution_history[-100:]
            if h.get("template_id") == template_id
        ]
        
        recent_success_rate = None
        if recent_outcomes:
            recent_successes = sum(
                1 for h in recent_outcomes if h.get("outcome") == "success"
            )
            recent_success_rate = recent_successes / len(recent_outcomes)
        
        return {
            "template_id": template_id,
            "title": template.title,
            "issue_type": template.issue_type,
            "times_used": template.times_used,
            "times_successful": template.times_successful,
            "overall_success_rate": template.success_rate,
            "recent_success_rate": recent_success_rate,
            "avg_resolution_days": template.avg_resolution_days,
            "estimated_effort_hours": template.estimated_effort_hours,
            "recent_outcomes_count": len(recent_outcomes),
            "trending": self._calculate_trend(recent_outcomes)
        }
    
    def _calculate_trend(self, recent_outcomes: List[Dict]) -> str:
        """Calculate if template performance is trending up, down, or stable."""
        if len(recent_outcomes) < 10:
            return "insufficient_data"
        
        # Compare first half vs second half success rates
        mid = len(recent_outcomes) // 2
        first_half = recent_outcomes[:mid]
        second_half = recent_outcomes[mid:]
        
        first_success = sum(1 for h in first_half if h.get("outcome") == "success") / len(first_half)
        second_success = sum(1 for h in second_half if h.get("outcome") == "success") / len(second_half)
        
        diff = second_success - first_success
        
        if diff > 0.1:
            return "improving"
        elif diff < -0.1:
            return "declining"
        else:
            return "stable"

    def get_statistics(self) -> Dict:
        total_templates = len(self.templates)
        by_type = defaultdict(int)
        by_role = defaultdict(int)
        
        for t in self.templates.values():
            by_type[t.issue_type] += 1
            by_role[t.responsible_role] += 1
        
        return {
            'total_templates': total_templates,
            'by_issue_type': dict(by_type),
            'by_responsible_role': dict(by_role),
            'total_resolutions_recorded': sum(t.times_used for t in self.templates.values()),
            'history_count': len(self.resolution_history)
        }
    
    def save(self, output_dir: Path):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        templates_data = {tid: asdict(t) for tid, t in self.templates.items()}
        with open(output_dir / 'resolution_templates.json', 'w') as f:
            json.dump(templates_data, f, indent=2, default=str)
        
        if self.embeddings:
            np.savez(output_dir / 'template_embeddings.npz',
                    **{k: v for k, v in self.embeddings.items()})
        
        with open(output_dir / 'resolution_history.json', 'w') as f:
            json.dump(self.resolution_history, f, indent=2, default=str)
        
        with open(output_dir / 'genome_config.json', 'w') as f:
            json.dump(asdict(self.config), f, indent=2)
        
        logger.info(f"Genome saved to {output_dir}")


# =============================================================================
# MAIN RUNNER FUNCTION
# =============================================================================

def run_resolution_genome(
    issues_path: Path,
    output_dir: Path,
    config: Optional[ResolutionGenomeConfig] = None
) -> Dict[str, Any]:
    """Run Resolution Genome pipeline with Assignment Loop."""
    logger.info("=" * 70)
    logger.info("SANCHALAK AI - RESOLUTION GENOME ENGINE v1.2")
    logger.info("=" * 70)
    
    start_time = datetime.now()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    genome_dir = output_dir / 'resolution_genome'
    genome_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize genome
    genome = ResolutionGenome(config=config or ResolutionGenomeConfig())
    genome.initialize()
    
    # Load issues from PostgreSQL
    logger.info(f"\nLoading issues from PostgreSQL...")
    
    from src.database.pg_data_service import PostgreSQLDataService
    db_service = PostgreSQLDataService()
    issues_df = db_service.get_patient_issues()
    
    logger.info(f"Loaded {len(issues_df):,} patient records")
    logger.info(f"Columns: {len(issues_df.columns)}")
    
    # Run assignment loop
    patient_recs_df, role_queue_df, issue_summary_df = genome.assign_resolutions_to_patients(issues_df)
    
    # Save outputs
    logger.info("\nSaving outputs...")
    
    if len(patient_recs_df) > 0:
        # Save to PostgreSQL
        from src.database.pg_writer import safe_to_postgres
        success, msg = safe_to_postgres(patient_recs_df, 'patient_recommendations', if_exists='replace')
        if success:
            logger.info(f"✅ Saved patient_recommendations to PostgreSQL: {len(patient_recs_df):,} rows")
        else:
            logger.warning(f"PostgreSQL write failed, using CSV fallback: {msg}")
        
        patient_recs_df.to_csv(genome_dir / 'patient_recommendations.csv', index=False)

        
        primary_recs = patient_recs_df[patient_recs_df['is_primary_issue'] == True]
        primary_recs.to_csv(genome_dir / 'patient_primary_recommendations.csv', index=False)
    
    if len(role_queue_df) > 0:
        role_queue_df.to_csv(genome_dir / 'role_task_queue.csv', index=False)
    
    if len(issue_summary_df) > 0:
        issue_summary_df.to_csv(genome_dir / 'issue_type_summary.csv', index=False)
    
    genome.save(genome_dir)
    
    # Generate summary
    stats = genome.get_statistics()
    
    role_counts = {}
    if len(patient_recs_df) > 0:
        role_counts = patient_recs_df['responsible_role'].value_counts().to_dict()
    
    summary = {
        'run_timestamp': datetime.now().isoformat(),
        'version': '1.2',
        'genome_statistics': stats,
        'assignment_results': {
            'patients_with_issues': int(patient_recs_df['patient_key'].nunique()) if len(patient_recs_df) > 0 else 0,
            'total_recommendations': int(len(patient_recs_df)),
            'primary_recommendations': int(len(patient_recs_df[patient_recs_df['is_primary_issue'] == True])) if len(patient_recs_df) > 0 else 0,
            'issue_types_covered': int(len(issue_summary_df)) if len(issue_summary_df) > 0 else 0,
            'roles_assigned': len(role_counts),
            'recommendations_by_role': {k: int(v) for k, v in role_counts.items()},
        },
        'effort_estimation': {
            'total_hours': float(patient_recs_df['estimated_effort_hours'].sum()) if len(patient_recs_df) > 0 else 0,
            'avg_hours_per_patient': float(patient_recs_df.groupby('patient_key')['estimated_effort_hours'].sum().mean()) if len(patient_recs_df) > 0 else 0,
        },
        'duration_seconds': (datetime.now() - start_time).total_seconds()
    }
    
    with open(genome_dir / 'genome_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("RESOLUTION GENOME COMPLETE")
    logger.info("=" * 70)
    
    logger.info(f"\nGenome Statistics:")
    logger.info(f"  Total Templates:    {stats['total_templates']}")
    logger.info(f"  Issue Types:        {len(stats['by_issue_type'])}")
    logger.info(f"  Responsible Roles:  {len(stats['by_responsible_role'])}")
    
    logger.info(f"\nAssignment Results:")
    logger.info(f"  Patients with Issues:    {summary['assignment_results']['patients_with_issues']:,}")
    logger.info(f"  Total Recommendations:   {summary['assignment_results']['total_recommendations']:,}")
    logger.info(f"  Primary Recommendations: {summary['assignment_results']['primary_recommendations']:,}")
    logger.info(f"  Issue Types Covered:     {summary['assignment_results']['issue_types_covered']}")
    
    if role_counts:
        logger.info(f"\nRecommendations by Role:")
        for role, count in sorted(role_counts.items(), key=lambda x: -x[1]):
            logger.info(f"    {role}: {count:,}")
    
    logger.info(f"\nEffort Estimation:")
    logger.info(f"  Total Hours:           {summary['effort_estimation']['total_hours']:,.1f}")
    logger.info(f"  Avg Hours per Patient: {summary['effort_estimation']['avg_hours_per_patient']:.2f}")
    
    logger.info(f"\nDuration: {summary['duration_seconds']:.2f} seconds")
    logger.info(f"Output: {genome_dir}")
    
    return summary


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from config.settings import DATA_PROCESSED

# PostgreSQL data access
try:
    from src.database.pg_data_service import get_data_service
    from src.database.pg_writer import get_pg_writer
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

    
    # issues_path no longer needed as we pull from DB, but keeping signature for compatibility
    issues_path = Path("DB") 
    output_dir = DATA_PROCESSED / 'analytics'
    
    run_resolution_genome(issues_path, output_dir)