"""
SANCHALAK AI - Configuration Settings
==============================================
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List

# ============================================
# PATH CONFIGURATION
# ============================================

# Project root (auto-detected)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Data paths
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_OUTPUTS = PROJECT_ROOT / "data" / "outputs"

# Output paths
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"

# Create directories if they don't exist
for dir_path in [DATA_RAW, DATA_PROCESSED, DATA_OUTPUTS, LOGS_DIR, MODELS_DIR, KNOWLEDGE_BASE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# ============================================
# FILE PATTERNS CONFIGURATION
# ============================================

@dataclass
class FilePatternConfig:
    """Configuration for identifying file types from filenames."""
    
    # Patterns for each data source type
    # Order matters for priority matching
    patterns: Dict[str, List[str]] = field(default_factory=lambda: {
        'cpid_edc_metrics': [
            'CPID_EDC_Metrics', 'CPID_EDC Metrics', 'CPID EDC Metrics',
            'EDC_Metrics', 'EDC Metrics'
        ],
        'visit_projection': [
            'Visit_Projection_Tracker', 'Visit Projection Tracker',
            'Visit_Projection', 'Visit Projection',
            'Missing_visit', 'Missing visit'
        ],
        'missing_lab_ranges': [
            'Missing_Lab_Name_and_Missing_Ranges',
            'Missing_Lab_Name_And_Missing_Ranges',
            'Missing_Lab_Name', 'Missing Lab',
            'Missing_LNR', 'Missing LNR',
            'Missing Lab & Range'
        ],
        'sae_dashboard': [
            'eSAE_Dashboard', 'eSAE Dashboard', 'eSAE_dashboard',
            'SAE_Dashboard', 'SAE Dashboard',
            'eSAE_updated', 'eSAE dashboard_DM',
            'SAE Dashboard_Standard', 'SAE Dashboard_DM'
        ],
        'inactivated_forms': [
            'Inactivated Folders, Forms and Records',
            'Inactivated Forms, Folders and Records',
            'Inactivated Forms and Folders',
            'Inactivated Form Folder',
            'Inactivated page', 'Inactivated_',
            'Inactivated report', 'inactivated'
        ],
        'missing_pages': [
            'Global_Missing_Pages_Report', 'Global Missing Pages',
            'Missing_Pages_Report', 'Missing Pages Report',
            'Missing_Pages', 'Missing_Page'
        ],
        'compiled_edrr': [
            'Compiled_EDRR', 'Compiled EDRR', 'EDRR'
        ],
        'coding_meddra': [
            'GlobalCodingReport_MedDRA', 'GlobalCoding Report_MedDRA',
            'Global Coding Report_Medra', 'MedDRA', 'Medra', 'MEDDRA'
        ],
        'coding_whodrug': [
            'GlobalCodingReport_WHODD', 'GlobalCoding Report_WHODD',
            'GlobalCodingReport_WHODrug', 'GlobalCoding Report_WHODrug',
            'Global Coding Report_WHODD', 'WHODrug', 'WHODD', 'WHOdra'
        ]
    })
    
    # Exclusion rules to prevent misidentification
    exclusions: Dict[str, List[str]] = field(default_factory=lambda: {
        'visit_projection': ['Missing_Lab', 'LNR', 'Lab_Name', 'Missing_Pages'],
        'missing_pages': ['Missing_Lab', 'LNR', 'Lab_Name']
    })


# ============================================
# DATA QUALITY INDEX (DQI) CONFIGURATION
# ============================================

@dataclass
class DQIConfig:
    """Configuration for DQI calculation."""
    
    # Component weights (must sum to 1.0)
    weights: Dict[str, float] = field(default_factory=lambda: {
        'safety_score': 0.25,      # SAE discrepancies
        'query_score': 0.20,       # Open queries
        'completeness_score': 0.15, # Missing visits/pages
        'coding_score': 0.12,      # Uncoded terms
        'lab_score': 0.10,         # Lab issues
        'sdv_score': 0.08,         # SDV completion
        'signature_score': 0.05,   # Overdue signatures
        'edrr_score': 0.05         # Third-party issues
    })
    
    # Age multipliers (days outstanding)
    age_multipliers: Dict[str, float] = field(default_factory=lambda: {
        '0-7': 1.0,
        '8-14': 1.1,
        '15-30': 1.3,
        '31-60': 1.5,
        '60+': 1.6
    })
    
    # Criticality multipliers
    criticality_multipliers: Dict[str, float] = field(default_factory=lambda: {
        'safety': 1.5,
        'high': 1.3,
        'medium': 1.1,
        'low': 1.0
    })
    
    # DQI bands
    bands: Dict[str, tuple] = field(default_factory=lambda: {
        'pristine': (95, 100),
        'excellent': (85, 94.99),
        'good': (75, 84.99),
        'moderate': (65, 74.99),
        'concerning': (50, 64.99),
        'critical': (0, 49.99)
    })


# ============================================
# CLEAN PATIENT CRITERIA
# ============================================

@dataclass
class CleanPatientConfig:
    """Configuration for Clean Patient derivation."""
    
    # Tier 1: Clinical Clean (Hard Blocks)
    tier1_criteria: Dict[str, str] = field(default_factory=lambda: {
        'missing_visits': '== 0',
        'missing_pages': '== 0',
        'open_queries': '== 0',
        'sdv_complete': '== 100',
        'signatures_complete': '== True',
        'coding_complete': '== True',
        'no_broken_signatures': '== True'
    })
    
    # Tier 2: Operational Clean (Soft Blocks)
    tier2_criteria: Dict[str, str] = field(default_factory=lambda: {
        'lab_issues': '== 0',
        'sae_pending': '== 0',
        'edrr_issues': '== 0',
        'overdue_crfs': '== 0',
        'inactivated_issues': '== 0'
    })


# ============================================
# LOGGING CONFIGURATION
# ============================================

LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {module}:{function}:{line} | {message}"
LOG_ROTATION = "10 MB"
LOG_RETENTION = "7 days"


@dataclass
class ReportQualityConfig:
    """Thresholds for report quality and conditional formatting."""
    dqi_target: float = 90.0
    clean_rate_target: float = 0.70
    query_resolution_days_target: float = 7.0
    data_entry_lag_target: float = 3.0
    signature_completion_target: float = 1.0  # 100%
    sdv_completion_target: float = 0.95
    missing_pages_threshold_pct: float = 5.0
    sae_reconciliation_target: float = 1.0

REPORT_QUALITY = ReportQualityConfig()

# ============================================
# EXPORT CONFIGURATION INSTANCES
# ============================================


FILE_PATTERNS = FilePatternConfig()
DQI_CONFIG = DQIConfig()
CLEAN_PATIENT_CONFIG = CleanPatientConfig()