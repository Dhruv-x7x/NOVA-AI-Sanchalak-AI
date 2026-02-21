# src/agents/resolver_enhanced.py
"""
SANCHALAK AI - Enhanced RESOLVER Agent v1.0

Purpose: Generate prioritized action plans, integrate resolution genome,
         calculate cascade impact, and provide role-based task assignments.

Features:
- Resolution genome integration for proven solutions
- Action plan generation with dependencies
- Cascade impact analysis
- Priority-based task sequencing
- Effort estimation and ROI calculation
- Role-based task assignment
- Quick wins identification
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from enum import Enum
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ActionPriority(Enum):
    """Priority levels for actions."""
    CRITICAL = "critical"    # Must do immediately (safety, regulatory)
    HIGH = "high"            # Do within 3 days
    MEDIUM = "medium"        # Do within 7 days
    LOW = "low"              # Do within 14 days
    OPTIONAL = "optional"    # Nice to have


class ActionStatus(Enum):
    """Status of an action."""
    PENDING = "pending"
    APPROVED = "approved"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class ActionCategory(Enum):
    """Categories of actions."""
    DATA_QUALITY = "data_quality"
    SAFETY = "safety"
    MONITORING = "monitoring"
    SITE_MANAGEMENT = "site_management"
    PROCESS = "process"
    RESOURCE = "resource"
    COMMUNICATION = "communication"
    TRAINING = "training"


class ResponsibleRole(Enum):
    """Roles responsible for actions."""
    CRA = "CRA"
    DATA_MANAGER = "Data Manager"
    SITE_COORDINATOR = "Site Coordinator"
    SAFETY_DATA_MANAGER = "Safety Data Manager"
    SAFETY_PHYSICIAN = "Safety Physician"
    MEDICAL_CODER = "Medical Coder"
    STUDY_LEAD = "Study Lead"
    CTM = "Clinical Trial Manager"
    SITE = "Site"
    PI = "Principal Investigator"


@dataclass
class ResolutionTemplate:
    """A resolution template from the genome."""
    template_id: str
    issue_type: str
    title: str
    description: str
    steps: List[str]
    responsible_role: ResponsibleRole
    effort_hours: float
    success_rate: float
    prerequisites: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    def __hash__(self):
        """Make ResolutionTemplate hashable by template_id."""
        return hash(self.template_id)
    
    def __eq__(self, other):
        """Equality based on template_id."""
        if isinstance(other, ResolutionTemplate):
            return self.template_id == other.template_id
        return False
    
    def to_dict(self) -> Dict:
        return {
            "template_id": self.template_id,
            "issue_type": self.issue_type,
            "title": self.title,
            "description": self.description,
            "steps": self.steps,
            "responsible_role": self.responsible_role.value,
            "effort_hours": self.effort_hours,
            "success_rate": self.success_rate,
            "prerequisites": self.prerequisites,
            "tags": self.tags
        }


@dataclass
class Action:
    """A single action in an action plan."""
    action_id: str
    title: str
    description: str
    category: ActionCategory
    priority: ActionPriority
    responsible_role: ResponsibleRole
    entity_id: str  # patient_key, site_id, study_id
    entity_type: str  # patient, site, study
    issue_type: str
    steps: List[str] = field(default_factory=list)
    effort_hours: float = 1.0
    impact_score: float = 0.0  # 0-100
    confidence: float = 0.0  # 0-1
    dependencies: List[str] = field(default_factory=list)  # action_ids
    unlocks: List[str] = field(default_factory=list)  # What this action enables
    status: ActionStatus = ActionStatus.PENDING
    due_date: Optional[datetime] = None
    template_id: Optional[str] = None
    success_rate: float = 0.85
    requires_approval: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            "action_id": self.action_id,
            "title": self.title,
            "description": self.description,
            "category": self.category.value,
            "priority": self.priority.value,
            "responsible_role": self.responsible_role.value,
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "issue_type": self.issue_type,
            "steps": self.steps,
            "effort_hours": self.effort_hours,
            "impact_score": round(self.impact_score, 2),
            "confidence": round(self.confidence, 3),
            "dependencies": self.dependencies,
            "unlocks": self.unlocks,
            "status": self.status.value,
            "due_date": self.due_date.strftime("%Y-%m-%d") if self.due_date else None,
            "template_id": self.template_id,
            "success_rate": self.success_rate,
            "requires_approval": self.requires_approval,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class CascadeImpact:
    """Impact analysis of resolving an issue."""
    source_issue: str
    source_entity: str
    direct_impact: Dict[str, Any]
    cascade_effects: List[Dict[str, Any]]
    total_issues_resolved: int
    total_patients_unblocked: int
    dqi_improvement: float
    db_lock_acceleration_days: float
    effort_hours: float
    roi_score: float  # Impact per hour of effort
    
    def to_dict(self) -> Dict:
        return {
            "source_issue": self.source_issue,
            "source_entity": self.source_entity,
            "direct_impact": self.direct_impact,
            "cascade_effects": self.cascade_effects,
            "total_issues_resolved": self.total_issues_resolved,
            "total_patients_unblocked": self.total_patients_unblocked,
            "dqi_improvement": round(self.dqi_improvement, 2),
            "db_lock_acceleration_days": round(self.db_lock_acceleration_days, 1),
            "effort_hours": self.effort_hours,
            "roi_score": round(self.roi_score, 2)
        }


@dataclass
class ActionPlan:
    """A complete action plan with prioritized actions."""
    plan_id: str
    title: str
    description: str
    entity_id: str
    entity_type: str
    actions: List[Action] = field(default_factory=list)
    quick_wins: List[Action] = field(default_factory=list)
    cascade_impacts: List[CascadeImpact] = field(default_factory=list)
    total_effort_hours: float = 0.0
    expected_impact: Dict[str, Any] = field(default_factory=dict)
    timeline_days: int = 0
    risk_factors: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    
    def add_action(self, action: Action):
        """Add an action and update totals."""
        self.actions.append(action)
        self.total_effort_hours += action.effort_hours
        if action.effort_hours <= 2 and action.impact_score >= 50:
            self.quick_wins.append(action)
    
    def get_actions_by_priority(self) -> Dict[str, List[Action]]:
        """Group actions by priority."""
        result = {}
        for action in self.actions:
            priority = action.priority.value
            if priority not in result:
                result[priority] = []
            result[priority].append(action)
        return result
    
    def get_actions_by_role(self) -> Dict[str, List[Action]]:
        """Group actions by responsible role."""
        result = {}
        for action in self.actions:
            role = action.responsible_role.value
            if role not in result:
                result[role] = []
            result[role].append(action)
        return result
    
    def to_dict(self) -> Dict:
        return {
            "plan_id": self.plan_id,
            "title": self.title,
            "description": self.description,
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "actions": [a.to_dict() for a in self.actions],
            "quick_wins": [a.to_dict() for a in self.quick_wins],
            "cascade_impacts": [c.to_dict() for c in self.cascade_impacts],
            "total_effort_hours": round(self.total_effort_hours, 1),
            "expected_impact": self.expected_impact,
            "timeline_days": self.timeline_days,
            "risk_factors": self.risk_factors,
            "assumptions": self.assumptions,
            "actions_by_priority": {k: len(v) for k, v in self.get_actions_by_priority().items()},
            "actions_by_role": {k: len(v) for k, v in self.get_actions_by_role().items()},
            "created_at": self.created_at.isoformat()
        }


@dataclass
class ResolverResult:
    """Complete result of a resolution operation."""
    result_id: str
    query: str
    action_plans: List[ActionPlan] = field(default_factory=list)
    top_actions: List[Action] = field(default_factory=list)
    cascade_analysis: List[CascadeImpact] = field(default_factory=list)
    summary: str = ""
    recommendations: List[str] = field(default_factory=list)
    total_effort_hours: float = 0.0
    expected_outcomes: Dict[str, Any] = field(default_factory=dict)
    data_sources: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        return {
            "result_id": self.result_id,
            "query": self.query,
            "action_plans": [p.to_dict() for p in self.action_plans],
            "top_actions": [a.to_dict() for a in self.top_actions],
            "cascade_analysis": [c.to_dict() for c in self.cascade_analysis],
            "summary": self.summary,
            "recommendations": self.recommendations,
            "total_effort_hours": round(self.total_effort_hours, 1),
            "expected_outcomes": self.expected_outcomes,
            "data_sources": self.data_sources,
            "duration_seconds": round(self.duration_seconds, 2),
            "created_at": self.created_at.isoformat()
        }


class ResolutionGenome:
    """Resolution genome with proven solution templates."""
    
    # Master template library
    TEMPLATES = {
        'sdv_incomplete': [
            ResolutionTemplate(
                template_id="RES-SDV-001",
                issue_type="sdv_incomplete",
                title="Focused SDV Visit",
                description="Schedule dedicated SDV visit to complete source data verification",
                steps=[
                    "Review outstanding SDV items in EDC",
                    "Prepare SDV checklist for visit",
                    "Schedule on-site or remote visit with site",
                    "Complete SDV for all outstanding CRFs",
                    "Document SDV completion in monitoring report"
                ],
                responsible_role=ResponsibleRole.CRA,
                effort_hours=4.0,
                success_rate=0.92,
                prerequisites=["Source documents available", "Site access confirmed"],
                tags=["monitoring", "sdv", "on-site"]
            ),
            ResolutionTemplate(
                template_id="RES-SDV-002",
                issue_type="sdv_incomplete",
                title="Remote SDV Session",
                description="Conduct remote SDV using certified copies or remote access",
                steps=[
                    "Request certified copies from site",
                    "Schedule remote verification session",
                    "Complete SDV using remote access tools",
                    "Document verification in EDC"
                ],
                responsible_role=ResponsibleRole.CRA,
                effort_hours=2.5,
                success_rate=0.85,
                prerequisites=["Remote access available", "Certified copies provided"],
                tags=["remote", "sdv"]
            )
        ],
        'open_queries': [
            ResolutionTemplate(
                template_id="RES-QRY-001",
                issue_type="open_queries",
                title="Query Resolution Call",
                description="Schedule call with site to resolve open queries",
                steps=[
                    "Prepare list of open queries with details",
                    "Schedule call with site coordinator",
                    "Walk through each query with site",
                    "Provide guidance on expected responses",
                    "Set deadline for query responses"
                ],
                responsible_role=ResponsibleRole.DATA_MANAGER,
                effort_hours=1.5,
                success_rate=0.88,
                tags=["query", "communication"]
            ),
            ResolutionTemplate(
                template_id="RES-QRY-002",
                issue_type="open_queries",
                title="Batch Query Reminder",
                description="Send consolidated reminder for all open queries",
                steps=[
                    "Generate query aging report",
                    "Prepare email with query summary",
                    "Send to site with response deadline",
                    "Follow up if no response in 48 hours"
                ],
                responsible_role=ResponsibleRole.DATA_MANAGER,
                effort_hours=0.5,
                success_rate=0.75,
                tags=["query", "email", "batch"]
            ),
            ResolutionTemplate(
                template_id="RES-QRY-003",
                issue_type="open_queries",
                title="Query Escalation to PI",
                description="Escalate unresolved queries to Principal Investigator",
                steps=[
                    "Identify queries >14 days old",
                    "Prepare escalation summary for PI",
                    "Send escalation email to PI",
                    "Copy site coordinator and CRA",
                    "Track PI response"
                ],
                responsible_role=ResponsibleRole.CRA,
                effort_hours=1.0,
                success_rate=0.90,
                prerequisites=["Queries aged >14 days", "Initial reminders sent"],
                tags=["query", "escalation", "PI"]
            )
        ],
        'signature_gaps': [
            ResolutionTemplate(
                template_id="RES-SIG-001",
                issue_type="signature_gaps",
                title="PI Signature Session",
                description="Schedule dedicated PI signature session",
                steps=[
                    "Generate list of unsigned CRFs",
                    "Contact PI to schedule signature time",
                    "Prepare signature queue in EDC",
                    "Support PI during signature session",
                    "Verify all signatures completed"
                ],
                responsible_role=ResponsibleRole.SITE_COORDINATOR,
                effort_hours=2.0,
                success_rate=0.95,
                tags=["signature", "PI", "session"]
            ),
            ResolutionTemplate(
                template_id="RES-SIG-002",
                issue_type="signature_gaps",
                title="Signature Reminder",
                description="Send signature reminder to PI/Sub-I",
                steps=[
                    "Generate unsigned CRF report",
                    "Send reminder email to PI",
                    "Include link to EDC signature page",
                    "Follow up in 24 hours if not completed"
                ],
                responsible_role=ResponsibleRole.SITE_COORDINATOR,
                effort_hours=0.25,
                success_rate=0.70,
                tags=["signature", "reminder"]
            )
        ],
        'broken_signatures': [
            ResolutionTemplate(
                template_id="RES-BRK-001",
                issue_type="broken_signatures",
                title="Re-signature After Data Correction",
                description="Obtain re-signature after data was corrected post-signature",
                steps=[
                    "Identify CRFs with broken signatures",
                    "Review what data changes caused the break",
                    "Notify PI of re-signature requirement",
                    "PI reviews changes and re-signs",
                    "Verify signature completion"
                ],
                responsible_role=ResponsibleRole.SITE_COORDINATOR,
                effort_hours=0.5,
                success_rate=0.92,
                prerequisites=["Data corrections completed"],
                tags=["signature", "re-sign"]
            )
        ],
        'sae_dm_pending': [
            ResolutionTemplate(
                template_id="RES-SAE-001",
                issue_type="sae_dm_pending",
                title="SAE Reconciliation",
                description="Reconcile SAE data between EDC and safety database",
                steps=[
                    "Pull SAE listing from both systems",
                    "Identify discrepancies",
                    "Investigate root cause of each discrepancy",
                    "Make corrections in appropriate system",
                    "Document reconciliation completion"
                ],
                responsible_role=ResponsibleRole.SAFETY_DATA_MANAGER,
                effort_hours=3.0,
                success_rate=0.90,
                tags=["safety", "reconciliation", "SAE"]
            ),
            ResolutionTemplate(
                template_id="RES-SAE-002",
                issue_type="sae_dm_pending",
                title="SAE Data Entry Verification",
                description="Verify SAE data entry accuracy and completeness",
                steps=[
                    "Compare SAE source documents to EDC entry",
                    "Verify all required fields completed",
                    "Check coding accuracy",
                    "Raise queries for any discrepancies",
                    "Document verification"
                ],
                responsible_role=ResponsibleRole.SAFETY_DATA_MANAGER,
                effort_hours=1.5,
                success_rate=0.88,
                tags=["safety", "verification"]
            )
        ],
        'sae_safety_pending': [
            ResolutionTemplate(
                template_id="RES-SAF-001",
                issue_type="sae_safety_pending",
                title="Medical Review of SAE",
                description="Complete medical review and causality assessment",
                steps=[
                    "Review SAE narrative and source documents",
                    "Assess relatedness to study treatment",
                    "Complete causality assessment",
                    "Document medical opinion",
                    "Sign off SAE review"
                ],
                responsible_role=ResponsibleRole.SAFETY_PHYSICIAN,
                effort_hours=2.0,
                success_rate=0.95,
                prerequisites=["SAE data entry complete"],
                tags=["safety", "medical", "review"]
            )
        ],
        'missing_visits': [
            ResolutionTemplate(
                template_id="RES-VIS-001",
                issue_type="missing_visits",
                title="Schedule Missed Visit",
                description="Schedule patient visit to complete missed assessments",
                steps=[
                    "Contact patient to schedule visit",
                    "Confirm visit date and time",
                    "Prepare visit checklist",
                    "Complete visit assessments",
                    "Enter data in EDC"
                ],
                responsible_role=ResponsibleRole.SITE_COORDINATOR,
                effort_hours=2.0,
                success_rate=0.80,
                tags=["visit", "scheduling"]
            ),
            ResolutionTemplate(
                template_id="RES-VIS-002",
                issue_type="missing_visits",
                title="Protocol Deviation Documentation",
                description="Document missed visit as protocol deviation if not recoverable",
                steps=[
                    "Confirm visit cannot be rescheduled",
                    "Assess impact on patient data",
                    "Complete protocol deviation form",
                    "Submit for review",
                    "Update EDC accordingly"
                ],
                responsible_role=ResponsibleRole.CRA,
                effort_hours=1.0,
                success_rate=1.0,
                tags=["deviation", "documentation"]
            )
        ],
        'missing_pages': [
            ResolutionTemplate(
                template_id="RES-PG-001",
                issue_type="missing_pages",
                title="Complete Missing CRF Pages",
                description="Enter data for missing CRF pages",
                steps=[
                    "Identify which pages are missing",
                    "Locate source documents",
                    "Enter data in EDC",
                    "Mark pages for review/signature"
                ],
                responsible_role=ResponsibleRole.SITE_COORDINATOR,
                effort_hours=1.5,
                success_rate=0.90,
                tags=["data_entry", "CRF"]
            )
        ],
        'meddra_uncoded': [
            ResolutionTemplate(
                template_id="RES-MED-001",
                issue_type="meddra_uncoded",
                title="Medical Term Coding",
                description="Code adverse event terms to MedDRA dictionary",
                steps=[
                    "Review verbatim term",
                    "Search MedDRA dictionary",
                    "Select appropriate PT/LLT",
                    "Apply coding",
                    "Document coding rationale if needed"
                ],
                responsible_role=ResponsibleRole.MEDICAL_CODER,
                effort_hours=0.1,
                success_rate=0.95,
                tags=["coding", "MedDRA"]
            ),
            ResolutionTemplate(
                template_id="RES-MED-002",
                issue_type="meddra_uncoded",
                title="Coding Query to Site",
                description="Send query to site for clarification on ambiguous terms",
                steps=[
                    "Identify ambiguous verbatim term",
                    "Prepare clarification query",
                    "Send query to site",
                    "Code once clarification received"
                ],
                responsible_role=ResponsibleRole.MEDICAL_CODER,
                effort_hours=0.25,
                success_rate=0.85,
                prerequisites=["Initial coding attempt failed"],
                tags=["coding", "query"]
            )
        ],
        'whodrug_uncoded': [
            ResolutionTemplate(
                template_id="RES-WHO-001",
                issue_type="whodrug_uncoded",
                title="Drug Name Coding",
                description="Code medication to WHODrug dictionary",
                steps=[
                    "Review verbatim drug name",
                    "Search WHODrug dictionary",
                    "Select appropriate drug code",
                    "Apply coding"
                ],
                responsible_role=ResponsibleRole.MEDICAL_CODER,
                effort_hours=0.1,
                success_rate=0.92,
                tags=["coding", "WHODrug"]
            )
        ],
        'lab_issues': [
            ResolutionTemplate(
                template_id="RES-LAB-001",
                issue_type="lab_issues",
                title="Lab Range Configuration",
                description="Configure missing laboratory reference ranges",
                steps=[
                    "Identify lab with missing ranges",
                    "Obtain reference ranges from lab",
                    "Configure ranges in system",
                    "Validate configuration"
                ],
                responsible_role=ResponsibleRole.DATA_MANAGER,
                effort_hours=2.0,
                success_rate=0.95,
                tags=["lab", "configuration"]
            )
        ],
        'edrr_issues': [
            ResolutionTemplate(
                template_id="RES-EDR-001",
                issue_type="edrr_issues",
                title="Third-Party Data Reconciliation",
                description="Reconcile discrepancies with external data sources",
                steps=[
                    "Identify reconciliation discrepancies",
                    "Review source data from third party",
                    "Compare with EDC data",
                    "Resolve discrepancies",
                    "Document reconciliation"
                ],
                responsible_role=ResponsibleRole.DATA_MANAGER,
                effort_hours=1.5,
                success_rate=0.85,
                tags=["reconciliation", "third_party"]
            )
        ],
        'inactivated_forms': [
            ResolutionTemplate(
                template_id="RES-INA-001",
                issue_type="inactivated_forms",
                title="Review Inactivated Forms",
                description="Review and validate reason for form inactivation",
                steps=[
                    "Review list of inactivated forms",
                    "Verify inactivation reason is documented",
                    "Confirm no data loss",
                    "Document review completion"
                ],
                responsible_role=ResponsibleRole.DATA_MANAGER,
                effort_hours=0.5,
                success_rate=0.98,
                tags=["forms", "review"]
            )
        ],
        'high_query_volume': [
            ResolutionTemplate(
                template_id="RES-HQV-001",
                issue_type="high_query_volume",
                title="Site Retraining",
                description="Provide retraining to address high query volume",
                steps=[
                    "Analyze query patterns by type",
                    "Identify training gaps",
                    "Prepare targeted training materials",
                    "Conduct training session with site",
                    "Monitor query rate post-training"
                ],
                responsible_role=ResponsibleRole.CRA,
                effort_hours=3.0,
                success_rate=0.80,
                tags=["training", "query"]
            )
        ]
    }
    
    def __init__(self):
        self.templates = self.TEMPLATES
        self._build_indices()
        logger.info(f"ResolutionGenome initialized with {self._count_templates()} templates")
    
    def _build_indices(self):
        """Build search indices for O(1) lookup."""
        self._tag_index = {}
        self._text_index = [] # List of (text, template) for scanning
        
        for issue_type, templates in self.templates.items():
            for tmpl in templates:
                # Index by tags
                for tag in tmpl.tags:
                    tag_lower = tag.lower()
                    if tag_lower not in self._tag_index:
                        self._tag_index[tag_lower] = []
                    self._tag_index[tag_lower].append(tmpl)
                    
                # Store text for broader search
                full_text = f"{tmpl.title} {tmpl.description} {' '.join(tmpl.tags)}".lower()
                self._text_index.append((full_text, tmpl))

    def _count_templates(self) -> int:
        return sum(len(templates) for templates in self.templates.values())
    
    def get_templates_for_issue(self, issue_type: str) -> List[ResolutionTemplate]:
        """Get all templates for an issue type."""
        return self.templates.get(issue_type, [])

    
    def get_best_template(self, issue_type: str) -> Optional[ResolutionTemplate]:
        """Get the best template (highest success rate) for an issue type."""
        templates = self.get_templates_for_issue(issue_type)
        if templates:
            return max(templates, key=lambda t: t.success_rate)
        return None
    
    def search_templates(self, query: str) -> List[ResolutionTemplate]:
        """Search templates by keyword using optimized indices."""
        query_lower = query.lower()
        results = set()
        
        # O(1) Tag Exact Match
        if query_lower in self._tag_index:
            results.update(self._tag_index[query_lower])
            
        # Linear scan on pre-computed text strings (faster than accessing object props)
        for text, tmpl in self._text_index:
            if query_lower in text:
                results.add(tmpl)
        
        return list(results)


class ResolverDataLoader:
    """Loads data for the resolver agent from PostgreSQL."""
    
    def __init__(self):
        try:
             from src.database.pg_data_service import PostgreSQLDataService
             self.db_service = PostgreSQLDataService()
             logger.info("ResolverDataLoader connected to PostgreSQL")
        except ImportError:
             logger.error("PostgreSQLDataService not available")
             self.db_service = None
        
        self._cache: Dict[str, pd.DataFrame] = {}
    
    def _load_from_db(self, name: str, table_name: str) -> Optional[pd.DataFrame]:
        if name in self._cache:
            return self._cache[name]
        
        if self.db_service:
            try:
                # Basic select * from table
                # Ensure the table names match your migration strategy
                df = self.db_service.execute_query(f"SELECT * FROM {table_name}")
                self._cache[name] = df
                logger.info(f"Loaded {name} from DB: {len(df)} rows")
                return df
            except Exception as e:
                logger.error(f"Error loading {name} from DB: {e}")
                return None
        return None
    
    @property
    def patient_issues(self) -> Optional[pd.DataFrame]:
        # Utilizing existing method if available or falling back to query
        if self.db_service:
            return self.db_service.get_patient_issues()
        return None
    
    @property
    def patient_cascade(self) -> Optional[pd.DataFrame]:
        # Using execute_query as there might not be a dedicated method yet
        # Assuming table name 'patient_cascade_analysis' from previous Parquet filename
        return self._load_from_db("patient_cascade", "patient_cascade_analysis")
    
    @property
    def patient_clean_status(self) -> Optional[pd.DataFrame]:
        if self.db_service:
             return self.db_service.get_patient_clean_status()
        return None
    
    @property
    def site_benchmarks(self) -> Optional[pd.DataFrame]:
        if self.db_service:
            return self.db_service.get_site_benchmarks()
        return None
    
    def get_site_issues(self, site_id: str) -> Dict[str, Any]:
        """Get issue summary for a site."""
        issues_df = self.patient_issues
        if issues_df is None or issues_df.empty:
            return {}
        
        site_data = issues_df[issues_df['site_id'] == site_id]
        if site_data.empty:
            return {}
        
        result = {
            'patient_count': len(site_data),
            'patients_with_issues': int(site_data['has_issues'].sum()) if 'has_issues' in site_data.columns else 0,
            'total_issues': int(site_data['total_issues'].sum()) if 'total_issues' in site_data.columns else 0
        }
        
        # Count by issue type
        issue_cols = [c for c in site_data.columns if c.startswith('issue_')] # Adjust if prefix removed
        result['issue_counts'] = {
            col.replace('issue_', ''): int(site_data[col].sum())
            for col in issue_cols
        }
        
        return result
    
    def get_patient_issues(self, patient_key: str) -> Dict[str, Any]:
        """Get issues for a specific patient."""
        issues_df = self.patient_issues
        if issues_df is None or issues_df.empty:
            return {}
        
        patient_data = issues_df[issues_df['patient_key'] == patient_key]
        if patient_data.empty:
            return {}
        
        return patient_data.iloc[0].to_dict()
    
    def get_cascade_data(self, patient_key: str) -> Dict[str, Any]:
        """Get cascade analysis for a patient."""
        cascade_df = self.patient_cascade
        if cascade_df is None or cascade_df.empty:
            return {}
        
        cascade_data = cascade_df[cascade_df['patient_key'] == patient_key]
        if cascade_data.empty:
            return {}
        
        return cascade_data.iloc[0].to_dict()


class CascadeAnalyzer:
    """Analyzes cascade impact of resolving issues."""
    
    # Issue dependency graph
    DEPENDENCIES = {
        'missing_visits': ['sdv_incomplete', 'signature_gaps', 'open_queries'],
        'missing_pages': ['sdv_incomplete', 'signature_gaps', 'open_queries'],
        'open_queries': ['signature_gaps'],
        'sdv_incomplete': ['signature_gaps'],
        'sae_dm_pending': ['sae_safety_pending'],
        'broken_signatures': ['signature_gaps'],
        'meddra_uncoded': [],
        'whodrug_uncoded': [],
        'lab_issues': [],
        'edrr_issues': [],
        'inactivated_forms': [],
        'signature_gaps': [],
        'sae_safety_pending': [],
        'high_query_volume': ['open_queries']
    }
    
    # Impact weights (how much fixing this issue helps DB Lock)
    IMPACT_WEIGHTS = {
        'missing_visits': 10.0,
        'missing_pages': 8.0,
        'open_queries': 6.0,
        'sdv_incomplete': 5.0,
        'signature_gaps': 7.0,
        'broken_signatures': 5.0,
        'sae_dm_pending': 9.0,
        'sae_safety_pending': 8.0,
        'meddra_uncoded': 2.0,
        'whodrug_uncoded': 2.0,
        'lab_issues': 3.0,
        'edrr_issues': 4.0,
        'inactivated_forms': 1.0,
        'high_query_volume': 3.0
    }
    
    def calculate_cascade_impact(
        self,
        issue_type: str,
        issue_count: int,
        entity_id: str
    ) -> CascadeImpact:
        """Calculate the cascade impact of resolving an issue type."""
        
        # Direct impact
        direct_weight = self.IMPACT_WEIGHTS.get(issue_type, 1.0)
        direct_impact = {
            'issue_type': issue_type,
            'issues_resolved': issue_count,
            'impact_weight': direct_weight,
            'direct_score': issue_count * direct_weight
        }
        
        # Cascade effects
        cascade_effects = []
        downstream = self.DEPENDENCIES.get(issue_type, [])
        
        for downstream_issue in downstream:
            downstream_weight = self.IMPACT_WEIGHTS.get(downstream_issue, 1.0)
            # Assume 30% of downstream issues get unblocked
            estimated_unblocked = int(issue_count * 0.3)
            
            cascade_effects.append({
                'issue_type': downstream_issue,
                'estimated_unblocked': estimated_unblocked,
                'impact_weight': downstream_weight,
                'cascade_score': estimated_unblocked * downstream_weight
            })
        
        # Total calculations
        total_issues = issue_count + sum(e['estimated_unblocked'] for e in cascade_effects)
        total_score = direct_impact['direct_score'] + sum(e['cascade_score'] for e in cascade_effects)
        
        # Estimate patients unblocked (roughly 1 patient per 2 issues)
        patients_unblocked = total_issues // 2
        
        # Estimate DQI improvement (0.1 points per weighted issue resolved)
        dqi_improvement = total_score * 0.1
        
        # Estimate DB Lock acceleration (1 day per 100 weighted score)
        db_lock_days = total_score / 100
        
        # Effort estimation
        effort_hours = issue_count * 0.5  # Base estimate
        
        # ROI score
        roi = total_score / max(effort_hours, 0.1)
        
        return CascadeImpact(
            source_issue=issue_type,
            source_entity=entity_id,
            direct_impact=direct_impact,
            cascade_effects=cascade_effects,
            total_issues_resolved=total_issues,
            total_patients_unblocked=patients_unblocked,
            dqi_improvement=dqi_improvement,
            db_lock_acceleration_days=db_lock_days,
            effort_hours=effort_hours,
            roi_score=roi
        )


class EnhancedResolverAgent:
    """
    Enhanced RESOLVER Agent for action plan generation.
    
    Capabilities:
    - Resolution genome integration
    - Action plan generation
    - Cascade impact analysis
    - Priority-based task sequencing
    - Role-based task assignment
    """
    
    def __init__(self, llm_wrapper=None):
        self.data_loader = ResolverDataLoader()
        self.genome = ResolutionGenome()
        self.cascade_analyzer = CascadeAnalyzer()
        self.llm = llm_wrapper
        self._action_counter = 0
        self._plan_counter = 0
        self._result_counter = 0
        
        logger.info("EnhancedResolverAgent initialized")
    
    def _generate_id(self, prefix: str) -> str:
        if prefix == "ACT":
            self._action_counter += 1
            return f"ACT-{self._action_counter:04d}"
        elif prefix == "PLN":
            self._plan_counter += 1
            return f"PLN-{self._plan_counter:04d}"
        elif prefix == "RES":
            self._result_counter += 1
            return f"RES-{self._result_counter:04d}"
        return f"{prefix}-{datetime.now().strftime('%H%M%S')}"
    
    def _determine_priority(self, issue_type: str, count: int) -> ActionPriority:
        """Determine action priority based on issue type and count."""
        # Safety issues are always critical
        if issue_type in ['sae_dm_pending', 'sae_safety_pending']:
            return ActionPriority.CRITICAL
        
        # High volume issues
        if count > 50:
            return ActionPriority.HIGH
        elif count > 20:
            return ActionPriority.MEDIUM
        elif count > 5:
            return ActionPriority.LOW
        else:
            return ActionPriority.OPTIONAL
    
    def _calculate_impact_score(self, issue_type: str, count: int) -> float:
        """Calculate impact score (0-100) for an action."""
        weight = self.cascade_analyzer.IMPACT_WEIGHTS.get(issue_type, 1.0)
        # Logarithmic scaling for count
        count_factor = np.log1p(count) / np.log1p(100)  # Normalize to 0-1
        
        return min(100, weight * 10 * (1 + count_factor))
    
    def create_action_from_template(
        self,
        template: ResolutionTemplate,
        entity_id: str,
        entity_type: str,
        issue_count: int = 1
    ) -> Action:
        """Create an action from a resolution template."""
        priority = self._determine_priority(template.issue_type, issue_count)
        impact = self._calculate_impact_score(template.issue_type, issue_count)
        
        # Calculate due date based on priority
        due_days = {
            ActionPriority.CRITICAL: 1,
            ActionPriority.HIGH: 3,
            ActionPriority.MEDIUM: 7,
            ActionPriority.LOW: 14,
            ActionPriority.OPTIONAL: 30
        }
        due_date = datetime.now() + timedelta(days=due_days.get(priority, 7))
        
        # Adjust effort for volume
        effort = template.effort_hours * (1 + np.log1p(issue_count) / 10)
        
        return Action(
            action_id=self._generate_id("ACT"),
            title=f"{template.title} ({issue_count} issues)",
            description=template.description,
            category=self._get_category(template.issue_type),
            priority=priority,
            responsible_role=template.responsible_role,
            entity_id=entity_id,
            entity_type=entity_type,
            issue_type=template.issue_type,
            steps=template.steps,
            effort_hours=effort,
            impact_score=impact,
            confidence=template.success_rate,
            dependencies=[],
            unlocks=self.cascade_analyzer.DEPENDENCIES.get(template.issue_type, []),
            status=ActionStatus.PENDING,
            due_date=due_date,
            template_id=template.template_id,
            success_rate=template.success_rate,
            requires_approval=priority == ActionPriority.CRITICAL
        )
    
    def _get_category(self, issue_type: str) -> ActionCategory:
        """Map issue type to action category."""
        category_map = {
            'sdv_incomplete': ActionCategory.MONITORING,
            'open_queries': ActionCategory.DATA_QUALITY,
            'signature_gaps': ActionCategory.DATA_QUALITY,
            'broken_signatures': ActionCategory.DATA_QUALITY,
            'sae_dm_pending': ActionCategory.SAFETY,
            'sae_safety_pending': ActionCategory.SAFETY,
            'missing_visits': ActionCategory.SITE_MANAGEMENT,
            'missing_pages': ActionCategory.DATA_QUALITY,
            'meddra_uncoded': ActionCategory.DATA_QUALITY,
            'whodrug_uncoded': ActionCategory.DATA_QUALITY,
            'lab_issues': ActionCategory.DATA_QUALITY,
            'edrr_issues': ActionCategory.DATA_QUALITY,
            'inactivated_forms': ActionCategory.DATA_QUALITY,
            'high_query_volume': ActionCategory.TRAINING
        }
        return category_map.get(issue_type, ActionCategory.DATA_QUALITY)
    
    def create_action_plan_for_site(self, site_id: str) -> ActionPlan:
        """Create a comprehensive action plan for a site."""
        site_issues = self.data_loader.get_site_issues(site_id)
        
        if not site_issues:
            return ActionPlan(
                plan_id=self._generate_id("PLN"),
                title=f"Action Plan for {site_id}",
                description="No issues found for this site",
                entity_id=site_id,
                entity_type="site"
            )
        
        plan = ActionPlan(
            plan_id=self._generate_id("PLN"),
            title=f"Action Plan for {site_id}",
            description=f"Comprehensive action plan to resolve {site_issues.get('total_issues', 0)} issues",
            entity_id=site_id,
            entity_type="site"
        )
        
        # Create actions for each issue type
        issue_counts = site_issues.get('issue_counts', {})
        
        for issue_type, count in sorted(issue_counts.items(), key=lambda x: -x[1]):
            if count > 0:
                template = self.genome.get_best_template(issue_type)
                if template:
                    action = self.create_action_from_template(
                        template, site_id, "site", count
                    )
                    plan.add_action(action)
                    
                    # Add cascade impact
                    cascade = self.cascade_analyzer.calculate_cascade_impact(
                        issue_type, count, site_id
                    )
                    plan.cascade_impacts.append(cascade)
        
        # Sort actions by priority and impact
        plan.actions.sort(key=lambda a: (
            -{'critical': 4, 'high': 3, 'medium': 2, 'low': 1, 'optional': 0}[a.priority.value],
            -a.impact_score
        ))
        
        # Calculate expected impact
        plan.expected_impact = {
            'issues_addressed': site_issues.get('total_issues', 0),
            'estimated_dqi_improvement': sum(c.dqi_improvement for c in plan.cascade_impacts),
            'estimated_db_lock_acceleration': sum(c.db_lock_acceleration_days for c in plan.cascade_impacts),
            'patients_affected': site_issues.get('patients_with_issues', 0)
        }
        
        # Estimate timeline
        plan.timeline_days = int(plan.total_effort_hours / 8) + 1  # Assuming 8 hours/day
        
        plan.assumptions = [
            "Resources available as assigned",
            "Site cooperation maintained",
            "No new major issues arise"
        ]
        
        plan.risk_factors = [
            "Staff availability may vary",
            "Some issues may be more complex than estimated"
        ]
        
        return plan
    
    def create_action_plan_for_patient(self, patient_key: str) -> ActionPlan:
        """Create an action plan for a specific patient."""
        patient_issues = self.data_loader.get_patient_issues(patient_key)
        
        if not patient_issues:
            return ActionPlan(
                plan_id=self._generate_id("PLN"),
                title=f"Action Plan for {patient_key}",
                description="No issues found for this patient",
                entity_id=patient_key,
                entity_type="patient"
            )
        
        plan = ActionPlan(
            plan_id=self._generate_id("PLN"),
            title=f"Action Plan for {patient_key}",
            description=f"Actions to resolve patient issues",
            entity_id=patient_key,
            entity_type="patient"
        )
        
        # Find issue types for this patient
        for key, value in patient_issues.items():
            if key.startswith('issue_') and key != 'has_any_issue' and value == 1:
                issue_type = key.replace('issue_', '')
                count_key = f'count_{issue_type}'
                count = patient_issues.get(count_key, 1)
                
                template = self.genome.get_best_template(issue_type)
                if template:
                    action = self.create_action_from_template(
                        template, patient_key, "patient", count
                    )
                    plan.add_action(action)
        
        return plan
    
    def create_portfolio_action_plan(self, top_n: int = 20) -> ActionPlan:
        """Create a portfolio-level action plan focusing on highest impact actions."""
        if self.data_loader.patient_issues is None:
            return ActionPlan(
                plan_id=self._generate_id("PLN"),
                title="Portfolio Action Plan",
                description="No data available",
                entity_id="portfolio",
                entity_type="portfolio"
            )
        
        df = self.data_loader.patient_issues
        
        plan = ActionPlan(
            plan_id=self._generate_id("PLN"),
            title="Portfolio Action Plan",
            description="High-impact actions across all sites",
            entity_id="portfolio",
            entity_type="portfolio"
        )
        
        # Aggregate issues by site and type
        issue_cols = [c for c in df.columns if c.startswith('issue_') and c != 'has_any_issue']
        
        # Calculate site-level impact opportunities
        site_opportunities = []
        
        for site_id in df['site_id'].unique():
            site_data = df[df['site_id'] == site_id]
            
            for col in issue_cols:
                issue_type = col.replace('issue_', '')
                count = int(site_data[col].sum())
                
                if count > 0:
                    impact = self._calculate_impact_score(issue_type, count)
                    site_opportunities.append({
                        'site_id': site_id,
                        'issue_type': issue_type,
                        'count': count,
                        'impact': impact
                    })
        
        # Sort by impact and take top N
        site_opportunities.sort(key=lambda x: -x['impact'])
        
        for opp in site_opportunities[:top_n]:
            template = self.genome.get_best_template(opp['issue_type'])
            if template:
                action = self.create_action_from_template(
                    template,
                    opp['site_id'],
                    "site",
                    opp['count']
                )
                action.title = f"[{opp['site_id']}] {action.title}"
                plan.add_action(action)
        
        # Calculate totals
        total_issues = int(df['total_issues'].sum())
        plan.expected_impact = {
            'actions_planned': len(plan.actions),
            'total_portfolio_issues': total_issues,
            'coverage_rate': len(plan.actions) / max(len(site_opportunities), 1)
        }
        
        return plan
    
    def identify_quick_wins(self, entity_id: str = "portfolio", max_effort: float = 2.0) -> List[Action]:
        """Identify quick win actions (high impact, low effort)."""
        if entity_id == "portfolio":
            plan = self.create_portfolio_action_plan(top_n=50)
        elif entity_id.startswith("Site_"):
            plan = self.create_action_plan_for_site(entity_id)
        else:
            plan = self.create_action_plan_for_patient(entity_id)
        
        quick_wins = [
            action for action in plan.actions
            if action.effort_hours <= max_effort and action.impact_score >= 30
        ]
        
        # Sort by ROI (impact per effort hour)
        quick_wins.sort(key=lambda a: -a.impact_score / max(a.effort_hours, 0.1))
        
        return quick_wins[:10]
    
    def resolve_from_query(self, query: str) -> ResolverResult:
        """Generate resolution plan from natural language query."""
        start_time = datetime.now()
        query_lower = query.lower()
        
        result = ResolverResult(
            result_id=self._generate_id("RES"),
            query=query
        )
        
        # Extract entity
        entity_id = self._extract_entity(query)
        
        # Determine what kind of resolution is needed
        if 'quick win' in query_lower or 'easy' in query_lower or 'fast' in query_lower:
            quick_wins = self.identify_quick_wins(entity_id)
            result.top_actions = quick_wins
            result.summary = self._generate_quick_wins_summary(quick_wins, entity_id)
        
        elif 'plan' in query_lower or 'how' in query_lower or 'resolve' in query_lower:
            if entity_id == "portfolio":
                plan = self.create_portfolio_action_plan()
            elif entity_id.startswith("Site_"):
                plan = self.create_action_plan_for_site(entity_id)
            else:
                plan = self.create_action_plan_for_patient(entity_id)
            
            result.action_plans.append(plan)
            result.top_actions = plan.actions[:10]
            result.cascade_analysis = plan.cascade_impacts
            result.total_effort_hours = plan.total_effort_hours
            result.summary = self._generate_plan_summary(plan)
        
        elif 'cascade' in query_lower or 'impact' in query_lower:
            # Focus on cascade analysis
            if entity_id.startswith("Site_"):
                site_issues = self.data_loader.get_site_issues(entity_id)
                for issue_type, count in site_issues.get('issue_counts', {}).items():
                    if count > 0:
                        cascade = self.cascade_analyzer.calculate_cascade_impact(
                            issue_type, count, entity_id
                        )
                        result.cascade_analysis.append(cascade)
            
            result.summary = self._generate_cascade_summary(result.cascade_analysis, entity_id)
        
        else:
            # Default to action plan
            if entity_id.startswith("Site_"):
                plan = self.create_action_plan_for_site(entity_id)
            else:
                plan = self.create_portfolio_action_plan(top_n=10)
            
            result.action_plans.append(plan)
            result.top_actions = plan.actions[:5]
            result.summary = self._generate_plan_summary(plan)
        
        # Generate recommendations
        result.recommendations = self._generate_recommendations(result)
        result.duration_seconds = (datetime.now() - start_time).total_seconds()
        
        return result
    
    def _extract_entity(self, query: str) -> str:
        """Extract entity ID from query."""
        import re
        
        site_match = re.search(r'Site[_\s]?(\d+)', query, re.IGNORECASE)
        if site_match:
            return f"Site_{site_match.group(1)}"
        
        study_match = re.search(r'Study[_\s]?(\d+)', query, re.IGNORECASE)
        if study_match:
            return f"Study_{study_match.group(1)}"
        
        patient_match = re.search(r'(Study_\d+\|Site_\d+\|Subject_\d+)', query)
        if patient_match:
            return patient_match.group(1)
        
        return "portfolio"
    
    def _generate_quick_wins_summary(self, quick_wins: List[Action], entity_id: str) -> str:
        """Generate summary for quick wins."""
        if not quick_wins:
            return f"No quick wins identified for {entity_id}"
        
        total_effort = sum(a.effort_hours for a in quick_wins)
        total_impact = sum(a.impact_score for a in quick_wins)
        
        summary = f"""
QUICK WINS IDENTIFIED: {entity_id}
{'=' * 50}
Total Quick Wins: {len(quick_wins)}
Total Effort: {total_effort:.1f} hours
Combined Impact Score: {total_impact:.0f}

TOP QUICK WINS:
"""
        for i, action in enumerate(quick_wins[:5], 1):
            summary += f"""
{i}. {action.title}
   Effort: {action.effort_hours:.1f} hrs | Impact: {action.impact_score:.0f}
   Role: {action.responsible_role.value}
   Priority: {action.priority.value.upper()}
"""
        
        return summary.strip()
    
    def _generate_plan_summary(self, plan: ActionPlan) -> str:
        """Generate summary for action plan."""
        priority_counts = plan.get_actions_by_priority()
        role_counts = plan.get_actions_by_role()
        
        summary = f"""
ACTION PLAN: {plan.title}
{'=' * 50}
Entity: {plan.entity_id}
Total Actions: {len(plan.actions)}
Quick Wins: {len(plan.quick_wins)}
Total Effort: {plan.total_effort_hours:.1f} hours
Timeline: ~{plan.timeline_days} days

BY PRIORITY:
"""
        for priority in ['critical', 'high', 'medium', 'low', 'optional']:
            count = len(priority_counts.get(priority, []))
            if count > 0:
                summary += f"  {priority.upper()}: {count}\n"
        
        summary += "\nBY ROLE:\n"
        for role, actions in sorted(role_counts.items(), key=lambda x: -len(x[1])):
            summary += f"  {role}: {len(actions)} actions\n"
        
        if plan.expected_impact:
            summary += f"""
EXPECTED IMPACT:
  Issues Addressed: {plan.expected_impact.get('issues_addressed', 0)}
  DQI Improvement: +{plan.expected_impact.get('estimated_dqi_improvement', 0):.1f} points
  DB Lock Acceleration: {plan.expected_impact.get('estimated_db_lock_acceleration', 0):.1f} days
"""
        
        return summary.strip()
    
    def _generate_cascade_summary(self, cascades: List[CascadeImpact], entity_id: str) -> str:
        """Generate cascade analysis summary."""
        if not cascades:
            return f"No cascade analysis available for {entity_id}"
        
        # Sort by ROI
        cascades.sort(key=lambda c: -c.roi_score)
        
        summary = f"""
CASCADE IMPACT ANALYSIS: {entity_id}
{'=' * 50}
Total Issue Types Analyzed: {len(cascades)}

TOP IMPACT OPPORTUNITIES (by ROI):
"""
        for i, cascade in enumerate(cascades[:5], 1):
            summary += f"""
{i}. {cascade.source_issue.upper()}
   Direct Issues: {cascade.direct_impact.get('issues_resolved', 0)}
   Cascade Effects: {len(cascade.cascade_effects)} downstream types
   Total Unblocked: {cascade.total_issues_resolved} issues, {cascade.total_patients_unblocked} patients
   DQI Improvement: +{cascade.dqi_improvement:.1f}
   DB Lock Acceleration: {cascade.db_lock_acceleration_days:.1f} days
   Effort: {cascade.effort_hours:.1f} hours
   ROI Score: {cascade.roi_score:.1f}
"""
        
        return summary.strip()
    
    def _generate_recommendations(self, result: ResolverResult) -> List[str]:
        """Generate recommendations based on result."""
        recommendations = []
        
        if result.top_actions:
            # Get highest priority action
            critical_actions = [a for a in result.top_actions if a.priority == ActionPriority.CRITICAL]
            if critical_actions:
                recommendations.append(
                    f"URGENT: Complete {len(critical_actions)} critical action(s) immediately"
                )
            
            # Role-based recommendations
            roles = set(a.responsible_role for a in result.top_actions)
            for role in roles:
                role_actions = [a for a in result.top_actions if a.responsible_role == role]
                if len(role_actions) >= 3:
                    recommendations.append(
                        f"{role.value}: {len(role_actions)} actions assigned - prioritize by due date"
                    )
        
        if result.cascade_analysis:
            # Find highest ROI opportunity
            best_cascade = max(result.cascade_analysis, key=lambda c: c.roi_score)
            recommendations.append(
                f"Highest ROI: Address {best_cascade.source_issue} for {best_cascade.roi_score:.1f}x return"
            )
        
        if not recommendations:
            recommendations.append("Review action plan and assign owners")
            recommendations.append("Track progress weekly")
        
        return recommendations
    
    def process(self, query: str, context: Dict = None) -> Dict:
        """Main processing method for orchestrator integration."""
        result = self.resolve_from_query(query)
        
        return {
            "result_id": result.result_id,
            "query": result.query,
            "action_plans": [p.to_dict() for p in result.action_plans],
            "top_actions": [a.to_dict() for a in result.top_actions],
            "cascade_analysis": [c.to_dict() for c in result.cascade_analysis],
            "summary": result.summary,
            "recommendations": result.recommendations,
            "total_effort_hours": result.total_effort_hours,
            "expected_outcomes": result.expected_outcomes,
            "duration_seconds": result.duration_seconds
        }


def get_resolver_agent(llm_wrapper=None) -> EnhancedResolverAgent:
    """Factory function to get resolver agent instance."""
    return EnhancedResolverAgent(llm_wrapper=llm_wrapper)