# src/generation/auto_summarizer.py
"""
SANCHALAK AI - Auto-Summarization Engine v1.0
Phase 6.5: Patient Summaries, Site Narratives, Investigation Summaries, Daily Digest

Features:
- Patient-level summaries with issue highlights
- Site narratives with performance analysis
- Investigation summaries with root cause insights
- Daily digest for stakeholders
- Executive summaries for leadership
- Trend analysis narratives
- Action item extraction
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from config.settings import DATA_PROCESSED, DATA_OUTPUTS

# PostgreSQL integration
from src.database.pg_data_service import get_data_service
from src.database.pg_writer import get_pg_writer


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class SummaryType(Enum):
    """Types of summaries that can be generated."""
    PATIENT = "patient"
    SITE = "site"
    STUDY = "study"
    PORTFOLIO = "portfolio"
    INVESTIGATION = "investigation"
    DAILY_DIGEST = "daily_digest"
    WEEKLY_DIGEST = "weekly_digest"
    EXECUTIVE = "executive"
    ISSUE = "issue"
    ACTION = "action"


class SummaryFormat(Enum):
    """Output formats for summaries."""
    TEXT = "text"
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"


class Severity(Enum):
    """Severity levels for issues and findings."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class Finding:
    """A finding or insight from the data."""
    finding_id: str
    category: str
    title: str
    description: str
    severity: Severity
    metric_value: Optional[float] = None
    metric_name: Optional[str] = None
    recommendation: Optional[str] = None
    affected_entities: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'finding_id': self.finding_id,
            'category': self.category,
            'title': self.title,
            'description': self.description,
            'severity': self.severity.value,
            'metric_value': self.metric_value,
            'metric_name': self.metric_name,
            'recommendation': self.recommendation,
            'affected_entities': self.affected_entities
        }


@dataclass
class ActionItem:
    """An action item extracted from analysis."""
    action_id: str
    title: str
    description: str
    priority: str
    owner: str
    due_date: Optional[datetime] = None
    status: str = "pending"
    related_findings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'action_id': self.action_id,
            'title': self.title,
            'description': self.description,
            'priority': self.priority,
            'owner': self.owner,
            'due_date': self.due_date.isoformat() if self.due_date else None,
            'status': self.status,
            'related_findings': self.related_findings
        }


@dataclass
class Summary:
    """A generated summary."""
    summary_id: str
    summary_type: SummaryType
    entity_id: str
    entity_name: str
    title: str
    executive_summary: str
    sections: List[Dict[str, Any]]
    findings: List[Finding]
    action_items: List[ActionItem]
    metrics: Dict[str, Any]
    generated_at: datetime
    format: SummaryFormat = SummaryFormat.MARKDOWN
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'summary_id': self.summary_id,
            'summary_type': self.summary_type.value,
            'entity_id': self.entity_id,
            'entity_name': self.entity_name,
            'title': self.title,
            'executive_summary': self.executive_summary,
            'sections': self.sections,
            'findings': [f.to_dict() for f in self.findings],
            'action_items': [a.to_dict() for a in self.action_items],
            'metrics': self.metrics,
            'generated_at': self.generated_at.isoformat(),
            'format': self.format.value,
            'metadata': self.metadata
        }
    
    def to_markdown(self) -> str:
        """Convert summary to markdown format."""
        md = []
        
        # Title
        md.append("# {}".format(self.title))
        md.append("")
        md.append("*Generated: {}*".format(self.generated_at.strftime('%Y-%m-%d %H:%M')))
        md.append("")
        
        # Executive Summary
        md.append("## Executive Summary")
        md.append("")
        md.append(self.executive_summary)
        md.append("")
        
        # Key Metrics
        if self.metrics:
            md.append("## Key Metrics")
            md.append("")
            md.append("| Metric | Value |")
            md.append("|--------|-------|")
            for key, value in self.metrics.items():
                formatted_key = key.replace('_', ' ').title()
                if isinstance(value, float):
                    formatted_value = "{:.2f}".format(value)
                elif isinstance(value, int):
                    formatted_value = "{:,}".format(value)
                else:
                    formatted_value = str(value)
                md.append("| {} | {} |".format(formatted_key, formatted_value))
            md.append("")
        
        # Sections
        for section in self.sections:
            md.append("## {}".format(section.get('title', 'Section')))
            md.append("")
            md.append(section.get('content', ''))
            md.append("")
            
            # Section items if present
            if 'items' in section:
                for item in section['items']:
                    md.append("- {}".format(item))
                md.append("")
        
        # Findings
        if self.findings:
            md.append("## Key Findings")
            md.append("")
            for finding in self.findings:
                severity_icon = {
                    Severity.CRITICAL: "🔴",
                    Severity.HIGH: "🟠",
                    Severity.MEDIUM: "🟡",
                    Severity.LOW: "🟢",
                    Severity.INFO: "ℹ️"
                }.get(finding.severity, "•")
                
                md.append("{} **{}**: {}".format(severity_icon, finding.title, finding.description))
                if finding.recommendation:
                    md.append("   - *Recommendation*: {}".format(finding.recommendation))
                md.append("")
        
        # Action Items
        if self.action_items:
            md.append("## Action Items")
            md.append("")
            md.append("| Priority | Action | Owner | Due Date |")
            md.append("|----------|--------|-------|----------|")
            for action in self.action_items:
                due = action.due_date.strftime('%Y-%m-%d') if action.due_date else "TBD"
                md.append("| {} | {} | {} | {} |".format(
                    action.priority, action.title, action.owner, due
                ))
            md.append("")
        
        return "\n".join(md)
    
    def to_html(self) -> str:
        """Convert summary to HTML format."""
        # Convert markdown to basic HTML
        md = self.to_markdown()
        
        html = ["<!DOCTYPE html>", "<html>", "<head>"]
        html.append("<title>{}</title>".format(self.title))
        html.append("<style>")
        html.append("""
            body { font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }
            h1 { color: #1a365d; border-bottom: 3px solid #38a169; padding-bottom: 10px; }
            h2 { color: #2c5282; margin-top: 25px; }
            table { border-collapse: collapse; width: 100%; margin: 15px 0; }
            th { background-color: #1a365d; color: white; padding: 10px; text-align: left; }
            td { padding: 8px; border-bottom: 1px solid #ddd; }
            tr:nth-child(even) { background-color: #f8f9fa; }
            .critical { color: #e53e3e; }
            .high { color: #dd6b20; }
            .medium { color: #d69e2e; }
            .low { color: #38a169; }
            .info { color: #3182ce; }
            .metric-card { background: #ebf8ff; padding: 15px; border-radius: 8px; margin: 10px 0; }
        """)
        html.append("</style>")
        html.append("</head>")
        html.append("<body>")
        
        # Simple markdown to HTML conversion
        lines = md.split('\n')
        in_table = False
        in_list = False
        
        for line in lines:
            if line.startswith('# '):
                html.append("<h1>{}</h1>".format(line[2:]))
            elif line.startswith('## '):
                html.append("<h2>{}</h2>".format(line[3:]))
            elif line.startswith('| ') and '|' in line[1:]:
                if not in_table:
                    html.append("<table>")
                    in_table = True
                cells = [c.strip() for c in line.split('|')[1:-1]]
                if '---' in line:
                    continue  # Skip separator row
                tag = 'th' if html[-1] == '<table>' else 'td'
                html.append("<tr>{}</tr>".format(''.join("<{}>{}</{}>".format(tag, c, tag) for c in cells)))
            elif in_table and not line.startswith('|'):
                html.append("</table>")
                in_table = False
            elif line.startswith('- '):
                if not in_list:
                    html.append("<ul>")
                    in_list = True
                html.append("<li>{}</li>".format(line[2:]))
            elif in_list and not line.startswith('- ') and not line.startswith('   '):
                html.append("</ul>")
                in_list = False
                if line.strip():
                    html.append("<p>{}</p>".format(line))
            elif line.startswith('*') and line.endswith('*'):
                html.append("<p><em>{}</em></p>".format(line.strip('*')))
            elif line.strip():
                html.append("<p>{}</p>".format(line))
        
        if in_table:
            html.append("</table>")
        if in_list:
            html.append("</ul>")
        
        html.append("</body>")
        html.append("</html>")
        
        return "\n".join(html)


# =============================================================================
# DATA LOADER
# =============================================================================

class SummaryDataLoader:
    """Loads data required for summaries."""
    
    def __init__(self):
        self.data_path = Path(DATA_PROCESSED)
        self.data = {}
        self._load_data()
    
    def _load_data(self):
        """Load all required data files."""
        files = {
            'patients': 'upr/unified_patient_record.parquet',
            'issues': 'analytics/patient_issues.parquet',
            'dqi': 'analytics/patient_dqi_enhanced.parquet',
            'clean_status': 'analytics/patient_clean_status.parquet',
            'dblock': 'analytics/patient_dblock_status.parquet',
            'sites': 'analytics/site_benchmarks.parquet',
            'cascade': 'analytics/patient_cascade_analysis.parquet'
        }
        
        for name, file_path in files.items():
            full_path = self.data_path / file_path
            if full_path.exists():
                try:
                    self.data[name] = pd.read_parquet(full_path)
                except Exception as e:
                    print("Warning: Could not load {}: {}".format(name, e))
    
    def get_patient_data(self, patient_key: str) -> Optional[Dict]:
        """Get all data for a specific patient."""
        result = {}
        
        for name, df in self.data.items():
            if 'patient_key' in df.columns:
                patient_df = df[df['patient_key'] == patient_key]
                if len(patient_df) > 0:
                    result[name] = patient_df.iloc[0].to_dict()
        
        return result if result else None
    
    def get_site_data(self, site_id: str) -> Dict:
        """Get aggregated data for a specific site."""
        result = {'site_id': site_id}
        
        for name, df in self.data.items():
            if 'site_id' in df.columns:
                site_df = df[df['site_id'] == site_id]
                if len(site_df) > 0:
                    result['{}_count'.format(name)] = len(site_df)
                    
                    # Aggregate numeric columns
                    numeric_cols = site_df.select_dtypes(include=[np.number]).columns
                    for col in numeric_cols:
                        if col not in ['patient_key']:
                            result['{}_{}_mean'.format(name, col)] = site_df[col].mean()
                            result['{}_{}_sum'.format(name, col)] = site_df[col].sum()
        
        # Get site benchmark data
        if 'sites' in self.data:
            sites_df = self.data['sites']
            if 'site_id' in sites_df.columns:
                site_bench = sites_df[sites_df['site_id'] == site_id]
                if len(site_bench) > 0:
                    result['benchmark'] = site_bench.iloc[0].to_dict()
        
        return result
    
    def get_study_data(self, study_id: str) -> Dict:
        """Get aggregated data for a specific study."""
        result = {'study_id': study_id}
        
        for name, df in self.data.items():
            if 'study_id' in df.columns:
                study_df = df[df['study_id'] == study_id]
                if len(study_df) > 0:
                    result['{}_count'.format(name)] = len(study_df)
                    
                    numeric_cols = study_df.select_dtypes(include=[np.number]).columns
                    for col in numeric_cols[:10]:  # Limit columns
                        if col not in ['patient_key']:
                            result['{}_{}_mean'.format(name, col)] = study_df[col].mean()
        
        return result
    
    def get_portfolio_data(self) -> Dict:
        """Get portfolio-level aggregated data."""
        result = {}
        
        if 'patients' in self.data:
            df = self.data['patients']
            result['total_patients'] = len(df)
            result['total_studies'] = df['study_id'].nunique() if 'study_id' in df.columns else 0
            result['total_sites'] = df['site_id'].nunique() if 'site_id' in df.columns else 0
        
        if 'dqi' in self.data:
            df = self.data['dqi']
            if 'dqi_score' in df.columns:
                result['mean_dqi'] = df['dqi_score'].mean()
                result['min_dqi'] = df['dqi_score'].min()
                result['max_dqi'] = df['dqi_score'].max()
        
        if 'clean_status' in self.data:
            df = self.data['clean_status']
            if 'tier1_clean' in df.columns:
                result['tier1_clean_rate'] = df['tier1_clean'].mean() * 100
            if 'tier2_clean' in df.columns:
                result['tier2_clean_rate'] = df['tier2_clean'].mean() * 100
        
        if 'dblock' in self.data:
            df = self.data['dblock']
            if 'dblock_ready' in df.columns:
                result['dblock_ready_rate'] = df['dblock_ready'].mean() * 100
            if 'dblock_eligible' in df.columns:
                result['dblock_eligible_count'] = df['dblock_eligible'].sum()
        
        if 'issues' in self.data:
            df = self.data['issues']
            if 'total_issues' in df.columns:
                result['total_issues'] = df['total_issues'].sum()
                result['patients_with_issues'] = (df['total_issues'] > 0).sum()
        
        return result


# =============================================================================
# PATIENT SUMMARIZER
# =============================================================================

class PatientSummarizer:
    """Generates patient-level summaries."""
    
    def __init__(self, data_loader: SummaryDataLoader):
        self.data = data_loader
    
    def summarize(self, patient_key: str) -> Summary:
        """Generate a summary for a specific patient."""
        patient_data = self.data.get_patient_data(patient_key)
        
        if not patient_data:
            return self._empty_summary(patient_key)
        
        # Extract key information
        patients_info = patient_data.get('patients', {})
        issues_info = patient_data.get('issues', {})
        dqi_info = patient_data.get('dqi', {})
        clean_info = patient_data.get('clean_status', {})
        dblock_info = patient_data.get('dblock', {})
        
        # Build metrics
        metrics = {
            'study_id': patients_info.get('study_id', 'Unknown'),
            'site_id': patients_info.get('site_id', 'Unknown'),
            'status': patients_info.get('subject_status', 'Unknown'),
            'dqi_score': dqi_info.get('dqi_score', 0),
            'dqi_band': dqi_info.get('dqi_band', 'Unknown'),
            'tier1_clean': clean_info.get('tier1_clean', False),
            'tier2_clean': clean_info.get('tier2_clean', False),
            'dblock_ready': dblock_info.get('dblock_ready', False),
            'total_issues': issues_info.get('total_issues', 0)
        }
        
        # Generate findings
        findings = self._analyze_patient_issues(patient_data)
        
        # Generate action items
        action_items = self._generate_patient_actions(patient_data, findings)
        
        # Build sections
        sections = [
            {
                'title': 'Patient Overview',
                'content': self._build_patient_overview(patient_data)
            },
            {
                'title': 'Data Quality Status',
                'content': self._build_dqi_section(patient_data)
            },
            {
                'title': 'Issue Summary',
                'content': self._build_issues_section(patient_data)
            },
            {
                'title': 'DB Lock Readiness',
                'content': self._build_dblock_section(patient_data)
            }
        ]
        
        # Executive summary
        exec_summary = self._generate_patient_executive_summary(patient_data, metrics)
        
        return Summary(
            summary_id="SUM-PAT-{}".format(hashlib.md5(patient_key.encode()).hexdigest()[:8]),
            summary_type=SummaryType.PATIENT,
            entity_id=patient_key,
            entity_name="Patient {}".format(patient_key.split('|')[-1] if '|' in patient_key else patient_key),
            title="Patient Summary: {}".format(patient_key),
            executive_summary=exec_summary,
            sections=sections,
            findings=findings,
            action_items=action_items,
            metrics=metrics,
            generated_at=datetime.now()
        )
    
    def _empty_summary(self, patient_key: str) -> Summary:
        return Summary(
            summary_id="SUM-PAT-EMPTY",
            summary_type=SummaryType.PATIENT,
            entity_id=patient_key,
            entity_name=patient_key,
            title="Patient Summary: {}".format(patient_key),
            executive_summary="No data available for patient {}.".format(patient_key),
            sections=[],
            findings=[],
            action_items=[],
            metrics={},
            generated_at=datetime.now()
        )
    
    def _analyze_patient_issues(self, patient_data: Dict) -> List[Finding]:
        findings = []
        issues = patient_data.get('issues', {})
        
        issue_checks = [
            ('issue_sdv_incomplete', 'SDV Incomplete', Severity.HIGH, 'CRA'),
            ('issue_open_queries', 'Open Queries', Severity.HIGH, 'Data Manager'),
            ('issue_signature_gaps', 'Signature Gaps', Severity.MEDIUM, 'Site'),
            ('issue_sae_dm_pending', 'SAE DM Pending', Severity.CRITICAL, 'Safety DM'),
            ('issue_sae_safety_pending', 'SAE Safety Pending', Severity.CRITICAL, 'Safety Physician'),
            ('issue_missing_visits', 'Missing Visits', Severity.HIGH, 'Site'),
            ('issue_missing_pages', 'Missing Pages', Severity.MEDIUM, 'Site'),
            ('issue_broken_signatures', 'Broken Signatures', Severity.MEDIUM, 'Site'),
        ]
        
        for col, title, severity, owner in issue_checks:
            if issues.get(col, False):
                findings.append(Finding(
                    finding_id="F-{}".format(col.upper()),
                    category="Issue",
                    title=title,
                    description="Patient has {} requiring attention.".format(title.lower()),
                    severity=severity,
                    recommendation="Assign to {} for resolution.".format(owner)
                ))
        
        return findings
    
    def _generate_patient_actions(self, patient_data: Dict, findings: List[Finding]) -> List[ActionItem]:
        actions = []
        
        for i, finding in enumerate(findings[:5]):  # Top 5 actions
            priority = {
                Severity.CRITICAL: "Critical",
                Severity.HIGH: "High",
                Severity.MEDIUM: "Medium",
                Severity.LOW: "Low"
            }.get(finding.severity, "Medium")
            
            due_days = {
                Severity.CRITICAL: 1,
                Severity.HIGH: 3,
                Severity.MEDIUM: 7,
                Severity.LOW: 14
            }.get(finding.severity, 7)
            
            actions.append(ActionItem(
                action_id="ACT-{:03d}".format(i + 1),
                title="Resolve: {}".format(finding.title),
                description=finding.recommendation or "Address this issue.",
                priority=priority,
                owner=finding.recommendation.split()[-1] if finding.recommendation else "TBD",
                due_date=datetime.now() + timedelta(days=due_days),
                related_findings=[finding.finding_id]
            ))
        
        return actions
    
    def _build_patient_overview(self, patient_data: Dict) -> str:
        patients = patient_data.get('patients', {})
        return """
This patient is enrolled at **{site}** in **{study}**.
Current status: **{status}**.
        """.format(
            site=patients.get('site_id', 'Unknown'),
            study=patients.get('study_id', 'Unknown'),
            status=patients.get('subject_status', 'Unknown')
        ).strip()
    
    def _build_dqi_section(self, patient_data: Dict) -> str:
        dqi = patient_data.get('dqi', {})
        score = dqi.get('dqi_score', 0)
        band = dqi.get('dqi_band', 'Unknown')
        
        return """
The patient's Data Quality Index (DQI) is **{:.1f}** ({}).
This score reflects the overall data quality across all metrics.
        """.format(score, band).strip()
    
    def _build_issues_section(self, patient_data: Dict) -> str:
        issues = patient_data.get('issues', {})
        total = issues.get('total_issues', 0)
        primary = issues.get('primary_issue', 'None')
        
        if total == 0:
            return "This patient has no outstanding issues. ✅"
        
        return """
This patient has **{} issue(s)** requiring attention.
Primary issue type: **{}**.
        """.format(total, primary).strip()
    
    def _build_dblock_section(self, patient_data: Dict) -> str:
        dblock = patient_data.get('dblock', {})
        ready = dblock.get('dblock_ready', False)
        eligible = dblock.get('dblock_eligible', False)
        
        if not eligible:
            return "This patient is not eligible for database lock."
        
        if ready:
            return "This patient is **ready for database lock**. ✅"
        
        reason = dblock.get('blocking_reason', 'Unknown')
        days = dblock.get('days_to_ready', 0)
        
        return """
This patient is **not ready** for database lock.
Blocking reason: **{}**.
Estimated days to ready: **{:.0f}**.
        """.format(reason, days).strip()
    
    def _generate_patient_executive_summary(self, patient_data: Dict, metrics: Dict) -> str:
        dqi = metrics.get('dqi_score', 0)
        issues = metrics.get('total_issues', 0)
        status = metrics.get('status', 'Unknown')
        dblock = metrics.get('dblock_ready', False)
        
        if issues == 0 and dblock:
            quality = "excellent"
            action = "No action required."
        elif issues <= 2:
            quality = "good"
            action = "Minor issues require attention."
        elif issues <= 5:
            quality = "fair"
            action = "Several issues need resolution."
        else:
            quality = "needs improvement"
            action = "Multiple issues require immediate attention."
        
        return """
Patient at **{site}** ({study}) has **{quality}** data quality with DQI of **{dqi:.1f}**.
Status: {status}. DB Lock Ready: {dblock}.
{action}
        """.format(
            site=metrics.get('site_id', 'Unknown'),
            study=metrics.get('study_id', 'Unknown'),
            quality=quality,
            dqi=dqi,
            status=status,
            dblock="Yes ✅" if dblock else "No ❌",
            action=action
        ).strip()


# =============================================================================
# SITE SUMMARIZER
# =============================================================================

class SiteSummarizer:
    """Generates site-level narrative summaries."""
    
    def __init__(self, data_loader: SummaryDataLoader):
        self.data = data_loader
    
    def summarize(self, site_id: str) -> Summary:
        """Generate a narrative summary for a specific site."""
        site_data = self.data.get_site_data(site_id)
        
        if not site_data or site_data.get('patients_count', 0) == 0:
            return self._empty_summary(site_id)
        
        # Build metrics
        metrics = self._extract_site_metrics(site_data)
        
        # Generate findings
        findings = self._analyze_site_performance(site_data, metrics)
        
        # Generate action items
        action_items = self._generate_site_actions(findings)
        
        # Build sections
        sections = [
            {
                'title': 'Site Overview',
                'content': self._build_site_overview(site_data, metrics)
            },
            {
                'title': 'Performance Analysis',
                'content': self._build_performance_section(metrics)
            },
            {
                'title': 'Data Quality',
                'content': self._build_site_dqi_section(metrics)
            },
            {
                'title': 'Issue Breakdown',
                'content': self._build_site_issues_section(site_data)
            },
            {
                'title': 'Recommendations',
                'content': self._build_recommendations_section(findings)
            }
        ]
        
        # Executive summary
        exec_summary = self._generate_site_executive_summary(site_data, metrics, findings)
        
        return Summary(
            summary_id="SUM-SITE-{}".format(hashlib.md5(site_id.encode()).hexdigest()[:8]),
            summary_type=SummaryType.SITE,
            entity_id=site_id,
            entity_name=site_id,
            title="Site Performance Summary: {}".format(site_id),
            executive_summary=exec_summary,
            sections=sections,
            findings=findings,
            action_items=action_items,
            metrics=metrics,
            generated_at=datetime.now()
        )
    
    def _empty_summary(self, site_id: str) -> Summary:
        return Summary(
            summary_id="SUM-SITE-EMPTY",
            summary_type=SummaryType.SITE,
            entity_id=site_id,
            entity_name=site_id,
            title="Site Summary: {}".format(site_id),
            executive_summary="No data available for site {}.".format(site_id),
            sections=[],
            findings=[],
            action_items=[],
            metrics={},
            generated_at=datetime.now()
        )
    
    def _extract_site_metrics(self, site_data: Dict) -> Dict:
        metrics = {
            'patient_count': site_data.get('patients_count', 0),
            'issues_count': site_data.get('issues_count', 0)
        }
        
        # Get from benchmark data
        benchmark = site_data.get('benchmark', {})
        if benchmark:
            metrics['composite_score'] = benchmark.get('composite_score', 0)
            metrics['performance_tier'] = benchmark.get('performance_tier', 'Unknown')
            metrics['percentile'] = benchmark.get('percentile', 0)
            metrics['dqi_mean'] = benchmark.get('dqi_mean', 0)
            metrics['tier2_clean_rate'] = benchmark.get('tier2_clean_rate', 0)
            metrics['dblock_ready_rate'] = benchmark.get('dblock_ready_rate', 0)
        
        # Calculate from aggregated data
        if 'dqi_dqi_score_mean' in site_data:
            metrics['dqi_mean'] = site_data['dqi_dqi_score_mean']
        
        if 'issues_total_issues_sum' in site_data:
            metrics['total_issues'] = site_data['issues_total_issues_sum']
        
        return metrics
    
    def _analyze_site_performance(self, site_data: Dict, metrics: Dict) -> List[Finding]:
        findings = []
        
        # Performance tier finding
        tier = metrics.get('performance_tier', 'Unknown')
        if tier in ['At Risk', 'Needs Improvement']:
            findings.append(Finding(
                finding_id="F-TIER",
                category="Performance",
                title="Site Performance Below Target",
                description="Site is classified as '{}' based on composite scoring.".format(tier),
                severity=Severity.HIGH if tier == 'At Risk' else Severity.MEDIUM,
                recommendation="Implement performance improvement plan."
            ))
        elif tier in ['Strong', 'Exceptional']:
            findings.append(Finding(
                finding_id="F-TIER-GOOD",
                category="Performance",
                title="Strong Site Performance",
                description="Site is classified as '{}' - maintain current practices.".format(tier),
                severity=Severity.INFO
            ))
        
        # DQI finding
        dqi = metrics.get('dqi_mean', 0)
        if dqi < 85:
            findings.append(Finding(
                finding_id="F-DQI-LOW",
                category="Data Quality",
                title="DQI Below Target",
                description="Mean DQI of {:.1f} is below the 85 target.".format(dqi),
                severity=Severity.HIGH,
                metric_value=dqi,
                metric_name="Mean DQI",
                recommendation="Focus on data quality improvement initiatives."
            ))
        
        # Clean rate finding
        clean_rate = metrics.get('tier2_clean_rate', 0)
        if clean_rate < 50:
            findings.append(Finding(
                finding_id="F-CLEAN-LOW",
                category="Clean Status",
                title="Low Clean Patient Rate",
                description="Only {:.1f}% of patients are Tier 2 clean.".format(clean_rate),
                severity=Severity.MEDIUM,
                metric_value=clean_rate,
                metric_name="Clean Rate",
                recommendation="Address blocking issues for clean status."
            ))
        
        # Issue count finding
        total_issues = metrics.get('total_issues', 0)
        patient_count = metrics.get('patient_count', 1)
        issues_per_patient = total_issues / max(patient_count, 1)
        
        if issues_per_patient > 2:
            findings.append(Finding(
                finding_id="F-ISSUES-HIGH",
                category="Issues",
                title="High Issue Density",
                description="Average of {:.1f} issues per patient.".format(issues_per_patient),
                severity=Severity.HIGH,
                metric_value=issues_per_patient,
                metric_name="Issues/Patient",
                recommendation="Conduct root cause analysis and implement corrective actions."
            ))
        
        return findings
    
    def _generate_site_actions(self, findings: List[Finding]) -> List[ActionItem]:
        actions = []
        
        for i, finding in enumerate(findings):
            if finding.severity in [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM]:
                priority = {
                    Severity.CRITICAL: "Critical",
                    Severity.HIGH: "High",
                    Severity.MEDIUM: "Medium"
                }.get(finding.severity, "Medium")
                
                actions.append(ActionItem(
                    action_id="ACT-SITE-{:03d}".format(i + 1),
                    title="Address: {}".format(finding.title),
                    description=finding.recommendation or finding.description,
                    priority=priority,
                    owner="Site Lead",
                    due_date=datetime.now() + timedelta(days=7),
                    related_findings=[finding.finding_id]
                ))
        
        return actions
    
    def _build_site_overview(self, site_data: Dict, metrics: Dict) -> str:
        return """
**{}** has **{:,}** patients enrolled.
Performance Tier: **{}** ({}th percentile).
        """.format(
            site_data.get('site_id', 'Unknown'),
            metrics.get('patient_count', 0),
            metrics.get('performance_tier', 'Unknown'),
            metrics.get('percentile', 0)
        ).strip()
    
    def _build_performance_section(self, metrics: Dict) -> str:
        return """
**Composite Score**: {:.1f}/100
**Performance Tier**: {}
**Percentile Rank**: {}th

The composite score reflects overall site performance across data quality, 
clean patient rate, and DB lock readiness metrics.
        """.format(
            metrics.get('composite_score', 0),
            metrics.get('performance_tier', 'Unknown'),
            metrics.get('percentile', 0)
        ).strip()
    
    def _build_site_dqi_section(self, metrics: Dict) -> str:
        dqi = metrics.get('dqi_mean', 0)
        status = "excellent" if dqi >= 95 else "good" if dqi >= 85 else "needs improvement"
        
        return """
**Mean DQI**: {:.1f}
**Status**: {}

Data quality at this site is {}.
        """.format(dqi, status.title(), status).strip()
    
    def _build_site_issues_section(self, site_data: Dict) -> str:
        total = site_data.get('issues_total_issues_sum', 0)
        patients = site_data.get('issues_count', 1)
        
        return """
**Total Issues**: {:,.0f}
**Patients with Issues**: {:,}
**Issues per Patient**: {:.1f}
        """.format(total, patients, total / max(patients, 1)).strip()
    
    def _build_recommendations_section(self, findings: List[Finding]) -> str:
        if not findings:
            return "No specific recommendations at this time. Continue monitoring."
        
        recs = []
        for finding in findings:
            if finding.recommendation:
                recs.append("- {}".format(finding.recommendation))
        
        return "\n".join(recs) if recs else "Review findings and develop action plan."
    
    def _generate_site_executive_summary(self, site_data: Dict, metrics: Dict, findings: List[Finding]) -> str:
        tier = metrics.get('performance_tier', 'Unknown')
        dqi = metrics.get('dqi_mean', 0)
        patients = metrics.get('patient_count', 0)
        
        critical_findings = [f for f in findings if f.severity == Severity.CRITICAL]
        high_findings = [f for f in findings if f.severity == Severity.HIGH]
        
        if tier in ['Exceptional', 'Strong']:
            status = "performing well"
            action = "Continue current practices and maintain performance."
        elif tier == 'Average':
            status = "performing adequately"
            action = "Focus on improvement opportunities."
        else:
            status = "needs attention"
            action = "Implement performance improvement initiatives."
        
        alerts = ""
        if critical_findings:
            alerts = " **{} critical issue(s)** require immediate attention.".format(len(critical_findings))
        elif high_findings:
            alerts = " {} high-priority issue(s) identified.".format(len(high_findings))
        
        return """
**{site}** has **{patients:,}** patients and is {status} with a composite score of **{score:.1f}** 
and mean DQI of **{dqi:.1f}**. Performance tier: **{tier}**.{alerts}

{action}
        """.format(
            site=site_data.get('site_id', 'Unknown'),
            patients=patients,
            status=status,
            score=metrics.get('composite_score', 0),
            dqi=dqi,
            tier=tier,
            alerts=alerts,
            action=action
        ).strip()


# =============================================================================
# DAILY DIGEST GENERATOR
# =============================================================================

class DailyDigestGenerator:
    """Generates daily digest summaries for stakeholders."""
    
    def __init__(self, data_loader: SummaryDataLoader):
        self.data = data_loader
    
    def generate(self, 
                 study_id: Optional[str] = None,
                 recipient_role: str = "Study Lead") -> Summary:
        """Generate a daily digest summary."""
        
        if study_id:
            portfolio_data = self.data.get_study_data(study_id)
            entity_id = study_id
            entity_name = study_id
        else:
            portfolio_data = self.data.get_portfolio_data()
            entity_id = "portfolio"
            entity_name = "Portfolio"
        
        # Build metrics
        metrics = self._extract_digest_metrics(portfolio_data)
        
        # Generate findings
        findings = self._analyze_daily_changes(portfolio_data, metrics)
        
        # Generate action items
        action_items = self._generate_digest_actions(findings)
        
        # Build sections based on recipient role
        sections = self._build_digest_sections(portfolio_data, metrics, findings, recipient_role)
        
        # Executive summary
        exec_summary = self._generate_digest_executive_summary(portfolio_data, metrics, findings)
        
        return Summary(
            summary_id="SUM-DAILY-{}".format(datetime.now().strftime('%Y%m%d')),
            summary_type=SummaryType.DAILY_DIGEST,
            entity_id=entity_id,
            entity_name=entity_name,
            title="Daily Digest - {}".format(datetime.now().strftime('%B %d, %Y')),
            executive_summary=exec_summary,
            sections=sections,
            findings=findings,
            action_items=action_items,
            metrics=metrics,
            generated_at=datetime.now(),
            metadata={'recipient_role': recipient_role}
        )
    
    def _extract_digest_metrics(self, data: Dict) -> Dict:
        return {
            'total_patients': data.get('total_patients', 0),
            'total_studies': data.get('total_studies', 0),
            'total_sites': data.get('total_sites', 0),
            'mean_dqi': data.get('mean_dqi', 0),
            'tier1_clean_rate': data.get('tier1_clean_rate', 0),
            'tier2_clean_rate': data.get('tier2_clean_rate', 0),
            'dblock_ready_rate': data.get('dblock_ready_rate', 0),
            'dblock_eligible': data.get('dblock_eligible_count', 0),
            'total_issues': data.get('total_issues', 0),
            'patients_with_issues': data.get('patients_with_issues', 0)
        }
    
    def _analyze_daily_changes(self, data: Dict, metrics: Dict) -> List[Finding]:
        findings = []
        
        # DQI status
        dqi = metrics.get('mean_dqi', 0)
        if dqi >= 95:
            findings.append(Finding(
                finding_id="F-DQI-STATUS",
                category="Data Quality",
                title="Excellent Data Quality",
                description="Portfolio DQI of {:.1f} exceeds target.".format(dqi),
                severity=Severity.INFO,
                metric_value=dqi,
                metric_name="Mean DQI"
            ))
        elif dqi < 85:
            findings.append(Finding(
                finding_id="F-DQI-LOW",
                category="Data Quality",
                title="DQI Below Target",
                description="Portfolio DQI of {:.1f} is below 85 target.".format(dqi),
                severity=Severity.HIGH,
                metric_value=dqi,
                metric_name="Mean DQI",
                recommendation="Prioritize data quality improvement."
            ))
        
        # Clean rate
        clean_rate = metrics.get('tier2_clean_rate', 0)
        if clean_rate < 50:
            findings.append(Finding(
                finding_id="F-CLEAN-RATE",
                category="Clean Status",
                title="Clean Rate Below 50%",
                description="Only {:.1f}% of patients are Tier 2 clean.".format(clean_rate),
                severity=Severity.MEDIUM,
                metric_value=clean_rate,
                metric_name="Tier 2 Clean Rate",
                recommendation="Focus on resolving blocking issues."
            ))
        
        # DB Lock readiness
        dblock_rate = metrics.get('dblock_ready_rate', 0)
        if dblock_rate < 20:
            findings.append(Finding(
                finding_id="F-DBLOCK-LOW",
                category="DB Lock",
                title="Low DB Lock Readiness",
                description="Only {:.1f}% of eligible patients are DB Lock ready.".format(dblock_rate),
                severity=Severity.MEDIUM,
                metric_value=dblock_rate,
                metric_name="DB Lock Ready Rate",
                recommendation="Accelerate cleanup activities."
            ))
        
        # Issue volume
        patients_with_issues = metrics.get('patients_with_issues', 0)
        total_patients = metrics.get('total_patients', 1)
        issue_rate = (patients_with_issues / max(total_patients, 1)) * 100
        
        if issue_rate > 50:
            findings.append(Finding(
                finding_id="F-ISSUE-RATE",
                category="Issues",
                title="High Issue Rate",
                description="{:.1f}% of patients have outstanding issues.".format(issue_rate),
                severity=Severity.HIGH,
                metric_value=issue_rate,
                metric_name="Issue Rate",
                recommendation="Conduct portfolio-wide issue resolution campaign."
            ))
        
        return findings
    
    def _generate_digest_actions(self, findings: List[Finding]) -> List[ActionItem]:
        actions = []
        
        for i, finding in enumerate(findings):
            if finding.severity in [Severity.CRITICAL, Severity.HIGH]:
                actions.append(ActionItem(
                    action_id="ACT-DIGEST-{:03d}".format(i + 1),
                    title=finding.title,
                    description=finding.recommendation or finding.description,
                    priority="High" if finding.severity == Severity.HIGH else "Critical",
                    owner="Study Lead",
                    due_date=datetime.now() + timedelta(days=3),
                    related_findings=[finding.finding_id]
                ))
        
        return actions
    
    def _build_digest_sections(self, data: Dict, metrics: Dict, findings: List[Finding], role: str) -> List[Dict]:
        sections = []
        
        # Key Metrics Section
        sections.append({
            'title': 'Key Metrics at a Glance',
            'content': """
**Patients**: {:,} across {} studies and {:,} sites
**Mean DQI**: {:.1f}
**Tier 2 Clean**: {:.1f}%
**DB Lock Ready**: {:.1f}%
**Patients with Issues**: {:,}
            """.format(
                metrics.get('total_patients', 0),
                metrics.get('total_studies', 0),
                metrics.get('total_sites', 0),
                metrics.get('mean_dqi', 0),
                metrics.get('tier2_clean_rate', 0),
                metrics.get('dblock_ready_rate', 0),
                metrics.get('patients_with_issues', 0)
            ).strip()
        })
        
        # Highlights Section
        highlights = []
        if metrics.get('mean_dqi', 0) >= 95:
            highlights.append("✅ DQI exceeds target at {:.1f}".format(metrics.get('mean_dqi', 0)))
        if metrics.get('tier2_clean_rate', 0) >= 60:
            highlights.append("✅ Clean rate on track at {:.1f}%".format(metrics.get('tier2_clean_rate', 0)))
        
        if highlights:
            sections.append({
                'title': 'Highlights',
                'content': '\n'.join(highlights)
            })
        
        # Attention Required Section
        attention = []
        for finding in findings:
            if finding.severity in [Severity.CRITICAL, Severity.HIGH]:
                attention.append("⚠️ {}: {}".format(finding.title, finding.description))
        
        if attention:
            sections.append({
                'title': 'Attention Required',
                'content': '\n'.join(attention)
            })
        
        # Role-specific section
        if role == "Study Lead":
            sections.append({
                'title': 'Study Lead Focus Areas',
                'content': """
- Review site performance dashboard
- Follow up on escalated issues
- Prepare for upcoming sponsor meeting
                """.strip()
            })
        elif role == "CRA":
            sections.append({
                'title': 'CRA Priorities Today',
                'content': """
- Complete pending SDV activities
- Follow up on open queries at assigned sites
- Review patient visit schedules
                """.strip()
            })
        elif role == "Data Manager":
            sections.append({
                'title': 'Data Manager Priorities',
                'content': """
- Review and resolve open queries
- Monitor coding completion
- Check SAE reconciliation status
                """.strip()
            })
        
        return sections
    
    def _generate_digest_executive_summary(self, data: Dict, metrics: Dict, findings: List[Finding]) -> str:
        total_patients = metrics.get('total_patients', 0)
        dqi = metrics.get('mean_dqi', 0)
        clean_rate = metrics.get('tier2_clean_rate', 0)
        issues = metrics.get('patients_with_issues', 0)
        
        critical = len([f for f in findings if f.severity == Severity.CRITICAL])
        high = len([f for f in findings if f.severity == Severity.HIGH])
        
        alerts = ""
        if critical > 0:
            alerts = " **{} critical** and".format(critical)
        if high > 0:
            alerts = "{} **{} high-priority** items need attention.".format(alerts, high)
        else:
            alerts = " All metrics within acceptable ranges."
        
        return """
**Daily Summary for {}**

Portfolio has **{:,}** patients with mean DQI of **{:.1f}** and **{:.1f}%** Tier 2 clean rate.
Currently **{:,}** patients have outstanding issues.{}
        """.format(
            datetime.now().strftime('%B %d, %Y'),
            total_patients,
            dqi,
            clean_rate,
            issues,
            alerts
        ).strip()


# =============================================================================
# INVESTIGATION SUMMARIZER
# =============================================================================

class InvestigationSummarizer:
    """Generates investigation summaries with root cause insights."""
    
    def __init__(self, data_loader: SummaryDataLoader):
        self.data = data_loader
    
    def summarize(self, 
                  entity_id: str,
                  entity_type: str = "site",
                  issue_focus: Optional[str] = None) -> Summary:
        """Generate an investigation summary."""
        
        if entity_type == "site":
            entity_data = self.data.get_site_data(entity_id)
        elif entity_type == "study":
            entity_data = self.data.get_study_data(entity_id)
        else:
            entity_data = self.data.get_portfolio_data()
            entity_id = "portfolio"
        
        # Build investigation context
        investigation_context = self._build_investigation_context(entity_data, issue_focus)
        
        # Generate hypotheses
        hypotheses = self._generate_hypotheses(entity_data, issue_focus)
        
        # Build findings from hypotheses
        findings = self._hypotheses_to_findings(hypotheses)
        
        # Generate action items
        action_items = self._generate_investigation_actions(findings)
        
        # Build sections
        sections = [
            {
                'title': 'Investigation Scope',
                'content': self._build_scope_section(entity_id, entity_type, issue_focus)
            },
            {
                'title': 'Current State Analysis',
                'content': self._build_current_state_section(entity_data)
            },
            {
                'title': 'Root Cause Hypotheses',
                'content': self._build_hypotheses_section(hypotheses)
            },
            {
                'title': 'Evidence Summary',
                'content': self._build_evidence_section(entity_data, hypotheses)
            },
            {
                'title': 'Recommended Actions',
                'content': self._build_actions_section(action_items)
            }
        ]
        
        # Executive summary
        exec_summary = self._generate_investigation_summary(entity_data, hypotheses)
        
        return Summary(
            summary_id="SUM-INV-{}".format(hashlib.md5("{}{}".format(entity_id, datetime.now()).encode()).hexdigest()[:8]),
            summary_type=SummaryType.INVESTIGATION,
            entity_id=entity_id,
            entity_name=entity_id,
            title="Investigation Summary: {}".format(entity_id),
            executive_summary=exec_summary,
            sections=sections,
            findings=findings,
            action_items=action_items,
            metrics=investigation_context,
            generated_at=datetime.now(),
            metadata={'entity_type': entity_type, 'issue_focus': issue_focus}
        )
    
    def _build_investigation_context(self, entity_data: Dict, issue_focus: Optional[str]) -> Dict:
        context = {
            'investigation_date': datetime.now().isoformat(),
            'issue_focus': issue_focus or "general"
        }
        
        if 'patients_count' in entity_data:
            context['patient_count'] = entity_data['patients_count']
        if 'issues_total_issues_sum' in entity_data:
            context['total_issues'] = entity_data['issues_total_issues_sum']
        
        benchmark = entity_data.get('benchmark', {})
        if benchmark:
            context['performance_tier'] = benchmark.get('performance_tier', 'Unknown')
            context['composite_score'] = benchmark.get('composite_score', 0)
        
        return context
    
    def _generate_hypotheses(self, entity_data: Dict, issue_focus: Optional[str]) -> List[Dict]:
        hypotheses = []
        
        benchmark = entity_data.get('benchmark', {})
        tier = benchmark.get('performance_tier', 'Unknown')
        
        # Performance-based hypotheses
        if tier in ['At Risk', 'Needs Improvement']:
            hypotheses.append({
                'id': 'H1',
                'title': 'Resource Constraint',
                'description': 'Site may have insufficient staff to handle patient volume.',
                'confidence': 75,
                'evidence': ['Low performance tier', 'High issue density'],
                'recommendation': 'Assess staffing levels and workload distribution.'
            })
            
            hypotheses.append({
                'id': 'H2',
                'title': 'Training Gap',
                'description': 'Staff may need additional training on data entry procedures.',
                'confidence': 65,
                'evidence': ['Query patterns suggest data entry errors'],
                'recommendation': 'Schedule refresher training session.'
            })
        
        # Issue-specific hypotheses
        if issue_focus == 'sdv_incomplete' or issue_focus is None:
            hypotheses.append({
                'id': 'H3',
                'title': 'CRA Availability',
                'description': 'CRA monitoring visits may be delayed or insufficient.',
                'confidence': 70,
                'evidence': ['SDV completion rate below target'],
                'recommendation': 'Review CRA visit schedule and prioritize SDV.'
            })
        
        if issue_focus == 'open_queries' or issue_focus is None:
            hypotheses.append({
                'id': 'H4',
                'title': 'Query Response Process',
                'description': 'Site query response process may be inefficient.',
                'confidence': 60,
                'evidence': ['High query aging', 'Query volume trends'],
                'recommendation': 'Implement query escalation procedure.'
            })
        
        if issue_focus == 'signature_gaps' or issue_focus is None:
            hypotheses.append({
                'id': 'H5',
                'title': 'PI Availability',
                'description': 'Principal Investigator may have limited availability for signatures.',
                'confidence': 80,
                'evidence': ['Signature delays correlated with PI schedule'],
                'recommendation': 'Schedule dedicated signature sessions with PI.'
            })
        
        return hypotheses
    
    def _hypotheses_to_findings(self, hypotheses: List[Dict]) -> List[Finding]:
        findings = []
        
        for hyp in hypotheses:
            severity = Severity.HIGH if hyp['confidence'] >= 70 else Severity.MEDIUM
            
            findings.append(Finding(
                finding_id="F-{}".format(hyp['id']),
                category="Root Cause",
                title=hyp['title'],
                description=hyp['description'],
                severity=severity,
                metric_value=hyp['confidence'],
                metric_name="Confidence %",
                recommendation=hyp['recommendation']
            ))
        
        return findings
    
    def _generate_investigation_actions(self, findings: List[Finding]) -> List[ActionItem]:
        actions = []
        
        for i, finding in enumerate(findings):
            actions.append(ActionItem(
                action_id="ACT-INV-{:03d}".format(i + 1),
                title="Investigate: {}".format(finding.title),
                description=finding.recommendation or finding.description,
                priority="High" if finding.severity == Severity.HIGH else "Medium",
                owner="Investigation Lead",
                due_date=datetime.now() + timedelta(days=5),
                related_findings=[finding.finding_id]
            ))
        
        return actions
    
    def _build_scope_section(self, entity_id: str, entity_type: str, issue_focus: Optional[str]) -> str:
        focus_text = issue_focus if issue_focus else "all issue types"
        return """
**Entity**: {} ({})
**Focus Area**: {}
**Investigation Date**: {}
        """.format(entity_id, entity_type, focus_text, datetime.now().strftime('%Y-%m-%d')).strip()
    
    def _build_current_state_section(self, entity_data: Dict) -> str:
        patients = entity_data.get('patients_count', 0)
        issues = entity_data.get('issues_total_issues_sum', 0)
        benchmark = entity_data.get('benchmark', {})
        
        return """
**Patient Count**: {:,}
**Total Issues**: {:,.0f}
**Performance Tier**: {}
**Composite Score**: {:.1f}
        """.format(
            patients,
            issues,
            benchmark.get('performance_tier', 'Unknown'),
            benchmark.get('composite_score', 0)
        ).strip()
    
    def _build_hypotheses_section(self, hypotheses: List[Dict]) -> str:
        if not hypotheses:
            return "No specific hypotheses generated."
        
        lines = []
        for hyp in hypotheses:
            lines.append("**{} - {}** (Confidence: {}%)".format(
                hyp['id'], hyp['title'], hyp['confidence']
            ))
            lines.append(hyp['description'])
            lines.append("")
        
        return "\n".join(lines).strip()
    
    def _build_evidence_section(self, entity_data: Dict, hypotheses: List[Dict]) -> str:
        evidence_items = []
        
        for hyp in hypotheses:
            for ev in hyp.get('evidence', []):
                if ev not in evidence_items:
                    evidence_items.append(ev)
        
        if not evidence_items:
            return "Evidence collection in progress."
        
        return "\n".join(["- {}".format(e) for e in evidence_items])
    
    def _build_actions_section(self, actions: List[ActionItem]) -> str:
        if not actions:
            return "No immediate actions required."
        
        lines = []
        for action in actions:
            lines.append("- **{}**: {} (Owner: {})".format(
                action.priority, action.title, action.owner
            ))
        
        return "\n".join(lines)
    
    def _generate_investigation_summary(self, entity_data: Dict, hypotheses: List[Dict]) -> str:
        num_hypotheses = len(hypotheses)
        high_confidence = len([h for h in hypotheses if h['confidence'] >= 70])
        
        return """
Investigation identified **{}** potential root causes, with **{}** having high confidence (≥70%).
Key areas of focus include resource constraints, process efficiency, and stakeholder availability.
Review the detailed hypotheses and implement recommended actions.
        """.format(num_hypotheses, high_confidence).strip()


# =============================================================================
# EXECUTIVE SUMMARIZER
# =============================================================================

class ExecutiveSummarizer:
    """Generates executive-level summaries."""
    
    def __init__(self, data_loader: SummaryDataLoader):
        self.data = data_loader
    
    def summarize(self, study_id: Optional[str] = None) -> Summary:
        """Generate an executive summary."""
        
        if study_id:
            data = self.data.get_study_data(study_id)
            entity_id = study_id
            entity_name = study_id
        else:
            data = self.data.get_portfolio_data()
            entity_id = "portfolio"
            entity_name = "Portfolio"
        
        metrics = self._extract_executive_metrics(data)
        findings = self._analyze_executive_insights(metrics)
        action_items = self._generate_executive_actions(findings)
        
        sections = [
            {'title': 'Portfolio Overview', 'content': self._build_overview(metrics)},
            {'title': 'Key Performance Indicators', 'content': self._build_kpi_section(metrics)},
            {'title': 'Risk Assessment', 'content': self._build_risk_section(findings)},
            {'title': 'Strategic Recommendations', 'content': self._build_strategy_section(findings)}
        ]
        
        exec_summary = self._generate_executive_summary_text(metrics, findings)
        
        return Summary(
            summary_id="SUM-EXEC-{}".format(datetime.now().strftime('%Y%m%d')),
            summary_type=SummaryType.EXECUTIVE,
            entity_id=entity_id,
            entity_name=entity_name,
            title="Executive Summary - {}".format(entity_name),
            executive_summary=exec_summary,
            sections=sections,
            findings=findings,
            action_items=action_items,
            metrics=metrics,
            generated_at=datetime.now()
        )
    
    def _extract_executive_metrics(self, data: Dict) -> Dict:
        return {
            'total_patients': data.get('total_patients', 0),
            'total_studies': data.get('total_studies', 0),
            'total_sites': data.get('total_sites', 0),
            'mean_dqi': data.get('mean_dqi', 0),
            'tier1_clean_rate': data.get('tier1_clean_rate', 0),
            'tier2_clean_rate': data.get('tier2_clean_rate', 0),
            'dblock_ready_rate': data.get('dblock_ready_rate', 0),
            'total_issues': data.get('total_issues', 0)
        }
    
    def _analyze_executive_insights(self, metrics: Dict) -> List[Finding]:
        findings = []
        
        dqi = metrics.get('mean_dqi', 0)
        if dqi >= 95:
            findings.append(Finding(
                finding_id="F-EXEC-DQI",
                category="Data Quality",
                title="Excellent Data Quality",
                description="Portfolio DQI of {:.1f} indicates strong data quality management.".format(dqi),
                severity=Severity.INFO
            ))
        elif dqi < 85:
            findings.append(Finding(
                finding_id="F-EXEC-DQI-LOW",
                category="Data Quality",
                title="Data Quality Risk",
                description="Portfolio DQI of {:.1f} is below target threshold.".format(dqi),
                severity=Severity.HIGH,
                recommendation="Escalate data quality improvement initiatives."
            ))
        
        clean_rate = metrics.get('tier2_clean_rate', 0)
        if clean_rate < 50:
            findings.append(Finding(
                finding_id="F-EXEC-CLEAN",
                category="Clean Status",
                title="Clean Rate Concern",
                description="Only {:.1f}% of patients are Tier 2 clean.".format(clean_rate),
                severity=Severity.MEDIUM,
                recommendation="Allocate additional resources to clean patient initiatives."
            ))
        
        return findings
    
    def _generate_executive_actions(self, findings: List[Finding]) -> List[ActionItem]:
        actions = []
        
        for i, finding in enumerate(findings):
            if finding.severity in [Severity.HIGH, Severity.CRITICAL]:
                actions.append(ActionItem(
                    action_id="ACT-EXEC-{:03d}".format(i + 1),
                    title=finding.title,
                    description=finding.recommendation or finding.description,
                    priority="High",
                    owner="Executive Sponsor",
                    due_date=datetime.now() + timedelta(days=7)
                ))
        
        return actions
    
    def _build_overview(self, metrics: Dict) -> str:
        return """
The portfolio encompasses **{:,}** patients across **{}** studies and **{:,}** sites.
        """.format(
            metrics.get('total_patients', 0),
            metrics.get('total_studies', 0),
            metrics.get('total_sites', 0)
        ).strip()
    
    def _build_kpi_section(self, metrics: Dict) -> str:
        return """
| KPI | Value | Target | Status |
|-----|-------|--------|--------|
| Mean DQI | {:.1f} | 90.0 | {} |
| Tier 2 Clean | {:.1f}% | 70% | {} |
| DB Lock Ready | {:.1f}% | 30% | {} |
        """.format(
            metrics.get('mean_dqi', 0),
            "✅" if metrics.get('mean_dqi', 0) >= 90 else "⚠️",
            metrics.get('tier2_clean_rate', 0),
            "✅" if metrics.get('tier2_clean_rate', 0) >= 70 else "⚠️",
            metrics.get('dblock_ready_rate', 0),
            "✅" if metrics.get('dblock_ready_rate', 0) >= 30 else "⚠️"
        ).strip()
    
    def _build_risk_section(self, findings: List[Finding]) -> str:
        risks = [f for f in findings if f.severity in [Severity.HIGH, Severity.CRITICAL]]
        
        if not risks:
            return "No significant risks identified at this time."
        
        lines = []
        for risk in risks:
            lines.append("⚠️ **{}**: {}".format(risk.title, risk.description))
        
        return "\n".join(lines)
    
    def _build_strategy_section(self, findings: List[Finding]) -> str:
        recs = [f.recommendation for f in findings if f.recommendation]
        
        if not recs:
            return "Continue current operational practices."
        
        return "\n".join(["- {}".format(r) for r in recs])
    
    def _generate_executive_summary_text(self, metrics: Dict, findings: List[Finding]) -> str:
        dqi = metrics.get('mean_dqi', 0)
        clean = metrics.get('tier2_clean_rate', 0)
        patients = metrics.get('total_patients', 0)
        
        risks = len([f for f in findings if f.severity in [Severity.HIGH, Severity.CRITICAL]])
        
        status = "on track" if dqi >= 90 and clean >= 60 else "needs attention"
        
        return """
Portfolio performance is **{}** with **{:,}** patients, DQI of **{:.1f}**, and **{:.1f}%** Tier 2 clean rate.
**{}** high-priority items require executive attention.
        """.format(status, patients, dqi, clean, risks).strip()


# =============================================================================
# MAIN AUTO-SUMMARIZER ENGINE
# =============================================================================

class AutoSummarizer:
    """
    Main auto-summarization engine.
    
    Provides a unified interface for all summary types.
    """
    
    def __init__(self):
        self.data_loader = SummaryDataLoader()
        self.patient_summarizer = PatientSummarizer(self.data_loader)
        self.site_summarizer = SiteSummarizer(self.data_loader)
        self.daily_digest = DailyDigestGenerator(self.data_loader)
        self.investigation_summarizer = InvestigationSummarizer(self.data_loader)
        self.executive_summarizer = ExecutiveSummarizer(self.data_loader)
        
        # Statistics
        self.stats = {
            'summaries_generated': 0,
            'by_type': {}
        }
    
    def summarize_patient(self, patient_key: str) -> Summary:
        """Generate a patient summary."""
        summary = self.patient_summarizer.summarize(patient_key)
        self._track_stats(SummaryType.PATIENT)
        return summary
    
    def summarize_site(self, site_id: str) -> Summary:
        """Generate a site narrative summary."""
        summary = self.site_summarizer.summarize(site_id)
        self._track_stats(SummaryType.SITE)
        return summary
    
    def generate_daily_digest(self, 
                              study_id: Optional[str] = None,
                              recipient_role: str = "Study Lead") -> Summary:
        """Generate a daily digest."""
        summary = self.daily_digest.generate(study_id, recipient_role)
        self._track_stats(SummaryType.DAILY_DIGEST)
        return summary
    
    def summarize_investigation(self,
                                entity_id: str,
                                entity_type: str = "site",
                                issue_focus: Optional[str] = None) -> Summary:
        """Generate an investigation summary."""
        summary = self.investigation_summarizer.summarize(entity_id, entity_type, issue_focus)
        self._track_stats(SummaryType.INVESTIGATION)
        return summary
    
    def generate_executive_summary(self, study_id: Optional[str] = None) -> Summary:
        """Generate an executive summary."""
        summary = self.executive_summarizer.summarize(study_id)
        self._track_stats(SummaryType.EXECUTIVE)
        return summary
    
    def summarize(self, 
                  summary_type: SummaryType,
                  entity_id: Optional[str] = None,
                  **kwargs) -> Summary:
        """
        Unified summarization interface.
        
        Args:
            summary_type: Type of summary to generate
            entity_id: Entity identifier (patient_key, site_id, study_id)
            **kwargs: Additional arguments specific to summary type
            
        Returns:
            Summary object
        """
        if summary_type == SummaryType.PATIENT:
            return self.summarize_patient(entity_id)
        elif summary_type == SummaryType.SITE:
            return self.summarize_site(entity_id)
        elif summary_type == SummaryType.DAILY_DIGEST:
            return self.generate_daily_digest(entity_id, kwargs.get('recipient_role', 'Study Lead'))
        elif summary_type == SummaryType.INVESTIGATION:
            return self.summarize_investigation(
                entity_id or 'portfolio',
                kwargs.get('entity_type', 'site'),
                kwargs.get('issue_focus')
            )
        elif summary_type == SummaryType.EXECUTIVE:
            return self.generate_executive_summary(entity_id)
        else:
            raise ValueError("Unsupported summary type: {}".format(summary_type))
    
    def _track_stats(self, summary_type: SummaryType):
        self.stats['summaries_generated'] += 1
        type_key = summary_type.value
        self.stats['by_type'][type_key] = self.stats['by_type'].get(type_key, 0) + 1
    
    def get_stats(self) -> Dict:
        return self.stats
    
    def save_summary(self, summary: Summary, output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Save summary to files in multiple formats.
        
        Returns:
            Dict mapping format to file path
        """
        output_dir = output_dir or str(Path(DATA_OUTPUTS) / 'summaries')
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        base_name = "{}_{}".format(summary.summary_type.value, summary.summary_id)
        saved_files = {}
        
        # Save markdown
        md_path = Path(output_dir) / "{}.md".format(base_name)
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(summary.to_markdown())
        saved_files['markdown'] = str(md_path)
        
        # Save HTML
        html_path = Path(output_dir) / "{}.html".format(base_name)
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(summary.to_html())
        saved_files['html'] = str(html_path)
        
        # Save JSON
        json_path = Path(output_dir) / "{}.json".format(base_name)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary.to_dict(), f, indent=2, default=str)
        saved_files['json'] = str(json_path)
        
        return saved_files


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_auto_summarizer = None

def get_auto_summarizer() -> AutoSummarizer:
    """Get or create the auto-summarizer singleton."""
    global _auto_summarizer
    if _auto_summarizer is None:
        _auto_summarizer = AutoSummarizer()
    return _auto_summarizer


def summarize_patient(patient_key: str) -> Summary:
    """Quick patient summary."""
    return get_auto_summarizer().summarize_patient(patient_key)


def summarize_site(site_id: str) -> Summary:
    """Quick site summary."""
    return get_auto_summarizer().summarize_site(site_id)


def daily_digest(study_id: Optional[str] = None, role: str = "Study Lead") -> Summary:
    """Quick daily digest."""
    return get_auto_summarizer().generate_daily_digest(study_id, role)


def executive_summary(study_id: Optional[str] = None) -> Summary:
    """Quick executive summary."""
    return get_auto_summarizer().generate_executive_summary(study_id)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SANCHALAK AI - Auto-Summarization Engine v1.0")
    print("=" * 60)
    
    summarizer = get_auto_summarizer()
    
    print("\n📊 Data loaded:")
    portfolio = summarizer.data_loader.get_portfolio_data()
    for key, value in portfolio.items():
        if isinstance(value, float):
            print("   {}: {:.2f}".format(key, value))
        elif isinstance(value, int):
            print("   {}: {:,}".format(key, value))
    
    print("\n✅ Auto-Summarization Engine Ready")
    print("\nAvailable functions:")
    print("  - summarize_patient(patient_key)")
    print("  - summarize_site(site_id)")
    print("  - daily_digest(study_id, role)")
    print("  - executive_summary(study_id)")