
"""
SANCHALAK AI - Template Engine v1.2
Jinja2-based template system with multi-format support
Hardened for high-contrast production reporting.
"""

import os
import json
import logging
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib

from jinja2 import Environment, select_autoescape, BaseLoader, TemplateNotFound
from markupsafe import Markup

# Configure logging
logger = logging.getLogger(__name__)

class ReportType(Enum):
    """12 supported report types"""
    CRA_MONITORING = "cra_monitoring"
    SITE_PERFORMANCE = "site_performance"
    SPONSOR_UPDATE = "sponsor_update"
    MEETING_PACK = "meeting_pack"
    SAFETY_NARRATIVE = "safety_narrative"
    INSPECTION_PREP = "inspection_prep"
    QUERY_SUMMARY = "query_summary"
    SITE_NEWSLETTER = "site_newsletter"
    EXECUTIVE_BRIEF = "executive_brief"
    DB_LOCK_READINESS = "db_lock_readiness"
    ISSUE_ESCALATION = "issue_escalation"
    DAILY_DIGEST = "daily_digest"
    REGIONAL_SUMMARY = "regional_summary"
    CODING_STATUS = "coding_status"
    ENROLLMENT_TRACKER = "enrollment_tracker"
    PATIENT_RISK = "patient_risk"

class OutputFormat(Enum):
    """Supported output formats"""
    HTML = "html"
    PDF = "pdf"
    WORD = "docx"
    POWERPOINT = "pptx"
    MARKDOWN = "md"
    JSON = "json"
    TEXT = "txt"

@dataclass
class ReportMetadata:
    """Metadata for generated reports"""
    report_id: str
    report_type: ReportType
    title: str
    generated_at: datetime
    generated_by: str
    version: str = "1.0"
    classification: str = "Internal"
    
    def to_dict(self) -> Dict:
        return {
            'report_id': self.report_id,
            'report_type': self.report_type.value,
            'title': self.title,
            'generated_at': self.generated_at.isoformat(),
            'generated_by': self.generated_by,
            'version': self.version,
            'classification': self.classification
        }


@dataclass
class ReportTemplate:
    """Template definition"""
    template_id: str
    report_type: ReportType
    name: str
    description: str
    template_file: str
    required_variables: List[str]

@dataclass
class GeneratedReport:
    """Generated report output"""
    report_id: str
    metadata: ReportMetadata
    content: str
    format: OutputFormat
    file_path: Optional[str] = None
    generation_time_ms: int = 0
    
    @property
    def html_content(self) -> str:
        return self.content if self.format == OutputFormat.HTML else ""

class TemplateEngine:
    """Core engine for rendering reports"""
    
    def __init__(self):
        self.env = Environment(
            loader=BaseLoader(),
            autoescape=select_autoescape(['html', 'xml'])
        )
        self._register_filters(self.env)
        self.templates = self._initialize_templates()
        
    def _register_filters(self, env: Environment):
        env.filters['format_date'] = lambda d: d.strftime('%Y-%m-%d') if isinstance(d, datetime) else d
        env.filters['format_datetime'] = lambda d: d.strftime('%Y-%m-%d %H:%M:%S') if isinstance(d, datetime) else d
        env.filters['format_currency'] = lambda n: f"${n:,.2f}" if isinstance(n, (int, float)) else n
        env.filters['format_number'] = lambda n: f"{int(n):,}" if isinstance(n, (int, float)) else n
        env.filters['format_percent'] = lambda x: f"{x*100:.1f}%" if isinstance(x, (int, float)) else x
        env.filters['format_decimal'] = lambda x, p=2: f"{x:.{p}f}" if isinstance(x, (int, float)) else x
        env.filters['priority_badge'] = self._priority_badge
        env.filters['risk_color'] = self._risk_to_color
        env.filters['dqi_band'] = self._dqi_to_band
        env.filters['title_case'] = lambda s: str(s).replace('_', ' ').title()
        env.filters['trend_arrow'] = lambda t: "↑" if t > 0 else "↓" if t < 0 else "→"
        env.filters['ljust'] = lambda s, n: str(s).ljust(n)
        env.filters['rjust'] = lambda s, n: str(s).rjust(n)

    # =========================================================================
    # MACROS (Format Aware)
    # =========================================================================

    def _macro_horizontal_bar(self, value: float, max_val: float = 100, color: str = "#3b82f6", width: int = 120, output_format: Any = "html") -> str:
        """Renders a horizontal bar chart (SVG for HTML, ASCII for Text)."""
        fmt = str(output_format.value if hasattr(output_format, 'value') else output_format).lower()
        pct = min(100, max(0, (value / max_val) * 100)) if max_val > 0 else 0
        
        if fmt in ('txt', 'text'):
            bar_w = 20
            fill = int((pct/100)*bar_w)
            return f"[{'#'*fill}{'-'*(bar_w-fill)}] {value:.1f}"
            
        return Markup(f'<svg width="{width}" height="14" style="vertical-align:middle"><rect width="{width}" height="14" rx="3" fill="#e2e8f0" /><rect width="{(pct/100)*width}" height="14" rx="3" fill="{color}" /></svg> <span style="font-size:11px;margin-left:5px">{value:.1f}</span>')

    def _macro_donut_chart(self, value: float, total: float = 100, color: str = "#3b82f6", size: int = 40, output_format: Any = "html") -> str:
        """Renders a donut chart (SVG for HTML, Percentage for Text)."""
        fmt = str(output_format.value if hasattr(output_format, 'value') else output_format).lower()
        pct = min(100, max(0, (value / total) * 100)) if total > 0 else 0
        
        if fmt in ('txt', 'text'):
            return f"{int(pct)}%"
            
        r, c = size*0.4, 2*3.14159*size*0.4
        offset = c - (pct/100)*c
        return Markup(f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" style="vertical-align:middle"><circle cx="{size/2}" cy="{size/2}" r="{r}" fill="transparent" stroke="#e2e8f0" stroke-width="{size*0.15}" /><circle cx="{size/2}" cy="{size/2}" r="{r}" fill="transparent" stroke="{color}" stroke-width="{size*0.15}" stroke-dasharray="{c}" stroke-dashoffset="{offset}" stroke-linecap="round" transform="rotate(-90 {size/2} {size/2})" /><text x="50%" y="55%" dominant-baseline="middle" text-anchor="middle" font-size="{size*0.25}" font-weight="bold" fill="#1e293b">{int(pct)}%</text></svg>')

    def _macro_sparkline(self, data: List[float], color: str = "#3b82f6", width: int = 80, height: int = 25, output_format: Any = "html") -> str:
        """Renders a sparkline (SVG for HTML, Trend text for Text)."""
        fmt = str(output_format.value if hasattr(output_format, 'value') else output_format).lower()
        if not data or len(data) < 2: 
            return "No trend" if fmt in ('txt', 'text') else ""
            
        if fmt in ('txt', 'text'):
            return "UP" if data[-1]>data[0] else "DOWN"
            
        mi, ma = min(data), max(data)
        rng = (ma - mi) or 1
        pts = [f"{(i/(len(data)-1))*width},{height-((v-mi)/rng)*height}" for i,v in enumerate(data)]
        return Markup(f'<svg width="{width}" height="{height}" style="vertical-align:middle"><path d="M {" ".join(pts)}" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" /></svg>')

    def _macro_progress_bar(self, val: float, target: float = 1.0, color: str = "#3b82f6", output_format: Any = "html") -> str:
        """Renders a progress bar (HTML for HTML, ASCII for Text)."""
        fmt = str(output_format.value if hasattr(output_format, 'value') else output_format).lower()
        pct = min(100, max(0, (val/target)*100)) if target > 0 else 0
        
        if fmt in ('txt', 'text'):
            return f"{val*100:.1f}% / {target*100:.1f}%"
            
        return Markup(f'<div class="bar-container"><div class="bar-track"><div class="bar-fill" style="width:{pct}%;background:{color}"></div></div><div class="bar-label">{val*100:.1f}% / {target*100:.1f}%</div></div>')

    def _macro_rag_dot(self, status: str, output_format: Any = "html") -> str:
        """Renders a status indicator (Dot for HTML, Text for Text)."""
        fmt = str(output_format.value if hasattr(output_format, 'value') else output_format).lower()
        s = str(status).lower()
        
        if fmt in ('txt', 'text'):
            return "[!]" if s in ('red','danger','critical','blocked','at risk','below target') else "(v)" if s in ('green','success','on-track','ready') else "[*]"
            
        color = "#10b981" if s in ('green','success','on-track','ready') else "#f59e0b" if s in ('amber','warning','pending') else "#ef4444" if s in ('red','danger','critical','blocked','at risk') else "#94a3b8"
        return Markup(f'<span style="height:10px;width:10px;background-color:{color};border-radius:50%;display:inline-block;margin-right:5px"></span>')

    def _macro_trend_indicator(self, cur: float, prev: float, output_format: Any = "html") -> str:
        """Renders a trend indicator (Arrow for HTML, text for Text)."""
        fmt = str(output_format.value if hasattr(output_format, 'value') else output_format).lower()
        if prev == 0: return "-"
        d = ((cur - prev)/prev)*100
        a = "↑" if d > 0 else "↓" if d < 0 else "→"
        
        if fmt in ('txt', 'text'):
            return f"{a} {abs(d):.1f}%"
            
        c = "#10b981" if d > 0 else "#ef4444" if d < 0 else "#94a3b8"
        return Markup(f'<span style="color:{c};font-weight:bold">{a} {abs(d):.1f}%</span>')

    def _priority_badge(self, p: str) -> str:
        c = {'critical':'#fee2e2;#991b1b', 'high':'#ffedd5;#9a3412', 'medium':'#fef9c3;#854d0e', 'low':'#dcfce7;#166534'}.get(str(p).lower(), '#f1f5f9;#475569')
        bg, txt = c.split(';')
        return Markup(f'<span style="padding:2px 8px;border-radius:4px;font-size:11px;font-weight:700;text-transform:uppercase;background:{bg};color:{txt};border:1px solid rgba(0,0,0,0.1)">{p}</span>')

    def _risk_to_color(self, r: str) -> str:
        return {'low':'#10b981', 'medium':'#f59e0b', 'high':'#f97316', 'critical':'#ef4444'}.get(str(r).lower(), '#64748b')

    def _dqi_to_band(self, s: float) -> str:
        return "Pristine" if s >= 90 else "Excellent" if s >= 80 else "Good" if s >= 70 else "Critical"

    def _initialize_templates(self) -> Dict[str, ReportTemplate]:
        templates = [
            ReportTemplate("cra_monitoring", ReportType.CRA_MONITORING, "CRA Monitoring Report", "Summary", "cra.html", []),
            ReportTemplate("site_performance", ReportType.SITE_PERFORMANCE, "Site Performance Summary", "Metrics", "site.html", []),
            ReportTemplate("executive_brief", ReportType.EXECUTIVE_BRIEF, "Executive Brief", "Overview", "exec.html", []),
            ReportTemplate("db_lock_readiness", ReportType.DB_LOCK_READINESS, "DB Lock Readiness", "Assessment", "lock.html", []),
            ReportTemplate("query_summary", ReportType.QUERY_SUMMARY, "Query Summary", "Status", "query.html", []),
            ReportTemplate("sponsor_update", ReportType.SPONSOR_UPDATE, "Sponsor Update", "Monthly", "sponsor.html", []),
            ReportTemplate("meeting_pack", ReportType.MEETING_PACK, "Meeting Pack", "Slides", "meeting.html", []),
            ReportTemplate("safety_narrative", ReportType.SAFETY_NARRATIVE, "Safety Narrative", "SAE", "safety.html", []),
            ReportTemplate("regional_summary", ReportType.REGIONAL_SUMMARY, "Regional Summary", "Global", "region.html", []),
            ReportTemplate("coding_status", ReportType.CODING_STATUS, "Coding Status", "Medical", "coding.html", []),
            ReportTemplate("enrollment_tracker", ReportType.ENROLLMENT_TRACKER, "Enrollment Tracker", "Recruitment", "enroll.html", []),
            ReportTemplate("patient_risk", ReportType.PATIENT_RISK, "Patient Risk Analysis", "Risk", "risk.html", [])
        ]
        return {t.template_id: t for t in templates}

    def render(self, template_id: str, variables: Dict[str, Any], output_format: OutputFormat = OutputFormat.HTML, generated_by: str = "System") -> GeneratedReport:
        template_info = self.templates.get(template_id)
        if not template_info: raise ValueError(f"Template {template_id} not found")
            
        metadata = ReportMetadata(
            report_id=f"RPT-{template_id.upper()}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            report_type=template_info.report_type,
            title=template_info.name,
            generated_at=datetime.now(),
            generated_by=generated_by
        )
        
        render_vars = {**variables, '_metadata': metadata.to_dict(), 'now': datetime.now, 'common_css': self._get_common_css(), 'output_format': output_format}
        
        self.env.globals.update({
            'horizontal_bar': lambda *a, **k: self._macro_horizontal_bar(*a, **k, output_format=output_format),
            'donut_chart': lambda *a, **k: self._macro_donut_chart(*a, **k, output_format=output_format),
            'sparkline': lambda *a, **k: self._macro_sparkline(*a, **k, output_format=output_format),
            'progress_bar': lambda *a, **k: self._macro_progress_bar(*a, **k, output_format=output_format),
            'rag_dot': lambda *a, **k: self._macro_rag_dot(*a, **k, output_format=output_format),
            'trend_indicator': lambda *a, **k: self._macro_trend_indicator(*a, **k, output_format=output_format)
        })

        t_str = self._get_inline_text_template(template_id) if output_format == OutputFormat.TEXT else self._get_inline_template(template_id)
        
        try:
            # Inject mapping fix for sdv_rate and clean_rate
            if 'metrics' in variables:
                m = variables['metrics']
                if isinstance(m, dict):
                    if 'sdv_rate' not in m: m['sdv_rate'] = m.get('sdv_completion_rate', 100.0)
                    if 'clean_rate' not in m: m['clean_rate'] = m.get('clean_crf_pct', 0.0)
                elif hasattr(m, 'sdv_completion_rate'):
                    # It's a DotDict or object
                    if not hasattr(m, 'sdv_rate'): setattr(m, 'sdv_rate', getattr(m, 'sdv_completion_rate', 100.0))
                    if not hasattr(m, 'clean_rate'): setattr(m, 'clean_rate', getattr(m, 'clean_crf_pct', 0.0))
            
            content = self.env.from_string(t_str).render(**render_vars)
            return GeneratedReport(report_id=metadata.report_id, metadata=metadata, content=content, format=output_format)
        except Exception as e:
            logger.error(f"Render failed: {e}")
            raise

    def _get_common_css(self) -> str:
        return """
        :root { --primary: #1e293b; --secondary: #334155; --accent: #3b82f6; --success: #10b981; --warning: #f59e0b; --danger: #ef4444; --bg: #f8fafc; }
        body { font-family: 'Inter', -apple-system, sans-serif !important; margin: 0 !important; padding: 0 !important; background: var(--bg) !important; color: #1e293b !important; line-height: 1.6 !important; }
        @media print { body { background: white !important; } .container { box-shadow: none !important; border: none !important; width: 100% !important; max-width: none !important; margin: 0 !important; } @page { size: A4; margin: 15mm; } }
        .classification-banner { background: #0f172a !important; color: white !important; text-align: center !important; padding: 4px !important; font-size: 10px !important; font-weight: bold !important; text-transform: uppercase !important; }
        .container { max-width: 1100px !important; margin: 20px auto !important; background: white !important; border-radius: 12px !important; box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important; overflow: hidden !important; border: 1px solid #e2e8f0 !important; text-align: left !important; color: #1e293b !important; }
        .header { background: linear-gradient(135deg, #1e293b 0%, #334155 100%) !important; color: white !important; padding: 40px !important; }
        .header h1 { margin: 0 !important; font-size: 32px !important; font-weight: 800 !important; color: white !important; display: flex !important; align-items: center !important; gap: 14px !important; }
        .header h1::before { content: '' !important; display: inline-block !important; width: 40px !important; height: 40px !important; min-width: 40px !important; background-image: url(data:image/png;base64,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) !important; background-size: contain !important; background-repeat: no-repeat !important; }
        .header .meta { margin-top: 15px !important; display: flex !important; gap: 20px !important; font-size: 14px !important; opacity: 0.9 !important; color: white !important; }
        .header .meta span { color: white !important; }
        .content { padding: 40px !important; background: white !important; color: #1e293b !important; }
        .section { margin-bottom: 40px !important; }
        .section-header { border-bottom: 2px solid #f1f5f9 !important; padding-bottom: 12px !important; margin-bottom: 25px !important; }
        .section h2 { margin: 0 !important; font-size: 20px !important; font-weight: 700 !important; color: #0f172a !important; text-transform: uppercase !important; }
        .kpi-grid { display: grid !important; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)) !important; gap: 20px !important; margin-bottom: 30px !important; }
        .card { background: white !important; border: 1px solid #e2e8f0 !important; padding: 25px !important; border-radius: 12px !important; color: #1e293b !important; }
        .card-value { font-size: 36px !important; font-weight: 800 !important; color: #0f172a !important; margin-bottom: 5px !important; }
        .card-label { font-size: 12px !important; color: #64748b !important; font-weight: 600 !important; text-transform: uppercase !important; }
        .card-sub { font-size: 11px !important; color: #94a3b8 !important; }
        table { width: 100% !important; border-collapse: collapse !important; margin: 10px 0 !important; }
        th { background: #f8fafc !important; padding: 12px 15px !important; text-align: left !important; font-size: 12px !important; color: #475569 !important; text-transform: uppercase !important; border-bottom: 2px solid #e2e8f0 !important; }
        td { padding: 12px 15px !important; border-bottom: 1px solid #f1f5f9 !important; font-size: 14px !important; color: #1e293b !important; }
        .ai-narrative { background: #f0f9ff !important; border: 1px solid #bae6fd !important; padding: 25px !important; border-radius: 12px !important; margin-bottom: 30px !important; position: relative !important; color: #0369a1 !important; }
        .ai-narrative h3 { margin-top: 0 !important; color: #0369a1 !important; font-size: 16px !important; }
        .ai-narrative p { color: #0369a1 !important; }
        .findings-box { display: grid !important; grid-template-columns: 1fr 1fr !important; gap: 30px !important; }
        .findings-panel { background: #fdf2f2 !important; border: 1px solid #fecaca !important; padding: 20px !important; border-radius: 8px !important; color: #991b1b !important; }
        .findings-panel h3 { color: #991b1b !important; margin-top: 0 !important; font-size: 16px !important; }
        .findings-panel ul { color: #991b1b !important; }
        .findings-panel li { color: #991b1b !important; }
        .success-panel { background: #f0fdf4 !important; border: 1px solid #bbf7d0 !important; padding: 20px !important; border-radius: 8px !important; color: #166534 !important; }
        .success-panel h3 { color: #166534 !important; margin-top: 0 !important; font-size: 16px !important; }
        .success-panel ul { color: #166534 !important; }
        .success-panel li { color: #166534 !important; }
        .bar-container { width: 100% !important; margin: 5px 0 !important; }
        .bar-track { height: 8px !important; background: #e2e8f0 !important; border-radius: 4px !important; overflow: hidden !important; }
        .bar-fill { height: 100% !important; border-radius: 4px !important; }
        .bar-label { font-size: 10px !important; color: #64748b !important; margin-top: 4px !important; text-align: right !important; }
        .footer { background: #f8fafc !important; padding: 30px 40px !important; border-top: 1px solid #e2e8f0 !important; color: #94a3b8 !important; font-size: 11px !important; display: flex !important; justify-content: space-between !important; }
        """

    def _get_inline_template(self, template_id: str) -> str:
        # Logo embedded as base64 (48x48 Sanchalak AI logo)
        _LOGO_B64 = "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"

        # Standard Shell Wrapper with logo in header
        shell_start = """<html><head><meta charset="UTF-8"><style>{{ common_css }}</style></head><body><div class="classification-banner">CONFIDENTIAL - FOR INTERNAL USE ONLY</div><div class="container">"""
        shell_end = """<div class="footer"><div style="display:flex;align-items:center;gap:8px;"><img src="data:image/png;base64,""" + _LOGO_B64 + """" style="width:20px;height:20px;" />Sanchalak AI | Generated by {{ _metadata.generated_by }}</div><div>Version {{ _metadata.version }} | Page 1 of 1</div></div></div></body></html>"""
        
        templates = {
            "cra_monitoring": shell_start + """
                <div class="header"><h1>CRA Monitoring Report</h1><div class="meta"><span><strong>SITE:</strong> {{ site_id }}</span><span><strong>CRA:</strong> {{ cra_name }}</span><span><strong>DATE:</strong> {{ visit_date | format_date }}</span></div></div>
                <div class="content">
                    <!-- AI_EXEC_SUMMARY -->
                    <div class="kpi-grid">
                        <div class="card"><div class="card-label">Total Patients</div><div class="card-value">{{ site_data.total_patients | format_number }}</div><div class="card-sub">{{ rag_dot('on-track') }} On Track</div></div>
                        <div class="card"><div class="card-label">DQI Score</div><div class="card-value">{{ site_data.dqi_score | format_decimal(1) }}</div><div class="card-sub">{{ rag_dot('on-track' if site_data.dqi_score >= 90 else 'warning') }} {{ site_data.dqi_score | dqi_band }}</div></div>
                        <div class="card"><div class="card-label">Clean Rate</div><div class="card-value">{{ site_data.clean_rate | format_percent }}</div><div class="card-sub">{{ progress_bar(site_data.clean_rate, 0.7, '#10b981') }}</div></div>
                        <div class="card"><div class="card-label">Open Queries</div><div class="card-value">{{ site_data.open_queries | format_number }}</div><div class="card-sub">{{ rag_dot('danger' if site_data.open_queries > 50 else 'success') }} Action Required</div></div>
                    </div>
                    <div class="section"><div class="section-header"><h2>Detailed Quality Metrics</h2></div><table><thead><tr><th>Metric Category</th><th>Performance</th><th>Target</th><th>Status</th></tr></thead><tbody>
                        {% for m in site_data.metrics %}<tr><td><strong>{{ m.name }}</strong></td><td>{{ horizontal_bar(m.value, 100, '#3b82f6') }}</td><td>{{ m.target | format_decimal(1) }}%</td><td>{% if m.value >= m.target %}<span style="color:var(--success);font-weight:bold">PASS</span>{% else %}<span style="color:var(--danger);font-weight:bold">FAIL</span>{% endif %}</td></tr>{% endfor %}
                    </tbody></table></div>
                    <!-- AI_FINDINGS -->
                    <!-- AI_RECOMMENDATIONS -->
                    {% if (findings or recommendations) and not ai_recommendations_injected %}
                    <div class="section" id="standard-observations">
                        <div class="section-header"><h2>Site Observations</h2></div>
                        <div class="findings-box">
                            {% if findings %}
                            <div class="findings-panel"><h3>Critical Issues</h3><ul>{% for f in findings %}<li>{{ f }}</li>{% endfor %}</ul></div>
                            {% endif %}
                            {% if recommendations %}
                            <div class="success-panel"><h3>Recommendations</h3><ul>{% for r in recommendations %}<li><strong>{{ r.title }}:</strong> {{ r.description }}</li>{% endfor %}</ul></div>
                            {% endif %}
                        </div>
                    </div>
                    {% endif %}
                </div>""" + shell_end,
            "site_performance": shell_start + """
                <div class="header" style="background:linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%)"><h1>Site Performance Summary</h1><div class="meta"><span><strong>STUDY:</strong> {{ study_id or 'All' }}</span><span><strong>SITE:</strong> {{ site_id or 'Global' }}</span></div></div>
                <div class="content"><!-- AI_EXEC_SUMMARY -->
                    <div class="kpi-grid">
                        <div class="card"><div class="card-label">Avg DQI Score</div><div class="card-value">{{ metrics.dqi_score | format_decimal(1) }}</div></div>
                        <div class="card"><div class="card-label">Avg Clean Rate</div><div class="card-value">{{ metrics.clean_rate | format_percent }}</div></div>
                        <div class="card"><div class="card-label">SDV Completion</div><div class="card-value">{{ metrics.sdv_rate | format_percent }}</div><div>{{ donut_chart(metrics.sdv_rate, 100) if metrics.sdv_rate > 1 else donut_chart(metrics.sdv_rate * 100, 100) }}</div></div>
                        <div class="card"><div class="card-label">Open Issues</div><div class="card-value">{{ metrics.total_issues | format_number }}</div></div>
                    </div>
                    <div class="section"><div class="section-header"><h2>Performance Trends</h2></div><!-- AI_FINDINGS --><table><thead><tr><th>Metric</th><th>Current</th><th>Previous</th><th>Trend</th><th>Analysis</th></tr></thead><tbody>
                        {% for t in trends %}<tr><td><strong>{{ t.name }}</strong></td><td>{{ t.current | format_decimal(1) }}</td><td>{{ t.previous | format_decimal(1) }}</td><td>{{ trend_indicator(t.current, t.previous) }}</td><td>{{ sparkline([t.previous, t.current*0.9, t.current], '#3b82f6') }}</td></tr>{% endfor %}
                    </tbody></table></div>
                    <div class="section"><div class="section-header"><h2>Action Plans</h2></div><table><thead><tr><th>Site ID</th><th>DQI Score</th><th>Clean Rate</th><th>Status</th></tr></thead><tbody>
                        {% for s in bottom_performers[:5] %}<tr><td><strong>{{ s.site_id }}</strong></td><td>{{ horizontal_bar(s.dqi_score, 100, '#ef4444') }}</td><td>{{ s.clean_rate | format_percent }}</td><td><span style="color:var(--danger)">At Risk</span></td></tr>{% endfor %}
                    </tbody></table><!-- AI_RECOMMENDATIONS --></div>
                </div>""" + shell_end,
            "executive_brief": shell_start + """
                <div class="header" style="background:#0f172a"><h1>Executive Brief</h1><div class="meta"><span><strong>STUDY:</strong> {{ study_id }}</span><span><strong>DATE:</strong> {{ report_date | format_date }}</span></div></div>
                <div class="content"><!-- AI_EXEC_SUMMARY -->
                    <div class="kpi-grid">
                        <div class="card" style="border-top:4px solid var(--accent)"><div class="card-label">Study DQI</div><div class="card-value">{{ key_metrics.dqi | format_decimal(1) }}</div></div>
                        <div class="card" style="border-top:4px solid var(--success)"><div class="card-label">Clean Rate</div><div class="card-value">{{ key_metrics.clean_rate | format_percent }}</div></div>
                        <div class="card" style="border-top:4px solid var(--warning)"><div class="card-label">DB Lock Ready</div><div class="card-value">{{ key_metrics.dblock_ready | format_percent }}</div></div>
                        <div class="card" style="border-top:4px solid var(--danger)"><div class="card-label">Issue Density</div><div class="card-value">{{ key_metrics.issue_rate | format_decimal(2) }}</div></div>
                    </div>
                    <div class="findings-box">
                        {% if highlights %}
                        <div class="findings-panel"><h3>Highlights</h3><!-- AI_FINDINGS --><ul>{% for h in highlights %}<li>{{ h }}</li>{% endfor %}</ul></div>
                        {% endif %}
                        {% if concerns %}
                        <div class="findings-panel"><h3>Concerns</h3><ul>{% for c in concerns %}<li>{{ c }}</li>{% endfor %}</ul></div>
                        {% endif %}
                    </div>
                    <!-- AI_RECOMMENDATIONS -->
                    {% if next_actions and not ai_recommendations_injected %}
                    <div class="section" id="standard-actions"><div class="section-header"><h2>Action Tracker</h2></div><table><thead><tr><th>Action</th><th>Owner</th><th>Priority</th></tr></thead><tbody>
                        {% for a in next_actions %}<tr><td><strong>{{ a.action }}</strong></td><td>{{ a.owner }}</td><td>{{ a.priority | priority_badge }}</td></tr>{% endfor %}
                    </tbody></table></div>
                    {% endif %}
                </div>""" + shell_end,
            "db_lock_readiness": shell_start + """
                <div class="header" style="background:#0f172a;border-bottom:6px solid var(--success)"><h1>DB Lock Readiness</h1><div class="meta"><span><strong>STUDY:</strong> {{ study_id }}</span><span><strong>TARGET:</strong> {{ target_date | format_date }}</span></div></div>
                <div class="content"><!-- AI_EXEC_SUMMARY -->
                    <div style="display:grid;grid-template-columns:300px 1fr;gap:40px;margin-bottom:40px">
                        <div class="card" style="background:#f0fdf4;border:2px solid #bbf7d0;text-align:center"><div class="card-label">Overall Readiness</div><div class="card-value" style="color:#15803d;font-size:64px">{{ readiness_data.ready_rate | format_percent }}</div></div>
                        <div><h3>Component Readiness</h3><!-- AI_FINDINGS -->{% for cat in readiness_data.categories %}<div style="margin-bottom:15px"><div><strong>{{ cat.name }}</strong>: {{ cat.rate | format_percent }}</div>{{ horizontal_bar(cat.rate*100, 100) }}</div>{% endfor %}</div>
                    </div>
                    <div class="section"><div class="section-header"><h2>Site Breakdown</h2></div>
                        {% if readiness_data.sites %}
                        <table><thead><tr><th>Site</th><th>Total</th><th>Ready</th><th>Status</th></tr></thead><tbody>
                            {% for s in readiness_data.sites[:10] %}<tr><td><strong>{{ s.site_id }}</strong></td><td>{{ s.patients }}</td><td>{{ s.ready }}</td><td><span class="badge">{{ 'READY' if (s.ready/s.patients) >= 0.9 else 'PENDING' }}</span></td></tr>{% endfor %}
                        </tbody></table>
                        {% endif %}
                        <!-- AI_RECOMMENDATIONS -->
                    </div>
                </div>""" + shell_end,
            "query_summary": shell_start + """
                <div class="header" style="background:linear-gradient(135deg, #ca8a04 0%, #a16207 100%)"><h1>Query Resolution Summary</h1><div class="meta"><span><strong>ENTITY:</strong> {{ entity_id }}</span></div></div>
                <div class="content"><!-- AI_EXEC_SUMMARY -->
                    <div class="kpi-grid">
                        <div class="card"><div class="card-label">Total</div><div class="card-value">{{ query_data.total }}</div></div>
                        <div class="card"><div class="card-label">Open</div><div class="card-value" style="color:var(--danger)">{{ query_data.open }}</div></div>
                        <div class="card"><div class="card-label">Resolved</div><div class="card-value" style="color:var(--success)">{{ query_data.resolved }}</div></div>
                        <div class="card"><div class="card-label">Avg Days</div><div class="card-value">{{ query_data.avg_days }}</div></div>
                    </div>
                    <div class="section"><div class="section-header"><h2>Top Categories</h2></div><!-- AI_FINDINGS --><table><thead><tr><th>Category</th><th>Count</th></tr></thead><tbody>
                        {% for t in top_issues %}<tr><td><strong>{{ t.category }}</strong></td><td>{{ t.count }}</td></tr>{% endfor %}
                    </tbody></table></div>
                </div>""" + shell_end,
            "sponsor_update": shell_start + """
                <div class="header"><h1>Sponsor Status Update</h1><div class="meta"><span><strong>STUDY:</strong> {{ study_id }}</span></div></div>
                <div class="content"><!-- AI_EXEC_SUMMARY -->
                    <div class="kpi-grid">
                        <div class="card"><div class="card-label">Enrolled</div><div class="card-value">{{ study_metrics.patients }}</div></div>
                        <div class="card"><div class="card-label">Sites</div><div class="card-value">{{ study_metrics.sites }}</div></div>
                        <div class="card"><div class="card-label">DQI</div><div class="card-value">{{ study_metrics.dqi | format_decimal(1) }}</div></div>
                        <div class="card"><div class="card-label">Readiness</div><div class="card-value">{{ study_metrics.dblock_ready | format_percent }}</div></div>
                    </div>
                    <div class="findings-box">
                        <div class="success-panel"><h3>Highlights</h3><!-- AI_FINDINGS --><ul>{% for h in highlights %}<li>{{ h }}</li>{% endfor %}</ul></div>
                        <div class="findings-panel"><h3>Risks</h3><ul>{% for r in risks %}<li>{{ r.description }}</li>{% endfor %}</ul></div>
                    </div>
                </div>""" + shell_end,
            "meeting_pack": shell_start + """
                <div class="header" style="background:#0f172a"><h1>{{ meeting_type | upper }}</h1><div class="meta"><span><strong>DATE:</strong> {{ meeting_date | format_date }}</span></div></div>
                <div class="content">
                    <div class="section"><h2>Agenda</h2><ul>{% for e in agenda %}<li><strong>{{ e.item }}</strong> ({{ e.duration }})</li>{% endfor %}</ul></div>
                    <div class="kpi-grid">
                        <div class="card"><div class="card-label">Patients</div><div class="card-value">{{ study_data.patients }}</div></div>
                        <div class="card"><div class="card-label">Clean Status</div><div class="card-value">{{ study_data.clean_rate | format_percent }}</div></div>
                    </div>
                    <div class="section"><h2>Action Items</h2><!-- AI_RECOMMENDATIONS --><table><thead><tr><th>Action</th><th>Owner</th><th>Due</th></tr></thead><tbody>
                        {% for a in action_items %}<tr><td><strong>{{ a.action }}</strong></td><td>{{ a.owner }}</td><td>{{ a.due }}</td></tr>{% endfor %}
                    </tbody></table></div>
                </div>""" + shell_end,
            "safety_narrative": shell_start + """
                <div class="header" style="background:white;color:black;border-bottom:3px solid black"><h1>Clinical Narrative Summary</h1><div class="meta" style="color:#666"><span><strong>SUBJECT:</strong> {{ patient_id }}</span><span><strong>SAE:</strong> {{ sae_id }}</span></div></div>
                <div class="content">
                    <div class="section"><h2>Patient Info</h2><p>Age: {{ event_details.age }} | Gender: {{ event_details.gender }}</p></div>
                    <div class="section"><h2>Narrative</h2><div style="padding:20px;border:1px solid #ddd;font-family:serif;background:#fff">{% for p in narrative_summary %}<p>{{ p }}</p>{% endfor %}</div></div>
                </div>""" + shell_end,
            "regional_summary": shell_start + """
                <div class="header" style="background:linear-gradient(135deg, #2563eb 0%, #1e40af 100%)"><h1>Regional Performance Summary</h1></div>
                <div class="content">
                    <div class="kpi-grid">{% for r in regions[:3] %}<div class="card"><div class="card-label">{{ r.region }}</div><div class="card-value">{{ r.avg_dqi | format_decimal(1) }}</div></div>{% endfor %}</div>
                    <table><thead><tr><th>Region</th><th>Avg DQI</th><th>Sites</th></tr></thead><tbody>
                        {% for r in regions %}<tr><td><strong>{{ r.region }}</strong></td><td>{{ horizontal_bar(r.avg_dqi) }}</td><td>{{ r.sites }}</td></tr>{% endfor %}
                    </tbody></table>
                </div>""" + shell_end,
            "coding_status": shell_start + """
                <div class="header" style="background:linear-gradient(135deg, #7c3aed 0%, #5b21b6 100%)"><h1>Medical Coding Status</h1></div>
                <div class="content">
                    <div style="display:flex;gap:40px">
                        <div class="card"><h3>MedDRA</h3>{{ donut_chart(coding_data.meddra.completion, 100) }}</div>
                        <div class="card"><h3>WHODrug</h3>{{ donut_chart(coding_data.whodrug.completion, 100) }}</div>
                    </div>
                </div>""" + shell_end,
            "enrollment_tracker": shell_start + """
                <div class="header" style="background:linear-gradient(135deg, #059669 0%, #065f46 100%)"><h1>Enrollment Tracker</h1></div>
                <div class="content">
                    <div class="card" style="text-align:center"><div class="card-label">Overall Progress</div><div class="card-value" style="color:var(--success)">{{ enrollment_data.overall_pct | format_decimal(1) }}%</div>{{ horizontal_bar(enrollment_data.overall_pct, 100) }}</div>
                    <div class="section"><h2>Site Enrollment</h2><table><thead><tr><th>Site ID</th><th>Enrolled</th><th>Avg DQI</th></tr></thead><tbody>
                        {% for s in enrollment_data.sites %}<tr><td><strong>{{ s.site_id }}</strong></td><td>{{ s.enrolled }}</td><td>{{ s.avg_dqi | format_decimal(1) }}</td></tr>{% endfor %}
                    </tbody></table></div>
                </div>""" + shell_end,
            "patient_risk": shell_start + """
                <div class="header" style="background:linear-gradient(135deg, #dc2626 0%, #991b1b 100%)"><h1>Patient Risk Analysis</h1></div>
                <div class="content">
                    <div class="kpi-grid">
                        <div class="card"><div class="card-label">Critical Risk</div><div class="card-value" style="color:var(--danger)">{{ patient_data.critical_count }}</div></div>
                        <div class="card"><div class="card-label">Avg DQI</div><div class="card-value">{{ patient_data.avg_dqi | format_decimal(1) }}</div></div>
                    </div>
                    <table><thead><tr><th>Risk Factor</th><th>Affected</th></tr></thead><tbody>
                        {% for f in patient_data.risk_factors %}<tr><td><strong>{{ f.factor }}</strong></td><td>{{ f.count }}</td></tr>{% endfor %}
                    </tbody></table>
                </div>""" + shell_end
        }
        return templates.get(template_id, shell_start + f"<h1>{template_id.replace('_',' ').title()}</h1><div class='content'><!-- AI_EXEC_SUMMARY --><p>Template content for {template_id} goes here.</p><!-- AI_FINDINGS --><!-- AI_RECOMMENDATIONS --></div>" + shell_end)

    def _get_inline_text_template(self, template_id: str) -> str:
        h = "="*80 + "\n SANCHALAK AI - {{ _metadata.title | upper }}\n" + "="*80 + "\nID: {{ _metadata.report_id }} | Generated: {{ _metadata.generated_at }}\n" + "-"*80 + "\n"
        f = "\n" + "-"*80 + "\n(c) 2026 Sanchalak AI | Version 1.1 | CONFIDENTIAL\n" + "="*80
        
        t = {
            "cra_monitoring": h + """
1. EXECUTIVE SUMMARY
[AI_EXEC_SUMMARY]
Data quality at Site {{ site_id }} remains stable. Primary focus: query resolution and SDV.

2. KEY PERFORMANCE INDICATORS
Metric                  Value       Status
--------------------------------------------------------------------------------
Total Patients          {{ site_data.total_patients | format_number }}         [OK]
DQI Score               {{ site_data.dqi_score | format_decimal(1) }}        [{{ site_data.dqi_score | dqi_band | upper }}]
Clean Rate              {{ site_data.clean_rate | format_percent }}       [LAG]
Open Queries            {{ site_data.open_queries | format_number }}         [OK]

3. DETAILED QUALITY METRICS
{% for m in site_data.metrics -%}
{{ m.name | ljust(24) }}{{ horizontal_bar(m.value) }}  Target: {{ m.target }}%
{% endfor %}

4. SITE OBSERVATIONS
[AI_FINDINGS]
Critical Issues:
{% for f in findings %} - {{ f }} {% endfor %}

[AI_RECOMMENDATIONS]
Recommendations:
{% for r in recommendations %} - {{ r.title }}: {{ r.description }} {% endfor %}
""" + f
        }
        return t.get(template_id, h + "1. OVERVIEW\nReport data for {{ study_id or 'Portfolio' }}.\n[AI_EXEC_SUMMARY]\n[AI_FINDINGS]\n[AI_RECOMMENDATIONS]\n" + f)

def get_template_engine() -> TemplateEngine:
    return TemplateEngine()
