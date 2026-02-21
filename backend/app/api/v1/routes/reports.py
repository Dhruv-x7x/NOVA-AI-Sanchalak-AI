"""
Report Routes
=============
Report generation endpoints for all 12 report types.
Integrated with src.generation.report_generators for production-grade output.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Response
from fastapi.responses import FileResponse, StreamingResponse
from typing import Optional, Dict, Any, List
from datetime import datetime
import io
import sys
import os
import pandas as pd
import logging
from pathlib import Path
import inspect

logger = logging.getLogger(__name__)

# Add project root to path for importing report generators
current_path = Path(__file__).resolve()
for parent in current_path.parents:
    if (parent / "src").exists():
        PROJECT_ROOT = parent
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        break
else:
    PROJECT_ROOT = current_path.parents[5]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from app.models.schemas import ReportRequest, ReportResponse
from app.core.security import get_current_user
from app.services.database import get_data_service

import json
from datetime import datetime, date

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle datetime and date objects."""
    def default(self, o):
        if isinstance(o, (datetime, date)):
            return o.isoformat()
        return super().default(o)

router = APIRouter()

@router.get("/types")
@router.get("/list")
@router.get("")
@router.get("/")
async def list_reports(
    current_user: dict = Depends(get_current_user)
):
    """List available reports for the user."""
    try:
        from src.generation.report_generators import ReportGeneratorFactory
        types = ReportGeneratorFactory.list_report_types()
        
        metadata = {
            'cra_monitoring': {"name": "CRA Monitoring Report", "description": "Site visit and monitoring summary", "category": "General", "icon": "user"},
            'site_performance': {"name": "Site Performance Report", "description": "Comprehensive site metrics", "category": "General", "icon": "building"},
            'executive_brief': {"name": "Executive Brief", "description": "High-level portfolio overview", "category": "Executive", "icon": "activity"},
            'db_lock_readiness': {"name": "DB Lock Readiness", "description": "Database lock preparation status", "category": "Operations", "icon": "shield"},
            'query_summary': {"name": "Query Summary", "description": "Data query status and trends", "category": "Data Management", "icon": "activity"},
            'coding_status': {"name": "Coding Status", "description": "MedDRA/WHODrug coding status", "category": "Data Management", "icon": "file-text"},
            'safety_narrative': {"name": "Safety Narrative", "description": "Safety event narratives", "category": "Safety", "icon": "shield"},
            'patient_risk': {"name": "Patient Risk Analysis", "description": "Individual patient risk assessment", "category": "Safety", "icon": "alert-triangle"},
            'regional_summary': {"name": "Regional Summary", "description": "Regional performance breakdown", "category": "Operations", "icon": "activity"},
            'enrollment_tracker': {"name": "Enrollment Tracker", "description": "Recruitment and retention metrics", "category": "Operations", "icon": "users"}
        }
        
        report_list = []
        for r_type in types:
            info = metadata.get(r_type, {"name": r_type.replace('_', ' ').title(), "description": "Generated report", "category": "General", "icon": "file-text"})
            report_list.append({"id": r_type, **info})
            
        return {"report_types": report_list}
    except Exception as e:
        logger.error(f"Failed to list reports: {e}")
        return {"report_types": []}

@router.get("/generate/{report_type}")
async def generate_report_get(
    report_type: str,
    site_id: Optional[str] = None,
    study_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Generate a report via GET request (legacy/frontend support)."""
    request = ReportRequest(
        report_type=report_type,
        site_id=site_id,
        study_id=study_id
    )
    return await generate_report_api(request, current_user)

@router.post("/generate")
@router.post("")
@router.post("/")
async def generate_report_api(
    request: ReportRequest,
    current_user: dict = Depends(get_current_user)
):
    """Generate a report of the specified type."""
    try:
        from src.generation.report_generators import ReportGeneratorFactory, OutputFormat
        
        # Normalize report type
        report_type = request.report_type.lower().replace(' ', '_').replace('-', '_')
        if report_type == 'clinical_summary':
            report_type = 'executive_brief'
            
        available_types = ReportGeneratorFactory.list_report_types()
        if report_type not in available_types:
            for t in available_types:
                if report_type in t or t in report_type:
                    report_type = t
                    break
            else:
                report_type = "cra_monitoring"

        generator = ReportGeneratorFactory.get_generator(report_type)
        sig = inspect.signature(generator.generate)
        
        # Prepare all possible params
        target_fmt = OutputFormat.HTML
        if request.format == "pdf": target_fmt = OutputFormat.PDF
        elif request.format in ("txt", "text"): target_fmt = OutputFormat.TEXT
        elif request.format in ("xlsx", "excel"): target_fmt = OutputFormat.XLSX
        
        all_params = {
            "study_id": request.study_id,
            "site_id": request.site_id,
            "sites": [request.site_id] if request.site_id else None,
            "date_range_days": request.date_range_days,
            "cra_name": current_user.get("full_name", current_user.get("username", "System User")),
            "output_formats": [target_fmt]
        }
        
        # Filter params based on what the generator actually accepts
        params = {k: v for k, v in all_params.items() if k in sig.parameters}
        
        # Ensure we have a site if the generator needs it
        data_service = get_data_service()
        if ('site_id' in sig.parameters and not params.get('site_id')) or \
           ('sites' in sig.parameters and not params.get('sites')):
            sites_df = data_service.get_sites()
            if not sites_df.empty:
                first_site = sites_df.iloc[0]["site_id"]
                if 'site_id' in sig.parameters: params['site_id'] = first_site
                if 'sites' in sig.parameters: params['sites'] = [first_site]
        
        results = generator.generate(**params)
        if not results:
            raise HTTPException(status_code=500, detail="No output from generator")
            
        result = results[0]
        
        # Determine media type and content handling
        media_type = "text/html"
        is_binary = False
        
        if target_fmt == OutputFormat.HTML:
            report_content = result.html_content
        elif target_fmt == OutputFormat.XLSX:
            report_content = result.content # Binary bytes
            media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            is_binary = True
        elif target_fmt == OutputFormat.PDF:
            report_content = result.content # Binary bytes
            media_type = "application/pdf"
            is_binary = True
        else:
            # Fallback for other formats
            if isinstance(result.content, bytes):
                try:
                    report_content = result.content.decode('utf-8')
                except UnicodeDecodeError:
                    report_content = result.content
                    is_binary = True
            else:
                report_content = result.content

        # If it's a binary format like Excel or PDF, enrich with AI then return directly
        if is_binary:
            # For XLSX, regenerate with AI Analysis sheet
            if target_fmt == OutputFormat.XLSX:
                try:
                    gen_vars = result.variables if hasattr(result, 'variables') else {}
                    km = gen_vars.get('key_metrics', {})
                    ai_context = {
                        "entity_id": request.site_id or request.study_id or "Portfolio",
                        "study_id": request.study_id or "Global",
                        "report_type": report_type,
                        "metrics": {
                            "total_patients": km.get('total_patients', 0),
                            "total_sites": km.get('total_sites', 0),
                            "total_studies": km.get('total_studies', 0),
                            "mean_dqi": km.get('mean_dqi', 85.0),
                            "clean_rate": km.get('clean_rate', 0.0),
                            "db_lock_ready": km.get('dblock_ready', 0.0),
                            "total_open_queries": km.get('total_open_queries', 0),
                            "total_missing_pages": km.get('total_missing_pages', 0),
                            "total_saes": km.get('total_saes', 0),
                            "total_protocol_deviations": km.get('total_protocol_deviations', 0),
                        },
                        "raw_data": {k: v for k, v in gen_vars.items() if k not in ('common_css', '_metadata')}
                    }
                    from src.knowledge.document_engine import GenerativeDocumentEngine
                    gen_engine = GenerativeDocumentEngine()
                    ai_res = gen_engine.generate_structured_report(report_type, ai_context)
                    ai_sections = ai_res.get('sections', {})
                    logger.info(f"AI sections for XLSX: {list(ai_sections.keys()) if ai_sections else 'EMPTY'}")
                    
                    # Regenerate XLSX with AI sections
                    from src.generation.export_engine import XLSXExporter
                    xlsx_exporter = XLSXExporter()
                    report_content = xlsx_exporter.generate(report_type, gen_vars, ai_sections=ai_sections)
                    logger.info(f"XLSX regenerated with AI: {len(report_content)} bytes")
                except Exception as ai_xlsx_err:
                    logger.warning(f"AI enrichment for XLSX failed, using original: {ai_xlsx_err}", exc_info=True)
            
            return Response(
                content=report_content,
                media_type=media_type,
                headers={
                    "Content-Disposition": f"attachment; filename={report_type}_{datetime.now().strftime('%Y%m%d')}.{target_fmt.value}"
                }
            )

        # =========================================================================
        # AI & RAG INJECTION (Phase 3) — Report.txt Quality
        # =========================================================================
        try:
            # Prepare context using generator's OWN variables (Ground Truth)
            gen_vars = result.variables if hasattr(result, 'variables') else {}
            
            def get_deep_metric(keys, default=0.0):
                curr = gen_vars
                for k in keys:
                    if isinstance(curr, dict): curr = curr.get(k, {})
                    else: return default
                return curr if isinstance(curr, (int, float)) else default

            # Sanitize raw data for context
            def sanitize_for_ai(obj):
                if isinstance(obj, dict):
                    return {k: sanitize_for_ai(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [sanitize_for_ai(x) for x in obj]
                elif isinstance(obj, (datetime, date)):
                    return obj.isoformat()
                return obj

            # Extract key metrics with full portfolio totals — UNIVERSAL across all report types
            km = gen_vars.get('key_metrics', {})
            
            # === Universal data source scanning ===
            site_data = gen_vars.get('site_data', gen_vars.get('site_details', {}))
            study_breakdown = gen_vars.get('study_breakdown', [])
            readiness_data = gen_vars.get('readiness_data', {})
            patient_data = gen_vars.get('patient_data', {})
            enrollment_data = gen_vars.get('enrollment_data', {})
            query_data = gen_vars.get('query_data', {})
            sae_summary = gen_vars.get('sae_summary', gen_vars.get('event_details', {}))
            coding_data = gen_vars.get('coding_data', {})
            regions = gen_vars.get('regions', [])
            
            # === Smart patient count ===
            total_patients = (
                km.get('total_patients') or km.get('patients')
                or (site_data.get('total_patients') if isinstance(site_data, dict) else 0)
                or patient_data.get('total_patients')
                or enrollment_data.get('total_enrolled')
                or sae_summary.get('total_patients')
                or sum(s.get('patients', 0) for s in study_breakdown)
                or sum(s.get('enrolled', 0) for s in enrollment_data.get('study_enrollment', []))
                or 0
            )
            
            # === Smart site count ===
            total_sites = (
                km.get('total_sites')
                or enrollment_data.get('total_sites')
                or readiness_data.get('total_sites')
                or sum(s.get('sites', 0) for s in study_breakdown)
                or len(regions)
                or (1 if isinstance(site_data, dict) and site_data.get('site_id') else 0)
                or 0
            )
            
            # === Smart study count ===
            study_assessments = readiness_data.get('study_assessments', [])
            total_studies = (
                km.get('total_studies')
                or enrollment_data.get('total_studies')
                or readiness_data.get('total_studies')
                or len(study_breakdown)
                or len(study_assessments)
                or len(query_data.get('per_study', []))
                or 0
            )
            
            # === DQI Score ===
            mean_dqi = (
                km.get('mean_dqi') or km.get('dqi')
                or (site_data.get('dqi_score') if isinstance(site_data, dict) else 0)
                or patient_data.get('avg_dqi')
                or 0
            )
            
            # === Clean Rate ===
            clean_rate = (
                km.get('clean_rate')
                or (site_data.get('clean_rate') if isinstance(site_data, dict) else 0)
                or 0
            )
            
            # === Open queries ===
            total_open_queries = (
                km.get('total_open_queries') or km.get('open_queries')
                or query_data.get('open')
                or (site_data.get('open_queries') if isinstance(site_data, dict) else 0)
                or readiness_data.get('total_open_queries')
                or sum(s.get('open_queries', 0) for s in study_breakdown)
                or sum(s.get('open_queries', 0) for s in study_assessments)
                or 0
            )
            
            # === SAEs ===
            total_saes = (
                km.get('total_saes')
                or sae_summary.get('total_sae_pending')
                or sum(s.get('sae_count', s.get('sae_pending', 0)) for s in study_breakdown)
                or sum(s.get('sae_pending', 0) for s in study_assessments)
                or 0
            )
            
            # === Missing pages ===
            total_missing_pages = (
                km.get('total_missing_pages')
                or readiness_data.get('total_missing_pages')
                or sum(s.get('missing_pages', 0) for s in study_breakdown)
                or sum(s.get('missing_pages', 0) for s in study_assessments)
                or 0
            )
            
            # === Protocol deviations ===
            total_deviations = (
                km.get('total_protocol_deviations')
                or sum(s.get('protocol_deviations', 0) for s in study_breakdown)
                or sum(s.get('protocol_deviations', 0) for s in study_assessments)
                or 0
            )
            
            # === DB Lock readiness ===
            db_lock_ready = (
                km.get('dblock_ready')
                or readiness_data.get('ready_rate')
                or 0
            )
            
            context = {
                "entity_id": request.site_id or request.study_id or "Portfolio",
                "study_id": request.study_id or "Global",
                "report_type": report_type,
                "metrics": {
                    "total_patients": total_patients,
                    "total_sites": total_sites,
                    "total_studies": total_studies,
                    "mean_dqi": mean_dqi,
                    "clean_rate": clean_rate,
                    "db_lock_ready": db_lock_ready,
                    "total_open_queries": total_open_queries,
                    "total_missing_pages": total_missing_pages,
                    "total_saes": total_saes,
                    "total_protocol_deviations": total_deviations,
                    "tier1_clean_rate": km.get('tier1_clean_rate', 0),
                    "tier2_clean_rate": km.get('tier2_clean_rate', 0),
                },
                "raw_data": sanitize_for_ai({k: v for k, v in gen_vars.items() if k not in ('common_css', '_metadata')})
            }
            
            from src.knowledge.document_engine import GenerativeDocumentEngine
            gen_engine = GenerativeDocumentEngine()
            
            ai_res = gen_engine.generate_structured_report(report_type, context)
            sections = ai_res.get('sections', {})
            is_fallback = ai_res.get('metadata', {}).get('fallback', False)
            
            if target_fmt == OutputFormat.HTML:
                # ---- EXECUTIVE SUMMARY (DeepAnalyze Protocol) ----
                title_prefix = "DeepAnalyze — Executive Assessment"
                badge_label = "AI-Powered Analysis"
                badge_bg = "#dbeafe"
                badge_color = "#1e40af"
                if is_fallback:
                    title_prefix = "Data Insight (Offline Analysis)"
                    badge_label = "Rule-Based Fallback"
                    badge_bg = "#fef3c7"
                    badge_color = "#92400e"
                
                ai_exec_html = f"""
                <div class="ai-narrative" id="ai-exec-summary" style="background:linear-gradient(135deg, #eff6ff 0%, #f0f9ff 100%) !important; border:1px solid #93c5fd !important; padding:30px !important; border-radius:12px !important; margin-bottom:30px !important; color:#1e3a5f !important; text-align:left !important; position:relative !important;">
                    <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:16px;">
                        <h3 style="color:#1e3a8a !important; margin:0 !important; font-weight:800 !important; font-size:18px !important; letter-spacing:-0.3px;">{title_prefix}</h3>
                        <span style="background:{badge_bg}; color:{badge_color}; padding:4px 12px; border-radius:20px; font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:0.5px;">{badge_label}</span>
                    </div>
                    <p style="color:#1e3a5f !important; font-size:14px !important; line-height:1.8 !important; white-space:pre-line; margin:0 !important;">{sections.get('executive_summary', 'Analysis pending.')}</p>
                    <div style="margin-top:16px; padding-top:12px; border-top:1px solid #bfdbfe; font-size:10px; color:#64748b; display:flex; gap:16px;">
                        <span>Source: Sanchalak AI Engine</span>
                        <span>Status: {'RAG Active' if not is_fallback else 'Offline Fallback'}</span>
                        <span>Model: {ai_res.get('metadata', {}).get('model', 'N/A')}</span>
                    </div>
                </div>
                """
                
                # ---- KEY FINDINGS ----
                raw_findings = sections.get('key_findings', [])
                findings_texts = []
                for f in raw_findings:
                    if isinstance(f, str):
                        findings_texts.append(f)
                    elif isinstance(f, dict):
                        # Extract meaningful text from dict findings
                        parts = []
                        for key in ('finding', 'title', 'category', 'metric', 'status', 'threshold'):
                            if f.get(key):
                                parts.append(str(f[key]))
                        findings_texts.append(' — '.join(parts) if parts else str(f))
                    else:
                        findings_texts.append(str(f))
                findings_list = "".join([f"<li style='color:#1e293b !important; margin-bottom:10px !important; text-align:left !important; line-height:1.6 !important; padding-left:4px;'>{t}</li>" for t in findings_texts])
                ai_findings_html = f"""
                <div id="ai-findings" style="background:#f8fafc !important; border:1px solid #e2e8f0 !important; padding:24px !important; border-radius:10px !important; margin-bottom:24px !important; text-align:left !important; width:100% !important; border-left:4px solid #3b82f6 !important;">
                    <h3 style="color:#0f172a !important; margin-top:0 !important; font-size:16px !important; font-weight:700 !important; text-align:left !important; text-transform:uppercase; letter-spacing:0.5px;">Key Findings (DeepAnalyze)</h3>
                    <ul style="font-size:13px !important; padding-left:20px !important; color:#1e293b !important; margin-top:12px !important; text-align:left !important; list-style-type:disc;">{findings_list}</ul>
                </div>
                """
                
                # ---- RISK ANALYSIS TABLE ----
                risk_table = sections.get('risk_table', sections.get('risk_analysis', []))
                risk_html = ""
                if isinstance(risk_table, list) and risk_table:
                    risk_rows = ""
                    for r in risk_table:
                        impact = r.get('impact', 'Moderate')
                        impact_color = '#dc2626' if impact == 'Critical' else '#ea580c' if impact == 'High' else '#2563eb'
                        risk_rows += f"""<tr>
                            <td style='color:#1e293b; padding:10px; border-bottom:1px solid #f1f5f9;'><strong>{r.get('category', '')}</strong></td>
                            <td style='color:#1e293b; padding:10px; border-bottom:1px solid #f1f5f9;'>{r.get('current_metric', '')}</td>
                            <td style='color:#1e293b; padding:10px; border-bottom:1px solid #f1f5f9;'>{r.get('threshold', '')}</td>
                            <td style='color:#1e293b; padding:10px; border-bottom:1px solid #f1f5f9;'>{r.get('gap', '')}</td>
                            <td style='color:{impact_color}; padding:10px; border-bottom:1px solid #f1f5f9; font-weight:600;'>{impact}</td>
                            <td style='color:#1e293b; padding:10px; border-bottom:1px solid #f1f5f9;'>{", ".join(r.get("highest_risk_studies", [])) if isinstance(r.get("highest_risk_studies"), list) else r.get("highest_risk_studies", "")}</td>
                        </tr>"""
                    risk_html = f"""
                    <div class="section" id="ai-risk-analysis" style="margin-top:30px; text-align:left; width:100%;">
                        <h2 style="color:#0f172a; font-size:18px; font-weight:700; margin-bottom:15px; text-transform:uppercase; letter-spacing:0.5px;">Portfolio Risk Analysis (DeepAnalyze)</h2>
                        <table style="width:100%; border-collapse:collapse; background:white; border:1px solid #e2e8f0; border-radius:8px; overflow:hidden;">
                            <thead style="background:#1e293b;"><tr>
                                <th style="padding:10px; text-align:left; color:white; font-size:11px; text-transform:uppercase;">Risk Area</th>
                                <th style="padding:10px; text-align:left; color:white; font-size:11px; text-transform:uppercase;">Current</th>
                                <th style="padding:10px; text-align:left; color:white; font-size:11px; text-transform:uppercase;">Threshold</th>
                                <th style="padding:10px; text-align:left; color:white; font-size:11px; text-transform:uppercase;">Gap</th>
                                <th style="padding:10px; text-align:left; color:white; font-size:11px; text-transform:uppercase;">Impact</th>
                                <th style="padding:10px; text-align:left; color:white; font-size:11px; text-transform:uppercase;">Highest Risk</th>
                            </tr></thead>
                            <tbody>{risk_rows}</tbody>
                        </table>
                    </div>
                    """
                
                # ---- RECOMMENDATIONS TABLE ----
                recs = sections.get('recommendations', [])
                recs_rows = ""
                for r in recs:
                    if isinstance(r, str):
                        # Recommendation is a plain string
                        recs_rows += f"""<tr>
                            <td style='color:#475569; padding:10px; border-bottom:1px solid #f1f5f9;' colspan='2'><strong>{r}</strong></td>
                            <td style='color:#1e293b; padding:10px; border-bottom:1px solid #f1f5f9;'>—</td>
                            <td style='background:#f0fdf4; color:#16a34a; padding:10px; border-bottom:1px solid #f1f5f9; font-weight:600; text-align:center;'>Medium</td>
                            <td style='color:#1e293b; padding:10px; border-bottom:1px solid #f1f5f9;'>—</td>
                        </tr>"""
                    elif isinstance(r, dict):
                        prio = r.get('priority', 'Medium')
                        prio_bg = '#fef2f2' if prio == 'Critical' else '#fffbeb' if prio == 'High' else '#f0fdf4'
                        prio_color = '#dc2626' if prio == 'Critical' else '#ea580c' if prio == 'High' else '#16a34a'
                        recs_rows += f"""<tr>
                            <td style='color:#475569; padding:10px; border-bottom:1px solid #f1f5f9; font-style:italic;'>{r.get('area', r.get('category', ''))}</td>
                            <td style='color:#1e293b; padding:10px; border-bottom:1px solid #f1f5f9;'><strong>{r.get('action', r.get('recommendation', r.get('description', '')))}</strong></td>
                            <td style='color:#1e293b; padding:10px; border-bottom:1px solid #f1f5f9;'>{r.get('owner', r.get('responsible', ''))}</td>
                            <td style='background:{prio_bg}; color:{prio_color}; padding:10px; border-bottom:1px solid #f1f5f9; font-weight:600; text-align:center;'>{prio}</td>
                            <td style='color:#1e293b; padding:10px; border-bottom:1px solid #f1f5f9;'>{r.get('target', r.get('timeline', ''))}</td>
                        </tr>"""
                ai_recs_html = f"""
                <div class="section" id="ai-recommendations" style="margin-top:40px; border-top:1px solid #e2e8f0; padding-top:30px; text-align:left; width:100%;">
                    <h2 style="color:#0f172a; font-size:20px; margin:0 0 20px 0; font-weight:700; text-transform:uppercase; letter-spacing:0.5px;">Strategic Recommendations (DeepAnalyze)</h2>
                    <table style="width:100%; border-collapse:collapse; background:white; border:1px solid #e2e8f0; border-radius:8px; overflow:hidden;">
                        <thead style="background:#f8fafc;"><tr>
                            <th style="padding:10px; text-align:left; color:#475569; font-size:11px; text-transform:uppercase;">Area</th>
                            <th style="padding:10px; text-align:left; color:#475569; font-size:11px; text-transform:uppercase;">Action Item</th>
                            <th style="padding:10px; text-align:left; color:#475569; font-size:11px; text-transform:uppercase;">Owner</th>
                            <th style="padding:10px; text-align:left; color:#475569; font-size:11px; text-transform:uppercase;">Priority</th>
                            <th style="padding:10px; text-align:left; color:#475569; font-size:11px; text-transform:uppercase;">Target</th>
                        </tr></thead>
                        <tbody>{recs_rows}</tbody>
                    </table>
                </div>
                """
                
                # ---- CONCLUSION ----
                conclusion = sections.get('conclusion', '')
                conclusion_html = ""
                if conclusion:
                    conclusion_html = f"""
                    <div class="section" id="ai-conclusion" style="margin-top:30px; background:linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 100%); border:1px solid #86efac; padding:28px; border-radius:12px; text-align:left;">
                        <h2 style="color:#166534; font-size:18px; font-weight:700; margin-top:0; text-transform:uppercase; letter-spacing:0.5px;">Conclusion & Action Targets</h2>
                        <p style="color:#15803d; font-size:14px; line-height:1.8; white-space:pre-line; margin-bottom:0;">{conclusion}</p>
                        <div style="margin-top:14px; padding-top:10px; border-top:1px solid #bbf7d0; font-size:10px; color:#64748b; display:flex; align-items:center; gap:8px;">
                            <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:#22c55e;"></span>
                            Prepared by: Sanchalak AI DeepAnalyze Engine — Data Steward, Medical Monitor &amp; Regulatory Affairs
                        </div>
                    </div>
                    """
                
                # ---- INJECT INTO HTML ----
                if '<!-- AI_EXEC_SUMMARY -->' in report_content:
                    report_content = report_content.replace('<!-- AI_EXEC_SUMMARY -->', ai_exec_html)
                else:
                    report_content = report_content.replace('<div class="content">', f'<div class="content">{ai_exec_html}', 1)
                
                if '<!-- AI_FINDINGS -->' in report_content:
                    report_content = report_content.replace('<!-- AI_FINDINGS -->', ai_findings_html)
                
                # Inject risk table + recommendations + conclusion before footer
                combined_injection = risk_html + ai_recs_html + conclusion_html
                combined_injection += '<style>#standard-actions, #standard-observations, #standard-observations-header { display: none !important; }</style>'
                
                if '<!-- AI_RECOMMENDATIONS -->' in report_content:
                    report_content = report_content.replace('<!-- AI_RECOMMENDATIONS -->', combined_injection)
                else:
                    if '</div>\n                        <div class="footer">' in report_content:
                        report_content = report_content.replace('</div>\n                        <div class="footer">', f'{combined_injection}</div>\n                        <div class="footer">', 1)
                    else:
                        report_content = report_content.replace('</div></body>', f'{combined_injection}</div></body>', 1)
            else:
                # Injected TEXT Blocks
                ai_exec_text = f"\nAI INTERPRETATION ({'RAG' if not is_fallback else 'OFFLINE'}):\n{sections.get('executive_summary', 'Analysis pending.')}\n"
                findings_text = "\nKEY FINDINGS:\n" + "\n".join([f"* {f}" for f in sections.get('key_findings', [])]) + "\n"
                
                # Risk table text
                risk_table = sections.get('risk_table', sections.get('risk_analysis', []))
                risk_text = "\nPORTFOLIO RISK ANALYSIS:\n"
                if isinstance(risk_table, list):
                    risk_text += f"{'Risk Area':<25}{'Current':<20}{'Threshold':<15}{'Gap':<15}{'Impact'}\n"
                    risk_text += "-"*85 + "\n"
                    for r in risk_table:
                        risk_text += f"{str(r.get('category',''))[:23]:<25}{str(r.get('current_metric',''))[:18]:<20}{str(r.get('threshold',''))[:13]:<15}{str(r.get('gap',''))[:13]:<15}{r.get('impact','')}\n"
                
                recs_text = "\nSTRATEGIC RECOMMENDATIONS (AI-GENERATED):\n"
                recs_text += f"{'Area':<18}{'Action Item':<32}{'Owner':<16}{'Priority':<12}{'Target'}\n"
                recs_text += "-"*95 + "\n"
                for r in sections.get('recommendations', []):
                    recs_text += f"{str(r.get('area',''))[:16]:<18}{str(r.get('action',''))[:30]:<32}{str(r.get('owner',''))[:14]:<16}{str(r.get('priority',''))[:10]:<12}{r.get('target','')}\n"
                
                conclusion_text = f"\nCONCLUSION:\n{sections.get('conclusion', '')}\n" if sections.get('conclusion') else ""
                
                report_content = report_content.replace('[AI_EXEC_SUMMARY]', ai_exec_text)
                report_content = report_content.replace('[AI_FINDINGS]', findings_text)
                report_content = report_content.replace('[AI_RECOMMENDATIONS]', risk_text + recs_text + conclusion_text)

        except Exception as ai_err:
            logger.warning(f"RAG Structured injection failed: {ai_err}", exc_info=True)

        return {
            "report_id": result.report_id,
            "report_content": report_content,
            "content": report_content,
            "generated_at": result.generated_at.isoformat(),
            "report_type": report_type,
            "format": target_fmt.value,
            "metadata": {
                "site_id": request.site_id,
                "study_id": request.study_id,
                "generated_by": current_user.get("username")
            }
        }
    except Exception as e:
        logger.error(f"Report generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/download/{report_type}")
async def download_report(
    report_type: str,
    format: str = "pdf",
    site_id: Optional[str] = None,
    study_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Generate and download a binary report."""
    try:
        from src.generation.report_generators import ReportGeneratorFactory, OutputFormat
        
        fmt_map = {
            "pdf": OutputFormat.PDF, "docx": OutputFormat.DOCX, "csv": OutputFormat.CSV,
            "json": OutputFormat.JSON, "xlsx": OutputFormat.XLSX, "excel": OutputFormat.XLSX,
            "txt": OutputFormat.TEXT, "text": OutputFormat.TEXT, "html": OutputFormat.HTML
        }
        target_format = fmt_map.get(format.lower(), OutputFormat.PDF)
        generator = ReportGeneratorFactory.get_generator(report_type)
        
        params = {
            "study_id": study_id,
            "site_id": site_id,
            "cra_name": current_user.get("full_name", "User"),
            "output_formats": [target_format]
        }
        
        sig = inspect.signature(generator.generate)
        params = {k: v for k, v in params.items() if k in sig.parameters}
        
        results = generator.generate(**params)
        if not results:
            raise HTTPException(status_code=500, detail="Failed to generate download")
            
        result = results[0]
        
        # XLSX generation via XLSXExporter with AI Analysis sheet
        if target_format == OutputFormat.XLSX:
            try:
                from src.generation.export_engine import XLSXExporter
                xlsx_exporter = XLSXExporter()
                gen_vars = result.variables if hasattr(result, 'variables') else {}
                
                # Generate AI sections for the AI Analysis sheet
                ai_sections = None
                try:
                    # Use same universal metrics extraction as the generate endpoint
                    km = gen_vars.get('key_metrics', {})
                    site_data_dl = gen_vars.get('site_data', gen_vars.get('site_details', {}))
                    study_breakdown_dl = gen_vars.get('study_breakdown', [])
                    readiness_data_dl = gen_vars.get('readiness_data', {})
                    patient_data_dl = gen_vars.get('patient_data', {})
                    enrollment_data_dl = gen_vars.get('enrollment_data', {})
                    query_data_dl = gen_vars.get('query_data', {})
                    sae_summary_dl = gen_vars.get('sae_summary', gen_vars.get('event_details', {}))
                    study_assessments_dl = readiness_data_dl.get('study_assessments', [])
                    
                    dl_patients = (km.get('total_patients') or km.get('patients')
                        or (site_data_dl.get('total_patients') if isinstance(site_data_dl, dict) else 0)
                        or patient_data_dl.get('total_patients') or enrollment_data_dl.get('total_enrolled')
                        or sae_summary_dl.get('total_patients') or sum(s.get('patients', 0) for s in study_breakdown_dl) or 0)
                    dl_sites = (km.get('total_sites') or enrollment_data_dl.get('total_sites')
                        or readiness_data_dl.get('total_sites') or sum(s.get('sites', 0) for s in study_breakdown_dl)
                        or (1 if isinstance(site_data_dl, dict) and site_data_dl.get('site_id') else 0) or 0)
                    dl_studies = (km.get('total_studies') or enrollment_data_dl.get('total_studies')
                        or readiness_data_dl.get('total_studies') or len(study_breakdown_dl)
                        or len(study_assessments_dl) or len(query_data_dl.get('per_study', [])) or 0)
                    dl_dqi = (km.get('mean_dqi') or km.get('dqi')
                        or (site_data_dl.get('dqi_score') if isinstance(site_data_dl, dict) else 0)
                        or patient_data_dl.get('avg_dqi') or 0)
                    dl_queries = (km.get('total_open_queries') or km.get('open_queries')
                        or query_data_dl.get('open') or (site_data_dl.get('open_queries') if isinstance(site_data_dl, dict) else 0)
                        or readiness_data_dl.get('total_open_queries') or sum(s.get('open_queries', 0) for s in study_breakdown_dl) or 0)
                    dl_saes = (km.get('total_saes') or sae_summary_dl.get('total_sae_pending')
                        or sum(s.get('sae_count', s.get('sae_pending', 0)) for s in study_breakdown_dl) or 0)
                    dl_clean = (km.get('clean_rate') or (site_data_dl.get('clean_rate') if isinstance(site_data_dl, dict) else 0) or 0)
                    
                    ai_context = {
                        "entity_id": site_id or study_id or "Portfolio",
                        "study_id": study_id or "Global",
                        "report_type": report_type,
                        "metrics": {
                            "total_patients": dl_patients,
                            "total_sites": dl_sites,
                            "total_studies": dl_studies,
                            "mean_dqi": dl_dqi,
                            "clean_rate": dl_clean,
                            "total_open_queries": dl_queries,
                            "total_saes": dl_saes,
                            "total_missing_pages": km.get('total_missing_pages', readiness_data_dl.get('total_missing_pages', 0)),
                            "total_protocol_deviations": km.get('total_protocol_deviations', sum(s.get('protocol_deviations', 0) for s in study_breakdown_dl)),
                        },
                        "raw_data": {k: v for k, v in gen_vars.items() if k not in ('common_css', '_metadata')}
                    }
                    from src.knowledge.document_engine import GenerativeDocumentEngine
                    gen_engine = GenerativeDocumentEngine()
                    ai_res = gen_engine.generate_structured_report(report_type, ai_context)
                    ai_sections = ai_res.get('sections', {})
                except Exception as ai_err:
                    logger.warning(f"AI generation for download XLSX failed: {ai_err}")
                
                xlsx_bytes = xlsx_exporter.generate(report_type, gen_vars, ai_sections=ai_sections)
                return Response(
                    content=xlsx_bytes,
                    media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    headers={"Content-Disposition": f"attachment; filename={report_type}_{datetime.now().strftime('%Y%m%d')}.xlsx"}
                )
            except Exception as xlsx_err:
                logger.warning(f"XLSX export failed, falling back to HTML: {xlsx_err}")
        
        if not result.content and result.html_content:
            printable_html = f"<html><body onload='window.print()'>{result.html_content}</body></html>"
            return Response(content=printable_html, media_type="text/html")
            
        return Response(
            content=result.content,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename=report.{format}"}
        )
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
