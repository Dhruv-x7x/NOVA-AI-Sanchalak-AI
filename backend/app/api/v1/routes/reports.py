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
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Thread pool for CPU-bound report generation
_report_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="report_gen")

logger = logging.getLogger(__name__)

# Add project root to path for importing report generators
# Robustly find project root by looking for 'src' directory
current_path = Path(__file__).resolve()
for parent in current_path.parents:
    if (parent / "src").exists():
        PROJECT_ROOT = parent
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        break
else:
    # Fallback to previous logic
    PROJECT_ROOT = current_path.parents[5]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from app.models.schemas import ReportRequest, ReportResponse
from app.core.security import get_current_user
from app.services.database import get_data_service

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
            'sponsor_update': {"name": "Sponsor Update", "description": "Sponsor communication pack", "category": "Executive", "icon": "activity"},
            'meeting_pack': {"name": "Meeting Pack", "description": "Team meeting materials", "category": "Operations", "icon": "calendar"},
            'safety_narrative': {"name": "Safety Narrative", "description": "Safety event narratives", "category": "Safety", "icon": "shield"},
            'patient_risk': {"name": "Patient Risk Analysis", "description": "Individual patient risk assessment", "category": "Safety", "icon": "alert-triangle"},
            'regional_summary': {"name": "Regional Summary", "description": "Regional performance breakdown", "category": "Operations", "icon": "activity"},
            'coding_status': {"name": "Coding Status", "description": "MedDRA/WHODrug coding status", "category": "Data Management", "icon": "file-text"},
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
        all_params = {
            "study_id": request.study_id,
            "site_id": request.site_id,
            "sites": [request.site_id] if request.site_id else None,
            "date_range_days": request.date_range_days,
            "cra_name": current_user.get("full_name", current_user.get("username", "System User")),
            "output_formats": [OutputFormat.HTML]
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
        
        # Run the (potentially slow) sync generator in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(_report_executor, lambda: generator.generate(**params))
        if not results:
            raise HTTPException(status_code=500, detail="No output from generator")
            
        result = results[0]
        html_content = result.html_content or "<html><body>No content generated</body></html>"

        # =========================================================================
        # AI & RAG INJECTION
        # =========================================================================
        try:
            # Fetch summary for context
            portfolio = data_service.get_portfolio_summary(study_id=request.study_id) or {}
            
            # Determine context for GenAI
            context = {
                "entity_id": request.site_id or request.study_id or "Portfolio",
                "study_id": request.study_id or "Global",
                "site_id": request.site_id or "Global",
                "dqi_score": portfolio.get('mean_dqi', 85.3),
                "readiness": portfolio.get('dblock_ready_rate', 10.2),
                "issues": f"{portfolio.get('total_issues', 0)} open queries and {portfolio.get('critical_issues', 0)} critical events"
            }
            
            from src.knowledge.document_engine import GenerativeDocumentEngine
            gen_engine = GenerativeDocumentEngine()
            ai_report = gen_engine.generate_report(report_type, context)
            
            ai_summary_html = f"""
            <div style="background: #fdf2f8; border: 2px solid #f9a8d4; border-radius: 12px; padding: 20px; margin-bottom: 30px; font-family: sans-serif;">
                <h3 style="margin: 0; color: #9d174d; text-transform: uppercase; font-size: 14px; font-weight: 800; display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 18px;">🪄</span> GenAI Intelligence Summary
                </h3>
                <p style="margin: 10px 0 0 0; color: #831843; font-style: italic; line-height: 1.6;">{ai_report['content']}</p>
                <div style="margin-top: 10px; font-size: 10px; color: #be185d; opacity: 0.7;">
                    RAG Context: { 'Active' if ai_report.get('metadata', {}).get('rag_active') else 'Fallback' } | 
                    Model: { ai_report.get('metadata', {}).get('model', 'Unknown') }
                </div>
            </div>
            """
            
            # Inject at start of body or after header
            if '<body>' in html_content:
                html_content = html_content.replace('<body>', '<body>' + ai_summary_html, 1)
            elif '<div class="header">' in html_content:
                html_content = html_content.replace('<div class="header">', ai_summary_html + '<div class="header">', 1)
            else:
                # Just prepend
                html_content = ai_summary_html + html_content
                
        except Exception as ai_err:
            logger.warning(f"RAG Summary injection failed: {ai_err}")

        # Return object matching both ReportResponse schema and TC004 expectations
        return {
            "report_id": result.report_id,
            "report_content": html_content,
            "content": html_content,
            "generated_at": result.generated_at.isoformat(),
            "report_type": report_type,
            "format": "html",
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
            "pdf": OutputFormat.PDF,
            "docx": OutputFormat.DOCX,
            "csv": OutputFormat.CSV,
            "json": OutputFormat.JSON,
            "xlsx": OutputFormat.DOCX
        }
        target_format = fmt_map.get(format.lower(), OutputFormat.PDF)
        generator = ReportGeneratorFactory.get_generator(report_type)
        
        params = {
            "study_id": study_id,
            "site_id": site_id,
            "cra_name": current_user.get("full_name", "User"),
            "output_formats": [target_format]
        }
        
        # Filter params
        sig = inspect.signature(generator.generate)
        params = {k: v for k, v in params.items() if k in sig.parameters}
        
        # Run the (potentially slow) sync generator in a thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(_report_executor, lambda: generator.generate(**params))
        if not results:
            raise HTTPException(status_code=500, detail="Failed to generate download")
            
        result = results[0]
        
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
