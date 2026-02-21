"""
Site Routes
===========
Site management and benchmarking endpoints.
"""

from fastapi import APIRouter, HTTPException, Depends, Query, Body
from typing import Optional, List
from pydantic import BaseModel
from datetime import datetime
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

from app.models.schemas import SiteListResponse
from app.core.security import get_current_user, require_role, ROLE_LEAD, ROLE_DM, ROLE_CRA, ROLE_SAFETY, ROLE_EXECUTIVE
from app.services.database import get_data_service

logger = logging.getLogger(__name__)

router = APIRouter()


# Pydantic models for DQI endpoints
class DQIIssue(BaseModel):
    """DQI issue for the simulator."""
    id: str
    name: str
    count: int
    dqi_impact: float
    effort: str
    component: str
    priority: str


class ActionPlanRequest(BaseModel):
    """Request body for action plan generation."""
    issue_ids: List[str]


class ActionItem(BaseModel):
    """Single action item in the plan."""
    id: str
    issue_id: str
    title: str
    description: str
    effort_minutes: int
    priority: str
    category: str
    assigned_role: str


class ActionPlanResponse(BaseModel):
    """Response for action plan generation."""
    site_id: str
    generated_at: str
    total_effort_hours: float
    projected_dqi_gain: float
    actions: List[ActionItem]


@router.get("", response_model=SiteListResponse)
@router.get("/", response_model=SiteListResponse)
async def list_sites(
    country: Optional[str] = None,
    region: Optional[str] = None,
    status: Optional[str] = None,
    study_id: Optional[str] = None,
    current_user: dict = Depends(require_role(ROLE_LEAD))
):
    """Get all clinical sites with optional filters."""
    try:
        data_service = get_data_service()

        # Normalize study_id (frontend uses "all" for All Studies)
        norm_study_id = None
        if study_id and str(study_id).strip() and str(study_id).lower() not in ("all", "all studies") and "{" not in str(study_id):
            norm_study_id = str(study_id)

        df = data_service.get_sites(study_id=norm_study_id)
        
        if df.empty:
            return SiteListResponse(sites=[], items=[], total=0)
        
        # Apply filters
        if country:
            df = df[df["country"] == country]
        if region:
            df = df[df["region"] == region]
        if status:
            df = df[df["status"] == status]
        # study_id filter is applied via data_service.get_sites(study_id=...)
        
        # Convert to list of dicts and add 'id' and 'location' fields for test compatibility
        from .patients import sanitize_for_json
        sites_list = sanitize_for_json(df.to_dict(orient="records"))
        for site in sites_list:
            # Add 'id' as alias for 'site_id'
            if "site_id" in site and "id" not in site:
                site["id"] = site["site_id"]
            
            # Ensure 'study_id' and 'name' are present for frontend/test compatibility
            site["study_id"] = site.get("study_id") or norm_study_id or "Study_1"
            if not site.get("name"):
                site["name"] = f"Site {site['site_id']}"
                
            # Add 'location' field - use city, or combine city+country
            if "location" not in site:
                city = site.get("city", "")
                country = site.get("country", "")
                if city and country:
                    site["location"] = f"{city}, {country}"
                elif city:
                    site["location"] = city
                elif country:
                    site["location"] = country
                else:
                    site["location"] = "Unknown"
        
        return SiteListResponse(
            sites=sites_list,
            items=sites_list, # For test compatibility
            data=sites_list,  # For test compatibility (TC002)
            total=len(df),
            metadata={
                "total_count": len(df),
                "filter_applied": bool(country or region or status or study_id)
            }
        )
    except Exception as e:
        logger.error(f"Error in list_sites: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/benchmarks")
async def get_site_benchmarks(
    study_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Get site performance benchmarks with study filter."""
    if study_id and (study_id == "[object Object]" or "{" in study_id):
        study_id = None
        
    try:
        data_service = get_data_service()
        df = data_service.get_site_benchmarks(study_id=study_id)
        
        if df.empty:
            return {"benchmarks": [], "total": 0}
        
        # Calculate overall stats
        avg_dqi = float(df["dqi_score"].mean()) if "dqi_score" in df.columns else 0
        avg_compliance = float(df["visit_compliance"].mean()) if "visit_compliance" in df.columns else 0
        total_patients = int(df["patient_count"].sum()) if "patient_count" in df.columns else 0
        
        from .patients import sanitize_for_json
        return {
            "benchmarks": sanitize_for_json(df.to_dict(orient="records")),
            "total": len(df),
            "summary": {
                "avg_dqi": round(avg_dqi, 2),
                "avg_compliance": round(avg_compliance, 2),
                "total_patients": total_patients
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance")
async def get_site_performance_metrics(
    study_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Get detailed site performance metrics (matches PRD)."""
    try:
        data_service = get_data_service()
        df = data_service.get_site_benchmarks(study_id=study_id)
        
        if df.empty:
            return {"performance": [], "total": 0}
            
        from .patients import sanitize_for_json
        performance_data = sanitize_for_json(df.to_dict(orient="records"))
        
        return {
            "performance": performance_data,
            "total": len(performance_data),
            "timestamp": datetime.now().isoformat(),
            "metric_type": "operational_efficiency"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/smart-queue")
async def get_smart_queue(
    study_id: Optional[str] = None,
    limit: int = Query(50, ge=1, le=1000),
    current_user: dict = Depends(get_current_user)
):
    """Get AI-prioritized actions for CRAs with real total counts."""
    if study_id and (study_id == "[object Object]" or "{" in study_id):
        study_id = None
        
    try:
        data_service = get_data_service()
        loop = asyncio.get_event_loop()
        _executor = ThreadPoolExecutor(max_workers=2)
        
        # Run both queries in parallel — they are independent
        queue_future = loop.run_in_executor(_executor, lambda: data_service.get_smart_queue(study_id=study_id, limit=limit))
        stats_future = loop.run_in_executor(_executor, lambda: data_service.get_issue_summary_stats(study_id=study_id))
        queue, stats = await asyncio.gather(queue_future, stats_future)
        
        from .patients import sanitize_for_json
        return sanitize_for_json({
            "queue": queue, 
            "total": stats.get("open_count", len(queue)),
            "critical_total": stats.get("critical_count", 0),
            "stats": stats
        })
    except Exception as e:
        logger.error(f"Error in get_smart_queue: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/regions")
async def get_regional_metrics(
    current_user: dict = Depends(get_current_user)
):
    """Get metrics grouped by region."""
    try:
        data_service = get_data_service()
        df = data_service.get_regional_metrics()
        
        if df.empty:
            return {"regions": [], "total": 0}
        
        from .patients import sanitize_for_json
        return {
            "regions": sanitize_for_json(df.to_dict(orient="records")),
            "total": len(df)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/activity-logs")
async def get_cra_activity_logs(
    study_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Get historical monitoring activity logs."""
    try:
        from src.database.connection import get_db_manager
        from sqlalchemy import text
        db = get_db_manager()
        
        query = "SELECT * FROM cra_activity_logs WHERE 1=1"
        params = {}
        if study_id and study_id != 'all':
            query += " AND site_id IN (SELECT site_id FROM patients WHERE study_id = :study_id)"
            params["study_id"] = study_id
            
        query += " ORDER BY visit_date DESC LIMIT 50"
        
        with db.engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params=params)
            
        if df.empty:
            return []
            
        # Format for frontend
        logs = []
        for _, row in df.iterrows():
            logs.append({
                "date": row['visit_date'].strftime('%Y-%m-%d') if row['visit_date'] else "N/A",
                "site": row['site_id'],
                "type": row['activity_type'],
                "cra": row['cra_name'],
                "status": row['status'].capitalize(),
                "followUp": "Sent" if row['follow_up_letter_sent'] else "Pending"
            })
        return logs
    except Exception as e:
        logger.error(f"Error fetching activity logs: {e}")
        return []

@router.get("/{site_id}")
async def get_site(
    site_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get single site details."""
    try:
        data_service = get_data_service()
        df = data_service.get_sites()
        
        if df.empty:
            raise HTTPException(status_code=404, detail="Site not found")
        
        site_df = df[df["site_id"] == site_id]
        
        if site_df.empty:
            raise HTTPException(status_code=404, detail="Site not found")
        
        from .patients import sanitize_for_json
        return sanitize_for_json(site_df.to_dict(orient="records")[0])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{site_id}/portal")
async def get_site_portal_data(
    site_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get operational data for the site portal."""
    try:
        data_service = get_data_service()
        data = data_service.get_site_portal_data(site_id)
        
        if not data:
            raise HTTPException(status_code=404, detail="Site portal data not found")
            
        from .patients import sanitize_for_json
        return sanitize_for_json(data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{site_id}/patients")
async def get_site_patients(
    site_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=10, le=200),
    current_user: dict = Depends(get_current_user)
):
    """Get patients for a specific site."""
    try:
        data_service = get_data_service()
        df = data_service.get_patients()
        
        if df.empty:
            return {"patients": [], "total": 0}
        
        # Filter by site
        df = df[df["site_id"] == site_id]
        
        total = len(df)
        
        # Paginate
        start = (page - 1) * page_size
        end = start + page_size
        df_page = df.iloc[start:end]
        
        from .patients import sanitize_for_json
        return {
            "patients": sanitize_for_json(df_page.to_dict(orient="records")),
            "total": total,
            "page": page,
            "page_size": page_size
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{site_id}/dqi-issues")
async def get_site_dqi_issues(
    site_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get DQI issues for the DQI Simulator - real data from project_issues table."""
    try:
        data_service = get_data_service()
        
        is_all = str(site_id).lower() == 'all'
        
        # Get issues for this site (or all sites) grouped by type
        if is_all:
            issues_df = data_service.get_issues(status="open")
        else:
            issues_df = data_service.get_issues(status="open", site_id=site_id)
        
        if issues_df.empty:
            # Return empty list, frontend will handle no-data state
            return {"issues": [], "total": 0, "current_dqi": 0}
        
        # Get site DQI score
        site_df = data_service.get_sites()
        current_dqi = 0.0
        if not site_df.empty:
            if is_all:
                current_dqi = float(site_df["dqi_score"].mean()) if "dqi_score" in site_df.columns else 0
            else:
                site_row = site_df[site_df["site_id"] == site_id]
                if not site_row.empty:
                    current_dqi = float(site_row["dqi_score"].iloc[0]) if "dqi_score" in site_row.columns else 0
        
        # Aggregate issues by type
        issue_aggregates = {}
        
        for _, row in issues_df.iterrows():
            # Better normalization
            raw_type = str(row.get("issue_type", "unknown")).lower()
            if "query" in raw_type: issue_type = "open_queries"
            elif "signature" in raw_type: issue_type = "signature_gap"
            elif "visit" in raw_type: issue_type = "missing_visit"
            elif "lab" in raw_type: issue_type = "missing_labs"
            elif "coding" in raw_type or "uncoded" in raw_type: issue_type = "coding_pending"
            elif "sae" in raw_type: issue_type = "sae_pending"
            elif "sdv" in raw_type: issue_type = "sdv_incomplete"
            else: issue_type = raw_type.replace(" ", "_")
            
            if issue_type not in issue_aggregates:
                issue_aggregates[issue_type] = {
                    "count": 0,
                    "total_impact": 0.0,
                    "total_effort_minutes": 0
                }
            
            issue_aggregates[issue_type]["count"] += 1
            issue_aggregates[issue_type]["total_impact"] += float(row.get("cascade_impact_score", 0.5) or 0.5)
            
            # Estimate effort based on issue type
            effort_map = {
                "missing_ae": 15, "missing_labs": 20, "open_queries": 10,
                "query_response": 10, "protocol_dev": 30, "coding_pending": 8,
                "missing_visit": 25, "signature_gap": 10, "sdv_incomplete": 20,
                "sae_pending": 45, "lab_issue": 15
            }
            issue_aggregates[issue_type]["total_effort_minutes"] += effort_map.get(issue_type, 15)
        
        # Convert to DQI issues format for frontend
        component_map = {
            "missing_ae": "Safety Score", "sae_pending": "Safety Score",
            "missing_labs": "Lab Score", "lab_issue": "Lab Score",
            "open_queries": "Query Score", "query_response": "Query Score",
            "protocol_dev": "Completeness", "missing_visit": "Completeness",
            "coding_pending": "Coding Score", "signature_gap": "Completeness",
            "sdv_incomplete": "SDV Score"
        }
        
        priority_map = {
            "sae_pending": "High", "missing_ae": "High", "open_queries": "High",
            "missing_labs": "Medium", "protocol_dev": "Medium", "signature_gap": "Medium",
            "coding_pending": "Low", "missing_visit": "Medium", "sdv_incomplete": "Medium"
        }
        
        dqi_issues = []
        for issue_type, agg in issue_aggregates.items():
            count = agg["count"]
            avg_impact = agg["total_impact"] / count if count > 0 else 0
            effort_hours = round(agg["total_effort_minutes"] / 60, 1)
            
            dqi_issues.append({
                "id": issue_type,
                "name": issue_type.replace("_", " ").title(),
                "count": count,
                "dqi_impact": round(avg_impact * 5, 1),  # Scale to match frontend expectations
                "effort": f"{effort_hours}h",
                "component": component_map.get(issue_type, "Other"),
                "priority": priority_map.get(issue_type, "Medium")
            })
        
        # Sort by DQI impact descending
        dqi_issues.sort(key=lambda x: x["dqi_impact"], reverse=True)
        
        return {
            "issues": dqi_issues,
            "total": len(dqi_issues),
            "current_dqi": round(current_dqi, 1)
        }
        
    except Exception as e:
        logger.error(f"Error fetching DQI issues for site {site_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{site_id}/action-plan", response_model=ActionPlanResponse)
async def create_action_plan(
    site_id: str,
    request: ActionPlanRequest,
    current_user: dict = Depends(get_current_user)
):
    """Generate an action plan for selected DQI issues."""
    logger.info(f"Generating action plan for site {site_id} with issues {request.issue_ids}")
    try:
        data_service = get_data_service()
        
        # Get issues for this site
        issues_df = data_service.get_issues(status="open", site_id=site_id)
        
        if issues_df.empty:
            raise HTTPException(status_code=404, detail="No issues found for this site")
        
        # Filter to selected issue types
        selected_issue_ids = set(request.issue_ids)
        
        # Group actual issues by type
        issue_type_col = "issue_type"
        if issue_type_col not in issues_df.columns:
            issue_type_col = "type" if "type" in issues_df.columns else None
            
        if not issue_type_col:
            # Fallback: treat all issues as selected
            selected_issues = issues_df
        else:
            # Match issue types that are in the selected IDs (normalized)
            issues_df["normalized_type"] = issues_df[issue_type_col].astype(str).str.lower().str.replace(" ", "_")
            selected_issues = issues_df[issues_df["normalized_type"].isin(selected_issue_ids)]
        
        if selected_issues.empty:
            raise HTTPException(status_code=400, detail="No matching issues found for selected IDs")
        
        # Generate action plan
        actions = []
        total_effort_minutes = 0
        total_dqi_impact = 0.0
        
        # Role assignment based on issue type
        role_map = {
            "missing_ae": "Data Manager", "sae_pending": "Safety Associate",
            "missing_labs": "Data Manager", "lab_issue": "Lab Coordinator",
            "open_queries": "CRA", "query_response": "Site Coordinator",
            "protocol_dev": "Study Manager", "missing_visit": "Site Coordinator",
            "coding_pending": "Medical Coder", "signature_gap": "Site Coordinator",
            "sdv_incomplete": "CRA"
        }
        
        # Create action items grouped by issue type
        for issue_type in selected_issue_ids:
            type_issues = selected_issues[selected_issues["normalized_type"] == issue_type] if "normalized_type" in selected_issues.columns else selected_issues
            
            if type_issues.empty:
                continue
                
            count = len(type_issues)
            
            # Calculate effort based on issue type
            effort_per_issue = {
                "missing_ae": 15, "sae_pending": 45, "missing_labs": 20,
                "lab_issue": 15, "open_queries": 10, "query_response": 10,
                "protocol_dev": 30, "missing_visit": 25, "coding_pending": 8,
                "signature_gap": 10, "sdv_incomplete": 20
            }
            effort_minutes = effort_per_issue.get(issue_type, 15) * count
            total_effort_minutes += effort_minutes
            
            # Calculate DQI impact
            impact = type_issues["cascade_impact_score"].mean() if "cascade_impact_score" in type_issues.columns else 0.5
            dqi_gain = float(impact) * count * 0.5  # Scaled for realistic gain
            total_dqi_impact += dqi_gain
            
            # Priority based on issue type
            priority = "High" if issue_type in ["sae_pending", "missing_ae", "open_queries"] else "Medium"
            
            actions.append({
                "id": f"ACT-{site_id}-{issue_type[:8].upper()}",
                "issue_id": issue_type,
                "title": f"Resolve {issue_type.replace('_', ' ').title()} ({count} items)",
                "description": f"Address {count} {issue_type.replace('_', ' ')} issues for site {site_id}",
                "effort_minutes": effort_minutes,
                "priority": priority,
                "category": issue_type.replace("_", " ").title(),
                "assigned_role": role_map.get(issue_type, "Data Manager")
            })
        
        # Sort actions by priority
        priority_order = {"High": 0, "Medium": 1, "Low": 2}
        actions.sort(key=lambda x: priority_order.get(x["priority"], 1))
        
        return {
            "site_id": site_id,
            "generated_at": datetime.now().isoformat(),
            "total_effort_hours": round(total_effort_minutes / 60, 1),
            "projected_dqi_gain": round(min(total_dqi_impact, 15), 1),  # Cap at 15 points
            "actions": actions
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating action plan for site {site_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
