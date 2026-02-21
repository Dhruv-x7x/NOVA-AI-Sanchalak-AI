"""
Study Routes
============
Study management endpoints.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Optional

from app.models.schemas import StudyListResponse
from app.core.security import get_current_user
from app.services.database import get_data_service

router = APIRouter()


@router.get("", response_model=StudyListResponse)
async def list_studies(
    phase: Optional[str] = None,
    status: Optional[str] = None,
    therapeutic_area: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Get all studies with optional filters."""
    try:
        data_service = get_data_service()
        df = data_service.get_studies()
        
        if df.empty:
            return StudyListResponse(studies=[], total=0)
        
        # Apply filters
        if phase:
            df = df[df["phase"] == phase]
        if status:
            df = df[df["status"] == status]
        if therapeutic_area:
            df = df[df["therapeutic_area"] == therapeutic_area]
        
        # Note: Patient counts and avg DQI are now aggregated by the Data Service
        # for optimal performance via PostgreSQL SQL.
        
        from .patients import sanitize_for_json
        return StudyListResponse(
            studies=sanitize_for_json(df.to_dict(orient="records")),
            total=len(df)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{study_id}")
async def get_study(
    study_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get single study details."""
    if study_id in ("Unknown", "all", "Multiple"):
        return {
            "study_id": "all",
            "name": "Global Portfolio",
            "status": "Active",
            "phase": "III",
            "therapeutic_area": "Various",
            "current_enrollment": 1250,
            "target_enrollment": 1500,
            "start_date": "2023-01-01",
            "end_date": "2024-12-31"
        }
        
    try:
        data_service = get_data_service()
        df = data_service.get_studies()
        
        if df.empty:
            raise HTTPException(status_code=404, detail="Study not found")
        
        study_df = df[df["study_id"] == study_id]
        
        if study_df.empty:
            raise HTTPException(status_code=404, detail="Study not found")
        
        from .patients import sanitize_for_json
        return sanitize_for_json(study_df.to_dict(orient="records")[0])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{study_id}/sites")
async def get_study_sites(
    study_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get sites participating in a study."""
    try:
        data_service = get_data_service()
        
        # Get patients to find sites in this study
        patients_df = data_service.get_patients()
        
        if patients_df.empty:
            return {"sites": [], "total": 0}
        
        # Filter by study
        study_patients = patients_df[patients_df["study_id"] == study_id]
        
        if study_patients.empty:
            return {"sites": [], "total": 0}
        
        # Get unique sites
        site_ids = study_patients["site_id"].unique().tolist()
        
        # Get site details
        sites_df = data_service.get_sites()
        sites_df = sites_df[sites_df["site_id"].isin(site_ids)]
        
        from .patients import sanitize_for_json
        return {
            "sites": sanitize_for_json(sites_df.to_dict(orient="records")),
            "total": len(sites_df)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{study_id}/patients")
async def get_study_patients(
    study_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get patients in a study."""
    try:
        data_service = get_data_service()
        df = data_service.get_patients()
        
        if df.empty:
            return {"patients": [], "total": 0}
        
        # Filter by study
        df = df[df["study_id"] == study_id]
        
        from .patients import sanitize_for_json
        return {
            "patients": sanitize_for_json(df.to_dict(orient="records")),
            "total": len(df)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
