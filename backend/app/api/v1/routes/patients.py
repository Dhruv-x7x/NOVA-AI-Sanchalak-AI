"""
Patient Routes
==============
Patient CRUD and analytics endpoints.
"""

from fastapi import APIRouter, HTTPException, status as fastapi_status, Depends, Query
from typing import Optional, List, Dict, Any, Union
import numpy as np
import math
import pandas as pd
from datetime import datetime, date
import sys
import os
import logging
from pathlib import Path

api_logger = logging.getLogger("uvicorn.error")

# Add project root to path for ML imports
# backend/app/api/v1/routes/patients.py -> parents[5] is sanchalak_ai
PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.models.schemas import PatientListResponse, PatientDetail, PatientSearchRequest
from app.core.security import get_current_user
from app.services.database import get_data_service

router = APIRouter()


def sanitize_for_json(data: Any) -> Any:
    """
    Sanitize data for JSON serialization by replacing NaN/Inf values with None
    and converting datetime/Timestamp objects to ISO format strings.
    Works with lists of dicts (from DataFrame.to_dict(orient='records')).
    """
    import pandas as pd
    from datetime import datetime, date
    
    if isinstance(data, list):
        return [sanitize_for_json(item) for item in data]
    elif isinstance(data, dict):
        return {k: sanitize_for_json(v) for k, v in data.items()}
    elif isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return None
        return data
    elif isinstance(data, (np.floating, np.integer, np.number)):
        if np.isnan(data) or np.isinf(data):
            return None
        return float(data) if isinstance(data, (np.floating, float)) else int(data)
    elif isinstance(data, (pd.Timestamp, datetime)):
        return data.isoformat() if pd.notna(data) else None
    elif isinstance(data, date):
        return data.isoformat() if data else None
    elif pd.isna(data):
        return None
    else:
        return data


@router.get("", response_model=PatientListResponse)
@router.head("")
async def list_patients(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    site_id: Optional[str] = None,
    study_id: Optional[str] = None,
    status: Optional[str] = None,
    risk_level: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Get paginated list of patients with optional filters."""
    # Guard against React Query objects or "all" string
    if study_id and (study_id == "[object Object]" or "{" in study_id or study_id.lower() == "all"):
        study_id = None
    if site_id and (site_id == "[object Object]" or "{" in site_id or site_id.lower() == "all"):
        site_id = None
        
    try:
        data_service = get_data_service()
        df = data_service.get_patients(study_id=study_id, site_id=site_id)
        
        if df.empty:
            return PatientListResponse(patients=[], total=0, page=page, page_size=page_size)
        
        # Apply other filters in memory if they were not pushed to SQL
        if status and status != 'all':
            df = df[df["status"] == status]
        if risk_level and risk_level != 'all':
            df = df[df["risk_level"] == risk_level]
        
        total = len(df)
        
        # Paginate
        start = (page - 1) * page_size
        end = start + page_size
        df_page = df.iloc[start:end].copy()
        
        # Fill NaN values before converting to dict to avoid JSON serialization errors
        for col in df_page.columns:
            if df_page[col].dtype == 'object':
                df_page.loc[:, col] = df_page[col].fillna('')
            else:
                df_page.loc[:, col] = df_page[col].fillna(0)
        
        # Convert to dict and sanitize for JSON
        patients: List[Dict[str, Any]] = sanitize_for_json(df_page.to_dict(orient="records"))
        
        # Add 'id' field as alias for 'patient_key' for frontend/test compatibility
        for patient in patients:
            if "patient_key" in patient and "id" not in patient:
                patient["id"] = patient["patient_key"]
        
        return PatientListResponse(
            patients=patients,
            items=patients, # For test compatibility
            data=patients,  # For test compatibility
            total=total,
            page=page,
            page_size=page_size
        )
    except Exception as e:
        api_logger.error(f"Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search")
async def search_patients(
    q: str = Query(..., min_length=2),
    limit: int = Query(20, ge=1, le=100),
    current_user: dict = Depends(get_current_user)
):
    """Search patients by key or site."""
    try:
        data_service = get_data_service()
        df = data_service.search_patients(q, limit)
        
        if df.empty:
            return {"patients": [], "total": 0}
        
        return {
            "patients": sanitize_for_json(df.to_dict(orient="records")),
            "total": len(df)
        }
    except Exception as e:
        api_logger.error(f"Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dqi")
async def get_patient_dqi(
    study_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Get patient DQI scores with study filter."""
    if study_id and (study_id == "[object Object]" or "{" in study_id):
        study_id = None
        
    try:
        data_service = get_data_service()
        df = data_service.get_patient_dqi(study_id=study_id)
        
        if df.empty:
            return {"data": [], "total": 0}
            
        return {
            "data": sanitize_for_json(df.to_dict(orient="records")),
            "total": len(df)
        }
    except Exception as e:
        api_logger.error(f"Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/clean-status")
async def get_patient_clean_status(
    study_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Get patient clean status tiers with study filter."""
    if study_id and (study_id == "[object Object]" or "{" in study_id):
        study_id = None
        
    try:
        data_service = get_data_service()
        df = data_service.get_patient_clean_status(study_id=study_id)
        
        if df.empty:
            return {"data": [], "total": 0}
            
        return {
            "data": sanitize_for_json(df.to_dict(orient="records")),
            "total": len(df)
        }
    except Exception as e:
        api_logger.error(f"Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dblock-status")
async def get_patient_dblock_status(
    study_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Get patient DB lock readiness with study filter."""
    if study_id and (study_id == "[object Object]" or "{" in study_id):
        study_id = None
        
    try:
        data_service = get_data_service()
        df = data_service.get_patient_dblock_status(study_id=study_id)
        
        if df.empty:
            return {"data": [], "total": 0}
        
        # Calculate summary stats
        total = len(df)
        ready = int(df["is_db_lock_ready"].sum()) if "is_db_lock_ready" in df.columns else 0
        
        return {
            "data": sanitize_for_json(df.to_dict(orient="records")),
            "total": total,
            "ready_count": ready,
            "ready_rate": round(ready / total * 100, 2) if total > 0 else 0
        }
    except Exception as e:
        api_logger.error(f"Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/issues")
async def get_patient_issues(
    current_user: dict = Depends(get_current_user)
):
    """Get patient issues summary."""
    try:
        data_service = get_data_service()
        df = data_service.get_patient_issues()
        
        if df.empty:
            return {"data": [], "total": 0}
        
        # Calculate summary
        with_issues = int(df["has_issues"].sum()) if "has_issues" in df.columns else 0
        
        return {
            "data": sanitize_for_json(df.to_dict(orient="records")),
            "total": len(df),
            "with_issues": with_issues
        }
    except Exception as e:
        api_logger.error(f"Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{patient_key}/risk-explanation")
async def get_patient_risk_explanation(
    patient_key: str,
    current_user: dict = Depends(get_current_user)
):
    """Get SHAP-powered ML explanation for a patient's risk level."""
    try:
        import logging
        api_logger = logging.getLogger("uvicorn.error")
        api_logger.info(f"ML Request: SHAP for patient {patient_key}")
        
        data_service = get_data_service()
        patient = data_service.get_patient(patient_key)
        
        if not patient:
            api_logger.warning(f"Patient not found for SHAP: {patient_key}")
            # Try to fetch patient by loosely matching key if not found
            df = data_service.get_patients(limit=100)
            if not df.empty:
                match = df[df['patient_key'].str.contains(patient_key.split('/')[-1])]
                if not match.empty:
                    patient = match.to_dict('records')[0]
                    api_logger.info(f"Fuzzy match found for SHAP: {patient['patient_key']}")
            
            if not patient:
                raise HTTPException(status_code=404, detail="Patient not found for SHAP analysis")
            
        try:
            from src.ml.inference import MLInferenceCore
            ml_core = MLInferenceCore()
        except ImportError:
            api_logger.error("Could not import src.ml.inference. Using mock ML Core.")
            class MockMLCore:
                def get_risk_explanation(self, key, features):
                    return {"patient_key": key, "risk_level": "Low", "risk_score": 0.1, "feature_impacts": [], "model_version": "mock-v1"}
            ml_core = MockMLCore()
        
        # Extract features for ML model
        features = {
            "dqi_score": float(patient.get("dqi_score", 100)),
            "open_issues_count": int(patient.get("open_issues_count", 0)),
            "missing_signatures": int(patient.get("missing_signatures", 0)),
            "coding_pending": int(patient.get("coding_pending", 0))
        }
        
        explanation = ml_core.get_risk_explanation(patient['patient_key'], features)
        return sanitize_for_json(explanation)
        
    except Exception as e:
        import logging
        err_logger = logging.getLogger("uvicorn.error")
        err_logger.error(f"SHAP API Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{patient_key}")
async def get_patient(
    patient_key: str,
    current_user: dict = Depends(get_current_user)
):
    """Get single patient details."""
    try:
        data_service = get_data_service()
        patient = data_service.get_patient(patient_key)
        
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")
        
        return sanitize_for_json(patient)
    except HTTPException:
        raise
    except Exception as e:
        api_logger.error(f"Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
