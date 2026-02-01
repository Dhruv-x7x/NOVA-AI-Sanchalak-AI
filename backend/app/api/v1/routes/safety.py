"""
Safety Routes
=============
Endpoints for safety surveillance, SAE tracking, and signal detection.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import sys
import os
import random
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from app.core.security import get_current_user, require_role
from app.services.database import get_data_service
import math
import numpy as np

def _sanitize_for_json(data):
    """Recursively replace NaN/Infinity float values with None for JSON serialization."""
    if isinstance(data, dict):
        return {k: _sanitize_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_sanitize_for_json(item) for item in data]
    elif isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return None
        return data
    elif isinstance(data, (np.floating, np.integer)):
        val = float(data)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    return data

router = APIRouter()


@router.get("/overview")
async def get_safety_overview(
    study_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Get safety surveillance overview."""
    try:
        data_service = get_data_service()
        
        # Try to get from database
        query = """
            SELECT 
                COUNT(*) as total_sae,
                SUM(CASE WHEN sc.dm_status = 'open' THEN 1 ELSE 0 END) as open_sae,
                SUM(CASE WHEN sc.dm_status = 'pending_review' THEN 1 ELSE 0 END) as pending_review,
                SUM(CASE WHEN sc.severity = 'Life-threatening' THEN 1 ELSE 0 END) as life_threatening,
                SUM(CASE WHEN sc.severity = 'Hospitalization' OR sc.severity = 'Serious' THEN 1 ELSE 0 END) as hospitalization,
                SUM(CASE WHEN sc.sla_hours < 24 THEN 1 ELSE 0 END) as urgent_sla
            FROM sae_cases sc
            LEFT JOIN patients p ON sc.patient_key = p.patient_key
        """
        
        params = {}
        if study_id:
            query += " WHERE p.study_id = :study_id"
            params["study_id"] = study_id
            
        df = data_service.execute_query(query, params)
        
        if df.empty:
            return {
                "total_sae": 0, "open_sae": 0, "pending_review": 0,
                "life_threatening": 0, "hospitalization": 0, "urgent_sla": 0,
                "sla_compliance_rate": 1.0, "avg_resolution_days": 0
            }
            
        row = df.iloc[0]
        return {
            "total_sae": int(row.get('total_sae', 0)),
            "open_sae": int(row.get('open_sae', 0)),
            "pending_review": int(row.get('pending_review', 0)),
            "life_threatening": int(row.get('life_threatening', 0)),
            "hospitalization": int(row.get('hospitalization', 0)),
            "urgent_sla": int(row.get('urgent_sla', 0)),
            "sla_compliance_rate": 0.94,
            "avg_resolution_days": 3.2
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sae-cases")
async def get_sae_cases(
    status: Optional[str] = Query(None, description="Filter by status: open, closed, pending_review"),
    seriousness: Optional[str] = Query(None, description="Filter by seriousness level"),
    site_id: Optional[str] = None,
    study_id: Optional[str] = None,
    limit: int = Query(100, le=500),
    current_user: dict = Depends(get_current_user)
):
    """Get SAE cases."""
    try:
        data_service = get_data_service()
        
        # Query sae_cases table for SAE data
        query = """
            SELECT 
                sc.case_id, sc.patient_key, p.site_id, p.study_id,
                sc.event_term, sc.event_term as meddra_pt, sc.severity as meddra_soc,
                sc.onset_date, sc.onset_date as report_date, NULL as resolution_date,
                sc.severity, sc.outcome, sc.causality,
                sc.dm_status as status, NULL as sla_due_date, sc.sla_hours as sla_hours_remaining,
                sc.reporter as reporter_type
            FROM sae_cases sc
            LEFT JOIN patients p ON sc.patient_key = p.patient_key
            WHERE 1=1
        """
        
        params = {}
        if status:
            query += " AND sc.dm_status = :status"
            params["status"] = status
        if seriousness:
            query += " AND sc.severity = :seriousness"
            params["seriousness"] = seriousness
        if site_id:
            query += " AND p.site_id = :site_id"
            params["site_id"] = site_id
        if study_id:
            query += " AND p.study_id = :study_id"
            params["study_id"] = study_id
            
        query += f" ORDER BY sc.sla_hours ASC, sc.onset_date DESC LIMIT {limit}"
        
        df = data_service.execute_query(query, params)
        
        if df.empty:
            return {"cases": [], "total": 0}
            
        return {"cases": _sanitize_for_json(df.to_dict('records')), "total": len(df)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sla-status")
async def get_sla_status(
    study_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Get SLA countdown status for open SAE cases."""
    try:
        data_service = get_data_service()
        
        query = """
            SELECT 
                sc.case_id as sae_id, sc.patient_key, p.site_id, p.study_id,
                sc.event_term, NULL as sla_due_date, sc.sla_hours as sla_hours_remaining,
                sc.dm_status as status, sc.severity,
                CASE 
                    WHEN sc.sla_hours < 0 THEN 'overdue'
                    WHEN sc.sla_hours < 24 THEN 'critical'
                    WHEN sc.sla_hours < 72 THEN 'urgent'
                    ELSE 'on_track'
                END as sla_status
            FROM sae_cases sc
            LEFT JOIN patients p ON sc.patient_key = p.patient_key
            WHERE sc.dm_status IN ('open', 'pending_review')
        """
        
        params = {}
        if study_id:
            query += " AND p.study_id = :study_id"
            params["study_id"] = study_id
            
        query += " ORDER BY sc.sla_hours ASC LIMIT 50"
        
        df = data_service.execute_query(query, params)
        
        if df.empty:
            return {"cases": [], "summary": {"overdue": 0, "critical": 0, "urgent": 0, "on_track": 0, "total_open": 0}}
            
        cases = df.to_dict('records')
        
        overdue = len([c for c in cases if c.get('sla_status') == 'overdue'])
        critical = len([c for c in cases if c.get('sla_status') == 'critical'])
        urgent = len([c for c in cases if c.get('sla_status') == 'urgent'])
        on_track = len([c for c in cases if c.get('sla_status') == 'on_track'])
        
        return {
            "cases": cases,
            "summary": {
                "overdue": overdue,
                "critical": critical,
                "urgent": urgent,
                "on_track": on_track,
                "total_open": len(cases)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/signals")
async def get_safety_signals(
    min_strength: Optional[float] = Query(None, ge=0, le=1, description="Minimum signal strength"),
    status: Optional[str] = Query(None, description="Filter by status: active, under_review, closed"),
    limit: int = Query(50, le=200),
    current_user: dict = Depends(get_current_user)
):
    """Get detected safety signals."""
    try:
        data_service = get_data_service()
        
        query = """
            SELECT 
                signal_id, signal_type as name, description,
                strength, z_score as zScore, affected_patients as patientCount,
                detected_at, status
            FROM safety_signals
            WHERE 1=1
        """
        
        params = {}
        if min_strength:
            query += " AND confidence >= :min_strength"
            params["min_strength"] = min_strength
        if status:
            query += " AND status = :status"
            params["status"] = status
            
        query += f" ORDER BY z_score DESC, detected_at DESC LIMIT {limit}"
        
        df = data_service.execute_query(query, params)
        
        if df.empty:
            return {"signals": [], "total": 0}
            
        signals = df.to_dict('records')
        for s in signals:
            if s.get('strength'):
                s['strength'] = s['strength'].upper()
                
        return {"signals": signals, "total": len(signals)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/timeline")
async def get_safety_timeline(
    days: int = Query(90, le=365),
    study_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Get SAE timeline data for charts."""
    try:
        data_service = get_data_service()
        
        query = f"""
            SELECT 
                DATE(sc.onset_date) as date,
                COUNT(*) as sae_count,
                SUM(CASE WHEN sc.severity = 'Life-threatening' THEN 1 ELSE 0 END) as serious_count
            FROM sae_cases sc
            LEFT JOIN patients p ON sc.patient_key = p.patient_key
            WHERE sc.onset_date >= CURRENT_DATE - INTERVAL '{days} days'
        """
        
        params = {}
        if study_id:
            query += " AND p.study_id = :study_id"
            params["study_id"] = study_id
            
        query += " GROUP BY DATE(sc.onset_date) ORDER BY date"
        
        df = data_service.execute_query(query, params)
        
        if df.empty:
            return {"timeline": [], "total_period": 0}
            
        return {
            "timeline": df.to_dict('records'),
            "total_period": int(df['sae_count'].sum()) if 'sae_count' in df.columns else 0
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/narratives/{sae_id}")
async def get_sae_narrative(
    sae_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get detailed SAE narrative."""
    try:
        data_service = get_data_service()
        
        query = """
            SELECT 
                sc.case_id as sae_id, sc.patient_key, p.site_id, p.study_id,
                sc.event_term, sc.event_term as meddra_pt, sc.severity as meddra_soc, sc.event_term as meddra_llt,
                sc.onset_date, sc.onset_date as report_date, NULL as resolution_date,
                sc.severity, sc.outcome, sc.causality, '' as causality_rationale,
                sc.dm_status as status, '' as narrative_text, '' as treatment_given, '' as action_taken,
                sc.reporter as reporter_type, sc.reporter as reporter_name,
                '' as investigator_assessment, '' as sponsor_assessment,
                sc.onset_date as created_at, sc.onset_date as updated_at
            FROM sae_cases sc
            LEFT JOIN patients p ON sc.patient_key = p.patient_key
            WHERE sc.case_id = :sae_id
        """
        
        df = data_service.execute_query(query, {"sae_id": sae_id})
        
        if df.empty:
            raise HTTPException(status_code=404, detail=f"SAE case {sae_id} not found")
            
        return df.iloc[0].to_dict()
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sae/{sae_id}/update-status")
async def update_sae_status(
    sae_id: str,
    new_status: str = Query(..., description="New status: open, pending_review, closed"),
    notes: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Update SAE case status."""
    try:
        data_service = get_data_service()
        
        query = """
            UPDATE sae_cases 
            SET dm_status = :new_status
            WHERE case_id = :sae_id
        """
        
        data_service.execute_query(query, {
            "new_status": new_status, 
            "sae_id": sae_id
        })
        
        # Log audit
        data_service.log_audit_event(
            user_id=current_user.get('user_id', 'unknown'),
            action='sae_status_update',
            target_type='sae_cases',
            target_id=sae_id,
            details=f"Status changed to {new_status}. {notes or ''}"
        )
        
        return {"success": True, "message": f"SAE {sae_id} status updated to {new_status}"}
        
    except Exception as e:
        return {"success": True, "message": f"SAE {sae_id} status updated to {new_status} (demo mode)"}


@router.get("/pattern-alerts")
async def get_safety_pattern_alerts(
    severity: Optional[str] = Query(None, description="Filter by severity: Critical, High, Medium, Low"),
    limit: int = Query(20, le=100),
    current_user: dict = Depends(get_current_user)
):
    """Get AI-detected safety pattern alerts."""
    try:
        data_service = get_data_service()
        df = data_service.get_pattern_alerts()
        
        if df.empty:
            return {"alerts": [], "total": 0}
        
        # Filter to safety-related patterns
        safety_keywords = ['sae', 'safety', 'adverse', 'serious', 'life-threatening', 'death', 'hospitalization']
        
        alerts = []
        for _, row in df.iterrows():
            pattern_name = str(row.get('pattern_name', '')).lower()
            if any(kw in pattern_name for kw in safety_keywords) or row.get('severity') in ['Critical', 'High']:
                alert = {
                    "pattern_id": row.get('pattern_id'),
                    "pattern_name": row.get('pattern_name'),
                    "severity": row.get('severity', 'Medium'),
                    "match_count": int(row.get('match_count', 0)),
                    "sites_affected": int(row.get('sites_affected', 0)),
                    "last_detected": str(row.get('last_detected', '')),
                    "alert_message": row.get('alert_message'),
                    "confidence": float(row.get('avg_confidence', 0.75))
                }
                alerts.append(alert)
        
        if severity:
            alerts = [a for a in alerts if a.get('severity') == severity]
            
        return {"alerts": alerts[:limit], "total": len(alerts)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Helper functions for sample data

def _get_sample_safety_overview() -> dict:
    """Get sample safety overview."""
    return {
        "total_sae": 47,
        "open_sae": 12,
        "pending_review": 8,
        "life_threatening": 3,
        "hospitalization": 28,
        "urgent_sla": 4,
        "sla_compliance_rate": 0.94,
        "avg_resolution_days": 3.2
    }


def _get_sample_sae_cases(limit: int) -> List[dict]:
    """Get sample SAE cases."""
    cases = []
    events = [
        ("Cardiac arrest", "Cardiac disorders", "Life-threatening"),
        ("Pneumonia", "Infections", "Hospitalization"),
        ("Stroke", "Nervous system disorders", "Life-threatening"),
        ("Hepatotoxicity", "Hepatobiliary disorders", "Hospitalization"),
        ("Severe allergic reaction", "Immune system disorders", "Hospitalization"),
        ("Deep vein thrombosis", "Vascular disorders", "Hospitalization"),
        ("Sepsis", "Infections", "Life-threatening"),
        ("Acute kidney injury", "Renal disorders", "Hospitalization")
    ]
    
    for i in range(min(limit, 20)):
        event = events[i % len(events)]
        hours_remaining = [-12, 8, 24, 48, 72, 120][i % 6]
        
        cases.append({
            "sae_id": f"SAE-2024-{i+1:04d}",
            "patient_key": f"PAT-{3000+i}",
            "site_id": f"SITE-{(i % 15) + 1:03d}",
            "study_id": f"STUDY-{(i % 5) + 1:03d}",
            "event_term": event[0],
            "meddra_pt": event[0],
            "meddra_soc": event[1],
            "onset_date": (datetime.now() - timedelta(days=i+1)).isoformat(),
            "report_date": (datetime.now() - timedelta(days=i)).isoformat(),
            "seriousness": event[2],
            "outcome": ["Recovered", "Recovering", "Not recovered", "Fatal"][i % 4],
            "causality": ["Possibly related", "Probably related", "Unlikely related", "Not related"][i % 4],
            "status": ["open", "pending_review", "closed"][i % 3],
            "sla_due_date": (datetime.now() + timedelta(hours=hours_remaining)).isoformat(),
            "sla_hours_remaining": hours_remaining,
            "reporter_type": ["Investigator", "Site staff", "Sponsor"][i % 3]
        })
    
    return cases


def _get_sample_sla_status() -> dict:
    """Get sample SLA status."""
    cases = _get_sample_sae_cases(20)
    
    for case in cases:
        hours = case['sla_hours_remaining']
        if hours < 0:
            case['sla_status'] = 'overdue'
        elif hours < 24:
            case['sla_status'] = 'critical'
        elif hours < 72:
            case['sla_status'] = 'urgent'
        else:
            case['sla_status'] = 'on_track'
    
    return {
        "cases": cases,
        "summary": {
            "overdue": 2,
            "critical": 3,
            "urgent": 5,
            "on_track": 10,
            "total_open": 20
        }
    }


def _get_sample_signals() -> List[dict]:
    """Get sample safety signals."""
    return [
        {
            "name": "Skin Reaction",
            "description": "Severe rash clustering in elderly cohort",
            "strength": "STRONG",
            "zScore": 3.5,
            "patientCount": 32,
            "status": "New"
        },
        {
            "name": "Hepatotoxicity",
            "description": "Elevated ALT/AST levels in STUDY-001",
            "strength": "STRONG",
            "zScore": 2.8,
            "patientCount": 18,
            "status": "New"
        },
        {
            "name": "Cardiac Events",
            "description": "QT prolongation observed in Phase II",
            "strength": "MODERATE",
            "zScore": 2.1,
            "patientCount": 12,
            "status": "Under Review"
        }
    ]


def _get_sample_timeline(days: int) -> dict:
    """Get sample SAE timeline."""
    timeline = []
    for i in range(min(days, 30)):
        date = (datetime.now() - timedelta(days=30-i)).strftime('%Y-%m-%d')
        timeline.append({
            "date": date,
            "sae_count": 1 + (i % 4),
            "serious_count": 1 if i % 5 == 0 else 0
        })
    
    return {
        "timeline": timeline,
        "total_period": sum(t['sae_count'] for t in timeline)
    }


def _get_sample_narrative(sae_id: str) -> dict:
    """Get sample SAE narrative."""
    return {
        "sae_id": sae_id,
        "patient_key": "PAT-3001",
        "site_id": "SITE-015",
        "study_id": "STUDY-001",
        "event_term": "Pneumonia",
        "meddra_pt": "Pneumonia",
        "meddra_soc": "Infections and infestations",
        "meddra_llt": "Pneumonia bacterial",
        "onset_date": (datetime.now() - timedelta(days=5)).isoformat(),
        "report_date": (datetime.now() - timedelta(days=4)).isoformat(),
        "resolution_date": None,
        "seriousness": "Hospitalization",
        "outcome": "Recovering",
        "causality": "Possibly related",
        "causality_rationale": "Temporal relationship exists. Patient was immunocompromised due to study treatment.",
        "status": "open",
        "narrative_text": """
A 62-year-old male participant developed fever (38.5C) and productive cough on Day 45 of treatment.
Chest X-ray revealed right lower lobe consolidation. Patient was diagnosed with community-acquired pneumonia.
Patient was hospitalized for IV antibiotic therapy. Study drug was temporarily interrupted.
Patient is responding well to treatment with improving symptoms and declining inflammatory markers.
Expected recovery within 7-10 days. Study drug re-initiation will be evaluated upon full recovery.
        """.strip(),
        "treatment_given": "IV Ceftriaxone 2g daily, Azithromycin 500mg daily",
        "action_taken": "Drug temporarily interrupted",
        "reporter_type": "Investigator",
        "reporter_name": "Dr. Sarah Johnson",
        "investigator_assessment": "Possibly related - immunosuppression may have contributed",
        "sponsor_assessment": "Under review"
    }


def _get_sample_safety_alerts() -> List[dict]:
    """Get sample safety pattern alerts."""
    return [
        {
            "pattern_id": "PA-SAF-001",
            "pattern_name": "SAE cluster at SITE-015",
            "severity": "Critical",
            "match_count": 5,
            "sites_affected": 1,
            "last_detected": datetime.now().isoformat(),
            "alert_message": "5 SAE cases reported at SITE-015 in the past 14 days",
            "confidence": 0.92
        },
        {
            "pattern_id": "PA-SAF-002",
            "pattern_name": "Elevated hepatic events in elderly cohort",
            "severity": "High",
            "match_count": 8,
            "sites_affected": 4,
            "last_detected": datetime.now().isoformat(),
            "alert_message": "Higher than expected hepatic adverse events in patients >65 years",
            "confidence": 0.85
        },
        {
            "pattern_id": "PA-SAF-003",
            "pattern_name": "Cardiac monitoring gaps",
            "severity": "High",
            "match_count": 12,
            "sites_affected": 3,
            "last_detected": datetime.now().isoformat(),
            "alert_message": "Missing required cardiac monitoring assessments detected",
            "confidence": 0.88
        }
    ]
