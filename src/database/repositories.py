"""
SANCHALAK AI - Data Repositories
=====================================
Data access layer with CRUD operations for all entities.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_

from .models import Patient, ClinicalSite as Site, Study, ProjectIssue as Issue, ResolutionAction as Resolution, AuditLog
from .connection import get_db_manager

logger = logging.getLogger(__name__)


class BaseRepository:
    """Base repository with common CRUD operations."""
    
    def __init__(self, session: Optional[Session] = None):
        """
        Initialize repository.
        
        Args:
            session: SQLAlchemy session (optional, will create if not provided)
        """
        self._session = session
        self._owns_session = session is None
    
    @property
    def session(self) -> Session:
        """Get or create session."""
        if self._session is None:
            self._session = get_db_manager().get_session()
        return self._session
    
    def commit(self) -> None:
        """Commit current transaction."""
        self.session.commit()
    
    def rollback(self) -> None:
        """Rollback current transaction."""
        self.session.rollback()
    
    def close(self) -> None:
        """Close session if we own it."""
        if self._owns_session and self._session:
            self._session.close()


class PatientRepository(BaseRepository):
    """Repository for Patient operations."""
    
    def get_by_key(self, patient_key: str) -> Optional[Patient]:
        """Get patient by primary key."""
        return self.session.query(Patient).filter(Patient.patient_key == patient_key).first()
    
    def get_all(self, limit: int = 1000, offset: int = 0) -> List[Patient]:
        """Get all patients with pagination."""
        return self.session.query(Patient).offset(offset).limit(limit).all()
    
    def get_by_study(self, study_id: str) -> List[Patient]:
        """Get all patients in a study."""
        return self.session.query(Patient).filter(Patient.study_id == study_id).all()
    
    def get_by_site(self, site_id: str) -> List[Patient]:
        """Get all patients at a site."""
        return self.session.query(Patient).filter(Patient.site_id == site_id).all()
    
    def get_high_priority(self, limit: int = 100) -> List[Patient]:
        """Get high priority patients."""
        return self.session.query(Patient).filter(
            Patient.priority_tier.in_(['critical', 'high'])
        ).order_by(Patient.action_priority_rank).limit(limit).all()
    
    def get_not_clean(self) -> List[Patient]:
        """Get patients not yet clean."""
        return self.session.query(Patient).filter(
            Patient.tier1_clean == False
        ).all()
    
    def get_dblock_ready(self) -> List[Patient]:
        """Get patients ready for DB lock."""
        return self.session.query(Patient).filter(
            Patient.is_db_lock_ready == True
        ).all()
    
    def count(self) -> int:
        """Get total patient count."""
        return self.session.query(func.count(Patient.patient_key)).scalar()
    
    def count_by_priority(self) -> Dict[str, int]:
        """Get patient counts by priority tier."""
        results = self.session.query(
            Patient.priority_tier,
            func.count(Patient.patient_key)
        ).group_by(Patient.priority_tier).all()
        return {tier: count for tier, count in results}
    
    def get_dqi_stats(self) -> Dict[str, float]:
        """Get DQI statistics."""
        stats = self.session.query(
            func.avg(Patient.dqi_score).label('mean'),
            func.min(Patient.dqi_score).label('min'),
            func.max(Patient.dqi_score).label('max'),
            func.count(Patient.patient_key).label('count')
        ).first()
        return {
            'mean': stats.mean,
            'min': stats.min,
            'max': stats.max,
            'count': stats.count
        }
    
    def create(self, patient: Patient) -> Patient:
        """Create a new patient."""
        self.session.add(patient)
        self.session.flush()
        return patient
    
    def bulk_create(self, patients: List[Patient]) -> int:
        """Bulk create patients."""
        self.session.bulk_save_objects(patients)
        self.session.flush()
        return len(patients)
    
    def update(self, patient_key: str, data: Dict[str, Any]) -> Optional[Patient]:
        """Update patient by key."""
        patient = self.get_by_key(patient_key)
        if patient:
            for key, value in data.items():
                if hasattr(patient, key):
                    setattr(patient, key, value)
            patient.updated_at = datetime.utcnow()
            self.session.flush()
        return patient
    
    def delete(self, patient_key: str) -> bool:
        """Delete patient by key."""
        patient = self.get_by_key(patient_key)
        if patient:
            self.session.delete(patient)
            self.session.flush()
            return True
        return False


class SiteRepository(BaseRepository):
    """Repository for Site operations."""
    
    def get_by_id(self, site_id: str) -> Optional[Site]:
        """Get site by ID."""
        return self.session.query(Site).filter(Site.site_id == site_id).first()
    
    def get_all(self) -> List[Site]:
        """Get all sites."""
        return self.session.query(Site).all()
    
    def get_by_study(self, study_id: str) -> List[Site]:
        """Get all sites in a study."""
        return self.session.query(Site).filter(Site.study_id == study_id).all()
    
    def get_by_country(self, country: str) -> List[Site]:
        """Get all sites in a country."""
        return self.session.query(Site).filter(Site.country == country).all()
    
    def count(self) -> int:
        """Get total site count."""
        return self.session.query(func.count(Site.site_id)).scalar()
    
    def create(self, site: Site) -> Site:
        """Create a new site."""
        self.session.add(site)
        self.session.flush()
        return site


class StudyRepository(BaseRepository):
    """Repository for Study operations."""
    
    def get_by_id(self, study_id: str) -> Optional[Study]:
        """Get study by ID."""
        return self.session.query(Study).filter(Study.study_id == study_id).first()
    
    def get_all(self) -> List[Study]:
        """Get all studies."""
        return self.session.query(Study).all()
    
    def get_active(self) -> List[Study]:
        """Get active studies."""
        return self.session.query(Study).filter(Study.status == 'active').all()
    
    def count(self) -> int:
        """Get total study count."""
        return self.session.query(func.count(Study.study_id)).scalar()
    
    def create(self, study: Study) -> Study:
        """Create a new study."""
        self.session.add(study)
        self.session.flush()
        return study


class IssueRepository(BaseRepository):
    """Repository for Issue operations."""
    
    def get_by_id(self, issue_id: int) -> Optional[Issue]:
        """Get issue by ID."""
        return self.session.query(Issue).filter(Issue.issue_id == issue_id).first()
    
    def get_by_patient(self, patient_key: str) -> List[Issue]:
        """Get all issues for a patient."""
        return self.session.query(Issue).filter(Issue.patient_key == patient_key).all()
    
    def get_open(self) -> List[Issue]:
        """Get all open issues."""
        return self.session.query(Issue).filter(Issue.status == 'open').all()
    
    def get_critical(self) -> List[Issue]:
        """Get critical issues."""
        return self.session.query(Issue).filter(Issue.severity == 'critical').all()
    
    def count_by_type(self) -> Dict[str, int]:
        """Get issue counts by type."""
        results = self.session.query(
            Issue.issue_type,
            func.count(Issue.issue_id)
        ).group_by(Issue.issue_type).all()
        return {issue_type: count for issue_type, count in results}
    
    def create(self, issue: Issue) -> Issue:
        """Create a new issue."""
        self.session.add(issue)
        self.session.flush()
        return issue


class AuditLogRepository(BaseRepository):
    """Repository for AuditLog operations."""
    
    def log_change(
        self,
        table_name: str,
        record_id: str,
        action: str,
        old_values: Optional[Dict] = None,
        new_values: Optional[Dict] = None,
        user_id: Optional[str] = None,
        reason: Optional[str] = None
    ) -> AuditLog:
        """
        Log a data change.
        
        Args:
            table_name: Name of the table
            record_id: Primary key of the record
            action: INSERT, UPDATE, or DELETE
            old_values: Previous values (for UPDATE/DELETE)
            new_values: New values (for INSERT/UPDATE)
            user_id: User making the change
            reason: Reason for the change
        """
        log = AuditLog(
            table_name=table_name,
            record_id=record_id,
            action=action,
            old_values=old_values,
            new_values=new_values,
            user_id=user_id,
            reason=reason,
            timestamp=datetime.utcnow()
        )
        self.session.add(log)
        self.session.flush()
        return log
    
    def get_by_record(self, table_name: str, record_id: str) -> List[AuditLog]:
        """Get audit history for a record."""
        return self.session.query(AuditLog).filter(
            and_(AuditLog.table_name == table_name, AuditLog.record_id == record_id)
        ).order_by(AuditLog.timestamp.desc()).all()
    
    def get_recent(self, limit: int = 100) -> List[AuditLog]:
        """Get recent audit logs."""
        return self.session.query(AuditLog).order_by(
            AuditLog.timestamp.desc()
        ).limit(limit).all()
