"""
SANCHALAK AI - Parquet to PostgreSQL Migration Utility
===========================================================
One-time migration script to load existing parquet files into PostgreSQL.

Usage:
    python -m src.database.parquet_migrator --migrate-all
    python -m src.database.parquet_migrator --validate
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import logging
from sqlalchemy.orm import Session

from src.database.connection import get_db_manager
from src.database.models import (
    Patient, ClinicalSite, Study, ProjectIssue, Query,
    Visit, LabResult, AdverseEvent
)
from src.database.extended_models import (
    AnomalyDetection, PatternMatch
)

logger = logging.getLogger(__name__)


class ParquetMigrator:
    """Migrate data from parquet files to PostgreSQL."""
    
    def __init__(self):
        self.db_manager = get_db_manager()
        self.data_dir = Path("data")
        self.processed_dir = self.data_dir / "processed"
        self.analytics_dir = self.processed_dir / "analytics"
        self.upr_dir = self.processed_dir / "upr"
        
        self.migration_report = {
            "started_at": datetime.utcnow(),
            "files_processed": [],
            "records_migrated": {},
            "errors": []
        }
    
    def migrate_all(self) -> Dict:
        """Run complete migration."""
        logger.info("Starting complete parquet to PostgreSQL migration...")
        
        try:
            # Phase 1: Core UPR data (if exists)
            self._migrate_upr()
            
            # Phase 2: Analytics outputs
            self._migrate_analytics()
            
            # Phase 3: ML outputs
            self._migrate_ml_outputs()
            
            self.migration_report["completed_at"] = datetime.utcnow()
            self.migration_report["status"] = "success"
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            self.migration_report["status"] = "failed"
            self.migration_report["error"] = str(e)
        
        return self.migration_report
    
    def _migrate_upr(self):
        """Migrate Unified Patient Record if parquet exists."""
        upr_path = self.upr_dir / "unified_patient_record.parquet"
        
        if not upr_path.exists():
            logger.info("No UPR parquet file found - skipping")
            return
        
        logger.info(f"Migrating UPR from {upr_path}...")
        
        try:
            df = pd.read_parquet(upr_path)
            logger.info(f"Loaded {len(df)} patient records from parquet")
            
            session = self.db_manager.get_session()
            
            # Map parquet columns to Patient model
            migrated = 0
            for _, row in df.iterrows():
                try:
                    # Check if patient already exists
                    existing = session.query(Patient).filter_by(
                        patient_key=row.get('patient_key')
                    ).first()
                    
                    if existing:
                        # Update existing
                        self._update_patient_from_row(existing, row)
                    else:
                        # Create new
                        patient = self._create_patient_from_row(row, session)
                        session.add(patient)
                    
                    migrated += 1
                    
                    if migrated % 1000 == 0:
                        session.commit()
                        logger.info(f"Migrated {migrated} patients...")
                
                except Exception as e:
                    logger.error(f"Error migrating patient {row.get('patient_key')}: {e}")
                    self.migration_report["errors"].append({
                        "file": "upr",
                        "record": row.get('patient_key'),
                        "error": str(e)
                    })
            
            session.commit()
            session.close()
            
            self.migration_report["records_migrated"]["patients"] = migrated
            self.migration_report["files_processed"].append(str(upr_path))
            logger.info(f"Successfully migrated {migrated} patients")
            
        except Exception as e:
            logger.error(f"Failed to migrate UPR: {e}")
            raise
    
    def _create_patient_from_row(self, row: pd.Series, session: Session) -> Patient:
        """Create Patient object from parquet row."""
        
        # Ensure site and study exist (create dummy if needed)
        site_id = row.get('site_id', row.get('site', 'UNKNOWN'))
        study_id = row.get('study_id', row.get('study', 'STUDY-001'))
        
        # Check/create site
        site = session.query(ClinicalSite).filter_by(site_id=site_id).first()
        if not site:
            site = ClinicalSite(
                site_id=site_id,
                name=f"Site {site_id}",
                country=row.get('country', 'Unknown'),
                region=row.get('region', 'North America')
            )
            session.add(site)
        
        # Check/create study
        study = session.query(Study).filter_by(study_id=study_id).first()
        if not study:
            study = Study(
                study_id=study_id,
                name=f"Study {study_id}",
                protocol_number=f"PROTO-{study_id}"
            )
            session.add(study)
        
        # Create patient
        enrollment_val = row.get('enrollment_date')
        age_val = row.get('age_at_enrollment')
        risk_score_val = row.get('risk_score')
        dqi_score_val = row.get('dqi_score')
        open_issues_val = row.get('open_issues_count')
        open_queries_val = row.get('open_queries_count')

        patient = Patient(
            patient_key=str(row.get('patient_key', '')),
            study_id=study_id,
            site_id=site_id,
            status=str(row.get('status', 'active')),
            enrollment_date=pd.to_datetime(enrollment_val) if enrollment_val is not None and pd.notna(enrollment_val) else None,
            age_at_enrollment=int(age_val) if age_val is not None and pd.notna(age_val) else None,
            gender=str(row.get('gender', '')) if row.get('gender') is not None and pd.notna(row.get('gender')) else None,
            clean_status_tier=str(row.get('clean_status_tier', 'tier_0')),
            is_db_lock_ready=bool(row.get('is_db_lock_ready', False)),
            risk_level=str(row.get('risk_level', 'low')),
            risk_score=float(risk_score_val) if risk_score_val is not None and pd.notna(risk_score_val) else 0.0,
            dqi_score=float(dqi_score_val) if dqi_score_val is not None and pd.notna(dqi_score_val) else 100.0,
            has_issues=bool(row.get('has_issues', False)),
            open_issues_count=int(open_issues_val) if open_issues_val is not None and pd.notna(open_issues_val) else 0,
            open_queries_count=int(open_queries_val) if open_queries_val is not None and pd.notna(open_queries_val) else 0,
        )

        
        return patient
    
    def _update_patient_from_row(self, patient: Patient, row: pd.Series):
        """Update existing patient with parquet data."""
        # Update key metrics
        if pd.notna(row.get('dqi_score')):
            patient.dqi_score = float(row.get('dqi_score'))
        if pd.notna(row.get('risk_score')):
            patient.risk_score = float(row.get('risk_score'))
        if pd.notna(row.get('clean_status_tier')):
            patient.clean_status_tier = row.get('clean_status_tier')
        if pd.notna(row.get('is_db_lock_ready')):
            patient.is_db_lock_ready = bool(row.get('is_db_lock_ready'))
    
    def _migrate_analytics(self):
        """Migrate analytics parquet files."""
        analytics_files = {
            "patient_issues.parquet": self._migrate_issues,
            "patient_anomalies.parquet": self._migrate_anomalies,
        }
        
        for filename, migrate_func in analytics_files.items():
            filepath = self.analytics_dir / filename
            if filepath.exists():
                try:
                    migrate_func(filepath)
                except Exception as e:
                    logger.error(f"Failed to migrate {filename}: {e}")
                    self.migration_report["errors"].append({
                        "file": filename,
                        "error": str(e)
                    })
    
    def _migrate_issues(self, filepath: Path):
        """Migrate patient issues from parquet."""
        logger.info(f"Migrating issues from {filepath}...")
        
        df = pd.read_parquet(filepath)
        session = self.db_manager.get_session()
        
        migrated = 0
        for _, row in df.iterrows():
            try:
                # Check if issue already exists
                existing = session.query(ProjectIssue).filter_by(
                    issue_id=row.get('issue_id')
                ).first()
                
                if not existing:
                    issue = ProjectIssue(
                        issue_id=row.get('issue_id', None),
                        patient_key=row.get('patient_key'),
                        site_id=row.get('site_id'),
                        category=row.get('category', 'data_quality'),
                        issue_type=row.get('issue_type', 'unknown'),
                        description=row.get('description', 'Migrated issue'),
                        priority=row.get('priority', 'medium'),
                        status=row.get('status', 'open'),
                    )
                    session.add(issue)
                    migrated += 1
            
            except Exception as e:
                logger.error(f"Error migrating issue: {e}")
        
        session.commit()
        session.close()
        
        self.migration_report["records_migrated"]["issues"] = migrated
        self.migration_report["files_processed"].append(str(filepath))
        logger.info(f"Migrated {migrated} issues")
    
    def _migrate_anomalies(self, filepath: Path):
        """Migrate anomaly detections from parquet."""
        logger.info(f"Migrating anomalies from {filepath}...")
        
        df = pd.read_parquet(filepath)
        session = self.db_manager.get_session()
        
        migrated = 0
        for _, row in df.iterrows():
            try:
                anomaly = AnomalyDetection(
                    entity_type='patient',
                    entity_id=row.get('patient_key'),
                    anomaly_type=row.get('anomaly_type', 'unknown'),
                    anomaly_score=float(row.get('anomaly_score', 0.0)),
                    severity=row.get('severity', 'low'),
                    description=row.get('description', 'Migrated anomaly'),
                )
                session.add(anomaly)
                migrated += 1
            
            except Exception as e:
                logger.error(f"Error migrating anomaly: {e}")
        
        session.commit()
        session.close()
        
        self.migration_report["records_migrated"]["anomalies"] = migrated
        self.migration_report["files_processed"].append(str(filepath))
        logger.info(f"Migrated {migrated} anomalies")
    
    def _migrate_ml_outputs(self):
        """Migrate ML model outputs."""
        # Pattern matches, recommendations, etc.
        logger.info("ML outputs migration - skipping (will be regenerated)")
    
    def validate_migration(self) -> Dict:
        """Validate migrated data."""
        logger.info("Validating migration...")
        
        session = self.db_manager.get_session()
        
        validation = {
            "patients_count": session.query(Patient).count(),
            "sites_count": session.query(ClinicalSite).count(),
            "studies_count": session.query(Study).count(),
            "issues_count": session.query(ProjectIssue).count(),
            "anomalies_count": session.query(AnomalyDetection).count(),
        }
        
        session.close()
        
        logger.info(f"Validation results: {validation}")
        return validation


def main():
    """Run migration from command line."""
    import argparse
    from src.database.pg_data_service import get_data_service
    from src.database.pg_writer import get_pg_writer


    
    parser = argparse.ArgumentParser(description="Migrate parquet files to PostgreSQL")
    parser.add_argument("--migrate-all", action="store_true", help="Run full migration")
    parser.add_argument("--validate", action="store_true", help="Validate migration")
    
    args = parser.parse_args()
    
    migrator = ParquetMigrator()
    
    if args.migrate_all:
        report = migrator.migrate_all()
        print("\n=== MIGRATION REPORT ===")
        print(f"Status: {report['status']}")
        print(f"Files processed: {len(report['files_processed'])}")
        print(f"Records migrated: {report['records_migrated']}")
        if report['errors']:
            print(f"Errors: {len(report['errors'])}")
    
    if args.validate:
        validation = migrator.validate_migration()
        print("\n=== VALIDATION RESULTS ===")
        for key, value in validation.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
