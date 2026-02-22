"""
a6on-i - PostgreSQL Data Writer
==========================================
Helper utilities to write data to PostgreSQL instead of Parquet files.
Provides drop-in replacements for parquet operations.
"""

import pandas as pd
import logging
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
from sqlalchemy.orm import Session
from sqlalchemy import text

from src.database.connection import get_db_manager
from src.database.models import (
    Patient, ClinicalSite, Study, ProjectIssue, Query,
    Visit, LabResult, AdverseEvent, ResolutionAction
)
from src.database.extended_models import (
    AnomalyDetection, PatternMatch, PipelineRun
)

logger = logging.getLogger(__name__)


class PostgreSQLWriter:
    """
    Drop-in replacement for parquet file operations.
    Writes data to PostgreSQL instead of parquet files.
    """
    
    def __init__(self):
        self.db_manager = get_db_manager()
    
    def safe_to_postgres(self, df: pd.DataFrame, table_name: str, 
                         if_exists: str = 'replace') -> Tuple[bool, str]:
        """
        Save DataFrame to PostgreSQL table.
        
        Args:
            df: DataFrame to save
            table_name: Target table name
            if_exists: 'replace', 'append', or 'fail'
        
        Returns:
            (success, message)
        """
        try:
            # If replace, use CASCADE to handle dependent views
            if if_exists == 'replace':
                with self.db_manager.engine.begin() as conn:
                    conn.execute(text(f"DROP TABLE IF EXISTS {table_name} CASCADE"))
                # After drop, we must use 'fail' or 'append' in to_sql, 
                # but 'fail' is fine since we just dropped it.
                # Actually, to_sql with if_exists='replace' will create it.
                # But since we already dropped it, 'append' will create it too if it doesn't exist.
                # Let's use 'replace' but we already cleared it with CASCADE.
            
            # Use pandas to_sql for bulk insert
            # Reduced chunksize to 500 and removed method='multi' to prevent MemoryError with 264 columns
            logger.info(f"Writing {len(df)} records to {table_name} in chunks of 500...")
            df.to_sql(
                name=table_name,
                con=self.db_manager.engine,
                if_exists=if_exists,
                index=False,
                chunksize=500
            )
            
            logger.info(f"Saved {len(df)} records to PostgreSQL table: {table_name}")
            return True, f"Saved {len(df)} records"

            
        except Exception as e:
            import traceback
            error_msg = f"PostgreSQL write failed: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            return False, error_msg

    
    def update_patients_bulk(self, df: pd.DataFrame) -> Tuple[int, int]:
        """
        Bulk update patient records from DataFrame.
        
        Returns:
            (updated_count, created_count)
        """
        session = self.db_manager.get_session()
        updated = 0
        created = 0
        
        try:
            for _, row in df.iterrows():
                patient_key = row.get('patient_key')
                
                # Check if exists
                existing = session.query(Patient).filter_by(
                    patient_key=patient_key
                ).first()
                
                if existing:
                    # Update
                    for col in df.columns:
                        if hasattr(existing, col) and col != 'patient_key':
                            setattr(existing, col, row.get(col))
                    updated += 1
                else:
                    # Create new
                    patient_data = row.to_dict()
                    patient = Patient(**{k: v for k, v in patient_data.items() 
                                       if hasattr(Patient, k)})
                    session.add(patient)
                    created += 1
                
                if (updated + created) % 1000 == 0:
                    session.commit()
            
            session.commit()
            logger.info(f"Updated {updated} patients, created {created} patients")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Bulk update failed: {e}")
            raise
        finally:
            session.close()
        
        return updated, created
    
    def write_issues(self, df: pd.DataFrame) -> int:
        """Write issues to database."""
        session = self.db_manager.get_session()
        count = 0
        
        try:
            for _, row in df.iterrows():
                issue = ProjectIssue(
                    patient_key=row.get('patient_key'),
                    site_id=row.get('site_id'),
                    category=row.get('category', 'data_quality'),
                    issue_type=row.get('issue_type', 'unknown'),
                    description=row.get('description', ''),
                    priority=row.get('priority', 'medium'),
                    status=row.get('status', 'open'),
                )
                session.add(issue)
                count += 1
                
                if count % 1000 == 0:
                    session.commit()
            
            session.commit()
            logger.info(f"Wrote {count} issues to database")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Write issues failed: {e}")
            raise
        finally:
            session.close()
        
        return count
    
    def write_anomalies(self, df: pd.DataFrame) -> int:
        """Write anomaly detections to database."""
        session = self.db_manager.get_session()
        count = 0
        
        try:
            for _, row in df.iterrows():
                anomaly = AnomalyDetection(
                    entity_type=row.get('entity_type', 'patient'),
                    entity_id=row.get('entity_id') or row.get('patient_key'),
                    anomaly_type=row.get('anomaly_type', 'unknown'),
                    anomaly_score=float(row.get('anomaly_score', 0.0)),
                    severity=row.get('severity', 'low'),
                    description=row.get('description', ''),
                )
                session.add(anomaly)
                count += 1
                
                if count % 1000 == 0:
                    session.commit()
            
            session.commit()
            logger.info(f"Wrote {count} anomalies to database")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Write anomalies failed: {e}")
            raise
        finally:
            session.close()
        
        return count


# Global instance
_pg_writer: Optional[PostgreSQLWriter] = None


def get_pg_writer() -> PostgreSQLWriter:
    """Get singleton PostgreSQL writer instance."""
    global _pg_writer
    if _pg_writer is None:
        _pg_writer = PostgreSQLWriter()
    return _pg_writer


def safe_to_postgres(df: pd.DataFrame, table_name: str, 
                     if_exists: str = 'replace') -> Tuple[bool, str]:
    """
    Drop-in replacement for safe_to_parquet().
    Saves DataFrame to PostgreSQL instead of parquet file.
    """
    writer = get_pg_writer()
    return writer.safe_to_postgres(df, table_name, if_exists)
