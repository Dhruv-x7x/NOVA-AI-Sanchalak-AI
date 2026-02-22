"""
a6on-i - Parquet to PostgreSQL Migration
===================================================
Migrate existing parquet data to PostgreSQL database.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.database.connection import get_db_manager
from src.database.models import Base, Patient, Site, Study, AuditLog

# PostgreSQL integration
from src.database.pg_data_service import get_data_service
from src.database.pg_writer import get_pg_writer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def migrate_studies(df: pd.DataFrame, session) -> int:
    """Migrate study data."""
    logger.info("Migrating studies...")
    
    # Extract unique studies
    study_col = 'study_id' if 'study_id' in df.columns else 'project_name'
    unique_studies = df[study_col].unique()
    
    count = 0
    for study_id in unique_studies:
        if pd.isna(study_id):
            continue
            
        existing = session.query(Study).filter(Study.study_id == str(study_id)).first()
        if not existing:
            study = Study(
                study_id=str(study_id),
                study_name=str(study_id),
                status='active',
            )
            session.add(study)
            count += 1
    
    session.commit()
    logger.info(f"  Created {count} studies")
    return count


def migrate_sites(df: pd.DataFrame, session) -> int:
    """Migrate site data."""
    logger.info("Migrating sites...")
    
    site_col = 'site_id' if 'site_id' in df.columns else 'site'
    study_col = 'study_id' if 'study_id' in df.columns else 'project_name'
    
    # Get unique site-study combinations
    site_data = df[[site_col, study_col]].drop_duplicates()
    
    # Track created sites to avoid duplicates
    created_sites = set()
    
    count = 0
    for _, row in site_data.iterrows():
        site_id = str(row[site_col])
        study_id = str(row[study_col])
        
        if pd.isna(row[site_col]) or pd.isna(row[study_col]):
            continue
        
        # Use composite key to check uniqueness
        composite_key = f"{study_id}_{site_id}"
        
        if composite_key in created_sites:
            continue
        
        # Check if site already exists in DB
        existing = session.query(Site).filter(Site.site_id == composite_key).first()
        if existing:
            continue
        
        # Get region if available
        region = df.loc[df[site_col] == row[site_col], 'region'].iloc[0] if 'region' in df.columns else None
        
        site = Site(
            site_id=composite_key,  # Use composite key
            study_id=study_id,
            region=str(region) if not pd.isna(region) else None,
            status='active',
        )
        session.add(site)
        created_sites.add(composite_key)
        count += 1
    
    session.commit()
    logger.info(f"  Created {count} sites")
    return count


def migrate_patients(df: pd.DataFrame, session, batch_size: int = 1000) -> int:
    """Migrate patient data in batches."""
    logger.info(f"Migrating {len(df)} patients...")
    
    # Column mapping from parquet to database
    column_mapping = {
        'patient_key': 'patient_key',
        'site_id': 'site_id',
        'study_id': 'study_id',
        'project_name': 'study_id',
        'site': 'site_id',
        'subject': 'subject',
        'subject_id': 'subject_id',
        'region': 'region',
        'country': 'country',
        'subject_status': 'subject_status',
        'dqi_score': 'dqi_score',
        'dqi_band': 'dqi_band',
        'tier1_clean': 'tier1_clean',
        'tier2_clean': 'tier2_clean',
        'is_db_lock_ready': 'is_db_lock_ready',
        'dblock_tier1_ready': 'dblock_tier1_ready',
        'total_issues': 'total_issues',
        'total_open_queries': 'total_open_queries',
        'dm_queries': 'dm_queries',
        'total_crfs': 'total_crfs',
        'risk_level': 'risk_level',
        'priority_tier': 'priority_tier',
    }
    
    # Columns that exist in Patient model
    patient_columns = [c.name for c in Patient.__table__.columns]
    
    count = 0
    for start in tqdm(range(0, len(df), batch_size), desc="Migrating patients"):
        batch = df.iloc[start:start + batch_size]
        patients = []
        
        for _, row in batch.iterrows():
            # Build patient data from row
            patient_data = {}
            
            for parquet_col, db_col in column_mapping.items():
                if parquet_col in row.index and db_col in patient_columns:
                    value = row[parquet_col]
                    if pd.isna(value):
                        value = None
                    elif isinstance(value, (bool, int, float, str)):
                        pass  # Keep as is
                    else:
                        value = str(value)
                    patient_data[db_col] = value
            
            # Ensure required fields
            if 'patient_key' not in patient_data or patient_data['patient_key'] is None:
                continue
            
            # Get study_id first
            if 'study_id' not in patient_data or patient_data['study_id'] is None:
                patient_data['study_id'] = str(row.get('project_name', 'Unknown'))
            
            # Build composite site_id to match sites table
            original_site_id = patient_data.get('site_id') or str(row.get('site', 'Unknown'))
            patient_data['site_id'] = f"{patient_data['study_id']}_{original_site_id}"
            
            # Collect remaining columns as extended attributes
            extended = {}
            for col in row.index:
                if col not in column_mapping and col not in ['_source_file', '_ingestion_ts']:
                    val = row[col]
                    if not pd.isna(val):
                        try:
                            extended[col] = float(val) if isinstance(val, (int, float)) else str(val)
                        except:
                            extended[col] = str(val)
            
            if extended:
                patient_data['extended_attributes'] = extended
            
            patients.append(Patient(**patient_data))
        
        # Bulk insert
        session.bulk_save_objects(patients)
        session.commit()
        count += len(patients)
    
    logger.info(f"  Created {count} patients")
    return count


def run_migration():
    """Run full migration from parquet to PostgreSQL."""
    print("=" * 70)
    print("a6on-i - PARQUET TO POSTGRESQL MIGRATION")
    print("=" * 70)
    print(f"Started: {datetime.now()}")
    
    # Initialize database
    db_manager = get_db_manager()
    
    # Create tables
    print("\n1. Creating database tables...")
    db_manager.create_tables(drop_existing=True)
    print("   ✅ Tables created")
    
    # Load UPR data
    upr_path = Path("data/processed/upr/unified_patient_record.parquet")
    if not upr_path.exists():
        print(f"   ❌ UPR file not found: {upr_path}")
        return
    
    print("\n2. Loading UPR data...")
    df = pd.read_parquet(upr_path)
    print(f"   ✅ Loaded {len(df)} patients with {len(df.columns)} columns")
    
    # Run migrations
    with db_manager.session() as session:
        print("\n3. Migrating data...")
        
        # Studies first (parent table)
        study_count = migrate_studies(df, session)
        
        # Sites second (references studies)
        site_count = migrate_sites(df, session)
        
        # Patients last (references both)
        patient_count = migrate_patients(df, session)
        
        # Log migration
        audit_repo_session = session
        audit_log = AuditLog(
            table_name='migration',
            record_id='parquet_to_postgres',
            action='MIGRATE',
            new_values={
                'studies': study_count,
                'sites': site_count,
                'patients': patient_count,
                'source': str(upr_path),
            },
            user_id='system',
            reason='Initial migration from parquet files',
        )
        session.add(audit_log)
    
    print("\n" + "=" * 70)
    print("MIGRATION COMPLETE")
    print("=" * 70)
    print(f"   Studies: {study_count}")
    print(f"   Sites: {site_count}")
    print(f"   Patients: {patient_count}")
    print(f"\nFinished: {datetime.now()}")


if __name__ == "__main__":
    run_migration()
