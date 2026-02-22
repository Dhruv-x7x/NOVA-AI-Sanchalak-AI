"""
a6on-i - Simplified Parquet to PostgreSQL Migration
==============================================================
Migrate existing parquet data to PostgreSQL database.
Simplified version without foreign key constraints.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sqlalchemy import create_engine, text
from src.database.config import get_database_url

# PostgreSQL integration
from src.database.pg_data_service import get_data_service
from src.database.pg_writer import get_pg_writer


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_simple_migration():
    """Run simplified migration from parquet to PostgreSQL."""
    print("=" * 70)
    print("a6on-i - SIMPLIFIED MIGRATION")
    print("=" * 70)
    print(f"Started: {datetime.now()}")
    
    # Create engine
    engine = create_engine(get_database_url())
    
    # Load UPR data
    upr_path = Path("data/processed/upr/unified_patient_record.parquet")
    if not upr_path.exists():
        print(f"   ❌ UPR file not found: {upr_path}")
        return
    
    print("\n1. Loading UPR data...")
    df = pd.read_parquet(upr_path)
    print(f"   ✅ Loaded {len(df)} patients with {len(df.columns)} columns")
    
    # Select key columns for patients table
    key_columns = [
        'patient_key', 'site_id', 'study_id', 'subject', 'subject_id',
        'region', 'country', 'subject_status',
        'dqi_score', 'dqi_band',
        'tier1_clean', 'tier2_clean',
        'is_db_lock_ready', 'dblock_tier1_ready',
        'total_issues', 'total_open_queries', 'dm_queries',
        'total_crfs', 'risk_level', 'priority_tier'
    ]
    
    # Only keep columns that exist
    available_columns = [c for c in key_columns if c in df.columns]
    patients_df = df[available_columns].copy()
    
    print("\n2. Creating tables...")
    with engine.connect() as conn:
        # Drop and recreate patients table (simplified)
        conn.execute(text("DROP TABLE IF EXISTS patients CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS sites CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS studies CASCADE"))
        conn.execute(text("DROP TABLE IF EXISTS audit_logs CASCADE"))
        conn.commit()
        
        # Create studies table
        conn.execute(text("""
            CREATE TABLE studies (
                study_id VARCHAR(50) PRIMARY KEY,
                study_name VARCHAR(255),
                status VARCHAR(50) DEFAULT 'active',
                created_at TIMESTAMP DEFAULT NOW()
            )
        """))
        
        # Create sites table
        conn.execute(text("""
            CREATE TABLE sites (
                site_id VARCHAR(100) PRIMARY KEY,
                study_id VARCHAR(50),
                region VARCHAR(100),
                country VARCHAR(100),
                status VARCHAR(50) DEFAULT 'active',
                created_at TIMESTAMP DEFAULT NOW()
            )
        """))
        
        # Create patients table
        conn.execute(text("""
            CREATE TABLE patients (
                patient_key VARCHAR(100) PRIMARY KEY,
                site_id VARCHAR(100),
                study_id VARCHAR(50),
                subject VARCHAR(50),
                subject_id VARCHAR(50),
                region VARCHAR(100),
                country VARCHAR(100),
                subject_status VARCHAR(50),
                dqi_score FLOAT,
                dqi_band VARCHAR(20),
                tier1_clean BOOLEAN,
                tier2_clean BOOLEAN,
                is_db_lock_ready BOOLEAN,
                dblock_tier1_ready BOOLEAN,
                total_issues FLOAT,
                total_open_queries FLOAT,
                dm_queries FLOAT,
                total_crfs FLOAT,
                risk_level VARCHAR(20),
                priority_tier VARCHAR(20),
                created_at TIMESTAMP DEFAULT NOW()
            )
        """))
        
        # Create audit_logs table
        conn.execute(text("""
            CREATE TABLE audit_logs (
                log_id SERIAL PRIMARY KEY,
                table_name VARCHAR(100),
                record_id VARCHAR(100),
                action VARCHAR(20),
                timestamp TIMESTAMP DEFAULT NOW()
            )
        """))
        conn.commit()
    print("   ✅ Tables created")
    
    print("\n3. Migrating studies...")
    studies_df = df[['study_id']].drop_duplicates()
    studies_df['study_name'] = studies_df['study_id']
    studies_df['status'] = 'active'
    studies_df.to_sql('studies', engine, if_exists='append', index=False, method='multi')
    print(f"   ✅ Created {len(studies_df)} studies")
    
    print("\n4. Migrating sites...")
    sites_df = df[['site_id', 'study_id', 'region']].drop_duplicates(subset=['site_id', 'study_id'])
    sites_df['site_id'] = sites_df['study_id'] + '_' + sites_df['site_id']  # Composite key
    sites_df['status'] = 'active'
    sites_df.to_sql('sites', engine, if_exists='append', index=False, method='multi')
    print(f"   ✅ Created {len(sites_df)} sites")
    
    print("\n5. Migrating patients...")
    # Update site_id to composite key
    patients_df['site_id'] = patients_df['study_id'] + '_' + patients_df['site_id']
    
    # Convert boolean columns
    for col in ['tier1_clean', 'tier2_clean', 'is_db_lock_ready', 'dblock_tier1_ready']:
        if col in patients_df.columns:
            patients_df[col] = patients_df[col].astype(bool)
    
    # Insert in batches
    batch_size = 5000
    for i in tqdm(range(0, len(patients_df), batch_size), desc="Inserting"):
        batch = patients_df.iloc[i:i+batch_size]
        batch.to_sql('patients', engine, if_exists='append', index=False, method='multi')
    
    print(f"   ✅ Created {len(patients_df)} patients")
    
    # Verify
    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM patients")).scalar()
        print(f"\n✅ Verification: {result} patients in database")
    
    print("\n" + "=" * 70)
    print("MIGRATION COMPLETE")
    print("=" * 70)
    print(f"   Studies: {len(studies_df)}")
    print(f"   Sites: {len(sites_df)}")
    print(f"   Patients: {len(patients_df)}")
    print(f"\nFinished: {datetime.now()}")


if __name__ == "__main__":
    run_simple_migration()
