"""
SANCHALAK AI - Knowledge Graph Sync Worker
==============================================
Background worker to synchronize 9 heterogeneous data sources from PostgreSQL 
into the Neo4j Knowledge Graph in real-time.
"""

import sys
import io

# Fix console encoding for emojis
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import time
import logging
import signal
from datetime import datetime
from pathlib import Path
import pandas as pd
from sqlalchemy import text

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.connection import get_db_manager
from src.knowledge.neo4j_graph import get_graph_service

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(PROJECT_ROOT / "logs" / "neo4j_sync.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Neo4jSyncWorker:
    def __init__(self, interval_seconds: int = 300): # Sync every 5 mins
        self.interval = interval_seconds
        self.running = True
        self._db_manager = get_db_manager()
        self.graph_service = get_graph_service()
        
        # Register signals
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)
        
    def stop(self, *args):
        logger.info("Stopping Neo4j sync worker...")
        self.running = False
        
    def sync(self):
        """Synchronize data from PG to Neo4j."""
        logger.info("Starting Knowledge Graph synchronization...")
        
        if self.graph_service.uses_mock:
            logger.warning("Neo4j service is in mock mode. Real sync skipped.")
            return

        try:
            with self._db_manager.engine.connect() as conn:
                # 1. Sync Studies
                studies = pd.read_sql(text("SELECT * FROM studies"), conn)
                for _, s in studies.iterrows():
                    self.graph_service.upsert_study(str(s['study_id']), str(s['name']))
                
                # 2. Sync Sites
                sites = pd.read_sql(text("SELECT * FROM clinical_sites"), conn)
                for _, s in sites.iterrows():
                    self.graph_service.upsert_site(str(s['site_id']), str(s['name']), str(s.get('region', '')))
                
                # 3. Sync Patients
                patients = pd.read_sql(text("SELECT patient_key, site_id, study_id FROM patients"), conn)
                for _, p in patients.iterrows():
                    self.graph_service.upsert_patient(str(p['patient_key']), str(p['site_id']))
                
                # 4. Sync Issues and Dependencies (Cascade Engine Foundation)
                issues = pd.read_sql(text("SELECT * FROM project_issues WHERE status = 'open'"), conn)
                for _, i in issues.iterrows():
                    self.graph_service.upsert_issue(
                        str(i['issue_id']), str(i['patient_key']), str(i['issue_type']), 
                        str(i['priority']), float(i.get('cascade_impact_score') or 0.0)
                    )
                
                # 5. Create IssueType aggregate nodes and cascade BLOCKS relationships
                all_types = [
                    "missing_visits", "missing_pages", "open_queries", "sdv_incomplete",
                    "signature_gaps", "broken_signatures", "sae_dm_pending", "sae_safety_pending",
                    "meddra_uncoded", "whodrug_uncoded", "lab_issues", "edrr_issues",
                    "inactivated_forms", "high_query_volume", "db_lock"
                ]
                cascade_rules = [
                    ('missing_visits', 'missing_pages'), ('missing_visits', 'open_queries'),
                    ('missing_visits', 'sdv_incomplete'), ('missing_visits', 'signature_gaps'),
                    ('missing_pages', 'open_queries'), ('missing_pages', 'sdv_incomplete'),
                    ('missing_pages', 'signature_gaps'),
                    ('open_queries', 'signature_gaps'), ('open_queries', 'db_lock'),
                    ('sdv_incomplete', 'signature_gaps'), ('sdv_incomplete', 'db_lock'),
                    ('signature_gaps', 'db_lock'),
                    ('broken_signatures', 'signature_gaps'), ('broken_signatures', 'db_lock'),
                    ('sae_dm_pending', 'sae_safety_pending'), ('sae_dm_pending', 'db_lock'),
                    ('sae_safety_pending', 'db_lock'),
                    ('meddra_uncoded', 'db_lock'), ('whodrug_uncoded', 'db_lock'),
                    ('lab_issues', 'db_lock'), ('edrr_issues', 'db_lock'),
                    ('inactivated_forms', 'db_lock'), ('high_query_volume', 'open_queries'),
                ]
                
                if not self.graph_service.uses_mock and self.graph_service._driver:
                    with self.graph_service._driver.session() as session:
                        # Create aggregate type nodes
                        for type_key in all_types:
                            session.run("""
                                MERGE (t:Issue {id: $type_id, type: $type_key})
                                SET t.priority = CASE $type_key WHEN 'db_lock' THEN 'Target' ELSE 'Cascade' END
                            """, type_id=f"CASCADE_{type_key.upper()}", type_key=type_key)
                        
                        # Create BLOCKS between aggregate nodes
                        for source, target in cascade_rules:
                            session.run("""
                                MATCH (a:Issue {id: $src_id})
                                MATCH (b:Issue {id: $tgt_id})
                                MERGE (a)-[:BLOCKS]->(b)
                            """, src_id=f"CASCADE_{source.upper()}", tgt_id=f"CASCADE_{target.upper()}")
                
                logger.info(f"Sync complete: {len(studies)} studies, {len(sites)} sites, {len(patients)} patients, {len(cascade_rules)} cascade rules")
                
        except Exception as e:
            logger.error(f"Sync failed: {e}")

    def run(self):
        logger.info(f"🚀 Knowledge Graph Sync Worker started (Interval: {self.interval}s)")
        
        while self.running:
            try:
                self.sync()
                if self.running:
                    time.sleep(self.interval)
            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                time.sleep(60)

if __name__ == "__main__":
    (PROJECT_ROOT / "logs").mkdir(exist_ok=True)
    worker = Neo4jSyncWorker()
    worker.run()
