"""
a6on-i - Knowledge Graph Seeding Script
=================================================
Seeds the Neo4j graph with data from PostgreSQL and analytics.
"""

import os
import sys
import logging
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.knowledge.neo4j_graph import get_graph_service
from src.database.connection import get_db_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def seed_graph():
    """Seed the Neo4j graph with current trial data."""
    logger.info("Starting Knowledge Graph seeding...")
    
    graph = get_graph_service()
    db = get_db_manager()
    
    if graph.uses_mock:
        logger.warning("Neo4j not available, seeding mock service (for testing only)")
    
    try:
        # 1. Load UPR data for core entities
        logger.info("Loading UPR data from PostgreSQL...")
        with db.engine.connect() as conn:
            upr_df = pd.read_sql("SELECT * FROM unified_patient_record", conn)
        
        if upr_df.empty:
            logger.warning("No patient data found in PostgreSQL. Skipping core entity build.")
        else:
            logger.info(f"Building entity graph with {len(upr_df)} patients...")
            graph.build_entity_graph(upr_df)
        
        # 2. Add Visits
        logger.info("Loading visits...")
        with db.engine.connect() as conn:
            visits_df = pd.read_sql("SELECT * FROM visits", conn)
        if not visits_df.empty:
            graph.add_visit_nodes(visits_df)
            
        # 3. Add Issues and Blockers
        logger.info("Loading issues and blockers...")
        with db.engine.connect() as conn:
            issues_df = pd.read_sql("SELECT * FROM project_issues", conn)
        
        if not issues_df.empty:
            # Map columns for graph service
            issues_df = issues_df.rename(columns={'category': 'type'})
            graph.add_issues_batch(issues_df)
            
            # Simulated blockers for testing cascade logic
            logger.info("Adding simulated blocking relationships for testing...")
            # Link some issues to block others for demonstration
            # In a real system, this would come from a dependency table
            with graph._driver.session() as session:
                if not graph.uses_mock:
                    session.run("""
                        MATCH (i1:Issue), (i2:Issue)
                        WHERE i1.id <> i2.id AND i1.type = 'MISSING_PAGES' AND i2.type = 'sdv_incomplete'
                        WITH i1, i2 LIMIT 50
                        MERGE (i1)-[:BLOCKS]->(i2)
                    """)
                    session.run("""
                        MATCH (i1:Issue), (i2:Issue)
                        WHERE i1.id <> i2.id AND i1.type = 'open_queries' AND i2.type = 'DBLOCK_READY'
                        WITH i1, i2 LIMIT 50
                        MERGE (i1)-[:BLOCKS]->(i2)
                    """)
        
        stats = graph.get_graph_stats()
        logger.info(f"Seeding completed successfully! Stats: {stats}")
        return True
        
    except Exception as e:
        logger.error(f"Seeding failed: {e}")
        return False

if __name__ == "__main__":
    success = seed_graph()
    sys.exit(0 if success else 1)
