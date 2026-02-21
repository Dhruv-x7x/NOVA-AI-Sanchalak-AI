"""
SANCHALAK AI - Neo4j Knowledge Graph Service
=================================================
Entity graph for Cascade Intelligence and relationship queries.

Enhanced v1.1:
- Connection pooling support
- Production mode enforcement
- Health check with detailed diagnostics
- Connection retry with exponential backoff
"""

import os
import logging
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from threading import Lock
from functools import wraps

import pandas as pd

logger = logging.getLogger(__name__)

# ============== Configuration ==============
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# Production mode: if True, fail if Neo4j unavailable (no mock fallback)
NEO4J_PRODUCTION_MODE = os.getenv("NEO4J_PRODUCTION_MODE", "false").lower() == "true"

# Connection pool settings
NEO4J_MAX_POOL_SIZE = int(os.getenv("NEO4J_MAX_POOL_SIZE", "50"))
NEO4J_CONNECTION_TIMEOUT = int(os.getenv("NEO4J_CONNECTION_TIMEOUT", "30"))
NEO4J_MAX_RETRY_ATTEMPTS = int(os.getenv("NEO4J_MAX_RETRY_ATTEMPTS", "3"))

try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    if NEO4J_PRODUCTION_MODE:
        raise ImportError("neo4j-driver required in production mode. Install with: pip install neo4j")
    logger.warning("neo4j-driver not installed, using mock graph")


def retry_on_connection_error(max_retries: int = 3, base_delay: float = 1.0):
    """Decorator for retrying Neo4j operations on connection errors."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_str = str(e).lower()
                    if any(x in error_str for x in ['connection', 'timeout', 'unavailable']):
                        last_error = e
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"Neo4j connection error (attempt {attempt + 1}/{max_retries}): {e}")
                        if attempt < max_retries - 1:
                            time.sleep(delay)
                    else:
                        raise
            raise last_error
        return wrapper
    return decorator


@dataclass
class CascadePath:
    """Represents a cascade path through the entity graph."""
    source_id: str
    source_type: str
    path_nodes: List[Dict[str, Any]]
    path_relationships: List[str]
    total_impact: float
    affected_patients: int
    unlocked_actions: List[str]


class MockGraphService:
    """Mock service when Neo4j is not available."""
    
    def __init__(self):
        self._entities = {}
        self._relationships = []
    
    def add_entity(self, entity_type: str, entity_id: str, properties: Dict = None):
        self._entities[f"{entity_type}:{entity_id}"] = {
            "type": entity_type,
            "id": entity_id,
            "properties": properties or {}
        }
    
    def add_relationship(self, from_type: str, from_id: str, rel_type: str, to_type: str, to_id: str):
        self._relationships.append({
            "from": f"{from_type}:{from_id}",
            "type": rel_type,
            "to": f"{to_type}:{to_id}"
        })
    
    def get_cascade_path(self, issue_id: str, max_depth: int = 3) -> CascadePath:
        """Get mock cascade path."""
        return CascadePath(
            source_id=issue_id,
            source_type="issue",
            path_nodes=[
                {"type": "issue", "id": issue_id, "label": f"Issue {issue_id}"},
                {"type": "patient", "id": "P001", "label": "Patient P001"},
                {"type": "site", "id": "SITE001", "label": "Site SITE001"}
            ],
            path_relationships=["AFFECTS", "AT_SITE"],
            total_impact=15.5,
            affected_patients=12,
            unlocked_actions=["Resolve Query", "Complete SDV", "Sign CRF"]
        )
    
    def query_downstream(self, entity_type: str, entity_id: str) -> List[Dict]:
        """Get downstream dependencies."""
        return [
            {"type": "patient", "id": "P001", "impact": 5.2},
            {"type": "patient", "id": "P002", "impact": 3.1}
        ]
    
    def close(self):
        pass


class Neo4jGraphService:
    """
    Neo4j-based knowledge graph service for Cascade Intelligence.
    
    Provides:
    - Entity graph management (Study → Site → Patient → Issue)
    - Cascade path computation
    - Downstream impact analysis
    - Pattern detection via graph algorithms
    
    Enhanced Features (v1.1):
    - Connection pooling with configurable pool size
    - Production mode enforcement (no mock fallback)
    - Health check with detailed diagnostics
    - Connection retry with exponential backoff
    - Query metrics tracking
    """
    
    _instance = None
    _lock = Lock()
    
    def __init__(self, production_mode: Optional[bool] = None):
        """
        Initialize Neo4j Graph Service.
        
        Args:
            production_mode: If True, fail if Neo4j unavailable (no mock).
                            If None, reads from NEO4J_PRODUCTION_MODE env var.
        """
        self._driver = None
        self._connected = False
        self._production_mode = production_mode if production_mode is not None else NEO4J_PRODUCTION_MODE
        self._use_mock = not NEO4J_AVAILABLE
        self._mock_service = None
        
        # Metrics tracking
        self._metrics = {
            "queries_executed": 0,
            "queries_failed": 0,
            "last_query_time": None,
            "connection_attempts": 0,
            "connection_failures": 0,
            "last_health_check": None
        }
        
        # Initialize mock service only if allowed
        if self._use_mock and not self._production_mode:
            self._mock_service = MockGraphService()
    
    @retry_on_connection_error(max_retries=NEO4J_MAX_RETRY_ATTEMPTS)
    def connect(self) -> bool:
        """
        Connect to Neo4j database with connection pooling.
        
        Returns:
            True if connected (or using mock in non-production mode)
            
        Raises:
            ConnectionError: If production mode and Neo4j unavailable
        """
        self._metrics["connection_attempts"] += 1
        
        if self._use_mock:
            if self._production_mode:
                raise ConnectionError(
                    "Neo4j driver not available and production mode is enabled. "
                    "Install with: pip install neo4j"
                )
            self._connected = True
            logger.info("Using mock Neo4j service (non-production mode)")
            return True
        
        try:
            self._driver = GraphDatabase.driver(
                NEO4J_URI,
                auth=(NEO4J_USER, NEO4J_PASSWORD),
                max_connection_pool_size=NEO4J_MAX_POOL_SIZE,
                connection_timeout=NEO4J_CONNECTION_TIMEOUT,
                connection_acquisition_timeout=NEO4J_CONNECTION_TIMEOUT * 2
            )
            # Test connection
            with self._driver.session(database=NEO4J_DATABASE) as session:
                result = session.run("RETURN 1 as test")
                result.single()
            
            self._connected = True
            logger.info(f"Connected to Neo4j: {NEO4J_URI} (pool_size={NEO4J_MAX_POOL_SIZE})")
            return True
            
        except Exception as e:
            self._metrics["connection_failures"] += 1
            
            if self._production_mode:
                raise ConnectionError(
                    f"Neo4j connection failed in production mode: {e}. "
                    f"Check NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD environment variables."
                )
            
            logger.warning(f"Neo4j connection failed: {e}, falling back to mock service")
            self._use_mock = True
            self._mock_service = MockGraphService()
            self._connected = True
            return True
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    @property
    def uses_mock(self) -> bool:
        return self._use_mock
    
    def _ensure_connected(self):
        if not self._connected:
            self.connect()
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check Neo4j health and return detailed status.
        
        Returns comprehensive diagnostics including:
        - Connection status and mode (real vs mock)
        - Graph statistics (nodes, relationships)
        - Query metrics (executed, failed, latency)
        - Configuration details
        """
        self._ensure_connected()
        self._metrics["last_health_check"] = datetime.now().isoformat()
        
        status = {
            "connected": self._connected,
            "using_mock": self._use_mock,
            "production_mode": self._production_mode,
            "uri": NEO4J_URI if not self._use_mock else "mock://localhost",
            "database": NEO4J_DATABASE,
            "status": "healthy" if self._connected else "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "metrics": self._metrics.copy()
        }
        
        if self._connected and not self._use_mock:
            try:
                # Get graph statistics
                start_time = time.time()
                stats = self.get_graph_stats()
                query_time_ms = (time.time() - start_time) * 1000
                
                status.update({
                    "nodes": sum(v for k, v in stats.items() if k not in ["relationships", "using_mock"]),
                    "relationships": stats.get("relationships", 0),
                    "node_counts": {k: v for k, v in stats.items() if k not in ["relationships", "using_mock"]},
                    "query_latency_ms": round(query_time_ms, 2),
                    "pool_size": NEO4J_MAX_POOL_SIZE
                })
                
                # Check connection pool health
                if self._driver:
                    status["driver_active"] = True
                    
            except Exception as e:
                status["error"] = str(e)
                status["status"] = "degraded"
                logger.warning(f"Health check encountered error: {e}")
        
        elif self._use_mock:
            # Mock service stats
            status["mock_entities"] = len(self._mock_service._entities) if self._mock_service else 0
            status["mock_relationships"] = len(self._mock_service._relationships) if self._mock_service else 0
        
        return status
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get detailed connection information for debugging."""
        return {
            "uri": NEO4J_URI,
            "database": NEO4J_DATABASE,
            "user": NEO4J_USER,
            "production_mode": self._production_mode,
            "using_mock": self._use_mock,
            "connected": self._connected,
            "pool_size": NEO4J_MAX_POOL_SIZE,
            "connection_timeout": NEO4J_CONNECTION_TIMEOUT,
            "max_retries": NEO4J_MAX_RETRY_ATTEMPTS,
            "metrics": self._metrics
        }
    
    def verify_schema(self) -> Dict[str, Any]:
        """
        Verify that the required schema exists in Neo4j.
        
        Returns dict with constraint and index status.
        """
        if self._use_mock:
            return {"using_mock": True, "schema_valid": True}
        
        self._ensure_connected()
        
        required_labels = ["Study", "Site", "Patient", "Issue", "Visit", "LabResult"]
        required_relationships = ["CONTAINS", "HAS_PATIENT", "HAS_ISSUE", "BLOCKS", "HAS_VISIT", "HAS_LAB"]
        
        schema_status = {
            "labels_found": [],
            "labels_missing": [],
            "constraints": [],
            "indexes": [],
            "schema_valid": True
        }
        
        try:
            with self._driver.session(database=NEO4J_DATABASE) as session:
                # Check for labels
                label_result = session.run("CALL db.labels()")
                existing_labels = [r["label"] for r in label_result]
                
                for label in required_labels:
                    if label in existing_labels:
                        schema_status["labels_found"].append(label)
                    else:
                        schema_status["labels_missing"].append(label)
                
                # Check constraints
                try:
                    constraint_result = session.run("SHOW CONSTRAINTS")
                    schema_status["constraints"] = [dict(r) for r in constraint_result]
                except Exception:
                    # Older Neo4j version
                    pass
                
                # Check indexes
                try:
                    index_result = session.run("SHOW INDEXES")
                    schema_status["indexes"] = [dict(r) for r in index_result]
                except Exception:
                    pass
                
                if schema_status["labels_missing"]:
                    schema_status["schema_valid"] = False
                    
        except Exception as e:
            schema_status["error"] = str(e)
            schema_status["schema_valid"] = False
            
        return schema_status
    
    # ============== Graph Build Methods ==============
    
    def build_entity_graph(self, upr_df: pd.DataFrame):
        """
        Build entity graph from Unified Patient Record.
        
        Creates nodes: Study, Site, Patient, Issue
        Creates relationships: CONTAINS, HAS_PATIENT, HAS_ISSUE, BLOCKS
        """
        self._ensure_connected()
        
        if self._use_mock:
            # Build mock graph
            for _, row in upr_df.iterrows():
                self._mock_service.add_entity("Study", row.get('study_id', 'UNK'))
                self._mock_service.add_entity("Site", row.get('site_id', 'UNK'))
                self._mock_service.add_entity("Patient", row.get('patient_key', 'UNK'))
            return
        
        # Real Neo4j build
        with self._driver.session() as session:
            # Create constraints (idempotent)
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (s:Study) REQUIRE s.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (st:Site) REQUIRE st.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Patient) REQUIRE p.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (i:Issue) REQUIRE i.id IS UNIQUE")
            
            # Build studies
            studies = upr_df[['study_id']].drop_duplicates()
            for _, row in studies.iterrows():
                session.run(
                    "MERGE (s:Study {id: $id}) SET s.name = $id",
                    id=row['study_id']
                )
            
            # Build sites with relationships
            site_cols = ['study_id', 'site_id']
            if 'region' in upr_df.columns: site_cols.append('region')
            if 'country' in upr_df.columns: site_cols.append('country')
            
            sites = upr_df[site_cols].drop_duplicates()
            for _, row in sites.iterrows():
                session.run("""
                    MERGE (st:Site {id: $site_id})
                    SET st.region = $region, st.country = $country
                    WITH st
                    MATCH (s:Study {id: $study_id})
                    MERGE (s)-[:CONTAINS]->(st)
                """, site_id=row['site_id'], study_id=row['study_id'],
                     region=row.get('region', ''), country=row.get('country', ''))
            
            # Build patients with relationships
            for _, row in upr_df.iterrows():
                session.run("""
                    MERGE (p:Patient {id: $patient_key})
                    SET p.dqi = $dqi, p.status = $status
                    WITH p
                    MATCH (st:Site {id: $site_id})
                    MERGE (st)-[:HAS_PATIENT]->(p)
                """, patient_key=row['patient_key'],
                     dqi=row.get('dqi_score', 0),
                     status=row.get('subject_status', 'Active'),
                     site_id=row['site_id'])
            
            logger.info(f"Built graph with {len(upr_df)} patients")

    def add_visit_nodes(self, visits_df: pd.DataFrame):
        """Add Visit nodes and link to Patients."""
        self._ensure_connected()
        
        if self._use_mock:
            return

        with self._driver.session() as session:
            # Create constraints
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (v:Visit) REQUIRE v.id IS UNIQUE")
            
            # Batch create visits
            visit_list = visits_df.to_dict('records')
            session.run("""
                UNWIND $visits as v
                MERGE (visit:Visit {id: v.visit_id})
                SET visit.name = v.visit_name,
                    visit.status = v.status,
                    visit.date = v.scheduled_date
                WITH visit, v
                MATCH (p:Patient {id: v.patient_key})
                MERGE (p)-[:HAS_VISIT]->(visit)
            """, visits=visit_list)
            
            logger.info(f"Added {len(visits_df)} visits")

    def add_lab_nodes(self, labs_df: pd.DataFrame):
        """Add LabResult nodes (only high risk/abnormal) and link to Patients."""
        self._ensure_connected()
        
        if self._use_mock:
            return

        with self._driver.session() as session:
            # Create constraints
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (l:LabResult) REQUIRE l.id IS UNIQUE")
            
            # Batch create labs
            lab_list = labs_df.to_dict('records')
            session.run("""
                UNWIND $labs as l
                MERGE (lab:LabResult {id: l.lab_id})
                SET lab.test = l.test_name,
                    lab.val = l.result_value,
                    lab.unit = l.unit
                WITH lab, l
                MATCH (p:Patient {id: l.patient_key})
                MERGE (p)-[:HAS_LAB]->(lab)
            """, labs=lab_list)
            
            logger.info(f"Added {len(labs_df)} lab results")
    
    def add_issues_batch(self, issues_df: pd.DataFrame):
        """
        Add multiple issues in batch.
        Expected columns: issue_id, patient_key, type, priority
        """
        self._ensure_connected()
        
        if self._use_mock:
            for _, row in issues_df.iterrows():
                self._mock_service.add_entity("Issue", row['issue_id'], {
                    "type": row.get('type', 'Unknown'), 
                    "priority": row.get('priority', 'Low')
                })
                if 'patient_key' in row:
                    self._mock_service.add_relationship("Patient", row['patient_key'], "HAS_ISSUE", "Issue", row['issue_id'])
            return
            
        with self._driver.session() as session:
            # Prepare list of dicts for UNWIND
            issue_list = issues_df.to_dict('records')
            
            # Batch merge issues and link to patients
            session.run("""
                UNWIND $issues as issue
                MERGE (i:Issue {id: issue.issue_id})
                SET i.type =  coalesce(issue.type, 'Unknown'), 
                    i.priority = coalesce(issue.priority, 'Low')
                WITH i, issue
                MATCH (p:Patient {id: issue.patient_key})
                MERGE (p)-[:HAS_ISSUE]->(i)
            """, issues=issue_list)
            
            logger.info(f"Batch added {len(issues_df)} issues")

    def add_issue_node(self, issue_id: str, patient_key: str, issue_type: str, 
                       priority: str, blocking_issues: List[str] = None):
        """Add an issue node with blocking relationships."""
        self._ensure_connected()
        
        if self._use_mock:
            self._mock_service.add_entity("Issue", issue_id, {
                "type": issue_type, "priority": priority
            })
            self._mock_service.add_relationship("Patient", patient_key, "HAS_ISSUE", "Issue", issue_id)
            return
        
        with self._driver.session() as session:
            # Create issue and link to patient
            session.run("""
                MERGE (i:Issue {id: $issue_id})
                SET i.type = $issue_type, i.priority = $priority
                WITH i
                MATCH (p:Patient {id: $patient_key})
                MERGE (p)-[:HAS_ISSUE]->(i)
            """, issue_id=issue_id, patient_key=patient_key,
                 issue_type=issue_type, priority=priority)
            
            # Add blocking relationships
            if blocking_issues:
                for blocked_id in blocking_issues:
                    session.run("""
                        MATCH (i1:Issue {id: $issue_id})
                        MATCH (i2:Issue {id: $blocked_id})
                        MERGE (i1)-[:BLOCKS]->(i2)
                    """, issue_id=issue_id, blocked_id=blocked_id)
    
    # ============== Cascade Analysis Methods ==============
    
    def get_cascade_path(self, issue_id: str, max_depth: int = 3) -> CascadePath:
        """
        Get the cascade path for resolving an issue.
        
        Shows what gets unlocked when this issue is resolved.
        """
        self._ensure_connected()
        
        if self._use_mock:
            return self._mock_service.get_cascade_path(issue_id, max_depth)
        
        with self._driver.session() as session:
            result = session.run("""
                MATCH path = (i:Issue {id: $issue_id})-[:BLOCKS*1..]->(blocked:Issue)
                WITH blocked, length(path) as depth
                WHERE depth <= $max_depth
                MATCH (blocked)<-[:HAS_ISSUE]-(p:Patient)
                RETURN blocked.id as blocked_issue, 
                       blocked.type as issue_type,
                       p.id as patient_id,
                       p.dqi as patient_dqi,
                       depth
                ORDER BY depth
            """, issue_id=issue_id, max_depth=max_depth)
            
            records = list(result)
            
            if not records:
                # No downstream issues, check patient impact
                patient_result = session.run("""
                    MATCH (i:Issue {id: $issue_id})<-[:HAS_ISSUE]-(p:Patient)
                    RETURN p.id as patient_id, p.dqi as dqi
                """, issue_id=issue_id)
                patient = patient_result.single()
                
                return CascadePath(
                    source_id=issue_id,
                    source_type="issue",
                    path_nodes=[{"type": "issue", "id": issue_id}],
                    path_relationships=[],
                    total_impact=5.0,
                    affected_patients=1 if patient else 0,
                    unlocked_actions=["Direct resolution"]
                )
            
            # Build cascade path from results
            path_nodes = [{"type": "issue", "id": issue_id}]
            affected_patients = set()
            
            for record in records:
                path_nodes.append({
                    "type": "issue",
                    "id": record["blocked_issue"],
                    "issue_type": record["issue_type"]
                })
                affected_patients.add(record["patient_id"])
            
            return CascadePath(
                source_id=issue_id,
                source_type="issue",
                path_nodes=path_nodes,
                path_relationships=["BLOCKS"] * (len(path_nodes) - 1),
                total_impact=len(records) * 3.5,  # Estimate DQI impact
                affected_patients=len(affected_patients),
                unlocked_actions=[f"Resolve {r['issue_type']}" for r in records[:5]]
            )
    
    def get_site_cascade_opportunities(self, site_id: str, top_n: int = 5) -> List[Dict]:
        """
        Find top cascade opportunities for a site.
        
        Returns issues that unlock the most downstream improvements.
        """
        self._ensure_connected()
        
        if self._use_mock:
            return [
                {"issue_id": "ISS001", "type": "open_query", "impact": 15.2, "unlocks": 8},
                {"issue_id": "ISS002", "type": "missing_signature", "impact": 12.1, "unlocks": 5},
                {"issue_id": "ISS003", "type": "sdv_pending", "impact": 8.5, "unlocks": 3}
            ][:top_n]
        
        with self._driver.session() as session:
            result = session.run("""
                MATCH (st:Site {id: $site_id})-[:HAS_PATIENT]->(p:Patient)-[:HAS_ISSUE]->(i:Issue)
                OPTIONAL MATCH (i)-[:BLOCKS*1..3]->(blocked:Issue)
                WITH i, count(blocked) as blocked_count
                RETURN i.id as issue_id, 
                       i.type as issue_type, 
                       i.priority as priority,
                       blocked_count,
                       blocked_count * 3.5 as estimated_impact
                ORDER BY blocked_count DESC
                LIMIT $top_n
            """, site_id=site_id, top_n=top_n)
            
            return [
                {
                    "issue_id": r["issue_id"],
                    "type": r["issue_type"],
                    "priority": r["priority"],
                    "impact": r["estimated_impact"],
                    "unlocks": r["blocked_count"]
                }
                for r in result
            ]
    
    def get_patient_dependencies(self, patient_key: str) -> Dict[str, Any]:
        """Get issue dependencies for a patient."""
        self._ensure_connected()
        
        if self._use_mock:
            return {
                "patient_key": patient_key,
                "total_issues": 5,
                "blocking_issues": 2,
                "blocked_issues": 3,
                "dependency_graph": []
            }
        
        with self._driver.session() as session:
            result = session.run("""
                MATCH (p:Patient {id: $patient_key})-[:HAS_ISSUE]->(i:Issue)
                OPTIONAL MATCH (i)-[:BLOCKS]->(blocked:Issue)
                OPTIONAL MATCH (blocker:Issue)-[:BLOCKS]->(i)
                RETURN i.id as issue_id,
                       i.type as issue_type,
                       collect(DISTINCT blocked.id) as blocks,
                       collect(DISTINCT blocker.id) as blocked_by
            """, patient_key=patient_key)
            
            issues = []
            blocking_count = 0
            blocked_count = 0
            
            for r in result:
                issues.append({
                    "id": r["issue_id"],
                    "type": r["issue_type"],
                    "blocks": [b for b in r["blocks"] if b],
                    "blocked_by": [b for b in r["blocked_by"] if b]
                })
                if r["blocks"]:
                    blocking_count += 1
                if r["blocked_by"]:
                    blocked_count += 1
            
            return {
                "patient_key": patient_key,
                "total_issues": len(issues),
                "blocking_issues": blocking_count,
                "blocked_issues": blocked_count,
                "dependency_graph": issues
            }
    
    # ============== Utility Methods ==============
    
    def execute_query(self, cypher: str, **params) -> List[Dict[str, Any]]:
        """Execute an arbitrary Cypher query and return results as list of dicts."""
        self._ensure_connected()
        self._metrics["queries_executed"] += 1
        
        if self._use_mock:
            return []
        
        try:
            with self._driver.session(database=NEO4J_DATABASE) as session:
                result = session.run(cypher, **params)
                return [dict(record) for record in result]
        except Exception as e:
            self._metrics["queries_failed"] += 1
            logger.error(f"Cypher query failed: {e}")
            raise
    
    def upsert_study(self, study_id: str, name: str):
        """Upsert a Study node."""
        self._ensure_connected()
        if self._use_mock:
            return
        with self._driver.session(database=NEO4J_DATABASE) as session:
            session.run(
                "MERGE (s:Study {id: $id}) SET s.name = $name",
                id=study_id, name=name
            )
    
    def upsert_site(self, site_id: str, name: str, region: str = ""):
        """Upsert a Site node."""
        self._ensure_connected()
        if self._use_mock:
            return
        with self._driver.session(database=NEO4J_DATABASE) as session:
            session.run(
                "MERGE (st:Site {id: $id}) SET st.name = $name, st.region = $region",
                id=site_id, name=name, region=region
            )
    
    def upsert_patient(self, patient_key: str, site_id: str):
        """Upsert a Patient node and link to site."""
        self._ensure_connected()
        if self._use_mock:
            return
        with self._driver.session(database=NEO4J_DATABASE) as session:
            session.run("""
                MERGE (p:Patient {id: $patient_key})
                WITH p
                MATCH (st:Site {id: $site_id})
                MERGE (st)-[:HAS_PATIENT]->(p)
            """, patient_key=patient_key, site_id=site_id)
    
    def upsert_issue(self, issue_id: str, patient_key: str, issue_type: str,
                     priority: str, impact: float = 0.0):
        """Upsert an Issue node and link to patient."""
        self._ensure_connected()
        if self._use_mock:
            return
        with self._driver.session(database=NEO4J_DATABASE) as session:
            session.run("""
                MERGE (i:Issue {id: $issue_id})
                SET i.type = $issue_type, i.priority = $priority, i.impact = $impact
                WITH i
                MATCH (p:Patient {id: $patient_key})
                MERGE (p)-[:HAS_ISSUE]->(i)
            """, issue_id=issue_id, patient_key=patient_key,
                 issue_type=issue_type, priority=priority, impact=impact)
    
    def get_cascade_topology(self) -> Dict[str, List[str]]:
        """Return BLOCKS topology as {source_type: [target_types]} from the graph."""
        self._ensure_connected()
        
        if self._use_mock:
            return {}
        
        try:
            with self._driver.session(database=NEO4J_DATABASE) as session:
                result = session.run("""
                    MATCH (a:Issue)-[:BLOCKS]->(b:Issue)
                    RETURN a.type as source, collect(DISTINCT b.type) as targets
                """)
                return {r["source"]: r["targets"] for r in result if r["source"]}
        except Exception as e:
            logger.error(f"get_cascade_topology failed: {e}")
            return {}
    
    def create_cascade_schema(self):
        """Create constraints and indexes for cascade-specific nodes."""
        self._ensure_connected()
        if self._use_mock:
            return
        with self._driver.session(database=NEO4J_DATABASE) as session:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (s:Study) REQUIRE s.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (st:Site) REQUIRE st.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Patient) REQUIRE p.id IS UNIQUE")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (i:Issue) REQUIRE i.id IS UNIQUE")
            # Indexes for cascade queries
            session.run("CREATE INDEX IF NOT EXISTS FOR (i:Issue) ON (i.type)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (i:Issue) ON (i.priority)")
            session.run("CREATE INDEX IF NOT EXISTS FOR (p:Patient) ON (p.site_id)")
            logger.info("Cascade schema created/verified")
    
    def get_graph_stats(self) -> Dict[str, int]:
        """Get graph statistics."""
        self._ensure_connected()
        
        if self._use_mock:
            return {
                "studies": len([e for e in self._mock_service._entities if "Study:" in e]),
                "sites": len([e for e in self._mock_service._entities if "Site:" in e]),
                "patients": len([e for e in self._mock_service._entities if "Patient:" in e]),
                "issues": len([e for e in self._mock_service._entities if "Issue:" in e]),
                "relationships": len(self._mock_service._relationships),
                "using_mock": True
            }
        
        with self._driver.session(database=NEO4J_DATABASE) as session:
            stats = {}
            for label in ["Study", "Site", "Patient", "Issue"]:
                result = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
                stats[label.lower() + "s"] = result.single()["count"]
            
            rel_result = session.run("MATCH ()-[r]->() RETURN count(r) as count")
            stats["relationships"] = rel_result.single()["count"]
            stats["using_mock"] = False
            
            return stats
    
    def clear_graph(self):
        """Clear all nodes and relationships (use with caution!)."""
        self._ensure_connected()
        
        if self._use_mock:
            self._mock_service._entities.clear()
            self._mock_service._relationships.clear()
            return
        
        with self._driver.session(database=NEO4J_DATABASE) as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.warning("Cleared entire Neo4j graph")
    
    def close(self):
        """Close Neo4j connection."""
        if self._driver:
            self._driver.close()
            self._connected = False
        if self._mock_service:
            self._mock_service.close()


# Singleton accessor
_graph_service: Optional[Neo4jGraphService] = None


def get_graph_service() -> Neo4jGraphService:
    """Get singleton Neo4j graph service."""
    global _graph_service
    if _graph_service is None:
        _graph_service = Neo4jGraphService()
        _graph_service.connect()
    return _graph_service


def reset_graph_service():
    """Reset the singleton (for testing)."""
    global _graph_service
    if _graph_service:
        _graph_service.close()
    _graph_service = None


# Alias for backward compatibility
get_knowledge_graph = get_graph_service
