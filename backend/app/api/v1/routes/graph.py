"""
Graph Routes
=============
Endpoints for Neo4j graph queries and cascade visualization.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Optional, List
from datetime import datetime
import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from app.core.security import get_current_user
from app.services.database import get_data_service

router = APIRouter()


import logging

logger = logging.getLogger(__name__)

def get_graph_service():
    """Get Neo4j graph service if available with fast failure."""
    from app.config import settings
    if not settings.NEO4J_ENABLED:
        return None
        
    try:
        from src.knowledge.neo4j_graph import get_graph_service
        return get_graph_service()
    except Exception as e:
        logger.warning(f"Neo4j not available or connection failed: {e}")
        return None


@router.get("/nodes")
async def get_graph_nodes(
    node_type: Optional[str] = Query(None, description="Filter by node type: Patient, Site, Issue, Study"),
    limit: int = Query(100, le=500),
    current_user: dict = Depends(get_current_user)
):
    """Get graph nodes for visualization."""
    try:
        graph = get_graph_service()
        
        if graph:
            # Query Neo4j
            if node_type:
                query = f"MATCH (n:{node_type}) RETURN n LIMIT {limit}"
            else:
                query = f"MATCH (n) RETURN n LIMIT {limit}"
                
            results = graph.execute_query(query)
            nodes = []
            for record in results:
                node = record.get('n', {})
                nodes.append({
                    "id": node.get('id') or node.get('node_id'),
                    "label": node.get('name') or node.get('label') or str(node.get('id', '')),
                    "type": node_type or 'Unknown',
                    "properties": dict(node)
                })
            return {"nodes": nodes, "total": len(nodes)}
        
        # Fall back to sample data
        return {"nodes": _get_sample_nodes(node_type, limit), "total": limit}
        
    except Exception as e:
        return {"nodes": _get_sample_nodes(node_type, limit), "total": limit}


@router.get("/edges")
async def get_graph_edges(
    source_type: Optional[str] = None,
    target_type: Optional[str] = None,
    relationship: Optional[str] = None,
    limit: int = Query(200, le=1000),
    current_user: dict = Depends(get_current_user)
):
    """Get graph edges/relationships."""
    try:
        graph = get_graph_service()
        
        if graph:
            query = "MATCH (a)-[r]->(b) "
            if source_type:
                query = f"MATCH (a:{source_type})-[r]->(b) "
            if target_type:
                query = query.replace("(b)", f"(b:{target_type})")
            if relationship:
                query = query.replace("[r]", f"[r:{relationship}]")
            query += f"RETURN a, r, b LIMIT {limit}"
            
            results = graph.execute_query(query)
            edges = []
            for record in results:
                edges.append({
                    "source": record.get('a', {}).get('id'),
                    "target": record.get('b', {}).get('id'),
                    "relationship": relationship or 'RELATED_TO',
                    "properties": dict(record.get('r', {}))
                })
            return {"edges": edges, "total": len(edges)}
        
        return {"edges": _get_sample_edges(limit), "total": limit}
        
    except Exception:
        return {"edges": _get_sample_edges(limit), "total": limit}


@router.get("/cascade-path/{issue_id}")
async def get_cascade_path(
    issue_id: str,
    max_depth: int = Query(3, ge=1, le=5),
    current_user: dict = Depends(get_current_user)
):
    """Get cascade impact path for an issue."""
    try:
        graph = get_graph_service()
        
        if graph:
            query = f"""
                MATCH path = (i:Issue {{issue_id: '{issue_id}'}})-[*1..{max_depth}]->(n)
                RETURN path
                LIMIT 100
            """
            results = graph.execute_query(query)
            
            nodes = set()
            edges = []
            
            for record in results:
                path = record.get('path')
                if path:
                    for node in path.nodes:
                        nodes.add(node.get('id'))
                    for rel in path.relationships:
                        edges.append({
                            "source": rel.start_node.get('id'),
                            "target": rel.end_node.get('id'),
                            "type": rel.type
                        })
            
            return {
                "issue_id": issue_id,
                "nodes": list(nodes),
                "edges": edges,
                "depth": max_depth
            }
        
        # Sample cascade path
        return _get_sample_cascade_path(issue_id, max_depth)
        
    except Exception:
        return _get_sample_cascade_path(issue_id, max_depth)


@router.get("/dependencies/{entity_id}")
async def get_entity_dependencies(
    entity_id: str,
    entity_type: str = Query(..., description="Entity type: Patient, Site, Issue, Study"),
    direction: str = Query("both", description="Direction: incoming, outgoing, both"),
    current_user: dict = Depends(get_current_user)
):
    """Get dependencies for an entity."""
    try:
        graph = get_graph_service()
        
        if graph:
            if direction == "incoming":
                query = f"MATCH (n)-[r]->(e:{entity_type} {{id: '{entity_id}'}}) RETURN n, r, e"
            elif direction == "outgoing":
                query = f"MATCH (e:{entity_type} {{id: '{entity_id}'}})-[r]->(n) RETURN e, r, n"
            else:
                query = f"MATCH (e:{entity_type} {{id: '{entity_id}'}})-[r]-(n) RETURN e, r, n"
            
            results = graph.execute_query(query)
            
            dependencies = {
                "incoming": [],
                "outgoing": []
            }
            
            for record in results:
                dep = {
                    "id": record.get('n', {}).get('id'),
                    "type": record.get('n', {}).get('type'),
                    "relationship": record.get('r', {}).get('type', 'RELATED')
                }
                if direction == "incoming" or (direction == "both" and record.get('n')):
                    dependencies["incoming"].append(dep)
                else:
                    dependencies["outgoing"].append(dep)
            
            return {
                "entity_id": entity_id,
                "entity_type": entity_type,
                "dependencies": dependencies
            }
        
        return _get_sample_dependencies(entity_id, entity_type)
        
    except Exception:
        return _get_sample_dependencies(entity_id, entity_type)


@router.get("/cascade-analysis")
async def get_cascade_analysis(
    site_id: Optional[str] = None,
    study_id: Optional[str] = None,
    min_impact: float = Query(0.0, ge=0, le=1),
    limit: int = Query(100, le=1000),
    current_user: dict = Depends(get_current_user)
):
    """Get cascade analysis from PostgreSQL."""
    try:
        data_service = get_data_service()
        df = data_service.get_cascade_analysis(study_id=study_id)
        
        if df.empty:
            return {"analysis": _get_sample_cascade_analysis(), "total": 50}
        
        # Filter
        if site_id:
            df = df[df['site_id'] == site_id]
        if study_id:
            df = df[df['study_id'] == study_id]
        if min_impact > 0:
            impact_col = 'cascade_impact_score' if 'cascade_impact_score' in df.columns else 'avg_cascade_impact'
            if impact_col in df.columns:
                df = df[df[impact_col] >= min_impact]
        
        # Sort by impact
        impact_col = 'cascade_impact_score' if 'cascade_impact_score' in df.columns else 'avg_cascade_impact'
        if impact_col in df.columns:
            df = df.sort_values(impact_col, ascending=False)
        
        df = df.head(limit)
        
        from .patients import sanitize_for_json
        return {"analysis": sanitize_for_json(df.to_dict('records')), "total": len(df)}
        
    except Exception:
        return {"analysis": _get_sample_cascade_analysis(), "total": 50}


@router.get("/visualization-data")
async def get_visualization_data(
    site_id: Optional[str] = None,
    study_id: Optional[str] = None,
    include_issues: bool = True,
    include_patients: bool = True,
    limit: int = Query(100, le=300),
    current_user: dict = Depends(get_current_user)
):
    """Get complete graph data for frontend visualization - uses real PostgreSQL data."""
    try:
        data_service = get_data_service()
        
        nodes = []
        edges = []
        node_ids = set()  # Track unique node IDs
        
        # First try Neo4j if available
        graph = get_graph_service()
        if graph:
            try:
                # Get data from Neo4j
                query = """
                    MATCH (s:Site)<-[:ENROLLED_AT]-(p:Patient)
                    OPTIONAL MATCH (p)<-[:AFFECTS]-(i:Issue)
                    RETURN s, p, i
                    LIMIT 200
                """
                results = graph.execute_query(query)
                if results:
                    for record in results:
                        site = record.get('s')
                        patient = record.get('p')
                        issue = record.get('i')
                        
                        if site and site.get('site_id') not in node_ids:
                            node_ids.add(site.get('site_id'))
                            nodes.append({
                                "id": site.get('site_id'),
                                "label": site.get('name', site.get('site_id')),
                                "type": "Site",
                                "color": "#3498db",
                                "size": 30,
                                "dqi_score": float(site.get('dqi_score', 0)) if site.get('dqi_score') else 0
                            })
                        
                        if patient and patient.get('patient_key') not in node_ids:
                            node_ids.add(patient.get('patient_key'))
                            nodes.append({
                                "id": patient.get('patient_key'),
                                "label": str(patient.get('patient_key', ''))[:12],
                                "type": "Patient",
                                "color": "#2ecc71",
                                "size": 12
                            })
                            if site:
                                edges.append({
                                    "source": patient.get('patient_key'),
                                    "target": site.get('site_id'),
                                    "type": "ENROLLED_AT"
                                })
                        
                        if issue and issue.get('issue_id') not in node_ids:
                            node_ids.add(issue.get('issue_id'))
                            priority = issue.get('priority', 'Medium')
                            nodes.append({
                                "id": issue.get('issue_id'),
                                "label": str(issue.get('issue_type', 'Issue'))[:15],
                                "type": "Issue",
                                "color": _get_priority_color(priority),
                                "size": 15,
                                "priority": priority
                            })
                            if patient:
                                edges.append({
                                    "source": issue.get('issue_id'),
                                    "target": patient.get('patient_key'),
                                    "type": "AFFECTS"
                                })
                    
                    if nodes:
                        return {
                            "nodes": nodes,
                            "edges": edges,
                            "node_count": len(nodes),
                            "edge_count": len(edges),
                            "source": "neo4j"
                        }
            except Exception as neo4j_error:
                print(f"Neo4j query failed, falling back to PostgreSQL: {neo4j_error}")
        
        # Fall back to PostgreSQL - get real data
        # Get sites
        sites_df = data_service.get_sites()
        if not sites_df.empty:
            # Filter out mock sites
            sites_df = sites_df[~sites_df['site_id'].str.startswith('Site_')]
            
            if study_id and study_id != 'all' and 'study_id' in sites_df.columns:
                sites_df = sites_df[sites_df['study_id'] == study_id]
            if site_id and site_id != 'all':
                sites_df = sites_df[sites_df['site_id'] == site_id]
            
            for _, site in sites_df.head(25).iterrows():
                site_id_val = site.get('site_id')
                if site_id_val and site_id_val not in node_ids:
                    node_ids.add(site_id_val)
                    nodes.append({
                        "id": site_id_val,
                        "label": site.get('name', site_id_val) or site_id_val,
                        "type": "Site",
                        "color": "#3498db",
                        "size": 30,
                        "dqi_score": float(site.get('dqi_score', 0)) if site.get('dqi_score') else 0
                    })
        
        # Get patients with their sites
        if include_patients:
            patients_df = data_service.get_patients(limit=min(limit, 80), study_id=study_id if study_id != 'all' else None)
            if not patients_df.empty:
                # Filter out mock patients  
                patients_df = patients_df[~patients_df['patient_key'].str.startswith('Study_')]
                
                if site_id and site_id != 'all' and 'site_id' in patients_df.columns:
                    patients_df = patients_df[patients_df['site_id'] == site_id]
                
                for _, patient in patients_df.head(60).iterrows():
                    patient_key = patient.get('patient_key')
                    if patient_key and patient_key not in node_ids:
                        node_ids.add(patient_key)
                        
                        # Color by risk level
                        risk = patient.get('risk_level', 'Low')
                        color = "#e74c3c" if risk == 'High' else "#f1c40f" if risk == 'Medium' else "#2ecc71"
                        
                        nodes.append({
                            "id": patient_key,
                            "label": str(patient_key)[:12],
                            "type": "Patient",
                            "color": color,
                            "size": 12,
                            "dqi_score": float(patient.get('dqi_score', 0)) if patient.get('dqi_score') else 0,
                            "risk_level": risk
                        })
                        
                        # Edge to site
                        patient_site = patient.get('site_id')
                        if patient_site and patient_site in node_ids:
                            edges.append({
                                "source": patient_key,
                                "target": patient_site,
                                "type": "ENROLLED_AT"
                            })
        
        # Get issues
        if include_issues:
            issues_df = data_service.get_issues(limit=min(limit, 100), study_id=study_id if study_id != 'all' else None)
            if not issues_df.empty:
                if site_id and site_id != 'all' and 'site_id' in issues_df.columns:
                    issues_df = issues_df[issues_df['site_id'] == site_id]
                
                for _, issue in issues_df.head(80).iterrows():
                    issue_id_val = issue.get('issue_id')
                    if issue_id_val and issue_id_val not in node_ids:
                        node_ids.add(issue_id_val)
                        priority = issue.get('priority', 'Medium')
                        nodes.append({
                            "id": issue_id_val,
                            "label": str(issue.get('issue_type', 'Issue'))[:15],
                            "type": "Issue",
                            "color": _get_priority_color(priority),
                            "size": 15,
                            "priority": priority,
                            "status": issue.get('status')
                        })
                        
                        # Edge to patient
                        patient_key = issue.get('patient_key')
                        if patient_key and patient_key in node_ids:
                            edges.append({
                                "source": issue_id_val,
                                "target": patient_key,
                                "type": "AFFECTS"
                            })
                        
                        # Edge to site if no patient edge
                        issue_site = issue.get('site_id')
                        if issue_site and issue_site in node_ids and not patient_key:
                            edges.append({
                                "source": issue_id_val,
                                "target": issue_site,
                                "type": "BELONGS_TO"
                            })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "source": "postgresql"
        }
        
    except Exception as e:
        print(f"Visualization data error: {e}")
        # Return minimal data on error
        return {
            "nodes": [],
            "edges": [],
            "node_count": 0,
            "edge_count": 0,
            "error": str(e)
        }


# Helper functions

def _get_priority_color(priority: str) -> str:
    """Get color for priority level."""
    colors = {
        "Critical": "#e74c3c",
        "High": "#e67e22",
        "Medium": "#f1c40f",
        "Low": "#27ae60"
    }
    return colors.get(priority, "#95a5a6")


def _get_sample_nodes(node_type: Optional[str], limit: int) -> List[dict]:
    """Generate sample nodes."""
    nodes = []
    
    if not node_type or node_type == "Site":
        for i in range(min(20, limit)):
            nodes.append({
                "id": f"SITE-{i+1:03d}",
                "label": f"Site {i+1}",
                "type": "Site",
                "properties": {"dqi_score": 75 + (i % 20), "patient_count": 50 + i * 10}
            })
    
    if not node_type or node_type == "Patient":
        for i in range(min(30, limit)):
            nodes.append({
                "id": f"PAT-{i+1:04d}",
                "label": f"Patient {i+1}",
                "type": "Patient",
                "properties": {"risk_level": ["Low", "Medium", "High"][i % 3]}
            })
    
    if not node_type or node_type == "Issue":
        for i in range(min(50, limit)):
            nodes.append({
                "id": f"ISS-{i+1:05d}",
                "label": f"Issue {i+1}",
                "type": "Issue",
                "properties": {"priority": ["Low", "Medium", "High", "Critical"][i % 4]}
            })
    
    return nodes[:limit]


def _get_sample_edges(limit: int) -> List[dict]:
    """Generate sample edges."""
    edges = []
    
    for i in range(min(limit, 100)):
        edges.append({
            "source": f"PAT-{(i % 30)+1:04d}",
            "target": f"SITE-{(i % 20)+1:03d}",
            "relationship": "ENROLLED_AT",
            "properties": {}
        })
        
        edges.append({
            "source": f"ISS-{i+1:05d}",
            "target": f"PAT-{(i % 30)+1:04d}",
            "relationship": "AFFECTS",
            "properties": {"impact": 0.5 + (i % 5) * 0.1}
        })
    
    return edges[:limit]


def _get_sample_cascade_path(issue_id: str, max_depth: int) -> dict:
    """Generate sample cascade path with realistic trial intelligence."""
    nodes = [issue_id]
    edges = []
    
    # Level 1: Patients affected
    for i in range(3):
        node_id = f"PAT-{1000+i}"
        nodes.append(node_id)
        edges.append({
            "source": issue_id,
            "target": node_id,
            "type": "BLOCKS_CLEAN_STATUS"
        })
        
        # Level 2: Downstream Issues (Cascade)
        if max_depth >= 2:
            for j in range(2):
                child_id = f"ISS-{2000+i*10+j}"
                nodes.append(child_id)
                edges.append({
                    "source": node_id,
                    "target": child_id,
                    "type": "CASCADES_TO"
                })
                
                # Level 3: Site/Study Impact
                if max_depth >= 3:
                    grandchild_id = f"SITE-{(i+j) % 20 + 1:03d}"
                    nodes.append(grandchild_id)
                    edges.append({
                        "source": child_id,
                        "target": grandchild_id,
                        "type": "IMPACTS_SITE_DQI"
                    })
    
    return {
        "issue_id": issue_id,
        "nodes": list(set(nodes)),
        "edges": edges,
        "depth": max_depth,
        "total_impact_score": 0.84,
        "affected_patients": 12,
        "affected_sites": 3,
        "unlocked_actions": [
            "Enable Tier-1 Clean Status for 3 patients",
            "Resolve blocking SAE narrative gap",
            "Restore Site 104 performance benchmark"
        ],
        "cascade_enabled": True,
        "source": "simulated_knowledge_graph"
    }


def _get_sample_dependencies(entity_id: str, entity_type: str) -> dict:
    """Generate sample dependencies."""
    return {
        "entity_id": entity_id,
        "entity_type": entity_type,
        "dependencies": {
            "incoming": [
                {"id": "ISS-00001", "type": "Issue", "relationship": "BLOCKS"},
                {"id": "ISS-00002", "type": "Issue", "relationship": "BLOCKS"},
                {"id": "PROC-001", "type": "Process", "relationship": "REQUIRES"}
            ],
            "outgoing": [
                {"id": "SITE-001", "type": "Site", "relationship": "IMPACTS"},
                {"id": "PAT-0001", "type": "Patient", "relationship": "AFFECTS"},
                {"id": "PAT-0002", "type": "Patient", "relationship": "AFFECTS"}
            ]
        },
        "total_dependencies": 6
    }


@router.get("/cascade-issue-types")
async def get_cascade_issue_types(
    study_id: Optional[str] = None,
    site_id: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Get issue counts by type for cascade analysis — real data from UPR/project_issues."""
    try:
        data_service = get_data_service()
        from sqlalchemy import text
        
        with data_service._db_manager.engine.connect() as conn:
            upr_exists = conn.execute(text(
                "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'unified_patient_record')"
            )).scalar()
            
            params = {}
            where = "WHERE 1=1"
            if study_id and str(study_id).lower() not in ['all', 'all studies', 'multiple', '{']:
                where += " AND study_id = :study_id"
                params["study_id"] = str(study_id)
            if site_id and str(site_id).lower() not in ['all', 'multiple', 'all sites']:
                upr_site_id = data_service._map_upr_site_id(site_id)
                where += " AND site_id = :site_id"
                params["site_id"] = str(upr_site_id)
            
            if upr_exists:
                q = f"""
                SELECT
                    SUM(CASE WHEN has_missing_visits = 1 THEN 1 ELSE 0 END)       as missing_visits,
                    SUM(CASE WHEN has_missing_pages = 1 THEN 1 ELSE 0 END)        as missing_pages,
                    SUM(CASE WHEN total_queries > 0 THEN 1 ELSE 0 END)            as open_queries,
                    SUM(CASE WHEN sdv_pending_count > 0 THEN 1 ELSE 0 END)        as sdv_incomplete,
                    SUM(CASE WHEN has_overdue_signatures = 1 THEN 1 ELSE 0 END)   as signature_gaps,
                    SUM(CASE WHEN has_broken_signatures = 1 THEN 1 ELSE 0 END)    as broken_signatures,
                    SUM(CASE WHEN has_sae_dm_pending = 1 THEN 1 ELSE 0 END)       as sae_dm_pending,
                    SUM(CASE WHEN has_sae_safety_pending = 1 THEN 1 ELSE 0 END)   as sae_safety_pending,
                    SUM(CASE WHEN has_meddra_uncoded = 1 THEN 1 ELSE 0 END)       as meddra_uncoded,
                    SUM(CASE WHEN has_whodrug_uncoded = 1 THEN 1 ELSE 0 END)      as whodrug_uncoded,
                    SUM(CASE WHEN has_lab_issues = 1 THEN 1 ELSE 0 END)           as lab_issues,
                    SUM(CASE WHEN edrr_edrr_issue_count > 0 THEN 1 ELSE 0 END)    as edrr_issues,
                    SUM(CASE WHEN has_inactivated_forms = 1 THEN 1 ELSE 0 END)    as inactivated_forms,
                    SUM(CASE WHEN total_queries > 5 THEN 1 ELSE 0 END)            as high_query_volume,
                    COALESCE(SUM(total_open_issues), 0)                           as total_open_issues
                FROM unified_patient_record {where}
                """
            else:
                # Fallback: aggregate from project_issues
                pi_where = "WHERE LOWER(status) = 'open'"
                if study_id and str(study_id).lower() != 'all':
                    pi_where += " AND site_id IN (SELECT site_id FROM patients WHERE study_id = :study_id)"
                q = f"""
                SELECT
                    SUM(CASE WHEN issue_type = 'missing_visit' THEN 1 ELSE 0 END) as missing_visits,
                    SUM(CASE WHEN issue_type = 'missing_page' THEN 1 ELSE 0 END) as missing_pages,
                    SUM(CASE WHEN issue_type = 'query' THEN 1 ELSE 0 END) as open_queries,
                    SUM(CASE WHEN issue_type = 'sdv' THEN 1 ELSE 0 END) as sdv_incomplete,
                    SUM(CASE WHEN issue_type = 'signature' THEN 1 ELSE 0 END) as signature_gaps,
                    0 as broken_signatures, 0 as sae_dm_pending, 0 as sae_safety_pending,
                    0 as meddra_uncoded, 0 as whodrug_uncoded, 0 as lab_issues,
                    0 as edrr_issues, 0 as inactivated_forms, 0 as high_query_volume,
                    COUNT(*) as total_open_issues
                FROM project_issues {pi_where}
                """
            
            row = conn.execute(text(q), params).fetchone()
            
            issue_counts = {
                "missing_visits": int(row[0] or 0),
                "missing_pages": int(row[1] or 0),
                "open_queries": int(row[2] or 0),
                "sdv_incomplete": int(row[3] or 0),
                "signature_gaps": int(row[4] or 0),
                "broken_signatures": int(row[5] or 0),
                "sae_dm_pending": int(row[6] or 0),
                "sae_safety_pending": int(row[7] or 0),
                "meddra_uncoded": int(row[8] or 0),
                "whodrug_uncoded": int(row[9] or 0),
                "lab_issues": int(row[10] or 0),
                "edrr_issues": int(row[11] or 0),
                "inactivated_forms": int(row[12] or 0),
                "high_query_volume": int(row[13] or 0),
                "db_lock": 0  # Target — not an issue count
            }
            total_open = int(row[14] or 0)  # From total_open_issues column
        
        source = "unified_patient_record" if upr_exists else "project_issues"
        
        return {
            "issue_counts": issue_counts,
            "total_issues": total_open,
            "critical_issues": issue_counts["sae_dm_pending"] + issue_counts["sae_safety_pending"],
            "study_id": study_id,
            "site_id": site_id,
            "source": source
        }
        
    except Exception as e:
        logger.error(f"cascade-issue-types error: {e}")
        return {
            "issue_counts": {k: 0 for k in [
                "missing_visits", "missing_pages", "open_queries", "sdv_incomplete",
                "signature_gaps", "broken_signatures", "sae_dm_pending", "sae_safety_pending",
                "meddra_uncoded", "whodrug_uncoded", "lab_issues", "edrr_issues",
                "inactivated_forms", "high_query_volume", "db_lock"
            ]},
            "total_issues": 0,
            "critical_issues": 0,
            "error": str(e),
            "source": "error"
        }


def _get_sample_cascade_analysis() -> List[dict]:
    """Generate sample cascade analysis."""
    analysis = []
    for i in range(50):
        analysis.append({
            "patient_key": f"PAT-{1000+i}",
            "site_id": f"SITE-{(i % 20)+1:03d}",
            "study_id": f"STUDY-{(i % 5)+1:03d}",
            "cascade_impact_score": round(0.3 + (i % 7) * 0.1, 2),
            "open_issues_count": i % 5,
            "open_queries_count": i % 8,
            "dqi_score": 70 + (i % 25),
            "risk_score": 20 + (i % 60),
            "blocking_issues": i % 3,
            "has_issues": i % 3 != 0
        })
    
    return sorted(analysis, key=lambda x: x['cascade_impact_score'], reverse=True)


def _get_sample_visualization_data() -> dict:
    """Generate complete sample visualization data."""
    nodes = []
    edges = []
    
    # Sites
    for i in range(15):
        nodes.append({
            "id": f"SITE-{i+1:03d}",
            "label": f"Site {i+1}",
            "type": "Site",
            "color": "#3498db",
            "size": 30,
            "dqi_score": 75 + (i % 20)
        })
    
    # Patients
    for i in range(40):
        site_idx = i % 15
        nodes.append({
            "id": f"PAT-{i+1:04d}",
            "label": f"P-{i+1}",
            "type": "Patient",
            "color": "#2ecc71",
            "size": 12,
            "dqi_score": 70 + (i % 25)
        })
        edges.append({
            "source": f"PAT-{i+1:04d}",
            "target": f"SITE-{site_idx+1:03d}",
            "type": "ENROLLED_AT"
        })
    
    # Issues
    for i in range(30):
        priority = ["Critical", "High", "Medium", "Low"][i % 4]
        patient_idx = i % 40
        nodes.append({
            "id": f"ISS-{i+1:05d}",
            "label": f"Issue {i+1}",
            "type": "Issue",
            "color": _get_priority_color(priority),
            "size": 15,
            "priority": priority
        })
        edges.append({
            "source": f"ISS-{i+1:05d}",
            "target": f"PAT-{patient_idx+1:04d}",
            "type": "AFFECTS"
        })
    
    return {
        "nodes": nodes,
        "edges": edges,
        "node_count": len(nodes),
        "edge_count": len(edges)
    }
@router.get("/cascade-topology")
async def get_cascade_topology(
    study_id: Optional[str] = None,
    site_id: Optional[str] = None
):
    """
    Get the full topology of issue dependencies for the cascade graph.
    Returns a mapping of issue types and what they block.
    Tries Neo4j first, then falls back to clinical defaults.
    """
    from app.config import settings
    
    # Clinical cascade dependency rules (last-resort fallback)
    default_topology = {
        "missing_visits": ["missing_pages", "open_queries", "sdv_incomplete", "signature_gaps"],
        "missing_pages": ["open_queries", "sdv_incomplete", "signature_gaps"],
        "open_queries": ["signature_gaps", "db_lock"],
        "sdv_incomplete": ["signature_gaps", "db_lock"],
        "signature_gaps": ["db_lock"],
        "broken_signatures": ["signature_gaps", "db_lock"],
        "sae_dm_pending": ["sae_safety_pending", "db_lock"],
        "sae_safety_pending": ["db_lock"],
        "meddra_uncoded": ["db_lock"],
        "whodrug_uncoded": ["db_lock"],
        "lab_issues": ["db_lock"],
        "edrr_issues": ["db_lock"],
        "inactivated_forms": ["db_lock"],
        "high_query_volume": ["open_queries"],
    }
    
    # 1. Try Neo4j graph service
    gs = get_graph_service()
    if gs and gs.is_connected and not gs.uses_mock:
        try:
            neo4j_topology = gs.get_cascade_topology()
            if neo4j_topology:
                return {"source": "neo4j", "topology": neo4j_topology}
            else:
                logger.info("Neo4j returned empty topology, using clinical defaults")
        except Exception as e:
            logger.warning(f"Neo4j topology query failed: {e}")
    
    # 2. Fallback to clinical defaults
    return {"source": "clinical_defaults", "topology": default_topology}


@router.post("/cascade-seed")
async def seed_cascade_graph(
    current_user: dict = Depends(get_current_user)
):
    """Trigger on-demand PostgreSQL → Neo4j sync for cascade data."""
    gs = get_graph_service()
    if not gs or gs.uses_mock:
        return {"status": "skipped", "reason": "Neo4j not available"}
    
    try:
        data_service = get_data_service()
        from sqlalchemy import text
        import pandas as pd
        
        with data_service._db_manager.engine.connect() as conn:
            # 1. Create schema
            gs.create_cascade_schema()
            
            # 2. Seed studies/sites/patients from UPR
            upr_exists = conn.execute(text(
                "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'unified_patient_record')"
            )).scalar()
            
            if upr_exists:
                upr_df = pd.read_sql(text("SELECT * FROM unified_patient_record LIMIT 500"), conn)
                gs.build_entity_graph(upr_df)
                patients_count = len(upr_df)
            else:
                patients_count = 0
            
            # 3. Seed issues
            issues_df = pd.read_sql(text(
                "SELECT issue_id, patient_key, issue_type as type, priority FROM project_issues WHERE status = 'open' LIMIT 1000"
            ), conn)
            if not issues_df.empty:
                gs.add_issues_batch(issues_df)
            
            # 4. Create IssueType aggregate nodes
            all_types = [
                "missing_visits", "missing_pages", "open_queries", "sdv_incomplete",
                "signature_gaps", "broken_signatures", "sae_dm_pending", "sae_safety_pending",
                "meddra_uncoded", "whodrug_uncoded", "lab_issues", "edrr_issues",
                "inactivated_forms", "high_query_volume", "db_lock"
            ]
            with gs._driver.session() as session:
                for type_key in all_types:
                    session.run("""
                        MERGE (t:Issue {id: $type_id, type: $type_key})
                        SET t.priority = CASE $type_key WHEN 'db_lock' THEN 'Target' ELSE 'Cascade' END
                    """, type_id=f"CASCADE_{type_key.upper()}", type_key=type_key)
            
            # 5. Create BLOCKS relationships between aggregate type nodes
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
            
            with gs._driver.session() as session:
                for source, target in cascade_rules:
                    session.run("""
                        MATCH (a:Issue {id: $src_id})
                        MATCH (b:Issue {id: $tgt_id})
                        MERGE (a)-[:BLOCKS]->(b)
                    """, src_id=f"CASCADE_{source.upper()}", tgt_id=f"CASCADE_{target.upper()}")
            
            return {
                "status": "success",
                "patients_seeded": patients_count,
                "issues_seeded": len(issues_df),
                "cascade_rules_applied": len(cascade_rules),
                "type_nodes_created": len(all_types)
            }
    except Exception as e:
        logger.error(f"Cascade seed failed: {e}")
        return {"status": "error", "error": str(e)}
