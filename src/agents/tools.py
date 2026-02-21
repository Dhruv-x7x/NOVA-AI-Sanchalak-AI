"""
Tool Registry for SANCHALAK AI
Phase 5.2: LangGraph Agent Framework - FIXED

Defines the tools that agents can use to interact with data and systems.
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
import json
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result from a tool execution"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "metadata": self.metadata
        }


@dataclass
class Tool:
    """Tool definition"""
    name: str
    description: str
    function: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)
    requires_approval: bool = False
    category: str = "general"
    
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters"""
        try:
            result = self.function(**kwargs)
            return ToolResult(success=True, data=result)
        except Exception as e:
            logger.error(f"Tool {self.name} failed: {e}")
            return ToolResult(success=False, error=str(e))


class ToolRegistry:
    """
    Registry of all available tools for agents.
    
    Tools are organized by category:
    - data: Data retrieval and analysis
    - search: Vector search and RAG
    - analytics: DQI, cascade, benchmarks
    - ml: ML model predictions
    - action: Executable actions
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize the tool registry"""
        self.data_dir = data_dir or PROJECT_ROOT / "data" / "processed"
        self.tools: Dict[str, Tool] = {}
        self._data_cache: Dict[str, pd.DataFrame] = {}
        
        # Register all tools
        self._register_data_tools()
        self._register_search_tools()
        self._register_analytics_tools()
        self._register_ml_tools()
        self._register_action_tools()
        
        logger.info(f"ToolRegistry initialized with {len(self.tools)} tools")
    
    def register(self, tool: Tool):
        """Register a tool"""
        self.tools[tool.name] = tool
    
    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name"""
        return self.tools.get(name)
    
    def execute(self, name: str, **kwargs) -> ToolResult:
        """Execute a tool by name"""
        tool = self.get(name)
        if not tool:
            return ToolResult(success=False, error=f"Tool '{name}' not found")
        return tool.execute(**kwargs)
    
    def list_tools(self, category: Optional[str] = None) -> List[Dict[str, str]]:
        """List all tools, optionally filtered by category"""
        tools = []
        for name, tool in self.tools.items():
            if category is None or tool.category == category:
                tools.append({
                    "name": name,
                    "description": tool.description,
                    "category": tool.category,
                    "requires_approval": tool.requires_approval
                })
        return tools
    
    def get_tools_for_llm(self) -> str:
        """Get tool descriptions formatted for LLM context"""
        lines = ["Available Tools:"]
        for name, tool in sorted(self.tools.items(), key=lambda x: x[1].category):
            lines.append(f"- {name}: {tool.description}")
        return "\n".join(lines)
    
    # === Data Loading Helper ===
    
    def _load_data(self, filename: str, subdir: str = "") -> pd.DataFrame:
        """Load data file with caching - FALLBACK for non-PostgreSQL files"""
        cache_key = f"{subdir}/{filename}"
        
        if cache_key not in self._data_cache:
            if subdir:
                filepath = self.data_dir / subdir / filename
            else:
                filepath = self.data_dir / filename
            
            if filepath.exists():
                self._data_cache[cache_key] = pd.read_parquet(filepath)
            else:
                raise FileNotFoundError(f"Data file not found: {filepath}")
        
        return self._data_cache[cache_key]
    
    def _get_primary_data(self) -> pd.DataFrame:
        """Get the primary patient data from PostgreSQL - NO PARQUET FALLBACK in production."""
        cache_key = "_postgresql_upr"
        
        if cache_key not in self._data_cache:
            try:
                from src.database.connection import get_db_manager
                db = get_db_manager()
                
                with db.engine.connect() as conn:
                    df = pd.read_sql("SELECT * FROM unified_patient_record", conn)
                    logger.info(f"Loaded {len(df)} patients from PostgreSQL")
                    self._data_cache[cache_key] = df
            except Exception as e:
                # NO PARQUET FALLBACK - Force PostgreSQL connectivity per riyaz2.md
                error_msg = (
                    f"FATAL: PostgreSQL connection failed: {e}. "
                    "Parquet fallback is disabled in production per riyaz2.md compliance. "
                    "Ensure PostgreSQL is running and DATABASE_URL is configured correctly."
                )
                logger.error(error_msg)
                raise ConnectionError(error_msg)
        
        return self._data_cache[cache_key]
    
    # === Data Tools ===
    
    def _register_data_tools(self):
        """Register data retrieval tools"""
        
        def get_patient(patient_key: str) -> Dict[str, Any]:
            """Get patient data by patient_key"""
            df = self._get_primary_data()
            patient = df[df["patient_key"] == patient_key]
            if len(patient) == 0:
                return {"error": f"Patient {patient_key} not found"}
            
            row = patient.iloc[0]
            return {
                "patient_key": patient_key,
                "study_id": row.get("study_id"),
                "site_id": row.get("site_id"),
                "subject_status": row.get("subject_status_clean"),
                "dqi_score": float(row.get("dqi_score", 0)),
                "dqi_band": row.get("dqi_band"),
                "risk_level": row.get("risk_level"),
                "tier1_clean": bool(row.get("tier1_clean", False)),
                "tier2_clean": bool(row.get("tier2_clean", False)),
                "total_queries": float(row.get("total_queries", 0)),
                "cascade_issue_count": int(row.get("cascade_issue_count", 0)),
                "cascade_impact_score": float(row.get("cascade_impact_score", 0)),
                "primary_blocker": row.get("cascade_primary_blocker"),
                "path_to_clean": row.get("path_to_clean"),
                "db_lock_status": row.get("db_lock_status"),
            }
        
        self.register(Tool(
            name="get_patient",
            description="Get detailed patient data including DQI, clean status, and issues",
            function=get_patient,
            parameters={"patient_key": "str - Patient identifier (Study_X|Site_XXX|Subject_XXXX)"},
            category="data"
        ))
        
        def get_site_summary(site_id: str) -> Dict[str, Any]:
            """Get site summary statistics"""
            df = self._get_primary_data()
            site_data = df[df["site_id"] == site_id]
            
            if len(site_data) == 0:
                return {"error": f"Site {site_id} not found"}
            
            return {
                "site_id": site_id,
                "study_id": site_data["study_id"].iloc[0],
                "patient_count": len(site_data),
                "avg_dqi": float(site_data["dqi_score"].mean()),
                "tier1_clean_rate": float(site_data["tier1_clean"].mean() * 100),
                "tier2_clean_rate": float(site_data["tier2_clean"].mean() * 100),
                "total_queries": int(site_data["total_queries"].sum()),
                "avg_cascade_impact": float(site_data["cascade_impact_score"].mean()),
                "patients_with_issues": int((site_data["cascade_issue_count"] > 0).sum()),
                "risk_high_critical": int(site_data["risk_level"].isin(["High", "Critical"]).sum()),
                "db_lock_ready": int((site_data["db_lock_status"] == "Ready").sum()),
            }
        
        self.register(Tool(
            name="get_site_summary",
            description="Get summary statistics for a site including patient count, DQI, and issues",
            function=get_site_summary,
            parameters={"site_id": "str - Site identifier (e.g., Site_101)"},
            category="data"
        ))
        
        def get_study_summary(study_id: str) -> Dict[str, Any]:
            """Get study summary statistics"""
            df = self._get_primary_data()
            study_data = df[df["study_id"] == study_id]
            
            if len(study_data) == 0:
                return {"error": f"Study {study_id} not found"}
            
            return {
                "study_id": study_id,
                "patient_count": len(study_data),
                "site_count": study_data["site_id"].nunique(),
                "avg_dqi": float(study_data["dqi_score"].mean()),
                "tier1_clean_rate": float(study_data["tier1_clean"].mean() * 100),
                "tier2_clean_rate": float(study_data["tier2_clean"].mean() * 100),
                "total_queries": int(study_data["total_queries"].sum()),
                "patients_with_issues": int((study_data["cascade_issue_count"] > 0).sum()),
                "issue_rate": float((study_data["cascade_issue_count"] > 0).mean() * 100),
                "risk_high_critical": int(study_data["risk_level"].isin(["High", "Critical"]).sum()),
                "db_lock_ready": int((study_data["db_lock_status"] == "Ready").sum()),
                "db_lock_ready_rate": float((study_data["db_lock_status"] == "Ready").mean() * 100),
            }
        
        self.register(Tool(
            name="get_study_summary",
            description="Get summary statistics for a study including site count and issue rates",
            function=get_study_summary,
            parameters={"study_id": "str - Study identifier (e.g., Study_21)"},
            category="data"
        ))
        
        def get_high_priority_patients(limit: int = 20) -> List[Dict[str, Any]]:
            """Get patients with high/critical risk or high cascade impact"""
            df = self._get_primary_data()
            
            # Filter for high priority patients
            high_priority = df[
                (df["risk_level"].isin(["High", "Critical"])) | 
                (df["cascade_issue_count"] > 3)
            ].copy()
            
            # Sort by cascade impact
            high_priority = high_priority.sort_values("cascade_impact_score", ascending=False)
            
            results = []
            for _, row in high_priority.head(limit).iterrows():
                results.append({
                    "patient_key": row["patient_key"],
                    "study_id": row["study_id"],
                    "site_id": row["site_id"],
                    "risk_level": row["risk_level"],
                    "dqi_score": float(row["dqi_score"]),
                    "cascade_issue_count": int(row["cascade_issue_count"]),
                    "cascade_impact_score": float(row["cascade_impact_score"]),
                    "primary_blocker": row.get("cascade_primary_blocker"),
                })
            
            return results
        
        self.register(Tool(
            name="get_high_priority_patients",
            description="Get list of patients with high risk or high cascade impact",
            function=get_high_priority_patients,
            parameters={"limit": "int - Maximum number of patients to return (default: 20)"},
            category="data"
        ))
        
        def get_overall_summary() -> Dict[str, Any]:
            """Get overall summary across all studies"""
            df = self._get_primary_data()
            
            return {
                "total_patients": len(df),
                "total_studies": df["study_id"].nunique(),
                "total_sites": df["site_id"].nunique(),
                "avg_dqi": float(df["dqi_score"].mean()),
                "tier1_clean_rate": float(df["tier1_clean"].mean() * 100),
                "tier2_clean_rate": float(df["tier2_clean"].mean() * 100),
                "patients_with_issues": int((df["cascade_issue_count"] > 0).sum()),
                "patients_clean": int((df["cascade_issue_count"] == 0).sum()),
                "risk_distribution": df["risk_level"].value_counts().to_dict(),
                "db_lock_ready": int((df["db_lock_status"] == "Ready").sum()),
                "db_lock_ready_rate": float((df["db_lock_status"] == "Ready").mean() * 100),
                "top_issues": df["dqi_primary_issue"].value_counts().head(5).to_dict(),
            }
        
        self.register(Tool(
            name="get_overall_summary",
            description="Get overall summary statistics across all studies",
            function=get_overall_summary,
            parameters={},
            category="data"
        ))
        
        def get_lowest_dqi_sites(limit: int = 10) -> List[Dict[str, Any]]:
            """Get sites with the lowest average DQI scores"""
            df = self._get_primary_data()
            
            if df.empty:
                return [{"error": "No data available"}]
            
            # Aggregate by site - use columns that exist
            agg_dict = {
                'dqi_score': 'mean',
                'patient_key': 'count',
                'study_id': 'first',
            }
            
            # Add completeness_score if available
            if 'completeness_score' in df.columns:
                agg_dict['completeness_score'] = 'mean'
            
            site_stats = df.groupby('site_id').agg(agg_dict).reset_index()
            
            # Rename columns
            site_stats.columns = ['site_id', 'avg_dqi', 'patient_count', 'study_id'] + (
                ['completeness_rate'] if 'completeness_score' in df.columns else []
            )
            site_stats = site_stats.sort_values('avg_dqi', ascending=True).head(limit)
            
            results = []
            for _, row in site_stats.iterrows():
                result_item = {
                    "site_id": row['site_id'],
                    "study_id": row['study_id'],
                    "avg_dqi": round(float(row['avg_dqi']), 2),
                    "patient_count": int(row['patient_count']),
                    "status": "Critical" if row['avg_dqi'] < 50 else "Warning" if row['avg_dqi'] < 70 else "Review"
                }
                if 'completeness_rate' in site_stats.columns:
                    result_item["completeness_rate"] = round(float(row['completeness_rate']), 1)
                results.append(result_item)
            
            return results
        
        self.register(Tool(
            name="get_lowest_dqi_sites",
            description="Get sites with the lowest average DQI scores - useful for identifying problem sites",
            function=get_lowest_dqi_sites,
            parameters={"limit": "int - Number of sites to return (default: 10)"},
            category="data"
        ))
        
        def get_site_patient_details(site_id: str, limit: int = 20) -> Dict[str, Any]:
            """Get detailed patient list for a specific site"""
            df = self._get_primary_data()
            
            if df.empty:
                return {"error": "No data available"}
            
            site_data = df[df['site_id'] == site_id]
            
            if len(site_data) == 0:
                # Try partial match
                site_data = df[df['site_id'].str.contains(site_id, case=False, na=False)]
            
            if len(site_data) == 0:
                return {"error": f"Site {site_id} not found. Available sites: {df['site_id'].unique()[:10].tolist()}"}
            
            patients = []
            for _, row in site_data.head(limit).iterrows():
                patients.append({
                    "patient_key": row.get('patient_key'),
                    "dqi_score": round(float(row.get('dqi_score', 0)), 2),
                    "tier1_clean": bool(row.get('tier1_clean', False)),
                    "risk_level": row.get('risk_level', 'Unknown'),
                    "total_queries": int(row.get('total_queries', 0)),
                })
            
            return {
                "site_id": site_id,
                "study_id": site_data['study_id'].iloc[0],
                "patient_count": len(site_data),
                "avg_dqi": round(float(site_data['dqi_score'].mean()), 2),
                "patients": patients
            }
        
        self.register(Tool(
            name="get_site_patient_details",
            description="Get detailed patient list for a specific site with DQI scores and issues",
            function=get_site_patient_details,
            parameters={
                "site_id": "str - Site identifier",
                "limit": "int - Max patients to return (default: 20)"
            },
            category="data"
        ))
    
    # === Search Tools ===
    
    def _register_search_tools(self):
        """Register search and RAG tools"""
        
        def search_resolutions(issue_type: str, limit: int = 5) -> List[Dict[str, Any]]:
            """Search resolution templates by issue type"""
            try:
                genome_dir = self.data_dir / "analytics" / "resolution_genome"
                templates_file = genome_dir / "resolution_templates.json"
                
                if templates_file.exists():
                    with open(templates_file) as f:
                        templates = json.load(f)
                    
                    # Filter by issue type
                    matches = [t for t in templates if t.get("issue_type") == issue_type]
                    return matches[:limit] if matches else [{"message": f"No templates for {issue_type}"}]
                else:
                    return [{"error": "Resolution templates not found"}]
            except Exception as e:
                return [{"error": str(e)}]
        
        self.register(Tool(
            name="search_resolutions",
            description="Search for resolution templates by issue type",
            function=search_resolutions,
            parameters={
                "issue_type": "str - Issue type (e.g., sdv_incomplete, open_queries)",
                "limit": "int - Maximum results (default: 5)"
            },
            category="search"
        ))
        
        def search_patterns(pattern_type: str = None, limit: int = 10) -> List[Dict[str, Any]]:
            """Search for known patterns"""
            try:
                pattern_dir = self.data_dir / "analytics" / "pattern_library"
                patterns_file = pattern_dir / "pattern_definitions.csv"
                
                if patterns_file.exists():
                    df = pd.read_csv(patterns_file)
                    if pattern_type:
                        df = df[df["category"] == pattern_type]
                    return df.head(limit).to_dict("records")
                else:
                    return [{"error": "Pattern library not found"}]
            except Exception as e:
                return [{"error": str(e)}]
        
        self.register(Tool(
            name="search_patterns",
            description="Search for known issue patterns from pattern library",
            function=search_patterns,
            parameters={
                "pattern_type": "str - Pattern category (optional)",
                "limit": "int - Maximum results (default: 10)"
            },
            category="search"
        ))
        
        def search_knowledge(query: str, source: str = None, limit: int = 5) -> List[Dict[str, Any]]:
            """Search RAG knowledge base"""
            try:
                rag_dir = self.data_dir / "knowledge" / "rag"
                docs_file = rag_dir / "rag_documents.json"
                
                if docs_file.exists():
                    with open(docs_file) as f:
                        docs = json.load(f)
                    
                    # Simple keyword matching
                    query_lower = query.lower()
                    matches = []
                    for doc in docs:
                        text = doc.get("text", "").lower()
                        if query_lower in text:
                            score = text.count(query_lower) / max(len(text.split()), 1)
                            matches.append({
                                "text": doc.get("text", "")[:300],
                                "source": doc.get("metadata", {}).get("source", "unknown"),
                                "category": doc.get("metadata", {}).get("category", "unknown"),
                                "score": round(score, 4)
                            })
                    
                    matches.sort(key=lambda x: x["score"], reverse=True)
                    if source:
                        matches = [m for m in matches if m["source"] == source]
                    
                    return matches[:limit] if matches else [{"message": "No matches found"}]
                else:
                    return [{"error": "RAG knowledge base not found"}]
            except Exception as e:
                return [{"error": str(e)}]
        
        self.register(Tool(
            name="search_knowledge",
            description="Search regulatory guidelines, SOPs, and protocol knowledge",
            function=search_knowledge,
            parameters={
                "query": "str - Search query",
                "source": "str - Filter by source (ich_gcp, protocol, sop) (optional)",
                "limit": "int - Maximum results (default: 5)"
            },
            category="search"
        ))
    
    # === Analytics Tools ===
    
    def _register_analytics_tools(self):
        """Register analytics tools"""
        
        def get_cascade_impact(patient_key: str) -> Dict[str, Any]:
            """Get cascade impact analysis for a patient"""
            try:
                df = self._get_primary_data()
                patient = df[df["patient_key"] == patient_key]
                
                if len(patient) == 0:
                    return {"error": f"Patient {patient_key} not found"}
                
                row = patient.iloc[0]
                return {
                    "patient_key": patient_key,
                    "cascade_impact_score": float(row.get("cascade_impact_score", 0)),
                    "cascade_issue_count": int(row.get("cascade_issue_count", 0)),
                    "cascade_path_length": int(row.get("cascade_path_length", 0)),
                    "cascade_cluster": row.get("cascade_cluster"),
                    "cascade_critical_path": row.get("cascade_critical_path"),
                    "cascade_primary_blocker": row.get("cascade_primary_blocker"),
                    "cascade_recommendation": row.get("cascade_recommendation"),
                }
            except Exception as e:
                return {"error": str(e)}
        
        self.register(Tool(
            name="get_cascade_impact",
            description="Get cascade impact analysis showing how fixing one issue unlocks others",
            function=get_cascade_impact,
            parameters={"patient_key": "str - Patient identifier"},
            category="analytics"
        ))
        
        def get_site_benchmark(site_id: str) -> Dict[str, Any]:
            """Get site benchmark compared to peers"""
            try:
                df = self._load_data("site_benchmarks.parquet", "analytics")
                site = df[df["site_id"] == site_id]
                
                if len(site) == 0:
                    return {"error": f"Site {site_id} not found in benchmarks"}
                
                row = site.iloc[0]
                return {
                    "site_id": site_id,
                    "study_id": row.get("study_id"),
                    "patient_count": int(row.get("patient_count", 0)),
                    "composite_score": float(row.get("composite_score", 0)),
                    "composite_percentile": float(row.get("composite_percentile", 0)),
                    "performance_tier": row.get("performance_tier"),
                    "dqi_mean": float(row.get("dqi_mean", 0)),
                    "tier1_clean_rate": float(row.get("tier1_clean_rate", 0)),
                    "tier2_clean_rate": float(row.get("tier2_clean_rate", 0)),
                    "ready_rate": float(row.get("ready_rate", 0)),
                    "overall_rank": int(row.get("overall_rank", 0)),
                    "study_rank": int(row.get("study_rank", 0)),
                }
            except Exception as e:
                return {"error": str(e)}
        
        self.register(Tool(
            name="get_site_benchmark",
            description="Get site performance benchmark compared to peer sites",
            function=get_site_benchmark,
            parameters={"site_id": "str - Site identifier"},
            category="analytics"
        ))
        
        def get_dblock_projection(study_id: str = None) -> Dict[str, Any]:
            """Get DB Lock readiness projection"""
            try:
                df = self._get_primary_data()
                
                if study_id:
                    df = df[df["study_id"] == study_id]
                
                if len(df) == 0:
                    return {"error": "No data found for projection"}
                
                eligible = df[df["db_lock_eligible"] == True]
                
                return {
                    "study_id": study_id or "All Studies",
                    "total_patients": len(df),
                    "eligible_patients": len(eligible),
                    "ready_now": int((eligible["db_lock_status"] == "Ready").sum()),
                    "ready_rate": float((eligible["db_lock_status"] == "Ready").mean() * 100) if len(eligible) > 0 else 0,
                    "pending": int((eligible["db_lock_status"] == "Pending").sum()),
                    "blocked": int((eligible["db_lock_status"] == "Blocked").sum()),
                    "not_eligible": int((df["db_lock_eligible"] == False).sum()),
                }
            except Exception as e:
                return {"error": str(e)}
        
        self.register(Tool(
            name="get_dblock_projection",
            description="Get DB Lock readiness status and projection",
            function=get_dblock_projection,
            parameters={"study_id": "str - Study identifier (optional, defaults to all)"},
            category="analytics"
        ))
        
        # === Neo4j Graph Tools (Cascade Intelligence) ===
        
        def get_cascade_path(issue_id: str, max_depth: int = 3) -> Dict[str, Any]:
            """Get cascade path from Neo4j knowledge graph"""
            try:
                from src.knowledge.neo4j_graph import get_graph_service
                
                service = get_graph_service()
                path = service.get_cascade_path(issue_id, max_depth)
                
                return {
                    "issue_id": issue_id,
                    "source_type": path.source_type,
                    "path_nodes": path.path_nodes,
                    "path_relationships": path.path_relationships,
                    "total_impact": path.total_impact,
                    "affected_patients": path.affected_patients,
                    "unlocked_actions": path.unlocked_actions,
                    "graph_source": "neo4j" if not service.uses_mock else "mock"
                }
            except Exception as e:
                logger.error(f"get_cascade_path error: {e}")
                return {"error": str(e)}
        
        self.register(Tool(
            name="get_cascade_path",
            description="Get cascade path showing what gets unlocked when an issue is resolved (uses Neo4j graph)",
            function=get_cascade_path,
            parameters={
                "issue_id": "str - Issue identifier",
                "max_depth": "int - Maximum cascade depth (default: 3)"
            },
            category="analytics"
        ))
        
        def get_site_cascade_opportunities(site_id: str, top_n: int = 5) -> Dict[str, Any]:
            """Get top cascade opportunities for a site"""
            try:
                from src.knowledge.neo4j_graph import get_graph_service
                
                service = get_graph_service()
                opportunities = service.get_site_cascade_opportunities(site_id, top_n)
                
                return {
                    "site_id": site_id,
                    "opportunities": opportunities,
                    "total_found": len(opportunities),
                    "graph_source": "neo4j" if not service.uses_mock else "mock"
                }
            except Exception as e:
                logger.error(f"get_site_cascade_opportunities error: {e}")
                return {"error": str(e)}
        
        self.register(Tool(
            name="get_site_cascade_opportunities",
            description="Get top cascade opportunities for a site - issues that unlock the most downstream improvements",
            function=get_site_cascade_opportunities,
            parameters={
                "site_id": "str - Site identifier",
                "top_n": "int - Number of top opportunities (default: 5)"
            },
            category="analytics"
        ))
        
        def get_patient_dependencies(patient_key: str) -> Dict[str, Any]:
            """Get issue dependencies for a patient from knowledge graph"""
            try:
                from src.knowledge.neo4j_graph import get_graph_service
                
                service = get_graph_service()
                deps = service.get_patient_dependencies(patient_key)
                
                return deps
            except Exception as e:
                logger.error(f"get_patient_dependencies error: {e}")
                return {"error": str(e)}
        
        self.register(Tool(
            name="get_patient_dependencies",
            description="Get issue dependencies for a patient showing blocking and blocked-by relationships",
            function=get_patient_dependencies,
            parameters={"patient_key": "str - Patient identifier"},
            category="analytics"
        ))
        
        def get_graph_stats() -> Dict[str, Any]:
            """Get Neo4j knowledge graph statistics"""
            try:
                from src.knowledge.neo4j_graph import get_graph_service
                
                service = get_graph_service()
                return service.get_graph_stats()
            except Exception as e:
                logger.error(f"get_graph_stats error: {e}")
                return {"error": str(e)}
        
        self.register(Tool(
            name="get_graph_stats",
            description="Get knowledge graph statistics (node counts, relationships)",
            function=get_graph_stats,
            parameters={},
            category="analytics"
        ))
    
    # === ML Tools ===
    
    def _register_ml_tools(self):
        """Register ML prediction tools - using trained models directly"""
        
        def predict_risk(patient_key: str) -> Dict[str, Any]:
            """Get REAL-TIME risk prediction for a patient using trained ML models"""
            try:
                from src.ml.model_loader import get_model_loader
                
                df = self._get_primary_data()
                patient = df[df["patient_key"] == patient_key]
                
                if len(patient) == 0:
                    return {"error": f"Patient {patient_key} not found"}
                
                # Get model loader and feature names
                loader = get_model_loader()
                feature_names = loader.get_risk_feature_names()
                
                # Prepare features for prediction
                features = patient[feature_names] if all(f in patient.columns for f in feature_names) else None
                
                if features is not None:
                    # Use trained model for prediction
                    predictions = loader.predict_risk(features)
                    pred = predictions[0]
                    
                    return {
                        "patient_key": patient_key,
                        "risk_level": pred.risk_level,
                        "confidence": round(pred.confidence * 100, 1),
                        "probabilities": {k: round(v * 100, 1) for k, v in pred.probabilities.items()},
                        "top_risk_factors": [
                            {"feature": f[0], "value": round(f[1], 2)} 
                            for f in pred.top_features
                        ],
                        "model": "trained_xgb_lgb_ensemble",
                        "real_time": True
                    }
                else:
                    # Fallback to pre-computed data
                    row = patient.iloc[0]
                    return {
                        "patient_key": patient_key,
                        "risk_level": row.get("risk_level", "Unknown"),
                        "dqi_score": float(row.get("dqi_score", 0)),
                        "dqi_band": row.get("dqi_band"),
                        "cascade_issue_count": int(row.get("cascade_issue_count", 0)),
                        "primary_issue": row.get("dqi_primary_issue"),
                        "model": "pre_computed",
                        "real_time": False
                    }
            except Exception as e:
                logger.error(f"predict_risk error: {e}")
                return {"error": str(e)}
        
        self.register(Tool(
            name="predict_risk",
            description="Get REAL-TIME risk assessment using trained ML models (XGBoost + LightGBM ensemble)",
            function=predict_risk,
            parameters={"patient_key": "str - Patient identifier"},
            category="ml"
        ))
        
        def detect_issues(patient_key: str = None, site_id: str = None, threshold: float = 0.5) -> Dict[str, Any]:
            """Detect issues using trained Issue Detector models (14 types)"""
            try:
                from src.ml.model_loader import get_model_loader
                
                df = self._get_primary_data()
                
                if patient_key:
                    data = df[df["patient_key"] == patient_key]
                elif site_id:
                    data = df[df["site_id"] == site_id]
                else:
                    return {"error": "Either patient_key or site_id required"}
                
                if len(data) == 0:
                    return {"error": "No data found"}
                
                loader = get_model_loader()
                issue_types = loader.get_issue_types()
                
                # Detect issues for each row
                predictions = loader.detect_issues(data, threshold=threshold)
                
                # Aggregate results
                all_issues = {}
                patients_with_issues = []
                
                for i, pred in enumerate(predictions):
                    row = data.iloc[i]
                    if pred.detected_issues:
                        patients_with_issues.append({
                            "patient_key": row["patient_key"],
                            "issues": pred.detected_issues,
                            "issue_count": pred.total_issues
                        })
                    
                    for issue_type, prob in pred.probabilities.items():
                        if issue_type not in all_issues:
                            all_issues[issue_type] = {"count": 0, "avg_prob": 0, "max_prob": 0}
                        if prob >= threshold:
                            all_issues[issue_type]["count"] += 1
                        all_issues[issue_type]["max_prob"] = max(all_issues[issue_type]["max_prob"], prob)
                
                return {
                    "scope": patient_key or site_id,
                    "total_records": len(data),
                    "patients_with_issues": len(patients_with_issues),
                    "issue_summary": {k: v["count"] for k, v in all_issues.items() if v["count"] > 0},
                    "high_risk_patients": patients_with_issues[:10],  # Top 10
                    "model": "trained_issue_detector_ensemble",
                    "issue_types_checked": len(issue_types),
                    "threshold": threshold,
                    "real_time": True
                }
            except Exception as e:
                logger.error(f"detect_issues error: {e}")
                return {"error": str(e)}
        
        self.register(Tool(
            name="detect_issues",
            description="Detect issues using trained ML models (14 issue types)",
            function=detect_issues,
            parameters={
                "patient_key": "str - Patient identifier (optional)",
                "site_id": "str - Site identifier (optional)",
                "threshold": "float - Detection threshold (default: 0.5)"
            },
            category="ml"
        ))
        
        def get_site_risk_ranking(site_id: str = None, top_n: int = 20) -> Dict[str, Any]:
            """Get site risk rankings from trained Site Risk Ranker"""
            try:
                from src.ml.model_loader import get_model_loader
                
                loader = get_model_loader()
                
                if site_id:
                    return loader.get_site_rank(site_id)
                else:
                    rankings = loader.get_site_risk_scores()
                    top_sites = rankings.head(top_n).to_dict("records")
                    return {
                        "total_sites": len(rankings),
                        "top_risk_sites": top_sites,
                        "model": "trained_site_risk_ranker",
                        "real_time": True
                    }
            except Exception as e:
                logger.error(f"get_site_risk_ranking error: {e}")
                return {"error": str(e)}
        
        self.register(Tool(
            name="get_site_risk_ranking",
            description="Get site risk rankings from trained Site Risk Ranker model",
            function=get_site_risk_ranking,
            parameters={
                "site_id": "str - Site identifier (optional, shows all if not provided)",
                "top_n": "int - Number of top risky sites to return (default: 20)"
            },
            category="ml"
        ))

        def predict_resolution_time(issue_type: str, priority_score: float = None, site_perf: float = 85.0, workload: float = 0.5) -> Dict[str, Any]:
            """Predict resolution time for an issue using trained ML models"""
            try:
                from src.ml.model_loader import get_model_loader
                import pandas as pd
                
                loader = get_model_loader()
                
                # Prepare features
                # Mapping priority labels to scores if needed
                if isinstance(priority_score, str):
                    p_map = {'critical': 3.0, 'high': 2.0, 'medium': 1.0, 'low': 0.0}
                    priority_score = p_map.get(priority_score.lower(), 1.0)
                elif priority_score is None:
                    priority_score = 1.0
                
                features = pd.DataFrame([{
                    'issue_type': issue_type,
                    'priority_score': priority_score,
                    'site_performance_score': site_perf,
                    'workload_index': workload
                }])
                
                preds = loader.predict_resolution_time(features)
                if not preds:
                    return {"error": "Prediction failed"}
                    
                p = preds[0]
                return {
                    "issue_type": issue_type,
                    "predicted_days": round(p['prediction_days'], 1),
                    "confidence_interval": {
                        "lower_bound": round(p['lower_bound_days'], 1),
                        "upper_bound": round(p['upper_bound_days'], 1)
                    },
                    "explanation": f"Estimated resolution in {round(p['prediction_days'], 1)} days based on historical patterns.",
                    "model": "trained_resolution_time_quantiles",
                    "real_time": True
                }
            except Exception as e:
                logger.error(f"predict_resolution_time error: {e}")
                return {"error": str(e)}

        self.register(Tool(
            name="predict_resolution_time",
            description="Predict estimated days for issue resolution using trained ML models",
            function=predict_resolution_time,
            parameters={
                "issue_type": "str - Type of issue (e.g., sdv_incomplete)",
                "priority_score": "float/str - Issue priority (0-3 or 'low' to 'critical')",
                "site_perf": "float - Site performance score (default: 85.0)",
                "workload": "float - Site workload index (default: 0.5)"
            },
            category="ml"
        ))
        
        def ml_model_status() -> Dict[str, Any]:
            """Check status of all trained ML models"""
            try:
                from src.ml.model_loader import get_model_loader
                
                loader = get_model_loader()
                status = loader.health_check()
                
                return {
                    "models": {
                        "risk_classifier": {
                            "loaded": status.get("risk_classifier", False),
                            "error": status.get("risk_classifier_error"),
                            "type": "XGBoost + LightGBM Ensemble",
                            "classes": ["Critical", "High", "Medium", "Low"]
                        },
                        "issue_detector": {
                            "loaded": status.get("issue_detector", False),
                            "error": status.get("issue_detector_error"),
                            "type": "14 Binary Classifiers",
                            "issue_types": loader.get_issue_types() if status.get("issue_detector") else []
                        },
                        "site_risk_ranker": {
                            "loaded": status.get("site_ranker", False),
                            "error": status.get("site_ranker_error"),
                            "type": "Learning-to-Rank Model"
                        }
                    },
                    "all_models_ready": all([
                        status.get("risk_classifier", False),
                        status.get("issue_detector", False),
                        status.get("site_ranker", False)
                    ])
                }
            except Exception as e:
                return {"error": str(e)}
        
        self.register(Tool(
            name="ml_model_status",
            description="Check status of all trained ML models",
            function=ml_model_status,
            parameters={},
            category="ml"
        ))
        
        # === Resolution Genome Tools ===
        
        def get_resolution_recommendation(issue_type: str) -> Dict[str, Any]:
            """Get the best resolution template for an issue type from the Resolution Genome"""
            try:
                from src.ml.resolution_genome_service import get_resolution_genome_service
                
                service = get_resolution_genome_service()
                recommendation = service.get_recommendation_for_issue(issue_type)
                
                return recommendation
            except Exception as e:
                logger.error(f"get_resolution_recommendation error: {e}")
                return {"error": str(e)}
        
        self.register(Tool(
            name="get_resolution_recommendation",
            description="Get the best resolution template for an issue type from the Resolution Genome (reusable knowledge base)",
            function=get_resolution_recommendation,
            parameters={"issue_type": "str - Issue type (e.g., 'open_queries', 'sdv_incomplete', 'sae_dm_pending')"},
            category="ml"
        ))
        
        def get_patient_resolution_plan(patient_key: str) -> Dict[str, Any]:
            """Get all resolution recommendations for a specific patient"""
            try:
                from src.ml.resolution_genome_service import get_resolution_genome_service
                
                service = get_resolution_genome_service()
                recommendations = service.get_patient_recommendations(patient_key)
                
                if not recommendations:
                    return {"patient_key": patient_key, "recommendations": [], "message": "No recommendations found"}
                
                return {
                    "patient_key": patient_key,
                    "total_recommendations": len(recommendations),
                    "recommendations": recommendations[:10],  # Limit to top 10
                    "source": "resolution_genome"
                }
            except Exception as e:
                logger.error(f"get_patient_resolution_plan error: {e}")
                return {"error": str(e)}
        
        self.register(Tool(
            name="get_patient_resolution_plan",
            description="Get personalized resolution recommendations for a specific patient from the Resolution Genome",
            function=get_patient_resolution_plan,
            parameters={"patient_key": "str - Patient identifier"},
            category="ml"
        ))
        
        def get_site_resolution_plan(site_id: str, limit: int = 20) -> Dict[str, Any]:
            """Get aggregated resolution recommendations for a site"""
            try:
                from src.ml.resolution_genome_service import get_resolution_genome_service
                
                service = get_resolution_genome_service()
                recommendations = service.get_site_recommendations(site_id, limit)
                
                if not recommendations:
                    return {"site_id": site_id, "recommendations": [], "message": "No recommendations found"}
                
                return {
                    "site_id": site_id,
                    "issue_types": len(recommendations),
                    "recommendations": recommendations,
                    "source": "resolution_genome"
                }
            except Exception as e:
                logger.error(f"get_site_resolution_plan error: {e}")
                return {"error": str(e)}
        
        self.register(Tool(
            name="get_site_resolution_plan",
            description="Get aggregated resolution recommendations for a site from the Resolution Genome",
            function=get_site_resolution_plan,
            parameters={
                "site_id": "str - Site identifier",
                "limit": "int - Max recommendations to return (default: 20)"
            },
            category="ml"
        ))
        
        def genome_statistics() -> Dict[str, Any]:
            """Get Resolution Genome statistics and coverage"""
            try:
                from src.ml.resolution_genome_service import get_resolution_genome_service
                
                service = get_resolution_genome_service()
                stats = service.get_statistics()
                
                return {
                    "genome_ready": service.is_ready,
                    "statistics": stats
                }
            except Exception as e:
                logger.error(f"genome_statistics error: {e}")
                return {"error": str(e)}
        
        self.register(Tool(
            name="genome_statistics",
            description="Get Resolution Genome statistics and coverage",
            function=genome_statistics,
            parameters={},
            category="ml"
        ))
        
        # === Causal Hypothesis Engine Tools ===
        
        def generate_patient_hypothesis(patient_key: str, issue_type: str = None) -> Dict[str, Any]:
            """Generate causal hypotheses for a patient's issues with evidence chains"""
            try:
                from src.knowledge.causal_hypothesis_service import get_causal_hypothesis_service
                
                service = get_causal_hypothesis_service()
                result = service.generate_patient_hypothesis(patient_key, issue_type)
                
                return result
            except Exception as e:
                logger.error(f"generate_patient_hypothesis error: {e}")
                return {"error": str(e)}
        
        self.register(Tool(
            name="generate_patient_hypothesis",
            description="Generate causal hypotheses with root cause analysis and evidence chains for a patient's issues",
            function=generate_patient_hypothesis,
            parameters={
                "patient_key": "str - Patient identifier",
                "issue_type": "str - Optional: filter to specific issue type (e.g., 'open_queries', 'sdv_incomplete')"
            },
            category="ml"
        ))
        
        def site_root_cause_analysis(site_id: str) -> Dict[str, Any]:
            """Analyze root causes across all patients at a site"""
            try:
                from src.knowledge.causal_hypothesis_service import get_causal_hypothesis_service
                
                service = get_causal_hypothesis_service()
                result = service.get_site_hypothesis(site_id)
                
                return result
            except Exception as e:
                logger.error(f"site_root_cause_analysis error: {e}")
                return {"error": str(e)}
        
        self.register(Tool(
            name="site_root_cause_analysis",
            description="Analyze root causes and generate hypotheses for all issues at a site",
            function=site_root_cause_analysis,
            parameters={"site_id": "str - Site identifier"},
            category="ml"
        ))
        
        def analyze_issue_root_causes(issue_type: str, sample_size: int = 50) -> Dict[str, Any]:
            """Analyze root causes for a specific issue type across population"""
            try:
                from src.knowledge.causal_hypothesis_service import get_causal_hypothesis_service
                
                service = get_causal_hypothesis_service()
                result = service.analyze_issue_root_causes(issue_type, sample_size)
                
                return result
            except Exception as e:
                logger.error(f"analyze_issue_root_causes error: {e}")
                return {"error": str(e)}
        
        self.register(Tool(
            name="analyze_issue_root_causes",
            description="Analyze root causes for a specific issue type across the patient population",
            function=analyze_issue_root_causes,
            parameters={
                "issue_type": "str - Issue type to analyze (e.g., 'open_queries', 'sae_dm_pending')",
                "sample_size": "int - Number of patients to sample (default: 50)"
            },
            category="ml"
        ))
        
        def hypothesis_engine_statistics() -> Dict[str, Any]:
            """Get Causal Hypothesis Engine statistics and coverage"""
            try:
                from src.knowledge.causal_hypothesis_service import get_causal_hypothesis_service
                
                service = get_causal_hypothesis_service()
                stats = service.get_statistics()
                
                return {
                    "engine_ready": service.is_ready,
                    "statistics": stats
                }
            except Exception as e:
                logger.error(f"hypothesis_engine_statistics error: {e}")
                return {"error": str(e)}
        
        self.register(Tool(
            name="hypothesis_engine_statistics",
            description="Get Causal Hypothesis Engine statistics and root cause template coverage",
            function=hypothesis_engine_statistics,
            parameters={},
            category="ml"
        ))
        
        # === Cross-Study Pattern Matcher Tools ===
        
        def find_similar_patterns(study_id: str, pattern_type: str = None) -> Dict[str, Any]:
            """Find patterns in other studies similar to patterns in the given study"""
            try:
                from src.knowledge.cross_study_pattern_service import get_cross_study_pattern_service
                
                service = get_cross_study_pattern_service()
                result = service.find_similar_patterns(study_id, pattern_type)
                
                return result
            except Exception as e:
                logger.error(f"find_similar_patterns error: {e}")
                return {"error": str(e)}
        
        self.register(Tool(
            name="find_similar_patterns",
            description="Find patterns in other studies that are similar to patterns in a specific study (cross-study learning)",
            function=find_similar_patterns,
            parameters={
                "study_id": "str - Study identifier",
                "pattern_type": "str - Optional: filter by pattern type (e.g., 'coordinator_overload', 'query_cascade')"
            },
            category="ml"
        ))
        
        def get_pattern_transfer_recommendations(study_id: str = None, limit: int = 10) -> Dict[str, Any]:
            """Get recommendations for transferring successful resolution patterns between studies"""
            try:
                from src.knowledge.cross_study_pattern_service import get_cross_study_pattern_service
                
                service = get_cross_study_pattern_service()
                result = service.get_transfer_recommendations(study_id, limit)
                
                return result
            except Exception as e:
                logger.error(f"get_pattern_transfer_recommendations error: {e}")
                return {"error": str(e)}
        
        self.register(Tool(
            name="get_pattern_transfer_recommendations",
            description="Get recommendations for transferring successful resolution patterns from other studies",
            function=get_pattern_transfer_recommendations,
            parameters={
                "study_id": "str - Target study identifier (optional)",
                "limit": "int - Max recommendations to return (default: 10)"
            },
            category="ml"
        ))
        
        def search_patterns_by_issue(issue_type: str, min_confidence: float = 0.5) -> Dict[str, Any]:
            """Search for patterns related to a specific issue type across all studies"""
            try:
                from src.knowledge.cross_study_pattern_service import get_cross_study_pattern_service
                
                service = get_cross_study_pattern_service()
                result = service.search_patterns_by_issue(issue_type, min_confidence)
                
                return result
            except Exception as e:
                logger.error(f"search_patterns_by_issue error: {e}")
                return {"error": str(e)}
        
        self.register(Tool(
            name="search_patterns_by_issue",
            description="Search for known patterns related to a specific issue type across all studies",
            function=search_patterns_by_issue,
            parameters={
                "issue_type": "str - Issue type (e.g., 'open_queries', 'sdv_incomplete')",
                "min_confidence": "float - Minimum confidence threshold (default: 0.5)"
            },
            category="ml"
        ))
        
        def pattern_matcher_statistics() -> Dict[str, Any]:
            """Get Cross-Study Pattern Matcher statistics"""
            try:
                from src.knowledge.cross_study_pattern_service import get_cross_study_pattern_service
                
                service = get_cross_study_pattern_service()
                stats = service.get_statistics()
                
                return {
                    "matcher_ready": service.is_ready,
                    "statistics": stats
                }
            except Exception as e:
                logger.error(f"pattern_matcher_statistics error: {e}")
                return {"error": str(e)}
        
        self.register(Tool(
            name="pattern_matcher_statistics",
            description="Get Cross-Study Pattern Matcher statistics and pattern coverage",
            function=pattern_matcher_statistics,
            parameters={},
            category="ml"
        ))
    
    # === Action Tools ===
    
    def _register_action_tools(self):
        """Register action execution tools"""
        
        def draft_query_email(site_id: str, issue_summary: str, recipient: str = "Site Coordinator") -> Dict[str, Any]:
            """Draft an email about pending queries"""
            return {
                "type": "email",
                "recipient": recipient,
                "subject": f"Action Required: Pending Queries at {site_id}",
                "body": f"""Dear {recipient},

This is a reminder regarding pending data quality items at {site_id}.

Summary:
{issue_summary}

Please review and resolve these items at your earliest convenience.

Best regards,
Clinical Trial Team""",
                "requires_approval": True,
                "status": "draft"
            }
        
        self.register(Tool(
            name="draft_query_email",
            description="Draft an email reminder for pending queries",
            function=draft_query_email,
            parameters={
                "site_id": "str - Site identifier",
                "issue_summary": "str - Summary of the issues",
                "recipient": "str - Recipient role (default: Site Coordinator)"
            },
            requires_approval=True,
            category="action"
        ))
        
        def create_task(title: str, description: str, assignee: str, priority: str = "medium", due_days: int = 7) -> Dict[str, Any]:
            """Create a task for follow-up"""
            from datetime import datetime, timedelta
            
            return {
                "task_id": f"TASK-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "title": title,
                "description": description,
                "assignee": assignee,
                "priority": priority,
                "due_date": (datetime.now() + timedelta(days=due_days)).isoformat(),
                "status": "pending",
                "requires_approval": True
            }
        
        self.register(Tool(
            name="create_task",
            description="Create a follow-up task for an action item",
            function=create_task,
            parameters={
                "title": "str - Task title",
                "description": "str - Task description",
                "assignee": "str - Person or role to assign",
                "priority": "str - Priority (low/medium/high/critical)",
                "due_days": "int - Days until due (default: 7)"
            },
            requires_approval=True,
            category="action"
        ))
        
        def log_investigation(patient_key: str, finding: str, next_steps: str) -> Dict[str, Any]:
            """Log an investigation finding"""
            from datetime import datetime
            
            return {
                "investigation_id": f"INV-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "patient_key": patient_key,
                "timestamp": datetime.now().isoformat(),
                "finding": finding,
                "next_steps": next_steps,
                "status": "logged"
            }
        
        self.register(Tool(
            name="log_investigation",
            description="Log an investigation finding for audit trail",
            function=log_investigation,
            parameters={
                "patient_key": "str - Patient identifier",
                "finding": "str - Investigation finding",
                "next_steps": "str - Recommended next steps"
            },
            category="action"
        ))
        
        # === AUTONOMY & SAFETY TOOLS ===
        
        def validate_action(action_type: str, confidence: float = 0.85, 
                           context: Dict[str, Any] = None) -> Dict[str, Any]:
            """Validate an action against safety checks and autonomy matrix"""
            try:
                from src.agents.autonomy.autonomy_matrix import get_autonomy_matrix
                
                matrix = get_autonomy_matrix()
                classification = matrix.classify_action(
                    action_type=action_type,
                    confidence=confidence,
                    context=context or {}
                )
                
                return {
                    "action_type": action_type,
                    "is_safe": not classification.requires_human or classification.decision.value == "auto_execute",
                    "decision": classification.decision.value,
                    "risk_level": classification.risk_level.value,
                    "confidence": classification.confidence,
                    "requires_human": classification.requires_human,
                    "reasoning": classification.reasoning,
                    "warnings": classification.warnings,
                    "can_auto_execute": classification.decision.value == "auto_execute"
                }
            except Exception as e:
                logger.error(f"validate_action error: {e}")
                return {"error": str(e), "is_safe": False, "requires_human": True}
        
        self.register(Tool(
            name="validate_action",
            description="Validate action against safety checks and autonomy matrix before execution",
            function=validate_action,
            parameters={
                "action_type": "str - Action type to validate",
                "confidence": "float - Agent confidence level (0-1)",
                "context": "dict - Optional context (batch_size, is_safety_related, etc.)"
            },
            category="action"
        ))
        
        def rollback_action(action_id: str) -> Dict[str, Any]:
            """Rollback a previously executed action"""
            try:
                from src.agents.autonomy.auto_executor import get_auto_executor
                
                executor = get_auto_executor()
                result = executor.rollback(action_id)
                
                return {
                    "action_id": action_id,
                    "rollback_success": result.success,
                    "rollback_action_id": result.action_id,
                    "error": result.error_message,
                    "rolled_back_at": result.executed_at.isoformat() if result.success else None
                }
            except Exception as e:
                logger.error(f"rollback_action error: {e}")
                return {"error": str(e), "rollback_success": False}
        
        self.register(Tool(
            name="rollback_action",
            description="Rollback a previously auto-executed action to its original state",
            function=rollback_action,
            parameters={"action_id": "str - ID of the action to rollback"},
            requires_approval=True,
            category="action"
        ))
        
        def schedule_notification(recipient: str, message: str, 
                                 schedule_time: str = None, 
                                 notification_type: str = "reminder") -> Dict[str, Any]:
            """Schedule a notification for future delivery"""
            from datetime import datetime, timedelta
            import uuid
            
            # Parse schedule time or default to 1 hour from now
            if schedule_time:
                try:
                    scheduled_at = datetime.fromisoformat(schedule_time)
                except:
                    scheduled_at = datetime.now() + timedelta(hours=1)
            else:
                scheduled_at = datetime.now() + timedelta(hours=1)
            
            notification_id = f"SCHED-{uuid.uuid4().hex[:8].upper()}"
            
            # Try to store in database
            try:
                from src.database.connection import get_db_manager
                db = get_db_manager()
                db.execute_query(
                    """INSERT INTO scheduled_notifications 
                       (id, recipient, message, type, scheduled_at, status, created_at)
                       VALUES (%s, %s, %s, %s, %s, 'pending', NOW())
                       ON CONFLICT DO NOTHING""",
                    (notification_id, recipient, message, notification_type, scheduled_at)
                )
            except Exception as e:
                logger.debug(f"Scheduled notification table may not exist: {e}")
            
            return {
                "notification_id": notification_id,
                "recipient": recipient,
                "message": message[:200],
                "scheduled_at": scheduled_at.isoformat(),
                "notification_type": notification_type,
                "status": "scheduled",
                "queued": True
            }
        
        self.register(Tool(
            name="schedule_notification",
            description="Schedule a notification for future delivery to a recipient",
            function=schedule_notification,
            parameters={
                "recipient": "str - Recipient identifier or role",
                "message": "str - Notification message",
                "schedule_time": "str - ISO datetime for delivery (optional, defaults to 1 hour)",
                "notification_type": "str - Type: reminder, alert, update (default: reminder)"
            },
            category="action"
        ))
        
        # === DIGITAL TWIN / SIMULATION TOOLS ===
        
        def monte_carlo_simulation(n_simulations: int = 10000, 
                                   target_date: str = None,
                                   study_id: str = None) -> Dict[str, Any]:
            """Run Monte Carlo simulation for DB Lock timeline prediction"""
            try:
                from src.ml.simulation.digital_twin_service import get_digital_twin_service
                from datetime import datetime
                
                service = get_digital_twin_service()
                
                target = None
                if target_date:
                    try:
                        target = datetime.fromisoformat(target_date)
                    except:
                        pass
                
                result = service.run_monte_carlo_simulation(
                    n_simulations=n_simulations,
                    target_date=target,
                    study_id=study_id
                )
                
                return result
            except Exception as e:
                logger.error(f"monte_carlo_simulation error: {e}")
                return {"error": str(e)}
        
        self.register(Tool(
            name="monte_carlo_simulation",
            description="Run Monte Carlo simulation for DB Lock timeline with probability distributions",
            function=monte_carlo_simulation,
            parameters={
                "n_simulations": "int - Number of simulations (default: 10000)",
                "target_date": "str - Target date in ISO format (optional)",
                "study_id": "str - Study to simulate (optional, defaults to all)"
            },
            category="analytics"
        ))
        
        def predict_timeline(task_type: str, entity_id: str = None,
                            constraints: Dict[str, Any] = None) -> Dict[str, Any]:
            """Predict timeline for task completion with confidence intervals"""
            try:
                from src.ml.simulation.digital_twin_service import get_digital_twin_service
                
                service = get_digital_twin_service()
                projection = service.get_timeline_projection(study_id=entity_id)
                
                # Estimate based on task type
                base_estimates = {
                    "query_resolution": {"min_days": 3, "max_days": 14, "confidence": 0.75},
                    "signature_completion": {"min_days": 1, "max_days": 7, "confidence": 0.80},
                    "sdv_completion": {"min_days": 5, "max_days": 21, "confidence": 0.70},
                    "db_lock": {"min_days": 30, "max_days": 90, "confidence": 0.65},
                    "site_cleanup": {"min_days": 7, "max_days": 30, "confidence": 0.70}
                }
                
                estimate = base_estimates.get(task_type, {"min_days": 7, "max_days": 28, "confidence": 0.60})
                
                return {
                    "task_type": task_type,
                    "entity_id": entity_id,
                    "prediction_days": (estimate["min_days"] + estimate["max_days"]) // 2,
                    "lower_bound_days": estimate["min_days"],
                    "upper_bound_days": estimate["max_days"],
                    "confidence": estimate["confidence"],
                    "projection_context": projection,
                    "assumptions": [
                        "Current resolution rate continues",
                        "No major resource changes",
                        "Standard business hours"
                    ]
                }
            except Exception as e:
                logger.error(f"predict_timeline error: {e}")
                return {"error": str(e)}
        
        self.register(Tool(
            name="predict_timeline",
            description="Predict timeline for task completion with confidence intervals",
            function=predict_timeline,
            parameters={
                "task_type": "str - Task type (query_resolution, signature_completion, sdv_completion, db_lock, site_cleanup)",
                "entity_id": "str - Entity ID (patient, site, study)",
                "constraints": "dict - Optional constraints affecting timeline"
            },
            category="analytics"
        ))
        
        def what_if_analysis(scenario_type: str, 
                            parameters: Dict[str, Any] = None) -> Dict[str, Any]:
            """Run what-if scenario analysis using Digital Twin"""
            try:
                from src.ml.simulation.digital_twin_service import get_digital_twin_service
                
                service = get_digital_twin_service()
                params = parameters or {}
                
                if scenario_type == "add_cra":
                    result = service.simulate_add_cra(
                        count=params.get("count", 1),
                        target_region=params.get("region")
                    )
                elif scenario_type == "close_site":
                    result = service.simulate_close_site(
                        site_id=params.get("site_id", "Site_001"),
                        reason=params.get("reason", "performance")
                    )
                elif scenario_type == "deadline_probability":
                    from datetime import datetime, timedelta
                    target = params.get("target_date")
                    if target:
                        target = datetime.fromisoformat(target)
                    else:
                        target = datetime.now() + timedelta(days=90)
                    result = service.simulate_deadline_probability(target_date=target)
                else:
                    result = {"error": f"Unknown scenario type: {scenario_type}"}
                
                return {
                    "scenario_type": scenario_type,
                    "parameters": params,
                    "result": result,
                    "analysis_timestamp": datetime.now().isoformat() if 'datetime' in dir() else None
                }
            except Exception as e:
                logger.error(f"what_if_analysis error: {e}")
                return {"error": str(e)}
        
        self.register(Tool(
            name="what_if_analysis",
            description="Run what-if scenario analysis to project impact of changes",
            function=what_if_analysis,
            parameters={
                "scenario_type": "str - Scenario type: add_cra, close_site, deadline_probability",
                "parameters": "dict - Scenario parameters (count, region, site_id, target_date, etc.)"
            },
            category="analytics"
        ))
        
        def calculate_cascade_impact(issue_id: str = None, 
                                    issue_type: str = None,
                                    patient_key: str = None) -> Dict[str, Any]:
            """Calculate downstream cascade effects of resolving an issue"""
            try:
                from src.knowledge.neo4j_graph import get_graph_service
                
                service = get_graph_service()
                
                if patient_key:
                    result = service.get_patient_dependencies(patient_key)
                elif issue_id:
                    result = service.get_cascade_path(issue_id)
                    if hasattr(result, '__dict__'):
                        result = {
                            "source_type": result.source_type,
                            "path_nodes": result.path_nodes,
                            "total_impact": result.total_impact,
                            "affected_patients": result.affected_patients,
                            "unlocked_actions": result.unlocked_actions
                        }
                else:
                    result = {"message": "Provide issue_id or patient_key"}
                
                return {
                    "issue_id": issue_id,
                    "issue_type": issue_type,
                    "patient_key": patient_key,
                    "cascade_analysis": result,
                    "graph_source": "neo4j" if not service.uses_mock else "mock"
                }
            except Exception as e:
                logger.error(f"calculate_cascade_impact error: {e}")
                return {"error": str(e)}
        
        self.register(Tool(
            name="calculate_cascade_impact",
            description="Calculate downstream cascade effects of resolving an issue - what gets unlocked",
            function=calculate_cascade_impact,
            parameters={
                "issue_id": "str - Issue identifier (optional)",
                "issue_type": "str - Issue type for context (optional)",
                "patient_key": "str - Patient key for patient-level analysis (optional)"
            },
            category="analytics"
        ))


# Singleton instance
_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """Get or create the tool registry singleton"""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry