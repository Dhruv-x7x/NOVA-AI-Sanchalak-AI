"""
Sanchalak AI — Zenith V4 Agentic Orchestrator
=============================================
ReAct-style agent that uses Gemini function calling to chain multiple
clinical-trial tools together.  The LLM decides which tools to call,
in what order, and synthesises a single actionable answer.

Tools available:
  1.  run_sql_query           — execute any read-only SQL against PostgreSQL
  2.  get_portfolio_summary   — quick portfolio KPIs
  3.  get_site_details        — detailed metrics for a specific site
  4.  get_patient_details     — detailed metrics for a specific patient
  5.  run_monte_carlo         — 10 000-run timeline simulation (DB-lock)
  6.  get_cascade_analysis    — issue dependency graph from project_issues
  7.  get_dqi_breakdown       — 8-component DQI for site or study
  8.  get_risk_distribution   — risk-level distribution across portfolio
  9.  run_drift_check         — ML model drift summary
 10.  get_issue_summary       — open issues grouped by category
 --- Digital Twin & What-If Simulation ---
 11.  simulate_site_closure   — Monte Carlo impact of closing a site
 12.  simulate_add_resource   — 10k-run impact of adding CRAs/DMs/staff
 13.  simulate_deadline       — probability of meeting a target date
 14.  simulate_process_change — impact of implementing a process improvement
 15.  simulate_dqi_fix        — DQI improvement from fixing specific issue types

No frontend changes required — the assistant endpoint still returns
{ summary, agent_chain, steps, tools_used, confidence, recommendations }.
"""

import json
import logging
import re
import time
import traceback
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import types

from config.llm_config import get_config

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# TOOL DEFINITIONS  (Gemini FunctionDeclaration format)
# ═══════════════════════════════════════════════════════════════

TOOL_DECLARATIONS = [
    types.FunctionDeclaration(
        name="run_sql_query",
        description=(
            "Execute a read-only SQL SELECT query against the clinical trial PostgreSQL database. "
            "Use this for any data retrieval: patient counts, site metrics, adverse events, queries, visits, CRA activity, issues. "
            "The database has tables: patients, clinical_sites, adverse_events, visits, project_issues, queries, cra_activity_logs, unified_patient_record. "
            "IMPORTANT: site_id format is 'Site NNN' (e.g. 'Site 468'). study_id format is 'Study_N'. "
            "risk_level values are Title Case: 'No Risk','Low','Medium','High','Critical'. "
            "Country column uses ISO-3 codes (USA, IND, DEU, GBR, JPN, etc). Region: 'AMERICA','ASIA','EMEA'. "
            "patients table has NO country column — JOIN with clinical_sites for geographic data. "
            "There are NO financial/cost columns in any table."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "sql": types.Schema(
                    type=types.Type.STRING,
                    description="A valid PostgreSQL SELECT statement. Must start with SELECT. No DDL/DML.",
                ),
                "purpose": types.Schema(
                    type=types.Type.STRING,
                    description="Brief explanation of what this query retrieves.",
                ),
            },
            required=["sql", "purpose"],
        ),
    ),
    types.FunctionDeclaration(
        name="get_portfolio_summary",
        description=(
            "Get high-level portfolio KPIs: total patients, total sites, total studies, "
            "mean DQI, DB-lock readiness %, high-risk patient count, critical issues count. "
            "Use this as a starting point for broad questions about the trial portfolio."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={},
        ),
    ),
    types.FunctionDeclaration(
        name="get_site_details",
        description=(
            "Get detailed metrics for a specific clinical site: DQI score, performance score, "
            "risk level, patient count, enrollment rate, query resolution days, open issues, "
            "country, region, principal investigator."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "site_id": types.Schema(
                    type=types.Type.STRING,
                    description="Site identifier, e.g. 'Site 468'. Format: 'Site NNN'.",
                ),
            },
            required=["site_id"],
        ),
    ),
    types.FunctionDeclaration(
        name="get_patient_details",
        description=(
            "Get detailed metrics for a specific patient: risk score, DQI score, "
            "clean status tier, open queries, open issues, SAE count, missing visits/pages."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "patient_key": types.Schema(
                    type=types.Type.STRING,
                    description="Patient key or search term, e.g. 'Subject 48147' or full key 'Study_1|Site 10|Subject 17'.",
                ),
            },
            required=["patient_key"],
        ),
    ),
    types.FunctionDeclaration(
        name="run_monte_carlo_simulation",
        description=(
            "Run a 10,000-iteration Monte Carlo simulation to predict DB-lock timeline. "
            "Returns P10/P25/P50/P75/P90 percentile estimates (in days) and confidence interval. "
            "Optionally provide a study_id to scope the simulation to one study. "
            "Use this when users ask about timelines, deadlines, 'when will we finish', "
            "'how long until DB lock', or probability of meeting a target date."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "study_id": types.Schema(
                    type=types.Type.STRING,
                    description="Optional study ID to scope simulation, e.g. 'Study_1'. If omitted, runs portfolio-wide.",
                ),
            },
        ),
    ),
    types.FunctionDeclaration(
        name="get_cascade_analysis",
        description=(
            "Get cascade/dependency analysis of issues — shows which issue types block others, "
            "impact scores, and resolution priority order. Based on project_issues table. "
            "Use this when users ask about root causes, issue dependencies, cascade effects, "
            "or 'what should we fix first'."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "study_id": types.Schema(
                    type=types.Type.STRING,
                    description="Optional study ID to filter, e.g. 'Study_1'.",
                ),
            },
        ),
    ),
    types.FunctionDeclaration(
        name="get_dqi_breakdown",
        description=(
            "Get the 8-component DQI (Data Quality Index) breakdown: Safety (25%), Query (20%), "
            "Completeness (15%), Coding (12%), Lab (10%), SDV (8%), Signature (5%), EDRR (5%). "
            "Shows which components are dragging down quality. Can be filtered by site_id or study_id."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "site_id": types.Schema(
                    type=types.Type.STRING,
                    description="Optional site ID to filter, e.g. 'Site 468'.",
                ),
                "study_id": types.Schema(
                    type=types.Type.STRING,
                    description="Optional study ID to filter, e.g. 'Study_1'.",
                ),
            },
        ),
    ),
    types.FunctionDeclaration(
        name="get_risk_distribution",
        description=(
            "Get the distribution of patients across risk levels (No Risk, Low, Medium, High, Critical) "
            "with counts and percentages. Can be filtered by study_id or site_id."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "study_id": types.Schema(
                    type=types.Type.STRING,
                    description="Optional study ID to filter.",
                ),
                "site_id": types.Schema(
                    type=types.Type.STRING,
                    description="Optional site ID to filter.",
                ),
            },
        ),
    ),
    types.FunctionDeclaration(
        name="run_drift_check",
        description=(
            "Run a LIVE drift detection check on a predictive model using current database data. "
            "Splits patients into baseline (first 60%) and current (last 40%) windows, then runs "
            "PSI (Population Stability Index) and KS (Kolmogorov-Smirnov) tests on each feature. "
            "Supported model names: 'dqi_predictor', 'risk_classifier', 'anomaly_detector'. "
            "Returns per-feature drift severity (none/low/medium/high/critical), PSI scores, "
            "p-values, and actionable recommendations. Use when users ask about model drift, "
            "model health, AI governance, retraining needs, or 'run drift check'."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "model_name": types.Schema(
                    type=types.Type.STRING,
                    description="Model to check: 'dqi_predictor', 'risk_classifier', or 'anomaly_detector'. Default: 'dqi_predictor'.",
                ),
            },
        ),
    ),
    types.FunctionDeclaration(
        name="get_issue_summary",
        description=(
            "Get a summary of open issues grouped by category and priority, "
            "with counts of affected sites, affected patients, and average cascade impact. "
            "Can be filtered by study_id or site_id."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "study_id": types.Schema(
                    type=types.Type.STRING,
                    description="Optional study ID to filter.",
                ),
                "site_id": types.Schema(
                    type=types.Type.STRING,
                    description="Optional site ID to filter.",
                ),
            },
        ),
    ),
    # ── Digital Twin & What-If Simulation Tools ──────────────
    types.FunctionDeclaration(
        name="simulate_site_closure",
        description=(
            "Run a 10,000-iteration Monte Carlo simulation to predict the impact of closing a clinical site. "
            "Models patient transfer success rates (Beta distribution), timeline delays (Gamma), "
            "and cost impacts. Returns success probability, delay percentiles, risk score, and recommendations. "
            "Use when users ask 'what if we close Site X', 'should we shut down this site', or site closure impact."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "site_id": types.Schema(
                    type=types.Type.STRING,
                    description="Site to simulate closing, e.g. 'Site 468'. Format: 'Site NNN'.",
                ),
            },
            required=["site_id"],
        ),
    ),
    types.FunctionDeclaration(
        name="simulate_add_resource",
        description=(
            "Run a 10,000-iteration Monte Carlo simulation to predict the impact of adding staff resources. "
            "Models productivity with ramp-up uncertainty, calculates issues resolved in 90 days, timeline improvement, "
            "6-month cost, and ROI. Resource types: CRA, data_manager, site_coordinator, safety_physician, medical_coder, study_lead. "
            "Use when users ask 'what if we add more CRAs', 'should we hire staff', 'resource planning'."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "resource_type": types.Schema(
                    type=types.Type.STRING,
                    description="Type: 'CRA', 'data_manager', 'site_coordinator', 'safety_physician', 'medical_coder', or 'study_lead'.",
                ),
                "count": types.Schema(
                    type=types.Type.INTEGER,
                    description="Number of resources to add. Default: 1.",
                ),
            },
            required=["resource_type"],
        ),
    ),
    types.FunctionDeclaration(
        name="simulate_deadline",
        description=(
            "Run a 10,000-iteration Monte Carlo simulation to estimate the probability of meeting a target "
            "completion date. Uses real pending SDV, open queries, and signature counts from the database "
            "with DB-derived resolution rates. Returns probability of on-time completion, assessment, "
            "timeline percentiles, and recommended actions. "
            "Use when users ask 'can we meet [date]', 'will we finish by [date]', 'deadline feasibility'."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "target_date": types.Schema(
                    type=types.Type.STRING,
                    description="Target deadline in YYYY-MM-DD format, e.g. '2025-12-31'.",
                ),
            },
            required=["target_date"],
        ),
    ),
    types.FunctionDeclaration(
        name="simulate_process_change",
        description=(
            "Run a 10,000-iteration Monte Carlo simulation to predict impact of a process improvement. "
            "Models improvement factor with adoption-rate and implementation-delay uncertainty. "
            "Returns issues reduced, net days saved, and probability of positive net impact. "
            "Use when users ask 'what if we implement [process]', 'how much would centralized monitoring help', "
            "'impact of automated query generation'."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "process_name": types.Schema(
                    type=types.Type.STRING,
                    description="Name of the process change, e.g. 'automated query generation', 'risk-based SDV', 'centralized monitoring'.",
                ),
                "expected_improvement": types.Schema(
                    type=types.Type.NUMBER,
                    description="Expected improvement factor 0.0-1.0 (e.g. 0.15 = 15%). Default: 0.15.",
                ),
            },
            required=["process_name"],
        ),
    ),
    types.FunctionDeclaration(
        name="simulate_dqi_fix",
        description=(
            "Simulate DQI (Data Quality Index) improvement from fixing specific issue types. "
            "Uses the 8-component DQI model (Safety 25%, Query 20%, Completeness 15%, Coding 12%, "
            "Lab 10%, SDV 8%, Signature 5%, EDRR 5%) to calculate projected score improvement. "
            "Issue types: 'open_query', 'missing_visit', 'pending_sae', 'unsigned_form', "
            "'incomplete_sdv', 'coding_issue', 'lab_issue', 'missing_page', 'protocol_deviation', 'edrr_issue'. "
            "Use when users ask 'how will DQI improve if we fix queries', 'DQI impact of resolving SAEs', "
            "'what should we fix to raise DQI'."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "site_id": types.Schema(
                    type=types.Type.STRING,
                    description="Optional site ID to scope, e.g. 'Site 468'. Omit for portfolio-wide.",
                ),
                "issue_types": types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(type=types.Type.STRING),
                    description="Issue types to fix. Omit to simulate fixing all types.",
                ),
            },
        ),
    ),
]


# ═══════════════════════════════════════════════════════════════
# SCHEMA CONTEXT  (given to the LLM so it can write correct SQL)
# ═══════════════════════════════════════════════════════════════

SCHEMA_CONTEXT = """
Available Tables & Important Columns (PostgreSQL):
1. clinical_sites (site_id TEXT 'Site NNN', name, country ISO-3, region TEXT 'AMERICA'|'ASIA'|'EMEA',
   performance_score, risk_level TEXT 'low'|'medium'|'high', dqi_score, enrollment_rate,
   query_resolution_days, patient_count, open_issues, principal_investigator)
2. patients (patient_key TEXT 'Study_N|Site NNN|Subject NNNNN', study_id, site_id, status, clean_status_tier,
   risk_level TEXT 'No Risk'|'Low'|'Medium'|'High'|'Critical', risk_score, dqi_score,
   open_queries_count, open_issues_count, pct_missing_visits, pct_missing_pages,
   is_clean_patient BOOL, is_db_lock_ready BOOL, sdtm_ready BOOL, has_sae BOOL, sae_count)
   NOTE: patients has NO country/region — JOIN clinical_sites for geography.
3. visits (visit_id, patient_key, visit_name, visit_number, status, deviation_days, sdv_complete, data_entry_complete, is_in_window)
4. adverse_events (ae_id, patient_key, ae_term, severity, causality, is_sae, is_ongoing, reported_date, meddra_pt, meddra_soc)
5. project_issues (issue_id, site_id, patient_key, category, issue_type, description, priority, severity, status, cascade_impact_score, study_id)
6. queries (query_id, patient_key, field_name, form_name, query_text, query_type, status, age_days)
7. cra_activity_logs (log_id, site_id, cra_name, activity_type, visit_date, status, follow_up_letter_sent)

Rules: Always prefix columns with table aliases. Limit to 20 unless aggregating.
risk_level in patients is Title Case. No financial/cost columns exist anywhere.
For site-specific: WHERE cs.site_id = 'Site 468'. For patient: WHERE p.patient_key LIKE '%Subject 48147%'.
For study: WHERE p.study_id = 'Study_5'. For country: WHERE cs.country = 'DEU'.
Country codes: USA, IND, DEU, GBR, JPN, FRA, BRA, CAN, AUS, CHN, KOR, etc.
Region mapping: 'EMEA' = Europe + Middle East + Africa; 'ASIA' = Asia-Pacific; 'AMERICA' = Americas.
"""


# ═══════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = f"""You are the Sanchalak AI Assistant (Zenith V4), an autonomous clinical trial analyst.

You have access to tools that let you query the clinical trial database, run Monte Carlo simulations,
analyze issue cascades, check model drift, and run Digital Twin what-if scenarios.

EFFICIENCY RULES — CRITICAL:
- For simple lookups ("where is Site X", "show me patient Y", "how many patients"),
  call ONE tool (get_site_details, get_patient_details, or a single run_sql_query)
  and IMMEDIATELY produce your final answer. Do NOT enrich with follow-up queries
  unless the user explicitly asks for more detail.
- Only chain multiple tools for genuinely complex analytical questions that require
  cross-referencing multiple data sources.
- After each tool result, ask yourself: "Do I already have enough data to answer?"
  If yes, STOP calling tools and synthesise your answer.

STRATEGY:
- Simple data lookups → 1 tool call, then answer
- For complex analytical questions: chain multiple tools (e.g. SQL → Monte Carlo → cascade analysis)
- For "why" questions: use cascade analysis + DQI breakdown to find root causes
- For "when/timeline" questions: use Monte Carlo simulation or simulate_deadline
- For "what should we prioritize" questions: combine risk distribution + issue summary + cascade
- For "what if" questions (Digital Twin): use simulate_site_closure, simulate_add_resource,
  simulate_process_change, or simulate_deadline
- For DQI improvement planning: use simulate_dqi_fix to model fix impact, then recommend actions
- For resource planning: use simulate_add_resource to compare adding different staff types

DATABASE SCHEMA:
{SCHEMA_CONTEXT}

RULES:
1. NEVER invent data. Only use data returned by tools.
2. Use exact identifiers (Site 1, Site 468, CRA 0, Study_1).
3. No financial/cost data exists — never reference budgets or dollars.
4. When writing SQL, always use the correct column names and table aliases.
5. Format responses in markdown with ## headers, tables, and bullet points.
6. Always end with actionable recommendations.
7. If a tool call fails, explain what happened and try an alternative approach.
8. For multi-part questions, address each part systematically.
9. MINIMIZE tool calls. Prefer dedicated tools (get_site_details, get_patient_details)
   over raw SQL when they fit. Answer as soon as you have sufficient data.
"""


# ═══════════════════════════════════════════════════════════════
# TOOL EXECUTOR
# ═══════════════════════════════════════════════════════════════

class ToolExecutor:
    """Executes tool calls against real system backends."""

    def __init__(self):
        self._db_manager = None

    def _get_db(self):
        if self._db_manager is None:
            try:
                from src.database.connection import get_db_manager
                self._db_manager = get_db_manager()
            except Exception as e:
                logger.error(f"DB manager init failed: {e}")
        return self._db_manager

    def _execute_sql_raw(self, sql: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Execute SQL and return (rows, error)."""
        from sqlalchemy import text

        db = self._get_db()
        if db is None or db.engine is None:
            return [], "Database connection unavailable."

        # Safety check
        s = sql.strip().lower()
        if not s.startswith("select"):
            return [], "Only SELECT queries are allowed."
        blocked = ["drop ", "delete ", "update ", "insert ", "alter ", "truncate ", "create ", "grant "]
        for b in blocked:
            if b in s:
                return [], f"Blocked keyword: {b.strip()}"

        try:
            with db.engine.connect() as conn:
                result = conn.execute(text(sql))
                rows = result.fetchall()
                columns = list(result.keys())
                data = []
                for row in rows:
                    d = dict(zip(columns, row))
                    for k, v in d.items():
                        if isinstance(v, datetime):
                            d[k] = v.isoformat()
                        elif isinstance(v, Decimal):
                            d[k] = float(v)
                    data.append(d)
                return data, None
        except Exception as e:
            return [], str(e)

    # ── Individual tool handlers ─────────────────────────────

    def run_sql_query(self, sql: str, purpose: str = "") -> Dict[str, Any]:
        logger.info(f"[TOOL] run_sql_query: {purpose} | SQL: {sql[:120]}...")
        data, err = self._execute_sql_raw(sql)
        if err:
            return {"error": err, "rows": 0, "data": []}
        return {"rows": len(data), "data": data[:50], "purpose": purpose}  # cap at 50 rows

    def get_portfolio_summary(self) -> Dict[str, Any]:
        logger.info("[TOOL] get_portfolio_summary")
        sql = """
            SELECT
                COUNT(DISTINCT p.patient_key) AS total_patients,
                COUNT(DISTINCT p.site_id) AS total_sites,
                COUNT(DISTINCT p.study_id) AS total_studies,
                ROUND(AVG(p.dqi_score)::numeric, 1) AS mean_dqi,
                ROUND(100.0 * SUM(CASE WHEN p.is_db_lock_ready THEN 1 ELSE 0 END) / NULLIF(COUNT(*),0), 1) AS db_lock_ready_pct,
                SUM(CASE WHEN p.risk_level IN ('High', 'Critical') THEN 1 ELSE 0 END) AS high_risk_patients,
                SUM(CASE WHEN p.is_clean_patient THEN 1 ELSE 0 END) AS clean_patients
            FROM patients p
        """
        data, err = self._execute_sql_raw(sql)
        if err:
            return {"error": err}
        return data[0] if data else {"error": "No data"}

    def get_site_details(self, site_id: str) -> Dict[str, Any]:
        logger.info(f"[TOOL] get_site_details: {site_id}")
        sql = f"""
            SELECT
                cs.site_id, cs.name, cs.country, cs.region,
                cs.patient_count,
                ROUND(cs.dqi_score::numeric, 1) AS dqi_score,
                ROUND(cs.performance_score::numeric, 1) AS performance_score,
                cs.risk_level,
                cs.enrollment_rate,
                ROUND(cs.query_resolution_days::numeric, 1) AS query_resolution_days,
                cs.principal_investigator,
                cs.open_issues,
                (SELECT COUNT(*) FROM patients p WHERE p.site_id = cs.site_id AND p.risk_level IN ('High','Critical')) AS high_risk_patients,
                (SELECT ROUND(100.0 * SUM(CASE WHEN p.is_db_lock_ready THEN 1 ELSE 0 END) / NULLIF(COUNT(*),0), 1) FROM patients p WHERE p.site_id = cs.site_id) AS db_lock_ready_pct
            FROM clinical_sites cs
            WHERE cs.site_id = '{site_id}'
        """
        data, err = self._execute_sql_raw(sql)
        if err:
            return {"error": err}
        return data[0] if data else {"error": f"Site {site_id} not found"}

    def get_patient_details(self, patient_key: str) -> Dict[str, Any]:
        logger.info(f"[TOOL] get_patient_details: {patient_key}")
        # Support partial match (e.g. "Subject 48147")
        where = f"p.patient_key LIKE '%{patient_key}%'" if '|' not in patient_key else f"p.patient_key = '{patient_key}'"
        sql = f"""
            SELECT
                p.patient_key, p.site_id, p.study_id, p.status,
                p.risk_level, ROUND(p.risk_score::numeric, 2) AS risk_score,
                ROUND(p.dqi_score::numeric, 1) AS dqi_score,
                p.clean_status_tier, p.open_queries_count, p.open_issues_count,
                p.pct_missing_visits, p.pct_missing_pages,
                p.has_sae, p.sae_count, p.is_db_lock_ready, p.sdtm_ready
            FROM patients p
            WHERE {where}
            LIMIT 5
        """
        data, err = self._execute_sql_raw(sql)
        if err:
            return {"error": err}
        if not data:
            return {"error": f"Patient '{patient_key}' not found"}
        return {"patients": data, "count": len(data)}

    def run_monte_carlo_simulation(self, study_id: str = None) -> Dict[str, Any]:
        logger.info(f"[TOOL] run_monte_carlo_simulation: study_id={study_id}")
        try:
            # Get current trial state from DB — use REAL counts, not fabricated multipliers
            where_patient = f"WHERE p.study_id = '{study_id}'" if study_id else ""

            sql = f"""
                SELECT
                    COUNT(*) AS total_patients,
                    SUM(CASE WHEN p.is_clean_patient THEN 1 ELSE 0 END) AS clean_patients,
                    SUM(CASE WHEN p.is_db_lock_ready THEN 1 ELSE 0 END) AS db_lock_ready,
                    SUM(p.open_queries_count) AS total_open_queries,
                    ROUND(AVG(p.dqi_score)::numeric, 1) AS avg_dqi,
                    COUNT(DISTINCT p.site_id) AS active_sites
                FROM patients p
                {where_patient}
            """
            data, err = self._execute_sql_raw(sql)
            if err:
                return {"error": err}

            state_data = data[0] if data else {}
            total_patients = int(state_data.get("total_patients", 0))
            open_queries = int(state_data.get("total_open_queries", 0))
            db_lock_ready = int(state_data.get("db_lock_ready", 0))
            active_sites = int(state_data.get("active_sites", 0))

            # Get REAL pending SDV and data-entry counts from visits table
            # NOTE: visits.patient_key uses different format than patients.patient_key
            # (PAT-NNNNN vs Study_N|Site NNN|Subject NNNNN) so we can't JOIN.
            # Query visits directly for portfolio-wide counts.
            sdv_sql = """
                SELECT
                    COALESCE(SUM(CASE WHEN NOT v.sdv_complete THEN 1 ELSE 0 END), 0) AS pending_sdv,
                    COALESCE(SUM(CASE WHEN NOT v.data_entry_complete THEN 1 ELSE 0 END), 0) AS pending_de
                FROM visits v
            """
            sdv_data, sdv_err = self._execute_sql_raw(sdv_sql)
            if sdv_err:
                pending_sdv = 0
                pending_sigs = 0
            else:
                sdv_row = sdv_data[0] if sdv_data else {}
                pending_sdv = int(sdv_row.get("pending_sdv", 0) or 0)
                pending_sigs = int(sdv_row.get("pending_de", 0) or 0)

            from src.ml.simulation.monte_carlo_engine import MonteCarloEngine, TrialState
            engine = MonteCarloEngine(n_simulations=10000)
            trial_state = TrialState(
                study_id=study_id or "portfolio",
                total_patients=total_patients,
                db_lock_ready=db_lock_ready,
                open_queries=open_queries,
                pending_signatures=pending_sigs,
                pending_sdv=pending_sdv,
                active_sites=active_sites,
                avg_dqi=float(state_data.get("avg_dqi", 75)),
            )
            # Load rates from DB
            trial_state._load_real_rates()

            result = engine.simulate_dblock_timeline(trial_state)
            return {
                "simulation": "DB-Lock Timeline",
                "n_simulations": 10000,
                "current_state": {
                    "total_patients": total_patients,
                    "db_lock_ready": db_lock_ready,
                    "open_queries": open_queries,
                    "pending_sdv": pending_sdv,
                    "pending_signatures": pending_sigs,
                    "active_sites": active_sites,
                    "avg_dqi": float(state_data.get("avg_dqi", 75)),
                },
                "timeline_days": result.to_dict(),
                "study_id": study_id or "all_studies",
            }
        except Exception as e:
            logger.error(f"Monte Carlo failed: {e}")
            return {"error": str(e)}

    def get_cascade_analysis(self, study_id: str = None) -> Dict[str, Any]:
        logger.info(f"[TOOL] get_cascade_analysis: study_id={study_id}")
        where = f"WHERE pi.status != 'resolved'" + (f" AND pi.study_id = '{study_id}'" if study_id else "")
        sql = f"""
            SELECT
                pi.category,
                pi.issue_type,
                pi.priority,
                COUNT(*) AS issue_count,
                COUNT(DISTINCT pi.site_id) AS sites_affected,
                COUNT(DISTINCT pi.patient_key) AS patients_affected,
                ROUND(AVG(pi.cascade_impact_score)::numeric, 2) AS avg_cascade_impact,
                ROUND(MAX(pi.cascade_impact_score)::numeric, 2) AS max_cascade_impact
            FROM project_issues pi
            {where}
            GROUP BY pi.category, pi.issue_type, pi.priority
            ORDER BY avg_cascade_impact DESC, issue_count DESC
            LIMIT 20
        """
        data, err = self._execute_sql_raw(sql)
        if err:
            return {"error": err}

        # Also get top blocking issues
        blocking_sql = f"""
            SELECT pi.issue_type, pi.category, pi.priority,
                   ROUND(pi.cascade_impact_score::numeric, 2) AS cascade_impact,
                   pi.description, pi.site_id
            FROM project_issues pi
            {where}
            ORDER BY pi.cascade_impact_score DESC NULLS LAST
            LIMIT 10
        """
        blocking_data, _ = self._execute_sql_raw(blocking_sql)

        return {
            "issue_groups": data,
            "top_blocking_issues": blocking_data or [],
            "total_groups": len(data),
            "study_id": study_id or "all_studies",
        }

    def get_dqi_breakdown(self, site_id: str = None, study_id: str = None) -> Dict[str, Any]:
        logger.info(f"[TOOL] get_dqi_breakdown: site_id={site_id}, study_id={study_id}")
        conditions = []
        if site_id:
            conditions.append(f"p.site_id = '{site_id}'")
        if study_id:
            conditions.append(f"p.study_id = '{study_id}'")
        where = "WHERE " + " AND ".join(conditions) if conditions else ""

        sql = f"""
            SELECT
                ROUND(AVG(p.dqi_score)::numeric, 1) AS overall_dqi,
                COUNT(*) AS patient_count,
                SUM(CASE WHEN p.dqi_score >= 90 THEN 1 ELSE 0 END) AS excellent_count,
                SUM(CASE WHEN p.dqi_score >= 80 AND p.dqi_score < 90 THEN 1 ELSE 0 END) AS good_count,
                SUM(CASE WHEN p.dqi_score >= 70 AND p.dqi_score < 80 THEN 1 ELSE 0 END) AS fair_count,
                SUM(CASE WHEN p.dqi_score < 70 THEN 1 ELSE 0 END) AS poor_count,
                ROUND(AVG(p.open_queries_count)::numeric, 1) AS avg_open_queries,
                ROUND(AVG(p.open_issues_count)::numeric, 1) AS avg_open_issues,
                ROUND(AVG(p.pct_missing_visits)::numeric, 1) AS avg_missing_visits_pct,
                SUM(CASE WHEN p.has_sae THEN 1 ELSE 0 END) AS patients_with_sae,
                ROUND(100.0 * SUM(CASE WHEN p.is_db_lock_ready THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1) AS db_lock_ready_pct,
                ROUND(100.0 * SUM(CASE WHEN p.is_clean_patient THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1) AS clean_patient_pct
            FROM patients p
            {where}
        """
        data, err = self._execute_sql_raw(sql)
        if err:
            return {"error": err}

        result = data[0] if data else {}

        # Get DQI component explanation
        result["dqi_components"] = {
            "safety": {"weight": "25%", "description": "SAE pending issues — penalty: -15 per pending SAE"},
            "query": {"weight": "20%", "description": "Open queries — penalty: -3 per open query"},
            "completeness": {"weight": "15%", "description": "Missing visits/pages — penalty: -8 per missing item"},
            "coding": {"weight": "12%", "description": "MedDRA/WHODrug uncoded terms — penalty: -5 per uncoded"},
            "lab": {"weight": "10%", "description": "Lab issues — penalty: -10 per lab issue"},
            "sdv": {"weight": "8%", "description": "SDV verification completion rate"},
            "signature": {"weight": "5%", "description": "Overdue signatures, weighted by age"},
            "edrr": {"weight": "5%", "description": "Third-party reconciliation issues"},
        }
        result["scope"] = {"site_id": site_id, "study_id": study_id}
        return result

    def get_risk_distribution(self, study_id: str = None, site_id: str = None) -> Dict[str, Any]:
        logger.info(f"[TOOL] get_risk_distribution: study_id={study_id}, site_id={site_id}")
        conditions = []
        if study_id:
            conditions.append(f"p.study_id = '{study_id}'")
        if site_id:
            conditions.append(f"p.site_id = '{site_id}'")
        where = "WHERE " + " AND ".join(conditions) if conditions else ""

        sql = f"""
            SELECT
                p.risk_level,
                COUNT(*) AS patient_count,
                ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER(), 1) AS pct
            FROM patients p
            {where}
            GROUP BY p.risk_level
            ORDER BY CASE p.risk_level
                WHEN 'Critical' THEN 5 WHEN 'High' THEN 4
                WHEN 'Medium' THEN 3 WHEN 'Low' THEN 2 ELSE 1 END DESC
        """
        data, err = self._execute_sql_raw(sql)
        if err:
            return {"error": err}
        return {"distribution": data, "scope": {"study_id": study_id, "site_id": site_id}}

    def run_drift_check(self, model_name: str = "dqi_predictor") -> Dict[str, Any]:
        logger.info(f"[TOOL] run_drift_check: model={model_name}")
        try:
            import pandas as pd
            import numpy as np

            # Define features for each model type
            MODEL_FEATURES = {
                "dqi_predictor": {
                    "target": "dqi_score",
                    "features": ["open_queries_count", "open_issues_count", "pct_missing_visits",
                                  "pct_missing_pages", "sae_count", "risk_score"],
                },
                "risk_classifier": {
                    "target": "risk_score",
                    "features": ["dqi_score", "open_queries_count", "open_issues_count",
                                  "pct_missing_visits", "sae_count"],
                },
                "anomaly_detector": {
                    "target": "risk_score",
                    "features": ["dqi_score", "open_queries_count", "open_issues_count",
                                  "pct_missing_visits", "pct_missing_pages", "sae_count"],
                },
            }

            config = MODEL_FEATURES.get(model_name, MODEL_FEATURES["dqi_predictor"])
            all_cols = config["features"] + [config["target"], "patient_key"]
            col_list = ", ".join(f"p.{c}" for c in all_cols)

            sql = f"SELECT {col_list} FROM patients p ORDER BY p.patient_key"
            data, err = self._execute_sql_raw(sql)
            if err:
                return {"error": err}
            if len(data) < 50:
                return {"error": "Insufficient data for drift analysis (need ≥50 patients)"}

            df = pd.DataFrame(data)
            for c in config["features"] + [config["target"]]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna()

            # Split: first 60% = baseline, last 40% = current
            split = int(len(df) * 0.6)
            baseline_df = df.iloc[:split]
            current_df = df.iloc[split:]

            # Run drift detection via the real DriftDetector
            from src.ml.governance.drift_detector import DriftDetector
            detector = DriftDetector()

            # Build baselines from the first window
            feature_cols = [c for c in config["features"] + [config["target"]] if c in baseline_df.columns]
            detector.create_baseline(model_name, "live_check", baseline_df[feature_cols])

            # Detect drift on the current window
            results = detector.detect_feature_drift(model_name, current_df[feature_cols])

            # Format results
            drift_results = []
            overall_severity = "none"
            severity_rank = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}

            for r in results:
                sev = r.severity.value if hasattr(r.severity, 'value') else str(r.severity)
                drift_results.append({
                    "feature": r.feature_name,
                    "drift_type": r.drift_type.value if hasattr(r.drift_type, 'value') else str(r.drift_type),
                    "severity": sev,
                    "psi_score": round(r.statistic, 4) if r.statistic else None,
                    "p_value": round(r.p_value, 4) if r.p_value else None,
                    "baseline_mean": round(r.baseline_mean, 4) if r.baseline_mean else None,
                    "current_mean": round(r.current_mean, 4) if r.current_mean else None,
                    "details": r.details,
                    "recommendations": r.recommendations if r.recommendations else [],
                })
                if severity_rank.get(sev, 0) > severity_rank.get(overall_severity, 0):
                    overall_severity = sev

            drifted_count = sum(1 for r in drift_results if r["severity"] != "none")

            return {
                "model_name": model_name,
                "check_type": "live_drift_detection",
                "baseline_window": f"first {split} patients ({split} records)",
                "current_window": f"last {len(df) - split} patients ({len(df) - split} records)",
                "total_features_checked": len(drift_results),
                "drifted_features": drifted_count,
                "overall_severity": overall_severity,
                "feature_results": drift_results,
                "methods_used": ["PSI (Population Stability Index)", "KS Test (Kolmogorov-Smirnov)"],
                "governance": "21 CFR Part 11 compliant — all drift events audit-logged",
            }

        except Exception as e:
            logger.error(f"Drift check failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"error": str(e), "model_name": model_name}

    def get_issue_summary(self, study_id: str = None, site_id: str = None) -> Dict[str, Any]:
        logger.info(f"[TOOL] get_issue_summary: study_id={study_id}, site_id={site_id}")
        conditions = ["pi.status != 'resolved'"]
        if study_id:
            conditions.append(f"pi.study_id = '{study_id}'")
        if site_id:
            conditions.append(f"pi.site_id = '{site_id}'")
        where = "WHERE " + " AND ".join(conditions)

        sql = f"""
            SELECT
                pi.category, pi.priority,
                COUNT(*) AS issue_count,
                COUNT(DISTINCT pi.site_id) AS sites_affected,
                COUNT(DISTINCT pi.patient_key) AS patients_affected,
                ROUND(AVG(pi.cascade_impact_score)::numeric, 2) AS avg_cascade_impact
            FROM project_issues pi
            {where}
            GROUP BY pi.category, pi.priority
            ORDER BY CASE pi.priority
                WHEN 'critical' THEN 4 WHEN 'high' THEN 3
                WHEN 'medium' THEN 2 ELSE 1 END DESC,
                issue_count DESC
        """
        data, err = self._execute_sql_raw(sql)
        if err:
            return {"error": err}

        total_issues = sum(r.get("issue_count", 0) for r in data)
        critical = sum(r.get("issue_count", 0) for r in data if r.get("priority") == "critical")
        return {
            "summary": data,
            "total_open_issues": total_issues,
            "critical_issues": critical,
            "scope": {"study_id": study_id, "site_id": site_id},
        }

    # ── Digital Twin & What-If Simulation handlers ─────────

    def simulate_site_closure(self, site_id: str) -> Dict[str, Any]:
        """Simulate closing a clinical site using 10k Monte Carlo iterations."""
        logger.info(f"[TOOL] simulate_site_closure: {site_id}")
        try:
            # Pull real site data from PostgreSQL
            sql = f"""
                SELECT cs.site_id, cs.patient_count, cs.performance_score,
                       cs.dqi_score, cs.risk_level, cs.name, cs.country, cs.region
                FROM clinical_sites cs
                WHERE cs.site_id = '{site_id}'
            """
            data, err = self._execute_sql_raw(sql)
            if err:
                return {"error": err}
            if not data:
                return {"error": f"Site {site_id} not found"}

            site = data[0]
            patient_count = int(site.get("patient_count", 0))
            perf_score = float(site.get("performance_score") or 50)

            from src.ml.simulation.monte_carlo_engine import MonteCarloEngine
            engine = MonteCarloEngine(n_simulations=10000)

            result = engine.simulate_site_closure_impact(
                site_id=site_id,
                site_patient_count=patient_count,
                avg_transfer_success_rate=0.85,
                site_performance_score=perf_score,
            )

            return {
                "simulation": f"Site Closure Impact: {site_id}",
                "site_info": {
                    "site_id": site.get("site_id"),
                    "name": site.get("name"),
                    "country": site.get("country"),
                    "region": site.get("region"),
                    "patient_count": patient_count,
                    "performance_score": perf_score,
                    "dqi_score": site.get("dqi_score"),
                    "risk_level": site.get("risk_level"),
                },
                "n_simulations": 10000,
                "results": result.to_dict(),
            }
        except Exception as e:
            logger.error(f"simulate_site_closure failed: {e}\n{traceback.format_exc()}")
            return {"error": str(e)}

    def simulate_add_resource(self, resource_type: str, count: int = 1) -> Dict[str, Any]:
        """Simulate impact of adding staff resources using 10k Monte Carlo iterations."""
        logger.info(f"[TOOL] simulate_add_resource: {resource_type} x{count}")
        try:
            import numpy as np

            # Pull real open-issue count from DB
            sql = """
                SELECT COUNT(*) AS total_issues,
                       SUM(CASE WHEN pi.priority = 'critical' THEN 1 ELSE 0 END) AS critical_issues,
                       SUM(CASE WHEN pi.priority = 'high' THEN 1 ELSE 0 END) AS high_issues
                FROM project_issues pi
                WHERE pi.status != 'resolved'
            """
            data, err = self._execute_sql_raw(sql)
            if err:
                return {"error": err}

            row = data[0] if data else {}
            total_issues = int(row.get("total_issues", 50000) or 50000)

            # Productivity (issues/day) and cost ($/month) by role
            RESOURCE_MAP = {
                "cra":                {"productivity": 8,  "cost": 12000},
                "data_manager":       {"productivity": 15, "cost": 10000},
                "site_coordinator":   {"productivity": 5,  "cost": 6000},
                "safety_physician":   {"productivity": 3,  "cost": 25000},
                "medical_coder":      {"productivity": 50, "cost": 8000},
                "study_lead":         {"productivity": 2,  "cost": 18000},
            }

            key = resource_type.lower().replace(" ", "_")
            resource = RESOURCE_MAP.get(key, RESOURCE_MAP["cra"])
            base_prod = resource["productivity"]
            monthly_cost = resource["cost"] * count

            rng = np.random.default_rng(42)
            n_sims = 10000

            # Monte Carlo parameters
            prod_mult = rng.normal(1.0, 0.15, n_sims).clip(0.5, 1.5)
            ramp_up = rng.triangular(14, 30, 60, n_sims)

            effective_days = 90 - ramp_up * 0.5
            daily_prod = prod_mult * base_prod * count
            issues_resolved_90d = daily_prod * effective_days

            # Timeline improvement
            baseline_daily = 100  # current staff baseline
            new_daily = baseline_daily + daily_prod
            current_days_left = total_issues / baseline_daily
            new_days_left = total_issues / new_daily
            timeline_improvement = current_days_left - new_days_left

            # ROI
            cost_6m = monthly_cost * 6
            value_saved = timeline_improvement * 10000  # $10k/day value
            roi = value_saved / cost_6m

            avg_roi = float(np.mean(roi))
            recs = []
            if avg_roi > 1.5:
                recs.append(f"Strong ROI of {avg_roi:.1f}x — recommend proceeding with {count} {resource_type}")
            elif avg_roi > 1.0:
                recs.append(f"Moderate ROI of {avg_roi:.1f}x — proceed with monitoring")
            else:
                recs.append(f"Low ROI of {avg_roi:.1f}x — consider alternatives")
            recs.append(f"Focus new {resource_type} on highest-cascade-impact sites first")

            return {
                "simulation": f"Add {count} {resource_type}",
                "n_simulations": n_sims,
                "current_open_issues": total_issues,
                "results": {
                    "issues_resolved_90d": {
                        "P10": round(float(np.percentile(issues_resolved_90d, 10))),
                        "P50": round(float(np.percentile(issues_resolved_90d, 50))),
                        "P90": round(float(np.percentile(issues_resolved_90d, 90))),
                        "mean": round(float(np.mean(issues_resolved_90d))),
                    },
                    "timeline_improvement_days": {
                        "P10": round(float(np.percentile(timeline_improvement, 10)), 1),
                        "P50": round(float(np.percentile(timeline_improvement, 50)), 1),
                        "P90": round(float(np.percentile(timeline_improvement, 90)), 1),
                        "mean": round(float(np.mean(timeline_improvement)), 1),
                    },
                    "cost_6_months": cost_6m,
                    "roi": {
                        "P50": round(float(np.percentile(roi, 50)), 2),
                        "mean": round(avg_roi, 2),
                    },
                    "probability_positive_roi": round(float(np.mean(roi > 1.0)) * 100, 1),
                },
                "recommendations": recs,
            }
        except Exception as e:
            logger.error(f"simulate_add_resource failed: {e}\n{traceback.format_exc()}")
            return {"error": str(e)}

    def simulate_deadline(self, target_date: str) -> Dict[str, Any]:
        """Simulate probability of meeting a target completion date."""
        logger.info(f"[TOOL] simulate_deadline: target={target_date}")
        try:
            target = datetime.strptime(target_date, "%Y-%m-%d")

            # Get current trial state from DB
            sql = """
                SELECT
                    COUNT(*) AS total_patients,
                    SUM(CASE WHEN p.is_clean_patient THEN 1 ELSE 0 END) AS clean_patients,
                    SUM(CASE WHEN p.is_db_lock_ready THEN 1 ELSE 0 END) AS db_lock_ready,
                    SUM(p.open_queries_count) AS total_open_queries,
                    ROUND(AVG(p.dqi_score)::numeric, 1) AS avg_dqi,
                    COUNT(DISTINCT p.site_id) AS active_sites
                FROM patients p
            """
            data, err = self._execute_sql_raw(sql)
            if err:
                return {"error": err}

            state = data[0] if data else {}

            # Real pending SDV/signatures from visits table
            sdv_sql = """
                SELECT
                    COALESCE(SUM(CASE WHEN NOT v.sdv_complete THEN 1 ELSE 0 END), 0) AS pending_sdv,
                    COALESCE(SUM(CASE WHEN NOT v.data_entry_complete THEN 1 ELSE 0 END), 0) AS pending_de
                FROM visits v
            """
            sdv_data, _ = self._execute_sql_raw(sdv_sql)
            sdv_row = sdv_data[0] if sdv_data else {}

            from src.ml.simulation.monte_carlo_engine import MonteCarloEngine, TrialState
            engine = MonteCarloEngine(n_simulations=10000)

            trial_state = TrialState(
                study_id="portfolio",
                total_patients=int(state.get("total_patients", 0)),
                db_lock_ready=int(state.get("db_lock_ready", 0)),
                open_queries=int(state.get("total_open_queries", 0)),
                pending_signatures=int(sdv_row.get("pending_de", 0) or 0),
                pending_sdv=int(sdv_row.get("pending_sdv", 0) or 0),
                active_sites=int(state.get("active_sites", 0)),
                avg_dqi=float(state.get("avg_dqi", 75)),
            )
            trial_state._load_real_rates()

            result = engine.simulate_deadline_probability(trial_state, target)

            return {
                "simulation": f"Deadline Feasibility: {target_date}",
                "n_simulations": 10000,
                "current_state": {
                    "total_patients": int(state.get("total_patients", 0)),
                    "db_lock_ready": int(state.get("db_lock_ready", 0)),
                    "open_queries": int(state.get("total_open_queries", 0)),
                    "pending_sdv": int(sdv_row.get("pending_sdv", 0) or 0),
                    "pending_signatures": int(sdv_row.get("pending_de", 0) or 0),
                },
                "results": result,
            }
        except Exception as e:
            logger.error(f"simulate_deadline failed: {e}\n{traceback.format_exc()}")
            return {"error": str(e)}

    def simulate_process_change(self, process_name: str, expected_improvement: float = 0.15) -> Dict[str, Any]:
        """Simulate impact of a process improvement using 10k Monte Carlo iterations."""
        logger.info(f"[TOOL] simulate_process_change: {process_name} (improvement={expected_improvement})")
        try:
            import numpy as np

            # Pull current open-issue count
            sql = """
                SELECT COUNT(*) AS total_issues
                FROM project_issues pi WHERE pi.status != 'resolved'
            """
            data, err = self._execute_sql_raw(sql)
            if err:
                return {"error": err}
            total_issues = int(data[0].get("total_issues", 50000)) if data else 50000

            rng = np.random.default_rng(42)
            n_sims = 10000

            # Monte Carlo: triangular distributions for uncertain parameters
            improvement = rng.triangular(
                expected_improvement * 0.5, expected_improvement, expected_improvement * 1.5, n_sims
            )
            adoption_rate = rng.triangular(0.50, 0.80, 1.0, n_sims)
            impl_delay = rng.triangular(7, 14, 30, n_sims)

            effective_improvement = improvement * adoption_rate
            issues_reduced = total_issues * effective_improvement
            days_saved = issues_reduced / 100  # 100 issues/day baseline resolution
            net_days_saved = days_saved - impl_delay * (1 - adoption_rate)

            avg_eff = float(np.mean(effective_improvement))
            recs = [
                f"{'Strong' if avg_eff > 0.10 else 'Modest'} expected improvement of {avg_eff*100:.0f}%",
                f"Implementation delay: ~{np.mean(impl_delay):.0f} days — plan change management",
                f"Net days saved (P50): {np.percentile(net_days_saved, 50):.0f} days",
            ]

            return {
                "simulation": f"Process Change: {process_name}",
                "n_simulations": n_sims,
                "current_open_issues": total_issues,
                "results": {
                    "effective_improvement_pct": {
                        "P10": round(float(np.percentile(effective_improvement * 100, 10)), 1),
                        "P50": round(float(np.percentile(effective_improvement * 100, 50)), 1),
                        "P90": round(float(np.percentile(effective_improvement * 100, 90)), 1),
                        "mean": round(float(np.mean(effective_improvement) * 100), 1),
                    },
                    "issues_reduced": {
                        "P10": round(float(np.percentile(issues_reduced, 10))),
                        "P50": round(float(np.percentile(issues_reduced, 50))),
                        "P90": round(float(np.percentile(issues_reduced, 90))),
                        "mean": round(float(np.mean(issues_reduced))),
                    },
                    "net_days_saved": {
                        "P10": round(float(np.percentile(net_days_saved, 10)), 1),
                        "P50": round(float(np.percentile(net_days_saved, 50)), 1),
                        "P90": round(float(np.percentile(net_days_saved, 90)), 1),
                        "mean": round(float(np.mean(net_days_saved)), 1),
                    },
                    "probability_net_positive": round(float(np.mean(net_days_saved > 0)) * 100, 1),
                },
                "recommendations": recs,
            }
        except Exception as e:
            logger.error(f"simulate_process_change failed: {e}\n{traceback.format_exc()}")
            return {"error": str(e)}

    def simulate_dqi_fix(self, site_id: str = None, issue_types: list = None) -> Dict[str, Any]:
        """Simulate DQI improvement from fixing specific issue types."""
        logger.info(f"[TOOL] simulate_dqi_fix: site_id={site_id}, issues={issue_types}")
        try:
            # 8-component DQI model weights
            DQI_WEIGHTS = {
                "safety": 0.25, "query": 0.20, "completeness": 0.15,
                "coding": 0.12, "lab": 0.10, "sdv": 0.08,
                "signature": 0.05, "edrr": 0.05,
            }

            # Issue type → (DQI component, penalty per issue)
            ISSUE_MAP = {
                "open_query":        ("query",        3.0),
                "missing_visit":     ("completeness", 8.0),
                "pending_sae":       ("safety",      15.0),
                "unsigned_form":     ("signature",    5.0),
                "incomplete_sdv":    ("sdv",          4.0),
                "coding_issue":      ("coding",       5.0),
                "lab_issue":         ("lab",         10.0),
                "missing_page":      ("completeness", 6.0),
                "protocol_deviation":("safety",       8.0),
                "edrr_issue":        ("edrr",         7.0),
            }

            fix_types = issue_types if issue_types else list(ISSUE_MAP.keys())

            # Pull current DQI and issue metrics from DB
            where = f"WHERE p.site_id = '{site_id}'" if site_id else ""
            sql = f"""
                SELECT
                    ROUND(AVG(p.dqi_score)::numeric, 2) AS current_dqi,
                    COUNT(*) AS patient_count,
                    ROUND(AVG(p.open_queries_count)::numeric, 2) AS avg_open_queries,
                    ROUND(AVG(p.open_issues_count)::numeric, 2) AS avg_open_issues,
                    ROUND(AVG(p.pct_missing_visits)::numeric, 2) AS avg_missing_visits,
                    ROUND(AVG(p.pct_missing_pages)::numeric, 2) AS avg_missing_pages,
                    SUM(CASE WHEN p.has_sae THEN 1 ELSE 0 END) AS patients_with_sae,
                    SUM(p.sae_count) AS total_saes
                FROM patients p {where}
            """
            data, err = self._execute_sql_raw(sql)
            if err:
                return {"error": err}

            row = data[0] if data else {}
            current_dqi = float(row.get("current_dqi") or 75)
            patient_count = int(row.get("patient_count") or 0)
            avg_queries = float(row.get("avg_open_queries") or 0)
            avg_issues = float(row.get("avg_open_issues") or 0)
            avg_missing_visits = float(row.get("avg_missing_visits") or 0)
            total_saes = float(row.get("total_saes") or 0)

            # Estimate per-issue count and calculate penalty relief per component
            component_improvement = {}
            fix_details = []
            for issue_type in fix_types:
                if issue_type not in ISSUE_MAP:
                    continue
                component, penalty_per = ISSUE_MAP[issue_type]
                weight = DQI_WEIGHTS.get(component, 0.10)

                # Estimate avg count per patient for this issue type
                if issue_type == "open_query":
                    est_count = avg_queries
                elif issue_type == "missing_visit":
                    est_count = avg_missing_visits * 5
                elif issue_type == "missing_page":
                    est_count = float(row.get("avg_missing_pages") or 0) * 3
                elif issue_type == "pending_sae":
                    est_count = total_saes / max(patient_count, 1) * 0.3
                elif issue_type == "protocol_deviation":
                    est_count = avg_issues * 0.15
                else:
                    est_count = avg_issues * 0.12

                improvement = min(weight * 100, est_count * penalty_per * weight)
                component_improvement.setdefault(component, 0.0)
                component_improvement[component] += improvement
                fix_details.append({
                    "issue_type": issue_type,
                    "component": component,
                    "estimated_improvement": round(improvement, 2),
                })

            total_improvement = sum(component_improvement.values())
            projected_dqi = min(100, current_dqi + total_improvement)

            # Build component breakdown
            breakdown = {}
            for comp, imp in component_improvement.items():
                breakdown[comp] = {
                    "weight": f"{DQI_WEIGHTS.get(comp, 0) * 100:.0f}%",
                    "improvement": round(imp, 2),
                }

            # Recommendations
            recs = []
            if component_improvement:
                best = max(component_improvement, key=component_improvement.get)
                recs.append(f"Biggest DQI gain: fixing {best} issues (+{component_improvement[best]:.1f} pts)")
            if total_improvement > 5:
                recs.append(f"Significant DQI uplift possible (+{total_improvement:.1f} pts)")
            elif total_improvement > 1:
                recs.append(f"Moderate DQI uplift (+{total_improvement:.1f} pts) — target remaining bottlenecks")
            else:
                recs.append(f"Minimal DQI uplift (+{total_improvement:.1f} pts) — broader intervention needed")

            return {
                "simulation": "DQI Improvement Simulation",
                "scope": {"site_id": site_id or "all_sites", "patient_count": patient_count},
                "current_dqi": current_dqi,
                "projected_dqi": round(projected_dqi, 2),
                "total_improvement": round(total_improvement, 2),
                "issues_fixed": fix_types,
                "fix_details": fix_details,
                "component_breakdown": breakdown,
                "dqi_weights": DQI_WEIGHTS,
                "recommendations": recs,
            }
        except Exception as e:
            logger.error(f"simulate_dqi_fix failed: {e}\n{traceback.format_exc()}")
            return {"error": str(e)}

    # ── Dispatch ─────────────────────────────────────────────

    def execute(self, function_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch a function call to the appropriate handler."""
        handler = getattr(self, function_name, None)
        if handler is None:
            return {"error": f"Unknown tool: {function_name}"}
        try:
            return handler(**args)
        except Exception as e:
            logger.error(f"Tool {function_name} failed: {e}\n{traceback.format_exc()}")
            return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════
# AGENTIC ORCHESTRATOR V4
# ═══════════════════════════════════════════════════════════════

class AgentOrchestratorV4:
    """
    ReAct-style agentic orchestrator using Gemini function calling.
    The LLM autonomously decides which tools to call and chains them.
    """

    MAX_TOOL_ROUNDS = 4  # Safety limit — simple queries should finish in 1-2 rounds

    def __init__(self):
        config = get_config()
        self._client = genai.Client(api_key=config.gemini.api_key)
        self._model = config.gemini.model
        self._executor = ToolExecutor()
        self._tools = [types.Tool(function_declarations=TOOL_DECLARATIONS)]

        logger.info(f"AgentOrchestratorV4 initialised with {len(TOOL_DECLARATIONS)} tools, model={self._model}")

    async def run(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        Main entry point.  Sends the query to Gemini with tool declarations,
        then enters a loop: execute tool calls → feed results back → repeat
        until the LLM produces a final text answer.

        Args:
            query: The user's current question.
            context: Optional dict with role, study_id, etc.
            conversation_history: Optional list of prior {role, content} turns
                                  for multi-turn context (last N messages).
        """
        start_time = time.time()
        q = query.lower().strip()

        # ── Greeting short-circuit (only if no prior conversation) ─
        greetings = ["hi", "hello", "who are you", "what are you", "hey", "hola", "hi there"]
        if q in greetings and not conversation_history:
            return {
                "summary": (
                    "Hello! I'm the **Sanchalak AI Assistant** — your autonomous clinical trial analyst. "
                    "I can query the database, run Monte Carlo simulations, analyze issue cascades, "
                    "check model drift, and more — all from a single question.\n\n"
                    "Try asking me something like:\n"
                    "- *Which sites are most likely to delay DB lock, and why?*\n"
                    "- *Run a simulation for Study_1 timeline*\n"
                    "- *What are the top risk factors across the portfolio?*"
                ),
                "agent_chain": ["COMMUNICATOR"],
                "steps": [{"agent": "COMMUNICATOR", "thought": "Greeting.", "action": "respond_directly"}],
                "tools_used": ["conversation_engine"],
                "confidence": 1.0,
                "recommendations": [{"action": "Ask a clinical trial question to see the agent in action", "impact": "High"}],
            }

        # ── Build initial messages ──────────────────────────────
        role_ctx = ""
        if context and context.get("role"):
            role = context["role"].lower()
            role_map = {
                "lead": "User is a Study Lead — focus on strategic portfolio view and timelines.",
                "dm": "User is a Data Manager — focus on DQI, clean patients, DB-lock readiness.",
                "cra": "User is a CRA — focus on site monitoring, visit compliance, issue escalation.",
                "safety": "User is a Safety Officer — focus on SAE/AE, safety signals, regulatory.",
                "executive": "User is an Executive — focus on high-level KPIs and risk summary.",
            }
            role_ctx = role_map.get(role, "")

        system_instruction = SYSTEM_PROMPT
        if role_ctx:
            system_instruction += f"\n\nUSER ROLE CONTEXT: {role_ctx}"

        # ── Inject conversation history for multi-turn ──────────
        contents: List[types.Content] = []

        if conversation_history:
            # Keep last 6 turns (3 user + 3 assistant) to stay within token budget
            recent = conversation_history[-6:]
            for turn in recent:
                role_str = turn.get("role", "user")
                text = turn.get("content", "")
                if not text:
                    continue
                # Gemini uses "user" and "model" roles
                gemini_role = "model" if role_str == "assistant" else "user"
                contents.append(types.Content(
                    role=gemini_role,
                    parts=[types.Part.from_text(text=text)],
                ))

        # Always add the current query as the final user turn
        contents.append(types.Content(
            role="user",
            parts=[types.Part.from_text(text=query)],
        ))

        # ── Agent loop ──────────────────────────────────────────
        steps: List[Dict[str, Any]] = [
            {"agent": "SUPERVISOR", "thought": "Classifying intent and planning tool usage.", "action": "plan_tools"}
        ]
        tools_used: List[str] = []
        tool_round = 0
        final_text = ""

        while tool_round < self.MAX_TOOL_ROUNDS:
            tool_round += 1

            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        tools=self._tools,
                        system_instruction=system_instruction,
                        temperature=0.3,
                    ),
                )
            except Exception as e:
                logger.error(f"Gemini API error in round {tool_round}: {e}")
                final_text = f"I encountered an error communicating with the AI service: {str(e)}"
                break

            # Check if the response contains function calls
            candidate = response.candidates[0] if response.candidates else None
            if not candidate or not candidate.content or not candidate.content.parts:
                final_text = response.text or "I wasn't able to generate a response. Please try rephrasing your question."
                break

            parts = candidate.content.parts

            # Separate text parts and function calls
            text_parts = []
            function_calls = []
            for part in parts:
                if part.function_call:
                    function_calls.append(part.function_call)
                elif part.text:
                    text_parts.append(part.text)

            if not function_calls:
                # No more tool calls — we have the final answer
                final_text = "\n".join(text_parts) if text_parts else (response.text or "")
                break

            # ── Execute function calls ──────────────────────────
            # Add model's response (with function calls) to conversation
            contents.append(candidate.content)

            function_response_parts = []
            for fc in function_calls:
                fn_name = fc.name
                fn_args = dict(fc.args) if fc.args else {}

                logger.info(f"[Round {tool_round}] Calling tool: {fn_name}({json.dumps(fn_args, default=str)[:200]})")

                # Execute the tool
                result = self._executor.execute(fn_name, fn_args)
                tools_used.append(fn_name)

                # Truncate large results to avoid token overflow
                result_str = json.dumps(result, default=str)
                if len(result_str) > 15000:
                    # Keep structure but truncate data arrays
                    if isinstance(result, dict) and "data" in result and isinstance(result["data"], list):
                        result["data"] = result["data"][:15]
                        result["_truncated"] = True
                        result["_note"] = f"Showing 15 of {result.get('rows', '?')} rows"
                    result_str = json.dumps(result, default=str)[:15000]

                steps.append({
                    "agent": "EXECUTOR",
                    "thought": f"Executing tool: {fn_name}",
                    "action": fn_name,
                    "observation": f"Returned {len(result_str)} chars",
                })

                function_response_parts.append(
                    types.Part.from_function_response(
                        name=fn_name,
                        response=result,
                    )
                )

            # Add function responses to conversation
            contents.append(types.Content(
                role="user",
                parts=function_response_parts,
            ))

        # ── Force synthesis if loop exhausted without final text ─
        if not final_text and tools_used:
            logger.info("Max rounds reached — forcing synthesis without tools")
            try:
                # One more call WITHOUT tools to force a text answer
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_text(
                        text="You have gathered enough data. Now provide your final comprehensive analysis in markdown. Do NOT call any more tools."
                    )],
                ))
                synth_response = self._client.models.generate_content(
                    model=self._model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.3,
                    ),
                )
                final_text = synth_response.text or ""
            except Exception as e:
                logger.error(f"Forced synthesis failed: {e}")
                final_text = "I gathered data from multiple tools but was unable to synthesise a final response. Please try a more specific question."

        # ── Post-processing ─────────────────────────────────────
        elapsed_ms = (time.time() - start_time) * 1000

        # Clean up the response
        final_text = self._clean_response(final_text)

        # Compute confidence based on tool success
        tool_count = len(tools_used)
        errors = sum(1 for s in steps if "error" in s.get("observation", "").lower())
        confidence = min(0.98, 0.60 + (tool_count * 0.08) - (errors * 0.15))
        confidence = max(0.40, confidence)

        # Build agent chain
        agent_chain = ["SUPERVISOR"]
        if tools_used:
            agent_chain.extend(["DIAGNOSTIC", "EXECUTOR"])
        agent_chain.extend(["RESOLVER", "COMMUNICATOR"])

        # Add final steps
        steps.append({
            "agent": "RESOLVER",
            "thought": "Synthesizing multi-tool results into actionable analysis.",
            "action": "synthesize",
            "observation": f"Used {tool_count} tools across {tool_round} rounds.",
        })

        # Build recommendations
        recommendations = self._build_recommendations(tools_used, confidence)

        return {
            "summary": final_text,
            "agent_chain": agent_chain,
            "steps": steps,
            "tools_used": list(set(tools_used)),
            "confidence": round(confidence, 2),
            "recommendations": recommendations,
            "metadata": {
                "tool_calls": tool_count,
                "rounds": tool_round,
                "elapsed_ms": round(elapsed_ms, 0),
            },
        }

    @staticmethod
    def _clean_response(text: str) -> str:
        """Clean LLM output artifacts."""
        # Strip thinking tags
        for tag in ("think", "Analyze", "Understand", "Code"):
            text = re.sub(rf'<{tag}>[\s\S]*?</{tag}>', '', text)
        # Kill repetition bugs
        text = re.sub(r'(ML\|){2,}', 'Analysis', text)
        text = re.sub(r'(CRA\|){2,}', 'CRA Monitoring', text)
        return text.strip()

    @staticmethod
    def _build_recommendations(tools_used: List[str], confidence: float) -> List[Dict[str, str]]:
        recs = []
        if "run_monte_carlo_simulation" in tools_used:
            recs.append({"action": "Review the P50/P90 timeline estimates and plan resource allocation accordingly", "impact": "High"})
        if "get_cascade_analysis" in tools_used:
            recs.append({"action": "Address highest-cascade-impact issues first for maximum downstream benefit", "impact": "High"})
        if "simulate_site_closure" in tools_used:
            recs.append({"action": "Evaluate patient transfer logistics before finalizing site closure decision", "impact": "High"})
        if "simulate_add_resource" in tools_used:
            recs.append({"action": "Compare ROI across resource types before committing to hiring", "impact": "High"})
        if "simulate_deadline" in tools_used:
            recs.append({"action": "If probability < 70%, escalate and consider resource augmentation or scope reduction", "impact": "High"})
        if "simulate_process_change" in tools_used:
            recs.append({"action": "Build a change management plan to maximize adoption rate", "impact": "Medium"})
        if "simulate_dqi_fix" in tools_used:
            recs.append({"action": "Prioritize the component with largest projected DQI improvement", "impact": "High"})
        if confidence < 0.70:
            recs.append({"action": "Cross-check findings manually — confidence is below threshold", "impact": "High"})
        if not recs:
            recs.append({"action": "Review site-specific risk assessments for targeted follow-up actions", "impact": "Medium"})
        return recs


# ═══════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════

_orchestrator_v4: Optional[AgentOrchestratorV4] = None

def get_orchestrator_v4() -> AgentOrchestratorV4:
    global _orchestrator_v4
    if _orchestrator_v4 is None:
        _orchestrator_v4 = AgentOrchestratorV4()
    return _orchestrator_v4
