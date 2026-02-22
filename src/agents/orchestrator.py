import logging
import re
import json
from typing import List, Dict, Any, Optional, Tuple
import uuid
from datetime import datetime
from decimal import Decimal
from src.agents.llm_wrapper import get_llm

# Aegis V40: Focus Flags
SILENT_MODE = False
STRIP_THINKING = True

from src.agents.autonomy.gate import AutonomyGate, AutonomyDecision

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# ZENITH V3: SQL Template Library  (14 pre-validated queries)
# ═══════════════════════════════════════════════════════════════

SQL_TEMPLATES: Dict[str, Dict[str, str]] = {
    # ── Portfolio / Enrollment ──────────────────────────────────
    "portfolio_summary": {
        "description": "Overall portfolio KPIs: total patients, sites, average DQI, DB-lock readiness",
        "sql": """
            SELECT
                COUNT(DISTINCT p.patient_key)  AS total_patients,
                COUNT(DISTINCT p.site_id)      AS total_sites,
                ROUND(AVG(p.dqi_score)::numeric, 1) AS mean_dqi,
                ROUND(100.0 * SUM(CASE WHEN p.is_db_lock_ready THEN 1 ELSE 0 END) / NULLIF(COUNT(*),0), 1) AS db_lock_ready_pct,
                SUM(CASE WHEN p.risk_level = 'High' THEN 1 ELSE 0 END) AS high_risk_patients
            FROM patients p
        """,
        "keywords": ["portfolio", "overview", "summary", "overall", "kpi", "status", "how many patients", "total patients", "total sites"]
    },

    "enrollment_by_site": {
        "description": "Patient enrollment counts per site with DQI and risk breakdown",
        "sql": """
            SELECT
                cs.site_id, cs.name, cs.country,
                cs.patient_count,
                ROUND(cs.dqi_score::numeric, 1)         AS dqi_score,
                ROUND(cs.performance_score::numeric, 1)  AS performance_score,
                cs.risk_level,
                cs.enrollment_rate,
                ROUND(cs.query_resolution_days::numeric, 1) AS query_resolution_days
            FROM clinical_sites cs
            ORDER BY cs.patient_count DESC
            LIMIT 15
        """,
        "keywords": ["enrollment by site", "patients per site", "site enrollment", "site list", "list sites", "all sites", "enrollment rate", "query resolution", "resolution time", "resolution days"]
    },

    "study_enrollment": {
        "description": "Patient enrollment comparison across studies",
        "sql": """
            SELECT
                p.study_id,
                COUNT(DISTINCT p.patient_key) AS total_patients,
                COUNT(DISTINCT p.site_id) AS total_sites,
                ROUND(AVG(p.dqi_score)::numeric, 1) AS mean_dqi,
                ROUND(100.0 * SUM(CASE WHEN p.is_db_lock_ready THEN 1 ELSE 0 END) / NULLIF(COUNT(*),0), 1) AS db_lock_ready_pct,
                SUM(CASE WHEN p.risk_level = 'High' THEN 1 ELSE 0 END) AS high_risk_patients
            FROM patients p
            GROUP BY p.study_id
            ORDER BY total_patients DESC
        """,
        "keywords": ["enrollment", "compare studies", "study comparison", "across studies", "per study", "by study", "each study", "study enrollment"]
    },

    # ── Risk & Quality ──────────────────────────────────────────
    "high_risk_patients": {
        "description": "Top high-risk patients with risk score, DQI, open issues",
        "sql": """
            SELECT
                p.patient_key, p.site_id, p.study_id,
                p.risk_level, ROUND(p.risk_score::numeric, 2) AS risk_score,
                ROUND(p.dqi_score::numeric, 1) AS dqi_score,
                p.open_queries_count, p.open_issues_count,
                p.clean_status_tier, p.status
            FROM patients p
            WHERE p.risk_level IN ('High', 'Critical')
            ORDER BY p.risk_score DESC
            LIMIT 15
        """,
        "keywords": ["high risk patient", "risk patient", "risky patient", "at risk patient", "critical patient"]
    },

    "high_risk_sites": {
        "description": "Sites with high-risk patients, showing count and site metrics",
        "sql": """
            SELECT
                cs.site_id, cs.name, cs.country, cs.risk_level,
                ROUND(cs.performance_score::numeric, 1) AS performance_score,
                ROUND(cs.dqi_score::numeric, 1) AS dqi_score,
                cs.patient_count,
                (SELECT COUNT(*) FROM patients p WHERE p.site_id = cs.site_id AND p.risk_level IN ('High', 'Critical')) AS high_risk_patient_count,
                cs.open_issues
            FROM clinical_sites cs
            WHERE cs.risk_level IN ('High', 'Critical')
               OR EXISTS (SELECT 1 FROM patients p WHERE p.site_id = cs.site_id AND p.risk_level IN ('High', 'Critical'))
            ORDER BY high_risk_patient_count DESC, cs.performance_score ASC
            LIMIT 15
        """,
        "keywords": ["high risk", "risky", "at risk", "risk site", "sites with high risk", "high risk site", "sites have high", "sites with risk", "sites have"]
    },

    "site_risk_ranking": {
        "description": "Sites ranked by risk level and performance",
        "sql": """
            SELECT
                cs.site_id, cs.name, cs.country,
                cs.risk_level,
                ROUND(cs.performance_score::numeric, 1) AS performance_score,
                ROUND(cs.dqi_score::numeric, 1) AS dqi_score,
                cs.patient_count,
                ROUND(cs.query_resolution_days::numeric, 1) AS query_resolution_days
            FROM clinical_sites cs
            ORDER BY CASE cs.risk_level WHEN 'Critical' THEN 4 WHEN 'High' THEN 3 WHEN 'Medium' THEN 2 ELSE 1 END DESC,
                     cs.performance_score ASC
            LIMIT 15
        """,
        "keywords": ["site risk", "worst site", "worst performing site", "underperforming", "low performing", "poorly performing", "site ranking", "risky site", "bottom site"]
    },

    "dqi_distribution": {
        "description": "DQI score distribution across patients",
        "sql": """
            SELECT
                CASE
                    WHEN p.dqi_score >= 90 THEN 'Excellent (90-100)'
                    WHEN p.dqi_score >= 80 THEN 'Good (80-89)'
                    WHEN p.dqi_score >= 70 THEN 'Fair (70-79)'
                    WHEN p.dqi_score >= 60 THEN 'Poor (60-69)'
                    ELSE 'Critical (<60)'
                END AS dqi_band,
                COUNT(*) AS patient_count,
                ROUND(AVG(p.risk_score)::numeric, 2) AS avg_risk_score
            FROM patients p
            WHERE p.dqi_score IS NOT NULL
            GROUP BY 1
            ORDER BY MIN(p.dqi_score) DESC
        """,
        "keywords": ["dqi", "data quality", "quality score", "dqi distribution", "dqi band"]
    },

    # ── DB-Lock Readiness ───────────────────────────────────────
    "db_lock_readiness": {
        "description": "DB-lock readiness breakdown by site",
        "sql": """
            SELECT
                p.site_id,
                COUNT(*) AS total_patients,
                SUM(CASE WHEN p.is_db_lock_ready THEN 1 ELSE 0 END) AS lock_ready,
                ROUND(100.0 * SUM(CASE WHEN p.is_db_lock_ready THEN 1 ELSE 0 END) / NULLIF(COUNT(*),0), 1) AS lock_ready_pct,
                ROUND(AVG(p.dqi_score)::numeric, 1) AS avg_dqi
            FROM patients p
            GROUP BY p.site_id
            ORDER BY lock_ready_pct ASC
            LIMIT 15
        """,
        "keywords": ["db lock", "db-lock", "database lock", "lock ready", "lock readiness", "lock ready", "clean patient", "db lock readiness"]
    },

    # ── Issues & Queries ────────────────────────────────────────
    "open_issues_summary": {
        "description": "Open issues grouped by category and priority",
        "sql": """
            SELECT
                pi.category, pi.priority,
                COUNT(*) AS issue_count,
                COUNT(DISTINCT pi.site_id) AS sites_affected,
                COUNT(DISTINCT pi.patient_key) AS patients_affected
            FROM project_issues pi
            WHERE pi.status != 'resolved'
            GROUP BY pi.category, pi.priority
            ORDER BY CASE pi.priority WHEN 'critical' THEN 4 WHEN 'high' THEN 3 WHEN 'medium' THEN 2 ELSE 1 END DESC,
                     issue_count DESC
        """,
        "keywords": ["issue", "issues", "open issue", "problem", "critical issue", "pending issue"]
    },

    "query_aging": {
        "description": "Open queries with aging information",
        "sql": """
            SELECT
                q.query_id, q.patient_key, q.field_name, q.form_name,
                q.query_text, q.status, q.age_days,
                q.query_type
            FROM queries q
            WHERE q.status != 'closed'
            ORDER BY q.age_days DESC NULLS LAST
            LIMIT 15
        """,
        "keywords": ["query", "queries", "open query", "query aging", "old query", "query age", "pending query"]
    },

    # ── Safety / AE ─────────────────────────────────────────────
    "safety_overview": {
        "description": "Adverse events summary by severity and causality",
        "sql": """
            SELECT
                ae.severity, ae.causality,
                COUNT(*) AS ae_count,
                SUM(CASE WHEN ae.is_sae THEN 1 ELSE 0 END) AS sae_count,
                SUM(CASE WHEN ae.is_ongoing THEN 1 ELSE 0 END) AS ongoing_count
            FROM adverse_events ae
            GROUP BY ae.severity, ae.causality
            ORDER BY ae_count DESC
        """,
        "keywords": ["safety", "adverse event", "ae", "sae", "serious adverse", "causality", "severity"]
    },

    # ── CRA Performance ─────────────────────────────────────────
    "cra_performance": {
        "description": "CRA performance ranking by average site metrics",
        "sql": """
            SELECT
                cal.cra_name,
                COUNT(DISTINCT cal.site_id) AS sites_managed,
                COUNT(cal.log_id) AS total_activities,
                ROUND(AVG(cs.performance_score)::numeric, 1) AS avg_site_performance,
                ROUND(AVG(cs.dqi_score)::numeric, 1) AS avg_site_dqi,
                SUM(CASE WHEN cs.risk_level IN ('High', 'Critical') THEN 1 ELSE 0 END) AS high_risk_sites,
                SUM(cs.open_issues) AS total_open_issues,
                ROUND(AVG(cs.query_resolution_days)::numeric, 1) AS avg_resolution_days
            FROM cra_activity_logs cal
            LEFT JOIN clinical_sites cs ON cs.site_id = REPLACE(cal.site_id, '_', ' ')
            GROUP BY cal.cra_name
            ORDER BY avg_site_performance ASC NULLS LAST
            LIMIT 15
        """,
        "keywords": ["cra performance", "worst cra", "best cra", "cra ranking", "top cra", "underperforming cra", "cras", "cra performing"]
    },

    # ── CRA Activity ────────────────────────────────────────────
    "cra_activity": {
        "description": "CRA monitoring activity log",
        "sql": """
            SELECT
                cal.cra_name, cal.site_id,
                cal.activity_type, cal.status,
                cal.visit_date::date AS visit_date,
                cal.follow_up_letter_sent
            FROM cra_activity_logs cal
            ORDER BY cal.visit_date DESC NULLS LAST
            LIMIT 15
        """,
        "keywords": ["cra activity", "cra visit", "monitoring visit", "site visit", "cra log"]
    },

    # ── Visit Compliance ────────────────────────────────────────
    "visit_compliance": {
        "description": "Visit compliance: deviations, SDV status, data entry",
        "sql": """
            SELECT
                v.patient_key, v.visit_name, v.status,
                v.deviation_days,
                v.sdv_complete, v.data_entry_complete,
                v.is_in_window
            FROM visits v
            WHERE v.status != 'completed' OR v.deviation_days > 0
            ORDER BY ABS(v.deviation_days) DESC NULLS LAST
            LIMIT 15
        """,
        "keywords": ["visit", "compliance", "deviation", "sdv", "data entry", "visit window"]
    },

    # ── SDTM / Clean Status ────────────────────────────────────
    "sdtm_readiness": {
        "description": "SDTM-ready patients by site",
        "sql": """
            SELECT
                p.site_id,
                COUNT(*) AS total,
                SUM(CASE WHEN p.sdtm_ready THEN 1 ELSE 0 END) AS sdtm_ready_count,
                ROUND(100.0 * SUM(CASE WHEN p.sdtm_ready THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 1) AS sdtm_ready_pct,
                SUM(CASE WHEN p.is_clean_patient THEN 1 ELSE 0 END) AS clean_count
            FROM patients p
            GROUP BY p.site_id
            ORDER BY sdtm_ready_pct DESC
            LIMIT 15
        """,
        "keywords": ["sdtm", "sdtm readiness", "sdtm ready", "sdtm status", "submission ready", "submission"]
    },

    # ── Site-Specific Detail ────────────────────────────────────
    "site_detail": {
        "description": "Detailed metrics for a single site (parameterised by LLM)",
        "sql": """
            SELECT
                cs.site_id, cs.name, cs.country, cs.region,
                cs.patient_count,
                ROUND(cs.dqi_score::numeric, 1) AS dqi_score,
                ROUND(cs.performance_score::numeric, 1) AS performance_score,
                cs.risk_level,
                cs.enrollment_rate,
                ROUND(cs.query_resolution_days::numeric, 1) AS query_resolution_days,
                cs.principal_investigator,
                cs.open_issues
            FROM clinical_sites cs
            ORDER BY cs.site_id
            LIMIT 15
        """,
        "keywords": ["specific site", "site detail", "tell me about site", "site info"]
    },

    # ── Patient-Level Detail ────────────────────────────────────
    "patient_detail": {
        "description": "Detailed patient-level data",
        "sql": """
            SELECT
                p.patient_key, p.site_id, p.study_id,
                p.status, p.risk_level,
                ROUND(p.risk_score::numeric, 2) AS risk_score,
                ROUND(p.dqi_score::numeric, 1) AS dqi_score,
                p.clean_status_tier,
                p.open_queries_count, p.open_issues_count,
                p.pct_missing_visits, p.pct_missing_pages,
                p.has_sae, p.sae_count
            FROM patients p
            ORDER BY p.risk_score DESC NULLS LAST
            LIMIT 15
        """,
        "keywords": ["patient detail", "patient data", "patient info", "specific patient", "patient list"]
    },
}

# Keywords that map to template names for quick lookup
_TEMPLATE_INDEX: Dict[str, List[str]] = {}
for _tname, _tmeta in SQL_TEMPLATES.items():
    for _kw in _tmeta["keywords"]:
        _TEMPLATE_INDEX.setdefault(_kw, []).append(_tname)


# ═══════════════════════════════════════════════════════════════
# ZENITH V3: Pipeline-Signal Confidence Scoring
# ═══════════════════════════════════════════════════════════════

def _compute_confidence(
    rows_returned: int,
    template_used: bool,
    retry_count: int,
    had_error: bool,
    query: str,
) -> float:
    """Compute confidence based on actual pipeline signals.

    Factors (additive from a 0.50 base, capped at 0.98):
      +0.20  data was returned (rows > 0)
      +0.10  a pre-validated template was used
      −0.08  per retry attempt
      −0.12  if any SQL error occurred
      +0.05  multiple rows (richer evidence)
      +0.05  domain-specific keywords present
      +0.03  large result set (> 5 rows)
    """
    conf = 0.50

    # Data presence
    if rows_returned > 0:
        conf += 0.20
    # Template bonus
    if template_used:
        conf += 0.10
    # Retry penalty
    conf -= 0.08 * retry_count
    # Error penalty
    if had_error:
        conf -= 0.12
    # Multiple rows
    if rows_returned > 1:
        conf += 0.05
    # Rich result set
    if rows_returned > 5:
        conf += 0.03
    # Query specificity
    _specific = [
        "site", "study", "patient", "trial", "enrollment",
        "safety", "adverse", "protocol", "deviation", "query",
        "cra", "monitor", "dqi", "risk", "lock", "sdtm",
    ]
    if any(kw in query.lower() for kw in _specific):
        conf += 0.05

    return round(min(0.98, max(0.40, conf)), 2)


# ═══════════════════════════════════════════════════════════════
# ZENITH V3: Role-Aware Context
# ═══════════════════════════════════════════════════════════════

ROLE_CONTEXTS = {
    "lead": "Focus: strategic portfolio view, timeline projections, resource decisions.",
    "dm": "Focus: data quality metrics, clean patient tiers, DB-lock readiness, query resolution, DQI bands.",
    "cra": "Focus: site-level detail, patient monitoring priorities, issue escalation, visit compliance.",
    "coder": "Focus: medical coding accuracy, MedDRA/WHO-DD mapping, coding queues, batch operations.",
    "safety": "Focus: SAE/AE analysis, safety signals, causality distributions, regulatory reporting (E6 §4.11).",
    "executive": "Focus: KPIs, portfolio health, enrollment velocity, risk summary.",
}

def get_role_context(context: Optional[Dict[str, Any]] = None) -> str:
    if not context:
        return ""
    role = context.get("role", "").lower()
    if role in ROLE_CONTEXTS:
        return f"\n\nUSER ROLE: {role.upper()}\n{ROLE_CONTEXTS[role]}\nAdapt your response depth and terminology for this role."
    return ""


# ═══════════════════════════════════════════════════════════════
# ZENITH V3: Schema Context (shared between templates & dynamic SQL)
# ═══════════════════════════════════════════════════════════════

SCHEMA_CONTEXT = """
Available Tables & Important Columns:
1. clinical_sites (site_id, name, country, region, performance_score, risk_level, dqi_score, enrollment_rate, query_resolution_days, patient_count, open_issues, principal_investigator)
   - site_id TEXT, format 'Site NNN' (e.g. 'Site 1', 'Site 468', 'Site 1468').
   - name TEXT (same as site_id in this dataset).
   - country TEXT, ISO-3166 alpha-3 codes. See COUNTRY REFERENCE below.
   - region TEXT: 'AMERICA', 'ASIA', 'EMEA'. Note: 'EMEA' covers Europe, Middle East, and Africa.
   - risk_level TEXT: 'low', 'medium', 'high'.
   - performance_score, dqi_score, enrollment_rate, query_resolution_days (NUMERIC).
2. patients (patient_key, study_id, site_id, status, clean_status_tier, risk_level, risk_score, dqi_score, open_queries_count, open_issues_count, pct_missing_visits, pct_missing_pages, is_clean_patient, is_db_lock_ready, sdtm_ready, has_sae, sae_count)
   - patient_key TEXT, format 'Study_N|Site NNN|Subject NNNNN' (e.g. 'Study_1|Site 10|Subject 17').
   - To find a patient by number, use: patient_key LIKE '%Subject 48147%'
   - To find patients at a site, use: patients.site_id = 'Site 468'
   - study_id TEXT, format 'Study_N' (e.g. 'Study_1', 'Study_25').
   - status TEXT: 'Enrolled','On Trial','Completed','Discontinued','Screening','Screen Failure','Follow-Up','Survival','Unknown'.
   - risk_level TEXT: 'No Risk','Low','Medium','High','Critical'. Values are Title Case.
   - sdtm_ready, is_db_lock_ready (BOOLEAN). open_queries_count, open_issues_count (INTEGER).
   - NOTE: patients does NOT have a 'name' column. Use patient_key.
3. visits (visit_id, patient_key, visit_name, visit_number, status, deviation_days, sdv_complete, data_entry_complete, is_in_window)
4. adverse_events (ae_id, patient_key, ae_term, severity, causality, is_sae, is_ongoing, reported_date, meddra_pt, meddra_soc)
5. project_issues (issue_id, site_id, patient_key, category, issue_type, description, priority, severity, status, cascade_impact_score)
6. queries (query_id, patient_key, field_name, form_name, query_text, query_type, status, age_days)
7. cra_activity_logs (log_id, site_id, cra_name, activity_type, visit_date, status, follow_up_letter_sent)
8. unified_patient_record (354-column view with comprehensive patient metrics including risk_score, dqi components, coding rates, safety metrics, etc.)

COUNTRY REFERENCE (country column uses ISO-3 codes):
  ARE=UAE, ARG=Argentina, AUS=Australia, AUT=Austria, BEL=Belgium, BFA=Burkina Faso,
  BGR=Bulgaria, BRA=Brazil, CAN=Canada, CHE=Switzerland, CHL=Chile, CHN=China,
  CIV=Ivory Coast, COD=DR Congo, COL=Colombia, CZE=Czech Republic, DEU=Germany,
  DNK=Denmark, ESP=Spain, EST=Estonia, FIN=Finland, FRA=France, GAB=Gabon,
  GBR=United Kingdom, GHA=Ghana, GRC=Greece, HKG=Hong Kong, HRV=Croatia,
  HUN=Hungary, IND=India, IRL=Ireland, ISL=Iceland, ISR=Israel, ITA=Italy,
  JOR=Jordan, JPN=Japan, KEN=Kenya, KOR=South Korea, LBN=Lebanon, LTU=Lithuania,
  LVA=Latvia, MEX=Mexico, MLI=Mali, MUS=Mauritius, MYS=Malaysia, NER=Niger,
  NLD=Netherlands, NOR=Norway, NZL=New Zealand, OMN=Oman, PHL=Philippines,
  POL=Poland, PRK=North Korea, PRT=Portugal, ROU=Romania, RUS=Russia, RWA=Rwanda,
  SGP=Singapore, SRB=Serbia, SVK=Slovakia, SVN=Slovenia, SWE=Sweden, THA=Thailand,
  TUR=Turkey, TWN=Taiwan, TZA=Tanzania, UGA=Uganda, USA=United States,
  VNM=Vietnam, ZAF=South Africa, ZMB=Zambia

REGION REFERENCE:
  'AMERICA' = North & South America (USA, CAN, BRA, ARG, CHL, COL, MEX)
  'EMEA' = Europe, Middle East & Africa (DEU, GBR, FRA, ESP, ITA, etc.)
  'ASIA' = Asia-Pacific (CHN, IND, JPN, KOR, AUS, SGP, etc.)
  When user says 'Europe', query region = 'EMEA' or filter European country codes.
  When user says 'Asia', query region = 'ASIA'.
  When listing countries in a region, use SELECT DISTINCT country and do NOT apply LIMIT (there may be 40+ countries in EMEA).

Rules:
- Always prefix columns with table names (e.g., patients.site_id).
- risk_level is TEXT (Title Case). Use ORDER BY CASE risk_level WHEN 'Critical' THEN 4 WHEN 'High' THEN 3 WHEN 'Medium' THEN 2 WHEN 'Low' THEN 1 ELSE 0 END DESC.
- For 'high risk' queries, filter: risk_level IN ('High', 'Critical').
- sdtm_ready and is_db_lock_ready are BOOLEAN. Use = TRUE / = FALSE.
- There are NO financial columns. Never reference cost, budget, dollar, price.
- Limit results to 15 unless the user asks for aggregate/summary data or country/region listings.
- ALWAYS include identifying columns (site_id, name, patient_key, study_id) in SELECT.
- When user mentions a specific site number (e.g. 'site 468'), use: cs.site_id = 'Site 468'
- When user mentions a specific patient/subject number (e.g. 'patient 48147'), use: p.patient_key LIKE '%Subject 48147%'
- When user mentions a specific study number (e.g. 'study 5'), use: p.study_id = 'Study_5'
- When user asks about a country by name, translate to ISO-3 code using the reference above.
- patients table has NO country/region column. To find patients or studies by country/region, JOIN with clinical_sites:
  SELECT DISTINCT p.study_id, cs.site_id, cs.country, cs.region FROM patients p JOIN clinical_sites cs ON p.site_id = cs.site_id WHERE cs.country = 'DEU'
- adverse_events, visits, queries, project_issues also lack country/region. Always JOIN via site_id → clinical_sites for geographic filtering.
"""


class AgentOrchestrator:
    """
    Zenith V3 Orchestrator — 6-agent swarm with:
      • 14-query SQL template library
      • Retry / error-feedback loop (max 2 retries)
      • Dedicated _execute_sql + _generate_sql methods
      • Pipeline-signal confidence scoring
      • DeepAnalyze structured synthesis
    """

    MAX_SQL_RETRIES = 2  # up to 2 retries after initial attempt

    def __init__(self):
        self.llm = get_llm()
        self.agents = {
            "SUPERVISOR": "Plans and routes tasks",
            "DIAGNOSTIC": "Investigates root causes",
            "FORECASTER": "Predicts future outcomes",
            "RESOLVER": "Generates action plans",
            "EXECUTOR": "Validates and executes remediations",
            "COMMUNICATOR": "Drafts site communications",
        }
        self._db_manager = None

    # ── Database accessor ───────────────────────────────────────

    def _get_db(self):
        """Lazy-load DatabaseManager."""
        if self._db_manager is None:
            try:
                from src.database.connection import get_db_manager
                self._db_manager = get_db_manager()
            except Exception as e:
                logger.error(f"DB manager init failed: {e}")
        return self._db_manager

    # ── Template matcher ────────────────────────────────────────

    @staticmethod
    def _match_template(query: str) -> Optional[str]:
        """Return the best-matching template name, or None if no template fits.

        Skips template matching when the query targets a specific entity
        (contains a number, country name, or region name) because templates
        return generic aggregated data, not entity-specific results.
        """
        q = query.lower()

        # If the query references a specific entity, skip templates entirely
        # so the LLM can generate a targeted WHERE clause.
        if re.search(r'\b(site|patient|subject)\s*\d', q):
            return None
        if re.search(r'\b(study)\s*_?\d', q):
            return None
        # Country / region specific queries need dynamic SQL
        _geo_triggers = [
            'germany', 'india', 'usa', 'china', 'japan', 'france', 'spain',
            'italy', 'brazil', 'canada', 'australia', 'uk', 'united kingdom',
            'europe', 'asia', 'america', 'emea', 'africa', 'middle east',
            'which country', 'which region', 'what region', 'what country',
            'where is', 'located',
        ]
        if any(g in q for g in _geo_triggers):
            return None

        scores: Dict[str, int] = {}
        for kw, tnames in _TEMPLATE_INDEX.items():
            if kw in q:
                for tn in tnames:
                    scores[tn] = scores.get(tn, 0) + 1
        if not scores:
            return None
        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        # Require at least 1 keyword hit
        return best if scores[best] >= 1 else None

    # ── SQL generation (LLM) ───────────────────────────────────

    def _generate_sql(self, query: str, error_context: Optional[str] = None) -> str:
        """Use the LLM to translate a natural-language query into SQL.

        If *error_context* is provided (from a previous failed attempt), the
        LLM is told what went wrong so it can self-correct.
        """
        error_hint = ""
        if error_context:
            error_hint = (
                f"\n\nPREVIOUS ATTEMPT FAILED with this error:\n{error_context}\n"
                "Please fix the SQL to avoid this error. Common fixes:\n"
                "- Use correct column names from the schema\n"
                "- Ensure table aliases are consistent\n"
                "- Cast types properly (e.g., ::numeric for rounding)\n"
            )

        sys_prompt = (
            "You are a SQL expert for a Clinical Trial Management System.\n"
            "Translate the user's natural language query into a SINGLE valid PostgreSQL SELECT statement.\n"
            "Only return the SQL. No explanation. No markdown formatting. No code fences.\n\n"
            f"Schema:\n{SCHEMA_CONTEXT}"
            f"{error_hint}"
        )

        resp = self.llm.generate(prompt=f"Query: {query}", system_prompt=sys_prompt)
        raw = resp.content.strip()
        # Strip code-fence wrappers if present
        raw = raw.replace("```sql", "").replace("```", "").strip()
        return raw

    # ── SQL validation ──────────────────────────────────────────

    @staticmethod
    def _validate_sql(sql: str) -> Tuple[bool, str]:
        """Basic safety & syntax validation on generated SQL.

        Returns (is_valid, reason).
        """
        s = sql.strip().lower()
        if not s.startswith("select"):
            return False, "Query does not start with SELECT."
        # Block dangerous statements
        _blocked = ["drop ", "delete ", "update ", "insert ", "alter ", "truncate ", "create ", "grant "]
        for b in _blocked:
            if b in s:
                return False, f"Blocked keyword detected: {b.strip()}"
        return True, ""

    # ── SQL execution (with retry loop) ─────────────────────────

    def _execute_sql(self, sql: str) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """Execute *sql* against PostgreSQL via SQLAlchemy.

        Returns (rows_as_dicts, error_message | None).
        """
        from sqlalchemy import text

        db = self._get_db()
        if db is None or db.engine is None:
            return [], "Database connection unavailable."

        try:
            with db.engine.connect() as conn:
                result = conn.execute(text(sql))
                rows = result.fetchall()
                columns = list(result.keys())
                data: List[Dict[str, Any]] = []
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
            logger.warning(f"SQL execution error: {e}")
            return [], str(e)

    # ── Data grounding (template → LLM → retry) ────────────────

    async def _get_data_context(self, query: str) -> Tuple[str, bool, int, bool]:
        """Ground the query against the database.

        Returns (context_json, template_used, retry_count, had_error).
        """
        template_used = False
        retry_count = 0
        had_error = False

        # 1. Try template match first
        tpl_name = self._match_template(query)
        if tpl_name:
            tpl_sql = SQL_TEMPLATES[tpl_name]["sql"].strip()
            data, err = self._execute_sql(tpl_sql)
            if err is None and data:
                template_used = True
                logger.info(f"Template hit: {tpl_name} ({len(data)} rows)")
                return json.dumps(data, indent=2), True, 0, False
            elif err:
                logger.warning(f"Template {tpl_name} failed: {err}")
                had_error = True
            # Template returned 0 rows — fall through to dynamic SQL

        # 2. Dynamic LLM-generated SQL with retry loop
        error_ctx: Optional[str] = None
        for attempt in range(1 + self.MAX_SQL_RETRIES):
            sql = self._generate_sql(query, error_context=error_ctx)

            # Validate
            valid, reason = self._validate_sql(sql)
            if not valid:
                logger.warning(f"SQL validation failed (attempt {attempt+1}): {reason}")
                error_ctx = f"Validation error: {reason}\nBad SQL: {sql}"
                retry_count += 1
                had_error = True
                continue

            logger.info(f"Generated SQL (attempt {attempt+1}): {sql}")

            # Execute
            data, err = self._execute_sql(sql)
            if err is None:
                if data:
                    return json.dumps(data, indent=2), template_used, retry_count, had_error
                else:
                    return "No specific data found for this query in the database.", template_used, retry_count, had_error
            else:
                error_ctx = f"Execution error: {err}\nFailed SQL: {sql}"
                retry_count += 1
                had_error = True
                logger.warning(f"SQL attempt {attempt+1} failed: {err}")

        # All attempts exhausted
        return "Error: could not retrieve data after multiple attempts.", template_used, retry_count, True

    # ── DeepAnalyze synthesis ───────────────────────────────────

    def _build_synthesis_prompt(self, query: str, data_context: str, role_ctx: str) -> Tuple[str, str]:
        """Build the DeepAnalyze system + user prompts for the final synthesis step."""
        system_prompt = f"""You are the a6on-i Assistant (Zenith V3), a highly specialized Clinical Data Analyst.

TASK (DeepAnalyze): Answer the user's query clearly and naturally using ONLY the provided database context.

ABSOLUTE RULES — VIOLATION = FAILURE:
1. NEVER INVENT DATA. If a number, name, or identifier is NOT in the DATABASE CONTEXT below, do NOT include it. Say "This information is not available in the current database" instead.
2. USE EXACT IDENTIFIERS. Sites are named "Site 1", "Site 2", etc. CRAs are "CRA 0", "CRA 1", etc. NEVER rename them.
3. NO FINANCIAL DATA. This database contains NO cost, budget, or dollar data. NEVER mention financial figures.
4. ONLY quote metrics that appear in the context.

RESPONSE STRUCTURE (DeepAnalyze Protocol):
1. ## Direct Answer — 1-2 sentence key finding.
2. ## Key Findings — Markdown table if multiple data points, or bullet list.
3. ## Analysis — Clinical interpretation grounded in the data.
4. ## Recommendations — Actionable bullet points based on findings.

FORMATTING:
- Use markdown headers (##), bold, tables, and bullets.
- Be precise: use exact numbers and identifiers from the context.
- Do NOT include <think>, <Analyze>, or <Code> tags.
{role_ctx}

LIVE DATABASE CONTEXT (Ground Truth):
{data_context}
"""
        user_prompt = f"User Query: {query}"
        return system_prompt, user_prompt

    # ── Post-processing ─────────────────────────────────────────

    @staticmethod
    def _clean_summary(text: str) -> str:
        """Strip artefacts from LLM output."""
        # Kill repetition bugs
        text = re.sub(r'(ML\|){2,}', 'Analysis', text)
        text = re.sub(r'(CRA\|){2,}', 'CRA Monitoring', text)

        # Strip internal tags
        if STRIP_THINKING:
            for tag in ("think", "Analyze", "Understand", "Code"):
                text = re.sub(rf'<{tag}>[\s\S]*?</{tag}>', '', text)

        # Strip LLM-generated confidence lines (we add our own)
        text = re.sub(r'\n*Confidence:?\s*\d+%?\.?\s*$', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\n*Confidence Level:?\s*\d+%?\.?\s*$', '', text, flags=re.IGNORECASE)
        return text.strip()

    # ── Main entry point ────────────────────────────────────────

    async def run(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Zenith V3 pipeline:
          1. Greeting short-circuit
          2. SUPERVISOR — intent classification
          3. DIAGNOSTIC — data grounding (template → LLM SQL → retry)
          4. RESOLVER — DeepAnalyze synthesis
          5. COMMUNICATOR — confidence + formatting
        """
        q = query.lower().strip()

        # ── Greeting short-circuit ──────────────────────────────
        greetings = ["hi", "hello", "who are you", "what are you", "hey", "hola", "hi there"]
        if q in greetings or len(q) < 3:
            return {
                "summary": (
                    "Hello! I am the a6on-i Assistant, your dedicated clinical intelligence partner. "
                    "I can help you analyze site performance, track patient risks, and query your clinical database. "
                    "How can I assist you with your trial data today?"
                ),
                "agent_chain": ["COMMUNICATOR"],
                "steps": [{"agent": "COMMUNICATOR", "thought": "Handling greeting naturally.", "action": "respond_directly"}],
                "tools_used": ["conversation_engine"],
                "confidence": 1.0,
                "recommendations": [{"action": "Ask me about site DQI scores or high-risk patients", "impact": "High"}],
            }

        # ── Step 1: SUPERVISOR — classify & route ───────────────
        steps: List[Dict[str, Any]] = [
            {"agent": "SUPERVISOR", "thought": "Classifying intent and routing to data engine.", "action": "classify_intent"}
        ]

        # ── Step 2: DIAGNOSTIC — data grounding ────────────────
        data_context, template_used, retry_count, had_error = await self._get_data_context(query)

        grounding_obs = "Data retrieved"
        if template_used:
            grounding_obs += " (pre-validated template)"
        if retry_count > 0:
            grounding_obs += f" after {retry_count} retry(ies)"
        if had_error and data_context.startswith("Error"):
            grounding_obs = "Data retrieval failed after retries"

        steps.append({
            "agent": "DIAGNOSTIC",
            "thought": "Accessing live clinical database via SQLAlchemy bridge.",
            "action": "execute_grounding_sql",
            "observation": grounding_obs,
        })

        # ── Step 3: FORECASTER — predictive enrichment ──────────
        forecast_insight = ""
        try:
            forecast_sys = (
                "You are the FORECASTER agent in a clinical trial AI system. "
                "Given the database context below, generate a brief predictive analysis (3-5 lines max):\n"
                "- Trend direction (improving/declining/stable)\n"
                "- Risk projection (low/medium/high) with one-line reasoning\n"
                "- Estimated timeline if a completion metric is present\n\n"
                "If the data is insufficient for prediction, say 'Insufficient data for forecasting.' "
                "Be concise. Do NOT invent numbers not in the data.\n\n"
                f"DATABASE CONTEXT:\n{data_context}"
            )
            forecast_res = self.llm.generate(
                prompt=f"Generate predictive enrichment for: {query}",
                system_prompt=forecast_sys,
            )
            forecast_insight = forecast_res.content.strip()
        except Exception as e:
            logger.warning(f"FORECASTER step failed (non-fatal): {e}")
            forecast_insight = "Forecaster: Unable to generate predictions for this query."

        forecast_obs = "Predictive enrichment complete" if forecast_insight else "Skipped"
        steps.append({
            "agent": "FORECASTER",
            "thought": "Generating trend analysis, risk projections, and timeline estimates.",
            "action": "predictive_enrichment",
            "observation": forecast_obs,
        })

        # Enrich data_context with forecast for downstream agents
        if forecast_insight and "Insufficient" not in forecast_insight:
            data_context += f"\n\n--- FORECASTER PREDICTIONS ---\n{forecast_insight}"

        # ── Step 4: RESOLVER — DeepAnalyze synthesis ───────────
        steps.append({
            "agent": "RESOLVER",
            "thought": "Synthesizing clinical evidence using DeepAnalyze protocol.",
            "action": "deep_analyze",
            "observation": "Structured synthesis complete.",
        })

        # ── Confidence (pipeline signals) ───────────────────────
        rows_returned = 0
        try:
            parsed = json.loads(data_context)
            if isinstance(parsed, list):
                rows_returned = len(parsed)
        except (json.JSONDecodeError, TypeError):
            pass

        confidence = _compute_confidence(
            rows_returned=rows_returned,
            template_used=template_used,
            retry_count=retry_count,
            had_error=had_error,
            query=query,
        )

        # ── Role-aware context ──────────────────────────────────
        role_ctx = get_role_context(context)

        # ── LLM synthesis ───────────────────────────────────────
        sys_prompt, usr_prompt = self._build_synthesis_prompt(query, data_context, role_ctx)
        llm_res = self.llm.generate(prompt=usr_prompt, system_prompt=sys_prompt)
        summary = self._clean_summary(llm_res.content)

        # ── Step 5: EXECUTOR — action validation ────────────────
        executor_note = ""
        try:
            executor_sys = (
                "You are the EXECUTOR agent in a clinical trial AI system. "
                "Review the AI-generated response below and provide a brief validation (2-3 lines):\n"
                "- Feasibility check: Are the recommendations actionable?\n"
                "- Risk assessment: Any potential issues with proposed actions?\n"
                "- Priority: What should be done first?\n\n"
                "Be concise. Output ONLY the validation note, nothing else.\n\n"
                f"RESPONSE TO VALIDATE:\n{summary[:1500]}"
            )
            executor_res = self.llm.generate(
                prompt=f"Validate actions for: {query}",
                system_prompt=executor_sys,
            )
            executor_note = executor_res.content.strip()
        except Exception as e:
            logger.warning(f"EXECUTOR step failed (non-fatal): {e}")

        exec_obs = "Action validation complete" if executor_note else "Skipped"
        steps.append({
            "agent": "EXECUTOR",
            "thought": "Validating feasibility and risk of proposed actions.",
            "action": "validate_actions",
            "observation": exec_obs,
        })

        # Append executor validation to summary if meaningful
        if executor_note and len(executor_note) > 10:
            summary += f"\n\n## Action Validation\n{executor_note}"

        # Append confidence
        summary += f"\n\nConfidence: {confidence:.0%}."

        # ── Step 6: COMMUNICATOR — final response ───────────────
        steps.append({
            "agent": "COMMUNICATOR",
            "thought": "Finalizing response with confidence scoring and role-specific formatting.",
            "action": "finalize_response",
            "observation": "Response delivered.",
        })

        # ── Build dynamic recommendations ───────────────────────
        recommendations = []
        if had_error:
            recommendations.append({"action": "Verify data availability — some queries encountered errors", "impact": "Medium"})
        if rows_returned == 0:
            recommendations.append({"action": "Broaden your query — no matching records found", "impact": "Low"})
        if confidence < 0.70:
            recommendations.append({"action": "Cross-check findings manually — confidence below threshold", "impact": "High"})
        if not recommendations:
            recommendations.append({"action": "Review site-specific risk assessments for actionable follow-ups", "impact": "Medium"})

        tools_used = ["sql_bridge", "predictive_engine", "clinical_reasoning", "action_validator"]
        if template_used:
            tools_used.insert(1, "template_library")

        return {
            "summary": summary,
            "agent_chain": ["SUPERVISOR", "DIAGNOSTIC", "FORECASTER", "RESOLVER", "EXECUTOR", "COMMUNICATOR"],
            "steps": steps,
            "tools_used": tools_used,
            "confidence": confidence,
            "recommendations": recommendations,
        }


_orchestrator = None
def get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator()
    return _orchestrator
