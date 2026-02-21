
import logging
import json
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
import os
from src.knowledge.rag_knowledge_base import RAGKnowledgeBase, RetrievalChain
from src.agents.llm_wrapper import get_llm

logger = logging.getLogger(__name__)


def _build_study_table_text(study_breakdown: list) -> str:
    """Build a compact text table of study-level data for the AI prompt context."""
    if not study_breakdown:
        return "No study-level breakdown available."
    
    lines = ["Study | Patients | Sites | DQI% | OpenQ | MissingPg | SAEs | Deviations | Clean%"]
    lines.append("-" * 85)
    for s in study_breakdown[:30]:  # Cap at 30 studies to save tokens
        lines.append(
            f"{s.get('study_id', '?'):>6} | {s.get('patients', 0):>8,} | {s.get('sites', 0):>5} | "
            f"{s.get('dqi', 0):>5.1f} | {s.get('open_queries', 0):>5,} | {s.get('missing_pages', 0):>9,} | "
            f"{s.get('sae_count', 0):>4,} | {s.get('protocol_deviations', 0):>10,} | {s.get('clean_rate', 0):>5.1f}"
        )
    return "\n".join(lines)


def _build_portfolio_summary_text(metrics: dict, study_breakdown: list) -> str:
    """Build a compact portfolio-level summary for the AI prompt."""
    tp = metrics.get('total_patients', 0)
    ts = metrics.get('total_sites', 0)
    tst = metrics.get('total_studies', len(study_breakdown))
    dqi = metrics.get('mean_dqi', 0)
    clean = metrics.get('clean_rate', 0)
    toq = metrics.get('total_open_queries', sum(s.get('open_queries', 0) for s in study_breakdown))
    tmp = metrics.get('total_missing_pages', sum(s.get('missing_pages', 0) for s in study_breakdown))
    tsa = metrics.get('total_saes', sum(s.get('sae_count', 0) for s in study_breakdown))
    tpd = metrics.get('total_protocol_deviations', sum(s.get('protocol_deviations', 0) for s in study_breakdown))
    dr = metrics.get('dblock_ready', 0)
    
    qpp = f"{toq/tp:.2f}" if tp > 0 else "N/A"
    return (
        f"Portfolio: {tst} studies, {tp:,} patients, {ts:,} sites\n"
        f"Avg DQI: {dqi}% | Clean Rate: {clean:.1%} | DB Lock Ready: {dr:.1%}\n"
        f"Open Queries: {toq:,} ({qpp} per patient)" + (f" | Missing Pages: {tmp:,}" if tmp else "") + "\n"
        f"SAEs: {tsa:,} | Protocol Deviations: {tpd:,}"
    )


class GenerativeDocumentEngine:
    """
    Advanced RAG-powered engine for generating professional, context-aware clinical trial reports.
    Produces structured, per-section narratives using ICH-GCP guidelines and SOPs.
    """
    
    def __init__(self):
        self.llm = get_llm()
        self.kb = RAGKnowledgeBase()
        self.retriever = None
        
        # Initialize RAG if index exists
        if self.kb.load_index():
            self.retriever = RetrievalChain(self.kb)
            logger.info("RAG Knowledge Base linked to Document Engine")
        else:
            logger.warning("RAG Index not found. Using zero-shot generation.")

    def _robust_json_parse(self, json_str: str) -> dict:
        """Multi-pass JSON parser to handle common LLM output quirks."""
        # Pass 1: Basic cleanup
        cleaned = re.sub(r',\s*([}\]])', r'\1', json_str)   # trailing commas
        cleaned = re.sub(r'//.*?$', '', cleaned, flags=re.MULTILINE)  # comments
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', ' ', cleaned)  # control chars
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        
        # Pass 2: Fix unescaped newlines/tabs inside JSON string values
        cleaned = re.sub(r'(?<=": ")(.*?)(?=")', lambda m: m.group(0).replace('\n', '\\n').replace('\t', '\\t'), cleaned, flags=re.DOTALL)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Pass 3: Single-quoted keys/values
        cleaned = re.sub(r"'(\w+)':", r'"\1":', cleaned)
        cleaned = re.sub(r":\s*'([^']*?)'", r': "\1"', cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Pass 4: Fix unescaped double-quotes inside string values
        # Pattern: find string values and escape inner quotes
        def _escape_inner_quotes(s):
            result = []
            in_string = False
            escape_next = False
            i = 0
            while i < len(s):
                c = s[i]
                if escape_next:
                    result.append(c)
                    escape_next = False
                elif c == '\\':
                    result.append(c)
                    escape_next = True
                elif c == '"':
                    if not in_string:
                        in_string = True
                        result.append(c)
                    else:
                        # Look ahead: is this the end of the string?
                        rest = s[i+1:].lstrip()
                        if rest and rest[0] in (',', '}', ']', ':'):
                            in_string = False
                            result.append(c)
                        elif rest and rest[0] == '"':
                            # Next token starts a new string — close current
                            in_string = False
                            result.append(c)
                        else:
                            # Unescaped quote inside string — escape it
                            result.append('\\"')
                            i += 1
                            continue
                else:
                    result.append(c)
                i += 1
            return ''.join(result)

        try:
            fixed = _escape_inner_quotes(cleaned)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

        # Pass 5: Fix missing commas between elements (e.g., "}\n{" or "]\n[" or '"\n"')
        fixed = re.sub(r'"\s*\n\s*"', '",\n"', cleaned)         # "value"\n"key" -> "value","key"
        fixed = re.sub(r'}\s*\n\s*{', '},\n{', fixed)           # }\n{ -> },\n{
        fixed = re.sub(r'"\s*\n\s*{', '",\n{', fixed)           # "\n{ -> ",\n{
        fixed = re.sub(r'}\s*\n\s*"', '},\n"', fixed)           # }\n" -> },\n"
        fixed = re.sub(r'"\s*\n\s*\[', '",\n[', fixed)          # "\n[ -> ",\n[
        fixed = re.sub(r'\]\s*\n\s*"', '],\n"', fixed)          # ]\n" -> ],\n"
        fixed = re.sub(r',\s*([}\]])', r'\1', fixed)            # re-clean trailing commas
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

        # Pass 6: Truncated JSON — auto-close brackets
        truncated = fixed.rstrip()
        open_braces = truncated.count('{') - truncated.count('}')
        open_brackets = truncated.count('[') - truncated.count(']')
        # Strip trailing partial string/value
        truncated = re.sub(r'[^}\]"0-9a-zA-Z]\s*$', '', truncated)
        if not truncated.endswith(('}', ']', '"')):
            truncated += '"'
        truncated += ']' * max(0, open_brackets) + '}' * max(0, open_braces)
        try:
            return json.loads(truncated)
        except json.JSONDecodeError:
            pass

        # Pass 7: Last resort — extract sections via regex and build dict
        logger.warning("All JSON parse attempts failed, extracting sections via regex")
        result = {}
        
        # Extract executive_summary
        es_match = re.search(r'"executive_summary"\s*:\s*"((?:[^"\\]|\\.)*)(?:")', json_str, re.DOTALL)
        if es_match:
            result['executive_summary'] = es_match.group(1).replace('\\n', '\n').replace('\\"', '"')
        
        # Extract key_findings as list of strings
        kf_match = re.search(r'"key_findings"\s*:\s*\[(.*?)\]', json_str, re.DOTALL)
        if kf_match:
            findings = re.findall(r'"((?:[^"\\]|\\.)+)"', kf_match.group(1))
            result['key_findings'] = [f.replace('\\n', '\n').replace('\\"', '"') for f in findings]
        
        # Extract conclusion
        cn_match = re.search(r'"conclusion"\s*:\s*"((?:[^"\\]|\\.)*)(?:")', json_str, re.DOTALL)
        if cn_match:
            result['conclusion'] = cn_match.group(1).replace('\\n', '\n').replace('\\"', '"')

        if result:
            return result
        
        raise json.JSONDecodeError("All JSON parse attempts failed", json_str, 0)

    def _normalize_sections(self, report_data: dict, metrics: dict) -> dict:
        """Guarantee all required sections exist with minimum viable content."""
        dqi = float(metrics.get('mean_dqi', metrics.get('dqi', 85)) or 85)
        clean = float(metrics.get('clean_rate', 0) or 0)
        tp = int(metrics.get('total_patients', 0) or 0)
        toq = int(metrics.get('total_open_queries', 0) or 0)
        tsa = int(metrics.get('total_saes', 0) or 0)
        tpd = int(metrics.get('total_protocol_deviations', 0) or 0)
        tmp = int(metrics.get('total_missing_pages', 0) or 0)
        qpp = toq / tp if tp > 0 else 0
        dev_pct = tpd / tp * 100 if tp > 0 else 0

        # Ensure executive_summary
        if not report_data.get('executive_summary'):
            report_data['executive_summary'] = 'Analysis pending — no executive summary generated.'

        # Ensure key_findings (minimum 3)
        findings = report_data.get('key_findings', [])
        if len(findings) < 3:
            defaults = [
                f"Average DQI: {dqi:.1f}% ({'above' if dqi >= 90 else 'below'} the 90% target)",
                f"Clean-patient rate: {clean:.0%} ({'meeting' if clean >= 0.8 else 'below'} the 80% threshold)",
                f"Open queries: {toq:,} ({qpp:.2f} per patient)" if tp > 0 else f"Open queries: {toq:,}",
                f"SAEs reported: {tsa:,}" if tsa > 0 else "No significant SAE exposure detected",
                f"Protocol deviations: {tpd:,} ({dev_pct:.1f}% of patients)" if tp > 0 else f"Protocol deviations: {tpd:,}",
            ]
            for d in defaults:
                if len(findings) >= 5:
                    break
                if not any(d[:20] in str(f) for f in findings):
                    findings.append(d)
            report_data['key_findings'] = findings

        # Ensure risk_table (minimum 3 rows)
        risk_table = report_data.get('risk_table', report_data.get('risk_analysis', []))
        if not isinstance(risk_table, list) or len(risk_table) < 3:
            risk_table = [
                {"category": "Data Quality Index", "current_metric": f"Avg = {dqi:.0f}%", "threshold": ">= 90%", "gap": f"{dqi - 90:+.0f} pts", "impact": "Moderate" if dqi >= 80 else "High"},
                {"category": "Open Query Rate", "current_metric": f"{qpp:.2f} q/patient" if tp > 0 else "N/A", "threshold": "<= 0.05 q/patient", "gap": f"+{qpp - 0.05:.2f}" if tp > 0 else "N/A", "impact": "High" if qpp > 0.05 else "Low"},
                {"category": "Clean-Patient Rate", "current_metric": f"{clean:.0%}", "threshold": ">= 80%", "gap": f"{clean - 0.8:+.0%}", "impact": "Critical" if clean < 0.5 else "High" if clean < 0.8 else "Low"},
                {"category": "Missing Source Pages", "current_metric": f"{tmp:,} pages", "threshold": "< 500", "gap": f"+{tmp - 500:,}" if tmp > 500 else "On target", "impact": "Moderate"},
                {"category": "Protocol Deviations", "current_metric": f"{dev_pct:.0f}% of patients" if tp > 0 else f"{tpd:,}", "threshold": "<= 5%", "gap": f"+{dev_pct - 5:.0f}%" if dev_pct > 5 else "On target", "impact": "High" if dev_pct > 5 else "Low"},
            ]
            report_data['risk_table'] = risk_table
            report_data['risk_analysis'] = risk_table

        # Ensure recommendations (minimum 6)
        recs = report_data.get('recommendations', [])
        if not isinstance(recs, list) or len(recs) < 3:
            recs = [
                {"area": "Query Management", "action": f"Triage and resolve {toq:,} open queries — prioritize aging > 14 days", "owner": "Data Manager", "priority": "High", "target": "Reduce open queries by 30% in 60 days"},
                {"area": "Data Completeness", "action": f"Chase {tmp:,} missing CRF pages with sites", "owner": "CRA", "priority": "High", "target": "All missing pages uploaded within 14 days"},
                {"area": "Protocol Compliance", "action": "Root-cause analysis for top deviation categories", "owner": "Medical Monitor", "priority": "Medium", "target": "CAPA plans within 30 days"},
                {"area": "Safety Oversight", "action": "Ensure all SAE narratives are current and reconciled", "owner": "Safety Officer", "priority": "High", "target": "SAE queries closed within 48 hours"},
                {"area": "DB Lock Readiness", "action": "Define clean-patient criteria and track lock-readiness per study", "owner": "Data Manager", "priority": "Critical", "target": f"Achieve >= 80% clean rate (current: {clean:.0%})"},
                {"area": "Governance", "action": "Activate KRI monitoring and schedule quarterly portfolio review", "owner": "Regulatory", "priority": "Medium", "target": "KRI dashboards live within 30 days"},
            ]
            report_data['recommendations'] = recs

        # Ensure conclusion
        if not report_data.get('conclusion'):
            report_data['conclusion'] = (
                f"Immediate priorities: (1) Reduce open queries by >= 30% within 60 days, "
                f"(2) Achieve >= 80% clean-patient rate for lock-ready studies, "
                f"(3) Close all SAE-related queries within regulatory timelines."
            )

        # Ensure interpretations
        if not report_data.get('interpretations'):
            report_data['interpretations'] = {
                "dqi": "Pristine" if dqi >= 95 else "On target" if dqi >= 90 else "Below target" if dqi >= 80 else "Critical",
                "clean_rate": "Lock-ready" if clean >= 0.8 else "Needs improvement"
            }

        return report_data

    def generate_structured_report(self, report_type: str, data_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive, structured AI analysis for a report.
        Returns a dictionary containing narratives for specific report sections.
        """
        logger.info(f"Generating structured AI insights for {report_type}")
        metrics = data_context.get('metrics', {})
        
        # 1. Get RAG Context
        query = f"Clinical Data Management {report_type.replace('_', ' ')} analysis for {data_context.get('study_id', 'portfolio')}"
        rag_context = ""
        if self.retriever:
            rag_context = self.kb.get_relevant_context(query, max_tokens=800)
        
        # 2. Extract structured data for prompt
        raw_data = data_context.get('raw_data', {})
        study_breakdown = raw_data.get('study_breakdown', data_context.get('study_breakdown', []))
        
        # Build compact text representations instead of dumping raw JSON
        portfolio_text = _build_portfolio_summary_text(metrics, study_breakdown)
        study_table_text = _build_study_table_text(study_breakdown)
        
        # 3. Build professional DeepAnalyze prompt
        system_prompt = f"""You are the Sanchalak AI Generative Document Engine — an expert Clinical Data Manager, Medical Monitor, and Regulatory Affairs specialist.

Your task: produce a publication-quality Clinical Data Management report following the DeepAnalyze Protocol.

DeepAnalyze Protocol for Reports:
1. Direct Assessment — Lead with the bottom-line portfolio health status.
2. Key Findings — Cite specific study numbers, exact metrics, and quantified thresholds.
3. Risk Analysis — Compare against ICH-GCP benchmarks and industry best practices.
4. Actionable Recommendations — Each must be specific, measurable, assigned, and time-bound.

REFERENCE GUIDELINES:
{rag_context}

STRICT OUTPUT RULES:
- Write in professional, clinical, objective language. NO marketing jargon, NO vague statements.
- EVERY finding must cite specific study numbers and exact values from the data provided.
- Compare metrics against these industry thresholds: DQI >= 90%, clean rate >= 80%, query rate <= 0.05/patient, deviations <= 5%, SDV completion >= 100%.
- Identify root causes and risk cascades (e.g., low clean rate → delays DB lock → regulatory risk).
- DO NOT include <think>, <Analyze>, or reasoning tags.
- Output ONLY a valid JSON object. No markdown fences, no text before/after.
- Use double quotes for all JSON keys and string values."""

        prompt = f"""Analyze this clinical trial portfolio data for a {report_type.replace('_', ' ')} report using the DeepAnalyze Protocol.

PORTFOLIO DATA:
{portfolio_text}

STUDY-LEVEL BREAKDOWN:
{study_table_text}

Return a JSON object with exactly these keys:

1. "executive_summary": 2-3 paragraphs following DeepAnalyze structure:
   Para 1 (Direct Assessment): Start with portfolio scope (X studies, Y patients, Z sites across N countries). State overall DQI, clean rate, query burden, and SAE volume. Give a one-sentence bottom-line verdict (e.g., "The portfolio is NOT lock-ready" or "Data quality meets threshold but query backlog poses timeline risk").
   Para 2 (Analysis): Identify the top 2-3 risk areas with specific study numbers and their impact. Explain risk cascades — how one metric failure leads to downstream consequences for regulatory timelines.
   Para 3 (Outlook): Summarize the path forward — what must change for DB lock readiness, regulatory submission, or patient safety compliance.

2. "key_findings": List of 6-8 STRING observations. Each must follow this pattern: "[Metric]: [Specific value] — [Clinical significance]". Examples:
   - "Clean-Patient Rate: 0% across all 18 studies — no study meets the 80% threshold required for DB lock, creating regulatory timeline risk"
   - "Query Backlog: Study STUDY-021 accounts for 28% of all open queries (2,092 of 7,354) — requires immediate query management intervention"
   - "Safety Exposure: 847 SAEs reported across portfolio — ongoing pharmacovigilance review and SUSAR assessment required"
   Each finding MUST be a plain string, not a dict. Focus on: DQI outliers, clean rate gaps, query-heavy studies, safety signals, protocol deviations, enrollment milestones.

3. "risk_table": List of 6 risk objects, each with: "category", "current_metric", "threshold", "gap", "impact" (Critical/High/Moderate/Low), "highest_risk_studies" (list of study IDs).
   Required categories: Query Backlog, Data Completeness, Protocol Compliance, Patient Safety, Clean-Patient Readiness, KRI Coverage.
   Rate impact based on: Critical = blocks regulatory submission, High = delays timeline > 30 days, Moderate = requires monitoring, Low = acceptable.

4. "recommendations": List of 6-8 objects. Each with: "area", "action" (specific, measurable), "owner" (CRA/Data Manager/Medical Monitor/Safety Officer/Regulatory/Sponsor), "priority" (Critical/High/Medium), "target" (quantified target with specific timeline).
   Recommendations must be SMART (Specific, Measurable, Assigned, Realistic, Time-bound).
   Example: {{"area": "Query Management", "action": "Implement daily query resolution targets for Study STUDY-021, prioritizing critical safety queries", "owner": "Data Manager", "priority": "Critical", "target": "Reduce open queries by 40% within 45 days"}}

5. "conclusion": Start with 1-sentence verdict. Then provide exactly 3 quantified action targets as bullet points, e.g.:
   "• Target 1: Reduce open query rate to <= 0.05 queries/patient within 60 days
   • Target 2: Achieve >= 80% clean-patient rate for top 5 enrollment studies within 90 days  
   • Target 3: Complete 100% SDV for all SAE patients within 30 days"

6. "interpretations": Dictionary mapping metric names to 1-sentence clinical interpretations grounded in the data.

Return ONLY the JSON object."""

        # 4. Call LLM and Parse
        try:
            llm_res = self.llm.generate(prompt=prompt, system_prompt=system_prompt)
            raw = llm_res.content or ""
            
            # Strip <think> blocks from DeepSeek-R1 based models
            raw = re.sub(r'<think>[\s\S]*?</think>', '', raw).strip()
            raw = re.sub(r'<Analyze>[\s\S]*?</Analyze>', '', raw).strip()
            
            # Strip markdown code fences (```json ... ```)
            fence_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', raw)
            if fence_match:
                raw = fence_match.group(1).strip()
            
            # Extract JSON object
            json_match = re.search(r'\{[\s\S]*\}', raw)
            if json_match:
                json_str = json_match.group()
                report_data = self._robust_json_parse(json_str)
            else:
                logger.warning("LLM did not return JSON. Extracting insights from plain text.")
                report_data = {
                    "executive_summary": raw[:2000],
                    "key_findings": [s.strip() for s in raw.split('\n') if s.strip() and len(s.strip()) > 20][:5],
                    "risk_table": [],
                    "recommendations": [],
                    "conclusion": "",
                    "interpretations": {}
                }
            
            # Ensure backward compatibility: map risk_table to risk_analysis if needed
            if 'risk_table' in report_data and 'risk_analysis' not in report_data:
                report_data['risk_analysis'] = report_data['risk_table']
            
            # NORMALIZE: guarantee all sections have minimum viable content
            report_data = self._normalize_sections(report_data, metrics)
            
            return {
                "report_id": f"AI-REP-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "sections": report_data,
                "metadata": {
                    "rag_active": self.retriever is not None,
                    "model": llm_res.model,
                    "generated_at": datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Structured AI generation failed: {e}")
            return self._generate_fallback_narrative(report_type, data_context)

    def _generate_fallback_narrative(self, report_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive rule-based fallback for when AI fails or is unavailable."""
        metrics = context.get('metrics', context)
        raw_data = context.get('raw_data', {})
        study_breakdown = raw_data.get('study_breakdown', context.get('study_breakdown', []))
        
        dqi = float(metrics.get('mean_dqi', metrics.get('dqi', metrics.get('avg_dqi', 85.0))) or 85.0)
        clean = float(metrics.get('clean_rate', metrics.get('tier1_clean_rate', 0.0)) or 0.0)
        ready = float(metrics.get('db_lock_ready', metrics.get('ready_rate', 0.0)) or 0.0)
        tp = int(metrics.get('total_patients', 0) or 0)
        ts = int(metrics.get('total_sites', 0) or 0)
        toq = int(metrics.get('total_open_queries', 0) or 0)
        tmp = int(metrics.get('total_missing_pages', 0) or 0)
        tsa = int(metrics.get('total_saes', 0) or 0)
        tpd = int(metrics.get('total_protocol_deviations', 0) or 0)
        num_studies = int(metrics.get('total_studies', len(study_breakdown)) or 0)
        
        dqi_interp = "Pristine data quality" if dqi >= 95 else "Data quality on target" if dqi >= 90 else "Data quality below target" if dqi >= 80 else "Critical data quality risk"
        clean_interp = "Exceptional cleaning progress" if clean >= 0.9 else "Cleaning on track" if clean >= 0.7 else "Cleaning lag — no study lock-ready"
        
        # Build comprehensive summary
        qpp_str = f"{toq/tp:.2f}" if tp > 0 else "N/A"
        summary = (
            f"Portfolio: {num_studies} studies ({tp:,} patients, {ts:,} sites). Reporting Date: {datetime.now().strftime('%d %b %Y')}. "
            f"Overall data quality is {'acceptable' if dqi >= 80 else 'below acceptable thresholds'} (average DQI = {dqi:.0f}/100) "
            f"but clean-patient rate is {clean:.0%}, {'meeting' if clean >= 0.8 else 'falling short of'} the 80% lock-readiness threshold. "
            f"Open query burden: {toq:,} active queries ({qpp_str} queries/patient). "
        )
        if tsa > 0:
            summary += f"Safety exposure: {tsa:,} SAEs reported, demanding continued vigilance. "
        if tpd > 0:
            summary += f"Protocol deviation load: {tpd:,} deviations across the portfolio."
        
        # Build risk table
        qpp_val = toq / tp if tp > 0 else 0
        dev_pct = tpd / tp * 100 if tp > 0 else 0
        risk_table = [
            {"category": "Data Quality Index", "current_metric": f"Avg = {dqi:.0f}%", "threshold": ">= 90%", "gap": f"{dqi - 90:+.0f} pts", "impact": "Moderate" if dqi >= 80 else "High"},
            {"category": "Open Query Rate", "current_metric": f"{toq:,} open / {tp:,} pts = {qpp_val:.2f} q/patient" if tp > 0 else "N/A", "threshold": "<= 0.05 q/patient", "gap": f"+{qpp_val - 0.05:.2f} q/patient" if tp > 0 else "N/A", "impact": "High"},
            {"category": "Clean-Patient Rate", "current_metric": f"{clean:.0%}", "threshold": ">= 80%", "gap": f"{clean - 0.8:+.0%}", "impact": "Critical" if clean < 0.5 else "High"},
            {"category": "Missing Source Pages", "current_metric": f"{tmp:,} pages", "threshold": "< 500", "gap": f"+{tmp - 500:,}" if tmp > 500 else "On target", "impact": "Moderate"},
            {"category": "Protocol Deviations", "current_metric": f"{tpd:,} ({dev_pct:.0f}% of pts)" if tp > 0 else f"{tpd:,}", "threshold": "<= 5%", "gap": f"+{dev_pct - 5:.0f}%" if tp > 0 else "N/A", "impact": "High"},
        ]
        
        return {
            "report_id": f"RULE-REP-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "sections": {
                "executive_summary": summary,
                "key_findings": [
                    f"Average DQI: {dqi:.1f}% ({dqi_interp})",
                    f"Clean-patient rate: {clean:.0%} — {'lock-ready' if clean >= 0.8 else 'not lock-ready'}",
                    f"Open queries: {toq:,} ({toq / tp:.2f} per patient)" if tp > 0 else f"Open queries: {toq:,}",
                    f"SAEs reported: {tsa:,}" if tsa > 0 else "No SAE data available",
                    f"Protocol deviations: {tpd:,}" if tpd > 0 else "No deviation data available"
                ],
                "risk_table": risk_table,
                "risk_analysis": risk_table,
                "recommendations": [
                    {"area": "Query Management", "action": f"Resolve top-{min(5, num_studies)} query-heavy studies", "owner": "Data Manager", "priority": "High", "target": "Reduce open queries by 30% in 60 days"},
                    {"area": "Data Completeness", "action": f"Reconcile {tmp:,} missing CRF pages", "owner": "CRA", "priority": "High", "target": "Upload within 7 days"},
                    {"area": "Protocol Compliance", "action": "Root-cause analysis for top deviation studies", "owner": "Medical Monitor", "priority": "Medium", "target": "CAPA within 30 days"},
                    {"area": "Safety Oversight", "action": "Quarterly safety-monitoring committee review", "owner": "Safety Officer", "priority": "High", "target": "All SAE queries closed within 48h"},
                    {"area": "DB Lock Readiness", "action": "Define clean-patient criteria and lock-readiness dashboard", "owner": "Data Manager", "priority": "Critical", "target": "Clean rate >= 80% for lock-ready studies"},
                    {"area": "Governance", "action": "Activate KRIs and quarterly portfolio review", "owner": "Regulatory", "priority": "Medium", "target": "At least 2 KRIs active"}
                ],
                "conclusion": (
                    f"The portfolio demonstrates solid enrollment yet data-quality maturity is lagging. "
                    f"Immediate actions required: (1) Reduce open queries by >= 30% within 60 days, "
                    f"(2) Achieve >= 80% clean-patient rate for at least three studies, "
                    f"(3) Bring protocol deviation rates below 10% through targeted CAPA."
                ),
                "interpretations": {"dqi": dqi_interp, "clean_rate": clean_interp}
            },
            "metadata": {"fallback": True, "model": "RuleEngine-2.0"}
        }

    def generate_report(self, report_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Legacy wrapper for compatibility."""
        res = self.generate_structured_report(report_type, context)
        return {
            "content": res["sections"]["executive_summary"],
            "report_id": res["report_id"],
            "type": report_type,
            "generated_at": res["metadata"].get("generated_at", datetime.utcnow().isoformat()),
            "metadata": res["metadata"]
        }

    def generate_safety_narrative(self, patient_data: Dict[str, Any], sae_details: Dict[str, Any]) -> str:
        """Specific helper for safety narratives."""
        res = self.generate_structured_report("safety_narrative", {**patient_data, **sae_details})
        return res["sections"]["executive_summary"]
