# a6on-i — Pitch Deck
### Team PARZIVAL: Riyaz, Dhruv, Preetam | NOVA AI Hackathon 2026

---

## Slide 1 — The Problem

### Clinical Trials Are Drowning in Data

> The average Phase III trial generates **3.6 million data points** across hundreds of sites and thousands of patients. Data managers spend **60-70% of their time** on manual data review instead of analysis.

**What goes wrong today:**

- **Delayed database lock** — missed deadlines cost sponsors **$600K–$8M per day**
- **Manual risk detection** — CRAs manually review spreadsheets to spot issues
- **No predictive visibility** — teams react to problems instead of preventing them
- **Siloed information** — safety, quality, and operational data live in separate systems
- **No AI assistance** — analysts copy-paste data into ChatGPT with no system awareness

**Bottom line:** Billions of dollars are lost to preventable delays, and no existing tool gives clinical teams an intelligent, unified command center.

---

## Slide 2 — Our Solution: a6on-i

### An AI-Powered Clinical Trial Intelligence Platform

a6on-i is a **full-stack platform** that unifies clinical trial data into a single dashboard with an autonomous AI analyst that can query, simulate, and explain — just like having a senior data manager available 24/7.

**What makes it different:**

| Traditional Tools | a6on-i |
|---|---|
| Static dashboards, manual refresh | Real-time, role-aware views |
| Export to Excel → manual analysis | AI assistant answers in natural language |
| Spreadsheet risk flags | ML-powered risk classification (XGBoost + LightGBM) |
| No timeline prediction | 10,000-run Monte Carlo simulations |
| No root-cause analysis | Cascade dependency intelligence |
| No model governance | PSI/KS drift detection, 21 CFR Part 11 audit trail |

**Scale:** 57,974 patients · 2,216 sites · 23 studies · 47 PostgreSQL tables — all real, production-grade data.

---

## Slide 3 — The AI Assistant (Core Innovation)

### A "Claude Code for Clinical Trials"

Our AI assistant is not a chatbot — it's an **autonomous agent** powered by Gemini function calling that chains multiple tools to answer complex questions.

**How it works:**

```
User: "What's causing delays across the portfolio?"
        ↓
   SUPERVISOR (plans tool calls)
        ↓
   [Round 1] → get_portfolio_summary() + get_issue_summary() + get_cascade_analysis() + run_monte_carlo()
        ↓
   [Round 2] → run_sql_query("SELECT study_id, category, COUNT(*)...")
        ↓
   [Round 3] → get_dqi_breakdown(study_id="Study_21") + run_sql_query(SAE details)
        ↓
   SYNTHESIZER → Comprehensive markdown report with tables, root causes, and recommendations
```

**10 Tools Available to the Agent:**

| # | Tool | What It Does |
|---|---|---|
| 1 | `run_sql_query` | Dynamic SQL against 47-table PostgreSQL database |
| 2 | `get_portfolio_summary` | Portfolio-wide KPIs in one call |
| 3 | `get_site_details` | Deep-dive metrics for any site |
| 4 | `get_patient_details` | Patient-level risk and quality scores |
| 5 | `run_monte_carlo_simulation` | 10,000-iteration DB-lock timeline prediction |
| 6 | `get_cascade_analysis` | Issue dependency and root-cause analysis |
| 7 | `get_dqi_breakdown` | 8-component Data Quality Index drill-down |
| 8 | `get_risk_distribution` | Patient risk level distribution |
| 9 | `run_drift_check` | Live PSI/KS model drift detection |
| 10 | `get_issue_summary` | Open issues by category and priority |

**Conversation memory** — the agent remembers prior turns, so users can say "drill into that" or "yes, show me the breakdown" naturally.

---

## Slide 4 — The Intelligence Engine (Under the Hood)

### Six Modules Working Together

**1. Data Quality Index (DQI)**
- 8 weighted components: Safety (25%), Query (20%), Completeness (15%), Coding (12%), Lab (10%), SDV (8%), Signature (5%), EDRR (5%)
- Every patient scored 0–100 in real time

**2. Risk Classification**
- 14 rule-based issue detectors (SAE pending, overdue signatures, missing visits, etc.)
- ML ensemble (XGBoost + LightGBM) with SHAP explainability
- 5-tier classification: No Risk → Low → Medium → High → Critical

**3. Monte Carlo Simulator**
- 10,000 parallel simulations for DB-lock timeline prediction
- P10/P25/P50/P75/P90 percentile estimates
- Site closure impact analysis ("what if we close Site 468?")

**4. Cascade Intelligence**
- Maps which issue types block other issue types
- Identifies highest-impact root causes to fix first
- Powered by cascade_impact_score across all open issues

**5. ML Governance & Drift Detection**
- Population Stability Index (PSI) + Kolmogorov-Smirnov tests
- Live comparison: baseline window vs. current window
- 21 CFR Part 11 compliant audit trail

**6. Role-Aware Dashboard**
- 5 roles: Study Lead, Data Manager, CRA, Safety Officer, Executive
- Each sees the metrics and alerts most relevant to their job
- Built with React 18, TanStack Query 5, Tailwind CSS, Radix UI

---

## Slide 5 — Why a6on-i Wins

### Not Just a Dashboard — An Autonomous Analyst

| Capability | Medidata Rave | Veeva Vault | Oracle InForm | **a6on-i** |
|---|---|---|---|---|
| Real-time DQI scoring | ❌ | ❌ | ❌ | ✅ 8-component |
| AI assistant with tool calling | ❌ | ❌ | ❌ | ✅ 10 tools |
| Monte Carlo timeline simulation | ❌ | ❌ | ❌ | ✅ 10K runs |
| Live model drift detection | ❌ | ❌ | ❌ | ✅ PSI + KS |
| Cascade root-cause analysis | ❌ | ❌ | ❌ | ✅ auto-prioritize |
| Conversation memory | ❌ | ❌ | ❌ | ✅ multi-turn |
| One-command setup | ❌ | ❌ | ❌ | ✅ `python run.py` |

**Speed:** AI assistant answers complex multi-tool queries in 10–25 seconds.

**Scale:** Battle-tested on 57,974 real patient records, not toy data.

**Governance:** Every AI decision is traceable, auditable, 21 CFR Part 11 ready.

---

## Slide 6 — Tech Stack

```
┌─────────────────────────────────────────────────────────┐
│  FRONTEND        React 18 · TypeScript · Tailwind CSS   │
│                  TanStack Query 5 · Zustand · Radix UI  │
├─────────────────────────────────────────────────────────┤
│  AI AGENT        Gemini 3 Flash · Function Calling      │
│                  10 Tools · ReAct Loop · Conv. Memory    │
├─────────────────────────────────────────────────────────┤
│  BACKEND         FastAPI · SQLAlchemy 2.0 · Pydantic 2  │
│                  JWT Auth · Role-Based Access            │
├─────────────────────────────────────────────────────────┤
│  ML ENGINE       XGBoost · LightGBM · Isolation Forest  │
│                  SHAP · Monte Carlo · Drift Detector     │
├─────────────────────────────────────────────────────────┤
│  DATABASE        PostgreSQL 16 (47 tables, 57K records) │
│                  Neo4j 5 (cascade graph) · ChromaDB      │
└─────────────────────────────────────────────────────────┘
```

---

## Slide 7 — Live Demo Results

### Real Queries → Real Answers

| Query | Tools Chained | Time | What Happened |
|---|---|---|---|
| "Give me a portfolio summary" | 4 tools | 9.5s | KPIs + risk dist + issues + drift status |
| "How is Site 468 performing?" | 5 tools, 3 rounds | 25s | Site metrics + DQI + issues + risk + SQL drill-down |
| "When will Study_1 achieve DB lock?" | Monte Carlo + 4 tools | 15s | P50: 50 days, P90: 73 days + bottleneck sites |
| "Run drift check on DQI predictor" | Drift detector + SQL | 12s | PSI scores for 7 features, severity per feature |
| "Yes, show me the breakdown" | DQI × 4 studies + SQL | 8s | **Remembered context** — drilled into Study_15/19 |
| "What's causing delays?" | 6 tools, 13 calls | 25s | Root-caused to Safety issues in Study_21 |

**Every answer includes:** markdown tables, actionable recommendations, confidence scores, and the full agent trace (which tools were called and why).

---

## Slide 8 — Thank You

### a6on-i — Intelligence That Acts

> *"Don't just show me the data. Tell me what's wrong, why it happened, and what to do about it."*

**Team PARZIVAL**
- Riyaz · Dhruv · Preetam

**Try it yourself:**
```bash
git clone https://github.com/PARZIVALPRIME/NOVA-AI-Sanchalak-AI.git
python run.py
```

📄 Full documentation: `Documentation.pdf`
🧠 Model weights: [github.com/PARZIVALPRIME/a6on-i_ai](https://github.com/PARZIVALPRIME/a6on-i_ai)
