# Product Requirement Document (PRD) - NOVA-AI Sanchalak

## 1. Product Overview
**NOVA-AI Sanchalak** is an advanced AI-driven Clinical Trial Management and Orchestration platform. It leverages a sophisticated multi-agent architecture to automate complex workflows, detect anomalies in clinical data, and predict trial outcomes with high precision. The "Sanchalak" (Sanskrit for "Conductor" or "Controller") launcher provides a unified, cross-platform interface for managing the entire ecosystem, including database orchestration and service lifecycle.

## 2. Problem Statement
Clinical trials suffer from data Silos, manual oversight bottlenecks, and delayed risk identification. Coordinating between sites, patients, and regulatory requirements involves immense manual labor and high error rates.

## 3. Core Features

### 3.1 Sanchalak Launcher (Unified Command Center)
- **Multi-Mode Operation**: Supports both local PostgreSQL and Docker-based containerized database environments.
- **Auto-Detect Technology**: Automatically identifies system capabilities (Node.js, Python, Docker) and configures the environment accordingly.
- **Service Orchestration**: Manages the synchronized startup of the FastAPI backend and Vite frontend.
- **Data Integrity**: Includes automated database restoration and sanitization tools for consistent environment setup.

### 3.2 Multi-Agent Orchestration System
The platform utilizes a collaborative agent framework located in `src/agents`:
- **Supervisor Agent**: Overlooks the entire agent network, delegating tasks and resolving conflicts.
- **Diagnostic Agent**: Performs deep analysis of system logs and clinical data to identify root causes of issues.
- **Forecaster Agent**: Predicts trial timelines, enrollment trends, and potential site bottlenecks.
- **Executor Agent**: Handles the execution of recurring tasks and automated responses.
- **Resolver Agent**: Proposes solutions for detected anomalies based on historical "Resolution Genomes".
- **Communicator Agent**: Manages stakeholder notifications and data dissemination.

### 3.3 Advanced ML Engines
Integrated ML modules located in `src/ml` provide deep intelligence:
- **Anomaly detector**: Real-time identification of outliers in clinical submissions.
- **Issue detector**: Categorizes and prioritizes project-level risks.
- **Resolution Genome**: A specialized knowledge base for learning from past trial resolutions to improve future accuracy.
- **Site Risk Ranker**: Evaluates clinical sites based on DQI (Data Quality Index) and historical performance.

### 3.4 Governance and Compliance
- **Data Privacy**: Built-in modules for handling sensitive patient information.
- **Audit Trails**: Extensive tracing of agent decisions and ML model inferences.
- **Compliance Guardrails**: Ensuring all actions align with clinical trial regulations.

## 4. Technical Architecture

### 4.1 Technology Stack
- **Frontend**: Vite, React, Vanilla CSS (Premium Design Focus).
- **Backend**: Python (FastAPI), Uvicorn.
- **Database**: PostgreSQL (Managed locally or via Docker).
- **AI/ML**: PyTorch/TensorFlow (for ML engines), LLM wrappers for agent logic.

### 4.2 Module Structure
- `backend/`: API services and business logic.
- `frontend/`: Interactive dashboards and management UI.
- `src/`: Core logic repository.
    - `agents/`: Agent logic and collaboration protocols.
    - `ml/`: Predictive models and data science modules.
    - `knowledge/`: Vector stores and RAG-based documentation systems.

## 5. User Personas
1. **Clinical Trial Manager**: Uses the Forecaster and Supervisor to oversee multi-site trial progress.
2. **Site Operations Lead**: Monitors Site Risk and DQI scores via the dashboard.
3. **Data Scientist**: Refines ML models and monitors Anomaly Detector performance.
4. **IT Administrator**: Uses Sanchalak launcher for deployment and infrastructure management.

## 6. Future Roadmap
- **Real-time Streaming**: Enhanced site telemetry via the `streaming` module.
- **Advanced Fine-tuning**: Direct UI-based model retraining for trial-specific needs.
- **Extended Collaboration**: Multi-user agent workspaces for global teams.
