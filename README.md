# TrialPulse Nexus

<img width="1683" height="800" alt="executive1" src="https://github.com/user-attachments/assets/c6e61270-bfe7-489a-8bb5-e4ce99137b12" />

This is submission by Team PARZIVAL (Riyaz, Dhruv, Preetam) for the NOVA AI Hackathon 2026.

---

## 📋 Prerequisites
- **Python 3.10+**
- **Node.js 18+**
- **PostgreSQL 14+** OR **Docker Desktop**
- **Neo4j** (Optional)
- **Ollama** (Optional)

## 🚀 Quick Start

### Option 1: One-Command Launch (Recommended)
```bash
python run.py
```
This auto-detects your environment and starts everything.

### Option 2: Choose Your Database Mode
```bash
python run.py --docker        # Use Docker for PostgreSQL (no local install needed)
python run.py --local         # Use local PostgreSQL installation
python run.py --frontend-only # Just run the UI (for preview)
python run.py --skip-db       # Skip database setup (use existing data)
```

> [!TIP]
> **No PostgreSQL installed?** Use `--docker` - it will automatically start PostgreSQL in a container.

> [!WARNING]
> **First-time installation may take 10-15 minutes** depending on your network speed and storage. The full dependency installation is **~10GB** (Python ML libraries + Node modules). Ensure you have sufficient disk space and a stable internet connection.

## ⚙️ Setup

### 1. Environment Configuration
```bash
cp .env.example .env
```
Edit `.env` and set:
- `DB_PASSWORD` - Your PostgreSQL password
- `GROQ_API_KEY` - For cloud AI features (get one at [console.groq.com](https://console.groq.com))

### 2. Download Database (Required)
Download the database dump and place it in the `database/` folder:
```
database/reproduction_dump.sql
```
📥 **[Download Database Dump](https://drive.google.com/file/d/1rurYOMNUhG6ncy-e7EI7jDXl6NU9y_K2/view?usp=sharing)** (~212 MB)

## 📊 Model Weights & Research

All ML model weights (including fine-tuned models), research notebooks, and result plots are available at:

🔗 **[trialpulse_nexus Repository](https://github.com/PARZIVALPRIME/trialpulse_nexus)**

## 📚 Documentation

Full technical documentation is included in this repository:

📄 **[Documentation.pdf](./Documentation.pdf)** - Complete 6-module technical documentation

## 📦 What `run.py` Does

1. ✅ Checks Python, Node.js, and database availability
2. ✅ Creates Python virtual environment (if needed)
3. ✅ Installs all dependencies (pip + npm)
4. ✅ Sets up the database with 57,974 patient records
5. ✅ Starts backend API on `http://127.0.0.1:8000`
6. ✅ Starts frontend on `http://localhost:5173`
7. ✅ Handles graceful shutdown with Ctrl+C

---

## 🔧 Manual Setup (Alternative)

If you prefer manual setup over the launcher:

### 1. Database Setup
```bash
python setup_repro_db.py
```

### 2. Backend
```bash
cd backend
python -m venv venv
# Windows: .\venv\Scripts\activate
# macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

### 3. Frontend
```bash
cd frontend
npm install
npm run dev
```

---

## 🧪 Optional Features

### Neo4j Knowledge Graph
```bash
python scripts/seed_neo4j_cascade.py
python scripts/seed_graph.py
```

### Local AI Model (Ollama)
```bash
ollama create trialpulse-nexus -f Modelfile
```

---
**TrialPulse Nexus**

