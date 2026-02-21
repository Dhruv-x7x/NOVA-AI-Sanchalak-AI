# Sanchalak AI

<img width="1683" height="800" alt="executive1" src="https://github.com/user-attachments/assets/c6e61270-bfe7-489a-8bb5-e4ce99137b12" />

This is submission by Team PARZIVAL (Riyaz, Dhruv, Preetam) for the NOVA AI Hackathon 2026.

---

## 📋 Prerequisites
- **Python 3.10+**
- **Node.js 18+**
- **PostgreSQL 14+** OR **Docker Desktop**
- **Neo4j** (Optional)
- **Ollama** (Optional)

> [!WARNING]
> **First-time installation may take 10-15 minutes** depending on your network speed and storage. The full dependency installation is **~10GB** (Python ML libraries + Node modules). Ensure you have sufficient disk space and a stable internet connection.

> [!NOTE]
> **Initial dashboard load may take 15-30 seconds** as the database initializes and caches data. Subsequent loads will be faster.

---

## 🔑 Free API Keys (Shared)

We're providing free API keys for evaluation purposes:

```
GROQ_API_KEY=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
```

> [!TIP]
> These are shared keys for hackathon judges and evaluators. Gemini is the primary AI provider; Groq is the fallback.

---

## 🚀 Quick Start (Unified Launcher)

### 1. Environment Configuration
```bash
cp .env.example .env
```
Edit `.env` and set:
- `DB_PASSWORD` - Your PostgreSQL password
- `GOOGLE_API_KEY` - Use the shared key above or get one at [aistudio.google.com](https://aistudio.google.com/apikey)
- `GROQ_API_KEY` - (fallback) Use the shared key above or get one at [console.groq.com](https://console.groq.com)

### 2. Download Database (Required)
Download the database dump and place it in the `database/` folder:
```
database/reproduction_dump.sql
```
📥 **[Download Database Dump](https://drive.google.com/file/d/1rurYOMNUhG6ncy-e7EI7jDXl6NU9y_K2/view?usp=sharing)** (~212 MB)

### 3. Run the Application
```bash
python run.py                 # Auto-detect best mode
python run.py --docker        # Use Docker for PostgreSQL
python run.py --local         # Use local PostgreSQL
python run.py --frontend-only # Just run the UI (for preview)
python run.py --skip-db       # Skip database setup
```

> [!TIP]
> **No PostgreSQL installed?** Use `--docker` - it will automatically start PostgreSQL in a container.

**What `run.py` does:**
1. ✅ Checks Python, Node.js, and database availability
2. ✅ Creates Python virtual environment (if needed)
3. ✅ Installs all dependencies (pip + npm)
4. ✅ Sets up the database with 57,974 patient records
5. ✅ Starts backend API on `http://127.0.0.1:8000`
6. ✅ Starts frontend on `http://localhost:5173`
7. ✅ Handles graceful shutdown with Ctrl+C

### 4. Access the Application
- **Frontend:** http://localhost:5173
- **Backend API:** http://127.0.0.1:8000
- **API Docs:** http://127.0.0.1:8000/docs

---

## 🛠️ Manual Setup (Alternative)

<details>
<summary>Click to expand manual setup instructions</summary>

### Database Setup
```bash
# Using Docker:
docker run -d --name trialplus-postgres -p 5432:5432 \
  -e POSTGRES_PASSWORD=chitti \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_DB=trialpulse_test \
  postgres:16

# Wait for PostgreSQL to start, then restore dump:
docker cp database/reproduction_dump.sql trialplus-postgres:/tmp/dump.sql
docker exec trialplus-postgres psql -U postgres -d trialpulse_test -f /tmp/dump.sql -q
```

Or use local PostgreSQL:
```bash
python setup_repro_db.py
```

### Backend Setup
```bash
cd backend
python -m venv venv

# Windows:
.\venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### Frontend Setup (new terminal)
```bash
cd frontend
npm install
npm run dev
```

</details>

---

## 📊 Model Weights & Research

All ML model weights (including fine-tuned models), research notebooks, and result plots are available at:

🔗 **[Sanchalak AI Models Repository](https://github.com/PARZIVALPRIME/sanchalak_ai)**

## 📚 Documentation

Full technical documentation is included in this repository:

📄 **[Documentation.pdf](./Documentation.pdf)** - Complete 6-module technical documentation

---

## 🧪 Optional Features

### Neo4j Knowledge Graph (Cascade Intelligence)
```bash
python scripts/seed_neo4j_cascade.py
```

---
**Sanchalak AI**
