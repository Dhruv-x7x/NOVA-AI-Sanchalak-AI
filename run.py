#!/usr/bin/env python3
"""
a6on-i - Unified Launcher
====================================
Cross-platform launcher script with Docker support.

Usage:
    python run.py                    # Auto-detect best mode
    python run.py --docker           # Use Docker for PostgreSQL
    python run.py --local            # Use local PostgreSQL
    python run.py --frontend-only    # Just run frontend (no backend)
    python run.py --skip-db          # Skip database setup
"""

import argparse
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.resolve()
BACKEND_DIR = PROJECT_ROOT / "backend"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
DATABASE_DUMP = PROJECT_ROOT / "database" / "reproduction_dump.sql"
ENV_FILE = PROJECT_ROOT / ".env"

# Docker settings
DOCKER_CONTAINER_NAME = "a6on-i-postgres"
DOCKER_IMAGE = "postgres:16"

# Load settings from .env if available
def _load_env_value(key, default):
    """Load a value from .env file if it exists."""
    if ENV_FILE.exists():
        with open(ENV_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith(f'{key}='):
                    return line.split('=', 1)[1].strip().strip('"').strip("'")
    return default

POSTGRES_PORT = int(_load_env_value('DB_PORT', '5432'))
POSTGRES_USER = "postgres"
POSTGRES_PASSWORD = _load_env_value('DB_PASSWORD', 'chitti')
POSTGRES_DB = "trialpulse_test"

# Server ports
BACKEND_PORT = 8000
FRONTEND_PORT = 5173

# Process holders for cleanup
running_processes = []


# =============================================================================
# UTILITIES
# =============================================================================

def is_windows():
    return platform.system() == "Windows"


def print_header(text):
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print('=' * 60)


def print_step(icon, text):
    print(f"[{icon}]  {text}")


def print_error(text):
    print(f"\n[ERROR] {text}")


def print_success(text):
    print(f"\n[SUCCESS] {text}")


def run_cmd(cmd, cwd=None, capture=False, check=True):
    """Run a command and optionally capture output."""
    try:
        if capture:
            result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=check)
            return result.stdout.strip()
        else:
            subprocess.run(cmd, cwd=cwd, check=check)
            return True
    except subprocess.CalledProcessError as e:
        if capture:
            return None
        raise e
    except FileNotFoundError:
        return None


def command_exists(cmd):
    """Check if a command exists in PATH."""
    return shutil.which(cmd) is not None


def wait_for_postgres(host="127.0.0.1", port=5432, timeout=30):
    """Wait for PostgreSQL to accept connections."""
    import socket
    start = time.time()
    while time.time() - start < timeout:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            sock.close()
            if result == 0:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


# =============================================================================
# DEPENDENCY CHECKS
# =============================================================================

def check_python():
    """Check Python version."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print_error(f"Python 3.10+ required, found {version.major}.{version.minor}")
        return False
    print_step("PY", f"Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_node():
    """Check Node.js installation."""
    if not command_exists("node"):
        print_error("Node.js not found. Install from https://nodejs.org/")
        return False
    version = run_cmd(["node", "--version"], capture=True)
    if version:
        print_step("JS", f"Node.js {version}")
    return True


def check_npm():
    """Check npm installation."""
    if not command_exists("npm"):
        print_error("npm not found.")
        return False
    return True


def check_docker():
    """Check Docker availability."""
    if not command_exists("docker"):
        return False
    # Check if Docker daemon is running
    result = run_cmd(["docker", "info"], capture=True, check=False)
    if result is None:
        return False
    if "error" in str(result).lower():
        return False
    print_step("DK", "Docker available")
    return True


def find_local_psql():
    """Find local PostgreSQL installation."""
    # Check PATH first
    if command_exists("psql"):
        print_step("DB", "PostgreSQL found in PATH")
        return "psql"
    
    # Windows-specific paths
    if is_windows():
        base_paths = [
            Path(r"C:\Program Files\PostgreSQL"),
            Path(r"C:\Program Files (x86)\PostgreSQL"),
        ]
        for base in base_paths:
            if base.exists():
                for version in ["18", "17", "16", "15", "14", "13", "12"]:
                    psql = base / version / "bin" / "psql.exe"
                    if psql.exists():
                        print_step("DB", f"PostgreSQL {version} found")
                        return str(psql)
    
    # macOS Homebrew
    homebrew_psql = Path("/opt/homebrew/bin/psql")
    if homebrew_psql.exists():
        print_step("DB", "PostgreSQL found (Homebrew)")
        return str(homebrew_psql)
    
    # Linux common paths
    linux_psql = Path("/usr/bin/psql")
    if linux_psql.exists():
        print_step("DB", "PostgreSQL found")
        return str(linux_psql)
    
    return None


# =============================================================================
# DOCKER MANAGEMENT
# =============================================================================

def docker_container_exists():
    """Check if our container exists."""
    result = run_cmd(
        ["docker", "ps", "-a", "--filter", f"name={DOCKER_CONTAINER_NAME}", "--format", "{{.Names}}"],
        capture=True, check=False
    )
    return result == DOCKER_CONTAINER_NAME


def docker_container_running():
    """Check if our container is running."""
    result = run_cmd(
        ["docker", "ps", "--filter", f"name={DOCKER_CONTAINER_NAME}", "--format", "{{.Names}}"],
        capture=True, check=False
    )
    return result == DOCKER_CONTAINER_NAME


def docker_remove_container():
    """Remove existing container (fresh start)."""
    if docker_container_exists():
        print_step("RM", f"Removing old container '{DOCKER_CONTAINER_NAME}'...")
        run_cmd(["docker", "rm", "-f", DOCKER_CONTAINER_NAME], check=False)


def docker_start_postgres():
    """Start PostgreSQL in Docker."""
    print_step("DK", "Starting PostgreSQL in Docker...")
    
    # Always remove old container for fresh start
    docker_remove_container()
    
    # Start new container
    cmd = [
        "docker", "run", "-d",
        "--name", DOCKER_CONTAINER_NAME,
        "-p", f"{POSTGRES_PORT}:5432",
        "-e", f"POSTGRES_PASSWORD={POSTGRES_PASSWORD}",
        "-e", f"POSTGRES_USER={POSTGRES_USER}",
        "-e", f"POSTGRES_DB={POSTGRES_DB}",
        DOCKER_IMAGE
    ]
    
    try:
        run_cmd(cmd)
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to start Docker container: {e}")
        return False
    
    # Wait for PostgreSQL to be ready
    print_step("WT", "Waiting for PostgreSQL to start...")
    if not wait_for_postgres(timeout=30):
        print_error("PostgreSQL failed to start in time")
        return False
    
    # Extra wait for initialization
    time.sleep(2)
    print_step("OK", "PostgreSQL is ready")
    return True


def docker_restore_dump():
    """Restore database dump into Docker container."""
    if not DATABASE_DUMP.exists():
        print_error(f"Database dump not found: {DATABASE_DUMP}")
        return False
    
    print_step("LD", "Restoring database dump (this may take 1-2 minutes)...")
    
    # Read and sanitize the dump to remove incompatible settings
    print_step("FX", "Sanitizing dump for compatibility...")
    with open(DATABASE_DUMP, 'r', encoding='utf-8', errors='ignore') as f:
        dump_content = f.read()
    
    # Remove settings that may not exist in target PostgreSQL version
    incompatible_settings = [
        'transaction_timeout',
        'idle_session_timeout',
        'statement_timeout_in_transaction_blocks',
    ]
    
    lines = dump_content.split('\n')
    filtered_lines = []
    for line in lines:
        skip = False
        for setting in incompatible_settings:
            if setting in line.lower():
                skip = True
                break
        if not skip:
            filtered_lines.append(line)
    
    sanitized_content = '\n'.join(filtered_lines)
    
    # Write sanitized dump to temp file
    sanitized_dump = PROJECT_ROOT / "database" / "sanitized_dump.sql"
    with open(sanitized_dump, 'w', encoding='utf-8') as f:
        f.write(sanitized_content)
    
    # Copy sanitized dump file into container
    run_cmd(["docker", "cp", str(sanitized_dump), f"{DOCKER_CONTAINER_NAME}:/tmp/dump.sql"])
    
    # Restore it
    start = time.time()
    cmd = [
        "docker", "exec", DOCKER_CONTAINER_NAME,
        "psql", "-U", POSTGRES_USER, "-d", POSTGRES_DB, "-f", "/tmp/dump.sql", "-q"
    ]
    
    try:
        run_cmd(cmd)
        duration = time.time() - start
        print_step("OK", f"Database restored in {duration:.1f}s")
        # Clean up sanitized dump
        if sanitized_dump.exists():
            sanitized_dump.unlink()
        
        # Create indexes for faster queries
        print_step("IX", "Creating database indexes for performance...")
        index_cmd = [
            "docker", "exec", DOCKER_CONTAINER_NAME,
            "psql", "-U", POSTGRES_USER, "-d", POSTGRES_DB, "-c",
            """
            CREATE INDEX IF NOT EXISTS idx_patients_study ON patients(study_id);
            CREATE INDEX IF NOT EXISTS idx_patients_site ON patients(site_id);
            CREATE INDEX IF NOT EXISTS idx_patients_dqi ON patients(dqi_score);
            CREATE INDEX IF NOT EXISTS idx_patients_key ON patients(patient_key);
            CREATE INDEX IF NOT EXISTS idx_patients_risk ON patients(risk_score);
            CREATE INDEX IF NOT EXISTS idx_issues_status ON project_issues(status);
            CREATE INDEX IF NOT EXISTS idx_issues_site ON project_issues(site_id);
            CREATE INDEX IF NOT EXISTS idx_issues_patient ON project_issues(patient_key);
            CREATE INDEX IF NOT EXISTS idx_sites_dqi ON clinical_sites(dqi_score);
            """
        ]
        run_cmd(index_cmd, check=False)  # Don't fail if indexes already exist
        print_step("OK", "Database indexes created")
        
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to restore database: {e}")
        return False



def docker_cleanup():
    """Stop and remove Docker container."""
    if docker_container_exists():
        print_step("ST", "Stopping Docker container...")
        run_cmd(["docker", "rm", "-f", DOCKER_CONTAINER_NAME], check=False)


# =============================================================================
# LOCAL POSTGRESQL
# =============================================================================

def local_restore_dump(psql_path):
    """Restore dump to local PostgreSQL — only if database is empty or missing."""
    if not DATABASE_DUMP.exists():
        print_error(f"Database dump not found: {DATABASE_DUMP}")
        return False
    
    env = os.environ.copy()
    env["PGPASSWORD"] = POSTGRES_PASSWORD
    port_str = str(POSTGRES_PORT)
    
    # Check if database already exists and has data
    try:
        import psycopg2
        conn = psycopg2.connect(
            host="127.0.0.1", port=POSTGRES_PORT,
            user=POSTGRES_USER, password=POSTGRES_PASSWORD,
            dbname=POSTGRES_DB
        )
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM patients")
        count = cur.fetchone()[0]
        cur.close()
        conn.close()
        
        if count > 0:
            print_step("OK", f"Database already has {count:,} patients — skipping restore")
            return True
        else:
            print_step("!!", "Database exists but is empty — will restore")
    except Exception:
        print_step("!!", "Database not found or inaccessible — will create and restore")
    
    # Force disconnect users and drop existing database
    print_step("RM", f"Dropping existing {POSTGRES_DB} (forcing disconnects)...")
    terminate_env = env.copy()
    terminate_env["PAGER"] = ""
    terminate_cmd = [
        psql_path, "-h", "127.0.0.1", "-p", port_str, "-U", POSTGRES_USER, "-d", "postgres", 
        "-c", f"SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '{POSTGRES_DB}' AND pid <> pg_backend_pid();"
    ]
    subprocess.run(terminate_cmd, env=terminate_env, capture_output=True)
    
    drop_cmd = [
        psql_path, "-h", "127.0.0.1", "-p", port_str, "-U", POSTGRES_USER, "-d", "postgres", 
        "-c", f"DROP DATABASE IF EXISTS {POSTGRES_DB} WITH (FORCE);"
    ]
    subprocess.run(drop_cmd, env=terminate_env, capture_output=True)
    
    # Create database
    print_step("DB", f"Creating {POSTGRES_DB}...")
    subprocess.run([psql_path, "-h", "127.0.0.1", "-p", port_str, "-U", POSTGRES_USER, "-d", "postgres",
          "-c", f"CREATE DATABASE {POSTGRES_DB};"], env=env, check=True)
    
    # Restore dump
    print_step("LD", "Restoring database dump (this may take 1-2 minutes)...")
    start = time.time()
    
    # Fix for Windows PAGER issue where cat is not available
    restore_env = env.copy()
    restore_env["PAGER"] = "" # Disable pager
    
    subprocess.run(
        [psql_path, "-h", "127.0.0.1", "-p", port_str, "-U", POSTGRES_USER, "-d", POSTGRES_DB,
         "-f", str(DATABASE_DUMP), "-q"],
        env=restore_env, check=True
    )
    duration = time.time() - start
    print_step("OK", f"Database restored in {duration:.1f}s")
    return True


# =============================================================================
# VENV AND DEPENDENCIES
# =============================================================================

def get_venv_python():
    """Get path to venv Python executable."""
    # Check root .venv first (preferred), then backend/venv
    root_venv = PROJECT_ROOT / ".venv"
    backend_venv = BACKEND_DIR / "venv"
    
    if is_windows():
        root_path = root_venv / "Scripts" / "python.exe"
        backend_path = backend_venv / "Scripts" / "python.exe"
    else:
        root_path = root_venv / "bin" / "python"
        backend_path = backend_venv / "bin" / "python"
    
    if root_path.exists():
        return root_path
    return backend_path


def get_venv_pip():
    """Get path to venv pip executable."""
    root_venv = PROJECT_ROOT / ".venv"
    backend_venv = BACKEND_DIR / "venv"
    
    if is_windows():
        root_path = root_venv / "Scripts" / "pip.exe"
        backend_path = backend_venv / "Scripts" / "pip.exe"
    else:
        root_path = root_venv / "bin" / "pip"
        backend_path = backend_venv / "bin" / "pip"
    
    if root_path.exists():
        return root_path
    return backend_path


def setup_backend_venv():
    """Create and setup backend virtual environment."""
    venv_python = get_venv_python()
    
    if not venv_python.exists():
        print_step("VN", "Creating Python virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", str(BACKEND_DIR / "venv")], check=True)
    
    # Check if dependencies are installed
    requirements = BACKEND_DIR / "requirements.txt"
    if requirements.exists():
        print_step("PK", "Installing backend dependencies...")
        subprocess.run([str(get_venv_pip()), "install", "-r", str(requirements), "-q"], check=True)
    
    return True


def setup_frontend():
    """Install frontend dependencies."""
    node_modules = FRONTEND_DIR / "node_modules"
    
    if not node_modules.exists():
        print_step("PK", "Installing frontend dependencies...")
        try:
            subprocess.run(
                ["npm", "install"], 
                cwd=FRONTEND_DIR, 
                check=True,
                shell=is_windows()  # npm needs shell on Windows
            )
        except FileNotFoundError:
            print_error("npm not found. Please install Node.js from https://nodejs.org/")
            print("   After installing, restart your terminal and run again.")
            return False
    else:
        print_step("OK", "Frontend dependencies already installed")
    
    return True


# =============================================================================
# SERVER MANAGEMENT
# =============================================================================

def start_backend():
    """Start the backend server."""
    print_step("RK", f"Starting backend on http://127.0.0.1:{BACKEND_PORT}")
    
    venv_python = get_venv_python()
    cmd = [
        str(venv_python), "-m", "uvicorn", 
        "app.main:app", 
        "--host", "127.0.0.1", 
        "--port", str(BACKEND_PORT),
        "--reload",
        "--reload-dir", str(BACKEND_DIR),
        "--reload-dir", str(PROJECT_ROOT / "src"),
        "--reload-dir", str(PROJECT_ROOT / "config"),
    ]
    
    # Start in background (logs stream to console)
    process = subprocess.Popen(
        cmd, 
        cwd=BACKEND_DIR
    )
    running_processes.append(("Backend", process))
    return process


def start_frontend():
    """Start the frontend dev server."""
    print_step("RK", f"Starting frontend on http://localhost:{FRONTEND_PORT}")
    
    cmd = ["npm", "run", "dev"]
    
    # Start in background (logs stream to console)
    process = subprocess.Popen(
        cmd,
        cwd=FRONTEND_DIR,
        shell=True  # npm needs shell on Windows/Linux usually works better too
    )
    running_processes.append(("Frontend", process))
    return process


def cleanup_processes(signum=None, frame=None):
    """Clean up running processes on exit."""
    print("\n")
    print_step("ST", "Shutting down...")
    
    for name, proc in running_processes:
        if proc.poll() is None:
            print_step("..", f"Stopping {name}...")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    
    # Cleanup Docker if it was used
    if docker_container_running():
        docker_cleanup()
    
    print_success("Shutdown complete")
    sys.exit(0)


def validate_db_data():
    """Verify that the database actually contains trial data."""
    try:
        import psycopg2
        conn = psycopg2.connect(
            host="127.0.0.1",
            port=POSTGRES_PORT,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            dbname=POSTGRES_DB
        )
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM patients")
        count = cur.fetchone()[0]
        cur.close()
        conn.close()
        
        if count == 0:
            print_error("Database is EMPTY! Clinical reports will use MOCK data.")
            print("   Please run without --skip-db to restore the data dump.")
            return False
        
        print_step("OK", f"Database validated: {count:,} patients found")
        return True
    except Exception as e:
        print_error(f"Database validation failed: {e}")
        return False


def monitor_processes():
    """Monitor running processes and display output."""
    print_success("All services started!")
    
    # Run data validation in background
    time.sleep(2)
    validate_db_data()
    
    print(f"\n   Backend:  http://127.0.0.1:{BACKEND_PORT}")
    print(f"   Frontend: http://localhost:{FRONTEND_PORT}")
    print(f"\n   Press Ctrl+C to stop all services\n")
    print("-" * 60)
    
    try:
        while True:
            for name, proc in running_processes:
                if proc.poll() is not None:
                    print_error(f"{name} process died unexpectedly")
                    cleanup_processes()
                    return
            time.sleep(1)
    except KeyboardInterrupt:
        cleanup_processes()


# =============================================================================
# MAIN MODES
# =============================================================================

def run_full_docker():
    """Run everything with Docker PostgreSQL."""
    print_header("Docker Mode - Full Setup")
    
    if not check_docker():
        print_error("Docker is not available. Install Docker Desktop.")
        return False
    
    if not docker_start_postgres():
        return False
    
    if not docker_restore_dump():
        docker_cleanup()
        return False
    
    if not setup_backend_venv():
        docker_cleanup()
        return False
    
    if not setup_frontend():
        docker_cleanup()
        return False
    
    start_backend()
    time.sleep(2)  # Give backend time to start
    start_frontend()
    
    monitor_processes()
    return True


def run_full_local():
    """Run everything with local PostgreSQL."""
    print_header("Local Mode - Full Setup")
    
    psql_path = find_local_psql()
    if not psql_path:
        print_error("Local PostgreSQL not found.")
        print("   Install PostgreSQL or use --docker mode")
        return False
    
    try:
        if not local_restore_dump(psql_path):
            return False
    except subprocess.CalledProcessError as e:
        print_error(f"Database setup failed: {e}")
        print("   Check your PostgreSQL password in .env")
        return False
    
    if not setup_backend_venv():
        return False
    
    if not setup_frontend():
        return False
    
    start_backend()
    time.sleep(2)
    start_frontend()
    
    monitor_processes()
    return True


def run_frontend_only():
    """Run just the frontend."""
    print_header("Frontend Only Mode")
    
    print_step("!!", "Backend not running - API calls will fail")
    
    if not setup_frontend():
        return False
    
    start_frontend()
    monitor_processes()
    return True


def run_skip_db():
    """Run backend and frontend, skip database setup."""
    print_header("Skip Database Mode")
    
    print_step("!!", "Skipping database setup - using existing data")
    
    if not setup_backend_venv():
        return False
    
    if not setup_frontend():
        return False
    
    start_backend()
    time.sleep(2)
    start_frontend()
    
    monitor_processes()
    return True


def auto_detect_mode():
    """Auto-detect the best mode to run."""
    print_header("Auto-Detect Mode")
    
    # Check what's available
    has_docker = check_docker()
    has_local_psql = find_local_psql() is not None
    
    if has_local_psql:
        print_step("->", "Using local PostgreSQL")
        return run_full_local()
    elif has_docker:
        print_step("->", "Using Docker PostgreSQL")
        return run_full_docker()
    else:
        print_step("!!", "No database available - running frontend only")
        return run_frontend_only()


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    # Only run if it's the main process
    if os.environ.get("RUN_PY_STARTED") == "1":
        return
    os.environ["RUN_PY_STARTED"] = "1"
    
    parser = argparse.ArgumentParser(
        description="a6on-i - Unified Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                # Auto-detect best mode
  python run.py --docker       # Use Docker for PostgreSQL
  python run.py --local        # Use local PostgreSQL
  python run.py --frontend-only # Just run frontend
  python run.py --skip-db      # Skip database setup
        """
    )
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--docker", action="store_true", help="Use Docker for PostgreSQL")
    group.add_argument("--local", action="store_true", help="Use local PostgreSQL")
    group.add_argument("--frontend-only", action="store_true", help="Run frontend only (no backend)")
    group.add_argument("--skip-db", action="store_true", help="Skip database setup")
    
    args = parser.parse_args()
    
    # Setup signal handlers for cleanup
    signal.signal(signal.SIGINT, cleanup_processes)
    signal.signal(signal.SIGTERM, cleanup_processes)
    
    # Use ASCII for robustness against encoding issues
    print("\n" + "="*62)
    print("           a6on-i - Unified Launcher              ")
    print("="*62 + "\n")
    
    # Check basic dependencies
    print_header("Checking Dependencies")
    
    if not check_python():
        sys.exit(1)
    if not check_node():
        sys.exit(1)
    if not check_npm():
        sys.exit(1)

    print("\n[!!] IMPORTANT: Ensure your .env file has the correct DB_PASSWORD!")
    print("     If you see 500 errors, check your database credentials.\n")
    
    # Run appropriate mode
    try:
        if args.docker:
            success = run_full_docker()
        elif args.local:
            success = run_full_local()
        elif args.frontend_only:
            success = run_frontend_only()
        elif args.skip_db:
            success = run_skip_db()
        else:
            success = auto_detect_mode()
        
        if not success:
            sys.exit(1)
            
    except KeyboardInterrupt:
        cleanup_processes()
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        cleanup_processes()
        sys.exit(1)


if __name__ == "__main__":
    main()
