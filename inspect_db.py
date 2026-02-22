
import os
import sys
import pandas as pd
from sqlalchemy import text
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.database.connection import get_db_manager
    db_manager = get_db_manager()
    
    with db_manager.engine.connect() as conn:
        print("--- Listing Views ---")
        views = conn.execute(text("SELECT table_name FROM information_schema.views WHERE table_schema = 'public'")).fetchall()
        for v in views:
            print(f"View: {v[0]}")
        sys.exit(0)
        
except Exception as e:
    print(f"Error: {e}")
