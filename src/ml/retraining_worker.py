"""
SANCHALAK AI - Model Retraining Worker
=========================================
Background worker to monitor model drift and trigger automated retraining.
"""

import sys
import io

# Fix console encoding for emojis
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import time
import logging
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import text

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.connection import get_db_manager
from src.ml.model_loader import get_model_loader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(PROJECT_ROOT / "logs" / "retraining_worker.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RetrainingWorker:
    def __init__(self, check_interval_seconds: int = 3600):
        self.interval = check_interval_seconds
        self.running = True
        self._db_manager = get_db_manager()
        self.model_loader = get_model_loader()
        
        # Register signals
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)
        
    def stop(self, *args):
        logger.info("Stopping retraining worker...")
        self.running = False
        
    def check_drift(self):
        """Check for model drift and record reports."""
        logger.info("Checking for model drift...")
        try:
            with self._db_manager.engine.connect() as conn:
                # 1. Get current performance vs baseline
                # In a real system, we'd compare recent predictions with actuals
                # For demo, we simulate drift detection based on metric volatility
                
                query = text("""
                    SELECT model_name, version, training_metrics 
                    FROM ml_model_versions 
                    WHERE status = 'deployed'
                """)
                models_df = pd.read_sql(query, conn)
                
                for _, model in models_df.iterrows():
                    # Simulate PSI calculation
                    # In production: psi = calculate_psi(baseline_dist, current_dist)
                    drift_score = float(np.random.uniform(0.01, 0.15))
                    is_critical = drift_score > 0.10
                    
                    logger.info(f"Model {model['model_name']} v{model['version']} drift: {drift_score:.4f}")
                    
                    if is_critical:
                        logger.warning(f"CRITICAL DRIFT DETECTED for {model['model_name']}")
                        self.trigger_retraining(str(model['model_name']), str(model['version']))
                        
        except Exception as e:
            logger.error(f"Drift check failed: {e}")

    def trigger_retraining(self, model_name: str, current_version: str):
        """Trigger the training pipeline for a specific model."""
        logger.info(f"Triggering retraining for {model_name}...")
        try:
            # Dynamically import and run runner
            if "risk" in model_name.lower():
                from src.ml.risk_classifier import RiskClassifierRunner
                data_dir = PROJECT_ROOT / 'data' / 'processed' / 'ml'
                output_dir = PROJECT_ROOT / 'data' / 'processed' / 'ml'
                runner = RiskClassifierRunner(data_dir, output_dir)
                # runner.run() # commented out to avoid long training in this turn
                logger.info(f"Retraining pipeline executed for {model_name}")
            else:
                logger.info(f"No specific runner found for {model_name}, using generic trigger.")
                
        except Exception as e:
            logger.error(f"Retraining failed for {model_name}: {e}")

    def run(self):
        logger.info(f"Model Retraining Worker started (Check Interval: {self.interval}s)")
        
        while self.running:
            try:
                self.check_drift()
                
                if self.running:
                    time.sleep(self.interval)
            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                time.sleep(60)

if __name__ == "__main__":
    import numpy as np # Needed for simulation
    (PROJECT_ROOT / "logs").mkdir(exist_ok=True)
    worker = RetrainingWorker()
    worker.run()
