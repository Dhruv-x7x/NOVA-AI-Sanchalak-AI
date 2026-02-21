"""
SANCHALAK AI - Temporal Snapshot Worker
==========================================
Background worker to capture hourly snapshots and detect deltas.
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
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.temporal_snapshots import get_snapshot_service

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(PROJECT_ROOT / "logs" / "snapshot_worker.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SnapshotWorker:
    def __init__(self, interval_seconds: int = 3600):
        self.interval = interval_seconds
        self.running = True
        self.service = get_snapshot_service()
        
        # Register signals
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)
        
    def stop(self, *args):
        logger.info("Stopping snapshot worker...")
        self.running = False
        
    def run(self):
        logger.info(f"Temporal Snapshot Worker started (Interval: {self.interval}s)")
        
        while self.running:
            try:
                start_time = time.time()
                
                # 1. Capture snapshot
                logger.info("Capturing trial state snapshot...")
                snapshot_result = self.service.capture_snapshot()
                
                if snapshot_result.get("status") == "success":
                    logger.info(f"Snapshot captured: {snapshot_result.get('metrics_captured')} metrics")
                    
                    # 2. Detect deltas and anomalies
                    logger.info("Detecting delta changes...")
                    deltas = self.service.detect_deltas(hours_back=1)
                    if deltas:
                        logger.info(f"ALERT: Detected {len(deltas)} anomalies in metrics")
                else:
                    logger.warning(f"Snapshot failed: {snapshot_result.get('error')}")
                
                # Sleep until next interval
                elapsed = time.time() - start_time
                sleep_time = max(0, self.interval - elapsed)
                
                if self.running:
                    logger.info(f"Sleeping for {sleep_time:.1f}s until next snapshot...")
                    time.sleep(sleep_time)
                    
            except Exception as e:
                logger.error(f"Worker loop error: {e}")
                time.sleep(60) # Wait a bit before retry on fatal error

if __name__ == "__main__":
    # Ensure logs directory exists
    (PROJECT_ROOT / "logs").mkdir(exist_ok=True)
    
    worker = SnapshotWorker()
    worker.run()
