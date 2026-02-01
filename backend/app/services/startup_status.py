"""
Startup Status Tracker
======================
Singleton service to track application startup progress.
"""

from datetime import datetime
from typing import Optional
import threading


class StartupStatusTracker:
    """Thread-safe singleton to track startup progress."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.ready = False
        self.stage = "initializing"
        self.percent = 0
        self.details = "Starting up..."
        self.started_at = datetime.utcnow()
        self._initialized = True
    
    def update(self, stage: str, percent: int, details: str = ""):
        """Update startup progress."""
        self.stage = stage
        self.percent = min(max(percent, 0), 100)
        self.details = details or stage.replace("_", " ").title()
        
        if percent >= 100:
            self.ready = True
    
    def mark_ready(self):
        """Mark startup as complete."""
        self.ready = True
        self.stage = "ready"
        self.percent = 100
        self.details = "System ready"
    
    def reset(self):
        """Reset for new startup cycle."""
        self.ready = False
        self.stage = "initializing"
        self.percent = 0
        self.details = "Starting up..."
        self.started_at = datetime.utcnow()
    
    def get_status(self) -> dict:
        """Get current startup status."""
        elapsed = (datetime.utcnow() - self.started_at).total_seconds()
        return {
            "ready": self.ready,
            "stage": self.stage,
            "percent": self.percent,
            "details": self.details,
            "elapsed_seconds": round(elapsed, 1)
        }


# Global instance
startup_status = StartupStatusTracker()


def get_startup_status() -> StartupStatusTracker:
    """Get the global startup status tracker."""
    return startup_status
