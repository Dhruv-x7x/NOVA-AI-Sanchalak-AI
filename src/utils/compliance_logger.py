"""
a6on-i - Compliance & Audit Engine (21 CFR Part 11)
============================================================
Ensures all data modifications are logged with non-repudiation hashing.
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional
from sqlalchemy import text, select, desc
from src.database.connection import get_db_manager
from src.database.models import AuditLog

logger = logging.getLogger(__name__)

class ComplianceLogger:
    """
    Handles immutable audit trail logging with SHA-256 integrity chaining.
    """
    
    def __init__(self):
        self._db = get_db_manager()

    def _get_previous_checksum(self) -> Optional[str]:
        """Fetch the checksum of the most recent audit log entry."""
        try:
            with self._db.engine.connect() as conn:
                res = conn.execute(
                    text("SELECT checksum FROM audit_logs ORDER BY timestamp DESC LIMIT 1")
                ).fetchone()
                return res[0] if res else None
        except Exception as e:
            logger.error(f"Failed to fetch previous checksum: {e}")
            return None

    def log_change(self, 
                   user_id: str, 
                   action: str, 
                   entity_type: str, 
                   entity_id: str,
                   field_name: Optional[str] = None,
                   old_value: Any = None,
                   new_value: Any = None,
                   reason: str = "System update",
                   ip_address: str = "0.0.0.0",
                   user_agent: str = "Internal") -> str:
        """
        Record a verifiable change in the audit trail.
        """
        timestamp = datetime.utcnow()
        prev_checksum = self._get_previous_checksum()
        
        # Prepare payload for hashing
        payload = {
            "timestamp": timestamp.isoformat(),
            "user_id": user_id,
            "action": action,
            "entity": f"{entity_type}:{entity_id}",
            "previous": prev_checksum or "GENESIS"
        }
        
        # Create SHA-256 hash (the digital fingerprint)
        payload_str = json.dumps(payload, sort_keys=True)
        checksum = hashlib.sha256(payload_str.encode()).hexdigest()
        
        try:
            with self._db.engine.connect() as conn:
                conn.execute(
                    text("""
                        INSERT INTO audit_logs 
                        (log_id, timestamp, user_id, action, entity_type, entity_id, 
                         field_name, old_value, new_value, reason, ip_address, 
                         user_agent, checksum, previous_checksum)
                        VALUES 
                        (:id, :ts, :u_id, :act, :e_type, :e_id, :f_name, :old, :new, 
                         :reason, :ip, :ua, :hash, :prev)
                    """),
                    {
                        "id": f"AUD-{hashlib.md5(payload_str.encode()).hexdigest()[:8]}",
                        "ts": timestamp,
                        "u_id": user_id,
                        "act": action,
                        "e_type": entity_type,
                        "e_id": entity_id,
                        "f_name": field_name,
                        "old": str(old_value) if old_value is not None else None,
                        "new": str(new_value) if new_value is not None else None,
                        "reason": reason,
                        "ip": ip_address,
                        "ua": user_agent,
                        "hash": checksum,
                        "prev": prev_checksum
                    }
                )
                conn.commit()
            
            logger.info(f"Compliance audit logged: {action} on {entity_type}/{entity_id}")
            return checksum
        except Exception as e:
            logger.error(f"CRITICAL: Failed to write compliance log: {e}")
            raise

_logger = None

def get_compliance_logger() -> ComplianceLogger:
    global _logger
    if _logger is None:
        _logger = ComplianceLogger()
    return _logger
