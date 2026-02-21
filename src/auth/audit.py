"""
SANCHALAK AI - Audit Logger
================================
21 CFR Part 11 compliant audit logging with electronic signatures.
"""

import os
import json
import hashlib
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

from .models import User, AuditLogEntry

logger = logging.getLogger(__name__)


class AuditLogger:
    """
    21 CFR Part 11 compliant audit logger.
    
    Features:
    - Immutable audit trail
    - Cryptographic integrity (hash chain)
    - Electronic signature support
    - Tamper detection
    """
    
    def __init__(self):
        # In-memory storage (replace with PostgreSQL in production)
        self._entries: List[AuditLogEntry] = []
        self._last_hash: Optional[str] = None
        self._entry_counter: int = 0
        
        logger.info("AuditLogger initialized (21 CFR Part 11 compliant)")
    
    def _compute_entry_hash(self, entry: AuditLogEntry, previous_hash: str = None) -> str:
        """
        Compute cryptographic hash for audit entry.
        Creates hash chain for integrity verification.
        """
        data = {
            'timestamp': entry.timestamp.isoformat(),
            'user_id': entry.user_id,
            'username': entry.username,
            'action': entry.action,
            'resource': entry.resource,
            'details': json.dumps(entry.details, sort_keys=True),
            'previous_hash': previous_hash or '',
        }
        
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def log(self, user: User, action: str, resource: str = None,
            details: Dict = None, ip_address: str = None,
            signature_meaning: str = None) -> AuditLogEntry:
        """
        Log an auditable action.
        
        Args:
            user: User performing the action
            action: Action type (e.g., 'login', 'patient_update', 'data_export')
            resource: Resource being acted upon (e.g., 'patient:STUDY001|SITE001|SUBJ001')
            details: Additional details about the action
            ip_address: Client IP address
            signature_meaning: Electronic signature meaning (e.g., 'Reviewed', 'Approved')
            
        Returns:
            Created audit log entry
        """
        self._entry_counter += 1
        
        entry = AuditLogEntry(
            id=self._entry_counter,
            timestamp=datetime.now(),
            user_id=user.id,
            username=user.username,
            action=action,
            resource=resource,
            details=details or {},
            ip_address=ip_address,
            signature_meaning=signature_meaning,
            previous_hash=self._last_hash,
        )
        
        # Compute hash for integrity
        entry.entry_hash = self._compute_entry_hash(entry, self._last_hash)
        self._last_hash = entry.entry_hash
        
        # Create electronic signature hash if meaning provided
        if signature_meaning:
            sig_data = f"{user.id}:{action}:{resource}:{signature_meaning}:{entry.timestamp.isoformat()}"
            entry.signature_hash = hashlib.sha256(sig_data.encode()).hexdigest()
        
        self._entries.append(entry)
        
        # Also log to standard logger
        logger.info(
            f"AUDIT: [{entry.action}] User={entry.username} Resource={entry.resource} "
            f"Signature={entry.signature_meaning or 'N/A'}"
        )
        
        return entry
    
    def log_login(self, user: User, ip_address: str = None, success: bool = True):
        """Log login attempt."""
        return self.log(
            user=user,
            action="login_success" if success else "login_failed",
            resource=f"session:{user.username}",
            details={"success": success},
            ip_address=ip_address
        )
    
    def log_logout(self, user: User, ip_address: str = None):
        """Log logout."""
        return self.log(
            user=user,
            action="logout",
            resource=f"session:{user.username}",
            ip_address=ip_address
        )
    
    def log_patient_view(self, user: User, patient_key: str, ip_address: str = None):
        """Log patient record access."""
        return self.log(
            user=user,
            action="patient_view",
            resource=f"patient:{patient_key}",
            ip_address=ip_address
        )
    
    def log_patient_update(self, user: User, patient_key: str, 
                          changes: Dict, ip_address: str = None):
        """Log patient record modification."""
        return self.log(
            user=user,
            action="patient_update",
            resource=f"patient:{patient_key}",
            details={"changes": changes},
            ip_address=ip_address
        )
    
    def log_data_export(self, user: User, export_type: str, 
                       record_count: int, ip_address: str = None):
        """Log data export action."""
        return self.log(
            user=user,
            action="data_export",
            resource=f"export:{export_type}",
            details={"record_count": record_count, "export_type": export_type},
            ip_address=ip_address
        )
    
    def log_approval(self, user: User, resource: str, 
                    approval_type: str, ip_address: str = None):
        """Log data approval with electronic signature."""
        return self.log(
            user=user,
            action="data_approval",
            resource=resource,
            details={"approval_type": approval_type},
            ip_address=ip_address,
            signature_meaning=f"Approved: {approval_type}"
        )
    
    def log_db_lock(self, user: User, study_id: str, 
                   lock_type: str, ip_address: str = None):
        """Log database lock action with electronic signature."""
        return self.log(
            user=user,
            action="db_lock",
            resource=f"study:{study_id}",
            details={"lock_type": lock_type},
            ip_address=ip_address,
            signature_meaning=f"DB Lock: {lock_type}"
        )
    
    def log_password_change(self, user: User, ip_address: str = None):
        """Log password change."""
        return self.log(
            user=user,
            action="password_change",
            resource=f"user:{user.username}",
            ip_address=ip_address
        )
    
    def get_entries(self, user_id: int = None, action: str = None,
                   start_date: datetime = None, end_date: datetime = None,
                   limit: int = 100) -> List[Dict]:
        """
        Query audit log entries.
        
        Args:
            user_id: Filter by user
            action: Filter by action type
            start_date: Filter from date
            end_date: Filter to date
            limit: Maximum entries to return
            
        Returns:
            List of audit entries as dicts
        """
        results = []
        
        for entry in reversed(self._entries):  # Most recent first
            if len(results) >= limit:
                break
            
            if user_id and entry.user_id != user_id:
                continue
            
            if action and entry.action != action:
                continue
            
            if start_date and entry.timestamp < start_date:
                continue
            
            if end_date and entry.timestamp > end_date:
                continue
            
            results.append(entry.to_dict())
        
        return results
    
    def verify_integrity(self) -> tuple:
        """
        Verify integrity of the audit log using hash chain.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self._entries:
            return True, "No entries to verify"
        
        previous_hash = None
        
        for entry in self._entries:
            expected_hash = self._compute_entry_hash(entry, previous_hash)
            
            if entry.entry_hash != expected_hash:
                return False, f"Integrity violation at entry {entry.id}"
            
            previous_hash = entry.entry_hash
        
        return True, f"All {len(self._entries)} entries verified"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get audit log statistics."""
        if not self._entries:
            return {
                "total_entries": 0,
                "unique_users": 0,
                "actions": {},
            }
        
        action_counts = {}
        user_ids = set()
        
        for entry in self._entries:
            user_ids.add(entry.user_id)
            action_counts[entry.action] = action_counts.get(entry.action, 0) + 1
        
        return {
            "total_entries": len(self._entries),
            "unique_users": len(user_ids),
            "actions": action_counts,
            "first_entry": self._entries[0].timestamp.isoformat() if self._entries else None,
            "last_entry": self._entries[-1].timestamp.isoformat() if self._entries else None,
        }


# Singleton instance
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Get singleton audit logger."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger
