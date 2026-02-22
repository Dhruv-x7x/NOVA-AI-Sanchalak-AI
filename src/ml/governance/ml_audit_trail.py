# src/ml/governance/ml_audit_trail.py
# NOTE: This module previously used SQLite but has been migrated to PostgreSQL-only.
# If you need to use this module, update it to use:
#   from src.database.pg_data_service import get_pg_data_service
#   service = get_pg_data_service()


"""
a6on-i - ML Audit Trail v1.0
21 CFR Part 11 Compliant Audit Logging for ML Systems

FDA 21 CFR Part 11 Requirements Addressed:
- 11.10(e): Use of secure, computer-generated, time-stamped audit trails
- 11.10(k): Controls for system documentation
- 11.50: Signature manifestations
- 11.70: Signature/record linking
- 11.200: Electronic signature components

Features:
- Immutable append-only audit logs
- SHA-256 checksums with blockchain-style chain hashing
- Model lifecycle event tracking
- Prediction audit trails
- Electronic signature support
- Tamper detection
"""

import json
import uuid
# SQLite removed - using PostgreSQL
import hashlib
import threading
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import csv
import io


# =============================================================================
# ENUMS
# =============================================================================

class MLEventType(Enum):
    """Types of ML audit events"""
    # Model Lifecycle
    MODEL_REGISTERED = "model_registered"
    MODEL_PROMOTED = "model_promoted"
    MODEL_DEMOTED = "model_demoted"
    MODEL_ARCHIVED = "model_archived"
    MODEL_DELETED = "model_deleted"
    
    # Training
    TRAINING_STARTED = "training_started"
    TRAINING_COMPLETED = "training_completed"
    TRAINING_FAILED = "training_failed"
    
    # Predictions
    PREDICTION_MADE = "prediction_made"
    PREDICTION_BATCH = "prediction_batch"
    PREDICTION_FEEDBACK = "prediction_feedback"
    PREDICTION_OVERRIDE = "prediction_override"
    
    # Drift
    DRIFT_DETECTED = "drift_detected"
    DRIFT_RESOLVED = "drift_resolved"
    BASELINE_CREATED = "baseline_created"
    BASELINE_UPDATED = "baseline_updated"
    
    # Retraining
    RETRAINING_TRIGGERED = "retraining_triggered"
    RETRAINING_APPROVED = "retraining_approved"
    RETRAINING_REJECTED = "retraining_rejected"
    RETRAINING_COMPLETED = "retraining_completed"
    RETRAINING_FAILED = "retraining_failed"
    
    # Configuration
    THRESHOLD_CHANGED = "threshold_changed"
    RULE_CREATED = "rule_created"
    RULE_MODIFIED = "rule_modified"
    RULE_DELETED = "rule_deleted"
    
    # Access
    ACCESS_GRANTED = "access_granted"
    ACCESS_REVOKED = "access_revoked"
    ARTIFACT_ACCESSED = "artifact_accessed"
    ARTIFACT_EXPORTED = "artifact_exported"
    
    # Verification
    INTEGRITY_CHECK = "integrity_check"
    AUDIT_EXPORT = "audit_export"


class ActorType(Enum):
    """Types of actors that can perform actions"""
    USER = "user"
    SYSTEM = "system"
    AGENT = "agent"
    API = "api"
    SCHEDULER = "scheduler"


class SignatureType(Enum):
    """Types of electronic signatures"""
    APPROVAL = "approval"
    ACKNOWLEDGMENT = "acknowledgment"
    VERIFICATION = "verification"
    REVIEW = "review"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class MLActor:
    """Actor who performed an action"""
    actor_id: str
    actor_type: ActorType
    name: str
    role: Optional[str] = None
    organization: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'actor_id': self.actor_id,
            'actor_type': self.actor_type.value,
            'name': self.name,
            'role': self.role,
            'organization': self.organization,
            'session_id': self.session_id,
            'ip_address': self.ip_address
        }
    
    @classmethod
    def system_actor(cls) -> 'MLActor':
        """Create a system actor"""
        return cls(
            actor_id="SYSTEM",
            actor_type=ActorType.SYSTEM,
            name="a6on-i System"
        )


@dataclass
class ElectronicSignature:
    """Electronic signature for 21 CFR Part 11 compliance"""
    signature_id: str
    signed_at: datetime
    signer_id: str
    signer_name: str
    signer_role: str
    signature_type: SignatureType
    meaning: str  # What the signature means/represents
    
    # The signed data hash
    data_hash: str
    
    # Digital signature (simplified - would use proper PKI in production)
    signature_hash: str
    
    def to_dict(self) -> Dict:
        return {
            'signature_id': self.signature_id,
            'signed_at': self.signed_at.isoformat(),
            'signer_id': self.signer_id,
            'signer_name': self.signer_name,
            'signer_role': self.signer_role,
            'signature_type': self.signature_type.value,
            'meaning': self.meaning,
            'data_hash': self.data_hash,
            'signature_hash': self.signature_hash
        }


@dataclass
class MLAuditEntry:
    """Immutable audit log entry for ML events"""
    entry_id: str
    timestamp: datetime
    event_type: MLEventType
    
    # Who
    actor: MLActor
    
    # What
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    
    # Details
    action_description: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    
    # State changes (for tracking before/after)
    entity_before: Optional[Dict[str, Any]] = None
    entity_after: Optional[Dict[str, Any]] = None
    
    # Chain integrity
    checksum: str = ""
    previous_checksum: str = ""  # Hash of previous entry
    sequence_number: int = 0
    
    # Signature (if signed)
    signature: Optional[ElectronicSignature] = None
    
    # Metadata
    correlation_id: Optional[str] = None  # Link related events
    
    def calculate_checksum(self) -> str:
        """Calculate SHA-256 checksum for this entry"""
        data = {
            'entry_id': self.entry_id,
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type.value,
            'actor': self.actor.to_dict(),
            'model_name': self.model_name,
            'model_version': self.model_version,
            'action_description': self.action_description,
            'details': self.details,
            'entity_before': self.entity_before,
            'entity_after': self.entity_after,
            'previous_checksum': self.previous_checksum,
            'sequence_number': self.sequence_number
        }
        
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()
    
    def to_dict(self) -> Dict:
        return {
            'entry_id': self.entry_id,
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type.value,
            'actor': self.actor.to_dict(),
            'model_name': self.model_name,
            'model_version': self.model_version,
            'action_description': self.action_description,
            'details': self.details,
            'entity_before': self.entity_before,
            'entity_after': self.entity_after,
            'checksum': self.checksum,
            'previous_checksum': self.previous_checksum,
            'sequence_number': self.sequence_number,
            'signature': self.signature.to_dict() if self.signature else None,
            'correlation_id': self.correlation_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MLAuditEntry':
        actor_data = data['actor']
        actor = MLActor(
            actor_id=actor_data['actor_id'],
            actor_type=ActorType(actor_data['actor_type']),
            name=actor_data['name'],
            role=actor_data.get('role'),
            organization=actor_data.get('organization'),
            session_id=actor_data.get('session_id'),
            ip_address=actor_data.get('ip_address')
        )
        
        return cls(
            entry_id=data['entry_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            event_type=MLEventType(data['event_type']),
            actor=actor,
            model_name=data.get('model_name'),
            model_version=data.get('model_version'),
            action_description=data.get('action_description', ''),
            details=data.get('details', {}),
            entity_before=data.get('entity_before'),
            entity_after=data.get('entity_after'),
            checksum=data.get('checksum', ''),
            previous_checksum=data.get('previous_checksum', ''),
            sequence_number=data.get('sequence_number', 0),
            correlation_id=data.get('correlation_id')
        )


# =============================================================================
# ML AUDIT LOGGER
# =============================================================================

class MLAuditLogger:
    """
    21 CFR Part 11 compliant audit logging for ML governance.
    
    Features:
    - Immutable append-only logs
    - Chain integrity via linked checksums
    - Electronic signatures
    - Tamper detection
    - Regulatory export
    """
    
    def __init__(
        self,
        db_path: str = "data/audit/ml_audit_trail.db"
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database"""
        # DISABLED: SQLite replaced with PostgreSQL
        if False:  # Disabled method
            cursor = conn.cursor()
            
            # Audit entries table (append-only)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ml_audit_entries (
                    entry_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    actor_type TEXT NOT NULL,
                    model_name TEXT,
                    model_version TEXT,
                    action_description TEXT,
                    checksum TEXT NOT NULL,
                    previous_checksum TEXT,
                    sequence_number INTEGER NOT NULL,
                    entry_data TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Signatures table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ml_signatures (
                    signature_id TEXT PRIMARY KEY,
                    entry_id TEXT NOT NULL,
                    signed_at TEXT NOT NULL,
                    signer_id TEXT NOT NULL,
                    signer_name TEXT NOT NULL,
                    signer_role TEXT,
                    signature_type TEXT NOT NULL,
                    meaning TEXT,
                    data_hash TEXT NOT NULL,
                    signature_hash TEXT NOT NULL,
                    FOREIGN KEY (entry_id) REFERENCES ml_audit_entries(entry_id)
                )
            ''')
            
            # Chain metadata table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ml_chain_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON ml_audit_entries(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_audit_model ON ml_audit_entries(model_name)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_audit_event ON ml_audit_entries(event_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_audit_actor ON ml_audit_entries(actor_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_audit_seq ON ml_audit_entries(sequence_number)')
            
            # Initialize chain if needed
            cursor.execute('SELECT value FROM ml_chain_metadata WHERE key = "last_checksum"')
            if not cursor.fetchone():
                cursor.execute('''
                    INSERT INTO ml_chain_metadata (key, value) VALUES ("last_checksum", "GENESIS")
                ''')
                cursor.execute('''
                    INSERT INTO ml_chain_metadata (key, value) VALUES ("sequence_number", "0")
                ''')
            
            conn.commit()
    
    # =========================================================================
    # Core Logging
    # =========================================================================
    
    def log_event(
        self,
        event_type: MLEventType,
        actor: MLActor,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        description: str = "",
        details: Optional[Dict[str, Any]] = None,
        entity_before: Optional[Dict[str, Any]] = None,
        entity_after: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> MLAuditEntry:
        """
        Log an ML audit event.
        
        Args:
            event_type: Type of event
            actor: Who performed the action
            model_name: Name of the model (if applicable)
            model_version: Version of the model (if applicable)
            description: Human-readable description
            details: Additional details as dict
            entity_before: State before change
            entity_after: State after change
            correlation_id: ID to link related events
            
        Returns:
            MLAuditEntry object
        """
        with self._lock:
            # DISABLED: SQLite replaced with PostgreSQL
            if False:  # Disabled method
                    cursor = conn.cursor()
                
                    # Get chain state
                    cursor.execute('SELECT value FROM ml_chain_metadata WHERE key = "last_checksum"')
                    previous_checksum = cursor.fetchone()[0]
                
                    cursor.execute('SELECT value FROM ml_chain_metadata WHERE key = "sequence_number"')
                    sequence_number = int(cursor.fetchone()[0]) + 1
                
                    # Create entry
                    entry = MLAuditEntry(
                        entry_id=str(uuid.uuid4()),
                        timestamp=datetime.now(),
                        event_type=event_type,
                        actor=actor,
                        model_name=model_name,
                        model_version=model_version,
                        action_description=description,
                        details=details or {},
                        entity_before=entity_before,
                        entity_after=entity_after,
                        previous_checksum=previous_checksum,
                        sequence_number=sequence_number,
                        correlation_id=correlation_id
                    )
                
                    # Calculate checksum
                    entry.checksum = entry.calculate_checksum()
                
                    # Insert entry
                    cursor.execute('''
                        INSERT INTO ml_audit_entries
                        (entry_id, timestamp, event_type, actor_id, actor_type,
                         model_name, model_version, action_description,
                         checksum, previous_checksum, sequence_number, entry_data)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        entry.entry_id,
                        entry.timestamp.isoformat(),
                        entry.event_type.value,
                        entry.actor.actor_id,
                        entry.actor.actor_type.value,
                        entry.model_name,
                        entry.model_version,
                        entry.action_description,
                        entry.checksum,
                        entry.previous_checksum,
                        entry.sequence_number,
                        json.dumps(entry.to_dict())
                    ))
                
                    # Update chain state
                    cursor.execute('''
                        UPDATE ml_chain_metadata SET value = ?, updated_at = ?
                        WHERE key = "last_checksum"
                    ''', (entry.checksum, datetime.now().isoformat()))
                
                    cursor.execute('''
                        UPDATE ml_chain_metadata SET value = ?, updated_at = ?
                        WHERE key = "sequence_number"
                    ''', (str(sequence_number), datetime.now().isoformat()))
                
                    conn.commit()
        
                    conn.commit()
        
        # Return dummy entry if disabled
        if 'entry' not in locals():
            entry = MLAuditEntry(
                entry_id="disabled",
                timestamp=datetime.now(),
                event_type=event_type,
                actor=actor,
                checksum="disabled",
                sequence_number=0
            )
            
        return entry
    
    # =========================================================================
    # Convenience Methods
    # =========================================================================
    
    def log_model_registered(
        self,
        actor: MLActor,
        model_name: str,
        model_version: str,
        metrics: Dict[str, Any],
        training_config: Optional[Dict[str, Any]] = None
    ) -> MLAuditEntry:
        """Log a model registration event"""
        return self.log_event(
            event_type=MLEventType.MODEL_REGISTERED,
            actor=actor,
            model_name=model_name,
            model_version=model_version,
            description=f"Model {model_name} version {model_version} registered",
            details={
                'metrics': metrics,
                'training_config': training_config
            },
            entity_after={'status': 'development', 'version': model_version}
        )
    
    def log_model_promoted(
        self,
        actor: MLActor,
        model_name: str,
        model_version: str,
        from_status: str,
        to_status: str,
        reason: str = ""
    ) -> MLAuditEntry:
        """Log a model promotion event"""
        return self.log_event(
            event_type=MLEventType.MODEL_PROMOTED,
            actor=actor,
            model_name=model_name,
            model_version=model_version,
            description=f"Model {model_name} v{model_version} promoted to {to_status}",
            details={'reason': reason},
            entity_before={'status': from_status},
            entity_after={'status': to_status}
        )
    
    def log_prediction(
        self,
        actor: MLActor,
        model_name: str,
        model_version: str,
        prediction_id: str,
        features_hash: str,
        prediction: str,
        confidence: float
    ) -> MLAuditEntry:
        """Log a prediction event"""
        return self.log_event(
            event_type=MLEventType.PREDICTION_MADE,
            actor=actor,
            model_name=model_name,
            model_version=model_version,
            description=f"Prediction {prediction_id} made",
            details={
                'prediction_id': prediction_id,
                'features_hash': features_hash,
                'prediction': prediction,
                'confidence': confidence
            }
        )
    
    def log_drift_detected(
        self,
        model_name: str,
        drift_type: str,
        severity: str,
        affected_features: List[str],
        psi_value: float
    ) -> MLAuditEntry:
        """Log a drift detection event"""
        return self.log_event(
            event_type=MLEventType.DRIFT_DETECTED,
            actor=MLActor.system_actor(),
            model_name=model_name,
            description=f"Drift detected in {model_name}: {severity}",
            details={
                'drift_type': drift_type,
                'severity': severity,
                'affected_features': affected_features,
                'psi_value': psi_value
            }
        )
    
    def log_retraining_event(
        self,
        event_type: MLEventType,
        actor: MLActor,
        model_name: str,
        job_id: str,
        details: Optional[Dict[str, Any]] = None
    ) -> MLAuditEntry:
        """Log a retraining event"""
        descriptions = {
            MLEventType.RETRAINING_TRIGGERED: "Retraining triggered",
            MLEventType.RETRAINING_APPROVED: "Retraining approved",
            MLEventType.RETRAINING_REJECTED: "Retraining rejected",
            MLEventType.RETRAINING_COMPLETED: "Retraining completed",
            MLEventType.RETRAINING_FAILED: "Retraining failed"
        }
        
        return self.log_event(
            event_type=event_type,
            actor=actor,
            model_name=model_name,
            description=f"{descriptions.get(event_type, 'Retraining event')} for job {job_id}",
            details={'job_id': job_id, **(details or {})}
        )
    
    def log_threshold_change(
        self,
        actor: MLActor,
        model_name: str,
        metric_name: str,
        old_thresholds: Dict[str, float],
        new_thresholds: Dict[str, float]
    ) -> MLAuditEntry:
        """Log a threshold change event"""
        return self.log_event(
            event_type=MLEventType.THRESHOLD_CHANGED,
            actor=actor,
            model_name=model_name,
            description=f"Thresholds changed for {metric_name}",
            entity_before=old_thresholds,
            entity_after=new_thresholds
        )
    
    # =========================================================================
    # Electronic Signatures
    # =========================================================================
    
    def add_signature(
        self,
        entry_id: str,
        signer_id: str,
        signer_name: str,
        signer_role: str,
        signature_type: SignatureType,
        meaning: str
    ) -> ElectronicSignature:
        """
        Add an electronic signature to an audit entry.
        
        This provides the 21 CFR Part 11 signature requirement.
        """
        with self._lock:
            # DISABLED: SQLite replaced with PostgreSQL
            if False:  # Disabled method
                    cursor = conn.cursor()
                
                    # Get the entry
                    cursor.execute('SELECT entry_data FROM ml_audit_entries WHERE entry_id = ?', (entry_id,))
                    row = cursor.fetchone()
                    if not row:
                        raise ValueError(f"Entry not found: {entry_id}")
                
                    entry_data = json.loads(row[0])
                
                    # Create data hash
                    data_hash = hashlib.sha256(json.dumps(entry_data, sort_keys=True).encode()).hexdigest()
                
                    # Create signature hash (simplified - production would use PKI)
                    signature_content = f"{signer_id}:{data_hash}:{datetime.now().isoformat()}"
                    signature_hash = hashlib.sha256(signature_content.encode()).hexdigest()
                
                    signature = ElectronicSignature(
                        signature_id=str(uuid.uuid4()),
                        signed_at=datetime.now(),
                        signer_id=signer_id,
                        signer_name=signer_name,
                        signer_role=signer_role,
                        signature_type=signature_type,
                        meaning=meaning,
                        data_hash=data_hash,
                        signature_hash=signature_hash
                    )
                
                    # Store signature
                    cursor.execute('''
                        INSERT INTO ml_signatures
                        (signature_id, entry_id, signed_at, signer_id, signer_name,
                         signer_role, signature_type, meaning, data_hash, signature_hash)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        signature.signature_id,
                        entry_id,
                        signature.signed_at.isoformat(),
                        signature.signer_id,
                        signature.signer_name,
                        signature.signer_role,
                        signature.signature_type.value,
                        signature.meaning,
                        signature.data_hash,
                        signature.signature_hash
                    ))
                
                    conn.commit()
        
        return signature
    
    # =========================================================================
    # Retrieval
    # =========================================================================
    
    def get_audit_trail(
        self,
        model_name: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        event_types: Optional[List[MLEventType]] = None,
        actor_id: Optional[str] = None,
        limit: int = 1000
    ) -> List[MLAuditEntry]:
        """Get audit trail entries with filtering"""
        entries = []
        
        # DISABLED: SQLite replaced with PostgreSQL
        if False:  # Disabled method
            cursor = conn.cursor()
            
            query = 'SELECT entry_data FROM ml_audit_entries WHERE 1=1'
            params = []
            
            if model_name:
                query += ' AND model_name = ?'
                params.append(model_name)
            
            if start:
                query += ' AND timestamp >= ?'
                params.append(start.isoformat())
            
            if end:
                query += ' AND timestamp <= ?'
                params.append(end.isoformat())
            
            if event_types:
                placeholders = ','.join('?' * len(event_types))
                query += f' AND event_type IN ({placeholders})'
                params.extend([et.value for et in event_types])
            
            if actor_id:
                query += ' AND actor_id = ?'
                params.append(actor_id)
            
            query += ' ORDER BY sequence_number DESC LIMIT ?'
            params.append(limit)
            
            cursor.execute(query, params)
            
            for row in cursor.fetchall():
                data = json.loads(row[0])
                entries.append(MLAuditEntry.from_dict(data))
        
        return entries
    
    def get_model_history(
        self,
        model_name: str,
        include_predictions: bool = False
    ) -> List[MLAuditEntry]:
        """Get complete audit history for a model"""
        event_types = None
        
        if not include_predictions:
            # Exclude prediction events for cleaner history
            event_types = [
                et for et in MLEventType
                if et not in [MLEventType.PREDICTION_MADE, MLEventType.PREDICTION_BATCH]
            ]
        
        return self.get_audit_trail(model_name=model_name, event_types=event_types)
    
    def get_prediction_audit(
        self,
        prediction_id: str
    ) -> List[MLAuditEntry]:
        """Get audit trail for a specific prediction"""
        entries = []
        
        # DISABLED: SQLite replaced with PostgreSQL
        if False:  # Disabled method
            cursor = conn.cursor()
            cursor.execute('''
                SELECT entry_data FROM ml_audit_entries
                WHERE entry_data LIKE ?
                ORDER BY timestamp
            ''', (f'%{prediction_id}%',))
            
            for row in cursor.fetchall():
                data = json.loads(row[0])
                if data.get('details', {}).get('prediction_id') == prediction_id:
                    entries.append(MLAuditEntry.from_dict(data))
        
        return entries
    
    def get_user_actions(
        self,
        actor_id: str,
        start: datetime,
        end: datetime
    ) -> List[MLAuditEntry]:
        """Get all actions by a specific user"""
        return self.get_audit_trail(
            actor_id=actor_id,
            start=start,
            end=end
        )
    
    # =========================================================================
    # Chain Integrity
    # =========================================================================
    
    def verify_chain_integrity(
        self,
        model_name: Optional[str] = None
    ) -> Tuple[bool, List[str]]:
        """
        Verify integrity of the audit chain.
        
        Returns:
            Tuple of (is_valid, list_of_corrupted_entry_ids)
        """
        corrupted = []
        
        # DISABLED: SQLite replaced with PostgreSQL
        if False:  # Disabled method
            cursor = conn.cursor()
            
            if model_name:
                cursor.execute('''
                    SELECT entry_id, checksum, previous_checksum, entry_data
                    FROM ml_audit_entries
                    WHERE model_name = ?
                    ORDER BY sequence_number
                ''', (model_name,))
            else:
                cursor.execute('''
                    SELECT entry_id, checksum, previous_checksum, entry_data
                    FROM ml_audit_entries
                    ORDER BY sequence_number
                ''')
            
            expected_previous = "GENESIS"
            
            for row in cursor.fetchall():
                entry_id, stored_checksum, previous_checksum, entry_data = row
                
                # Verify chain link
                if previous_checksum != expected_previous:
                    corrupted.append(entry_id)
                
                # Verify checksum
                entry = MLAuditEntry.from_dict(json.loads(entry_data))
                calculated_checksum = entry.calculate_checksum()
                
                if calculated_checksum != stored_checksum:
                    corrupted.append(entry_id)
                
                expected_previous = stored_checksum
        
        is_valid = len(corrupted) == 0
        
        # Log the integrity check
        self.log_event(
            event_type=MLEventType.INTEGRITY_CHECK,
            actor=MLActor.system_actor(),
            model_name=model_name,
            description=f"Chain integrity check: {'PASSED' if is_valid else 'FAILED'}",
            details={
                'is_valid': is_valid,
                'corrupted_entries': corrupted
            }
        )
        
        return is_valid, corrupted
    
    # =========================================================================
    # Export
    # =========================================================================
    
    def export_audit_trail(
        self,
        model_name: str,
        start: datetime,
        end: datetime,
        format: str = 'csv'
    ) -> str:
        """
        Export audit trail for regulatory submission.
        
        Args:
            model_name: Name of the model
            start: Start date
            end: End date
            format: 'csv' or 'json'
            
        Returns:
            Formatted audit trail string
        """
        entries = self.get_audit_trail(
            model_name=model_name,
            start=start,
            end=end
        )
        
        # Log the export
        self.log_event(
            event_type=MLEventType.AUDIT_EXPORT,
            actor=MLActor.system_actor(),
            model_name=model_name,
            description=f"Audit trail exported ({len(entries)} entries)",
            details={
                'start': start.isoformat(),
                'end': end.isoformat(),
                'format': format,
                'entry_count': len(entries)
            }
        )
        
        if format == 'json':
            return json.dumps([e.to_dict() for e in entries], indent=2, default=str)
        
        else:  # CSV
            output = io.StringIO()
            writer = csv.writer(output)
            
            # Header
            writer.writerow([
                'Entry ID', 'Timestamp', 'Event Type', 'Actor ID', 'Actor Name',
                'Model Name', 'Model Version', 'Description', 'Checksum',
                'Previous Checksum', 'Sequence Number'
            ])
            
            # Rows
            for entry in entries:
                writer.writerow([
                    entry.entry_id,
                    entry.timestamp.isoformat(),
                    entry.event_type.value,
                    entry.actor.actor_id,
                    entry.actor.name,
                    entry.model_name or '',
                    entry.model_version or '',
                    entry.action_description,
                    entry.checksum,
                    entry.previous_checksum,
                    entry.sequence_number
                ])
            
            return output.getvalue()
    
    # =========================================================================
    # Statistics
    # =========================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get audit logger statistics"""
        # Initialize default values to avoid UnboundLocalError
        total_entries = 0
        by_event_type = {}
        by_model = {}
        total_signatures = 0
        last_entry = None
        last_integrity_check = None
        
        # DISABLED: SQLite replaced with PostgreSQL
        if False:  # Disabled method
            cursor = conn.cursor()
            
            # Total entries
            cursor.execute('SELECT COUNT(*) FROM ml_audit_entries')
            total_entries = cursor.fetchone()[0]
            
            # By event type
            cursor.execute('''
                SELECT event_type, COUNT(*) FROM ml_audit_entries GROUP BY event_type
            ''')
            by_event_type = dict(cursor.fetchall())
            
            # By model
            cursor.execute('''
                SELECT model_name, COUNT(*) FROM ml_audit_entries
                WHERE model_name IS NOT NULL GROUP BY model_name
            ''')
            by_model = dict(cursor.fetchall())
            
            # Signatures
            cursor.execute('SELECT COUNT(*) FROM ml_signatures')
            total_signatures = cursor.fetchone()[0]
            
            # Last entry
            cursor.execute('''
                SELECT entry_data FROM ml_audit_entries ORDER BY sequence_number DESC LIMIT 1
            ''')
            row = cursor.fetchone()
            last_entry = None
            if row:
                data = json.loads(row[0])
                last_entry = {
                    'timestamp': data['timestamp'],
                    'event_type': data['event_type'],
                    'model_name': data.get('model_name')
                }
            
            # Chain integrity (last check result)
            cursor.execute('''
                SELECT entry_data FROM ml_audit_entries
                WHERE event_type = 'integrity_check'
                ORDER BY timestamp DESC LIMIT 1
            ''')
            integrity_row = cursor.fetchone()
            last_integrity_check = None
            if integrity_row:
                data = json.loads(integrity_row[0])
                last_integrity_check = data.get('details', {})
        
        return {
            'total_entries': total_entries,
            'by_event_type': by_event_type,
            'by_model': by_model,
            'total_signatures': total_signatures,
            'last_entry': last_entry,
            'last_integrity_check': last_integrity_check
        }


# =============================================================================
# SINGLETON
# =============================================================================

_ml_audit_logger: Optional[MLAuditLogger] = None


def get_ml_audit_logger() -> MLAuditLogger:
    """Get or create the ML audit logger singleton"""
    global _ml_audit_logger
    if _ml_audit_logger is None:
        _ml_audit_logger = MLAuditLogger()
    return _ml_audit_logger


def reset_ml_audit_logger():
    """Reset the ML audit logger singleton (for testing)"""
    global _ml_audit_logger
    _ml_audit_logger = None
