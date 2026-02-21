"""
SANCHALAK AI - User & Role Models
======================================
Database models for authentication with 21 CFR Part 11 compliance.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class Role(Enum):
    """User roles with hierarchical permissions."""
    ADMIN = "admin"                    # Full system access
    STUDY_LEAD = "study_lead"          # Study management, approvals
    CRA = "cra"                        # Site monitoring, data review
    DATA_MANAGER = "data_manager"      # Query resolution, data entry
    SAFETY_OFFICER = "safety_officer"  # SAE review, safety reports
    VIEWER = "viewer"                  # Read-only access
    
    @property
    def level(self) -> int:
        """Role hierarchy level (higher = more permissions)."""
        levels = {
            Role.ADMIN: 100,
            Role.STUDY_LEAD: 80,
            Role.SAFETY_OFFICER: 70,
            Role.CRA: 60,
            Role.DATA_MANAGER: 50,
            Role.VIEWER: 10,
        }
        return levels.get(self, 0)


@dataclass
class User:
    """User entity for authentication."""
    id: int
    username: str
    email: str
    password_hash: str
    role: Role
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    password_changed_at: datetime = field(default_factory=datetime.now)
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None
    
    # 21 CFR Part 11 fields
    signature_meaning: Optional[str] = None  # e.g., "Reviewed", "Approved"
    electronic_signature_enabled: bool = False
    
    # Profile
    full_name: Optional[str] = None
    department: Optional[str] = None
    study_access: List[str] = field(default_factory=list)  # Allowed study IDs
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'role': self.role.value,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'full_name': self.full_name,
            'department': self.department,
            'study_access': self.study_access,
        }
    
    def to_session_dict(self) -> Dict[str, Any]:
        """Minimal data for session storage."""
        return {
            'id': self.id,
            'username': self.username,
            'role': self.role.value,
            'full_name': self.full_name,
        }
    
    @property
    def is_locked(self) -> bool:
        if self.locked_until is None:
            return False
        return datetime.now() < self.locked_until


@dataclass
class AuditLogEntry:
    """21 CFR Part 11 compliant audit log entry."""
    id: Optional[int]
    timestamp: datetime
    user_id: int
    username: str
    action: str                        # e.g., "login", "patient_update", "approve_data"
    resource: Optional[str] = None     # e.g., "patient:STUDY001|SITE001|SUBJ001"
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    # 21 CFR Part 11 electronic signature
    signature_hash: Optional[str] = None
    signature_meaning: Optional[str] = None
    
    # Integrity
    previous_hash: Optional[str] = None
    entry_hash: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'user_id': self.user_id,
            'username': self.username,
            'action': self.action,
            'resource': self.resource,
            'details': self.details,
            'ip_address': self.ip_address,
            'signature_meaning': self.signature_meaning,
        }
    
    @classmethod
    def create(cls, user: User, action: str, resource: str = None, 
               details: Dict = None, ip_address: str = None) -> 'AuditLogEntry':
        """Factory method to create audit entry."""
        return cls(
            id=None,
            timestamp=datetime.now(),
            user_id=user.id,
            username=user.username,
            action=action,
            resource=resource,
            details=details or {},
            ip_address=ip_address,
        )


@dataclass 
class Session:
    """User session for tracking active logins."""
    session_id: str
    user_id: int
    created_at: datetime
    expires_at: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    is_valid: bool = True
    
    @property
    def is_expired(self) -> bool:
        return datetime.now() > self.expires_at


# Permission definitions
PERMISSIONS = {
    "view_dashboard": [Role.VIEWER, Role.DATA_MANAGER, Role.CRA, Role.SAFETY_OFFICER, Role.STUDY_LEAD, Role.ADMIN],
    "view_patients": [Role.DATA_MANAGER, Role.CRA, Role.SAFETY_OFFICER, Role.STUDY_LEAD, Role.ADMIN],
    "edit_patients": [Role.DATA_MANAGER, Role.CRA, Role.STUDY_LEAD, Role.ADMIN],
    "resolve_queries": [Role.DATA_MANAGER, Role.CRA, Role.ADMIN],
    "approve_data": [Role.STUDY_LEAD, Role.ADMIN],
    "view_safety": [Role.SAFETY_OFFICER, Role.STUDY_LEAD, Role.ADMIN],
    "manage_sae": [Role.SAFETY_OFFICER, Role.ADMIN],
    "view_ml": [Role.DATA_MANAGER, Role.STUDY_LEAD, Role.ADMIN],
    "manage_users": [Role.ADMIN],
    "view_audit": [Role.STUDY_LEAD, Role.ADMIN],
    "export_data": [Role.STUDY_LEAD, Role.ADMIN],
    "db_lock": [Role.STUDY_LEAD, Role.ADMIN],
    "ai_assistant": [Role.CRA, Role.DATA_MANAGER, Role.STUDY_LEAD, Role.ADMIN],
}


def has_permission(user: User, permission: str) -> bool:
    """Check if user has a specific permission."""
    allowed_roles = PERMISSIONS.get(permission, [])
    return user.role in allowed_roles
