"""
a6on-i - Authentication Module
=========================================
21 CFR Part 11 compliant authentication with JWT and RBAC.
"""

from .authentication import AuthService, get_auth_service
from .jwt_handler import JWTHandler, get_jwt_handler
from .models import User, Role, AuditLogEntry
from .authorization import RBACAuthorizer, require_role, require_permission
from .audit import AuditLogger, get_audit_logger

__all__ = [
    'AuthService',
    'get_auth_service',
    'JWTHandler',
    'get_jwt_handler',
    'User',
    'Role',
    'AuditLogEntry',
    'RBACAuthorizer',
    'require_role',
    'require_permission',
    'AuditLogger',
    'get_audit_logger',
]
