"""
SANCHALAK AI - RBAC Authorization
======================================
Role-based access control with permission checking.
"""

import logging
from functools import wraps
from typing import Callable, List, Optional, Set

from .models import User, Role, PERMISSIONS, has_permission

logger = logging.getLogger(__name__)


class RBACAuthorizer:
    """
    Role-Based Access Control authorizer.
    
    Provides:
    - Permission checking
    - Role hierarchy validation
    - Study-level access control
    """
    
    def __init__(self):
        self.permissions = PERMISSIONS
    
    def check_permission(self, user: User, permission: str) -> bool:
        """
        Check if user has a specific permission.
        
        Args:
            user: The user to check
            permission: Permission name (e.g., 'view_patients')
            
        Returns:
            True if permitted
        """
        if user is None or not user.is_active:
            return False
        
        return has_permission(user, permission)
    
    def check_role(self, user: User, required_roles: List[Role]) -> bool:
        """
        Check if user has one of the required roles.
        
        Args:
            user: The user to check
            required_roles: List of allowed roles
            
        Returns:
            True if user has one of the required roles
        """
        if user is None or not user.is_active:
            return False
        
        return user.role in required_roles
    
    def check_role_level(self, user: User, minimum_level: int) -> bool:
        """
        Check if user's role level meets minimum.
        
        Args:
            user: The user to check
            minimum_level: Minimum role level required
            
        Returns:
            True if user's role level >= minimum
        """
        if user is None or not user.is_active:
            return False
        
        return user.role.level >= minimum_level
    
    def check_study_access(self, user: User, study_id: str) -> bool:
        """
        Check if user has access to a specific study.
        
        Args:
            user: The user to check
            study_id: Study identifier
            
        Returns:
            True if user has access to the study
        """
        if user is None or not user.is_active:
            return False
        
        # Admins have access to all studies
        if user.role == Role.ADMIN:
            return True
        
        # Check specific study access
        if not user.study_access:
            return True  # No restrictions = all studies
        
        return study_id in user.study_access
    
    def get_user_permissions(self, user: User) -> Set[str]:
        """
        Get all permissions for a user.
        
        Returns:
            Set of permission names
        """
        if user is None or not user.is_active:
            return set()
        
        return {
            perm for perm, roles in self.permissions.items()
            if user.role in roles
        }
    
    def get_accessible_pages(self, user: User) -> List[str]:
        """
        Get list of dashboard pages user can access.
        
        Returns:
            List of page identifiers
        """
        if user is None or not user.is_active:
            return []
        
        permissions = self.get_user_permissions(user)
        
        # Map permissions to pages
        page_permissions = {
            "executive_overview": "view_dashboard",
            "study_lead": "approve_data",
            "site_analysis": "view_patients",
            "digital_twin": "view_dashboard",
            "ai_assistant": "ai_assistant",
            "ml_governance": "view_ml",
            "collaboration_hub": "view_dashboard",
            "audit_log": "view_audit",
        }
        
        return [
            page for page, required_perm in page_permissions.items()
            if required_perm in permissions
        ]


# Decorator functions for easy permission checking

def require_role(*roles: Role):
    """
    Decorator to require specific roles.
    
    Usage:
        @require_role(Role.ADMIN, Role.STUDY_LEAD)
        def protected_function(user, ...):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Find user in args or kwargs
            user = kwargs.get('user') or (args[0] if args else None)
            
            if not isinstance(user, User):
                raise PermissionError("User not provided")
            
            if user.role not in roles:
                raise PermissionError(
                    f"Role '{user.role.value}' not permitted. Required: {[r.value for r in roles]}"
                )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def require_permission(permission: str):
    """
    Decorator to require specific permission.
    
    Usage:
        @require_permission("approve_data")
        def protected_function(user, ...):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Find user in args or kwargs
            user = kwargs.get('user') or (args[0] if args else None)
            
            if not isinstance(user, User):
                raise PermissionError("User not provided")
            
            if not has_permission(user, permission):
                raise PermissionError(
                    f"Permission '{permission}' required. User role: {user.role.value}"
                )
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


# Singleton instance
_rbac: Optional[RBACAuthorizer] = None


def get_rbac_authorizer() -> RBACAuthorizer:
    """Get singleton RBAC authorizer."""
    global _rbac
    if _rbac is None:
        _rbac = RBACAuthorizer()
    return _rbac
