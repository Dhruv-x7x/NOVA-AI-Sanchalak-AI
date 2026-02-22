"""
a6on-i - Authentication Service
==========================================
Core authentication logic with 21 CFR Part 11 compliance.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple

from .models import User, Role, Session
from .jwt_handler import get_jwt_handler
from .password import get_password_handler, PasswordPolicy

logger = logging.getLogger(__name__)


class AuthService:
    """
    Authentication service for a6on-i.
    
    Features:
    - Secure login with account lockout
    - JWT token management
    - Session tracking
    - Password management
    - 21 CFR Part 11 compliant logging
    """
    
    MAX_FAILED_ATTEMPTS = 5
    LOCKOUT_DURATION = timedelta(minutes=30)
    
    def __init__(self):
        self.jwt_handler = get_jwt_handler()
        self.password_handler = get_password_handler()
        
        # In-memory user store (replace with PostgreSQL in production)
        self._users: Dict[int, User] = {}
        self._users_by_username: Dict[str, User] = {}
        self._sessions: Dict[str, Session] = {}
        
        # Create default users
        self._create_default_users()
        
        logger.info("AuthService initialized")
    
    def _create_default_users(self):
        """Create default system users."""
        default_users = [
            ("admin", "Admin@a6on-i2024!", Role.ADMIN, "System Administrator"),
            ("study_lead", "StudyLead@2024!", Role.STUDY_LEAD, "Dr. Sarah Chen"),
            ("cra", "CRA@Monitor2024!", Role.CRA, "Michael Thompson"),
            ("data_manager", "DataMgr@2024!", Role.DATA_MANAGER, "Emily Rodriguez"),
            ("viewer", "Viewer@Read2024!", Role.VIEWER, "Demo User"),
        ]
        
        for i, (username, password, role, full_name) in enumerate(default_users, start=1):
            password_hash = self.password_handler.hash_password(password)
            user = User(
                id=i,
                username=username,
                email=f"{username}@a6on-i.com",
                password_hash=password_hash,
                role=role,
                full_name=full_name,
                is_active=True,
            )
            self._users[user.id] = user
            self._users_by_username[user.username] = user
        
        logger.info(f"Created {len(default_users)} default users")
    
    def authenticate(self, username: str, password: str, 
                    ip_address: str = None) -> Tuple[bool, Optional[Dict], str]:
        """
        Authenticate a user with username and password.
        
        Args:
            username: User's username
            password: User's password
            ip_address: Client IP for audit logging
            
        Returns:
            Tuple of (success, token_dict, message)
        """
        # Find user
        user = self._users_by_username.get(username)
        
        if user is None:
            logger.warning(f"Login attempt for non-existent user: {username}")
            return False, None, "Invalid username or password"
        
        # Check if account is locked
        if user.is_locked:
            remaining = (user.locked_until - datetime.now()).total_seconds() / 60
            logger.warning(f"Login attempt for locked account: {username}")
            return False, None, f"Account is locked. Try again in {remaining:.0f} minutes."
        
        # Check if account is active
        if not user.is_active:
            logger.warning(f"Login attempt for inactive account: {username}")
            return False, None, "Account is deactivated. Contact administrator."
        
        # Verify password
        if not self.password_handler.verify_password(password, user.password_hash):
            # Increment failed attempts
            user.failed_login_attempts += 1
            
            if user.failed_login_attempts >= self.MAX_FAILED_ATTEMPTS:
                user.locked_until = datetime.now() + self.LOCKOUT_DURATION
                logger.warning(f"Account locked after {self.MAX_FAILED_ATTEMPTS} failed attempts: {username}")
                return False, None, "Account locked due to too many failed attempts."
            
            remaining = self.MAX_FAILED_ATTEMPTS - user.failed_login_attempts
            return False, None, f"Invalid password. {remaining} attempts remaining."
        
        # Successful login - reset failed attempts
        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login = datetime.now()
        
        # Check password expiry
        is_expired, days_remaining = PasswordPolicy.check_expiry(user.password_changed_at)
        
        # Create tokens
        tokens = self.jwt_handler.create_token_pair(
            user_id=user.id,
            username=user.username,
            role=user.role.value
        )
        
        # Add user info to response
        tokens["user"] = user.to_session_dict()
        
        if is_expired:
            tokens["password_expired"] = True
            tokens["message"] = "Password has expired. Please change your password."
        elif days_remaining <= 14:
            tokens["password_warning"] = True
            tokens["message"] = f"Password expires in {days_remaining} days."
        
        logger.info(f"Successful login: {username} from {ip_address}")
        
        return True, tokens, "Login successful"
    
    def verify_session(self, access_token: str) -> Tuple[bool, Optional[User]]:
        """
        Verify an access token and return the user.
        
        Returns:
            Tuple of (is_valid, user)
        """
        is_valid, claims = self.jwt_handler.verify_access_token(access_token)
        
        if not is_valid or claims is None:
            return False, None
        
        user_id = int(claims.get("sub", 0))
        user = self._users.get(user_id)
        
        if user is None or not user.is_active:
            return False, None
        
        return True, user
    
    def refresh_session(self, refresh_token: str) -> Tuple[bool, Optional[Dict]]:
        """
        Refresh an expired access token.
        
        Returns:
            Tuple of (success, new_tokens)
        """
        is_valid, claims = self.jwt_handler.verify_refresh_token(refresh_token)
        
        if not is_valid or claims is None:
            return False, None
        
        user_id = int(claims.get("sub", 0))
        user = self._users.get(user_id)
        
        if user is None or not user.is_active:
            return False, None
        
        # Create new access token
        new_access = self.jwt_handler.create_access_token(
            user_id=user.id,
            username=user.username,
            role=user.role.value
        )
        
        return True, {
            "access_token": new_access,
            "token_type": "bearer",
            "user": user.to_session_dict()
        }
    
    def logout(self, access_token: str) -> bool:
        """
        Logout user and invalidate token.
        
        Returns:
            True if successful
        """
        self.jwt_handler.revoke_token(access_token)
        logger.info("User logged out, token revoked")
        return True
    
    def change_password(self, user_id: int, old_password: str, 
                       new_password: str) -> Tuple[bool, str]:
        """
        Change a user's password.
        
        Returns:
            Tuple of (success, message)
        """
        user = self._users.get(user_id)
        
        if user is None:
            return False, "User not found"
        
        # Verify old password
        if not self.password_handler.verify_password(old_password, user.password_hash):
            return False, "Current password is incorrect"
        
        # Validate and hash new password
        is_valid, new_hash, violations = self.password_handler.validate_and_hash(
            new_password, user_id
        )
        
        if not is_valid:
            return False, violations[0] if violations else "Invalid password"
        
        # Update password
        self.password_handler.add_to_history(user_id, user.password_hash)
        user.password_hash = new_hash
        user.password_changed_at = datetime.now()
        
        logger.info(f"Password changed for user: {user.username}")
        
        return True, "Password changed successfully"
    
    def get_user(self, user_id: int) -> Optional[User]:
        """Get user by ID."""
        return self._users.get(user_id)
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        return self._users_by_username.get(username)
    
    def list_users(self) -> list:
        """List all users."""
        return [u.to_dict() for u in self._users.values()]


# Singleton instance
_auth_service: Optional[AuthService] = None


def get_auth_service() -> AuthService:
    """Get singleton auth service."""
    global _auth_service
    if _auth_service is None:
        _auth_service = AuthService()
    return _auth_service
