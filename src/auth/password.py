"""
a6on-i - Password Handler
====================================
Secure password hashing and policy validation for 21 CFR Part 11 compliance.
"""

import re
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Tuple, List

logger = logging.getLogger(__name__)

# Try to import bcrypt, fall back to hashlib
try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False
    logger.warning("bcrypt not installed. Using PBKDF2 fallback.")


class PasswordPolicy:
    """
    Password policy for 21 CFR Part 11 compliance.
    
    Requirements:
    - Minimum 12 characters
    - At least 1 uppercase letter
    - At least 1 lowercase letter
    - At least 1 digit
    - At least 1 special character
    - Cannot be same as last 5 passwords
    - Must be changed every 90 days
    """
    
    MIN_LENGTH = 12
    MAX_LENGTH = 128
    REQUIRE_UPPERCASE = True
    REQUIRE_LOWERCASE = True
    REQUIRE_DIGIT = True
    REQUIRE_SPECIAL = True
    HISTORY_COUNT = 5
    MAX_AGE_DAYS = 90
    
    SPECIAL_CHARS = "!@#$%^&*()_+-=[]{}|;':\",./<>?"
    
    @classmethod
    def validate(cls, password: str) -> Tuple[bool, List[str]]:
        """
        Validate password against policy.
        
        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        violations = []
        
        if len(password) < cls.MIN_LENGTH:
            violations.append(f"Password must be at least {cls.MIN_LENGTH} characters")
        
        if len(password) > cls.MAX_LENGTH:
            violations.append(f"Password cannot exceed {cls.MAX_LENGTH} characters")
        
        if cls.REQUIRE_UPPERCASE and not re.search(r'[A-Z]', password):
            violations.append("Password must contain at least one uppercase letter")
        
        if cls.REQUIRE_LOWERCASE and not re.search(r'[a-z]', password):
            violations.append("Password must contain at least one lowercase letter")
        
        if cls.REQUIRE_DIGIT and not re.search(r'\d', password):
            violations.append("Password must contain at least one digit")
        
        if cls.REQUIRE_SPECIAL and not any(c in cls.SPECIAL_CHARS for c in password):
            violations.append(f"Password must contain at least one special character ({cls.SPECIAL_CHARS[:10]}...)")
        
        # Check for common patterns
        if password.lower() in ['password', 'password123', 'a6on-i', 'nexus123']:
            violations.append("Password is too common")
        
        return len(violations) == 0, violations
    
    @classmethod
    def check_expiry(cls, password_changed_at: datetime) -> Tuple[bool, int]:
        """
        Check if password has expired.
        
        Returns:
            Tuple of (is_expired, days_until_expiry)
        """
        if password_changed_at is None:
            return True, 0
        
        expiry_date = password_changed_at + timedelta(days=cls.MAX_AGE_DAYS)
        days_remaining = (expiry_date - datetime.now()).days
        
        return days_remaining <= 0, days_remaining


class PasswordHandler:
    """Secure password hashing and verification."""
    
    def __init__(self):
        self.rounds = 12  # bcrypt work factor
        self._password_history: dict = {}  # user_id -> list of hashes
    
    def hash_password(self, password: str) -> str:
        """
        Hash a password securely.
        
        Returns:
            Hashed password string
        """
        if BCRYPT_AVAILABLE:
            salt = bcrypt.gensalt(rounds=self.rounds)
            hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
            return hashed.decode('utf-8')
        else:
            # PBKDF2 fallback
            import secrets
            salt = secrets.token_hex(16)
            hashed = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt.encode('utf-8'),
                100000  # iterations
            )
            return f"pbkdf2:{salt}:{hashed.hex()}"
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """
        Verify a password against its hash.
        
        Returns:
            True if password matches
        """
        try:
            if BCRYPT_AVAILABLE and not hashed.startswith('pbkdf2:'):
                return bcrypt.checkpw(
                    password.encode('utf-8'),
                    hashed.encode('utf-8')
                )
            elif hashed.startswith('pbkdf2:'):
                # PBKDF2 verification
                _, salt, stored_hash = hashed.split(':')
                new_hash = hashlib.pbkdf2_hmac(
                    'sha256',
                    password.encode('utf-8'),
                    salt.encode('utf-8'),
                    100000
                )
                return new_hash.hex() == stored_hash
            else:
                return False
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False
    
    def check_password_history(self, user_id: int, password: str) -> bool:
        """
        Check if password was recently used.
        
        Returns:
            True if password is NOT in history (safe to use)
        """
        history = self._password_history.get(user_id, [])
        
        for old_hash in history:
            if self.verify_password(password, old_hash):
                return False
        
        return True
    
    def add_to_history(self, user_id: int, password_hash: str):
        """Add a password hash to user's history."""
        if user_id not in self._password_history:
            self._password_history[user_id] = []
        
        history = self._password_history[user_id]
        history.append(password_hash)
        
        # Keep only last N passwords
        if len(history) > PasswordPolicy.HISTORY_COUNT:
            self._password_history[user_id] = history[-PasswordPolicy.HISTORY_COUNT:]
    
    def validate_and_hash(self, password: str, user_id: int = None) -> Tuple[bool, str, List[str]]:
        """
        Validate password policy and hash if valid.
        
        Returns:
            Tuple of (is_valid, hash_or_empty, violations)
        """
        is_valid, violations = PasswordPolicy.validate(password)
        
        if not is_valid:
            return False, "", violations
        
        # Check history if user_id provided
        if user_id and not self.check_password_history(user_id, password):
            return False, "", [f"Cannot reuse one of your last {PasswordPolicy.HISTORY_COUNT} passwords"]
        
        password_hash = self.hash_password(password)
        
        return True, password_hash, []


# Singleton instance
_password_handler: PasswordHandler = None


def get_password_handler() -> PasswordHandler:
    """Get singleton password handler."""
    global _password_handler
    if _password_handler is None:
        _password_handler = PasswordHandler()
    return _password_handler
