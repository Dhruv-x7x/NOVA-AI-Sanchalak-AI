"""
SANCHALAK AI - JWT Token Handler
=====================================
Secure JWT token creation and validation for authentication.
"""

import os
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Try to import jwt library
try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    logger.warning("PyJWT not installed. Using fallback token handler.")


class JWTHandler:
    """
    JWT token handler for secure authentication.
    
    Features:
    - Access tokens (short-lived)
    - Refresh tokens (longer-lived)
    - Token blacklisting
    - Secure key management
    """
    
    def __init__(self):
        # Secret key from environment or generate secure default
        self.secret_key = os.getenv("JWT_SECRET_KEY", self._generate_default_key())
        self.algorithm = os.getenv("JWT_ALGORITHM", "HS256")
        
        # Token expiry settings
        self.access_token_expiry = timedelta(
            minutes=int(os.getenv("JWT_ACCESS_EXPIRY_MINUTES", "30"))
        )
        self.refresh_token_expiry = timedelta(
            days=int(os.getenv("JWT_REFRESH_EXPIRY_DAYS", "7"))
        )
        
        # Blacklisted tokens (should use Redis in production)
        self._blacklist: set = set()
        
        logger.info("JWTHandler initialized")
    
    def _generate_default_key(self) -> str:
        """Generate a default secret key (not for production!)."""
        import uuid
        key = f"sanchalak-ai-{uuid.uuid4().hex}"
        logger.warning("Using auto-generated JWT secret. Set JWT_SECRET_KEY for production!")
        return key
    
    def create_access_token(self, user_id: int, username: str, role: str, 
                           additional_claims: Dict = None) -> str:
        """
        Create a short-lived access token.
        
        Args:
            user_id: User's unique ID
            username: User's username
            role: User's role
            additional_claims: Extra data to include in token
            
        Returns:
            Encoded JWT token string
        """
        now = datetime.utcnow()
        
        payload = {
            "sub": str(user_id),
            "username": username,
            "role": role,
            "type": "access",
            "iat": now,
            "exp": now + self.access_token_expiry,
            "jti": self._generate_token_id(),
        }
        
        if additional_claims:
            payload.update(additional_claims)
        
        if JWT_AVAILABLE:
            return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        else:
            return self._fallback_encode(payload)
    
    def create_refresh_token(self, user_id: int) -> str:
        """Create a longer-lived refresh token."""
        now = datetime.utcnow()
        
        payload = {
            "sub": str(user_id),
            "type": "refresh",
            "iat": now,
            "exp": now + self.refresh_token_expiry,
            "jti": self._generate_token_id(),
        }
        
        if JWT_AVAILABLE:
            return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        else:
            return self._fallback_encode(payload)
    
    def create_token_pair(self, user_id: int, username: str, role: str) -> Dict[str, str]:
        """Create both access and refresh tokens."""
        return {
            "access_token": self.create_access_token(user_id, username, role),
            "refresh_token": self.create_refresh_token(user_id),
            "token_type": "bearer",
            "expires_in": int(self.access_token_expiry.total_seconds()),
        }
    
    def verify_token(self, token: str) -> Tuple[bool, Optional[Dict]]:
        """
        Verify a token and return its claims.
        
        Returns:
            Tuple of (is_valid, claims_dict)
        """
        if not token:
            return False, None
        
        # Check blacklist
        if token in self._blacklist:
            logger.warning("Token is blacklisted")
            return False, None
        
        try:
            if JWT_AVAILABLE:
                claims = jwt.decode(
                    token, 
                    self.secret_key, 
                    algorithms=[self.algorithm]
                )
            else:
                claims = self._fallback_decode(token)
            
            return True, claims
            
        except jwt.ExpiredSignatureError if JWT_AVAILABLE else Exception:
            logger.debug("Token expired")
            return False, None
        except jwt.InvalidTokenError if JWT_AVAILABLE else Exception as e:
            logger.warning(f"Invalid token: {e}")
            return False, None
    
    def verify_access_token(self, token: str) -> Tuple[bool, Optional[Dict]]:
        """Verify an access token specifically."""
        is_valid, claims = self.verify_token(token)
        
        if is_valid and claims.get("type") != "access":
            return False, None
        
        return is_valid, claims
    
    def verify_refresh_token(self, token: str) -> Tuple[bool, Optional[Dict]]:
        """Verify a refresh token specifically."""
        is_valid, claims = self.verify_token(token)
        
        if is_valid and claims.get("type") != "refresh":
            return False, None
        
        return is_valid, claims
    
    def refresh_access_token(self, refresh_token: str, username: str, role: str) -> Optional[str]:
        """Use a refresh token to get a new access token."""
        is_valid, claims = self.verify_refresh_token(refresh_token)
        
        if not is_valid:
            return None
        
        user_id = int(claims["sub"])
        return self.create_access_token(user_id, username, role)
    
    def revoke_token(self, token: str):
        """Add token to blacklist."""
        self._blacklist.add(token)
        logger.info("Token revoked")
    
    def _generate_token_id(self) -> str:
        """Generate unique token ID."""
        import uuid
        return uuid.uuid4().hex
    
    def _fallback_encode(self, payload: Dict) -> str:
        """Fallback encoding when PyJWT not available."""
        import json
        import base64
        
        # Simple base64 encoding (NOT SECURE - for development only)
        payload_json = json.dumps(payload, default=str)
        encoded = base64.urlsafe_b64encode(payload_json.encode()).decode()
        
        # Add a simple signature
        signature = hashlib.sha256(
            (encoded + self.secret_key).encode()
        ).hexdigest()[:16]
        
        return f"{encoded}.{signature}"
    
    def _fallback_decode(self, token: str) -> Dict:
        """Fallback decoding when PyJWT not available."""
        import json
        import base64
        
        parts = token.split(".")
        if len(parts) != 2:
            raise ValueError("Invalid token format")
        
        encoded, signature = parts
        
        # Verify signature
        expected_sig = hashlib.sha256(
            (encoded + self.secret_key).encode()
        ).hexdigest()[:16]
        
        if signature != expected_sig:
            raise ValueError("Invalid signature")
        
        # Decode payload
        payload_json = base64.urlsafe_b64decode(encoded.encode()).decode()
        payload = json.loads(payload_json)
        
        # Check expiry
        if "exp" in payload:
            exp_time = datetime.fromisoformat(payload["exp"])
            if datetime.utcnow() > exp_time:
                raise ValueError("Token expired")
        
        return payload


# Singleton instance
_jwt_handler: Optional[JWTHandler] = None


def get_jwt_handler() -> JWTHandler:
    """Get singleton JWT handler."""
    global _jwt_handler
    if _jwt_handler is None:
        _jwt_handler = JWTHandler()
    return _jwt_handler
