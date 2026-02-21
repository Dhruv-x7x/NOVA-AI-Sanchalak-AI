"""
Security Module - JWT Authentication & Password Hashing
"""
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
import hashlib
import hmac
import os
from fastapi import HTTPException, status, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.config import settings

# Bearer token security - auto_error=False allows us to handle missing tokens gracefully
security = HTTPBearer(auto_error=False)

# Simple password hashing using SHA256 with salt (avoiding bcrypt compatibility issues)
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    # For demo purposes, using simple comparison
    # In production, use proper hashing
    return get_password_hash(plain_password) == hashed_password


def get_password_hash(password: str) -> str:
    """Generate password hash using SHA256"""
    # Simple hash for demo - in production use bcrypt with proper version
    salt = "sanchalak_ai_salt_2024"
    return hashlib.sha256(f"{salt}{password}".encode()).hexdigest()


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire, "type": "access"})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def create_refresh_token(data: dict) -> str:
    """Create JWT refresh token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def decode_token(token: str) -> Dict[str, Any]:
    """Decode and validate JWT token"""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )


# Default anonymous user for test mode
ANONYMOUS_USER = {
    "user_id": "anonymous",
    "username": "anonymous",
    "role": "lead",
    "email": "anonymous@test.com",
    "full_name": "Anonymous Test User",
}


def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Dict[str, Any]:
    """Get current user from JWT token - ALWAYS allows access in test mode"""
    
    # HYPER-PERMISSIVE TEST MODE: Always allow access
    if settings.TEST_MODE:
        # If credentials provided, try to decode them
        if credentials and credentials.credentials:
            try:
                payload = decode_token(credentials.credentials)
                return {
                    "user_id": payload.get("sub", "test-user"),
                    "username": payload.get("username", "testuser"),
                    "role": payload.get("role", "lead"),
                    "email": payload.get("email", "test@test.com"),
                    "full_name": payload.get("full_name", "Test User"),
                }
            except:
                pass  # Fall through to anonymous user
        
        # Return anonymous user for any unauthenticated request in test mode
        return ANONYMOUS_USER
    
    # PRODUCTION MODE: Require valid authentication
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = credentials.credentials
    
    try:
        payload = decode_token(token)
    except HTTPException:
        raise
    
    if payload.get("type") != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type",
        )
    
    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
        )
    
    return {
        "user_id": user_id,
        "username": payload.get("username"),
        "role": payload.get("role"),
        "email": payload.get("email"),
        "full_name": payload.get("full_name"),
    }


def require_role(*allowed_roles: str):
    """Dependency to require specific roles"""
    def role_checker(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        user_role = current_user.get("role", "")
        if user_role not in allowed_roles:
            # In test mode, allow any role
            if settings.TEST_MODE:
                return current_user
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{user_role}' not authorized. Required: {allowed_roles}"
            )
        return current_user
    return role_checker


# Role constants
ROLE_LEAD = "lead"
ROLE_DM = "dm"
ROLE_CRA = "cra"
ROLE_CODER = "coder"
ROLE_SAFETY = "safety"
ROLE_EXECUTIVE = "executive"

ALL_ROLES = [ROLE_LEAD, ROLE_DM, ROLE_CRA, ROLE_CODER, ROLE_SAFETY, ROLE_EXECUTIVE]
