"""
a6on-i - Streamlit Auth Middleware
=============================================
Middleware for protecting Streamlit pages with authentication.
"""

import logging
from typing import Optional, Callable, List
import streamlit as st

from .authentication import get_auth_service, AuthService
from .authorization import get_rbac_authorizer, RBACAuthorizer
from .models import User, Role
from .audit import get_audit_logger

logger = logging.getLogger(__name__)


def init_auth_session():
    """Initialize authentication in Streamlit session state."""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'user' not in st.session_state:
        st.session_state.user = None
    if 'access_token' not in st.session_state:
        st.session_state.access_token = None
    if 'refresh_token' not in st.session_state:
        st.session_state.refresh_token = None


def get_current_user() -> Optional[User]:
    """Get the current authenticated user."""
    init_auth_session()
    
    if not st.session_state.authenticated:
        return None
    
    # Verify token is still valid
    auth = get_auth_service()
    is_valid, user = auth.verify_session(st.session_state.access_token)
    
    if not is_valid:
        # Try to refresh
        if st.session_state.refresh_token:
            success, new_tokens = auth.refresh_session(st.session_state.refresh_token)
            if success:
                st.session_state.access_token = new_tokens['access_token']
                return auth.get_user(int(new_tokens['user']['id']))
        
        # Clear session
        logout()
        return None
    
    return user


def login(username: str, password: str) -> tuple:
    """
    Attempt to login user.
    
    Returns:
        Tuple of (success, message)
    """
    init_auth_session()
    auth = get_auth_service()
    
    success, tokens, message = auth.authenticate(username, password)
    
    if success and tokens:
        st.session_state.authenticated = True
        st.session_state.access_token = tokens['access_token']
        st.session_state.refresh_token = tokens['refresh_token']
        st.session_state.user = tokens['user']
        
        # Log login
        audit = get_audit_logger()
        user = auth.get_user(tokens['user']['id'])
        if user:
            audit.log_login(user)
        
        return True, message
    
    return False, message


def logout():
    """Logout current user."""
    init_auth_session()
    
    if st.session_state.access_token:
        auth = get_auth_service()
        auth.logout(st.session_state.access_token)
        
        # Log logout
        if st.session_state.user:
            audit = get_audit_logger()
            user = auth.get_user(st.session_state.user['id'])
            if user:
                audit.log_logout(user)
    
    st.session_state.authenticated = False
    st.session_state.user = None
    st.session_state.access_token = None
    st.session_state.refresh_token = None


def require_auth(func: Callable = None, roles: List[Role] = None, 
                permission: str = None):
    """
    Decorator/function to require authentication for a page.
    
    Usage as decorator:
        @require_auth
        def my_page():
            ...
        
        @require_auth(roles=[Role.ADMIN])
        def admin_page():
            ...
    
    Usage as guard:
        if not require_auth():
            return
    """
    def wrapper(page_func: Callable = None):
        def inner():
            init_auth_session()
            
            if not st.session_state.authenticated:
                show_login_page()
                return
            
            user = get_current_user()
            if user is None:
                show_login_page()
                return
            
            # Check roles if specified
            if roles:
                rbac = get_rbac_authorizer()
                if not rbac.check_role(user, roles):
                    show_access_denied(f"Required roles: {[r.value for r in roles]}")
                    return
            
            # Check permission if specified
            if permission:
                rbac = get_rbac_authorizer()
                if not rbac.check_permission(user, permission):
                    show_access_denied(f"Required permission: {permission}")
                    return
            
            # Call the page function
            if page_func:
                return page_func()
        
        return inner
    
    # If called without arguments
    if func is not None:
        return wrapper(func)
    
    # If called with arguments
    return wrapper


def show_login_page():
    """Display login form."""
    st.markdown("""
    <style>
    .login-container {
        max-width: 400px;
        margin: 100px auto;
        padding: 40px;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.95), rgba(15, 23, 42, 0.98));
        border-radius: 16px;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    .login-title {
        text-align: center;
        color: white;
        font-size: 28px;
        margin-bottom: 30px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("## 🔐 a6on-i Login")
        st.markdown("---")
        
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            submitted = st.form_submit_button("Login", use_container_width=True)
            
            if submitted:
                if username and password:
                    success, message = login(username, password)
                    
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
                else:
                    st.warning("Please enter both username and password")
        
        st.markdown("---")
        st.markdown("""
        **Demo Accounts:**
        - `admin` / `Admin@a6on-i2024!`
        - `study_lead` / `StudyLead@2024!`
        - `cra` / `CRA@Monitor2024!`
        """)


def show_access_denied(reason: str = None):
    """Display access denied message."""
    st.error("🚫 Access Denied")
    
    if reason:
        st.warning(f"Reason: {reason}")
    
    st.info("Please contact your administrator if you need access to this page.")
    
    if st.button("Go to Home"):
        st.switch_page("pages/executive_overview.py")


def show_user_menu():
    """Display user menu in sidebar."""
    init_auth_session()
    
    if st.session_state.authenticated and st.session_state.user:
        user = st.session_state.user
        
        with st.sidebar:
            st.markdown("---")
            st.markdown(f"**👤 {user.get('full_name', user.get('username'))}**")
            st.caption(f"Role: {user.get('role', 'Unknown').title()}")
            
            if st.button("🚪 Logout", key="logout_btn"):
                logout()
                st.rerun()
