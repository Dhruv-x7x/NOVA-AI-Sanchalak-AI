"""
a6on-i - Phase 11.2 Error Handling System v1.0
# NOTE: This module previously used SQLite but has been migrated to PostgreSQL-only.
# If you need to use this module, update it to use:
#   from src.database.pg_data_service import get_pg_data_service
#   service = get_pg_data_service()


Comprehensive error handling with:
- Graceful degradation
- Fallback modes
- Error reporting
- Recovery procedures
- Circuit breaker pattern
- Error aggregation and analysis
"""

import os
import sys
import json
# SQLite removed - using PostgreSQL
import sqlite3  # kept for type annotations only
import hashlib
import traceback
import threading
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from enum import Enum
from functools import wraps
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# ENUMS
# =============================================================================

class ErrorSeverity(Enum):
    """Error severity levels"""
    CRITICAL = "critical"      # System cannot continue
    HIGH = "high"              # Major functionality impaired
    MEDIUM = "medium"          # Some functionality impaired
    LOW = "low"                # Minor issue, workaround available
    WARNING = "warning"        # Potential issue, no immediate impact


class ErrorCategory(Enum):
    """Error categories for classification"""
    DATA = "data"                      # Data loading/processing errors
    DATABASE = "database"              # Database connection/query errors
    ML_MODEL = "ml_model"              # ML model errors
    LLM = "llm"                        # LLM/agent errors
    NETWORK = "network"                # Network/API errors
    VALIDATION = "validation"          # Validation errors
    CONFIGURATION = "configuration"    # Configuration errors
    RESOURCE = "resource"              # Resource exhaustion
    PERMISSION = "permission"          # Permission/access errors
    TIMEOUT = "timeout"                # Timeout errors
    DEPENDENCY = "dependency"          # External dependency errors
    UNKNOWN = "unknown"                # Unclassified errors


class RecoveryAction(Enum):
    """Recovery action types"""
    RETRY = "retry"                    # Retry the operation
    FALLBACK = "fallback"              # Use fallback mechanism
    SKIP = "skip"                      # Skip and continue
    DEGRADE = "degrade"                # Graceful degradation
    CIRCUIT_BREAK = "circuit_break"    # Open circuit breaker
    ESCALATE = "escalate"              # Escalate to human
    RESTART = "restart"                # Restart component
    ABORT = "abort"                    # Abort operation


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"          # Normal operation
    OPEN = "open"              # Failing, reject requests
    HALF_OPEN = "half_open"    # Testing if recovered


class FallbackMode(Enum):
    """Fallback modes for degraded operation"""
    FULL = "full"              # Full functionality
    CACHED = "cached"          # Use cached data only
    STATIC = "static"          # Use static/default values
    MINIMAL = "minimal"        # Minimal functionality
    READONLY = "readonly"      # Read-only mode
    OFFLINE = "offline"        # Offline mode
    DISABLED = "disabled"      # Feature disabled


class ComponentStatus(Enum):
    """Component health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"
    OFFLINE = "offline"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ErrorContext:
    """Context information for an error"""
    component: str
    operation: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    entity_type: Optional[str] = None
    entity_id: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ErrorRecord:
    """Complete error record"""
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    context: ErrorContext
    exception_type: Optional[str] = None
    exception_message: Optional[str] = None
    recovery_action: Optional[RecoveryAction] = None
    recovery_result: Optional[str] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        result = asdict(self)
        result['category'] = self.category.value
        result['severity'] = self.severity.value
        result['recovery_action'] = self.recovery_action.value if self.recovery_action else None
        result['context'] = asdict(self.context)
        result['context']['timestamp'] = self.context.timestamp.isoformat()
        result['created_at'] = self.created_at.isoformat()
        result['resolved_at'] = self.resolved_at.isoformat() if self.resolved_at else None
        return result


@dataclass
class FallbackConfig:
    """Configuration for fallback behavior"""
    component: str
    fallback_mode: FallbackMode
    cache_ttl_seconds: int = 3600
    static_value: Any = None
    fallback_function: Optional[Callable] = None
    enabled: bool = True


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    component: str
    failure_threshold: int = 5          # Failures before opening
    success_threshold: int = 3          # Successes to close
    timeout_seconds: int = 60           # Time before half-open
    half_open_max_calls: int = 3        # Max calls in half-open


@dataclass
class CircuitBreakerState:
    """Current state of a circuit breaker"""
    component: str
    state: CircuitState
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    last_success_time: Optional[datetime] = None
    opened_at: Optional[datetime] = None
    half_open_calls: int = 0


@dataclass
class ComponentHealth:
    """Health status of a component"""
    component: str
    status: ComponentStatus
    fallback_mode: FallbackMode
    circuit_state: CircuitState
    error_rate: float  # Errors per minute
    last_error: Optional[datetime] = None
    last_success: Optional[datetime] = None
    message: Optional[str] = None


@dataclass
class RecoveryProcedure:
    """Recovery procedure definition"""
    procedure_id: str
    name: str
    description: str
    category: ErrorCategory
    severity: ErrorSeverity
    steps: List[Dict[str, Any]]
    auto_execute: bool = False
    requires_approval: bool = False
    estimated_duration_seconds: int = 60


# =============================================================================
# DATABASE MANAGER
# =============================================================================

class ErrorDatabaseManager:
    """Thread-safe database manager for error handling"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._write_lock = threading.Lock()
        self._init_database()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection"""
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            self._local.connection = None  # DB persistence disabled
            pass  # PRAGMA removed
        return self._local.connection
    
    def _init_database(self):
        """Initialize database tables"""
        conn = self._get_connection()
        
        # Error records table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS error_records (
                error_id TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                severity TEXT NOT NULL,
                message TEXT NOT NULL,
                exception_type TEXT,
                exception_message TEXT,
                component TEXT,
                operation TEXT,
                user_id TEXT,
                session_id TEXT,
                entity_type TEXT,
                entity_id TEXT,
                parameters TEXT,
                stack_trace TEXT,
                recovery_action TEXT,
                recovery_result TEXT,
                resolved INTEGER DEFAULT 0,
                resolved_at TEXT,
                resolved_by TEXT,
                created_at TEXT NOT NULL
            )
        """)
        
        # Circuit breaker state table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS circuit_breaker_state (
                component TEXT PRIMARY KEY,
                state TEXT NOT NULL,
                failure_count INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                last_failure_time TEXT,
                last_success_time TEXT,
                opened_at TEXT,
                half_open_calls INTEGER DEFAULT 0,
                updated_at TEXT
            )
        """)
        
        # Fallback cache table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS fallback_cache (
                cache_key TEXT PRIMARY KEY,
                component TEXT NOT NULL,
                data TEXT,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL
            )
        """)
        
        # Recovery procedures table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS recovery_procedures (
                procedure_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                category TEXT,
                severity TEXT,
                steps TEXT,
                auto_execute INTEGER DEFAULT 0,
                requires_approval INTEGER DEFAULT 0,
                estimated_duration_seconds INTEGER DEFAULT 60,
                created_at TEXT
            )
        """)
        
        # Error aggregation table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS error_aggregation (
                aggregation_id TEXT PRIMARY KEY,
                period_start TEXT NOT NULL,
                period_end TEXT NOT NULL,
                component TEXT,
                category TEXT,
                severity TEXT,
                error_count INTEGER DEFAULT 0,
                unique_errors INTEGER DEFAULT 0,
                created_at TEXT
            )
        """)
        
        conn.commit()
    
    def execute_write(self, sql: str, params: tuple = ()) -> bool:
        """Execute write operation with locking"""
        with self._write_lock:
            try:
                conn = self._get_connection()
                conn.execute(sql, params)
                conn.commit()
                return True
            except Exception as e:
                logger.error(f"Database write error: {e}")
                return False
    
    def execute_read(self, sql: str, params: tuple = ()) -> List[sqlite3.Row]:
        """Execute read operation"""
        try:
            conn = self._get_connection()
            cursor = conn.execute(sql, params)
            return cursor.fetchall()
        except Exception as e:
            logger.error(f"Database read error: {e}")
            return []
    
    def execute_read_one(self, sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
        """Execute read operation returning single row"""
        try:
            conn = self._get_connection()
            cursor = conn.execute(sql, params)
            return cursor.fetchone()
        except Exception as e:
            logger.error(f"Database read error: {e}")
            return None


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================

class CircuitBreaker:
    """Circuit breaker implementation"""
    
    def __init__(self, db_manager: ErrorDatabaseManager):
        self.db = db_manager
        self._configs: Dict[str, CircuitBreakerConfig] = {}
        self._states: Dict[str, CircuitBreakerState] = {}
        self._lock = threading.Lock()
    
    def configure(self, config: CircuitBreakerConfig):
        """Configure circuit breaker for a component"""
        with self._lock:
            self._configs[config.component] = config
            if config.component not in self._states:
                self._states[config.component] = CircuitBreakerState(
                    component=config.component,
                    state=CircuitState.CLOSED
                )
                self._save_state(config.component)
    
    def _get_config(self, component: str) -> CircuitBreakerConfig:
        """Get config or return default"""
        return self._configs.get(component, CircuitBreakerConfig(component=component))
    
    def _get_state(self, component: str) -> CircuitBreakerState:
        """Get current state for component"""
        if component not in self._states:
            # Try to load from database
            row = self.db.execute_read_one(
                "SELECT * FROM circuit_breaker_state WHERE component = ?",
                (component,)
            )
            if row:
                self._states[component] = CircuitBreakerState(
                    component=component,
                    state=CircuitState(row['state']),
                    failure_count=row['failure_count'] or 0,
                    success_count=row['success_count'] or 0,
                    last_failure_time=datetime.fromisoformat(row['last_failure_time']) if row['last_failure_time'] else None,
                    last_success_time=datetime.fromisoformat(row['last_success_time']) if row['last_success_time'] else None,
                    opened_at=datetime.fromisoformat(row['opened_at']) if row['opened_at'] else None,
                    half_open_calls=row['half_open_calls'] or 0
                )
            else:
                self._states[component] = CircuitBreakerState(
                    component=component,
                    state=CircuitState.CLOSED
                )
        return self._states[component]
    
    def _save_state(self, component: str):
        """Save circuit breaker state to database"""
        state = self._states.get(component)
        if not state:
            return
        
        self.db.execute_write("""
            INSERT OR REPLACE INTO circuit_breaker_state 
            (component, state, failure_count, success_count, last_failure_time,
             last_success_time, opened_at, half_open_calls, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            component,
            state.state.value,
            state.failure_count,
            state.success_count,
            state.last_failure_time.isoformat() if state.last_failure_time else None,
            state.last_success_time.isoformat() if state.last_success_time else None,
            state.opened_at.isoformat() if state.opened_at else None,
            state.half_open_calls,
            datetime.now().isoformat()
        ))
    
    def can_execute(self, component: str) -> Tuple[bool, str]:
        """Check if component can execute (circuit allows)"""
        with self._lock:
            state = self._get_state(component)
            config = self._get_config(component)
            
            if state.state == CircuitState.CLOSED:
                return True, "Circuit closed - normal operation"
            
            elif state.state == CircuitState.OPEN:
                # Check if timeout has passed
                if state.opened_at:
                    elapsed = (datetime.now() - state.opened_at).total_seconds()
                    if elapsed >= config.timeout_seconds:
                        # Transition to half-open
                        state.state = CircuitState.HALF_OPEN
                        state.half_open_calls = 0
                        self._save_state(component)
                        return True, "Circuit half-open - testing recovery"
                return False, f"Circuit open - rejecting requests (timeout: {config.timeout_seconds}s)"
            
            else:  # HALF_OPEN
                if state.half_open_calls < config.half_open_max_calls:
                    state.half_open_calls += 1
                    self._save_state(component)
                    return True, f"Circuit half-open - test call {state.half_open_calls}/{config.half_open_max_calls}"
                return False, "Circuit half-open - max test calls reached"
    
    def record_success(self, component: str):
        """Record successful execution"""
        with self._lock:
            state = self._get_state(component)
            config = self._get_config(component)
            
            state.success_count += 1
            state.last_success_time = datetime.now()
            
            if state.state == CircuitState.HALF_OPEN:
                if state.success_count >= config.success_threshold:
                    # Transition to closed
                    state.state = CircuitState.CLOSED
                    state.failure_count = 0
                    state.success_count = 0
                    state.opened_at = None
                    logger.info(f"Circuit breaker CLOSED for {component}")
            
            self._save_state(component)
    
    def record_failure(self, component: str):
        """Record failed execution"""
        with self._lock:
            state = self._get_state(component)
            config = self._get_config(component)
            
            state.failure_count += 1
            state.last_failure_time = datetime.now()
            
            if state.state == CircuitState.CLOSED:
                if state.failure_count >= config.failure_threshold:
                    # Transition to open
                    state.state = CircuitState.OPEN
                    state.opened_at = datetime.now()
                    logger.warning(f"Circuit breaker OPEN for {component}")
            
            elif state.state == CircuitState.HALF_OPEN:
                # Any failure in half-open reopens circuit
                state.state = CircuitState.OPEN
                state.opened_at = datetime.now()
                state.success_count = 0
                logger.warning(f"Circuit breaker reopened for {component}")
            
            self._save_state(component)
    
    def get_state(self, component: str) -> CircuitBreakerState:
        """Get current circuit breaker state"""
        with self._lock:
            return self._get_state(component)
    
    def reset(self, component: str):
        """Manually reset circuit breaker to closed"""
        with self._lock:
            state = self._get_state(component)
            state.state = CircuitState.CLOSED
            state.failure_count = 0
            state.success_count = 0
            state.opened_at = None
            state.half_open_calls = 0
            self._save_state(component)
            logger.info(f"Circuit breaker manually reset for {component}")


# =============================================================================
# FALLBACK MANAGER
# =============================================================================

class FallbackManager:
    """Manages fallback behavior for degraded operation"""
    
    def __init__(self, db_manager: ErrorDatabaseManager):
        self.db = db_manager
        self._configs: Dict[str, FallbackConfig] = {}
        self._current_modes: Dict[str, FallbackMode] = {}
        self._lock = threading.Lock()
    
    def configure(self, config: FallbackConfig):
        """Configure fallback for a component"""
        with self._lock:
            self._configs[config.component] = config
            if config.component not in self._current_modes:
                self._current_modes[config.component] = FallbackMode.FULL
    
    def set_fallback_mode(self, component: str, mode: FallbackMode):
        """Set fallback mode for a component"""
        with self._lock:
            self._current_modes[component] = mode
            logger.info(f"Fallback mode set to {mode.value} for {component}")
    
    def get_fallback_mode(self, component: str) -> FallbackMode:
        """Get current fallback mode"""
        return self._current_modes.get(component, FallbackMode.FULL)
    
    def cache_data(self, component: str, cache_key: str, data: Any):
        """Cache data for fallback use"""
        config = self._configs.get(component)
        ttl = config.cache_ttl_seconds if config else 3600
        expires_at = datetime.now() + timedelta(seconds=ttl)
        
        self.db.execute_write("""
            INSERT OR REPLACE INTO fallback_cache 
            (cache_key, component, data, created_at, expires_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            cache_key,
            component,
            json.dumps(data) if not isinstance(data, str) else data,
            datetime.now().isoformat(),
            expires_at.isoformat()
        ))
    
    def get_cached_data(self, component: str, cache_key: str) -> Optional[Any]:
        """Get cached data for fallback"""
        row = self.db.execute_read_one("""
            SELECT data, expires_at FROM fallback_cache 
            WHERE cache_key = ? AND component = ?
        """, (cache_key, component))
        
        if row:
            expires_at = datetime.fromisoformat(row['expires_at'])
            if expires_at > datetime.now():
                try:
                    return json.loads(row['data'])
                except:
                    return row['data']
        return None
    
    def get_fallback_value(self, component: str, cache_key: str, 
                          default: Any = None) -> Tuple[Any, str]:
        """Get fallback value based on current mode"""
        mode = self.get_fallback_mode(component)
        config = self._configs.get(component)
        
        if mode == FallbackMode.FULL:
            return None, "Full mode - no fallback needed"
        
        elif mode == FallbackMode.CACHED:
            cached = self.get_cached_data(component, cache_key)
            if cached is not None:
                return cached, "Using cached data"
            return default, "Cache miss - using default"
        
        elif mode == FallbackMode.STATIC:
            if config and config.static_value is not None:
                return config.static_value, "Using static value"
            return default, "No static value - using default"
        
        elif mode == FallbackMode.MINIMAL:
            return default, "Minimal mode - using default"
        
        elif mode == FallbackMode.READONLY:
            return None, "Read-only mode - writes disabled"
        
        elif mode == FallbackMode.OFFLINE:
            cached = self.get_cached_data(component, cache_key)
            if cached is not None:
                return cached, "Offline mode - using cached data"
            return default, "Offline mode - using default"
        
        else:  # DISABLED
            return None, "Feature disabled"
    
    def execute_with_fallback(self, component: str, cache_key: str,
                             primary_func: Callable, 
                             fallback_func: Optional[Callable] = None,
                             default: Any = None,
                             **kwargs) -> Tuple[Any, bool, str]:
        """Execute with automatic fallback on failure"""
        mode = self.get_fallback_mode(component)
        
        # If disabled, don't even try
        if mode == FallbackMode.DISABLED:
            return None, False, "Feature disabled"
        
        # Try primary function
        try:
            result = primary_func(**kwargs)
            # Cache successful result
            if mode != FallbackMode.READONLY:
                self.cache_data(component, cache_key, result)
            return result, True, "Primary function succeeded"
        
        except Exception as e:
            logger.warning(f"Primary function failed for {component}: {e}")
            
            # Try fallback function if provided
            if fallback_func:
                try:
                    result = fallback_func(**kwargs)
                    return result, True, "Fallback function succeeded"
                except Exception as e2:
                    logger.warning(f"Fallback function failed for {component}: {e2}")
            
            # Use cached or default value
            value, message = self.get_fallback_value(component, cache_key, default)
            return value, value is not None, message


# =============================================================================
# ERROR HANDLER
# =============================================================================

class ErrorHandler:
    """Central error handling and reporting"""
    
    def __init__(self, db_manager: ErrorDatabaseManager,
                 circuit_breaker: CircuitBreaker,
                 fallback_manager: FallbackManager):
        self.db = db_manager
        self.circuit_breaker = circuit_breaker
        self.fallback_manager = fallback_manager
        self._error_counts: Dict[str, int] = {}
        self._lock = threading.Lock()
    
    def _generate_error_id(self) -> str:
        """Generate unique error ID"""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
        hash_suffix = hashlib.md5(f"{timestamp}{threading.current_thread().ident}".encode()).hexdigest()[:8]
        return f"ERR-{timestamp[:14]}-{hash_suffix}"
    
    def _classify_error(self, exception: Exception, context: ErrorContext) -> Tuple[ErrorCategory, ErrorSeverity]:
        """Classify error category and severity"""
        exception_type = type(exception).__name__
        message = str(exception).lower()
        
        # Category classification
        category = ErrorCategory.UNKNOWN
        severity = ErrorSeverity.MEDIUM
        
        # Database errors
        if 'sqlite' in exception_type.lower() or 'database' in message or 'sql' in message:
            category = ErrorCategory.DATABASE
            severity = ErrorSeverity.HIGH
        
        # Network errors
        elif 'connection' in message or 'timeout' in message or 'network' in message:
            if 'timeout' in message:
                category = ErrorCategory.TIMEOUT
            else:
                category = ErrorCategory.NETWORK
            severity = ErrorSeverity.HIGH
        
        # File/Data errors
        elif 'file' in message or 'not found' in message or 'parquet' in message:
            category = ErrorCategory.DATA
            severity = ErrorSeverity.HIGH
        
        # Permission errors
        elif 'permission' in message or 'access' in message or 'denied' in message:
            category = ErrorCategory.PERMISSION
            severity = ErrorSeverity.HIGH
        
        # Validation errors
        elif 'validation' in message or 'invalid' in message or 'required' in message:
            category = ErrorCategory.VALIDATION
            severity = ErrorSeverity.LOW
        
        # Resource errors
        elif 'memory' in message or 'resource' in message or 'exhausted' in message:
            category = ErrorCategory.RESOURCE
            severity = ErrorSeverity.CRITICAL
        
        # Configuration errors
        elif 'config' in message or 'setting' in message:
            category = ErrorCategory.CONFIGURATION
            severity = ErrorSeverity.MEDIUM
        
        # ML model errors
        elif context.component and ('ml' in context.component.lower() or 
              'model' in context.component.lower() or 'predict' in context.operation.lower()):
            category = ErrorCategory.ML_MODEL
            severity = ErrorSeverity.HIGH
        
        # LLM/Agent errors
        elif context.component and ('agent' in context.component.lower() or 
              'llm' in context.component.lower()):
            category = ErrorCategory.LLM
            severity = ErrorSeverity.MEDIUM
        
        return category, severity
    
    def handle_error(self, exception: Exception, context: ErrorContext,
                    auto_recover: bool = True) -> ErrorRecord:
        """Handle an error with optional automatic recovery"""
        # Generate error ID
        error_id = self._generate_error_id()
        
        # Classify error
        category, severity = self._classify_error(exception, context)
        
        # Get stack trace
        stack_trace = traceback.format_exc()
        
        # Create error record
        error = ErrorRecord(
            error_id=error_id,
            category=category,
            severity=severity,
            message=str(exception),
            context=context,
            exception_type=type(exception).__name__,
            exception_message=str(exception),
            created_at=datetime.now()
        )
        
        # Update error counts
        with self._lock:
            key = f"{context.component}:{category.value}"
            self._error_counts[key] = self._error_counts.get(key, 0) + 1
        
        # Record failure in circuit breaker
        self.circuit_breaker.record_failure(context.component)
        
        # Attempt recovery if enabled
        if auto_recover:
            recovery_action, recovery_result = self._attempt_recovery(error)
            error.recovery_action = recovery_action
            error.recovery_result = recovery_result
        
        # Save error to database
        self._save_error(error)
        
        # Log error
        log_level = {
            ErrorSeverity.CRITICAL: logging.CRITICAL,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.WARNING: logging.WARNING
        }.get(severity, logging.ERROR)
        
        logger.log(log_level, f"[{error_id}] {category.value.upper()}: {exception} in {context.component}.{context.operation}")
        
        return error
    
    def _attempt_recovery(self, error: ErrorRecord) -> Tuple[RecoveryAction, str]:
        """Attempt automatic recovery based on error type"""
        category = error.category
        severity = error.severity
        component = error.context.component
        
        # Check circuit breaker state
        cb_state = self.circuit_breaker.get_state(component)
        
        # Determine recovery action
        if severity == ErrorSeverity.CRITICAL:
            # Critical errors require escalation
            return RecoveryAction.ESCALATE, "Critical error - escalated for human review"
        
        if cb_state.state == CircuitState.OPEN:
            # Circuit is open - use fallback
            self.fallback_manager.set_fallback_mode(component, FallbackMode.CACHED)
            return RecoveryAction.FALLBACK, f"Circuit open - switched to cached mode"
        
        if category == ErrorCategory.TIMEOUT:
            return RecoveryAction.RETRY, "Timeout - will retry"
        
        if category == ErrorCategory.NETWORK:
            self.fallback_manager.set_fallback_mode(component, FallbackMode.OFFLINE)
            return RecoveryAction.FALLBACK, "Network error - switched to offline mode"
        
        if category == ErrorCategory.DATA:
            self.fallback_manager.set_fallback_mode(component, FallbackMode.CACHED)
            return RecoveryAction.FALLBACK, "Data error - using cached data"
        
        if category == ErrorCategory.RESOURCE:
            return RecoveryAction.DEGRADE, "Resource exhaustion - degrading functionality"
        
        if category == ErrorCategory.VALIDATION:
            return RecoveryAction.SKIP, "Validation error - skipping operation"
        
        # Default: try fallback
        return RecoveryAction.FALLBACK, "Using fallback mechanism"
    
    def _save_error(self, error: ErrorRecord):
        """Save error to database"""
        self.db.execute_write("""
            INSERT INTO error_records 
            (error_id, category, severity, message, exception_type, exception_message,
             component, operation, user_id, session_id, entity_type, entity_id,
             parameters, stack_trace, recovery_action, recovery_result, 
             resolved, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            error.error_id,
            error.category.value,
            error.severity.value,
            error.message,
            error.exception_type,
            error.exception_message,
            error.context.component,
            error.context.operation,
            error.context.user_id,
            error.context.session_id,
            error.context.entity_type,
            error.context.entity_id,
            json.dumps(error.context.parameters),
            error.context.stack_trace,
            error.recovery_action.value if error.recovery_action else None,
            error.recovery_result,
            0,
            error.created_at.isoformat()
        ))
    
    def resolve_error(self, error_id: str, resolved_by: str, 
                     resolution_notes: Optional[str] = None) -> bool:
        """Mark an error as resolved"""
        return self.db.execute_write("""
            UPDATE error_records 
            SET resolved = 1, resolved_at = ?, resolved_by = ?,
                recovery_result = COALESCE(recovery_result || ' | ', '') || ?
            WHERE error_id = ?
        """, (
            datetime.now().isoformat(),
            resolved_by,
            resolution_notes or "Manually resolved",
            error_id
        ))
    
    def get_error(self, error_id: str) -> Optional[ErrorRecord]:
        """Get error by ID"""
        row = self.db.execute_read_one(
            "SELECT * FROM error_records WHERE error_id = ?",
            (error_id,)
        )
        if row:
            return self._row_to_error(row)
        return None
    
    def _row_to_error(self, row: sqlite3.Row) -> ErrorRecord:
        """Convert database row to ErrorRecord"""
        context = ErrorContext(
            component=row['component'] or 'unknown',
            operation=row['operation'] or 'unknown',
            user_id=row['user_id'],
            session_id=row['session_id'],
            entity_type=row['entity_type'],
            entity_id=row['entity_id'],
            parameters=json.loads(row['parameters']) if row['parameters'] else {},
            stack_trace=row['stack_trace'],
            timestamp=datetime.fromisoformat(row['created_at'])
        )
        
        return ErrorRecord(
            error_id=row['error_id'],
            category=ErrorCategory(row['category']),
            severity=ErrorSeverity(row['severity']),
            message=row['message'],
            context=context,
            exception_type=row['exception_type'],
            exception_message=row['exception_message'],
            recovery_action=RecoveryAction(row['recovery_action']) if row['recovery_action'] else None,
            recovery_result=row['recovery_result'],
            resolved=bool(row['resolved']),
            resolved_at=datetime.fromisoformat(row['resolved_at']) if row['resolved_at'] else None,
            resolved_by=row['resolved_by'],
            created_at=datetime.fromisoformat(row['created_at'])
        )
    
    def get_recent_errors(self, limit: int = 50, 
                         category: Optional[ErrorCategory] = None,
                         severity: Optional[ErrorSeverity] = None,
                         component: Optional[str] = None,
                         unresolved_only: bool = False) -> List[ErrorRecord]:
        """Get recent errors with optional filters"""
        sql = "SELECT * FROM error_records WHERE 1=1"
        params = []
        
        if category:
            sql += " AND category = ?"
            params.append(category.value)
        
        if severity:
            sql += " AND severity = ?"
            params.append(severity.value)
        
        if component:
            sql += " AND component = ?"
            params.append(component)
        
        if unresolved_only:
            sql += " AND resolved = 0"
        
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        
        rows = self.db.execute_read(sql, tuple(params))
        return [self._row_to_error(row) for row in rows]
    
    def get_error_summary(self, hours: int = 24) -> Dict:
        """Get error summary for the last N hours"""
        since = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        # Total errors
        total = self.db.execute_read_one(
            "SELECT COUNT(*) as count FROM error_records WHERE created_at >= ?",
            (since,)
        )
        
        # By severity
        by_severity = self.db.execute_read(
            """SELECT severity, COUNT(*) as count 
               FROM error_records WHERE created_at >= ?
               GROUP BY severity""",
            (since,)
        )
        
        # By category
        by_category = self.db.execute_read(
            """SELECT category, COUNT(*) as count 
               FROM error_records WHERE created_at >= ?
               GROUP BY category""",
            (since,)
        )
        
        # By component
        by_component = self.db.execute_read(
            """SELECT component, COUNT(*) as count 
               FROM error_records WHERE created_at >= ?
               GROUP BY component ORDER BY count DESC LIMIT 10""",
            (since,)
        )
        
        # Unresolved count
        unresolved = self.db.execute_read_one(
            "SELECT COUNT(*) as count FROM error_records WHERE resolved = 0",
            ()
        )
        
        return {
            'period_hours': hours,
            'total_errors': total['count'] if total else 0,
            'by_severity': {row['severity']: row['count'] for row in by_severity},
            'by_category': {row['category']: row['count'] for row in by_category},
            'by_component': {row['component']: row['count'] for row in by_component},
            'unresolved': unresolved['count'] if unresolved else 0
        }


# =============================================================================
# RECOVERY MANAGER
# =============================================================================

class RecoveryManager:
    """Manages recovery procedures"""
    
    def __init__(self, db_manager: ErrorDatabaseManager,
                 error_handler: ErrorHandler,
                 circuit_breaker: CircuitBreaker,
                 fallback_manager: FallbackManager):
        self.db = db_manager
        self.error_handler = error_handler
        self.circuit_breaker = circuit_breaker
        self.fallback_manager = fallback_manager
        self._procedures: Dict[str, RecoveryProcedure] = {}
        self._init_default_procedures()
    
    def _init_default_procedures(self):
        """Initialize default recovery procedures"""
        procedures = [
            RecoveryProcedure(
                procedure_id="PROC-CIRCUIT-RESET",
                name="Reset Circuit Breaker",
                description="Manually reset circuit breaker to closed state",
                category=ErrorCategory.UNKNOWN,
                severity=ErrorSeverity.MEDIUM,
                steps=[
                    {"action": "reset_circuit_breaker", "component": "{component}"},
                    {"action": "verify_health", "component": "{component}"},
                    {"action": "log_recovery", "message": "Circuit breaker reset"}
                ],
                auto_execute=False,
                requires_approval=True
            ),
            RecoveryProcedure(
                procedure_id="PROC-CACHE-CLEAR",
                name="Clear Component Cache",
                description="Clear cached data for a component",
                category=ErrorCategory.DATA,
                severity=ErrorSeverity.LOW,
                steps=[
                    {"action": "clear_cache", "component": "{component}"},
                    {"action": "log_recovery", "message": "Cache cleared"}
                ],
                auto_execute=True,
                requires_approval=False
            ),
            RecoveryProcedure(
                procedure_id="PROC-FALLBACK-RESTORE",
                name="Restore Full Functionality",
                description="Restore component from fallback to full mode",
                category=ErrorCategory.UNKNOWN,
                severity=ErrorSeverity.MEDIUM,
                steps=[
                    {"action": "set_fallback_mode", "component": "{component}", "mode": "full"},
                    {"action": "verify_health", "component": "{component}"},
                    {"action": "log_recovery", "message": "Restored to full mode"}
                ],
                auto_execute=False,
                requires_approval=True
            ),
            RecoveryProcedure(
                procedure_id="PROC-DATA-RELOAD",
                name="Reload Data",
                description="Reload data from source files",
                category=ErrorCategory.DATA,
                severity=ErrorSeverity.HIGH,
                steps=[
                    {"action": "set_fallback_mode", "component": "{component}", "mode": "cached"},
                    {"action": "reload_data", "component": "{component}"},
                    {"action": "set_fallback_mode", "component": "{component}", "mode": "full"},
                    {"action": "log_recovery", "message": "Data reloaded"}
                ],
                auto_execute=False,
                requires_approval=True,
                estimated_duration_seconds=300
            ),
            RecoveryProcedure(
                procedure_id="PROC-GRACEFUL-DEGRADE",
                name="Graceful Degradation",
                description="Switch to degraded mode to maintain partial functionality",
                category=ErrorCategory.UNKNOWN,
                severity=ErrorSeverity.HIGH,
                steps=[
                    {"action": "set_fallback_mode", "component": "{component}", "mode": "minimal"},
                    {"action": "notify_users", "message": "Operating in degraded mode"},
                    {"action": "log_recovery", "message": "Switched to degraded mode"}
                ],
                auto_execute=True,
                requires_approval=False
            )
        ]
        
        for proc in procedures:
            self._procedures[proc.procedure_id] = proc
            self._save_procedure(proc)
    
    def _save_procedure(self, procedure: RecoveryProcedure):
        """Save procedure to database"""
        self.db.execute_write("""
            INSERT OR REPLACE INTO recovery_procedures
            (procedure_id, name, description, category, severity, steps,
             auto_execute, requires_approval, estimated_duration_seconds, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            procedure.procedure_id,
            procedure.name,
            procedure.description,
            procedure.category.value if procedure.category else None,
            procedure.severity.value if procedure.severity else None,
            json.dumps(procedure.steps),
            1 if procedure.auto_execute else 0,
            1 if procedure.requires_approval else 0,
            procedure.estimated_duration_seconds,
            datetime.now().isoformat()
        ))
    
    def get_procedure(self, procedure_id: str) -> Optional[RecoveryProcedure]:
        """Get recovery procedure by ID"""
        return self._procedures.get(procedure_id)
    
    def list_procedures(self) -> List[RecoveryProcedure]:
        """List all recovery procedures"""
        return list(self._procedures.values())
    
    def execute_procedure(self, procedure_id: str, component: str,
                         executed_by: str) -> Tuple[bool, str]:
        """Execute a recovery procedure"""
        procedure = self.get_procedure(procedure_id)
        if not procedure:
            return False, f"Procedure {procedure_id} not found"
        
        results = []
        for step in procedure.steps:
            action = step.get('action')
            
            try:
                if action == 'reset_circuit_breaker':
                    self.circuit_breaker.reset(component)
                    results.append(f"Circuit breaker reset for {component}")
                
                elif action == 'set_fallback_mode':
                    mode = FallbackMode(step.get('mode', 'full'))
                    self.fallback_manager.set_fallback_mode(component, mode)
                    results.append(f"Fallback mode set to {mode.value}")
                
                elif action == 'clear_cache':
                    self.db.execute_write(
                        "DELETE FROM fallback_cache WHERE component = ?",
                        (component,)
                    )
                    results.append(f"Cache cleared for {component}")
                
                elif action == 'verify_health':
                    # Placeholder - would check actual component health
                    results.append(f"Health verified for {component}")
                
                elif action == 'log_recovery':
                    message = step.get('message', 'Recovery action completed')
                    logger.info(f"Recovery: {message} for {component} by {executed_by}")
                    results.append(message)
                
                elif action == 'notify_users':
                    message = step.get('message', 'System notification')
                    results.append(f"Users notified: {message}")
                
                else:
                    results.append(f"Unknown action: {action}")
            
            except Exception as e:
                return False, f"Step failed: {action} - {e}"
        
        return True, " | ".join(results)
    
    def get_recommended_procedure(self, error: ErrorRecord) -> Optional[RecoveryProcedure]:
        """Get recommended recovery procedure for an error"""
        # Match by category
        for proc in self._procedures.values():
            if proc.category == error.category:
                return proc
        
        # Default graceful degradation
        if error.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            return self._procedures.get('PROC-GRACEFUL-DEGRADE')
        
        return None


# =============================================================================
# ERROR HANDLING SYSTEM (MAIN CLASS)
# =============================================================================

class ErrorHandlingSystem:
    """Main error handling system integrating all components"""
    
    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_path = PROJECT_ROOT / "data" / "error_handling" / "error_handling.db"
        
        self.db_path = db_path
        self.db = ErrorDatabaseManager(db_path)
        self.circuit_breaker = CircuitBreaker(self.db)
        self.fallback_manager = FallbackManager(self.db)
        self.error_handler = ErrorHandler(self.db, self.circuit_breaker, self.fallback_manager)
        self.recovery_manager = RecoveryManager(
            self.db, self.error_handler, 
            self.circuit_breaker, self.fallback_manager
        )
        
        # Initialize default circuit breaker configs
        self._init_default_configs()
    
    def _init_default_configs(self):
        """Initialize default configurations"""
        # Default circuit breaker configs for common components
        components = [
            "data_loader", "ml_models", "llm_agents", 
            "dashboard", "database", "analytics"
        ]
        
        for component in components:
            self.circuit_breaker.configure(CircuitBreakerConfig(
                component=component,
                failure_threshold=5,
                success_threshold=3,
                timeout_seconds=60
            ))
            
            self.fallback_manager.configure(FallbackConfig(
                component=component,
                fallback_mode=FallbackMode.FULL,
                cache_ttl_seconds=3600
            ))
    
    def handle(self, exception: Exception, 
              component: str, 
              operation: str,
              **context_kwargs) -> ErrorRecord:
        """Main entry point for handling errors"""
        context = ErrorContext(
            component=component,
            operation=operation,
            **context_kwargs
        )
        return self.error_handler.handle_error(exception, context)
    
    def with_error_handling(self, component: str, operation: str,
                           fallback_value: Any = None,
                           **context_kwargs):
        """Decorator for error handling"""
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Check circuit breaker
                can_exec, reason = self.circuit_breaker.can_execute(component)
                if not can_exec:
                    logger.warning(f"Circuit open for {component}: {reason}")
                    value, _ = self.fallback_manager.get_fallback_value(
                        component, operation, fallback_value
                    )
                    return value
                
                try:
                    result = func(*args, **kwargs)
                    self.circuit_breaker.record_success(component)
                    return result
                except Exception as e:
                    self.handle(e, component, operation, **context_kwargs)
                    value, _ = self.fallback_manager.get_fallback_value(
                        component, operation, fallback_value
                    )
                    return value
            return wrapper
        return decorator
    
    def get_component_health(self, component: str) -> ComponentHealth:
        """Get health status for a component"""
        cb_state = self.circuit_breaker.get_state(component)
        fallback_mode = self.fallback_manager.get_fallback_mode(component)
        
        # Calculate error rate (last hour)
        since = (datetime.now() - timedelta(hours=1)).isoformat()
        errors = self.db.execute_read_one(
            """SELECT COUNT(*) as count FROM error_records 
               WHERE component = ? AND created_at >= ?""",
            (component, since)
        )
        error_count = errors['count'] if errors else 0
        error_rate = error_count / 60.0  # errors per minute
        
        # Determine status
        if cb_state.state == CircuitState.OPEN:
            status = ComponentStatus.FAILING
        elif cb_state.state == CircuitState.HALF_OPEN:
            status = ComponentStatus.DEGRADED
        elif fallback_mode != FallbackMode.FULL:
            status = ComponentStatus.DEGRADED
        elif error_rate > 1.0:
            status = ComponentStatus.DEGRADED
        else:
            status = ComponentStatus.HEALTHY
        
        return ComponentHealth(
            component=component,
            status=status,
            fallback_mode=fallback_mode,
            circuit_state=cb_state.state,
            error_rate=error_rate,
            last_error=cb_state.last_failure_time,
            last_success=cb_state.last_success_time,
            message=f"Error rate: {error_rate:.2f}/min"
        )
    
    def get_system_health(self) -> Dict:
        """Get overall system health"""
        components = [
            "data_loader", "ml_models", "llm_agents",
            "dashboard", "database", "analytics"
        ]
        
        component_health = {}
        overall_status = ComponentStatus.HEALTHY
        
        for component in components:
            health = self.get_component_health(component)
            component_health[component] = {
                'status': health.status.value,
                'fallback_mode': health.fallback_mode.value,
                'circuit_state': health.circuit_state.value,
                'error_rate': health.error_rate
            }
            
            if health.status == ComponentStatus.FAILING:
                overall_status = ComponentStatus.FAILING
            elif health.status == ComponentStatus.DEGRADED and overall_status == ComponentStatus.HEALTHY:
                overall_status = ComponentStatus.DEGRADED
        
        # Error summary
        error_summary = self.error_handler.get_error_summary(hours=24)
        
        return {
            'overall_status': overall_status.value,
            'components': component_health,
            'error_summary': error_summary,
            'timestamp': datetime.now().isoformat()
        }
    
    def recover_component(self, component: str, 
                         procedure_id: Optional[str] = None,
                         executed_by: str = "system") -> Tuple[bool, str]:
        """Attempt to recover a component"""
        if procedure_id:
            return self.recovery_manager.execute_procedure(
                procedure_id, component, executed_by
            )
        
        # Auto-select procedure based on state
        cb_state = self.circuit_breaker.get_state(component)
        
        if cb_state.state == CircuitState.OPEN:
            return self.recovery_manager.execute_procedure(
                "PROC-CIRCUIT-RESET", component, executed_by
            )
        
        fallback_mode = self.fallback_manager.get_fallback_mode(component)
        if fallback_mode != FallbackMode.FULL:
            return self.recovery_manager.execute_procedure(
                "PROC-FALLBACK-RESTORE", component, executed_by
            )
        
        return True, "Component appears healthy, no recovery needed"
    
    def get_statistics(self) -> Dict:
        """Get error handling statistics"""
        return {
            'error_summary': self.error_handler.get_error_summary(hours=24),
            'recovery_procedures': len(self.recovery_manager.list_procedures()),
            'timestamp': datetime.now().isoformat()
        }


# =============================================================================
# SINGLETON AND CONVENIENCE FUNCTIONS
# =============================================================================

_error_handling_system: Optional[ErrorHandlingSystem] = None
_system_lock = threading.Lock()


def get_error_handling_system() -> ErrorHandlingSystem:
    """Get singleton instance of error handling system"""
    global _error_handling_system
    with _system_lock:
        if _error_handling_system is None:
            _error_handling_system = ErrorHandlingSystem()
        return _error_handling_system


def reset_error_handling_system():
    """Reset singleton for testing"""
    global _error_handling_system
    with _system_lock:
        _error_handling_system = None


def handle_error(exception: Exception, component: str, 
                operation: str, **kwargs) -> ErrorRecord:
    """Convenience function to handle an error"""
    return get_error_handling_system().handle(exception, component, operation, **kwargs)


def with_error_handling(component: str, operation: str, 
                       fallback_value: Any = None, **kwargs):
    """Convenience decorator for error handling"""
    return get_error_handling_system().with_error_handling(
        component, operation, fallback_value, **kwargs
    )


def get_system_health() -> Dict:
    """Get system health status"""
    return get_error_handling_system().get_system_health()


def recover_component(component: str, procedure_id: Optional[str] = None,
                     executed_by: str = "system") -> Tuple[bool, str]:
    """Recover a component"""
    return get_error_handling_system().recover_component(
        component, procedure_id, executed_by
    )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enums
    'ErrorSeverity',
    'ErrorCategory', 
    'RecoveryAction',
    'CircuitState',
    'FallbackMode',
    'ComponentStatus',
    
    # Data Classes
    'ErrorContext',
    'ErrorRecord',
    'FallbackConfig',
    'CircuitBreakerConfig',
    'CircuitBreakerState',
    'ComponentHealth',
    'RecoveryProcedure',
    
    # Main Classes
    'ErrorHandlingSystem',
    'ErrorHandler',
    'CircuitBreaker',
    'FallbackManager',
    'RecoveryManager',
    
    # Convenience Functions
    'get_error_handling_system',
    'reset_error_handling_system',
    'handle_error',
    'with_error_handling',
    'get_system_health',
    'recover_component',
]
