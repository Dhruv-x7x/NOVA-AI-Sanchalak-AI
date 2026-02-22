"""
a6on-i - Phase 11.3: Performance Optimization System v1.0
# NOTE: This module previously used SQLite but has been migrated to PostgreSQL-only.
# If you need to use this module, update it to use:
#   from src.database.pg_data_service import get_pg_data_service
#   service = get_pg_data_service()


Features:
- Multi-layer caching (memory, disk, Redis-compatible)
- Lazy loading with deferred execution
- Query optimization and profiling
- Response time monitoring and targets
- Performance metrics and reporting
"""

import os
import sys
import time
import json
import pickle
import hashlib
# SQLite removed - using PostgreSQL
import threading
import functools
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import OrderedDict
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# ENUMS
# =============================================================================

class CacheLayer(Enum):
    """Cache layer types"""
    MEMORY = "memory"
    DISK = "disk"
    REDIS = "redis"  # Redis-compatible interface (uses disk for demo)

class CachePolicy(Enum):
    """Cache eviction policies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    FIFO = "fifo"  # First In First Out

class LoadingStrategy(Enum):
    """Data loading strategies"""
    EAGER = "eager"  # Load immediately
    LAZY = "lazy"  # Load on first access
    DEFERRED = "deferred"  # Load in background
    ON_DEMAND = "on_demand"  # Load only when explicitly requested

class PerformanceLevel(Enum):
    """Performance level classifications"""
    EXCELLENT = "excellent"  # < 100ms
    GOOD = "good"  # 100-500ms
    ACCEPTABLE = "acceptable"  # 500ms-1s
    SLOW = "slow"  # 1-3s
    CRITICAL = "critical"  # > 3s

class OptimizationStatus(Enum):
    """Query optimization status"""
    OPTIMIZED = "optimized"
    NEEDS_OPTIMIZATION = "needs_optimization"
    CANNOT_OPTIMIZE = "cannot_optimize"
    PENDING = "pending"

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CacheEntry:
    """Single cache entry"""
    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime]
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    size_bytes: int = 0
    layer: CacheLayer = CacheLayer.MEMORY
    tags: List[str] = field(default_factory=list)
    
    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    @property
    def ttl_seconds(self) -> Optional[float]:
        if self.expires_at is None:
            return None
        remaining = (self.expires_at - datetime.now()).total_seconds()
        return max(0, remaining)

@dataclass
class CacheConfig:
    """Cache configuration"""
    max_memory_items: int = 1000
    max_memory_bytes: int = 100 * 1024 * 1024  # 100MB
    max_disk_bytes: int = 1024 * 1024 * 1024  # 1GB
    default_ttl_seconds: int = 3600  # 1 hour
    eviction_policy: CachePolicy = CachePolicy.LRU
    enable_disk_cache: bool = True
    enable_compression: bool = False
    warmup_on_start: bool = False

@dataclass
class CacheStats:
    """Cache statistics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    memory_items: int = 0
    memory_bytes: int = 0
    disk_items: int = 0
    disk_bytes: int = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

@dataclass
class LazyObject:
    """Lazy-loaded object wrapper"""
    loader: Callable
    loaded: bool = False
    value: Any = None
    load_time_ms: float = 0.0
    error: Optional[str] = None
    
    def get(self) -> Any:
        if not self.loaded:
            start = time.time()
            try:
                self.value = self.loader()
                self.loaded = True
            except Exception as e:
                self.error = str(e)
                raise
            finally:
                self.load_time_ms = (time.time() - start) * 1000
        return self.value

@dataclass
class QueryProfile:
    """Query execution profile"""
    query_id: str
    query_hash: str
    query_text: str
    execution_time_ms: float
    rows_scanned: int = 0
    rows_returned: int = 0
    cache_hit: bool = False
    optimization_applied: bool = False
    optimization_suggestions: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ResponseTimeTarget:
    """Response time SLA target"""
    operation: str
    target_ms: float
    warning_ms: float
    critical_ms: float
    current_avg_ms: float = 0.0
    current_p95_ms: float = 0.0
    current_p99_ms: float = 0.0
    samples: int = 0
    violations: int = 0
    
    @property
    def status(self) -> PerformanceLevel:
        if self.current_avg_ms < self.target_ms:
            return PerformanceLevel.EXCELLENT
        elif self.current_avg_ms < self.warning_ms:
            return PerformanceLevel.GOOD
        elif self.current_avg_ms < self.critical_ms:
            return PerformanceLevel.ACCEPTABLE
        else:
            return PerformanceLevel.CRITICAL

@dataclass
class PerformanceMetrics:
    """Overall performance metrics"""
    timestamp: datetime
    cache_hit_rate: float
    avg_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    queries_per_second: float
    memory_usage_bytes: int
    active_connections: int
    slow_queries_count: int
    error_rate: float

# =============================================================================
# MULTI-LAYER CACHE MANAGER
# =============================================================================

class MemoryCache:
    """In-memory LRU cache"""
    
    def __init__(self, max_items: int = 1000, max_bytes: int = 100 * 1024 * 1024):
        self.max_items = max_items
        self.max_bytes = max_bytes
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._current_bytes = 0
    
    def get(self, key: str) -> Optional[CacheEntry]:
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if entry.is_expired:
                    self._remove(key)
                    return None
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                entry.access_count += 1
                entry.last_accessed = datetime.now()
                return entry
            return None
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None,
            tags: List[str] = None) -> bool:
        with self._lock:
            # Calculate size
            try:
                size = len(pickle.dumps(value))
            except:
                size = 1024  # Default estimate
            
            # Evict if necessary
            while (len(self._cache) >= self.max_items or 
                   self._current_bytes + size > self.max_bytes):
                if not self._cache:
                    break
                # Remove oldest (LRU)
                oldest_key = next(iter(self._cache))
                self._remove(oldest_key)
            
            # Create entry
            expires_at = None
            if ttl_seconds:
                expires_at = datetime.now() + timedelta(seconds=ttl_seconds)
            
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.now(),
                expires_at=expires_at,
                size_bytes=size,
                layer=CacheLayer.MEMORY,
                tags=tags or []
            )
            
            self._cache[key] = entry
            self._current_bytes += size
            return True
    
    def _remove(self, key: str):
        if key in self._cache:
            entry = self._cache.pop(key)
            self._current_bytes -= entry.size_bytes
    
    def delete(self, key: str) -> bool:
        with self._lock:
            if key in self._cache:
                self._remove(key)
                return True
            return False
    
    def clear(self):
        with self._lock:
            self._cache.clear()
            self._current_bytes = 0
    
    def get_stats(self) -> Dict:
        with self._lock:
            return {
                'items': len(self._cache),
                'bytes': self._current_bytes,
                'max_items': self.max_items,
                'max_bytes': self.max_bytes
            }
    
    def invalidate_by_tag(self, tag: str) -> int:
        with self._lock:
            keys_to_remove = [
                k for k, v in self._cache.items() 
                if tag in v.tags
            ]
            for key in keys_to_remove:
                self._remove(key)
            return len(keys_to_remove)

class DiskCache:
    """Disk-based cache with SQLite backend"""
    
    def __init__(self, db_path: Path, max_bytes: int = 1024 * 1024 * 1024):
        self.db_path = db_path
        self.max_bytes = max_bytes
        self._lock = threading.RLock()
        self._init_db()
    
    def _init_db(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if False:  # DB persistence disabled
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    created_at TEXT,
                    expires_at TEXT,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TEXT,
                    size_bytes INTEGER,
                    tags TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_expires ON cache(expires_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_accessed ON cache(last_accessed)")
    
    def get(self, key: str) -> Optional[CacheEntry]:
        with self._lock:
            try:
                if False:  # DB persistence disabled
                    cursor = conn.execute(
                        "SELECT * FROM cache WHERE key = ?", (key,)
                    )
                    row = cursor.fetchone()
                    if row:
                        expires_at = datetime.fromisoformat(row[3]) if row[3] else None
                        if expires_at and datetime.now() > expires_at:
                            conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                            return None
                        
                        # Update access info
                        conn.execute("""
                            UPDATE cache SET access_count = access_count + 1,
                            last_accessed = ? WHERE key = ?
                        """, (datetime.now().isoformat(), key))
                        
                        return CacheEntry(
                            key=row[0],
                            value=pickle.loads(row[1]),
                            created_at=datetime.fromisoformat(row[2]),
                            expires_at=expires_at,
                            access_count=row[4] + 1,
                            last_accessed=datetime.now(),
                            size_bytes=row[6],
                            layer=CacheLayer.DISK,
                            tags=json.loads(row[7]) if row[7] else []
                        )
            except Exception as e:
                logger.error(f"Disk cache get error: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None,
            tags: List[str] = None) -> bool:
        with self._lock:
            try:
                value_bytes = pickle.dumps(value)
                size = len(value_bytes)
                
                expires_at = None
                if ttl_seconds:
                    expires_at = (datetime.now() + timedelta(seconds=ttl_seconds)).isoformat()
                
                if False:  # DB persistence disabled
                    conn.execute("""
                        INSERT OR REPLACE INTO cache 
                        (key, value, created_at, expires_at, size_bytes, tags)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        key, value_bytes, datetime.now().isoformat(),
                        expires_at, size, json.dumps(tags or [])
                    ))
                return True
            except Exception as e:
                logger.error(f"Disk cache set error: {e}")
                return False
    
    def delete(self, key: str) -> bool:
        with self._lock:
            try:
                if False:  # DB persistence disabled
                    conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                return True
            except Exception as e:
                logger.error(f"Disk cache delete error: {e}")
                return False
    
    def clear(self):
        with self._lock:
            try:
                if False:  # DB persistence disabled
                    conn.execute("DELETE FROM cache")
            except Exception as e:
                logger.error(f"Disk cache clear error: {e}")
    
    def cleanup_expired(self) -> int:
        with self._lock:
            try:
                if False:  # DB persistence disabled
                    cursor = conn.execute("""
                        DELETE FROM cache WHERE expires_at IS NOT NULL 
                        AND expires_at < ?
                    """, (datetime.now().isoformat(),))
                    return cursor.rowcount
            except:
                return 0
    
    def get_stats(self) -> Dict:
        with self._lock:
            try:
                if False:  # DB persistence disabled
                    cursor = conn.execute(
                        "SELECT COUNT(*), SUM(size_bytes) FROM cache"
                    )
                    row = cursor.fetchone()
                    return {
                        'items': row[0] or 0,
                        'bytes': row[1] or 0,
                        'max_bytes': self.max_bytes
                    }
            except:
                return {'items': 0, 'bytes': 0, 'max_bytes': self.max_bytes}

class CacheManager:
    """Multi-layer cache manager"""
    
    def __init__(self, config: CacheConfig = None, cache_dir: Path = None):
        self.config = config or CacheConfig()
        self.cache_dir = cache_dir or Path("data/cache")
        
        # Initialize layers
        self.memory = MemoryCache(
            max_items=self.config.max_memory_items,
            max_bytes=self.config.max_memory_bytes
        )
        
        if self.config.enable_disk_cache:
            self.disk = DiskCache(
                self.cache_dir / "disk_cache.db",
                max_bytes=self.config.max_disk_bytes
            )
        else:
            self.disk = None
        
        # Statistics
        self._stats = CacheStats()
        self._lock = threading.RLock()
    
    def get(self, key: str, layer: CacheLayer = None) -> Optional[Any]:
        """Get value from cache"""
        with self._lock:
            # Try memory first
            if layer is None or layer == CacheLayer.MEMORY:
                entry = self.memory.get(key)
                if entry:
                    self._stats.hits += 1
                    return entry.value
            
            # Try disk
            if self.disk and (layer is None or layer == CacheLayer.DISK):
                entry = self.disk.get(key)
                if entry:
                    self._stats.hits += 1
                    # Promote to memory
                    if layer is None:
                        self.memory.set(key, entry.value, 
                                       int(entry.ttl_seconds) if entry.ttl_seconds else None,
                                       entry.tags)
                    return entry.value
            
            self._stats.misses += 1
            return None
    
    def set(self, key: str, value: Any, ttl_seconds: int = None,
            layer: CacheLayer = CacheLayer.MEMORY, tags: List[str] = None) -> bool:
        """Set value in cache"""
        ttl = ttl_seconds or self.config.default_ttl_seconds
        
        if layer == CacheLayer.MEMORY or layer is None:
            self.memory.set(key, value, ttl, tags)
        
        if self.disk and (layer == CacheLayer.DISK or layer == CacheLayer.REDIS):
            self.disk.set(key, value, ttl, tags)
        
        return True
    
    def delete(self, key: str) -> bool:
        """Delete from all layers"""
        result = self.memory.delete(key)
        if self.disk:
            result = result or self.disk.delete(key)
        return result
    
    def invalidate_by_tag(self, tag: str) -> int:
        """Invalidate all entries with a specific tag"""
        count = self.memory.invalidate_by_tag(tag)
        # Note: disk invalidation by tag would need additional implementation
        return count
    
    def clear(self, layer: CacheLayer = None):
        """Clear cache"""
        if layer is None or layer == CacheLayer.MEMORY:
            self.memory.clear()
        if self.disk and (layer is None or layer == CacheLayer.DISK):
            self.disk.clear()
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        with self._lock:
            mem_stats = self.memory.get_stats()
            disk_stats = self.disk.get_stats() if self.disk else {'items': 0, 'bytes': 0}
            
            self._stats.memory_items = mem_stats['items']
            self._stats.memory_bytes = mem_stats['bytes']
            self._stats.disk_items = disk_stats['items']
            self._stats.disk_bytes = disk_stats['bytes']
            
            return self._stats
    
    def cleanup(self) -> int:
        """Cleanup expired entries"""
        count = 0
        if self.disk:
            count = self.disk.cleanup_expired()
        return count

# =============================================================================
# LAZY LOADING MANAGER
# =============================================================================

class LazyLoader:
    """Manages lazy loading of data and resources"""
    
    def __init__(self, cache_manager: CacheManager = None):
        self.cache = cache_manager
        self._loaders: Dict[str, LazyObject] = {}
        self._lock = threading.RLock()
        self._background_threads: List[threading.Thread] = []
    
    def register(self, key: str, loader: Callable, 
                strategy: LoadingStrategy = LoadingStrategy.LAZY) -> None:
        """Register a lazy loader"""
        with self._lock:
            lazy_obj = LazyObject(loader=loader)
            self._loaders[key] = lazy_obj
            
            if strategy == LoadingStrategy.EAGER:
                # Load immediately
                lazy_obj.get()
            elif strategy == LoadingStrategy.DEFERRED:
                # Load in background
                thread = threading.Thread(target=lazy_obj.get, daemon=True)
                thread.start()
                self._background_threads.append(thread)
    
    def get(self, key: str) -> Any:
        """Get a lazy-loaded value"""
        with self._lock:
            if key not in self._loaders:
                raise KeyError(f"No loader registered for key: {key}")
            
            lazy_obj = self._loaders[key]
            
            # Check cache first
            if self.cache:
                cached = self.cache.get(f"lazy:{key}")
                if cached is not None:
                    return cached
            
            # Load the value
            value = lazy_obj.get()
            
            # Cache the result
            if self.cache:
                self.cache.set(f"lazy:{key}", value)
            
            return value
    
    def is_loaded(self, key: str) -> bool:
        """Check if a value is loaded"""
        with self._lock:
            if key in self._loaders:
                return self._loaders[key].loaded
            return False
    
    def get_load_time(self, key: str) -> float:
        """Get load time for a key"""
        with self._lock:
            if key in self._loaders:
                return self._loaders[key].load_time_ms
            return 0.0
    
    def reload(self, key: str) -> Any:
        """Force reload a value"""
        with self._lock:
            if key in self._loaders:
                lazy_obj = self._loaders[key]
                lazy_obj.loaded = False
                lazy_obj.value = None
                value = lazy_obj.get()
                if self.cache:
                    self.cache.set(f"lazy:{key}", value)
                return value
            raise KeyError(f"No loader registered for key: {key}")
    
    def get_stats(self) -> Dict:
        """Get lazy loading statistics"""
        with self._lock:
            loaded = sum(1 for v in self._loaders.values() if v.loaded)
            errors = sum(1 for v in self._loaders.values() if v.error)
            total_load_time = sum(v.load_time_ms for v in self._loaders.values() if v.loaded)
            
            return {
                'total_loaders': len(self._loaders),
                'loaded': loaded,
                'pending': len(self._loaders) - loaded,
                'errors': errors,
                'total_load_time_ms': total_load_time,
                'avg_load_time_ms': total_load_time / loaded if loaded > 0 else 0
            }

# =============================================================================
# QUERY OPTIMIZER
# =============================================================================

class QueryOptimizer:
    """Query optimization and profiling"""
    
    def __init__(self, db_path: Path = None):
        self.db_path = db_path or Path("data/performance/query_profiles.db")
        self._profiles: Dict[str, List[QueryProfile]] = {}
        self._optimization_cache: Dict[str, str] = {}
        self._lock = threading.RLock()
        self._init_db()
    
    def _init_db(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if False:  # DB persistence disabled
            conn.execute("""
                CREATE TABLE IF NOT EXISTS query_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query_id TEXT,
                    query_hash TEXT,
                    query_text TEXT,
                    execution_time_ms REAL,
                    rows_scanned INTEGER,
                    rows_returned INTEGER,
                    cache_hit INTEGER,
                    optimization_applied INTEGER,
                    suggestions TEXT,
                    timestamp TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_query_hash ON query_profiles(query_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON query_profiles(timestamp)")
    
    def _hash_query(self, query: str) -> str:
        """Generate hash for query"""
        normalized = ' '.join(query.lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()[:16]
    
    def profile_query(self, query: str, execution_time_ms: float,
                     rows_scanned: int = 0, rows_returned: int = 0,
                     cache_hit: bool = False) -> QueryProfile:
        """Profile a query execution"""
        query_hash = self._hash_query(query)
        query_id = f"Q-{datetime.now().strftime('%Y%m%d%H%M%S')}-{query_hash[:8]}"
        
        # Generate optimization suggestions
        suggestions = self._analyze_query(query, execution_time_ms, rows_scanned, rows_returned)
        
        profile = QueryProfile(
            query_id=query_id,
            query_hash=query_hash,
            query_text=query[:500],  # Truncate long queries
            execution_time_ms=execution_time_ms,
            rows_scanned=rows_scanned,
            rows_returned=rows_returned,
            cache_hit=cache_hit,
            optimization_applied=query_hash in self._optimization_cache,
            optimization_suggestions=suggestions
        )
        
        # Store profile
        with self._lock:
            if query_hash not in self._profiles:
                self._profiles[query_hash] = []
            self._profiles[query_hash].append(profile)
            
            # Persist to database
            self._save_profile(profile)
        
        return profile
    
    def _analyze_query(self, query: str, execution_time_ms: float,
                      rows_scanned: int, rows_returned: int) -> List[str]:
        """Analyze query and generate optimization suggestions"""
        suggestions = []
        query_lower = query.lower()
        
        # Time-based suggestions
        if execution_time_ms > 1000:
            suggestions.append("Query exceeds 1 second - consider optimization")
        
        # Scan efficiency
        if rows_scanned > 0 and rows_returned > 0:
            efficiency = rows_returned / rows_scanned
            if efficiency < 0.1:
                suggestions.append(f"Low scan efficiency ({efficiency:.1%}) - add appropriate filters or indexes")
        
        # Pattern-based suggestions
        if 'select *' in query_lower:
            suggestions.append("Avoid SELECT * - specify only needed columns")
        
        if 'where' not in query_lower and 'select' in query_lower:
            suggestions.append("No WHERE clause - consider adding filters")
        
        if query_lower.count('join') > 2:
            suggestions.append("Multiple JOINs detected - consider denormalization or query splitting")
        
        if 'order by' in query_lower and 'limit' not in query_lower:
            suggestions.append("ORDER BY without LIMIT - consider adding LIMIT")
        
        if 'like' in query_lower and '%' in query:
            if query.find('%') < query.find("'"):  # Leading wildcard
                suggestions.append("Leading wildcard in LIKE - consider full-text search")
        
        return suggestions
    
    def _save_profile(self, profile: QueryProfile):
        try:
            if False:  # DB persistence disabled
                conn.execute("""
                    INSERT INTO query_profiles 
                    (query_id, query_hash, query_text, execution_time_ms,
                     rows_scanned, rows_returned, cache_hit, optimization_applied,
                     suggestions, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    profile.query_id, profile.query_hash, profile.query_text,
                    profile.execution_time_ms, profile.rows_scanned, profile.rows_returned,
                    1 if profile.cache_hit else 0,
                    1 if profile.optimization_applied else 0,
                    json.dumps(profile.optimization_suggestions),
                    profile.timestamp.isoformat()
                ))
        except Exception as e:
            logger.error(f"Error saving query profile: {e}")
    
    def get_slow_queries(self, threshold_ms: float = 1000, limit: int = 20) -> List[QueryProfile]:
        """Get slow queries"""
        try:
            if False:  # DB persistence disabled
                cursor = conn.execute("""
                    SELECT * FROM query_profiles 
                    WHERE execution_time_ms > ?
                    ORDER BY execution_time_ms DESC
                    LIMIT ?
                """, (threshold_ms, limit))
                
                profiles = []
                for row in cursor.fetchall():
                    profiles.append(QueryProfile(
                        query_id=row[1],
                        query_hash=row[2],
                        query_text=row[3],
                        execution_time_ms=row[4],
                        rows_scanned=row[5],
                        rows_returned=row[6],
                        cache_hit=bool(row[7]),
                        optimization_applied=bool(row[8]),
                        optimization_suggestions=json.loads(row[9]) if row[9] else [],
                        timestamp=datetime.fromisoformat(row[10])
                    ))
                return profiles
        except Exception as e:
            logger.error(f"Error getting slow queries: {e}")
            return []
    
    def get_query_stats(self, query_hash: str = None) -> Dict:
        """Get query statistics"""
        try:
            if False:  # DB persistence disabled
                if query_hash:
                    cursor = conn.execute("""
                        SELECT COUNT(*), AVG(execution_time_ms), 
                               MIN(execution_time_ms), MAX(execution_time_ms),
                               SUM(CASE WHEN cache_hit = 1 THEN 1 ELSE 0 END)
                        FROM query_profiles WHERE query_hash = ?
                    """, (query_hash,))
                else:
                    cursor = conn.execute("""
                        SELECT COUNT(*), AVG(execution_time_ms),
                               MIN(execution_time_ms), MAX(execution_time_ms),
                               SUM(CASE WHEN cache_hit = 1 THEN 1 ELSE 0 END)
                        FROM query_profiles
                    """)
                
                row = cursor.fetchone()
                return {
                    'total_queries': row[0] or 0,
                    'avg_time_ms': row[1] or 0,
                    'min_time_ms': row[2] or 0,
                    'max_time_ms': row[3] or 0,
                    'cache_hits': row[4] or 0,
                    'cache_hit_rate': (row[4] or 0) / row[0] if row[0] else 0
                }
        except Exception as e:
            logger.error(f"Error getting query stats: {e}")
            return {}
    
    def suggest_indexes(self) -> List[Dict]:
        """Analyze queries and suggest indexes"""
        suggestions = []
        
        try:
            if False:  # DB persistence disabled
                # Get frequently slow queries
                cursor = conn.execute("""
                    SELECT query_hash, query_text, COUNT(*), AVG(execution_time_ms)
                    FROM query_profiles
                    WHERE execution_time_ms > 500
                    GROUP BY query_hash
                    HAVING COUNT(*) > 5
                    ORDER BY AVG(execution_time_ms) DESC
                    LIMIT 10
                """)
                
                for row in cursor.fetchall():
                    query_text = row[1].lower()
                    
                    # Extract potential index columns
                    columns = self._extract_filter_columns(query_text)
                    
                    if columns:
                        suggestions.append({
                            'query_hash': row[0],
                            'frequency': row[2],
                            'avg_time_ms': row[3],
                            'suggested_columns': columns,
                            'suggestion': f"Consider index on columns: {', '.join(columns)}"
                        })
        except Exception as e:
            logger.error(f"Error suggesting indexes: {e}")
        
        return suggestions
    
    def _extract_filter_columns(self, query: str) -> List[str]:
        """Extract columns used in WHERE, JOIN, ORDER BY"""
        columns = []
        
        # Simple pattern matching for common patterns
        import re
        
        # WHERE column = 
        where_matches = re.findall(r'where\s+(\w+)\s*[=<>]', query)
        columns.extend(where_matches)
        
        # AND/OR column =
        and_matches = re.findall(r'(?:and|or)\s+(\w+)\s*[=<>]', query)
        columns.extend(and_matches)
        
        # ORDER BY column
        order_matches = re.findall(r'order\s+by\s+(\w+)', query)
        columns.extend(order_matches)
        
        return list(set(columns))

# =============================================================================
# RESPONSE TIME MONITOR
# =============================================================================

class ResponseTimeMonitor:
    """Monitor and track response times against targets"""
    
    def __init__(self, db_path: Path = None):
        self.db_path = db_path or Path("data/performance/response_times.db")
        self._targets: Dict[str, ResponseTimeTarget] = {}
        self._measurements: Dict[str, List[float]] = {}
        self._lock = threading.RLock()
        self._max_measurements = 1000  # Keep last 1000 per operation
        self._init_db()
        self._init_default_targets()
    
    def _init_db(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if False:  # DB persistence disabled
            conn.execute("""
                CREATE TABLE IF NOT EXISTS response_times (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    operation TEXT,
                    response_time_ms REAL,
                    timestamp TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS targets (
                    operation TEXT PRIMARY KEY,
                    target_ms REAL,
                    warning_ms REAL,
                    critical_ms REAL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_op_time ON response_times(operation, timestamp)")
    
    def _init_default_targets(self):
        """Initialize default response time targets"""
        defaults = [
            ("dashboard_load", 500, 1000, 3000),
            ("data_refresh", 5000, 10000, 30000),
            ("query_execution", 100, 500, 2000),
            ("report_generation", 2000, 5000, 15000),
            ("ai_recommendation", 1000, 3000, 10000),
            ("cache_lookup", 10, 50, 200),
            ("api_response", 200, 500, 2000),
            ("page_render", 300, 800, 2000),
        ]
        
        for op, target, warning, critical in defaults:
            self.set_target(op, target, warning, critical)
    
    def set_target(self, operation: str, target_ms: float, 
                   warning_ms: float, critical_ms: float) -> None:
        """Set response time target for an operation"""
        with self._lock:
            self._targets[operation] = ResponseTimeTarget(
                operation=operation,
                target_ms=target_ms,
                warning_ms=warning_ms,
                critical_ms=critical_ms
            )
            
            # Persist
            try:
                if False:  # DB persistence disabled
                    conn.execute("""
                        INSERT OR REPLACE INTO targets 
                        (operation, target_ms, warning_ms, critical_ms)
                        VALUES (?, ?, ?, ?)
                    """, (operation, target_ms, warning_ms, critical_ms))
            except Exception as e:
                logger.error(f"Error saving target: {e}")
    
    def record(self, operation: str, response_time_ms: float) -> None:
        """Record a response time measurement"""
        with self._lock:
            # Update in-memory measurements
            if operation not in self._measurements:
                self._measurements[operation] = []
            
            self._measurements[operation].append(response_time_ms)
            
            # Trim to max size
            if len(self._measurements[operation]) > self._max_measurements:
                self._measurements[operation] = self._measurements[operation][-self._max_measurements:]
            
            # Update target stats
            if operation in self._targets:
                target = self._targets[operation]
                measurements = self._measurements[operation]
                
                target.samples = len(measurements)
                target.current_avg_ms = sum(measurements) / len(measurements)
                
                sorted_ms = sorted(measurements)
                p95_idx = int(len(sorted_ms) * 0.95)
                p99_idx = int(len(sorted_ms) * 0.99)
                
                target.current_p95_ms = sorted_ms[min(p95_idx, len(sorted_ms) - 1)]
                target.current_p99_ms = sorted_ms[min(p99_idx, len(sorted_ms) - 1)]
                
                if response_time_ms > target.critical_ms:
                    target.violations += 1
            
            # Persist (async would be better in production)
            try:
                if False:  # DB persistence disabled
                    conn.execute("""
                        INSERT INTO response_times (operation, response_time_ms, timestamp)
                        VALUES (?, ?, ?)
                    """, (operation, response_time_ms, datetime.now().isoformat()))
            except Exception as e:
                logger.error(f"Error recording response time: {e}")
    
    def get_target(self, operation: str) -> Optional[ResponseTimeTarget]:
        """Get target for an operation"""
        return self._targets.get(operation)
    
    def get_all_targets(self) -> Dict[str, ResponseTimeTarget]:
        """Get all targets"""
        return dict(self._targets)
    
    def get_violations(self, hours: int = 24) -> List[Dict]:
        """Get recent SLA violations"""
        violations = []
        
        for op, target in self._targets.items():
            if target.current_avg_ms > target.critical_ms:
                violations.append({
                    'operation': op,
                    'target_ms': target.target_ms,
                    'current_avg_ms': target.current_avg_ms,
                    'violation_rate': target.violations / target.samples if target.samples > 0 else 0,
                    'status': target.status.value
                })
        
        return violations
    
    def get_performance_report(self) -> Dict:
        """Get performance report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'operations': {},
            'summary': {
                'total_operations': len(self._targets),
                'healthy': 0,
                'warning': 0,
                'critical': 0
            }
        }
        
        for op, target in self._targets.items():
            status = target.status
            report['operations'][op] = {
                'target_ms': target.target_ms,
                'current_avg_ms': target.current_avg_ms,
                'p95_ms': target.current_p95_ms,
                'p99_ms': target.current_p99_ms,
                'samples': target.samples,
                'violations': target.violations,
                'status': status.value
            }
            
            if status in [PerformanceLevel.EXCELLENT, PerformanceLevel.GOOD]:
                report['summary']['healthy'] += 1
            elif status == PerformanceLevel.ACCEPTABLE:
                report['summary']['warning'] += 1
            else:
                report['summary']['critical'] += 1
        
        return report

# =============================================================================
# PERFORMANCE OPTIMIZATION SYSTEM (Main Interface)
# =============================================================================

class PerformanceOptimizationSystem:
    """Main interface for performance optimization"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, base_dir: Path = None):
        if self._initialized:
            return
        
        self.base_dir = base_dir or Path("data/performance")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.cache = CacheManager(
            cache_dir=self.base_dir / "cache"
        )
        
        self.lazy_loader = LazyLoader(cache_manager=self.cache)
        
        self.query_optimizer = QueryOptimizer(
            db_path=self.base_dir / "query_profiles.db"
        )
        
        self.response_monitor = ResponseTimeMonitor(
            db_path=self.base_dir / "response_times.db"
        )
        
        # Performance history
        self._metrics_history: List[PerformanceMetrics] = []
        
        self._initialized = True
        logger.info("PerformanceOptimizationSystem initialized")
    
    # -------------------------------------------------------------------------
    # CACHING INTERFACE
    # -------------------------------------------------------------------------
    
    def cache_get(self, key: str) -> Optional[Any]:
        """Get from cache"""
        return self.cache.get(key)
    
    def cache_set(self, key: str, value: Any, ttl_seconds: int = None,
                  tags: List[str] = None) -> bool:
        """Set in cache"""
        return self.cache.set(key, value, ttl_seconds, tags=tags)
    
    def cache_delete(self, key: str) -> bool:
        """Delete from cache"""
        return self.cache.delete(key)
    
    def cache_invalidate_tag(self, tag: str) -> int:
        """Invalidate cache entries by tag"""
        return self.cache.invalidate_by_tag(tag)
    
    def cache_clear(self) -> None:
        """Clear all caches"""
        self.cache.clear()
    
    def cache_stats(self) -> CacheStats:
        """Get cache statistics"""
        return self.cache.get_stats()
    
    # -------------------------------------------------------------------------
    # LAZY LOADING INTERFACE
    # -------------------------------------------------------------------------
    
    def register_lazy_loader(self, key: str, loader: Callable,
                            strategy: LoadingStrategy = LoadingStrategy.LAZY) -> None:
        """Register a lazy loader"""
        self.lazy_loader.register(key, loader, strategy)
    
    def lazy_get(self, key: str) -> Any:
        """Get lazy-loaded value"""
        return self.lazy_loader.get(key)
    
    def lazy_reload(self, key: str) -> Any:
        """Force reload a lazy value"""
        return self.lazy_loader.reload(key)
    
    def lazy_stats(self) -> Dict:
        """Get lazy loading statistics"""
        return self.lazy_loader.get_stats()
    
    # -------------------------------------------------------------------------
    # QUERY OPTIMIZATION INTERFACE
    # -------------------------------------------------------------------------
    
    def profile_query(self, query: str, execution_time_ms: float,
                     rows_scanned: int = 0, rows_returned: int = 0,
                     cache_hit: bool = False) -> QueryProfile:
        """Profile a query"""
        return self.query_optimizer.profile_query(
            query, execution_time_ms, rows_scanned, rows_returned, cache_hit
        )
    
    def get_slow_queries(self, threshold_ms: float = 1000) -> List[QueryProfile]:
        """Get slow queries"""
        return self.query_optimizer.get_slow_queries(threshold_ms)
    
    def get_query_stats(self) -> Dict:
        """Get query statistics"""
        return self.query_optimizer.get_query_stats()
    
    def suggest_indexes(self) -> List[Dict]:
        """Get index suggestions"""
        return self.query_optimizer.suggest_indexes()
    
    # -------------------------------------------------------------------------
    # RESPONSE TIME INTERFACE
    # -------------------------------------------------------------------------
    
    def set_response_target(self, operation: str, target_ms: float,
                           warning_ms: float, critical_ms: float) -> None:
        """Set response time target"""
        self.response_monitor.set_target(operation, target_ms, warning_ms, critical_ms)
    
    def record_response_time(self, operation: str, response_time_ms: float) -> None:
        """Record a response time"""
        self.response_monitor.record(operation, response_time_ms)
    
    def get_response_target(self, operation: str) -> Optional[ResponseTimeTarget]:
        """Get response target"""
        return self.response_monitor.get_target(operation)
    
    def get_sla_violations(self) -> List[Dict]:
        """Get SLA violations"""
        return self.response_monitor.get_violations()
    
    def get_performance_report(self) -> Dict:
        """Get performance report"""
        return self.response_monitor.get_performance_report()
    
    # -------------------------------------------------------------------------
    # DECORATORS
    # -------------------------------------------------------------------------
    
    def cached(self, ttl_seconds: int = 3600, tags: List[str] = None):
        """Decorator for caching function results"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Generate cache key
                key_parts = [func.__name__]
                key_parts.extend(str(a) for a in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = hashlib.md5(':'.join(key_parts).encode()).hexdigest()
                
                # Try cache
                cached = self.cache_get(cache_key)
                if cached is not None:
                    return cached
                
                # Execute and cache
                result = func(*args, **kwargs)
                self.cache_set(cache_key, result, ttl_seconds, tags)
                return result
            return wrapper
        return decorator
    
    def timed(self, operation: str = None):
        """Decorator for timing function execution"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                op_name = operation or func.__name__
                start = time.time()
                try:
                    return func(*args, **kwargs)
                finally:
                    elapsed = (time.time() - start) * 1000
                    self.record_response_time(op_name, elapsed)
            return wrapper
        return decorator
    
    # -------------------------------------------------------------------------
    # OVERALL STATISTICS
    # -------------------------------------------------------------------------
    
    def get_statistics(self) -> Dict:
        """Get overall performance statistics"""
        cache_stats = self.cache_stats()
        lazy_stats = self.lazy_stats()
        query_stats = self.get_query_stats()
        perf_report = self.get_performance_report()
        
        return {
            'cache': {
                'hit_rate': cache_stats.hit_rate,
                'memory_items': cache_stats.memory_items,
                'memory_bytes': cache_stats.memory_bytes,
                'disk_items': cache_stats.disk_items,
                'disk_bytes': cache_stats.disk_bytes,
                'hits': cache_stats.hits,
                'misses': cache_stats.misses
            },
            'lazy_loading': lazy_stats,
            'queries': query_stats,
            'response_times': perf_report['summary'],
            'operations': perf_report['operations']
        }
    
    def get_health_status(self) -> Dict:
        """Get system health status"""
        cache_stats = self.cache_stats()
        violations = self.get_sla_violations()
        
        # Determine overall health
        if len(violations) > 3:
            status = 'critical'
        elif len(violations) > 0:
            status = 'degraded'
        elif cache_stats.hit_rate < 0.5:
            status = 'warning'
        else:
            status = 'healthy'
        
        return {
            'status': status,
            'cache_hit_rate': cache_stats.hit_rate,
            'sla_violations': len(violations),
            'memory_usage_bytes': cache_stats.memory_bytes,
            'timestamp': datetime.now().isoformat()
        }

# =============================================================================
# SINGLETON ACCESS & CONVENIENCE FUNCTIONS
# =============================================================================

_performance_system: Optional[PerformanceOptimizationSystem] = None

def get_performance_system() -> PerformanceOptimizationSystem:
    """Get singleton performance system"""
    global _performance_system
    if _performance_system is None:
        _performance_system = PerformanceOptimizationSystem()
    return _performance_system

def reset_performance_system():
    """Reset for testing"""
    global _performance_system
    _performance_system = None

# Convenience functions
def cache_get(key: str) -> Optional[Any]:
    return get_performance_system().cache_get(key)

def cache_set(key: str, value: Any, ttl: int = 3600, tags: List[str] = None) -> bool:
    return get_performance_system().cache_set(key, value, ttl, tags)

def cache_delete(key: str) -> bool:
    return get_performance_system().cache_delete(key)

def record_timing(operation: str, time_ms: float):
    get_performance_system().record_response_time(operation, time_ms)

def get_performance_stats() -> Dict:
    return get_performance_system().get_statistics()

def get_health() -> Dict:
    return get_performance_system().get_health_status()

# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Main class
    'PerformanceOptimizationSystem',
    
    # Components
    'CacheManager',
    'LazyLoader',
    'QueryOptimizer',
    'ResponseTimeMonitor',
    
    # Data classes
    'CacheEntry',
    'CacheConfig',
    'CacheStats',
    'LazyObject',
    'QueryProfile',
    'ResponseTimeTarget',
    'PerformanceMetrics',
    
    # Enums
    'CacheLayer',
    'CachePolicy',
    'LoadingStrategy',
    'PerformanceLevel',
    'OptimizationStatus',
    
    # Convenience functions
    'get_performance_system',
    'reset_performance_system',
    'cache_get',
    'cache_set',
    'cache_delete',
    'record_timing',
    'get_performance_stats',
    'get_health',
]
