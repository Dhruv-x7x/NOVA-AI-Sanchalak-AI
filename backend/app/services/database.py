"""
Database Service Bridge
=======================
Bridges FastAPI to the existing PostgreSQL data service.
Includes a TTL cache layer to avoid redundant DB round-trips.
"""

import sys
import os
import importlib
import time
import hashlib
import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Add src to path for importing existing data service
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Keep a global instance
_service_instance: Any = None

# ── TTL Cache ──────────────────────────────────────────────────
_cache: dict = {}           # key → (timestamp, value)
_CACHE_TTL = 30             # seconds — fresh enough for dashboards, fast enough to feel instant

def _cache_key(method: str, **kwargs) -> str:
    """Build a deterministic cache key from method name + args."""
    raw = f"{method}:{json.dumps(kwargs, sort_keys=True, default=str)}"
    return hashlib.md5(raw.encode()).hexdigest()

def _cached_call(method_name: str, **kwargs):
    """Call a data-service method with TTL caching."""
    key = _cache_key(method_name, **kwargs)
    now = time.monotonic()
    if key in _cache:
        ts, val = _cache[key]
        if now - ts < _CACHE_TTL:
            return val
    svc = get_data_service()
    fn = getattr(svc, method_name)
    result = fn(**kwargs)
    _cache[key] = (now, result)
    return result

def clear_ttl_cache():
    """Flush entire TTL cache."""
    global _cache
    _cache.clear()

class CachedDataService:
    """Proxy that wraps the real PostgreSQL data service with TTL caching.
    
    All read methods are cached for _CACHE_TTL seconds.
    Mutating methods (if any) bypass the cache.
    """
    _SKIP_CACHE = {'health_check', '_initialize', '_get_session', '_map_site_id', '_map_upr_site_id'}

    def __init__(self, real_service):
        self._real = real_service

    def __getattr__(self, name):
        attr = getattr(self._real, name)
        if not callable(attr) or name.startswith('_') or name in self._SKIP_CACHE:
            return attr

        def cached_wrapper(*args, **kwargs):
            key = _cache_key(name, args=args, **kwargs)
            now = time.monotonic()
            if key in _cache:
                ts, val = _cache[key]
                if now - ts < _CACHE_TTL:
                    return val
            result = attr(*args, **kwargs)
            _cache[key] = (now, result)
            return result

        return cached_wrapper


_raw_service_instance: Any = None

def get_data_service():
    """Get singleton instance of PostgreSQL data service wrapped with TTL cache."""
    global _service_instance, _raw_service_instance
    
    # Lazy import to avoid circular dependencies
    import src.database.pg_data_service as pg_mod
    
    if _raw_service_instance is None or not hasattr(_raw_service_instance, 'get_dqi_distribution'):
        importlib.reload(pg_mod)
        _raw_service_instance = pg_mod.PostgreSQLDataService()
        _service_instance = CachedDataService(_raw_service_instance)
    
    return _service_instance


def clear_data_service_cache():
    """Clear lru_cache, internal service cache, and TTL cache."""
    global _service_instance, _raw_service_instance
    _service_instance = None
    _raw_service_instance = None
    clear_ttl_cache()
    
    import src.database.pg_data_service as pg_mod
    importlib.reload(pg_mod)
    
    return True


# Export commonly used functions — hot-path queries use TTL cache
def get_patients(limit=None, study_id=None, site_id=None):
    return _cached_call('get_patients', limit=limit, study_id=study_id, site_id=site_id)

def get_patient(patient_key: str):
    return get_data_service().get_patient(patient_key)

def search_patients(query: str, limit: int = 20):
    return get_data_service().search_patients(query, limit)

def get_sites():
    return _cached_call('get_sites')

def get_site_benchmarks(study_id=None):
    return _cached_call('get_site_benchmarks', study_id=study_id)

def get_studies(limit=None):
    return _cached_call('get_studies', limit=limit)

def get_issues(status=None, limit=2000, study_id=None, site_id=None):
    return _cached_call('get_issues', status=status, limit=limit, study_id=study_id, site_id=site_id)

def get_patient_issues():
    return _cached_call('get_patient_issues')

def get_queries(status=None):
    return _cached_call('get_queries', status=status)

def get_portfolio_summary(study_id=None):
    return _cached_call('get_portfolio_summary', study_id=study_id)

def get_patient_dqi(study_id=None):
    return _cached_call('get_patient_dqi', study_id=study_id)

def get_patient_clean_status(study_id=None):
    return _cached_call('get_patient_clean_status', study_id=study_id)

def get_patient_dblock_status(study_id=None):
    return _cached_call('get_patient_dblock_status', study_id=study_id)

def get_regional_metrics(study_id=None):
    return _cached_call('get_regional_metrics', study_id=study_id)

def get_ml_models():
    return _cached_call('get_ml_models')

def get_dqi_distribution(study_id=None):
    return _cached_call('get_dqi_distribution', study_id=study_id)

def get_pattern_alerts(study_id=None):
    return _cached_call('get_pattern_alerts', study_id=study_id)

def get_cascade_analysis(study_id=None):
    return _cached_call('get_cascade_analysis', study_id=study_id)

def get_signatures_summary(study_id: Optional[str] = None):
    return _cached_call('get_signatures_summary', study_id=study_id)

def get_prediction_data(study_id: Optional[str] = None, site_id: Optional[str] = None):
    return _cached_call('get_prediction_data', study_id=study_id, site_id=site_id)

def health_check():
    return get_data_service().health_check()
