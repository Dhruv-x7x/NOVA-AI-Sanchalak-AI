"""
SANCHALAK AI - Database Data Loader
=======================================
Data loader that reads from PostgreSQL instead of parquet files.
Can be used as a drop-in replacement for the file-based loader.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

import pandas as pd
from sqlalchemy import text

from src.database.connection import get_db_manager
from src.database.config import DatabaseConfig

# PostgreSQL integration
from src.database.pg_data_service import get_data_service
from src.database.pg_writer import get_pg_writer


logger = logging.getLogger(__name__)


class DatabaseDataLoader:
    """
    Data loader that reads from PostgreSQL database.
    Provides the same interface as DashboardDataLoader for compatibility.
    """
    
    def __init__(self):
        """Initialize database data loader."""
        self._db_manager = None
        self._cache = {}
        self._cache_time = None
        self._cache_ttl = 300  # 5 minutes
    
    @property
    def db_manager(self):
        """Get database manager (lazy initialization)."""
        if self._db_manager is None:
            try:
                self._db_manager = get_db_manager()
            except Exception as e:
                logger.error(f"Database connection failed: {e}")
                raise
        return self._db_manager
    
    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid."""
        if self._cache_time is None:
            return False
        return (datetime.now() - self._cache_time).seconds < self._cache_ttl
    
    def get_patients_df(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Get patients dataframe from database.
        
        Args:
            limit: Optional limit on number of rows
            
        Returns:
            DataFrame with patient data
        """
        cache_key = f"patients_{limit}"
        if cache_key in self._cache and self._is_cache_valid():
            return self._cache[cache_key]
        
        query = "SELECT * FROM patients"
        if limit:
            query += f" LIMIT {limit}"
        
        with self.db_manager.engine.connect() as conn:
            df = pd.read_sql(query, conn)
        
        self._cache[cache_key] = df
        self._cache_time = datetime.now()
        return df
    
    def get_sites_df(self) -> pd.DataFrame:
        """Get sites dataframe from database."""
        if "sites" in self._cache and self._is_cache_valid():
            return self._cache["sites"]
        
        with self.db_manager.engine.connect() as conn:
            df = pd.read_sql("SELECT * FROM sites", conn)
        
        self._cache["sites"] = df
        self._cache_time = datetime.now()
        return df
    
    def get_studies_df(self) -> pd.DataFrame:
        """Get studies dataframe from database."""
        if "studies" in self._cache and self._is_cache_valid():
            return self._cache["studies"]
        
        with self.db_manager.engine.connect() as conn:
            df = pd.read_sql("SELECT * FROM studies", conn)
        
        self._cache["studies"] = df
        self._cache_time = datetime.now()
        return df
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Get portfolio summary statistics.
        
        Returns:
            Dictionary with summary stats
        """
        with self.db_manager.engine.connect() as conn:
            # Get counts
            patient_count = conn.execute(text("SELECT COUNT(*) FROM patients")).scalar()
            site_count = conn.execute(text("SELECT COUNT(*) FROM sites")).scalar()
            study_count = conn.execute(text("SELECT COUNT(*) FROM studies")).scalar()
            
            # Get DQI stats
            dqi_stats = conn.execute(text("""
                SELECT 
                    AVG(dqi_score) as mean_dqi,
                    MIN(dqi_score) as min_dqi,
                    MAX(dqi_score) as max_dqi
                FROM patients
            """)).fetchone()
            
            # Get clean counts
            clean_stats = conn.execute(text("""
                SELECT 
                    SUM(CASE WHEN tier1_clean = true THEN 1 ELSE 0 END) as tier1_clean_count,
                    SUM(CASE WHEN tier2_clean = true THEN 1 ELSE 0 END) as tier2_clean_count,
                    SUM(CASE WHEN is_db_lock_ready = true THEN 1 ELSE 0 END) as dblock_ready_count
                FROM patients
            """)).fetchone()
            
            # Get priority distribution
            priority_dist = conn.execute(text("""
                SELECT priority_tier, COUNT(*) as count
                FROM patients
                GROUP BY priority_tier
            """)).fetchall()
        
        return {
            'total_patients': patient_count,
            'total_sites': site_count,
            'total_studies': study_count,
            'mean_dqi': float(dqi_stats[0]) if dqi_stats[0] else 0,
            'min_dqi': float(dqi_stats[1]) if dqi_stats[1] else 0,
            'max_dqi': float(dqi_stats[2]) if dqi_stats[2] else 0,
            'tier1_clean_count': int(clean_stats[0]) if clean_stats[0] else 0,
            'tier2_clean_count': int(clean_stats[1]) if clean_stats[1] else 0,
            'dblock_ready_count': int(clean_stats[2]) if clean_stats[2] else 0,
            'priority_distribution': {row[0]: row[1] for row in priority_dist},
        }
    
    def get_high_priority_patients(self, limit: int = 100) -> pd.DataFrame:
        """Get high priority patients."""
        query = f"""
            SELECT * FROM patients 
            WHERE priority_tier IN ('critical', 'high')
            ORDER BY dqi_score ASC
            LIMIT {limit}
        """
        with self.db_manager.engine.connect() as conn:
            return pd.read_sql(query, conn)
    
    def get_patients_by_study(self, study_id: str) -> pd.DataFrame:
        """Get patients for a specific study."""
        query = f"SELECT * FROM patients WHERE study_id = '{study_id}'"
        with self.db_manager.engine.connect() as conn:
            return pd.read_sql(query, conn)
    
    def get_patients_by_site(self, site_id: str) -> pd.DataFrame:
        """Get patients for a specific site."""
        query = f"SELECT * FROM patients WHERE site_id LIKE '%{site_id}'"
        with self.db_manager.engine.connect() as conn:
            return pd.read_sql(query, conn)
    
    def get_dqi_distribution(self) -> Dict[str, int]:
        """Get DQI score distribution."""
        query = """
            SELECT 
                CASE 
                    WHEN dqi_score >= 95 THEN 'Elite'
                    WHEN dqi_score >= 90 THEN 'Optimal'
                    WHEN dqi_score >= 85 THEN 'Standard'
                    WHEN dqi_score >= 80 THEN 'Risk'
                    ELSE 'Critical'
                END as dqi_band,
                COUNT(*) as count
            FROM patients
            GROUP BY dqi_band
        """
        with self.db_manager.engine.connect() as conn:
            result = conn.execute(text(query)).fetchall()
        return {row[0]: row[1] for row in result}
    
    def search_patients(self, query: str, limit: int = 50) -> pd.DataFrame:
        """
        Search patients by key, subject, or site.
        
        Args:
            query: Search string
            limit: Maximum results
            
        Returns:
            DataFrame with matching patients
        """
        sql = f"""
            SELECT * FROM patients 
            WHERE patient_key ILIKE '%{query}%'
               OR subject ILIKE '%{query}%'
               OR site_id ILIKE '%{query}%'
            LIMIT {limit}
        """
        with self.db_manager.engine.connect() as conn:
            return pd.read_sql(sql, conn)
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute custom SQL query."""
        with self.db_manager.engine.connect() as conn:
            return pd.read_sql(query, conn)
    
    def health_check(self) -> bool:
        """Check database connectivity."""
        return self.db_manager.health_check()
    
    def clear_cache(self) -> None:
        """Clear the data cache."""
        self._cache = {}
        self._cache_time = None


# Singleton instance
_db_loader: Optional[DatabaseDataLoader] = None


def get_db_loader() -> DatabaseDataLoader:
    """Get singleton database loader instance."""
    global _db_loader
    if _db_loader is None:
        _db_loader = DatabaseDataLoader()
    return _db_loader
