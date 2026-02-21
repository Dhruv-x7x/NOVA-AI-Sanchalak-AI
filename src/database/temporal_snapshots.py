"""
SANCHALAK AI - Temporal Snapshots Service
==============================================
Captures hourly state saves for trend analysis and delta tracking.

Features:
- Capture point-in-time snapshots of entity metrics
- Delta detection with anomaly flagging
- Trend analysis over configurable time periods
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
from sqlalchemy import text

from src.database.connection import get_db_manager

logger = logging.getLogger(__name__)


class TemporalSnapshotService:
    """Service for managing temporal snapshots and delta tracking."""
    
    _instance: Optional['TemporalSnapshotService'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._db_manager = None
        self._initialize()
        self._initialized = True
    
    def _initialize(self):
        """Initialize database connection."""
        try:
            self._db_manager = get_db_manager()
            self._ensure_tables()
            logger.info("TemporalSnapshotService initialized")
        except Exception as e:
            logger.error(f"TemporalSnapshotService initialization failed: {e}")
            raise
    
    def _ensure_tables(self):
        """Ensure temporal snapshot tables exist."""
        create_snapshots_table = text("""
            CREATE TABLE IF NOT EXISTS temporal_snapshots (
                id SERIAL PRIMARY KEY,
                snapshot_timestamp TIMESTAMP NOT NULL DEFAULT NOW(),
                entity_type VARCHAR(50) NOT NULL,
                entity_id VARCHAR(100) NOT NULL,
                metric_name VARCHAR(100) NOT NULL,
                metric_value FLOAT NOT NULL,
                dqi_score FLOAT,
                clean_status VARCHAR(20),
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        
        create_deltas_table = text("""
            CREATE TABLE IF NOT EXISTS delta_changes (
                id SERIAL PRIMARY KEY,
                detected_at TIMESTAMP NOT NULL DEFAULT NOW(),
                entity_type VARCHAR(50) NOT NULL,
                entity_id VARCHAR(100) NOT NULL,
                metric_name VARCHAR(100) NOT NULL,
                previous_value FLOAT,
                current_value FLOAT,
                change_magnitude FLOAT,
                change_percentage FLOAT,
                is_anomaly BOOLEAN DEFAULT FALSE,
                anomaly_score FLOAT
            )
        """)
        
        create_indexes = [
            text("CREATE INDEX IF NOT EXISTS idx_snapshots_entity ON temporal_snapshots(entity_type, entity_id)"),
            text("CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp ON temporal_snapshots(snapshot_timestamp)"),
            text("CREATE INDEX IF NOT EXISTS idx_delta_entity ON delta_changes(entity_type, entity_id)"),
            text("CREATE INDEX IF NOT EXISTS idx_delta_anomaly ON delta_changes(is_anomaly) WHERE is_anomaly = TRUE"),
        ]
        
        try:
            with self._db_manager.engine.connect() as conn:
                conn.execute(create_snapshots_table)
                conn.execute(create_deltas_table)
                for idx in create_indexes:
                    try:
                        conn.execute(idx)
                    except Exception:
                        pass  # Index may already exist
                conn.commit()
                logger.info("Temporal snapshot tables verified")
        except Exception as e:
            logger.warning(f"Could not create temporal tables: {e}")
    
    def capture_snapshot(self) -> Dict[str, Any]:
        """Capture current state snapshot for all entities."""
        try:
            with self._db_manager.engine.connect() as conn:
                timestamp = datetime.now()
                
                # Get site-level aggregates
                site_query = text("""
                    SELECT site_id, 
                           AVG(dqi_score) as avg_dqi,
                           COUNT(*) as patient_count,
                           SUM(CASE WHEN clean_status_tier IN ('TIER_2', 'DB_LOCK_READY') THEN 1 ELSE 0 END) as clean_count,
                           SUM(CASE WHEN is_db_lock_ready = TRUE THEN 1 ELSE 0 END) as dblock_ready_count
                    FROM patients
                    GROUP BY site_id
                """)
                site_df = pd.read_sql(site_query, conn)
                
                if site_df.empty:
                    logger.warning("No patient data available for snapshot")
                    return {"status": "no_data", "timestamp": timestamp.isoformat()}
                
                # Insert site snapshots
                insert_count = 0
                for _, row in site_df.iterrows():
                    # Insert avg_dqi metric
                    conn.execute(text("""
                        INSERT INTO temporal_snapshots 
                        (snapshot_timestamp, entity_type, entity_id, metric_name, metric_value, dqi_score)
                        VALUES (:ts, 'site', :site_id, 'avg_dqi', :value, :dqi)
                    """), {
                        "ts": timestamp,
                        "site_id": row['site_id'],
                        "value": float(row['avg_dqi']) if row['avg_dqi'] else 0.0,
                        "dqi": float(row['avg_dqi']) if row['avg_dqi'] else 0.0
                    })
                    
                    # Insert patient_count metric
                    conn.execute(text("""
                        INSERT INTO temporal_snapshots 
                        (snapshot_timestamp, entity_type, entity_id, metric_name, metric_value)
                        VALUES (:ts, 'site', :site_id, 'patient_count', :value)
                    """), {
                        "ts": timestamp,
                        "site_id": row['site_id'],
                        "value": float(row['patient_count'])
                    })
                    
                    # Insert clean_rate metric
                    clean_rate = (row['clean_count'] / row['patient_count'] * 100) if row['patient_count'] > 0 else 0
                    conn.execute(text("""
                        INSERT INTO temporal_snapshots 
                        (snapshot_timestamp, entity_type, entity_id, metric_name, metric_value)
                        VALUES (:ts, 'site', :site_id, 'clean_rate', :value)
                    """), {
                        "ts": timestamp,
                        "site_id": row['site_id'],
                        "value": float(clean_rate)
                    })
                    
                    insert_count += 3
                
                # Get study-level aggregates
                study_query = text("""
                    SELECT study_id, 
                           AVG(dqi_score) as avg_dqi,
                           COUNT(*) as patient_count
                    FROM patients
                    GROUP BY study_id
                """)
                study_df = pd.read_sql(study_query, conn)
                
                for _, row in study_df.iterrows():
                    conn.execute(text("""
                        INSERT INTO temporal_snapshots 
                        (snapshot_timestamp, entity_type, entity_id, metric_name, metric_value, dqi_score)
                        VALUES (:ts, 'study', :study_id, 'avg_dqi', :value, :dqi)
                    """), {
                        "ts": timestamp,
                        "study_id": row['study_id'],
                        "value": float(row['avg_dqi']) if row['avg_dqi'] else 0.0,
                        "dqi": float(row['avg_dqi']) if row['avg_dqi'] else 0.0
                    })
                    insert_count += 1
                
                conn.commit()
                logger.info(f"Captured {insert_count} snapshot metrics at {timestamp}")
                
                return {
                    "status": "success",
                    "timestamp": timestamp.isoformat(),
                    "metrics_captured": insert_count,
                    "sites_captured": len(site_df),
                    "studies_captured": len(study_df)
                }
                
        except Exception as e:
            logger.error(f"Error capturing snapshot: {e}")
            return {"status": "error", "error": str(e)}
    
    def detect_deltas(self, hours_back: int = 1, anomaly_threshold: float = 10.0) -> List[Dict]:
        """
        Detect changes between current and previous snapshot.
        
        Args:
            hours_back: How many hours back to compare
            anomaly_threshold: Percentage change to flag as anomaly
            
        Returns:
            List of detected delta changes
        """
        try:
            with self._db_manager.engine.connect() as conn:
                # Get latest two snapshots per entity/metric
                delta_query = text("""
                    WITH ranked AS (
                        SELECT *, 
                               ROW_NUMBER() OVER (
                                   PARTITION BY entity_type, entity_id, metric_name 
                                   ORDER BY snapshot_timestamp DESC
                               ) as rn
                        FROM temporal_snapshots
                        WHERE snapshot_timestamp > NOW() - INTERVAL ':hours hours'
                    )
                    SELECT 
                        a.entity_type, 
                        a.entity_id, 
                        a.metric_name,
                        a.metric_value as current_value,
                        b.metric_value as previous_value,
                        (a.metric_value - b.metric_value) as change_magnitude,
                        CASE WHEN b.metric_value != 0 
                             THEN ((a.metric_value - b.metric_value) / b.metric_value * 100)
                             ELSE 0 END as change_percentage
                    FROM ranked a
                    JOIN ranked b ON a.entity_type = b.entity_type 
                                  AND a.entity_id = b.entity_id 
                                  AND a.metric_name = b.metric_name
                    WHERE a.rn = 1 AND b.rn = 2
                """.replace(':hours', str(hours_back * 2)))
                
                delta_df = pd.read_sql(delta_query, conn)
                
                if delta_df.empty:
                    return []
                
                # Flag anomalies (changes exceeding threshold)
                delta_df['is_anomaly'] = abs(delta_df['change_percentage']) > anomaly_threshold
                delta_df['anomaly_score'] = abs(delta_df['change_percentage']) / anomaly_threshold
                
                anomalies = delta_df[delta_df['is_anomaly']]
                
                # Insert delta records for anomalies
                for _, row in anomalies.iterrows():
                    conn.execute(text("""
                        INSERT INTO delta_changes 
                        (entity_type, entity_id, metric_name, previous_value, current_value,
                         change_magnitude, change_percentage, is_anomaly, anomaly_score)
                        VALUES (:entity_type, :entity_id, :metric_name, :prev, :curr,
                                :magnitude, :pct, TRUE, :score)
                    """), {
                        "entity_type": row['entity_type'],
                        "entity_id": row['entity_id'],
                        "metric_name": row['metric_name'],
                        "prev": float(row['previous_value']) if row['previous_value'] else 0.0,
                        "curr": float(row['current_value']) if row['current_value'] else 0.0,
                        "magnitude": float(row['change_magnitude']) if row['change_magnitude'] else 0.0,
                        "pct": float(row['change_percentage']) if row['change_percentage'] else 0.0,
                        "score": float(row['anomaly_score']) if row['anomaly_score'] else 0.0
                    })
                
                conn.commit()
                
                return anomalies.to_dict('records')
                
        except Exception as e:
            logger.error(f"Error detecting deltas: {e}")
            return []
    
    def get_trend_data(
        self, 
        entity_type: str, 
        entity_id: str, 
        metric_name: str, 
        days: int = 7
    ) -> pd.DataFrame:
        """
        Get historical trend data for an entity.
        
        Args:
            entity_type: 'site', 'study', 'region'
            entity_id: Entity identifier
            metric_name: Metric to retrieve
            days: Number of days of history
            
        Returns:
            DataFrame with timestamp and metric_value columns
        """
        try:
            with self._db_manager.engine.connect() as conn:
                query = text("""
                    SELECT snapshot_timestamp, metric_value
                    FROM temporal_snapshots
                    WHERE entity_type = :entity_type 
                      AND entity_id = :entity_id 
                      AND metric_name = :metric_name
                      AND snapshot_timestamp > NOW() - INTERVAL ':days days'
                    ORDER BY snapshot_timestamp
                """.replace(':days', str(days)))
                
                df = pd.read_sql(query, conn, params={
                    "entity_type": entity_type,
                    "entity_id": entity_id,
                    "metric_name": metric_name
                })
                
                return df
                
        except Exception as e:
            logger.error(f"Error getting trend data: {e}")
            return pd.DataFrame()
    
    def get_recent_anomalies(self, hours: int = 24, limit: int = 50) -> pd.DataFrame:
        """Get recent anomalies detected."""
        try:
            with self._db_manager.engine.connect() as conn:
                query = text(f"""
                    SELECT * FROM delta_changes
                    WHERE is_anomaly = TRUE
                      AND detected_at > NOW() - INTERVAL '{hours} hours'
                    ORDER BY anomaly_score DESC
                    LIMIT {limit}
                """)
                
                return pd.read_sql(query, conn)
                
        except Exception as e:
            logger.error(f"Error getting anomalies: {e}")
            return pd.DataFrame()
    
    def get_snapshot_stats(self) -> Dict[str, Any]:
        """Get statistics about snapshots."""
        try:
            with self._db_manager.engine.connect() as conn:
                # Count snapshots
                count_query = text("SELECT COUNT(*) as cnt FROM temporal_snapshots")
                count_result = pd.read_sql(count_query, conn)
                snapshot_count = int(count_result['cnt'].values[0]) if not count_result.empty else 0
                
                # Count anomalies
                anomaly_query = text("SELECT COUNT(*) as cnt FROM delta_changes WHERE is_anomaly = TRUE")
                anomaly_result = pd.read_sql(anomaly_query, conn)
                anomaly_count = int(anomaly_result['cnt'].values[0]) if not anomaly_result.empty else 0
                
                # Latest snapshot
                latest_query = text("SELECT MAX(snapshot_timestamp) as latest FROM temporal_snapshots")
                latest_result = pd.read_sql(latest_query, conn)
                latest = latest_result['latest'].values[0] if not latest_result.empty else None
                
                return {
                    "total_snapshots": snapshot_count,
                    "total_anomalies": anomaly_count,
                    "latest_snapshot": str(latest) if latest else None
                }
                
        except Exception as e:
            logger.error(f"Error getting snapshot stats: {e}")
            return {"error": str(e)}


# Singleton accessor
_snapshot_service: Optional[TemporalSnapshotService] = None


def get_snapshot_service() -> TemporalSnapshotService:
    """Get singleton temporal snapshot service."""
    global _snapshot_service
    if _snapshot_service is None:
        _snapshot_service = TemporalSnapshotService()
    return _snapshot_service


def reset_snapshot_service():
    """Reset the singleton (for testing)."""
    global _snapshot_service
    _snapshot_service = None
