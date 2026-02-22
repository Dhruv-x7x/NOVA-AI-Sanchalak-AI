"""
a6on-i - UPR Real-time Updater
=========================================
Apply streaming events to PostgreSQL database in real-time.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional

from sqlalchemy import text

from .event_processor import ProcessedEvent, get_event_processor
from src.database.connection import get_db_manager

# PostgreSQL integration
from src.database.pg_data_service import get_data_service
from src.database.pg_writer import get_pg_writer


logger = logging.getLogger(__name__)


class UPRUpdater:
    """
    Apply real-time updates to the Unified Patient Record in PostgreSQL.
    
    Handles:
    - Increments/decrements for counters
    - Direct field updates
    - Audit logging
    """
    
    def __init__(self):
        self._db_manager = None
        self._processor = get_event_processor()
        self._update_count = 0
    
    @property
    def db_manager(self):
        if self._db_manager is None:
            self._db_manager = get_db_manager()
        return self._db_manager
    
    def apply_event(self, event: Dict[str, Any]) -> bool:
        """
        Process and apply an event to the database.
        
        Args:
            event: Raw event from Kafka
            
        Returns:
            True if update was applied successfully
        """
        # Process the event
        processed = self._processor.process(event)
        if processed is None:
            return False
        
        return self.apply_processed_event(processed)
    
    def apply_processed_event(self, processed: ProcessedEvent) -> bool:
        """
        Apply a processed event to the database.
        
        Args:
            processed: Processed event with changes
            
        Returns:
            True if update was applied successfully
        """
        try:
            if processed.entity_type == "patient":
                return self._update_patient(processed)
            elif processed.entity_type == "site":
                return self._update_site(processed)
            elif processed.entity_type == "study":
                return self._update_study(processed)
            else:
                logger.warning(f"Unknown entity type: {processed.entity_type}")
                return False
        except Exception as e:
            logger.error(f"Failed to apply event {processed.event_id}: {e}")
            return False
    
    def _update_patient(self, processed: ProcessedEvent) -> bool:
        """Update patient record."""
        patient_key = processed.entity_id
        
        # Build SQL update
        set_clauses = []
        params = {"patient_key": patient_key}
        
        for field, operation in processed.changes.items():
            op_type = operation[0]
            
            if op_type == "set":
                value = operation[1]
                set_clauses.append(f"{field} = :{field}")
                params[field] = value
            elif op_type == "increment":
                amount = operation[1]
                set_clauses.append(f"{field} = COALESCE({field}, 0) + {amount}")
            elif op_type == "decrement":
                amount = operation[1]
                set_clauses.append(f"{field} = GREATEST(0, COALESCE({field}, 0) - {amount})")
            elif op_type == "noop":
                continue
        
        if not set_clauses:
            return True  # Nothing to update
        
        sql = f"""
            UPDATE patients 
            SET {', '.join(set_clauses)}
            WHERE patient_key = :patient_key
        """
        
        with self.db_manager.engine.connect() as conn:
            result = conn.execute(text(sql), params)
            conn.commit()
            
            if result.rowcount > 0:
                self._update_count += 1
                logger.debug(f"Updated patient {patient_key}: {list(processed.changes.keys())}")
                return True
            else:
                logger.warning(f"Patient not found: {patient_key}")
                return False
    
    def _update_site(self, processed: ProcessedEvent) -> bool:
        """Update site record."""
        site_id = processed.entity_id
        
        set_clauses = []
        params = {"site_id": site_id}
        
        for field, operation in processed.changes.items():
            op_type = operation[0]
            if op_type == "set":
                set_clauses.append(f"{field} = :{field}")
                params[field] = operation[1]
        
        if not set_clauses:
            return True
        
        sql = f"""
            UPDATE sites 
            SET {', '.join(set_clauses)}
            WHERE site_id = :site_id
        """
        
        with self.db_manager.engine.connect() as conn:
            result = conn.execute(text(sql), params)
            conn.commit()
            return result.rowcount > 0
    
    def _update_study(self, processed: ProcessedEvent) -> bool:
        """Update study record."""
        study_id = processed.entity_id
        
        set_clauses = []
        params = {"study_id": study_id}
        
        for field, operation in processed.changes.items():
            op_type = operation[0]
            if op_type == "set":
                set_clauses.append(f"{field} = :{field}")
                params[field] = operation[1]
        
        if not set_clauses:
            return True
        
        sql = f"""
            UPDATE studies 
            SET {', '.join(set_clauses)}
            WHERE study_id = :study_id
        """
        
        with self.db_manager.engine.connect() as conn:
            result = conn.execute(text(sql), params)
            conn.commit()
            return result.rowcount > 0
    
    @property
    def update_count(self) -> int:
        """Get total number of updates applied."""
        return self._update_count
    
    def reset_count(self):
        """Reset update counter."""
        self._update_count = 0


# Singleton
_updater: Optional[UPRUpdater] = None


def get_upr_updater() -> UPRUpdater:
    """Get singleton UPR updater."""
    global _updater
    if _updater is None:
        _updater = UPRUpdater()
    return _updater
