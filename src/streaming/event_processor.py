"""
SANCHALAK AI - Event Processor
===================================
Transform streaming events into actionable UPR updates.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ProcessedEvent:
    """Result of event processing."""
    event_id: str
    event_type: str
    entity_type: str  # patient, site, study
    entity_id: str
    action: str  # update, create, delete
    changes: Dict[str, Any]
    priority: str = "normal"  # low, normal, high, critical
    processed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class EventProcessor:
    """
    Process streaming events and prepare UPR updates.
    
    Transforms raw Kafka events into structured database operations.
    """
    
    def __init__(self):
        self._processors = {
            # Query events
            "query_created": self._process_query_created,
            "query_resolved": self._process_query_resolved,
            
            # Patient events
            "patient_updated": self._process_patient_updated,
            "patient_status_changed": self._process_patient_status_changed,
            "patient_dqi_changed": self._process_patient_dqi_changed,
            
            # Site events
            "site_metrics_updated": self._process_site_metrics_updated,
            
            # Issue events
            "issue_detected": self._process_issue_detected,
            "issue_resolved": self._process_issue_resolved,
        }
    
    def process(self, event: Dict[str, Any]) -> Optional[ProcessedEvent]:
        """
        Process a raw event and return structured update.
        
        Args:
            event: Raw event from Kafka
            
        Returns:
            ProcessedEvent with database changes, or None if not applicable
        """
        event_type = event.get("event_type", "unknown")
        processor = self._processors.get(event_type)
        
        if processor is None:
            logger.warning(f"No processor for event type: {event_type}")
            return None
        
        try:
            return processor(event)
        except Exception as e:
            logger.error(f"Failed to process {event_type}: {e}")
            return None
    
    def _process_query_created(self, event: Dict) -> ProcessedEvent:
        """Process query created event."""
        payload = event.get("payload", {})
        patient_key = payload.get("patient_key", event.get("key", "unknown"))
        
        return ProcessedEvent(
            event_id=event.get("event_id", ""),
            event_type="query_created",
            entity_type="patient",
            entity_id=patient_key,
            action="update",
            changes={
                "total_open_queries": ("increment", 1),
                "dm_queries": ("increment", 1) if payload.get("query_type") == "DM" else ("noop",),
            },
            priority="high"
        )
    
    def _process_query_resolved(self, event: Dict) -> ProcessedEvent:
        """Process query resolved event."""
        payload = event.get("payload", {})
        patient_key = payload.get("patient_key", event.get("key", "unknown"))
        
        return ProcessedEvent(
            event_id=event.get("event_id", ""),
            event_type="query_resolved",
            entity_type="patient",
            entity_id=patient_key,
            action="update",
            changes={
                "total_open_queries": ("decrement", 1),
                "dm_queries": ("decrement", 1) if payload.get("query_type") == "DM" else ("noop",),
            },
            priority="normal"
        )
    
    def _process_patient_updated(self, event: Dict) -> ProcessedEvent:
        """Process patient data update event."""
        payload = event.get("payload", {})
        patient_key = payload.get("patient_key", event.get("key", "unknown"))
        
        # Direct field updates
        changes = {k: ("set", v) for k, v in payload.get("changes", {}).items()}
        
        return ProcessedEvent(
            event_id=event.get("event_id", ""),
            event_type="patient_updated",
            entity_type="patient",
            entity_id=patient_key,
            action="update",
            changes=changes,
            priority="normal"
        )
    
    def _process_patient_status_changed(self, event: Dict) -> ProcessedEvent:
        """Process patient status change event."""
        payload = event.get("payload", {})
        patient_key = payload.get("patient_key", event.get("key", "unknown"))
        
        new_status = payload.get("new_status", "")
        
        # Determine actions based on new status
        changes = {
            "subject_status": ("set", new_status),
        }
        
        # Update flags based on status
        if new_status == "completed":
            changes["is_active"] = ("set", False)
        elif new_status == "withdrawn":
            changes["is_active"] = ("set", False)
        elif new_status == "enrolled":
            changes["is_active"] = ("set", True)
        
        return ProcessedEvent(
            event_id=event.get("event_id", ""),
            event_type="patient_status_changed",
            entity_type="patient",
            entity_id=patient_key,
            action="update",
            changes=changes,
            priority="high"
        )
    
    def _process_patient_dqi_changed(self, event: Dict) -> ProcessedEvent:
        """Process DQI score change event."""
        payload = event.get("payload", {})
        patient_key = payload.get("patient_key", event.get("key", "unknown"))
        
        new_dqi = payload.get("new_dqi", 0)
        
        # Determine DQI band
        if new_dqi >= 95:
            dqi_band = "Elite"
        elif new_dqi >= 90:
            dqi_band = "Optimal"
        elif new_dqi >= 85:
            dqi_band = "Standard"
        elif new_dqi >= 80:
            dqi_band = "Risk"
        else:
            dqi_band = "Critical"
        
        # Determine priority tier based on DQI
        if new_dqi < 80:
            priority_tier = "critical"
        elif new_dqi < 85:
            priority_tier = "high"
        elif new_dqi < 90:
            priority_tier = "medium"
        else:
            priority_tier = "low"
        
        return ProcessedEvent(
            event_id=event.get("event_id", ""),
            event_type="patient_dqi_changed",
            entity_type="patient",
            entity_id=patient_key,
            action="update",
            changes={
                "dqi_score": ("set", new_dqi),
                "dqi_band": ("set", dqi_band),
                "priority_tier": ("set", priority_tier),
            },
            priority="critical" if new_dqi < 80 else "normal"
        )
    
    def _process_site_metrics_updated(self, event: Dict) -> ProcessedEvent:
        """Process site metrics update event."""
        payload = event.get("payload", {})
        site_id = payload.get("site_id", event.get("key", "unknown"))
        
        changes = {k: ("set", v) for k, v in payload.get("metrics", {}).items()}
        
        return ProcessedEvent(
            event_id=event.get("event_id", ""),
            event_type="site_metrics_updated",
            entity_type="site",
            entity_id=site_id,
            action="update",
            changes=changes,
            priority="low"
        )
    
    def _process_issue_detected(self, event: Dict) -> ProcessedEvent:
        """Process issue detected event."""
        payload = event.get("payload", {})
        patient_key = payload.get("patient_key", event.get("key", "unknown"))
        
        return ProcessedEvent(
            event_id=event.get("event_id", ""),
            event_type="issue_detected",
            entity_type="patient",
            entity_id=patient_key,
            action="update",
            changes={
                "total_issues": ("increment", 1),
            },
            priority="high"
        )
    
    def _process_issue_resolved(self, event: Dict) -> ProcessedEvent:
        """Process issue resolved event."""
        payload = event.get("payload", {})
        patient_key = payload.get("patient_key", event.get("key", "unknown"))
        
        return ProcessedEvent(
            event_id=event.get("event_id", ""),
            event_type="issue_resolved",
            entity_type="patient",
            entity_id=patient_key,
            action="update",
            changes={
                "total_issues": ("decrement", 1),
            },
            priority="normal"
        )


# Singleton
_processor: Optional[EventProcessor] = None


def get_event_processor() -> EventProcessor:
    """Get singleton event processor."""
    global _processor
    if _processor is None:
        _processor = EventProcessor()
    return _processor
