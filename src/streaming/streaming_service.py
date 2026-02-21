"""
SANCHALAK AI - Unified Streaming Service
=============================================
Provides unified access to Kafka streaming for the dashboard.
"""

import os
import logging
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class StreamStatus:
    """Status of the streaming service."""
    connected: bool
    broker: str
    last_event_time: Optional[datetime]
    events_sent: int
    events_received: int
    error: Optional[str]


class StreamingService:
    """
    Unified streaming service for Sanchalak AI.
    
    Provides:
    - Event publishing (producer)
    - Event subscription (consumer)
    - Connection status tracking
    - Graceful fallback when Kafka unavailable
    """
    
    _instance = None
    
    def __init__(self):
        self._producer = None
        self._consumer = None
        self._connected = False
        self._last_event_time = None
        self._events_sent = 0
        self._events_received = 0
        self._error = None
        self._subscribers: Dict[str, List[Callable]] = {}
        self._consumer_thread = None
        
        # Initialize
        self._initialize()
    
    def _initialize(self):
        """Initialize streaming connections."""
        try:
            from .kafka_producer import get_event_producer
            self._producer = get_event_producer()
            self._connected = self._producer._connected
            
            if self._connected:
                logger.info("StreamingService: Producer connected")
        except Exception as e:
            logger.warning(f"StreamingService: Producer init failed: {e}")
            self._error = str(e)
            self._connected = False
    
    @property
    def is_connected(self) -> bool:
        """Check if streaming is connected."""
        return self._connected
    
    @property
    def uses_mock(self) -> bool:
        """Check if using mock producer."""
        if self._producer:
            from .kafka_producer import KAFKA_AVAILABLE
            return not KAFKA_AVAILABLE
        return True
    
    def get_status(self) -> StreamStatus:
        """Get current streaming status."""
        return StreamStatus(
            connected=self._connected,
            broker="localhost:9092",  # From config
            last_event_time=self._last_event_time,
            events_sent=self._events_sent,
            events_received=self._events_received,
            error=self._error
        )
    
    # ============== Publishing Methods ==============
    
    def publish_patient_update(self, patient_key: str, changes: Dict[str, Any]) -> bool:
        """Publish patient update event."""
        if not self._producer:
            return False
        
        success = self._producer.publish_patient_updated(patient_key, changes)
        if success:
            self._events_sent += 1
            self._last_event_time = datetime.utcnow()
        return success
    
    def publish_issue_detected(self, patient_key: str, issue_data: Dict[str, Any]) -> bool:
        """Publish issue detected event."""
        if not self._producer:
            return False
        
        success = self._producer.publish_issue_detected(patient_key, issue_data)
        if success:
            self._events_sent += 1
            self._last_event_time = datetime.utcnow()
        return success
    
    def publish_issue_resolved(self, patient_key: str, resolution_data: Dict[str, Any]) -> bool:
        """Publish issue resolved event."""
        if not self._producer:
            return False
        
        success = self._producer.publish_issue_resolved(patient_key, resolution_data)
        if success:
            self._events_sent += 1
            self._last_event_time = datetime.utcnow()
        return success
    
    def publish_dqi_change(self, patient_key: str, old_dqi: float, new_dqi: float) -> bool:
        """Publish DQI change event."""
        if not self._producer:
            return False
        
        success = self._producer.publish_patient_dqi_changed(patient_key, old_dqi, new_dqi)
        if success:
            self._events_sent += 1
            self._last_event_time = datetime.utcnow()
        return success
    
    def publish_site_update(self, site_id: str, metrics: Dict[str, Any]) -> bool:
        """Publish site metrics update."""
        if not self._producer:
            return False
        
        success = self._producer.publish_site_metrics_updated(site_id, metrics)
        if success:
            self._events_sent += 1
            self._last_event_time = datetime.utcnow()
        return success
    
    def publish_drift_detected(self, model_id: str, drift_data: Dict[str, Any]) -> bool:
        """Publish ML model drift detection event."""
        if not self._producer:
            return False
            
        success = self._producer.publish_drift_detected(model_id, drift_data)
        if success:
            self._events_sent += 1
            self._last_event_time = datetime.utcnow()
        return success
    
    # ============== Subscription Methods ==============
    
    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to an event type."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)
    
    def unsubscribe(self, event_type: str, callback: Callable):
        """Unsubscribe from an event type."""
        if event_type in self._subscribers:
            self._subscribers[event_type] = [
                cb for cb in self._subscribers[event_type] if cb != callback
            ]
    
    def _notify_subscribers(self, event_type: str, data: Dict[str, Any]):
        """Notify all subscribers of an event."""
        if event_type in self._subscribers:
            for callback in self._subscribers[event_type]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Subscriber callback error: {e}")
    
    # ============== Consumer Management ==============
    
    def start_consumer(self, topics: List[str] = None):
        """Start the consumer thread."""
        import threading
        
        if self._consumer_thread and self._consumer_thread.is_alive():
            logger.warning("Consumer already running")
            return
        
        default_topics = [
            "patient_updates",
            "issue_events", 
            "site_updates"
        ]
        
        def consumer_loop():
            try:
                from .kafka_consumer import get_event_consumer
                consumer = get_event_consumer()
                consumer.start(
                    topics=topics or default_topics,
                    callback=self._handle_event
                )
            except Exception as e:
                logger.error(f"Consumer loop error: {e}")
                self._error = str(e)
        
        self._consumer_thread = threading.Thread(target=consumer_loop, daemon=True)
        self._consumer_thread.start()
        logger.info("Consumer thread started")
    
    def stop_consumer(self):
        """Stop the consumer thread."""
        if self._consumer:
            try:
                self._consumer.stop()
            except Exception:
                pass
        logger.info("Consumer stopped")
    
    def _handle_event(self, event: Dict[str, Any]):
        """Handle incoming event from consumer."""
        self._events_received += 1
        self._last_event_time = datetime.utcnow()
        
        event_type = event.get("event_type", "unknown")
        self._notify_subscribers(event_type, event)
    
    # ============== Utility Methods ==============
    
    def flush(self):
        """Flush pending messages."""
        if self._producer:
            self._producer.flush()
    
    def close(self):
        """Close streaming connections."""
        self.stop_consumer()
        if self._producer:
            self._producer.close()
        self._connected = False


# Singleton accessor
_service_instance: Optional[StreamingService] = None


def get_streaming_service() -> StreamingService:
    """Get singleton StreamingService instance."""
    global _service_instance
    if _service_instance is None:
        _service_instance = StreamingService()
    return _service_instance


def reset_streaming_service():
    """Reset the singleton (for testing)."""
    global _service_instance
    if _service_instance:
        _service_instance.close()
    _service_instance = None
