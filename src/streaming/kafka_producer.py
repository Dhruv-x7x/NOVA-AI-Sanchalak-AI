"""
a6on-i - Kafka Event Producer
========================================
Publish events to Kafka topics for real-time processing.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict

from .stream_config import get_stream_config

logger = logging.getLogger(__name__)

# Try to import kafka, fall back to mock if not available
try:
    from kafka import KafkaProducer as RealKafkaProducer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    logger.warning("kafka-python not installed. Using mock producer.")


@dataclass
class Event:
    """Base event structure."""
    event_id: str
    event_type: str
    timestamp: str
    source: str
    payload: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class MockKafkaProducer:
    """Mock producer for when Kafka is not available."""
    
    def __init__(self, **kwargs):
        self.messages = []
        logger.info("MockKafkaProducer initialized (Kafka not available)")
    
    def send(self, topic: str, value: Any, key: Any = None):
        self.messages.append({"topic": topic, "value": value, "key": key})
        val_str = str(value)
        logger.debug(f"Mock sent to {topic}: {val_str[:100]}...")
        return self
    
    def flush(self):
        pass
    
    def close(self):
        pass


class EventProducer:
    """
    Kafka event producer for publishing a6on-i events.
    
    Usage:
        producer = get_event_producer()
        producer.publish_patient_update(patient_key, {"dqi_score": 95.5})
    """
    
    def __init__(self):
        self.config = get_stream_config()
        self._producer = None
        self._connected = False
    
    def connect(self) -> bool:
        """Connect to Kafka broker."""
        if self._producer is not None:
            return self._connected
        
        try:
            if KAFKA_AVAILABLE:
                self._producer = RealKafkaProducer(
                    bootstrap_servers=self.config.bootstrap_servers,
                    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                    key_serializer=lambda k: k.encode('utf-8') if k else None,
                    acks=self.config.acks,
                    retries=self.config.retries,
                )
                self._connected = True
                logger.info(f"Connected to Kafka: {self.config.bootstrap_servers}")
            else:
                self._producer = MockKafkaProducer()
                self._connected = True
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            self._producer = MockKafkaProducer()
            self._connected = True
        
        return self._connected
    
    def _ensure_connected(self):
        """Ensure producer is connected."""
        if not self._connected:
            self.connect()
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID."""
        import uuid
        return f"evt_{uuid.uuid4().hex[:12]}"
    
    def publish(self, event_type: str, key: str, payload: Dict[str, Any], source: str = "system") -> bool:
        """
        Publish an event to Kafka.
        
        Args:
            event_type: Type of event (e.g., 'patient_updated')
            key: Message key (usually entity ID)
            payload: Event data
            source: Event source identifier
            
        Returns:
            True if published successfully
        """
        self._ensure_connected()
        
        event = Event(
            event_id=self._generate_event_id(),
            event_type=event_type,
            timestamp=datetime.utcnow().isoformat(),
            source=source,
            payload=payload
        )
        
        topic = self.config.get_topic(event_type)
        
        try:
            self._producer.send(topic, value=event.to_dict(), key=key)
            logger.debug(f"Published {event_type} to {topic}")
            return True
        except Exception as e:
            logger.error(f"Failed to publish event: {e}")
            return False
    
    # Convenience methods for common events
    
    def publish_query_created(self, patient_key: str, query_data: Dict) -> bool:
        """Publish query created event."""
        return self.publish("query_created", patient_key, query_data, "query_service")
    
    def publish_query_resolved(self, patient_key: str, query_data: Dict) -> bool:
        """Publish query resolved event."""
        return self.publish("query_resolved", patient_key, query_data, "query_service")
    
    def publish_patient_updated(self, patient_key: str, changes: Dict) -> bool:
        """Publish patient data updated event."""
        return self.publish("patient_updated", patient_key, changes, "data_service")
    
    def publish_patient_status_changed(self, patient_key: str, old_status: str, new_status: str) -> bool:
        """Publish patient status change event."""
        return self.publish("patient_status_changed", patient_key, {
            "old_status": old_status,
            "new_status": new_status
        }, "status_service")
    
    def publish_patient_dqi_changed(self, patient_key: str, old_dqi: float, new_dqi: float) -> bool:
        """Publish DQI score change event."""
        return self.publish("patient_dqi_changed", patient_key, {
            "old_dqi": old_dqi,
            "new_dqi": new_dqi,
            "delta": new_dqi - old_dqi
        }, "dqi_service")
    
    def publish_site_metrics_updated(self, site_id: str, metrics: Dict) -> bool:
        """Publish site metrics update event."""
        return self.publish("site_metrics_updated", site_id, metrics, "site_service")
    
    def publish_issue_detected(self, patient_key: str, issue_data: Dict) -> bool:
        """Publish issue detected event."""
        return self.publish("issue_detected", patient_key, issue_data, "issue_detector")
    
    def publish_issue_resolved(self, patient_key: str, resolution_data: Dict) -> bool:
        """Publish issue resolved event."""
        return self.publish("issue_resolved", patient_key, resolution_data, "resolver")
    
    def publish_drift_detected(self, model_id: str, drift_data: Dict) -> bool:
        """Publish ML model drift detected event."""
        return self.publish("ml_drift_detected", model_id, drift_data, "drift_detector")
    
    def flush(self):
        """Flush pending messages."""
        if self._producer:
            self._producer.flush()
    
    def close(self):
        """Close the producer."""
        if self._producer:
            self._producer.close()
            self._producer = None
            self._connected = False


# Singleton instance
_producer: Optional[EventProducer] = None


def get_event_producer() -> EventProducer:
    """Get singleton event producer."""
    global _producer
    if _producer is None:
        _producer = EventProducer()
        _producer.connect()
    return _producer
