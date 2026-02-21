"""
SANCHALAK AI - Stream Configuration
========================================
Kafka broker, topic, and consumer group configuration.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class StreamConfig:
    """Kafka streaming configuration."""
    
    # Kafka Enabled Flag - set to 'true' to use real Kafka
    enabled: bool = os.getenv("KAFKA_ENABLED", "false").lower() == "true"
    
    # Kafka Broker Settings
    bootstrap_servers: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    security_protocol: str = os.getenv("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT")
    
    # SASL authentication (for production Kafka like Confluent Cloud)
    sasl_mechanism: str = os.getenv("KAFKA_SASL_MECHANISM", "PLAIN")
    sasl_username: str = os.getenv("KAFKA_SASL_USERNAME", "")
    sasl_password: str = os.getenv("KAFKA_SASL_PASSWORD", "")
    
    # Consumer Settings
    consumer_group: str = os.getenv("KAFKA_CONSUMER_GROUP", "sanchalak-ai")
    auto_offset_reset: str = "earliest"
    enable_auto_commit: bool = True
    session_timeout_ms: int = 30000
    heartbeat_interval_ms: int = 10000
    
    # Producer Settings
    acks: str = "all"
    retries: int = 3
    batch_size: int = 16384
    linger_ms: int = 5
    
    # Topics
    topics: Dict[str, str] = field(default_factory=lambda: {
        # Query Events
        "query_created": "sanchalak.query.created",
        "query_resolved": "sanchalak.query.resolved",
        "query_escalated": "sanchalak.query.escalated",
        
        # Patient Events
        "patient_updated": "sanchalak.patient.updated",
        "patient_status_changed": "sanchalak.patient.status_changed",
        "patient_dqi_changed": "sanchalak.patient.dqi_changed",
        
        # Site Events
        "site_metrics_updated": "sanchalak.site.metrics_updated",
        "site_status_changed": "sanchalak.site.status_changed",
        
        # Issue Events
        "issue_detected": "sanchalak.issue.detected",
        "issue_resolved": "sanchalak.issue.resolved",
        "issue_escalated": "sanchalak.issue.escalated",
        
        # Real-Data Streams (riyaz2.md)
        "patient_vitals": "sanchalak.patient.vitals",
        "safety_sae": "sanchalak.safety.sae",
        
        # System Events
        "data_refresh": "sanchalak.system.data_refresh",
        "model_updated": "sanchalak.system.model_updated",
    })
    
    # Partitions per topic
    partitions: int = int(os.getenv("KAFKA_PARTITIONS", "3"))
    replication_factor: int = int(os.getenv("KAFKA_REPLICATION", "1"))
    
    def get_topic(self, event_type: str) -> str:
        """Get topic name for an event type."""
        return self.topics.get(event_type, f"sanchalak.unknown.{event_type}")
    
    def get_all_topics(self) -> List[str]:
        """Get all topic names."""
        return list(self.topics.values())
    
    def get_producer_config(self) -> Dict:
        """Get Kafka producer configuration."""
        return {
            "bootstrap_servers": self.bootstrap_servers,
            "security_protocol": self.security_protocol,
            "acks": self.acks,
            "retries": self.retries,
            "batch_size": self.batch_size,
            "linger_ms": self.linger_ms,
        }
    
    def get_consumer_config(self) -> Dict:
        """Get Kafka consumer configuration."""
        return {
            "bootstrap_servers": self.bootstrap_servers,
            "security_protocol": self.security_protocol,
            "group_id": self.consumer_group,
            "auto_offset_reset": self.auto_offset_reset,
            "enable_auto_commit": self.enable_auto_commit,
        }


# Singleton instance
_config: Optional[StreamConfig] = None


def get_stream_config() -> StreamConfig:
    """Get singleton stream configuration."""
    global _config
    if _config is None:
        _config = StreamConfig()
    return _config
