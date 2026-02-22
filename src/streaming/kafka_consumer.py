"""
a6on-i - Kafka Event Consumer (Production-Ready)
============================================================
Subscribe to Kafka topics and process events in real-time.

Features:
- Real Kafka consumer with consumer groups
- Dead letter queue for failed messages
- Retry logic with exponential backoff
- Health checks and metrics
- Graceful shutdown

Per riyaz.md Phase 3: Streaming & Real-Time
"""

import json
import logging
import threading
import time
import os
from datetime import datetime
from typing import Dict, Any, Callable, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from .stream_config import get_stream_config

logger = logging.getLogger(__name__)

# Try to import kafka, fall back to mock if not available
try:
    from kafka import KafkaConsumer as RealKafkaConsumer
    from kafka import KafkaProducer as RealKafkaProducer
    from kafka.errors import KafkaError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    KafkaError = Exception  # Fallback
    logger.warning("kafka-python not installed. Using mock consumer.")


class MockKafkaConsumer:
    """Mock consumer for when Kafka is not available."""
    
    def __init__(self, *topics, **kwargs):
        self.topics = topics
        self.messages = []
        self._running = False
        logger.info(f"MockKafkaConsumer initialized for topics: {topics}")
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if not self._running or not self.messages:
            raise StopIteration
        return self.messages.pop(0)
    
    def subscribe(self, topics: List[str]):
        self.topics = topics
    
    def close(self):
        self._running = False
    
    def add_mock_message(self, topic: str, key: str, value: Dict):
        """Add a mock message for testing."""
        self.messages.append(MockMessage(topic, key, value))


@dataclass
class MockMessage:
    """Mock Kafka message."""
    topic: str
    key: str
    value: Dict
    
    def __post_init__(self):
        if isinstance(self.key, str):
            self.key = self.key.encode('utf-8')
        if isinstance(self.value, dict):
            self.value = json.dumps(self.value).encode('utf-8')


@dataclass
class ConsumerMetrics:
    """Track consumer metrics for monitoring."""
    messages_received: int = 0
    messages_processed: int = 0
    messages_failed: int = 0
    messages_retried: int = 0
    messages_to_dlq: int = 0
    last_message_time: Optional[str] = None
    processing_errors: Dict[str, int] = field(default_factory=lambda: defaultdict(int))


class DeadLetterQueue:
    """
    Dead Letter Queue for failed messages.
    
    Messages that fail processing after max retries are sent here
    for manual review or automated reprocessing later.
    """
    
    def __init__(self, dlq_topic: str = "a6on-i.dlq"):
        self.dlq_topic = dlq_topic
        self._producer = None
        self._local_queue: List[Dict] = []  # Fallback when Kafka unavailable
    
    def _get_producer(self):
        """Get or create Kafka producer for DLQ."""
        if self._producer is None and KAFKA_AVAILABLE:
            try:
                config = get_stream_config()
                self._producer = RealKafkaProducer(
                    bootstrap_servers=config.bootstrap_servers,
                    value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                    key_serializer=lambda k: k.encode('utf-8') if k else None
                )
            except Exception as e:
                logger.error(f"Failed to create DLQ producer: {e}")
        return self._producer
    
    def send(self, original_message: Dict, error: str, retry_count: int):
        """Send a failed message to the dead letter queue."""
        dlq_message = {
            "original_message": original_message,
            "error": str(error),
            "retry_count": retry_count,
            "failed_at": datetime.now().isoformat(),
            "source_topic": original_message.get("_source_topic", "unknown")
        }
        
        producer = self._get_producer()
        if producer:
            try:
                producer.send(
                    self.dlq_topic,
                    key=original_message.get("event_id", "unknown"),
                    value=dlq_message
                )
                producer.flush()
                logger.warning(f"Message sent to DLQ: {dlq_message['source_topic']}")
            except Exception as e:
                logger.error(f"Failed to send to DLQ: {e}")
                self._local_queue.append(dlq_message)
        else:
            # Store locally when Kafka unavailable
            self._local_queue.append(dlq_message)
            logger.warning(f"Message stored in local DLQ (Kafka unavailable)")
    
    def get_local_queue_size(self) -> int:
        """Get number of messages in local fallback queue."""
        return len(self._local_queue)
    
    def get_local_queue_messages(self, limit: int = 100) -> List[Dict]:
        """Retrieve messages from local queue for reprocessing."""
        return self._local_queue[:limit]
    
    def clear_local_queue(self) -> int:
        """Clear local queue and return count of cleared messages."""
        count = len(self._local_queue)
        self._local_queue = []
        return count


class EventConsumer:
    """
    Production-ready Kafka event consumer for processing a6on-i events.
    
    Features:
    - Consumer groups for horizontal scaling
    - Dead letter queue for failed messages
    - Retry with exponential backoff
    - Metrics and health checks
    - Graceful shutdown
    
    Usage:
        consumer = get_event_consumer()
        consumer.register_handler('patient_updated', my_handler)
        consumer.start()
    """
    
    # Retry configuration
    MAX_RETRIES = 3
    INITIAL_RETRY_DELAY = 1.0  # seconds
    MAX_RETRY_DELAY = 30.0  # seconds
    BACKOFF_MULTIPLIER = 2.0
    
    def __init__(self):
        self.config = get_stream_config()
        self._consumer = None
        self._handlers: Dict[str, List[Callable]] = {}
        self._running = False
        self._thread = None
        self._lock = threading.Lock()
        
        # Production features
        self.metrics = ConsumerMetrics()
        self.dlq = DeadLetterQueue()
        self._health_check_interval = 30  # seconds
        self._last_health_check = None
    
    def connect(self, topics: Optional[List[str]] = None) -> bool:
        """Connect to Kafka broker and subscribe to topics."""
        if topics is None:
            topics = self.config.get_all_topics()
        
        try:
            if KAFKA_AVAILABLE and self._should_use_real_kafka():
                self._consumer = RealKafkaConsumer(
                    *topics,
                    bootstrap_servers=self.config.bootstrap_servers,
                    group_id=self.config.consumer_group,
                    value_deserializer=lambda v: json.loads(v.decode('utf-8')),
                    key_deserializer=lambda k: k.decode('utf-8') if k else None,
                    auto_offset_reset=self.config.auto_offset_reset,
                    enable_auto_commit=self.config.enable_auto_commit,
                    # Production settings
                    session_timeout_ms=30000,
                    heartbeat_interval_ms=10000,
                    max_poll_interval_ms=300000,
                    max_poll_records=500,
                )
                logger.info(f"Connected to Kafka ({self.config.bootstrap_servers}), subscribed to {len(topics)} topics")
                return True
            else:
                self._consumer = MockKafkaConsumer(*topics)
                logger.info("Using mock consumer (Kafka not available or disabled)")
                return True
        except Exception as e:
            logger.error(f"Failed to connect to Kafka: {e}")
            self._consumer = MockKafkaConsumer(*topics)
            return True
    
    def _should_use_real_kafka(self) -> bool:
        """Check if real Kafka should be used based on environment."""
        # Check environment variable
        use_mock = os.environ.get("a6on-i_KAFKA_MOCK", "false").lower()
        if use_mock in ("true", "1", "yes"):
            return False
        
        # Check if bootstrap servers are configured
        if not self.config.bootstrap_servers or self.config.bootstrap_servers == "localhost:9092":
            # Check if there's a real server configured
            env_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS")
            if env_servers:
                self.config.bootstrap_servers = env_servers
                return True
            return False
        
        return True
    
    def register_handler(self, event_type: str, handler: Callable[[Dict], None]):
        """
        Register a handler function for an event type.
        
        Args:
            event_type: Type of event to handle
            handler: Function to call when event is received
        """
        with self._lock:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            self._handlers[event_type].append(handler)
        logger.debug(f"Registered handler for {event_type}")
    
    def unregister_handler(self, event_type: str, handler: Callable[[Dict], None]) -> bool:
        """Remove a handler for an event type."""
        with self._lock:
            if event_type in self._handlers and handler in self._handlers[event_type]:
                self._handlers[event_type].remove(handler)
                return True
        return False
    
    def _process_message_with_retry(self, message) -> bool:
        """Process a message with retry logic."""
        retry_count = 0
        delay = self.INITIAL_RETRY_DELAY
        last_error = None
        
        # Decode message
        try:
            if isinstance(message.value, bytes):
                event = json.loads(message.value.decode('utf-8'))
            else:
                event = message.value
            event["_source_topic"] = message.topic
        except Exception as e:
            logger.error(f"Failed to decode message: {e}")
            self.metrics.messages_failed += 1
            return False
        
        while retry_count <= self.MAX_RETRIES:
            try:
                self._process_event(event, message.topic)
                self.metrics.messages_processed += 1
                return True
            except Exception as e:
                last_error = e
                retry_count += 1
                self.metrics.messages_retried += 1
                
                if retry_count <= self.MAX_RETRIES:
                    logger.warning(
                        f"Processing failed (attempt {retry_count}/{self.MAX_RETRIES}), "
                        f"retrying in {delay:.1f}s: {e}"
                    )
                    time.sleep(delay)
                    delay = min(delay * self.BACKOFF_MULTIPLIER, self.MAX_RETRY_DELAY)
        
        # All retries exhausted - send to DLQ
        logger.error(f"Max retries exceeded, sending to DLQ: {last_error}")
        self.dlq.send(event, str(last_error), retry_count)
        self.metrics.messages_failed += 1
        self.metrics.messages_to_dlq += 1
        self.metrics.processing_errors[str(type(last_error).__name__)] += 1
        return False
    
    def _process_event(self, event: Dict, topic: str):
        """Process a single event by finding and calling appropriate handlers."""
        event_type = event.get('event_type', 'unknown')
        
        # Find handlers for this event type
        handlers = self._handlers.get(event_type, [])
        if not handlers:
            # Try topic-based matching
            for registered_type, topic_name in self.config.topics.items():
                if topic_name == topic:
                    handlers = self._handlers.get(registered_type, [])
                    break
        
        if not handlers:
            logger.debug(f"No handlers for event type: {event_type}")
            return
        
        # Execute handlers
        for handler in handlers:
            handler(event)  # Let exceptions propagate for retry logic
    
    def _consume_loop(self):
        """Main consumption loop with health checks."""
        logger.info("Starting event consumption loop")
        
        try:
            for message in self._consumer:
                if not self._running:
                    break
                
                self.metrics.messages_received += 1
                self.metrics.last_message_time = datetime.now().isoformat()
                
                self._process_message_with_retry(message)
                
                # Periodic health check
                self._check_health()
                
        except Exception as e:
            logger.error(f"Consumer loop error: {e}")
        
        logger.info("Event consumption loop stopped")
    
    def _check_health(self):
        """Perform periodic health checks."""
        now = time.time()
        if self._last_health_check is None or (now - self._last_health_check) > self._health_check_interval:
            self._last_health_check = now
            
            # Log metrics
            logger.info(
                f"Consumer health: received={self.metrics.messages_received}, "
                f"processed={self.metrics.messages_processed}, "
                f"failed={self.metrics.messages_failed}, "
                f"dlq={self.metrics.messages_to_dlq}"
            )
    
    def start(self, blocking: bool = False):
        """
        Start consuming events.
        
        Args:
            blocking: If True, block the current thread; otherwise run in background
        """
        if self._consumer is None:
            self.connect()
        
        self._running = True
        
        if blocking:
            self._consume_loop()
        else:
            self._thread = threading.Thread(target=self._consume_loop, daemon=True)
            self._thread.start()
            logger.info("Started consumer in background thread")
    
    def stop(self, timeout: float = 10.0):
        """Stop consuming events gracefully."""
        logger.info("Stopping consumer...")
        self._running = False
        
        if self._consumer:
            try:
                self._consumer.close()
            except Exception as e:
                logger.warning(f"Error closing consumer: {e}")
            self._consumer = None
        
        if self._thread:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("Consumer thread did not stop in time")
            self._thread = None
        
        logger.info("Consumer stopped")
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current consumer metrics."""
        return {
            "messages_received": self.metrics.messages_received,
            "messages_processed": self.metrics.messages_processed,
            "messages_failed": self.metrics.messages_failed,
            "messages_retried": self.metrics.messages_retried,
            "messages_to_dlq": self.metrics.messages_to_dlq,
            "last_message_time": self.metrics.last_message_time,
            "processing_errors": dict(self.metrics.processing_errors),
            "dlq_local_queue_size": self.dlq.get_local_queue_size(),
            "is_running": self._running,
            "handlers_registered": len(self._handlers)
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check and return status."""
        status = "healthy"
        issues = []
        
        # Check if running
        if not self._running:
            status = "stopped"
            issues.append("Consumer is not running")
        
        # Check error rate
        if self.metrics.messages_received > 0:
            error_rate = self.metrics.messages_failed / self.metrics.messages_received
            if error_rate > 0.1:  # More than 10% errors
                status = "degraded"
                issues.append(f"High error rate: {error_rate:.1%}")
        
        # Check DLQ backlog
        dlq_size = self.dlq.get_local_queue_size()
        if dlq_size > 100:
            status = "degraded" if status == "healthy" else status
            issues.append(f"DLQ backlog: {dlq_size} messages")
        
        return {
            "status": status,
            "issues": issues,
            "metrics": self.get_metrics(),
            "timestamp": datetime.now().isoformat()
        }


# Singleton instance
_consumer: Optional[EventConsumer] = None


def get_event_consumer() -> EventConsumer:
    """Get singleton event consumer."""
    global _consumer
    if _consumer is None:
        _consumer = EventConsumer()
    return _consumer
