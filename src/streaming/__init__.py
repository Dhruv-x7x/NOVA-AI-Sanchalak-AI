"""
a6on-i - Streaming Package
=====================================
Real-time event streaming with Apache Kafka.
"""

from .stream_config import StreamConfig, get_stream_config
from .kafka_producer import EventProducer, get_event_producer
from .kafka_consumer import EventConsumer, get_event_consumer
from .event_processor import EventProcessor, get_event_processor
from .upr_updater import UPRUpdater, get_upr_updater

__all__ = [
    'StreamConfig',
    'get_stream_config',
    'EventProducer',
    'get_event_producer',
    'EventConsumer',
    'get_event_consumer',
    'EventProcessor',
    'get_event_processor',
    'UPRUpdater',
    'get_upr_updater',
]
