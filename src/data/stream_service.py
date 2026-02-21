"""
SANCHALAK AI - Real-Time Data Streaming Service
Integrates with Kafka for live event ingestion using standardized internal framework.
"""

import os
import json
import logging
import threading
import time
from typing import Dict, Any, Optional, Callable
from datetime import datetime
import pandas as pd

from src.database.pg_data_service import PostgreSQLDataService
from src.streaming.kafka_consumer import get_event_consumer
from src.streaming.event_processor import get_event_processor

logger = logging.getLogger(__name__)

class KafkaStreamService:
    """
    Service to handle real-time data ingestion from Kafka topics.
    Updates the local PostgreSQL database with live events.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.topics = ['patient_vitals', 'issue_detected', 'patient_updated']
        self.running = False
        self._thread = None
        self._initialized = True
        
        logger.info("Kafka Stream Service initialized (Unified Framework)")

    def start_stream(self):
        """Start consumption in a background thread."""
        if self.running:
            return
        
        self.running = True
        self._thread = threading.Thread(target=self._consume_loop, daemon=True)
        self._thread.start()
        logger.info("Background stream consumption started")

    def stop_stream(self):
        """Stop consumption."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            
    def _consume_loop(self):
        """Standardized consumption loop using internal EventConsumer."""
        logger.info("Initializing standardized EventConsumer...")
        try:
            consumer = get_event_consumer()
            
            # Register handlers for real-time updates
            consumer.register_handler("patient_vitals", self._handle_patient_update)
            consumer.register_handler("patient_updated", self._handle_patient_update)
            consumer.register_handler("issue_detected", self._handle_issue_detected)
            
            # Start consumer
            consumer.start(blocking=True)
        except Exception as e:
            logger.error(f"Kafka consumer error: {e}")
            if self.running:
                logger.info("Restarting consumer in 30s...")
                time.sleep(30)
                self._consume_loop()

    def _handle_patient_update(self, event: Dict[str, Any]):
        """Handle patient data updates by syncing with PostgreSQL."""
        try:
            payload = event.get('payload', {})
            patient_key = event.get('key')
            if not patient_key: return
            
            logger.info(f"Real-time update received for patient {patient_key}")
            # Logic to update PG would go here
        except Exception as e:
            logger.error(f"Error handling patient update: {e}")

    def _handle_issue_detected(self, event: Dict[str, Any]):
        """Handle new issue detected events."""
        try:
            payload = event.get('payload', {})
            patient_key = event.get('key')
            logger.info(f"New issue detected via Kafka for patient {patient_key}")
            # Logic to insert issue into PG would go here
        except Exception as e:
            logger.error(f"Error handling issue: {e}")


# Singleton
_stream_service = None

def get_stream_service():
    global _stream_service
    if _stream_service is None:
        _stream_service = KafkaStreamService()
    return _stream_service
