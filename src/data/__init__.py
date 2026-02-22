"""
Data processing modules for a6on-i.

Modules are imported explicitly to avoid circular imports and RuntimeWarnings.
Use: from src.data.ingestion import DataIngestionEngine
"""

__all__ = ['DataIngestionEngine', 'DataCleaningEngine', 'UPRBuilder']

# Do NOT auto-import here - import explicitly when needed
