"""
SANCHALAK AI - Conversation Memory v1.0
SQLite-based persistence for agent conversations.
# NOTE: This module previously used SQLite but has been migrated to PostgreSQL-only.
# If you need to use this module, update it to use:
#   from src.database.pg_data_service import get_pg_data_service
#   service = get_pg_data_service()


Features:
- Thread-based conversation storage
- Message history with metadata
- Context window management (configurable)
- Search/filter capabilities
"""

import json
import logging
# SQLite removed - using PostgreSQL
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import threading

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A single message in a conversation."""
    message_id: str
    thread_id: str
    role: str  # 'user', 'assistant', 'system', 'error'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'message_id': self.message_id,
            'thread_id': self.thread_id,
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


@dataclass
class Conversation:
    """A conversation thread."""
    thread_id: str
    title: str
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    messages: List[Message] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def message_count(self) -> int:
        return len(self.messages)
    
    @property
    def last_message(self) -> Optional[Message]:
        return self.messages[-1] if self.messages else None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'thread_id': self.thread_id,
            'title': self.title,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'message_count': self.message_count,
            'metadata': self.metadata
        }


class ConversationMemory:
    """
    Persistent conversation memory using SQLite.
    
    Usage:
        memory = get_conversation_memory()
        
        # Start or resume a conversation
        thread_id = memory.create_thread("My Query")
        
        # Add messages
        memory.add_message(thread_id, "user", "Show critical sites")
        memory.add_message(thread_id, "assistant", "Here are the sites...")
        
        # Get history
        messages = memory.get_messages(thread_id, limit=10)
        
        # List recent conversations
        threads = memory.list_threads(limit=20)
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, db_path: Optional[Path] = None):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, db_path: Optional[Path] = None):
        if self._initialized:
            return
        
        if db_path is None:
            db_path = Path(__file__).parent.parent.parent.parent / "data" / "collaboration" / "conversations.db"
        
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._context_window = 10  # Keep last N messages for context
        self._init_db()
        self._initialized = True
        logger.info(f"ConversationMemory initialized at {self.db_path}")
    
    def _init_db(self):
        """Initialize database schema."""
        # DISABLED: SQLite replaced with PostgreSQL
        if False:
             pass
    
    def create_thread(self, title: str, metadata: Optional[Dict] = None) -> str:
        """Create a new conversation thread."""
        import uuid
        thread_id = f"thread_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        now = datetime.now().isoformat()
        
        # DISABLED: SQLite replaced with PostgreSQL
        if False:
             pass
        
        logger.info(f"Created thread: {thread_id}")
        return thread_id
    
    def add_message(
        self,
        thread_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """Add a message to a thread."""
        import uuid
        message_id = f"msg_{uuid.uuid4().hex[:12]}"
        now = datetime.now().isoformat()
        
        # DISABLED: SQLite replaced with PostgreSQL
        if False:
             pass
        
        return message_id
    
    def get_messages(
        self,
        thread_id: str,
        limit: Optional[int] = None,
        include_system: bool = True
    ) -> List[Message]:
        """Get messages for a thread."""
        limit = limit or self._context_window
        
        rows = []
        # DISABLED: SQLite replaced with PostgreSQL
        if False:
             pass
        
        messages = []
        for row in reversed(rows):  # Reverse to get chronological order
            messages.append(Message(
                message_id=row[0],
                thread_id=row[1],
                role=row[2],
                content=row[3],
                timestamp=datetime.fromisoformat(row[4]),
                metadata=json.loads(row[5]) if row[5] else {}
            ))
        
        return messages
    
    def get_thread(self, thread_id: str) -> Optional[Conversation]:
        """Get a conversation thread with messages."""
        row = None
        # DISABLED: SQLite replaced with PostgreSQL
        if False:
             pass
        
        if not row:
            return None
        
        messages = self.get_messages(thread_id, limit=100)
        
        return Conversation(
            thread_id=row[0],
            title=row[1],
            created_at=datetime.fromisoformat(row[2]),
            updated_at=datetime.fromisoformat(row[3]),
            messages=messages,
            metadata=json.loads(row[4]) if row[4] else {}
        )
    
    def list_threads(self, limit: int = 20) -> List[Dict[str, Any]]:
        """List recent conversation threads."""
        rows = []
        # DISABLED: SQLite replaced with PostgreSQL
        if False:
             pass
        
        return [
            {
                'thread_id': row[0],
                'title': row[1],
                'created_at': row[2],
                'updated_at': row[3],
                'message_count': row[5],
                'preview': self._get_thread_preview(row[0])
            }
            for row in rows
        ]
    
    def _get_thread_preview(self, thread_id: str) -> str:
        """Get the first user message as preview."""
        row = None
        # DISABLED: SQLite replaced with PostgreSQL
        if False:
             pass
        
        if row:
            content = row[0]
            return content[:100] + '...' if len(content) > 100 else content
        return ""
    
    def delete_thread(self, thread_id: str):
        """Delete a thread and its messages."""
        # DISABLED: SQLite replaced with PostgreSQL
        if False:
             pass
        logger.info(f"Deleted thread: {thread_id}")
    
    def get_context_for_llm(self, thread_id: str, max_tokens: int = 4000) -> List[Dict[str, str]]:
        """Get messages formatted for LLM context."""
        messages = self.get_messages(thread_id, limit=self._context_window)
        
        context = []
        total_chars = 0
        char_limit = max_tokens * 4  # Approximate chars to tokens
        
        for msg in messages:
            if total_chars + len(msg.content) > char_limit:
                break
            context.append({
                'role': msg.role,
                'content': msg.content
            })
            total_chars += len(msg.content)
        
        return context
    
    def search_messages(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search messages across all threads."""
        rows = []
        # DISABLED: SQLite replaced with PostgreSQL
        if False:
             pass
        
        return [
            {
                'message_id': row[0],
                'thread_id': row[1],
                'role': row[2],
                'content': row[3][:200],
                'timestamp': row[4],
                'thread_title': row[5]
            }
            for row in rows
        ]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        thread_count = 0
        message_count = 0
        oldest = None
        # DISABLED: SQLite replaced with PostgreSQL
        if False:
             pass
        
        return {
            'total_threads': thread_count,
            'total_messages': message_count,
            'oldest_thread': oldest,
            'db_size_bytes': self.db_path.stat().st_size if self.db_path.exists() else 0
        }


# Singleton accessor
_memory_instance: Optional[ConversationMemory] = None


def get_conversation_memory() -> ConversationMemory:
    """Get the global ConversationMemory instance."""
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = ConversationMemory()
    return _memory_instance


# Export public API
__all__ = [
    'ConversationMemory',
    'Conversation',
    'Message',
    'get_conversation_memory'
]
