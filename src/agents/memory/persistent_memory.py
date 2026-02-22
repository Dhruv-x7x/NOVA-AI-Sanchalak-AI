"""
a6on-i - Persistent Agent Memory
===============================================
Store and retrieve agent learnings in PostgreSQL for cross-session memory.

Reference: riyaz.md - Improvement #22
"""

import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Memory:
    """A single memory/learning item."""
    memory_id: str
    agent_id: str
    context: str
    learning: str
    confidence: float
    created_at: datetime
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    relevance_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "agent_id": self.agent_id,
            "context": self.context[:200],
            "learning": self.learning,
            "confidence": round(self.confidence, 2),
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat() if self.last_accessed else None,
            "access_count": self.access_count,
            "relevance_score": round(self.relevance_score, 2)
        }


class PersistentAgentMemory:
    """
    Persistent agent memory with PostgreSQL storage.
    
    Features:
    - Store learnings across sessions
    - Semantic retrieval by context
    - Confidence decay over time
    - Memory consolidation
    
    Usage:
        memory = PersistentAgentMemory(agent_id="diagnostic_agent")
        
        # Store a learning
        memory.remember(
            context="Patient with high cascade impact and SDV incomplete",
            learning="SDV issues at this site often caused by staffing changes",
            confidence=0.85
        )
        
        # Recall relevant memories
        memories = memory.recall("SDV completion delays", limit=5)
    """
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self._memories: Dict[str, Memory] = {}
        self._db_manager = None
        self._embedding_model = None
        
        # Load existing memories from database
        self._load_from_database()
        
        logger.info(f"PersistentAgentMemory initialized for {agent_id}")
    
    def _get_db_manager(self):
        """Get database manager lazily."""
        if self._db_manager is None:
            try:
                from src.database.connection import get_db_manager
                self._db_manager = get_db_manager()
            except Exception:
                pass
        return self._db_manager
    
    def _get_embedding_model(self):
        """Get embedding model for semantic search."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception:
                pass
        return self._embedding_model
    
    def _load_from_database(self):
        """Load existing memories from PostgreSQL."""
        try:
            db = self._get_db_manager()
            if db and hasattr(db, 'engine'):
                from sqlalchemy import text
                with db.engine.connect() as conn:
                    result = conn.execute(text("""
                        SELECT memory_id, agent_id, context, learning, confidence,
                               created_at, last_accessed, access_count, metadata
                        FROM agent_memories
                        WHERE agent_id = :agent_id
                        ORDER BY confidence DESC, access_count DESC
                        LIMIT 1000
                    """), {"agent_id": self.agent_id})
                    
                    for row in result:
                        memory = Memory(
                            memory_id=row[0],
                            agent_id=row[1],
                            context=row[2],
                            learning=row[3],
                            confidence=row[4],
                            created_at=row[5],
                            last_accessed=row[6],
                            access_count=row[7] or 0,
                            metadata=eval(row[8]) if row[8] else {}
                        )
                        self._memories[memory.memory_id] = memory
                    
                    logger.info(f"Loaded {len(self._memories)} memories for {self.agent_id}")
        except Exception as e:
            logger.debug(f"Memory loading skipped: {e}")
    
    def remember(
        self,
        context: str,
        learning: str,
        confidence: float,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Store a learning in agent memory.
        
        Args:
            context: The context in which this learning applies
            learning: The actual learning/insight
            confidence: How confident the agent is (0-1)
            metadata: Additional metadata
            
        Returns:
            Memory ID
        """
        memory_id = f"MEM-{uuid.uuid4().hex[:8].upper()}"
        
        memory = Memory(
            memory_id=memory_id,
            agent_id=self.agent_id,
            context=context,
            learning=learning,
            confidence=min(1.0, max(0.0, confidence)),
            created_at=datetime.now(),
            metadata=metadata or {}
        )
        
        # Check for similar existing memories
        similar = self._find_similar(context, threshold=0.9)
        if similar:
            # Consolidate with existing memory
            existing = similar[0]
            existing.confidence = (existing.confidence + confidence) / 2
            existing.access_count += 1
            existing.last_accessed = datetime.now()
            self._persist_memory(existing)
            return existing.memory_id
        
        self._memories[memory_id] = memory
        self._persist_memory(memory)
        
        logger.info(f"Memory stored: {memory_id} for {self.agent_id}")
        
        return memory_id
    
    def _find_similar(self, context: str, threshold: float = 0.8) -> List[Memory]:
        """Find memories with similar context."""
        similar = []
        
        model = self._get_embedding_model()
        if model:
            try:
                import numpy as np
                
                query_embedding = model.encode(context)
                
                for memory in self._memories.values():
                    memory_embedding = model.encode(memory.context)
                    
                    # Cosine similarity
                    similarity = np.dot(query_embedding, memory_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(memory_embedding)
                    )
                    
                    if similarity >= threshold:
                        memory.relevance_score = float(similarity)
                        similar.append(memory)
            except Exception as e:
                logger.debug(f"Embedding similarity failed: {e}")
        else:
            # Fallback: simple keyword matching
            context_words = set(context.lower().split())
            for memory in self._memories.values():
                memory_words = set(memory.context.lower().split())
                overlap = len(context_words & memory_words) / max(1, len(context_words | memory_words))
                if overlap >= threshold:
                    memory.relevance_score = overlap
                    similar.append(memory)
        
        return sorted(similar, key=lambda x: x.relevance_score, reverse=True)
    
    def recall(self, context: str, limit: int = 5) -> List[Memory]:
        """
        Retrieve relevant memories for a context.
        
        Args:
            context: The current context to find relevant memories for
            limit: Maximum number of memories to return
            
        Returns:
            List of relevant Memory objects
        """
        # Find semantically similar memories
        similar = self._find_similar(context, threshold=0.3)[:limit * 2]
        
        # Also get high-confidence recent memories
        recent_confident = [
            m for m in self._memories.values()
            if m.confidence >= 0.7 and (datetime.now() - m.created_at).days < 30
        ]
        
        # Combine and deduplicate
        all_memories = list({m.memory_id: m for m in similar + recent_confident}.values())
        
        # Sort by relevance and confidence
        all_memories.sort(key=lambda x: (x.relevance_score * 0.6 + x.confidence * 0.4), reverse=True)
        
        # Update access count
        for memory in all_memories[:limit]:
            memory.access_count += 1
            memory.last_accessed = datetime.now()
        
        return all_memories[:limit]
    
    def _persist_memory(self, memory: Memory):
        """Persist memory to PostgreSQL."""
        try:
            db = self._get_db_manager()
            if db and hasattr(db, 'engine'):
                from sqlalchemy import text
                with db.engine.connect() as conn:
                    conn.execute(text("""
                        INSERT INTO agent_memories 
                        (memory_id, agent_id, context, learning, confidence,
                         created_at, last_accessed, access_count, metadata)
                        VALUES (:memory_id, :agent_id, :context, :learning, :confidence,
                                :created_at, :last_accessed, :access_count, :metadata)
                        ON CONFLICT (memory_id) DO UPDATE SET
                            confidence = :confidence,
                            last_accessed = :last_accessed,
                            access_count = :access_count
                    """), {
                        "memory_id": memory.memory_id,
                        "agent_id": memory.agent_id,
                        "context": memory.context,
                        "learning": memory.learning,
                        "confidence": memory.confidence,
                        "created_at": memory.created_at,
                        "last_accessed": memory.last_accessed,
                        "access_count": memory.access_count,
                        "metadata": str(memory.metadata)
                    })
                    conn.commit()
        except Exception as e:
            logger.debug(f"Memory persistence skipped: {e}")
    
    def forget(self, memory_id: str) -> bool:
        """Remove a memory."""
        if memory_id in self._memories:
            del self._memories[memory_id]
            
            try:
                db = self._get_db_manager()
                if db and hasattr(db, 'engine'):
                    from sqlalchemy import text
                    with db.engine.connect() as conn:
                        conn.execute(text(
                            "DELETE FROM agent_memories WHERE memory_id = :memory_id"
                        ), {"memory_id": memory_id})
                        conn.commit()
            except Exception:
                pass
            
            return True
        return False
    
    def consolidate(self, min_confidence: float = 0.3, max_age_days: int = 90):
        """
        Consolidate memories - remove low-confidence old memories.
        
        Args:
            min_confidence: Minimum confidence to keep
            max_age_days: Maximum age for low-confidence memories
        """
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        to_remove = []
        
        for memory_id, memory in self._memories.items():
            if memory.confidence < min_confidence and memory.created_at < cutoff_date:
                to_remove.append(memory_id)
        
        for memory_id in to_remove:
            self.forget(memory_id)
        
        logger.info(f"Consolidated {len(to_remove)} memories for {self.agent_id}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics."""
        confidences = [m.confidence for m in self._memories.values()]
        
        return {
            "agent_id": self.agent_id,
            "total_memories": len(self._memories),
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0,
            "high_confidence_count": sum(1 for c in confidences if c >= 0.7),
            "recent_count": sum(
                1 for m in self._memories.values()
                if (datetime.now() - m.created_at).days < 7
            )
        }


# Memory cache for agents
_agent_memories: Dict[str, PersistentAgentMemory] = {}


def get_agent_memory(agent_id: str) -> PersistentAgentMemory:
    """Get or create persistent memory for an agent."""
    if agent_id not in _agent_memories:
        _agent_memories[agent_id] = PersistentAgentMemory(agent_id)
    return _agent_memories[agent_id]
