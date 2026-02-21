"""
SANCHALAK AI - Vector Store for RAG
========================================
ChromaDB-based vector store for semantic search and RAG.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Check ChromaDB availability
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("chromadb not installed - using mock vector store")


@dataclass
class SearchResult:
    """Result from vector search."""
    id: str
    text: str
    metadata: Dict[str, Any]
    score: float


class MockVectorStore:
    """Mock vector store when ChromaDB is not available."""
    
    def __init__(self):
        self._documents = []
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        self._documents.extend(documents)
    
    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        # Simple keyword matching
        query_lower = query.lower()
        results = []
        
        for i, doc in enumerate(self._documents[:top_k]):
            text = doc.get("text", "").lower()
            score = 1.0 if query_lower in text else 0.5
            results.append(SearchResult(
                id=doc.get("id", f"doc_{i}"),
                text=doc.get("text", "")[:500],
                metadata=doc.get("metadata", {}),
                score=score
            ))
        
        return results[:top_k]
    
    def count(self) -> int:
        return len(self._documents)


class VectorStore:
    """
    ChromaDB-based vector store for RAG capabilities.
    
    Provides:
    - Document embedding and storage
    - Semantic similarity search
    - Resolution genome pattern matching
    - Hybrid search (keyword + semantic)
    """
    
    _instance = None
    
    def __init__(self, persist_dir: Optional[str] = None):
        self._persist_dir = persist_dir or "data/vector_store"
        self._client = None
        self._collection = None
        self._initialized = False
        self._use_mock = not CHROMADB_AVAILABLE
        self._mock_store = MockVectorStore() if self._use_mock else None
    
    def initialize(self) -> bool:
        """Initialize vector store connection."""
        if self._initialized:
            return True
        
        if self._use_mock:
            self._initialized = True
            logger.info("VectorStore: Using mock store (ChromaDB not available)")
            return True
        
        try:
            # Create persist directory
            Path(self._persist_dir).mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB
            self._client = chromadb.Client(Settings(
                persist_directory=self._persist_dir,
                anonymized_telemetry=False
            ))
            
            # Get or create collection
            self._collection = self._client.get_or_create_collection(
                name="sanchalak_rag",
                metadata={"hnsw:space": "cosine"}
            )
            
            self._initialized = True
            logger.info(f"VectorStore: Initialized with {self._collection.count()} documents")
            return True
            
        except Exception as e:
            logger.error(f"VectorStore initialization failed: {e}")
            self._use_mock = True
            self._mock_store = MockVectorStore()
            self._initialized = True
            return True
    
    @property
    def is_ready(self) -> bool:
        if not self._initialized:
            self.initialize()
        return self._initialized
    
    @property
    def uses_mock(self) -> bool:
        return self._use_mock
    
    def count(self) -> int:
        """Get document count."""
        if self._use_mock:
            return self._mock_store.count()
        if self._collection:
            return self._collection.count()
        return 0
    
    # ============== Indexing Methods ==============
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> int:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of {"id", "text", "metadata"} dicts
            batch_size: Batch size for insertion
            
        Returns:
            Number of documents added
        """
        if not self.is_ready:
            return 0
        
        if self._use_mock:
            self._mock_store.add_documents(documents)
            return len(documents)
        
        try:
            added = 0
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i+batch_size]
                
                ids = [d["id"] for d in batch]
                texts = [d["text"] for d in batch]
                metadatas = [d.get("metadata", {}) for d in batch]
                
                self._collection.add(
                    ids=ids,
                    documents=texts,
                    metadatas=metadatas
                )
                added += len(batch)
            
            logger.info(f"Added {added} documents to vector store")
            return added
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return 0
    
    def add_resolution_patterns(self, patterns: List[Dict[str, Any]]) -> int:
        """Add resolution genome patterns to vector store."""
        documents = []
        
        for pattern in patterns:
            doc = {
                "id": f"pattern_{pattern.get('pattern_id', '')}",
                "text": f"{pattern.get('issue_type', '')} {pattern.get('description', '')} {pattern.get('resolution_steps', '')}",
                "metadata": {
                    "type": "resolution_pattern",
                    "issue_type": pattern.get("issue_type", ""),
                    "success_rate": pattern.get("success_rate", 0),
                    "avg_time_hours": pattern.get("avg_time_hours", 0)
                }
            }
            documents.append(doc)
        
        return self.add_documents(documents)
    
    def add_knowledge_docs(self, docs: List[Dict[str, Any]]) -> int:
        """Add knowledge base documents (SOPs, guidelines, protocols)."""
        documents = []
        
        for doc in docs:
            documents.append({
                "id": f"kb_{doc.get('id', '')}",
                "text": doc.get("content", doc.get("text", "")),
                "metadata": {
                    "type": "knowledge_base",
                    "source": doc.get("source", "unknown"),
                    "category": doc.get("category", "general"),
                    "title": doc.get("title", "")
                }
            })
        
        return self.add_documents(documents)
    
    # ============== Search Methods ==============
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_type: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Semantic search for similar documents.
        
        Args:
            query: Search query text
            top_k: Number of results
            filter_type: Optional filter by document type
            
        Returns:
            List of SearchResult objects
        """
        if not self.is_ready:
            return []
        
        if self._use_mock:
            return self._mock_store.search(query, top_k)
        
        try:
            where_filter = None
            if filter_type:
                where_filter = {"type": filter_type}
            
            results = self._collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_filter
            )
            
            search_results = []
            if results and results["ids"] and len(results["ids"]) > 0:
                for i, doc_id in enumerate(results["ids"][0]):
                    search_results.append(SearchResult(
                        id=doc_id,
                        text=results["documents"][0][i] if results["documents"] else "",
                        metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                        score=1.0 - (results["distances"][0][i] if results["distances"] else 0)
                    ))
            
            return search_results
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def search_resolutions(
        self,
        issue_type: str,
        context: str = "",
        top_k: int = 5
    ) -> List[SearchResult]:
        """Search for resolution patterns by issue type."""
        query = f"{issue_type} {context}".strip()
        return self.search(query, top_k, filter_type="resolution_pattern")
    
    def search_knowledge(
        self,
        query: str,
        source: Optional[str] = None,
        top_k: int = 5
    ) -> List[SearchResult]:
        """Search knowledge base documents."""
        results = self.search(query, top_k * 2, filter_type="knowledge_base")
        
        if source:
            results = [r for r in results if r.metadata.get("source") == source]
        
        return results[:top_k]
    
    # ============== RAG Methods ==============
    
    def get_context_for_query(
        self,
        query: str,
        max_tokens: int = 2000
    ) -> str:
        """
        Get relevant context for RAG augmentation.
        
        Args:
            query: User query
            max_tokens: Approximate max context size
            
        Returns:
            Concatenated context string
        """
        results = self.search(query, top_k=5)
        
        context_parts = []
        current_length = 0
        
        for result in results:
            text_length = len(result.text.split())
            if current_length + text_length > max_tokens // 4:
                break
            context_parts.append(f"[{result.metadata.get('type', 'doc')}] {result.text}")
            current_length += text_length
        
        return "\n\n".join(context_parts)
    
    def get_resolution_context(
        self,
        issue_type: str,
        patient_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get resolution context for an issue.
        
        Args:
            issue_type: Type of issue
            patient_context: Patient-specific context
            
        Returns:
            Resolution context with patterns and recommendations
        """
        # Search for similar patterns
        patterns = self.search_resolutions(issue_type, top_k=3)
        
        # Search for relevant knowledge
        knowledge = self.search_knowledge(issue_type, top_k=2)
        
        return {
            "issue_type": issue_type,
            "similar_patterns": [
                {
                    "id": p.id,
                    "description": p.text[:200],
                    "success_rate": p.metadata.get("success_rate", 0),
                    "score": p.score
                }
                for p in patterns
            ],
            "relevant_knowledge": [
                {
                    "source": k.metadata.get("source", ""),
                    "title": k.metadata.get("title", ""),
                    "excerpt": k.text[:200]
                }
                for k in knowledge
            ],
            "patient_context": patient_context
        }
    
    # ============== Utility Methods ==============
    
    def clear(self):
        """Clear all documents from the store."""
        if self._use_mock:
            self._mock_store._documents.clear()
            return
        
        if self._collection:
            # Delete and recreate collection
            self._client.delete_collection("sanchalak_rag")
            self._collection = self._client.create_collection(
                name="sanchalak_rag",
                metadata={"hnsw:space": "cosine"}
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return {
            "initialized": self._initialized,
            "using_mock": self._use_mock,
            "document_count": self.count(),
            "persist_dir": self._persist_dir
        }


# Singleton accessor
_store_instance: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Get singleton VectorStore instance."""
    global _store_instance
    if _store_instance is None:
        _store_instance = VectorStore()
        _store_instance.initialize()
    return _store_instance


def reset_vector_store():
    """Reset the singleton (for testing)."""
    global _store_instance
    _store_instance = None
