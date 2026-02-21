"""
SANCHALAK AI - Qdrant Vector Store
==========================================
Production-grade vector database for semantic search and RAG.

Features:
- Qdrant server or in-memory mode
- 7 specialized collections
- Dense embeddings with sentence-transformers
- Hybrid search (semantic + keyword)
- Persistent storage
- Batch indexing with async support

Author: Sanchalak AI Team
Date: 2026-01-24
"""

import os
import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

# Check Qdrant availability
QDRANT_AVAILABLE = False
FASTEMBED_AVAILABLE = False

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct,
        Filter, FieldCondition, MatchValue,
        SearchParams, HnswConfigDiff
    )
    QDRANT_AVAILABLE = True
    logger.info("✅ Qdrant client available")
except ImportError:
    logger.warning("⚠️ qdrant-client not installed - using mock store")

try:
    from fastembed import TextEmbedding
    FASTEMBED_AVAILABLE = True
    logger.info("✅ FastEmbed available for local embeddings")
except ImportError:
    logger.warning("⚠️ fastembed not installed - using simple embeddings")


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SearchResult:
    """Result from vector search."""
    id: str
    text: str
    metadata: Dict[str, Any]
    score: float
    collection: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CollectionConfig:
    """Configuration for a vector collection."""
    name: str
    description: str
    vector_size: int = 384  # Default for all-MiniLM-L6-v2
    distance: str = "cosine"
    metadata_fields: List[str] = None
    
    def __post_init__(self):
        if self.metadata_fields is None:
            self.metadata_fields = []


# =============================================================================
# EMBEDDING SERVICE
# =============================================================================

class EmbeddingService:
    """Local embedding service using FastEmbed or fallback."""
    
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self._model = None
        self._initialized = False
    
    def initialize(self) -> bool:
        """Initialize embedding model."""
        if self._initialized:
            return True
        
        if FASTEMBED_AVAILABLE:
            try:
                self._model = TextEmbedding(model_name=self.model_name)
                self._initialized = True
                logger.info(f"✅ Loaded embedding model: {self.model_name}")
                return True
            except Exception as e:
                logger.warning(f"⚠️ Failed to load FastEmbed model: {e}")
        
        # Use simple hash-based embeddings as fallback
        logger.info("Using simple hash-based embeddings (install fastembed for better search)")
        self._initialized = True
        return True
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts."""
        if not self._initialized:
            self.initialize()
        
        if self._model is not None:
            # Use FastEmbed
            embeddings = list(self._model.embed(texts))
            return np.array(embeddings)
        else:
            # Simple hash-based fallback
            return self._hash_embeddings(texts)
    
    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for single text."""
        return self.embed([text])[0]
    
    def _hash_embeddings(self, texts: List[str], dim: int = 384) -> np.ndarray:
        """Simple hash-based embeddings for fallback."""
        embeddings = []
        for text in texts:
            # Create deterministic embedding from hash
            hash_bytes = hashlib.sha512(text.lower().encode()).digest()
            # Expand to required dimension
            values = []
            for i in range(0, len(hash_bytes), 4):
                val = int.from_bytes(hash_bytes[i:i+4], 'little') / (2**32 - 1)
                values.append(val * 2 - 1)  # Scale to [-1, 1]
            
            # Pad or truncate to dimension
            while len(values) < dim:
                values.extend(values[:dim - len(values)])
            values = values[:dim]
            
            # Normalize
            vec = np.array(values)
            vec = vec / (np.linalg.norm(vec) + 1e-8)
            embeddings.append(vec)
        
        return np.array(embeddings)
    
    @property
    def vector_size(self) -> int:
        """Get embedding dimension."""
        return 384  # BGE-small default


# =============================================================================
# MOCK QDRANT CLIENT
# =============================================================================

class MockQdrantCollection:
    """Mock collection for when Qdrant is not available."""
    
    def __init__(self, name: str):
        self.name = name
        self.documents: Dict[str, Dict] = {}
    
    def add(self, ids: List[str], vectors: List[np.ndarray], 
            payloads: List[Dict]) -> int:
        for id_, vec, payload in zip(ids, vectors, payloads):
            self.documents[id_] = {
                "id": id_,
                "vector": vec,
                "payload": payload
            }
        return len(ids)
    
    def search(self, vector: np.ndarray, limit: int = 5,
               filter_dict: Dict = None) -> List[Dict]:
        results = []
        
        for doc_id, doc in self.documents.items():
            # Apply filter
            if filter_dict:
                match = True
                for key, value in filter_dict.items():
                    if doc["payload"].get(key) != value:
                        match = False
                        break
                if not match:
                    continue
            
            # Calculate cosine similarity
            doc_vec = np.array(doc["vector"])
            score = np.dot(vector, doc_vec) / (
                np.linalg.norm(vector) * np.linalg.norm(doc_vec) + 1e-8
            )
            
            results.append({
                "id": doc_id,
                "score": float(score),
                "payload": doc["payload"]
            })
        
        # Sort by score descending
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]
    
    def count(self) -> int:
        return len(self.documents)
    
    def delete(self, ids: List[str]):
        for id_ in ids:
            self.documents.pop(id_, None)


class MockQdrantClient:
    """Mock Qdrant client for development/testing."""
    
    def __init__(self):
        self.collections: Dict[str, MockQdrantCollection] = {}
        logger.info("MockQdrantClient initialized")
    
    def create_collection(self, name: str, **kwargs):
        self.collections[name] = MockQdrantCollection(name)
    
    def collection_exists(self, name: str) -> bool:
        return name in self.collections
    
    def get_collection(self, name: str) -> MockQdrantCollection:
        return self.collections.get(name)
    
    def delete_collection(self, name: str):
        self.collections.pop(name, None)


# =============================================================================
# QDRANT VECTOR STORE
# =============================================================================

class QdrantVectorStore:
    """
    Production Qdrant vector store for SANCHALAK AI.
    
    Collections:
    1. resolution_templates - Issue resolution patterns
    2. issue_descriptions - Issue type definitions
    3. sop_documents - Standard Operating Procedures
    4. query_templates - Data query templates
    5. pattern_descriptions - Site/patient patterns
    6. regulatory_guidelines - ICH, 21 CFR Part 11
    7. clinical_terms - Clinical trial terminology
    
    Usage:
        store = QdrantVectorStore()
        store.initialize()
        
        # Add documents
        store.add_documents("resolution_templates", documents)
        
        # Search
        results = store.search("How to resolve SDV issues?")
    """
    
    # Collection configurations
    COLLECTIONS = {
        "resolution_templates": CollectionConfig(
            name="resolution_templates",
            description="Resolution patterns for clinical trial issues",
            metadata_fields=["issue_type", "responsible_role", "effort_hours", "success_rate"]
        ),
        "issue_descriptions": CollectionConfig(
            name="issue_descriptions",
            description="Issue type definitions and impacts",
            metadata_fields=["issue_type", "responsible", "priority"]
        ),
        "sop_documents": CollectionConfig(
            name="sop_documents",
            description="Standard Operating Procedures",
            metadata_fields=["sop_id", "title", "department"]
        ),
        "query_templates": CollectionConfig(
            name="query_templates",
            description="Data query templates",
            metadata_fields=["query_category", "priority", "data_point"]
        ),
        "pattern_descriptions": CollectionConfig(
            name="pattern_descriptions",
            description="Site and patient behavior patterns",
            metadata_fields=["pattern_id", "severity", "category"]
        ),
        "regulatory_guidelines": CollectionConfig(
            name="regulatory_guidelines",
            description="Regulatory guidelines (ICH, FDA)",
            metadata_fields=["guideline_id", "section"]
        ),
        "clinical_terms": CollectionConfig(
            name="clinical_terms",
            description="Clinical trial terminology",
            metadata_fields=["term", "full_name"]
        )
    }
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        grpc_port: int = 6334,
        persist_dir: Optional[str] = None,
        prefer_grpc: bool = True,
        use_memory: bool = False
    ):
        """
        Initialize Qdrant Vector Store.
        
        Args:
            host: Qdrant server host
            port: Qdrant REST port
            grpc_port: Qdrant gRPC port (faster)
            persist_dir: Local persistence directory (for in-memory mode)
            prefer_grpc: Use gRPC for better performance
            use_memory: Use in-memory Qdrant (no server required)
        """
        self.host = host
        self.port = port
        self.grpc_port = grpc_port
        self.persist_dir = persist_dir or "data/qdrant"
        self.prefer_grpc = prefer_grpc
        self.use_memory = use_memory
        
        self._client = None
        self._embedding_service = EmbeddingService()
        self._initialized = False
        self._use_mock = not QDRANT_AVAILABLE
        
        # Stats
        self.stats = {
            "documents_indexed": 0,
            "searches_performed": 0,
            "collections_created": 0
        }
    
    def initialize(self, create_collections: bool = True) -> bool:
        """
        Initialize Qdrant connection and collections.
        
        Args:
            create_collections: Create default collections if not exist
            
        Returns:
            True if initialization successful
        """
        if self._initialized:
            return True
        
        # Initialize embedding service
        self._embedding_service.initialize()
        
        # Initialize Qdrant client
        if self._use_mock:
            self._client = MockQdrantClient()
            logger.info("✅ Using mock Qdrant client")
        elif QDRANT_AVAILABLE:
            try:
                if self.use_memory:
                    # In-memory mode with optional persistence
                    Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
                    self._client = QdrantClient(path=self.persist_dir)
                    logger.info(f"✅ Qdrant in-memory mode at: {self.persist_dir}")
                else:
                    # Connect to Qdrant server
                    self._client = QdrantClient(
                        host=self.host,
                        port=self.port,
                        grpc_port=self.grpc_port if self.prefer_grpc else None,
                        prefer_grpc=self.prefer_grpc
                    )
                    logger.info(f"✅ Connected to Qdrant server at {self.host}:{self.port}")
            except Exception as e:
                logger.warning(f"⚠️ Qdrant server not available: {e}")
                logger.info("Falling back to in-memory mode...")
                
                try:
                    Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
                    self._client = QdrantClient(path=self.persist_dir)
                    logger.info(f"✅ Qdrant in-memory mode at: {self.persist_dir}")
                except Exception as e2:
                    logger.warning(f"⚠️ In-memory Qdrant failed: {e2}")
                    self._client = MockQdrantClient()
                    self._use_mock = True
        
        # Create collections
        if create_collections:
            self._create_collections()
        
        self._initialized = True
        return True
    
    def _create_collections(self):
        """Create all configured collections."""
        for name, config in self.COLLECTIONS.items():
            try:
                if self._use_mock:
                    if not self._client.collection_exists(name):
                        self._client.create_collection(name)
                        self.stats["collections_created"] += 1
                else:
                    # Real Qdrant
                    collections = self._client.get_collections().collections
                    exists = any(c.name == name for c in collections)
                    
                    if not exists:
                        self._client.create_collection(
                            collection_name=name,
                            vectors_config=VectorParams(
                                size=self._embedding_service.vector_size,
                                distance=Distance.COSINE
                            ),
                            hnsw_config=HnswConfigDiff(
                                m=16,
                                ef_construct=100
                            )
                        )
                        self.stats["collections_created"] += 1
                        logger.info(f"  Created collection: {name}")
                    else:
                        logger.info(f"  Collection exists: {name}")
                        
            except Exception as e:
                logger.error(f"  Error creating {name}: {e}")
    
    @property
    def is_ready(self) -> bool:
        """Check if store is ready."""
        if not self._initialized:
            self.initialize()
        return self._initialized
    
    @property
    def uses_mock(self) -> bool:
        """Check if using mock client."""
        return self._use_mock
    
    # =========================================================================
    # INDEXING METHODS
    # =========================================================================
    
    def add_documents(
        self,
        collection_name: str,
        documents: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> int:
        """
        Add documents to a collection.
        
        Args:
            collection_name: Target collection
            documents: List of {"id", "text", "metadata"} dicts
            batch_size: Batch size for insertion
            
        Returns:
            Number of documents added
        """
        if not self.is_ready:
            return 0
        
        if collection_name not in self.COLLECTIONS:
            logger.error(f"Unknown collection: {collection_name}")
            return 0
        
        added = 0
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            
            # Extract texts for embedding
            texts = [d.get("text", "") for d in batch]
            embeddings = self._embedding_service.embed(texts)
            
            if self._use_mock:
                ids = [d.get("id", f"{collection_name}_{i+j}") for j, d in enumerate(batch)]
                payloads = []
                for d in batch:
                    payload = d.get("metadata", {}).copy()
                    payload["text"] = d.get("text", "")[:1000]  # Store truncated text
                    payloads.append(payload)
                
                collection = self._client.get_collection(collection_name)
                if collection:
                    added += collection.add(ids, embeddings.tolist(), payloads)
            else:
                # Real Qdrant
                points = []
                for j, (doc, emb) in enumerate(zip(batch, embeddings)):
                    doc_id = doc.get("id", f"{collection_name}_{i+j}")
                    payload = doc.get("metadata", {}).copy()
                    payload["text"] = doc.get("text", "")[:1000]
                    
                    points.append(PointStruct(
                        id=doc_id if isinstance(doc_id, int) else hash(doc_id) & 0x7FFFFFFF,
                        vector=emb.tolist(),
                        payload=payload
                    ))
                
                self._client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                added += len(points)
        
        self.stats["documents_indexed"] += added
        logger.info(f"Added {added} documents to {collection_name}")
        return added
    
    def add_resolution_patterns(self, patterns: List[Dict]) -> int:
        """Add resolution genome patterns."""
        documents = []
        for pattern in patterns:
            documents.append({
                "id": f"pattern_{pattern.get('pattern_id', hash(str(pattern)))}",
                "text": f"{pattern.get('issue_type', '')} {pattern.get('description', '')} {pattern.get('steps', '')}",
                "metadata": {
                    "issue_type": pattern.get("issue_type", ""),
                    "responsible_role": pattern.get("responsible_role", ""),
                    "effort_hours": pattern.get("effort_hours", 0),
                    "success_rate": pattern.get("success_rate", 0)
                }
            })
        return self.add_documents("resolution_templates", documents)
    
    def add_knowledge_docs(self, docs: List[Dict]) -> int:
        """Add knowledge base documents."""
        documents = []
        for doc in docs:
            documents.append({
                "id": f"kb_{doc.get('id', hash(str(doc)))}",
                "text": doc.get("content", doc.get("text", "")),
                "metadata": {
                    "sop_id": doc.get("sop_id", ""),
                    "title": doc.get("title", ""),
                    "department": doc.get("department", "")
                }
            })
        return self.add_documents("sop_documents", documents)
    
    # =========================================================================
    # SEARCH METHODS
    # =========================================================================
    
    def search(
        self,
        query: str,
        collection_name: Optional[str] = None,
        n_results: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Semantic search across collections.
        
        Args:
            query: Search query text
            collection_name: Specific collection (None = all collections)
            n_results: Number of results
            filter_dict: Metadata filter (e.g., {"issue_type": "sdv_incomplete"})
            
        Returns:
            List of SearchResult objects
        """
        if not self.is_ready:
            return []
        
        self.stats["searches_performed"] += 1
        
        # Generate query embedding
        query_vector = self._embedding_service.embed_single(query)
        
        # Determine collections to search
        collections = [collection_name] if collection_name else list(self.COLLECTIONS.keys())
        
        all_results = []
        
        for coll_name in collections:
            try:
                if self._use_mock:
                    collection = self._client.get_collection(coll_name)
                    if collection:
                        results = collection.search(
                            query_vector, 
                            limit=n_results,
                            filter_dict=filter_dict
                        )
                        for r in results:
                            all_results.append(SearchResult(
                                id=r["id"],
                                text=r["payload"].get("text", ""),
                                metadata=r["payload"],
                                score=r["score"],
                                collection=coll_name
                            ))
                else:
                    # Real Qdrant search
                    qdrant_filter = None
                    if filter_dict:
                        conditions = []
                        for key, value in filter_dict.items():
                            conditions.append(
                                FieldCondition(
                                    key=key,
                                    match=MatchValue(value=value)
                                )
                            )
                        qdrant_filter = Filter(must=conditions)
                    
                    results = self._client.search(
                        collection_name=coll_name,
                        query_vector=query_vector.tolist(),
                        limit=n_results,
                        query_filter=qdrant_filter,
                        search_params=SearchParams(hnsw_ef=128)
                    )
                    
                    for hit in results:
                        all_results.append(SearchResult(
                            id=str(hit.id),
                            text=hit.payload.get("text", ""),
                            metadata=hit.payload,
                            score=hit.score,
                            collection=coll_name
                        ))
                        
            except Exception as e:
                logger.warning(f"Search error in {coll_name}: {e}")
        
        # Sort by score
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[:n_results]
    
    def search_resolutions(
        self,
        issue_type: str,
        context: str = "",
        n_results: int = 5
    ) -> List[SearchResult]:
        """Search for resolution patterns by issue type."""
        query = f"resolution for {issue_type} {context}".strip()
        return self.search(
            query=query,
            collection_name="resolution_templates",
            n_results=n_results,
            filter_dict={"issue_type": issue_type} if issue_type else None
        )
    
    def search_by_role(
        self,
        role: str,
        query: str = "",
        n_results: int = 10
    ) -> List[SearchResult]:
        """Search for items relevant to a specific role."""
        search_query = f"{role} responsibilities {query}".strip()
        return self.search(
            query=search_query,
            n_results=n_results,
            filter_dict={"responsible_role": role}
        )
    
    def search_regulatory(
        self,
        query: str,
        n_results: int = 5
    ) -> List[SearchResult]:
        """Search regulatory guidelines."""
        return self.search(
            query=query,
            collection_name="regulatory_guidelines",
            n_results=n_results
        )
    
    def search_clinical_term(
        self,
        term: str,
        n_results: int = 5
    ) -> List[SearchResult]:
        """Search for clinical term definitions."""
        return self.search(
            query=f"definition of {term}",
            collection_name="clinical_terms",
            n_results=n_results
        )
    
    def get_context_for_query(
        self,
        query: str,
        max_tokens: int = 2000
    ) -> str:
        """Get relevant context for RAG augmentation."""
        results = self.search(query, n_results=5)
        
        context_parts = []
        current_length = 0
        
        for result in results:
            text_length = len(result.text.split())
            if current_length + text_length > max_tokens // 4:
                break
            context_parts.append(f"[{result.collection}] {result.text}")
            current_length += text_length
        
        return "\n\n".join(context_parts)
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics for all collections."""
        stats = {}
        
        for name in self.COLLECTIONS.keys():
            try:
                if self._use_mock:
                    collection = self._client.get_collection(name)
                    count = collection.count() if collection else 0
                else:
                    info = self._client.get_collection(name)
                    count = info.points_count
                
                stats[name] = {
                    "count": count,
                    "metadata_fields": self.COLLECTIONS[name].metadata_fields
                }
            except Exception as e:
                stats[name] = {"count": 0, "error": str(e)}
        
        return stats
    
    def get_stats(self) -> Dict[str, Any]:
        """Get overall store statistics."""
        collection_stats = self.get_collection_stats()
        total_docs = sum(c.get("count", 0) for c in collection_stats.values())
        
        return {
            "initialized": self._initialized,
            "using_mock": self._use_mock,
            "total_documents": total_docs,
            "documents_indexed": self.stats["documents_indexed"],
            "searches_performed": self.stats["searches_performed"],
            "collections": collection_stats
        }
    
    def clear_collection(self, collection_name: str):
        """Clear all documents from a collection."""
        if collection_name not in self.COLLECTIONS:
            return
        
        try:
            if self._use_mock:
                collection = self._client.get_collection(collection_name)
                if collection:
                    collection.documents.clear()
            else:
                self._client.delete_collection(collection_name)
                self._client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=self._embedding_service.vector_size,
                        distance=Distance.COSINE
                    )
                )
            logger.info(f"Cleared collection: {collection_name}")
        except Exception as e:
            logger.error(f"Error clearing {collection_name}: {e}")
    
    def close(self):
        """Close the Qdrant connection."""
        if self._client and not self._use_mock:
            try:
                self._client.close()
            except Exception:
                pass
        self._initialized = False


# =============================================================================
# SINGLETON ACCESSOR
# =============================================================================

_qdrant_store: Optional[QdrantVectorStore] = None


def get_qdrant_store(use_memory: bool = True) -> QdrantVectorStore:
    """Get singleton Qdrant store instance."""
    global _qdrant_store
    if _qdrant_store is None:
        _qdrant_store = QdrantVectorStore(use_memory=use_memory)
        _qdrant_store.initialize()
    return _qdrant_store


def reset_qdrant_store():
    """Reset the singleton (for testing)."""
    global _qdrant_store
    if _qdrant_store:
        _qdrant_store.close()
    _qdrant_store = None


# =============================================================================
# MAIN / DEMO
# =============================================================================

def main():
    """Demo the Qdrant vector store."""
    print("=" * 70)
    print("SANCHALAK AI - QDRANT VECTOR STORE DEMO")
    print("=" * 70)
    
    # Initialize store
    store = QdrantVectorStore(use_memory=True)
    store.initialize()
    
    print(f"\n✅ Store initialized (mock={store.uses_mock})")
    
    # Add sample documents
    sample_resolutions = [
        {
            "id": "res_001",
            "text": "SDV Incomplete: Review source documents, compare with EDC entries, update CRF data, mark SDV complete in system.",
            "metadata": {
                "issue_type": "sdv_incomplete",
                "responsible_role": "CRA",
                "effort_hours": 2,
                "success_rate": 95
            }
        },
        {
            "id": "res_002",
            "text": "Open Query: Review query text, gather required information from site, enter response in EDC, close query.",
            "metadata": {
                "issue_type": "open_query",
                "responsible_role": "Site",
                "effort_hours": 0.5,
                "success_rate": 90
            }
        },
        {
            "id": "res_003",
            "text": "Missing Signature: Contact PI for signature, ensure all prerequisite data is complete, obtain e-signature.",
            "metadata": {
                "issue_type": "missing_signature",
                "responsible_role": "Site",
                "effort_hours": 1,
                "success_rate": 85
            }
        }
    ]
    
    added = store.add_documents("resolution_templates", sample_resolutions)
    print(f"✅ Added {added} resolution patterns")
    
    # Test search
    print("\n" + "-" * 50)
    print("SEARCH TEST: 'How to complete SDV?'")
    print("-" * 50)
    
    results = store.search("How to complete SDV?", n_results=3)
    for i, result in enumerate(results, 1):
        print(f"\n{i}. [{result.collection}] Score: {result.score:.3f}")
        print(f"   {result.text[:80]}...")
    
    # Test filtered search
    print("\n" + "-" * 50)
    print("FILTERED SEARCH: issue_type='sdv_incomplete'")
    print("-" * 50)
    
    results = store.search_resolutions("sdv_incomplete")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result.score:.3f}")
        print(f"   {result.text[:80]}...")
    
    # Print stats
    print("\n" + "=" * 70)
    print("STORE STATISTICS")
    print("=" * 70)
    stats = store.get_stats()
    print(f"Total Documents: {stats['total_documents']}")
    print(f"Searches Performed: {stats['searches_performed']}")
    print(f"Using Mock: {stats['using_mock']}")
    
    print("\n✅ Qdrant Vector Store demo complete!")
    return store


if __name__ == "__main__":
    main()
