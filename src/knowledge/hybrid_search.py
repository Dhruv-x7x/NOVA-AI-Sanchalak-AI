"""
SANCHALAK AI - Hybrid RAG Search Engine
================================================
Implements Reciprocal Rank Fusion (RRF) of:
- Qdrant/ChromaDB vector search (semantic similarity)
- BM25 keyword search (lexical matching)

Reference: SOLUTION.md L275-303
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Search result with combined score."""
    document_id: str
    text: str
    score: float
    source: str
    category: str
    vector_score: Optional[float] = None
    bm25_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "text": self.text[:500],  # Truncate for display
            "score": round(self.score, 4),
            "source": self.source,
            "category": self.category,
            "vector_score": round(self.vector_score, 4) if self.vector_score else None,
            "bm25_score": round(self.bm25_score, 4) if self.bm25_score else None,
            "metadata": self.metadata
        }


class BM25Index:
    """
    Simple BM25 (Okapi BM25) implementation for keyword search.
    
    BM25 Formula:
    score(D, Q) = Σ IDF(q) * (f(q,D) * (k1 + 1)) / (f(q,D) + k1 * (1 - b + b * |D|/avgdl))
    
    Where:
    - f(q,D) = term frequency of query term q in document D
    - |D| = length of document D
    - avgdl = average document length
    - k1, b = free parameters (typically k1=1.5, b=0.75)
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: List[Dict[str, Any]] = []
        self.doc_lengths: List[int] = []
        self.avgdl: float = 0.0
        self.doc_freqs: Dict[str, int] = defaultdict(int)
        self.term_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}
        self._built = False
    
    def add_documents(self, documents: List[Dict[str, Any]], text_field: str = "text"):
        """Add documents to the index."""
        for doc in documents:
            text = doc.get(text_field, "")
            tokens = self._tokenize(text)
            
            self.documents.append(doc)
            self.doc_lengths.append(len(tokens))
            
            # Term frequencies for this document
            tf = defaultdict(int)
            for token in tokens:
                tf[token] += 1
            self.term_freqs.append(dict(tf))
            
            # Document frequencies
            for token in set(tokens):
                self.doc_freqs[token] += 1
        
        self._built = False
    
    def build(self):
        """Build the index (compute IDF values)."""
        if not self.documents:
            logger.warning("No documents to build index from")
            return
        
        n_docs = len(self.documents)
        self.avgdl = sum(self.doc_lengths) / n_docs if n_docs > 0 else 0
        
        # Compute IDF for each term
        for term, df in self.doc_freqs.items():
            # IDF = log((N - n(q) + 0.5) / (n(q) + 0.5))
            self.idf[term] = np.log((n_docs - df + 0.5) / (df + 0.5) + 1)
        
        self._built = True
        logger.info(f"BM25 index built: {n_docs} documents, {len(self.idf)} unique terms")
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization (lowercase, split on non-alphanumeric)."""
        import re
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def search(self, query: str, limit: int = 20) -> List[Tuple[int, float]]:
        """
        Search the index with BM25 scoring.
        
        Returns list of (doc_index, score) tuples.
        """
        if not self._built:
            self.build()
        
        if not self.documents:
            return []
        
        query_tokens = self._tokenize(query)
        scores = []
        
        for doc_idx, tf in enumerate(self.term_freqs):
            score = 0.0
            doc_len = self.doc_lengths[doc_idx]
            
            for token in query_tokens:
                if token not in self.idf:
                    continue
                
                idf = self.idf[token]
                term_freq = tf.get(token, 0)
                
                # BM25 scoring
                numerator = term_freq * (self.k1 + 1)
                denominator = term_freq + self.k1 * (1 - self.b + self.b * doc_len / max(1, self.avgdl))
                score += idf * (numerator / denominator)
            
            scores.append((doc_idx, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:limit]


class HybridSearchEngine:
    """
    Hybrid search engine combining vector and BM25 search.
    
    Uses Reciprocal Rank Fusion (RRF) to combine results:
    RRF(d) = Σ 1 / (k + rank_i(d))
    
    Where:
    - k = constant (typically 60)
    - rank_i(d) = rank of document d in ranking i
    
    Usage:
        engine = HybridSearchEngine(vector_store, bm25_index)
        results = engine.search("patient with missing signature", alpha=0.7)
    """
    
    def __init__(
        self,
        vector_store = None,
        bm25_index: Optional[BM25Index] = None,
        rrf_k: int = 60
    ):
        """
        Initialize hybrid search engine.
        
        Args:
            vector_store: Qdrant/ChromaDB vector store (with search method)
            bm25_index: BM25Index for keyword search
            rrf_k: RRF constant (default 60)
        """
        self.vector_store = vector_store
        self.bm25_index = bm25_index or BM25Index()
        self.rrf_k = rrf_k
        self._documents: List[Dict[str, Any]] = []
        logger.info("HybridSearchEngine initialized")
    
    def add_documents(self, documents: List[Dict[str, Any]], text_field: str = "text"):
        """Add documents to both indexes."""
        self._documents = documents
        
        # Add to BM25 index
        self.bm25_index.add_documents(documents, text_field)
        self.bm25_index.build()
        
        # Add to vector store if available
        if self.vector_store and hasattr(self.vector_store, 'add_documents'):
            try:
                self.vector_store.add_documents(documents)
            except Exception as e:
                logger.warning(f"Could not add to vector store: {e}")
        
        logger.info(f"Added {len(documents)} documents to hybrid search")
    
    def search(
        self,
        query: str,
        alpha: float = 0.7,
        limit: int = 10
    ) -> List[SearchResult]:
        """
        Perform hybrid search with RRF fusion.
        
        Args:
            query: Search query
            alpha: Weight for vector results (0.7 = 70% vector, 30% BM25)
            limit: Maximum results to return
            
        Returns:
            List of SearchResult objects with fused scores
        """
        # Get results from both sources
        vector_results = self._vector_search(query, limit=limit * 2)
        bm25_results = self._bm25_search(query, limit=limit * 2)
        
        # Apply RRF fusion
        fused_results = self._reciprocal_rank_fusion(
            vector_results,
            bm25_results,
            alpha=alpha
        )
        
        return fused_results[:limit]
    
    def _vector_search(self, query: str, limit: int = 20) -> List[Tuple[str, float]]:
        """Perform vector search if available."""
        if not self.vector_store:
            return []
        
        try:
            if hasattr(self.vector_store, 'search'):
                results = self.vector_store.search(query, limit=limit)
                # Normalize to (doc_id, score) format
                if results and isinstance(results[0], dict):
                    return [(r.get('id', str(i)), r.get('score', 0)) for i, r in enumerate(results)]
                return results
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
        
        return []
    
    def _bm25_search(self, query: str, limit: int = 20) -> List[Tuple[str, float]]:
        """Perform BM25 search."""
        results = self.bm25_index.search(query, limit=limit)
        
        # Convert to (doc_id, score) format
        return [(str(idx), score) for idx, score in results]
    
    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Tuple[str, float]],
        bm25_results: List[Tuple[str, float]],
        alpha: float = 0.7
    ) -> List[SearchResult]:
        """
        Combine results using Reciprocal Rank Fusion.
        
        RRF(d) = alpha * (1 / (k + vector_rank)) + (1-alpha) * (1 / (k + bm25_rank))
        """
        # Build rank maps
        vector_ranks = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(vector_results)}
        bm25_ranks = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(bm25_results)}
        
        # Get score maps
        vector_scores = {doc_id: score for doc_id, score in vector_results}
        bm25_scores = {doc_id: score for doc_id, score in bm25_results}
        
        # All unique document IDs
        all_doc_ids = set(vector_ranks.keys()) | set(bm25_ranks.keys())
        
        # Calculate RRF scores
        rrf_scores = []
        for doc_id in all_doc_ids:
            v_rank = vector_ranks.get(doc_id, len(vector_results) + 100)  # Penalty for missing
            b_rank = bm25_ranks.get(doc_id, len(bm25_results) + 100)
            
            # RRF formula with alpha weighting
            rrf_score = (
                alpha * (1.0 / (self.rrf_k + v_rank)) +
                (1 - alpha) * (1.0 / (self.rrf_k + b_rank))
            )
            
            rrf_scores.append((doc_id, rrf_score, vector_scores.get(doc_id), bm25_scores.get(doc_id)))
        
        # Sort by RRF score
        rrf_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Convert to SearchResult objects
        results = []
        for doc_id, rrf_score, v_score, b_score in rrf_scores:
            try:
                doc_idx = int(doc_id)
                if 0 <= doc_idx < len(self._documents):
                    doc = self._documents[doc_idx]
                    results.append(SearchResult(
                        document_id=doc_id,
                        text=doc.get('text', ''),
                        score=rrf_score,
                        source=doc.get('metadata', {}).get('source', 'unknown'),
                        category=doc.get('metadata', {}).get('category', 'unknown'),
                        vector_score=v_score,
                        bm25_score=b_score,
                        metadata=doc.get('metadata', {})
                    ))
            except (ValueError, IndexError):
                continue
        
        return results
    
    def search_with_filters(
        self,
        query: str,
        filters: Dict[str, Any] = None,
        alpha: float = 0.7,
        limit: int = 10
    ) -> List[SearchResult]:
        """
        Search with metadata filters.
        
        Args:
            query: Search query
            filters: Metadata filters (e.g., {"source": "ich_gcp"})
            alpha: Weight for vector results
            limit: Maximum results
            
        Returns:
            Filtered search results
        """
        # Get base results
        results = self.search(query, alpha=alpha, limit=limit * 3)
        
        if not filters:
            return results[:limit]
        
        # Apply filters
        filtered = []
        for result in results:
            match = True
            for key, value in filters.items():
                if result.metadata.get(key) != value:
                    if key == 'source' and result.source != value:
                        match = False
                        break
                    if key == 'category' and result.category != value:
                        match = False
                        break
            
            if match:
                filtered.append(result)
            
            if len(filtered) >= limit:
                break
        
        return filtered
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get search engine statistics."""
        return {
            "total_documents": len(self._documents),
            "bm25_terms": len(self.bm25_index.idf),
            "vector_store_available": self.vector_store is not None,
            "rrf_k": self.rrf_k,
            "avg_doc_length": round(self.bm25_index.avgdl, 2) if self.bm25_index.avgdl else 0
        }


# Singleton instance
_engine: Optional[HybridSearchEngine] = None


def get_hybrid_search_engine() -> HybridSearchEngine:
    """Get or create the hybrid search engine singleton."""
    global _engine
    if _engine is None:
        # Try to get vector store
        vector_store = None
        try:
            from src.knowledge.qdrant_store import get_qdrant_store
            vector_store = get_qdrant_store()
        except Exception:
            try:
                from src.knowledge.vector_store import get_vector_store
                vector_store = get_vector_store()
            except Exception:
                logger.info("No vector store available, using BM25-only mode")
        
        _engine = HybridSearchEngine(vector_store=vector_store)
        
        # Load default documents if available
        try:
            from src.knowledge.rag_knowledge_base import RAGKnowledgeBase
            kb = RAGKnowledgeBase()
            if hasattr(kb, 'get_all_documents'):
                docs = kb.get_all_documents()
                if docs:
                    _engine.add_documents(docs)
        except Exception as e:
            logger.debug(f"Could not load default documents: {e}")
    
    return _engine
