# File: src/knowledge/__init__.py
"""
SANCHALAK AI - Knowledge Module

Components:
- Embedding Pipeline (4.1) - Requires sentence_transformers
- Vector Store - ChromaDB (4.2) - Requires chromadb
- RAG Knowledge Base (4.3) - Requires sentence_transformers, chromadb
- Causal Hypothesis Engine (4.4) - Core, no external ML dependencies
- Cross-Study Pattern Matcher (4.5) - Core, no external ML dependencies
"""

from pathlib import Path

# Module version
__version__ = "1.0.0"

# Core components (may fail due to transitive dependencies like torch/sentence_transformers)
CausalHypothesisEngine = None
CrossStudyPatternMatcher = None

try:
    from .causal_hypothesis_engine import CausalHypothesisEngine
except (ImportError, OSError) as e:
    import logging
    logging.getLogger(__name__).warning(f"CausalHypothesisEngine not available: {e}")

try:
    from .cross_study_pattern_matcher import CrossStudyPatternMatcher
except (ImportError, OSError) as e:
    import logging
    logging.getLogger(__name__).warning(f"CrossStudyPatternMatcher not available: {e}")

# Optional components with external dependencies
EmbeddingPipeline = None
VectorStore = None
RAGKnowledgeBase = None

try:
    from .embedding_pipeline import EmbeddingPipeline
except (ImportError, OSError) as e:
    import logging
    logging.getLogger(__name__).warning(f"EmbeddingPipeline not available: {e}")

try:
    from .vector_store import VectorStore
except (ImportError, OSError, ValueError) as e:
    import logging
    logging.getLogger(__name__).warning(f"VectorStore not available: {e}")

try:
    from .rag_knowledge_base import RAGKnowledgeBase
except (ImportError, OSError, ValueError) as e:
    import logging
    logging.getLogger(__name__).warning(f"RAGKnowledgeBase not available: {e}")

# Public API (explicit exports)
__all__ = [
    'EmbeddingPipeline',
    'VectorStore',
    'RAGKnowledgeBase',
    'CausalHypothesisEngine',
    'CrossStudyPatternMatcher'
]
