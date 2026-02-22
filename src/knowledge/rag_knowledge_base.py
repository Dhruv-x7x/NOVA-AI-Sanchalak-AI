# File: src/knowledge/rag_knowledge_base.py
"""
RAG Knowledge Base for a6on-i
Phase 4.3: Complete RAG system with document chunking, embedding, and retrieval.
Version: 1.1 - Fixed chunking issues
"""

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import re

import numpy as np
from sentence_transformers import SentenceTransformer

# Local imports
from .documents.ich_gcp_guidelines import (
    ICH_GCP_GUIDELINES,
    get_ich_gcp_guidelines,
    get_section,
    get_all_sections,
    search_guidelines
)
from .documents.protocol_knowledge import (
    PROTOCOL_KNOWLEDGE_BASE,
    get_protocol_knowledge,
    get_protocol_section,
    search_protocol_knowledge
)
from .documents.sop_documents import (
    SOP_DOCUMENTS,
    get_all_sops,
    get_sop_by_id,
    search_sops
)


class DocumentChunker:
    """Chunks documents into smaller pieces for embedding."""
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        min_chunk_size: int = 150
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Chunk text into smaller pieces with metadata.
        
        Args:
            text: Text to chunk
            metadata: Metadata to attach to each chunk
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text or not text.strip():
            return []
        
        # Clean text
        text = self._clean_text(text)
        
        # If text is short enough, return as single chunk
        if len(text) <= self.chunk_size:
            return [self._create_chunk(text, 0, metadata)]
        
        # Split into paragraphs first for better semantic coherence
        paragraphs = self._split_into_paragraphs(text)
        
        # Build chunks from paragraphs
        chunks = []
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para_length = len(para)
            
            # If single paragraph is too long, split by sentences
            if para_length > self.chunk_size:
                # Save current chunk if exists
                if current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    if len(chunk_text) >= self.min_chunk_size:
                        chunks.append(self._create_chunk(chunk_text, len(chunks), metadata))
                    current_chunk = []
                    current_length = 0
                
                # Split long paragraph by sentences
                sentences = self._split_into_sentences(para)
                for sentence in sentences:
                    if current_length + len(sentence) > self.chunk_size and current_chunk:
                        chunk_text = ' '.join(current_chunk)
                        if len(chunk_text) >= self.min_chunk_size:
                            chunks.append(self._create_chunk(chunk_text, len(chunks), metadata))
                        # Keep overlap
                        overlap_count = max(1, len(current_chunk) // 3)
                        current_chunk = current_chunk[-overlap_count:]
                        current_length = sum(len(s) for s in current_chunk)
                    
                    current_chunk.append(sentence)
                    current_length += len(sentence)
                continue
            
            # If adding this paragraph exceeds chunk size
            if current_length + para_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunk_text = '\n\n'.join(current_chunk)
                if len(chunk_text) >= self.min_chunk_size:
                    chunks.append(self._create_chunk(chunk_text, len(chunks), metadata))
                
                # Start new chunk with overlap (keep last paragraph)
                if current_chunk and len(current_chunk[-1]) < self.chunk_overlap:
                    current_chunk = [current_chunk[-1]]
                    current_length = len(current_chunk[0])
                else:
                    current_chunk = []
                    current_length = 0
            
            current_chunk.append(para)
            current_length += para_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(self._create_chunk(chunk_text, len(chunks), metadata))
        
        return chunks
    
    def chunk_by_sections(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Chunk text by logical sections (numbered headings or major breaks).
        
        Args:
            text: Text with section numbering
            metadata: Base metadata
            
        Returns:
            List of section chunks
        """
        if not text or not text.strip():
            return []
        
        # Multiple patterns for section detection
        patterns = [
            r'\n(\d+(?:\.\d+)*)\s+([A-Z][^\n]+)',  # "1.1 Title"
            r'\n(\d+(?:\.\d+)*)\s*\)\s*([^\n]+)',   # "1.1) Title"
            r'\n([A-Z][A-Z\s]{2,}:)',               # "SECTION TITLE:"
            r'\n(\d+)\.\s+([A-Z][^\n]+)',           # "1. Title"
        ]
        
        # Try to find sections
        all_sections = []
        
        for pattern in patterns:
            matches = list(re.finditer(pattern, text))
            if matches and len(matches) >= 3:  # At least 3 sections
                all_sections = matches
                break
        
        # If no clear sections found, use paragraph-based chunking
        if not all_sections:
            return self.chunk_text(text, metadata)
        
        chunks = []
        
        # Process text before first section
        first_match = all_sections[0]
        intro_text = text[:first_match.start()].strip()
        if intro_text and len(intro_text) >= self.min_chunk_size:
            intro_chunks = self.chunk_text(intro_text, {
                **(metadata or {}),
                'section_type': 'introduction'
            })
            chunks.extend(intro_chunks)
        
        # Process each section
        for i, match in enumerate(all_sections):
            # Get section content (until next section or end)
            start = match.start()
            end = all_sections[i + 1].start() if i + 1 < len(all_sections) else len(text)
            section_text = text[start:end].strip()
            
            # Extract section info
            groups = match.groups()
            section_id = groups[0] if groups else ""
            section_title = groups[1] if len(groups) > 1 else ""
            
            section_metadata = {
                **(metadata or {}),
                'section_number': section_id,
                'section_title': section_title.strip() if section_title else "",
                'section_type': 'numbered'
            }
            
            # Chunk this section
            if len(section_text) > self.chunk_size * 1.5:
                sub_chunks = self.chunk_text(section_text, section_metadata)
                chunks.extend(sub_chunks)
            elif len(section_text) >= self.min_chunk_size:
                chunks.append(self._create_chunk(section_text, len(chunks), section_metadata))
        
        return chunks
    
    def chunk_by_headers(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Chunk text by detecting header patterns (ALL CAPS lines, numbered lists, etc.)
        Better for protocol knowledge base format.
        """
        if not text or not text.strip():
            return []
        
        # Split by major headers (ALL CAPS lines)
        header_pattern = r'\n([A-Z][A-Z\s\-\(\)]{5,}:?)\n'
        
        parts = re.split(header_pattern, text)
        
        if len(parts) < 3:  # No clear headers, use regular chunking
            return self.chunk_text(text, metadata)
        
        chunks = []
        current_header = None
        
        i = 0
        while i < len(parts):
            part = parts[i].strip()
            
            # Check if this is a header
            if re.match(r'^[A-Z][A-Z\s\-\(\)]{5,}:?$', part):
                current_header = part.rstrip(':')
                i += 1
                continue
            
            if part and len(part) >= self.min_chunk_size:
                section_metadata = {
                    **(metadata or {}),
                    'section_title': current_header or '',
                    'section_type': 'header'
                }
                
                # If content is long, chunk it
                if len(part) > self.chunk_size * 1.5:
                    sub_chunks = self.chunk_text(part, section_metadata)
                    chunks.extend(sub_chunks)
                else:
                    # Include header in chunk for context
                    if current_header:
                        part = f"{current_header}\n\n{part}"
                    chunks.append(self._create_chunk(part, len(chunks), section_metadata))
            
            i += 1
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace but preserve paragraph breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        # Remove special characters that might cause issues
        text = text.replace('\x00', '')
        return text.strip()
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        paragraphs = re.split(r'\n\n+', text)
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Handle common abbreviations
        text = re.sub(r'(\b(?:Dr|Mr|Mrs|Ms|Prof|Inc|Ltd|etc|e\.g|i\.e))\.\s', r'\1<DOT> ', text)
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Restore dots
        sentences = [s.replace('<DOT>', '.') for s in sentences]
        return [s.strip() for s in sentences if s.strip()]
    
    def _create_chunk(
        self,
        text: str,
        chunk_index: int,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Create a chunk dictionary with metadata."""
        chunk_id = hashlib.md5(text.encode()).hexdigest()[:12]
        
        return {
            'chunk_id': f"chunk_{chunk_id}",
            'chunk_index': chunk_index,
            'text': text,
            'char_count': len(text),
            'word_count': len(text.split()),
            'metadata': metadata or {}
        }


class RAGKnowledgeBase:
    """
    RAG Knowledge Base for clinical trial operations.
    Provides semantic search and retrieval augmented generation capabilities.
    """
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        knowledge_dir: str = None,
        chunk_size: int = 500,
        chunk_overlap: int = 100
    ):
        """
        Initialize the RAG Knowledge Base.
        
        Args:
            embedding_model: SentenceTransformer model name
            knowledge_dir: Directory to store/load knowledge base
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between chunks
        """
        self.embedding_model_name = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Set up paths
        if knowledge_dir:
            self.knowledge_dir = Path(knowledge_dir)
        else:
            # Try to find the data directory in the project root first
            # Knowledge base is in src/knowledge, so project root is 3 levels up
            try:
                project_root = Path(__file__).parent.parent.parent
                root_kb_path = project_root / "data" / "processed" / "knowledge" / "rag"
                
                # Check if the root directory exists and contains the index files
                if root_kb_path.exists() and (root_kb_path / "rag_embeddings.npy").exists():
                    self.knowledge_dir = root_kb_path
                    print(f"📍 Using RAG Index from project root: {self.knowledge_dir}")
                else:
                    self.knowledge_dir = Path("data/processed/knowledge/rag")
            except Exception:
                self.knowledge_dir = Path("data/processed/knowledge/rag")
        
        # Only create if it doesn't exist (avoid issues with read-only roots if applicable)
        if not self.knowledge_dir.exists():
            self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.chunker = DocumentChunker(chunk_size, chunk_overlap, min_chunk_size=150)
        self.model = None
        self.documents = []
        self.embeddings = None
        self.index_built = False
        
        # Document sources
        self.sources = {
            'ich_gcp': 'ICH-GCP E6(R2) Guidelines',
            'protocol': 'Protocol Knowledge Base',
            'sop': 'Standard Operating Procedures'
        }
    
    def initialize(self) -> None:
        """Initialize the embedding model."""
        print(f"Loading embedding model: {self.embedding_model_name}")
        self.model = SentenceTransformer(self.embedding_model_name)
        print(f"Model loaded. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def load_all_documents(self) -> Dict[str, int]:
        """
        Load and chunk all document sources.
        
        Returns:
            Dictionary with document counts per source
        """
        if self.model is None:
            self.initialize()
        
        stats = {}
        
        # Load ICH-GCP Guidelines
        print("\n📚 Loading ICH-GCP Guidelines...")
        ich_chunks = self._load_ich_gcp()
        stats['ich_gcp'] = len(ich_chunks)
        self.documents.extend(ich_chunks)
        
        # Load Protocol Knowledge
        print("📚 Loading Protocol Knowledge Base...")
        protocol_chunks = self._load_protocol_knowledge()
        stats['protocol'] = len(protocol_chunks)
        self.documents.extend(protocol_chunks)
        
        # Load SOPs
        print("📚 Loading Standard Operating Procedures...")
        sop_chunks = self._load_sops()
        stats['sop'] = len(sop_chunks)
        self.documents.extend(sop_chunks)
        
        print(f"\n✅ Total documents loaded: {len(self.documents)}")
        return stats
    
    def _load_ich_gcp(self) -> List[Dict[str, Any]]:
        """Load and chunk ICH-GCP guidelines."""
        chunks = []
        guidelines = get_ich_gcp_guidelines()
        
        for section in guidelines.get('sections', []):
            metadata = {
                'source': 'ich_gcp',
                'source_name': self.sources['ich_gcp'],
                'document_type': 'regulatory_guideline',
                'section_id': section.get('section_id', ''),
                'title': section.get('title', ''),
                'version': guidelines.get('document_info', {}).get('version', 'E6(R2)')
            }
            
            content = section.get('content', '')
            if content:
                # Use section-based chunking for ICH-GCP
                section_chunks = self.chunker.chunk_by_sections(content, metadata)
                if not section_chunks:
                    section_chunks = self.chunker.chunk_text(content, metadata)
                chunks.extend(section_chunks)
        
        print(f"  ├── ICH-GCP: {len(chunks)} chunks from {len(guidelines.get('sections', []))} sections")
        return chunks
    
    def _load_protocol_knowledge(self) -> List[Dict[str, Any]]:
        """Load and chunk protocol knowledge base - FIXED VERSION."""
        chunks = []
        knowledge = get_protocol_knowledge()
        
        for section in knowledge.get('sections', []):
            metadata = {
                'source': 'protocol',
                'source_name': self.sources['protocol'],
                'document_type': 'knowledge_base',
                'section_id': section.get('section_id', ''),
                'category': section.get('category', ''),
                'title': section.get('title', ''),
                'keywords': section.get('keywords', []),
                'related_sections': section.get('related_sections', [])
            }
            
            content = section.get('content', '')
            if content:
                # Add title to content for better context
                full_content = f"{section.get('title', '')}\n\n{content}"
                
                # Use header-based chunking for protocol knowledge
                # This works better with the ALL CAPS header format
                section_chunks = self.chunker.chunk_by_headers(full_content, metadata)
                
                if not section_chunks or len(section_chunks) < 2:
                    # Fall back to text chunking if headers not found
                    section_chunks = self.chunker.chunk_text(full_content, metadata)
                
                chunks.extend(section_chunks)
        
        print(f"  ├── Protocol KB: {len(chunks)} chunks from {len(knowledge.get('sections', []))} sections")
        return chunks
    
    def _load_sops(self) -> List[Dict[str, Any]]:
        """Load and chunk SOP documents."""
        chunks = []
        sops = get_all_sops()
        
        for sop in sops.get('sops', []):
            metadata = {
                'source': 'sop',
                'source_name': self.sources['sop'],
                'document_type': 'sop',
                'sop_id': sop.get('sop_id', ''),
                'title': sop.get('title', ''),
                'version': sop.get('version', ''),
                'department': sop.get('department', ''),
                'category': sop.get('category', ''),
                'keywords': sop.get('keywords', []),
                'roles': sop.get('roles', [])
            }
            
            # Combine objective, scope, and content
            content_parts = []
            if sop.get('objective'):
                content_parts.append(f"OBJECTIVE: {sop['objective']}")
            if sop.get('scope'):
                content_parts.append(f"SCOPE: {sop['scope']}")
            if sop.get('content'):
                content_parts.append(sop['content'])
            
            content = '\n\n'.join(content_parts)
            
            if content:
                # Use section-based chunking for SOPs (numbered format)
                sop_chunks = self.chunker.chunk_by_sections(content, metadata)
                if not sop_chunks or len(sop_chunks) < 2:
                    sop_chunks = self.chunker.chunk_text(content, metadata)
                chunks.extend(sop_chunks)
        
        print(f"  └── SOPs: {len(chunks)} chunks from {len(sops.get('sops', []))} SOPs")
        return chunks
    
    def build_index(self) -> None:
        """Build the embedding index for all documents."""
        if not self.documents:
            raise ValueError("No documents loaded. Call load_all_documents() first.")
        
        if self.model is None:
            self.initialize()
        
        print(f"\n🔄 Building embedding index for {len(self.documents)} documents...")
        
        # Extract texts
        texts = [doc['text'] for doc in self.documents]
        
        # Generate embeddings
        self.embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        self.index_built = True
        print(f"✅ Index built. Shape: {self.embeddings.shape}")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        source_filter: Optional[str] = None,
        category_filter: Optional[str] = None,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search the knowledge base.
        
        Args:
            query: Search query
            top_k: Number of results to return
            source_filter: Filter by source ('ich_gcp', 'protocol', 'sop')
            category_filter: Filter by category
            min_score: Minimum similarity score
            
        Returns:
            List of search results with scores
        """
        if not self.index_built:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)[0]
        
        # Calculate similarities
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Apply filters
        mask = np.ones(len(self.documents), dtype=bool)
        
        if source_filter:
            for i, doc in enumerate(self.documents):
                if doc['metadata'].get('source') != source_filter:
                    mask[i] = False
        
        if category_filter:
            for i, doc in enumerate(self.documents):
                if doc['metadata'].get('category') != category_filter:
                    mask[i] = False
        
        # Apply mask to similarities
        masked_similarities = similarities.copy()
        masked_similarities[~mask] = -1
        
        # Get top-k indices
        top_indices = np.argsort(masked_similarities)[::-1][:top_k]
        
        # Build results
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score >= min_score:
                doc = self.documents[idx]
                results.append({
                    'chunk_id': doc['chunk_id'],
                    'text': doc['text'],
                    'score': score,
                    'source': doc['metadata'].get('source', ''),
                    'source_name': doc['metadata'].get('source_name', ''),
                    'title': doc['metadata'].get('title', ''),
                    'section_id': doc['metadata'].get('section_id', ''),
                    'category': doc['metadata'].get('category', ''),
                    'metadata': doc['metadata']
                })
        
        return results
    
    def search_with_context(
        self,
        query: str,
        top_k: int = 3,
        context_window: int = 1
    ) -> Dict[str, Any]:
        """
        Search and return results with surrounding context.
        
        Args:
            query: Search query
            top_k: Number of main results
            context_window: Number of surrounding chunks to include
            
        Returns:
            Search results with context
        """
        # Get base results
        results = self.search(query, top_k=top_k)
        
        # Add context for each result
        for result in results:
            chunk_idx = None
            for i, doc in enumerate(self.documents):
                if doc['chunk_id'] == result['chunk_id']:
                    chunk_idx = i
                    break
            
            if chunk_idx is not None:
                # Get surrounding chunks from same source
                context_before = []
                context_after = []
                source = result['source']
                
                # Look for context chunks
                for offset in range(1, context_window + 1):
                    # Before
                    if chunk_idx - offset >= 0:
                        prev_doc = self.documents[chunk_idx - offset]
                        if prev_doc['metadata'].get('source') == source:
                            context_before.insert(0, prev_doc['text'])
                    
                    # After
                    if chunk_idx + offset < len(self.documents):
                        next_doc = self.documents[chunk_idx + offset]
                        if next_doc['metadata'].get('source') == source:
                            context_after.append(next_doc['text'])
                
                result['context_before'] = context_before
                result['context_after'] = context_after
        
        return {
            'query': query,
            'results': results,
            'total_results': len(results)
        }
    
    def get_relevant_context(
        self,
        query: str,
        max_tokens: int = 2000
    ) -> str:
        """
        Get relevant context for a query, suitable for LLM input.
        
        Args:
            query: Search query
            max_tokens: Maximum context length (approximate)
            
        Returns:
            Formatted context string
        """
        results = self.search(query, top_k=5, min_score=0.3)
        
        context_parts = []
        total_length = 0
        max_chars = max_tokens * 4  # Approximate chars per token
        
        for result in results:
            source_info = f"[{result['source_name']}]"
            if result.get('title'):
                source_info += f" {result['title']}"
            
            chunk = f"{source_info}\n{result['text']}\n"
            
            if total_length + len(chunk) > max_chars:
                break
            
            context_parts.append(chunk)
            total_length += len(chunk)
        
        return "\n---\n".join(context_parts)
    
    def save_index(self) -> str:
        """Save the knowledge base index to disk."""
        if not self.index_built:
            raise ValueError("No index to save. Call build_index() first.")
        
        # Save embeddings
        embeddings_path = self.knowledge_dir / "rag_embeddings.npy"
        np.save(embeddings_path, self.embeddings)
        
        # Save documents
        documents_path = self.knowledge_dir / "rag_documents.json"
        with open(documents_path, 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, indent=2, ensure_ascii=False)
        
        # Save metadata
        metadata = {
            'created_at': datetime.now().isoformat(),
            'version': '1.1',
            'embedding_model': self.embedding_model_name,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'total_documents': len(self.documents),
            'embedding_dimension': self.embeddings.shape[1] if self.embeddings is not None else 0,
            'sources': {
                source: sum(1 for d in self.documents if d['metadata'].get('source') == source)
                for source in self.sources.keys()
            },
            'categories': {}
        }
        
        # Count categories
        for doc in self.documents:
            cat = doc['metadata'].get('category', '')
            if cat:
                metadata['categories'][cat] = metadata['categories'].get(cat, 0) + 1
        
        metadata_path = self.knowledge_dir / "rag_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Index saved to {self.knowledge_dir}")
        return str(self.knowledge_dir)
    
    def load_index(self) -> bool:
        """
        Load a previously saved index.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        embeddings_path = self.knowledge_dir / "rag_embeddings.npy"
        documents_path = self.knowledge_dir / "rag_documents.json"
        metadata_path = self.knowledge_dir / "rag_metadata.json"
        
        if not all(p.exists() for p in [embeddings_path, documents_path, metadata_path]):
            print("⚠️ No saved index found.")
            return False
        
        try:
            # Load embeddings
            self.embeddings = np.load(embeddings_path)
            
            # Load documents
            with open(documents_path, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
            
            # Load metadata
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Initialize model if needed
            if self.model is None:
                self.initialize()
            
            self.index_built = True
            
            print(f"✅ Index loaded: {len(self.documents)} documents")
            print(f"   Version: {metadata.get('version', 'unknown')}")
            print(f"   Created: {metadata.get('created_at', 'unknown')}")
            print(f"   Sources: {metadata.get('sources', {})}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading index: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        if not self.documents:
            return {'status': 'empty', 'total_documents': 0}
        
        # Source distribution
        source_counts = {}
        category_counts = {}
        total_chars = 0
        total_words = 0
        chunk_sizes = []
        
        for doc in self.documents:
            source = doc['metadata'].get('source', 'unknown')
            source_counts[source] = source_counts.get(source, 0) + 1
            
            category = doc['metadata'].get('category', '')
            if category:
                category_counts[category] = category_counts.get(category, 0) + 1
            
            char_count = doc.get('char_count', len(doc['text']))
            total_chars += char_count
            total_words += doc.get('word_count', len(doc['text'].split()))
            chunk_sizes.append(char_count)
        
        return {
            'status': 'ready' if self.index_built else 'documents_loaded',
            'total_documents': len(self.documents),
            'total_characters': total_chars,
            'total_words': total_words,
            'avg_chunk_size': total_chars // len(self.documents) if self.documents else 0,
            'min_chunk_size': min(chunk_sizes) if chunk_sizes else 0,
            'max_chunk_size': max(chunk_sizes) if chunk_sizes else 0,
            'source_distribution': source_counts,
            'category_distribution': category_counts,
            'embedding_dimension': self.embeddings.shape[1] if self.embeddings is not None else 0,
            'index_built': self.index_built
        }


class RetrievalChain:
    """
    Retrieval chain for RAG-based question answering.
    Combines knowledge retrieval with context formatting for LLM input.
    """
    
    def __init__(self, knowledge_base: RAGKnowledgeBase):
        """
        Initialize the retrieval chain.
        
        Args:
            knowledge_base: Initialized RAGKnowledgeBase instance
        """
        self.kb = knowledge_base
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        source_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            source_filter: Optional source filter
            
        Returns:
            List of relevant documents
        """
        return self.kb.search(query, top_k=top_k, source_filter=source_filter)
    
    def format_context(
        self,
        documents: List[Dict[str, Any]],
        max_length: int = 3000
    ) -> str:
        """
        Format retrieved documents as context for LLM.
        
        Args:
            documents: Retrieved documents
            max_length: Maximum context length
            
        Returns:
            Formatted context string
        """
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(documents, 1):
            # Format source citation
            source = doc.get('source_name', doc.get('source', 'Unknown'))
            title = doc.get('title', '')
            section = doc.get('section_id', '')
            category = doc.get('category', '')
            
            citation = f"[Source {i}: {source}"
            if title:
                citation += f" - {title}"
            if section:
                citation += f" (Section {section})"
            if category:
                citation += f" | Category: {category}"
            citation += "]"
            
            # Format content
            content = doc.get('text', '')
            score = doc.get('score', 0)
            
            formatted = f"{citation}\n{content}\n(Relevance: {score:.2f})\n"
            
            # Check length
            if current_length + len(formatted) > max_length:
                # Truncate if necessary
                remaining = max_length - current_length - 100
                if remaining > 200:
                    formatted = f"{citation}\n{content[:remaining]}...\n"
                    context_parts.append(formatted)
                break
            
            context_parts.append(formatted)
            current_length += len(formatted)
        
        return "\n---\n".join(context_parts)
    
    def create_prompt(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Create a prompt for the LLM with retrieved context.
        
        Args:
            query: User query
            context: Retrieved context
            system_prompt: Optional system prompt
            
        Returns:
            Dictionary with system and user prompts
        """
        if system_prompt is None:
            system_prompt = """You are a clinical trial operations expert assistant for a6on-i.
Your role is to provide accurate, helpful information about clinical trial operations, data management, 
regulatory compliance, and best practices.

When answering questions:
1. Use the provided context from ICH-GCP guidelines, protocol knowledge, and SOPs
2. Be specific and cite relevant sections when applicable
3. If the context doesn't contain enough information, say so
4. Provide actionable recommendations when appropriate
5. Maintain compliance with GCP and regulatory requirements

Always prioritize patient safety and data integrity in your recommendations."""
        
        user_prompt = f"""Based on the following knowledge base context, please answer the question.

CONTEXT:
{context}

QUESTION: {query}

Please provide a comprehensive answer based on the context above. If the context doesn't fully address the question, indicate what additional information might be needed."""
        
        return {
            'system': system_prompt,
            'user': user_prompt,
            'context': context,
            'query': query
        }
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        source_filter: Optional[str] = None,
        return_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Complete RAG query pipeline.
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            source_filter: Optional source filter
            return_sources: Whether to return source documents
            
        Returns:
            Dictionary with prompt and sources
        """
        # Retrieve relevant documents
        documents = self.retrieve(question, top_k=top_k, source_filter=source_filter)
        
        # Format context
        context = self.format_context(documents)
        
        # Create prompt
        prompt = self.create_prompt(question, context)
        
        result = {
            'prompt': prompt,
            'retrieval_count': len(documents)
        }
        
        if return_sources:
            result['sources'] = [
                {
                    'source': doc.get('source_name', doc.get('source', '')),
                    'title': doc.get('title', ''),
                    'section_id': doc.get('section_id', ''),
                    'category': doc.get('category', ''),
                    'score': doc.get('score', 0),
                    'text_preview': doc.get('text', '')[:200] + '...' if len(doc.get('text', '')) > 200 else doc.get('text', '')
                }
                for doc in documents
            ]
        
        return result


class TopicSearch:
    """
    Topic-based search for the knowledge base.
    Provides specialized search for common clinical trial topics.
    """
    
    # Predefined topics with related queries
    TOPICS = {
        'informed_consent': {
            'name': 'Informed Consent',
            'queries': ['informed consent process', 'consent requirements', 'subject consent', 'consent form'],
            'sources': ['ich_gcp', 'sop']
        },
        'safety_reporting': {
            'name': 'Safety Reporting',
            'queries': ['SAE reporting', 'serious adverse event', 'safety report', 'adverse event'],
            'sources': ['ich_gcp', 'protocol', 'sop']
        },
        'data_quality': {
            'name': 'Data Quality',
            'queries': ['data quality', 'DQI', 'data integrity', 'query management', 'data quality index'],
            'sources': ['protocol', 'sop']
        },
        'monitoring': {
            'name': 'Site Monitoring',
            'queries': ['monitoring visit', 'SDV', 'source data verification', 'CRA monitoring'],
            'sources': ['ich_gcp', 'protocol', 'sop']
        },
        'database_lock': {
            'name': 'Database Lock',
            'queries': ['database lock', 'DB lock', 'data freeze', 'lock readiness', 'pre-lock'],
            'sources': ['protocol', 'sop']
        },
        'protocol_deviation': {
            'name': 'Protocol Deviations',
            'queries': ['protocol deviation', 'protocol violation', 'non-compliance', 'deviation reporting'],
            'sources': ['ich_gcp', 'protocol', 'sop']
        },
        'essential_documents': {
            'name': 'Essential Documents',
            'queries': ['essential documents', 'TMF', 'trial master file', 'document retention'],
            'sources': ['ich_gcp', 'sop']
        },
        'irb_iec': {
            'name': 'IRB/IEC',
            'queries': ['IRB', 'IEC', 'ethics committee', 'institutional review board', 'ethics approval'],
            'sources': ['ich_gcp', 'sop']
        },
        'investigator': {
            'name': 'Investigator Responsibilities',
            'queries': ['investigator responsibilities', 'PI duties', 'principal investigator', 'investigator qualifications'],
            'sources': ['ich_gcp', 'protocol']
        },
        'sponsor': {
            'name': 'Sponsor Responsibilities',
            'queries': ['sponsor responsibilities', 'sponsor duties', 'sponsor oversight', 'quality management'],
            'sources': ['ich_gcp', 'protocol']
        }
    }
    
    def __init__(self, knowledge_base: RAGKnowledgeBase):
        """Initialize with knowledge base."""
        self.kb = knowledge_base
    
    def search_topic(
        self,
        topic: str,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Search for a predefined topic.
        
        Args:
            topic: Topic key from TOPICS
            top_k: Number of results per query
            
        Returns:
            Combined search results for the topic
        """
        if topic not in self.TOPICS:
            available = list(self.TOPICS.keys())
            raise ValueError(f"Unknown topic: {topic}. Available: {available}")
        
        topic_info = self.TOPICS[topic]
        all_results = []
        seen_chunks = set()
        
        for query in topic_info['queries']:
            results = self.kb.search(query, top_k=top_k // len(topic_info['queries']) + 2)
            for result in results:
                if result['chunk_id'] not in seen_chunks:
                    seen_chunks.add(result['chunk_id'])
                    all_results.append(result)
        
        # Sort by score
        all_results.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'topic': topic,
            'topic_name': topic_info['name'],
            'results': all_results[:top_k],
            'total_found': len(all_results)
        }
    
    def get_available_topics(self) -> List[Dict[str, str]]:
        """Get list of available topics."""
        return [
            {'key': key, 'name': info['name']}
            for key, info in self.TOPICS.items()
        ]


def build_rag_knowledge_base(
    output_dir: str = "data/processed/knowledge/rag",
    save_index: bool = True
) -> RAGKnowledgeBase:
    """
    Build the complete RAG knowledge base.
    
    Args:
        output_dir: Directory to save the index
        save_index: Whether to save the index to disk
        
    Returns:
        Initialized RAGKnowledgeBase
    """
    print("=" * 60)
    print("a6on-i - RAG Knowledge Base Builder v1.1")
    print("=" * 60)
    
    # Initialize knowledge base with improved chunking parameters
    kb = RAGKnowledgeBase(
        knowledge_dir=output_dir,
        chunk_size=500,
        chunk_overlap=100
    )
    kb.initialize()
    
    # Load all documents
    stats = kb.load_all_documents()
    
    # Build index
    kb.build_index()
    
    # Save if requested
    if save_index:
        kb.save_index()
    
    # Print statistics
    print("\n" + "=" * 60)
    print("KNOWLEDGE BASE STATISTICS")
    print("=" * 60)
    
    full_stats = kb.get_statistics()
    print(f"Total Documents: {full_stats['total_documents']}")
    print(f"Total Characters: {full_stats['total_characters']:,}")
    print(f"Total Words: {full_stats['total_words']:,}")
    print(f"Chunk Size: min={full_stats['min_chunk_size']}, avg={full_stats['avg_chunk_size']}, max={full_stats['max_chunk_size']}")
    print(f"Embedding Dimension: {full_stats['embedding_dimension']}")
    print(f"\nSource Distribution:")
    for source, count in full_stats['source_distribution'].items():
        print(f"  - {source}: {count} chunks")
    
    print(f"\nCategory Distribution:")
    for category, count in sorted(full_stats['category_distribution'].items(), key=lambda x: -x[1]):
        print(f"  - {category}: {count} chunks")
    
    return kb
