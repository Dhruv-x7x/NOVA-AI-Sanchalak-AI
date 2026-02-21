# src/knowledge/vector_store.py
"""
SANCHALAK AI - ChromaDB Vector Store v1.0
Phase 4.2: Persistent vector database with collections and filtered search

Features:
- 7 collections (one per knowledge category)
- Metadata filtering (issue_type, priority, role, category)
- Hybrid search (semantic + keyword)
- Persistent storage
- Batch indexing
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

import chromadb
from chromadb.config import Settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """
    ChromaDB-based vector store for SANCHALAK AI knowledge artifacts.
    
    Collections:
    1. resolution_templates - Resolution patterns for issues
    2. issue_descriptions - Issue type definitions
    3. sop_documents - Standard Operating Procedures
    4. query_templates - Data query templates
    5. pattern_descriptions - Site/patient patterns
    6. regulatory_guidelines - ICH, 21 CFR Part 11
    7. clinical_terms - Clinical trial terminology
    """
    
    # Collection configuration
    COLLECTIONS = {
        "resolution_templates": {
            "description": "Resolution patterns for clinical trial issues",
            "metadata_fields": ["issue_type", "responsible_role", "effort_hours", "success_rate"]
        },
        "issue_descriptions": {
            "description": "Issue type definitions and impacts",
            "metadata_fields": ["issue_type", "responsible", "priority"]
        },
        "sop_documents": {
            "description": "Standard Operating Procedures",
            "metadata_fields": ["sop_id", "title", "department"]
        },
        "query_templates": {
            "description": "Data query templates",
            "metadata_fields": ["query_category", "priority", "data_point"]
        },
        "pattern_descriptions": {
            "description": "Site and patient behavior patterns",
            "metadata_fields": ["pattern_id", "severity", "category"]
        },
        "regulatory_guidelines": {
            "description": "Regulatory guidelines (ICH, FDA)",
            "metadata_fields": ["guideline_id", "section"]
        },
        "clinical_terms": {
            "description": "Clinical trial terminology",
            "metadata_fields": ["term", "full_name"]
        }
    }
    
    def __init__(
        self,
        persist_directory: Optional[Path] = None,
        embedding_dir: Optional[Path] = None
    ):
        """
        Initialize the Vector Store.
        
        Args:
            persist_directory: Directory for ChromaDB persistence
            embedding_dir: Directory containing embeddings from Phase 4.1
        """
        # Setup paths
        if persist_directory is None:
            # Try to find in project root first
            try:
                project_root = Path(__file__).parent.parent.parent
                root_persist_path = project_root / "data" / "processed" / "knowledge" / "chromadb"
                if root_persist_path.exists():
                    self.persist_dir = root_persist_path
                else:
                    self.persist_dir = Path("data/processed/knowledge/chromadb")
            except Exception:
                self.persist_dir = Path("data/processed/knowledge/chromadb")
        else:
            self.persist_dir = Path(persist_directory)
        
        if not self.persist_dir.exists():
            self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        if embedding_dir is None:
            # Try to find in project root first
            try:
                project_root = Path(__file__).parent.parent.parent
                root_embedding_path = project_root / "data" / "processed" / "knowledge" / "embeddings"
                if root_embedding_path.exists():
                    self.embedding_dir = root_embedding_path
                else:
                    self.embedding_dir = Path("data/processed/knowledge/embeddings")
            except Exception:
                self.embedding_dir = Path("data/processed/knowledge/embeddings")
        else:
            self.embedding_dir = Path(embedding_dir)
        
        # Initialize ChromaDB client with persistence
        logger.info(f"Initializing ChromaDB at: {self.persist_dir}")
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Track collections
        self.collections: Dict[str, chromadb.Collection] = {}
        
        # Statistics
        self.stats = {
            "collections_created": 0,
            "total_documents": 0,
            "indexing_time": 0
        }
    
    def create_collections(self, reset: bool = False) -> Dict[str, chromadb.Collection]:
        """
        Create all collections in ChromaDB.
        
        Args:
            reset: If True, delete existing collections first
            
        Returns:
            Dictionary of collection name -> Collection object
        """
        logger.info("=" * 60)
        logger.info("CREATING CHROMADB COLLECTIONS")
        logger.info("=" * 60)
        
        if reset:
            logger.warning("Resetting all collections...")
            for name in self.COLLECTIONS.keys():
                try:
                    self.client.delete_collection(name)
                    logger.info(f"  Deleted existing collection: {name}")
                except Exception:
                    pass
        
        for name, config in self.COLLECTIONS.items():
            try:
                # Get or create collection
                collection = self.client.get_or_create_collection(
                    name=name,
                    metadata={
                        "description": config["description"],
                        "created_at": datetime.now().isoformat(),
                        "hnsw:space": "cosine"  # Use cosine similarity
                    }
                )
                self.collections[name] = collection
                self.stats["collections_created"] += 1
                logger.info(f"  Created collection: {name}")
                
            except Exception as e:
                logger.error(f"  Error creating {name}: {e}")
        
        logger.info(f"Created {len(self.collections)} collections")
        return self.collections
    
    def index_embeddings(self) -> Dict[str, int]:
        """
        Index all embeddings from Phase 4.1 into ChromaDB collections.
        
        Returns:
            Dictionary of collection name -> document count
        """
        logger.info("=" * 60)
        logger.info("INDEXING EMBEDDINGS INTO CHROMADB")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        indexed_counts = {}
        
        for collection_name in self.COLLECTIONS.keys():
            count = self._index_collection(collection_name)
            indexed_counts[collection_name] = count
            self.stats["total_documents"] += count
        
        self.stats["indexing_time"] = (datetime.now() - start_time).total_seconds()
        
        logger.info("=" * 60)
        logger.info("INDEXING COMPLETE")
        logger.info(f"Total Documents: {self.stats['total_documents']}")
        logger.info(f"Indexing Time: {self.stats['indexing_time']:.2f}s")
        logger.info("=" * 60)
        
        return indexed_counts
    
    def _index_collection(self, collection_name: str) -> int:
        """Index a single collection from embedding files."""
        
        # Load embeddings
        embeddings_file = self.embedding_dir / f"{collection_name}_embeddings.npy"
        metadata_file = self.embedding_dir / f"{collection_name}_metadata.json"
        texts_file = self.embedding_dir / f"{collection_name}_texts.json"
        
        if not all(f.exists() for f in [embeddings_file, metadata_file, texts_file]):
            logger.warning(f"  Missing files for {collection_name}, skipping")
            return 0
        
        # Load data
        embeddings = np.load(embeddings_file)
        with open(metadata_file, 'r') as f:
            metadata_list = json.load(f)
        with open(texts_file, 'r') as f:
            texts = json.load(f)
        
        if len(embeddings) == 0:
            logger.warning(f"  No embeddings for {collection_name}")
            return 0
        
        # Get or create collection
        if collection_name not in self.collections:
            self.collections[collection_name] = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        
        collection = self.collections[collection_name]
        
        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []
        embedding_list = []
        
        for i, (text, meta, emb) in enumerate(zip(texts, metadata_list, embeddings)):
            # Generate unique ID
            doc_id = meta.get('id', f"{collection_name}_{i}")
            
            # Clean metadata (ChromaDB only accepts str, int, float, bool)
            clean_meta = self._clean_metadata(meta)
            clean_meta['category'] = collection_name
            
            ids.append(doc_id)
            documents.append(text)
            metadatas.append(clean_meta)
            embedding_list.append(emb.tolist())
        
        # Add to collection (ChromaDB handles duplicates by ID)
        try:
            # First try to delete existing docs with same IDs
            try:
                collection.delete(ids=ids)
            except Exception:
                pass
            
            # Add documents
            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embedding_list
            )
            
            logger.info(f"  Indexed {len(ids)} documents into {collection_name}")
            return len(ids)
            
        except Exception as e:
            logger.error(f"  Error indexing {collection_name}: {e}")
            return 0
    
    def _clean_metadata(self, meta: Dict) -> Dict:
        """Clean metadata for ChromaDB (only str, int, float, bool allowed)."""
        clean = {}
        for key, value in meta.items():
            if value is None:
                continue
            elif isinstance(value, (str, int, float, bool)):
                clean[key] = value
            elif isinstance(value, (list, dict)):
                clean[key] = json.dumps(value)
            else:
                clean[key] = str(value)
        return clean
    
    # =========================================================================
    # SEARCH METHODS
    # =========================================================================
    
    def search(
        self,
        query: str,
        collection_name: Optional[str] = None,
        n_results: int = 5,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for similar documents.
        
        Args:
            query: Search query text
            collection_name: Specific collection to search (None = all)
            n_results: Number of results per collection
            where: Metadata filter (e.g., {"issue_type": "sdv_incomplete"})
            where_document: Document content filter (e.g., {"$contains": "SDV"})
            
        Returns:
            List of search results with scores
        """
        results = []
        
        collections_to_search = (
            [collection_name] if collection_name 
            else list(self.collections.keys())
        )
        
        for coll_name in collections_to_search:
            if coll_name not in self.collections:
                continue
            
            collection = self.collections[coll_name]
            
            try:
                # Build query parameters
                query_params = {
                    "query_texts": [query],
                    "n_results": n_results
                }
                
                if where:
                    query_params["where"] = where
                if where_document:
                    query_params["where_document"] = where_document
                
                # Execute query
                response = collection.query(**query_params)
                
                # Process results
                if response and response['ids'] and response['ids'][0]:
                    for i, doc_id in enumerate(response['ids'][0]):
                        results.append({
                            "id": doc_id,
                            "collection": coll_name,
                            "document": response['documents'][0][i] if response['documents'] else "",
                            "metadata": response['metadatas'][0][i] if response['metadatas'] else {},
                            "distance": response['distances'][0][i] if response['distances'] else 0,
                            "score": 1 - response['distances'][0][i] if response['distances'] else 0
                        })
                        
            except Exception as e:
                logger.warning(f"Search error in {coll_name}: {e}")
        
        # Sort by score (highest first)
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results[:n_results]
    
    def search_by_issue_type(
        self,
        issue_type: str,
        n_results: int = 5
    ) -> List[Dict]:
        """
        Search for resolutions by issue type.
        
        Args:
            issue_type: Issue type (e.g., "sdv_incomplete", "open_queries")
            n_results: Number of results
            
        Returns:
            List of matching resolutions
        """
        return self.search(
            query=f"Resolution for {issue_type}",
            collection_name="resolution_templates",
            n_results=n_results,
            where={"issue_type": issue_type}
        )
    
    def search_by_role(
        self,
        role: str,
        query: str = "",
        n_results: int = 10
    ) -> List[Dict]:
        """
        Search for items relevant to a specific role.
        
        Args:
            role: Role (e.g., "CRA", "Site", "Data Manager")
            query: Additional search query
            n_results: Number of results
            
        Returns:
            List of relevant items
        """
        search_query = f"{role} responsibilities {query}" if query else f"{role} tasks and responsibilities"
        
        results = []
        
        # Search resolution templates by role
        resolution_results = self.search(
            query=search_query,
            collection_name="resolution_templates",
            n_results=n_results,
            where={"responsible_role": role}
        )
        results.extend(resolution_results)
        
        # Search issue descriptions by role
        issue_results = self.search(
            query=search_query,
            collection_name="issue_descriptions",
            n_results=n_results,
            where={"responsible": role}
        )
        results.extend(issue_results)
        
        # Sort and return
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:n_results]
    
    def search_by_priority(
        self,
        priority: str,
        query: str = "",
        n_results: int = 10
    ) -> List[Dict]:
        """
        Search for items by priority level.
        
        Args:
            priority: Priority level (e.g., "Critical", "High", "Medium", "Low")
            query: Additional search query
            n_results: Number of results
            
        Returns:
            List of matching items
        """
        search_query = f"{priority} priority issues {query}" if query else f"{priority} priority"
        
        return self.search(
            query=search_query,
            n_results=n_results,
            where={"priority": priority}
        )
    
    def search_regulatory(
        self,
        query: str,
        n_results: int = 5
    ) -> List[Dict]:
        """
        Search regulatory guidelines.
        
        Args:
            query: Search query (e.g., "audit trail", "electronic signatures")
            n_results: Number of results
            
        Returns:
            List of matching guidelines
        """
        return self.search(
            query=query,
            collection_name="regulatory_guidelines",
            n_results=n_results
        )
    
    def search_clinical_term(
        self,
        term: str,
        n_results: int = 5
    ) -> List[Dict]:
        """
        Search for clinical term definitions.
        
        Args:
            term: Term to search (e.g., "SDV", "SAE", "CRF")
            n_results: Number of results
            
        Returns:
            List of matching terms
        """
        return self.search(
            query=f"Definition of {term}",
            collection_name="clinical_terms",
            n_results=n_results
        )
    
    def hybrid_search(
        self,
        query: str,
        keyword: Optional[str] = None,
        collection_name: Optional[str] = None,
        n_results: int = 5
    ) -> List[Dict]:
        """
        Hybrid search combining semantic and keyword matching.
        
        Args:
            query: Semantic search query
            keyword: Keyword to filter by (in document text)
            collection_name: Specific collection
            n_results: Number of results
            
        Returns:
            List of matching items
        """
        where_document = {"$contains": keyword} if keyword else None
        
        return self.search(
            query=query,
            collection_name=collection_name,
            n_results=n_results,
            where_document=where_document
        )
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics for all collections."""
        stats = {}
        
        for name, collection in self.collections.items():
            count = collection.count()
            stats[name] = {
                "count": count,
                "metadata_fields": self.COLLECTIONS.get(name, {}).get("metadata_fields", [])
            }
        
        return stats
    
    def get_all_issue_types(self) -> List[str]:
        """Get all unique issue types in resolution templates."""
        if "resolution_templates" not in self.collections:
            return []
        
        collection = self.collections["resolution_templates"]
        results = collection.get(include=["metadatas"])
        
        issue_types = set()
        for meta in results.get('metadatas', []):
            if meta and 'issue_type' in meta:
                issue_types.add(meta['issue_type'])
        
        return sorted(list(issue_types))
    
    def get_all_roles(self) -> List[str]:
        """Get all unique responsible roles."""
        if "resolution_templates" not in self.collections:
            return []
        
        collection = self.collections["resolution_templates"]
        results = collection.get(include=["metadatas"])
        
        roles = set()
        for meta in results.get('metadatas', []):
            if meta and 'responsible_role' in meta:
                roles.add(meta['responsible_role'])
        
        return sorted(list(roles))
    
    def save_index_summary(self):
        """Save index summary to JSON."""
        summary = {
            "created_at": datetime.now().isoformat(),
            "persist_directory": str(self.persist_dir),
            "stats": self.stats,
            "collections": self.get_collection_stats(),
            "issue_types": self.get_all_issue_types(),
            "roles": self.get_all_roles()
        }
        
        summary_path = self.persist_dir / "index_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved index summary to {summary_path}")
        return summary


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Run the vector store indexing pipeline."""
    print("=" * 70)
    print("SANCHALAK AI - PHASE 4.2: CHROMADB VECTOR STORE")
    print("=" * 70)
    
    # Initialize vector store
    store = VectorStore()
    
    # Create collections
    store.create_collections(reset=True)
    
    # Index embeddings
    indexed = store.index_embeddings()
    
    # Print collection stats
    print("\n" + "=" * 70)
    print("COLLECTION STATISTICS")
    print("=" * 70)
    stats = store.get_collection_stats()
    for name, info in stats.items():
        print(f"  {name}: {info['count']} documents")
    
    # Print available filters
    print("\n" + "-" * 70)
    print("AVAILABLE FILTERS")
    print("-" * 70)
    print(f"Issue Types: {store.get_all_issue_types()}")
    print(f"Roles: {store.get_all_roles()}")
    
    # Test searches
    print("\n" + "=" * 70)
    print("TESTING SEARCH FUNCTIONALITY")
    print("=" * 70)
    
    # Test 1: Basic search
    print("\n[TEST 1] Basic Search: 'How to complete SDV?'")
    print("-" * 50)
    results = store.search("How to complete SDV?", n_results=3)
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r['collection']}] Score: {r['score']:.3f}")
        print(f"     {r['document'][:70]}...")
    
    # Test 2: Search by issue type
    print("\n[TEST 2] Search by Issue Type: 'sdv_incomplete'")
    print("-" * 50)
    results = store.search_by_issue_type("sdv_incomplete", n_results=3)
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r['collection']}] Score: {r['score']:.3f}")
        print(f"     {r['document'][:70]}...")
    
    # Test 3: Search by role
    print("\n[TEST 3] Search by Role: 'CRA'")
    print("-" * 50)
    results = store.search_by_role("CRA", n_results=3)
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r['collection']}] Score: {r['score']:.3f}")
        print(f"     {r['document'][:70]}...")
    
    # Test 4: Regulatory search
    print("\n[TEST 4] Regulatory Search: 'audit trail requirements'")
    print("-" * 50)
    results = store.search_regulatory("audit trail requirements", n_results=3)
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r['collection']}] Score: {r['score']:.3f}")
        print(f"     {r['document'][:70]}...")
    
    # Test 5: Clinical term search
    print("\n[TEST 5] Clinical Term: 'SAE'")
    print("-" * 50)
    results = store.search_clinical_term("SAE", n_results=3)
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r['collection']}] Score: {r['score']:.3f}")
        print(f"     {r['document'][:70]}...")
    
    # Test 6: Hybrid search
    print("\n[TEST 6] Hybrid Search: semantic='safety' + keyword='SAE'")
    print("-" * 50)
    results = store.hybrid_search("safety procedures", keyword="SAE", n_results=3)
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r['collection']}] Score: {r['score']:.3f}")
        print(f"     {r['document'][:70]}...")
    
    # Save summary
    store.save_index_summary()
    
    print("\n" + "=" * 70)
    print("PHASE 4.2 COMPLETE")
    print("=" * 70)
    print(f"Total Documents Indexed: {store.stats['total_documents']}")
    print(f"Collections Created: {store.stats['collections_created']}")
    print(f"Indexing Time: {store.stats['indexing_time']:.2f}s")
    print(f"Persist Directory: {store.persist_dir}")
    print("=" * 70)
    
    return store


if __name__ == "__main__":
    main()