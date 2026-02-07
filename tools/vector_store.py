"""
================================================================================
SurvyAI Vector Store - Semantic Search and Embeddings
================================================================================

This module provides a robust vector database implementation using ChromaDB
for local storage and persistence. It supports multiple embedding providers
for flexibility between local (free) and cloud (higher quality) options.

FEATURES:
---------
1. Local ChromaDB storage with persistence
2. Multiple embedding providers:
   - Local: Sentence Transformers (free, offline capable)
   - Cloud: OpenAI embeddings (higher quality, requires API key)
3. Multiple collections for different data types:
   - documents: PDF, Word, text files
   - drawings: AutoCAD entities and metadata
   - coordinates: Survey points and coordinates
   - conversations: Chat history for context
4. Semantic search with similarity scoring
5. Metadata filtering and retrieval

USAGE:
------
```python
from tools.vector_store import VectorStore

# Initialize with local embeddings (free)
store = VectorStore(embedding_provider="local")

# Or with OpenAI embeddings (higher quality)
store = VectorStore(embedding_provider="openai")

# Add documents
store.add_documents([
    {"content": "Survey report for parcel 123", "metadata": {"type": "report"}},
    {"content": "Property boundaries description", "metadata": {"type": "legal"}}
])

# Search
results = store.search("property boundaries", top_k=5)
```

COLLECTIONS:
------------
- documents: General text documents
- drawings: AutoCAD drawing data
- coordinates: Survey coordinate data
- conversations: Agent conversation history

Author: SurvyAI Team
License: MIT
================================================================================
"""

# ==============================================================================
# IMPORTS
# ==============================================================================

from __future__ import annotations

import hashlib
import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

# ChromaDB - Local vector database
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    from chromadb.utils import embedding_functions
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None

# Sentence Transformers - Local embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

# OpenAI for cloud embeddings
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

# ==============================================================================
# LOGGING
# ==============================================================================

logger = logging.getLogger(__name__)

# ==============================================================================
# CONSTANTS
# ==============================================================================

# Default local embedding model (small, fast, good quality)
DEFAULT_LOCAL_MODEL = "all-MiniLM-L6-v2"

# Default OpenAI embedding model (cost-effective, good quality)
DEFAULT_OPENAI_MODEL = "text-embedding-3-small"

# Collection names
COLLECTION_DOCUMENTS = "documents"
COLLECTION_DRAWINGS = "drawings"
COLLECTION_COORDINATES = "coordinates"
COLLECTION_CONVERSATIONS = "conversations"

# Default persistence directory
DEFAULT_PERSIST_DIR = ".survyai_vectordb"


# ==============================================================================
# EMBEDDING PROVIDERS
# ==============================================================================

class LocalEmbeddingProvider:
    """
    Local embedding provider using Sentence Transformers.
    
    This provider runs entirely on the local machine and doesn't require
    any API keys or internet connection after the model is downloaded.
    
    Attributes:
        model_name: Name of the Sentence Transformers model
        model: The loaded SentenceTransformer model
    """
    
    def __init__(self, model_name: str = DEFAULT_LOCAL_MODEL):
        """
        Initialize the local embedding provider.
        
        Args:
            model_name: Name of the Sentence Transformers model to use.
                       Default is 'all-MiniLM-L6-v2' (fast and efficient).
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for local embeddings. "
                "Install with: pip install sentence-transformers"
            )
        
        self.model_name = model_name
        logger.info(f"Loading local embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self._dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"✓ Local embedding model loaded (dimension: {self._dimension})")
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (as lists of floats)
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query text.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector as a list of floats
        """
        return self.embed([text])[0]
    
    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension


class OpenAIEmbeddingProvider:
    """
    OpenAI embedding provider using the OpenAI API.
    
    This provider uses OpenAI's embedding models for higher quality
    embeddings. Requires an OpenAI API key.
    
    Attributes:
        model_name: Name of the OpenAI embedding model
        client: OpenAI API client
    """
    
    def __init__(
        self, 
        api_key: str,
        model_name: str = DEFAULT_OPENAI_MODEL
    ):
        """
        Initialize the OpenAI embedding provider.
        
        Args:
            api_key: OpenAI API key
            model_name: Name of the OpenAI embedding model to use.
                       Default is 'text-embedding-3-small' (cost-effective).
        """
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "openai is required for OpenAI embeddings. "
                "Install with: pip install openai"
            )
        
        if not api_key or not api_key.strip():
            raise ValueError(
                "OPENAI_API_KEY is required for OpenAI embeddings. "
                "Please set it in your .env file."
            )
        
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)
        
        # Dimension lookup for OpenAI models
        self._dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        self._dimension = self._dimensions.get(model_name, 1536)
        
        logger.info(f"✓ OpenAI embedding provider initialized (model: {model_name})")
    
    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (as lists of floats)
        """
        # OpenAI API has a limit on texts per request
        batch_size = 100
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                model=self.model_name,
                input=batch
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query text.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector as a list of floats
        """
        return self.embed([text])[0]
    
    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return self._dimension


# ==============================================================================
# VECTOR STORE
# ==============================================================================

class VectorStore:
    """
    Robust vector database implementation using ChromaDB.
    
    This class provides a unified interface for storing and searching
    vector embeddings with support for multiple embedding providers
    and collections for different data types.
    
    Attributes:
        persist_directory: Directory for ChromaDB persistence
        embedding_provider: The embedding provider instance
        client: ChromaDB client
        collections: Dictionary of ChromaDB collections
    """
    
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        embedding_provider: Literal["local", "openai"] = "local",
        openai_api_key: Optional[str] = None,
        local_model_name: str = DEFAULT_LOCAL_MODEL,
        openai_model_name: str = DEFAULT_OPENAI_MODEL,
    ):
        """
        Initialize the vector store.
        
        Args:
            persist_directory: Directory for ChromaDB persistence.
                              Defaults to '.survyai_vectordb' in current dir.
            embedding_provider: Which embedding provider to use:
                               - "local": Sentence Transformers (free, offline)
                               - "openai": OpenAI embeddings (requires API key)
            openai_api_key: OpenAI API key (required if using "openai" provider)
            local_model_name: Name of local embedding model
            openai_model_name: Name of OpenAI embedding model
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "chromadb is required for vector storage. "
                "Install with: pip install chromadb"
            )
        
        # Set persistence directory
        self.persist_directory = persist_directory or os.path.join(
            os.getcwd(), DEFAULT_PERSIST_DIR
        )
        
        # Ensure directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing VectorStore at: {self.persist_directory}")
        
        # Initialize embedding provider
        self._init_embedding_provider(
            embedding_provider,
            openai_api_key,
            local_model_name,
            openai_model_name
        )
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize collections
        self.collections: Dict[str, Any] = {}
        self._init_collections()
        
        logger.info("✓ VectorStore initialized successfully")
    
    def _init_embedding_provider(
        self,
        provider_type: str,
        openai_api_key: Optional[str],
        local_model_name: str,
        openai_model_name: str
    ):
        """Initialize the embedding provider based on configuration."""
        if provider_type == "openai":
            if not openai_api_key:
                # Try to get from environment
                openai_api_key = os.environ.get("OPENAI_API_KEY", "")
            
            if not openai_api_key:
                logger.warning(
                    "OpenAI API key not provided, falling back to local embeddings"
                )
                provider_type = "local"
            else:
                try:
                    self.embedding_provider = OpenAIEmbeddingProvider(
                        api_key=openai_api_key,
                        model_name=openai_model_name
                    )
                    self.provider_type = "openai"
                    return
                except Exception as e:
                    logger.warning(f"Failed to initialize OpenAI embeddings: {e}")
                    logger.warning("Falling back to local embeddings")
                    provider_type = "local"
        
        # Local embeddings (default/fallback)
        self.embedding_provider = LocalEmbeddingProvider(
            model_name=local_model_name
        )
        self.provider_type = "local"
    
    def _init_collections(self):
        """Initialize or get existing collections with cosine similarity."""
        for name in [COLLECTION_DOCUMENTS, COLLECTION_DRAWINGS, 
                     COLLECTION_COORDINATES, COLLECTION_CONVERSATIONS]:
            try:
                try:
                    existing = self.client.get_collection(name=name)
                    if existing.metadata.get("hnsw:space") != "cosine":
                        logger.warning(f"Recreating '{name}' with cosine similarity...")
                        self.client.delete_collection(name=name)
                        self.collections[name] = self.client.create_collection(
                            name=name, metadata={"hnsw:space": "cosine"}
                        )
                    else:
                        self.collections[name] = existing
                except Exception:
                    self.collections[name] = self.client.create_collection(
                        name=name, metadata={"hnsw:space": "cosine"}
                    )
                logger.debug(f"Collection '{name}' initialized (count: {self.collections[name].count()})")
            except Exception as e:
                logger.error(f"Failed to initialize collection '{name}': {e}")
                raise
    
    def _generate_id(self, content: str, metadata: Optional[Dict] = None) -> str:
        """Generate a unique ID for a document based on content hash."""
        hash_input = content
        if metadata:
            hash_input += json.dumps(metadata, sort_keys=True)
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    # ==========================================================================
    # DOCUMENT OPERATIONS
    # ==========================================================================
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        collection: str = COLLECTION_DOCUMENTS
    ) -> List[str]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents with content, optional metadata and id
            collection: Which collection to add to
            
        Returns:
            List of document IDs that were added
        """
        if collection not in self.collections:
            raise ValueError(f"Unknown collection: {collection}")
        
        if not documents:
            return []
        
        # Prepare data for ChromaDB
        ids, contents, metadatas = [], [], []
        now = datetime.now().isoformat()
        
        for doc in documents:
            content = doc.get("content", "")
            if not content:
                continue
            
            metadata = doc.get("metadata", {}).copy()
            metadata.update({"added_at": now, "collection": collection})
            
            ids.append(doc.get("id") or self._generate_id(content, metadata))
            contents.append(content)
            metadatas.append(metadata)
        
        if not contents:
            return []
        
        # Generate embeddings and add to ChromaDB
        logger.info(f"Generating embeddings for {len(contents)} documents...")
        try:
            self.collections[collection].add(
                ids=ids,
                embeddings=self.embedding_provider.embed(contents),
                documents=contents,
                metadatas=metadatas
            )
            logger.info(f"✓ Added {len(ids)} documents to '{collection}'")
            return ids
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise
    
    def add_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        collection: str = COLLECTION_DOCUMENTS,
        doc_id: Optional[str] = None
    ) -> str:
        """
        Add a single text document to the vector store.
        
        Args:
            text: The text content
            metadata: Optional metadata dict
            collection: Which collection to add to
            doc_id: Optional document ID
            
        Returns:
            The document ID
        """
        ids = self.add_documents(
            [{"content": text, "metadata": metadata or {}, "id": doc_id}],
            collection=collection
        )
        return ids[0] if ids else ""
    
    # ==========================================================================
    # SEARCH OPERATIONS
    # ==========================================================================
    
    def search(
        self,
        query: str,
        collection: str = COLLECTION_DOCUMENTS,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
        include_embeddings: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents using semantic search.
        
        Args:
            query: The search query text
            collection: Which collection to search
            top_k: Number of results to return
            where: Optional metadata filter (ChromaDB where clause)
            include_embeddings: Whether to include embeddings in results
            
        Returns:
            List of search results, each with:
            - id: Document ID
            - content: Document text
            - metadata: Document metadata
            - score: Similarity score (higher is more similar)
            
        Example:
            >>> results = store.search("property boundaries", top_k=3)
            >>> for r in results:
            ...     print(f"{r['score']:.2f}: {r['content'][:50]}...")
        """
        if collection not in self.collections:
            raise ValueError(f"Unknown collection: {collection}")
        
        if not query:
            logger.warning("Empty query provided")
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_provider.embed_query(query)
        
        # Build include list
        include = ["documents", "metadatas", "distances"]
        if include_embeddings:
            include.append("embeddings")
        
        # Execute search
        try:
            results = self.collections[collection].query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where,
                include=include
            )
            
            # Format results (convert distance to similarity score for cosine)
            if not (results and results["ids"] and results["ids"][0]):
                return []
            
            formatted_results = [
                {
                    "id": doc_id,
                    "content": results["documents"][0][i] if results["documents"] else "",
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "score": 1 - results["distances"][0][i] if results["distances"] else 0,
                    **({"embedding": results["embeddings"][0][i]} if include_embeddings and results.get("embeddings") else {})
                }
                for i, doc_id in enumerate(results["ids"][0])
            ]
            
            logger.debug(f"Search returned {len(formatted_results)} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    def search_all_collections(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search across all collections.
        
        Args:
            query: The search query text
            top_k: Number of results per collection
            where: Optional metadata filter
            
        Returns:
            Dictionary mapping collection names to their results
        """
        results = {}
        for collection_name in self.collections:
            try:
                results[collection_name] = self.search(
                    query=query,
                    collection=collection_name,
                    top_k=top_k,
                    where=where
                )
            except Exception as e:
                logger.warning(f"Search in '{collection_name}' failed: {e}")
                results[collection_name] = []
        return results
    
    # ==========================================================================
    # SPECIALIZED ADD METHODS
    # ==========================================================================
    
    def add_autocad_entities(
        self,
        entities: List[Dict[str, Any]],
        drawing_name: str = ""
    ) -> List[str]:
        """
        Add AutoCAD entities to the drawings collection.
        
        Args:
            entities: List of AutoCAD entity dictionaries with:
                     - type: Entity type (LINE, POLYLINE, TEXT, etc.)
                     - layer: Layer name
                     - content: Text content (for TEXT/MTEXT entities)
                     - properties: Other entity properties
            drawing_name: Name of the source drawing
            
        Returns:
            List of document IDs
        """
        documents = []
        for entity in entities:
            # Create searchable content from entity data
            content_parts = [
                f"Entity Type: {entity.get('type', 'unknown')}",
                f"Layer: {entity.get('layer', 'unknown')}",
            ]
            
            if entity.get("content"):
                content_parts.append(f"Content: {entity.get('content')}")
            
            if entity.get("properties"):
                for key, value in entity.get("properties", {}).items():
                    content_parts.append(f"{key}: {value}")
            
            content = "\n".join(content_parts)
            
            metadata = {
                "entity_type": entity.get("type", "unknown"),
                "layer": entity.get("layer", "unknown"),
                "drawing_name": drawing_name,
                "source": "autocad"
            }
            
            documents.append({"content": content, "metadata": metadata})
        
        return self.add_documents(documents, collection=COLLECTION_DRAWINGS)
    
    def add_coordinates(
        self,
        coordinates: List[Dict[str, Any]],
        source: str = ""
    ) -> List[str]:
        """
        Add coordinate data to the coordinates collection.
        
        Args:
            coordinates: List of coordinate dictionaries with:
                        - x/easting: X coordinate
                        - y/northing: Y coordinate
                        - z/elevation: Optional Z coordinate
                        - name/point_id: Point identifier
                        - description: Optional description
            source: Source of the coordinates (file name, etc.)
            
        Returns:
            List of document IDs
        """
        documents = []
        for coord in coordinates:
            # Extract coordinate values
            x = coord.get("x") or coord.get("easting") or coord.get("E") or 0
            y = coord.get("y") or coord.get("northing") or coord.get("N") or 0
            z = coord.get("z") or coord.get("elevation") or coord.get("Z") or None
            
            name = coord.get("name") or coord.get("point_id") or coord.get("id") or ""
            desc = coord.get("description") or coord.get("desc") or ""
            
            # Create searchable content
            content_parts = [
                f"Point: {name}" if name else "Point",
                f"Coordinates: X={x}, Y={y}",
            ]
            if z is not None:
                content_parts.append(f"Elevation: {z}")
            if desc:
                content_parts.append(f"Description: {desc}")
            
            content = "\n".join(content_parts)
            
            metadata = {
                "x": float(x),
                "y": float(y),
                "z": float(z) if z is not None else None,
                "point_name": str(name),
                "source": source,
                "type": "coordinate"
            }
            
            documents.append({"content": content, "metadata": metadata})
        
        return self.add_documents(documents, collection=COLLECTION_COORDINATES)
    
    def add_conversation(
        self,
        role: str,
        content: str,
        session_id: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a conversation message to the conversations collection.
        
        Args:
            role: Message role ("user", "assistant", "system")
            content: Message content
            session_id: Conversation session ID
            metadata: Optional additional metadata
            
        Returns:
            Document ID
        """
        now = datetime.now()
        meta = (metadata or {}).copy()
        meta.update({
            "role": role,
            "session_id": session_id,
            "timestamp": now.isoformat()
        })
        
        # Generate unique ID with session, timestamp, and content hash for ordering
        unique_id = f"{session_id}_{now.timestamp()}_{hashlib.md5(content.encode()).hexdigest()[:8]}"
        
        return self.add_text(
            text=content,
            metadata=meta,
            collection=COLLECTION_CONVERSATIONS,
            doc_id=unique_id
        )
    
    def get_recent_conversations(
        self,
        session_id: str,
        limit: int = 10,
        role: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get recent conversations from a specific session, ordered by timestamp.
        
        Args:
            session_id: Session ID to filter by
            limit: Maximum number of conversations to return
            role: Optional role filter ("user", "assistant", "system")
            
        Returns:
            List of conversation dicts with id, content, metadata, ordered by timestamp
        """
        if COLLECTION_CONVERSATIONS not in self.collections:
            return []
        
        try:
            where_clause = {"session_id": session_id}
            if role:
                where_clause["role"] = role
            
            results = self.collections[COLLECTION_CONVERSATIONS].get(
                where=where_clause, include=["documents", "metadatas"]
            )
            
            if not results or not results.get("ids"):
                return []
            
            # Format and sort by timestamp (most recent first)
            conversations = [
                {
                    "id": doc_id,
                    "content": results["documents"][i] if results["documents"] else "",
                    "metadata": results["metadatas"][i] if results["metadatas"] else {},
                    "timestamp": (results["metadatas"][i] if results["metadatas"] else {}).get("timestamp", "")
                }
                for i, doc_id in enumerate(results["ids"])
            ]
            conversations.sort(key=lambda x: x["timestamp"], reverse=True)
            return conversations[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get recent conversations: {e}")
            return []
    
    # ==========================================================================
    # COLLECTION MANAGEMENT
    # ==========================================================================
    
    def get_collection_count(self, collection: str = COLLECTION_DOCUMENTS) -> int:
        """Get the number of documents in a collection."""
        if collection not in self.collections:
            raise ValueError(f"Unknown collection: {collection}")
        return self.collections[collection].count()
    
    def get_all_counts(self) -> Dict[str, int]:
        """Get document counts for all collections."""
        return {name: col.count() for name, col in self.collections.items()}
    
    def clear_collection(self, collection: str) -> None:
        """
        Clear all documents from a collection.
        
        Args:
            collection: Name of the collection to clear
        """
        if collection not in self.collections:
            raise ValueError(f"Unknown collection: {collection}")
        
        # Delete and recreate the collection
        self.client.delete_collection(collection)
        self.collections[collection] = self.client.create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"✓ Cleared collection '{collection}'")
    
    def clear_all_collections(self) -> None:
        """Clear all documents from all collections."""
        for name in list(self.collections.keys()):
            self.clear_collection(name)
        logger.info("✓ Cleared all collections")
    
    def delete_by_ids(
        self,
        ids: List[str],
        collection: str = COLLECTION_DOCUMENTS
    ) -> None:
        """
        Delete documents by their IDs.
        
        Args:
            ids: List of document IDs to delete
            collection: Collection to delete from
        """
        if collection not in self.collections:
            raise ValueError(f"Unknown collection: {collection}")
        
        self.collections[collection].delete(ids=ids)
        logger.info(f"✓ Deleted {len(ids)} documents from '{collection}'")
    
    def get_by_id(
        self,
        doc_id: str,
        collection: str = COLLECTION_DOCUMENTS
    ) -> Optional[Dict[str, Any]]:
        """
        Get a document by its ID.
        
        Args:
            doc_id: Document ID
            collection: Collection to search
            
        Returns:
            Document dict or None if not found
        """
        if collection not in self.collections:
            raise ValueError(f"Unknown collection: {collection}")
        
        result = self.collections[collection].get(
            ids=[doc_id],
            include=["documents", "metadatas"]
        )
        
        if result and result["ids"]:
            return {
                "id": result["ids"][0],
                "content": result["documents"][0] if result["documents"] else "",
                "metadata": result["metadatas"][0] if result["metadatas"] else {}
            }
        return None
    
    # ==========================================================================
    # UTILITIES
    # ==========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with stats including:
            - persist_directory: Storage location
            - embedding_provider: Which provider is being used
            - embedding_dimension: Dimension of embeddings
            - collections: Count per collection
        """
        return {
            "persist_directory": self.persist_directory,
            "embedding_provider": self.provider_type,
            "embedding_model": (
                self.embedding_provider.model_name 
                if hasattr(self.embedding_provider, "model_name") 
                else "unknown"
            ),
            "embedding_dimension": self.embedding_provider.dimension,
            "collections": self.get_all_counts(),
            "total_documents": sum(self.get_all_counts().values())
        }
    
    def reset(self) -> None:
        """
        Reset the entire vector store (delete all data).
        
        WARNING: This will permanently delete all stored data!
        """
        logger.warning("Resetting vector store - all data will be deleted!")
        self.client.reset()
        self._init_collections()
        logger.info("✓ Vector store reset complete")


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def create_vector_store(
    persist_directory: Optional[str] = None,
    use_openai: bool = False,
    openai_api_key: Optional[str] = None
) -> VectorStore:
    """
    Convenience function to create a VectorStore instance.
    
    Args:
        persist_directory: Where to store the database
        use_openai: Whether to use OpenAI embeddings
        openai_api_key: OpenAI API key (if using OpenAI)
        
    Returns:
        Configured VectorStore instance
    """
    return VectorStore(
        persist_directory=persist_directory,
        embedding_provider="openai" if use_openai else "local",
        openai_api_key=openai_api_key
    )


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

__all__ = [
    "VectorStore",
    "LocalEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "create_vector_store",
    "COLLECTION_DOCUMENTS",
    "COLLECTION_DRAWINGS",
    "COLLECTION_COORDINATES",
    "COLLECTION_CONVERSATIONS",
]

