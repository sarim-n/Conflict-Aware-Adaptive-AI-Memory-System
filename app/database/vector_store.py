import logging
import os
import threading
from pathlib import Path
from typing import List, Optional, Tuple, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Synchronized FAISS vector store with SentenceTransformers embeddings.
    
    Uses faiss.IndexIDMap2 on top of IndexFlatIP (Inner Product) with normalized
    embeddings to provide cosine similarity search indexed directly by SQLite Memory.id.
    """

    _instance: Optional["VectorStoreManager"] = None
    _lock = threading.Lock()

    def __init__(
        self,
        model_name: Optional[str] = None,
        dimension: Optional[int] = None,
        index_path: Optional[str] = None,
    ):
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.dimension = dimension or settings.EMBEDDING_DIM
        self.index_path = index_path or settings.FAISS_INDEX_PATH
        self._rw_lock = threading.Lock()

        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)

        self._load_or_create_index()

    @classmethod
    def get_instance(cls) -> "VectorStoreManager":
        """Thread-safe singleton accessor."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (useful for isolated tests)."""
        with cls._lock:
            cls._instance = None

    def _init_empty_index(self) -> None:
        """Create a fresh IndexIDMap2 wrapping IndexFlatIP."""
        sub_index = faiss.IndexFlatIP(self.dimension)
        self.index = faiss.IndexIDMap2(sub_index)

    def _load_or_create_index(self) -> None:
        """Load existing index from disk or initialize an empty one."""
        with self._rw_lock:
            if os.path.exists(self.index_path):
                try:
                    logger.info(f"Loading FAISS index from {self.index_path}")
                    self.index = faiss.read_index(self.index_path)
                    logger.info(f"Loaded index with {self.index.ntotal} vectors")
                except Exception as e:
                    logger.warning(
                        f"Failed reading FAISS index from {self.index_path}: {e}. Creating new empty index."
                    )
                    self._init_empty_index()
            else:
                logger.info(f"No existing index at {self.index_path}. Creating new index.")
                self._init_empty_index()

    def save(self) -> None:
        """Persist index to disk."""
        target_path = Path(self.index_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(target_path))

    def add_memory_vector(self, memory_id: int, text: str) -> None:
        """Encode text, add/replace in FAISS index with memory_id, and persist."""
        with self._rw_lock:
            # Check if vector already exists; remove first if so to avoid duplicates
            try:
                self.index.remove_ids(np.array([memory_id], dtype=np.int64))
            except Exception:
                pass

            vector = self.model.encode([text], normalize_embeddings=True)
            vector = np.ascontiguousarray(vector, dtype=np.float32)
            ids = np.array([memory_id], dtype=np.int64)

            self.index.add_with_ids(vector, ids)
            self.save()
            logger.debug(f"Indexed memory id={memory_id} (ntotal={self.index.ntotal})")

    def search_similar(
        self, query_text: str, top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """Search top_k similar memories for query_text.
        
        Returns:
            List of (memory_id, cosine_similarity_score) tuples, sorted descending.
        """
        with self._rw_lock:
            if self.index.ntotal == 0:
                return []

            query_vec = self.model.encode([query_text], normalize_embeddings=True)
            query_vec = np.ascontiguousarray(query_vec, dtype=np.float32)

            k = min(top_k, self.index.ntotal)
            distances, indices = self.index.search(query_vec, k)

            results: List[Tuple[int, float]] = []
            for idx, dist in zip(indices[0], distances[0]):
                if idx != -1:
                    results.append((int(idx), float(dist)))

            return results

    def get_vector_by_id(self, memory_id: int) -> Optional[np.ndarray]:
        """Reconstruct the vector for memory_id from FAISS index."""
        with self._rw_lock:
            try:
                vec = self.index.reconstruct(int(memory_id))
                return np.array(vec, dtype=np.float32)
            except Exception:
                return None

    def rebuild_from_db(self, memories: List[Any]) -> None:
        """Rebuild entire FAISS index from a list of Memory records."""
        with self._rw_lock:
            self._init_empty_index()
            if not memories:
                self.save()
                return

            ids = np.array([m.id for m in memories], dtype=np.int64)
            texts = [m.content for m in memories]

            vectors = self.model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            )
            vectors = np.ascontiguousarray(vectors, dtype=np.float32)

            self.index.add_with_ids(vectors, ids)
            self.save()
            logger.info(
                f"Rebuilt FAISS index with {len(memories)} memories from database."
            )
