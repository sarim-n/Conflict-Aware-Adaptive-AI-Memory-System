import logging
import math
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from app.config import settings
from app.database import (
    Memory,
    MemoryStatus,
    get_db_session,
    get_memory_lineage,
    VectorStoreManager,
)

logger = logging.getLogger(__name__)

HISTORICAL_KEYWORDS = [
    "before",
    "previously",
    "used to",
    "history",
    "migrated from",
    "old",
    "past",
    "previous",
    "earlier",
    "prior",
]

USER_PROJECT_INTENT_KEYWORDS = [
    "database", "db", "project", "architecture", "stack", "framework",
    "backend", "frontend", "we", "our", "my", "i", "used to", "before",
    "history", "what are we", "what is our", "what did we", "embedding", "goal",
    "theme", "preference", "role", "title", "who am i", "what am i", "migrated",
    "using", "chatbot", "engine", "cache", "redis", "cloud", "model",
    "friend", "friends", "colleague", "colleagues", "coworker", "coworkers",
    "teammate", "teammates", "contact", "contacts", "hobby", "hobbies",
    "favorite", "favorites", "pet", "pets", "family", "brother", "sister"
]


class MemoryRetriever:
    """Memory retriever comparing pure FAISS baseline vs. Adaptive Composite Retrieval."""

    def __init__(self):
        self.vector_manager = VectorStoreManager.get_instance()

    def has_historical_intent(self, query: str) -> bool:
        """Check if query is asking for previous/historical facts."""
        query_low = query.lower()
        return any(keyword in query_low for keyword in HISTORICAL_KEYWORDS)

    def has_project_intent(self, query: str) -> bool:
        """Check if query specifically asks about user/project state or preferences."""
        query_low = query.lower()
        return any(keyword in query_low for keyword in USER_PROJECT_INTENT_KEYWORDS)

    # =========================================================================
    # 1. PURE FAISS BASELINE RETRIEVAL
    # =========================================================================
    def retrieve_basic(
        self, user_id: str, query: str, top_k: int = 5
    ) -> Dict[str, Any]:
        """Pure FAISS vector search baseline without status filtering or composite scoring.
        
        Demonstrates the baseline failure mode where old historical memories outrank
        new active memories based solely on raw embedding distance.
        """
        start_time = time.perf_counter()

        raw_results = self.vector_manager.search_similar(query, top_k=top_k * 2)

        results: List[Dict[str, Any]] = []
        with get_db_session() as session:
            for mem_id, sim in raw_results:
                mem = session.get(Memory, mem_id)
                if mem and mem.user_id == user_id:
                    results.append({
                        "id": mem.id,
                        "content": mem.content,
                        "status": mem.status.value,
                        "category": mem.category.value,
                        "similarity": round(float(sim), 4),
                        "importance": mem.importance,
                        "confidence": mem.confidence,
                        "created_at": mem.created_at.isoformat(),
                        "valid_until": mem.valid_until.isoformat() if mem.valid_until else None,
                    })
                if len(results) >= top_k:
                    break

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "query": query,
            "mode": "basic_vector",
            "latency_ms": latency_ms,
            "count": len(results),
            "results": results,
        }

    # =========================================================================
    # 2. ADAPTIVE COMPOSITE RETRIEVAL
    # =========================================================================
    def retrieve_adaptive(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        include_historical_context: bool = True,
    ) -> Dict[str, Any]:
        """Adaptive Memory Retrieval incorporating:
        1. BaseScore = w_s*Similarity + w_i*Importance + w_r*Recency + w_c*Confidence
        2. Status Gate (Psi):
           - ACTIVE: 1.0
           - TEMPORARY: 0.95 (session override boost)
           - HISTORICAL: 1.0 if query has historical intent, else 0.15 (demoted)
        3. Automatic historical lineage attachment for superseded memories.
        """
        start_time = time.perf_counter()
        is_historical = self.has_historical_intent(query)
        now_utc = datetime.now(timezone.utc)

        # 1. Fetch wide candidate pool from FAISS
        candidate_ids_sim: Dict[int, float] = {}
        raw_faiss = self.vector_manager.search_similar(query, top_k=top_k * 3)
        for mid, sim in raw_faiss:
            candidate_ids_sim[mid] = float(sim)

        # 2. Also fetch any active TEMPORARY session memories
        with get_db_session() as session:
            temp_memories = (
                session.query(Memory)
                .filter(
                    Memory.user_id == user_id,
                    Memory.status == MemoryStatus.TEMPORARY,
                )
                .all()
            )
            query_vec = None
            for tm in temp_memories:
                if tm.id not in candidate_ids_sim:
                    if query_vec is None:
                        query_vec = self.vector_manager.model.encode(
                            [query], normalize_embeddings=True
                        )[0]
                    t_vec = self.vector_manager.get_vector_by_id(tm.id)
                    sim = float(np.dot(t_vec, query_vec)) if t_vec is not None else 0.5
                    candidate_ids_sim[tm.id] = sim

            # 3. Retrieve and score candidates
            scored_candidates: List[Dict[str, Any]] = []

            for mid, similarity in candidate_ids_sim.items():
                mem = session.get(Memory, mid)
                if not mem or mem.user_id != user_id:
                    continue

                # Recency score calculation with exponential decay
                last_accessed = mem.last_accessed_at
                if last_accessed.tzinfo is None:
                    last_accessed = last_accessed.replace(tzinfo=timezone.utc)
                delta_seconds = max(0.0, (now_utc - last_accessed).total_seconds())
                delta_hours = delta_seconds / 3600.0
                recency = math.exp(-settings.RECENCY_DECAY_LAMBDA * delta_hours)

                # Base Composite Score
                base_score = (
                    settings.WEIGHT_SIMILARITY * similarity
                    + settings.WEIGHT_IMPORTANCE * mem.importance
                    + settings.WEIGHT_RECENCY * recency
                    + settings.WEIGHT_CONFIDENCE * mem.confidence
                )

                # Context-Aware Status Gate (Psi)
                if mem.status == MemoryStatus.ACTIVE:
                    psi = 1.0
                elif mem.status == MemoryStatus.TEMPORARY:
                    psi = 0.95
                elif mem.status == MemoryStatus.HISTORICAL:
                    psi = 1.0 if is_historical else 0.15
                else:  # ARCHIVED
                    psi = 0.05

                adaptive_score = base_score * psi

                # Build lineage chain if active and has lineage
                lineage_chain = []
                if include_historical_context and mem.status == MemoryStatus.ACTIVE:
                    full_lineage = get_memory_lineage(mem.id, db=session)
                    lineage_chain = [
                        {
                            "id": anc.id,
                            "content": anc.content,
                            "status": anc.status.value,
                            "valid_until": anc.valid_until.isoformat() if anc.valid_until else None,
                        }
                        for anc in full_lineage
                        if anc.id != mem.id
                    ]

                is_project = self.has_project_intent(query)
                is_relevant = (similarity >= 0.28) or (is_project and similarity >= 0.18)

                scored_candidates.append({
                    "id": mem.id,
                    "content": mem.content,
                    "subject": mem.subject,
                    "predicate": mem.predicate,
                    "object_val": mem.object_val,
                    "status": mem.status.value,
                    "category": mem.category.value,
                    "similarity": round(similarity, 4),
                    "importance": round(mem.importance, 4),
                    "recency": round(recency, 4),
                    "confidence": round(mem.confidence, 4),
                    "status_multiplier": psi,
                    "base_score": round(base_score, 4),
                    "adaptive_score": round(adaptive_score, 4),
                    "lineage_chain": lineage_chain,
                    "is_relevant": is_relevant,
                })

        # 4. Sort descending by adaptive_score
        scored_candidates.sort(key=lambda x: x["adaptive_score"], reverse=True)
        top_results = scored_candidates[:top_k]

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "query": query,
            "mode": "adaptive",
            "historical_intent": is_historical,
            "has_project_intent": self.has_project_intent(query),
            "latency_ms": latency_ms,
            "count": len(top_results),
            "results": top_results,
        }
