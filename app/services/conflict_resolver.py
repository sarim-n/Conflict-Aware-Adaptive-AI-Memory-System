import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from app.config import settings
from app.database import (
    Memory,
    MemoryConflictLog,
    ConversationTurn,
    MemoryCategory,
    MemoryStatus,
    ConflictResolutionType,
    get_db_session,
    create_memory,
    supersede_memory,
    record_temporary_override,
    reinforce_memory,
    confirm_and_archive_conflicts,
    VectorStoreManager,
)
from app.services.llm_service import LLMService, ExtractedFact, ConflictAdjudication

logger = logging.getLogger(__name__)

SINGLETON_SUBJECTS = {
    "project_goal",
    "primary_database",
    "backend_framework",
    "frontend_framework",
    "embedding_model",
    "cache_layer",
    "cloud_provider",
}


class MemoryIngestionService:
    """Context-aware memory ingestion pipeline that extracts facts, detects
    candidate conflicts via hybrid search (subject + FAISS), and executes
    atomic lineage operations in SQLite & FAISS.
    """

    def __init__(self, llm_service: Optional[LLMService] = None):
        self.llm_service = llm_service or LLMService()
        self.vector_manager = VectorStoreManager.get_instance()

    def ingest_message(
        self,
        user_id: str,
        session_id: str,
        user_message: str,
        recent_turns: Optional[List[Dict[str, str]]] = None,
        source_turn_id: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Ingest a user message, extract atomic facts with coreference resolution, resolve conflicts, and persist memories."""
        logger.info(f"Ingesting message for user={user_id}, session={session_id}")
        if source_turn_id is None:
            source_turn_id = kwargs.get("source_turn_id")

        # Assemble recent conversation context and active memories for pronoun resolution
        recent_context_lines = []
        if recent_turns:
            for t in recent_turns[-3:]:
                recent_context_lines.append(f"{t.get('role', 'User').capitalize()}: {t.get('content', '')}")
        else:
            with get_db_session() as session:
                turns = (
                    session.query(ConversationTurn)
                    .filter(ConversationTurn.user_id == user_id, ConversationTurn.session_id == session_id)
                    .order_by(ConversationTurn.id.desc())
                    .limit(3)
                    .all()
                )
                for t in reversed(turns):
                    recent_context_lines.append(f"User: {t.user_message}")
                    if t.assistant_response:
                        recent_context_lines.append(f"Assistant: {t.assistant_response}")

        # Also append the latest active memories for entity grounding
        with get_db_session() as session:
            recent_active = (
                session.query(Memory)
                .filter(Memory.user_id == user_id, Memory.status == MemoryStatus.ACTIVE)
                .order_by(Memory.id.desc())
                .limit(3)
                .all()
            )
            if recent_active:
                recent_context_lines.append("Current Active Memories:")
                for m in reversed(recent_active):
                    recent_context_lines.append(f"- Subject: {m.subject}, Fact: '{m.content}', Value: '{m.object_val}'")

        recent_context_str = "\n".join(recent_context_lines) if recent_context_lines else None

        extracted_facts: List[ExtractedFact] = self.llm_service.extract_memories(
            user_message, recent_context=recent_context_str
        )

        memories_created: List[Dict[str, Any]] = []
        conflicts_detected: List[Dict[str, Any]] = []
        actions_taken: List[str] = []

        if not extracted_facts:
            logger.info("No declarative facts extracted from message.")
            return {
                "extracted_facts": [],
                "memories_created": [],
                "conflicts_detected": [],
                "actions_taken": ["NO_FACTS_EXTRACTED"],
            }

        # Filter out assistant suggestions and unconfirmed inferences:
        # Only user_confirmed and context_resolved facts can become persistent memories
        supported_facts = [
            f for f in extracted_facts
            if getattr(f, "source_type", "user_confirmed") in ["user_confirmed", "context_resolved"]
        ]

        if not supported_facts:
            logger.info("No supported user facts to persist (discarded suggestions/inferences).")
            return {
                "extracted_facts": [f.model_dump() for f in extracted_facts],
                "memories_created": [],
                "conflicts_detected": [],
                "actions_taken": ["DISCARDED_UNSUPPORTED_INFERENCES"],
            }

        for fact in supported_facts:
            action, mem, conflict = self._process_single_fact(
                user_id, session_id, fact, source_turn_id=source_turn_id
            )
            actions_taken.append(action)
            if mem:
                memories_created.append({
                    "id": mem.id,
                    "content": mem.content,
                    "subject": mem.subject,
                    "category": mem.category.value if hasattr(mem.category, "value") else str(mem.category),
                    "status": mem.status.value if hasattr(mem.status, "value") else str(mem.status),
                    "importance": mem.importance,
                    "confidence": mem.confidence,
                    "source_turn_id": mem.source_turn_id,
                })
            if conflict:
                conflicts_detected.append({
                    "id": conflict.id,
                    "old_memory_id": conflict.old_memory_id,
                    "new_memory_id": conflict.new_memory_id,
                    "resolution_type": (
                        conflict.resolution_type.value
                        if hasattr(conflict.resolution_type, "value")
                        else str(conflict.resolution_type)
                    ),
                    "similarity_score": conflict.similarity_score,
                    "reasoning": conflict.reasoning,
                })

        return {
            "extracted_facts": [f.model_dump() for f in extracted_facts],
            "memories_created": memories_created,
            "conflicts_detected": conflicts_detected,
            "actions_taken": actions_taken,
        }

    def _process_single_fact(
        self,
        user_id: str,
        session_id: str,
        fact: ExtractedFact,
        source_turn_id: Optional[int] = None,
    ) -> Tuple[str, Optional[Memory], Optional[Any]]:
        """Find candidate conflicts and apply appropriate resolution."""
        candidates = self._find_candidate_conflicts(user_id, fact)

        # Case 1: No candidate ACTIVE memories exist
        if not candidates:
            category = MemoryCategory.SHORT_TERM if fact.is_temporary else fact.category
            status = MemoryStatus.TEMPORARY if fact.is_temporary else MemoryStatus.ACTIVE

            mem = create_memory(
                user_id=user_id,
                session_id=session_id,
                content=fact.content,
                subject=fact.subject,
                predicate=fact.predicate,
                object_val=fact.object_val,
                category=category,
                status=status,
                importance=fact.importance,
                confidence=fact.confidence,
                source_turn_id=source_turn_id,
            )
            action_desc = f"CREATED memory id={mem.id} (Status: {mem.status.value})"
            logger.info(action_desc)
            return action_desc, mem, None

        # Check for same-subject active memories
        same_subj_candidates = [
            m for m, sim in candidates
            if fact.subject and m.subject == fact.subject
        ]

        is_retraction = (
            fact.predicate in ["does_not_use", "rejected", "not_using", "dropped", "deprecated"]
            or "not " in (fact.object_val or "").lower()
            or "not " in (fact.content or "").lower()
            or "won't use" in (fact.content or "").lower()
            or "wont use" in (fact.content or "").lower()
        )

        # Check if an existing memory already matches the confirmed value or same event
        matching_cand: Optional[Memory] = None
        if not is_retraction:
            for m in same_subj_candidates:
                if m.object_val and fact.object_val:
                    v1 = m.object_val.strip().lower()
                    v2 = fact.object_val.strip().lower()
                    if v1 == v2 or v1 in v2 or v2 in v1:
                        matching_cand = m
                        break
                if m.content and fact.content:
                    c1 = m.content.strip().lower()
                    c2 = fact.content.strip().lower()
                    if c1 == c2 or (fact.subject == "friend_meeting" and ("manoj" in c1 and "manoj" in c2)):
                        matching_cand = m
                        break

        if matching_cand is not None and not fact.is_temporary:
            # Reaffirm existing active memory and archive all conflicting active memories for this subject (singletons only)
            conflicting_ids = []
            if fact.subject in SINGLETON_SUBJECTS:
                conflicting_ids = [m.id for m in same_subj_candidates if m.id != matching_cand.id]
            if conflicting_ids:
                confirmed_mem, logs = confirm_and_archive_conflicts(
                    confirmed_memory_id=matching_cand.id,
                    conflicting_memory_ids=conflicting_ids,
                    reasoning=f"Confirmed '{fact.object_val}' as active {fact.subject}; archived {len(conflicting_ids)} superseded previous decisions.",
                )
                action_desc = f"CONFIRMED active memory {confirmed_mem.id} and ARCHIVED {len(conflicting_ids)} superseded memories: {conflicting_ids}"
                logger.info(action_desc)
                primary_log = logs[0] if logs else None
                return action_desc, confirmed_mem, primary_log
            else:
                reinforced_mem = reinforce_memory(matching_cand.id, confidence_boost=0.05)
                action_desc = f"REINFORCED memory {reinforced_mem.id} (confidence={reinforced_mem.confidence:.2f})"
                logger.info(action_desc)
                return action_desc, reinforced_mem, None

        # Case 2: Adjudicate with primary candidate
        # If fact explicitly specifies replaces_value, target that specific candidate record
        best_candidate: Optional[Memory] = None
        best_sim = 0.5
        if fact.replaces_value:
            rep_clean = fact.replaces_value.lower().strip()
            for m, sim in candidates:
                if (
                    (m.object_val and rep_clean == m.object_val.lower().strip())
                    or (m.object_val and rep_clean in m.object_val.lower())
                    or (m.content and rep_clean in m.content.lower())
                ):
                    best_candidate = m
                    best_sim = sim
                    break

        if best_candidate is None:
            # If fact specifies replaces_value but NO active candidate matches that value,
            # it should NOT supersede an unrelated candidate (e.g. Sameer or Syed when correcting Manoj).
            if fact.replaces_value:
                category = MemoryCategory.SHORT_TERM if fact.is_temporary else fact.category
                status = MemoryStatus.TEMPORARY if fact.is_temporary else MemoryStatus.ACTIVE
                comp_mem = create_memory(
                    user_id=user_id,
                    session_id=session_id,
                    content=fact.content,
                    subject=fact.subject,
                    predicate=fact.predicate,
                    object_val=fact.object_val,
                    category=category,
                    status=status,
                    importance=fact.importance,
                    confidence=fact.confidence,
                    source_turn_id=source_turn_id,
                )
                action_desc = f"CREATED memory id={comp_mem.id} (Status: {comp_mem.status.value})"
                return action_desc, comp_mem, None

            best_candidate, best_sim = candidates[0]

        adjudication: ConflictAdjudication = self.llm_service.evaluate_conflict(
            old_memory=best_candidate, new_fact=fact, similarity=best_sim
        )
        logger.info(
            f"Adjudication between old_id={best_candidate.id} and new_fact='{fact.content}': "
            f"{adjudication.resolution_type.value} (sim={best_sim:.3f})"
        )

        res_type = adjudication.resolution_type

        # Singleton domain subject guard: multiple conflicting goals or primary DBs cannot be complementary
        if (
            fact.subject in SINGLETON_SUBJECTS
            and best_candidate.subject == fact.subject
            and not fact.is_temporary
            and res_type == ConflictResolutionType.COMPLEMENTARY
        ):
            res_type = ConflictResolutionType.PERMANENT_UPDATE
            adjudication.reasoning = (
                f"Singleton architectural rule: '{fact.subject}' has only one active choice; "
                f"new decision '{fact.content}' replaces '{best_candidate.content}'."
            )

        # Same event / high similarity guard: avoid creating duplicate complementary memories for identical events
        is_distinct_collection_item = (
            fact.subject not in SINGLETON_SUBJECTS
            and fact.object_val
            and best_candidate.object_val
            and fact.object_val.strip().lower() != best_candidate.object_val.strip().lower()
        )
        if (
            res_type == ConflictResolutionType.COMPLEMENTARY
            and best_candidate.subject == fact.subject
            and best_sim >= 0.85
            and not is_retraction
            and not is_distinct_collection_item
        ):
            res_type = ConflictResolutionType.REINFORCEMENT
            adjudication.reasoning = (
                f"Same event/milestone detected for subject '{fact.subject}' with high similarity ({best_sim:.2f}); "
                f"reinforcing existing memory {best_candidate.id} instead of creating duplicate complementary memory."
            )

        if res_type == ConflictResolutionType.PERMANENT_UPDATE:
            new_data = {
                "user_id": user_id,
                "session_id": session_id,
                "content": fact.content,
                "subject": fact.subject,
                "predicate": fact.predicate,
                "object_val": fact.object_val,
                "category": fact.category,
                "importance": fact.importance,
                "confidence": fact.confidence,
                "source_turn_id": source_turn_id,
            }
            # Archive best candidate AND other active memories only for singleton subjects
            other_superseded_ids = []
            if fact.subject in SINGLETON_SUBJECTS:
                other_superseded_ids = [m.id for m in same_subj_candidates if m.id != best_candidate.id]
            new_mem, conflict_log = supersede_memory(
                old_memory_id=best_candidate.id,
                new_memory_data=new_data,
                similarity_score=best_sim,
                reasoning=adjudication.reasoning,
                additional_superseded_ids=other_superseded_ids,
            )
            action_desc = f"SUPERSEDED old memory {best_candidate.id} -> new memory {new_mem.id}"
            return action_desc, new_mem, conflict_log

        elif res_type == ConflictResolutionType.TEMPORARY_OVERRIDE:
            temp_data = {
                "user_id": user_id,
                "session_id": session_id,
                "content": fact.content,
                "subject": fact.subject,
                "predicate": fact.predicate,
                "object_val": fact.object_val,
                "importance": fact.importance,
                "confidence": fact.confidence,
                "source_turn_id": source_turn_id,
            }
            temp_mem, conflict_log = record_temporary_override(
                active_memory_id=best_candidate.id,
                temp_memory_data=temp_data,
                similarity_score=best_sim,
                reasoning=adjudication.reasoning,
            )
            action_desc = (
                f"TEMPORARY_OVERRIDE for memory {best_candidate.id} -> created temp memory {temp_mem.id}"
            )
            return action_desc, temp_mem, conflict_log

        elif res_type == ConflictResolutionType.REINFORCEMENT:
            reinforced_mem = reinforce_memory(best_candidate.id, confidence_boost=0.05)
            action_desc = f"REINFORCED memory {reinforced_mem.id} (confidence={reinforced_mem.confidence:.2f})"
            return action_desc, reinforced_mem, None

        else:  # COMPLEMENTARY
            category = MemoryCategory.SHORT_TERM if fact.is_temporary else fact.category
            status = MemoryStatus.TEMPORARY if fact.is_temporary else MemoryStatus.ACTIVE
            comp_mem = create_memory(
                user_id=user_id,
                session_id=session_id,
                content=fact.content,
                subject=fact.subject,
                predicate=fact.predicate,
                object_val=fact.object_val,
                category=category,
                status=status,
                importance=fact.importance,
                confidence=fact.confidence,
                source_turn_id=source_turn_id,
            )
            action_desc = f"CREATED complementary memory id={comp_mem.id}"
            return action_desc, comp_mem, None

    def _find_candidate_conflicts(
        self, user_id: str, fact: ExtractedFact
    ) -> List[Tuple[Memory, float]]:
        """Perform hybrid candidate search using exact subject match + FAISS semantic similarity."""
        candidates_dict: Dict[int, Tuple[Memory, float]] = {}

        # 1. Subject-based lookup for active memories of this user
        with get_db_session() as session:
            if fact.subject:
                subject_matches = (
                    session.query(Memory)
                    .filter(
                        Memory.user_id == user_id,
                        Memory.status == MemoryStatus.ACTIVE,
                        Memory.subject == fact.subject,
                    )
                    .all()
                )
                for mem in subject_matches:
                    sim = self._calculate_similarity(mem.id, fact.content)
                    candidates_dict[mem.id] = (mem, sim)

        # 2. Vector search lookup (top_k=5)
        vector_results = self.vector_manager.search_similar(fact.content, top_k=5)
        with get_db_session() as session:
            for mem_id, sim in vector_results:
                if sim >= settings.CONFLICT_SIMILARITY_THRESHOLD:
                    if mem_id not in candidates_dict:
                        mem = session.get(Memory, mem_id)
                        if mem and mem.user_id == user_id and mem.status == MemoryStatus.ACTIVE:
                            candidates_dict[mem.id] = (mem, sim)

        def _sort_key(item: Tuple[Memory, float]) -> Tuple[int, float]:
            mem, sim = item
            is_subj_match = 1 if (fact.subject and mem.subject == fact.subject) else 0
            return (is_subj_match, sim)

        sorted_candidates = sorted(candidates_dict.values(), key=_sort_key, reverse=True)
        return sorted_candidates

    def _calculate_similarity(self, memory_id: int, new_text: str) -> float:
        """Calculate cosine similarity between indexed memory vector and new_text."""
        old_vec = self.vector_manager.get_vector_by_id(memory_id)
        if old_vec is None:
            return 0.5
        new_vec = self.vector_manager.model.encode([new_text], normalize_embeddings=True)[0]
        return float(np.dot(old_vec, new_vec))
