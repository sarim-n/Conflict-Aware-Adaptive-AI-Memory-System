import logging
from contextlib import contextmanager
from typing import Generator, List, Optional, Tuple, Any, Dict

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session

from app.config import settings
from app.database.models import (
    Base,
    Memory,
    MemoryConflictLog,
    MemoryCategory,
    MemoryStatus,
    ConflictResolutionType,
    utc_now,
)
from app.database.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

# Engine configuration for SQLite WAL mode
connect_args = {}
if "sqlite" in settings.DATABASE_URL:
    connect_args = {"check_same_thread": False}

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    echo=False,
    future=True,
)


@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection: Any, connection_record: Any) -> None:
    """Enable SQLite WAL mode and enforce foreign keys for concurrency and integrity."""
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA foreign_keys=ON;")
    finally:
        cursor.close()


SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


def init_db() -> None:
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized successfully.")


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Context manager for database sessions with automatic commit/rollback."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def create_memory(
    user_id: str,
    content: str,
    subject: Optional[str] = None,
    predicate: Optional[str] = None,
    object_val: Optional[str] = None,
    category: MemoryCategory = MemoryCategory.LONG_TERM,
    status: MemoryStatus = MemoryStatus.ACTIVE,
    importance: float = 0.5,
    confidence: float = 1.0,
    session_id: Optional[str] = None,
    source_turn_id: Optional[int] = None,
    db: Optional[Session] = None,
) -> Memory:
    """Insert a Memory row into SQLite and index its vector in FAISS atomically.
    
    Returns:
        The created Memory instance.
    """
    def _execute(session: Session) -> Memory:
        now = utc_now()
        memory = Memory(
            user_id=user_id,
            session_id=session_id,
            content=content,
            subject=subject,
            predicate=predicate,
            object_val=object_val,
            category=category,
            status=status,
            importance=importance,
            confidence=confidence,
            source_turn_id=source_turn_id,
            created_at=now,
            last_accessed_at=now,
            valid_from=now,
        )
        session.add(memory)
        session.flush()  # Ensures memory.id is populated before FAISS indexing

        # Add vector to FAISS
        v_manager = VectorStoreManager.get_instance()
        v_manager.add_memory_vector(memory.id, memory.content)

        return memory

    if db is not None:
        return _execute(db)
    with get_db_session() as session:
        return _execute(session)


def supersede_memory(
    old_memory_id: int,
    new_memory_data: dict,
    similarity_score: float,
    reasoning: str,
    additional_superseded_ids: Optional[List[int]] = None,
    db: Optional[Session] = None,
) -> Tuple[Memory, MemoryConflictLog]:
    """Atomically supersede existing memory/memories with a new active memory.
    
    1. Finds all applicable ACTIVE memories for the same logical subject.
    2. Marks them as HISTORICAL with valid_until = utc_now().
    3. Creates new memory as ACTIVE with supersedes_id = old_memory_id.
    4. Links reciprocal pointers (superseded_by_id = new_memory.id).
    5. Records conflict resolution in MemoryConflictLog audit records.
    6. Indexes new memory vector in FAISS (retaining old vectors for historical queries).
    7. Atomically verifies the persisted state before returning.
    
    Returns:
        Tuple of (new_memory, primary_conflict_log).
    """
    def _execute(session: Session) -> Tuple[Memory, MemoryConflictLog]:
        old_mem = session.get(Memory, old_memory_id)
        if not old_mem:
            raise ValueError(f"Memory with id={old_memory_id} not found.")

        now = utc_now()
        user_id = new_memory_data.get("user_id", old_mem.user_id)
        subject = new_memory_data.get("subject", old_mem.subject)

        # Collect all active memories that this new decision supersedes
        to_archive: Dict[int, Memory] = {old_mem.id: old_mem}

        if additional_superseded_ids:
            for aid in additional_superseded_ids:
                if aid != old_mem.id:
                    extra = session.get(Memory, aid)
                    if extra and extra.status == MemoryStatus.ACTIVE:
                        to_archive[extra.id] = extra

        # Also auto-discover any remaining ACTIVE memories for the same user and logical subject
        if subject:
            same_subj_active = (
                session.query(Memory)
                .filter(
                    Memory.user_id == user_id,
                    Memory.subject == subject,
                    Memory.status == MemoryStatus.ACTIVE,
                )
                .all()
            )
            for m in same_subj_active:
                to_archive[m.id] = m

        # 1. Archive superseded memories
        for m in to_archive.values():
            m.status = MemoryStatus.HISTORICAL
            m.valid_until = now

        # 2. Create new active memory
        new_mem = Memory(
            user_id=user_id,
            session_id=new_memory_data.get("session_id", old_mem.session_id),
            content=new_memory_data["content"],
            subject=subject,
            predicate=new_memory_data.get("predicate", old_mem.predicate),
            object_val=new_memory_data.get("object_val", old_mem.object_val),
            category=new_memory_data.get("category", old_mem.category),
            status=MemoryStatus.ACTIVE,
            importance=new_memory_data.get("importance", old_mem.importance),
            confidence=new_memory_data.get("confidence", 1.0),
            source_turn_id=new_memory_data.get("source_turn_id", None),
            supersedes_id=old_mem.id,
            created_at=now,
            last_accessed_at=now,
            valid_from=now,
            valid_until=None,
        )
        session.add(new_mem)
        session.flush()

        # 3. Link reciprocal pointers for all archived memories
        for m in to_archive.values():
            m.superseded_by_id = new_mem.id

        # 4. Record primary conflict log
        primary_conflict_log = MemoryConflictLog(
            user_id=new_mem.user_id,
            old_memory_id=old_mem.id,
            new_memory_id=new_mem.id,
            similarity_score=similarity_score,
            resolution_type=ConflictResolutionType.PERMANENT_UPDATE,
            reasoning=reasoning,
            detected_at=now,
        )
        session.add(primary_conflict_log)

        # 5. Record conflict logs for any additional superseded memories
        for extra_id, extra_mem in to_archive.items():
            if extra_id != old_mem.id:
                extra_log = MemoryConflictLog(
                    user_id=new_mem.user_id,
                    old_memory_id=extra_id,
                    new_memory_id=new_mem.id,
                    similarity_score=similarity_score,
                    resolution_type=ConflictResolutionType.PERMANENT_UPDATE,
                    reasoning=f"Superseded as an older active decision for subject '{subject}' by '{new_mem.content}'.",
                    detected_at=now,
                )
                session.add(extra_log)

        session.flush()

        # 6. Index new memory in FAISS
        v_manager = VectorStoreManager.get_instance()
        v_manager.add_memory_vector(new_mem.id, new_mem.content)

        # 7. Verification check of persisted state
        if new_mem.id is None or new_mem.status != MemoryStatus.ACTIVE:
            raise RuntimeError("Atomic persistence verification failed: new memory is not ACTIVE.")
        for m in to_archive.values():
            if m.status != MemoryStatus.HISTORICAL or m.valid_until is None or m.superseded_by_id != new_mem.id:
                raise RuntimeError(f"Atomic persistence verification failed: memory #{m.id} was not properly archived.")

        return new_mem, primary_conflict_log

    if db is not None:
        return _execute(db)
    with get_db_session() as session:
        return _execute(session)


def confirm_and_archive_conflicts(
    confirmed_memory_id: int,
    conflicting_memory_ids: List[int],
    reasoning: str = "Superseded by confirmed active decision.",
    db: Optional[Session] = None,
) -> Tuple[Memory, List[MemoryConflictLog]]:
    """Reinforce an existing active memory and archive all conflicting active memories for the same subject.
    
    1. Keeps confirmed_memory_id ACTIVE (reinforces confidence & updates last_accessed_at).
    2. Marks conflicting_memory_ids as HISTORICAL with valid_until = utc_now().
    3. Links their superseded_by_id = confirmed_memory_id.
    4. Logs conflict resolutions as PERMANENT_UPDATE in audit log.
    5. Verifies atomic persistence.
    """
    def _execute(session: Session) -> Tuple[Memory, List[MemoryConflictLog]]:
        confirmed_mem = session.get(Memory, confirmed_memory_id)
        if not confirmed_mem:
            raise ValueError(f"Memory with id={confirmed_memory_id} not found.")

        now = utc_now()
        confirmed_mem.status = MemoryStatus.ACTIVE
        confirmed_mem.valid_until = None
        confirmed_mem.last_accessed_at = now
        confirmed_mem.access_count += 1
        confirmed_mem.confidence = min(1.0, confirmed_mem.confidence + 0.05)

        logs: List[MemoryConflictLog] = []
        for cid in conflicting_memory_ids:
            if cid == confirmed_memory_id:
                continue
            old_mem = session.get(Memory, cid)
            if old_mem and old_mem.status == MemoryStatus.ACTIVE:
                old_mem.status = MemoryStatus.HISTORICAL
                old_mem.valid_until = now
                old_mem.superseded_by_id = confirmed_mem.id
                log = MemoryConflictLog(
                    user_id=confirmed_mem.user_id,
                    old_memory_id=old_mem.id,
                    new_memory_id=confirmed_mem.id,
                    similarity_score=1.0,
                    resolution_type=ConflictResolutionType.PERMANENT_UPDATE,
                    reasoning=reasoning,
                    detected_at=now,
                )
                session.add(log)
                logs.append(log)

        session.flush()

        # Verify state
        if confirmed_mem.status != MemoryStatus.ACTIVE:
            raise RuntimeError("Atomic persistence verification failed: confirmed memory is not ACTIVE.")
        for cid in conflicting_memory_ids:
            if cid != confirmed_memory_id:
                m = session.get(Memory, cid)
                if m and (m.status != MemoryStatus.HISTORICAL or m.valid_until is None or m.superseded_by_id != confirmed_mem.id):
                    raise RuntimeError(f"Atomic persistence verification failed: memory #{cid} was not properly archived.")

        return confirmed_mem, logs

    if db is not None:
        return _execute(db)
    with get_db_session() as session:
        return _execute(session)


def repair_legacy_conflicts(
    user_id: str = "default_user",
    target_subject: str = "project_goal",
    confirmed_memory_id: Optional[int] = 15,
    db: Optional[Session] = None,
) -> Dict[str, Any]:
    """Safely reconcile legacy conflicting ACTIVE memories for a user and subject.
    
    If confirmed_memory_id is provided, preserves it as ACTIVE and transitions all
    other active memories for that subject to HISTORICAL with reciprocal lineage.
    """
    def _execute(session: Session) -> Dict[str, Any]:
        active_mems = (
            session.query(Memory)
            .filter(
                Memory.user_id == user_id,
                Memory.subject == target_subject,
                Memory.status == MemoryStatus.ACTIVE,
            )
            .order_by(Memory.id.asc())
            .all()
        )
        if len(active_mems) <= 1:
            return {"repaired": False, "message": "No multiple active memories found."}

        active_ids = [m.id for m in active_mems]
        chosen_id = confirmed_memory_id if confirmed_memory_id in active_ids else active_ids[-1]
        superseded_ids = [mid for mid in active_ids if mid != chosen_id]

        chosen_mem, logs = confirm_and_archive_conflicts(
            confirmed_memory_id=chosen_id,
            conflicting_memory_ids=superseded_ids,
            reasoning=f"Legacy reconciliation: '{chosen_id}' confirmed as active {target_subject}; superseded previous decisions.",
            db=session,
        )
        return {
            "repaired": True,
            "confirmed_id": chosen_id,
            "archived_ids": superseded_ids,
            "logs_created": len(logs),
        }

    if db is not None:
        return _execute(db)
    with get_db_session() as session:
        return _execute(session)


def backfill_memory_categories(
    user_id: Optional[str] = None,
    db: Optional[Session] = None,
) -> Dict[str, Any]:
    """Safely backfill and reclassify existing memory records that were previously defaulted to LONG_TERM.

    Preserves all records, lifecycle statuses (ACTIVE, HISTORICAL, TEMPORARY), lineages, and attributions.
    Updates only category based on classification rules.
    """
    def _execute(session: Session) -> Dict[str, Any]:
        from app.services.llm_service import classify_memory_category

        query = session.query(Memory)
        if user_id:
            query = query.filter(Memory.user_id == user_id)
        memories = query.order_by(Memory.id.asc()).all()

        updated_count = 0
        changes = []

        for mem in memories:
            old_cat = mem.category
            is_temp = (mem.status == MemoryStatus.TEMPORARY)
            new_cat = classify_memory_category(
                content=mem.content,
                subject=mem.subject,
                predicate=mem.predicate,
                object_val=mem.object_val,
                is_temporary=is_temp,
            )
            if new_cat != old_cat:
                mem.category = new_cat
                updated_count += 1
                changes.append({
                    "id": mem.id,
                    "content": mem.content,
                    "old_category": old_cat.value if hasattr(old_cat, "value") else str(old_cat),
                    "new_category": new_cat.value if hasattr(new_cat, "value") else str(new_cat),
                })

        session.flush()
        return {
            "total_records": len(memories),
            "updated_records": updated_count,
            "changes": changes,
        }

    if db is not None:
        return _execute(db)
    with get_db_session() as session:
        return _execute(session)


def record_temporary_override(
    active_memory_id: int,
    temp_memory_data: dict,
    similarity_score: float,
    reasoning: str,
    db: Optional[Session] = None,
) -> Tuple[Memory, MemoryConflictLog]:
    """Record a temporary override memory without altering the active status of the original memory.
    
    1. Keeps original LTM/Semantic memory ACTIVE.
    2. Creates new memory as SHORT_TERM and TEMPORARY.
    3. Logs conflict resolution as TEMPORARY_OVERRIDE.
    4. Indexes new temporary memory vector in FAISS.
    
    Returns:
        Tuple of (new_temp_memory, conflict_log).
    """
    def _execute(session: Session) -> Tuple[Memory, MemoryConflictLog]:
        active_mem = session.get(Memory, active_memory_id)
        if not active_mem:
            raise ValueError(f"Memory with id={active_memory_id} not found.")

        now = utc_now()
        new_temp_mem = Memory(
            user_id=temp_memory_data.get("user_id", active_mem.user_id),
            session_id=temp_memory_data.get("session_id", active_mem.session_id),
            content=temp_memory_data["content"],
            subject=temp_memory_data.get("subject", active_mem.subject),
            predicate=temp_memory_data.get("predicate", active_mem.predicate),
            object_val=temp_memory_data.get("object_val", active_mem.object_val),
            category=MemoryCategory.SHORT_TERM,
            status=MemoryStatus.TEMPORARY,
            importance=temp_memory_data.get("importance", 0.5),
            confidence=temp_memory_data.get("confidence", 1.0),
            source_turn_id=temp_memory_data.get("source_turn_id", None),
            supersedes_id=None,
            created_at=now,
            last_accessed_at=now,
            valid_from=now,
        )
        session.add(new_temp_mem)
        session.flush()

        conflict_log = MemoryConflictLog(
            user_id=new_temp_mem.user_id,
            old_memory_id=active_mem.id,
            new_memory_id=new_temp_mem.id,
            similarity_score=similarity_score,
            resolution_type=ConflictResolutionType.TEMPORARY_OVERRIDE,
            reasoning=reasoning,
            detected_at=now,
        )
        session.add(conflict_log)
        session.flush()

        v_manager = VectorStoreManager.get_instance()
        v_manager.add_memory_vector(new_temp_mem.id, new_temp_mem.content)

        return new_temp_mem, conflict_log

    if db is not None:
        return _execute(db)
    with get_db_session() as session:
        return _execute(session)


def reinforce_memory(
    memory_id: int,
    confidence_boost: float = 0.05,
    db: Optional[Session] = None,
) -> Memory:
    """Reinforce an existing memory by incrementing access count, updating access time, and boosting confidence (capped at 1.0)."""
    def _execute(session: Session) -> Memory:
        mem = session.get(Memory, memory_id)
        if not mem:
            raise ValueError(f"Memory with id={memory_id} not found.")

        mem.access_count += 1
        mem.last_accessed_at = utc_now()
        mem.confidence = min(1.0, mem.confidence + confidence_boost)
        session.flush()
        return mem

    if db is not None:
        return _execute(db)
    with get_db_session() as session:
        return _execute(session)


def get_memory_lineage(
    memory_id: int,
    db: Optional[Session] = None,
) -> List[Memory]:
    """Retrieve the full chronological version history of a memory fact.
    
    Traverses backward (supersedes_id) to the root, then forward (superseded_by_id)
    to return records from oldest HISTORICAL to current ACTIVE.
    """
    def _execute(session: Session) -> List[Memory]:
        target = session.get(Memory, memory_id)
        if not target:
            return []

        # Traverse backward to find oldest ancestor
        curr = target
        visited_back = {curr.id}
        while curr.supersedes_id is not None:
            parent = session.get(Memory, curr.supersedes_id)
            if not parent or parent.id in visited_back:
                break
            visited_back.add(parent.id)
            curr = parent

        root = curr

        # Traverse forward to assemble chronological lineage
        lineage: List[Memory] = [root]
        visited_forward = {root.id}
        curr = root
        while curr.superseded_by_id is not None:
            child = session.get(Memory, curr.superseded_by_id)
            if not child or child.id in visited_forward:
                break
            visited_forward.add(child.id)
            lineage.append(child)
            curr = child

        # Also discover any predecessor memories pointing to any node in this lineage
        lineage_ids = [m.id for m in lineage]
        additional_preds = (
            session.query(Memory)
            .filter(Memory.superseded_by_id.in_(lineage_ids))
            .all()
        )
        for pred in additional_preds:
            if pred.id not in visited_forward and pred.id not in visited_back:
                lineage.append(pred)
                visited_back.add(pred.id)

        # Sort chronologically: historical predecessors first, then active
        lineage.sort(key=lambda m: (m.status == MemoryStatus.ACTIVE, m.id))
        return lineage

    if db is not None:
        return _execute(db)
    with get_db_session() as session:
        return _execute(session)
