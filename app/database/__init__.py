"""Database package exposing models, engine, and CRUD helpers."""

from app.database.models import (
    Base,
    Memory,
    MemoryConflictLog,
    ConversationTurn,
    MemoryCategory,
    MemoryStatus,
    ConflictResolutionType,
    utc_now,
)
from app.database.vector_store import VectorStoreManager
from app.database.db import (
    engine,
    SessionLocal,
    init_db,
    get_db_session,
    create_memory,
    supersede_memory,
    record_temporary_override,
    reinforce_memory,
    get_memory_lineage,
    confirm_and_archive_conflicts,
    repair_legacy_conflicts,
    backfill_memory_categories,
)

__all__ = [
    "Base",
    "Memory",
    "MemoryConflictLog",
    "ConversationTurn",
    "MemoryCategory",
    "MemoryStatus",
    "ConflictResolutionType",
    "utc_now",
    "VectorStoreManager",
    "engine",
    "SessionLocal",
    "init_db",
    "get_db_session",
    "create_memory",
    "supersede_memory",
    "record_temporary_override",
    "reinforce_memory",
    "get_memory_lineage",
    "confirm_and_archive_conflicts",
    "repair_legacy_conflicts",
    "backfill_memory_categories",
]
