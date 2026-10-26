from datetime import datetime, timezone
from enum import Enum
from typing import Optional, List

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    Enum as SQLEnum,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now() -> datetime:
    """Return timezone-aware current UTC datetime."""
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class MemoryCategory(str, Enum):
    SHORT_TERM = "SHORT_TERM"
    LONG_TERM = "LONG_TERM"
    EPISODIC = "EPISODIC"
    SEMANTIC = "SEMANTIC"


class MemoryStatus(str, Enum):
    ACTIVE = "ACTIVE"
    HISTORICAL = "HISTORICAL"
    TEMPORARY = "TEMPORARY"
    ARCHIVED = "ARCHIVED"


class ConflictResolutionType(str, Enum):
    PERMANENT_UPDATE = "PERMANENT_UPDATE"
    TEMPORARY_OVERRIDE = "TEMPORARY_OVERRIDE"
    REINFORCEMENT = "REINFORCEMENT"
    COMPLEMENTARY = "COMPLEMENTARY"


class Memory(Base):
    __tablename__ = "memories"

    # Primary key used as FAISS 64-bit vector ID
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # Identifiers
    user_id: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    session_id: Mapped[Optional[str]] = mapped_column(String(255), index=True, nullable=True)

    # Memory content
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # Structured triple columns for conflict matching
    subject: Mapped[Optional[str]] = mapped_column(String(255), index=True, nullable=True)
    predicate: Mapped[Optional[str]] = mapped_column(String(255), index=True, nullable=True)
    object_val: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)

    # Classification & Lifecycle
    category: Mapped[MemoryCategory] = mapped_column(
        SQLEnum(MemoryCategory, native_enum=False),
        nullable=False,
        default=MemoryCategory.LONG_TERM,
    )
    status: Mapped[MemoryStatus] = mapped_column(
        SQLEnum(MemoryStatus, native_enum=False),
        nullable=False,
        default=MemoryStatus.ACTIVE,
    )

    # Scoring metadata
    importance: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    access_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Timestamps (Timezone-aware UTC)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    last_accessed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    valid_from: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    valid_until: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Lineage pointers
    supersedes_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("memories.id"), nullable=True
    )
    superseded_by_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("memories.id"), nullable=True
    )

    # Optional origin turn
    source_turn_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("conversation_turns.id"), nullable=True
    )

    # Relationships
    supersedes: Mapped[Optional["Memory"]] = relationship(
        "Memory",
        foreign_keys=[supersedes_id],
        remote_side=[id],
        post_update=True,
    )
    superseded_by: Mapped[Optional["Memory"]] = relationship(
        "Memory",
        foreign_keys=[superseded_by_id],
        remote_side=[id],
        post_update=True,
    )

    __table_args__ = (
        Index("idx_memories_user_status_category", "user_id", "status", "category"),
    )

    def __repr__(self) -> str:
        return (
            f"<Memory(id={self.id}, user_id='{self.user_id}', "
            f"category={self.category.value if isinstance(self.category, MemoryCategory) else self.category}, "
            f"status={self.status.value if isinstance(self.status, MemoryStatus) else self.status}, "
            f"content='{self.content[:30]}...')>"
        )


class MemoryConflictLog(Base):
    __tablename__ = "memory_conflict_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    old_memory_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("memories.id"), nullable=True
    )
    new_memory_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("memories.id"), nullable=True
    )
    similarity_score: Mapped[float] = mapped_column(Float, nullable=False)
    resolution_type: Mapped[ConflictResolutionType] = mapped_column(
        SQLEnum(ConflictResolutionType, native_enum=False), nullable=False
    )
    reasoning: Mapped[str] = mapped_column(Text, nullable=False)
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    # Relationships
    old_memory: Mapped[Optional["Memory"]] = relationship(
        "Memory", foreign_keys=[old_memory_id]
    )
    new_memory: Mapped[Optional["Memory"]] = relationship(
        "Memory", foreign_keys=[new_memory_id]
    )

    def __repr__(self) -> str:
        return (
            f"<MemoryConflictLog(id={self.id}, old_id={self.old_memory_id}, "
            f"new_id={self.new_memory_id}, res='{self.resolution_type}', "
            f"sim={self.similarity_score:.3f})>"
        )


class ConversationTurn(Base):
    __tablename__ = "conversation_turns"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    session_id: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    user_message: Mapped[str] = mapped_column(Text, nullable=False)
    assistant_response: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    def __repr__(self) -> str:
        return (
            f"<ConversationTurn(id={self.id}, user_id='{self.user_id}', "
            f"session_id='{self.session_id}')>"
        )
