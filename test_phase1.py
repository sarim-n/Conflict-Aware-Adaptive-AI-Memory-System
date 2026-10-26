"""Phase 1 Verification Script for Conflict-Aware Adaptive AI Memory System.

Verifies:
1. Database initialization and clean table creation.
2. Initial memory creation (id=1, ACTIVE) and synchronized FAISS indexing.
3. Memory supersede operation (id=2 ACTIVE, id=1 HISTORICAL with valid_until, bidirectional linking, conflict log).
4. Temporary memory override (id=3 SHORT_TERM TEMPORARY, id=2 remains ACTIVE, conflict log).
5. FAISS semantic similarity search returning relevant memories.
6. Chronological lineage retrieval via get_memory_lineage(2) returning [id=1, id=2].
7. Memory reinforcement (access_count incremented, confidence boosted).
8. Vector reconstruction & index integrity.
"""

import os
import shutil
import sys
from pathlib import Path

# Ensure root dir in python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from sqlalchemy import text
from app.config import settings
from app.database.models import (
    Base,
    Memory,
    MemoryConflictLog,
    MemoryCategory,
    MemoryStatus,
    ConflictResolutionType,
)
from app.database.vector_store import VectorStoreManager
from app.database.db import (
    engine,
    init_db,
    get_db_session,
    create_memory,
    supersede_memory,
    record_temporary_override,
    reinforce_memory,
    get_memory_lineage,
)


def run_phase1_tests():
    print("=" * 70)
    print("🚀 STARTING PHASE 1 VERIFICATION TESTS")
    print("=" * 70)

    # Clean existing data for deterministic test run
    if os.path.exists(settings.FAISS_INDEX_PATH):
        try:
            os.remove(settings.FAISS_INDEX_PATH)
            print(f"🧹 Cleaned existing FAISS index at {settings.FAISS_INDEX_PATH}")
        except Exception as e:
            print(f"⚠️ Could not delete index: {e}")

    # Reset vector store singleton
    VectorStoreManager.reset_instance()

    # Remove SQLite db if exists
    db_file = Path(settings.DATABASE_URL.replace("sqlite:///", ""))
    engine.dispose()
    if db_file.exists():
        try:
            db_file.unlink()
            print(f"🧹 Cleaned existing SQLite DB at {db_file}")
        except Exception:
            # File is held by another running process (e.g. dev server); drop and recreate tables
            Base.metadata.drop_all(bind=engine)
            with engine.connect() as conn:
                try:
                    conn.execute(text("DELETE FROM sqlite_sequence;"))
                    conn.commit()
                except Exception:
                    pass
            print(f"🧹 Reset and truncated tables in {db_file}")

    # -------------------------------------------------------------
    # 1. DATABASE INITIALIZATION
    # -------------------------------------------------------------
    print("\n[Step 1] Initializing SQLite database...")
    init_db()
    assert db_file.exists(), f"Database file was not created at {db_file}"
    print(f"✅ SQLite database created at {db_file}")

    # -------------------------------------------------------------
    # 2. CREATE INITIAL MEMORY (id=1)
    # -------------------------------------------------------------
    print("\n[Step 2] Creating initial memory: 'Project uses MongoDB as primary database'...")
    user_id = "user_dev_001"
    session_id = "session_arch_42"

    mem1 = create_memory(
        user_id=user_id,
        session_id=session_id,
        content="Project uses MongoDB as primary database",
        subject="primary_database",
        predicate="uses",
        object_val="MongoDB",
        category=MemoryCategory.LONG_TERM,
        status=MemoryStatus.ACTIVE,
        importance=0.85,
        confidence=1.0,
    )

    print(f"  Created Memory: ID={mem1.id}, Status={mem1.status.value}, Content='{mem1.content}'")
    assert mem1.id == 1, f"Expected id=1, got {mem1.id}"
    assert mem1.status == MemoryStatus.ACTIVE, f"Expected ACTIVE, got {mem1.status}"
    assert mem1.valid_until is None, "valid_until should be None for active memory"

    # Verify FAISS indexing for id=1
    v_manager = VectorStoreManager.get_instance()
    assert v_manager.index.ntotal == 1, f"FAISS expected 1 vector, got {v_manager.index.ntotal}"
    vec1 = v_manager.get_vector_by_id(1)
    assert vec1 is not None, "Failed to reconstruct vector for id=1 from FAISS"
    assert len(vec1) == settings.EMBEDDING_DIM, f"Vector dimension mismatch: {len(vec1)} vs {settings.EMBEDDING_DIM}"
    print(f"✅ Memory id=1 stored in SQLite and indexed in FAISS (dim={len(vec1)})")

    # -------------------------------------------------------------
    # 3. SUPERSEDE MEMORY (id=1 -> id=2)
    # -------------------------------------------------------------
    print("\n[Step 3] Calling supersede_memory with: 'Project migrated from MongoDB to PostgreSQL'...")
    new_mem_data = {
        "content": "Project migrated from MongoDB to PostgreSQL",
        "subject": "primary_database",
        "predicate": "uses",
        "object_val": "PostgreSQL",
        "category": MemoryCategory.LONG_TERM,
        "importance": 0.90,
        "confidence": 1.0,
    }
    similarity_score = 0.88
    reasoning = "Technological migration updated active database choice from MongoDB to PostgreSQL."

    mem2, conflict_log = supersede_memory(
        old_memory_id=1,
        new_memory_data=new_mem_data,
        similarity_score=similarity_score,
        reasoning=reasoning,
    )

    print(f"  Created New Memory: ID={mem2.id}, Status={mem2.status.value}, Supersedes={mem2.supersedes_id}")
    print(f"  Conflict Log: Old={conflict_log.old_memory_id}, New={conflict_log.new_memory_id}, Type={conflict_log.resolution_type.value}")

    assert mem2.id == 2, f"Expected id=2, got {mem2.id}"
    assert mem2.status == MemoryStatus.ACTIVE, f"Expected id=2 ACTIVE, got {mem2.status}"
    assert mem2.supersedes_id == 1, f"Expected id=2 supersedes_id=1, got {mem2.supersedes_id}"

    # Verify old memory updated in database
    with get_db_session() as session:
        old_mem_refreshed = session.get(Memory, 1)
        assert old_mem_refreshed.status == MemoryStatus.HISTORICAL, (
            f"Expected id=1 to be HISTORICAL, got {old_mem_refreshed.status}"
        )
        assert old_mem_refreshed.valid_until is not None, "id=1 valid_until should be populated"
        assert old_mem_refreshed.superseded_by_id == 2, (
            f"Expected id=1 superseded_by_id=2, got {old_mem_refreshed.superseded_by_id}"
        )
        print(f"  Old Memory (id=1): Status={old_mem_refreshed.status.value}, Valid Until={old_mem_refreshed.valid_until}")

    # Verify FAISS contains both vectors (old + new)
    assert v_manager.index.ntotal == 2, f"FAISS should have 2 vectors, got {v_manager.index.ntotal}"
    assert v_manager.get_vector_by_id(1) is not None, "Old vector id=1 should still be present in FAISS"
    assert v_manager.get_vector_by_id(2) is not None, "New vector id=2 should be present in FAISS"
    print("✅ Supersede operation succeeded: id=1 is HISTORICAL, id=2 is ACTIVE, conflict logged.")

    # -------------------------------------------------------------
    # 4. TEMPORARY OVERRIDE (id=3)
    # -------------------------------------------------------------
    print("\n[Step 4] Calling record_temporary_override with: 'Using SQLite in-memory for local unit tests today'...")
    temp_mem_data = {
        "content": "Using SQLite in-memory for local unit tests today",
        "subject": "primary_database",
        "predicate": "temporary_test_db",
        "object_val": "SQLite in-memory",
        "importance": 0.40,
        "confidence": 0.95,
    }
    temp_sim_score = 0.72
    temp_reasoning = "Ephemeral testing override for current test execution session."

    mem3, temp_conflict_log = record_temporary_override(
        active_memory_id=2,
        temp_memory_data=temp_mem_data,
        similarity_score=temp_sim_score,
        reasoning=temp_reasoning,
    )

    print(f"  Created Temp Memory: ID={mem3.id}, Category={mem3.category.value}, Status={mem3.status.value}")
    print(f"  Conflict Log: Old={temp_conflict_log.old_memory_id}, New={temp_conflict_log.new_memory_id}, Type={temp_conflict_log.resolution_type.value}")

    assert mem3.id == 3, f"Expected id=3, got {mem3.id}"
    assert mem3.category == MemoryCategory.SHORT_TERM, f"Expected SHORT_TERM, got {mem3.category}"
    assert mem3.status == MemoryStatus.TEMPORARY, f"Expected TEMPORARY, got {mem3.status}"

    # Verify id=2 is still ACTIVE
    with get_db_session() as session:
        active_mem = session.get(Memory, 2)
        assert active_mem.status == MemoryStatus.ACTIVE, "Memory id=2 should remain ACTIVE"
        print(f"  Active Memory (id=2): Status={active_mem.status.value} (Preserved)")

    assert v_manager.index.ntotal == 3, f"FAISS should have 3 vectors, got {v_manager.index.ntotal}"
    print("✅ Temporary override succeeded: id=3 is SHORT_TERM/TEMPORARY, id=2 remains ACTIVE.")

    # -------------------------------------------------------------
    # 5. SEMANTIC SEARCH & LINEAGE TRAVERSAL
    # -------------------------------------------------------------
    print("\n[Step 5] Querying FAISS for: 'What database does the project use?'...")
    query = "What database does the project use?"
    search_results = v_manager.search_similar(query, top_k=5)

    print(f"  FAISS Search Results for '{query}':")
    returned_ids = [res[0] for res in search_results]
    for mem_id, score in search_results:
        with get_db_session() as session:
            m = session.get(Memory, mem_id)
            print(f"    - ID={mem_id} (Cosine Sim: {score:.4f}, Status: {m.status.value}) -> '{m.content}'")

    assert 1 in returned_ids, "Memory id=1 should be returned in top results"
    assert 2 in returned_ids, "Memory id=2 should be returned in top results"

    print("\n[Step 6] Testing get_memory_lineage(2)...")
    lineage = get_memory_lineage(2)
    print(f"  Lineage for memory id=2:")
    for m in lineage:
        print(f"    - ID={m.id}, Status={m.status.value}, valid_until={m.valid_until}, Content='{m.content}'")

    assert len(lineage) == 2, f"Expected lineage length 2, got {len(lineage)}"
    assert lineage[0].id == 1 and lineage[0].status == MemoryStatus.HISTORICAL, "First lineage item must be id=1 HISTORICAL"
    assert lineage[1].id == 2 and lineage[1].status == MemoryStatus.ACTIVE, "Second lineage item must be id=2 ACTIVE"
    print("✅ Lineage traversal verified: [Memory(id=1, HISTORICAL), Memory(id=2, ACTIVE)]")

    # Also test get_memory_lineage starting from id=1
    lineage_from_1 = get_memory_lineage(1)
    assert [m.id for m in lineage_from_1] == [1, 2], "Lineage from id=1 must also return [id=1, id=2]"
    print("✅ Lineage traversal from root id=1 also verified.")

    # -------------------------------------------------------------
    # 7. REINFORCE MEMORY
    # -------------------------------------------------------------
    print("\n[Step 7] Testing reinforce_memory(2)...")
    reinforced = reinforce_memory(2, confidence_boost=0.05)
    assert reinforced.access_count == 1, f"Expected access_count=1, got {reinforced.access_count}"
    assert reinforced.confidence == 1.0, f"Expected confidence capped at 1.0, got {reinforced.confidence}"
    print(f"✅ Memory id=2 reinforced: access_count={reinforced.access_count}, confidence={reinforced.confidence}")

    # -------------------------------------------------------------
    # 8. REBUILD FAISS FROM DB VERIFICATION
    # -------------------------------------------------------------
    print("\n[Step 8] Testing VectorStoreManager.rebuild_from_db()...")
    with get_db_session() as session:
        all_memories = session.query(Memory).all()
        v_manager.rebuild_from_db(all_memories)

    assert v_manager.index.ntotal == 3, f"Expected 3 vectors after rebuild, got {v_manager.index.ntotal}"
    print(f"✅ FAISS index rebuilt from SQLite cleanly ({v_manager.index.ntotal} vectors).")

    print("\n" + "=" * 70)
    print("🎉 ALL PHASE 1 VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_phase1_tests()
