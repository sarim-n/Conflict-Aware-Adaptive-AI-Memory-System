"""Phase 2 & 3 Verification Script:
LLM Fact Extraction, Conflict Resolution Engine, and Adaptive Composite Retrieval.

Verifies:
1. Turn 1 (Ingestion): Ingests user identity, MongoDB as primary database, and FastAPI.
2. Turn 2 (Permanent Conflict): Ingests migration to PostgreSQL, superseding MongoDB (PERMANENT_UPDATE).
3. Turn 3 (Temporary Override): Ingests ephemeral SQLite testing override (TEMPORARY_OVERRIDE).
4. Turn 4 (Retrieval Comparison): Compares Basic Vector Search vs Adaptive Composite Retrieval on 'What is our primary database?'.
   Asserts Adaptive ranks PostgreSQL #1 above MongoDB with lineage attached.
5. Turn 5 (Historical Query): Queries 'What database did we use previously before migrating?' and verifies historical intent surfaces MongoDB.
6. Turn 6 (Context-Aware Answer Generation): Calls LLMService.generate_response grounded in adaptive memories.
"""

import os
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
from app.database import (
    Base,
    Memory,
    MemoryStatus,
    MemoryCategory,
    ConflictResolutionType,
    engine,
    init_db,
    get_db_session,
    VectorStoreManager,
)
from app.services import (
    LLMService,
    MemoryIngestionService,
    MemoryRetriever,
)


def run_phase2_tests():
    print("=" * 75)
    print("🚀 STARTING PHASE 2 & 3 VERIFICATION TESTS")
    print("=" * 75)

    # 1. Clean environment
    if os.path.exists(settings.FAISS_INDEX_PATH):
        try:
            os.remove(settings.FAISS_INDEX_PATH)
            print(f"🧹 Cleaned existing FAISS index at {settings.FAISS_INDEX_PATH}")
        except Exception as e:
            print(f"⚠️ Could not delete index: {e}")

    db_file = Path(settings.DATABASE_URL.replace("sqlite:///", ""))
    engine.dispose()
    if db_file.exists():
        try:
            db_file.unlink()
            print(f"🧹 Cleaned existing SQLite DB at {db_file}")
        except Exception:
            Base.metadata.drop_all(bind=engine)
            with engine.connect() as conn:
                try:
                    conn.execute(text("DELETE FROM sqlite_sequence;"))
                    conn.commit()
                except Exception:
                    pass
            print(f"🧹 Reset and truncated tables in {db_file}")

    # Reset vector store and initialize database
    VectorStoreManager.reset_instance()
    init_db()
    print("✅ Fresh SQLite database and vector index initialized.")

    llm_service = LLMService()
    ingestion_service = MemoryIngestionService(llm_service=llm_service)
    retriever = MemoryRetriever()

    user_id = "user_aarav_001"
    session_id = "session_phase2_test"

    # -------------------------------------------------------------
    # 2. TURN 1: INITIAL INGESTION
    # -------------------------------------------------------------
    print("\n" + "-" * 60)
    print("[Turn 1] Ingesting initial architectural statement:")
    t1_msg = "Hi, I'm Aarav, a backend architect. Our project uses MongoDB as our primary database and FastAPI for the backend."
    print(f"User: \"{t1_msg}\"")

    res1 = ingestion_service.ingest_message(user_id=user_id, session_id=session_id, user_message=t1_msg)
    print(f"  Actions taken: {res1['actions_taken']}")
    print(f"  Memories created ({len(res1['memories_created'])}):")
    for m in res1["memories_created"]:
        print(f"    - ID={m['id']}, Subject='{m['subject']}', Status={m['status']}, Content='{m['content']}'")

    # Assertions for Turn 1
    with get_db_session() as session:
        db_memories = session.query(Memory).filter(Memory.user_id == user_id).all()
        assert len(db_memories) >= 2, f"Expected at least 2 memories, found {len(db_memories)}"
        mongo_mem = next((m for m in db_memories if "MongoDB" in m.content), None)
        assert mongo_mem is not None, "MongoDB memory was not created"
        assert mongo_mem.status == MemoryStatus.ACTIVE, f"Expected MongoDB ACTIVE, got {mongo_mem.status}"
        mongo_id = mongo_mem.id
        print(f"✅ Turn 1 Verified: MongoDB primary database stored as ACTIVE (ID={mongo_id}).")

    # -------------------------------------------------------------
    # 3. TURN 2: PERMANENT CONFLICT RESOLUTION (MIGRATION)
    # -------------------------------------------------------------
    print("\n" + "-" * 60)
    print("[Turn 2] Ingesting technological migration (Permanent Update):")
    t2_msg = "Update on the architecture: we officially migrated our primary database from MongoDB to PostgreSQL because we needed strict ACID transactions."
    print(f"User: \"{t2_msg}\"")

    res2 = ingestion_service.ingest_message(user_id=user_id, session_id=session_id, user_message=t2_msg)
    print(f"  Actions taken: {res2['actions_taken']}")
    print(f"  Conflicts detected ({len(res2['conflicts_detected'])}):")
    for c in res2["conflicts_detected"]:
        print(f"    - Old ID={c['old_memory_id']} -> New ID={c['new_memory_id']}, Res={c['resolution_type']}, Reason='{c['reasoning']}'")

    # Assertions for Turn 2
    with get_db_session() as session:
        old_m = session.get(Memory, mongo_id)
        assert old_m.status == MemoryStatus.HISTORICAL, f"Expected MongoDB HISTORICAL, got {old_m.status}"
        assert old_m.valid_until is not None, "Old memory valid_until must be set"
        postgres_id = old_m.superseded_by_id
        assert postgres_id is not None, "MongoDB superseded_by_id must point to new PostgreSQL memory"

        new_m = session.get(Memory, postgres_id)
        assert new_m.status == MemoryStatus.ACTIVE, f"Expected PostgreSQL ACTIVE, got {new_m.status}"
        assert new_m.supersedes_id == mongo_id, f"Expected PostgreSQL supersedes_id={mongo_id}, got {new_m.supersedes_id}"
        assert "PostgreSQL" in new_m.content, f"New memory content mismatch: {new_m.content}"

    assert len(res2["conflicts_detected"]) >= 1, "Expected at least 1 conflict log"
    conflict_entry = res2["conflicts_detected"][0]
    assert conflict_entry["resolution_type"] == ConflictResolutionType.PERMANENT_UPDATE.value
    print(f"✅ Turn 2 Verified: Conflict detected! MongoDB (ID={mongo_id}) is HISTORICAL, PostgreSQL (ID={postgres_id}) is ACTIVE.")

    # -------------------------------------------------------------
    # 4. TURN 3: TEMPORARY OVERRIDE
    # -------------------------------------------------------------
    print("\n" + "-" * 60)
    print("[Turn 3] Ingesting temporary session override:")
    t3_msg = "Just for running local unit tests today, I am using an in-memory SQLite database."
    print(f"User: \"{t3_msg}\"")

    res3 = ingestion_service.ingest_message(user_id=user_id, session_id=session_id, user_message=t3_msg)
    print(f"  Actions taken: {res3['actions_taken']}")
    print(f"  Conflicts detected ({len(res3['conflicts_detected'])}):")
    for c in res3["conflicts_detected"]:
        print(f"    - Old ID={c['old_memory_id']} -> New ID={c['new_memory_id']}, Res={c['resolution_type']}, Reason='{c['reasoning']}'")

    # Assertions for Turn 3
    with get_db_session() as session:
        temp_mems = session.query(Memory).filter(
            Memory.user_id == user_id,
            Memory.status == MemoryStatus.TEMPORARY
        ).all()
        assert len(temp_mems) >= 1, "Expected temporary memory created"
        temp_m = temp_mems[0]
        assert temp_m.category == MemoryCategory.SHORT_TERM, f"Expected SHORT_TERM, got {temp_m.category}"
        assert "SQLite" in temp_m.content, f"Content mismatch: {temp_m.content}"

        pg_m = session.get(Memory, postgres_id)
        assert pg_m.status == MemoryStatus.ACTIVE, f"PostgreSQL must remain ACTIVE, got {pg_m.status}"

    assert len(res3["conflicts_detected"]) >= 1
    assert res3["conflicts_detected"][0]["resolution_type"] == ConflictResolutionType.TEMPORARY_OVERRIDE.value
    print("✅ Turn 3 Verified: Temporary override logged. SQLite is TEMPORARY/SHORT_TERM, PostgreSQL remains ACTIVE.")

    # -------------------------------------------------------------
    # 5. TURN 4: RETRIEVAL COMPARISON (BASIC VS ADAPTIVE)
    # -------------------------------------------------------------
    print("\n" + "-" * 60)
    print("[Turn 4] Retrieval Comparison on present query:")
    query_present = "What is our primary database?"
    print(f"Query: \"{query_present}\"")

    basic_results = retriever.retrieve_basic(user_id=user_id, query=query_present, top_k=5)
    adaptive_results = retriever.retrieve_adaptive(user_id=user_id, query=query_present, top_k=5)

    print("\n  [A] Pure FAISS Baseline Retrieval:")
    for idx, r in enumerate(basic_results["results"], 1):
        print(f"      {idx}. [ID={r['id']}] Status={r['status']}, Sim={r['similarity']:.4f} -> '{r['content']}'")

    print("\n  [B] Adaptive Composite Retrieval (Status-Gated + Recency + Lineage):")
    for idx, r in enumerate(adaptive_results["results"], 1):
        lineage_preview = f" (Lineage: {[x['content'] for x in r['lineage_chain']]})" if r['lineage_chain'] else ""
        print(
            f"      {idx}. [ID={r['id']}] Status={r['status']}, BaseScore={r['base_score']:.4f}, "
            f"Psi={r['status_multiplier']}, AdaptiveScore={r['adaptive_score']:.4f} -> '{r['content']}'{lineage_preview}"
        )

    assert len(adaptive_results["results"]) > 0, "Adaptive retrieval returned no results"
    top_adaptive = adaptive_results["results"][0]
    assert top_adaptive["id"] == postgres_id, (
        f"Expected PostgreSQL (ID={postgres_id}) ranked #1 in adaptive retrieval, got ID={top_adaptive['id']} ({top_adaptive['content']})"
    )
    assert top_adaptive["status"] == "ACTIVE", "Rank #1 memory must be ACTIVE"
    assert len(top_adaptive["lineage_chain"]) > 0, "Rank #1 PostgreSQL memory must include superseded lineage"
    assert top_adaptive["lineage_chain"][0]["id"] == mongo_id, "Superseded lineage must reference MongoDB"
    print(f"✅ Turn 4 Verified: Adaptive Retrieval correctly ranked ACTIVE PostgreSQL (ID={postgres_id}) #1 with lineage chain!")

    # -------------------------------------------------------------
    # 6. TURN 5: HISTORICAL QUERY RETRIEVAL
    # -------------------------------------------------------------
    print("\n" + "-" * 60)
    print("[Turn 5] Testing Historical Intent Query:")
    query_history = "What database did we use previously before migrating?"
    print(f"Query: \"{query_history}\"")

    hist_adaptive_results = retriever.retrieve_adaptive(user_id=user_id, query=query_history, top_k=5)
    print(f"  Historical Intent Detected: {hist_adaptive_results['historical_intent']}")
    assert hist_adaptive_results["historical_intent"] is True, "Expected historical intent to be detected"

    print("  Adaptive Retrieval Results for Historical Query:")
    for idx, r in enumerate(hist_adaptive_results["results"], 1):
        print(
            f"      {idx}. [ID={r['id']}] Status={r['status']}, Psi={r['status_multiplier']}, "
            f"AdaptiveScore={r['adaptive_score']:.4f} -> '{r['content']}'"
        )

    mongo_hist_entry = next((r for r in hist_adaptive_results["results"] if r["id"] == mongo_id), None)
    assert mongo_hist_entry is not None, "Historical MongoDB record should be returned for historical query"
    assert mongo_hist_entry["status_multiplier"] == 1.0, f"Expected Psi=1.0 for historical query, got {mongo_hist_entry['status_multiplier']}"
    print(f"✅ Turn 5 Verified: Historical query detected! MongoDB (ID={mongo_id}) surfaced with full weight (Psi=1.0).")

    # -------------------------------------------------------------
    # 7. TURN 6: CONTEXT-AWARE ANSWER GENERATION
    # -------------------------------------------------------------
    print("\n" + "-" * 60)
    print("[Turn 6] Context-Aware Answer Generation:")
    answer = llm_service.generate_response(
        query=query_present,
        retrieved_memories=adaptive_results["results"],
        conflict_history=res2["conflicts_detected"],
    )
    print(f"\nAssistant Response:\n\"{answer}\"\n")
    assert "PostgreSQL" in answer, "Response should mention current database PostgreSQL"
    print("✅ Turn 6 Verified: Generated response is context-aware and grounded in active memories.")

    print("\n" + "=" * 75)
    print("🎉 ALL PHASE 2 & 3 VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    run_phase2_tests()
