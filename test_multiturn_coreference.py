"""Multi-Turn Context, Coreference/Pronoun Resolution, and General Knowledge Test Script.

Simulates:
1. Turn 1: 'hi' -> Casual greeting, no memories created, friendly assistant greeting.
2. Turn 2: 'we are going to build a rag chatbot engine' -> Stored as ACTIVE (subject='project_goal', is_temporary=False).
3. Turn 3: 'we are gonna use sentence transformer for vector embeddings' -> Stored as ACTIVE (subject='embedding_model').
4. Turn 4: 'no we wont use it' -> Coreference resolves 'it' to sentence transformer (subject='embedding_model'),
           triggers PERMANENT_UPDATE, marks Turn 3 memory as HISTORICAL, responds naturally asking for alternatives.
5. Turn 5: 'how many countries are there in the world' -> General knowledge question answered directly without apologizing or mentioning project memories.
6. Turn 6: 'What is our primary database and what did we use before?' -> Uses adaptive memory to state PostgreSQL (active) and MongoDB (historical).
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
    MemoryConflictLog,
    ConflictResolutionType,
    ConversationTurn,
    engine,
    init_db,
    get_db_session,
    create_memory,
    supersede_memory,
    VectorStoreManager,
)
from app.services import (
    LLMService,
    MemoryIngestionService,
    MemoryRetriever,
)


def run_multiturn_test():
    print("=" * 80)
    print("🚀 RUNNING MULTI-TURN CONTEXT & COREFERENCE RESOLUTION TEST")
    print("=" * 80)

    user_id = "test_user_multiturn"
    session_id = "session_coref_01"

    # Clean test user data safely respecting foreign key constraints
    with get_db_session() as session:
        session.query(ConversationTurn).filter(ConversationTurn.user_id == user_id).delete()
        session.query(MemoryConflictLog).filter(MemoryConflictLog.user_id == user_id).delete()
        session.query(Memory).filter(Memory.user_id == user_id).update(
            {"superseded_by_id": None, "supersedes_id": None}, synchronize_session=False
        )
        session.query(Memory).filter(Memory.user_id == user_id).delete()
        session.commit()

    llm_service = LLMService()
    ingestion_service = MemoryIngestionService(llm_service=llm_service)
    retriever = MemoryRetriever()

    chat_history = []

    def execute_turn(user_msg: str, step_name: str) -> dict:
        print(f"\n--- [{step_name}] User: \"{user_msg}\" ---")

        # Ingestion
        ingest_res = ingestion_service.ingest_message(
            user_id=user_id,
            session_id=session_id,
            user_message=user_msg,
            recent_turns=chat_history,
        )

        # Retrieval
        adaptive_res = retriever.retrieve_adaptive(
            user_id=user_id,
            query=user_msg,
            top_k=5,
        )

        # Response Generation
        response = llm_service.generate_response(
            query=user_msg,
            retrieved_memories=adaptive_res["results"],
            conflict_history=ingest_res["conflicts_detected"],
            ingestion_summary=ingest_res,
            recent_turns=chat_history,
        )

        chat_history.append({"role": "user", "content": user_msg})
        chat_history.append({"role": "assistant", "content": response})

        # Save turn to DB
        with get_db_session() as session:
            turn = ConversationTurn(
                user_id=user_id,
                session_id=session_id,
                user_message=user_msg,
                assistant_response=response,
            )
            session.add(turn)

        print(f"  Extracted Facts ({len(ingest_res['extracted_facts'])}):")
        for f in ingest_res["extracted_facts"]:
            print(f"    - [{f['subject']}] {f['predicate']} '{f['object_val']}' | content: '{f['content']}' (temp={f['is_temporary']})")

        print(f"  Conflicts ({len(ingest_res['conflicts_detected'])}):")
        for c in ingest_res["conflicts_detected"]:
            print(f"    - Type={c['resolution_type']} | Reasoning: {c['reasoning']}")

        print(f"  Assistant: \"{response}\"")

        return {
            "ingest": ingest_res,
            "retrieval": adaptive_res,
            "response": response,
        }

    # -------------------------------------------------------------
    # TURN 1: GREETING
    # -------------------------------------------------------------
    t1 = execute_turn("hi", "Turn 1: Greeting")
    assert len(t1["ingest"]["extracted_facts"]) == 0, "No facts should be extracted from casual greeting 'hi'"
    assert len(t1["response"]) > 0

    # -------------------------------------------------------------
    # TURN 2: PROJECT GOAL
    # -------------------------------------------------------------
    t2 = execute_turn("we are going to build a rag chatbot engine", "Turn 2: Project Goal")
    assert len(t2["ingest"]["extracted_facts"]) >= 1, "Should extract project_goal fact"
    fact_goal = t2["ingest"]["extracted_facts"][0]
    assert fact_goal["subject"] == "project_goal", f"Subject should be 'project_goal', got '{fact_goal['subject']}'"
    assert fact_goal["is_temporary"] is False, "Project goal must NOT be temporary"

    with get_db_session() as session:
        m_goal = session.query(Memory).filter(Memory.user_id == user_id, Memory.subject == "project_goal").first()
        assert m_goal is not None
        assert m_goal.status == MemoryStatus.ACTIVE
        print("  ✅ Verified: project_goal stored as ACTIVE.")

    # -------------------------------------------------------------
    # TURN 3: EMBEDDING MODEL CHOICE
    # -------------------------------------------------------------
    t3 = execute_turn("we are gonna use sentence transformer for vector embeddings", "Turn 3: Embedding Model")
    assert len(t3["ingest"]["extracted_facts"]) >= 1, "Should extract embedding_model fact"
    fact_emb = t3["ingest"]["extracted_facts"][0]
    assert fact_emb["subject"] == "embedding_model", f"Subject should be 'embedding_model', got '{fact_emb['subject']}'"
    assert fact_emb["is_temporary"] is False, "Embedding model choice must NOT be temporary"

    with get_db_session() as session:
        m_emb = session.query(Memory).filter(Memory.user_id == user_id, Memory.subject == "embedding_model").first()
        assert m_emb is not None
        assert m_emb.status == MemoryStatus.ACTIVE
        emb_id = m_emb.id
        print(f"  ✅ Verified: embedding_model stored as ACTIVE (ID={emb_id}).")

    # -------------------------------------------------------------
    # TURN 4: COREFERENCE NEGATION ("no we wont use it")
    # -------------------------------------------------------------
    t4 = execute_turn("no we wont use it", "Turn 4: Coreference Negation")
    assert len(t4["ingest"]["extracted_facts"]) >= 1, "Should extract resolved retraction"
    fact_retract = t4["ingest"]["extracted_facts"][0]
    assert fact_retract["subject"] == "embedding_model", (
        f"Subject should resolve 'it' to 'embedding_model', got '{fact_retract['subject']}'"
    )
    assert "sentence transformer" in fact_retract["content"].lower() or "not" in fact_retract["content"].lower()

    # Check conflict resolution
    assert len(t4["ingest"]["conflicts_detected"]) >= 1, "Must detect conflict with previous embedding memory"
    conf = t4["ingest"]["conflicts_detected"][0]
    assert conf["resolution_type"] == ConflictResolutionType.PERMANENT_UPDATE.value, (
        f"Expected PERMANENT_UPDATE, got {conf['resolution_type']}"
    )

    with get_db_session() as session:
        old_emb = session.get(Memory, emb_id)
        assert old_emb.status == MemoryStatus.HISTORICAL, f"Turn 3 memory should now be HISTORICAL, got {old_emb.status}"
        assert old_emb.valid_until is not None

        new_emb = session.get(Memory, old_emb.superseded_by_id)
        assert new_emb.status == MemoryStatus.ACTIVE
        print(f"  ✅ Verified: Memory #{emb_id} transitioned to HISTORICAL, superseded by #{new_emb.id} (ACTIVE).")

    # -------------------------------------------------------------
    # TURN 5: GENERAL KNOWLEDGE (Countries in the world)
    # -------------------------------------------------------------
    t5 = execute_turn("how many countries are there in the world", "Turn 5: General Trivia")
    resp_t5 = t5["response"]
    assert "195" in resp_t5 or "193" in resp_t5 or "countries" in resp_t5.lower(), (
        f"Expected direct answer about countries, got: {resp_t5}"
    )
    assert "provided memories do not contain" not in resp_t5.lower(), (
        "Must NOT apologize about memories for general questions!"
    )
    print("  ✅ Verified: General trivia answered directly without memory apologies.")

    # -------------------------------------------------------------
    # TURN 6: SEED DATABASE MIGRATION & TEST HISTORICAL QUERY
    # -------------------------------------------------------------
    # Seed MongoDB and PostgreSQL migration memories for this user
    m1 = create_memory(
        user_id=user_id,
        content="Project uses MongoDB as primary database",
        subject="primary_database",
        predicate="uses",
        object_val="MongoDB",
        category=MemoryCategory.LONG_TERM,
        status=MemoryStatus.ACTIVE,
    )
    m2, _ = supersede_memory(
        old_memory_id=m1.id,
        new_memory_data={
            "user_id": user_id,
            "content": "Project uses PostgreSQL as primary database",
            "subject": "primary_database",
            "predicate": "uses",
            "object_val": "PostgreSQL",
            "category": MemoryCategory.LONG_TERM,
        },
        similarity_score=0.85,
        reasoning="Migrated to PostgreSQL for strict transactions.",
    )

    t6 = execute_turn(
        "What is our primary database and what did we use before?",
        "Turn 6: Active + Historical Lineage Query",
    )
    resp_t6 = t6["response"]
    assert "postgresql" in resp_t6.lower(), "Response must mention active database PostgreSQL"
    assert "mongodb" in resp_t6.lower(), "Response must mention historical predecessor MongoDB"
    print("  ✅ Verified: Answer grounded in active database and historical lineage.")

    print("\n" + "=" * 80)
    print("🎉 ALL MULTI-TURN COREFERENCE & GENERAL KNOWLEDGE TESTS PASSED!")
    print("=" * 80)


if __name__ == "__main__":
    run_multiturn_test()
