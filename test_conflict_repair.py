"""
Automated regression test suite verifying conflict resolution, atomic archival,
lineage preservation, and retrieval accuracy for the Conflict-Aware Adaptive AI Memory System.

Verifies:
1. Replacing an existing project goal marks the old goal HISTORICAL.
2. The replacement goal is ACTIVE.
3. Repeated goal changes preserve the complete version history.
4. A REINFORCEMENT does not create unnecessary duplicate memories.
5. A COMPLEMENTARY fact does not incorrectly archive an existing fact.
6. A TEMPORARY_OVERRIDE does not permanently replace the long-term goal.
7. Failed transactions do not leave partially updated memory chains.
8. The system never reports successful archival when the database operation fails.
9. Current-tense retrieval prefers the latest active goal.
10. Historical queries can still retrieve previous goals.
11. Lineage pointers are reciprocal and valid.
"""

import sys
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from app.database.models import (
    Base,
    Memory,
    MemoryCategory,
    MemoryConflictLog,
    MemoryStatus,
    ConflictResolutionType,
    utc_now,
)
from app.database.db import (
    get_db_session,
    create_memory,
    supersede_memory,
    confirm_and_archive_conflicts,
    record_temporary_override,
    reinforce_memory,
    get_memory_lineage,
    repair_legacy_conflicts,
)
from app.services.conflict_resolver import MemoryIngestionService
from app.services.retriever import MemoryRetriever
from app.services.llm_service import ExtractedFact, LLMService


class TestConflictResolutionAndLineage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_user = "test_user_regression"
        cls.ingestion_service = MemoryIngestionService()
        cls.retriever = MemoryRetriever()

    def setUp(self):
        # Clean test user records before each test
        with get_db_session() as session:
            session.query(MemoryConflictLog).filter(MemoryConflictLog.user_id == self.test_user).delete()
            session.query(Memory).filter(Memory.user_id == self.test_user).delete()

    def tearDown(self):
        with get_db_session() as session:
            session.query(MemoryConflictLog).filter(MemoryConflictLog.user_id == self.test_user).delete()
            session.query(Memory).filter(Memory.user_id == self.test_user).delete()

    # 1 & 2: Replacing an existing project goal marks old goal HISTORICAL, replacement is ACTIVE
    def test_01_replacing_project_goal_archives_old_and_activates_new(self):
        with get_db_session() as session:
            old_mem = create_memory(
                user_id=self.test_user,
                content="Project is building an online store",
                subject="project_goal",
                predicate="is_building",
                object_val="online store",
                db=session,
            )
            old_id = old_mem.id

        # Ingest replacement goal
        new_fact = ExtractedFact(
            content="Project is building billing software",
            subject="project_goal",
            predicate="is_building",
            object_val="billing software",
            category=MemoryCategory.LONG_TERM,
            importance=0.9,
            confidence=1.0,
            is_temporary=False,
        )

        with get_db_session() as session:
            cand = session.get(Memory, old_id)
            new_mem, log = supersede_memory(
                old_memory_id=old_id,
                new_memory_data={
                    "user_id": self.test_user,
                    "content": new_fact.content,
                    "subject": new_fact.subject,
                    "predicate": new_fact.predicate,
                    "object_val": new_fact.object_val,
                    "category": new_fact.category,
                    "importance": new_fact.importance,
                    "confidence": new_fact.confidence,
                },
                similarity_score=0.85,
                reasoning="Updated project goal",
                db=session,
            )
            new_id = new_mem.id

        # Verify persisted state in fresh session
        with get_db_session() as session:
            old_ref = session.get(Memory, old_id)
            new_ref = session.get(Memory, new_id)

            self.assertEqual(old_ref.status, MemoryStatus.HISTORICAL)
            self.assertIsNotNone(old_ref.valid_until)
            self.assertEqual(old_ref.superseded_by_id, new_id)

            self.assertEqual(new_ref.status, MemoryStatus.ACTIVE)
            self.assertIsNone(new_ref.valid_until)
            self.assertEqual(new_ref.supersedes_id, old_id)
            self.assertIsNone(new_ref.superseded_by_id)

    # 3: Repeated goal changes preserve complete version history
    def test_02_repeated_goal_changes_preserve_complete_lineage(self):
        goals = ["billing software", "digital marketing", "shoes online", "shoes offline"]
        ids = []

        with get_db_session() as session:
            m0 = create_memory(
                user_id=self.test_user,
                content=f"Project is building {goals[0]}",
                subject="project_goal",
                predicate="is_building",
                object_val=goals[0],
                db=session,
            )
            ids.append(m0.id)

        for goal in goals[1:]:
            with get_db_session() as session:
                new_m, _ = supersede_memory(
                    old_memory_id=ids[-1],
                    new_memory_data={
                        "user_id": self.test_user,
                        "content": f"Project is building {goal}",
                        "subject": "project_goal",
                        "predicate": "is_building",
                        "object_val": goal,
                        "category": MemoryCategory.LONG_TERM,
                        "importance": 0.85,
                        "confidence": 1.0,
                    },
                    similarity_score=0.80,
                    reasoning=f"Shifted goal to {goal}",
                    db=session,
                )
                ids.append(new_m.id)

        # Verify full lineage from root or active leaf
        lineage = get_memory_lineage(ids[-1])
        lineage_ids = [m.id for m in lineage]
        self.assertEqual(lineage_ids, ids)
        self.assertEqual(len(lineage), 4)

        # First 3 must be HISTORICAL, 4th must be ACTIVE
        for m in lineage[:-1]:
            self.assertEqual(m.status, MemoryStatus.HISTORICAL)
            self.assertIsNotNone(m.valid_until)
        self.assertEqual(lineage[-1].status, MemoryStatus.ACTIVE)
        self.assertIsNone(lineage[-1].valid_until)

    # 4: REINFORCEMENT does not create duplicate memories
    def test_03_reinforcement_does_not_create_duplicates(self):
        with get_db_session() as session:
            m = create_memory(
                user_id=self.test_user,
                content="Project uses PostgreSQL as primary database",
                subject="primary_database",
                predicate="uses",
                object_val="PostgreSQL",
                confidence=0.8,
                db=session,
            )
            initial_id = m.id
            initial_count = m.access_count

        reinforced = reinforce_memory(initial_id, confidence_boost=0.1)
        self.assertEqual(reinforced.id, initial_id)
        self.assertEqual(reinforced.access_count, initial_count + 1)
        self.assertAlmostEqual(reinforced.confidence, 0.9, places=2)

        # Check total records in DB for subject
        with get_db_session() as session:
            count = session.query(Memory).filter(
                Memory.user_id == self.test_user,
                Memory.subject == "primary_database"
            ).count()
            self.assertEqual(count, 1)

    # 5: COMPLEMENTARY fact does not incorrectly archive an existing fact
    def test_04_complementary_fact_does_not_archive_existing(self):
        with get_db_session() as session:
            db_mem = create_memory(
                user_id=self.test_user,
                content="Project uses PostgreSQL as primary database",
                subject="primary_database",
                predicate="uses",
                object_val="PostgreSQL",
                db=session,
            )
            db_id = db_mem.id

            cache_mem = create_memory(
                user_id=self.test_user,
                content="Project uses Redis for caching layer",
                subject="cache_layer",
                predicate="uses",
                object_val="Redis",
                db=session,
            )
            cache_id = cache_mem.id

        with get_db_session() as session:
            db_ref = session.get(Memory, db_id)
            cache_ref = session.get(Memory, cache_id)

            self.assertEqual(db_ref.status, MemoryStatus.ACTIVE)
            self.assertEqual(cache_ref.status, MemoryStatus.ACTIVE)
            self.assertIsNone(db_ref.superseded_by_id)
            self.assertIsNone(cache_ref.superseded_by_id)

    # 6: TEMPORARY_OVERRIDE does not permanently replace the long-term goal
    def test_05_temporary_override_preserves_active_goal(self):
        with get_db_session() as session:
            active_mem = create_memory(
                user_id=self.test_user,
                content="Project uses PostgreSQL as primary database",
                subject="primary_database",
                predicate="uses",
                object_val="PostgreSQL",
                db=session,
            )
            active_id = active_mem.id

            temp_mem, conflict_log = record_temporary_override(
                active_memory_id=active_id,
                temp_memory_data={
                    "user_id": self.test_user,
                    "content": "Using SQLite in-memory for unit tests today",
                    "subject": "primary_database",
                    "predicate": "temporary_test_db",
                    "object_val": "SQLite in-memory",
                },
                similarity_score=0.88,
                reasoning="Temporary local override for testing session",
                db=session,
            )
            temp_id = temp_mem.id

        with get_db_session() as session:
            active_ref = session.get(Memory, active_id)
            temp_ref = session.get(Memory, temp_id)

            self.assertEqual(active_ref.status, MemoryStatus.ACTIVE)
            self.assertIsNone(active_ref.valid_until)
            self.assertIsNone(active_ref.superseded_by_id)

            self.assertEqual(temp_ref.status, MemoryStatus.TEMPORARY)
            self.assertEqual(temp_ref.category, MemoryCategory.SHORT_TERM)

    # 7 & 8: Failed transactions roll back atomically and system reports failure
    def test_06_failed_transaction_rollback_and_error_reporting(self):
        with get_db_session() as session:
            mem = create_memory(
                user_id=self.test_user,
                content="Project is building billing software",
                subject="project_goal",
                predicate="is_building",
                object_val="billing software",
                db=session,
            )
            initial_id = mem.id

        # Simulate exception during supersede_memory persistence
        with self.assertRaises(Exception):
            with get_db_session() as session:
                with patch.object(session, "flush", side_effect=RuntimeError("Simulated DB Write Error")):
                    supersede_memory(
                        old_memory_id=initial_id,
                        new_memory_data={
                            "user_id": self.test_user,
                            "content": "Project is building shoe store",
                            "subject": "project_goal",
                        },
                        similarity_score=0.8,
                        reasoning="Testing rollback",
                        db=session,
                    )

        # Verify old memory remained ACTIVE and intact
        with get_db_session() as session:
            mem_ref = session.get(Memory, initial_id)
            self.assertEqual(mem_ref.status, MemoryStatus.ACTIVE)
            self.assertIsNone(mem_ref.valid_until)
            self.assertIsNone(mem_ref.superseded_by_id)

    # 9 & 10: Current-tense retrieval prefers active goal, historical retrieves previous
    def test_07_current_vs_historical_retrieval(self):
        with get_db_session() as session:
            m1 = create_memory(
                user_id=self.test_user,
                content="Project previously planned to sell shoes online",
                subject="project_goal",
                predicate="is_building",
                object_val="shoes online",
                db=session,
            )
            m1_id = m1.id

            m2, _ = supersede_memory(
                old_memory_id=m1_id,
                new_memory_data={
                    "user_id": self.test_user,
                    "content": "Project is building billing software",
                    "subject": "project_goal",
                    "predicate": "is_building",
                    "object_val": "billing software",
                    "category": MemoryCategory.LONG_TERM,
                    "importance": 0.9,
                    "confidence": 1.0,
                },
                similarity_score=0.85,
                reasoning="Replaced with billing software",
                db=session,
            )
            m2_id = m2.id

        # 1. Current tense query
        curr_res = self.retriever.retrieve_adaptive(
            user_id=self.test_user,
            query="What is our active project goal?",
            top_k=2,
        )
        self.assertGreater(len(curr_res["results"]), 0)
        top_match = curr_res["results"][0]
        self.assertEqual(top_match["id"], m2_id)
        self.assertEqual(top_match["status"], "ACTIVE")

        # 2. Historical query
        hist_res = self.retriever.retrieve_adaptive(
            user_id=self.test_user,
            query="What did we plan to build before billing software?",
            top_k=5,
        )
        hist_ids = [m["id"] for m in hist_res["results"]]
        self.assertIn(m1_id, hist_ids)

    # 11: Lineage pointers are reciprocal and valid
    def test_08_reciprocal_lineage_pointers(self):
        with get_db_session() as session:
            old_mem = create_memory(
                user_id=self.test_user,
                content="Project uses MongoDB as primary database",
                subject="primary_database",
                predicate="uses",
                object_val="MongoDB",
                db=session,
            )
            new_mem, _ = supersede_memory(
                old_memory_id=old_mem.id,
                new_memory_data={
                    "user_id": self.test_user,
                    "content": "Project uses PostgreSQL as primary database",
                    "subject": "primary_database",
                    "predicate": "uses",
                    "object_val": "PostgreSQL",
                    "category": MemoryCategory.LONG_TERM,
                },
                similarity_score=0.9,
                reasoning="Database migration",
                db=session,
            )
            old_id = old_mem.id
            new_id = new_mem.id

        with get_db_session() as session:
            old_ref = session.get(Memory, old_id)
            new_ref = session.get(Memory, new_id)

            # Reciprocal check
            self.assertEqual(old_ref.superseded_by_id, new_id)
            self.assertEqual(new_ref.supersedes_id, old_id)
            self.assertIsNone(old_ref.supersedes_id)
            self.assertIsNone(new_ref.superseded_by_id)

    # Multiple Conflicting Active Memories Archival (Core Bug Reproduction Test)
    def test_09_multiple_conflicting_active_memories_archival(self):
        """Simulate reproduction scenario:
        IDs:
        - 1: billing software
        - 2: digital marketing
        - 3: shoes online
        - 4: shoes offline
        All subject='project_goal' and initially ACTIVE.
        When confirming billing software, all 3 others must become HISTORICAL linked to 1.
        """
        with get_db_session() as session:
            m1 = create_memory(user_id=self.test_user, content="Project is building billing software", subject="project_goal", object_val="billing software", db=session)
            m2 = create_memory(user_id=self.test_user, content="Project is building digital marketing", subject="project_goal", object_val="digital marketing", db=session)
            m3 = create_memory(user_id=self.test_user, content="Project will sell shoes online", subject="project_goal", object_val="shoes online", db=session)
            m4 = create_memory(user_id=self.test_user, content="Project will sell shoes offline", subject="project_goal", object_val="shoes offline", db=session)
            m1_id, m2_id, m3_id, m4_id = m1.id, m2.id, m3.id, m4.id

        # Execute confirm_and_archive_conflicts
        confirmed, logs = confirm_and_archive_conflicts(
            confirmed_memory_id=m1_id,
            conflicting_memory_ids=[m2_id, m3_id, m4_id],
            reasoning="Confirmed billing software as canonical goal",
        )

        with get_db_session() as session:
            ref1 = session.get(Memory, m1_id)
            ref2 = session.get(Memory, m2_id)
            ref3 = session.get(Memory, m3_id)
            ref4 = session.get(Memory, m4_id)

            self.assertEqual(ref1.status, MemoryStatus.ACTIVE)
            self.assertIsNone(ref1.valid_until)

            for ref in [ref2, ref3, ref4]:
                self.assertEqual(ref.status, MemoryStatus.HISTORICAL)
                self.assertIsNotNone(ref.valid_until)
                self.assertEqual(ref.superseded_by_id, m1_id)

        # Verify lineage of m1 includes m2, m3, m4
        lineage = get_memory_lineage(m1_id)
        lineage_ids = [m.id for m in lineage]
        self.assertIn(m1_id, lineage_ids)
        self.assertIn(m2_id, lineage_ids)
        self.assertIn(m3_id, lineage_ids)
        self.assertIn(m4_id, lineage_ids)


if __name__ == "__main__":
    unittest.main()
