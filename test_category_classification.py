"""Regression test suite for memory-category classification, validation, and conflict-resolution semantics.

Test Cases:
1. Project goal -> SEMANTIC.
2. Stable user-profile fact -> SEMANTIC.
3. Technology-stack choice -> LONG_TERM.
4. Specific migration or incident -> EPISODIC.
5. Temporary session override -> SHORT_TERM, with correct lifecycle status (TEMPORARY).
6. Superseding a semantic project goal preserves SEMANTIC on both records while updating their statuses and lineage.
7. Raju says travel time is 5 hours; Ravi says 6 hours. Preserve both sources and their reported estimates.
   When the user explicitly confirms that Ravi replaced Raju's estimate, apply the existing supersession behavior
   without losing the historical record or attribution.
8. Missing or invalid category values are handled safely (fallback, synonyms, validation).
9. Existing conflict-resolution and retrieval regression tests still pass.
"""

import sys
import unittest
from pathlib import Path

# Ensure root dir in python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.database.models import (
    Memory,
    MemoryCategory,
    MemoryStatus,
    MemoryConflictLog,
    ConflictResolutionType,
)
from app.database.db import (
    get_db_session,
    create_memory,
    supersede_memory,
    confirm_and_archive_conflicts,
    record_temporary_override,
    get_memory_lineage,
    backfill_memory_categories,
)
from app.services.llm_service import (
    ExtractedFact,
    LLMService,
    classify_memory_category,
)
from app.services.conflict_resolver import MemoryIngestionService
from app.services.retriever import MemoryRetriever


class TestCategoryClassification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_user = "test_user_category_regression"
        cls.llm_service = LLMService()
        cls.ingestion_service = MemoryIngestionService(llm_service=cls.llm_service)
        cls.retriever = MemoryRetriever()

    def setUp(self):
        # Clean only test user records before each test
        with get_db_session() as session:
            session.query(MemoryConflictLog).filter(MemoryConflictLog.user_id == self.test_user).delete()
            session.query(Memory).filter(Memory.user_id == self.test_user).update(
                {"superseded_by_id": None, "supersedes_id": None}, synchronize_session=False
            )
            session.query(Memory).filter(Memory.user_id == self.test_user).delete()
            session.commit()

    def tearDown(self):
        # Clean up test user records after each test
        with get_db_session() as session:
            session.query(MemoryConflictLog).filter(MemoryConflictLog.user_id == self.test_user).delete()
            session.query(Memory).filter(Memory.user_id == self.test_user).update(
                {"superseded_by_id": None, "supersedes_id": None}, synchronize_session=False
            )
            session.query(Memory).filter(Memory.user_id == self.test_user).delete()
            session.commit()

    # -------------------------------------------------------------------------
    # 1. Project goal -> SEMANTIC
    # -------------------------------------------------------------------------
    def test_1_project_goal_semantic(self):
        goal_texts = [
            "we are going to build a rag chatbot engine",
            "Project is building billing software",
            "We plan to build a real-time analytics dashboard",
            "Project will sell shoes online",
            "Project focuses on manufacturing units",
        ]
        for text in goal_texts:
            facts = self.llm_service._fallback_extract_memories(text)
            self.assertTrue(len(facts) > 0, f"Expected facts extracted for '{text}'")
            fact = facts[0]
            normalized_cat = self.llm_service._normalize_or_fallback_category(fact)
            self.assertEqual(
                normalized_cat,
                MemoryCategory.SEMANTIC,
                f"Goal '{text}' should be classified as SEMANTIC, got {normalized_cat}",
            )

        # Direct classifier check
        cat = classify_memory_category(
            content="Project is building a RAG chatbot engine",
            subject="project_goal",
            predicate="is_building",
            object_val="RAG chatbot engine",
        )
        self.assertEqual(cat, MemoryCategory.SEMANTIC)

    # -------------------------------------------------------------------------
    # 2. Stable user-profile fact -> SEMANTIC
    # -------------------------------------------------------------------------
    def test_2_stable_user_profile_semantic(self):
        user_texts = [
            "Hi, I'm Aarav, a backend architect.",
            "User is a principal cloud engineer",
            "User prefers dark mode and concise responses",
        ]
        for text in user_texts:
            facts = self.llm_service._fallback_extract_memories(text)
            if facts:
                fact = facts[0]
                normalized_cat = self.llm_service._normalize_or_fallback_category(fact)
                self.assertEqual(
                    normalized_cat,
                    MemoryCategory.SEMANTIC,
                    f"User profile '{text}' should be SEMANTIC, got {normalized_cat}",
                )

        # Direct classifier check
        cat = classify_memory_category(
            content="User Aarav is a backend architect",
            subject="user_profile",
            predicate="has_role",
            object_val="Aarav (backend architect)",
        )
        self.assertEqual(cat, MemoryCategory.SEMANTIC)

        cat_pref = classify_memory_category(
            content="User prefers dark mode and concise responses",
            subject="user_preference",
            predicate="prefers",
            object_val="dark mode",
        )
        self.assertEqual(cat_pref, MemoryCategory.SEMANTIC)

    # -------------------------------------------------------------------------
    # 3. Technology-stack choice -> LONG_TERM
    # -------------------------------------------------------------------------
    def test_3_technology_stack_choice_long_term(self):
        tech_cases = [
            ("Project uses PostgreSQL as primary database", "primary_database", "uses", "PostgreSQL"),
            ("Project uses FastAPI for the backend", "backend_framework", "uses", "FastAPI"),
            ("Project uses sentence transformer for vector embeddings", "embedding_model", "uses", "sentence transformer"),
            ("Project will NOT use sentence transformer for vector embeddings", "embedding_model", "does_not_use", "NOT sentence transformer"),
            ("Project uses React for the frontend", "frontend_framework", "uses", "React"),
        ]
        for content, subj, pred, obj in tech_cases:
            fact = ExtractedFact(
                content=content,
                subject=subj,
                predicate=pred,
                object_val=obj,
                category=None,  # test fallback classification
                is_temporary=False,
            )
            normalized_cat = self.llm_service._normalize_or_fallback_category(fact)
            self.assertEqual(
                normalized_cat,
                MemoryCategory.LONG_TERM,
                f"Tech stack fact '{content}' should be LONG_TERM, got {normalized_cat}",
            )

        # Direct extraction check
        extracted = self.llm_service._fallback_extract_memories(
            "we are gonna use sentence transformer for vector embeddings"
        )
        self.assertTrue(len(extracted) > 0)
        self.assertEqual(extracted[0].category, MemoryCategory.LONG_TERM)

    # -------------------------------------------------------------------------
    # 4. Specific migration or incident -> EPISODIC
    # -------------------------------------------------------------------------
    def test_4_specific_migration_or_incident_episodic(self):
        episodic_cases = [
            "On Tuesday, the team officially completed migration from MySQL to PostgreSQL",
            "Production deployment failed due to OOM errors yesterday",
            "User was promoted to Principal Backend Architect on Friday",
            "In sprint planning on Monday, team agreed to delay the export feature",
        ]
        for text in episodic_cases:
            cat = classify_memory_category(content=text, subject="system_event")
            self.assertEqual(
                cat,
                MemoryCategory.EPISODIC,
                f"Event '{text}' should be EPISODIC, got {cat}",
            )

            fact = ExtractedFact(
                content=text,
                subject="primary_database" if "migration" in text else "incident",
                predicate="occurred",
                object_val="event",
                category=None,
            )
            # Even if subject is primary_database, temporal/event meaning makes it EPISODIC
            normalized_cat = self.llm_service._normalize_or_fallback_category(fact)
            self.assertEqual(
                normalized_cat,
                MemoryCategory.EPISODIC,
                f"Event with subject '{fact.subject}' should be EPISODIC, got {normalized_cat}",
            )

    # -------------------------------------------------------------------------
    # 5. Temporary session override -> SHORT_TERM with TEMPORARY status
    # -------------------------------------------------------------------------
    def test_5_temporary_session_override_short_term(self):
        fact = ExtractedFact(
            content="Using SQLite in-memory for local unit tests today",
            subject="primary_database",
            predicate="temporary_test_db",
            object_val="SQLite in-memory",
            is_temporary=True,
        )
        self.assertEqual(fact.category, MemoryCategory.SHORT_TERM)
        normalized_cat = self.llm_service._normalize_or_fallback_category(fact)
        self.assertEqual(normalized_cat, MemoryCategory.SHORT_TERM)

        # Ingestion test
        mem = create_memory(
            user_id=self.test_user,
            session_id="session_temp_01",
            content=fact.content,
            subject=fact.subject,
            predicate=fact.predicate,
            object_val=fact.object_val,
            category=normalized_cat,
            status=MemoryStatus.TEMPORARY,
        )
        self.assertEqual(mem.category, MemoryCategory.SHORT_TERM)
        self.assertEqual(mem.status, MemoryStatus.TEMPORARY)

    # -------------------------------------------------------------------------
    # 6. Superseding a semantic project goal preserves SEMANTIC on both records
    # -------------------------------------------------------------------------
    def test_6_superseding_semantic_goal_preserves_category(self):
        # Initial goal
        old_fact = ExtractedFact(
            content="Project is building a RAG chatbot engine",
            subject="project_goal",
            predicate="is_building",
            object_val="RAG chatbot engine",
            category=MemoryCategory.SEMANTIC,
            is_temporary=False,
        )
        old_mem = create_memory(
            user_id=self.test_user,
            session_id="session_01",
            content=old_fact.content,
            subject=old_fact.subject,
            predicate=old_fact.predicate,
            object_val=old_fact.object_val,
            category=old_fact.category,
            status=MemoryStatus.ACTIVE,
        )
        self.assertEqual(old_mem.category, MemoryCategory.SEMANTIC)
        self.assertEqual(old_mem.status, MemoryStatus.ACTIVE)

        # Replacement goal
        new_fact = ExtractedFact(
            content="Project is building an enterprise billing platform",
            subject="project_goal",
            predicate="is_building",
            object_val="enterprise billing platform",
            category=MemoryCategory.SEMANTIC,
            is_temporary=False,
        )
        new_mem, log = supersede_memory(
            old_memory_id=old_mem.id,
            new_memory_data={
                "user_id": self.test_user,
                "content": new_fact.content,
                "subject": new_fact.subject,
                "predicate": new_fact.predicate,
                "object_val": new_fact.object_val,
                "category": new_fact.category,
            },
            similarity_score=0.88,
            reasoning="Pivoted project goal from chatbot to billing platform",
        )

        with get_db_session() as session:
            reloaded_old = session.get(Memory, old_mem.id)
            reloaded_new = session.get(Memory, new_mem.id)

            # Both records MUST retain SEMANTIC
            self.assertEqual(reloaded_old.category, MemoryCategory.SEMANTIC)
            self.assertEqual(reloaded_new.category, MemoryCategory.SEMANTIC)

            # Statuses MUST be properly updated
            self.assertEqual(reloaded_old.status, MemoryStatus.HISTORICAL)
            self.assertEqual(reloaded_new.status, MemoryStatus.ACTIVE)

            # Reciprocal lineage MUST be intact
            self.assertEqual(reloaded_old.superseded_by_id, reloaded_new.id)
            self.assertEqual(reloaded_new.supersedes_id, reloaded_old.id)
            self.assertIsNotNone(reloaded_old.valid_until)

    # -------------------------------------------------------------------------
    # 7. Raju says 5 hours; Ravi says 6 hours.
    #    Preserve both sources & estimates.
    #    When confirmed Ravi replaced Raju, apply supersession without losing attribution.
    # -------------------------------------------------------------------------
    def test_7_raju_ravi_travel_time_attribution_and_supersession(self):
        # Raju statement
        raju_fact = ExtractedFact(
            content="Raju reported that travel time from Bangalore to Chennai is 5 hours",
            subject="travel_time",
            predicate="reported",
            object_val="5 hours",
            category=MemoryCategory.EPISODIC,
            is_temporary=False,
        )
        self.assertEqual(raju_fact.category, MemoryCategory.EPISODIC)

        raju_mem = create_memory(
            user_id=self.test_user,
            session_id="session_route",
            content=raju_fact.content,
            subject=raju_fact.subject,
            predicate=raju_fact.predicate,
            object_val=raju_fact.object_val,
            category=raju_fact.category,
            status=MemoryStatus.ACTIVE,
        )

        # Ravi statement
        ravi_fact = ExtractedFact(
            content="Ravi reported that travel time from Bangalore to Chennai is 6 hours",
            subject="travel_time",
            predicate="reported",
            object_val="6 hours",
            category=MemoryCategory.EPISODIC,
            is_temporary=False,
        )
        self.assertEqual(ravi_fact.category, MemoryCategory.EPISODIC)

        # Adjudication between Raju and Ravi should be COMPLEMENTARY (two distinct sources)
        adj = self.llm_service._fallback_evaluate_conflict(
            old_memory=raju_mem,
            new_fact=ravi_fact,
            similarity=0.85,
        )
        self.assertEqual(
            adj.resolution_type,
            ConflictResolutionType.COMPLEMENTARY,
            "Distinct speaker reports should initially be preserved as COMPLEMENTARY.",
        )

        # Store Ravi's estimate alongside Raju's estimate
        ravi_mem = create_memory(
            user_id=self.test_user,
            session_id="session_route",
            content=ravi_fact.content,
            subject=ravi_fact.subject,
            predicate=ravi_fact.predicate,
            object_val=ravi_fact.object_val,
            category=ravi_fact.category,
            status=MemoryStatus.ACTIVE,
        )

        # Verify both memories exist and preserve their attributions and estimates
        with get_db_session() as session:
            m1 = session.get(Memory, raju_mem.id)
            m2 = session.get(Memory, ravi_mem.id)
            self.assertEqual(m1.status, MemoryStatus.ACTIVE)
            self.assertEqual(m2.status, MemoryStatus.ACTIVE)
            self.assertIn("Raju", m1.content)
            self.assertIn("5 hours", m1.content)
            self.assertIn("Ravi", m2.content)
            self.assertIn("6 hours", m2.content)

        # Now user explicitly confirms that Ravi's estimate replaced Raju's estimate
        confirmed_mem, logs = confirm_and_archive_conflicts(
            confirmed_memory_id=ravi_mem.id,
            conflicting_memory_ids=[raju_mem.id],
            reasoning="User confirmed Ravi's 6-hour estimate officially replaced Raju's earlier 5-hour estimate.",
        )

        with get_db_session() as session:
            archived_raju = session.get(Memory, raju_mem.id)
            active_ravi = session.get(Memory, ravi_mem.id)

            # Attribution and content MUST be preserved
            self.assertIn("Raju", archived_raju.content)
            self.assertIn("5 hours", archived_raju.content)
            self.assertIn("Ravi", active_ravi.content)
            self.assertIn("6 hours", active_ravi.content)

            # Categories MUST remain EPISODIC
            self.assertEqual(archived_raju.category, MemoryCategory.EPISODIC)
            self.assertEqual(active_ravi.category, MemoryCategory.EPISODIC)

            # Statuses & Lineage updated properly
            self.assertEqual(archived_raju.status, MemoryStatus.HISTORICAL)
            self.assertEqual(active_ravi.status, MemoryStatus.ACTIVE)
            self.assertEqual(archived_raju.superseded_by_id, active_ravi.id)
            self.assertIsNotNone(archived_raju.valid_until)

    # -------------------------------------------------------------------------
    # 8. Missing or invalid category values handled safely
    # -------------------------------------------------------------------------
    def test_8_missing_or_invalid_category_handling(self):
        # 1. Missing category on a project goal -> safely falls back to SEMANTIC
        f1 = ExtractedFact(
            content="Project is building a RAG chatbot engine",
            subject="project_goal",
            predicate="is_building",
            object_val="RAG chatbot engine",
            category=None,
        )
        self.assertEqual(f1.category, MemoryCategory.SEMANTIC)

        # 2. Invalid string category on database -> safely falls back to LONG_TERM
        f2 = ExtractedFact(
            content="Project uses PostgreSQL as primary database",
            subject="primary_database",
            predicate="uses",
            object_val="PostgreSQL",
            category="INVALID_BOGUS_CATEGORY",
        )
        self.assertEqual(f2.category, MemoryCategory.LONG_TERM)

        # 3. Invalid category on incident -> safely falls back to EPISODIC
        f3 = ExtractedFact(
            content="Production deployment failed due to OOM errors yesterday",
            subject="incident",
            predicate="failed",
            object_val="OOM errors",
            category="UNKNOWN_CAT",
        )
        self.assertEqual(f3.category, MemoryCategory.EPISODIC)

        # 4. Empty string category on temporary override -> safely falls back to SHORT_TERM
        f4 = ExtractedFact(
            content="Using SQLite in-memory for unit tests today",
            subject="primary_database",
            predicate="uses",
            object_val="SQLite",
            category="",
            is_temporary=True,
        )
        self.assertEqual(f4.category, MemoryCategory.SHORT_TERM)

        # 5. Synonym mapping: "TECH_STACK" -> LONG_TERM, "GOAL" -> SEMANTIC, "EVENT" -> EPISODIC
        f5 = ExtractedFact(
            content="Project uses FastAPI for backend",
            subject="backend_framework",
            predicate="uses",
            object_val="FastAPI",
            category="TECH_STACK",
        )
        self.assertEqual(f5.category, MemoryCategory.LONG_TERM)

        f6 = ExtractedFact(
            content="Project aims to provide enterprise billing",
            subject="project_goal",
            predicate="aims_to",
            object_val="enterprise billing",
            category="GOAL",
        )
        self.assertEqual(f6.category, MemoryCategory.SEMANTIC)

        f7 = ExtractedFact(
            content="Sprint 5 meeting completed on Tuesday",
            subject="milestone",
            predicate="completed",
            object_val="Sprint 5",
            category="EVENT",
        )
        self.assertEqual(f7.category, MemoryCategory.EPISODIC)

    # -------------------------------------------------------------------------
    # 9. Existing conflict resolution and retrieval still pass
    # -------------------------------------------------------------------------
    def test_9_conflict_resolution_and_retrieval_pass(self):
        # 1. Ingest initial database
        m1 = create_memory(
            user_id=self.test_user,
            session_id="sess_01",
            content="Project uses MongoDB as primary database",
            subject="primary_database",
            predicate="uses",
            object_val="MongoDB",
            category=MemoryCategory.LONG_TERM,
            status=MemoryStatus.ACTIVE,
        )

        # 2. Supersede with PostgreSQL
        m2, log = supersede_memory(
            old_memory_id=m1.id,
            new_memory_data={
                "user_id": self.test_user,
                "session_id": "sess_02",
                "content": "Project uses PostgreSQL as primary database",
                "subject": "primary_database",
                "predicate": "uses",
                "object_val": "PostgreSQL",
                "category": MemoryCategory.LONG_TERM,
            },
            similarity_score=0.92,
            reasoning="Migrated from MongoDB to PostgreSQL",
        )

        # 3. Retrieve lineage
        lineage = get_memory_lineage(m2.id)
        self.assertEqual(len(lineage), 2)
        self.assertEqual(lineage[0].id, m1.id)
        self.assertEqual(lineage[1].id, m2.id)
        self.assertEqual(lineage[0].status, MemoryStatus.HISTORICAL)
        self.assertEqual(lineage[1].status, MemoryStatus.ACTIVE)
        self.assertEqual(lineage[0].category, MemoryCategory.LONG_TERM)
        self.assertEqual(lineage[1].category, MemoryCategory.LONG_TERM)

        # 4. Adaptive retrieval for current database
        retrieval_res = self.retriever.retrieve_adaptive(
            user_id=self.test_user,
            query="What is our primary database?",
            top_k=3,
        )
        results = retrieval_res.get("results", [])
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0]["id"], m2.id)
        self.assertIn("PostgreSQL", results[0]["content"])
        self.assertEqual(results[0]["status"], "ACTIVE")


if __name__ == "__main__":
    unittest.main(verbosity=2)
