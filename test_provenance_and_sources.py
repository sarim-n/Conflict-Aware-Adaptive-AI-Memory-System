"""Regression tests for Source Type Distinction, Provenance, and Unsupported Inference Rejection.

Verifies:
1. Six supported facts are extracted correctly across the conversation turns.
2. Coreference resolution properly binds pronouns ('we' -> Manoj / Project) with provenance_context.
3. Assistant-generated suggestions and questions are NOT extracted as persistent memories.
4. Mall date retains exact input without any invented year.
5. Client meeting and deal finalization remain distinct events and are not merged into a single dated event.
6. Company ownership claim does not invent ownership shares or unstated owners.
7. Where supported by the schema, source_turn_id provenance is preserved.
8. Category classification, lifecycle status, conflict resolution, and lineage are preserved.
"""

import unittest
from datetime import datetime
from typing import List, Dict, Any, Optional

from app.database.models import (
    Memory,
    MemoryCategory,
    MemoryStatus,
    MemoryConflictLog,
    ConflictResolutionType,
    ConversationTurn,
)
from app.database.db import get_db_session, create_memory
from app.services.llm_service import LLMService, ExtractedFact
from app.services.conflict_resolver import MemoryIngestionService


class TestProvenanceAndSources(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.llm_service = LLMService()
        cls.ingestion_service = MemoryIngestionService(llm_service=cls.llm_service)

    def test_six_supported_facts_extracted_correctly(self):
        """Test that all 6 supported facts are extracted with appropriate categories and source types."""
        # Turn 1: Friend's name
        facts_1 = self.llm_service._fallback_extract_memories("my friend's name is manoj")
        self.assertEqual(len(facts_1), 1)
        self.assertEqual(facts_1[0].subject, "friend_name")
        self.assertEqual(facts_1[0].predicate, "has_name")
        self.assertEqual(facts_1[0].object_val, "Manoj")
        self.assertEqual(facts_1[0].category, MemoryCategory.SEMANTIC)
        self.assertEqual(facts_1[0].source_type, "user_confirmed")

        # Turn 2: Met 6 years ago with context resolving 'we'
        ctx_2 = "User: my friend's name is manoj\nAssistant: Got it—I've noted that your friend's name is Manoj."
        facts_2 = self.llm_service._fallback_extract_memories("we met 6 years ago", recent_context=ctx_2)
        self.assertEqual(len(facts_2), 1)
        self.assertEqual(facts_2[0].subject, "friend_meeting")
        self.assertEqual(facts_2[0].predicate, "met")
        self.assertEqual(facts_2[0].object_val, "Manoj")
        self.assertEqual(facts_2[0].category, MemoryCategory.EPISODIC)
        self.assertEqual(facts_2[0].source_type, "context_resolved")
        self.assertIn("manoj", facts_2[0].content.lower())
        self.assertIsNotNone(facts_2[0].provenance_context)

        # Turn 3: Went to mall on 19th October (with typo tolerance)
        facts_3 = self.llm_service._fallback_extract_memories("i went to mall on 19th ovtober")
        self.assertEqual(len(facts_3), 1)
        self.assertEqual(facts_3[0].subject, "mall_visit")
        self.assertEqual(facts_3[0].predicate, "went_to")
        self.assertEqual(facts_3[0].category, MemoryCategory.EPISODIC)
        self.assertEqual(facts_3[0].source_type, "user_confirmed")
        # Ensure no year was invented
        self.assertNotIn("202", facts_3[0].content)

        # Turn 4: Meet business client along with Manoj
        facts_4 = self.llm_service._fallback_extract_memories("i went to meet a business client along with manoj")
        self.assertEqual(len(facts_4), 1)
        self.assertEqual(facts_4[0].subject, "client_meeting")
        self.assertEqual(facts_4[0].predicate, "attended")
        self.assertEqual(facts_4[0].category, MemoryCategory.EPISODIC)
        self.assertEqual(facts_4[0].source_type, "user_confirmed")
        self.assertIn("manoj", facts_4[0].content.lower())

        # Turn 5: Finalised deal on supplying water bottles to client
        ctx_5 = "User: i went to meet a business client along with manoj"
        facts_5 = self.llm_service._fallback_extract_memories(
            "we finalised a deal on supplying water bottles to the client",
            recent_context=ctx_5,
        )
        self.assertEqual(len(facts_5), 1)
        self.assertEqual(facts_5[0].subject, "client_deal")
        self.assertEqual(facts_5[0].predicate, "finalised")
        self.assertEqual(facts_5[0].object_val, "water bottle supply")
        self.assertEqual(facts_5[0].category, MemoryCategory.EPISODIC)
        self.assertEqual(facts_5[0].source_type, "context_resolved")
        self.assertIsNotNone(facts_5[0].provenance_context)

        # Turn 6: Owner of successful water bottle company
        facts_6 = self.llm_service._fallback_extract_memories("we are the owner of a successfull water bottlw comapany")
        self.assertEqual(len(facts_6), 1)
        self.assertEqual(facts_6[0].subject, "company")
        self.assertEqual(facts_6[0].predicate, "owned_by")
        self.assertEqual(facts_6[0].object_val, "User")
        self.assertEqual(facts_6[0].content, "User owns a successful water bottle company")
        self.assertEqual(facts_6[0].category, MemoryCategory.SEMANTIC)
        self.assertEqual(facts_6[0].source_type, "context_resolved")
        self.assertIsNotNone(facts_6[0].provenance_context)

    def test_assistant_suggestions_do_not_become_persistent_memories(self):
        """Assistant questions/suggestions in context must not become memories."""
        # 1. Assistant suggestion about meeting Manoj at the mall
        ctx_mall = (
            "User: i went to mall on 19th ovtober\n"
            "Assistant: Sounds like a nice outing! Did you run into Manoj there?"
        )
        # If user message is unrelated or doesn't confirm, extractor must not extract Manoj at mall
        facts = self.llm_service.extract_memories(
            user_message="i went to mall on 19th ovtober",
            recent_context=ctx_mall,
        )
        for f in facts:
            self.assertFalse(
                "manoj" in f.content.lower() and "mall" in f.content.lower(),
                "Must not infer that Manoj attended the mall visit!",
            )

        # 2. Assistant suggestions about drafting contracts or marketing campaigns
        ctx_deal = (
            "User: we finalised a deal on supplying water bottles to the client\n"
            "Assistant: Do you need to draft the contract, set up the delivery schedule, or prepare marketing material?"
        )
        facts_deal = self.llm_service.extract_memories(
            user_message="we finalised a deal on supplying water bottles to the client",
            recent_context=ctx_deal,
        )
        for f in facts_deal:
            self.assertNotIn("draft contract", f.content.lower())
            self.assertNotIn("drafting", f.content.lower())
            self.assertNotIn("marketing campaign", f.content.lower())

        # 3. Assistant suggestions about expanding production or distribution channels
        ctx_company = (
            "User: we are the owner of a successfull water bottlw comapany\n"
            "Assistant: Are you looking to expand production, explore new distribution channels, or start a marketing campaign?"
        )
        facts_comp = self.llm_service.extract_memories(
            user_message="we are the owner of a successfull water bottlw comapany",
            recent_context=ctx_company,
        )
        for f in facts_comp:
            self.assertNotIn("expand production", f.content.lower())
            self.assertNotIn("distribution channel", f.content.lower())
            self.assertNotIn("marketing campaign", f.content.lower())

    def test_mall_visit_has_no_invented_year(self):
        """Verify that the mall visit retains only the provided date and no hallucinated year."""
        facts = self.llm_service.extract_memories(
            user_message="i went to mall on 19th ovtober",
            recent_context=None,
        )
        self.assertTrue(len(facts) >= 1)
        for f in facts:
            if "mall" in f.subject.lower() or "mall" in f.content.lower():
                self.assertNotIn("202", f.content, "Invented year found in mall visit memory!")
                self.assertIn("19", f.content)
                self.assertTrue(any(m in f.content.lower() for m in ["october", "ovtober"]))

    def test_meeting_and_deal_remain_separate_events(self):
        """Verify meeting and deal are not merged into a single dated event."""
        facts_meet = self.llm_service.extract_memories("i went to meet a business client along with manoj")
        facts_deal = self.llm_service.extract_memories(
            "we finalised a deal on supplying water bottles to the client",
            recent_context="User: i went to meet a business client along with manoj",
        )
        self.assertEqual(facts_meet[0].subject, "client_meeting")
        deal_fact = next((f for f in facts_deal if f.subject == "client_deal"), facts_deal[-1])
        self.assertEqual(deal_fact.subject, "client_deal")
        self.assertNotEqual(facts_meet[0].subject, deal_fact.subject)
        # Neither should have an invented date linking them
        self.assertNotIn("on the same day", deal_fact.content.lower())
        self.assertNotIn("same date", deal_fact.content.lower())

    def test_ownership_does_not_invent_shares_or_owners(self):
        """Verify ownership claim does not invent percentages or extra unstated owners."""
        facts = self.llm_service.extract_memories("we are the owner of a successfull water bottlw comapany")
        self.assertTrue(len(facts) >= 1)
        for f in facts:
            self.assertNotIn("50%", f.content)
            self.assertNotIn("100%", f.content)
            self.assertNotIn("shareholder", f.content.lower())
            self.assertNotIn("co-owner", f.content.lower())

    def test_source_turn_id_provenance_tracking(self):
        """Verify that passing source_turn_id properly persists the lineage in the database."""
        test_user = "test_provenance_user"
        test_session = "test_prov_session"

        # Simulate turn pre-registration
        with get_db_session() as session:
            turn = ConversationTurn(
                user_id=test_user,
                session_id=test_session,
                user_message="my friend's name is manoj",
                assistant_response="Got it!",
            )
            session.add(turn)
            session.flush()
            turn_id = turn.id

        # Ingest message with source_turn_id
        res = self.ingestion_service.ingest_message(
            user_id=test_user,
            session_id=test_session,
            user_message="my friend's name is manoj",
            source_turn_id=turn_id,
        )

        self.assertTrue(len(res["memories_created"]) >= 1)
        created_id = res["memories_created"][0]["id"]

        with get_db_session() as session:
            persisted = session.get(Memory, created_id)
            self.assertIsNotNone(persisted)
            self.assertEqual(persisted.source_turn_id, turn_id)

            # Clean up test user memories and turns safely
            session.query(Memory).filter(Memory.user_id == test_user).delete()
            session.query(ConversationTurn).filter(ConversationTurn.user_id == test_user).delete()

    def test_regression_repeated_friend_meeting_does_not_create_duplicate(self):
        """Regression test for Defect 1: Repeated meeting statement must reinforce existing active memory rather than create duplicate complementary memory."""
        test_user = "test_user_meeting_dedup"
        test_session = "sess_meeting_dedup"

        try:
            # First turn: friend's name
            self.ingestion_service.ingest_message(
                user_id=test_user,
                session_id=test_session,
                user_message="my friend's name is manoj",
            )
            # Second turn: met 6 years ago
            res1 = self.ingestion_service.ingest_message(
                user_id=test_user,
                session_id=test_session,
                user_message="we met 6 years ago",
                recent_turns=[{"role": "user", "content": "my friend's name is manoj"}],
            )
            self.assertTrue(len(res1["memories_created"]) >= 1)
            first_mem_id = res1["memories_created"][0]["id"]

            # Third turn: repeating the meeting in a new turn
            res2 = self.ingestion_service.ingest_message(
                user_id=test_user,
                session_id=test_session,
                user_message="we met 6 years ago",
                recent_turns=[
                    {"role": "user", "content": "my friend's name is manoj"},
                    {"role": "user", "content": "we met 6 years ago"},
                ],
            )

            # Must reinforce existing memory, not create a new one
            self.assertTrue(any("REINFORCED" in a for a in res2["actions_taken"]))
            self.assertEqual(res2["memories_created"][0]["id"], first_mem_id)

            # Verify in database: exactly 1 ACTIVE memory for friend_meeting exists for this user
            with get_db_session() as session:
                active_meetings = (
                    session.query(Memory)
                    .filter(
                        Memory.user_id == test_user,
                        Memory.subject == "friend_meeting",
                        Memory.status == MemoryStatus.ACTIVE,
                    )
                    .all()
                )
                self.assertEqual(len(active_meetings), 1)
                self.assertEqual(active_meetings[0].id, first_mem_id)
        finally:
            with get_db_session() as session:
                session.query(MemoryConflictLog).filter(MemoryConflictLog.user_id == test_user).delete()
                session.query(Memory).filter(Memory.user_id == test_user).delete()
                session.query(ConversationTurn).filter(ConversationTurn.user_id == test_user).delete()

    def test_regression_company_ownership_attributed_to_user_not_project(self):
        """Regression test for Defect 2: 'we are the owner of a successful water bottle company' must attribute ownership to User, not Project."""
        test_user = "test_user_ownership"
        test_session = "sess_ownership"

        try:
            res = self.ingestion_service.ingest_message(
                user_id=test_user,
                session_id=test_session,
                user_message="we are the owner of a successfull water bottlw comapany",
            )
            self.assertTrue(len(res["memories_created"]) >= 1)
            mem_data = res["memories_created"][0]
            self.assertEqual(mem_data["subject"], "company")
            self.assertIn("user owns", mem_data["content"].lower())
            self.assertNotIn("project", mem_data["content"].lower())

            with get_db_session() as session:
                persisted = session.get(Memory, mem_data["id"])
                self.assertIsNotNone(persisted)
                self.assertEqual(persisted.predicate, "owned_by")
                self.assertEqual(persisted.object_val, "User")
                self.assertNotIn("Project", persisted.content)
        finally:
            with get_db_session() as session:
                session.query(MemoryConflictLog).filter(MemoryConflictLog.user_id == test_user).delete()
                session.query(Memory).filter(Memory.user_id == test_user).delete()
                session.query(ConversationTurn).filter(ConversationTurn.user_id == test_user).delete()

    def test_regression_existing_database_records_lineage_preserved(self):
        """Verify that existing memories 29-35 and historical lineage in data/memory_system.db are preserved."""
        with get_db_session() as session:
            # ID 29, 30, 31, 32, 33, 34, 35
            m29 = session.get(Memory, 29)
            m30 = session.get(Memory, 30)
            m31 = session.get(Memory, 31)
            m32 = session.get(Memory, 32)
            m33 = session.get(Memory, 33)
            m34 = session.get(Memory, 34)
            m35 = session.get(Memory, 35)

            # All records must exist (not deleted or reset)
            self.assertIsNotNone(m29)
            self.assertIsNotNone(m30)
            self.assertIsNotNone(m31)
            self.assertIsNotNone(m32)
            self.assertIsNotNone(m33)
            self.assertIsNotNone(m34)
            self.assertIsNotNone(m35)

            # Lineage of ID 34: superseded by ID 36 in turn 40
            m36 = session.get(Memory, 36)
            if m36:
                self.assertEqual(m34.superseded_by_id, 36)
                self.assertEqual(m36.supersedes_id, 34)

    def test_regression_correction_friend_name_abdul_not_manoj(self):
        """Regression test for fact correction: 'My friend's name is Abdul, not Manoj.'
        
        Verifies:
        1. Identifies Abdul as the current value of friend_name.
        2. Marks the previous Manoj memory as HISTORICAL.
        3. Keeps the Abdul memory ACTIVE.
        4. Does NOT create a standalone active memory such as 'friend's name is NOT Manoj'.
        5. Preserves lineage pointers (supersedes_id / superseded_by_id).
        6. Retrieval returns Abdul as the active memory for friend's name.
        """
        from app.services.retriever import MemoryRetriever
        retriever = MemoryRetriever()
        test_user = "test_user_abdul_correction"
        test_session = "sess_abdul_correction"

        try:
            # Step 1: Initial fact stating friend's name is Manoj
            res1 = self.ingestion_service.ingest_message(
                user_id=test_user,
                session_id=test_session,
                user_message="my friend's name is manoj",
            )
            self.assertTrue(len(res1["memories_created"]) >= 1)
            manoj_mem_id = res1["memories_created"][0]["id"]

            # Step 2: User corrects fact: "My friend's name is Abdul, not Manoj."
            res2 = self.ingestion_service.ingest_message(
                user_id=test_user,
                session_id=test_session,
                user_message="My friend's name is Abdul, not Manoj.",
                recent_turns=[{"role": "user", "content": "my friend's name is manoj"}],
            )

            # Verification of extraction: positive fact extracted, no standalone negative memory
            extracted_facts = res2["extracted_facts"]
            abdul_fact = next((f for f in extracted_facts if f.get("subject") == "friend_name"), None)
            self.assertIsNotNone(abdul_fact)
            self.assertEqual(abdul_fact.get("object_val"), "Abdul")
            self.assertNotIn("NOT Manoj", [f.get("object_val") for f in extracted_facts])

            # Verification of conflict classification: must be a SUPERSEDED / PERMANENT_UPDATE
            self.assertTrue(any("SUPERSEDED" in a for a in res2["actions_taken"]))
            abdul_mem_id = res2["memories_created"][0]["id"]
            self.assertNotEqual(abdul_mem_id, manoj_mem_id)

            # Verification in database
            with get_db_session() as session:
                manoj_db = session.get(Memory, manoj_mem_id)
                abdul_db = session.get(Memory, abdul_mem_id)

                self.assertIsNotNone(manoj_db)
                self.assertIsNotNone(abdul_db)

                # Manoj marked HISTORICAL with reciprocal lineage
                self.assertEqual(manoj_db.status, MemoryStatus.HISTORICAL)
                self.assertEqual(manoj_db.superseded_by_id, abdul_mem_id)
                self.assertIsNotNone(manoj_db.valid_until)

                # Abdul kept ACTIVE with reciprocal lineage
                self.assertEqual(abdul_db.status, MemoryStatus.ACTIVE)
                self.assertEqual(abdul_db.supersedes_id, manoj_mem_id)
                self.assertEqual(abdul_db.object_val, "Abdul")

                # Verify NO standalone active memory such as "friend's name is NOT Manoj" exists
                all_active = (
                    session.query(Memory)
                    .filter(
                        Memory.user_id == test_user,
                        Memory.subject == "friend_name",
                        Memory.status == MemoryStatus.ACTIVE,
                    )
                    .all()
                )
                self.assertEqual(len(all_active), 1)
                self.assertEqual(all_active[0].id, abdul_mem_id)
                self.assertNotIn("not manoj", all_active[0].content.lower())

            # Verification of retrieval
            retrieval_res = retriever.retrieve_adaptive(
                user_id=test_user,
                query="what is my friend's name",
                top_k=5,
            )
            retrieved_items = retrieval_res.get("results", [])
            self.assertTrue(len(retrieved_items) >= 1)
            top_result = retrieved_items[0]
            self.assertEqual(top_result["id"], abdul_mem_id)
            self.assertEqual(top_result["status"], "ACTIVE")
            self.assertIn("Abdul", top_result["content"])

        finally:
            with get_db_session() as session:
                session.query(MemoryConflictLog).filter(MemoryConflictLog.user_id == test_user).delete()
                session.query(Memory).filter(Memory.user_id == test_user).delete()
                session.query(ConversationTurn).filter(ConversationTurn.user_id == test_user).delete()

    def test_regression_multi_friends_and_corrections(self):
        """Comprehensive regression test covering Cases A-F:
        A. 'My friends are Syed and Sameer.' -> Both names remain ACTIVE.
        B. Add Syed after Sameer. -> Sameer remains ACTIVE.
        C. 'My friend's name is Abdul, not Manoj.' -> Abdul ACTIVE, Manoj HISTORICAL.
        D. 'My friend is Syed.' -> Does not automatically deactivate Sameer.
        E. Add a new friend after a correction. -> Existing unrelated friends remain ACTIVE.
        F. Ask 'Who are my friends?' -> Retrieval returns all relevant ACTIVE friend records.
        """
        from app.services.retriever import MemoryRetriever
        retriever = MemoryRetriever()
        test_user = "test_user_multi_friends_cases"
        test_session = "sess_multi_friends_cases"

        try:
            # Case A: "My friends are Syed and Sameer." -> Both names remain ACTIVE
            res_a = self.ingestion_service.ingest_message(
                user_id=test_user,
                session_id=test_session,
                user_message="My friends are Syed and Sameer.",
            )
            created_a = res_a.get("memories_created", [])
            self.assertEqual(len(created_a), 2, "Expected 2 memories for Syed and Sameer")
            with get_db_session() as session:
                active_friends_a = (
                    session.query(Memory)
                    .filter(
                        Memory.user_id == test_user,
                        Memory.subject == "friend_name",
                        Memory.status == MemoryStatus.ACTIVE,
                    )
                    .all()
                )
                self.assertEqual(len(active_friends_a), 2)
                names_a = {m.object_val for m in active_friends_a}
                self.assertEqual(names_a, {"Syed", "Sameer"})
                for m in active_friends_a:
                    self.assertIsNone(m.superseded_by_id)

            # Clean test user for Case B & D: Sequential addition
            with get_db_session() as session:
                session.query(MemoryConflictLog).filter(MemoryConflictLog.user_id == test_user).delete()
                session.query(Memory).filter(Memory.user_id == test_user).delete()

            # Case B: Add Syed after Sameer -> Sameer remains ACTIVE
            res_b1 = self.ingestion_service.ingest_message(
                user_id=test_user,
                session_id=test_session,
                user_message="My friend's name is Sameer.",
            )
            sameer_mem_id = res_b1["memories_created"][0]["id"]

            res_b2 = self.ingestion_service.ingest_message(
                user_id=test_user,
                session_id=test_session,
                user_message="My friend's name is Syed.",
            )
            syed_mem_id = res_b2["memories_created"][0]["id"]
            self.assertNotEqual(sameer_mem_id, syed_mem_id)

            with get_db_session() as session:
                sameer_mem = session.get(Memory, sameer_mem_id)
                syed_mem = session.get(Memory, syed_mem_id)
                self.assertEqual(sameer_mem.status, MemoryStatus.ACTIVE, "Sameer must remain ACTIVE after adding Syed")
                self.assertEqual(syed_mem.status, MemoryStatus.ACTIVE, "Syed must be ACTIVE")
                self.assertIsNone(sameer_mem.superseded_by_id, "Sameer must not be superseded by adding Syed")

            # Case D: "My friend is Syed." -> Re-affirming or adding friend must not deactivate Sameer
            res_d = self.ingestion_service.ingest_message(
                user_id=test_user,
                session_id=test_session,
                user_message="My friend is Syed.",
            )
            with get_db_session() as session:
                sameer_mem_d = session.get(Memory, sameer_mem_id)
                self.assertEqual(sameer_mem_d.status, MemoryStatus.ACTIVE, "Sameer must remain ACTIVE after re-affirming Syed")

            # Clean test user for Case C, E, F: Correction and post-correction addition
            with get_db_session() as session:
                session.query(MemoryConflictLog).filter(MemoryConflictLog.user_id == test_user).delete()
                session.query(Memory).filter(Memory.user_id == test_user).delete()

            # Step 1: Initial friend Manoj
            res_c1 = self.ingestion_service.ingest_message(
                user_id=test_user,
                session_id=test_session,
                user_message="My friend's name is Manoj.",
            )
            manoj_id = res_c1["memories_created"][0]["id"]

            # Case C: "My friend's name is Abdul, not Manoj."
            res_c2 = self.ingestion_service.ingest_message(
                user_id=test_user,
                session_id=test_session,
                user_message="My friend's name is Abdul, not Manoj.",
            )
            abdul_id = res_c2["memories_created"][0]["id"]
            with get_db_session() as session:
                manoj_ref = session.get(Memory, manoj_id)
                abdul_ref = session.get(Memory, abdul_id)
                self.assertEqual(manoj_ref.status, MemoryStatus.HISTORICAL, "Manoj must be HISTORICAL")
                self.assertEqual(abdul_ref.status, MemoryStatus.ACTIVE, "Abdul must be ACTIVE")
                self.assertEqual(manoj_ref.superseded_by_id, abdul_id)
                self.assertEqual(abdul_ref.supersedes_id, manoj_id)

            # Case E: Add a new friend after a correction -> Existing unrelated friends remain ACTIVE
            res_e = self.ingestion_service.ingest_message(
                user_id=test_user,
                session_id=test_session,
                user_message="My friend's name is Sameer.",
            )
            sameer_new_id = res_e["memories_created"][0]["id"]
            with get_db_session() as session:
                abdul_ref = session.get(Memory, abdul_id)
                sameer_ref = session.get(Memory, sameer_new_id)
                manoj_ref = session.get(Memory, manoj_id)
                self.assertEqual(abdul_ref.status, MemoryStatus.ACTIVE, "Abdul must remain ACTIVE after adding Sameer")
                self.assertEqual(sameer_ref.status, MemoryStatus.ACTIVE, "Sameer must be ACTIVE")
                self.assertEqual(manoj_ref.status, MemoryStatus.HISTORICAL, "Manoj must remain HISTORICAL")

            # Case F: Ask "Who are my friends?" -> Retrieval returns all relevant ACTIVE friend records
            retrieval = retriever.retrieve_adaptive(
                user_id=test_user,
                query="Who are my friends?",
                top_k=5,
            )
            results = retrieval.get("results", [])
            active_retrieved_ids = [m["id"] for m in results if m["status"] == "ACTIVE"]
            self.assertIn(abdul_id, active_retrieved_ids, "Retrieval must include Abdul")
            self.assertIn(sameer_new_id, active_retrieved_ids, "Retrieval must include Sameer")

        finally:
            with get_db_session() as session:
                session.query(MemoryConflictLog).filter(MemoryConflictLog.user_id == test_user).delete()
                session.query(Memory).filter(Memory.user_id == test_user).delete()
                session.query(ConversationTurn).filter(ConversationTurn.user_id == test_user).delete()


if __name__ == "__main__":
    unittest.main()
