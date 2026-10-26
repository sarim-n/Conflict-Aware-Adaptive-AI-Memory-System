import unittest
from app.database import (
    Memory,
    MemoryCategory,
    MemoryStatus,
    MemoryConflictLog,
    ConversationTurn,
    init_db,
    get_db_session,
)
from app.services import (
    LLMService,
    MemoryIngestionService,
    MemoryRetriever,
)


class TestMultiValuedEntities(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()
        cls.llm_service = LLMService()
        cls.ingestion_service = MemoryIngestionService(llm_service=cls.llm_service)
        cls.retriever = MemoryRetriever()

    def test_add_friend_persists_across_sessions(self):
        user_id = "test_friend_user"
        session_1 = "sess_add_friend"
        session_2 = "sess_new_fresh_chat"

        try:
            # Step 1: User in session 1 asks to add a friend
            res1 = self.ingestion_service.ingest_message(
                user_id=user_id,
                session_id=session_1,
                user_message="add a new friend named John",
            )
            created = res1.get("memories_created", [])
            self.assertTrue(len(created) >= 1, "Expected memory to be created for friend John")
            john_id = created[0]["id"]

            with get_db_session() as session:
                mem = session.get(Memory, john_id)
                self.assertIsNotNone(mem)
                self.assertEqual(mem.status, MemoryStatus.ACTIVE)
                self.assertEqual(mem.object_val, "John")

            # Step 2: In a brand-new session (empty chat history), ask who friends are
            retrieval = self.retriever.retrieve_adaptive(
                user_id=user_id,
                query="who are my friends?",
                top_k=5,
            )
            results = retrieval.get("results", [])
            retrieved_ids = [m["id"] for m in results if m["status"] == "ACTIVE"]
            self.assertIn(john_id, retrieved_ids, "Fresh chat session must retrieve John from persistent storage")

            # Step 3: Add a second friend - both must remain ACTIVE
            res2 = self.ingestion_service.ingest_message(
                user_id=user_id,
                session_id=session_1,
                user_message="can you add a new friend named Alice",
            )
            created2 = res2.get("memories_created", [])
            self.assertTrue(len(created2) >= 1, "Expected memory to be created for friend Alice")
            alice_id = created2[0]["id"]

            with get_db_session() as session:
                m_john = session.get(Memory, john_id)
                m_alice = session.get(Memory, alice_id)
                self.assertEqual(m_john.status, MemoryStatus.ACTIVE, "John must remain ACTIVE")
                self.assertEqual(m_alice.status, MemoryStatus.ACTIVE, "Alice must be ACTIVE")

            # Step 4: Add colleague & hobby - both must persist as COMPLEMENTARY
            res_col = self.ingestion_service.ingest_message(
                user_id=user_id,
                session_id=session_1,
                user_message="add a new colleague Sarah",
            )
            self.assertTrue(len(res_col.get("memories_created", [])) >= 1)

            res_hobby = self.ingestion_service.ingest_message(
                user_id=user_id,
                session_id=session_1,
                user_message="my hobby is photography",
            )
            self.assertTrue(len(res_hobby.get("memories_created", [])) >= 1)

            # Fresh session retrieval for colleague & hobby
            ret_col = self.retriever.retrieve_adaptive(user_id=user_id, query="who is my colleague", top_k=5)
            self.assertTrue(any("Sarah" in m.get("content", "") or m.get("object_val") == "Sarah" for m in ret_col["results"]))

            ret_hobby = self.retriever.retrieve_adaptive(user_id=user_id, query="what is my hobby", top_k=5)
            self.assertTrue(any("photography" in m.get("content", "").lower() or m.get("object_val") == "photography" for m in ret_hobby["results"]))

        finally:
            with get_db_session() as session:
                session.query(MemoryConflictLog).filter(MemoryConflictLog.user_id == user_id).delete()
                session.query(Memory).filter(Memory.user_id == user_id).delete()
                session.query(ConversationTurn).filter(ConversationTurn.user_id == user_id).delete()


if __name__ == "__main__":
    unittest.main()
