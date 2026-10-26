import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.config import settings
from app.database import (
    Memory,
    MemoryConflictLog,
    ConversationTurn,
    MemoryStatus,
    MemoryCategory,
    ConflictResolutionType,
    init_db,
    get_db_session,
    get_memory_lineage,
    VectorStoreManager,
)
from app.services import (
    LLMService,
    MemoryIngestionService,
    MemoryRetriever,
)
from app.evaluation.benchmark import run_benchmark_suite

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -------------------------------------------------------------
# Lifespan: DB & FAISS startup initialization
# -------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing database and FAISS index on startup...")
    init_db()
    with get_db_session() as session:
        all_memories = session.query(Memory).all()
        VectorStoreManager.get_instance().rebuild_from_db(all_memories)
    logger.info(f"System ready. Indexed {len(all_memories)} memories in FAISS.")
    yield
    logger.info("Application shutting down.")


app = FastAPI(
    title="Conflict-Aware Adaptive AI Memory System",
    description="REST backend providing synchronized SQLite + FAISS memory extraction, conflict resolution, and adaptive retrieval.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Service Singletons
llm_service = LLMService()
ingestion_service = MemoryIngestionService(llm_service=llm_service)
retriever = MemoryRetriever()


# -------------------------------------------------------------
# Request & Response Schemas
# -------------------------------------------------------------
class ChatRequest(BaseModel):
    user_id: str = Field(default="default_user", description="Unique user identifier.")
    session_id: str = Field(default="default_session", description="Current conversation session ID.")
    message: str = Field(..., description="User message content.")


class RetrieveRequest(BaseModel):
    user_id: str = Field(default="default_user")
    query: str = Field(..., description="Query text to search for.")
    mode: str = Field(default="adaptive", description="'adaptive' or 'basic'")
    top_k: int = Field(default=5, ge=1, le=20)


# -------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------
@app.get("/")
def root():
    return {
        "status": "online",
        "system": "Conflict-Aware Adaptive AI Memory System",
        "docs": "/docs",
    }


@app.get("/api/health")
def health_check():
    v_mgr = VectorStoreManager.get_instance()
    return {
        "status": "healthy",
        "faiss_total_vectors": v_mgr.index.ntotal,
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
        "llm_provider": settings.LLM_PROVIDER,
    }


@app.post("/api/chat")
def chat_endpoint(req: ChatRequest):
    """Ingest message, resolve conflicts, retrieve adaptive memories, and generate response."""
    try:
        # 1. Fetch recent conversation history and pre-register turn for provenance tracking
        recent_history: List[Dict[str, str]] = []
        turn_id: Optional[int] = None
        with get_db_session() as session:
            past_turns = (
                session.query(ConversationTurn)
                .filter(ConversationTurn.user_id == req.user_id, ConversationTurn.session_id == req.session_id)
                .order_by(ConversationTurn.id.desc())
                .limit(4)
                .all()
            )
            for pt in reversed(past_turns):
                recent_history.append({"role": "user", "content": pt.user_message})
                if pt.assistant_response:
                    recent_history.append({"role": "assistant", "content": pt.assistant_response})

            current_turn = ConversationTurn(
                user_id=req.user_id,
                session_id=req.session_id,
                user_message=req.message,
                assistant_response="",
            )
            session.add(current_turn)
            session.flush()
            turn_id = current_turn.id

        # 2. Ingest message & detect/resolve conflicts with pronoun/coreference resolution and provenance
        ingest_result = ingestion_service.ingest_message(
            user_id=req.user_id,
            session_id=req.session_id,
            user_message=req.message,
            recent_turns=recent_history,
            source_turn_id=turn_id,
        )

        # 3. Retrieve adaptive context
        adaptive_result = retriever.retrieve_adaptive(
            user_id=req.user_id,
            query=req.message,
            top_k=5,
        )

        # 4. Generate context-aware grounded response
        response_text = llm_service.generate_response(
            query=req.message,
            retrieved_memories=adaptive_result["results"],
            conflict_history=ingest_result["conflicts_detected"],
            ingestion_summary=ingest_result,
            recent_turns=recent_history,
        )

        # 5. Update conversation turn with assistant response
        if turn_id:
            with get_db_session() as session:
                turn = session.get(ConversationTurn, turn_id)
                if turn:
                    turn.assistant_response = response_text

        return {
            "response": response_text,
            "session_id": req.session_id,
            "extracted_facts": ingest_result["extracted_facts"],
            "conflicts_detected": ingest_result["conflicts_detected"],
            "retrieved_memories": adaptive_result["results"],
            "actions_taken": ingest_result["actions_taken"],
        }
    except Exception as e:
        logger.exception("Error in /api/chat:")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/memories")
def get_memories(
    user_id: str = Query("default_user"),
    status: Optional[str] = Query(None, description="ACTIVE, HISTORICAL, TEMPORARY, ARCHIVED"),
    category: Optional[str] = Query(None, description="SHORT_TERM, LONG_TERM, EPISODIC, SEMANTIC"),
):
    """Retrieve memory records with optional status and category filters."""
    with get_db_session() as session:
        query = session.query(Memory).filter(Memory.user_id == user_id)
        if status:
            try:
                mem_status = MemoryStatus(status.upper())
                query = query.filter(Memory.status == mem_status)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid status '{status}'")

        if category:
            try:
                mem_cat = MemoryCategory(category.upper())
                query = query.filter(Memory.category == mem_cat)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid category '{category}'")

        records = query.order_by(Memory.id.desc()).all()

        return [
            {
                "id": m.id,
                "user_id": m.user_id,
                "session_id": m.session_id,
                "content": m.content,
                "subject": m.subject,
                "predicate": m.predicate,
                "object_val": m.object_val,
                "status": m.status.value,
                "category": m.category.value,
                "importance": m.importance,
                "confidence": m.confidence,
                "access_count": m.access_count,
                "created_at": m.created_at.isoformat(),
                "last_accessed_at": m.last_accessed_at.isoformat(),
                "valid_until": m.valid_until.isoformat() if m.valid_until else None,
                "supersedes_id": m.supersedes_id,
                "superseded_by_id": m.superseded_by_id,
            }
            for m in records
        ]


@app.get("/api/memories/{memory_id}/lineage")
def get_lineage(memory_id: int):
    """Get full chronological version history (oldest HISTORICAL to newest ACTIVE)."""
    with get_db_session() as session:
        lineage = get_memory_lineage(memory_id, db=session)
        if not lineage:
            raise HTTPException(status_code=404, detail=f"Memory id={memory_id} not found.")

        return [
            {
                "id": m.id,
                "content": m.content,
                "subject": m.subject,
                "status": m.status.value,
                "category": m.category.value,
                "valid_from": m.valid_from.isoformat(),
                "valid_until": m.valid_until.isoformat() if m.valid_until else None,
                "supersedes_id": m.supersedes_id,
                "superseded_by_id": m.superseded_by_id,
            }
            for m in lineage
        ]


@app.get("/api/conflicts")
def get_conflicts(user_id: Optional[str] = Query(None)):
    """Retrieve audit trail of detected memory conflicts and resolutions."""
    with get_db_session() as session:
        query = session.query(MemoryConflictLog)
        if user_id:
            query = query.filter(MemoryConflictLog.user_id == user_id)
        logs = query.order_by(MemoryConflictLog.id.desc()).all()

        results = []
        for c in logs:
            old_mem = session.get(Memory, c.old_memory_id) if c.old_memory_id else None
            new_mem = session.get(Memory, c.new_memory_id) if c.new_memory_id else None
            results.append({
                "id": c.id,
                "user_id": c.user_id,
                "old_memory_id": c.old_memory_id,
                "old_memory_content": old_mem.content if old_mem else None,
                "new_memory_id": c.new_memory_id,
                "new_memory_content": new_mem.content if new_mem else None,
                "similarity_score": round(c.similarity_score, 4),
                "resolution_type": c.resolution_type.value,
                "reasoning": c.reasoning,
                "detected_at": c.detected_at.isoformat(),
            })
        return results


@app.post("/api/retrieve")
def retrieve_endpoint(req: RetrieveRequest):
    """Execute either pure basic FAISS vector search or adaptive composite retrieval."""
    if req.mode.lower() == "basic":
        return retriever.retrieve_basic(user_id=req.user_id, query=req.query, top_k=req.top_k)
    else:
        return retriever.retrieve_adaptive(user_id=req.user_id, query=req.query, top_k=req.top_k)


@app.post("/api/evaluate")
def evaluate_endpoint():
    """Trigger automated benchmarking suite and return precision, accuracy, and latency metrics."""
    try:
        report = run_benchmark_suite()
        return report
    except Exception as e:
        logger.exception("Error during benchmark execution:")
        raise HTTPException(status_code=500, detail=str(e))
