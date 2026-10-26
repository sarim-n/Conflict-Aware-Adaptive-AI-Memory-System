import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

# Add project root to sys.path so app packages import cleanly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

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

# -------------------------------------------------------------
# Page Configuration & Modern Theme Styling
# -------------------------------------------------------------
st.set_page_config(
    page_title="Conflict-Aware Adaptive AI Memory System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for rich aesthetics & glassmorphic badges
st.markdown(
    """
    <style>
    .main { background-color: #0b0f19; }
    .stMetric {
        background: rgba(255, 255, 255, 0.04);
        padding: 16px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .badge-active {
        background-color: #059669;
        color: white;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-historical {
        background-color: #dc2626;
        color: white;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-temporary {
        background-color: #d97706;
        color: white;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-perm-update {
        background-color: #7c3aed;
        color: white;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .lineage-node {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 8px;
        padding: 12px;
        margin: 6px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize singletons in st.cache_resource
@st.cache_resource
def get_services():
    init_db()
    llm = LLMService()
    ingest = MemoryIngestionService(llm_service=llm)
    ret = MemoryRetriever()
    return llm, ingest, ret


llm_service, ingestion_service, retriever = get_services()

# Session State Initialization
if "user_id" not in st.session_state:
    st.session_state.user_id = "default_user"
if "session_id" not in st.session_state:
    st.session_state.session_id = f"sess_{int(time.time())}"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "latest_inspection" not in st.session_state:
    st.session_state.latest_inspection = None
if "benchmark_results" not in st.session_state:
    st.session_state.benchmark_results = None


# -------------------------------------------------------------
# Sidebar: Settings & Controls
# -------------------------------------------------------------
with st.sidebar:
    st.title("🧠 AI Memory System")
    st.caption("Conflict-Aware Adaptive Memory Layer")

    st.markdown("---")
    st.session_state.user_id = st.text_input("Active User ID", value=st.session_state.user_id)
    st.session_state.session_id = st.text_input("Session ID", value=st.session_state.session_id)

    st.markdown("### ⚙️ System Config")
    st.markdown(f"**LLM Provider:** `{settings.LLM_PROVIDER.upper()}`")
    st.markdown(f"**Embedding Model:** `{settings.EMBEDDING_MODEL_NAME.split('/')[-1]}`")
    st.markdown(f"**Conflict Threshold:** `{settings.CONFLICT_SIMILARITY_THRESHOLD}`")

    st.markdown("### ⚖️ Adaptive Weights")
    st.caption(
        f"• Similarity: {settings.WEIGHT_SIMILARITY * 100:.0f}%\n"
        f"• Importance: {settings.WEIGHT_IMPORTANCE * 100:.0f}%\n"
        f"• Recency: {settings.WEIGHT_RECENCY * 100:.0f}%\n"
        f"• Confidence: {settings.WEIGHT_CONFIDENCE * 100:.0f}%"
    )

    if st.button("🔄 Sync & Rebuild FAISS"):
        with get_db_session() as session:
            all_m = session.query(Memory).all()
            VectorStoreManager.get_instance().rebuild_from_db(all_m)
        st.success(f"Synchronized {len(all_m)} memories in FAISS!")

    if st.button("🧹 Clear User Chat"):
        st.session_state.chat_history = []
        st.session_state.latest_inspection = None
        st.rerun()


# -------------------------------------------------------------
# Main Application Tabs
# -------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "💬 Live Chat & Inspector",
    "🧠 Memory Stash & Lineage",
    "⚖️ Conflict Audit Log",
    "📊 Benchmarks & System Status",
])

# =============================================================
# TAB 1: LIVE CHAT & MEMORY INSPECTOR
# =============================================================
with tab1:
    col_chat, col_inspect = st.columns([1.1, 0.9])

    with col_chat:
        st.subheader("💬 Chat with Adaptive Memory")
        st.caption("Facts you mention are atomically ingested, conflict-checked, and remembered.")

        # Quick Starter Prompts
        st.markdown("**Quick Test Prompts:**")
        q_cols = st.columns(4)
        if q_cols[0].button("1. Build RAG Engine"):
            st.session_state.pending_prompt = "we are going to build a rag chatbot engine"
        if q_cols[1].button("2. Embeddings Choice"):
            st.session_state.pending_prompt = "we are gonna use sentence transformer for vector embeddings"
        if q_cols[2].button("3. Retract ('won't use it')"):
            st.session_state.pending_prompt = "no we wont use it"
        if q_cols[3].button("4. General Trivia"):
            st.session_state.pending_prompt = "how many countries are there in the world"

        # Display Message History
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat Input Box
        prompt = st.chat_input("Tell me a fact or ask a question...")
        if "pending_prompt" in st.session_state and st.session_state.pending_prompt:
            prompt = st.session_state.pending_prompt
            del st.session_state.pending_prompt

        if prompt:
            # 1. Append user message
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            # Pre-register turn to track provenance
            turn_id: Optional[int] = None
            with get_db_session() as session:
                turn = ConversationTurn(
                    user_id=st.session_state.user_id,
                    session_id=st.session_state.session_id,
                    user_message=prompt,
                    assistant_response="",
                )
                session.add(turn)
                session.flush()
                turn_id = turn.id

            # 2. Ingest message & resolve conflicts with recent turns and provenance
            past_chat = st.session_state.chat_history[:-1]
            ingest_res = ingestion_service.ingest_message(
                user_id=st.session_state.user_id,
                session_id=st.session_state.session_id,
                user_message=prompt,
                recent_turns=past_chat,
                source_turn_id=turn_id,
            )

            # 3. Retrieve adaptive context
            adaptive_res = retriever.retrieve_adaptive(
                user_id=st.session_state.user_id,
                query=prompt,
                top_k=5,
            )

            # 4. Generate response with conversation history & ingestion summary
            response = llm_service.generate_response(
                query=prompt,
                retrieved_memories=adaptive_res["results"],
                conflict_history=ingest_res["conflicts_detected"],
                ingestion_summary=ingest_res,
                recent_turns=past_chat,
            )

            # 5. Save assistant response to session state & DB
            st.session_state.chat_history.append({"role": "assistant", "content": response})

            if turn_id:
                with get_db_session() as session:
                    turn_to_update = session.get(ConversationTurn, turn_id)
                    if turn_to_update:
                        turn_to_update.assistant_response = response

            # 6. Save inspection payload
            st.session_state.latest_inspection = {
                "user_message": prompt,
                "ingest_res": ingest_res,
                "adaptive_res": adaptive_res,
                "response": response,
            }
            st.rerun()

    with col_inspect:
        st.subheader("🔍 Real-time Memory Inspector")
        insp = st.session_state.latest_inspection

        if insp is None:
            st.info("Send a message in the chat to inspect live memory extraction, conflict resolution, and adaptive scoring.")
        else:
            with st.expander("📌 Extracted Facts", expanded=True):
                facts = insp["ingest_res"].get("extracted_facts", [])
                if not facts:
                    st.write("No new declarative facts extracted from this message (e.g., pure question).")
                else:
                    for f in facts:
                        temp_tag = " [TEMPORARY]" if f.get("is_temporary") else ""
                        st.markdown(
                            f"• **`{f.get('subject')}`** `{f.get('predicate')}` **`{f.get('object_val')}`**{temp_tag}\n"
                            f"  *Canonical:* \"{f.get('content')}\" (Importance: {f.get('importance')}, Conf: {f.get('confidence')})"
                        )

            with st.expander("⚖️ Conflicts Detected & Actions", expanded=True):
                actions = insp["ingest_res"].get("actions_taken", [])
                conflicts = insp["ingest_res"].get("conflicts_detected", [])
                for a in actions:
                    st.markdown(f"• **Action:** `{a}`")
                for c in conflicts:
                    st.markdown(
                        f"• **Conflict Log #{c.get('id')}:** Old #{c.get('old_memory_id')} $\\rightarrow$ New #{c.get('new_memory_id')}\n"
                        f"  **Resolution:** `{c.get('resolution_type')}` | **Similarity:** `{c.get('similarity_score')}`\n"
                        f"  **Reasoning:** *{c.get('reasoning')}*"
                    )

            with st.expander("🎯 Retrieved Memories & Score Breakdown", expanded=True):
                ret_results = insp["adaptive_res"].get("results", [])
                is_hist = insp["adaptive_res"].get("historical_intent", False)
                st.caption(f"Historical intent detected: **{is_hist}** | Latency: **{insp['adaptive_res'].get('latency_ms')} ms**")

                if not ret_results:
                    st.write("No memories retrieved for this query.")
                else:
                    for idx, m in enumerate(ret_results, 1):
                        status_class = f"badge-{m['status'].lower()}"
                        st.markdown(
                            f"**{idx}. [{m['status']}]** {m['content']}  \n"
                            f"• *Similarity:* `{m.get('similarity')}` | *Recency:* `{m.get('recency')}` | "
                            f"*Gate $\\Psi$:* `{m.get('status_multiplier')}` $\\rightarrow$ **Adaptive Score: `{m.get('adaptive_score')}`**"
                        )
                        if m.get("lineage_chain"):
                            st.caption(f"↳ Superseded Lineage: {[x['content'] for x in m['lineage_chain']]}")
                        st.markdown("---")


# =============================================================
# TAB 2: MEMORY STASH & LINEAGE EXPLORER
# =============================================================
with tab2:
    st.subheader("🧠 Memory Stash & Version Lineage")
    st.caption("Inspect all stored atomic memories in SQLite and traverse chronological fact lineages.")

    f_col1, f_col2, f_col3 = st.columns([1, 1, 2])
    status_filter = f_col1.selectbox("Filter by Status", ["ALL", "ACTIVE", "HISTORICAL", "TEMPORARY", "ARCHIVED"])
    cat_filter = f_col2.selectbox("Filter by Category", ["ALL", "LONG_TERM", "SHORT_TERM", "SEMANTIC", "EPISODIC"])
    search_term = f_col3.text_input("Search content or subject...", "")

    with get_db_session() as session:
        q = session.query(Memory).filter(Memory.user_id == st.session_state.user_id)
        if status_filter != "ALL":
            q = q.filter(Memory.status == MemoryStatus(status_filter))
        if cat_filter != "ALL":
            q = q.filter(Memory.category == MemoryCategory(cat_filter))
        if search_term:
            q = q.filter(Memory.content.ilike(f"%{search_term}%") | Memory.subject.ilike(f"%{search_term}%"))

        memories = q.order_by(Memory.id.desc()).all()

    if not memories:
        st.info("No memories found matching the filters.")
    else:
        # Table of memories
        df_mem = pd.DataFrame([
            {
                "ID": m.id,
                "Status": m.status.value,
                "Category": m.category.value,
                "Subject": m.subject,
                "Predicate": m.predicate,
                "Object": m.object_val,
                "Content": m.content,
                "Importance": m.importance,
                "Confidence": m.confidence,
                "Supersedes": m.supersedes_id,
                "Superseded By": m.superseded_by_id,
                "Created At": m.created_at.strftime("%Y-%m-%d %H:%M:%S") if m.created_at else "",
            }
            for m in memories
        ])
        st.dataframe(df_mem, width="stretch", height=280)

        # Lineage Explorer Section
        st.markdown("### 🧬 Lineage Graph Explorer")
        st.caption("Select a memory ID to walk its bidirectional version lineage (oldest HISTORICAL to newest ACTIVE).")

        selected_id = st.selectbox(
            "Select Memory ID to explore lineage:",
            options=[m.id for m in memories],
            format_func=lambda x: f"ID #{x} — {next((m.content for m in memories if m.id == x), '')[:60]}",
        )

        if selected_id:
            with get_db_session() as session:
                chain = get_memory_lineage(selected_id, db=session)

            if len(chain) <= 1:
                st.info(f"Memory #{selected_id} is a standalone fact with no superseding lineage.")
            else:
                st.markdown(f"**Lineage Path ({len(chain)} versions):**")
                cols = st.columns(len(chain))
                for i, node in enumerate(chain):
                    with cols[i]:
                        status_color = "#dc2626" if node.status.value == "HISTORICAL" else "#059669"
                        valid_info = f"Until: {node.valid_until.strftime('%H:%M:%S')}" if node.valid_until else "Valid: CURRENT"
                        st.markdown(
                            f"""
                            <div class="lineage-node">
                                <span style="background:{status_color};color:white;padding:2px 6px;border-radius:6px;font-size:0.75rem;">
                                    {node.status.value} (v{i+1})
                                </span>
                                <h5 style="margin:8px 0 4px 0;">ID #{node.id}</h5>
                                <p style="font-size:0.85rem;margin:0;">{node.content}</p>
                                <hr style="margin:6px 0;opacity:0.2;"/>
                                <small style="color:#94a3b8;">{valid_info}</small>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        if i < len(chain) - 1:
                            st.write("➡️ *Superseded by*")


# =============================================================
# TAB 3: CONFLICT AUDIT LOG
# =============================================================
with tab3:
    st.subheader("⚖️ Conflict Resolution Audit Log")
    st.caption("Auditable log of detected knowledge contradictions, resolution policies, and LLM reasoning.")

    with get_db_session() as session:
        conflicts = (
            session.query(MemoryConflictLog)
            .filter(MemoryConflictLog.user_id == st.session_state.user_id)
            .order_by(MemoryConflictLog.id.desc())
            .all()
        )

    if not conflicts:
        st.info("No conflicts recorded for this user yet.")
    else:
        conflict_data = []
        for c in conflicts:
            with get_db_session() as session:
                old_m = session.get(Memory, c.old_memory_id) if c.old_memory_id else None
                new_m = session.get(Memory, c.new_memory_id) if c.new_memory_id else None
            conflict_data.append({
                "Conflict ID": c.id,
                "Resolution Type": c.resolution_type.value,
                "Similarity Score": f"{c.similarity_score:.4f}",
                "Old Memory (Superseded)": f"#{c.old_memory_id}: {old_m.content}" if old_m else "N/A",
                "New Memory (Active)": f"#{c.new_memory_id}: {new_m.content}" if new_m else "N/A",
                "Reasoning": c.reasoning,
                "Detected At": c.detected_at.strftime("%Y-%m-%d %H:%M:%S") if c.detected_at else "",
            })

        df_conflicts = pd.DataFrame(conflict_data)
        st.dataframe(df_conflicts, width="stretch", height=350)


# =============================================================
# TAB 4: BENCHMARKS & SYSTEM STATUS
# =============================================================
with tab4:
    st.subheader("📊 System Observability & Evaluation Benchmarks")
    st.caption("Benchmarking Pure FAISS Vector Search vs. Adaptive Status-Gated Composite Retrieval.")

    # High-level Metric Cards
    with get_db_session() as session:
        total_m = session.query(Memory).count()
        active_m = session.query(Memory).filter(Memory.status == MemoryStatus.ACTIVE).count()
        hist_m = session.query(Memory).filter(Memory.status == MemoryStatus.HISTORICAL).count()
        temp_m = session.query(Memory).filter(Memory.status == MemoryStatus.TEMPORARY).count()
        total_conf = session.query(MemoryConflictLog).count()

    m_cols = st.columns(5)
    m_cols[0].metric("Total Memories", total_m)
    m_cols[1].metric("Active Memories", active_m)
    m_cols[2].metric("Historical (Outdated)", hist_m)
    m_cols[3].metric("Temporary Overrides", temp_m)
    m_cols[4].metric("Conflicts Resolved", total_conf)

    st.markdown("---")

    # Benchmark Execution Button
    b_col1, b_col2 = st.columns([1, 3])
    if b_col1.button("🚀 Run Live Evaluation Benchmark", type="primary"):
        with st.spinner("Executing 5 standardized evaluation scenarios across SQLite & FAISS..."):
            st.session_state.benchmark_results = run_benchmark_suite()
        st.success("Benchmark completed successfully!")

    # Display Benchmark Results
    bench = st.session_state.benchmark_results
    if bench:
        metrics = bench["metrics"]

        st.markdown("### 🏆 Benchmark Comparison Metrics")
        b_metrics = st.columns(4)
        b_metrics[0].metric(
            "Adaptive Precision @ 1",
            f"{metrics['adaptive_retrieval_precision_pct']}%",
            f"+{metrics['precision_improvement_pct']}% vs Basic",
        )
        b_metrics[1].metric(
            "Basic Vector Precision @ 1",
            f"{metrics['basic_retrieval_precision_pct']}%",
        )
        b_metrics[2].metric(
            "Contradiction Detection F1",
            f"{metrics['contradiction_detection_acc_pct']}%",
        )
        b_metrics[3].metric(
            "Response Accuracy",
            f"{metrics['response_accuracy_pct']}%",
        )

        # Plotly Charts
        ch_col1, ch_col2 = st.columns(2)

        with ch_col1:
            fig_prec = go.Figure(data=[
                go.Bar(
                    name="Retrieval Precision @ 1 (%)",
                    x=["Basic Vector Search", "Adaptive Retrieval"],
                    y=[metrics["basic_retrieval_precision_pct"], metrics["adaptive_retrieval_precision_pct"]],
                    marker_color=["#ef4444", "#10b981"],
                    text=[f"{metrics['basic_retrieval_precision_pct']}%", f"{metrics['adaptive_retrieval_precision_pct']}%"],
                    textposition="auto",
                )
            ])
            fig_prec.update_layout(
                title="Retrieval Precision: Basic vs Adaptive",
                template="plotly_dark",
                yaxis=dict(range=[0, 110]),
                height=320,
            )
            st.plotly_chart(fig_prec, width="stretch")

        with ch_col2:
            fig_lat = go.Figure(data=[
                go.Bar(
                    name="Average Latency (ms)",
                    x=["Basic Vector Search", "Adaptive Retrieval"],
                    y=[metrics["avg_basic_latency_ms"], metrics["avg_adaptive_latency_ms"]],
                    marker_color=["#3b82f6", "#8b5cf6"],
                    text=[f"{metrics['avg_basic_latency_ms']} ms", f"{metrics['avg_adaptive_latency_ms']} ms"],
                    textposition="auto",
                )
            ])
            fig_lat.update_layout(
                title="Retrieval Latency Comparison",
                template="plotly_dark",
                yaxis_title="Milliseconds",
                height=320,
            )
            st.plotly_chart(fig_lat, width="stretch")

        # Scenario Breakdown Table
        st.markdown("### 📋 Scenario-by-Scenario Evaluation Results")
        scenarios_df = pd.DataFrame(bench["scenarios"])
        st.dataframe(scenarios_df, width="stretch")
    else:
        st.info("Click 'Run Live Evaluation Benchmark' to execute the test suite and visualize performance metrics.")
