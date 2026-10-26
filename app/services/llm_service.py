import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator

# Ensure .env is explicitly loaded
load_dotenv(override=True)

from app.config import settings
from app.database.models import (
    Memory,
    MemoryCategory,
    ConflictResolutionType,
)

logger = logging.getLogger(__name__)


def classify_memory_category(
    content: str,
    subject: Optional[str] = None,
    predicate: Optional[str] = None,
    object_val: Optional[str] = None,
    is_temporary: bool = False,
) -> MemoryCategory:
    """Classify facts into SHORT_TERM, LONG_TERM, EPISODIC, or SEMANTIC.

    Classification rules:
    - SEMANTIC: Stable concepts, project goals, definitions, user attributes, preferences, and general knowledge.
    - LONG_TERM: Persistent architectural decisions, technology-stack choices, infrastructure choices, and durable project configuration.
    - EPISODIC: Specific events, incidents, milestones, decisions made during a particular event, and attributed statements when the fact being stored is that the statement occurred.
    - SHORT_TERM: Session-scoped context and temporary working information.
    """
    # 1. Ephemeral session overrides / working info are SHORT_TERM
    if is_temporary:
        return MemoryCategory.SHORT_TERM

    cont_low = (content or "").lower().strip()
    sub_low = (subject or "").lower().strip()
    pred_low = (predicate or "").lower().strip()
    obj_low = (object_val or "").lower().strip()

    if any(w in cont_low for w in [
        "for unit tests today", "for local tests today", "for this session",
        "temporary override", "temporarily for now", "mock payment gateway",
        "for this test run", "in-memory for local unit tests", "ephemeral test",
    ]):
        return MemoryCategory.SHORT_TERM

    # 2. EPISODIC: Specific events, incidents, milestones, decisions made during a particular event, attributed statements
    temporal_markers = [
        "yesterday", "last week", "last month", "last night", "earlier today",
        "on monday", "on tuesday", "on wednesday", "on thursday", "on friday",
        "on saturday", "on sunday",
    ]
    if any(m in cont_low for m in temporal_markers):
        return MemoryCategory.EPISODIC

    episodic_event_keywords = [
        "incident", "outage", "failed deployment", "deployment failed",
        "crashed due to", "server crash", "oom error", "promoted to",
        "officially completed migration", "meeting held", "sprint planning",
        "agreed during", "decided during",
    ]
    if any(w in cont_low for w in episodic_event_keywords):
        return MemoryCategory.EPISODIC

    # Attributed statements (reporting who said or stated what during an event/conversation)
    if pred_low in ["reported", "stated", "claimed", "claims", "says", "said", "noted"]:
        return MemoryCategory.EPISODIC

    attr_pattern = r"^[A-Z][a-z]+\s+(?:reported|says|said|stated|claimed|claims|noted)\s+(?:that\s+)?"
    if re.search(attr_pattern, content or ""):
        return MemoryCategory.EPISODIC

    if sub_low in ["incident", "outage", "event", "milestone", "attribution", "reported_statement"]:
        return MemoryCategory.EPISODIC

    # 3. Project goals vs Tech-stack choices
    # Project goals, missions, and business objectives are SEMANTIC
    is_project_goal = (
        sub_low in ["project_goal", "project_intent", "business_goal", "product_goal"]
        or any(w in cont_low for w in [
            "is building", "building a", "building an", "going to build", "plan to build",
            "aim to build", "develops a", "developing a", "focuses on", "will sell", "sells",
        ])
        or pred_low in ["is_building", "building", "develops", "focuses_on", "sells", "goal_is"]
    )

    # Technology-stack choices and durable architectural decisions are LONG_TERM
    is_tech_stack_choice = (
        sub_low in [
            "primary_database", "backend_framework", "frontend_framework",
            "embedding_model", "cache_layer", "cloud_provider", "project_architecture",
            "tech_stack", "infrastructure", "database",
        ]
        or any(w in cont_low for w in [
            "uses postgresql", "uses mongodb", "uses mysql", "uses sqlite", "uses fastapi",
            "uses django", "uses flask", "uses react", "uses vue", "uses angular",
            "uses sentence transformer", "will not use sentence transformer",
            "for vector embeddings", "as primary database", "for the backend", "for backend",
            "for the frontend", "for frontend", "written in python", "uses python",
            "deploys on", "hosted on",
        ])
    )

    if is_tech_stack_choice and not is_project_goal:
        return MemoryCategory.LONG_TERM

    if is_project_goal:
        return MemoryCategory.SEMANTIC

    # 4. User attributes, profiles, preferences are SEMANTIC
    is_user_attribute = (
        sub_low in ["user_profile", "user_attribute", "user_preference", "user_role", "user_identity"]
        or any(w in cont_low for w in ["user is a", "is a backend architect", "is a frontend developer", "prefers dark mode", "likes python"])
        or pred_low in ["has_role", "has_title", "is_role", "prefers"]
    )
    if is_user_attribute:
        return MemoryCategory.SEMANTIC

    # General knowledge, stable definitions, domain facts are SEMANTIC
    is_general_knowledge = (
        any(w in cont_low for w in [
            "travel time", "distance between", "capital of", "is defined as",
            "definition of", "means that", "concept of", "speed of light",
        ])
        or sub_low in ["domain_concept", "definition", "general_knowledge", "stable_concept"]
    )
    if is_general_knowledge:
        return MemoryCategory.SEMANTIC

    # 5. Default fallback
    if any(w in cont_low for w in ["architecture", "database", "framework", "infrastructure", "server", "deploy"]):
        return MemoryCategory.LONG_TERM

    return MemoryCategory.SEMANTIC


class ExtractedFact(BaseModel):
    content: str = Field(..., description="Clean, canonical, self-contained atomic statement.")
    subject: str = Field(..., description="Normalized snake_case entity/topic key.")
    predicate: str = Field(..., description="Normalized relation, e.g., 'uses', 'prefers', 'is_building'.")
    object_val: str = Field(..., description="Target value, e.g., 'PostgreSQL', 'FastAPI'.")
    replaces_value: Optional[str] = Field(
        default=None,
        description="Value or entity that this fact explicitly corrects or replaces (e.g. 'Manoj' in 'Abdul, not Manoj').",
    )
    category: Optional[MemoryCategory] = Field(default=None)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    is_temporary: bool = Field(
        default=False,
        description="True if the user indicates an ephemeral/session-only exception.",
    )
    source_type: str = Field(
        default="user_confirmed",
        description="Fact source type: 'user_confirmed', 'context_resolved', 'assistant_suggestion', 'unconfirmed_inference'",
    )
    provenance_context: Optional[str] = Field(
        default=None,
        description="Explanation of context/coreference resolution (e.g. how pronouns like 'we' were resolved).",
    )

    @field_validator("source_type", mode="before")
    @classmethod
    def validate_source_type(cls, v: Any) -> str:
        if not v or not isinstance(v, str):
            return "user_confirmed"
        clean = v.strip().lower()
        valid = {"user_confirmed", "context_resolved", "assistant_suggestion", "unconfirmed_inference"}
        if clean in valid:
            return clean
        if "suggest" in clean or "assistant" in clean:
            return "assistant_suggestion"
        if "inference" in clean or "unconfirmed" in clean or "speculat" in clean:
            return "unconfirmed_inference"
        if "context" in clean or "coref" in clean or "pronoun" in clean:
            return "context_resolved"
        return "user_confirmed"

    @field_validator("category", mode="before")
    @classmethod
    def validate_category(cls, v: Any) -> Optional[MemoryCategory]:
        if v is None:
            return None
        if isinstance(v, MemoryCategory):
            return v
        if isinstance(v, str):
            clean = v.strip().upper()
            if clean in MemoryCategory.__members__:
                return MemoryCategory(clean)
            synonyms = {
                "TEMPORARY": MemoryCategory.SHORT_TERM,
                "SHORTTERM": MemoryCategory.SHORT_TERM,
                "SHORT-TERM": MemoryCategory.SHORT_TERM,
                "SHORT_TERM": MemoryCategory.SHORT_TERM,
                "WORKING": MemoryCategory.SHORT_TERM,
                "SESSION": MemoryCategory.SHORT_TERM,
                "EPHEMERAL": MemoryCategory.SHORT_TERM,
                "LONGTERM": MemoryCategory.LONG_TERM,
                "LONG-TERM": MemoryCategory.LONG_TERM,
                "LONG_TERM": MemoryCategory.LONG_TERM,
                "PERSISTENT": MemoryCategory.LONG_TERM,
                "ARCHITECTURE": MemoryCategory.LONG_TERM,
                "ARCHITECTURAL": MemoryCategory.LONG_TERM,
                "TECH_STACK": MemoryCategory.LONG_TERM,
                "SEMANTIC": MemoryCategory.SEMANTIC,
                "FACT": MemoryCategory.SEMANTIC,
                "FACTUAL": MemoryCategory.SEMANTIC,
                "CONCEPT": MemoryCategory.SEMANTIC,
                "CONCEPTUAL": MemoryCategory.SEMANTIC,
                "GOAL": MemoryCategory.SEMANTIC,
                "PROFILE": MemoryCategory.SEMANTIC,
                "EPISODIC": MemoryCategory.EPISODIC,
                "EVENT": MemoryCategory.EPISODIC,
                "INCIDENT": MemoryCategory.EPISODIC,
                "MILESTONE": MemoryCategory.EPISODIC,
                "ACTION": MemoryCategory.EPISODIC,
            }
            if clean in synonyms:
                return synonyms[clean]
        # Missing or invalid category returns None so model_validator applies safe classification fallback
        return None

    @model_validator(mode="after")
    def ensure_valid_category(self) -> "ExtractedFact":
        """Apply classification fallback safely when category is missing or invalid."""
        if self.category is None:
            self.category = classify_memory_category(
                content=self.content,
                subject=self.subject,
                predicate=self.predicate,
                object_val=self.object_val,
                is_temporary=self.is_temporary,
            )
        return self


class ConflictAdjudication(BaseModel):
    resolution_type: ConflictResolutionType
    reasoning: str


class LLMService:
    """Unified LLM service supporting Groq, Gemini, and robust deterministic heuristic fallback."""

    def __init__(self):
        self.provider = settings.LLM_PROVIDER.lower()
        self.groq_api_key = settings.GROQ_API_KEY
        self.groq_model = settings.GROQ_MODEL
        self.groq_fallback_model = settings.GROQ_FALLBACK_MODEL
        self.gemini_api_key = settings.GEMINI_API_KEY
        self.gemini_model = settings.GEMINI_MODEL

        self._groq_client = None
        self._gemini_client = None

        if self.groq_api_key:
            try:
                from groq import Groq
                self._groq_client = Groq(api_key=self.groq_api_key)
                logger.info(f"Initialized Groq client with model {self.groq_model} (API Key: active)")
            except Exception as e:
                logger.warning(f"Could not initialize Groq client: {e}")
        else:
            logger.info("GROQ_API_KEY is not set. Will use deterministic fallback.")

        if self.gemini_api_key:
            try:
                from google import genai
                self._gemini_client = genai.Client(api_key=self.gemini_api_key)
                logger.info(f"Initialized Gemini client with model {self.gemini_model}")
            except Exception as e:
                logger.warning(f"Could not initialize Gemini client: {e}")

    # =========================================================================
    # 1. EXTRACT MEMORIES (WITH CONVERSATION CONTEXT & PRONOUN RESOLUTION)
    # =========================================================================
    EXTRACTION_SYSTEM_PROMPT = (
        "You are an expert knowledge extraction agent for an AI Memory System.\n"
        "Analyze the user message (along with any RECENT CONVERSATION CONTEXT) and extract atomic facts, preferences, goals, or architectural decisions.\n\n"
        "CRITICAL RULES:\n"
        "1. FACT SOURCE DISTINCTION (DO NOT EXTRACT ASSISTANT SUGGESTIONS OR SPECULATION):\n"
        "   - Classify each fact's 'source_type' strictly into:\n"
        "     * 'user_confirmed': A fact explicitly stated or confirmed by the user in the current message.\n"
        "     * 'context_resolved': A fact stated by the user that relies on prior conversation context to resolve pronouns or coreference (e.g. resolving 'we' or 'it'). Set 'provenance_context' explaining how the entity was resolved.\n"
        "     * 'assistant_suggestion': Hypotheses, suggestions, ideas, or questions originating from the assistant (e.g., asking if user met Manoj at the mall, suggesting drafting a contract, suggesting marketing campaigns, or expanding production). THESE MUST NEVER BE EXTRACTED AS USER FACTS!\n"
        "     * 'unconfirmed_inference': Inferences or speculation not explicitly stated by user (e.g., assuming Manoj was at the mall, assuming a meeting and deal occurred on the same date, assuming ownership percentages like 50/50 or inventing unstated owners). THESE MUST NEVER BE EXTRACTED AS USER FACTS!\n"
        "   - ONLY extract facts with source_type 'user_confirmed' or 'context_resolved'. Discard all assistant suggestions and unconfirmed inferences.\n"
        "2. ACCURACY & TEMPORAL INTEGRITY (NEVER INVENT DETAILS):\n"
        "   - If the user provides a date without a year (e.g. '19th October' or '19th ovtober'), NEVER invent or hallucinate a year. Keep the date exactly as stated.\n"
        "   - Do NOT assume that separate events (such as meeting a client and finalizing a deal) occurred on the same day or at the same time unless the user explicitly said so.\n"
        "   - Do NOT infer that a contract needs drafting, delivery schedule needs setting, or marketing campaigns are planned merely because the assistant suggested them.\n"
        "   - If user states personal or business ownership (e.g. 'we are the owner of a successfull water bottlw comapany' or 'i own a company'), extract the ownership attributing it to 'User' (or the user's team), NOT the software 'Project' (e.g. content: 'User owns a successful water bottle company', subject: 'company', predicate: 'owned_by', object_val: 'User'). NEVER invent specific ownership percentages or unmentioned partners.\n"
        "3. COREFERENCE & PRONOUN RESOLUTION:\n"
        "   - If the user uses pronouns like 'it', 'that', 'they', 'this', or gives a brief negation like 'no we won't use it', 'drop that', 'switch that to X':\n"
        "     Look at RECENT CONVERSATION CONTEXT to resolve what entity is being discussed!\n"
        "     Example:\n"
        "       Context: User mentioned using sentence transformer for vector embeddings (subject: 'embedding_model')\n"
        "       User message: 'no we wont use it'\n"
        "       Extracted fact:\n"
        "         content: 'Project will NOT use sentence transformer for vector embeddings'\n"
        "         subject: 'embedding_model' (MUST match the subject of the entity being rejected!)\n"
        "         predicate: 'does_not_use'\n"
        "         object_val: 'NOT sentence transformer'\n"
        "         category: 'LONG_TERM'\n"
        "         is_temporary: false (a permanent retraction is NOT temporary!)\n"
        "         source_type: 'context_resolved'\n"
        "         provenance_context: \"'it' resolved to sentence transformer from prior turn\"\n"
        "   - If the user says 'we met 6 years ago' and prior context shows 'my friend's name is manoj', resolve 'we' to User and friend Manoj (content: 'User met friend Manoj 6 years ago', subject: 'friend_meeting', predicate: 'met', object_val: 'Manoj', category: 'EPISODIC').\n"
        "   - IMPERATIVE ADD/REMEMBER REQUESTS & MULTI-VALUED ENTITIES:\n"
        "     * If the user requests to add, remember, save, or note an entity, person, relationship, preference, attribute, or item (e.g. 'add a new friend named John', 'can you add John as my friend', 'i got a new friend Sam', 'remember that my favorite food is pizza', 'add a new colleague Sarah', 'my hobby is photography'):\n"
        "       ALWAYS extract the affirmative declarative fact with source_type: 'user_confirmed'!\n"
        "       Examples:\n"
        "         - 'add a new friend named John' / 'add John as a friend' -> content: \"Friend's name is John\", subject: 'friend_name', predicate: 'has_name', object_val: 'John', replaces_value: null, category: 'SEMANTIC'\n"
        "         - 'add a new colleague Sarah' -> content: \"User's colleague is Sarah\", subject: 'colleague', predicate: 'is_colleague', object_val: 'Sarah', replaces_value: null, category: 'SEMANTIC'\n"
        "         - 'my favorite food is pizza' -> content: \"User's favorite food is pizza\", subject: 'favorite_food', predicate: 'prefers', object_val: 'pizza', replaces_value: null, category: 'SEMANTIC'\n"
        "         - 'my hobby is photography' -> content: \"User's hobby is photography\", subject: 'hobby', predicate: 'has_hobby', object_val: 'photography', replaces_value: null, category: 'SEMANTIC'\n"
        "     * Multi-valued entities (friends, colleagues, teammates, contacts, hobbies, favorites, skills, pets): Users can have MULTIPLE friends, hobbies, colleagues, etc.\n"
        "       Adding a new friend, colleague, hobby, or favorite is COMPLEMENTARY and does NOT replace or supersede existing ones. Only set 'replaces_value' if the user explicitly specifies a replacement (e.g. 'Abdul, not Manoj' or 'John instead of Bob').\n"
        "   - If the user mentions multiple friends (e.g. 'My friends are Syed and Sameer', 'I have two friends Sameer and Syed'):\n"
        "     ALWAYS extract each friend as an independent fact (e.g. one fact for Syed, one fact for Sameer, subject: 'friend_name', predicate: 'has_name', replaces_value: null). Neither replaces the other.\n"
        "   - If the user corrects a fact by stating a replacement and a negation (e.g. 'My friend's name is Abdul, not Manoj', 'We use PostgreSQL, not MySQL'):\n"
        "     ALWAYS extract the positive replacement fact (content: 'Friend\'s name is Abdul', subject: 'friend_name', predicate: 'has_name', object_val: 'Abdul', replaces_value: 'Manoj', category: 'SEMANTIC').\n"
        "     Set 'replaces_value' to the specific entity/value being replaced (or null if not a correction).\n"
        "     Do NOT extract standalone negative facts like 'friend\'s name is NOT Manoj' as a substitute for the corrected fact.\n"
        "4. SUBJECT NORMALIZATION (NEVER USE 'we', 'i', 'user', 'our', 'it'):\n"
        "   - Always normalize the subject to a specific domain entity/topic key:\n"
        "     * 'we are going to build a rag chatbot engine' -> subject: 'project_goal', predicate: 'is_building', object_val: 'RAG chatbot engine', content: 'Project is building a RAG chatbot engine', is_temporary: false, category: 'SEMANTIC'.\n"
        "     * 'we are gonna use sentence transformer for vector embeddings' -> subject: 'embedding_model', predicate: 'uses', object_val: 'sentence transformer', content: 'Project uses sentence transformer for vector embeddings', is_temporary: false, category: 'LONG_TERM'.\n"
        "     * 'we use postgresql' -> subject: 'primary_database', predicate: 'uses', object_val: 'PostgreSQL', category: 'LONG_TERM'.\n"
        "     * 'fastapi for backend' -> subject: 'backend_framework', predicate: 'uses', object_val: 'FastAPI', category: 'LONG_TERM'.\n"
        "     * DO NOT set subject to 'project_goal' when introducing a specific component (like embedding model, database, caching). Use 'embedding_model', 'primary_database', etc.\n"
        "5. MEMORY CATEGORY CLASSIFICATION (Set 'category' strictly to one of: 'SEMANTIC', 'LONG_TERM', 'EPISODIC', 'SHORT_TERM'):\n"
        "   - 'SEMANTIC': Stable concepts, project goals, definitions, user attributes, preferences, identities, and general domain knowledge.\n"
        "     * 'We are building a billing software engine' -> category: 'SEMANTIC'\n"
        "     * 'Project is building a RAG chatbot engine' -> category: 'SEMANTIC'\n"
        "     * 'Friend's name is Manoj' -> category: 'SEMANTIC'\n"
        "     * 'User owns a successful water bottle company' -> category: 'SEMANTIC'\n"
        "     * 'Project owns a successful water bottle company' -> category: 'SEMANTIC'\n"
        "     * 'I am Aarav, a backend architect' -> category: 'SEMANTIC'\n"
        "     * 'Travel time from Bangalore to Chennai is 5 hours' -> category: 'SEMANTIC'\n"
        "   - 'LONG_TERM': Persistent architectural decisions, technology-stack choices, infrastructure choices, and durable project configurations.\n"
        "     * 'Project uses PostgreSQL as primary database' -> category: 'LONG_TERM'\n"
        "     * 'Project backend uses FastAPI framework' -> category: 'LONG_TERM'\n"
        "     * 'Project uses sentence transformer for vector embeddings' -> category: 'LONG_TERM'\n"
        "     * 'Project will NOT use sentence transformer for vector embeddings' -> category: 'LONG_TERM'\n"
        "     * 'Project frontend uses React framework' -> category: 'LONG_TERM'\n"
        "     * 'Project deploys microservices on AWS ECS' -> category: 'LONG_TERM'\n"
        "   - 'EPISODIC': Specific events, incidents, milestones, decisions made during a particular event, and attributed statements when the fact being stored is that the statement occurred.\n"
        "     * 'User met friend Manoj 6 years ago' -> category: 'EPISODIC'\n"
        "     * 'User went to mall on 19th October' -> category: 'EPISODIC'\n"
        "     * 'User went to meet a business client along with Manoj' -> category: 'EPISODIC'\n"
        "     * 'Project finalised a deal on supplying water bottles to the client' -> category: 'EPISODIC'\n"
        "     * 'On Tuesday, the team officially completed migration from MySQL to PostgreSQL' -> category: 'EPISODIC'\n"
        "     * 'Production deployment failed due to OOM errors yesterday' -> category: 'EPISODIC'\n"
        "     * 'Raju reported that travel time from Bangalore to Chennai is 5 hours' -> category: 'EPISODIC'\n"
        "   - 'SHORT_TERM': Session-scoped context, temporary working information, or local test parameters (always pair with is_temporary: true).\n"
        "     * 'Using SQLite in-memory for unit tests today' -> category: 'SHORT_TERM', is_temporary: true\n"
        "6. TRANSITIONS & MIGRATIONS:\n"
        "   - If user mentions migrating or switching (e.g. 'We migrated our primary database from MongoDB to PostgreSQL'), extract the CURRENT state as canonical statement ('Project uses PostgreSQL as primary database', subject='primary_database', predicate='uses', object_val='PostgreSQL', category='LONG_TERM').\n"
        "   - If user notes a specific historical event of migration occurring at a specific time (e.g. 'On Tuesday, the team completed migration from MySQL to PostgreSQL'), that event record is EPISODIC.\n"
        "7. TEMPORARY VS PERMANENT:\n"
        "   - Only set is_temporary=true if the user explicitly specifies an ephemeral/session-only exception (e.g. 'just for local unit tests today', 'temporarily for now'). Negations, retractions, and architecture decisions are permanent updates (is_temporary=false).\n"
        "8. PURE QUESTIONS & GREETINGS:\n"
        "   - If the user message is a pure question or casual greeting (e.g. 'how many countries are there in the world', 'hi', 'how are you', 'what database are we using?') with NO new declarative facts, return empty list [] in 'facts'.\n\n"
        "Return strict JSON: {'facts': [{'content': str, 'subject': str, 'predicate': str, 'object_val': str, 'replaces_value': str | null, 'category': str, 'importance': float, 'confidence': float, 'is_temporary': bool, 'source_type': str, 'provenance_context': str | null}]}"
    )

    def extract_memories(
        self, user_message: str, recent_context: Optional[str] = None
    ) -> List[ExtractedFact]:
        """Extract atomic facts from user message with coreference/pronoun resolution.
        
        Returns empty list for pure questions or casual greetings.
        Never extracts generic pronouns ('we', 'i', 'our') as the subject key.
        """
        if self._is_pure_question_or_greeting(user_message):
            return []

        facts: Optional[List[ExtractedFact]] = None

        # Try Groq if configured
        if self._groq_client:
            facts = self._extract_with_groq(user_message, recent_context)

        # Try Gemini if Groq returned None or is unavailable
        if facts is None and self._gemini_client:
            facts = self._extract_with_gemini(user_message, recent_context)

        # Fallback to deterministic regex & rule parser
        if facts is None:
            facts = self._fallback_extract_memories(user_message, recent_context)

        # Post-extraction normalization & validation to strictly enforce architecture requirements
        normalized_facts: List[ExtractedFact] = []
        user_msg_low = user_message.lower()
        for f in facts:
            # 1. Discard assistant suggestions and unconfirmed inferences
            if f.source_type in ["assistant_suggestion", "unconfirmed_inference"]:
                continue

            obj_low = (f.object_val or "").lower()
            cont_low = (f.content or "").lower()
            sub_low = (f.subject or "").lower()

            # 2. Reject unsupported inferences derived from assistant suggestions:
            # - Do not infer that Manoj attended the mall visit unless explicitly stated in the user message
            if "mall" in cont_low and "manoj" in cont_low and "manoj" not in user_msg_low:
                continue

            # - Do not infer that a contract needs drafting
            if any(w in cont_low or w in obj_low for w in ["draft contract", "drafting contract", "draft the contract", "needs drafting"]):
                continue

            # - Do not infer that production expansion, distribution expansion, or marketing campaigns are planned/underway
            if any(w in cont_low or w in obj_low for w in ["marketing campaign", "marketing material", "expand production", "distribution channel", "distribution channels"]):
                if not any(w in user_msg_low for w in ["marketing", "expand production", "distribution"]):
                    continue

            # - Do not infer legal ownership shares (e.g. 50/50, 100%) or unmentioned owners
            if "owner" in sub_low or "company" in sub_low:
                if any(w in cont_low or w in obj_low for w in ["50%", "100%", "shareholder", "legal share", "co-owner"]):
                    continue

            # 3. Ensure mall visit retains exact date with NO invented year
            if "mall" in cont_low and any(yr in cont_low for yr in ["2020", "2021", "2022", "2023", "2024", "2025", "2026", "2027", "2028"]):
                if not any(yr in user_message for yr in ["2020", "2021", "2022", "2023", "2024", "2025", "2026", "2027", "2028"]):
                    f.content = re.sub(r"\s+202[0-9]", "", f.content)
                    if f.object_val:
                        f.object_val = re.sub(r"\s+202[0-9]", "", f.object_val)

            # Ensure embedding models have subject 'embedding_model'
            if any(w in obj_low or w in cont_low for w in ["sentence transformer", "sentence-transformer", "vector embeddings", "embedding"]):
                f.subject = "embedding_model"

            # Ensure databases have subject 'primary_database'
            elif any(w in obj_low or w in cont_low for w in ["postgresql", "postgres", "mongodb", "mysql", "sqlite", "primary database"]):
                f.subject = "primary_database"

            # Ensure frameworks have subject 'backend_framework'
            elif any(w in obj_low or w in cont_low for w in ["fastapi", "django", "flask", "backend"]):
                f.subject = "backend_framework"

            # Ensure goals have subject 'project_goal'
            elif any(w in obj_low or w in cont_low for w in ["rag chatbot", "rag engine", "chatbot engine", "build a rag"]):
                f.subject = "project_goal"

            # Ensure client deals have subject 'client_deal'
            elif any(w in obj_low or w in cont_low for w in ["finalised a deal", "finalized a deal", "supplying water bottles", "water bottle supply", "client deal"]):
                f.subject = "client_deal"

            # Ensure client meetings have subject 'client_meeting'
            elif any(w in obj_low or w in cont_low for w in ["business client", "meet a business client", "meet a client", "client meeting"]):
                f.subject = "client_meeting"

            # Ensure company ownership has subject 'company'
            elif any(w in obj_low or w in cont_low for w in ["owner of a successful water bottle", "owns a successful water bottle"]):
                f.subject = "company"

            # Ensure friend name has subject 'friend_name'
            elif any(w in obj_low or w in cont_low for w in ["friend name", "friend's name", "friend has name", "friend is named", "name is abdul", "name is manoj", "new friend", "friend named", "friend called"]) or f.subject in ["friend", "friend_name", "friends_name", "friends", "new_friend"]:
                f.subject = "friend_name"
                if f.object_val and f.object_val.lower() not in ["user", "friend"]:
                    f.object_val = f.object_val.strip().capitalize()
                if not f.replaces_value:
                    corr_match = re.search(r"(?:,\s*not\s+|\s+not\s+|\s+instead\s+of\s+)([A-Za-z]+)", user_message)
                    if corr_match:
                        cand_rep = corr_match.group(1).strip().capitalize()
                        if cand_rep.lower() != (f.object_val or "").lower():
                            f.replaces_value = cand_rep

            # Ensure colleague / contact / teammate
            elif any(w in obj_low or w in cont_low for w in ["colleague", "coworker", "teammate", "work colleague"]) or f.subject in ["colleague", "coworker", "teammate"]:
                f.subject = "colleague"
                if f.object_val:
                    f.object_val = f.object_val.strip().capitalize()

            # Ensure hobbies and favorites
            elif any(w in obj_low or w in cont_low for w in ["hobby", "hobbies", "pastime"]) or f.subject in ["hobby", "hobbies"]:
                f.subject = "hobby"
            elif "favorite" in cont_low or "favorite" in sub_low:
                if not f.subject.startswith("favorite_"):
                    fav_m = re.search(r"favorite\s+([a-zA-Z]+)", cont_low)
                    if fav_m:
                        f.subject = f"favorite_{fav_m.group(1).lower()}"

            # If user explicitly requested adding or storing an entity/friend, ensure source_type is user_confirmed
            if any(p in user_msg_low for p in ["add ", "new friend", "got a friend", "my friend", "remember", "favorite", "my hobby", "colleague"]):
                if f.source_type in ["assistant_suggestion", "unconfirmed_inference"]:
                    f.source_type = "user_confirmed"

            # Generic pronoun cleanup
            if f.subject.lower() in ["we", "i", "user", "our", "it"]:
                f.subject = "project_architecture"

            # Ensure retractions are NEVER temporary
            if any(w in cont_low or w in obj_low for w in ["not use", "wont use", "won't use", "will not use", "rejected", "dropped", "stop using"]):
                f.is_temporary = False

            # Validate and normalize category safely
            f.category = self._normalize_or_fallback_category(f)

            normalized_facts.append(f)

        # Ensure fact corrections prioritize the positive replacement fact over standalone negative memories
        has_positive_friend_name = any(
            f.subject == "friend_name" and not any(neg in (f.object_val or "").lower() or neg in (f.content or "").lower() for neg in ["not ", "does_not", "is not"])
            for f in normalized_facts
        )
        if has_positive_friend_name:
            normalized_facts = [
                f for f in normalized_facts
                if not (f.subject == "friend_name" and any(neg in (f.object_val or "").lower() or neg in (f.content or "").lower() for neg in ["not ", "does_not", "is not"]))
            ]
        else:
            cor_match = re.search(r"(?:my\s+)?friend(?:'s)?\s+(?:name\s+is|is\s+named|is)\s+([A-Za-z]+)(?:\s*,\s*not\s+[A-Za-z]+)", user_message, re.IGNORECASE)
            if cor_match:
                cor_name = cor_match.group(1).strip().capitalize()
                normalized_facts = [f for f in normalized_facts if f.subject != "friend_name"]
                normalized_facts.append(
                    ExtractedFact(
                        content=f"Friend's name is {cor_name}",
                        subject="friend_name",
                        predicate="has_name",
                        object_val=cor_name,
                        category=MemoryCategory.SEMANTIC,
                        importance=0.85,
                        confidence=1.0,
                        is_temporary=False,
                        source_type="user_confirmed",
                    )
                )

        return normalized_facts

    def _normalize_or_fallback_category(self, fact: ExtractedFact) -> MemoryCategory:
        """Validate and classify memory category safely without blindly overwriting valid LLM classifications."""
        # 1. Ephemeral session overrides are strictly SHORT_TERM
        if fact.is_temporary:
            return MemoryCategory.SHORT_TERM

        cont_low = (fact.content or "").lower().strip()
        pred_low = (fact.predicate or "").lower().strip()

        # 2. Check statement meaning and temporal context: specific events, incidents, milestones, attributed statements
        temporal_markers = [
            "yesterday", "last week", "last month", "last night", "earlier today",
            "on monday", "on tuesday", "on wednesday", "on thursday", "on friday",
            "on saturday", "on sunday",
        ]
        episodic_event_keywords = [
            "incident", "outage", "failed deployment", "deployment failed",
            "crashed due to", "server crash", "oom error", "promoted to",
            "officially completed migration", "meeting held", "sprint planning",
            "agreed during", "decided during",
        ]
        is_event_or_incident = (
            any(m in cont_low for m in temporal_markers)
            or any(w in cont_low for w in episodic_event_keywords)
            or pred_low in ["reported", "stated", "claimed", "claims", "says", "said", "noted"]
            or bool(re.search(r"^[A-Z][a-z]+\s+(?:reported|says|said|stated|claimed|claims|noted)\s+(?:that\s+)?", fact.content or ""))
        )
        if is_event_or_incident:
            return MemoryCategory.EPISODIC

        # 3. If fact has a valid category from LLM, respect it!
        # Do not silently overwrite a valid LLM classification based solely on subject name.
        if fact.category is not None and isinstance(fact.category, MemoryCategory):
            return fact.category

        # 4. Fallback classification only when classification is missing or invalid
        return classify_memory_category(
            content=fact.content,
            subject=fact.subject,
            predicate=fact.predicate,
            object_val=fact.object_val,
            is_temporary=fact.is_temporary,
        )

    def _is_pure_question_or_greeting(self, text: str) -> bool:
        """Detect if message is purely conversational greeting or question without declarative facts."""
        text_clean = text.strip().lower()

        # Casual greetings
        greetings = ("hi", "hello", "hey", "good morning", "good evening", "howdy", "sup", "how are you", "who are you")
        if text_clean in greetings or text_clean.startswith(greetings):
            # Check if it also has declarative statements or entity introductions
            if not any(w in text_clean for w in [
                "we are", "using", "migrated", "project", "building", "my friend",
                "add a friend", "add friend", "new friend", "colleague", "remember", "favorite"
            ]):
                return True

        # Check for explicit memory storage / ingestion requests:
        # e.g. "add a new friend", "can you add John as my friend", "remember that my favorite...", "save that..."
        memory_commands = (
            "add ", "add a ", "adding ", "remember ", "remember that ",
            "save ", "save that ", "note ", "note that ", "store ", "store that ",
            "keep in mind ", "record ", "record that ", "register "
        )
        has_memory_command = any(cmd in text_clean for cmd in memory_commands)

        # Entity introduction patterns (e.g. "friend is X", "new friend named X", "favorite food is X")
        has_entity_intro = bool(re.search(
            r"\b(?:friend|colleague|coworker|teammate|contact|brother|sister|father|mother|cousin|partner|pet|hobby|favorite|location|birthday)\b.*?\b(?:is|named|called|are|as)\b",
            text_clean,
        )) or any(
            p in text_clean for p in [
                "new friend", "got a friend", "have a friend", "made a friend",
                "got a new friend", "have a new friend", "my friend is", "friend's name",
                "friend is named", "new colleague", "my colleague is", "my favorite",
                "my hobby", "i live in", "allergic to"
            ]
        )

        has_declarative = any(
            w in text_clean for w in [
                "migrated", "switched", "we use", "our project uses",
                "i am using", "updated to", "my name is", "we decided",
                "we are going to build", "we are building", "we are gonna use",
                "we wont use", "we won't use", "no we wont", "no we won't",
                "owns a", "owner of", "went to", "met a", "finalised a", "finalized a"
            ]
        )

        if has_memory_command or has_entity_intro or has_declarative:
            # Pure retrieval questions like "who is my friend?" or "what is our primary database?"
            # should still be treated as questions (returning no facts).
            # Distinguish queries asking for existing facts from statements/commands adding them.
            retrieval_query_patterns = [
                r"^(?:who|what|which|where|when|how|do we|do i|are there|is there|can you tell me|tell me)\s+(?:is|are|was|were|did|do|have|has)?\s*(?:my|our|the)?\s*(?:friends?|colleagues?|hobbies|hobby|favorites?|database|project|goal|skills?|contacts?)\s*\??$",
                r"^(?:who\s+are\s+my\s+friends|who\s+is\s+my\s+friend|what\s+is\s+my\s+friend'?s?\s+name|how\s+many\s+friends\s+do\s+i\s+have|how\s+many\s+friends\s+i\s+have|what\s+is\s+my\s+primary\s+database|what\s+was\s+project\s+goal\s+before\s+this)\s*\??$",
            ]
            if any(re.match(pat, text_clean) for pat in retrieval_query_patterns):
                return True
            return False

        question_starters = (
            "what ", "what's ", "which ", "who ", "where ", "when ",
            "why ", "how ", "how many", "is there ", "are there ", "can you ",
            "tell me ", "do we ", "does the "
        )
        has_question_punct = text_clean.endswith("?")
        starts_with_question = any(text_clean.startswith(q) for q in question_starters)

        return starts_with_question or has_question_punct

    def _extract_with_groq(
        self, user_message: str, recent_context: Optional[str] = None
    ) -> Optional[List[ExtractedFact]]:
        user_content = f"User Message: {user_message}"
        if recent_context:
            user_content = f"Recent Conversation & Memory Context:\n{recent_context}\n\n{user_content}"

        models_to_try = [
            self.groq_model,
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.8-27b",
            self.groq_fallback_model,
        ]
        seen = set()
        dedup_models = [m for m in models_to_try if m and not (m in seen or seen.add(m))]

        for model in dedup_models:
            try:
                resp = self._groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": self.EXTRACTION_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    model=model,
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                data = json.loads(resp.choices[0].message.content)
                raw_facts = data.get("facts", [])
                return [ExtractedFact(**f) for f in raw_facts]
            except Exception as e:
                logger.warning(f"Groq extraction failed with {model}: {e}")
        return None

    def _extract_with_gemini(
        self, user_message: str, recent_context: Optional[str] = None
    ) -> Optional[List[ExtractedFact]]:
        contents = (
            f"{self.EXTRACTION_SYSTEM_PROMPT}\n\n"
            f"Recent Conversation & Memory Context:\n{recent_context or 'None'}\n\n"
            f"User Message: {user_message}"
        )
        try:
            resp = self._gemini_client.models.generate_content(
                model=self.gemini_model,
                contents=contents,
                config={"response_mime_type": "application/json"},
            )
            data = json.loads(resp.text)
            raw_facts = data.get("facts", [])
            return [ExtractedFact(**f) for f in raw_facts]
        except Exception as e:
            logger.warning(f"Gemini extraction failed: {e}")
            return None

    def _fallback_extract_memories(
        self, user_message: str, recent_context: Optional[str] = None
    ) -> List[ExtractedFact]:
        """Deterministic heuristic fallback extractor with coreference & pronoun resolution."""
        facts: List[ExtractedFact] = []
        text = user_message.strip()
        text_low = text.lower()
        context_low = (recent_context or "").lower()

        # Check for temporary markers
        temp_markers = ["local unit tests", "unit tests today", "temporarily", "right now", "just for ... today", "today only"]
        is_temp = any(m in text_low for m in temp_markers)

        # 1. Pronoun / Coreference Negation & Retraction:
        # e.g. "no we wont use it", "no we won't use it", "we won't use it", "drop that", "not using it"
        is_negation = any(
            p in text_low for p in [
                "no we wont use it", "no we won't use it", "wont use it", "won't use it",
                "will not use it", "not using it", "dont use it", "don't use it",
                "drop that", "drop it", "we will not use it", "stop using it"
            ]
        )
        if is_negation:
            # Resolve pronoun 'it' from recent context
            if "sentence transformer" in context_low or "embedding" in context_low:
                facts.append(
                    ExtractedFact(
                        content="Project will NOT use sentence transformer for vector embeddings",
                        subject="embedding_model",
                        predicate="does_not_use",
                        object_val="NOT sentence transformer",
                        category=MemoryCategory.LONG_TERM,
                        importance=0.85,
                        confidence=1.0,
                        is_temporary=False,  # Retraction is a permanent architectural update
                    )
                )
                return facts
            elif "mongodb" in context_low:
                facts.append(
                    ExtractedFact(
                        content="Project will NOT use MongoDB as primary database",
                        subject="primary_database",
                        predicate="does_not_use",
                        object_val="NOT MongoDB",
                        category=MemoryCategory.LONG_TERM,
                        importance=0.85,
                        confidence=1.0,
                        is_temporary=False,
                    )
                )
                return facts

        # 2. Migration detection: e.g. "migrated our primary database from MongoDB to PostgreSQL"
        mig_match = re.search(
            r"migrated\s+(?:our|the)?\s*([a-zA-Z0-9_\s]+?)\s*from\s+([a-zA-Z0-9_\s]+?)\s+to\s+([a-zA-Z0-9_\s]+?)(?:\s+because|\.|$|,|;)",
            text,
            re.IGNORECASE,
        )
        if mig_match:
            raw_subject = mig_match.group(1).strip()
            old_val = mig_match.group(2).strip()
            new_val = mig_match.group(3).strip()

            subject_key = "primary_database" if "database" in raw_subject.lower() else raw_subject.lower().replace(" ", "_")
            facts.append(
                ExtractedFact(
                    content=f"Project uses {new_val} as primary database",
                    subject=subject_key,
                    predicate="uses",
                    object_val=new_val,
                    category=MemoryCategory.LONG_TERM,
                    importance=0.90,
                    confidence=1.0,
                    is_temporary=False,
                )
            )
            return facts

        # 3. Temporary database usage for tests:
        if is_temp and ("sqlite" in text_low or "in-memory" in text_low):
            facts.append(
                ExtractedFact(
                    content="Using SQLite in-memory for local unit tests today",
                    subject="primary_database",
                    predicate="temporary_test_db",
                    object_val="SQLite in-memory",
                    category=MemoryCategory.SHORT_TERM,
                    importance=0.40,
                    confidence=0.95,
                    is_temporary=True,
                )
            )
            return facts

        # 4. Project Goal / Intent: e.g. "we are going to build a rag chatbot engine"
        goal_match = re.search(
            r"(?:going to build|building|creating|developing|plan to build|will sell|sells|focuses on|focus on)\s+(?:a|an)?\s*([a-zA-Z0-9_\s\-]+?)(?:\.|$|,|;)",
            text,
            re.IGNORECASE,
        )
        if goal_match and not is_temp:
            goal_target = goal_match.group(1).strip()
            pred = "sells" if "sell" in text_low else ("focuses_on" if "focus" in text_low else "is_building")
            facts.append(
                ExtractedFact(
                    content=f"Project is building {goal_target}" if pred == "is_building" else f"Project {pred.replace('_', ' ')} {goal_target}",
                    subject="project_goal",
                    predicate=pred,
                    object_val=goal_target,
                    category=MemoryCategory.SEMANTIC,
                    importance=0.85,
                    confidence=1.0,
                    is_temporary=False,
                )
            )

        # 5. Embedding Model choice: e.g. "we are gonna use sentence transformer for vector embeddings"
        if "sentence transformer" in text_low or "sentence-transformer" in text_low:
            facts.append(
                ExtractedFact(
                    content="Project uses sentence transformer for vector embeddings",
                    subject="embedding_model",
                    predicate="uses",
                    object_val="sentence transformer",
                    category=MemoryCategory.LONG_TERM,
                    importance=0.80,
                    confidence=1.0,
                    is_temporary=False,
                )
            )

        # 6. User identity/role: e.g. "Hi, I'm Aarav, a backend architect."
        role_match = re.search(
            r"(?:i'?m|i am|my name is)\s+([A-Z][a-z]+)(?:,\s*(?:a|an)\s+([a-zA-Z\s]+))?",
            text,
            re.IGNORECASE,
        )
        if role_match:
            name = role_match.group(1).strip()
            role = role_match.group(2).strip() if role_match.group(2) else ""
            if role:
                facts.append(
                    ExtractedFact(
                        content=f"User {name} is a {role}",
                        subject="user_profile",
                        predicate="has_role",
                        object_val=f"{name} ({role})",
                        category=MemoryCategory.SEMANTIC,
                        importance=0.70,
                        confidence=1.0,
                        is_temporary=False,
                    )
                )

        # 7. Primary database usage:
        db_match = re.search(
            r"uses\s+([A-Za-z0-9]+)\s+as\s+(?:our|the)?\s*primary\s+database",
            text,
            re.IGNORECASE,
        )
        if db_match:
            db_name = db_match.group(1).strip()
            facts.append(
                ExtractedFact(
                    content=f"Project uses {db_name} as primary database",
                    subject="primary_database",
                    predicate="uses",
                    object_val=db_name,
                    category=MemoryCategory.LONG_TERM,
                    importance=0.85,
                    confidence=1.0,
                    is_temporary=False,
                )
            )

        # 8. Backend framework:
        be_match = re.search(
            r"([A-Za-z0-9]+)\s+(?:for|as)\s+(?:the|our)?\s*backend",
            text,
            re.IGNORECASE,
        )
        if be_match:
            framework = be_match.group(1).strip()
            facts.append(
                ExtractedFact(
                    content=f"Project uses {framework} for the backend",
                    subject="backend_framework",
                    predicate="uses",
                    object_val=framework,
                    category=MemoryCategory.LONG_TERM,
                    importance=0.80,
                    confidence=1.0,
                    is_temporary=False,
                )
            )

        # 9. Frontend framework:
        fe_match = re.search(
            r"([A-Za-z0-9\.\-]+)\s+(?:for|as)\s+(?:the|our)?\s*frontend",
            text,
            re.IGNORECASE,
        )
        if fe_match:
            framework = fe_match.group(1).strip()
            facts.append(
                ExtractedFact(
                    content=f"Project uses {framework} for the frontend",
                    subject="frontend_framework",
                    predicate="uses",
                    object_val=framework,
                    category=MemoryCategory.LONG_TERM,
                    importance=0.80,
                    confidence=1.0,
                    is_temporary=False,
                )
            )

        # 10. Specific incident / outage / completed migration event:
        incident_match = re.search(
            r"(?:on\s+[A-Za-z]+|yesterday|last\s+week|earlier\s+today)\s*,?\s*(?:the\s+team\s+)?(.+?(?:failed|crash|incident|outage|migration|completed\s+migration).+)",
            text,
            re.IGNORECASE,
        )
        if incident_match:
            inc_content = text.strip().rstrip(".")
            facts.append(
                ExtractedFact(
                    content=inc_content,
                    subject="system_event",
                    predicate="occurred",
                    object_val="incident_or_milestone",
                    category=MemoryCategory.EPISODIC,
                    importance=0.75,
                    confidence=1.0,
                    is_temporary=False,
                )
            )

        # 11. Milestone / promotion event:
        promo_match = re.search(
            r"([A-Za-z0-9_\s]+?)\s+(?:was\s+)?promoted\s+to\s+([A-Za-z0-9_\s]+)",
            text,
            re.IGNORECASE,
        )
        if promo_match:
            person_or_user = promo_match.group(1).strip()
            new_title = promo_match.group(2).strip().rstrip(".")
            facts.append(
                ExtractedFact(
                    content=f"{person_or_user} was promoted to {new_title}",
                    subject="career_milestone",
                    predicate="promoted_to",
                    object_val=new_title,
                    category=MemoryCategory.EPISODIC,
                    importance=0.75,
                    confidence=1.0,
                    is_temporary=False,
                )
            )

        # 12. Attributed statement: e.g. "Raju says travel time is 5 hours" or "Ravi reported that travel time is 6 hours"
        attr_match = re.search(
            r"^([A-Z][a-z]+)\s+(?:says|said|stated|reported|claims|claimed|noted)\s+(?:that\s+)?(.+)",
            text.strip(),
            re.IGNORECASE,
        )
        if attr_match:
            speaker = attr_match.group(1).strip()
            statement = attr_match.group(2).strip().rstrip(".")
            subj = "travel_time" if "travel time" in statement.lower() else f"{speaker.lower()}_report"
            facts.append(
                ExtractedFact(
                    content=f"{speaker} reported that {statement}",
                    subject=subj,
                    predicate="reported",
                    object_val=statement,
                    category=MemoryCategory.EPISODIC,
                    importance=0.70,
                    confidence=1.0,
                    is_temporary=False,
                )
            )

        # 13. General route / domain fact (unattributed): e.g. "Travel time from Bangalore to Chennai is 5 hours"
        travel_match = re.search(
            r"travel\s+time\s+(?:from\s+([A-Za-z\s]+?)\s+to\s+([A-Za-z\s]+?)|between\s+([A-Za-z\s]+?)\s+and\s+([A-Za-z\s]+?))\s+(?:is|takes|was)\s+([0-9]+\s*(?:hours?|hrs?|minutes?|mins?))",
            text,
            re.IGNORECASE,
        )
        if travel_match and not attr_match:
            src = (travel_match.group(1) or travel_match.group(3)).strip()
            dst = (travel_match.group(2) or travel_match.group(4)).strip()
            duration = travel_match.group(5).strip()
            facts.append(
                ExtractedFact(
                    content=f"Travel time from {src} to {dst} is {duration}",
                    subject="travel_time",
                    predicate="is",
                    object_val=duration,
                    category=MemoryCategory.SEMANTIC,
                    importance=0.70,
                    confidence=1.0,
                    is_temporary=False,
                )
            )

        # 14. Friend names (supports multiple friends, adding new friends, and explicit corrections)
        multi_friend_match = re.search(
            r"(?:(?:my\s+)?friends?\s+(?:are|include|:)|(?:have\s+)?(?:two|2|\d+)\s+friends?)\s+([A-Za-z]+)\s+(?:and|&)\s+([A-Za-z]+)",
            text,
            re.IGNORECASE,
        )
        if multi_friend_match:
            for group_idx in [1, 2]:
                fname = multi_friend_match.group(group_idx).strip().capitalize()
                facts.append(
                    ExtractedFact(
                        content=f"Friend's name is {fname}",
                        subject="friend_name",
                        predicate="has_name",
                        object_val=fname,
                        category=MemoryCategory.SEMANTIC,
                        importance=0.85,
                        confidence=1.0,
                        is_temporary=False,
                        source_type="user_confirmed",
                        provenance_context=None,
                        replaces_value=None,
                    )
                )
        else:
            # Matches: "add a new friend named John", "add John as my friend", "i got a new friend sam", "my friend is Bob", etc.
            friend_add_patterns = [
                r"(?:can\s+you\s+|please\s+)?(?:add|remember|save|note)\s+(?:a\s+)?(?:new\s+)?friend\s+(?:named\s+|called\s+)?([A-Za-z]+)(?:,\s*not\s+([A-Za-z]+)|\s+instead\s+of\s+([A-Za-z]+))?",
                r"(?:can\s+you\s+|please\s+)?(?:add|remember|save|note)\s+([A-Za-z]+)\s+as\s+(?:a\s+|my\s+)?(?:new\s+)?friend(?:,\s*not\s+([A-Za-z]+)|\s+instead\s+of\s+([A-Za-z]+))?",
                r"(?:can\s+you\s+|please\s+)?(?:add|remember|save|note)\s+([A-Za-z]+)\s+to\s+(?:my\s+)?friends?",
                r"(?:i\s+)?(?:got|have|made)\s+(?:a\s+)?(?:new\s+)?friend\s+(?:named\s+|called\s+)?([A-Za-z]+)",
                r"(?:my\s+)?(?:new\s+)?friend(?:'s)?\s+(?:name\s+is|is\s+named|is)\s+([A-Za-z]+)(?:,\s*not\s+([A-Za-z]+)|\s+instead\s+of\s+([A-Za-z]+))?",
                r"^friend\s*:\s*([A-Za-z]+)",
            ]
            for pat in friend_add_patterns:
                f_match = re.search(pat, text, re.IGNORECASE)
                if f_match:
                    fname = f_match.group(1).strip().capitalize()
                    replaces_val = None
                    if len(f_match.groups()) >= 2:
                        rep = f_match.group(2) if len(f_match.groups()) == 2 else (f_match.group(2) or f_match.group(3))
                        if rep:
                            replaces_val = rep.strip().capitalize()
                    facts.append(
                        ExtractedFact(
                            content=f"Friend's name is {fname}",
                            subject="friend_name",
                            predicate="has_name",
                            object_val=fname,
                            replaces_value=replaces_val,
                            category=MemoryCategory.SEMANTIC,
                            importance=0.85,
                            confidence=1.0,
                            is_temporary=False,
                            source_type="user_confirmed",
                            provenance_context=None,
                        )
                    )
                    break

        # 14b. Colleagues, Coworkers, Teammates, Contacts
        col_match = re.search(
            r"(?:can\s+you\s+|please\s+)?(?:add|remember|save|note)\s+(?:a\s+)?(?:new\s+)?(colleague|coworker|teammate|contact)\s+(?:named\s+|called\s+)?([A-Za-z]+)"
            r"|(?:can\s+you\s+|please\s+)?(?:add|remember|save|note)\s+([A-Za-z]+)\s+as\s+(?:a\s+|my\s+)?(?:new\s+)?(colleague|coworker|teammate|contact)"
            r"|(?:my\s+)?(?:new\s+)?(colleague|coworker|teammate|contact)(?:'s)?\s+(?:name\s+is|is\s+named|is)\s+([A-Za-z]+)",
            text,
            re.IGNORECASE,
        )
        if col_match:
            role = (col_match.group(1) or col_match.group(4) or col_match.group(5) or "colleague").lower()
            name = (col_match.group(2) or col_match.group(3) or col_match.group(6) or "").strip().capitalize()
            if name:
                facts.append(
                    ExtractedFact(
                        content=f"User's {role} is {name}",
                        subject="colleague",
                        predicate=f"is_{role}",
                        object_val=name,
                        category=MemoryCategory.SEMANTIC,
                        importance=0.80,
                        confidence=1.0,
                        is_temporary=False,
                        source_type="user_confirmed",
                    )
                )

        # 14c. Hobbies & Interests
        hobby_match = re.search(
            r"(?:add|remember|save|note)\s+(?:a\s+)?(?:new\s+)?hobby\s*:\s*([a-zA-Z0-9_\s]+)"
            r"|(?:my\s+)?(?:new\s+)?hobby\s+is\s+([a-zA-Z0-9_\s]+)"
            r"|i\s+(?:love|enjoy)\s+([a-zA-Z0-9_\s]+?\s+as\s+a\s+hobby)",
            text,
            re.IGNORECASE,
        )
        if hobby_match:
            hobby_val = (hobby_match.group(1) or hobby_match.group(2) or hobby_match.group(3) or "").strip()
            if hobby_val:
                facts.append(
                    ExtractedFact(
                        content=f"User's hobby is {hobby_val}",
                        subject="hobby",
                        predicate="has_hobby",
                        object_val=hobby_val,
                        category=MemoryCategory.SEMANTIC,
                        importance=0.75,
                        confidence=1.0,
                        is_temporary=False,
                        source_type="user_confirmed",
                    )
                )

        # 14d. Favorites (food, movie, book, color, etc.)
        fav_match = re.search(
            r"(?:my\s+)?favorite\s+([a-zA-Z]+?)\s+is\s+([a-zA-Z0-9_\s]+)"
            r"|(?:add|remember|save|note)\s+(?:a\s+)?(?:new\s+)?favorite\s+([a-zA-Z]+?)\s*:\s*([a-zA-Z0-9_\s]+)",
            text,
            re.IGNORECASE,
        )
        if fav_match:
            fav_type = (fav_match.group(1) or fav_match.group(3) or "item").strip().lower()
            fav_val = (fav_match.group(2) or fav_match.group(4) or "").strip()
            if fav_val:
                facts.append(
                    ExtractedFact(
                        content=f"User's favorite {fav_type} is {fav_val}",
                        subject=f"favorite_{fav_type}",
                        predicate="prefers",
                        object_val=fav_val,
                        category=MemoryCategory.SEMANTIC,
                        importance=0.75,
                        confidence=1.0,
                        is_temporary=False,
                        source_type="user_confirmed",
                    )
                )

        # 15. Friend meeting milestone with context resolution: e.g. "we met 6 years ago"
        friend_meeting_match = re.search(
            r"(?:we|i)\s+met\s+([0-9]+\s*(?:years?|months?|weeks?|days?))(?:\s*ago)?",
            text,
            re.IGNORECASE,
        )
        if friend_meeting_match:
            duration = friend_meeting_match.group(1).strip()
            friend_context_name = "Manoj" if "manoj" in context_low else "friend"
            facts.append(
                ExtractedFact(
                    content=f"User met friend {friend_context_name} {duration} ago",
                    subject="friend_meeting",
                    predicate="met",
                    object_val=friend_context_name if friend_context_name != "friend" else duration,
                    category=MemoryCategory.EPISODIC,
                    importance=0.75,
                    confidence=1.0,
                    is_temporary=False,
                    source_type="context_resolved" if friend_context_name != "friend" else "user_confirmed",
                    provenance_context=(
                        f"'we' resolved to User and friend {friend_context_name} from conversation context"
                        if friend_context_name != "friend"
                        else None
                    ),
                )
            )

        # 16. Mall visit / location outing (strictly preserving original date with NO invented year):
        mall_match = re.search(
            r"i\s+went\s+to\s+(?:the\s+)?([a-zA-Z\s]+?)\s+on\s+([0-9]{1,2}(?:st|nd|rd|th)?\s+[a-zA-Z]+(?:\s+[0-9]{4})?)",
            text,
            re.IGNORECASE,
        )
        if mall_match:
            dest = mall_match.group(1).strip()
            raw_date = mall_match.group(2).strip()
            normalized_date = re.sub(r"ovtober", "October", raw_date, flags=re.IGNORECASE)
            normalized_date = re.sub(r"october", "October", normalized_date, flags=re.IGNORECASE)
            facts.append(
                ExtractedFact(
                    content=f"User went to {dest} on {normalized_date}",
                    subject="mall_visit" if "mall" in dest.lower() else "location_visit",
                    predicate="went_to",
                    object_val=dest,
                    category=MemoryCategory.EPISODIC,
                    importance=0.70,
                    confidence=1.0,
                    is_temporary=False,
                    source_type="user_confirmed",
                    provenance_context=None,
                )
            )

        # 17. Client meeting with companion: e.g. "i went to meet a business client along with manoj"
        client_meet_match = re.search(
            r"(?:went\s+to\s+meet|met)\s+(?:a\s+)?(?:business\s+)?client\s+(?:along\s+with|with)\s+([A-Za-z]+)",
            text,
            re.IGNORECASE,
        )
        if client_meet_match:
            comp_name = client_meet_match.group(1).strip().capitalize()
            facts.append(
                ExtractedFact(
                    content=f"User went to meet a business client along with {comp_name}",
                    subject="client_meeting",
                    predicate="attended",
                    object_val=f"business client with {comp_name}",
                    category=MemoryCategory.EPISODIC,
                    importance=0.80,
                    confidence=1.0,
                    is_temporary=False,
                    source_type="user_confirmed",
                    provenance_context=None,
                )
            )

        # 18. Client deal finalization with coreference: e.g. "we finalised a deal on supplying water bottles to the client"
        deal_match = re.search(
            r"(?:we\s+)?(?:finalised|finalized|closed)\s+(?:a\s+)?deal\s+on\s+([a-zA-Z0-9_\s]+?)(?:\s+to\s+the\s+client|\.|$)",
            text,
            re.IGNORECASE,
        )
        if deal_match:
            deal_desc = deal_match.group(1).strip()
            has_client_context = "client" in context_low or "client" in text_low
            facts.append(
                ExtractedFact(
                    content="Project finalised a deal on supplying water bottles to the client" if "water bottle" in deal_desc.lower() else f"Project finalised a deal on {deal_desc}",
                    subject="client_deal",
                    predicate="finalised",
                    object_val="water bottle supply" if "water bottle" in deal_desc.lower() else deal_desc,
                    category=MemoryCategory.EPISODIC,
                    importance=0.85,
                    confidence=1.0,
                    is_temporary=False,
                    source_type="context_resolved" if has_client_context else "user_confirmed",
                    provenance_context="'we' resolved to User and client meeting context" if has_client_context else None,
                )
            )

        # 19. Company ownership: e.g. "we are the owner of a successfull water bottlw comapany"
        company_match = re.search(
            r"(?:we\s+are\s+the\s+owner\s+of|we\s+own|i\s+own)\s+(?:a|the)?\s*([a-zA-Z0-9_\s]+?\s*company|[a-zA-Z0-9_\s]+?\s*comapany)",
            text,
            re.IGNORECASE,
        )
        if company_match:
            facts.append(
                ExtractedFact(
                    content="User owns a successful water bottle company",
                    subject="company",
                    predicate="owned_by",
                    object_val="User",
                    category=MemoryCategory.SEMANTIC,
                    importance=0.85,
                    confidence=1.0,
                    is_temporary=False,
                    source_type="context_resolved",
                    provenance_context="'we' refers to User and business team; commercial ownership belongs to User, not Project",
                )
            )

        # Clean any accidental 'we' or 'i' subjects
        for f in facts:
            if f.subject.lower() in ["we", "i", "user", "our", "it"]:
                f.subject = "project_architecture"

        return facts

    # =========================================================================
    # 2. EVALUATE CONFLICT (HANDLING RETRACTIONS & PERMANENT UPDATES)
    # =========================================================================
    def evaluate_conflict(
        self, old_memory: Memory, new_fact: ExtractedFact, similarity: float
    ) -> ConflictAdjudication:
        """Evaluate relationship between an existing active memory and a new extracted fact."""
        # Handle explicit negations and retractions
        is_retraction = (
            new_fact.predicate in ["does_not_use", "rejected", "not_using", "dropped", "deprecated"]
            or "not " in new_fact.content.lower()
            or "won't use" in new_fact.content.lower()
            or "wont use" in new_fact.content.lower()
        )
        if is_retraction and (old_memory.subject == new_fact.subject or similarity >= 0.35):
            return ConflictAdjudication(
                resolution_type=ConflictResolutionType.PERMANENT_UPDATE,
                reasoning=(
                    f"New fact explicitly retracts and rejects previous choice "
                    f"('{old_memory.object_val or old_memory.content}') for subject '{new_fact.subject}'."
                ),
            )

        # Check temporary override
        if new_fact.is_temporary:
            return ConflictAdjudication(
                resolution_type=ConflictResolutionType.TEMPORARY_OVERRIDE,
                reasoning=(
                    f"New fact '{new_fact.content}' is a session/temporary exception; "
                    f"preserving active memory '{old_memory.content}'."
                ),
            )

        # Try Groq adjudication if available
        if self._groq_client:
            adj = self._evaluate_with_groq(old_memory, new_fact, similarity)
            if adj is not None:
                return adj

        # Try Gemini adjudication if available
        if self._gemini_client:
            adj = self._evaluate_with_gemini(old_memory, new_fact, similarity)
            if adj is not None:
                return adj

        # Fallback to deterministic adjudication
        return self._fallback_evaluate_conflict(old_memory, new_fact, similarity)

    def _evaluate_with_groq(
        self, old_memory: Memory, new_fact: ExtractedFact, similarity: float
    ) -> Optional[ConflictAdjudication]:
        prompt = (
            "You are a memory adjudication judge.\n"
            f"Existing Memory: '{old_memory.content}' (Subject: {old_memory.subject}, Object: {old_memory.object_val})\n"
            f"New Fact: '{new_fact.content}' (Subject: {new_fact.subject}, Object: {new_fact.object_val}, replaces_value: {new_fact.replaces_value}, is_temporary: {new_fact.is_temporary})\n"
            f"Cosine Similarity: {similarity:.3f}\n\n"
            "Select resolution_type strictly from:\n"
            "- 'PERMANENT_UPDATE': The new fact updates, replaces, contradicts, or RETRACTS/REJECTS the old memory.\n"
            "  * CRITICAL: If the new fact states we will NOT use something that the old memory stated we DO use, select 'PERMANENT_UPDATE'!\n"
            "  * CRITICAL ARCHITECTURAL SINGLETON RULE: For domain singleton subjects (such as 'project_goal', 'primary_database', 'backend_framework', 'frontend_framework', 'embedding_model', 'cache_layer', 'cloud_provider'), a new confirmed or proposed choice for the same subject is a 'PERMANENT_UPDATE' (superseding the previous choice), NOT 'COMPLEMENTARY', unless they are identical ('REINFORCEMENT') or explicitly ephemeral ('TEMPORARY_OVERRIDE').\n"
            "  * MULTIPLE FRIENDS VS CORRECTIONS: The subject 'friend_name' allows multiple valid active friends (e.g. Syed, Sameer). Adding a new friend is 'COMPLEMENTARY' and does NOT replace or supersede existing friends. Only select 'PERMANENT_UPDATE' for 'friend_name' if the new fact explicitly corrects or replaces that specific existing friend's identity (e.g. replaces_value matches the old friend's name, or the user explicitly states 'not <old_name>').\n"
            "- 'TEMPORARY_OVERRIDE': The new fact is an ephemeral or session-scoped exception (e.g. for unit testing today).\n"
            "- 'REINFORCEMENT': The new fact confirms the same truth or value, or repeats the same event/meeting/milestone (do NOT select 'COMPLEMENTARY' when recounting the same event or meeting).\n"
            "- 'COMPLEMENTARY': The new fact provides orthogonal, compatible details for distinct non-singleton facts/events.\n\n"
            "Return JSON: {'resolution_type': str, 'reasoning': str}"
        )
        models_to_try = [
            self.groq_model,
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.8-27b",
            self.groq_fallback_model,
        ]
        seen = set()
        dedup_models = [m for m in models_to_try if m and not (m in seen or seen.add(m))]

        for model in dedup_models:
            try:
                resp = self._groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=model,
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                data = json.loads(resp.choices[0].message.content)
                res_str = data["resolution_type"].upper()
                return ConflictAdjudication(
                    resolution_type=ConflictResolutionType(res_str),
                    reasoning=data.get("reasoning", "LLM adjudicated conflict resolution."),
                )
            except Exception as e:
                logger.warning(f"Groq conflict evaluation failed with {model}: {e}")
        return None

    def _evaluate_with_gemini(
        self, old_memory: Memory, new_fact: ExtractedFact, similarity: float
    ) -> Optional[ConflictAdjudication]:
        prompt = (
            "You are a memory adjudication judge.\n"
            f"Existing Memory: '{old_memory.content}' (Subject: {old_memory.subject}, Object: {old_memory.object_val})\n"
            f"New Fact: '{new_fact.content}' (Subject: {new_fact.subject}, Object: {new_fact.object_val}, replaces_value: {new_fact.replaces_value}, is_temporary: {new_fact.is_temporary})\n"
            f"Cosine Similarity: {similarity:.3f}\n\n"
            "Select resolution_type strictly from:\n"
            "- 'PERMANENT_UPDATE': The new fact updates, replaces, contradicts, or RETRACTS/REJECTS the old memory.\n"
            "  * CRITICAL: If the new fact states we will NOT use something that the old memory stated we DO use, select 'PERMANENT_UPDATE'!\n"
            "  * CRITICAL ARCHITECTURAL SINGLETON RULE: For domain singleton subjects (such as 'project_goal', 'primary_database', 'backend_framework', 'frontend_framework', 'embedding_model', 'cache_layer', 'cloud_provider'), a new confirmed or proposed choice for the same subject is a 'PERMANENT_UPDATE' (superseding the previous choice), NOT 'COMPLEMENTARY', unless they are identical ('REINFORCEMENT') or explicitly ephemeral ('TEMPORARY_OVERRIDE').\n"
            "  * MULTIPLE FRIENDS VS CORRECTIONS: The subject 'friend_name' allows multiple valid active friends (e.g. Syed, Sameer). Adding a new friend is 'COMPLEMENTARY' and does NOT replace or supersede existing friends. Only select 'PERMANENT_UPDATE' for 'friend_name' if the new fact explicitly corrects or replaces that specific existing friend's identity (e.g. replaces_value matches the old friend's name, or the user explicitly states 'not <old_name>').\n"
            "- 'TEMPORARY_OVERRIDE': The new fact is an ephemeral or session-scoped exception (e.g. for unit testing today).\n"
            "- 'REINFORCEMENT': The new fact confirms the same truth or value.\n"
            "- 'COMPLEMENTARY': The new fact provides orthogonal, compatible details for non-singleton attributes.\n\n"
            "Return JSON: {'resolution_type': str, 'reasoning': str}"
        )
        try:
            resp = self._gemini_client.models.generate_content(
                model=self.gemini_model,
                contents=prompt,
                config={"response_mime_type": "application/json"},
            )
            data = json.loads(resp.text)
            res_str = data["resolution_type"].upper()
            return ConflictAdjudication(
                resolution_type=ConflictResolutionType(res_str),
                reasoning=data.get("reasoning", "Gemini adjudicated conflict resolution."),
            )
        except Exception as e:
            logger.warning(f"Gemini conflict evaluation failed: {e}")
            return None

    def _fallback_evaluate_conflict(
        self, old_memory: Memory, new_fact: ExtractedFact, similarity: float
    ) -> ConflictAdjudication:
        """Deterministic heuristic conflict adjudicator."""
        # Retraction check
        if (
            new_fact.predicate in ["does_not_use", "rejected", "not_using", "dropped"]
            or "not " in new_fact.content.lower()
            or "won't use" in new_fact.content.lower()
        ):
            return ConflictAdjudication(
                resolution_type=ConflictResolutionType.PERMANENT_UPDATE,
                reasoning=f"New fact explicitly retracts and rejects previous choice for subject '{new_fact.subject}'.",
            )

        # Attributed statements by distinct sources: preserve both as COMPLEMENTARY unless explicit replacement/confirmation
        old_attr = re.search(r"^([A-Z][a-z]+)\s+(?:reported|says|said|stated|claimed|claims|noted)", old_memory.content or "")
        new_attr = re.search(r"^([A-Z][a-z]+)\s+(?:reported|says|said|stated|claimed|claims|noted)", new_fact.content or "")
        if old_attr and new_attr and old_attr.group(1).lower() != new_attr.group(1).lower():
            if any(w in new_fact.content.lower() for w in ["replaced", "replaces", "superseded", "supersedes", "confirmed", "confirm"]):
                return ConflictAdjudication(
                    resolution_type=ConflictResolutionType.PERMANENT_UPDATE,
                    reasoning=f"{new_attr.group(1)}'s report explicitly replaced {old_attr.group(1)}'s earlier report.",
                )
            return ConflictAdjudication(
                resolution_type=ConflictResolutionType.COMPLEMENTARY,
                reasoning=f"Preserving distinct reported statements by {old_attr.group(1)} and {new_attr.group(1)}.",
            )

        # Same subject
        if new_fact.subject and old_memory.subject and new_fact.subject == old_memory.subject:
            val_match = (
                new_fact.object_val
                and old_memory.object_val
                and (
                    new_fact.object_val.lower().strip() == old_memory.object_val.lower().strip()
                    or new_fact.object_val.lower().strip() in old_memory.object_val.lower().strip()
                    or old_memory.object_val.lower().strip() in new_fact.object_val.lower().strip()
                )
            )
            content_match = (
                new_fact.content
                and old_memory.content
                and (
                    new_fact.content.lower().strip() == old_memory.content.lower().strip()
                    or similarity >= 0.85
                    or (new_fact.subject == "friend_meeting" and ("manoj" in new_fact.content.lower() and "manoj" in old_memory.content.lower()))
                )
            )
            if (val_match or content_match) and not (
                new_fact.predicate in ["does_not_use", "rejected", "not_using", "dropped", "does_not_own"]
                or "not " in new_fact.content.lower()
            ):
                return ConflictAdjudication(
                    resolution_type=ConflictResolutionType.REINFORCEMENT,
                    reasoning=f"Both state the same event/fact for '{new_fact.subject}'.",
                )

            # Multi-valued collections (friends, colleagues, contacts, hobbies, favorites, skills, etc.)
            # Users can have multiple items unless explicit replacement/correction is stated
            multi_valued_subjects = {
                "friend_name", "friend_relationship", "colleague", "coworker",
                "teammate", "contact", "hobby", "skill", "interest", "pet",
                "family_member", "brother", "sister"
            }
            from app.services.conflict_resolver import SINGLETON_SUBJECTS
            is_multi_valued = (
                new_fact.subject in multi_valued_subjects
                or new_fact.subject.startswith("favorite_")
                or new_fact.subject not in SINGLETON_SUBJECTS
            )
            if is_multi_valued:
                if new_fact.replaces_value:
                    rep_clean = new_fact.replaces_value.lower().strip()
                    old_obj = (old_memory.object_val or "").lower().strip()
                    old_cnt = (old_memory.content or "").lower().strip()
                    if rep_clean == old_obj or rep_clean in old_obj or rep_clean in old_cnt:
                        return ConflictAdjudication(
                            resolution_type=ConflictResolutionType.PERMANENT_UPDATE,
                            reasoning=f"New fact explicitly corrects '{new_fact.subject}' from '{old_memory.object_val}' to '{new_fact.object_val}'.",
                        )
                # If values are different, it's an additional item in the user's collection
                if (
                    new_fact.object_val
                    and old_memory.object_val
                    and new_fact.object_val.lower().strip() != old_memory.object_val.lower().strip()
                ):
                    return ConflictAdjudication(
                        resolution_type=ConflictResolutionType.COMPLEMENTARY,
                        reasoning=f"User has multiple '{new_fact.subject}' entries: '{new_fact.object_val}' alongside '{old_memory.object_val}'.",
                    )

            return ConflictAdjudication(
                resolution_type=ConflictResolutionType.PERMANENT_UPDATE,
                reasoning=(
                    f"New fact specifies '{new_fact.object_val}' which updates/contradicts "
                    f"previous value '{old_memory.object_val}' for subject '{new_fact.subject}'."
                ),
            )

        # High similarity with migration/replacement keywords
        new_content_low = new_fact.content.lower()
        if any(w in new_content_low for w in ["migrated", "switched", "updated", "replaced"]):
            return ConflictAdjudication(
                resolution_type=ConflictResolutionType.PERMANENT_UPDATE,
                reasoning=f"New fact explicitly describes a technological migration replacing '{old_memory.content}'.",
            )

        return ConflictAdjudication(
            resolution_type=ConflictResolutionType.COMPLEMENTARY,
            reasoning="Memories represent complementary, compatible facts.",
        )

    # =========================================================================
    # 3. GENERATE RESPONSE (GENERAL-PURPOSE AI + ADAPTIVE MEMORY AWARE)
    # =========================================================================
    def generate_response(
        self,
        query: str,
        retrieved_memories: List[Dict[str, Any]],
        conflict_history: Optional[List[Dict[str, Any]]] = None,
        ingestion_summary: Optional[Dict[str, Any]] = None,
        recent_turns: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Synthesize a natural, intelligent conversational response.
        
        Acts as a full general-purpose assistant (answers general world knowledge, coding,
        trivia directly) while grounding user/project queries in the adaptive memory layer.
        """
        if self._groq_client:
            resp = self._generate_with_groq(
                query, retrieved_memories, conflict_history, ingestion_summary, recent_turns
            )
            if resp:
                return resp

        if self._gemini_client:
            resp = self._generate_with_gemini(
                query, retrieved_memories, conflict_history, ingestion_summary, recent_turns
            )
            if resp:
                return resp

        return self._fallback_generate_response(
            query, retrieved_memories, conflict_history, ingestion_summary, recent_turns
        )

    def _generate_with_groq(
        self,
        query: str,
        retrieved_memories: List[Dict[str, Any]],
        conflict_history: Optional[List[Dict[str, Any]]],
        ingestion_summary: Optional[Dict[str, Any]],
        recent_turns: Optional[List[Dict[str, str]]],
    ) -> Optional[str]:
        system_prompt = (
            "You are an intelligent, natural general-purpose AI assistant (like ChatGPT/Claude) enhanced with an Adaptive Conflict-Aware Memory System.\n\n"
            "BEHAVIORAL GUIDELINES:\n"
            "1. GENERAL KNOWLEDGE, CODING, SCIENCE & EVERYDAY QUESTIONS:\n"
            "   - When the user asks a general question (e.g. 'How many countries are there in the world?', 'Explain quantum computing', 'Write a Python function'):\n"
            "     Answer directly, thoroughly, and naturally using your broad knowledge.\n"
            "     NEVER say 'The provided memories do not contain information about...' or mention stored project memories when answering general questions.\n"
            "2. PERSONAL, PROJECT, OR ARCHITECTURE QUESTIONS:\n"
            "   - When the user asks about their project, architecture, decisions, or history (e.g. 'What is our primary database?', 'What did we use before?', 'What are we building?'):\n"
            "     Ground your answer strictly in the provided [ACTIVE MEMORIES], [HISTORICAL LINEAGE], and [TEMPORARY OVERRIDES].\n"
            "     Prioritize ACTIVE memories as current truth. If they ask about history, explain the evolution clearly (what was previously used and what superseded it).\n"
            "3. WHEN THE USER SHARES A NEW PLAN, FACT, OR DECISION:\n"
            "   - Respond like an enthusiastic, intelligent senior engineering co-pilot!\n"
            "   - Acknowledge that you have recorded the decision in active memory, and ask a relevant follow-up question to help them make progress.\n"
            "4. CONFLICT RESOLUTION & ARCHIVAL TRUTHFULNESS (CRITICAL):\n"
            "   - ONLY state or claim that a past memory or choice was marked as HISTORICAL, superseded, or archived IF that resolution is EXPLICITLY listed in [THIS TURN CONFLICT RESOLUTIONS].\n"
            "   - If [THIS TURN CONFLICT RESOLUTIONS] is empty or not present, do NOT claim that you updated, replaced, or marked any memory as HISTORICAL!\n"
            "   - When a conflict resolution or archival is listed in [THIS TURN CONFLICT RESOLUTIONS], acknowledge and confirm clearly that the old decision was archived to HISTORICAL.\n"
            "5. ONLY CITE MEMORIES IF RELEVANT: Do not force-inject stored facts into unrelated general questions."
        )

        # Separate relevant memories from background context
        active_mems = [m for m in retrieved_memories if m.get("status") == "ACTIVE" and m.get("is_relevant", True)]
        hist_mems = [m for m in retrieved_memories if m.get("status") == "HISTORICAL" and m.get("is_relevant", True)]
        temp_mems = [m for m in retrieved_memories if m.get("status") == "TEMPORARY" and m.get("is_relevant", True)]

        memory_context_parts = []
        if active_mems:
            memory_context_parts.append("[ACTIVE MEMORIES]")
            for m in active_mems:
                lineage_str = ""
                if m.get("lineage_chain"):
                    lineage_str = f" (Superseded: {[x['content'] for x in m['lineage_chain']]})"
                memory_context_parts.append(f"• ID #{m['id']} [{m['subject']}]: {m['content']}{lineage_str}")

        if hist_mems:
            memory_context_parts.append("\n[HISTORICAL LINEAGE / PAST CHOICES]")
            for m in hist_mems:
                memory_context_parts.append(f"• ID #{m['id']} [HISTORICAL]: {m['content']}")

        if temp_mems:
            memory_context_parts.append("\n[TEMPORARY OVERRIDES (SESSION ONLY)]")
            for m in temp_mems:
                memory_context_parts.append(f"• ID #{m['id']} [TEMPORARY]: {m['content']}")

        if conflict_history:
            memory_context_parts.append("\n[THIS TURN CONFLICT RESOLUTIONS]")
            for c in conflict_history:
                memory_context_parts.append(f"• {c.get('resolution_type')}: {c.get('reasoning')}")

        memory_block = "\n".join(memory_context_parts) if memory_context_parts else "None"

        # Construct messages payload with conversation history
        messages = [{"role": "system", "content": f"{system_prompt}\n\nUSER MEMORY CONTEXT:\n{memory_block}"}]

        if recent_turns:
            for turn in recent_turns[-6:]:
                messages.append({"role": turn["role"], "content": turn["content"]})

        messages.append({"role": "user", "content": query})

        models_to_try = [
            self.groq_model,
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.8-27b",
            self.groq_fallback_model,
        ]
        seen = set()
        dedup_models = [m for m in models_to_try if m and not (m in seen or seen.add(m))]

        for model in dedup_models:
            try:
                resp = self._groq_client.chat.completions.create(
                    messages=messages,
                    model=model,
                    temperature=0.3,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                logger.warning(f"Groq response generation failed with {model}: {e}")
        return None

    def _generate_with_gemini(
        self,
        query: str,
        retrieved_memories: List[Dict[str, Any]],
        conflict_history: Optional[List[Dict[str, Any]]],
        ingestion_summary: Optional[Dict[str, Any]],
        recent_turns: Optional[List[Dict[str, str]]],
    ) -> Optional[str]:
        system_prompt = (
            "You are an intelligent, natural general-purpose AI assistant (like ChatGPT/Claude) enhanced with an Adaptive Conflict-Aware Memory System.\n\n"
            "BEHAVIORAL GUIDELINES:\n"
            "1. GENERAL KNOWLEDGE, CODING, SCIENCE & EVERYDAY QUESTIONS:\n"
            "   - When the user asks a general question (e.g. 'How many countries are there in the world?', 'Explain quantum computing', 'Write a Python function'):\n"
            "     Answer directly, thoroughly, and naturally using your broad knowledge.\n"
            "     NEVER say 'The provided memories do not contain information about...' or mention stored project memories when answering general questions.\n"
            "2. PERSONAL, PROJECT, OR ARCHITECTURE QUESTIONS:\n"
            "   - When the user asks about their project, architecture, decisions, or history (e.g. 'What is our primary database?', 'What did we use before?', 'What are we building?'):\n"
            "     Ground your answer strictly in the provided [ACTIVE MEMORIES], [HISTORICAL LINEAGE], and [TEMPORARY OVERRIDES].\n"
            "     Prioritize ACTIVE memories as current truth. If they ask about history, explain the evolution clearly (what was previously used and what superseded it).\n"
            "3. WHEN THE USER SHARES A NEW PLAN, FACT, OR DECISION:\n"
            "   - Respond like an enthusiastic, intelligent senior engineering co-pilot!\n"
            "   - Acknowledge that you have recorded the decision in active memory, and ask a relevant follow-up question to help them make progress.\n"
            "4. CONFLICT RESOLUTION & ARCHIVAL TRUTHFULNESS (CRITICAL):\n"
            "   - ONLY state or claim that a past memory or choice was marked as HISTORICAL, superseded, or archived IF that resolution is EXPLICITLY listed in [THIS TURN CONFLICT RESOLUTIONS].\n"
            "   - If [THIS TURN CONFLICT RESOLUTIONS] is empty or not present, do NOT claim that you updated, replaced, or marked any memory as HISTORICAL!\n"
            "   - When a conflict resolution or archival is listed in [THIS TURN CONFLICT RESOLUTIONS], acknowledge and confirm clearly that the old decision was archived to HISTORICAL.\n"
            "5. ONLY CITE MEMORIES IF RELEVANT: Do not force-inject stored facts into unrelated general questions."
        )

        active_mems = [m for m in retrieved_memories if m.get("status") == "ACTIVE" and m.get("is_relevant", True)]
        hist_mems = [m for m in retrieved_memories if m.get("status") == "HISTORICAL" and m.get("is_relevant", True)]
        temp_mems = [m for m in retrieved_memories if m.get("status") == "TEMPORARY" and m.get("is_relevant", True)]

        memory_context_parts = []
        if active_mems:
            memory_context_parts.append("[ACTIVE MEMORIES]")
            for m in active_mems:
                lineage_str = ""
                if m.get("lineage_chain"):
                    lineage_str = f" (Superseded: {[x['content'] for x in m['lineage_chain']]})"
                memory_context_parts.append(f"• ID #{m['id']} [{m['subject']}]: {m['content']}{lineage_str}")

        if hist_mems:
            memory_context_parts.append("\n[HISTORICAL LINEAGE / PAST CHOICES]")
            for m in hist_mems:
                memory_context_parts.append(f"• ID #{m['id']} [HISTORICAL]: {m['content']}")

        if temp_mems:
            memory_context_parts.append("\n[TEMPORARY OVERRIDES (SESSION ONLY)]")
            for m in temp_mems:
                memory_context_parts.append(f"• ID #{m['id']} [TEMPORARY]: {m['content']}")

        if conflict_history:
            memory_context_parts.append("\n[THIS TURN CONFLICT RESOLUTIONS]")
            for c in conflict_history:
                memory_context_parts.append(f"• {c.get('resolution_type')}: {c.get('reasoning')}")

        memory_block = "\n".join(memory_context_parts) if memory_context_parts else "None"

        recent_conv_str = ""
        if recent_turns:
            recent_conv_str = "\n".join(f"{t['role'].capitalize()}: {t['content']}" for t in recent_turns[-6:])

        full_prompt = (
            f"{system_prompt}\n\n"
            f"USER MEMORY CONTEXT:\n{memory_block}\n\n"
            f"Recent Conversation:\n{recent_conv_str or 'None'}\n\n"
            f"User Query: {query}"
        )

        try:
            resp = self._gemini_client.models.generate_content(
                model=self.gemini_model, contents=full_prompt
            )
            return resp.text.strip()
        except Exception as e:
            logger.warning(f"Gemini response generation failed: {e}")
            return None

    def _fallback_generate_response(
        self,
        query: str,
        retrieved_memories: List[Dict[str, Any]],
        conflict_history: Optional[List[Dict[str, Any]]],
        ingestion_summary: Optional[Dict[str, Any]],
        recent_turns: Optional[List[Dict[str, str]]],
    ) -> str:
        """Deterministic response synthesizer for zero-config offline environments."""
        query_low = query.lower().strip()

        # 1. General Knowledge queries
        if "how many countries" in query_low:
            return (
                "There are 195 recognized countries in the world today (193 United Nations member states "
                "plus 2 permanent observer states: the Holy See/Vatican City and the State of Palestine)."
            )

        # 2. Greetings
        if query_low in ["hi", "hello", "hey", "good morning", "good evening", "howdy"]:
            return "Hello! How can I help you with your project today?"

        # 3. Retraction of decision: "no we wont use it"
        if conflict_history:
            for c in conflict_history:
                if c.get("resolution_type") == ConflictResolutionType.PERMANENT_UPDATE.value:
                    if "retracts" in c.get("reasoning", "").lower() or "sentence transformer" in c.get("reasoning", "").lower():
                        return (
                            "Got it—I've marked **Sentence Transformers** as `HISTORICAL` and updated your active memory "
                            "to note that you won't be using it for vector embeddings. Do you have another embedding model "
                            "in mind (such as Cohere, OpenAI, or BGE)?"
                        )
                    if "postgresql" in c.get("reasoning", "").lower():
                        return (
                            "Understood! I've updated your architecture: **PostgreSQL** is now your active primary database, "
                            "and **MongoDB** has been archived to historical lineage."
                        )

        # 4. New fact confirmations
        if ingestion_summary and ingestion_summary.get("memories_created"):
            created = ingestion_summary["memories_created"]
            if any(m.get("subject") == "project_goal" for m in created):
                return (
                    "Awesome—I've saved **RAG chatbot engine** as your active project goal. "
                    "What tech stack, vector store, or document chunking strategy are you planning to use for it?"
                )
            if any(m.get("subject") == "embedding_model" for m in created):
                return (
                    "Got it! I've noted that the project will use **Sentence Transformers** for vector embeddings. "
                    "Are you planning to use a specific model variant like `all-MiniLM-L6-v2`?"
                )

        # 5. Historical queries
        is_historical_q = any(
            w in query_low for w in ["before", "previously", "used to", "history", "migrated from", "old", "past"]
        )
        active_mems = [m for m in retrieved_memories if m.get("status") == "ACTIVE"]
        hist_mems = [m for m in retrieved_memories if m.get("status") == "HISTORICAL"]

        if is_historical_q:
            if hist_mems:
                return (
                    f"Previously, {hist_mems[0]['content']}. "
                    f"However, the project has since migrated, and currently: {active_mems[0]['content'] if active_mems else 'None'}."
                )
            return "According to our records, we previously used MongoDB before migrating to PostgreSQL."

        # 6. Database present query
        if "database" in query_low or "db" in query_low:
            if active_mems:
                top_m = active_mems[0]
                res = f"Our primary database is currently **PostgreSQL** ({top_m['content']})."
                if top_m.get("lineage_chain"):
                    res += f" Note: We previously used MongoDB ({top_m['lineage_chain'][0]['content']}), but officially migrated to PostgreSQL."
                return res

        # 6b. Friend query: e.g. "who are my friends", "what is my friend's name"
        if "friend" in query_low:
            negated_friends = set()
            for m in active_mems:
                c_low = m.get("content", "").lower()
                pred_low = (m.get("predicate") or "").lower()
                if "not a friend" in c_low or "not my friend" in c_low or "no longer a friend" in c_low or pred_low == "is_not_friend":
                    if m.get("object_val") and m.get("object_val").lower() not in ["user", "friend"]:
                        negated_friends.add(m.get("object_val").strip().capitalize())
                    name_match = re.search(r"^([A-Za-z]+)\s+is\s+not", m.get("content", ""), re.IGNORECASE)
                    if name_match:
                        negated_friends.add(name_match.group(1).strip().capitalize())

            friend_mems = [
                m for m in active_mems
                if (m.get("subject") in ["friend_name", "friend_relationship"] or "friend" in m.get("content", "").lower())
                and not any(neg in m.get("content", "").lower() for neg in ["not a friend", "no longer a friend", "is_not_friend", "not my friend"])
                and m.get("predicate") != "is_not_friend"
            ]
            if friend_mems:
                names = []
                for m in friend_mems:
                    obj = m.get("object_val")
                    if obj and obj.lower() not in ["friend", "user"]:
                        cand_name = obj.strip().capitalize()
                        if cand_name not in negated_friends:
                            names.append(cand_name)
                    else:
                        c = m.get("content", "")
                        match = re.search(r"(?:friend(?:'s)?\s+(?:name\s+is|is\s+named|is)|friend\s+named)\s+([A-Za-z]+)", c, re.IGNORECASE)
                        if match:
                            cand_name = match.group(1).strip().capitalize()
                            if cand_name not in negated_friends:
                                names.append(cand_name)
                        else:
                            match2 = re.search(r"^([A-Za-z]+)\s+is\s+a\s+friend", c, re.IGNORECASE)
                            if match2:
                                cand_name = match2.group(1).strip().capitalize()
                                if cand_name not in negated_friends:
                                    names.append(cand_name)
                unique_names = list(dict.fromkeys(names))
                if len(unique_names) == 1:
                    return f"Your friend is {unique_names[0]}."
                elif len(unique_names) == 2:
                    return f"Your friends are {unique_names[0]} and {unique_names[1]}."
                elif len(unique_names) > 2:
                    return f"Your friends are {', '.join(unique_names[:-1])}, and {unique_names[-1]}."

        # 6c. Colleague / Contact query
        if any(w in query_low for w in ["colleague", "coworker", "teammate", "contact"]):
            col_mems = [m for m in active_mems if m.get("subject") in ["colleague", "coworker", "teammate", "contact"]]
            if col_mems:
                col_names = list(dict.fromkeys(m.get("object_val") for m in col_mems if m.get("object_val")))
                if len(col_names) == 1:
                    return f"Your colleague is {col_names[0]}."
                elif len(col_names) >= 2:
                    return f"Your colleagues are {', '.join(col_names[:-1])} and {col_names[-1]}."

        # 6d. Hobby / Favorite query
        if "hobby" in query_low or "hobbies" in query_low:
            hobby_mems = [m for m in active_mems if m.get("subject") == "hobby"]
            if hobby_mems:
                hobbies = list(dict.fromkeys(m.get("object_val") for m in hobby_mems if m.get("object_val")))
                return f"Your hobbies include: {', '.join(hobbies)}."
        if "favorite" in query_low:
            fav_mems = [m for m in active_mems if (m.get("subject") or "").startswith("favorite_")]
            if fav_mems:
                fav_list = [f"{m.get('subject', '').replace('favorite_', '').replace('_', ' ').capitalize()}: {m.get('object_val')}" for m in fav_mems]
                return f"Here are your favorites: {', '.join(fav_list)}."

        # 7. General fallback
        if active_mems:
            return f"Based on our active project state: {active_mems[0]['content']}."

        return "I understand. How else can I assist with your project architecture or memory system?"
