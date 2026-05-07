import json
import math
import os
import sqlite3
import uuid
import time
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Optional
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

_token_encoder = None

langfuse_client = None

def get_langfuse():
    global langfuse_client
    if langfuse_client is None:
        try:
            from langfuse import get_client
            langfuse_client = get_client()
        except ImportError:
            logger.warning("Langfuse not installed")
            langfuse_client = False
        except Exception as e:
            logger.warning(f"Langfuse init failed: {e}")
            langfuse_client = False
    return langfuse_client if langfuse_client else None


class APIError(Exception):
    def __init__(self, message: str, status_code: int = 500, details: Optional[dict] = None):
        self.message = message
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


async def api_error_handler(request: Request, exc: APIError):
    logger.error(f"API Error: {exc.message} | Details: {exc.details}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.message,
            "details": exc.details,
            "path": str(request.url.path)
        }
    )


async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "path": str(request.url.path)
        }
    )


class TracingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        trace_id = str(uuid.uuid4())[:8]
        request.state.trace_id = trace_id
        start_time = time.time()
        
        # Log request
        print(f"[{trace_id}] {request.method} {request.url.path}")
        
        # Capture request body for input
        request_body = ""
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                if body:
                    request_body = body.decode('utf-8')[:500]
            except:
                pass
        
        # Try to create Langfuse trace
        lf = get_langfuse()
        request.state.langfuse_span = None
        if lf:
            try:
                with lf.start_as_current_observation(as_type="span", name=f"{request.method} {request.url.path}") as span:
                    request.state.langfuse_span = span
                    
                    # Set input
                    if request_body:
                        span.update(input=request_body)
                    
                    print(f"[{trace_id}] Langfuse trace started")
                    
                    response = await call_next(request)
                    
                    # Try to get response body
                    try:
                        body = b""
                        async for chunk in response.body_iterator:
                            body += chunk
                        
                        # Decode and limit response body
                        response_text = body.decode('utf-8')[:1000] if body else ""
                        if response_text:
                            span.update(output=response_text)
                        else:
                            span.update(output={"status": response.status_code})
                    except Exception as e:
                        # Fallback: just status
                        span.update(output={"status": response.status_code})
                    
                    # Recreate response with original body
                    from starlette.responses import Response
                    response = Response(
                        content=body,
                        status_code=response.status_code,
                        headers=dict(response.headers),
                        media_type=response.media_type
                    )
                    
                    print(f"[{trace_id}] Langfuse trace ended")
            except Exception as e:
                print(f"Langfuse error: {e}")
                response = await call_next(request)
        else:
            response = await call_next(request)
        
        duration = time.time() - start_time
        print(f"[{trace_id}] {response.status_code} | {duration:.3f}s")
        
        return response


def log_trace_step(request: Request, step: str, **kwargs):
    """Log step with trace_id."""
    trace_id = getattr(request.state, 'trace_id', 'unknown')
    params = " ".join([f"{k}={v}" for k, v in kwargs.items()])
    print(f"[{trace_id}] {step} | {params}")
    
    # Also log to Langfuse if available
    if hasattr(request.state, 'langfuse_span'):
        try:
            request.state.langfuse_span.update(metadata={**kwargs, "step": step})
        except Exception as e:
            print(f"Langfuse span update failed: {e}")


def get_token_encoder():
    global _token_encoder
    if _token_encoder is None:
        try:
            import tiktoken
            _token_encoder = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            logger.warning("tiktoken not installed, using fallback token count")
            _token_encoder = None
        except Exception as e:
            logger.warning(f"Failed to load tiktoken: {e}")
            _token_encoder = None
    return _token_encoder


def count_tokens(text: str) -> int:
    enc = get_token_encoder()
    if enc:
        return len(enc.encode(text))
    return int(len(text) * 0.3)


def truncate_to_max_tokens(context: str, max_tokens: int) -> str:
    if count_tokens(context) <= max_tokens:
        return context
    lines = context.split("\n")
    result = []
    for line in lines:
        if count_tokens("\n".join(result + [line])) > max_tokens:
            break
        result.append(line)
    return "\n".join(result)


IMPLICIT_AGENT_THRESHOLD = 60000

from src.models import (
    Citation,
    HealthResponse,
    Memory,
    RecallRequest,
    RecallResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    TurnRequest,
    TurnResponse,
    UserMemoriesResponse,
)

app = FastAPI(title="Memory Service", version="0.2.0")
app.add_middleware(TracingMiddleware)
app.add_exception_handler(APIError, api_error_handler)
app.add_exception_handler(Exception, generic_exception_handler)

# Use local data directory
DATA_DIR = os.environ.get("DATA_DIR", "/tmp/memory_data")
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = f"{DATA_DIR}/memory.db"

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
conn.row_factory = sqlite3.Row


def init_db():
    conn.execute("""
        CREATE TABLE IF NOT EXISTS turns (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            user_id TEXT,
            messages TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            metadata TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS facts (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            turn_id TEXT NOT NULL,
            type TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            content TEXT,
            confidence REAL DEFAULT 0.9,
            active INTEGER DEFAULT 1,
            supersedes TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_facts_user ON facts(user_id)
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_facts_active ON facts(user_id, active)
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_docs (
            user_id TEXT PRIMARY KEY,
            full_doc TEXT NOT NULL,
            sections TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    # New table: full conversation history per user
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_conversations (
            user_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            turn_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (user_id, session_id)
        )
    """)
    # MD chunks with embeddings for semantic search
    conn.execute("""
        CREATE TABLE IF NOT EXISTS md_chunks (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            section TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_md_chunks_user ON md_chunks(user_id)")
    conn.commit()


init_db()

AZURE_CONFIG = {
    "endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
    "api_key": os.environ.get("AZURE_OPENAI_KEY", ""),
    "deployment": os.environ.get("AZURE_DEPLOYMENT_NAME", "gpt-oss-120b"),
}

_azure_client = None


def get_azure_client():
    global _azure_client
    if not _azure_client and AZURE_CONFIG["endpoint"] and AZURE_CONFIG["api_key"]:
        try:
            from openai import AzureOpenAI
            _azure_client = AzureOpenAI(
                azure_endpoint=AZURE_CONFIG["endpoint"],
                api_key=AZURE_CONFIG["api_key"],
                api_version="2024-02-01",
            )
        except ImportError:
            logger.warning("openai package not installed")
        except Exception as e:
            logger.warning(f"Failed to initialize Azure client: {e}")
    return _azure_client


async def extract_facts_from_history(history: list, user_id: str, turn_id: str) -> list:
    """Extract facts from full conversation history using multiple passes."""
    client = get_azure_client()
    if not client:
        return []

    if not history:
        return []

    user_messages = []
    for turn_data in history:
        turn_json = turn_data.get("messages", [])
        for m in turn_json:
            if m.get("role") == "user":
                content = m.get("content", "")
                if content:
                    user_messages.append(content)

    user_text = "\n".join([f"[USER]: {msg}" for msg in user_messages])

    if len(user_text) > 15000:
        user_text = user_text[-15000:]

    try:
        # Pass 1
        response = client.chat.completions.create(
            model=AZURE_CONFIG["deployment"],
            messages=[
                {
                    "role": "system",
                    "content": "Extract ALL facts. Be exhaustive. Return ONLY valid JSON array.",
                },
                {
                    "role": "user",
                    "content": f"""Extract ONLY facts that USER explicitly mentioned about themselves.
Look for: times (XX minutes, XX hours), dates, numbers, prices, quantities, counts, locations, names, events, achievements, possessions, preferences, family, work, education, hobbies, health issues.

IMPORTANT: Only extract from USER messages. Ignore assistant responses.

Return ALL facts as JSON array.
Format: [{{"type": "fact", "key": "descriptive_key", "value": "exact fact", "confidence": 0.9}}]

USER MESSAGES:
{user_text}

Extract all facts (0-20):""",
                },
            ],
            temperature=0.1,
            max_tokens=3000,
        )

        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()

        facts = json.loads(content)
        if not isinstance(facts, list):
            facts = []

        print(f"Pass 1 extracted {len(facts)} facts")

        # Pass 2
        response2 = client.chat.completions.create(
            model=AZURE_CONFIG["deployment"],
            messages=[
                {"role": "system", "content": "Find specific details user mentioned."},
                {
                    "role": "user",
                    "content": f"""Search for any facts the user mentioned that might have been missed.
Look for: exact times (XX minutes XX seconds), dates, numbers, prices, quantities, counts, specific achievements, personal bests, family details, work history.
Only look in USER messages.

USER MESSAGES:
{user_text}

Return any missed facts as JSON array:""",
                },
            ],
            temperature=0.1,
            max_tokens=1500,
        )

        content2 = response2.choices[0].message.content.strip()
        if content2.startswith("```"):
            content2 = content2.split("```")[1]
            if content2.startswith("json"):
                content2 = content2[4:]
        content2 = content2.strip()

        more_facts = json.loads(content2)
        if isinstance(more_facts, list):
            facts.extend(more_facts)
            print(f"Pass 2 extracted {len(more_facts)} more facts")

        print(f"Total extracted {len(facts)} facts")
        return facts

    except (json.JSONDecodeError, Exception) as e:
        print(f"Extraction error: {e}")
        return []


def extract_facts_fallback(messages: list) -> list:
    """Simple rule-based fallback when LLM fails."""
    facts = []
    patterns = [
        (r"I live in (.+)", "location", 1),
        (r"I work at (.+)", "job", 1),
        (r"I work as (.+)", "job", 1),
        (r"I'm a (.+)", "job", 1),
        (r"I moved to (.+)", "location", 1),
        (r"my name is (.+)", "name", 1),
        (r"I bought (.+)", "purchase", 1),
    ]

    for msg in messages:
        if msg.get("role") != "user":
            continue
        text = msg.get("content", "")
        for pattern, ftype, conf in patterns:
            import re

            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                facts.append(
                    {
                        "type": ftype,
                        "key": ftype,
                        "value": match.group(1).strip(),
                        "confidence": conf,
                    }
                )
    return facts[:3]


def get_user_total_tokens(user_id: str) -> int:
    cur = conn.execute(
        """
        SELECT turn_json FROM user_conversations WHERE user_id=?
    """,
        (user_id,),
    )
    total = 0
    for row in cur.fetchall():
        try:
            data = json.loads(row["turn_json"])
            messages = data.get("messages", [])
            for m in messages:
                total += count_tokens(m.get("content", ""))
        except:
            pass
    return total


async def extract_implicit_facts(user_id: str, session_id: str) -> list:
    """Агент для извлечения неявных фактов из всей истории при достижении 60k токенов."""
    client = get_azure_client()
    if not client:
        return []

    total_tokens = get_user_total_tokens(user_id)
    if total_tokens < IMPLICIT_AGENT_THRESHOLD:
        return []

    print(f"Implicit agent triggered for {user_id}: {total_tokens} tokens")

    cur = conn.execute(
        """
        SELECT turn_json FROM user_conversations WHERE user_id=?
    """,
        (user_id,),
    )

    all_messages = []
    for row in cur.fetchall():
        try:
            data = json.loads(row["turn_json"])
            all_messages.extend(data.get("messages", []))
        except:
            pass

    if not all_messages:
        return []

    user_text = "\n".join(
        [
            f"[{m.get('role')}]: {m.get('content', '')}"
            for m in all_messages
            if m.get("content")
        ]
    )

    if len(user_text) > 50000:
        user_text = user_text[-50000:]

    try:
        response = client.chat.completions.create(
            model=AZURE_CONFIG["deployment"],
            messages=[
                {
                    "role": "system",
                    "content": "You're an agent who uncovers hidden facts. Analyze patterns and behavior.",
                },
                {
                    "role": "user",
                    "content": f"""You are an agent tasked with uncovering implicit facts. Analyze patterns and behavior. Examine the entire history and identify IMPLICIT facts and behavioral patterns.
                    Analyze the entire history and identify IMPLICIT facts and behavioral patterns. Implicit facts include:
                    - Recurring topics (asks about React every week)
                    - Patterns (back hurts several times → chronic pain)
                    - Preferences (always chooses certain restaurants)
                    - Behavioral patterns (gets tired after work, looks for news in the evenings)

                    Return a JSON array with the implicit facts:
                    [{{"type": "pattern", "key": "description", "value": "specific pattern", "confidence": 0.6}}]
                    History ({total_tokens} tokens):
                    {user_text}

                    Extract implicit facts (0–10):""",
                },
            ],
            temperature=0.2,
            max_tokens=4000,
        )

        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()

        facts = json.loads(content)
        if not isinstance(facts, list):
            return []

        print(f"Implicit agent extracted {len(facts)} patterns")
        return facts

    except Exception as e:
        print(f"Implicit agent error: {e}")
        return []


def chunk_md_by_headers(md_text: str) -> list[dict]:
    """Split MD by ## headers into chunks."""
    if not md_text:
        return []
    
    chunks = []
    current_section = "General"
    content = []
    
    for line in md_text.split('\n'):
        if line.startswith('## '):
            if content:
                chunks.append({
                    "section": current_section,
                    "content": "\n".join(content).strip()
                })
            current_section = line[3:].strip()
            content = []
        else:
            content.append(line)
    
    if content:
        chunks.append({
            "section": current_section,
            "content": "\n".join(content).strip()
        })
    
    return chunks


def get_embedding(text: str) -> list[float]:
    """Generate embedding using Azure text-embedding-3-large (3072 dim)."""
    try:
        client = get_azure_client()
        if not client:
            raise APIError("Azure client not available", status_code=503)

        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text
        )
        return response.data[0].embedding
    except APIError:
        raise
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        raise APIError("Failed to generate embedding", status_code=503, details={"error": str(e)})


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    if not a or not b:
        return 0.0
    
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


# Simple cache for query embeddings (TTL = 5 minutes)
_query_embedding_cache = {}
_CACHE_TTL_SECONDS = 300


def get_cached_query_embedding(query: str) -> Optional[list[float]]:
    """Get cached query embedding or None if not cached."""
    if query in _query_embedding_cache:
        return _query_embedding_cache[query]
    return None


def set_cached_query_embedding(query: str, embedding: list[float]):
    """Cache query embedding."""
    _query_embedding_cache[query] = embedding


def ensure_user_embeddings(user_id: str) -> bool:
    """Ensure user has embeddings, generate if missing. Returns True if embeddings exist."""
    cur = conn.execute("""
        SELECT COUNT(*) as cnt FROM md_chunks 
        WHERE user_id = ? AND embedding IS NOT NULL
    """, (user_id,))
    has_embeddings = cur.fetchone()["cnt"] > 0
    
    if has_embeddings:
        return True
    
    logger.info(f"Generating embeddings for user {user_id}...")
    
    # Get MD doc and create chunks if needed
    cur = conn.execute("SELECT full_doc FROM user_docs WHERE user_id=?", (user_id,))
    row = cur.fetchone()
    
    if row and row["full_doc"]:
        # Create chunks from MD
        chunks = chunk_md_by_headers(row["full_doc"])
        now = datetime.utcnow().isoformat() + "Z"
        
        # Delete old chunks and insert new ones
        conn.execute("DELETE FROM md_chunks WHERE user_id=?", (user_id,))
        
        for chunk in chunks:
            conn.execute("""
                INSERT INTO md_chunks (id, user_id, section, content, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (str(uuid.uuid4()), user_id, chunk['section'], chunk['content'], now))
    
    conn.commit()
    
    # Now generate embeddings
    cur = conn.execute("""
        SELECT id, section, content FROM md_chunks 
        WHERE user_id = ? AND embedding IS NULL
    """, (user_id,))
    
    for row in cur.fetchall():
        try:
            chunk_text = f"{row['section']}: {row['content']}"
            embedding = get_embedding(chunk_text)
            conn.execute("UPDATE md_chunks SET embedding = ? WHERE id = ?",
                        (json.dumps(embedding), row["id"]))
        except Exception as e:
            logger.warning(f"Embedding failed for chunk {row['id']}: {e}")
    
    conn.commit()
    logger.info(f"Embeddings generated for user {user_id}")
    return True
    
    conn.commit()
    logger.info(f"Embeddings generated for user {user_id}")
    return True


def semantic_search_chunks(query: str, user_id: str, top_k: int = 5) -> list[dict]:
    """Search MD chunks by embedding similarity with caching."""
    
    # Check query cache first
    query_emb = get_cached_query_embedding(query)
    if not query_emb:
        try:
            query_emb = get_embedding(query)
            set_cached_query_embedding(query, query_emb)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    # Ensure user has embeddings (lazy generation)
    ensure_user_embeddings(user_id)
    
    cur = conn.execute("""
        SELECT id, user_id, section, content, embedding 
        FROM md_chunks 
        WHERE user_id = ? AND embedding IS NOT NULL
    """, (user_id,))
    
    results = []
    for row in cur.fetchall():
        if row["embedding"]:
            try:
                chunk_emb = json.loads(row["embedding"])
                score = cosine_similarity(query_emb, chunk_emb)
                results.append({
                    "id": row["id"],
                    "section": row["section"],
                    "content": row["content"],
                    "score": score
                })
            except json.JSONDecodeError:
                continue
    
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def bm25_search_chunks(query: str, user_id: str, top_k: int = 5) -> list[dict]:
    """BM25 search fallback when embeddings fail."""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        return []
    
    cur = conn.execute("""
        SELECT id, section, content 
        FROM md_chunks 
        WHERE user_id = ?
    """, (user_id,))
    
    chunks = []
    for row in cur.fetchall():
        chunks.append({
            "id": row["id"],
            "section": row["section"],
            "content": row["content"]
        })
    
    if not chunks:
        return []
    
    corpus = [c["content"] for c in chunks]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query)
    
    results = []
    for i, chunk in enumerate(chunks):
        results.append({
            **chunk,
            "score": scores[i]
        })
    
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


async def store_facts(facts: list, user_id: str, session_id: str, turn_id: str):
    if not facts or not user_id:
        return

    now = datetime.utcnow().isoformat() + "Z"

    for fact in facts:
        if not isinstance(fact, dict):
            continue

        fact_key = fact.get("key", "").lower().strip()
        if not fact_key:
            continue

        cur = conn.execute(
            "SELECT id FROM facts WHERE user_id=? AND key=? AND active=1",
            (user_id, fact_key),
        )
        existing = cur.fetchone()

        supersedes = None
        if existing:
            conn.execute(
                "UPDATE facts SET active=0, updated_at=? WHERE id=?",
                (now, existing["id"]),
            )
            supersedes = existing["id"]

        fact_id = str(uuid.uuid4())
        conn.execute(
            """
            INSERT INTO facts (id, user_id, session_id, turn_id, type, key, value, content, confidence, active, supersedes, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
        """,
            (
                fact_id,
                user_id,
                session_id,
                turn_id,
                fact.get("type", "fact"),
                fact_key,
                str(fact.get("value", "")),
                json.dumps(fact),
                fact.get("confidence", 0.9),
                supersedes,
                now,
                now,
            ),
        )
    conn.commit()


async def update_user_doc_with_history(user_id: str):
    """Generate MD doc with session history format."""
    now = datetime.utcnow().isoformat() + "Z"
    
    # Get all sessions for this user
    cur = conn.execute(
        """
        SELECT session_id, turn_json, created_at FROM user_conversations
        WHERE user_id=?
        ORDER BY created_at ASC
    """,
        (user_id,),
    )
    sessions = cur.fetchall()

    if not sessions:
        return

    # Get facts
    cur = conn.execute(
        """
        SELECT key, value, type, session_id FROM facts
        WHERE user_id=? AND active=1
    """,
        (user_id,),
    )
    facts = cur.fetchall()

    # Group facts by key
    facts_by_key = {}
    for f in facts:
        key = f["key"]
        if key not in facts_by_key:
            facts_by_key[key] = []
        facts_by_key[key].append({"value": f["value"], "session_id": f["session_id"]})

    # Build MD with session history
    md_lines = [f"# User Profile: {user_id}", ""]

    # Group into sections
    sections = [
        "Employment",
        "Location",
        "Personal",
        "Hobbies",
        "Technology",
        "Preferences",
        "Other",
    ]

    for section in sections:
        md_lines.append(f"## {section}")

        # Find facts matching this section
        section_facts = []
        if section == "Employment":
            section_facts = facts_by_key.get("employment", []) + facts_by_key.get(
                "job", []
            )
        elif section == "Location":
            section_facts = facts_by_key.get("location", [])
        elif section == "Personal":
            section_facts = (
                facts_by_key.get("name", [])
                + facts_by_key.get("family", [])
                + facts_by_key.get("pet", [])
            )
        elif section == "Hobbies":
            section_facts = facts_by_key.get("hobby", []) + facts_by_key.get(
                "health", []
            )
        elif section == "Technology":
            section_facts = facts_by_key.get("technology", []) + facts_by_key.get(
                "education", []
            )
        elif section == "Preferences":
            section_facts = facts_by_key.get("preference_food", []) + facts_by_key.get(
                "preference_entertainment", []
            )

        for i, fact in enumerate(section_facts):
            is_current = " (current)" if i == len(section_facts) - 1 else ""
            md_lines.append(f"- [{fact['session_id']}] {fact['value']}{is_current}")

        if not section_facts:
            md_lines.append("- (no data)")
        md_lines.append("")

    md_doc = "\n".join(md_lines)

    conn.execute(
        """
        INSERT OR REPLACE INTO user_docs (user_id, full_doc, sections, updated_at)
        VALUES (?, ?, ?, ?)
    """,
        (user_id, md_doc, "{}", now),
    )

    # Generate chunks and embeddings immediately to fix the critical bug
    chunks = chunk_md_by_headers(md_doc)
    conn.execute("DELETE FROM md_chunks WHERE user_id=?", (user_id,))
    
    for chunk in chunks:
        if chunk["content"].strip():
            chunk_id = str(uuid.uuid4())
            embedding_json = None
            try:
                embedding = get_embedding(f"{chunk['section']}: {chunk['content']}")
                embedding_json = json.dumps(embedding)
            except Exception as e:
                logger.warning(f"Embedding failed for chunk section {chunk['section']}: {e}")

            conn.execute("""
                INSERT INTO md_chunks (id, user_id, section, content, embedding, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (chunk_id, user_id, chunk["section"], chunk["content"], embedding_json, now))

    conn.commit()
    logger.info(f"Updated MD for {user_id} with {len(sessions)} sessions and generated {len(chunks)} chunks.")


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok")


@app.post("/turns", response_model=TurnResponse, status_code=201)
async def create_turn(req: TurnRequest):
    try:
        turn_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat() + "Z"

        conn.execute(
            """
            INSERT INTO turns (id, session_id, user_id, messages, timestamp, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                turn_id,
                req.session_id,
                req.user_id,
                json.dumps([m.model_dump() for m in req.messages]),
                req.timestamp,
                json.dumps(req.metadata),
                now,
            ),
        )
        conn.commit()

        if req.user_id:
            turn_data = {
                "turn_id": turn_id,
                "session_id": req.session_id,
                "messages": [m.model_dump() for m in req.messages],
                "timestamp": req.timestamp,
                "metadata": req.metadata,
            }
            conn.execute(
                """
                INSERT OR REPLACE INTO user_conversations (user_id, session_id, turn_json, created_at)
                VALUES (?, ?, ?, ?)
            """,
                (req.user_id, req.session_id, json.dumps(turn_data), now),
            )
            conn.commit()

            try:
                full_history = get_user_conversation_history(req.user_id)
                messages = [m.model_dump() for m in req.messages]
                facts = await extract_facts_from_history(full_history, req.user_id, turn_id)

                if not facts:
                    facts = extract_facts_fallback(messages)
                    logger.info(f"Fallback extracted {len(facts)} facts")

                if facts:
                    await store_facts(facts, req.user_id, req.session_id, turn_id)
                    await update_user_doc_with_history(req.user_id)

                    implicit_facts = await extract_implicit_facts(req.user_id, req.session_id)
                    if implicit_facts:
                        await store_facts(implicit_facts, req.user_id, req.session_id, turn_id)
                        logger.info(f"Implicit agent stored {len(implicit_facts)} pattern facts")
            except Exception as e:
                logger.error(f"Fact extraction error: {e}")
                pass

        return TurnResponse(id=turn_id)
    except Exception as e:
        logger.exception(f"Failed to create turn: {e}")
        raise APIError("Failed to create turn", status_code=500, details={"error": str(e)})


def get_user_conversation_history(user_id: str) -> list:
    """Get all conversation turns for a user."""
    cur = conn.execute(
        """
        SELECT turn_json FROM user_conversations
        WHERE user_id=?
        ORDER BY created_at ASC
    """,
        (user_id,),
    )
    history = []
    for row in cur.fetchall():
        try:
            turn_data = json.loads(row["turn_json"])
            history.append(turn_data)
        except:
            pass
    return history


@app.post("/recall", response_model=RecallResponse)
async def recall(req: RecallRequest):
    user_id = req.user_id or "unknown"

    try:
        try:
            top_chunks = semantic_search_chunks(req.query, user_id, top_k=5)
            if top_chunks:
                # Instead of LLM answer generation, return raw chunks
                context = "\n\n".join([
                    f"## {c['section']}\n{c['content']}"
                    for c in top_chunks[:3]
                ])

                if req.max_tokens and count_tokens(context) > req.max_tokens:
                    context = truncate_to_max_tokens(context, req.max_tokens)

                citations = [
                    Citation(
                        turn_id=chunk.get("id", ""),
                        score=chunk.get("score", 0.9),
                        snippet=chunk["content"][:200]
                    )
                    for chunk in top_chunks[:3]
                ]

                logger.info(f"Recall (semantic): {len(top_chunks)} chunks, top score: {top_chunks[0].get('score', 0):.2f}")
                return RecallResponse(context=context, citations=citations)
        except Exception as e:
            logger.warning(f"Semantic search failed: {e}")

            try:
                top_chunks = bm25_search_chunks(req.query, user_id, top_k=5)
                if top_chunks:
                    context_parts = []
                    for chunk in top_chunks:
                        context_parts.append(f"## {chunk['section']}\n{chunk['content']}")
                    context = "\n\n".join(context_parts)

                    if req.max_tokens and count_tokens(context) > req.max_tokens:
                        context = truncate_to_max_tokens(context, req.max_tokens)

                    citations = [
                        Citation(
                            turn_id=chunk.get("id", ""),
                            score=chunk.get("score", 0.5),
                            snippet=chunk["content"][:200]
                        )
                        for chunk in top_chunks
                    ]

                    logger.info(f"BM25 fallback: {len(top_chunks)} chunks")
                    return RecallResponse(context=context, citations=citations)
            except Exception as bm25_e:
                logger.error(f"BM25 fallback failed: {bm25_e}")

        cur = conn.execute("""
            SELECT messages FROM turns
            WHERE user_id=? AND session_id=?
        """, (user_id, req.session_id))
        turn_result = cur.fetchone()

        context = ""
        if turn_result:
            try:
                msgs = json.loads(turn_result["messages"])
                context = "\n".join([
                    f'{m.get("role", "user").upper()}: "{m["content"]}"'
                    for m in msgs if m.get("content")
                ])
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to parse turn messages: {e}")

        if not context:
            cur = conn.execute("""
                SELECT messages FROM turns
                WHERE user_id=?
                ORDER BY created_at DESC
                LIMIT 1
            """, (user_id,))
            turn_result = cur.fetchone()
            if turn_result:
                try:
                    msgs = json.loads(turn_result["messages"])
                    context = "\n".join([
                        f'{m.get("role", "user").upper()}: "{m["content"]}"'
                        for m in msgs if m.get("content")
                    ])
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Failed to parse recent messages: {e}")

        if req.max_tokens and count_tokens(context) > req.max_tokens:
            context = truncate_to_max_tokens(context, req.max_tokens)

        return RecallResponse(context=context, citations=[])
    except Exception as e:
        logger.exception(f"Recall endpoint failed: {e}")
        raise APIError("Failed to recall memories", status_code=500, details={"error": str(e)})


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    try:
        user_id = req.user_id or "%"
        session_id = req.session_id or "%"

        cur = conn.execute(
            """
            SELECT f.value, f.key, f.user_id, f.session_id, t.timestamp, 1.0 as score
            FROM facts f
            LEFT JOIN turns t ON f.turn_id = t.id
            WHERE (f.user_id LIKE ? OR f.user_id IS NULL)
            AND (f.session_id LIKE ? OR ? = '%')
            AND (f.value LIKE ? OR f.key LIKE ?)
            LIMIT ?
        """,
            (
                user_id,
                session_id,
                session_id,
                f"%{req.query}%",
                f"%{req.query}%",
                req.limit,
            ),
        )

        results = []
        for row in cur.fetchall():
            results.append(
                SearchResult(
                    content=row["value"],
                    score=row["score"],
                    session_id=row["session_id"] or "",
                    timestamp=row["timestamp"] or "",
                    metadata={"key": row["key"]},
                )
            )

        return SearchResponse(results=results)
    except Exception as e:
        logger.exception(f"Search failed: {e}")
        raise APIError("Search failed", status_code=500, details={"error": str(e)})


@app.get("/users/{user_id}/memories", response_model=UserMemoriesResponse)
async def get_user_memories(user_id: str):
    try:
        cur = conn.execute(
            """
            SELECT id, type, key, value, confidence, session_id, turn_id, created_at, updated_at, supersedes, active
            FROM facts
            WHERE user_id=?
            ORDER BY created_at DESC
        """,
            (user_id,),
        )

        memories = []
        for row in cur.fetchall():
            memories.append(
                Memory(
                    id=row["id"],
                    type=row["type"],
                    key=row["key"],
                    value=row["value"],
                    confidence=row["confidence"],
                    source_session=row["session_id"],
                    source_turn=row["turn_id"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    supersedes=row["supersedes"],
                    active=bool(row["active"]),
                )
            )

        return UserMemoriesResponse(memories=memories)
    except Exception as e:
        logger.exception(f"Failed to get user memories: {e}")
        raise APIError("Failed to retrieve memories", status_code=500, details={"error": str(e), "user_id": user_id})


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    try:
        conn.execute("DELETE FROM facts WHERE session_id=?", (session_id,))
        conn.execute("DELETE FROM turns WHERE session_id=?", (session_id,))
        conn.execute("DELETE FROM user_conversations WHERE session_id=?", (session_id,))
        conn.commit()
        logger.info(f"Deleted session: {session_id}")
        return Response(status_code=204)
    except Exception as e:
        logger.exception(f"Failed to delete session {session_id}: {e}")
        raise APIError("Failed to delete session", status_code=500, details={"error": str(e), "session_id": session_id})


@app.delete("/users/{user_id}")
async def delete_user(user_id: str):
    try:
        conn.execute("DELETE FROM facts WHERE user_id=?", (user_id,))
        conn.execute("DELETE FROM turns WHERE user_id=?", (user_id,))
        conn.execute("DELETE FROM user_conversations WHERE user_id=?", (user_id,))
        conn.execute("DELETE FROM user_docs WHERE user_id=?", (user_id,))
        conn.execute("DELETE FROM md_chunks WHERE user_id=?", (user_id,))
        conn.commit()
        logger.info(f"Deleted user: {user_id}")
        return Response(status_code=204)
    except Exception as e:
        logger.exception(f"Failed to delete user {user_id}: {e}")
        raise APIError("Failed to delete user", status_code=500, details={"error": str(e), "user_id": user_id})
