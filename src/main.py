import os
import sqlite3
import uuid
import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from src.models import (
    TurnRequest, TurnResponse,
    RecallRequest, RecallResponse, Citation,
    SearchRequest, SearchResponse, SearchResult,
    UserMemoriesResponse, Memory,
    HealthResponse
)

app = FastAPI(title="Memory Service", version="0.2.0")

DB_PATH = "/app/data/memory.db"

os.makedirs("/app/data", exist_ok=True)

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
    conn.commit()

init_db()

AZURE_CONFIG = {
    "endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
    "api_key": os.environ.get("AZURE_OPENAI_KEY", ""),
    "deployment": os.environ.get("AZURE_DEPLOYMENT_NAME", "gpt-oss-120b")
}

async def extract_facts(messages: list, user_id: str, turn_id: str) -> list:
    if not AZURE_CONFIG["endpoint"] or not AZURE_CONFIG["api_key"]:
        return []

    try:
        from openai import AzureOpenAI
        client = AzureOpenAI(
            azure_endpoint=AZURE_CONFIG["endpoint"],
            api_key=AZURE_CONFIG["api_key"],
            api_version="2024-02-01"
        )

        messages_text = "\n".join([
            f"{m.get('role', 'user')}: {m.get('content', '')}"
            for m in messages
        ])

        prompt = f"""Extract structured facts from this conversation turn.

Return ONLY valid JSON array. Each fact should have:
- type: "fact" | "preference" | "opinion" | "event"
- key: short key like "location", "employment", "pet", "preference_food"
- value: the fact value
- confidence: 0.0-1.0

Examples:
- "I moved to Berlin" → {{"type": "fact", "key": "location", "value": "Berlin", "confidence": 0.95}}
- "I love pizza" → {{"type": "preference", "key": "food", "value": "pizza", "confidence": 0.9}}
- "I hate waiting" → {{"type": "opinion", "key": "patience", "value": "hates waiting", "confidence": 0.8}}

Conversation:
{messages_text}

Return JSON array of facts found (0-5 facts)."""

        response = client.chat.completions.create(
            model=AZURE_CONFIG["deployment"],
            messages=[
                {"role": "system", "content": "You extract structured facts from conversations. Return ONLY valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )

        content = response.choices[0].message.content
        facts = json.loads(content)
        if not isinstance(facts, list):
            facts = []
        return facts

    except Exception as e:
        print(f"Extraction error: {e}")
        return []


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
            (user_id, fact_key)
        )
        existing = cur.fetchone()

        supersedes = None
        if existing:
            conn.execute(
                "UPDATE facts SET active=0, updated_at=? WHERE id=?",
                (now, existing["id"])
            )
            supersedes = existing["id"]

        fact_id = str(uuid.uuid4())
        conn.execute("""
            INSERT INTO facts (id, user_id, session_id, turn_id, type, key, value, content, confidence, active, supersedes, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
        """, (
            fact_id, user_id, session_id, turn_id,
            fact.get("type", "fact"),
            fact_key,
            fact.get("value", ""),
            json.dumps(fact),
            fact.get("confidence", 0.9),
            supersedes,
            now, now
        ))

    conn.commit()


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok")


@app.post("/turns", response_model=TurnResponse, status_code=201)
async def create_turn(req: TurnRequest):
    turn_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat() + "Z"

    conn.execute("""
        INSERT INTO turns (id, session_id, user_id, messages, timestamp, metadata, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        turn_id,
        req.session_id,
        req.user_id,
        json.dumps([m.model_dump() for m in req.messages]),
        req.timestamp,
        json.dumps(req.metadata),
        now
    ))
    conn.commit()

    if req.user_id:
        messages = [m.model_dump() for m in req.messages]
        facts = await extract_facts(messages, req.user_id, turn_id)
        if facts:
            await store_facts(facts, req.user_id, req.session_id, turn_id)

    return TurnResponse(id=turn_id)


@app.post("/recall", response_model=RecallResponse)
async def recall(req: RecallRequest):
    user_id = req.user_id or "unknown"
    session_id = req.session_id
    query = req.query.lower()

    context_parts = []
    citations = []

    query_keywords = set(query.split())

    cur = conn.execute("""
        SELECT id, type, key, value, content, created_at
        FROM facts
        WHERE user_id=? AND active=1
        ORDER BY created_at DESC
        LIMIT 50
    """, (user_id,))

    facts = cur.fetchall()
    relevant_facts = []
    for fact in facts:
        fact_dict = dict(fact)
        fact_text = f"{fact_dict['key']} {fact_dict['value']}".lower()
        if any(kw in fact_text for kw in query_keywords) or not query_keywords:
            relevant_facts.append(fact_dict)
            citations.append(Citation(
                turn_id=fact_dict['id'],
                score=1.0,
                snippet=fact_dict['value']
            ))

    if relevant_facts:
        context_parts.append("## Known facts about this user")
        for fact in relevant_facts[:10]:
            context_parts.append(f"- {fact['key']}: {fact['value']}")

    cur = conn.execute("""
        SELECT id, messages, timestamp FROM turns
        WHERE user_id=? OR session_id=?
        ORDER BY created_at DESC
        LIMIT 10
    """, (user_id, session_id))

    turns = cur.fetchall()
    for turn in turns:
        msgs = json.loads(turn["messages"])
        for msg in msgs:
            content = msg.get("content", "").lower()
            if any(kw in content for kw in query_keywords):
                context_parts.append(f"\n## From conversation ({turn['timestamp']})")
                context_parts.append(f"- {msg['role']}: {msg['content'][:200]}")
                citations.append(Citation(
                    turn_id=turn["id"],
                    score=0.8,
                    snippet=msg['content'][:100]
                ))
                break

    cur = conn.execute("""
        SELECT messages, timestamp FROM turns
        WHERE session_id=? OR user_id=?
        ORDER BY created_at DESC
        LIMIT 5
    """, (session_id, user_id))

    recent_turns = cur.fetchall()
    if recent_turns:
        context_parts.append("\n## Recent conversations")
        for turn in recent_turns:
            msgs = json.loads(turn["messages"])
            for msg in msgs[-2:]:
                if msg.get("content"):
                    context_parts.append(f"- [{turn['timestamp']}] {msg['role']}: {msg['content'][:100]}")

    context = "\n".join(context_parts)

    tokens_estimate = len(context) // 4
    if tokens_estimate > req.max_tokens:
        lines = context.split("\n")
        context = "\n".join(lines[:10])

    if not context:
        context = ""

    return RecallResponse(context=context, citations=citations)


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    user_id = req.user_id or "%"
    session_id = req.session_id or "%"

    cur = conn.execute("""
        SELECT f.value, f.key, f.user_id, f.session_id, t.timestamp, 1.0 as score
        FROM facts f
        LEFT JOIN turns t ON f.turn_id = t.id
        WHERE (f.user_id LIKE ? OR f.user_id IS NULL)
        AND (f.session_id LIKE ? OR ? = '%')
        AND (f.value LIKE ? OR f.key LIKE ?)
        LIMIT ?
    """, (user_id, session_id, session_id, f"%{req.query}%", f"%{req.query}%", req.limit))

    results = []
    for row in cur.fetchall():
        results.append(SearchResult(
            content=row["value"],
            score=row["score"],
            session_id=row["session_id"] or "",
            timestamp=row["timestamp"] or "",
            metadata={"key": row["key"]}
        ))

    return SearchResponse(results=results)


@app.get("/users/{user_id}/memories", response_model=UserMemoriesResponse)
async def get_user_memories(user_id: str):
    cur = conn.execute("""
        SELECT id, type, key, value, confidence, session_id, turn_id, created_at, updated_at, supersedes, active
        FROM facts
        WHERE user_id=?
        ORDER BY created_at DESC
    """, (user_id,))

    memories = []
    for row in cur.fetchall():
        memories.append(Memory(
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
            active=bool(row["active"])
        ))

    return UserMemoriesResponse(memories=memories)


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    conn.execute("DELETE FROM facts WHERE session_id=?", (session_id,))
    conn.execute("DELETE FROM turns WHERE session_id=?", (session_id,))
    conn.commit()
    return JSONResponse(status_code=204)


@app.delete("/users/{user_id}")
async def delete_user(user_id: str):
    conn.execute("DELETE FROM facts WHERE user_id=?", (user_id,))
    conn.execute("DELETE FROM turns WHERE user_id=?", (user_id,))
    conn.commit()
    return JSONResponse(status_code=204)