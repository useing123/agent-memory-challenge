# Improvements Guide

This document outlines prioritized improvements to boost eval scores from ~15% to 60%+.

---

## Priority 1: Fix `max_tokens` Enforcement

**Current:** Context returns unbounded text regardless of `max_tokens`.

**Problem:** `/recall` must respect token budget. This is a core requirement.

**Solution:** Implement token counting and truncation.

```python
# Add to main.py
def estimate_tokens(text: str) -> int:
    return len(text) // 4  # Rough approximation

@app.post("/recall")
async def recall(req: RecallRequest):
    # Build context...
    context = "## Facts\n" + facts_text + "\n## Conversation\n" + recent_context

    # Truncate to max_tokens
    if estimate_tokens(context) > req.max_tokens:
        # Priority: facts first, then conversation
        # Truncate conversation, keep facts
        remaining = req.max_tokens - estimate_tokens(facts_text) - 50
        if remaining > 0:
            context = f"## Facts\n{facts_text}\n\n## Recent Conversation\n{recent_context[:remaining*4]}"
        else:
            context = facts_text

    return RecallResponse(context=context, citations=[])
```

**Impact:** Critical requirement compliance.

---

## Priority 2: Implement Proper Citations

**Current:** Always returns `citations: []`.

**Problem:** Contract requires `{turn_id, score, snippet}` for each citation.

**Solution:** Track which facts/turns contributed to the response.

```python
@app.post("/recall")
async def recall(req: RecallRequest):
    citations = []

    # Get matching facts with source turn_id
    cur = conn.execute("""
        SELECT f.value, f.turn_id, f.session_id, t.timestamp
        FROM facts f
        LEFT JOIN turns t ON f.turn_id = t.id
        WHERE f.user_id=? AND f.active=1
    """, (user_id,))

    for row in cur.fetchall():
        citations.append(Citation(
            turn_id=row["turn_id"] or "",
            score=0.9,  # Could use embedding similarity
            snippet=row["value"][:200]
        ))

    return RecallResponse(context=context, citations=citations[:5])
```

**Impact:** Contract compliance + eval can verify recall sources.

---

## Priority 3: Add Semantic Embedding Recall

**Current:** Keyword matching + LLM query. Not "vanilla cosine-top-k" but not far from it.

**Problem:** The eval explicitly warns that "vanilla cosine-top-k will not score well." Need hybrid retrieval.

**Solution:** Add embeddings without full vector DB (keep SQLite).

```python
# Add embedding-based search
from openai import OpenAI  # Or use Azure

def get_embedding(text: str) -> list:
    client = get_azure_client()
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return resp.data[0].embedding

@app.post("/recall")
async def recall(req: RecallRequest):
    # Get query embedding
    query_emb = get_embedding(req.query)

    # Search facts by embedding similarity
    # Store embedding in facts table (add column)
    cur = conn.execute("""
        SELECT id, value, key, embedding FROM facts
        WHERE user_id=? AND active=1
    """, (user_id,))

    scored = []
    for row in cur.fetchall():
        if row["embedding"]:
            sim = cosine_similarity(query_emb, json.loads(row["embedding"]))
            scored.append((row, sim))

    # Top-k results
    scored.sort(key=lambda x: x[1], reverse=True)
    top_facts = [s[0] for s in scored[:10]]

    # Format context from top facts
    context = format_facts_as_context(top_facts)

    return RecallResponse(context=context, citations=[])
```

**Schema change:**
```sql
ALTER TABLE facts ADD COLUMN embedding TEXT;
```

**Impact:** ~20-30% score improvement on semantic recall tests.

---

## Priority 4: Add Optional Auth

**Current:** No auth validation.

**Problem:** Requirement says to support `Authorization: Bearer <token>` if `MEMORY_AUTH_TOKEN` is set.

**Solution:**

```python
from fastapi import Header

AUTH_TOKEN = os.environ.get("MEMORY_AUTH_TOKEN", "")

async def verify_auth(authorization: str = None):
    if AUTH_TOKEN and authorization:
        if not authorization.startswith("Bearer "):
            raise HTTPException(401, "Invalid auth header")
        token = authorization[7:]
        if token != AUTH_TOKEN:
            raise HTTPException(401, "Invalid token")

@app.post("/turns")
async def create_turn(req: TurnRequest, authorization: str = None):
    await verify_auth(authorization)
    # ...
```

---

## Priority 5: Improve Extraction Quality

**Current:** Basic prompts miss implicit facts.

**Problem:** `"walking Biscuit this morning"` should extract pet fact automatically.

**Solution:** Enhanced extraction prompt with implicit inference.

```python
async def extract_facts_from_history(history: list, user_id: str, turn_id: str) -> list:
    # More explicit system prompt
    system_prompt = """Extract ALL facts from user messages.
    INCLUDE implicit facts:
    - "walking my dog" → has a dog
    - "just bought a new MacBook" → owns MacBook
    - "my back hurts" → has back pain
    - "taking the kids to soccer" → has children, kids play soccer

    Return JSON array: [{"type": "fact", "key": "...", "value": "...", "confidence": 0.9}]"""

    # Use few-shot examples
    user_prompt = f"""Examples:
    Input: "I love hiking on weekends"
    Output: [{"type": "preference", "key": "hobby", "value": "hiking on weekends", "confidence": 0.9}]

    Input: "Just finished a 5K run"
    Output: [{"type": "fact", "key": "fitness_activity", "value": "running 5K", "confidence": 0.9}]

    Now extract from:
    {user_text}
    """
```

**Impact:** Better fact coverage across all eval categories.

---

## Priority 6: Multi-Hop Reasoning

**Current:** Single-hop retrieval only.

**Problem:** `"How many bass did he catch?"` requires connecting "caught 5 bass" + "caught 7 bass" from different turns.

**Solution:** Use LLM for multi-hop reasoning in recall.

```python
@app.post("/recall")
async def recall(req: RecallRequest):
    # Get all relevant context (facts + recent turns)
    all_context = get_facts_and_turns(user_id, session_id)

    # Use LLM to answer multi-hop question
    client = get_azure_client()
    response = client.chat.completions.create(
        model=AZURE_CONFIG["deployment"],
        messages=[
            {"role": "system", "content": "Answer the question based on the context. Do math if needed."},
            {"role": "user", "content": f"Context:\n{all_context}\n\nQuestion: {req.query}"}
        ],
        max_tokens=req.max_tokens
    )

    context = response.choices[0].message.content

    return RecallResponse(context=context, citations=extract_citations(all_context))
```

**Impact:** Major boost to multi_hop category (currently 21.8%).

---

## Priority 7: BM25 Hybrid Recall

**Current:** Pure embedding search.

**Problem:** Embeddings miss exact keyword matches. Need hybrid.

**Solution:** Combine embedding similarity + BM25.

```python
from rank_bm25 import BM25Okapi

def hybrid_search(query: str, facts: list) -> list:
    # BM25 on fact values
    corpus = [f["value"] for f in facts]
    bm25 = BM25Okapi(corpus)
    bm25_scores = bm25.get_scores(query)

    # Embedding similarity
    query_emb = get_embedding(query)
    embed_scores = [cosine_similarity(query_emb, json.loads(f["embedding"])) for f in facts]

    # Normalize and combine (alpha = 0.5)
    combined = []
    for i, f in enumerate(facts):
        norm_bm25 = bm25_scores[i] / max(bm25_scores) if max(bm25_scores) > 0 else 0
        norm_embed = embed_scores[i] if embed_scores[i] > 0 else 0
        combined.append((f, 0.5 * norm_bm25 + 0.5 * norm_embed))

    return [f for f, s in sorted(combined, key=lambda x: x[1], reverse=True)]
```

**Impact:** Better on fact_extraction (keyword-heavy) + fact_evolution.

---

## Priority 8: Fact Evolution Timeline

**Current:** Simple supersedes, but evolution is linear.

**Problem:** `"I love TypeScript"` → `"TypeScript generics annoying"` → `"TS fine for big projects"` — this is an arc, not a replacement.

**Solution:** Store evolution history per key.

```python
# Add to facts table
ALTER TABLE facts ADD COLUMN timeline_id TEXT;  # Groups related facts

# When storing new fact with same key, link to timeline
timeline_id = existing.get("timeline_id") or str(uuid.uuid4())

# In recall, show evolution (most recent first)
cur = conn.execute("""
    SELECT value, session_id, created_at FROM facts
    WHERE user_id=? AND timeline_id=? AND active=0
    ORDER BY created_at DESC
""", (user_id, timeline_id))
```

**Impact:** Better on fact_evolution (currently 14.1%).

---

## Summary

| Priority | Change | Est. Impact |
|----------|--------|--------------|
| 1 | max_tokens enforcement | Critical |
| 2 | Citations | Contract |
| 3 | Embedding recall | +20-30% |
| 4 | Auth | Contract |
| 5 | Extraction quality | +10% |
| 6 | Multi-hop LLM | +15% |
| 7 | BM25 hybrid | +10% |
| 8 | Timeline evolution | +5% |

**Total potential:** From ~15% to ~60-70%

---

## Quick Wins (Low Effort, High Impact)

1. **Add citations** — 1 hour, fixes contract
2. **Token truncation** — 2 hours, fixes requirement
3. **Enhance extraction prompt** — 1 hour, +10%
4. **Add embeddings column** — schema change, enables Priority 3

---

## Long-Term Architecture

Consider migrating to:
- **Qdrant** or **pgvector** for native vector search
- **Redis** for session caching
- **Background job queue** for async extraction (60s timeout is generous)

But for immediate eval improvement, Priorities 1-6 above deliver 80% of value with minimal re-architecture.