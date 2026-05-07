# Memory Service

A Dockerized memory service for AI agents that persists conversation turns and provides recall capabilities.

## Quick Start

```bash
# Build and start
docker compose up -d

# Wait for health
until curl -sf http://localhost:8080/health; do sleep 1; done

# Run evaluation tests
python3 scripts/run_eval.py
```

## Project Structure

```
memory-service/
├── src/
│   ├── main.py          # FastAPI app, all 7 endpoints
│   └── models.py        # Pydantic models for request/response
├── scripts/
│   ├── run_eval.py      # Run 500 tests from fixtures
│   ├── generate_tests.py # Generate tests via LLM (not used)
│   └── download_dataset.py # Download LongMemEval
├── fixtures/
│   ├── eval_tests.json  # 500 test cases (LongMemEval)
│   └── test_data.json   # Manual test fixtures
├── tests/
│   └── test_contract.py # Contract tests (health, turns, recall, etc.)
├── Dockerfile           # Python 3.11 container
├── docker-compose.yml   # Runs on port 8080
├── .env.example         # Azure OpenAI keys (if needed)
├── CHANGELOG.md         # Iteration history
└── README.md            # This file
```

## What We've Built

### Current Status (v0.3.0)

| Component | Status |
|-----------|--------|
| All 7 HTTP endpoints | ✅ Working |
| Docker + volume | ✅ Working |
| Health check | ✅ Working |
| Evaluation fixtures | ✅ 500 tests |
| Extraction (LLM) | ❌ Not implemented |
| Fact storage | ❌ Not implemented |
| Recall with facts | ❌ Returns empty |

### Test Results

```
First 50 tests: 0% passed
Reason: No extraction — recall returns empty context
```

This is expected. The skeleton is working (endpoints respond correctly), but without extraction logic, there's no data to recall.

---

## Architecture (Planned)

```
POST /turns
    │
    ├─→ Save turn to storage
    │
    └─→ LLM extraction → Store facts
         │
         ├─ type: fact|preference|opinion|event
         ├─ key: employment|location|pet|...
         └─ value: "Berlin"|"Google"|"Biscuit"|...
         

POST /recall
    │
    ├─ Get user facts (active only)
    ├─ Get recent session messages
    ├─ Format context within max_tokens
    └─ Return {context, citations}
```

### Storage Schema (planned)

```sql
turns: id, session_id, user_id, messages, timestamp, metadata
facts: id, turn_id, user_id, type, key, value, content, confidence, active, created_at
```

---

## Endpoints

| Endpoint | Description | Current |
|----------|-------------|---------|
| `GET /health` | Liveness probe | ✅ |
| `POST /turns` | Store turn + extract facts | ⚠️ Stores only |
| `POST /recall` | Get context for agent | ⚠️ Returns empty |
| `POST /search` | Explicit search | ⚠️ Returns empty |
| `GET /users/{user_id}/memories` | List all memories | ⚠️ Returns empty |
| `DELETE /sessions/{session_id}` | Delete session | ✅ |
| `DELETE /users/{user_id}` | Delete user | ✅ |

---

## Configuration

### Environment Variables

```bash
# Azure OpenAI (for extraction) - optional
AZURE_OPENAI_ENDPOINT=https://...openai.azure.com/
AZURE_OPENAI_KEY=...
AZURE_DEPLOYMENT_NAME=gpt-oss-120b

# Optional: Auth token
MEMORY_AUTH_TOKEN=
```

### Fallback Strategy

If no LLM key provided:
- Option 1: Save raw messages as facts (simple fallback)
- Option 2: Use local embeddings (sentence-transformers)
- Option 3: Rule-based extraction (regex patterns)

---

## Running Tests

### Contract Tests

```bash
python3 -m pytest tests/test_contract.py -v
```

### Evaluation Tests

```bash
# Run first 50 tests
python3 scripts/run_eval.py

# Run all 500
# (edit scripts/run_eval.py to change limit)
```

### Smoke Test (from task.md)

```bash
curl -s http://localhost:8080/health
# {"status":"ok"}

curl -X POST http://localhost:8080/turns \
  -H 'Content-Type: application/json' \
  -d '{
    "session_id": "smoke-1",
    "user_id": "user-1",
    "messages": [
      {"role": "user", "content": "I just moved to Berlin from NYC last month."},
      {"role": "assistant", "content": "That sounds exciting!"}
    ],
    "timestamp": "2025-03-15T10:30:00Z",
    "metadata": {}
  }'

curl -X POST http://localhost:8080/recall \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "Where does this user live?",
    "session_id": "smoke-2",
    "user_id": "user-1",
    "max_tokens": 512
  }'
# Expected: context contains "Berlin"
```

---

## What's Next

1. **Extraction** — LLM extraction from turns → structured facts
2. **Fact storage** — SQLite with proper schema
3. **Recall logic** — Format facts + session messages within token budget
4. **Fact evolution** — Handle contradictions (new fact supersedes old)
5. **Smart retrieval** — Hybrid (embeddings + BM25) instead of simple

See CHANGELOG.md for iteration history.

---

## Test Dataset

We use **LongMemEval** (ICLR 2025) — 500 memory evaluation questions:

| Category | Count |
|----------|-------|
| fact_extraction | 259 |
| multi_hop | 133 |
| fact_evolution | 78 |
| preferences_opinions | 30 |

Each test has: chat history (turn), recall query, expected answer.