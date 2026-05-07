# Changelog

## v0.1.0 - Bare Minimum

I've initialized repository. Created docker container, smoke tests, and basic contract endpoints.

Now I'm brainstorming for possible solutions to the problem. 

How I will solve this without time to execute: Basic api endpoint for calling LLM with chat history and token limit. And memory for agent after passing limit with AI powered prune for chat history. Maybe also use some graph based extraction after reaching some use cases.

Now I'm looking for viewing and brainstorming existing solutions and create some mix of this maybe after researching existing solutions I can implement something that fits my needs. As I understand we need to create some type of long memory like in claude and beat other solutions on performance and scalability. Maybe later I also need to write my tests cases and show where my solution beat others.

My main focus is extraction methods also I should remember about this: 
```
We inspect /users/{user_id}/memories during review. If it returns raw message chunks instead of structured memories, that's a red flag.
```
I can't ask questions and I'll assume that I'm builing memory for some cli agent that should need to remember and extract information from chat history. And my main focus on extracting structured memories from chat history.

## v0.2.0 — Architecture Decision

**What changed:** Defined memory architecture with two independent memory systems.

**Why:** After analyzing task requirements and smoke test expectations, identified that:
- Smoke test expects extraction: "I just moved to Berlin" → structured fact
- They explicitly check: `/memories` returns `{type, key, value}`, NOT raw text
- `/recall` must return context with extracted facts, not empty response

**Architecture:**
- **Session Memory** — current conversation context (last N messages), highest priority
- **Global Memory** — facts extracted from ALL conversations, searchable

**Storage (SQLite):**
```
turns: id, session_id, user_id, messages, timestamp, metadata
facts: id, turn_id, user_id, type, key, value, content, confidence, active, created_at
```

**Extraction:**
- LLM extracts 0-5 facts per turn
- All facts stored as active=True
- No auto-deletion — LLM on recall decides what's relevant

**Contradiction handling:**
- Store ALL facts (no deletion)
- On recall: return all active facts + source context
- Let downstream agent (via instructions) decide what's current
- Pro: simple, handles gradual opinion evolution ("love TS" → "TS is okay")

**Recall priority:**
1. Current session messages (most recent)
2. User's global facts
3. Format as context within max_tokens

**Next:** Implement extraction pipeline. Need to decide on embedding approach and fallback if no LLM key provided.

---

## v0.3.0 — Test Dataset & Fixture Generation

**What changed:** Generated 500 test cases from LongMemEval dataset for eval.

**Why:** The task requires "recall quality fixture" with 3-5 conversations. Instead of manually writing few tests, found LongMemEval benchmark (ICLR 2025) with 500 pre-built memory evaluation cases.

**Observation:**
- LongMemEval: gradual multi-session filling → then recall query
- Higgsfield smoke test: immediate steps in sequence
- These are different patterns — may need to handle both

**Dataset details:**
- 500 questions across 4 categories
- fact_extraction: 259, multi_hop: 133, fact_evolution: 78, preferences_opinions: 30
- Each has: turn (chat history), question, expected_answer

**Format adaptation:**
- Converted LongMemEval's haystack_sessions → our POST /turns format
- Added recall_queries with expected_context_contains

**Next:** Implement extraction and recall to pass these tests. Need to handle both immediate recall (smoke test style) and multi-session patterns (LongMemEval style).

---

## v0.4.0 — LLM Extraction Working

**What changed:** Implemented LLM extraction via Azure OpenAI and SQLite storage.

**Why:** Smoke test was failing because recall returned empty context. Need extraction to pass.

**Implementation:**
- Azure OpenAI extracts facts from each turn (type, key, value, confidence)
- Facts stored in SQLite with active=1
- Contradiction handling: new fact with same key supersedes old (active=0)
- Recall returns all active facts + recent conversation messages

**Result:**
- Smoke test: ✅ PASSED
  - POST /recall now returns "location: Berlin" (was empty)
  - GET /memories returns structured facts (was empty)
- Eval tests (first 50): 10% passed (5/50)

**Issue identified:**
- Recall shows ALL facts, not filtered by query relevance
- LLM extracts relevant facts from conversation, but recall doesn't match query to facts

**Next:** Improve recall query matching — filter facts by query keywords or use embedding similarity.

---

## v0.4.1 — Query-Based Filtering

**What changed:** Added keyword-based filtering for recall to match query to relevant facts.

**Result:**
- Smoke test: PASSES (returns "location: Berlin" for "Where does this user live?")
- Eval tests (first 50): 12% passed (6/50)

**Analysis:**
- Smoke test works
- LongMemEval eval tests are complex: multi-session conversations, specific questions
- Issue: extraction misses some facts; query matching is basic keyword match
- Would need: better extraction prompt, embeddings for similarity, or more sessions in test data

**Next:** On previous hackathons I've researched about vectoring strategies and I will read existing solutions for inspiration. But now I want to create .md based + user_id based facts storage and .md based vector search. Maybe in future I'll change my solution if I'll get more scores on the eval tests.
