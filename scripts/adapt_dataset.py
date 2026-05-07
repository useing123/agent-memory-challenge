#!/usr/bin/env python3
"""
Adapt LongMemEval to our fixture format.
"""

import json
import random

with open("/home/useing123/Desktop/higgsfield-memory/fixtures/longmemeval_sample.json") as f:
    data = json.load(f)

CATEGORY_MAP = {
    "single-session-user": "fact_extraction",
    "single-session-assistant": "fact_extraction",
    "single-session-preference": "preferences_opinions",
    "temporal-reasoning": "fact_extraction",
    "knowledge-update": "fact_evolution",
    "multi-session": "multi_hop",
}

def convert_entry(entry):
    # Get session info
    session_ids = entry.get("haystack_session_ids", [])
    session_dates = entry.get("haystack_dates", [])

    session_id = session_ids[0] if session_ids else f"session-{random.randint(1,100)}"
    timestamp = session_dates[0] if session_dates else "2025-01-01T00:00:00Z"

    # Build messages from haystack_sessions (first 2 sessions for simplicity)
    messages = []
    haystack = entry.get("haystack_sessions", [])
    for session in haystack[:2]:
        for turn in session:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if content:  # Only add non-empty
                messages.append({
                    "role": role,
                    "content": content[:500]  # Truncate long content
                })

    # Limit to 10 messages
    messages = messages[:10]

    return {
        "category": CATEGORY_MAP.get(entry["question_type"], "fact_extraction"),
        "turn": {
            "session_id": session_id,
            "user_id": "user-1",
            "messages": messages,
            "timestamp": timestamp,
            "metadata": {}
        },
        "question": entry["question"],
        "expected_answer": entry["answer"],
        "question_type": entry["question_type"],
        "recall_queries": [
            {
                "query": entry["question"],
                "expected_context_contains": [entry["answer"]]
            }
        ]
    }

# Convert all entries
tests = [convert_entry(e) for e in data]

# Show category distribution
cats = {}
for t in tests:
    c = t["category"]
    cats[c] = cats.get(c, 0) + 1

print("Category distribution:")
for k, v in cats.items():
    print(f"  {k}: {v}")

# Save to our format
output = {
    "source": "LongMemEval (xiaowu0162/longmemeval-cleaned)",
    "total_tests": len(tests),
    "categories": cats,
    "tests": tests
}

with open("/home/useing123/Desktop/higgsfield-memory/fixtures/eval_tests.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\nSaved {len(tests)} tests to fixtures/eval_tests.json")