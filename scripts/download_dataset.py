#!/usr/bin/env python3
"""
Download and adapt LongMemEval dataset to our fixture format.
"""

import json
from datasets import load_dataset

# Download LongMemEval
print("Downloading LongMemEval dataset...")
ds = load_dataset("xiaowu0162/longmemeval-cleaned")

# Check structure
print(f"Available splits: {list(ds.keys())}")
print(f"Example entry keys: {ds['test'][0].keys()}")

# Map our categories to LongMemEval question_types
CATEGORY_MAP = {
    "single-session-user": "fact_extraction",
    "single-session-assistant": "fact_extraction",
    "single-session-preference": "preferences_opinions",
    "temporal-reasoning": "fact_extraction",
    "knowledge-update": "fact_evolution",
    "multi-session": "multi_hop",
}

# Convert to our format
def convert_entry(entry):
    session_id = entry["haystack_session_ids"][0] if entry["haystack_session_ids"] else "session-1"

    # Build messages from haystack_sessions
    messages = []
    for session in entry.get("haystack_sessions", [])[:3]:  # First 3 sessions
        for turn in session:
            messages.append({
                "role": turn.get("role", "user"),
                "content": turn.get("content", "")
            })

    return {
        "category": CATEGORY_MAP.get(entry["question_type"], "fact_extraction"),
        "turn": {
            "session_id": session_id,
            "user_id": "user-1",
            "messages": messages[:10],  # Limit messages
            "timestamp": entry.get("haystack_dates", ["2025-01-01T00:00:00Z"])[0],
            "metadata": {}
        },
        "question": entry["question"],
        "expected_answer": entry["answer"],
        "recall_queries": [
            {
                "query": entry["question"],
                "expected_context_contains": [entry["answer"]]
            }
        ]
    }

# Convert first 10 examples
tests = []
for i, entry in enumerate(ds['test'][:10]):
    try:
        converted = convert_entry(entry)
        tests.append(converted)
        print(f"Converted: {converted['category']} - {entry['question'][:50]}...")
    except Exception as e:
        print(f"Error converting entry {i}: {e}")

# Save to fixtures
output = {
    "source": "LongMemEval (xiaowu0162/longmemeval-cleaned)",
    "total_tests": len(tests),
    "tests": tests
}

output_path = "/home/useing123/Desktop/higgsfield-memory/fixtures/eval_tests.json"
with open(output_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nSaved {len(tests)} tests to {output_path}")