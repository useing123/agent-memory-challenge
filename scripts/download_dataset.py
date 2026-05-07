#!/usr/bin/env python3
"""
Download and adapt LongMemEval dataset to our fixture format.
Dataset: xiaowu0162/longmemeval-cleaned
- longmemeval_s_cleaned: ~40 sessions per test
- longmemeval_m_cleaned: ~500 sessions per test
"""

import json
import os
from datasets import load_dataset

# Download LongMemEval - use streaming to avoid large download
print("Downloading LongMemEval dataset (streaming)...")
ds = load_dataset("xiaowu0162/longmemeval-cleaned", split="longmemeval_s_cleaned", streaming=True)

# Check structure
print(f"Dataset type: {type(ds)}")
print("Dataset loaded successfully, will process entries...")

# Map our categories to LongMemEval question_types
CATEGORY_MAP = {
    "single-session-user": "fact_extraction",
    "single-session-assistant": "fact_extraction", 
    "single-session-preference": "preferences_opinions",
    "temporal-reasoning": "fact_extraction",
    "knowledge-update": "fact_evolution",
    "multi-session": "multi_hop",
}

def convert_entry(entry, use_full_sessions=True):
    """Convert LongMemEval entry to our fixture format.
    
    Key insight: Each test has 40-500 sessions representing one user persona.
    We feed all sessions to build memory, then ask one question.
    """
    
    # Use question_id as unique user_id for test isolation
    question_id = entry.get("question_id", "unknown")
    
    # Get all sessions or first N
    haystack_sessions = entry.get("haystack_sessions", [])
    if not use_full_sessions:
        haystack_sessions = haystack_sessions[:10]  # First 10 for quick testing
    
    # Build turns from all sessions
    turns = []
    for session_idx, session in enumerate(haystack_sessions):
        session_id = entry.get("haystack_session_ids", [f"session-{session_idx}"])[session_idx]
        
        for turn in session:
            turns.append({
                "session_id": session_id,
                "user_id": question_id,  # Use question_id as unique user_id
                "messages": [{
                    "role": turn.get("role", "user"),
                    "content": turn.get("content", "")
                }],
                "timestamp": entry.get("haystack_dates", ["2025-01-01T00:00:00Z"])[session_idx] if session_idx < len(entry.get("haystack_dates", [])) else "2025-01-01T00:00:00Z",
                "metadata": {"session_index": session_idx}
            })
    
    return {
        "question_id": question_id,
        "category": CATEGORY_MAP.get(entry.get("question_type", ""), "fact_extraction"),
        "question_type": entry.get("question_type", ""),
        "turns": turns,
        "question": entry.get("question", ""),
        "expected_answer": entry.get("answer", ""),
        "answer_session_ids": entry.get("answer_session_ids", []),
        "haystack_session_ids": entry.get("haystack_session_ids", []),
    }


# Convert all examples
print("Converting entries...")
tests = []

for i, entry in enumerate(ds):
    try:
        converted = convert_entry(entry, use_full_sessions=True)
        tests.append(converted)
        
        if (i + 1) % 50 == 0:
            print(f"Converted {i + 1} tests...")
            
    except Exception as e:
        print(f"Error converting entry {i}: {e}")

print(f"\nTotal converted: {len(tests)}")

# Category breakdown
category_counts = {}
for t in tests:
    cat = t.get("category", "unknown")
    category_counts[cat] = category_counts.get(cat, 0) + 1

print("Category breakdown:")
for cat, count in sorted(category_counts.items()):
    print(f"  {cat}: {count}")

# Save full dataset
output = {
    "source": "LongMemEval (xiaowu0162/longmemeval-cleaned)",
    "split": "longmemeval_s_cleaned",
    "total_tests": len(tests),
    "categories": category_counts,
"tests": tests
}

output_path = "/home/useing123/Desktop/higgsfield-memory/data/longmemeval_s_cleaned.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nSaved {len(tests)} tests to {output_path}")

# Also create a smaller test set (first 50)
small_output = {
    "source": "LongMemEval (xiaowu0162/longmemeval-cleaned)",
    "split": "longmemeval_s_cleaned",
    "total_tests": 50,
    "categories": {k: v for k, v in category_counts.items() if k in ["fact_extraction", "multi_hop", "fact_evolution", "preferences_opinions"]},
    "tests": tests[:50]
}

small_path = "/home/useing123/Desktop/higgsfield-memory/data/longmemeval_small.json"
with open(small_path, "w") as f:
    json.dump(small_output, f, indent=2)

print(f"Saved 50 tests to {small_path}")