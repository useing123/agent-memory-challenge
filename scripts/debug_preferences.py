#!/usr/bin/env python3
"""
Debug preferences_opinions test - single test with fresh user.
"""

import json
import requests
import sys
import uuid

BASE_URL = "http://localhost:8080"

def debug_test(test):
    # Use unique user_id
    user_id = f"debug-user-{uuid.uuid4().hex[:8]}"
    session_id = f"debug-session-{uuid.uuid4().hex[:8]}"
    
    print(f"User: {user_id}")
    print(f"Session: {session_id}")
    
    # Step 1: POST turn
    print("\n=== Step 1: POST turn ===")
    turn = test["turn"].copy()
    turn["user_id"] = user_id
    turn["session_id"] = session_id
    
    print(f"Messages in turn: {len(turn.get('messages', []))}")
    for i, msg in enumerate(turn.get("messages", [])[:5]):
        print(f"  {i}: {msg.get('role')}: {msg.get('content', '')[:80]}...")
    
    r = requests.post(f"{BASE_URL}/turns", json=turn, timeout=30)
    print(f"Status: {r.status_code}")
    turn_id = r.json().get("id")
    print(f"Turn ID: {turn_id}")
    
    # Step 2: Check extracted memories
    print("\n=== Step 2: Extracted memories ===")
    r = requests.get(f"{BASE_URL}/users/{user_id}/memories")
    memories = r.json().get("memories", [])
    print(f"Total memories: {len(memories)}")
    for m in memories[:5]:
        print(f"  - {m['type']}: {m['key']} = {m['value']}")
    
    # Step 3: POST recall
    print("\n=== Step 3: POST recall ===")
    query = test.get("recall_queries", [{}])[0].get("query", "") or test.get("question", "")
    expected = str(test.get("expected_answer", ""))
    
    print(f"Query: {query[:100]}")
    print(f"Expected: {expected[:100]}")
    
    recall_req = {
        "query": query,
        "session_id": session_id,
        "user_id": user_id,
        "max_tokens": 1024
    }
    
    r = requests.post(f"{BASE_URL}/recall", json=recall_req, timeout=10)
    result = r.json()
    context = result.get("context", "")
    citations = result.get("citations", [])
    
    print(f"\nContext (first 500 chars):\n{context[:500]}")
    print(f"\nCitations: {len(citations)}")
    
    # Check if found
    found = expected.lower() in context.lower() if expected and context else False
    print(f"\n=== RESULT: {'FOUND' if found else 'NOT FOUND'} ===")

if __name__ == "__main__":
    # Load first preferences test
    with open("fixtures/eval_tests.json") as f:
        data = json.load(f)
    
    prefs = [t for t in data["tests"] if t.get("category") == "preferences_opinions"]
    
    if not prefs:
        print("No preferences_opinions tests found")
        sys.exit(1)
    
    print(f"Testing {len(prefs)} preferences tests")
    print("="*50)
    
    # Test first one
    debug_test(prefs[0])