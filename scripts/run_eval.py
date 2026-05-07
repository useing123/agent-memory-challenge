#!/usr/bin/env python3
"""
Run evaluation tests against the memory service.
"""

import json
import requests
import sys
from datetime import datetime

BASE_URL = "http://localhost:8080"

def wait_for_service(timeout=30):
    """Wait for service to be ready."""
    start = datetime.now()
    while (datetime.now() - start).seconds < timeout:
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=2)
            if r.status_code == 200:
                print("✓ Service is ready")
                return True
        except:
            pass
    return False

def run_test(test):
    """Run a single test: POST turn, then POST recall, check if expected answer in context."""
    try:
        # POST turn
        turn = test["turn"]
        r = requests.post(f"{BASE_URL}/turns", json=turn, timeout=30)
        if r.status_code != 201:
            return {"error": f"turn failed: {r.status_code}"}
        
        turn_id = r.json().get("id")
        
        # POST recall
        query = test["recall_queries"][0]["query"] if test.get("recall_queries") else test.get("question", "")
        expected = test.get("expected_answer", "")
        
        recall_req = {
            "query": query,
            "session_id": turn["session_id"],
            "user_id": turn.get("user_id", "user-1"),
            "max_tokens": 1024
        }
        
        r = requests.post(f"{BASE_URL}/recall", json=recall_req, timeout=10)
        if r.status_code != 200:
            return {"error": f"recall failed: {r.status_code}"}
        
        result = r.json()
        context = result.get("context", "")
        
        # Check if expected answer is in context
        found = expected.lower() in context.lower()
        
        return {
            "found": found,
            "expected": expected[:50],
            "context_preview": context[:100] if context else "(empty)"
        }
        
    except Exception as e:
        return {"error": str(e)}

def main():
    # Load fixtures
    with open("fixtures/eval_tests.json") as f:
        data = json.load(f)
    
    tests = data["tests"]
    print(f"Loaded {len(tests)} tests")
    
    # Wait for service
    if not wait_for_service():
        print("✗ Service not ready")
        sys.exit(1)
    
    # Run tests (limit to 50 for quick check)
    results = {
        "passed": 0,
        "failed": 0,
        "errors": 0
    }
    
    sample_errors = []
    
    for i, test in enumerate(tests[:50]):
        cat = test.get("category", "unknown")
        result = run_test(test)
        
        if "error" in result:
            results["errors"] += 1
            if len(sample_errors) < 3:
                sample_errors.append(f"  [{cat}] {result['error']}")
        elif result.get("found"):
            results["passed"] += 1
        else:
            results["failed"] += 1
            if len(sample_errors) < 3:
                sample_errors.append(f"  [{cat}] expected: {result['expected']}, got: {result['context_preview']}")
        
        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/50")
    
    print(f"\n=== Results (first 50 tests) ===")
    print(f"Passed:  {results['passed']}")
    print(f"Failed:  {results['failed']}")
    print(f"Errors:  {results['errors']}")
    print(f"\nSuccess rate: {results['passed']/50*100:.1f}%")
    
    if sample_errors:
        print(f"\nSample issues:")
        for e in sample_errors:
            print(e)

if __name__ == "__main__":
    main()