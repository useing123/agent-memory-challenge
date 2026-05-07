#!/usr/bin/env python3
"""
Run tests for specific category with detailed logging.
"""

import json
import requests
import sys
from datetime import datetime

BASE_URL = "http://localhost:8080"

def wait_for_service(timeout=10):
    for _ in range(timeout):
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=2)
            if r.status_code == 200:
                return True
        except:
            pass
    return False

def run_category_tests(category, limit=20):
    with open("fixtures/eval_tests.json") as f:
        data = json.load(f)
    
    tests = [t for t in data["tests"] if t.get("category") == category][:limit]
    print(f"Running {len(tests)} tests for category: {category}")
    
    results = []
    for i, test in enumerate(tests):
        try:
            turn = test["turn"]
            r = requests.post(f"{BASE_URL}/turns", json=turn, timeout=30)
            if r.status_code != 201:
                results.append({"error": f"turn failed: {r.status_code}", "test": i})
                continue
            
            query = test.get("recall_queries", [{}])[0].get("query", "") or test.get("question", "")
            expected = str(test.get("expected_answer", ""))
            
            recall_req = {
                "query": query,
                "session_id": turn["session_id"],
                "user_id": turn.get("user_id", "user-1"),
                "max_tokens": 1024
            }
            
            r = requests.post(f"{BASE_URL}/recall", json=recall_req, timeout=10)
            result = r.json()
            context = result.get("context", "")
            
            found = expected.lower() in context.lower() if expected and context else False
            
            # Cleanup after each test for isolation
            try:
                requests.delete(f"{BASE_URL}/users/{turn.get('user_id', 'user-1')}", timeout=5)
            except:
                pass
            
            results.append({
                "test_id": i,
                "query": query[:100],
                "expected": expected[:200],
                "found": found,
                "context_preview": context[:300] if context else ""
            })
            
        except Exception as e:
            results.append({"error": str(e), "test": i})
        
        if (i + 1) % 5 == 0:
            print(f"  Progress: {i+1}/{len(tests)}")
    
    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 run_category.py <category> [limit]")
        print("Categories: fact_extraction, fact_evolution, multi_hop, preferences_opinions")
        sys.exit(1)
    
    category = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    
    if not wait_for_service():
        print("Service not ready!")
        sys.exit(1)
    
    results = run_category_tests(category, limit)
    
    passed = sum(1 for r in results if r.get("found"))
    print(f"\n=== {category}: {passed}/{len(results)} passed ===")
    
    # Save detailed results
    filename = f"results/{category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w") as f:
        json.dump({
            "category": category,
            "passed": passed,
            "total": len(results),
            "results": results
        }, f, indent=2)
    
    print(f"Results saved to {filename}")
    
    # Print some failures for debugging
    failures = [r for r in results if not r.get("found") and "error" not in r][:3]
    if failures:
        print(f"\nSample failures:")
        for f in failures:
            print(f"  Query: {f.get('query', '')[:80]}")
            print(f"  Expected: {f.get('expected', '')[:80]}")
            print(f"  Got: {f.get('context_preview', '')[:100]}...")
            print()