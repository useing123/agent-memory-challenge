#!/usr/bin/env python3
"""
Run full eval with category breakdown and save results.
"""

import json
import requests
import sys
from datetime import datetime

BASE_URL = "http://localhost:8080"

def wait_for_service(timeout=30):
    start = datetime.now()
    while (datetime.now() - start).seconds < timeout:
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=2)
            if r.status_code == 200:
                return True
        except:
            pass
    return False

def run_test(test):
    try:
        turn = test["turn"]
        r = requests.post(f"{BASE_URL}/turns", json=turn, timeout=30)
        if r.status_code != 201:
            return {"error": f"turn failed: {r.status_code}"}
        
        turn_id = r.json().get("id")
        query = test["recall_queries"][0]["query"] if test.get("recall_queries") else test.get("question", "")
        expected = str(test.get("expected_answer", ""))
        
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
        found = expected.lower() in context.lower() if expected and context else False
        
        return {
            "found": found,
            "expected": expected[:100],
            "actual_context": context[:500] if context else "",
            "query": query[:100],
            "category": test.get("category", "unknown")
        }
        
    except Exception as e:
        return {"error": str(e), "category": test.get("category", "unknown")}

def main():
    with open("fixtures/eval_tests.json") as f:
        data = json.load(f)
    
    tests = data["tests"]
    print(f"Loaded {len(tests)} tests")
    
    if not wait_for_service():
        print("Service not ready")
        sys.exit(1)
    
    # Track by category
    by_category = {}
    
    detailed_results = []
    
    for i, test in enumerate(tests):
        cat = test.get("category", "unknown")
        if cat not in by_category:
            by_category[cat] = {"passed": 0, "failed": 0, "errors": 0}
        
        result = run_test(test)
        
        detailed_results.append({
            "test_id": i,
            "category": cat,
            **result
        })
        
        if "error" in result:
            by_category[cat]["errors"] += 1
        elif result.get("found"):
            by_category[cat]["passed"] += 1
        else:
            by_category[cat]["failed"] += 1
        
        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{len(tests)}", flush=True)
    
    # Summary
    total_passed = sum(c["passed"] for c in by_category.values())
    total_failed = sum(c["failed"] for c in by_category.values())
    total_errors = sum(c["errors"] for c in by_category.values())
    total = total_passed + total_failed + total_errors
    current_passed = total_passed
    
    print(f"\n=== Full Eval Results (500 tests) ===")
    print(f"\nBy Category:")
    for cat, stats in sorted(by_category.items()):
        cat_total = stats["passed"] + stats["failed"] + stats["errors"]
        rate = stats["passed"] / cat_total * 100 if cat_total > 0 else 0
        print(f"  {cat}: {stats['passed']}/{cat_total} ({rate:.1f}%)")
    
    print(f"\nTotal: {total_passed}/{total} ({total_passed/total*100:.1f}%)")
    
    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "total_tests": total,
        "passed": total_passed,
        "failed": total_failed,
        "errors": total_errors,
        "success_rate": total_passed/total*100 if total > 0 else 0,
        "by_category": by_category,
        "tests": detailed_results[:50]
    }
    
    filename = f"results/eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {filename}")
    
    print(f"\nResults saved to results/")

if __name__ == "__main__":
    main()