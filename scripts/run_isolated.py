#!/usr/bin/env python3
"""
Test runner with progress bar and logging for evaluation.
"""

import json
import requests
import sys
import uuid
import argparse
import time
from datetime import datetime
import time
from datetime import datetime

BASE_URL = "http://localhost:8080"

def wait_for_service(timeout=30):
    """Wait for service to be ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=2)
            if r.status_code == 200:
                print("[OK] Service is ready")
                return True
        except:
            pass
    return False

def cleanup_user(user_id):
    """Clean up all data for a user."""
    try:
        requests.delete(f"{BASE_URL}/users/{user_id}", timeout=5)
    except:
        pass

def run_test(test, unique_user_id):
    """Run a single test with unique user_id."""
    turn = test["turn"]
    query = test.get("recall_queries", [{}])[0].get("query") or test.get("question", "")
    expected = str(test.get("expected_answer", ""))
    
    # Modify turn with unique user_id
    turn["user_id"] = unique_user_id
    
    # Post turn
    r = requests.post(f"{BASE_URL}/turns", json=turn, timeout=90)
    if r.status_code != 201:
        return {"error": f"turn failed: {r.status_code}", "expected": expected[:50]}
    
    # Recall with same user_id
    recall_req = {
        "query": query,
        "session_id": turn["session_id"],
        "user_id": unique_user_id,
        "max_tokens": 2048
    }
    
    r = requests.post(f"{BASE_URL}/recall", json=recall_req, timeout=30)
    if r.status_code != 200:
        return {"error": f"recall failed: {r.status_code}", "expected": expected[:50]}
    
    result = r.json()
    context = result.get("context", "")
    
    # Check if expected answer is in context
    found = expected.lower() in context.lower() if expected and context else False
    
    return {
        "found": found,
        "expected": expected[:100],
        "got": context[:200] if context else "(empty)"
    }

def print_progress_bar(current, total, prefix='Progress:', width=40):
    """Print a progress bar."""
    percent = current / total
    filled = int(width * percent)
    bar = '█' * filled + '░' * (width - filled)
    sys.stdout.write(f'\r{prefix} [{bar}] {current}/{total} ({percent*100:.1f}%)')
    sys.stdout.flush()

def run_category(category, limit=20, test_mode=False, verbose=False):
    """Run tests for a category."""
    # Load fixtures
    with open("fixtures/eval_tests.json") as f:
        data = json.load(f)
    
    tests = [t for t in data["tests"] if t.get("category") == category][:limit]
    print(f"\n=== Running {len(tests)} tests for: {category} ===")
    if test_mode:
        print("[MODE] Isolated - unique user per test")
    else:
        print("[MODE] Shared - same user for all")
    print()
    
    results = []
    start_time = time.time()
    
    for i, test in enumerate(tests):
        # Progress bar
        print_progress_bar(i, len(tests), f"[{category}]", width=30)
        
        unique_user_id = f"user-test-{uuid.uuid4().hex[:8]}" if test_mode else test["turn"].get("user_id", "user-1")
        
        try:
            result = run_test(test, unique_user_id)
            
            if "error" in result:
                results.append({"error": result["error"], "test": i, "expected": result.get("expected", "")})
                if verbose:
                    print(f"\n  [ERROR] Test {i}: {result['error']}")
            elif result.get("found"):
                results.append({"found": True, "test": i})
                if verbose:
                    print(f"\n  [PASS] Test {i}")
            else:
                results.append({
                    "found": False, 
                    "test": i,
                    "expected": result.get("expected", ""),
                    "got": result.get("got", "")[:200]
                })
                if verbose:
                    print(f"\n  [FAIL] Expected: {result.get('expected', '')[:60]}")
                    print(f"         Got: {result.get('got', '')[:80]}")
            
            # Cleanup in test mode
            if test_mode:
                cleanup_user(unique_user_id)
                
        except Exception as e:
            results.append({"error": str(e), "test": i})
            if verbose:
                print(f"\n  [EXCEPTION] {e}")
        
        # Small delay between tests
        time.sleep(0.2)
    
    elapsed = time.time() - start_time
    
    # Final progress bar
    print_progress_bar(len(tests), len(tests), f"[{category}]", width=30)
    print()  # Newline
    
    # Results
    passed = sum(1 for r in results if r.get("found"))
    failed = sum(1 for r in results if not r.get("found") and "error" not in r)
    errors = sum(1 for r in results if "error" in r)
    
    print(f"\n=== Results for {category} ===")
    print(f"Passed:  {passed}")
    print(f"Failed:  {failed}")
    print(f"Errors:  {errors}")
    print(f"Total:   {len(tests)}")
    print(f"Score:   {passed/len(tests)*100:.1f}%")
    print(f"Time:    {elapsed:.1f}s")
    
    # Sample failures
    if verbose:
        failures = [r for r in results if not r.get("found") and "error" not in r][:2]
        if failures:
            print(f"\nSample failures:")
            for f in failures:
                print(f"  - Expected: {f.get('expected', '')[:40]}...")
    
    # Save results
    filename = f"results/{category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w") as f:
        json.dump({
            "category": category,
            "passed": passed,
            "total": len(tests),
            "results": results
        }, f, indent=2)
    print(f"\nResults saved to: {filename}")
    
    return {
        "category": category,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "total": len(tests),
        "results": results,
        "elapsed": elapsed
    }

def main():
    parser = argparse.ArgumentParser(description="Run evaluation tests")
    parser.add_argument("category", nargs="?", help="Category to test (fact_extraction, multi_hop, fact_evolution, preferences_opinions)")
    parser.add_argument("limit", type=int, default=20, help="Number of tests")
    parser.add_argument("--test", action="store_true", help="Isolated test mode with unique user_ids")
    parser.add_argument("--all", action="store_true", help="Run all categories")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    if not wait_for_service(30):
        print("[ERROR] Service not ready!")
        sys.exit(1)
    
    if args.all:
        # Run all categories
        categories = ["fact_extraction", "multi_hop", "fact_evolution", "preferences_opinions"]
        all_results = {}
        
        for cat in categories:
            print(f"\n{'='*50}")
            result = run_category(cat, args.limit, test_mode=args.test, verbose=args.verbose)
            all_results[cat] = result
            time.sleep(1)  # Brief pause between categories
        
        # Summary
        print(f"\n{'='*50}")
        print("=== SUMMARY ===")
        total_passed = 0
        total_tests = 0
        for cat, r in all_results.items():
            print(f"{cat}: {r['passed']}/{r['total']} ({r['passed']/r['total']*100:.1f}%)")
            total_passed += r['passed']
            total_tests += r['total']
        
        print(f"\nOVERALL: {total_passed}/{total_tests} ({total_passed/total_tests*100:.1f}%)")
        
    elif args.category:
        result = run_category(args.category, args.limit, test_mode=args.test, verbose=args.verbose)
    else:
        print("Usage:")
        print("  python3 scripts/run_isolated.py fact_evolution 20")
        print("  python3 scripts/run_isolated.py fact_evolution 20 --test (isolated)")
        print("  python3 scripts/run_isolated.py --all 10")
        print("  python3 scripts/run_isolated.py --all 10 --test -v")

if __name__ == "__main__":
    main()