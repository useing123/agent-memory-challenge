#!/usr/bin/env python3
"""
Create mini test set (100 tests) with preserved category distribution.
"""

import json
import random

random.seed(42)

with open("/home/useing123/Desktop/higgsfield-memory/fixtures/eval_tests.json") as f:
    data = json.load(f)

tests = data["tests"]

# Group by category
by_cat = {}
for t in tests:
    cat = t.get("category", "unknown")
    if cat not in by_cat:
        by_cat[cat] = []
    by_cat[cat].append(t)

print("Original distribution:")
for cat, items in by_cat.items():
    print(f"  {cat}: {len(items)}")

# Create mini set with proportional distribution
mini = []
counts = {
    "fact_extraction": 40,
    "multi_hop": 25,
    "fact_evolution": 20,
    "preferences_opinions": 15,
}

for cat, count in counts.items():
    if cat in by_cat:
        samples = random.sample(by_cat[cat], min(count, len(by_cat[cat])))
        mini.extend(samples)

random.shuffle(mini)

print(f"\nMini test set: {len(mini)} tests")

# Save mini
output = {
    "source": "LongMemEval mini (100 tests)",
    "total_tests": len(mini),
    "tests": mini
}

with open("/home/useing123/Desktop/higgsfield-memory/fixtures/eval_tests_mini.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"Saved to fixtures/eval_tests_mini.json")