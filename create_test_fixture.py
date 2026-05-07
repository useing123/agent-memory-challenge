#!/usr/bin/env python3
"""
Create a small, balanced test fixture from the main LongMemEval dataset.
This allows for rapid testing without processing the entire 500-test suite.
"""
import json
import os
from collections import defaultdict

# --- Configuration ---
DATA_DIR = "data"
FULL_DATASET_PATH = os.path.join(DATA_DIR, "longmemeval_s_cleaned.json")
OUTPUT_FIXTURE_PATH = "fixtures/eval_tests_small.json"
SAMPLES_PER_CATEGORY = {
    "fact_extraction": 8,
    "multi_hop": 5,
    "fact_evolution": 4,
    "preferences_opinions": 3,
}
# ---

def create_small_fixture():
    """Reads the full dataset and creates a small, balanced sample."""
    print(f"Loading full dataset from {FULL_DATASET_PATH}...")
    try:
        with open(FULL_DATASET_PATH, "r") as f:
            full_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Full dataset not found at {FULL_DATASET_PATH}.")
        print("Please run 'python3 process_longmemeval.py' first.")
        return

    all_tests = full_data.get("tests", [])
    print(f"Loaded {len(all_tests)} total tests.")

    # Group tests by category
    tests_by_category = defaultdict(list)
    for test in all_tests:
        tests_by_category[test.get("category")].append(test)

    # Create a balanced sample
    sampled_tests = []
    for category, num_samples in SAMPLES_PER_CATEGORY.items():
        if category in tests_by_category:
            sampled_tests.extend(tests_by_category[category][:num_samples])
            print(f"Sampled {min(num_samples, len(tests_by_category[category]))} tests from '{category}'")

    # Create the final fixture object
    output_fixture = {
        "source": "Small sample from LongMemEval",
        "total_tests": len(sampled_tests),
        "tests": sampled_tests,
    }

    # Ensure fixtures directory exists
    os.makedirs(os.path.dirname(OUTPUT_FIXTURE_PATH), exist_ok=True)

    # Save the new fixture
    with open(OUTPUT_FIXTURE_PATH, "w") as f:
        json.dump(output_fixture, f, indent=2)

    print(f"\nSuccessfully created small test fixture with {len(sampled_tests)} tests at '{OUTPUT_FIXTURE_PATH}'")

if __name__ == "__main__":
    create_small_fixture()
