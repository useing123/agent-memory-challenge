# Alpha-Stage Memory Service (v1.2.1)

This repository contains a functional, Dockerized memory service for an AI agent. This is an **alpha version** submitted to demonstrate the core concepts, identify key challenges, and propose a robust, production-ready architecture.

The service correctly ingests conversational data, extracts structured facts, and performs semantic recall. However, several critical bugs and performance bottlenecks were identified during development (and documented in the `CHANGELOG.md`). The proposed final architecture, which addresses these issues, is detailed in `ARCHITECTURE.md`.

## Quick Start

### 1. Configure Environment
Create a `.env` file from the example and add your Azure OpenAI credentials. This is required for fact extraction and embedding.
```bash
cp .env.example .env
# Now, edit .env with your Azure keys and endpoint
```

### 2. Build and Run with Docker
This command builds the Docker image and runs the service in the background, passing your environment variables securely.
```bash
docker build -t memory-service:latest .
docker run -d -p 8080:8080 --name memory-service --env-file .env memory-service:latest
```

### 3. Setup Data
The evaluation relies on the `LongMemEval` dataset. A processing script is included to download and prepare the necessary files.
```bash
python3 process_longmemeval.py
```
This will download the raw dataset (277MB) into `data/` and create processed `longmemeval_*.json` files for the evaluation script.

### 4. Run Evaluation
Use the dedicated evaluation script to test the agent's memory performance against the `LongMemEval` benchmark.
```bash
# Run 10 tests from all categories in isolated mode (recommended)
python3 run_eval_new.py --all 10 --test -v

# Run 50 tests for a specific category
python3 run_eval_new.py fact_extraction 50 --test -v
```

## Architecture & Design (Alpha Version)

The current implementation uses a straightforward but flawed architecture, which is detailed in the `ARCHITECTURE.md` file.

*   **Backing Store:** SQLite. Chosen for its simplicity, zero-config setup, and because it's sufficient for the scale of this project. The database file is persisted via a Docker volume.
*   **Extraction Pipeline:** On every `POST /turns`, the service makes 1-2 calls to an Azure OpenAI model (`gpt-4o-mini` by default) to extract structured facts (`{key, value, type}`) from the conversation.
*   **Recall Strategy (Alpha):**
    1.  **MD Document Generation:** All extracted facts for a user are compiled into a single Markdown document.
    2.  **Chunking & Embedding:** This document is split into chunks by section headers. On every update, all chunks are deleted and re-embedded using Azure's `text-embedding-3-large`. **This is the primary performance bottleneck.**
    3.  **Semantic Search:** On `POST /recall`, the service performs a cosine similarity search over the embedded chunks to find the top-k most relevant pieces of context.

## Identified Issues & Future Architecture

This alpha implementation suffers from several critical issues:
1.  **Major Performance Bottleneck:** Re-embedding the entire user history on every single turn is extremely slow and expensive.
2.  **Sub-Optimal Recall:** The "vanilla cosine-top-k" recall strategy is explicitly discouraged in the project brief and will not perform well on multi-hop or keyword-sensitive queries.
3.  **Lack of Context Budgeting:** The strict priority logic for assembling context under a token budget is not yet implemented.

A robust, production-ready architecture is proposed in `ARCHITECTURE.md` and documented in the `CHANGELOG.md`. This future plan includes **hybrid search (BM25 + Semantic), Reciprocal Rank Fusion (RRF), and an LLM reranker** to meet the "Excellent" criteria outlined in the project brief.

## Running Tests

### Local Pytest
You can run the suite of unit and integration tests locally.
```bash
python3 -m pytest -v
```

### Evaluation Script
The primary evaluation tool is `run_eval_new.py`, which tests against the `LongMemEval` dataset.

**Modes:**
*   **Isolated Mode (`--test`):** **Recommended.** Creates a unique user for each test case and cleans up afterward. This prevents memory contamination between tests.
*   **Shared Mode (no flag):** Uses the same `user-id` for all tests, simulating one long-running conversation.

**Commands:**
```bash
# Run 10 tests from all categories in isolated mode with verbose output
python3 run_eval_new.py --all 10 --test -v

# Run all 50 tests from the 'fact_extraction' category in isolated mode
python3 run_eval_new.py fact_extraction 50 --test

# Run all tests using a different data file
python3 run_eval_new.py --all 100 --test --data data/longmemeval_100.json
```
