import pytest
import json
from httpx import AsyncClient, ASGITransport
from src.main import app

# Load the quality fixture
with open("fixtures/recall_quality_fixture.json") as f:
    fixture_data = json.load(f)

@pytest.mark.parametrize("test_case", fixture_data["tests"])
@pytest.mark.asyncio
async def test_recall_quality_fixture(test_case):
    """
    This test runs through the scenarios defined in the recall_quality_fixture.json file.
    It simulates a user's history and then asks a targeted recall question.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        user_id = test_case["turns"][0]["user_id"]
        
        # 1. Ingest all turns to build the memory
        for turn in test_case["turns"]:
            response = await client.post("/turns", json=turn)
            assert response.status_code == 201, f"Failed to post turn for fixture {test_case['fixture_id']}"

        # 2. Perform the recall query
        recall_query = test_case["recall_query"]
        recall_payload = {
            "query": recall_query["query"],
            "session_id": test_case["turns"][-1]["session_id"],
            "user_id": user_id,
            "max_tokens": 2048
        }
        response = await client.post("/recall", json=recall_payload)
        assert response.status_code == 200, f"Recall failed for fixture {test_case['fixture_id']}"
        
        recall_result = response.json()
        context = recall_result.get("context", "").lower()

        # 3. Assertions
        for expected in recall_query.get("expected_context_contains", []):
            assert expected.lower() in context, f"Expected to find '{expected}' in context for fixture {test_case['fixture_id']}"
            
        for not_expected in recall_query.get("expected_context_not_contains", []):
            assert not_expected.lower() not in context, f"Did not expect to find '{not_expected}' in context for fixture {test_case['fixture_id']}"

        # 4. Cleanup
        await client.delete(f"/users/{user_id}")
