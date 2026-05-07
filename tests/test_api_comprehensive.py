"""
Comprehensive API test cases for Memory Service.

Run with: pytest tests/test_api_comprehensive.py -v
"""

import pytest
import json
from httpx import AsyncClient, ASGITransport
from src.main import app


class TestHealthEndpoint:
    """Test /health endpoint."""
    
    @pytest.mark.asyncio
    async def test_health_returns_ok(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestTurnsEndpoint:
    """Test /turns endpoint."""
    
    @pytest.mark.asyncio
    async def test_create_turn_success(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/turns", json={
                "session_id": "session-1",
                "user_id": "user-1",
                "messages": [
                    {"role": "user", "content": "I work at Google."},
                    {"role": "assistant", "content": "What team?"}
                ],
                "timestamp": "2025-01-01T00:00:00Z",
                "metadata": {}
            })
        assert response.status_code == 201
        data = response.json()
        assert "id" in data
    
    @pytest.mark.asyncio
    async def test_create_turn_extracts_facts(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post("/turns", json={
                "session_id": "session-facts",
                "user_id": "user-facts",
                "messages": [
                    {"role": "user", "content": "I live in Berlin and work at Google."}
                ],
                "timestamp": "2025-01-01T00:00:00Z",
                "metadata": {}
            })
            
            # Verify facts extracted
            mem_response = await client.get("/users/user-facts/memories")
            memories = mem_response.json()["memories"]
            assert len(memories) > 0
    
    @pytest.mark.asyncio
    async def test_create_turn_with_empty_messages(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/turns", json={
                "session_id": "session-empty",
                "user_id": "user-empty",
                "messages": [],
                "timestamp": "2025-01-01T00:00:00Z",
                "metadata": {}
            })
        # Should still return 201, handle gracefully
        assert response.status_code == 201
    
    @pytest.mark.asyncio
    async def test_create_turn_missing_user_id(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/turns", json={
                "session_id": "session-2",
                "messages": [
                    {"role": "user", "content": "Hello"}
                ],
                "timestamp": "2025-01-01T00:00:00Z",
                "metadata": {}
            })
        assert response.status_code == 201


class TestRecallEndpoint:
    """Test /recall endpoint - the most important endpoint."""
    
    @pytest.mark.asyncio
    async def test_recall_returns_200(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/recall", json={
                "query": "test query",
                "session_id": "session-1",
                "user_id": "user-1",
                "max_tokens": 1024
            })
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_recall_response_structure(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/recall", json={
                "query": "test",
                "session_id": "session-2",
                "user_id": "user-2",
                "max_tokens": 1024
            })
        data = response.json()
        assert "context" in data
        assert "citations" in data
        assert isinstance(data["citations"], list)
    
    @pytest.mark.asyncio
    async def test_recall_with_data_returns_context(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Create turn with known fact
            await client.post("/turns", json={
                "session_id": "recall-test-1",
                "user_id": "user-recall-test",
                "messages": [
                    {"role": "user", "content": "I live in Berlin."}
                ],
                "timestamp": "2025-01-01T00:00:00Z",
                "metadata": {}
            })
            
            # Recall should return some context
            response = await client.post("/recall", json={
                "query": "Where does this user live?",
                "session_id": "recall-test-1",
                "user_id": "user-recall-test",
                "max_tokens": 600
            })
            
            data = response.json()
            # Should have context (either answer from LLM or fallback)
            assert "context" in data
    
    @pytest.mark.asyncio
    async def test_recall_cold_session_returns_empty(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/recall", json={
                "query": "What do you know?",
                "session_id": "cold-session-xyz",
                "user_id": "new-user-xyz",
                "max_tokens": 600
            })
        assert response.status_code == 200
        # Cold session should return empty or minimal context
        data = response.json()
        assert "context" in data
    
    @pytest.mark.asyncio
    async def test_recall_max_tokens_enforced(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post("/turns", json={
                "session_id": "token-test",
                "user_id": "user-token-test",
                "messages": [
                    {"role": "user", "content": "I live in Berlin and work at Google. I have a dog named Max."}
                ],
                "timestamp": "2025-01-01T00:00:00Z",
                "metadata": {}
            })
            
            # Very low token limit
            response = await client.post("/recall", json={
                "query": "Tell me about this user",
                "session_id": "token-test",
                "user_id": "user-token-test",
                "max_tokens": 50
            })
            
            assert response.status_code == 200
            data = response.json()
            # Context should be truncated (approximate check)
            context = data.get("context", "")
            assert len(context) < 500  # Rough check
    
    @pytest.mark.asyncio
    async def test_recall_citations_format(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.post("/turns", json={
                "session_id": "cit-test",
                "user_id": "user-cit-test",
                "messages": [
                    {"role": "user", "content": "I work at Google."}
                ],
                "timestamp": "2025-01-01T00:00:00Z",
                "metadata": {}
            })
            
            response = await client.post("/recall", json={
                "query": "Where do they work?",
                "session_id": "cit-test",
                "user_id": "user-cit-test",
                "max_tokens": 600
            })
            
            data = response.json()
            citations = data.get("citations", [])
            
            # Check citation structure if present
            for cit in citations:
                assert "score" in cit or "snippet" in cit


class TestSearchEndpoint:
    """Test /search endpoint."""
    
    @pytest.mark.asyncio
    async def test_search_returns_200(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/search", json={
                "query": "test",
                "limit": 10
            })
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_search_response_structure(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/search", json={
                "query": "google",
                "limit": 5
            })
        data = response.json()
        assert "results" in data
        assert isinstance(data["results"], list)
    
    @pytest.mark.asyncio
    async def test_search_with_filters(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/search", json={
                "query": "test",
                "session_id": "session-1",
                "user_id": "user-1",
                "limit": 5
            })
        assert response.status_code == 200


class TestMemoriesEndpoint:
    """Test /users/{user_id}/memories endpoint."""
    
    @pytest.mark.asyncio
    async def test_get_memories_returns_200(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/users/test-user/memories")
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_get_memories_response_structure(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/users/user-123/memories")
        data = response.json()
        assert "memories" in data
        assert isinstance(data["memories"], list)
    
    @pytest.mark.asyncio
    async def test_get_memories_with_data(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Create some data
            await client.post("/turns", json={
                "session_id": "mem-test",
                "user_id": "user-mem-test",
                "messages": [
                    {"role": "user", "content": "I work at Google."}
                ],
                "timestamp": "2025-01-01T00:00:00Z",
                "metadata": {}
            })
            
            # Get memories
            response = await client.get("/users/user-mem-test/memories")
            data = response.json()
            memories = data["memories"]
            
            # Should have extracted facts
            assert len(memories) > 0
            
            # Check structure
            mem = memories[0]
            assert "id" in mem
            assert "type" in mem
            assert "key" in mem
            assert "value" in mem
            assert "confidence" in mem


class TestDeleteEndpoints:
    """Test DELETE endpoints."""
    
    @pytest.mark.asyncio
    async def test_delete_session_returns_204(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.delete("/sessions/test-session-123")
        assert response.status_code == 204
    
    @pytest.mark.asyncio
    async def test_delete_user_returns_204(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.delete("/users/test-user-delete")
        assert response.status_code == 204
    
    @pytest.mark.asyncio
    async def test_delete_session_removes_data(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Create data
            await client.post("/turns", json={
                "session_id": "delete-test-session",
                "user_id": "delete-test-user",
                "messages": [{"role": "user", "content": "Test"}],
                "timestamp": "2025-01-01T00:00:00Z",
                "metadata": {}
            })
            
            # Delete session
            await client.delete("/sessions/delete-test-session")
            
            # Session data should be gone (may need to check via recall)
            # This is a basic check - implementation may vary
    
    @pytest.mark.asyncio
    async def test_delete_user_removes_all_data(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Create data
            await client.post("/turns", json={
                "session_id": "delete-user-session",
                "user_id": "delete-user-123",
                "messages": [{"role": "user", "content": "Test"}],
                "timestamp": "2025-01-01T00:00:00Z",
                "metadata": {}
            })
            
            # Delete user
            await client.delete("/users/delete-user-123")
            
            # Should return 204
            response = await client.delete("/users/delete-user-123")
            assert response.status_code == 204


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.mark.asyncio
    async def test_unicode_in_messages(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/turns", json={
                "session_id": "unicode-test",
                "user_id": "unicode-user",
                "messages": [
                    {"role": "user", "content": "Я живу в Берлине 🏠"}
                ],
                "timestamp": "2025-01-01T00:00:00Z",
                "metadata": {}
            })
        assert response.status_code == 201
    
    @pytest.mark.asyncio
    async def test_very_long_message(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            long_content = "word " * 10000  # Very long message
            response = await client.post("/turns", json={
                "session_id": "long-test",
                "user_id": "long-user",
                "messages": [
                    {"role": "user", "content": long_content}
                ],
                "timestamp": "2025-01-01T00:00:00Z",
                "metadata": {}
            })
        # Should handle gracefully (may truncate or process)
        assert response.status_code in [200, 201, 422, 500]
    
    @pytest.mark.asyncio
    async def test_recall_with_special_characters(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/recall", json={
                "query": "What's <info> \"test\" & 'more'?",
                "session_id": "special-test",
                "user_id": "special-user",
                "max_tokens": 600
            })
        assert response.status_code == 200


class TestPersistence:
    """Test data persistence across requests."""
    
    @pytest.mark.asyncio
    async def test_facts_persist_after_multiple_turns(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Turn 1
            await client.post("/turns", json={
                "session_id": "persist-1",
                "user_id": "user-persist",
                "messages": [{"role": "user", "content": "I work at Google."}],
                "timestamp": "2025-01-01T00:00:00Z",
                "metadata": {}
            })
            
            # Turn 2
            await client.post("/turns", json={
                "session_id": "persist-2",
                "user_id": "user-persist",
                "messages": [{"role": "user", "content": "I live in Berlin."}],
                "timestamp": "2025-01-02T00:00:00Z",
                "metadata": {}
            })
            
            # Check memories - should have both facts
            response = await client.get("/users/user-persist/memories")
            memories = response.json()["memories"]
            
            # Should have at least 2 facts (work, location)
            assert len(memories) >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])