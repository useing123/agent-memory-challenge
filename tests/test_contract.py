import pytest
from httpx import AsyncClient, ASGITransport
from src.main import app


@pytest.mark.asyncio
async def test_health():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_create_turn():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/turns", json={
            "session_id": "test-1",
            "user_id": "user-1",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"}
            ],
            "timestamp": "2025-01-01T00:00:00Z",
            "metadata": {}
        })
    assert response.status_code == 201
    assert "id" in response.json()


@pytest.mark.asyncio
async def test_recall():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/recall", json={
            "query": "test query",
            "session_id": "test-1",
            "user_id": "user-1",
            "max_tokens": 1024
        })
    assert response.status_code == 200
    data = response.json()
    assert "context" in data
    assert "citations" in data


@pytest.mark.asyncio
async def test_search():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/search", json={
            "query": "test",
            "session_id": None,
            "user_id": None,
            "limit": 10
        })
    assert response.status_code == 200
    assert "results" in response.json()


@pytest.mark.asyncio
async def test_get_user_memories():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/users/user-1/memories")
    assert response.status_code == 200
    assert "memories" in response.json()


@pytest.mark.asyncio
async def test_delete_session():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.delete("/sessions/test-session")
    assert response.status_code == 204


@pytest.mark.asyncio
async def test_delete_user():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.delete("/users/test-user")
    assert response.status_code == 204