from pydantic import BaseModel, Field
from typing import Optional, Any
from datetime import datetime


class Message(BaseModel):
    role: str
    content: str
    name: Optional[str] = None


class TurnRequest(BaseModel):
    session_id: str
    user_id: Optional[str] = None
    messages: list[Message]
    timestamp: str
    metadata: dict[str, Any] = {}


class TurnResponse(BaseModel):
    id: str


class RecallRequest(BaseModel):
    query: str
    session_id: str
    user_id: Optional[str] = None
    max_tokens: int = 1024


class Citation(BaseModel):
    turn_id: str
    score: float
    snippet: str


class RecallResponse(BaseModel):
    context: str
    citations: list[Citation] = []


class SearchRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    limit: int = 10


class SearchResult(BaseModel):
    content: str
    score: float
    session_id: str
    timestamp: str
    metadata: dict[str, Any] = {}


class SearchResponse(BaseModel):
    results: list[SearchResult] = []


class Memory(BaseModel):
    id: str
    type: str
    key: str
    value: str
    confidence: float
    source_session: str
    source_turn: str
    created_at: str
    updated_at: str
    supersedes: Optional[str] = None
    active: bool = True


class UserMemoriesResponse(BaseModel):
    memories: list[Memory] = []


class HealthResponse(BaseModel):
    status: str