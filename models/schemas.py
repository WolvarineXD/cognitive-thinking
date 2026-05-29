"""
Pydantic schemas for every data structure in the memory system.
These are the contracts between all components.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


# ── Enums ──────────────────────────────────────────────────────────────

class MemoryType(str, Enum):
    """The four memory categories."""
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    WORKING = "working"


class EmotionalValence(str, Enum):
    """Emotional coloring of a memory (affects retrieval weight)."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    URGENT = "urgent"


# ── Core Memory Record ────────────────────────────────────────────────

class MemoryRecord(BaseModel):
    """
    The universal memory unit.
    Every piece of stored knowledge becomes one of these.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    memory_type: MemoryType
    content: str                          # Raw text content
    summary: Optional[str] = None         # Compressed summary (filled by reflection)
    embedding: Optional[list[float]] = None  # Vector embedding

    # ── Metadata ──
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    access_count: int = 0
    source: Optional[str] = None          # Where this came from

    # ── Scoring ──
    importance: float = 0.5               # 0.0–1.0 importance score
    user_priority: float = 0.5            # 0.0–1.0 user-set priority
    emotional_valence: EmotionalValence = EmotionalValence.NEUTRAL
    decay_factor: float = 1.0             # Current decay multiplier

    # ── Extracted Info ──
    entities: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    intent: Optional[str] = None

    # ── Relationships ──
    related_memory_ids: list[str] = Field(default_factory=list)
    parent_id: Optional[str] = None       # For hierarchical memories
    episode_id: Optional[str] = None      # Group episodic memories

    # ── Flags ──
    is_archived: bool = False
    is_contradiction: bool = False         # Flagged as contradicting another memory
    contradicts_id: Optional[str] = None

    model_config = ConfigDict(
        json_encoders={datetime: lambda v: v.isoformat()},
    )


# ── Episode (group of related episodic memories) ──────────────────────

class Episode(BaseModel):
    """A group of related episodic memories forming one coherent interaction."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: Optional[str] = None
    memory_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    summary: Optional[str] = None
    tags: list[str] = Field(default_factory=list)


# ── Procedural Knowledge ─────────────────────────────────────────────

class ProcedureStep(BaseModel):
    """A single step in a procedural workflow."""
    order: int
    action: str
    details: Optional[str] = None
    conditions: Optional[str] = None      # When to apply this step


class Procedure(BaseModel):
    """A stored workflow / action pattern."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    steps: list[ProcedureStep] = Field(default_factory=list)
    success_rate: float = 0.5             # How often this pattern succeeds
    use_count: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_used: Optional[datetime] = None
    tags: list[str] = Field(default_factory=list)


# ── Graph Structures ──────────────────────────────────────────────────

class GraphEdge(BaseModel):
    """A relationship between two memory nodes."""
    source_id: str
    target_id: str
    relation_type: str                    # e.g., "related_to", "caused_by", "part_of"
    weight: float = 1.0                   # Strength of relationship
    created_at: datetime = Field(default_factory=datetime.utcnow)


class GraphNode(BaseModel):
    """A node in the memory graph (wraps a MemoryRecord)."""
    memory_id: str
    label: str
    node_type: MemoryType
    centrality: float = 0.0               # Computed graph centrality


# ── Retrieval Structures ──────────────────────────────────────────────

class RetrievalQuery(BaseModel):
    """What the retrieval engine receives."""
    query_text: str
    top_k: int = 10
    memory_types: Optional[list[MemoryType]] = None  # Filter by type
    min_importance: float = 0.0
    include_archived: bool = False
    time_range_hours: Optional[int] = None  # Only recent memories


class ScoredMemory(BaseModel):
    """A memory with its computed retrieval score breakdown."""
    memory: MemoryRecord
    total_score: float
    semantic_score: float = 0.0
    recency_score: float = 0.0
    importance_score: float = 0.0
    graph_score: float = 0.0
    priority_score: float = 0.0


class RetrievalResult(BaseModel):
    """Full retrieval response."""
    query: str
    results: list[ScoredMemory] = Field(default_factory=list)
    total_candidates: int = 0
    retrieval_time_ms: float = 0.0


# ── Reflection Structures ────────────────────────────────────────────

class ReflectionReport(BaseModel):
    """Output of a reflection cycle."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    memories_reviewed: int = 0
    patterns_extracted: list[str] = Field(default_factory=list)
    summaries_created: int = 0
    duplicates_merged: int = 0
    memories_pruned: int = 0
    contradictions_found: int = 0
    insights: list[str] = Field(default_factory=list)


# ── API Request/Response ─────────────────────────────────────────────

class MemoryInput(BaseModel):
    """What the user sends to store a memory."""
    content: str
    source: Optional[str] = "user_input"
    tags: list[str] = Field(default_factory=list)
    importance: Optional[float] = None
    memory_type: Optional[MemoryType] = None  # Auto-classified if None


class ConversationInput(BaseModel):
    """A conversation turn to process."""
    user_message: str
    context: Optional[str] = None
    session_id: Optional[str] = None


class MemoryResponse(BaseModel):
    """API response after storing."""
    memory_id: str
    memory_type: MemoryType
    entities_extracted: list[str]
    related_memories: int
    message: str


class QueryResponse(BaseModel):
    """API response for a memory query."""
    results: list[ScoredMemory]
    context_summary: Optional[str] = None
    total_memories_searched: int
    retrieval_time_ms: float
