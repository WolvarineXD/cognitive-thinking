"""
Episodic Memory — stores conversations, events, interactions, experiences.

Each "episode" is a group of related memories forming one coherent interaction.
Think of it like: "The user and I discussed resume AI on May 26th."
"""
from datetime import datetime
from typing import Optional

from memory.base import BaseMemory
from models.schemas import MemoryRecord, MemoryType, Episode
from storage.relational_store import RelationalStore
from storage.vector_store import VectorStore
from storage.graph_store import GraphStore
from core.embedding_engine import EmbeddingEngine


class EpisodicMemory(BaseMemory):
    """Manages episodic (event-based) memories."""

    def __init__(self, relational: RelationalStore, vector: VectorStore,
                 graph: GraphStore, embedder: EmbeddingEngine):
        self.relational = relational
        self.vector = vector
        self.graph = graph
        self.embedder = embedder
        self._current_episode: Optional[Episode] = None

    def start_episode(self, title: str = None) -> Episode:
        """Begin a new episode (e.g., a conversation session)."""
        episode = Episode(title=title)
        self._current_episode = episode
        self.relational.store_episode(episode)
        return episode

    def end_episode(self):
        """Close the current episode."""
        if self._current_episode:
            self._current_episode.ended_at = datetime.utcnow()
            self.relational.store_episode(self._current_episode)
            self._current_episode = None

    def store(self, memory: MemoryRecord) -> str:
        """Store an episodic memory, linking it to the current episode."""
        memory.memory_type = MemoryType.EPISODIC

        # Link to current episode if active
        if self._current_episode:
            memory.episode_id = self._current_episode.id
            self._current_episode.memory_ids.append(memory.id)
            self.relational.store_episode(self._current_episode)

        # Generate embedding
        if memory.embedding is None:
            memory.embedding = self.embedder.embed(memory.content)

        # Store across all layers
        self.relational.store_memory(memory)
        self.vector.store_embedding(memory)
        self.graph.add_memory_node(memory)
        self.graph.auto_link_by_entities(memory)

        return memory.id

    def retrieve(self, query: str, top_k: int = 5) -> list[MemoryRecord]:
        """Retrieve relevant episodic memories via semantic search."""
        query_embedding = self.embedder.embed(query)
        results = self.vector.search_similar(
            query_embedding, top_k=top_k, memory_type=MemoryType.EPISODIC
        )
        memories = []
        for r in results:
            mem = self.relational.get_memory(r["id"])
            if mem:
                self.relational.touch_memory(mem.id)
                memories.append(mem)
        return memories

    def get(self, memory_id: str) -> Optional[MemoryRecord]:
        return self.relational.get_memory(memory_id)

    def update(self, memory: MemoryRecord):
        memory.memory_type = MemoryType.EPISODIC
        if memory.embedding is None:
            memory.embedding = self.embedder.embed(memory.content)
        self.relational.update_memory(memory)
        self.vector.store_embedding(memory)

    def delete(self, memory_id: str):
        self.relational.delete_memory(memory_id)
        self.vector.delete_embedding(memory_id, MemoryType.EPISODIC)
        self.graph.remove_node(memory_id)

    def count(self) -> int:
        counts = self.relational.get_memory_count()
        return counts.get(MemoryType.EPISODIC.value, 0)

    def get_episode_memories(self, episode_id: str) -> list[MemoryRecord]:
        """Get all memories belonging to a specific episode."""
        all_episodic = self.relational.get_memories_by_type(MemoryType.EPISODIC, limit=1000)
        return [m for m in all_episodic if m.episode_id == episode_id]
