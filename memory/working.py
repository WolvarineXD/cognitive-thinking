"""
Working Memory — short-term active context.

This is the RAM of the system. Stores current conversation context,
immediate reasoning state, and temporary data.

Uses in-memory dict with TTL-based eviction.
Can be upgraded to Redis for production.
"""
import time
from typing import Optional
from memory.base import BaseMemory
from models.schemas import MemoryRecord, MemoryType
from config import config


class WorkingMemoryItem:
    """A working memory item with expiration."""
    def __init__(self, memory: MemoryRecord, ttl: int):
        self.memory = memory
        self.created_at = time.time()
        self.ttl = ttl
        self.last_accessed = time.time()

    @property
    def is_expired(self) -> bool:
        return (time.time() - self.last_accessed) > self.ttl

    def touch(self):
        self.last_accessed = time.time()


class WorkingMemory(BaseMemory):
    """In-memory short-term context store with TTL eviction."""

    def __init__(self):
        self._store: dict[str, WorkingMemoryItem] = {}
        self._max_items = config.working_memory.max_items
        self._ttl = config.working_memory.ttl_seconds

    def store(self, memory: MemoryRecord) -> str:
        memory.memory_type = MemoryType.WORKING
        self._evict_expired()

        # If at capacity, evict oldest
        if len(self._store) >= self._max_items:
            self._evict_oldest()

        self._store[memory.id] = WorkingMemoryItem(memory, self._ttl)
        return memory.id

    def retrieve(self, query: str, top_k: int = 5) -> list[MemoryRecord]:
        """Return all non-expired working memories (simple text match)."""
        self._evict_expired()
        query_lower = query.lower()
        scored = []
        for item in self._store.values():
            if not item.is_expired:
                # Simple relevance: how many query words appear in memory
                content_lower = item.memory.content.lower()
                words = query_lower.split()
                hits = sum(1 for w in words if w in content_lower)
                if hits > 0:
                    scored.append((hits, item.memory))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:top_k]]

    def get(self, memory_id: str) -> Optional[MemoryRecord]:
        item = self._store.get(memory_id)
        if item and not item.is_expired:
            item.touch()
            return item.memory
        return None

    def update(self, memory: MemoryRecord):
        if memory.id in self._store:
            self._store[memory.id].memory = memory
            self._store[memory.id].touch()

    def delete(self, memory_id: str):
        self._store.pop(memory_id, None)

    def count(self) -> int:
        self._evict_expired()
        return len(self._store)

    def get_all_active(self) -> list[MemoryRecord]:
        """Get all non-expired working memories."""
        self._evict_expired()
        return [item.memory for item in self._store.values()]

    def clear(self):
        """Clear all working memory."""
        self._store.clear()

    def _evict_expired(self):
        expired = [k for k, v in self._store.items() if v.is_expired]
        for k in expired:
            del self._store[k]

    def _evict_oldest(self):
        if not self._store:
            return
        oldest_key = min(self._store, key=lambda k: self._store[k].last_accessed)
        del self._store[oldest_key]
