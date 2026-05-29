"""
Abstract base class for all memory types.
Every memory layer must implement this interface.
"""
from abc import ABC, abstractmethod
from typing import Optional
from models.schemas import MemoryRecord


class BaseMemory(ABC):
    """Interface that all memory types must implement."""

    @abstractmethod
    def store(self, memory: MemoryRecord) -> str:
        """Store a memory. Returns memory ID."""
        ...

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[MemoryRecord]:
        """Retrieve relevant memories for a query."""
        ...

    @abstractmethod
    def get(self, memory_id: str) -> Optional[MemoryRecord]:
        """Get a specific memory by ID."""
        ...

    @abstractmethod
    def update(self, memory: MemoryRecord):
        """Update an existing memory."""
        ...

    @abstractmethod
    def delete(self, memory_id: str):
        """Delete a memory."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Get total number of memories in this layer."""
        ...
