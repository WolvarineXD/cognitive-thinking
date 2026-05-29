"""
Semantic Memory — stores generalized knowledge, concepts, abstractions, patterns.

Unlike episodic memory (specific events), semantic memory stores WHAT the user knows
and believes. Example: "User prefers deep AI systems over shallow chatbot projects."
"""
from typing import Optional
from memory.base import BaseMemory
from models.schemas import MemoryRecord, MemoryType
from storage.relational_store import RelationalStore
from storage.vector_store import VectorStore
from storage.graph_store import GraphStore
from core.embedding_engine import EmbeddingEngine


class SemanticMemory(BaseMemory):
    """Manages semantic (knowledge/concept) memories."""

    def __init__(self, relational: RelationalStore, vector: VectorStore,
                 graph: GraphStore, embedder: EmbeddingEngine):
        self.relational = relational
        self.vector = vector
        self.graph = graph
        self.embedder = embedder

    def store(self, memory: MemoryRecord) -> str:
        memory.memory_type = MemoryType.SEMANTIC

        if memory.embedding is None:
            memory.embedding = self.embedder.embed(memory.content)

        # Check for near-duplicates before storing
        existing = self.vector.search_similar(
            memory.embedding, top_k=1, memory_type=MemoryType.SEMANTIC
        )
        if existing and existing[0]["similarity"] > 0.95:
            # Near duplicate — update existing instead of creating new
            existing_mem = self.relational.get_memory(existing[0]["id"])
            if existing_mem:
                existing_mem.access_count += 1
                existing_mem.importance = min(1.0, existing_mem.importance + 0.05)
                self.relational.update_memory(existing_mem)
                return existing_mem.id

        self.relational.store_memory(memory)
        self.vector.store_embedding(memory)
        self.graph.add_memory_node(memory)
        self.graph.auto_link_by_entities(memory)
        return memory.id

    def retrieve(self, query: str, top_k: int = 5) -> list[MemoryRecord]:
        query_embedding = self.embedder.embed(query)
        results = self.vector.search_similar(
            query_embedding, top_k=top_k, memory_type=MemoryType.SEMANTIC
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
        memory.memory_type = MemoryType.SEMANTIC
        if memory.embedding is None:
            memory.embedding = self.embedder.embed(memory.content)
        self.relational.update_memory(memory)
        self.vector.store_embedding(memory)

    def delete(self, memory_id: str):
        self.relational.delete_memory(memory_id)
        self.vector.delete_embedding(memory_id, MemoryType.SEMANTIC)
        self.graph.remove_node(memory_id)

    def count(self) -> int:
        counts = self.relational.get_memory_count()
        return counts.get(MemoryType.SEMANTIC.value, 0)
