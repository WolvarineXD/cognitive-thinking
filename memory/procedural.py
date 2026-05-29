"""
Procedural Memory — stores workflows, action patterns, reasoning chains.

Example: "How to evaluate resumes effectively" gets stored as a series of
steps that the AI can recall and execute when similar tasks arise.
"""
from datetime import datetime
from typing import Optional

from memory.base import BaseMemory
from models.schemas import MemoryRecord, MemoryType, Procedure, ProcedureStep
from storage.relational_store import RelationalStore
from storage.vector_store import VectorStore
from storage.graph_store import GraphStore
from core.embedding_engine import EmbeddingEngine


class ProceduralMemory(BaseMemory):
    """Manages procedural (workflow/strategy) memories."""

    def __init__(self, relational: RelationalStore, vector: VectorStore,
                 graph: GraphStore, embedder: EmbeddingEngine):
        self.relational = relational
        self.vector = vector
        self.graph = graph
        self.embedder = embedder

    def store(self, memory: MemoryRecord) -> str:
        memory.memory_type = MemoryType.PROCEDURAL

        if memory.embedding is None:
            memory.embedding = self.embedder.embed(memory.content)

        self.relational.store_memory(memory)
        self.vector.store_embedding(memory)
        self.graph.add_memory_node(memory)
        self.graph.auto_link_by_entities(memory)
        return memory.id

    def store_procedure(self, name: str, description: str,
                        steps: list[dict], tags: list[str] = None) -> str:
        """Store a structured procedure (workflow)."""
        proc_steps = [
            ProcedureStep(order=i + 1, action=s["action"],
                          details=s.get("details"), conditions=s.get("conditions"))
            for i, s in enumerate(steps)
        ]
        procedure = Procedure(
            name=name, description=description,
            steps=proc_steps, tags=tags or []
        )
        self.relational.store_procedure(procedure)

        # Also store as a memory record for semantic retrieval
        content = f"Procedure: {name}\n{description}\nSteps:\n"
        for step in proc_steps:
            content += f"  {step.order}. {step.action}"
            if step.details:
                content += f" — {step.details}"
            content += "\n"

        memory = MemoryRecord(
            memory_type=MemoryType.PROCEDURAL,
            content=content,
            tags=tags or [],
            importance=0.7,  # Procedures are inherently important
        )
        return self.store(memory)

    def retrieve(self, query: str, top_k: int = 5) -> list[MemoryRecord]:
        query_embedding = self.embedder.embed(query)
        results = self.vector.search_similar(
            query_embedding, top_k=top_k, memory_type=MemoryType.PROCEDURAL
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
        memory.memory_type = MemoryType.PROCEDURAL
        if memory.embedding is None:
            memory.embedding = self.embedder.embed(memory.content)
        self.relational.update_memory(memory)
        self.vector.store_embedding(memory)

    def delete(self, memory_id: str):
        self.relational.delete_memory(memory_id)
        self.vector.delete_embedding(memory_id, MemoryType.PROCEDURAL)
        self.graph.remove_node(memory_id)

    def count(self) -> int:
        counts = self.relational.get_memory_count()
        return counts.get(MemoryType.PROCEDURAL.value, 0)
