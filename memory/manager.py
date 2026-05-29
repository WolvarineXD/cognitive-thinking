"""
Memory Manager — the unified orchestrator.

This is the central controller that routes memories to the correct layer,
coordinates storage across all backends, and provides a single API.
"""
from typing import Optional

from models.schemas import MemoryRecord, MemoryType, MemoryInput, Episode
from core.input_processor import InputProcessor
from core.memory_classifier import MemoryClassifier
from core.embedding_engine import EmbeddingEngine
from memory.episodic import EpisodicMemory
from memory.semantic import SemanticMemory
from memory.working import WorkingMemory
from memory.procedural import ProceduralMemory
from storage.relational_store import RelationalStore
from storage.vector_store import VectorStore
from storage.graph_store import GraphStore
from config import config, SQLITE_PATH, CHROMA_DIR, GRAPH_DIR


class MemoryManager:
    """Unified memory system orchestrator."""

    def __init__(self):
        # Initialize storage backends
        self.relational = RelationalStore(SQLITE_PATH)
        self.vector = VectorStore(CHROMA_DIR)
        self.graph = GraphStore(GRAPH_DIR)

        # Initialize core processors
        self.embedder = EmbeddingEngine()
        self.processor = InputProcessor()
        self.classifier = MemoryClassifier()

        # Initialize memory layers
        self.episodic = EpisodicMemory(self.relational, self.vector, self.graph, self.embedder)
        self.semantic = SemanticMemory(self.relational, self.vector, self.graph, self.embedder)
        self.working = WorkingMemory()
        self.procedural = ProceduralMemory(self.relational, self.vector, self.graph, self.embedder)

        # Map type → layer
        self._layers = {
            MemoryType.EPISODIC: self.episodic,
            MemoryType.SEMANTIC: self.semantic,
            MemoryType.WORKING: self.working,
            MemoryType.PROCEDURAL: self.procedural,
        }

    def ingest(self, content: str, source: str = "user_input",
               memory_type: Optional[MemoryType] = None,
               tags: list[str] = None,
               importance: Optional[float] = None) -> MemoryRecord:
        """
        Full ingestion pipeline:
        1. Process input (clean, extract entities, detect intent)
        2. Classify memory type (if not provided)
        3. Create MemoryRecord
        4. Store in appropriate layer
        """
        # Step 1: Process input
        processed = self.processor.process(content, source)

        # Step 2: Classify
        if memory_type is None:
            memory_type = self.classifier.classify(
                processed["cleaned_text"]
            )

        # Step 3: Create record
        memory = MemoryRecord(
            memory_type=memory_type,
            content=processed["cleaned_text"],
            source=source,
            entities=processed["entities"],
            tags=list(set((tags or []) + processed["tags"])),
            intent=processed["intent"],
            importance=importance if importance is not None else 0.5,
        )

        # Step 4: Store in correct layer
        layer = self._layers[memory_type]
        layer.store(memory)

        # Also add to working memory for immediate context
        if memory_type != MemoryType.WORKING:
            self.working.store(MemoryRecord(
                id=memory.id,
                memory_type=MemoryType.WORKING,
                content=memory.content,
                entities=memory.entities,
                tags=memory.tags,
            ))

        return memory

    def query(self, query_text: str, top_k: int = 10,
              memory_types: list[MemoryType] = None) -> list[MemoryRecord]:
        """
        Query across all memory layers.
        Returns combined results from all specified types.
        """
        if memory_types is None:
            memory_types = [MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.PROCEDURAL]

        all_results = []
        per_type_k = max(3, top_k // len(memory_types))

        for mem_type in memory_types:
            layer = self._layers[mem_type]
            results = layer.retrieve(query_text, top_k=per_type_k)
            all_results.extend(results)

        # Also check working memory
        working_results = self.working.retrieve(query_text, top_k=3)
        all_results.extend(working_results)

        # Deduplicate by ID
        seen = set()
        unique = []
        for mem in all_results:
            if mem.id not in seen:
                seen.add(mem.id)
                unique.append(mem)

        return unique[:top_k]

    def start_episode(self, title: str = None) -> Episode:
        """Start a new conversation episode."""
        return self.episodic.start_episode(title)

    def end_episode(self):
        """End the current episode."""
        self.episodic.end_episode()

    def get_stats(self) -> dict:
        """Get system-wide statistics."""
        return {
            "memory_counts": self.relational.get_memory_count(),
            "vector_stats": self.vector.get_collection_stats(),
            "graph_stats": self.graph.get_stats(),
            "working_memory_active": self.working.count(),
        }

    def close(self):
        """Cleanup resources."""
        self.relational.close()
