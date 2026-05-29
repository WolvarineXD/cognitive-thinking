"""
Hybrid Retrieval Engine — the HEART of the system.

Does NOT just do top-k cosine similarity.
Combines: semantic similarity + recency + importance + graph connectivity + user priority.

Score = 0.4×Semantic + 0.2×Recency + 0.2×Importance + 0.1×GraphConnectivity + 0.1×UserPriority
"""
import math
import time
from datetime import datetime
from typing import Optional

from models.schemas import (
    MemoryRecord, MemoryType, RetrievalQuery, ScoredMemory, RetrievalResult
)
from storage.relational_store import RelationalStore
from storage.vector_store import VectorStore
from storage.graph_store import GraphStore
from core.embedding_engine import EmbeddingEngine
from config import config


class HybridRetriever:
    """Multi-factor memory retrieval with weighted scoring."""

    def __init__(self, relational: RelationalStore, vector: VectorStore,
                 graph: GraphStore, embedder: EmbeddingEngine):
        self.relational = relational
        self.vector = vector
        self.graph = graph
        self.embedder = embedder
        self.weights = config.retrieval_weights

    def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        """Full hybrid retrieval pipeline."""
        start_time = time.time()

        # Step 1: Get candidates via semantic search (cast a wide net)
        query_embedding = self.embedder.embed(query.query_text)
        candidate_k = query.top_k * 3  # Fetch 3x more for reranking

        semantic_results = self.vector.search_similar(
            query_embedding,
            top_k=candidate_k,
            memory_type=query.memory_types[0] if query.memory_types and len(query.memory_types) == 1 else None,
            min_importance=query.min_importance,
            include_archived=query.include_archived,
        )

        # Step 2: Score each candidate with the full formula
        scored_memories = []
        for result in semantic_results:
            memory = self.relational.get_memory(result["id"])
            if memory is None:
                continue

            # Apply type filter
            if query.memory_types and memory.memory_type not in query.memory_types:
                continue

            # Apply time filter
            if query.time_range_hours:
                hours_ago = (datetime.utcnow() - memory.created_at).total_seconds() / 3600
                if hours_ago > query.time_range_hours:
                    continue

            # Compute all scoring factors
            semantic_score = result["similarity"]
            recency_score = self._compute_recency(memory)
            importance_score = memory.importance * memory.decay_factor
            graph_score = self.graph.get_connectivity_score(memory.id)
            priority_score = memory.user_priority

            # Weighted fusion
            total = (
                self.weights.semantic * semantic_score +
                self.weights.recency * recency_score +
                self.weights.importance * importance_score +
                self.weights.graph_connectivity * graph_score +
                self.weights.user_priority * priority_score
            )

            scored_memories.append(ScoredMemory(
                memory=memory,
                total_score=total,
                semantic_score=semantic_score,
                recency_score=recency_score,
                importance_score=importance_score,
                graph_score=graph_score,
                priority_score=priority_score,
            ))

        # Step 3: Sort by total score and take top_k
        scored_memories.sort(key=lambda x: x.total_score, reverse=True)
        top_results = scored_memories[:query.top_k]

        # Touch accessed memories (reinforcement)
        for sm in top_results:
            self.relational.touch_memory(sm.memory.id)

        elapsed_ms = (time.time() - start_time) * 1000

        return RetrievalResult(
            query=query.query_text,
            results=top_results,
            total_candidates=len(semantic_results),
            retrieval_time_ms=elapsed_ms,
        )

    def _compute_recency(self, memory: MemoryRecord) -> float:
        """
        Exponential decay based on time since creation.
        Score = exp(-λ × hours_since_creation)
        """
        hours_since = (datetime.utcnow() - memory.created_at).total_seconds() / 3600
        lambda_factor = config.decay.lambda_factor
        return math.exp(-lambda_factor * hours_since)

    def retrieve_simple(self, query_text: str, top_k: int = 5) -> list[ScoredMemory]:
        """Convenience method with sensible defaults."""
        query = RetrievalQuery(query_text=query_text, top_k=top_k)
        result = self.retrieve(query)
        return result.results
