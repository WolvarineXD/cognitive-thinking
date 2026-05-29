"""
Reranker — quality scoring and reranking of retrieved memories.

Prevents retrieval pollution by:
1. Scoring memory quality (coherence, specificity, freshness)
2. Filtering low-quality or contradicted memories
3. Diversity-aware reranking (avoid redundant results)
"""
from models.schemas import ScoredMemory
from core.embedding_engine import EmbeddingEngine


class Reranker:
    """Reranks and filters retrieved memories for quality."""

    def __init__(self, embedder: EmbeddingEngine):
        self.embedder = embedder

    def rerank(self, scored_memories: list[ScoredMemory],
               query_embedding: list[float],
               diversity_threshold: float = 0.85) -> list[ScoredMemory]:
        """
        Rerank memories with quality filtering and diversity enforcement.

        Steps:
        1. Filter contradicted/archived memories
        2. Score memory quality
        3. Enforce diversity (remove near-duplicates)
        4. Re-sort by adjusted score
        """
        # Step 1: Filter
        filtered = [
            sm for sm in scored_memories
            if not sm.memory.is_archived and not sm.memory.is_contradiction
        ]

        # Step 2: Quality scoring
        for sm in filtered:
            quality = self._compute_quality(sm)
            sm.total_score *= quality  # Adjust total score

        # Step 3: Diversity enforcement
        diverse = self._enforce_diversity(filtered, diversity_threshold)

        # Step 4: Re-sort
        diverse.sort(key=lambda x: x.total_score, reverse=True)

        return diverse

    def _compute_quality(self, sm: ScoredMemory) -> float:
        """
        Quality score (0-1) based on:
        - Content length (too short = low quality)
        - Access count (frequently accessed = validated)
        - Has entities (structured = higher quality)
        """
        mem = sm.memory
        score = 1.0

        # Penalize very short memories
        if len(mem.content) < 20:
            score *= 0.5
        elif len(mem.content) < 50:
            score *= 0.8

        # Boost frequently accessed memories
        if mem.access_count > 5:
            score *= 1.1
        elif mem.access_count > 10:
            score *= 1.2

        # Boost memories with extracted entities
        if mem.entities:
            score *= 1.05

        return min(1.5, score)  # Cap at 1.5x

    def _enforce_diversity(self, scored: list[ScoredMemory],
                           threshold: float) -> list[ScoredMemory]:
        """Remove near-duplicate results to ensure diversity."""
        if not scored:
            return []

        diverse = [scored[0]]
        for sm in scored[1:]:
            is_duplicate = False
            for kept in diverse:
                if (kept.memory.embedding and sm.memory.embedding):
                    sim = self._cosine_sim(kept.memory.embedding, sm.memory.embedding)
                    if sim > threshold:
                        is_duplicate = True
                        break
                # Fallback: check content overlap
                elif self._content_overlap(kept.memory.content, sm.memory.content) > 0.8:
                    is_duplicate = True
                    break

            if not is_duplicate:
                diverse.append(sm)

        return diverse

    def _cosine_sim(self, a: list[float], b: list[float]) -> float:
        """Quick cosine similarity without numpy dependency in hot path."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _content_overlap(self, a: str, b: str) -> float:
        """Jaccard similarity between word sets."""
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = words_a & words_b
        union = words_a | words_b
        return len(intersection) / len(union)
