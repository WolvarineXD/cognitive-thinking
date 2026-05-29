"""
Tests for the Retrieval system.
Tests hybrid retrieval scoring, reranking, and context packing.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from models.schemas import MemoryRecord, MemoryType, ScoredMemory
from retrieval.reranker import Reranker
from retrieval.context_packer import ContextPacker
from reflection.decay import DecayEngine
from datetime import datetime, timedelta


class TestReranker:
    @pytest.fixture
    def scored_memories(self):
        """Create sample scored memories."""
        memories = []
        for i in range(5):
            mem = MemoryRecord(
                memory_type=MemoryType.SEMANTIC,
                content=f"Memory number {i} about topic {i}",
                importance=0.5 + i * 0.1,
                access_count=i,
                entities=[f"entity_{i}"],
            )
            sm = ScoredMemory(
                memory=mem,
                total_score=0.9 - i * 0.1,
                semantic_score=0.9 - i * 0.1,
            )
            memories.append(sm)
        return memories

    def test_filters_archived(self):
        mem = MemoryRecord(
            memory_type=MemoryType.SEMANTIC,
            content="Archived memory",
            is_archived=True,
        )
        sm = ScoredMemory(memory=mem, total_score=0.9)

        # Mock embedder since we won't actually compute embeddings here
        class MockEmbedder:
            def similarity(self, a, b):
                return 0.5
        reranker = Reranker(MockEmbedder())
        result = reranker.rerank([sm], [0.0] * 384)
        assert len(result) == 0

    def test_filters_contradictions(self):
        mem = MemoryRecord(
            memory_type=MemoryType.SEMANTIC,
            content="Contradicted memory",
            is_contradiction=True,
        )
        sm = ScoredMemory(memory=mem, total_score=0.9)

        class MockEmbedder:
            def similarity(self, a, b):
                return 0.5
        reranker = Reranker(MockEmbedder())
        result = reranker.rerank([sm], [0.0] * 384)
        assert len(result) == 0

    def test_quality_penalizes_short_content(self):
        class MockEmbedder:
            def similarity(self, a, b):
                return 0.5
        reranker = Reranker(MockEmbedder())

        short_mem = ScoredMemory(
            memory=MemoryRecord(
                memory_type=MemoryType.SEMANTIC,
                content="Hi",
            ),
            total_score=0.9,
        )
        long_mem = ScoredMemory(
            memory=MemoryRecord(
                memory_type=MemoryType.SEMANTIC,
                content="This is a much longer memory with substantial content about AI systems",
            ),
            total_score=0.9,
        )

        result = reranker.rerank([short_mem, long_mem], [0.0] * 384)
        # Long memory should rank higher after quality scoring
        assert result[0].memory.content == long_mem.memory.content


class TestContextPacker:
    @pytest.fixture
    def packer(self):
        return ContextPacker()

    def test_packs_memories(self, packer):
        scored = []
        for i in range(3):
            mem = MemoryRecord(
                memory_type=MemoryType.SEMANTIC,
                content=f"Memory content about topic {i}",
                tags=[f"tag_{i}"],
                entities=[f"entity_{i}"],
            )
            scored.append(ScoredMemory(
                memory=mem,
                total_score=0.9 - i * 0.1,
                importance_score=0.5,
            ))

        result = packer.pack(scored, "test query")
        assert "Retrieved Memories" in result
        assert "Memory #1" in result

    def test_empty_input(self, packer):
        result = packer.pack([], "test query")
        assert result == ""

    def test_token_estimation(self, packer):
        tokens = packer.estimate_tokens("This is a test sentence")
        assert tokens > 0
        assert isinstance(tokens, int)


class TestDecayEngine:
    @pytest.fixture
    def decay(self):
        return DecayEngine()

    def test_recent_memory_high_decay(self, decay):
        mem = MemoryRecord(
            memory_type=MemoryType.SEMANTIC,
            content="Recent memory",
            last_accessed=datetime.utcnow(),
        )
        factor = decay.compute_decay(mem)
        assert factor > 0.9  # Very recent, should be close to 1.0

    def test_old_memory_lower_decay(self, decay):
        mem = MemoryRecord(
            memory_type=MemoryType.SEMANTIC,
            content="Old memory",
            last_accessed=datetime.utcnow() - timedelta(days=30),
        )
        factor = decay.compute_decay(mem)
        assert factor < 1.0  # Should have decayed

    def test_reinforcement_boosts(self, decay):
        mem = MemoryRecord(
            memory_type=MemoryType.SEMANTIC,
            content="Reinforced",
            decay_factor=0.5,
        )
        original = mem.decay_factor
        reinforced = decay.reinforce(mem)
        assert reinforced.decay_factor > original

    def test_prune_decision(self, decay):
        low_mem = MemoryRecord(
            memory_type=MemoryType.SEMANTIC,
            content="Low importance",
            importance=0.01,
            last_accessed=datetime.utcnow() - timedelta(days=90),
        )
        assert decay.should_prune(low_mem) is True

        high_mem = MemoryRecord(
            memory_type=MemoryType.SEMANTIC,
            content="High importance",
            importance=0.9,
            last_accessed=datetime.utcnow(),
        )
        assert decay.should_prune(high_mem) is False

    def test_minimum_floor(self, decay):
        mem = MemoryRecord(
            memory_type=MemoryType.SEMANTIC,
            content="Ancient memory",
            last_accessed=datetime.utcnow() - timedelta(days=365),
        )
        factor = decay.compute_decay(mem)
        assert factor >= 0.05  # Should never hit zero


class TestGraphStore:
    def test_add_and_retrieve_neighbors(self):
        import tempfile
        from storage.graph_store import GraphStore
        from models.schemas import GraphEdge

        with tempfile.TemporaryDirectory() as tmpdir:
            graph = GraphStore(Path(tmpdir))

            mem_a = MemoryRecord(
                memory_type=MemoryType.SEMANTIC, content="Memory A",
                entities=["AI"]
            )
            mem_b = MemoryRecord(
                memory_type=MemoryType.SEMANTIC, content="Memory B",
                entities=["AI"]
            )
            graph.add_memory_node(mem_a)
            graph.add_memory_node(mem_b)
            graph.add_relationship(mem_a.id, mem_b.id, "related_to")

            neighbors = graph.get_neighbors(mem_a.id)
            assert mem_b.id in neighbors

    def test_connectivity_score(self):
        import tempfile
        from storage.graph_store import GraphStore

        with tempfile.TemporaryDirectory() as tmpdir:
            graph = GraphStore(Path(tmpdir))

            # Create a hub node connected to many
            hub = MemoryRecord(
                memory_type=MemoryType.SEMANTIC, content="Hub"
            )
            graph.add_memory_node(hub)

            for i in range(5):
                spoke = MemoryRecord(
                    memory_type=MemoryType.SEMANTIC, content=f"Spoke {i}"
                )
                graph.add_memory_node(spoke)
                graph.add_relationship(hub.id, spoke.id, "related_to")

            score = graph.get_connectivity_score(hub.id)
            assert score > 0

    def test_stats(self):
        import tempfile
        from storage.graph_store import GraphStore

        with tempfile.TemporaryDirectory() as tmpdir:
            graph = GraphStore(Path(tmpdir))
            stats = graph.get_stats()
            assert "nodes" in stats
            assert "edges" in stats
            assert stats["nodes"] == 0


# Need Path import for graph tests
from pathlib import Path


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
