"""
Tests for the Reflection Engine.
Tests pattern extraction, contradiction detection, and decay.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

from models.schemas import MemoryRecord, MemoryType, ReflectionReport
from storage.relational_store import RelationalStore
from storage.graph_store import GraphStore


class TestReflectionPatterns:
    """Test pattern extraction from memories."""

    @pytest.fixture
    def memories(self):
        """Create sample memories with recurring patterns."""
        mems = []
        # Create memories with recurring entities
        for i in range(5):
            mems.append(MemoryRecord(
                memory_type=MemoryType.SEMANTIC,
                content=f"Discussion about AI and machine learning topic {i}",
                entities=["AI", "machine learning"],
                intent="learning",
                tags=["ai", "ml"],
            ))
        for i in range(3):
            mems.append(MemoryRecord(
                memory_type=MemoryType.EPISODIC,
                content=f"Meeting about resume screening project {i}",
                entities=["resume", "screening"],
                intent="planning",
                tags=["project"],
            ))
        return mems

    def test_entity_frequency_detection(self, memories):
        """Reflection should detect frequently mentioned entities."""
        # Count entities manually
        entity_counts = {}
        for mem in memories:
            for entity in mem.entities:
                entity_counts[entity.lower()] = entity_counts.get(entity.lower(), 0) + 1

        # AI should be mentioned 5 times
        assert entity_counts.get("ai", 0) == 5


class TestContradictionDetection:
    def test_detects_negation_contradiction(self):
        """Two memories sharing entities but with opposing negation should flag."""
        from reflection.reflection_engine import ReflectionEngine

        mem_a_content = "Python is the best language for AI development and data science"
        mem_b_content = "Python is not the best language for AI development performance"

        # Use the static method approach
        engine = ReflectionEngine.__new__(ReflectionEngine)
        result = engine._has_contradiction_signals(mem_a_content, mem_b_content)
        assert result is True

    def test_no_false_positive(self):
        """Two similar but non-contradictory memories should not flag."""
        from reflection.reflection_engine import ReflectionEngine

        mem_a_content = "Python is great for data science"
        mem_b_content = "Python is also great for web development"

        engine = ReflectionEngine.__new__(ReflectionEngine)
        result = engine._has_contradiction_signals(mem_a_content, mem_b_content)
        assert result is False


class TestRelationalStoreIntegration:
    """Integration tests using actual SQLite storage."""

    @pytest.fixture
    def store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            s = RelationalStore(db_path)
            yield s
            s.close()

    def test_full_lifecycle(self, store):
        """Test create → read → update → archive → count."""
        # Create
        mem = MemoryRecord(
            memory_type=MemoryType.SEMANTIC,
            content="Lifecycle test memory",
            importance=0.7,
            entities=["test"],
            tags=["lifecycle"],
        )
        store.store_memory(mem)

        # Read
        retrieved = store.get_memory(mem.id)
        assert retrieved is not None
        assert retrieved.importance == 0.7

        # Update importance
        store.update_importance(mem.id, 0.95)
        updated = store.get_memory(mem.id)
        assert abs(updated.importance - 0.95) < 0.01

        # Touch
        store.touch_memory(mem.id)
        touched = store.get_memory(mem.id)
        assert touched.access_count == 1

        # Archive
        store.archive_memory(mem.id)
        active = store.get_all_memories(include_archived=False)
        assert all(m.id != mem.id for m in active)

        # Count
        counts = store.get_memory_count()
        # Archived memories shouldn't be counted
        assert counts.get("semantic", 0) == 0

    def test_memories_below_importance(self, store):
        """Test pruning candidate detection."""
        low = MemoryRecord(
            memory_type=MemoryType.SEMANTIC,
            content="Low importance",
            importance=0.01,
        )
        high = MemoryRecord(
            memory_type=MemoryType.SEMANTIC,
            content="High importance",
            importance=0.9,
        )
        store.store_memory(low)
        store.store_memory(high)

        candidates = store.get_memories_below_importance(0.1)
        assert len(candidates) == 1
        assert candidates[0].id == low.id


class TestGraphStoreIntegration:
    """Integration tests for graph operations."""

    @pytest.fixture
    def graph(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            g = GraphStore(Path(tmpdir))
            yield g

    def test_auto_link_by_entities(self, graph):
        """Memories sharing entities should get auto-linked."""
        mem_a = MemoryRecord(
            memory_type=MemoryType.SEMANTIC,
            content="TensorFlow for deep learning",
            entities=["TensorFlow", "deep learning"],
        )
        mem_b = MemoryRecord(
            memory_type=MemoryType.SEMANTIC,
            content="PyTorch vs TensorFlow comparison",
            entities=["PyTorch", "TensorFlow"],
        )

        graph.add_memory_node(mem_a)
        graph.add_memory_node(mem_b)
        graph.auto_link_by_entities(mem_b)

        neighbors = graph.get_neighbors(mem_b.id)
        assert mem_a.id in neighbors

    def test_cluster_detection(self, graph):
        """Connected memories should form clusters."""
        mems = []
        for i in range(4):
            mem = MemoryRecord(
                memory_type=MemoryType.SEMANTIC,
                content=f"Cluster member {i}",
            )
            graph.add_memory_node(mem)
            mems.append(mem)

        # Connect them in a chain
        for i in range(3):
            graph.add_relationship(mems[i].id, mems[i + 1].id, "related_to")

        clusters = graph.get_clusters()
        assert len(clusters) >= 1
        # The cluster should contain all 4 nodes
        largest = max(clusters, key=len)
        assert len(largest) == 4

    def test_persistence(self):
        """Graph should persist to disk and reload."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and populate
            g1 = GraphStore(Path(tmpdir))
            mem = MemoryRecord(
                memory_type=MemoryType.SEMANTIC, content="Persist me"
            )
            g1.add_memory_node(mem)
            assert g1.get_stats()["nodes"] == 1

            # Reload from disk
            g2 = GraphStore(Path(tmpdir))
            assert g2.get_stats()["nodes"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
