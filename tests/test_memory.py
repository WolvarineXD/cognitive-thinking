"""
Tests for Memory layers and the Memory Manager.
Tests storage, retrieval, classification, and the full ingest pipeline.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import tempfile
from pathlib import Path

from config import SystemConfig
from models.schemas import MemoryRecord, MemoryType, EmotionalValence
from core.memory_classifier import MemoryClassifier
from storage.relational_store import RelationalStore


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_memory.db"
        store = RelationalStore(db_path)
        yield store
        store.close()


@pytest.fixture
def classifier():
    return MemoryClassifier()


class TestMemoryClassifier:
    def test_classifies_episodic(self, classifier):
        text = "Yesterday I talked to the team about the project"
        result = classifier.classify(text)
        assert result == MemoryType.EPISODIC

    def test_classifies_semantic(self, classifier):
        text = "Machine learning is a subset of artificial intelligence"
        result = classifier.classify(text)
        assert result == MemoryType.SEMANTIC

    def test_classifies_procedural(self, classifier):
        text = "Step 1: First, install the dependencies. Then build the pipeline."
        result = classifier.classify(text)
        assert result == MemoryType.PROCEDURAL

    def test_classifies_working(self, classifier):
        text = "Right now I'm temporarily holding this draft note"
        result = classifier.classify(text)
        assert result == MemoryType.WORKING

    def test_default_to_semantic(self, classifier):
        text = "Random text with no clear signals"
        result = classifier.classify(text)
        assert result == MemoryType.SEMANTIC

    def test_confidence_scores_sum_to_one(self, classifier):
        text = "How to build a pipeline step by step"
        scores = classifier.classify_with_confidence(text)
        total = sum(scores.values())
        assert abs(total - 1.0) < 0.01


class TestRelationalStore:
    def test_store_and_retrieve(self, temp_db):
        memory = MemoryRecord(
            memory_type=MemoryType.SEMANTIC,
            content="Python is a programming language",
            source="test",
            entities=["Python"],
            tags=["programming"],
        )
        temp_db.store_memory(memory)
        retrieved = temp_db.get_memory(memory.id)
        assert retrieved is not None
        assert retrieved.content == memory.content
        assert retrieved.memory_type == MemoryType.SEMANTIC

    def test_get_nonexistent_returns_none(self, temp_db):
        result = temp_db.get_memory("nonexistent-id")
        assert result is None

    def test_get_memories_by_type(self, temp_db):
        for i in range(5):
            mem = MemoryRecord(
                memory_type=MemoryType.EPISODIC,
                content=f"Episode memory {i}",
            )
            temp_db.store_memory(mem)
        for i in range(3):
            mem = MemoryRecord(
                memory_type=MemoryType.SEMANTIC,
                content=f"Semantic memory {i}",
            )
            temp_db.store_memory(mem)

        episodic = temp_db.get_memories_by_type(MemoryType.EPISODIC)
        semantic = temp_db.get_memories_by_type(MemoryType.SEMANTIC)
        assert len(episodic) == 5
        assert len(semantic) == 3

    def test_archive_memory(self, temp_db):
        memory = MemoryRecord(
            memory_type=MemoryType.SEMANTIC,
            content="To be archived",
        )
        temp_db.store_memory(memory)
        temp_db.archive_memory(memory.id)

        # Should not appear in non-archived results
        results = temp_db.get_memories_by_type(MemoryType.SEMANTIC, include_archived=False)
        assert all(r.id != memory.id for r in results)

        # Should appear in archived results
        results = temp_db.get_memories_by_type(MemoryType.SEMANTIC, include_archived=True)
        assert any(r.id == memory.id for r in results)

    def test_touch_memory(self, temp_db):
        memory = MemoryRecord(
            memory_type=MemoryType.SEMANTIC,
            content="Touch test",
        )
        temp_db.store_memory(memory)
        temp_db.touch_memory(memory.id)

        retrieved = temp_db.get_memory(memory.id)
        assert retrieved.access_count == 1

    def test_update_importance(self, temp_db):
        memory = MemoryRecord(
            memory_type=MemoryType.SEMANTIC,
            content="Importance test",
            importance=0.5,
        )
        temp_db.store_memory(memory)
        temp_db.update_importance(memory.id, 0.9)

        retrieved = temp_db.get_memory(memory.id)
        assert abs(retrieved.importance - 0.9) < 0.01

    def test_memory_count(self, temp_db):
        for i in range(3):
            temp_db.store_memory(MemoryRecord(
                memory_type=MemoryType.EPISODIC, content=f"ep {i}"
            ))
        for i in range(2):
            temp_db.store_memory(MemoryRecord(
                memory_type=MemoryType.SEMANTIC, content=f"sem {i}"
            ))

        counts = temp_db.get_memory_count()
        assert counts.get("episodic", 0) == 3
        assert counts.get("semantic", 0) == 2

    def test_delete_memory(self, temp_db):
        memory = MemoryRecord(
            memory_type=MemoryType.SEMANTIC,
            content="Delete me",
        )
        temp_db.store_memory(memory)
        temp_db.delete_memory(memory.id)
        assert temp_db.get_memory(memory.id) is None

    def test_search_by_tags(self, temp_db):
        mem1 = MemoryRecord(
            memory_type=MemoryType.SEMANTIC,
            content="Tagged memory",
            tags=["python", "ai"],
        )
        mem2 = MemoryRecord(
            memory_type=MemoryType.SEMANTIC,
            content="Other memory",
            tags=["java"],
        )
        temp_db.store_memory(mem1)
        temp_db.store_memory(mem2)

        results = temp_db.search_by_tags(["python"])
        assert len(results) == 1
        assert results[0].id == mem1.id

    def test_search_by_entities(self, temp_db):
        mem = MemoryRecord(
            memory_type=MemoryType.SEMANTIC,
            content="Entity memory",
            entities=["TensorFlow", "PyTorch"],
        )
        temp_db.store_memory(mem)

        results = temp_db.search_by_entities(["tensorflow"])
        assert len(results) == 1


class TestWorkingMemory:
    def test_store_and_retrieve(self):
        from memory.working import WorkingMemory
        wm = WorkingMemory()
        mem = MemoryRecord(
            memory_type=MemoryType.WORKING,
            content="Current conversation about AI",
        )
        wm.store(mem)
        assert wm.count() == 1

        results = wm.retrieve("AI conversation")
        assert len(results) >= 1

    def test_clear(self):
        from memory.working import WorkingMemory
        wm = WorkingMemory()
        for i in range(5):
            wm.store(MemoryRecord(
                memory_type=MemoryType.WORKING,
                content=f"Temp memory {i}",
            ))
        assert wm.count() == 5
        wm.clear()
        assert wm.count() == 0

    def test_get_by_id(self):
        from memory.working import WorkingMemory
        wm = WorkingMemory()
        mem = MemoryRecord(
            memory_type=MemoryType.WORKING,
            content="Specific memory",
        )
        wm.store(mem)
        retrieved = wm.get(mem.id)
        assert retrieved is not None
        assert retrieved.content == "Specific memory"


class TestMemoryRecord:
    def test_default_values(self):
        mem = MemoryRecord(
            memory_type=MemoryType.SEMANTIC,
            content="Test",
        )
        assert mem.importance == 0.5
        assert mem.access_count == 0
        assert mem.is_archived is False
        assert mem.emotional_valence == EmotionalValence.NEUTRAL
        assert mem.id is not None
        assert len(mem.id) > 0

    def test_serialization(self):
        mem = MemoryRecord(
            memory_type=MemoryType.EPISODIC,
            content="Serializable",
            entities=["test"],
            tags=["demo"],
        )
        json_str = mem.model_dump_json()
        assert "Serializable" in json_str
        assert "episodic" in json_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
