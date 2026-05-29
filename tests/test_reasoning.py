"""
Tests for the Reasoning Engine.
Verifies prepare_context context preparation, graph neighbor expansion, and LLM response generation mocks.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import tempfile
from pathlib import Path
from models.schemas import MemoryRecord, MemoryType, ScoredMemory, RetrievalResult
from core.reasoning_engine import ReasoningEngine
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker
from retrieval.context_packer import ContextPacker
from storage.graph_store import GraphStore
from storage.relational_store import RelationalStore
from core.embedding_engine import EmbeddingEngine


class MockEmbedder:
    def embed(self, text: str):
        return [0.1] * 384
    def similarity(self, a, b):
        return 0.9


class MockRetriever:
    def __init__(self, relational):
        self.relational = relational
        class MockWeights:
            semantic = 0.5
            recency = 0.1
            importance = 0.1
            graph_connectivity = 0.2
            user_priority = 0.1
        self.weights = MockWeights()
        
    def retrieve(self, query):
        mem = MemoryRecord(
            id="mem1",
            memory_type=MemoryType.SEMANTIC,
            content="This is retrieved memory 1 about AI projects",
        )
        sm = ScoredMemory(
            memory=mem,
            total_score=0.9,
            semantic_score=0.9,
        )
        return RetrievalResult(
            query=query.query_text,
            results=[sm],
            total_candidates=1,
            retrieval_time_ms=5.0,
        )
        
    def _compute_recency(self, memory):
        return 1.0


def test_reasoning_engine_prepare_context():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        relational = RelationalStore(db_path)
        graph = GraphStore(Path(tmpdir))
        
        try:
            # Save neighbor memory
            neighbor_mem = MemoryRecord(
                id="mem2",
                memory_type=MemoryType.SEMANTIC,
                content="This is the graph neighbor memory",
            )
            relational.store_memory(neighbor_mem)
            graph.add_memory_node(neighbor_mem)
            
            # Save original memory
            original_mem = MemoryRecord(
                id="mem1",
                memory_type=MemoryType.SEMANTIC,
                content="This is retrieved memory 1 about AI projects",
            )
            graph.add_memory_node(original_mem)
            graph.add_relationship("mem1", "mem2", "related")
            
            embedder = MockEmbedder()
            retriever = MockRetriever(relational)
            reranker = Reranker(embedder)
            packer = ContextPacker()
            
            engine = ReasoningEngine(retriever, reranker, packer, graph, embedder)
            prep = engine.prepare_context("AI projects", top_k=2)
            
            assert "context" in prep
            assert prep["graph_neighbors_found"] == 1
            assert len(prep["all_memories"]) == 2
            assert prep["all_memories"][0].memory.id == "mem1"
            assert prep["all_memories"][1].memory.id == "mem2"
            assert "This is retrieved memory 1" in prep["context"]
            assert "This is the graph neighbor memory" in prep["context"]
        finally:
            relational.close()
