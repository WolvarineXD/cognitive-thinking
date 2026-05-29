"""
Reasoning Engine — Step 8 of the pipeline.

Uses retrieved memories + knowledge graph context to generate
contextually-aware responses through an LLM.
"""
from typing import Optional
import time

from models.schemas import RetrievalQuery, ScoredMemory
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker
from retrieval.context_packer import ContextPacker
from storage.graph_store import GraphStore
from core.embedding_engine import EmbeddingEngine
from config import config

try:
    import ollama as ollama_client
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False


class ReasoningEngine:
    """Reasons over retrieved memories to generate informed responses."""

    SYSTEM_PROMPT = """You are an AI with a persistent memory system. You have access to 
retrieved memories from past interactions. Use these memories to:
1. Maintain continuity across conversations
2. Reference past discussions and preferences
3. Build on previously established knowledge
4. Acknowledge changes in the user's interests or preferences

When memories are provided, integrate them naturally into your reasoning.
Do not just list memories — synthesize them into coherent understanding."""

    def __init__(self, retriever: HybridRetriever, reranker: Reranker,
                 context_packer: ContextPacker, graph: GraphStore,
                 embedder: EmbeddingEngine):
        self.retriever = retriever
        self.reranker = reranker
        self.packer = context_packer
        self.graph = graph
        self.embedder = embedder

    def prepare_context(self, query: str, top_k: int = 5, relevance_threshold: float = 0.0) -> dict:
        """
        Runs the retrieval, reranking, graph expansion, and context packing pipeline.
        Returns a dict containing:
        - "context": packed context string
        - "reranked_memories": list of semantic ScoredMemory objects
        - "expanded_memories": list of graph-expanded ScoredMemory objects
        - "all_memories": combined list of memories
        - "retrieval_time_ms": elapsed time in milliseconds
        - "graph_neighbors_found": count of neighbor IDs found
        """
        start_time = time.time()

        # Step 1: Hybrid retrieval (semantic search + multi-factor scoring)
        retrieval_query = RetrievalQuery(query_text=query, top_k=top_k * 2)
        retrieval_result = self.retriever.retrieve(retrieval_query)

        # Step 2: Rerank (diversity and quality)
        query_embedding = self.embedder.embed(query)
        reranked = self.reranker.rerank(
            retrieval_result.results, query_embedding
        )[:top_k]

        # Step 3: Expand via graph (get related memories)
        expanded_ids = set()
        for sm in reranked:
            neighbors = self.graph.get_neighbors(sm.memory.id, depth=1)
            expanded_ids.update(neighbors)

        # Remove IDs that are already in reranked
        existing_ids = {sm.memory.id for sm in reranked}
        new_neighbor_ids = expanded_ids - existing_ids

        # Fetch and score neighboring memories using the retriever's scoring formula
        expanded_memories = []
        for mid in new_neighbor_ids:
            memory = self.retriever.relational.get_memory(mid)
            if memory and not memory.is_archived and not memory.is_contradiction:
                # Compute hybrid scoring factors
                neighbor_embedding = self.embedder.embed(memory.content)
                semantic_score = self.embedder.similarity(query_embedding, neighbor_embedding)
                recency_score = self.retriever._compute_recency(memory)
                importance_score = memory.importance * memory.decay_factor
                graph_score = self.graph.get_connectivity_score(memory.id)
                priority_score = memory.user_priority

                # Weighted fusion
                total = (
                    self.retriever.weights.semantic * semantic_score +
                    self.retriever.weights.recency * recency_score +
                    self.retriever.weights.importance * importance_score +
                    self.retriever.weights.graph_connectivity * graph_score +
                    self.retriever.weights.user_priority * priority_score
                )

                expanded_memories.append(ScoredMemory(
                    memory=memory,
                    total_score=total,
                    semantic_score=semantic_score,
                    recency_score=recency_score,
                    importance_score=importance_score,
                    graph_score=graph_score,
                    priority_score=priority_score,
                ))

        # Sort expanded memories by total score
        expanded_memories.sort(key=lambda x: x.total_score, reverse=True)

        # Combine: primary semantically-relevant ones, then graph neighbors
        all_memories = reranked + expanded_memories

        # Filter memories for context packing if threshold is specified
        memories_for_packing = all_memories
        if relevance_threshold > 0.0:
            # Filter primary semantic ones by threshold, but keep graph-expanded ones as they are explicitly related
            relevant_semantic = [sm for sm in reranked if sm.semantic_score > relevance_threshold]
            memories_for_packing = relevant_semantic + expanded_memories

        # Step 4: Pack context
        context = self.packer.pack(memories_for_packing, query, self.SYSTEM_PROMPT)

        elapsed_ms = (time.time() - start_time) * 1000

        return {
            "context": context,
            "reranked_memories": reranked,
            "expanded_memories": expanded_memories,
            "all_memories": all_memories,
            "retrieval_time_ms": elapsed_ms,
            "graph_neighbors_found": len(expanded_ids),
        }

    def reason(self, query: str, top_k: int = 5,
               additional_context: str = "") -> dict:
        """
        Full reasoning pipeline:
        1. Retrieve relevant memories (hybrid)
        2. Rerank for quality
        3. Expand via graph neighbors
        4. Pack context
        5. Generate response via LLM
        """
        prep = self.prepare_context(query, top_k=top_k)

        context = prep["context"]
        if additional_context:
            context = f"{additional_context}\n\n{context}"

        # Step 5: Generate response
        response = self._generate_response(query, context)

        return {
            "response": response,
            "memories_used": len(prep["all_memories"]),
            "retrieval_time_ms": prep["retrieval_time_ms"],
            "memories": [
                {
                    "id": sm.memory.id,
                    "content": sm.memory.content[:200],
                    "score": sm.total_score,
                    "type": sm.memory.memory_type.value,
                }
                for sm in prep["all_memories"]
            ],
            "graph_neighbors_found": prep["graph_neighbors_found"],
        }

    def _generate_response(self, query: str, context: str) -> str:
        """Generate a response using the LLM."""
        if not HAS_OLLAMA:
            return (
                f"[LLM not available — Ollama not installed]\n\n"
                f"Query: {query}\n\n"
                f"Retrieved Context:\n{context}"
            )

        try:
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
            ]
            if context:
                messages.append({
                    "role": "system",
                    "content": f"Here are relevant memories:\n\n{context}"
                })
            messages.append({"role": "user", "content": query})

            response = ollama_client.chat(
                model=config.llm.model,
                messages=messages,
                options={
                    "temperature": config.llm.temperature,
                    "num_predict": config.llm.max_tokens,
                }
            )
            return response["message"]["content"]

        except Exception as e:
            return f"[LLM Error: {str(e)}]\n\nRetrieved Context:\n{context}"
