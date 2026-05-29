"""
Interactive Demo -- Cognitive AI Memory System
==============================================

Run this script to see the entire system in action
without needing curl or a browser.

Usage:
    python demo.py
"""
import sys
import time
import json
import io

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def print_header(title: str):
    width = 60
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_step(step: int, description: str):
    print(f"\n{'-' * 50}")
    print(f"  Step {step}: {description}")
    print(f"{'-' * 50}")


def print_json(data: dict, indent: int = 2):
    print(json.dumps(data, indent=indent, default=str))


def main():
    print_header("[BRAIN] COGNITIVE AI MEMORY SYSTEM - LIVE DEMO")
    print("\nInitializing all subsystems...")
    print("  -> Loading embedding model (first run downloads ~90MB)...")

    start = time.time()

    # Import and initialize
    from memory.manager import MemoryManager
    from retrieval.hybrid_retriever import HybridRetriever
    from retrieval.reranker import Reranker
    from retrieval.context_packer import ContextPacker
    from reflection.reflection_engine import ReflectionEngine
    from reflection.decay import DecayEngine
    from graph.memory_graph import MemoryGraph
    from models.schemas import MemoryType, RetrievalQuery

    manager = MemoryManager()
    retriever = HybridRetriever(
        manager.relational, manager.vector, manager.graph, manager.embedder
    )
    reranker = Reranker(manager.embedder)
    packer = ContextPacker()
    reflection = ReflectionEngine(
        manager.relational, manager.vector, manager.graph, manager.embedder
    )
    decay = DecayEngine()
    memory_graph = MemoryGraph(manager.graph, manager.relational)

    elapsed = time.time() - start
    print(f"  -> System ready in {elapsed:.1f}s\n")

    # -- Step 1: Ingest Episodic Memories ---------------------
    print_step(1, "INGESTING EPISODIC MEMORIES (conversations/events)")

    episode = manager.start_episode("AI Architecture Discussion")
    print(f"  Started episode: {episode.id[:8]}...")

    episodic_inputs = [
        "Yesterday I discussed building a resume AI system with the team. We decided to use embeddings and cosine similarity for matching.",
        "The user asked about legal AI automation. They want to process contracts and extract key clauses automatically.",
        "We had a meeting about OCR pipeline optimization. The bottleneck is in the preprocessing stage.",
    ]

    for text in episodic_inputs:
        mem = manager.ingest(text, source="conversation")
        print(f"  [OK] Stored [{mem.memory_type.value}]: \"{text[:60]}...\"")
        print(f"    Entities: {mem.entities[:5]}")
        print(f"    Intent: {mem.intent}")

    manager.end_episode()
    print(f"\n  Episode closed with {len(episodic_inputs)} memories.")

    # -- Step 2: Ingest Semantic Memories ---------------------
    print_step(2, "INGESTING SEMANTIC MEMORIES (knowledge/patterns)")

    semantic_inputs = [
        "User prefers deep AI systems with proper architecture over quick shallow chatbot demos.",
        "Cosine similarity measures the angle between two vectors. Values range from -1 to 1, where 1 means identical direction.",
        "Retrieval-Augmented Generation (RAG) combines retrieval from a knowledge base with LLM generation for accurate responses.",
        "Transformer architectures use self-attention mechanisms to process sequences in parallel rather than sequentially.",
    ]

    for text in semantic_inputs:
        mem = manager.ingest(text, source="knowledge_base", importance=0.8)
        print(f"  [OK] Stored [{mem.memory_type.value}]: \"{text[:60]}...\"")

    # -- Step 3: Ingest Procedural Memories --------------------
    print_step(3, "INGESTING PROCEDURAL MEMORIES (workflows/strategies)")

    proc_inputs = [
        "How to evaluate resumes effectively: Step 1 - Parse PDF to text. Step 2 - Extract skills using NLP. Step 3 - Generate embeddings. Step 4 - Compare with job description. Step 5 - Score and rank candidates.",
        "How to build a RAG pipeline: First, chunk documents. Then generate embeddings. Next, store in vector database. Finally, retrieve and augment LLM context.",
    ]

    for text in proc_inputs:
        mem = manager.ingest(text, source="workflow", importance=0.7)
        print(f"  [OK] Stored [{mem.memory_type.value}]: \"{text[:60]}...\"")

    # -- Step 4: Hybrid Retrieval -----------------------------
    print_step(4, "HYBRID RETRIEVAL (multi-factor scoring)")

    queries = [
        "How does resume matching work?",
        "What are the user's preferences for AI projects?",
        "Tell me about vector embeddings and similarity",
    ]

    for query_text in queries:
        print(f"\n  Query: \"{query_text}\"")
        query = RetrievalQuery(query_text=query_text, top_k=3)
        result = retriever.retrieve(query)

        # Rerank
        query_emb = manager.embedder.embed(query_text)
        reranked = reranker.rerank(result.results, query_emb)

        print(f"  Found {result.total_candidates} candidates in {result.retrieval_time_ms:.1f}ms")
        print(f"  Top {len(reranked)} after reranking:")

        for i, sm in enumerate(reranked[:3]):
            print(f"\n    [{i+1}] Score: {sm.total_score:.4f}")
            print(f"        Semantic: {sm.semantic_score:.3f} | "
                  f"Recency: {sm.recency_score:.3f} | "
                  f"Importance: {sm.importance_score:.3f} | "
                  f"Graph: {sm.graph_score:.3f}")
            print(f"        Type: {sm.memory.memory_type.value}")
            print(f"        Content: \"{sm.memory.content[:80]}...\"")

    # -- Step 5: Context Packing ------------------------------
    print_step(5, "CONTEXT PACKING (token budget optimization)")

    query = RetrievalQuery(query_text="How to build AI systems?", top_k=5)
    result = retriever.retrieve(query)
    packed = packer.pack(result.results, "How to build AI systems?")

    print(f"  Packed {len(result.results)} memories into context")
    print(f"  Estimated tokens: {packer.estimate_tokens(packed)}")
    print(f"\n  --- Packed Context Preview ---")
    print(packed[:500])
    if len(packed) > 500:
        print(f"  ... [{len(packed) - 500} more chars]")

    # -- Step 6: Knowledge Graph ------------------------------
    print_step(6, "KNOWLEDGE GRAPH (relationships between memories)")

    stats = manager.graph.get_stats()
    print(f"  Graph Stats:")
    print(f"    Nodes: {stats['nodes']}")
    print(f"    Edges: {stats['edges']}")
    print(f"    Clusters: {stats['clusters']}")
    print(f"    Density: {stats['density']:.4f}")

    clusters = memory_graph.get_topic_clusters()
    if clusters:
        print(f"\n  Topic Clusters:")
        for i, cluster in enumerate(clusters[:5]):
            print(f"    Cluster {i+1}: {cluster['size']} memories")
            print(f"      Topics: {cluster['topics'][:5]}")

    # -- Step 7: Reflection -----------------------------------
    print_step(7, "REFLECTION ENGINE (maintenance & insights)")

    report = reflection.run_reflection_cycle()
    print(f"  Memories reviewed: {report.memories_reviewed}")
    print(f"  Duplicates merged: {report.duplicates_merged}")
    print(f"  Contradictions found: {report.contradictions_found}")
    print(f"  Summaries created: {report.summaries_created}")
    print(f"  Memories pruned: {report.memories_pruned}")
    if report.patterns_extracted:
        print(f"\n  Patterns Detected:")
        for pattern in report.patterns_extracted:
            print(f"    * {pattern}")
    if report.insights:
        print(f"\n  Insights:")
        for insight in report.insights:
            print(f"    > {insight}")

    # -- Step 8: System Stats ---------------------------------
    print_step(8, "SYSTEM STATISTICS")

    stats = manager.get_stats()
    print(f"  Memory Counts by Type:")
    for mem_type, count in stats["memory_counts"].items():
        print(f"    {mem_type}: {count}")

    print(f"\n  Vector Store Collections:")
    for name, count in stats["vector_stats"].items():
        print(f"    {name}: {count} embeddings")

    print(f"\n  Graph: {stats['graph_stats']}")
    print(f"  Working Memory Active: {stats['working_memory_active']}")

    # -- Step 9: Duplicate Detection Demo ---------------------
    print_step(9, "NEAR-DUPLICATE DETECTION (semantic dedup)")

    dup_text = "User prefers deep AI systems with proper architecture over quick shallow chatbot demos."
    mem = manager.ingest(dup_text, source="duplicate_test", importance=0.8)
    print(f"  Tried to ingest near-duplicate...")
    print(f"  Result: {mem.memory_type.value} (ID: {mem.id[:8]}...)")
    print(f"  -> Semantic memory layer auto-detects >95% similarity and reinforces existing memory instead of duplicating.")

    # -- Done -------------------------------------------------
    print_header("[DONE] DEMO COMPLETE")
    print(f"\n  All {sum(stats['memory_counts'].values())} memories stored across:")
    print(f"    - SQLite (metadata)")
    print(f"    - ChromaDB (embeddings)")
    print(f"    - NetworkX (knowledge graph)")
    print(f"\n  Data persisted in: data/")
    print(f"\n  To start the API server:")
    print(f"    uvicorn main:app --reload --port 8000")
    print(f"\n  API docs at: http://localhost:8000/docs\n")

    # Cleanup
    manager.close()


if __name__ == "__main__":
    main()
