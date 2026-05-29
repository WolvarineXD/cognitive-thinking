"""
Reflection Engine — the "consciousness" of the memory system.

Periodically:
1. Reviews all memories
2. Extracts patterns and insights
3. Summarizes clusters of related memories
4. Merges duplicates
5. Updates importance scores
6. Prunes low-value memories
7. Detects contradictions

Without reflection, memory systems become garbage dumps.
"""
from datetime import datetime
from typing import Optional

from models.schemas import MemoryRecord, MemoryType, ReflectionReport
from storage.relational_store import RelationalStore
from storage.vector_store import VectorStore
from storage.graph_store import GraphStore
from core.embedding_engine import EmbeddingEngine
from config import config

try:
    import ollama as ollama_client
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False


class ReflectionEngine:
    """Periodic memory review and optimization engine."""

    def __init__(self, relational: RelationalStore, vector: VectorStore,
                 graph: GraphStore, embedder: EmbeddingEngine):
        self.relational = relational
        self.vector = vector
        self.graph = graph
        self.embedder = embedder
        self.cfg = config.reflection

    def run_reflection_cycle(self) -> ReflectionReport:
        """Execute a full reflection cycle."""
        report = ReflectionReport()

        # Get all active memories
        memories = self.relational.get_all_memories(include_archived=False)
        report.memories_reviewed = len(memories)

        if not memories:
            return report

        # Step 1: Find and merge duplicates
        merged = self._merge_duplicates(memories)
        report.duplicates_merged = merged

        # Step 2: Detect contradictions
        contradictions = self._detect_contradictions(memories)
        report.contradictions_found = contradictions

        # Step 3: Extract patterns
        patterns = self._extract_patterns(memories)
        report.patterns_extracted = patterns

        # Step 4: Create summaries for clusters
        summaries = self._summarize_clusters(memories)
        report.summaries_created = summaries

        # Step 5: Update importance via decay
        self._apply_decay(memories)

        # Step 6: Prune low-value memories
        pruned = self._prune_low_value()
        report.memories_pruned = pruned

        # Step 7: Generate insights (if LLM available)
        insights = self._generate_insights(memories, patterns)
        report.insights = insights

        return report

    def _merge_duplicates(self, memories: list[MemoryRecord]) -> int:
        """Find and merge near-duplicate memories."""
        merged_count = 0
        threshold = self.cfg.similarity_merge_threshold

        # Group by type for efficiency
        by_type: dict[MemoryType, list[MemoryRecord]] = {}
        for mem in memories:
            by_type.setdefault(mem.memory_type, []).append(mem)

        for mem_type, type_memories in by_type.items():
            if len(type_memories) < 2:
                continue

            # Compare each pair (quadratic but necessary for quality)
            skip_ids = set()
            for i, mem_a in enumerate(type_memories):
                if mem_a.id in skip_ids:
                    continue
                for j, mem_b in enumerate(type_memories[i + 1:], i + 1):
                    if mem_b.id in skip_ids:
                        continue

                    # Quick length check before expensive embedding comparison
                    len_ratio = min(len(mem_a.content), len(mem_b.content)) / max(
                        len(mem_a.content), len(mem_b.content), 1
                    )
                    if len_ratio < 0.5:
                        continue

                    # Embedding similarity
                    if mem_a.embedding and mem_b.embedding:
                        sim = self.embedder.similarity(mem_a.embedding, mem_b.embedding)
                    else:
                        # Generate embeddings if missing
                        emb_a = self.embedder.embed(mem_a.content)
                        emb_b = self.embedder.embed(mem_b.content)
                        sim = self.embedder.similarity(emb_a, emb_b)

                    if sim > threshold:
                        # Merge: keep the one with higher importance/access
                        keep, remove = (mem_a, mem_b) if (
                            mem_a.importance + mem_a.access_count >
                            mem_b.importance + mem_b.access_count
                        ) else (mem_b, mem_a)

                        # Transfer knowledge from removed to kept
                        keep.access_count += remove.access_count
                        keep.importance = max(keep.importance, remove.importance)
                        keep.entities = list(set(keep.entities + remove.entities))
                        keep.tags = list(set(keep.tags + remove.tags))
                        keep.related_memory_ids.extend(remove.related_memory_ids)

                        self.relational.update_memory(keep)
                        self.relational.archive_memory(remove.id)
                        skip_ids.add(remove.id)
                        merged_count += 1

        return merged_count

    def _detect_contradictions(self, memories: list[MemoryRecord]) -> int:
        """Detect memories that contradict each other."""
        contradiction_count = 0

        # Focus on semantic memories (most likely to contradict)
        semantic = [m for m in memories if m.memory_type == MemoryType.SEMANTIC]

        for i, mem_a in enumerate(semantic):
            for mem_b in semantic[i + 1:]:
                # Check if they share entities but have opposing content
                shared_entities = set(e.lower() for e in mem_a.entities) & set(
                    e.lower() for e in mem_b.entities
                )
                if not shared_entities:
                    continue

                # Check for negation patterns
                if self._has_contradiction_signals(mem_a.content, mem_b.content):
                    # Flag both
                    mem_a.is_contradiction = True
                    mem_a.contradicts_id = mem_b.id
                    mem_b.is_contradiction = True
                    mem_b.contradicts_id = mem_a.id

                    # Keep the newer one as canonical
                    if mem_a.created_at < mem_b.created_at:
                        mem_a.importance *= 0.5  # Reduce old contradicted memory
                    else:
                        mem_b.importance *= 0.5

                    self.relational.update_memory(mem_a)
                    self.relational.update_memory(mem_b)
                    contradiction_count += 1

        return contradiction_count

    def _has_contradiction_signals(self, text_a: str, text_b: str) -> bool:
        """Check if two texts show signs of contradiction."""
        negation_words = {"not", "never", "don't", "doesn't", "isn't", "wasn't",
                          "no", "none", "neither", "nor", "without", "cannot",
                          "won't", "shouldn't", "wouldn't", "couldn't"}
        a_lower = set(text_a.lower().split())
        b_lower = set(text_b.lower().split())

        # Check if one has negation of what the other states
        a_negations = a_lower & negation_words
        b_negations = b_lower & negation_words

        # Simple heuristic: if one text has negation words and they share
        # significant vocabulary, they might contradict
        shared_words = (a_lower - negation_words) & (b_lower - negation_words)
        if len(shared_words) > 3 and (bool(a_negations) != bool(b_negations)):
            return True

        return False

    def _extract_patterns(self, memories: list[MemoryRecord]) -> list[str]:
        """Extract recurring patterns from memories."""
        patterns = []

        # Entity frequency analysis
        entity_counts: dict[str, int] = {}
        for mem in memories:
            for entity in mem.entities:
                entity_counts[entity.lower()] = entity_counts.get(entity.lower(), 0) + 1

        # Top recurring entities become patterns
        frequent = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
        for entity, count in frequent[:10]:
            if count >= 3:
                patterns.append(f"Recurring topic: '{entity}' (mentioned {count} times)")

        # Intent distribution
        intent_counts: dict[str, int] = {}
        for mem in memories:
            if mem.intent:
                intent_counts[mem.intent] = intent_counts.get(mem.intent, 0) + 1

        for intent, count in sorted(intent_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            if count >= 2:
                patterns.append(f"Common intent: '{intent}' ({count} occurrences)")

        return patterns

    def _summarize_clusters(self, memories: list[MemoryRecord]) -> int:
        """Summarize clusters of related memories into higher-level abstractions."""
        clusters = self.graph.get_clusters()
        summaries_created = 0

        for cluster_ids in clusters:
            if len(cluster_ids) < self.cfg.min_memories_for_summary:
                continue

            # Get memories in this cluster
            cluster_memories = [
                m for m in memories if m.id in cluster_ids
            ]
            if not cluster_memories:
                continue

            # Create a summary memory
            combined_content = "\n".join([m.content for m in cluster_memories[:10]])
            summary_text = self._create_summary(combined_content)

            if summary_text:
                summary_memory = MemoryRecord(
                    memory_type=MemoryType.SEMANTIC,
                    content=summary_text,
                    source="reflection_summary",
                    importance=0.8,
                    tags=["reflection_summary", "cluster_summary"],
                    entities=list(set(
                        e for m in cluster_memories for e in m.entities
                    ))[:15],
                )
                summary_memory.embedding = self.embedder.embed(summary_text)

                # Link summary to cluster members
                for mem in cluster_memories:
                    summary_memory.related_memory_ids.append(mem.id)
                    mem.parent_id = summary_memory.id
                    self.relational.update_memory(mem)

                self.relational.store_memory(summary_memory)
                self.vector.store_embedding(summary_memory)
                self.graph.add_memory_node(summary_memory)

                summaries_created += 1

        return summaries_created

    def _create_summary(self, text: str) -> Optional[str]:
        """Create a summary using LLM or fallback to truncation."""
        if HAS_OLLAMA:
            try:
                response = ollama_client.chat(
                    model=config.llm.model,
                    messages=[{
                        "role": "system",
                        "content": "Summarize the following memories into a concise, "
                                   "high-level abstraction. Focus on key concepts, "
                                   "relationships, and patterns. Be brief."
                    }, {
                        "role": "user",
                        "content": text[:2000]
                    }],
                    options={"temperature": 0.3}
                )
                return response["message"]["content"]
            except Exception:
                pass

        # Fallback: simple extractive summary
        sentences = text.split(". ")
        if len(sentences) <= 3:
            return text[:500]
        # Take first, middle, and last sentences
        summary = ". ".join([
            sentences[0],
            sentences[len(sentences) // 2],
            sentences[-1]
        ])
        return summary[:500]

    def _apply_decay(self, memories: list[MemoryRecord]):
        """Apply time-based decay to all memories."""
        now = datetime.utcnow()
        for mem in memories:
            hours_since_access = (now - mem.last_accessed).total_seconds() / 3600
            decay = max(
                config.decay.min_importance,
                mem.decay_factor * (1.0 - config.decay.lambda_factor * hours_since_access / 100)
            )
            if abs(decay - mem.decay_factor) > 0.01:  # Only update if changed meaningfully
                mem.decay_factor = decay
                self.relational.update_decay(mem.id, decay)

    def _prune_low_value(self) -> int:
        """Archive memories below the prune threshold."""
        candidates = self.relational.get_memories_below_importance(
            config.decay.prune_threshold
        )
        pruned = 0
        for mem in candidates:
            if mem.decay_factor < config.decay.prune_threshold:
                self.relational.archive_memory(mem.id)
                pruned += 1
        return pruned

    def _generate_insights(self, memories: list[MemoryRecord],
                            patterns: list[str]) -> list[str]:
        """Use LLM to generate higher-level insights from patterns."""
        if not HAS_OLLAMA or not patterns:
            return patterns  # Return patterns as-is

        try:
            prompt = (
                "Based on these observed patterns in a user's memory system, "
                "generate 2-3 high-level insights:\n\n"
                + "\n".join(patterns)
            )
            response = ollama_client.chat(
                model=config.llm.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.5}
            )
            insights = response["message"]["content"].strip().split("\n")
            return [i.strip() for i in insights if i.strip()]
        except Exception:
            return patterns
