"""
Context Packer — optimizes what goes into the LLM context window.

Problem: LLMs have limited context. You can't stuff infinite memories.
Solution: Intelligent packing that maximizes information density.
"""
from models.schemas import ScoredMemory
from config import config


class ContextPacker:
    """Packs retrieved memories into an optimal context string within token budget."""

    def __init__(self):
        self.budget = config.context_budget

    def pack(self, scored_memories: list[ScoredMemory],
             query: str, system_prompt: str = "") -> str:
        """
        Pack memories into a context string that fits the token budget.

        Strategy:
        1. Reserve tokens for query and system prompt
        2. Add memories in score order until budget exhausted
        3. Use summaries for lower-scored memories to save tokens
        """
        available_tokens = (
            self.budget.max_tokens -
            self.budget.reserved_for_query -
            self.budget.reserved_for_system
        )

        context_parts = []
        tokens_used = 0

        for i, sm in enumerate(scored_memories):
            # Use full content for top memories, summary for lower ones
            if i < 3 and sm.memory.content:
                text = sm.memory.content
            elif sm.memory.summary:
                text = sm.memory.summary
            else:
                text = sm.memory.content

            # Estimate tokens (rough: 1 token ≈ 4 chars)
            estimated_tokens = len(text) // 4

            if tokens_used + estimated_tokens > available_tokens:
                # Try truncating
                remaining_chars = (available_tokens - tokens_used) * 4
                if remaining_chars > 100:
                    text = text[:remaining_chars] + "..."
                    estimated_tokens = remaining_chars // 4
                else:
                    break  # No more room

            # Format the memory entry
            entry = self._format_memory_entry(sm, text, i + 1)
            context_parts.append(entry)
            tokens_used += estimated_tokens

        if not context_parts:
            return ""

        header = "## Retrieved Memories\n\n"
        return header + "\n---\n".join(context_parts)

    def _format_memory_entry(self, sm: ScoredMemory, text: str, rank: int) -> str:
        """Format a single memory for context injection."""
        mem = sm.memory
        lines = [
            f"**[Memory #{rank}]** (type: {mem.memory_type.value}, "
            f"score: {sm.total_score:.3f}, importance: {mem.importance:.2f})",
        ]
        if mem.tags:
            lines.append(f"Tags: {', '.join(mem.tags[:5])}")
        if mem.entities:
            lines.append(f"Entities: {', '.join(mem.entities[:5])}")
        lines.append(f"Content: {text}")
        return "\n".join(lines)

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars ≈ 1 token)."""
        return len(text) // 4
