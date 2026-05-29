"""
Context Compressor — reduces memory bloat through summarization and merging.
"""
from models.schemas import MemoryRecord
from core.embedding_engine import EmbeddingEngine
from config import config

try:
    import ollama as ollama_client
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False


class ContextCompressor:
    """Compresses and summarizes memories to prevent unbounded growth."""

    def __init__(self, embedder: EmbeddingEngine):
        self.embedder = embedder

    def summarize_memories(self, memories: list[MemoryRecord]) -> str:
        """Summarize a group of related memories into one coherent text."""
        if not memories:
            return ""
        combined = "\n---\n".join([m.content for m in memories[:20]])

        if HAS_OLLAMA:
            try:
                response = ollama_client.chat(
                    model=config.llm.model,
                    messages=[{
                        "role": "system",
                        "content": (
                            "You are a memory compression engine. Summarize the following "
                            "memories into a single coherent paragraph. Preserve key facts, "
                            "entities, relationships, and insights. Remove redundancy."
                        )
                    }, {
                        "role": "user",
                        "content": combined[:3000]
                    }],
                    options={"temperature": 0.2}
                )
                return response["message"]["content"]
            except Exception:
                pass

        # Fallback: extractive compression
        return self._extractive_summary(memories)

    def _extractive_summary(self, memories: list[MemoryRecord]) -> str:
        """Fallback extractive summarization."""
        # Take first sentence of each memory
        sentences = []
        for mem in memories[:10]:
            first_sentence = mem.content.split(". ")[0]
            if first_sentence not in sentences:
                sentences.append(first_sentence)
        return ". ".join(sentences)

    def find_duplicates(self, memories: list[MemoryRecord],
                        threshold: float = 0.92) -> list[tuple[str, str]]:
        """Find pairs of near-duplicate memories."""
        duplicates = []
        for i, mem_a in enumerate(memories):
            for mem_b in memories[i + 1:]:
                if mem_a.embedding and mem_b.embedding:
                    sim = self.embedder.similarity(mem_a.embedding, mem_b.embedding)
                    if sim > threshold:
                        duplicates.append((mem_a.id, mem_b.id))
        return duplicates
