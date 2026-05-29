"""
Memory Decay — implements forgetting curves and memory reinforcement.

Inspired by Ebbinghaus forgetting curve:
memories naturally fade unless reinforced through access.
"""
import math
from datetime import datetime
from models.schemas import MemoryRecord
from config import config


class DecayEngine:
    """Manages memory decay and reinforcement."""

    def __init__(self):
        self.cfg = config.decay

    def compute_decay(self, memory: MemoryRecord) -> float:
        """
        Compute current decay factor for a memory.
        Uses exponential decay: factor = exp(-λ × hours_since_last_access)
        """
        hours_since = (datetime.utcnow() - memory.last_accessed).total_seconds() / 3600
        raw_decay = math.exp(-self.cfg.lambda_factor * hours_since)

        # Boost based on access count (frequently accessed = more resistant to decay)
        access_resistance = min(1.0, memory.access_count * 0.02)
        adjusted = raw_decay + access_resistance * (1.0 - raw_decay)

        return max(self.cfg.min_importance, min(1.0, adjusted))

    def reinforce(self, memory: MemoryRecord) -> MemoryRecord:
        """Reinforce a memory (called when it's accessed/retrieved)."""
        memory.decay_factor = min(1.0, memory.decay_factor + self.cfg.reinforcement_boost)
        memory.access_count += 1
        memory.last_accessed = datetime.utcnow()
        return memory

    def should_prune(self, memory: MemoryRecord) -> bool:
        """Check if a memory should be pruned."""
        effective_importance = memory.importance * self.compute_decay(memory)
        return effective_importance < self.cfg.prune_threshold

    def batch_compute_decay(self, memories: list[MemoryRecord]) -> list[tuple[str, float]]:
        """Compute decay for a batch of memories. Returns list of (id, decay_factor)."""
        return [(m.id, self.compute_decay(m)) for m in memories]
