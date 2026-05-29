"""
Central configuration for the Cognitive Memory System.
All tunable parameters live here.
"""
from pathlib import Path
from pydantic import BaseModel, ConfigDict


# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma"
GRAPH_DIR = DATA_DIR / "graph"
SQLITE_PATH = DATA_DIR / "memory.db"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)
GRAPH_DIR.mkdir(exist_ok=True)


# ── Embedding Configuration ───────────────────────────────────────────
class EmbeddingConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_name: str = "all-MiniLM-L6-v2"
    dimension: int = 384  # MiniLM output dimension
    batch_size: int = 32
    normalize: bool = True


# ── Retrieval Weights ─────────────────────────────────────────────────
class RetrievalWeights(BaseModel):
    """Hybrid retrieval scoring weights. Must sum to 1.0."""
    semantic: float = 0.40
    recency: float = 0.20
    importance: float = 0.20
    graph_connectivity: float = 0.10
    user_priority: float = 0.10


# ── Memory Decay Configuration ────────────────────────────────────────
class DecayConfig(BaseModel):
    """Controls how memories lose importance over time."""
    lambda_factor: float = 0.01       # Decay rate (higher = faster decay)
    min_importance: float = 0.05      # Floor — never fully forget
    reinforcement_boost: float = 0.15  # Boost when memory is accessed
    prune_threshold: float = 0.02     # Below this → candidate for pruning


# ── Reflection Configuration ──────────────────────────────────────────
class ReflectionConfig(BaseModel):
    interval_minutes: int = 60          # How often reflection runs
    min_memories_for_summary: int = 5   # Min memories before summarization
    max_summary_tokens: int = 500       # Max tokens per summary
    similarity_merge_threshold: float = 0.92  # Above this → merge duplicates


# ── Working Memory ────────────────────────────────────────────────────
class WorkingMemoryConfig(BaseModel):
    max_items: int = 50          # Max items in working memory
    ttl_seconds: int = 3600      # Time-to-live: 1 hour default
    overflow_strategy: str = "evict_oldest"  # or "evict_least_important"


# ── Context Budget ────────────────────────────────────────────────────
class ContextBudgetConfig(BaseModel):
    max_tokens: int = 4096       # Max tokens for context window
    reserved_for_query: int = 512  # Reserved for the current query
    reserved_for_system: int = 256  # Reserved for system prompt


# ── LLM Configuration ────────────────────────────────────────────────
class LLMConfig(BaseModel):
    provider: str = "ollama"
    model: str = "llama3:8b"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.3
    max_tokens: int = 2048


# ── Master Config ─────────────────────────────────────────────────────
class SystemConfig(BaseModel):
    embedding: EmbeddingConfig = EmbeddingConfig()
    retrieval_weights: RetrievalWeights = RetrievalWeights()
    decay: DecayConfig = DecayConfig()
    reflection: ReflectionConfig = ReflectionConfig()
    working_memory: WorkingMemoryConfig = WorkingMemoryConfig()
    context_budget: ContextBudgetConfig = ContextBudgetConfig()
    llm: LLMConfig = LLMConfig()


# Singleton config instance
config = SystemConfig()
