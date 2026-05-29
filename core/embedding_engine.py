"""
Embedding Engine — generates vector representations of text.

Uses SentenceTransformers. Model is configurable via config.py.
This is Step 3 in the pipeline: bad embeddings destroy the entire system.
"""
from sentence_transformers import SentenceTransformer
from config import config
import numpy as np


class EmbeddingEngine:
    """Generates high-quality text embeddings."""

    def __init__(self):
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the model (it's heavy)."""
        if self._model is None:
            self._model = SentenceTransformer(config.embedding.model_name)
        return self._model

    def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text string."""
        embedding = self.model.encode(
            text,
            normalize_embeddings=config.embedding.normalize,
            show_progress_bar=False
        )
        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=config.embedding.normalize,
            batch_size=config.embedding.batch_size,
            show_progress_bar=False
        )
        return embeddings.tolist()

    def similarity(self, embedding_a: list[float], embedding_b: list[float]) -> float:
        """Compute cosine similarity between two embeddings."""
        a = np.array(embedding_a)
        b = np.array(embedding_b)
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def get_dimension(self) -> int:
        """Get the embedding dimension of the current model."""
        return self.model.get_sentence_embedding_dimension()
