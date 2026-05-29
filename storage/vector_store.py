"""
ChromaDB-based vector store for semantic memory retrieval.
Handles embedding storage and similarity search.
"""
import chromadb
from chromadb.config import Settings
from pathlib import Path
from typing import Optional
from models.schemas import MemoryRecord, MemoryType


class VectorStore:
    """ChromaDB wrapper for embedding storage and semantic search."""

    def __init__(self, persist_dir: Path):
        self.persist_dir = persist_dir
        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        self.collections: dict[str, chromadb.Collection] = {}
        self._initialize_collections()

    def _initialize_collections(self):
        for mem_type in MemoryType:
            name = f"memory_{mem_type.value}"
            self.collections[mem_type.value] = self.client.get_or_create_collection(
                name=name, metadata={"hnsw:space": "cosine"}
            )
        self.collections["all"] = self.client.get_or_create_collection(
            name="memory_all", metadata={"hnsw:space": "cosine"}
        )

    def store_embedding(self, memory: MemoryRecord):
        """Store embedding in both type-specific and master collection."""
        if memory.embedding is None:
            raise ValueError(f"Memory {memory.id} has no embedding")
        metadata = {
            "memory_type": memory.memory_type.value,
            "importance": memory.importance,
            "created_at": memory.created_at.isoformat(),
            "source": memory.source or "",
            "intent": memory.intent or "",
            "is_archived": str(memory.is_archived),
            "tags": ",".join(memory.tags),
            "entities": ",".join(memory.entities),
        }
        self.collections[memory.memory_type.value].upsert(
            ids=[memory.id], embeddings=[memory.embedding],
            metadatas=[metadata], documents=[memory.content]
        )
        self.collections["all"].upsert(
            ids=[memory.id], embeddings=[memory.embedding],
            metadatas=[metadata], documents=[memory.content]
        )

    def search_similar(self, query_embedding: list[float], top_k: int = 10,
                       memory_type: Optional[MemoryType] = None,
                       min_importance: float = 0.0,
                       include_archived: bool = False) -> list[dict]:
        """Find most similar memories. Returns list of dicts with id, content, distance, similarity, metadata."""
        collection = self.collections[memory_type.value] if memory_type else self.collections["all"]
        where_clauses = []
        if min_importance > 0:
            where_clauses.append({"importance": {"$gte": min_importance}})
        if not include_archived:
            where_clauses.append({"is_archived": "False"})
        where = None
        if len(where_clauses) == 1:
            where = where_clauses[0]
        elif len(where_clauses) > 1:
            where = {"$and": where_clauses}
        try:
            count = collection.count()
            if count == 0:
                return []
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, count),
                where=where if where else None,
                include=["documents", "metadatas", "distances"]
            )
        except Exception:
            return []
        if not results["ids"] or not results["ids"][0]:
            return []
        output = []
        for i, memory_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][i] if results["distances"] else 0.0
            output.append({
                "id": memory_id,
                "content": results["documents"][0][i] if results["documents"] else "",
                "distance": distance,
                "similarity": 1.0 - distance,
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {}
            })
        return output

    def delete_embedding(self, memory_id: str, memory_type: Optional[MemoryType] = None):
        """Remove an embedding from all collections."""
        try:
            self.collections["all"].delete(ids=[memory_id])
        except Exception:
            pass
        if memory_type:
            try:
                self.collections[memory_type.value].delete(ids=[memory_id])
            except Exception:
                pass
        else:
            for col in self.collections.values():
                try:
                    col.delete(ids=[memory_id])
                except Exception:
                    pass

    def get_collection_stats(self) -> dict[str, int]:
        return {name: col.count() for name, col in self.collections.items()}
