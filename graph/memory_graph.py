"""
Memory Graph — higher-level graph operations over the knowledge graph.

Provides:
- Subgraph extraction for context
- Path-based reasoning
- Community detection
- Graph-based recommendations
"""
from typing import Optional
from storage.graph_store import GraphStore
from storage.relational_store import RelationalStore
from models.schemas import MemoryRecord


class MemoryGraph:
    """High-level graph operations for memory reasoning."""

    def __init__(self, graph: GraphStore, relational: RelationalStore):
        self.graph = graph
        self.relational = relational

    def get_context_subgraph(self, memory_id: str, depth: int = 2) -> dict:
        """
        Extract a subgraph around a memory for context building.
        Returns nodes and edges within N hops.
        """
        neighbor_ids = self.graph.get_neighbors(memory_id, depth=depth)
        all_ids = [memory_id] + neighbor_ids

        nodes = []
        for mid in all_ids:
            mem = self.relational.get_memory(mid)
            if mem:
                nodes.append({
                    "id": mid,
                    "content": mem.content[:200],
                    "type": mem.memory_type.value,
                    "importance": mem.importance,
                })

        edges = []
        for mid in all_ids:
            if self.graph.graph.has_node(mid):
                for successor in self.graph.graph.successors(mid):
                    if successor in all_ids:
                        edge_data = self.graph.graph[mid][successor]
                        edges.append({
                            "source": mid,
                            "target": successor,
                            "relation": edge_data.get("relation_type", "related"),
                            "weight": edge_data.get("weight", 1.0),
                        })

        return {"nodes": nodes, "edges": edges}

    def find_reasoning_path(self, from_id: str, to_id: str) -> Optional[list[dict]]:
        """
        Find a reasoning path between two memories.
        Returns the memories along the path with their relationships.
        """
        path_ids = self.graph.find_shortest_path(from_id, to_id)
        if not path_ids:
            return None

        path = []
        for mid in path_ids:
            mem = self.relational.get_memory(mid)
            if mem:
                path.append({
                    "id": mid,
                    "content": mem.content[:200],
                    "type": mem.memory_type.value,
                })
        return path

    def get_topic_clusters(self) -> list[dict]:
        """Get clusters of related memories with their topics."""
        clusters = self.graph.get_clusters()
        result = []

        for cluster_ids in clusters:
            memories = []
            all_entities = []
            all_tags = []

            for mid in cluster_ids:
                mem = self.relational.get_memory(mid)
                if mem:
                    memories.append(mem)
                    all_entities.extend(mem.entities)
                    all_tags.extend(mem.tags)

            # Find dominant topics
            entity_counts = {}
            for e in all_entities:
                entity_counts[e.lower()] = entity_counts.get(e.lower(), 0) + 1

            top_topics = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:5]

            result.append({
                "size": len(cluster_ids),
                "memory_ids": list(cluster_ids),
                "topics": [t[0] for t in top_topics],
                "tags": list(set(all_tags))[:10],
            })

        return sorted(result, key=lambda x: x["size"], reverse=True)
