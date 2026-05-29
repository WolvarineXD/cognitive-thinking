"""
Graph store using NetworkX for memory relationship management.
This is the knowledge graph that makes the system elite — memories are not flat,
they form a web of relationships.
"""
import json
from pathlib import Path
from typing import Optional
import networkx as nx
from models.schemas import MemoryRecord, GraphEdge, MemoryType


class GraphStore:
    """NetworkX-based knowledge graph for memory relationships."""

    def __init__(self, persist_dir: Path):
        self.persist_dir = persist_dir
        self.graph_file = persist_dir / "memory_graph.json"
        self.graph = nx.DiGraph()
        self._load()

    def _load(self):
        """Load graph from disk if it exists."""
        if self.graph_file.exists():
            try:
                with open(self.graph_file, "r") as f:
                    data = json.load(f)
                self.graph = nx.node_link_graph(data)
            except Exception:
                self.graph = nx.DiGraph()

    def _save(self):
        """Persist graph to disk."""
        data = nx.node_link_data(self.graph)
        with open(self.graph_file, "w") as f:
            json.dump(data, f, default=str)

    # ── Node Operations ───────────────────────────────────────────────

    def add_memory_node(self, memory: MemoryRecord):
        """Add a memory as a node in the graph."""
        self.graph.add_node(
            memory.id,
            label=memory.content[:100],
            memory_type=memory.memory_type.value,
            importance=memory.importance,
            created_at=memory.created_at.isoformat(),
            entities=memory.entities,
            tags=memory.tags,
        )
        self._save()

    def remove_node(self, memory_id: str):
        """Remove a memory node and all its edges."""
        if self.graph.has_node(memory_id):
            self.graph.remove_node(memory_id)
            self._save()

    # ── Edge Operations ───────────────────────────────────────────────

    def add_edge(self, edge: GraphEdge):
        """Add a relationship between two memories."""
        if not self.graph.has_node(edge.source_id) or not self.graph.has_node(edge.target_id):
            return
        self.graph.add_edge(
            edge.source_id, edge.target_id,
            relation_type=edge.relation_type,
            weight=edge.weight,
            created_at=edge.created_at.isoformat()
        )
        self._save()

    def add_relationship(self, source_id: str, target_id: str,
                         relation_type: str = "related_to", weight: float = 1.0):
        """Convenience method to add a relationship."""
        edge = GraphEdge(
            source_id=source_id, target_id=target_id,
            relation_type=relation_type, weight=weight
        )
        self.add_edge(edge)

    def remove_edge(self, source_id: str, target_id: str):
        if self.graph.has_edge(source_id, target_id):
            self.graph.remove_edge(source_id, target_id)
            self._save()

    # ── Traversal & Retrieval ─────────────────────────────────────────

    def get_neighbors(self, memory_id: str, depth: int = 1) -> list[str]:
        """Get all connected memory IDs up to N hops away."""
        if not self.graph.has_node(memory_id):
            return []
        visited = set()
        frontier = {memory_id}
        for _ in range(depth):
            next_frontier = set()
            for node in frontier:
                for neighbor in list(self.graph.successors(node)) + list(self.graph.predecessors(node)):
                    if neighbor not in visited and neighbor != memory_id:
                        next_frontier.add(neighbor)
            visited.update(next_frontier)
            frontier = next_frontier
        return list(visited)

    def get_connectivity_score(self, memory_id: str) -> float:
        """Get normalized degree centrality for a memory node (0–1)."""
        if not self.graph.has_node(memory_id):
            return 0.0
        if self.graph.number_of_nodes() <= 1:
            return 0.0
        centrality = nx.degree_centrality(self.graph)
        return centrality.get(memory_id, 0.0)

    def get_all_centralities(self) -> dict[str, float]:
        """Compute centrality for all nodes."""
        if self.graph.number_of_nodes() == 0:
            return {}
        return nx.degree_centrality(self.graph)

    def find_shortest_path(self, source_id: str, target_id: str) -> Optional[list[str]]:
        """Find shortest path between two memories."""
        try:
            return nx.shortest_path(self.graph.to_undirected(), source_id, target_id)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def get_clusters(self) -> list[set[str]]:
        """Find clusters of strongly connected memories."""
        undirected = self.graph.to_undirected()
        return [comp for comp in nx.connected_components(undirected) if len(comp) > 1]

    def get_related_by_entities(self, entities: list[str]) -> list[str]:
        """Find memory nodes sharing entities with the given list."""
        results = []
        for node_id, data in self.graph.nodes(data=True):
            node_entities = data.get("entities", [])
            if any(e.lower() in [ne.lower() for ne in node_entities] for e in entities):
                results.append(node_id)
        return results

    def auto_link_by_entities(self, memory: MemoryRecord):
        """Automatically create edges to other memories sharing entities."""
        if not memory.entities:
            return
        related = self.get_related_by_entities(memory.entities)
        for related_id in related:
            if related_id != memory.id:
                shared = [e for e in memory.entities
                          if e.lower() in [ne.lower() for ne in
                                           self.graph.nodes[related_id].get("entities", [])]]
                weight = min(1.0, len(shared) * 0.3)
                self.add_relationship(memory.id, related_id, "shares_entities", weight)
                self.add_relationship(related_id, memory.id, "shares_entities", weight)

    def get_stats(self) -> dict:
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "clusters": len(self.get_clusters()),
            "density": nx.density(self.graph) if self.graph.number_of_nodes() > 0 else 0
        }
