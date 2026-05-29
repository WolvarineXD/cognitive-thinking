"""
SQLite-based relational store for memory metadata.
Handles structured queries, metadata lookups, and ACID guarantees.

This is NOT where embeddings live — that's the vector store.
This stores the structured metadata about memories.
"""
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from models.schemas import MemoryRecord, MemoryType, Episode, Procedure


class RelationalStore:
    """SQLite wrapper for structured memory metadata."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn: Optional[sqlite3.Connection] = None
        self._initialize()

    def _initialize(self):
        """Create tables if they don't exist."""
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")  # Better concurrency
        self.conn.execute("PRAGMA foreign_keys=ON")

        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                memory_type TEXT NOT NULL,
                content TEXT NOT NULL,
                summary TEXT,
                created_at TEXT NOT NULL,
                last_accessed TEXT NOT NULL,
                access_count INTEGER DEFAULT 0,
                source TEXT,
                importance REAL DEFAULT 0.5,
                user_priority REAL DEFAULT 0.5,
                emotional_valence TEXT DEFAULT 'neutral',
                decay_factor REAL DEFAULT 1.0,
                entities TEXT DEFAULT '[]',
                tags TEXT DEFAULT '[]',
                intent TEXT,
                related_memory_ids TEXT DEFAULT '[]',
                parent_id TEXT,
                episode_id TEXT,
                is_archived INTEGER DEFAULT 0,
                is_contradiction INTEGER DEFAULT 0,
                contradicts_id TEXT
            );

            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY,
                title TEXT,
                memory_ids TEXT DEFAULT '[]',
                created_at TEXT NOT NULL,
                ended_at TEXT,
                summary TEXT,
                tags TEXT DEFAULT '[]'
            );

            CREATE TABLE IF NOT EXISTS procedures (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                steps TEXT DEFAULT '[]',
                success_rate REAL DEFAULT 0.5,
                use_count INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                last_used TEXT,
                tags TEXT DEFAULT '[]'
            );

            CREATE TABLE IF NOT EXISTS reflection_logs (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                memories_reviewed INTEGER DEFAULT 0,
                patterns_extracted TEXT DEFAULT '[]',
                summaries_created INTEGER DEFAULT 0,
                duplicates_merged INTEGER DEFAULT 0,
                memories_pruned INTEGER DEFAULT 0,
                contradictions_found INTEGER DEFAULT 0,
                insights TEXT DEFAULT '[]'
            );

            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT DEFAULT 'New Chat',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );

            -- Indexes for fast lookups
            CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
            CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance);
            CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
            CREATE INDEX IF NOT EXISTS idx_memories_episode ON memories(episode_id);
            CREATE INDEX IF NOT EXISTS idx_memories_archived ON memories(is_archived);
            CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id);
            CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at);
        """)
        self.conn.commit()

    # ── Memory CRUD ────────────────────────────────────────────────────

    def store_memory(self, memory: MemoryRecord) -> str:
        """Store a memory record. Returns the memory ID."""
        self.conn.execute("""
            INSERT OR REPLACE INTO memories (
                id, memory_type, content, summary, created_at, last_accessed,
                access_count, source, importance, user_priority,
                emotional_valence, decay_factor, entities, tags, intent,
                related_memory_ids, parent_id, episode_id,
                is_archived, is_contradiction, contradicts_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            memory.id, memory.memory_type.value, memory.content, memory.summary,
            memory.created_at.isoformat(), memory.last_accessed.isoformat(),
            memory.access_count, memory.source, memory.importance,
            memory.user_priority, memory.emotional_valence.value,
            memory.decay_factor, json.dumps(memory.entities),
            json.dumps(memory.tags), memory.intent,
            json.dumps(memory.related_memory_ids), memory.parent_id,
            memory.episode_id, int(memory.is_archived),
            int(memory.is_contradiction), memory.contradicts_id
        ))
        self.conn.commit()
        return memory.id

    def get_memory(self, memory_id: str) -> Optional[MemoryRecord]:
        """Retrieve a single memory by ID."""
        row = self.conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_memory(row)

    def get_memories_by_type(
        self, memory_type: MemoryType,
        include_archived: bool = False,
        limit: int = 100
    ) -> list[MemoryRecord]:
        """Get all memories of a specific type."""
        query = "SELECT * FROM memories WHERE memory_type = ?"
        params: list = [memory_type.value]
        if not include_archived:
            query += " AND is_archived = 0"
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(query, params).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def get_all_memories(
        self, include_archived: bool = False,
        limit: int = 500
    ) -> list[MemoryRecord]:
        """Get all memories across all types."""
        query = "SELECT * FROM memories"
        params: list = []
        if not include_archived:
            query += " WHERE is_archived = 0"
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(query, params).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def update_memory(self, memory: MemoryRecord):
        """Update an existing memory record."""
        self.store_memory(memory)  # INSERT OR REPLACE handles this

    def delete_memory(self, memory_id: str):
        """Permanently delete a memory."""
        self.conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self.conn.commit()

    def archive_memory(self, memory_id: str):
        """Soft-delete: mark as archived."""
        self.conn.execute(
            "UPDATE memories SET is_archived = 1 WHERE id = ?", (memory_id,)
        )
        self.conn.commit()

    def touch_memory(self, memory_id: str):
        """Update last_accessed and increment access_count."""
        now = datetime.utcnow().isoformat()
        self.conn.execute("""
            UPDATE memories
            SET last_accessed = ?, access_count = access_count + 1
            WHERE id = ?
        """, (now, memory_id))
        self.conn.commit()

    def update_importance(self, memory_id: str, importance: float):
        """Update the importance score of a memory."""
        self.conn.execute(
            "UPDATE memories SET importance = ? WHERE id = ?",
            (max(0.0, min(1.0, importance)), memory_id)
        )
        self.conn.commit()

    def update_decay(self, memory_id: str, decay_factor: float):
        """Update the decay factor of a memory."""
        self.conn.execute(
            "UPDATE memories SET decay_factor = ? WHERE id = ?",
            (decay_factor, memory_id)
        )
        self.conn.commit()

    def get_memories_below_importance(self, threshold: float) -> list[MemoryRecord]:
        """Get memories below a certain importance threshold (prune candidates)."""
        rows = self.conn.execute(
            "SELECT * FROM memories WHERE importance < ? AND is_archived = 0",
            (threshold,)
        ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def search_by_tags(self, tags: list[str], limit: int = 50) -> list[MemoryRecord]:
        """Find memories containing any of the given tags."""
        results = []
        for row in self.conn.execute(
            "SELECT * FROM memories WHERE is_archived = 0"
        ).fetchall():
            memory_tags = json.loads(row["tags"])
            if any(tag in memory_tags for tag in tags):
                results.append(self._row_to_memory(row))
            if len(results) >= limit:
                break
        return results

    def search_by_entities(self, entities: list[str], limit: int = 50) -> list[MemoryRecord]:
        """Find memories containing any of the given entities."""
        results = []
        for row in self.conn.execute(
            "SELECT * FROM memories WHERE is_archived = 0"
        ).fetchall():
            memory_entities = json.loads(row["entities"])
            if any(e.lower() in [me.lower() for me in memory_entities] for e in entities):
                results.append(self._row_to_memory(row))
            if len(results) >= limit:
                break
        return results

    def get_memory_count(self) -> dict[str, int]:
        """Get count of memories by type."""
        rows = self.conn.execute(
            "SELECT memory_type, COUNT(*) as cnt FROM memories "
            "WHERE is_archived = 0 GROUP BY memory_type"
        ).fetchall()
        return {row["memory_type"]: row["cnt"] for row in rows}

    # ── Episode CRUD ──────────────────────────────────────────────────

    def store_episode(self, episode: Episode) -> str:
        """Store an episode."""
        self.conn.execute("""
            INSERT OR REPLACE INTO episodes (
                id, title, memory_ids, created_at, ended_at, summary, tags
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            episode.id, episode.title, json.dumps(episode.memory_ids),
            episode.created_at.isoformat(),
            episode.ended_at.isoformat() if episode.ended_at else None,
            episode.summary, json.dumps(episode.tags)
        ))
        self.conn.commit()
        return episode.id

    def get_episode(self, episode_id: str) -> Optional[Episode]:
        """Get an episode by ID."""
        row = self.conn.execute(
            "SELECT * FROM episodes WHERE id = ?", (episode_id,)
        ).fetchone()
        if row is None:
            return None
        return Episode(
            id=row["id"],
            title=row["title"],
            memory_ids=json.loads(row["memory_ids"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            ended_at=datetime.fromisoformat(row["ended_at"]) if row["ended_at"] else None,
            summary=row["summary"],
            tags=json.loads(row["tags"])
        )

    # ── Procedure CRUD ────────────────────────────────────────────────

    def store_procedure(self, procedure: Procedure) -> str:
        """Store a procedure."""
        self.conn.execute("""
            INSERT OR REPLACE INTO procedures (
                id, name, description, steps, success_rate,
                use_count, created_at, last_used, tags
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            procedure.id, procedure.name, procedure.description,
            json.dumps([s.model_dump() for s in procedure.steps]),
            procedure.success_rate, procedure.use_count,
            procedure.created_at.isoformat(),
            procedure.last_used.isoformat() if procedure.last_used else None,
            json.dumps(procedure.tags)
        ))
        self.conn.commit()
        return procedure.id

    # ── Helpers ────────────────────────────────────────────────────────

    def _row_to_memory(self, row: sqlite3.Row) -> MemoryRecord:
        """Convert a database row to a MemoryRecord."""
        return MemoryRecord(
            id=row["id"],
            memory_type=MemoryType(row["memory_type"]),
            content=row["content"],
            summary=row["summary"],
            created_at=datetime.fromisoformat(row["created_at"]),
            last_accessed=datetime.fromisoformat(row["last_accessed"]),
            access_count=row["access_count"],
            source=row["source"],
            importance=row["importance"],
            user_priority=row["user_priority"],
            emotional_valence=row["emotional_valence"],
            decay_factor=row["decay_factor"],
            entities=json.loads(row["entities"]),
            tags=json.loads(row["tags"]),
            intent=row["intent"],
            related_memory_ids=json.loads(row["related_memory_ids"]),
            parent_id=row["parent_id"],
            episode_id=row["episode_id"],
            is_archived=bool(row["is_archived"]),
            is_contradiction=bool(row["is_contradiction"]),
            contradicts_id=row["contradicts_id"]
        )

    # ── Conversation & Message Persistence ────────────────────────────

    def create_conversation(self, conv_id: str, title: str = "New Chat") -> str:
        """Create a new conversation. Returns conversation ID."""
        now = datetime.utcnow().isoformat()
        self.conn.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (conv_id, title, now, now)
        )
        self.conn.commit()
        return conv_id

    def list_conversations(self, limit: int = 50) -> list[dict]:
        """List conversations ordered by most recently updated."""
        rows = self.conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_conversation(self, conv_id: str) -> dict | None:
        """Get a single conversation by ID."""
        row = self.conn.execute(
            "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
            (conv_id,)
        ).fetchone()
        return dict(row) if row else None

    def update_conversation_title(self, conv_id: str, title: str):
        """Update conversation title."""
        self.conn.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            (title, datetime.utcnow().isoformat(), conv_id)
        )
        self.conn.commit()

    def touch_conversation(self, conv_id: str):
        """Update the updated_at timestamp for a conversation."""
        self.conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (datetime.utcnow().isoformat(), conv_id)
        )
        self.conn.commit()

    def delete_conversation(self, conv_id: str):
        """Delete a conversation and all its messages."""
        self.conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
        self.conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
        self.conn.commit()

    def save_message(self, msg_id: str, conv_id: str, role: str,
                     content: str, metadata: dict = None) -> str:
        """Save a chat message. Returns message ID."""
        self.conn.execute(
            "INSERT INTO messages (id, conversation_id, role, content, metadata, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (msg_id, conv_id, role, content, json.dumps(metadata or {}), datetime.utcnow().isoformat())
        )
        self.touch_conversation(conv_id)
        self.conn.commit()
        return msg_id

    def get_messages(self, conv_id: str) -> list[dict]:
        """Get all messages for a conversation, ordered chronologically."""
        rows = self.conn.execute(
            "SELECT id, role, content, metadata, created_at FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
            (conv_id,)
        ).fetchall()
        result = []
        for r in rows:
            msg = dict(r)
            msg["metadata"] = json.loads(msg["metadata"]) if msg["metadata"] else {}
            result.append(msg)
        return result

    def get_conversation_message_count(self, conv_id: str) -> int:
        """Get message count for a conversation."""
        row = self.conn.execute(
            "SELECT COUNT(*) FROM messages WHERE conversation_id = ?", (conv_id,)
        ).fetchone()
        return row[0] if row else 0

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
