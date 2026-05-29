"""
Cognitive AI Memory System — FastAPI Application (v3.0)
=======================================================

ChatGPT-style persistent chat with cognitive memory.
- Multiple independent conversations with full history
- Messages persist across page refreshes
- Memory retrieval with visual proof
- Streaming LLM responses via SSE
"""
import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from config import config
from memory.manager import MemoryManager
from models.schemas import MemoryType, RetrievalQuery
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker
from retrieval.context_packer import ContextPacker
from reflection.reflection_engine import ReflectionEngine
from core.reasoning_engine import ReasoningEngine

try:
    import ollama as ollama_client
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False


# ── Global State ──────────────────────────────────────────────────────
manager: MemoryManager = None
retriever: HybridRetriever = None
reranker: Reranker = None
packer: ContextPacker = None
reflection: ReflectionEngine = None
reasoning: ReasoningEngine = None
_reflection_task: asyncio.Task = None


async def periodic_reflection():
    """Background task that runs reflection at configured intervals."""
    while True:
        await asyncio.sleep(config.reflection.interval_minutes * 60)
        try:
            report = reflection.run_reflection_cycle()
            print(f"[Reflection] Reviewed {report.memories_reviewed}, "
                  f"merged {report.duplicates_merged}, pruned {report.memories_pruned}")
        except Exception as e:
            print(f"[Reflection Error] {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all systems on startup, cleanup on shutdown."""
    global manager, retriever, reranker, packer, reflection, _reflection_task, reasoning

    print("[Boot] Initializing Cognitive Memory System...")
    manager = MemoryManager()
    retriever = HybridRetriever(
        manager.relational, manager.vector, manager.graph, manager.embedder
    )
    reranker = Reranker(manager.embedder)
    packer = ContextPacker()
    reflection = ReflectionEngine(
        manager.relational, manager.vector, manager.graph, manager.embedder
    )
    reasoning = ReasoningEngine(
        retriever, reranker, packer, manager.graph, manager.embedder
    )
    _reflection_task = asyncio.create_task(periodic_reflection())

    stats = manager.get_stats()
    convs = manager.relational.list_conversations()
    print(f"[Boot] Ready. {stats['memory_counts']} memories, {len(convs)} conversations")

    yield

    if _reflection_task:
        _reflection_task.cancel()
    manager.close()
    print("[Shutdown] System closed.")


# ── FastAPI App ───────────────────────────────────────────────────────
app = FastAPI(title="Cognitive AI", version="3.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Request Models ───────────────────────────────────────────────────

class ChatMessage(BaseModel):
    message: str
    top_k: int = 5


class ConvCreate(BaseModel):
    title: str = "New Chat"


class ConvRename(BaseModel):
    title: str


# ── Conversation CRUD ────────────────────────────────────────────────

@app.post("/api/conversations")
async def create_conversation(req: ConvCreate):
    """Create a new conversation."""
    conv_id = str(uuid.uuid4())
    manager.relational.create_conversation(conv_id, req.title)
    return {"id": conv_id, "title": req.title}


@app.get("/api/conversations")
async def list_conversations():
    """List all conversations, most recent first."""
    convs = manager.relational.list_conversations()
    result = []
    for c in convs:
        msg_count = manager.relational.get_conversation_message_count(c["id"])
        result.append({**c, "message_count": msg_count})
    return result


@app.get("/api/conversations/{conv_id}")
async def get_conversation(conv_id: str):
    """Get a conversation with all its messages."""
    conv = manager.relational.get_conversation(conv_id)
    if not conv:
        raise HTTPException(404, "Conversation not found")
    messages = manager.relational.get_messages(conv_id)
    return {**conv, "messages": messages}


@app.put("/api/conversations/{conv_id}")
async def rename_conversation(conv_id: str, req: ConvRename):
    """Rename a conversation."""
    conv = manager.relational.get_conversation(conv_id)
    if not conv:
        raise HTTPException(404, "Conversation not found")
    manager.relational.update_conversation_title(conv_id, req.title)
    return {"id": conv_id, "title": req.title}


@app.delete("/api/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    """Delete a conversation and all its messages."""
    manager.relational.delete_conversation(conv_id)
    return {"deleted": True}


# ── Chat (Streaming SSE) ────────────────────────────────────────────

@app.post("/api/conversations/{conv_id}/chat")
async def chat_in_conversation(conv_id: str, req: ChatMessage):
    """
    Stream a chat response within a specific conversation.
    Full pipeline: ingest → retrieve → rerank → pack → stream LLM → save.
    Returns SSE with metadata, tokens, and completion event.
    """
    conv = manager.relational.get_conversation(conv_id)
    if not conv:
        raise HTTPException(404, "Conversation not found")

    user_msg = req.message.strip()
    if not user_msg:
        raise HTTPException(400, "Empty message")

    # ── Step 1: Save user message to DB ──
    user_msg_id = str(uuid.uuid4())

    # ── Step 2: Ingest into memory system (auto-classify) ──
    from core.memory_classifier import MemoryClassifier
    _classifier = MemoryClassifier()
    classification_scores = _classifier.classify_with_confidence(user_msg)

    user_memory = manager.ingest(
        content=user_msg,
        source=f"chat:{conv_id[:8]}",
        memory_type=None,  # Let the classifier decide!
        importance=0.6,
    )

    user_meta = {
        "entities": user_memory.entities[:10],
        "intent": user_memory.intent,
        "memory_type": user_memory.memory_type.value,
        "classification_scores": {k.value: round(v, 3) for k, v in classification_scores.items()},
    }
    manager.relational.save_message(user_msg_id, conv_id, "user", user_msg, user_meta)

    # ── Step 3-5: Retrieve, Rerank, Graph Expand and Pack Context ──
    prep = reasoning.prepare_context(user_msg, top_k=req.top_k, relevance_threshold=0.3)
    context = prep["context"]

    # ── Step 6: Build conversation history for LLM ──
    past_messages = manager.relational.get_messages(conv_id)
    # Take last N messages for conversation context (avoid token overflow)
    recent_history = past_messages[-20:]  # Last 10 exchanges

    # ── Build metadata event (includes both semantic candidates and graph-expanded neighbors) ──
    memories_data = [
        {
            "id": sm.memory.id[:8],
            "content": sm.memory.content[:200],
            "type": sm.memory.memory_type.value,
            "score": round(sm.total_score, 4),
            "semantic": round(sm.semantic_score, 3),
            "recency": round(sm.recency_score, 3),
            "importance": round(sm.importance_score, 3),
            "graph": round(sm.graph_score, 3),
        }
        for sm in prep["all_memories"]
    ]

    meta_event = {
        "type": "metadata",
        "user_msg_id": user_msg_id,
        "memories_used": memories_data,
        "entities_extracted": user_memory.entities[:10],
        "intent": user_memory.intent,
        "memory_type": user_memory.memory_type.value,
        "classification_scores": {k.value: round(v, 3) for k, v in classification_scores.items()},
        "retrieval_time_ms": round(prep["retrieval_time_ms"], 1),
        "total_candidates": len(prep["reranked_memories"]),
    }

    async def event_stream():
        yield f"data: {json.dumps(meta_event)}\n\n"

        full_response = ""
        ai_msg_id = str(uuid.uuid4())

        if HAS_OLLAMA:
            try:
                system_prompt = (
                    "You are Cognitive AI, an intelligent assistant with persistent memory. "
                    "You have access to a memory retrieval system that may provide context from past interactions. "
                    "IMPORTANT RULES:\n"
                    "1. ONLY reference past memories if they are CLEARLY and DIRECTLY relevant to what the user is saying RIGHT NOW.\n"
                    "2. Do NOT say 'I remember' unless the retrieved memories genuinely match the current topic.\n"
                    "3. If memories are not relevant, just respond naturally to the user's current message as a fresh conversation.\n"
                    "4. Be conversational, helpful, and concise.\n"
                    "5. If the user explicitly asks what you remember, then describe your memories.\n"
                    "6. Never fabricate or hallucinate memories — only reference what is actually provided in the context below."
                )

                messages = [{"role": "system", "content": system_prompt}]

                # Add memory context ONLY if there are relevant memories
                if context:
                    messages.append({
                        "role": "system",
                        "content": (
                            "The following memories were retrieved from your knowledge base. "
                            "Only reference these if they are DIRECTLY relevant to the user's current message. "
                            "If none are relevant, ignore them entirely.\n\n"
                            f"{context}"
                        )
                    })

                # Add conversation history
                for msg in recent_history[:-1]:  # Exclude the just-saved user message
                    messages.append({
                        "role": msg["role"] if msg["role"] in ("user", "assistant") else "user",
                        "content": msg["content"]
                    })

                # Current user message
                messages.append({"role": "user", "content": user_msg})

                stream = ollama_client.chat(
                    model=config.llm.model,
                    messages=messages,
                    stream=True,
                    options={
                        "temperature": config.llm.temperature,
                        "num_predict": config.llm.max_tokens,
                    }
                )
                for chunk in stream:
                    token = chunk["message"]["content"]
                    full_response += token
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

            except Exception as e:
                full_response = f"Error connecting to Ollama: {str(e)}"
                yield f"data: {json.dumps({'type': 'token', 'content': full_response})}\n\n"
        else:
            fallback = "Ollama is not running. Here's what I found in memory:\n\n"
            for sm in prep["all_memories"][:3]:
                fallback += f"- [{sm.memory.memory_type.value}] {sm.memory.content[:100]}\n"
            if not prep["all_memories"]:
                fallback += "No relevant memories yet. Tell me something!"
            full_response = fallback
            yield f"data: {json.dumps({'type': 'token', 'content': fallback})}\n\n"

        # Save AI response to DB
        ai_meta = {
            "memories_used": memories_data,
            "retrieval_time_ms": round(prep["retrieval_time_ms"], 1),
        }
        manager.relational.save_message(ai_msg_id, conv_id, "assistant", full_response, ai_meta)

        # Ingest AI response into memory (auto-classify)
        if full_response and len(full_response) > 20:
            manager.ingest(
                content=f"AI said: {full_response[:500]}",
                source=f"ai_response:{conv_id[:8]}",
                memory_type=None,  # Let classifier decide
                importance=0.3,
            )

        # Auto-title the conversation if it's the first message
        if conv.get("title") == "New Chat" and len(past_messages) <= 1:
            title = _generate_title(user_msg)
            manager.relational.update_conversation_title(conv_id, title)
            yield f"data: {json.dumps({'type': 'title', 'title': title})}\n\n"

        yield f"data: {json.dumps({'type': 'done', 'ai_msg_id': ai_msg_id})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


def _generate_title(first_message: str) -> str:
    """Generate a short title from the first user message."""
    # Try LLM
    if HAS_OLLAMA:
        try:
            response = ollama_client.chat(
                model=config.llm.model,
                messages=[{
                    "role": "user",
                    "content": f"Generate a very short title (3-6 words max) for a conversation starting with: \"{first_message[:100]}\". Reply with ONLY the title, nothing else."
                }],
                options={"temperature": 0.3, "num_predict": 20}
            )
            title = response["message"]["content"].strip().strip('"').strip("'")
            return title[:60]
        except Exception:
            pass

    # Fallback: truncate
    words = first_message.split()[:6]
    return " ".join(words) + ("..." if len(first_message.split()) > 6 else "")


# ── Utility Endpoints ────────────────────────────────────────────────

@app.post("/api/reflection")
async def run_reflection():
    """Manually trigger a reflection cycle."""
    report = reflection.run_reflection_cycle()
    return {
        "memories_reviewed": report.memories_reviewed,
        "duplicates_merged": report.duplicates_merged,
        "memories_pruned": report.memories_pruned,
        "contradictions_found": report.contradictions_found,
        "summaries_created": report.summaries_created,
        "patterns": report.patterns_extracted,
        "insights": report.insights,
    }


@app.get("/api/stats")
async def get_stats():
    stats = manager.get_stats()
    convs = manager.relational.list_conversations()
    return {**stats, "conversation_count": len(convs)}


@app.get("/api/memory-breakdown")
async def memory_breakdown():
    """Get detailed memory breakdown by type with recent samples."""
    counts = manager.relational.get_memory_count()
    working_count = manager.working.count()

    breakdown = {
        "episodic": {
            "count": counts.get("episodic", 0),
            "description": "Conversations & events",
            "recent": [],
        },
        "semantic": {
            "count": counts.get("semantic", 0),
            "description": "Facts & knowledge",
            "recent": [],
        },
        "procedural": {
            "count": counts.get("procedural", 0),
            "description": "Workflows & how-tos",
            "recent": [],
        },
        "working": {
            "count": working_count,
            "description": "Active short-term context",
            "recent": [],
        },
    }

    # Fetch 3 most recent for each persistent type
    for mem_type_str in ["episodic", "semantic", "procedural"]:
        mem_type = MemoryType(mem_type_str)
        recent = manager.relational.get_memories_by_type(mem_type, limit=3)
        breakdown[mem_type_str]["recent"] = [
            {
                "id": m.id[:8],
                "content": m.content[:120],
                "importance": round(m.importance, 2),
                "created_at": m.created_at.isoformat(),
            }
            for m in recent
        ]

    # Working memory recent
    working_items = manager.working.get_all_active()[:3]
    breakdown["working"]["recent"] = [
        {
            "id": m.id[:8],
            "content": m.content[:120],
            "importance": round(m.importance, 2),
            "created_at": m.created_at.isoformat(),
        }
        for m in working_items
    ]

    total = sum(b["count"] for b in breakdown.values())
    return {"breakdown": breakdown, "total": total}


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "ollama": HAS_OLLAMA,
        "model": config.llm.model,
        "timestamp": datetime.utcnow().isoformat(),
    }


# ── Serve Frontend ───────────────────────────────────────────────────

@app.get("/")
async def serve_frontend():
    html_path = STATIC_DIR / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
