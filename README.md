# 🧠 Cognitive AI Memory System

A production-grade, multi-layered cognitive memory architecture that mimics human memory systems — with episodic, semantic, procedural, and working memory, a reflection engine, knowledge graphs, and hybrid retrieval.

## Architecture

```
User Input
    │
    ▼
┌─────────────────────┐
│  Input Processor     │ ← Text cleaning, NER, intent analysis
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  Memory Classifier   │ ← Episodic / Semantic / Procedural / Working
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  Embedding Engine    │ ← SentenceTransformers (all-MiniLM-L6-v2)
└────────┬────────────┘
         ▼
┌─────────────────────────────────────────┐
│         Multi-Layer Storage             │
│  ┌──────────┐ ┌──────────┐ ┌─────────┐ │
│  │  SQLite   │ │ ChromaDB │ │NetworkX │ │
│  │ metadata  │ │ vectors  │ │  graph  │ │
│  └──────────┘ └──────────┘ └─────────┘ │
└─────────────────┬───────────────────────┘
                  ▼
┌─────────────────────┐
│  Hybrid Retriever    │ ← 5-factor weighted scoring
│  + Reranker          │ ← Quality filtering + diversity
│  + Context Packer    │ ← Token budget optimization
└────────┬────────────┘
         ▼
┌─────────────────────┐
│  Reasoning Engine    │ ← Ollama LLM with memory context
└─────────────────────┘

         ↕ (periodic)

┌─────────────────────┐
│  Reflection Engine   │ ← Dedup, decay, pruning, insights
└─────────────────────┘
```

## Features

- **4 Memory Types**: Episodic (events), Semantic (knowledge), Procedural (workflows), Working (short-term)
- **Hybrid Retrieval**: `0.4×Semantic + 0.2×Recency + 0.2×Importance + 0.1×Graph + 0.1×Priority`
- **Knowledge Graph**: Auto-linking by shared entities, centrality scoring, cluster detection
- **Reflection Engine**: Duplicate merging, contradiction detection, pattern extraction, decay
- **Context Optimization**: Token-budgeted packing with summary fallback
- **Near-Duplicate Detection**: >95% similarity auto-merges instead of duplicating
- **Memory Decay**: Ebbinghaus-inspired forgetting curves with reinforcement on access

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download spaCy model
python -m spacy download en_core_web_sm

# 4. Run the demo
python demo.py

# 5. Or start the API server
uvicorn main:app --reload --port 8000
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/memory/ingest` | Store a new memory (auto-classifies, embeds, graphs) |
| `POST` | `/memory/query` | Hybrid retrieval with full score breakdown |
| `POST` | `/memory/reason` | Memory-augmented LLM reasoning |
| `POST` | `/episode/start` | Start a conversation episode |
| `POST` | `/episode/end` | End current episode |
| `POST` | `/reflection/run` | Trigger reflection cycle |
| `GET`  | `/stats` | System statistics |
| `GET`  | `/health` | Health check |

### Example: Ingest

```bash
curl -X POST http://localhost:8000/memory/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "content": "User prefers deep AI systems over shallow chatbot projects",
    "tags": ["preference"],
    "importance": 0.9
  }'
```

### Example: Query

```bash
curl -X POST http://localhost:8000/memory/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What kind of projects does the user like?",
    "top_k": 5
  }'
```

## Project Structure

```
├── main.py                     # FastAPI application
├── config.py                   # All configuration
├── demo.py                     # Interactive demo script
├── requirements.txt            # Dependencies
│
├── core/                       # Processing pipeline
│   ├── input_processor.py      # Text cleaning, NER, intent
│   ├── memory_classifier.py    # Classify memory type
│   ├── embedding_engine.py     # Generate embeddings
│   └── reasoning_engine.py     # LLM-powered reasoning
│
├── memory/                     # Memory layers
│   ├── base.py                 # Abstract interface
│   ├── episodic.py             # Conversations, events
│   ├── semantic.py             # Knowledge, concepts
│   ├── working.py              # Short-term cache (TTL)
│   ├── procedural.py           # Workflows, strategies
│   └── manager.py              # Unified orchestrator
│
├── storage/                    # Storage backends
│   ├── vector_store.py         # ChromaDB
│   ├── graph_store.py          # NetworkX
│   └── relational_store.py     # SQLite
│
├── retrieval/                  # Retrieval engine
│   ├── hybrid_retriever.py     # 5-factor scoring
│   ├── reranker.py             # Quality + diversity
│   └── context_packer.py       # Token budget
│
├── reflection/                 # Memory maintenance
│   ├── reflection_engine.py    # Full reflection cycle
│   ├── compressor.py           # Summarization
│   └── decay.py                # Forgetting curves
│
├── graph/
│   └── memory_graph.py         # High-level graph ops
│
├── models/
│   └── schemas.py              # All Pydantic models
│
└── tests/                      # Test suite
    ├── test_input_processor.py
    ├── test_memory.py
    ├── test_retrieval.py
    └── test_reflection.py
```

## Configuration

All parameters are in `config.py`:

| Config | Default | Description |
|--------|---------|-------------|
| Embedding model | `all-MiniLM-L6-v2` | Upgradeable to BGE-large |
| Retrieval semantic weight | 0.4 | Weight for cosine similarity |
| Decay λ | 0.01 | Forgetting rate |
| Reflection interval | 60 min | How often reflection runs |
| Working memory TTL | 3600s | Short-term memory lifetime |
| Context budget | 4096 tokens | Max LLM context |
| Similarity merge threshold | 0.92 | Duplicate detection |

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

## Requirements

- Python 3.10+
- ~500MB disk for embedding model (first run)
- Ollama (optional, for LLM reasoning/reflection)
