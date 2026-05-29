"""
Input Processor — Step 1 of the pipeline.

Handles:
    - Text cleaning
    - Chunking (for long inputs)
    - Entity extraction (using spaCy)
    - Intent analysis (keyword + pattern based)
    - Metadata extraction (timestamps, sources)
"""
import re
from datetime import datetime
from typing import Optional

# spaCy loaded lazily to avoid slow startup
_nlp = None


def _get_nlp():
    """Lazy-load spaCy model."""
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Model not installed — fall back to blank
            import spacy
            _nlp = spacy.blank("en")
    return _nlp


class InputProcessor:
    """Processes raw user input into structured, enriched data."""

    # Intent keywords mapping
    INTENT_PATTERNS = {
        "question": ["what", "how", "why", "when", "where", "who", "which", "?"],
        "command": ["do", "make", "create", "build", "run", "execute", "start", "stop"],
        "information": ["tell me", "explain", "describe", "define", "show"],
        "preference": ["i like", "i prefer", "i want", "i need", "i hate", "i love"],
        "feedback": ["good", "bad", "great", "terrible", "awesome", "awful", "nice"],
        "planning": ["plan", "strategy", "roadmap", "steps", "workflow", "pipeline"],
        "debugging": ["error", "bug", "fix", "broken", "issue", "problem", "crash"],
        "learning": ["learn", "understand", "study", "tutorial", "guide", "teach"],
    }

    def __init__(self):
        self.nlp = _get_nlp()

    def process(self, text: str, source: str = "user_input") -> dict:
        """
        Full processing pipeline on a piece of text.

        Returns dict with:
            - cleaned_text: normalized text
            - chunks: list of text chunks (for long inputs)
            - entities: extracted named entities
            - intent: detected intent category
            - metadata: timestamps, source, length info
            - tags: auto-generated tags from entities + intent
        """
        cleaned = self.clean_text(text)
        chunks = self.chunk_text(cleaned)
        entities = self.extract_entities(cleaned)
        intent = self.analyze_intent(cleaned)
        tags = self._generate_tags(entities, intent)

        return {
            "cleaned_text": cleaned,
            "chunks": chunks,
            "entities": entities,
            "intent": intent,
            "tags": tags,
            "metadata": {
                "source": source,
                "timestamp": datetime.utcnow().isoformat(),
                "original_length": len(text),
                "cleaned_length": len(cleaned),
                "chunk_count": len(chunks),
                "entity_count": len(entities),
            }
        }

    def clean_text(self, text: str) -> str:
        """Normalize and clean text input."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove control characters (keep newlines)
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        # Normalize quotes
        text = text.replace('\u201c', '"').replace('\u201d', '"')
        text = text.replace('\u2018', "'").replace('\u2019', "'")
        return text

    def chunk_text(self, text: str, max_chunk_size: int = 512,
                   overlap: int = 50) -> list[str]:
        """
        Split text into overlapping chunks.
        Short texts return a single chunk.
        """
        if len(text) <= max_chunk_size:
            return [text]

        chunks = []
        sentences = self._split_sentences(text)
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                # Keep overlap
                words = current_chunk.split()
                overlap_words = words[-overlap:] if len(words) > overlap else words
                current_chunk = " ".join(overlap_words) + " " + sentence
            else:
                current_chunk += " " + sentence

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks if chunks else [text]

    def extract_entities(self, text: str) -> list[str]:
        """Extract named entities using spaCy."""
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            entity_text = ent.text.strip()
            if entity_text and entity_text not in entities:
                entities.append(entity_text)

        # Also extract noun chunks as potential concepts
        try:
            for chunk in doc.noun_chunks:
                chunk_text = chunk.text.strip()
                if (len(chunk_text) > 3 and chunk_text not in entities
                        and chunk_text.lower() not in ["i", "you", "we", "they", "it"]):
                    entities.append(chunk_text)
        except Exception:
            pass  # Some spaCy models don't support noun chunks

        return entities[:20]  # Cap at 20 to prevent bloat

    def analyze_intent(self, text: str) -> str:
        """Classify the intent of the text based on keyword patterns."""
        text_lower = text.lower()

        scores = {}
        for intent, keywords in self.INTENT_PATTERNS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[intent] = score

        if not scores:
            return "general"

        return max(scores, key=scores.get)

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        return sentences if sentences else [text]

    def _generate_tags(self, entities: list[str], intent: str) -> list[str]:
        """Auto-generate tags from extracted info."""
        tags = [intent]
        # Add top entities as tags (lowercased, cleaned)
        for entity in entities[:5]:
            tag = re.sub(r'[^a-zA-Z0-9_\-]', '_', entity.lower()).strip('_')
            if tag and len(tag) > 2:
                tags.append(tag)
        return list(set(tags))
