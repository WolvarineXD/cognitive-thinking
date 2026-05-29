"""
Memory Classifier — Step 2 of the pipeline.

Classifies incoming text into memory types:
    - Episodic: personal experiences, conversations, events
    - Semantic: facts, concepts, general knowledge
    - Procedural: how-to, workflows, step-by-step processes
    - Working: temporary, context-only, short-lived

Uses a rule-based classifier with scoring.
Can be upgraded to a fine-tuned transformer later.
"""
import re
from models.schemas import MemoryType


class MemoryClassifier:
    """Classifies text into memory categories using weighted heuristics."""

    # Patterns that signal each memory type
    PATTERNS = {
        MemoryType.EPISODIC: {
            "keywords": [
                "i did", "we discussed", "yesterday", "last time", "remember when",
                "conversation", "meeting", "told me", "asked about", "talked about",
                "happened", "event", "experience", "session", "interaction",
                "you said", "i said", "earlier", "previously", "that time",
            ],
            "regex": [
                r"\bi\s+(did|was|went|saw|met|talked|asked|told)\b",
                r"\b(yesterday|last\s+week|last\s+time|earlier\s+today)\b",
                r"\bremember\s+when\b",
            ],
            "weight": 1.0,
        },
        MemoryType.SEMANTIC: {
            "keywords": [
                "is a", "means", "definition", "concept", "fact", "always",
                "generally", "usually", "knowledge", "theory", "principle",
                "refers to", "known as", "described as", "fundamentally",
                "in general", "typically", "abstract", "pattern",
            ],
            "regex": [
                r"\b(is|are|was|were)\s+(a|an|the)\s+\w+",
                r"\bdefined\s+as\b",
                r"\bknown\s+(as|for)\b",
            ],
            "weight": 1.0,
        },
        MemoryType.PROCEDURAL: {
            "keywords": [
                "how to", "step", "workflow", "pipeline", "process",
                "first", "then", "next", "finally", "procedure",
                "instructions", "method", "approach", "strategy",
                "algorithm", "recipe", "guide", "tutorial",
                "execute", "implement", "build", "create", "deploy",
            ],
            "regex": [
                r"\b(step\s+\d+|first|then|next|finally)\b",
                r"\bhow\s+to\b",
                r"\b(workflow|pipeline|procedure)\b",
            ],
            "weight": 1.0,
        },
        MemoryType.WORKING: {
            "keywords": [
                "right now", "currently", "at the moment", "temporary",
                "for now", "just", "quickly", "hold this", "note to self",
                "scratch", "temp", "draft", "placeholder",
            ],
            "regex": [
                r"\b(right\s+now|currently|at\s+the\s+moment)\b",
                r"\b(temporary|temp|scratch|draft)\b",
            ],
            "weight": 0.8,  # Lower weight — default to persistent memory
        },
    }

    def classify(self, text: str, context: str | None = None) -> MemoryType:
        """
        Classify text into a memory type.
        Returns the type with the highest confidence score.
        """
        scores = self._score_all(text, context)

        if not scores or max(scores.values()) == 0:
            return MemoryType.SEMANTIC  # Default: treat as knowledge

        return max(scores, key=scores.get)

    def classify_with_confidence(self, text: str,
                                  context: str | None = None) -> dict[MemoryType, float]:
        """Return all type scores normalized to 0-1."""
        scores = self._score_all(text, context)
        total = sum(scores.values())
        if total == 0:
            return {t: 0.25 for t in MemoryType}  # Uniform uncertainty
        return {t: s / total for t, s in scores.items()}

    def _score_all(self, text: str, context: str | None = None) -> dict[MemoryType, float]:
        """Compute raw scores for each memory type."""
        text_lower = text.lower()
        full_text = f"{text_lower} {(context or '').lower()}"
        scores = {}

        for mem_type, patterns in self.PATTERNS.items():
            score = 0.0

            # Keyword matching
            keyword_hits = sum(1 for kw in patterns["keywords"] if kw in full_text)
            score += keyword_hits * 0.5

            # Regex matching
            regex_hits = sum(1 for rx in patterns["regex"] if re.search(rx, full_text))
            score += regex_hits * 1.0

            # Apply type weight
            score *= patterns["weight"]

            scores[mem_type] = score

        return scores
