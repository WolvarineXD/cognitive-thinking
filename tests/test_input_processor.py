"""
Tests for the Input Processor.
Tests text cleaning, chunking, entity extraction, intent analysis.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from core.input_processor import InputProcessor


@pytest.fixture
def processor():
    return InputProcessor()


class TestTextCleaning:
    def test_removes_excessive_whitespace(self, processor):
        result = processor.clean_text("hello    world")
        assert result == "hello world"

    def test_strips_leading_trailing(self, processor):
        result = processor.clean_text("  hello world  ")
        assert result == "hello world"

    def test_normalizes_smart_quotes(self, processor):
        result = processor.clean_text("\u201cquoted\u201d")
        assert result == '"quoted"'

    def test_removes_control_characters(self, processor):
        result = processor.clean_text("hello\x00world")
        assert result == "helloworld"

    def test_preserves_normal_text(self, processor):
        text = "This is a normal sentence."
        assert processor.clean_text(text) == text


class TestChunking:
    def test_short_text_single_chunk(self, processor):
        text = "Short text"
        chunks = processor.chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_multiple_chunks(self, processor):
        text = "This is a sentence. " * 100  # ~2000 chars
        chunks = processor.chunk_text(text, max_chunk_size=200)
        assert len(chunks) > 1

    def test_chunks_not_empty(self, processor):
        text = "Word " * 200
        chunks = processor.chunk_text(text, max_chunk_size=100)
        for chunk in chunks:
            assert len(chunk.strip()) > 0


class TestIntentAnalysis:
    def test_question_intent(self, processor):
        assert processor.analyze_intent("What is machine learning?") == "question"

    def test_command_intent(self, processor):
        assert processor.analyze_intent("Create a new project") == "command"

    def test_preference_intent(self, processor):
        assert processor.analyze_intent("I prefer deep learning systems") == "preference"

    def test_planning_intent(self, processor):
        assert processor.analyze_intent("Plan the workflow and pipeline steps") == "planning"

    def test_general_intent_fallback(self, processor):
        assert processor.analyze_intent("abcxyz") == "general"


class TestEntityExtraction:
    def test_extracts_entities(self, processor):
        entities = processor.extract_entities("Google announced a new AI model in California")
        # Should extract at least some entities
        assert isinstance(entities, list)

    def test_caps_at_20(self, processor):
        # Very entity-rich text
        text = "Apple, Google, Microsoft, Amazon, Meta, Tesla, Netflix, " * 5
        entities = processor.extract_entities(text)
        assert len(entities) <= 20

    def test_empty_input(self, processor):
        entities = processor.extract_entities("")
        assert isinstance(entities, list)


class TestFullPipeline:
    def test_process_returns_all_keys(self, processor):
        result = processor.process("Tell me about machine learning")
        assert "cleaned_text" in result
        assert "chunks" in result
        assert "entities" in result
        assert "intent" in result
        assert "tags" in result
        assert "metadata" in result

    def test_metadata_has_timestamps(self, processor):
        result = processor.process("Test input")
        assert "timestamp" in result["metadata"]
        assert "source" in result["metadata"]

    def test_tags_include_intent(self, processor):
        result = processor.process("How to build a pipeline?")
        assert any(tag in result["tags"] for tag in ["question", "planning", "command"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
