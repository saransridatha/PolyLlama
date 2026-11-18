from __future__ import annotations

from app.router.classifier import (
    TASK_CATEGORIES,
    classify_task,
    estimate_tokens,
    parse_llm_classification,
)


def classify(content: str, **kwargs):
    return classify_task(messages=[{"role": "user", "content": content}], **kwargs)


def test_greeting_is_general():
    task = classify("Hello, how are you?")
    assert task.category == "general"
    assert task.confidence >= 0.7


def test_code_generation_detection():
    task = classify(
        "Write a Python function that reads a CSV file and returns a list of dictionaries."
    )
    assert task.category == "code_generation"
    assert task.capability == "code_generation"


def test_debugging_detection_prefers_debugging_over_generation():
    task = classify(
        "Why does this fail?\n```python\ndef f():\n    return 1/0\n```\n"
        "Traceback: ZeroDivisionError"
    )
    assert task.category == "debugging"
    assert "debugging" in task.scores


def test_summarization_detection():
    assert classify("Summarize the key points of this release note.").category == "summarization"


def test_translation_detection():
    assert classify("Translate this paragraph into French.").category == "translation"


def test_math_detection():
    assert classify("Calculate the integral of x^2 from 0 to 3.").category == "math"


def test_creative_detection():
    assert classify("Write a short poem about a lighthouse.").category == "creative_writing"


def test_classification_detection():
    assert classify("Classify the sentiment of this review: great product").category == "classification"


def test_json_detection_from_response_format():
    task = classify("Give me the result.", response_format={"type": "json_object"})
    assert task.category == "structured_json"
    assert task.needs_json is True


def test_json_detection_from_text():
    task = classify("Return this data as JSON with fields name and email.")
    assert task.category == "structured_json"


def test_long_context_detection():
    task = classify("Analyze the entire document and summarize every section. " + "word " * 8000)
    assert task.category in ("long_context", "summarization")
    assert task.long_context is True
    assert task.estimated_tokens > 4000


def test_tool_use_flag():
    task = classify(
        "Find the weather",
        tools=[{"type": "function", "function": {"name": "weather"}}],
    )
    assert task.needs_tools is True


def test_language_detection():
    task = classify("Bonjour, je voudrais une explication avec des examples pour ce sujet.")
    assert task.language == "french"
    task = classify("これを日本語で説明してください。")
    assert task.language == "japanese"


def test_vision_input():
    task = classify_task(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this chart?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                ],
            }
        ]
    )
    assert task.needs_vision is True
    assert task.category == "vision"


def test_token_estimate_handles_cjk():
    latin = estimate_tokens("hello world " * 10)
    cjk = estimate_tokens("你好世界" * 10)
    assert latin > 0
    assert cjk > 0


def test_llm_classification_parser():
    base = classify("Do something ambiguous")
    parsed = parse_llm_classification(
        '{"task": "debugging", "confidence": 0.88, "reason": "stack trace"}', base
    )
    assert parsed is not None
    assert parsed.category == "debugging"
    assert parsed.source == "llm"
    assert parsed.confidence == 0.88


def test_llm_classification_parser_rejects_garbage():
    base = classify("Hello")
    assert parse_llm_classification("not json", base) is None
    assert parse_llm_classification('{"task": "unknown_thing"}', base) is None


def test_all_categories_have_unique_names():
    assert len(TASK_CATEGORIES) == len(set(TASK_CATEGORIES))
