"""Fast deterministic task classification.

This is the first routing layer. It is intentionally explainable, sub-millisecond,
and dependency-free. A small local LLM is only consulted by the orchestrator when
this classifier reports ambiguity and the operator has enabled that layer.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

TASK_CATEGORIES = [
    "general",
    "reasoning",
    "coding",
    "code_generation",
    "debugging",
    "math",
    "summarization",
    "extraction",
    "translation",
    "creative_writing",
    "structured_json",
    "long_context",
    "classification",
    "vision",
    "embeddings",
]

TASK_CAPABILITY = {
    "general": "chat",
    "reasoning": "reasoning",
    "coding": "coding",
    "code_generation": "code_generation",
    "debugging": "debugging",
    "math": "math",
    "summarization": "summarization",
    "extraction": "extraction",
    "translation": "translation",
    "creative_writing": "creative",
    "structured_json": "json",
    "long_context": "long_context",
    "classification": "classification",
    "vision": "vision",
    "embeddings": "embeddings",
}

TASK_LABELS = {
    "general": "General conversation",
    "reasoning": "Complex reasoning",
    "coding": "Coding assistance",
    "code_generation": "Code generation",
    "debugging": "Debugging",
    "math": "Mathematics",
    "summarization": "Summarization",
    "extraction": "Text extraction",
    "translation": "Translation",
    "creative_writing": "Creative writing",
    "structured_json": "Structured JSON generation",
    "long_context": "Long-context analysis",
    "classification": "Lightweight classification",
    "vision": "Vision understanding",
    "embeddings": "Embeddings",
}


@dataclass(slots=True)
class TaskClassification:
    category: str
    confidence: float
    scores: dict[str, float]
    signals: list[str]
    language: str
    estimated_tokens: int
    complexity: str
    long_context: bool
    needs_json: bool
    needs_tools: bool
    needs_vision: bool
    ambiguous: bool
    source: str = "heuristic"
    rationale: str = ""
    alternatives: list[str] = field(default_factory=list)

    @property
    def capability(self) -> str:
        return TASK_CAPABILITY.get(self.category, "chat")

    @property
    def label(self) -> str:
        return TASK_LABELS.get(self.category, self.category)

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "label": self.label,
            "capability": self.capability,
            "confidence": round(self.confidence, 3),
            "scores": {k: round(v, 2) for k, v in sorted(self.scores.items(), key=lambda kv: -kv[1])},
            "signals": self.signals[:8],
            "language": self.language,
            "estimated_tokens": self.estimated_tokens,
            "complexity": self.complexity,
            "long_context": self.long_context,
            "needs_json": self.needs_json,
            "needs_tools": self.needs_tools,
            "needs_vision": self.needs_vision,
            "ambiguous": self.ambiguous,
            "source": self.source,
            "rationale": self.rationale,
            "alternatives": self.alternatives,
        }


_JSON_REQUEST = re.compile(
    r"(?i)(return|respond|output|reply|format).{0,30}\bjson\b|json\s+(schema|object|output)|\bas\s+json\b"
)
_CODE_HINT = re.compile(
    r"(?i)(```|def\s+\w+\s*\(|class\s+\w+|function\s+\w+\s*\(|import\s+\w+|"
    r"#include\s*<|public\s+static|const\s+\w+\s*=|SELECT\s+.+\s+FROM|"
    r"<\w+[^>]*>|npm\s+|pip\s+install|git\s+|docker\s+|kubectl\s+)"
)
_CODE_VERB = re.compile(
    r"(?i)\b(write|implement|create|build|generate|refactor|optimize|convert|add)\b.{0,40}"
    r"\b(code|function|class|script|program|component|endpoint|api|method|module|query|regex)\b"
)
_DEBUG = re.compile(
    r"(?i)\b(bug|error|exception|traceback|stack\s*trace|fails?|failing|broken|"
    r"crash(es|ed)?|segfault|undefined|null\s+pointer|debug|fix\s+this|why\s+(does|is).+fail)\b"
)
_MATH = re.compile(
    r"(?i)\b(calculate|solve|equation|integral|derivative|matrix|probability|prove|"
    r"theorem|algebra|geometry|statistics|arithmetic|percentage|factorial)\b|"
    r"\d+\s*[\+\-\*/^=]\s*\d+|\\frac|\\sum|∑|√"
)
_SUMMARY = re.compile(
    r"(?i)\b(summar(y|ize|ise)|tl;?dr|key\s+points|main\s+(points|ideas)|abstract|"
    r"condense|shorten|recap|overview\s+of)\b"
)
_EXTRACT = re.compile(
    r"(?i)\b(extract|pull\s+out|parse|find\s+all|list\s+all|collect|identify)\b.{0,50}"
    r"\b(name|email|phone|field|value|entity|date|address|invoice|data|information)\b|"
    r"\bconvert\s+(this|the)\s+(document|text|html|page)\s+(in)?to\s+(json|csv|table)\b"
)
_TRANSLATE = re.compile(
    r"(?i)\b(translate|translation|in\s+(french|spanish|german|japanese|chinese|hindi|"
    r"portuguese|italian|korean|arabic|russian)|to\s+(french|spanish|german|japanese|"
    r"chinese|hindi|portuguese|italian|korean|arabic|russian))\b"
)
_CREATIVE = re.compile(
    r"(?i)\b(story|poem|poetry|lyrics|novel|screenplay|scene|fiction|creative|"
    r"write\s+me\s+a|once\s+upon\s+a\s+time|haiku|sonnet)\b"
)
_CLASSIFY = re.compile(
    r"(?i)\b(classify|classification|categori[sz]e|sentiment|label\s+(this|the)|"
    r"is\s+this\s+(positive|negative|spam|toxic)|which\s+category)\b"
)
_REASONING = re.compile(
    r"(?i)\b(step[\s-]by[\s-]step|reasoning|derive|prove|analy[sz]e|compare|evaluate|"
    r"trade[\s-]?offs?|architecture|design\s+(a|the)|root\s+cause|implications|"
    r"explain\s+why|what\s+happens\s+if|strategy|plan|critique)\b"
)
_LONG = re.compile(
    r"(?i)\b(entire|whole|full)\s+(document|codebase|transcript|book|report|conversation)|"
    r"\ball\s+of\s+(the\s+)?(above|text|document)|long[\s-]context|throughout\s+the\s+document\b"
)
_CLASSIFICATION_INPUT = re.compile(
    r"(?i)^(is\s+this|classify|what\s+is\s+the\s+sentiment|label|categorize|spam\s+or\s+not)"
)
_CODE_LANGS = {
    "python",
    "javascript",
    "typescript",
    "rust",
    "go",
    "java",
    "c++",
    "c#",
    "sql",
    "html",
    "css",
    "bash",
    "shell",
    "yaml",
    "json",
    "react",
    "kubernetes",
    "docker",
}
_SCRIPT_RE = {
    "hindi": re.compile(r"[\u0900-\u097F]"),
    "arabic": re.compile(r"[\u0600-\u06FF]"),
    "cyrillic": re.compile(r"[\u0400-\u04FF]"),
    "japanese": re.compile(r"[\u3040-\u30FF]"),
    "chinese": re.compile(r"[\u4E00-\u9FFF]"),
    "korean": re.compile(r"[\uAC00-\uD7AF]"),
    "greek": re.compile(r"[\u0370-\u03FF]"),
    "hebrew": re.compile(r"[\u0590-\u05FF]"),
}
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]+")


def estimate_tokens(text: str) -> int:
    """Cheap tokenizer-neutral estimate. Roughly 3.6 chars/token for English."""
    if not text:
        return 0
    cjk = len(re.findall(r"[\u3000-\u9fff\u3040-\u30ff\uac00-\ud7af]", text))
    other = max(0, len(text) - cjk)
    return max(1, math.ceil(other / 3.6) + cjk)


def flatten_messages(messages: list[dict[str, Any]] | None) -> tuple[str, str, bool]:
    """Return (all_text, last_user_text, has_image)."""
    parts: list[str] = []
    last_user = ""
    image = False
    for msg in messages or []:
        role = str(msg.get("role") or "")
        content = msg.get("content")
        text = ""
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            chunks = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") in ("text", "input_text"):
                        chunks.append(str(part.get("text") or ""))
                    elif part.get("type") in ("image_url", "input_image", "image"):
                        image = True
                elif isinstance(part, str):
                    chunks.append(part)
            text = "\n".join(chunks)
        if text:
            parts.append(f"{role}: {text}")
            if role == "user":
                last_user = text
    return "\n".join(parts), last_user, image


def classify_task(
    *,
    messages: list[dict[str, Any]] | None,
    prompt: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    response_format: dict[str, Any] | None = None,
    max_tokens: int = 0,
) -> TaskClassification:
    all_text, last_user, has_image = flatten_messages(messages)
    if prompt:
        all_text = prompt if not all_text else all_text + "\n" + prompt
        last_user = prompt or last_user
    response_type = str((response_format or {}).get("type") or "")
    needs_json = response_type in ("json_object", "json_schema") or bool(_JSON_REQUEST.search(last_user))
    needs_tools = bool(tools)
    tokens = estimate_tokens(all_text) + sum(estimate_tokens(json.dumps(t, default=str)) for t in (tools or []))
    prompt_tokens = tokens
    scores: dict[str, float] = {"general": 1.0}
    signals: list[str] = []

    def add(category: str, weight: float, signal: str | None = None) -> None:
        scores[category] = scores.get(category, 0) + weight
        if signal:
            signals.append(signal)

    lower = last_user.lower()
    code_hits = _CODE_HINT.findall(last_user)
    has_code_block = "```" in last_user
    if needs_json:
        add("structured_json", 5.5, "JSON output requested")
    if has_image:
        add("vision", 8.0, "image content supplied")
        signals.append("multimodal input")
    if _TRANSLATE.search(last_user):
        add("translation", 5.5, "translation request")
    if _SUMMARY.search(last_user):
        add("summarization", 5.0, "summarization verb")
    if _EXTRACT.search(last_user):
        add("extraction", 5.0, "extraction pattern")
    if _CLASSIFY.search(last_user) or (
        len(last_user) < 600 and _CLASSIFICATION_INPUT.search(last_user.strip())
    ):
        add("classification", 4.2, "classification/labeling request")
    if _CREATIVE.search(last_user):
        add("creative_writing", 4.8, "creative-writing language")
    debug_hit = _DEBUG.search(last_user)
    if debug_hit:
        weight = 5.2 if has_code_block or code_hits or "```" in all_text else 3.6
        add("debugging", weight, "failure/error language")
    if _MATH.search(last_user):
        add("math", 4.4, "mathematical expression or vocabulary")
    code_request = _CODE_VERB.search(last_user)
    if code_request:
        add("code_generation", 5.2, "code-generation verb")
    if code_hits or has_code_block or any(lang in lower for lang in _CODE_LANGS):
        add("coding", 2.8 if not debug_hit else 3.2, "code syntax or language")
    if _REASONING.search(last_user):
        add("reasoning", 3.0, "reasoning/analysis vocabulary")
    if tokens > 6000:
        add("long_context", 3.5, f"large prompt (~{tokens} tokens)")
    if _LONG.search(last_user):
        add("long_context", 4.5, "whole-document/long-context request")
    if needs_tools:
        add("reasoning", 1.2, "tool use requested")
    if len(last_user) < 120 and re.match(
        r"(?i)^(hi|hello|hey|thanks|thank you|good (morning|evening)|how are you)\b", last_user.strip()
    ):
        add("general", 3.0, "greeting")

    # Code generation and debugging are distinct; debugging wins if both exist.
    if scores.get("debugging", 0) >= 3.5:
        scores["code_generation"] = scores.get("code_generation", 0) * 0.6
    if scores.get("structured_json", 0) >= 4:
        scores["extraction"] = scores.get("extraction", 0) * 0.7

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    top_category, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0
    margin = top_score - second_score
    confidence = 0.48 + min(0.28, top_score / 22.0) + min(0.18, margin / 18.0)
    if top_score >= 4 and margin >= 2.0:
        confidence += 0.06
    confidence = round(min(0.97, max(0.35, confidence)), 3)
    ambiguous = confidence < 0.62 or (margin < 1.2 and top_category != "general")
    if top_category == "general" and len(last_user) < 80:
        confidence = max(confidence, 0.8)
        ambiguous = False

    language = "english"
    for name, pattern in _SCRIPT_RE.items():
        if len(pattern.findall(last_user)) > 2:
            language = name
            break
    else:
        words = [w.lower() for w in _WORD_RE.findall(last_user)]
        if len(words) >= 6:
            common = {
                "french": {"le", "la", "les", "des", "une", "est", "pour", "avec", "vous"},
                "spanish": {"el", "la", "los", "una", "es", "para", "con", "que", "por"},
                "german": {"der", "die", "das", "und", "ist", "nicht", "mit", "für", "ein"},
                "portuguese": {"o", "a", "os", "uma", "não", "para", "com", "que", "por"},
                "italian": {"il", "la", "gli", "una", "non", "per", "con", "che", "sono"},
            }
            for lang, markers in common.items():
                if len(markers.intersection(words)) >= 2:
                    language = lang
                    break

    complexity = "low"
    if tokens > 500 or top_category in ("reasoning", "debugging", "math", "long_context"):
        complexity = "medium"
    if tokens > 4000 or (top_category in ("reasoning", "math") and top_score >= 5):
        complexity = "high"
    if top_category == "long_context":
        complexity = "high"
    needs_vision = has_image or top_category == "vision"
    required_tokens = prompt_tokens + max(0, max_tokens) + 64
    rationale = _rationale(top_category, signals, top_score, margin, language)
    return TaskClassification(
        category=top_category,
        confidence=confidence,
        scores=scores,
        signals=signals,
        language=language,
        estimated_tokens=prompt_tokens,
        complexity=complexity,
        long_context=top_category == "long_context" or tokens > 4000,
        needs_json=needs_json,
        needs_tools=needs_tools,
        needs_vision=needs_vision,
        ambiguous=ambiguous,
        rationale=rationale,
        alternatives=[cat for cat, _ in ranked[1:4]],
    )


def _rationale(category: str, signals: list[str], score: float, margin: float, language: str) -> str:
    label = TASK_LABELS.get(category, category)
    if signals:
        evidence = "; ".join(dict.fromkeys(signals[:3]))
        return f"{label} selected from prompt features: {evidence}."
    return f"{label} selected by default conversational heuristics (score {score:.1f}, margin {margin:.1f})."


_LLM_PROMPT = """Classify the user's request into exactly one category.
Categories: {categories}
Return ONLY a JSON object, no prose:
{{"task":"<category>","confidence":<0..1>,"reason":"<8 words max>"}}
Request:
{prompt}
JSON:"""


def build_llm_prompt(classification: TaskClassification, text: str) -> str:
    return _LLM_PROMPT.format(
        categories=", ".join(TASK_CATEGORIES),
        prompt=text[-4000:],
    )


def parse_llm_classification(payload: str, fallback: TaskClassification) -> TaskClassification | None:
    match = re.search(r"\{.*?\}", payload, re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    task = str(data.get("task") or "").strip().lower().replace(" ", "_")
    if task not in TASK_CATEGORIES:
        aliases = {
            "code": "coding",
            "code_generation": "code_generation",
            "coding": "coding",
            "debug": "debugging",
            "json": "structured_json",
            "creative": "creative_writing",
            "summary": "summarization",
            "long": "long_context",
            "chat": "general",
        }
        task = aliases.get(task, "")
    if not task:
        return None
    try:
        conf = float(data.get("confidence", fallback.confidence))
    except (TypeError, ValueError):
        conf = fallback.confidence
    scores = dict(fallback.scores)
    scores[task] = scores.get(task, 0) + 3.0
    rationale = f"Ambiguous heuristic result refined by lightweight router model: {data.get('reason', '')}".strip()
    result = TaskClassification(
        category=task,
        confidence=max(0.5, min(0.95, conf)),
        scores=scores,
        signals=fallback.signals + ["LLM classifier arbitration"],
        language=fallback.language,
        estimated_tokens=fallback.estimated_tokens,
        complexity=fallback.complexity,
        long_context=fallback.long_context or task == "long_context",
        needs_json=fallback.needs_json,
        needs_tools=fallback.needs_tools,
        needs_vision=fallback.needs_vision,
        ambiguous=False,
        source="llm",
        rationale=rationale,
        alternatives=[fallback.category],
    )
    return result
