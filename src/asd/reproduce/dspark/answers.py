"""Strict numeric answer extraction for the frozen GSM8K protocol."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

_NUMBER_BODY = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_FULL_NUMBER = re.compile(rf"(?:\\?\$\s*)?({_NUMBER_BODY})")
_REFERENCE = re.compile(rf"####\s*((?:\\?\$\s*)?{_NUMBER_BODY})")
_BOX_START = re.compile(r"\\boxed\s*\{")


@dataclass(frozen=True)
class AnswerExtraction:
    value: str | None
    rule: str
    raw: str | None


def normalize_number(raw: str) -> str | None:
    """Normalize a complete finite decimal without guessing from surrounding text."""

    if not isinstance(raw, str):
        return None
    candidate = raw.strip().replace(r"\$", "$")
    match = _FULL_NUMBER.fullmatch(candidate)
    if match is None:
        return None
    candidate = match.group(1).replace(",", "")
    try:
        value = Decimal(candidate)
    except InvalidOperation:
        return None
    if not value.is_finite():
        return None
    if value == 0:
        return "0"
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def extract_reference_answer(answer: str) -> AnswerExtraction:
    """Extract the last numeric GSM8K ``####`` answer marker."""

    if not isinstance(answer, str):
        return AnswerExtraction(None, "last_hashes", None)
    matches = list(_REFERENCE.finditer(answer))
    if not matches:
        return AnswerExtraction(None, "last_hashes", None)
    raw = matches[-1].group(1)
    return AnswerExtraction(normalize_number(raw), "last_hashes", raw)


def _complete_boxed_contents(text: str) -> list[str]:
    contents: list[str] = []
    for match in _BOX_START.finditer(text):
        content_start = match.end()
        depth = 1
        position = content_start
        while position < len(text) and depth:
            char = text[position]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            position += 1
        if depth == 0:
            contents.append(text[content_start : position - 1])
    return contents


def extract_model_answer(text: str) -> AnswerExtraction:
    """Extract only the final complete numeric ``\\boxed{}`` answer."""

    if not isinstance(text, str):
        return AnswerExtraction(None, "last_boxed", None)
    contents = _complete_boxed_contents(text)
    if not contents:
        return AnswerExtraction(None, "last_boxed", None)
    raw = contents[-1]
    return AnswerExtraction(normalize_number(raw), "last_boxed", raw)
