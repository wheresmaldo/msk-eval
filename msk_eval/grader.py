"""The AI grader ("LLM judge").

One call grades every rubric item for one answer, which keeps cost down.
The grader must return JSON; anything it skips or garbles is treated as
"not met, low confidence" so a human looks at it rather than it passing silently.
"""
from __future__ import annotations

import json
import re

from .providers import Provider
from .schema import Case, ItemGrade, Usage

ANSWER_PROMPT = (
    "A person sends this message to a general-purpose AI assistant. "
    "Reply to them directly, as a helpful assistant would, in under 150 words of plain text.\n\n"
    "Message:\n{prompt}"
)

GRADER_PROMPT = """You are grading an AI assistant's answer to a patient's musculoskeletal question.
An orthopedic clinician wrote the rubric. Judge only from the answer text. Be strict:
an item is met only if the answer clearly does what the item says.

Patient message:
<<<
{prompt}
>>>

AI answer:
<<<
{answer}
>>>

Rubric items:
{items}

Reply with only a JSON array, one object per rubric item, in this shape:
[{{"id": "r1", "met": true, "confidence": "high", "rationale": "under 20 words"}}]
Use confidence "low" when a clinician should double-check."""


def answer_prompt(prompt: str) -> str:
    return ANSWER_PROMPT.format(prompt=prompt)


def grader_prompt(case: Case, prompt: str, answer: str) -> str:
    items = "\n".join(f"{i.id}{' (MUST-PASS)' if i.must_pass else ''}: {i.text}" for i in case.rubric)
    return GRADER_PROMPT.format(prompt=prompt, answer=answer, items=items)


def parse_grades(case: Case, raw: str) -> list[ItemGrade]:
    data = _extract_json_array(raw)
    seen: dict[str, ItemGrade] = {}
    valid_ids = {i.id for i in case.rubric}
    for obj in data:
        if not isinstance(obj, dict) or str(obj.get("id")) not in valid_ids:
            continue
        item_id = str(obj["id"])
        seen[item_id] = ItemGrade(
            item_id=item_id,
            met=obj.get("met") is True,
            confidence="low" if str(obj.get("confidence", "high")).lower() == "low" else "high",
            rationale=str(obj.get("rationale", ""))[:300],
        )
    for item in case.rubric:
        if item.id not in seen:
            seen[item.id] = ItemGrade(item.id, False, "low", "Grader returned no grade for this item.")
    return [seen[i.id] for i in case.rubric]


def grade(provider: Provider, judge_model: str, case: Case, prompt: str, answer: str) -> tuple[list[ItemGrade], Usage]:
    raw, usage = provider.complete(judge_model, grader_prompt(case, prompt, answer))
    return parse_grades(case, raw), usage


def _extract_json_array(raw: str) -> list:
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []
