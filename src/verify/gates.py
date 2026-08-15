"""발행 전 하드 게이트 (기획안 §10, 델타 §D4·§D8).

게이트는 **조용히 넘어가지 않습니다.** 실패하면 예외이고, Actions는 빨간불로 남습니다.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from src.summarize.schema import FAITHFULNESS_THRESHOLD, Verdict


class GateViolation(RuntimeError):
    """게이트 위반. 발행을 막습니다."""


class ScopeViolation(GateViolation):
    """비공개 아이템이 공개 싱크에 도달했습니다 (델타 §D4)."""


#: 모델 ID에서 "계열"을 뽑는 규칙. `anthropic/claude-...`, `claude-opus-...`,
#: `openai/gpt-...` 처럼 제공사·제품명이 앞에 오는 관례를 씁니다.
_FAMILY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("anthropic", re.compile(r"\b(anthropic|claude)\b", re.I)),
    ("openai", re.compile(r"\b(openai|gpt-|o[1-9]-)", re.I)),
    ("google", re.compile(r"\b(google|gemini|gemma)\b", re.I)),
    ("meta", re.compile(r"\b(meta|llama)\b", re.I)),
    ("mistral", re.compile(r"\b(mistral|mixtral)\b", re.I)),
    ("qwen", re.compile(r"\b(qwen)\b", re.I)),
    ("deepseek", re.compile(r"\b(deepseek)\b", re.I)),
)


def model_family(model_id: str) -> str:
    """모델 ID에서 계열 이름을 뽑습니다. 못 알아보면 ID 전체를 계열로 봅니다."""
    for family, pattern in _FAMILY_PATTERNS:
        if pattern.search(model_id):
            return family
    return model_id.strip().lower()


def assert_verifier_differs(generator_model: str, verifier_model: str) -> None:
    """★ 검증자가 생성자와 같은 계열이면 실패 (기획안 §10 `test_verifier_differs`).

    동일가족 편향 때문에 검증이 성립하지 않습니다. 취향이 아니라 이전 프로젝트(AdNova)
    에서 실측한 결과입니다.

    **ID가 다른 것만으로는 부족합니다.** `claude-opus-5`로 생성하고 `claude-haiku-4-5`로
    검증하면 ID는 다르지만 같은 계열이라 편향이 그대로입니다. 계열로 비교합니다.
    """
    if not generator_model or not verifier_model:
        raise GateViolation(
            f"모델 ID가 비어 있습니다: generator={generator_model!r}, verifier={verifier_model!r}"
        )
    if generator_model.strip() == verifier_model.strip():
        raise GateViolation(f"검증자가 생성자와 동일합니다: {generator_model}")

    generator_family = model_family(generator_model)
    verifier_family = model_family(verifier_model)
    if generator_family == verifier_family:
        raise GateViolation(
            f"검증자가 생성자와 같은 계열입니다 ({generator_family}): "
            f"{generator_model} / {verifier_model}. "
            "동일가족 편향으로 검증이 성립하지 않습니다 (기획안 §10)"
        )


def assert_faithfulness_flagged(verdict: Verdict, flagged: bool) -> None:
    """★ 임계 미달 요약이 **플래그 없이** 발행되면 실패 (기획안 §10).

    임계 미달 자체는 발행을 막지 않습니다 — 기획안 §8 M2는 재생성 2회 후에도 미달이면
    "경고 플래그와 함께 발행하고 원장에 기록"하라고 했습니다. 막는 건 **플래그 누락**입니다.
    """
    if verdict.faithfulness < FAITHFULNESS_THRESHOLD and not flagged:
        raise GateViolation(
            f"faithfulness {verdict.faithfulness:.2f} < {FAITHFULNESS_THRESHOLD} 인데 "
            "경고 플래그가 없습니다"
        )


def assert_public_scope(items: Iterable[Any]) -> None:
    """★ 공개 싱크에 비공개 아이템이 도달하면 실패 (델타 §D4 `test_publish_scope`).

    채용공고가 공개 사이트에 올라가면 사람인 약관 조항 6(제휴 관계 오인)에 걸립니다.
    무료여도 걸립니다. 설정으로 우회할 수 없도록 어댑터가 scope를 리터럴로 고정하고,
    싱크가 여기서 다시 확인합니다.
    """
    leaked = [getattr(i, "id", repr(i)) for i in items if getattr(i, "publish_scope", None) != "public"]
    if leaked:
        raise ScopeViolation(f"비공개 아이템이 공개 싱크에 도달했습니다: {leaked}")
