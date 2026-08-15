"""생성자·검증자 출력 스키마 — **확정본** (델타 §D6.4).

키가 도착하기 전에 지금 고정하는 이유: 나중에 바꾸면 `runs.jsonl` 원장이 깨지고,
이미 기록된 실행과 비교가 불가능해집니다. 키를 기다리는 동안 정할 수 있는 건 정합니다.

`runs.jsonl`의 `summaries[]` 항목(기획안 §7)과 필드명을 맞춰 둡니다:

    {"item_id", "rank_score_stage1", "rank_score_stage2",
     "generator_model", "verifier_model", "faithfulness", "retries",
     "unsupported_claims"}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: faithfulness 채점 대상 필드. ★ `connection`이 빠져 있는 게 핵심입니다.
#:
#: `connection`("내 프로젝트와의 연결점")은 독자의 맥락에서 나오는 판단이라
#: **초록에 근거가 있을 수 없습니다.** 채점에 포함하면 모든 요약이 "근거 없는 주장"으로
#: 감점되어 지표가 무의미해집니다. 이걸 코드로 고정해 두지 않으면, 나중에 누가
#: 요약 전체를 검증자에게 넘기고 "왜 점수가 다 낮지"로 시간을 씁니다.
VERIFIABLE_FIELDS: tuple[str, ...] = ("problem", "method", "key_results", "limitations")

ALL_FIELDS: tuple[str, ...] = (*VERIFIABLE_FIELDS, "connection")

#: 기획안 §8 M2 — 임계 미달이면 재생성, 최대 2회.
FAITHFULNESS_THRESHOLD = 0.7
MAX_REGENERATIONS = 2

#: 요약 출력 언어. 기본값은 한국어(독자가 본인)이지만, **초록이 영어라 검증이
#: 교차언어가 됩니다.** 교차언어 판정은 동일언어 대비 약해서, faithfulness가 낮게
#: 나와도 요약이 나쁜 건지 검증이 약한 건지 구분되지 않습니다.
#: M2에서 같은 초록에 한국어/영어 요약을 각각 생성해 비교하면 이 교란을 분리할 수
#: 있습니다 — M1의 영/한 프로파일 비교와 같은 구조의 실험입니다.
#: prompts/paper.md 의 `{output_language}` 참조.
OUTPUT_LANGUAGES: tuple[str, ...] = ("ko", "en")
DEFAULT_OUTPUT_LANGUAGE = "ko"

#: 검증자가 붙이는 왜곡 유형. 프롬프트의 목록과 같아야 합니다.
DISTORTION_TYPES: tuple[str, ...] = (
    "numeric",
    "causal",
    "reversal",
    "overreach",
    "fabrication",
)


@dataclass
class PaperSummary:
    """생성자 출력. 기획안 §8 M2의 5개 항목."""

    item_id: str
    problem: str
    method: str
    key_results: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    connection: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "problem": self.problem,
            "method": self.method,
            "key_results": list(self.key_results),
            "limitations": list(self.limitations),
            "connection": self.connection,
        }

    def verifiable_text(self) -> str:
        """검증자에게 넘길 텍스트. `connection`은 제외됩니다."""
        parts: list[str] = []
        for name in VERIFIABLE_FIELDS:
            value = getattr(self, name)
            if isinstance(value, list):
                parts.extend(str(v) for v in value)
            elif value:
                parts.append(str(value))
        return "\n".join(parts)

    def verifiable_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in VERIFIABLE_FIELDS}

    @classmethod
    def from_dict(cls, data: dict[str, Any], item_id: str | None = None) -> PaperSummary:
        missing = [f for f in ("problem", "method") if not data.get(f)]
        if missing:
            raise ValueError(f"요약에 필수 필드가 비어 있습니다: {missing}")
        return cls(
            item_id=item_id or data["item_id"],
            problem=data["problem"],
            method=data["method"],
            key_results=list(data.get("key_results") or []),
            limitations=list(data.get("limitations") or []),
            connection=data.get("connection", ""),
        )


@dataclass
class UnsupportedClaim:
    claim: str
    field: str
    type: str
    why: str

    def __post_init__(self) -> None:
        if self.type not in DISTORTION_TYPES:
            raise ValueError(f"알 수 없는 왜곡 유형: {self.type!r} (허용: {list(DISTORTION_TYPES)})")

    def to_dict(self) -> dict[str, Any]:
        return {"claim": self.claim, "field": self.field, "type": self.type, "why": self.why}


@dataclass
class Verdict:
    """검증자 출력. `runs.jsonl`의 `summaries[]`로 그대로 들어갑니다."""

    faithfulness: float
    unsupported_claims: list[UnsupportedClaim] = field(default_factory=list)
    verifier_model: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.faithfulness <= 1.0:
            raise ValueError(f"faithfulness 는 0.0~1.0 이어야 합니다: {self.faithfulness}")

    @property
    def passed(self) -> bool:
        return self.faithfulness >= FAITHFULNESS_THRESHOLD

    def to_dict(self) -> dict[str, Any]:
        return {
            "faithfulness": round(self.faithfulness, 4),
            "unsupported_claims": [c.to_dict() for c in self.unsupported_claims],
            "verifier_model": self.verifier_model,
        }
