"""생성 → 검증 → 재생성 루프 (기획안_2 §6.3).

    생성 → 검증 → faithfulness ≥ 0.7 ?
                    ├ 예   → 발행
                    └ 아니오 → 재생성 (최대 2회)
                                └ 2회 후에도 미달 → flagged=True 로 발행 + 원장 기록

이 모듈이 지키는 계약 4개
------------------------
1. **`assert_verifier_differs` 를 검증 요청 직전에** 부릅니다 (기획안_2 §6.2 / R5).
   설정을 읽은 직후가 아니라 그 모델로 실제 요청을 보내기 직전이어야 합니다. 그래서
   루프 **안**, `verifier.verify()` 바로 위에 있습니다. 재시도마다 다시 부릅니다 —
   호출부가 도중에 검증자를 갈아끼울 수 있고, 게이트는 "함수로 존재"할 뿐 아무도
   부르지 않으면 통과한 적이 없는 것과 같습니다 (R5).
2. **재생성은 같은 프롬프트로 다시 부르지 않습니다** (기획안_2 §6.3). 직전 검증자의
   `unsupported_claims` 를 문장으로 만들어 생성자에게 넘깁니다. 생성자가 그 인자를
   받지 못하면 `RegenerationUnsupportedError` 로 **멈춥니다** — 같은 프롬프트로 조용히
   재호출하면 같은 실수가 그대로 나오고, 원장의 `retries` 는 "고쳐 봤다"는 거짓말이
   됩니다.
3. **임계 미달은 발행을 막지 않습니다. 막는 건 플래그 누락입니다** (기획안_2 §6.3).
   `flagged` 를 임계값에서 직접 유도하고 `assert_faithfulness_flagged` 로 그 계약을
   한 번 더 못박습니다 (publish/markdown.py 와 같은 형태).
4. **매 시도의 faithfulness 를 남깁니다** (기획안_2 §6.3 — "분포가 M2 DoD입니다").
   `VerifiedSummary.attempts` 와 원장 항목의 `attempts` 키입니다.

단가·모델 id 는 이 모듈이 정하지 않습니다. 생성자·검증자 객체가 들고 옵니다.
비용은 **호출 직후마다** 걷습니다 (`VerifiedSummary.cost_usd`) — 아이템 1건이
생성 최대 3회 + 검증 최대 3회이므로, 루프가 끝난 뒤 `generator.last_usage` 를 한 번
읽으면 **마지막 시도분만** 잡혀 재생성한 아이템의 비용이 조용히 줄어듭니다.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from src.core.models import Item
from src.summarize.generator import Generator
from src.summarize.schema import (
    DEFAULT_OUTPUT_LANGUAGE,
    MAX_REGENERATIONS,
    PaperSummary,
    UnsupportedClaim,
    Verdict,
)
from src.verify.faithfulness import Verifier
from src.verify.gates import assert_faithfulness_flagged, assert_verifier_differs


class RegenerationUnsupportedError(RuntimeError):
    """재생성이 필요한데 생성자가 피드백을 받지 못합니다 (기획안_2 §6.3).

    조용히 같은 프롬프트로 재호출하지 않기 위한 예외입니다. 생성자를
    `generate(item, output_language, *, feedback: str | None = None)` 로 확장하면
    해결됩니다 — `src/summarize/generator.py` 의 구현체 둘(`FakeGenerator`,
    `AnthropicGenerator`)은 이미 받습니다. `Generator` 프로토콜의 최소 계약은
    2인자이므로, 그 계약만 만족하는 구현체가 들어오면 여기서 멈춥니다.
    """


@dataclass(frozen=True, slots=True)
class Attempt:
    """시도 1회의 기록. `index=0` 이 최초 생성, 1부터가 재생성입니다."""

    index: int
    faithfulness: float
    unsupported_claims: tuple[UnsupportedClaim, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            # Verdict.to_dict 와 같은 자릿수 — 원장 안에서 표기가 갈리면 안 됩니다.
            "faithfulness": round(self.faithfulness, 4),
            "unsupported_claims": [c.to_dict() for c in self.unsupported_claims],
        }


@dataclass(frozen=True, slots=True)
class VerifiedSummary:
    """루프의 결과. 기획안_2 §6.3 이 요구하는 5가지를 그대로 들고 있습니다.

        summary   최종 요약 (통과했든, 2회 후 미달로 플래그됐든 마지막 것)
        verdict   그 요약에 대한 마지막 판정
        retries   재생성 횟수 (= 시도 수 - 1)
        flagged   임계 미달로 경고 플래그가 붙었는가
        attempts  매 시도의 faithfulness 와 지적 사항
    """

    summary: PaperSummary
    verdict: Verdict
    retries: int
    flagged: bool
    attempts: tuple[Attempt, ...]
    generator_model: str
    #: 이 아이템에 든 비용 합계 (생성 + 검증, 실패한 시도 포함). 단가를 모르는 호출이
    #: 하나라도 있으면 **None** 입니다 — 0.0 으로 적으면 원장의 비용이 거짓이 됩니다
    #: (기획안_2 §12). 사용량을 아예 보고하지 않는 Fake 구현체만 쓴 실행도 None 입니다.
    cost_usd: float | None = None

    @property
    def faithfulness_history(self) -> tuple[float, ...]:
        """시도 순서대로의 점수. 기획안_2 §6.3 의 "분포" 계산용."""
        return tuple(a.faithfulness for a in self.attempts)


def render_regeneration_feedback(claims: Sequence[UnsupportedClaim]) -> str:
    """직전 판정의 지적 사항 → 재생성 프롬프트에 넣을 한국어 지시문.

    기획안_2 §6.3: "재생성 시 직전 검증자의 `unsupported_claims` 를 프롬프트에
    넣으세요. 같은 프롬프트로 다시 부르면 같은 실수가 나옵니다."

    ★ 여기서 만든 문자열은 **그대로 넘깁니다.** 다시 `.format()` 하거나 f-string 에
    넣으면 인용된 초록·요약의 LaTeX 중괄호에서 터집니다 (기획안_2 §9.15).
    """
    lines = [
        "직전 시도의 요약은 검증에서 다음 지적을 받았습니다. 같은 실수를 반복하지 마세요.",
    ]
    if claims:
        for i, claim in enumerate(claims, start=1):
            lines.append(f"{i}. [{claim.field} / {claim.type}] {claim.claim}")
            if claim.why:
                lines.append(f"   → {claim.why}")
    else:
        # 점수는 미달인데 지목된 문장이 없는 경우. 여기서 "재생성 안 함"으로 빠지면
        # §6.3 위반이므로, 근거 없는 확장을 줄이라는 일반 지시라도 반드시 넣습니다.
        lines.append("(검증자가 특정 문장을 지목하지 않았습니다. 점수만 임계 미달입니다.)")
    lines.append(
        "초록에 근거가 없는 내용은 쓰지 말고, 근거를 댈 수 없는 항목은 비우세요. "
        "빈 칸이 틀린 문장보다 낫습니다."
    )
    return "\n".join(lines)


class _CostMeter:
    """호출 **직후** 사용량을 걷습니다 (기획안_2 §12).

    `last_usage` 를 나중에 한 번 읽는 방식은 재생성이 일어난 아이템에서 마지막
    시도분만 잡습니다. 호출 직후에 걷으면 어느 호출의 비용인지가 확정됩니다.

    사용량을 보고하지 않는 구현체(Fake)는 건너뜁니다 — 실제로 돈이 안 들었습니다.
    단가를 모른 채(`cost_usd is None`) 보고한 호출이 하나라도 있으면 합계는
    **모릅니다(None)**. 모르는 것을 0.0 으로 적으면 §12 의 비용 실측이 거짓이 됩니다.
    """

    def __init__(self) -> None:
        self._priced_total = 0.0
        self._reports = 0
        self._unpriced = False

    def collect(self, component: Any) -> None:
        usage = getattr(component, "last_usage", None)
        if usage is None:
            return
        self._reports += 1
        cost = getattr(usage, "cost_usd", None)
        if cost is None:
            self._unpriced = True
        else:
            self._priced_total += float(cost)

    @property
    def cost_usd(self) -> float | None:
        if self._reports == 0 or self._unpriced:
            return None
        return self._priced_total


def _accepts_feedback(generate: Callable[..., Any]) -> bool:
    """생성자가 `feedback` 인자를 받는지 시그니처로 판정합니다.

    `TypeError` 를 잡아서 판정하지 않는 이유: 생성자 **내부**에서 난 TypeError 까지
    "피드백 미지원"으로 삼켜 버립니다.
    """
    try:
        parameters = inspect.signature(generate).parameters
    except (TypeError, ValueError):  # 시그니처를 못 읽는 호출가능 객체
        return False
    if "feedback" in parameters:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())


def _generate(
    generator: Generator, item: Item, output_language: str, feedback: str | None
) -> PaperSummary:
    if feedback is None:
        return generator.generate(item, output_language)
    if not _accepts_feedback(generator.generate):
        raise RegenerationUnsupportedError(
            f"생성자 {generator.model_id!r} 가 feedback 인자를 받지 않습니다. "
            "같은 프롬프트로 다시 부르면 같은 실수가 나옵니다 (기획안_2 §6.3) — "
            "조용히 재호출하지 않고 여기서 멈춥니다. "
            "generate(item, output_language, *, feedback: str | None = None) 로 확장하세요"
        )
    return generator.generate(item, output_language, feedback=feedback)


def summarize_with_verification(
    item: Item,
    generator: Generator,
    verifier: Verifier,
    max_retries: int | None = None,
    *,
    output_language: str | None = None,
) -> VerifiedSummary:
    """아이템 1건을 요약하고, 임계 미달이면 재생성합니다 (기획안_2 §6.3).

    `max_retries` / `output_language` 기본값이 모듈 상수가 아니라 `None` 인 이유는
    기획안_2 §9.1 / R7 입니다 (정의 시점 바인딩 금지). 생략하면 각각
    `schema.MAX_REGENERATIONS`(=2) 와 `schema.DEFAULT_OUTPUT_LANGUAGE` 입니다.

    임계 미달로 끝나도 예외가 아닙니다 — `flagged=True` 로 돌려주고, 발행을 막는 건
    플래그 **누락**뿐입니다 (기획안_2 §6.3).
    """
    retries_allowed = MAX_REGENERATIONS if max_retries is None else int(max_retries)
    if retries_allowed < 0:
        raise ValueError(f"max_retries 는 0 이상이어야 합니다: {retries_allowed}")
    language = DEFAULT_OUTPUT_LANGUAGE if output_language is None else output_language

    attempts: list[Attempt] = []
    feedback: str | None = None
    summary: PaperSummary | None = None
    verdict: Verdict | None = None
    cost = _CostMeter()

    for index in range(retries_allowed + 1):
        summary = _generate(generator, item, language, feedback)
        cost.collect(generator)  # 호출 직후에 걷습니다 — 재생성분이 사라지지 않게

        # ★ 게이트는 **여기**입니다 — 그 모델로 실제 요청을 보내기 직전 (기획안_2 §6.2).
        #   설정 로드 시점에 한 번 부르고 마는 것은 §6.2 가 명시적으로 금지한 형태입니다.
        assert_verifier_differs(generator.model_id, verifier.model_id)
        verdict = verifier.verify(item.abstract, summary)
        cost.collect(verifier)

        attempts.append(
            Attempt(
                index=index,
                faithfulness=verdict.faithfulness,
                unsupported_claims=tuple(verdict.unsupported_claims),
            )
        )
        if verdict.passed:
            break
        feedback = render_regeneration_feedback(verdict.unsupported_claims)

    assert summary is not None and verdict is not None  # 루프는 최소 1회 돕니다

    flagged = not verdict.passed
    # 계약(플래그는 임계값에서 유도된다)을 게이트로 못박습니다 (R5). 발행 단계에서
    # 한 번 더 검사하지만, 미달 요약이 플래그 없이 이 함수를 **떠나지도** 못하게 합니다.
    assert_faithfulness_flagged(verdict, flagged)

    return VerifiedSummary(
        summary=summary,
        verdict=verdict,
        retries=len(attempts) - 1,
        flagged=flagged,
        attempts=tuple(attempts),
        generator_model=generator.model_id,
        cost_usd=cost.cost_usd,
    )


def ledger_summary_entry(
    result: VerifiedSummary,
    *,
    rank_score_stage1: float | None = None,
    rank_score_stage2: float | None = None,
) -> dict[str, Any]:
    """원장 `summaries[]` 항목 1건 (기획안_2 §4.4).

    필드명·순서를 §4.4 그대로 씁니다. 이 dict 를 만드는 곳이 여기 하나여야 필드명이
    갈리지 않습니다 (원장 쓰기 자체는 `ledger.RunRecord` 로만 — R3).

    - `rank_score_stage2` 는 **raw 로짓**입니다. sigmoid 를 씌우면 원장을 사람이 읽을
      수 없습니다 (기획안_2 §9.4). 여기서는 받은 값을 그대로 넣습니다.
    - 랭킹을 돌리지 않은 실행이면 `None` 입니다 — 키 삭제가 아니라 `null` 입니다
      (§4.4 규칙 2).
    - `attempts` 는 §4.4 의 9개 키 **뒤에 덧붙인** 키입니다. §6.3 이 "매 시도의
      faithfulness 를 전부 원장에 남기라"고 요구하는데 9개 키로는 마지막 시도밖에
      남지 않습니다. ledger.py 와 같은 원칙입니다 — 추가는 하되 삭제·개명은 없습니다.
    """
    return {
        "item_id": result.summary.item_id,
        "rank_score_stage1": rank_score_stage1,
        "rank_score_stage2": rank_score_stage2,
        "generator_model": result.generator_model,
        "verifier_model": result.verdict.verifier_model,
        "faithfulness": round(result.verdict.faithfulness, 4),
        "retries": result.retries,
        "unsupported_claims": [c.to_dict() for c in result.verdict.unsupported_claims],
        "flagged": result.flagged,
        "attempts": [a.to_dict() for a in result.attempts],
    }
