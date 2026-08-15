"""Faithfulness 검증자 (기획안 §8 M2, 델타 §D6.4).

지금 있는 것 / 없는 것
---------------------
    Verifier                 포트 (프로토콜)
    RuleBasedFakeVerifier    키 불필요. **수치 왜곡만** 잡습니다
    LLMVerifier              키 필요. M2 나머지에서 구현

`RuleBasedFakeVerifier`가 잡는 것과 못 잡는 것을 분명히 해 둡니다. 델타 §D6.4는 이걸
"진짜 faithfulness는 아니지만 파이프라인 배선과 게이트 동작을 테스트하기에 충분하고,
나중에 LLM 검증자의 **하한선 베이스라인**으로도 쓰인다"고 규정했습니다.

    ① 수치 조작        → 잡습니다. 초록에 없는 숫자가 요약에 있으면 근거 없음
    ② 없는 인과 주장    → **못 잡습니다.** 규칙으로 인과를 판정할 수 없습니다
    ③ 한계를 성과로 반전 → **못 잡습니다.** 의미 반전은 규칙 밖입니다

기획안 §8 M2 DoD("왜곡 3건을 검증자가 모두 잡아낸다")는 **LLM 검증자로만 판정 가능**
합니다. 왜곡 샘플 3건과 테스트 하네스는 지금 만들어 두었으니(`tests/fixtures/
distorted_summaries.yaml`), 키가 오는 날 구현체만 갈아끼우면 그날 판정됩니다.

②③을 규칙으로 흉내 내지 않은 이유: 제가 왜곡 샘플도 쓰고 규칙도 쓰면, 규칙이 제
샘플을 잡는 건 당연하고 아무것도 증명하지 못합니다. 하한선 베이스라인은 정직할 때만
베이스라인입니다.

알려진 한계 (LLM 검증자로 넘어갈 때 사라지는 것들)
------------------------------------------------
- **표기 차이를 못 넘습니다.** 초록이 `0.768`인데 요약이 `76.8%`면 불일치로 잡습니다.
  요약 프롬프트가 "수치는 초록에 나온 그대로" 를 요구하므로 실무상 드물지만, 이
  검증자의 점수를 절대 신뢰도로 읽으면 안 되는 이유입니다.
- **교차언어**: 영어 초록 ↔ 한국어 요약 조합에서 맨정수 오탐이 나기 쉬워
  `_BARE_INT_MIN` 아래를 버립니다. 그만큼 재현율을 포기한 것입니다.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from src.summarize.schema import PaperSummary, UnsupportedClaim, Verdict

#: 정수·소수·백분율을 잡습니다. 쉼표 자릿수 구분(1,024)도 포함합니다.
_NUMBER_RE = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(%?)")

#: 숫자로 보이지만 사실상 식별자·연도라 비교 의미가 없는 것들.
_YEAR_RE = re.compile(r"(?:19|20)\d{2}")

#: ★ 교차언어 오탐을 막는 하한선.
#:
#: 영어 초록은 작은 수를 **단어로** 씁니다("six object detection models"). 한국어
#: 요약은 같은 내용을 **숫자로** 씁니다("모델 6종"). 그래서 모든 숫자를 대조하면
#: `6`, `2단계`, `YOLOv5`의 `5` 같은 것이 전부 "초록에 없는 수치"로 잡힙니다.
#: 실제로 그렇게 짰다가 정상 요약이 0.3점을 받았습니다.
#:
#: 성능 수치는 거의 항상 **소수이거나, 백분율이거나, 세 자리 이상**입니다.
#: 그 외의 맨정수는 개수·차수·모델 버전이라 검사에서 뺍니다. 대신 "500장으로 학습"
#: 같은 세 자리 이상 조작은 그대로 잡힙니다.
_BARE_INT_MIN = 100


@runtime_checkable
class Verifier(Protocol):
    #: 원장에 기록되는 모델 식별자. 게이트가 생성자와 비교합니다.
    model_id: str

    def verify(self, abstract: str, summary: PaperSummary) -> Verdict: ...


def _numbers(text: str, *, measured_only: bool) -> set[str]:
    """텍스트의 수치를 정규화해 뽑습니다.

    `measured_only=True`  요약 쪽. 성능 수치로 볼 만한 것만 (오탐 방지)
    `measured_only=False` 초록 쪽. 전부 (요약의 수치가 대조될 수 있게)
    """
    found: set[str] = set()
    for raw, percent in _NUMBER_RE.findall(text):
        cleaned = raw.replace(",", "")
        if _YEAR_RE.fullmatch(cleaned):
            continue  # 연도는 성능 수치가 아니라 식별자
        try:
            value = float(cleaned)
        except ValueError:
            continue
        if measured_only:
            is_decimal = "." in cleaned
            if not (is_decimal or percent or value >= _BARE_INT_MIN):
                continue
        # 36.0 과 36 을 같게 봅니다. 표기 차이로 오탐이 나면 하한선 자격을 잃습니다.
        found.add(f"{value:g}")
    return found


class RuleBasedFakeVerifier:
    """수치 근거만 검사하는 규칙 검증자. 키가 필요 없습니다.

    LLM 검증자의 **하한선**입니다. 나중에 M2 평가에서 "LLM 검증자가 규칙 검증자보다
    얼마나 나은가"를 이 구현과 비교해 수치로 낼 수 있습니다.
    """

    model_id = "rulebased_fake/v1"

    def __init__(self, penalty_per_claim: float = 0.35) -> None:
        self.penalty_per_claim = penalty_per_claim

    def verify(self, abstract: str, summary: PaperSummary) -> Verdict:
        source_numbers = _numbers(abstract, measured_only=False)
        claims: list[UnsupportedClaim] = []

        for field_name in ("problem", "method", "key_results", "limitations"):
            value = getattr(summary, field_name)
            sentences = value if isinstance(value, list) else [value]
            for sentence in sentences:
                if not sentence:
                    continue
                unsupported = _numbers(sentence, measured_only=True) - source_numbers
                if unsupported:
                    claims.append(
                        UnsupportedClaim(
                            claim=sentence,
                            field=field_name,
                            type="numeric",
                            why=(
                                f"초록에 없는 수치: {', '.join(sorted(unsupported))}. "
                                "규칙 검증자는 수치만 봅니다"
                            ),
                        )
                    )

        score = max(0.0, 1.0 - self.penalty_per_claim * len(claims))
        return Verdict(
            faithfulness=score,
            unsupported_claims=claims,
            verifier_model=self.model_id,
        )
