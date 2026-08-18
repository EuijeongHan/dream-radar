"""프로파일 관심사 → 질의 벡터 · 점수 합성 (기획안_2 §5.2).

점수 합성은 M1의 실질적인 설계 결정입니다. 문서 하나에 대해 관심사 15개의 코사인이
나오는데, 이걸 **어떻게 하나로 접느냐**가 무엇이 상위에 오는지를 결정합니다.

| 방식 | 계산 | 무엇을 왜곡하는가 |
|:--|:--|:--|
| **가중 최대**(기본) | `max_i(cos(d, q_i) × w_i)` | 한 관심사에 강하게 맞는 논문이 이깁니다 |
| 가중 평균 | `Σ(cos × w) / Σw` | 모든 관심사에 고르게 애매한 논문이 이깁니다 |

기본값이 가중 최대인 근거: 관심사 15개는 서로 독립된 주제이고 "diffusion 편집" 논문이
"RAG 평가"와 유사할 이유가 없습니다. 평균을 쓰면 15개 중 14개의 낮은 유사도가 신호를
지웁니다. **두 방식을 다 잴 수 있게 만든 것은 의도입니다** — M1에서 조건 하나를 더
돌려 "왜 최대를 골랐나"에 수치로 답하기 위함입니다 (§5.2).

`exclude.soft` 는 **감점이지 하드 배제가 아닙니다**:

    penalty = max_j(cos(d, e_j) × p_j)          # 가장 강하게 걸리는 것 하나
    score   = base_score - penalty_weight × penalty

★ 하드 배제로 바꾸지 마세요 (§5.2 경고). 경계 사례를 통째로 버리면 골드셋에 관련으로
  라벨된 논문이 후보에서 사라져 **Hit@5의 상한 자체가 내려갑니다.** 예를 들어
  프로파일의 `survey`(penalty 0.3)는 "완전 배제하지 않음 — 분야 입문에 유용"이라고
  프로파일 주석이 명시하고 있습니다.

프로파일은 반드시 `src.core.config.load_profile()` 로 읽습니다 (재사용 규칙 R1).
경로를 문자열로 쓰면 `profile.papers_2.yaml` 이 생기는 순간 구버전을 읽고,
**테스트는 통과하고 결과만 틀립니다** (§9.9).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.core.config import load_profile
from src.rank.embed import Embedder, Vector, cosine

#: 점수 합성 방식 (§5.2). 상위 k 평균은 M1 범위 밖입니다 — 조건이 늘기만 하고
#: "왜 최대인가"에 답하는 데 필요하지 않습니다.
SCORING_METHODS: tuple[str, ...] = ("weighted_max", "weighted_mean")
DEFAULT_SCORING_METHOD = "weighted_max"

#: `exclude.soft` 감점 계수. **M1에서 확정할 값입니다** (§5.2 — "penalty_weight는
#: M1에서 결정"). 1.0은 "감점을 그대로 반영"이라는 중립 출발점이고, 프로파일의
#: `selection.penalty_weight` 가 있으면 그쪽이 우선합니다.
DEFAULT_PENALTY_WEIGHT = 1.0

#: 논문 채널 프로파일의 stem. **경로가 아니라 stem 입니다** (R1 / §4.5).
DEFAULT_PROFILE_STEM = "profile.papers"


@dataclass(frozen=True, slots=True)
class Interest:
    """관심사 1건. `label` 은 사람이 읽는 필드라 **임베딩하지 않습니다**.

    프로파일 주석이 명시한 규칙입니다: "label은 사람이 읽기 위한 필드이므로
    임베딩하지 마세요." label 을 붙여 임베딩하면 한국어 label 이 영어 초록과의
    정렬을 흐려 교차언어 대조 실험(조건 4)의 의미가 사라집니다.
    """

    id: str
    text: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError(f"관심사 {self.id!r} 의 text 가 비어 있습니다")
        if self.weight < 0:
            raise ValueError(f"관심사 {self.id!r} 의 weight 가 음수입니다: {self.weight}")


@dataclass(frozen=True, slots=True)
class Exclusion:
    """`exclude.soft` 1건. `penalty` 는 감점의 세기이지 배제 여부가 아닙니다."""

    text: str
    penalty: float = 1.0

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("exclude.soft 의 text 가 비어 있습니다")
        if self.penalty < 0:
            raise ValueError(f"exclude.soft penalty 가 음수입니다: {self.penalty}")


@dataclass(frozen=True, slots=True)
class ScoreBreakdown:
    """점수 하나와 **그 점수가 어떻게 나왔는지**.

    `best_interest` 를 같이 들고 다니는 것이 핵심입니다. 2차 랭킹(cross-encoder)의
    질의로 쓰이고(§5.1), 사람이 결과를 볼 때 "왜 이게 뽑혔나"에 답합니다.
    """

    score: float
    base: float
    penalty: float
    best_interest: str | None
    best_exclusion: str | None
    method: str


@dataclass(frozen=True, slots=True)
class QuerySet:
    """관심사·배제어의 텍스트와 벡터를 한 묶음으로.

    `profile_path` 는 **원장 기록용**입니다. 어느 프로파일로 돈 결과인지 남기지 않으면
    평가가 재현 불가입니다 (§4.4 규칙 4 / §4.5).
    """

    interests: tuple[Interest, ...]
    interest_vectors: tuple[Vector, ...]
    exclusions: tuple[Exclusion, ...] = ()
    exclusion_vectors: tuple[Vector, ...] = ()
    # 작업규약 §4.4-①의 grep 에 걸리지만 함정 9.1이 아닙니다: 경로가 아니라 스칼라이고,
    # 운영 경로에서는 `build_queryset` 이 `resolve_penalty_weight()` 로 **호출 시점에**
    # 값을 정해 넘깁니다. 이 기본값은 QuerySet 을 직접 만드는 경우의 안전값입니다.
    penalty_weight: float = DEFAULT_PENALTY_WEIGHT
    profile_path: str = ""
    model_id: str = ""
    revision: str = ""

    def __post_init__(self) -> None:
        if not self.interests:
            raise ValueError("관심사가 하나도 없습니다 — 프로파일을 확인하세요")
        if len(self.interests) != len(self.interest_vectors):
            raise ValueError(
                f"관심사 {len(self.interests)}건에 벡터 {len(self.interest_vectors)}건 — "
                "임베더가 순서를 바꿨거나 건수를 흘렸습니다"
            )
        if len(self.exclusions) != len(self.exclusion_vectors):
            raise ValueError(
                f"exclude {len(self.exclusions)}건에 벡터 {len(self.exclusion_vectors)}건"
            )

    def interest_text(self, interest_id: str) -> str:
        for interest in self.interests:
            if interest.id == interest_id:
                return interest.text
        raise KeyError(f"관심사를 찾을 수 없습니다: {interest_id!r}")


def parse_interests(profile: Mapping[str, Any]) -> tuple[Interest, ...]:
    """`interests:` 를 읽습니다. id 가 없으면 순번으로 채웁니다."""
    raw = profile.get("interests")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("프로파일에 interests 목록이 없습니다")
    interests: list[Interest] = []
    for index, entry in enumerate(raw):
        if not isinstance(entry, Mapping):
            raise ValueError(f"interests[{index}] 가 매핑이 아닙니다")
        interests.append(
            Interest(
                id=str(entry.get("id") or f"interest_{index}"),
                text=str(entry.get("text") or ""),
                weight=float(entry.get("weight", 1.0)),
            )
        )
    if not interests:
        raise ValueError("프로파일의 interests 가 비어 있습니다")
    return tuple(interests)


def parse_exclusions(profile: Mapping[str, Any]) -> tuple[Exclusion, ...]:
    """`exclude.soft:` 를 읽습니다. 없으면 빈 튜플입니다.

    `exclude.hard` 는 **읽지 않습니다.** 프로파일에도 없고, 있어서도 안 됩니다 (§5.2).
    """
    block = profile.get("exclude") or {}
    if not isinstance(block, Mapping):
        raise ValueError("exclude 가 매핑이 아닙니다")
    raw = block.get("soft") or []
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("exclude.soft 가 목록이 아닙니다")
    return tuple(
        Exclusion(text=str(entry.get("text") or ""), penalty=float(entry.get("penalty", 1.0)))
        for entry in raw
        if isinstance(entry, Mapping)
    )


def resolve_penalty_weight(
    profile: Mapping[str, Any] | None = None, penalty_weight: float | None = None
) -> float:
    """인자 > 프로파일 `selection.penalty_weight` > 모듈 기본값 순서로 해석합니다."""
    if penalty_weight is not None:
        return float(penalty_weight)
    if profile:
        selection = profile.get("selection") or {}
        if isinstance(selection, Mapping) and selection.get("penalty_weight") is not None:
            return float(selection["penalty_weight"])
    return DEFAULT_PENALTY_WEIGHT


def build_queryset(
    embedder: Embedder,
    *,
    profile: Mapping[str, Any] | None = None,
    profile_path: Path | str | None = None,
    stem: str | None = None,
    root: Path | str | None = None,
    penalty_weight: float | None = None,
) -> QuerySet:
    """프로파일 → 질의 벡터 목록 + 가중치.

    `profile` 을 안 주면 **`load_profile()` 로 최신 번호를 해석**합니다 (R1 / §4.5).
    경로를 하드코딩하지 않는 것이 규칙이고, 실제로 읽은 경로를 `QuerySet.profile_path`
    에 남겨 원장·`results.md` 가 재현 가능하게 합니다.

    관심사와 배제어를 **한 번의 `encode()` 로** 넘깁니다. 캐시 계층이 미스만 골라
    내부 임베더에 보내므로, 호출을 쪼개면 배치만 작아지고 얻는 게 없습니다.
    """
    resolved_path = "" if profile_path is None else str(profile_path)
    if profile is None:
        # ★ R1 — 경로 하드코딩 금지. stem 으로 최고 번호를 해석합니다.
        loaded_stem = stem or DEFAULT_PROFILE_STEM
        if root is None:
            profile, path = load_profile(loaded_stem)
        else:
            profile, path = load_profile(loaded_stem, root=root)
        resolved_path = str(path)

    interests = parse_interests(profile)
    exclusions = parse_exclusions(profile)

    # label 이 아니라 text 만 임베딩합니다 (Interest 독스트링 참조).
    texts = [interest.text for interest in interests] + [item.text for item in exclusions]
    vectors = embedder.encode(texts)
    if len(vectors) != len(texts):
        raise RuntimeError(
            f"질의 {len(texts)}건에 벡터 {len(vectors)}건 — 임베더가 건수를 흘렸습니다"
        )

    split = len(interests)
    return QuerySet(
        interests=interests,
        interest_vectors=tuple(vectors[:split]),
        exclusions=exclusions,
        exclusion_vectors=tuple(vectors[split:]),
        penalty_weight=resolve_penalty_weight(profile, penalty_weight),
        profile_path=resolved_path,
        model_id=getattr(embedder, "model_id", ""),
        revision=getattr(embedder, "revision", ""),
    )


def score_document(
    doc_vector: Sequence[float],
    queryset: QuerySet,
    *,
    method: str | None = None,
    penalty_weight: float | None = None,
) -> ScoreBreakdown:
    """문서 벡터 1건 → 점수 (§5.2).

    `method`·`penalty_weight` 기본값은 **호출 시점에** 해석합니다 (§9.1의 정신 —
    정의 시점 바인딩은 monkeypatch 를 무력화합니다).
    """
    resolved_method = method or DEFAULT_SCORING_METHOD
    if resolved_method not in SCORING_METHODS:
        raise ValueError(f"알 수 없는 점수 합성 방식: {resolved_method!r} (가능: {SCORING_METHODS})")
    weight = queryset.penalty_weight if penalty_weight is None else float(penalty_weight)

    base = 0.0
    best_interest: str | None = None
    if resolved_method == "weighted_max":
        best = None
        for interest, vector in zip(queryset.interests, queryset.interest_vectors):
            value = cosine(doc_vector, vector) * interest.weight
            if best is None or value > best:
                best = value
                best_interest = interest.id
        base = 0.0 if best is None else best
    else:  # weighted_mean — 비교용 조건 (§5.2). 관심사 15개라 희석이 심합니다
        total_weight = sum(interest.weight for interest in queryset.interests)
        if total_weight <= 0:
            raise ValueError("가중치 합이 0입니다 — 가중 평균을 계산할 수 없습니다")
        accumulated = 0.0
        best_value = None
        for interest, vector in zip(queryset.interests, queryset.interest_vectors):
            similarity = cosine(doc_vector, vector)
            accumulated += similarity * interest.weight
            scaled = similarity * interest.weight
            if best_value is None or scaled > best_value:
                best_value = scaled
                best_interest = interest.id  # 2차 랭킹 질의는 합성 방식과 무관하게 최고 관심사
        base = accumulated / total_weight

    strongest: float | None = None
    best_exclusion: str | None = None
    for exclusion, vector in zip(queryset.exclusions, queryset.exclusion_vectors):
        value = cosine(doc_vector, vector) * exclusion.penalty
        if strongest is None or value > strongest:
            strongest = value
            best_exclusion = exclusion.text
    # ★ 감점은 0 아래로 내려가지 않습니다. 코사인이 음수인 배제어를 그대로 빼면
    #   **감점이 가점으로 뒤집힙니다** — "로봇공학과 정반대인 논문"에 보너스를 주는
    #   꼴이고, 아무도 눈치채지 못합니다.
    #   (누적 변수를 0.0에서 시작해 우연히 막는 방식은 쓰지 않습니다. 그러면 클램프를
    #    지워도 테스트가 통과해 이 계약이 검사되지 않습니다 — 작업규약 §4.2로 실측.)
    penalty = max(0.0, strongest if strongest is not None else 0.0)

    return ScoreBreakdown(
        score=base - weight * penalty,
        base=base,
        penalty=penalty,
        best_interest=best_interest,
        best_exclusion=best_exclusion if penalty > 0 else None,
        method=resolved_method,
    )


def score_documents(
    doc_vectors: Sequence[Sequence[float]],
    queryset: QuerySet,
    *,
    method: str | None = None,
    penalty_weight: float | None = None,
) -> list[ScoreBreakdown]:
    return [
        score_document(vector, queryset, method=method, penalty_weight=penalty_weight)
        for vector in doc_vectors
    ]
