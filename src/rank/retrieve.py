"""1차 랭킹 — 후보 임베딩 → 점수 → `stage1_top_n` (기획안_2 §5.1 · §5.4 · §5.7).

여기서 정하는 것은 두 가지뿐입니다: **무엇을 임베딩할 것인가**와 **동점을 어떻게
깰 것인가.** 둘 다 조용히 틀리는 종류입니다.

동점 처리 (§5.4)
----------------
점수가 같으면 **`item_id` 사전순**입니다. 정렬이 불안정하면 같은 조건을 두 번 돌렸을
때 수치가 달라지고, M1 DoD의 "같은 명령을 두 번 돌려 같은 수치"(§5.8)가 깨집니다.
파이썬의 `sorted` 는 안정 정렬이라 입력 순서를 보존하는데, **후보 파일의 줄 순서는
수집 순서**라 조건마다 달라질 수 있습니다. 그래서 점수만이 아니라 id 까지 키에 넣습니다.

`min_score_threshold` (§5.7)
----------------------------
★ 프로파일의 `0.42` 는 **잠정치이고, 실측상 거의 전부를 통과시킵니다.**
  bge-m3 의 코사인은 무관한 텍스트끼리도 0.5~0.6 이 나옵니다(다국어 모델의 이방성).
  즉 0.42 는 필터로서 아무 일도 하지 않습니다. M1에서 골드셋의 정답/오답 점수 분포를
  그리고 **정답의 95%를 살리는 지점**으로 다시 잡은 뒤, 그 근거를 `eval/results.md`
  에 남기세요. 여기서는 설정을 읽어 적용만 하고 값을 코드에 박지 않습니다.

  주의: 감점(§5.2)이 들어간 뒤의 점수에 임계값을 겁니다. base 에 걸면 배제 대상이
  임계값을 넘어 들어옵니다.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.rank.embed import Embedder
from src.rank.profile import QuerySet, ScoreBreakdown, score_document

#: 프로파일 `selection:` 이 없을 때의 기본값 (§5.1 / profile.papers_1.yaml).
#: **함수 기본 인자에 직접 쓰지 마세요** — 호출 시점에 해석합니다 (§9.1 / R7).
DEFAULT_STAGE1_TOP_N = 30
DEFAULT_FINAL_N = 5
#: 기본은 "임계값 없음" 입니다. 잠정치 0.42 를 코드 기본값으로 박으면, 프로파일을
#: 안 준 호출이 조용히 잠정치로 걸러집니다 (§5.7).
DEFAULT_MIN_SCORE_THRESHOLD: float | None = None


class _Unset:
    """'인자를 주지 않았음' 을 나타내는 센티널."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - 디버깅 표시용
        return "UNSET"


#: ★ `None` 과 "미지정" 을 구분합니다.
#:
#:   min_score=UNSET  → 프로파일/기본값을 따른다
#:   min_score=None   → **임계값을 끈다** (명시적)
#:   min_score=0.42   → 그 값을 쓴다
#:
#: 둘을 `None` 하나로 뭉치면 "프로파일에 0.42가 있는데 이번 조건만 임계값 없이
#: 돌려보고 싶다"를 표현할 방법이 없습니다. §5.7이 요구하는 분포 관찰이 바로
#: 그 실행입니다 — 임계값을 끄고 정답/오답 점수 분포를 봐야 값을 정할 수 있습니다.
#:
#: 참고: 작업규약 §4.4-①의 grep("기본 인자에 모듈 상수")에 `= UNSET` 이 걸립니다.
#: **함정 9.1이 아닙니다.** 9.1은 "경로/설정 상수가 정의 시점에 박혀 monkeypatch가
#: 무효가 되는 것"이고, UNSET 은 값이 아니라 "인자 없음" 표식인 불변 센티널입니다
#: (아무도 갈아끼우지 않으며, 갈아끼우면 의미가 사라집니다). 실제 경로·설정 해석은
#: 전부 `stage1_settings()` 안에서 **호출 시점에** 일어납니다.
UNSET = _Unset()


@dataclass(frozen=True, slots=True)
class Stage1Settings:
    top_n: int | None
    final_n: int
    min_score: float | None


@dataclass(frozen=True, slots=True)
class RankedItem:
    """점수가 붙은 후보 1건.

    `Item` 은 `frozen=True` 라 점수를 써넣을 수 없습니다 (인벤토리 §2.2). 그래서
    점수를 별도 구조체로 들고 다닙니다 — 원본은 `item` 필드에 그대로 둡니다.
    """

    item_id: str
    score: float
    base: float
    penalty: float
    best_interest: str | None
    rank: int
    item: Any

    @property
    def title(self) -> str:
        return str(get_field(self.item, "title") or "")


def get_field(item: Any, key: str) -> Any:
    """`Item` 객체와 후보 JSONL 의 dict 를 **둘 다** 받습니다.

    운영 파이프라인은 `Item` 을 들고 있고, 평가(`eval/run_eval.py`)는 후보 파일에서
    읽은 dict 를 다룹니다. 둘을 위해 랭킹을 두 벌 만들 이유가 없습니다.
    """
    if isinstance(item, Mapping):
        return item.get(key)
    return getattr(item, key, None)


def document_text(item: Any) -> str:
    """임베딩 입력 = 제목 + 초록.

    제목만 쓰면 신호가 짧고, 초록만 쓰면 제목의 강한 단서를 버립니다.
    ★ 여기서 `.format()` 이나 f-string 재치환을 하지 마세요 — 초록의 10%에 LaTeX
      중괄호가 있어 `KeyError: '6π'` 로 터집니다 (§9.15). 문자열을 이어붙이기만 합니다.
    """
    title = str(get_field(item, "title") or "").strip()
    abstract = str(get_field(item, "abstract") or "").strip()
    if title and abstract:
        return f"{title}\n\n{abstract}"
    return title or abstract


def item_id_of(item: Any) -> str:
    item_id = get_field(item, "id")
    if not item_id:
        raise ValueError(f"후보에 id 가 없습니다: {item!r}")
    return str(item_id)


def stage1_settings(
    profile: Mapping[str, Any] | None = None,
    *,
    top_n: int | None | _Unset = UNSET,
    final_n: int | _Unset = UNSET,
    min_score: float | None | _Unset = UNSET,
) -> Stage1Settings:
    """프로파일 `selection:` 에서 읽고, 인자를 준 항목은 인자가 이깁니다.

    프로파일 실값 (profile.papers_1.yaml): `stage1_top_n: 30`, `final_n: 5`,
    `min_score_threshold: 0.42`(잠정치 — §5.7 경고 참조).

    `UNSET` 과 `None` 의 차이는 위 상수 주석을 보세요. `None` 은 "끈다" 입니다.
    """
    selection: Mapping[str, Any] = {}
    if profile:
        candidate = profile.get("selection")
        if isinstance(candidate, Mapping):
            selection = candidate

    resolved_top_n = selection.get("stage1_top_n") if isinstance(top_n, _Unset) else top_n
    resolved_final_n = selection.get("final_n") if isinstance(final_n, _Unset) else final_n
    resolved_min = (
        selection.get("min_score_threshold") if isinstance(min_score, _Unset) else min_score
    )
    if isinstance(top_n, _Unset) and resolved_top_n is None:
        resolved_top_n = DEFAULT_STAGE1_TOP_N
    if resolved_final_n is None:
        resolved_final_n = DEFAULT_FINAL_N
    if isinstance(min_score, _Unset) and resolved_min is None:
        resolved_min = DEFAULT_MIN_SCORE_THRESHOLD

    return Stage1Settings(
        top_n=int(resolved_top_n) if resolved_top_n is not None else None,
        final_n=int(resolved_final_n),
        min_score=float(resolved_min) if resolved_min is not None else None,
    )


def sort_ranked(rows: Iterable[tuple[str, ScoreBreakdown, Any]]) -> list[RankedItem]:
    """★ 정렬 계약: `(-score, item_id)` (§5.4).

    `rank` 는 1부터 시작하는 연속 번호입니다. 임계값으로 거른 뒤에 다시 매기지 않고,
    **자른 뒤에도 원래 순위를 유지**합니다 — 상위 30건의 rank 가 1..30 인 것과
    "전체에서 몇 등이었나"가 같아야 MRR 계산이 헷갈리지 않습니다.
    """
    ordered = sorted(rows, key=lambda row: (-row[1].score, row[0]))
    return [
        RankedItem(
            item_id=item_id,
            score=breakdown.score,
            base=breakdown.base,
            penalty=breakdown.penalty,
            best_interest=breakdown.best_interest,
            rank=index,
            item=item,
        )
        for index, (item_id, breakdown, item) in enumerate(ordered, start=1)
    ]


def score_items(
    items: Sequence[Any],
    embedder: Embedder,
    queryset: QuerySet,
    *,
    method: str | None = None,
    penalty_weight: float | None = None,
) -> list[RankedItem]:
    """후보 전체를 점수순으로 정렬해 돌려줍니다 (자르지 않습니다).

    임베딩은 **한 번의 `encode()`** 로 넘깁니다 — 캐시 계층이 미스만 골라 실모델에
    보내므로 배치가 클수록 이득이고, 719건을 한 건씩 부르면 배치가 무의미해집니다.
    """
    items = list(items)
    if not items:
        return []
    texts = [document_text(item) for item in items]
    vectors = embedder.encode(texts)
    if len(vectors) != len(items):
        raise RuntimeError(
            f"후보 {len(items)}건에 벡터 {len(vectors)}건 — 문서와 벡터가 어긋납니다"
        )
    rows = [
        (
            item_id_of(item),
            score_document(vector, queryset, method=method, penalty_weight=penalty_weight),
            item,
        )
        for item, vector in zip(items, vectors)
    ]
    return sort_ranked(rows)


def select_stage1(
    ranked: Sequence[RankedItem],
    settings: Stage1Settings | None = None,
    *,
    top_n: int | None | _Unset = UNSET,
    min_score: float | None | _Unset = UNSET,
) -> list[RankedItem]:
    """임계값으로 거른 뒤 상위 N 을 남깁니다 (§5.1 — 1차: 유사도 → top 30).

    순서가 중요합니다: **거른 뒤 자릅니다.** 자른 뒤 거르면 조용한 날에도 30건을
    채운 것처럼 보였다가 임계값에서 사라져 "왜 5건이 안 나오나"를 추적할 수 없습니다.
    """
    resolved = settings or Stage1Settings(
        top_n=DEFAULT_STAGE1_TOP_N,
        final_n=DEFAULT_FINAL_N,
        min_score=DEFAULT_MIN_SCORE_THRESHOLD,
    )
    limit = resolved.top_n if isinstance(top_n, _Unset) else top_n
    threshold = resolved.min_score if isinstance(min_score, _Unset) else min_score

    kept = list(ranked)
    if threshold is not None:
        # §5.7 — 감점이 반영된 최종 점수에 겁니다. base 에 걸면 배제 대상이 통과합니다.
        kept = [row for row in kept if row.score >= threshold]
    if limit is not None:
        if limit < 0:
            raise ValueError(f"top_n 이 음수입니다: {limit}")
        kept = kept[:limit]
    return kept


def run_stage1(
    items: Sequence[Any],
    embedder: Embedder,
    queryset: QuerySet,
    *,
    profile: Mapping[str, Any] | None = None,
    top_n: int | None | _Unset = UNSET,
    min_score: float | None | _Unset = UNSET,
    method: str | None = None,
    penalty_weight: float | None = None,
) -> list[RankedItem]:
    """점수 → 정렬 → 임계 → 상위 N. 원장에는 `len()` 을 `stage1_top_n` 으로 남기세요."""
    ranked = score_items(
        items, embedder, queryset, method=method, penalty_weight=penalty_weight
    )
    settings = stage1_settings(profile, top_n=top_n, min_score=min_score)
    return select_stage1(ranked, settings)
