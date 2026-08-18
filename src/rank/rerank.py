"""2차 랭킹 — cross-encoder (기획안_2 §5.1 · §9.4).

★ 함정 9.4 — CrossEncoder 는 기본으로 sigmoid 를 씌웁니다
--------------------------------------------------------
| 설정 | 범위 | 표준편차 |
|:--|:--|--:|
| 기본값 | `+0.0000 ~ +0.0751` | 0.029 |
| `activation_fn=Identity()` | `-10.08 ~ -2.51` | 2.92 |

순위는 단조변환이라 **동일합니다.** 문제는 원장과 임계값입니다: sigmoid 를 씌우면
무관한 쌍이 전부 0 근처로 뭉쳐서 `rank_score_stage2`(§4.4)를 사람이 읽을 수 없고,
`0.0000` 과 `0.0001` 이 로짓으로는 -9.2 와 -9.0 이라 임계값을 정할 수가 없습니다.
→ 평가·원장에는 **raw 로짓**. `eval/models.yaml` 의 `use_raw_logits: true` 가 계약입니다.

이 모듈은 그 계약을 두 겹으로 지킵니다:
  1. 생성 시 `activation_fn=torch.nn.Identity()` 를 넘긴다 (넘겼는지 테스트가 검사)
  2. 나온 점수가 전부 [0,1] 안에 뭉쳐 있으면 **RuntimeError 로 죽는다**
     — 누군가 1번을 되돌려도 조용히 sigmoid 값이 원장에 들어가지 않게.

질의를 무엇으로 할 것인가
-------------------------
cross-encoder 는 (질의, 문서) 쌍을 받습니다. 프로파일에는 관심사가 15개라 "질의"가
하나로 정해지지 않습니다. 여기서는 **1차 랭킹에서 그 문서를 끌어올린 관심사**의
text 를 질의로 씁니다 (`RankedItem.best_interest`). 근거: 가중 최대(§5.2)로 뽑힌
문서는 정의상 그 관심사와 가장 잘 맞고, 15개를 이어붙인 질의는 cross-encoder 의
512 토큰(models.yaml `max_length`)을 넘겨 뒤쪽이 잘립니다.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

from src.rank.embed import ModelSpec, load_model_spec, resolve_device
from src.rank.profile import QuerySet
from src.rank.retrieve import UNSET, RankedItem, _Unset, document_text

#: 최종 선정 건수 기본값 (§5.1 — 2차: cross-encoder → top 5).
DEFAULT_FINAL_N = 5

#: sigmoid 판정 기준. 쌍이 이보다 적으면 판정하지 않습니다 — 2~3건이 우연히
#: [0,1] 에 들어갈 확률은 낮지 않습니다. 실측 로짓 범위는 -10.08 ~ -2.51 입니다.
_RAW_LOGIT_MIN_SAMPLES = 3


@runtime_checkable
class Reranker(Protocol):
    """리랭커 포트 (§4.2). `score_pairs` 는 **raw 로짓**을 냅니다 (§9.4)."""

    @property
    def model_id(self) -> str: ...

    @property
    def revision(self) -> str: ...

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]: ...


@dataclass(frozen=True, slots=True)
class RerankedItem:
    """2차 점수가 붙은 후보. `stage1_score` 를 같이 들고 다니는 이유는,
    원장(§4.4)이 `rank_score_stage1` 과 `rank_score_stage2` 를 **둘 다** 요구하고
    1차/2차가 얼마나 다른 순서를 냈는지가 M1 비교 조건 3의 내용이기 때문입니다."""

    item_id: str
    stage1_score: float
    stage2_score: float
    query_id: str | None
    rank: int
    item: Any


def looks_like_probabilities(scores: Sequence[float]) -> bool:
    """전부 [0,1] 안이면 sigmoid 가 씌워졌을 가능성 (§9.4 진단).

    확정 판정이 아니라 신호입니다. 그래서 표본이 적으면 판정하지 않습니다.
    """
    values = [float(value) for value in scores]
    if len(values) < _RAW_LOGIT_MIN_SAMPLES:
        return False
    return all(0.0 <= value <= 1.0 for value in values)


class FakeReranker:
    """키·모델 없이 도는 결정적 리랭커 — **배선 테스트 전용**.

    ★ 랭킹 품질 평가에 쓰지 마세요. 자카드 겹침일 뿐 의미를 모릅니다.

    실모델의 관측 범위(-10.08 ~ -2.51)를 흉내 내도록 겹침을 로짓으로 사상합니다:
    겹침 0 → -10.0, 겹침 1 → +2.0. **음수가 나오는 것이 의도**입니다 — 반환값을
    확률로 착각하는 코드가 있으면 여기서 드러나야 합니다 (§9.4).
    """

    model_id = "fake_reranker"
    revision = "fake0001"

    _SLOPE = 12.0
    _INTERCEPT = -10.0

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        scores: list[float] = []
        for query, document in pairs:
            left = set(query.lower().split())
            right = set(document.lower().split())
            union = left | right
            overlap = len(left & right) / len(union) if union else 0.0
            scores.append(self._SLOPE * overlap + self._INTERCEPT)
        return scores


class BgeReranker:
    """`BAAI/bge-reranker-v2-m3` 어댑터 (로컬 CPU, 키 불필요 — CLAUDE.md §4).

    - `sentence_transformers`·`torch` 는 **점수 계산 시점에** import 합니다 (§9.10)
    - repo·revision·max_length 는 `eval/models.yaml` 에서 읽습니다 (R8 / §9.3)
    - `activation_fn=torch.nn.Identity()` 로 raw 로짓을 받습니다 (§9.4)
    - `factory`/`activation` 은 테스트 주입구입니다. 실모델을 받지 않고도
      "Identity 를 정말 넘기는가"를 검사할 수 있어야 합니다

    실측 (models.yaml): 30쌍 31.5초 (CPU). 2차 랭킹은 30건만 처리하므로 1분 이내입니다.
    """

    def __init__(
        self,
        spec: ModelSpec | None = None,
        *,
        device: str | None = None,
        models_path: Path | str | None = None,
        max_length: int | None = None,
        batch_size: int = 8,
        factory: Callable[..., Any] | None = None,
        activation: Any = None,
        strict_raw_logits: bool = True,
    ) -> None:
        self._spec = spec
        self._device_arg = device
        self._models_path = models_path
        self._max_length_arg = max_length
        self._batch_size = batch_size
        self._factory = factory
        self._activation = activation
        self._strict_raw_logits = strict_raw_logits
        self._model: Any = None
        self._device: str | None = None

    @property
    def spec(self) -> ModelSpec:
        if self._spec is None:
            self._spec = load_model_spec("reranker", self._models_path)
        return self._spec

    @property
    def model_id(self) -> str:
        return self.spec.id

    @property
    def revision(self) -> str:
        return self.spec.revision

    @property
    def max_length(self) -> int:
        if self._max_length_arg is not None:
            return int(self._max_length_arg)
        # models.yaml 의 512. 코드에 박지 않습니다 (R8).
        return int(self.spec.raw.get("max_length", 512))

    @property
    def device(self) -> str:
        if self._device is None:
            self._device = resolve_device(self._device_arg, self._models_path)
        return self._device

    def _identity_activation(self) -> Any:
        if self._activation is not None:
            return self._activation
        import torch  # noqa: PLC0415  — 지연 import (§9.10)

        return torch.nn.Identity()

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        factory = self._factory or _default_cross_encoder
        kwargs: dict[str, Any] = {
            "revision": self.spec.revision,  # ★ §9.3 — 빼면 미머지 PR 가중치가 옵니다
            "max_length": self.max_length,
            "device": self.device,
            "activation_fn": self._identity_activation(),  # ★ §9.4 — raw 로짓
        }
        try:
            self._model = factory(self.spec.repo, **kwargs)
        except TypeError as exc:
            # sentence-transformers 4.x 이하는 이름이 `activation_fct` 였습니다.
            # 이름이 안 맞을 때 **인자를 빼고 재시도하면 조용히 sigmoid 가 됩니다** —
            # 그건 §9.4 그 자체입니다. 이름만 바꿔 다시 넘기고, 그래도 안 되면 죽습니다.
            if "activation_fn" not in str(exc):
                raise
            kwargs["activation_fct"] = kwargs.pop("activation_fn")
            self._model = factory(self.spec.repo, **kwargs)
        return self._model

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        pairs = list(pairs)
        if not pairs:
            return []
        model = self._load()
        raw = model.predict(pairs, batch_size=self._batch_size, show_progress_bar=False)
        scores = [float(value) for value in raw]

        # ★ §9.4 2중 방어. models.yaml 이 raw 로짓을 계약했는데 전부 [0,1] 이면
        #   activation 이 어딘가에서 되돌려진 것입니다. 원장에 sigmoid 값이 들어가
        #   임계값을 못 정하게 되는 것보다 여기서 죽는 게 낫습니다.
        if (
            self._strict_raw_logits
            and bool(self.spec.raw.get("use_raw_logits"))
            and looks_like_probabilities(scores)
        ):
            raise RuntimeError(
                "리랭커 점수가 전부 [0,1] 입니다 — sigmoid 가 씌워진 것으로 보입니다 "
                "(기획안_2 §9.4). CrossEncoder(activation_fn=torch.nn.Identity()) 를 "
                f"확인하세요. 관측: min={min(scores):.4f} max={max(scores):.4f}"
            )
        return scores


def _default_cross_encoder(repo: str, **kwargs: Any) -> Any:
    """실모델 생성. **지연 import** 입니다 (§9.10)."""
    from sentence_transformers import CrossEncoder  # noqa: PLC0415

    return CrossEncoder(repo, **kwargs)


def query_for(row: RankedItem, queryset: QuerySet) -> tuple[str | None, str]:
    """`(관심사 id, 질의 텍스트)`. 1차에서 이 문서를 끌어올린 관심사를 씁니다.

    `best_interest` 가 없거나(빈 프로파일 등) 이름이 안 맞으면 첫 관심사로 물러섭니다.
    질의가 없으면 2차 랭킹을 아예 못 하므로, 여기서만은 폴백이 정당합니다.
    """
    if row.best_interest:
        try:
            return row.best_interest, queryset.interest_text(row.best_interest)
        except KeyError:
            pass
    first = queryset.interests[0]
    return first.id, first.text


def rerank_stage2(
    ranked: Sequence[RankedItem],
    reranker: Reranker,
    queryset: QuerySet,
    *,
    final_n: int | None | _Unset = UNSET,
    query_text: str | None = None,
) -> list[RerankedItem]:
    """1차 상위 N → cross-encoder → 최종 `final_n` 건 (§5.1).

    - 점수는 **raw 로짓**입니다. `rank_score_stage2` 에 그대로 넣으세요 (§4.4 / §9.4)
    - 동점은 `item_id` 사전순 (§5.4). 1차와 같은 계약입니다
    - `query_text` 를 주면 모든 쌍에 그 질의를 씁니다 (비교 조건용)
    - `final_n=None` 은 **자르지 않음**, 미지정은 `DEFAULT_FINAL_N`
      (retrieve.UNSET 주석 참조 — 평가에서는 전체 순위가 필요합니다. nDCG@10 을
      상위 5건만으로 계산하면 값이 조용히 낮아집니다)
    """
    rows = list(ranked)
    if not rows:
        return []

    queries: list[tuple[str | None, str]] = []
    for row in rows:
        if query_text is not None:
            queries.append((None, query_text))
        else:
            queries.append(query_for(row, queryset))

    pairs = [(query, document_text(row.item)) for (_, query), row in zip(queries, rows)]
    scores = reranker.score_pairs(pairs)
    if len(scores) != len(rows):
        raise RuntimeError(
            f"쌍 {len(rows)}건에 점수 {len(scores)}건 — 리랭커가 순서를 흘렸습니다"
        )

    ordered = sorted(
        zip(rows, queries, scores),
        key=lambda triple: (-triple[2], triple[0].item_id),  # ★ §5.4
    )
    limit = DEFAULT_FINAL_N if isinstance(final_n, _Unset) else final_n
    result = [
        RerankedItem(
            item_id=row.item_id,
            stage1_score=row.score,
            stage2_score=float(score),
            query_id=query_id,
            rank=index,
            item=row.item,
        )
        for index, (row, (query_id, _), score) in enumerate(ordered, start=1)
    ]
    if limit is not None:
        if limit < 0:
            raise ValueError(f"final_n 이 음수입니다: {limit}")
        result = result[:limit]
    return result


def build_reranker(
    kind: str = "fake",
    *,
    device: str | None = None,
    models_path: Path | str | None = None,
) -> Reranker:
    """이름으로 리랭커를 만듭니다 (§4.2). 알 수 없는 이름은 **조용한 폴백 없이** 예외."""
    if kind == "fake":
        return FakeReranker()
    if kind in ("bge", "bge_reranker_v2_m3", "reranker"):
        return BgeReranker(device=device, models_path=models_path)
    raise ValueError(f"알 수 없는 리랭커: {kind!r} (가능: 'fake', 'bge_reranker_v2_m3')")


def selection_settings(profile: Mapping[str, Any] | None = None) -> int:
    """프로파일 `selection.final_n` (기본 5). 코드에 5를 박지 않기 위한 한 줄입니다."""
    if profile:
        selection = profile.get("selection")
        if isinstance(selection, Mapping) and selection.get("final_n") is not None:
            return int(selection["final_n"])
    return DEFAULT_FINAL_N
