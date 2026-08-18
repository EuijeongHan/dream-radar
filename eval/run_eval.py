"""M1 정량 평가 러너 — Hit@5 / MRR / nDCG@10 과 조건 비교 (기획안_2 §5.4~5.6, §5.8).

    python -m eval.run_eval                                    # 기본 4조건
    python -m eval.run_eval --conditions baseline bge_m3       # 일부만
    python -m eval.run_eval --device cpu

이 파일의 담당은 **지표와 조건 비교**입니다. 랭킹 자체는 만들지 않습니다
--------------------------------------------------------------------------
- 임베딩·점수 합성·1차 선별·2차 리랭킹은 전부 `src/rank` 를 그대로 부릅니다
  (`build_queryset` → `score_items` → `select_stage1` → `rerank_stage2`).
  평가가 랭킹을 **다시 구현하면** 두 구현이 갈라지는 순간 results.md 의 수치가
  운영 파이프라인과 다른 코드의 것이 됩니다. 그게 이 프로젝트에서 가장 비싼 사고입니다
  (§5.3 임베딩 캐시도 `src.rank.embed.CachedEmbedder` 안에 있습니다).
- **원장(`data/runs.jsonl`)에 쓰지 않습니다.** 원장 1줄 = 파이프라인 1회 실행(§4.4)이고
  평가는 로컬 1회성 배치입니다. 평가의 기록물은 `eval/results.md` 입니다.
- **골드셋을 만들지 않습니다.** 랭커로 후보를 뽑아 라벨링하면 평가 대상이 정답셋의
  범위를 정하게 됩니다 (§9.11 풀링 편향). 골드셋은 `eval/label.py` 전수 라벨링의
  산출물이고, 없으면 이 러너는 **거부합니다.**

여기서 직접 구현하는 것은 두 가지뿐입니다
------------------------------------------
1. **지표** (§5.4) — 프로덕션 랭커가 관여할 일이 아닙니다
2. **베이스라인** (§5.5) — 임베딩을 쓰지 않는 어휘 매칭이라 `src/rank` 에 둘 것이
   아닙니다. 베이스라인이 약하면 개선폭이 과장되므로 성실하게 만듭니다

지표 정의는 §5.4 그대로입니다. 특히:
- 정답이 0건인 날은 **평가에서 제외**하고 제외 사실을 results.md 에 남깁니다
- 정답이 랭킹에 하나도 없으면 MRR 은 `1/∞` 가 아니라 **0**
- nDCG 의 IDCG 는 `min(|G_d|, 10)` 개가 최상위에 있을 때의 값
- 동점은 **(점수 내림차순, item_id 오름차순)** 고정 — 정렬이 불안정하면 같은 명령을
  두 번 돌렸을 때 수치가 달라집니다 (§5.8 재현성). `src.rank.retrieve.sort_ranked` 와
  같은 계약입니다
- **날짜별로 계산하고 평균냅니다.** 전체를 한 풀로 합치면 안 됩니다 (운영이 하루 단위)
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

# 후보 로딩은 라벨링 도구의 것을 재사용합니다. 중복 줄 접기(§9.6)가 거기 있고,
# 라벨링과 평가가 **다른 후보 집합**을 보면 골드셋과 랭킹이 어긋납니다.
from eval.label import load_candidates
from src.core.config import load_profile  # R1 — 프로파일 경로를 문자열로 쓰지 않습니다
from src.rank.embed import Embedder, build_embedder, load_models_config
from src.rank.profile import build_queryset
from src.rank.rerank import Reranker, build_reranker, rerank_stage2
from src.rank.retrieve import document_text, score_items, select_stage1, stage1_settings

#: 기본 경로들. 함수 기본 인자에 **직접 쓰지 않습니다** — 정의 시점에 바인딩되어
#: 테스트가 tmp 경로를 넘겨도 실제 `data/` 를 읽습니다 (§9.1 / R7). `None` + 호출 시점 해석.
GOLDSET_PATH = Path("eval/goldset.yaml")
RESULTS_PATH = Path("eval/results.md")

#: 운영 프로파일(영어)과 교차언어 대조군(한국어). 둘 다 **stem** 입니다 —
#: `resolve_profile()` 이 최고 번호를 고릅니다 (R1 / §9.9).
PROFILE_STEM = "profile.papers"
PROFILE_STEM_KO = "profile.papers.ko"

#: 조건 키 → 설명 (§5.5). 기본 4개 + §5.2 대조용 1개.
CONDITIONS: dict[str, str] = {
    "baseline": "베이스라인 — 프로파일 명사구 매칭 × weight (§5.5)",
    "bge_m3": "임베딩만 — 가중 최대 합성 (§5.2 기본값)",
    "bge_m3_rerank": "임베딩 + cross-encoder 리랭킹 — 최종 구성",
    "bge_m3_ko": "임베딩 + 한국어 프로파일 — 교차언어 대조",
    "bge_m3_wmean": "(선택) 임베딩만 — 가중 평균 합성 (§5.2 대조)",
}
DEFAULT_CONDITIONS: tuple[str, ...] = ("baseline", "bge_m3", "bge_m3_rerank", "bge_m3_ko")

#: 임베딩을 쓰는 조건. `baseline` 만 돌리면 모델 없이 완주해야 합니다 (CLAUDE.md §4).
EMBEDDING_CONDITIONS: frozenset[str] = frozenset(
    {"bge_m3", "bge_m3_rerank", "bge_m3_ko", "bge_m3_wmean"}
)


# ══════════════════════════════════════════════════════════════════════════
# 지표 — §5.4 정의 그대로
# ══════════════════════════════════════════════════════════════════════════


def rank_items(scores: Mapping[str, float]) -> list[str]:
    """점수 매핑을 랭킹 리스트로. **(점수 내림차순, item_id 오름차순)** 고정입니다.

    `src.rank.retrieve.sort_ranked` 와 같은 계약입니다. 동점 처리를 고정하지 않으면
    같은 조건을 두 번 돌렸을 때 수치가 달라집니다 (§5.4 / §5.8 재현성). dict 삽입
    순서에 기대면 후보 파일 줄 순서가 바뀌는 것만으로 Hit@5 가 흔들립니다.
    """
    return [item_id for item_id, _ in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))]


def hit_at_k(ranked: Sequence[str], gold: Iterable[str], k: int = 5) -> float | None:
    """상위 k 에 정답이 하나라도 있으면 1.0.

    `G_d = ∅` 이면 **None** — 그날은 평가에서 제외한다는 뜻입니다 (§5.4).
    0.0 을 돌려주면 "정답이 없는 날"과 "다 놓친 날"이 같은 값이 되어 평균이 왜곡됩니다.
    제외 사실은 `evaluate_condition()` 이 모아 results.md 에 남깁니다.
    """
    gold = set(gold)
    if not gold:
        return None
    return 1.0 if any(item_id in gold for item_id in ranked[:k]) else 0.0


def mrr(ranked: Sequence[str], gold: Iterable[str], k: int | None = None) -> float | None:
    """첫 정답의 역순위. 정답이 리스트에 하나도 없으면 **0.0** (`1/∞` 아님, §5.4).

    `k` 를 주면 상위 k 밖의 정답은 없는 것으로 봅니다 (§5.4 "top-k 밖이면 0").
    기본값 `None` 은 자르지 않음 — 후보 풀 전체를 랭킹하므로 기본은 전체 리스트입니다.
    """
    gold = set(gold)
    if not gold:
        return None
    window = ranked if k is None else ranked[:k]
    for position, item_id in enumerate(window, start=1):
        if item_id in gold:
            return 1.0 / position
    return 0.0


def ndcg_at_10(ranked: Sequence[str], gold: Iterable[str], k: int = 10) -> float | None:
    """이진 relevance nDCG@k. IDCG 는 `min(|G_d|, k)` 개가 최상위에 있을 때 값 (§5.4).

    IDCG 를 `|G_d|` 로 잡으면 정답이 k 개보다 많은 날 상한이 1.0 미만이 되어,
    "정답이 많은 날일수록 랭커가 못한다"는 착시가 생깁니다.
    """
    gold = set(gold)
    if not gold:
        return None
    dcg = sum(
        1.0 / math.log2(position + 1)
        for position, item_id in enumerate(ranked[:k], start=1)
        if item_id in gold
    )
    idcg = sum(1.0 / math.log2(position + 1) for position in range(1, min(len(gold), k) + 1))
    return dcg / idcg


@dataclass(frozen=True, slots=True)
class DateResult:
    """날짜 1일치 결과. `candidates` 열은 results.md 에서 **뺄 수 없습니다** (§9.12)."""

    date: str
    candidates: int
    relevant: int
    hit_at_5: float
    mrr: float
    ndcg_at_10: float
    #: 골드셋에 정답으로 라벨됐는데 그날 후보 풀에 없는 건수. 랭커의 잘못이 아니라
    #: 수집·중복제거 쪽 문제이므로 지표와 분리해 따로 기록합니다.
    gold_missing: int


@dataclass(frozen=True, slots=True)
class ConditionResult:
    condition: str
    per_date: tuple[DateResult, ...]
    #: 정답 0건이라 계산에서 **제외된** 날 (§5.4). 제외 사실을 남기는 게 요구사항입니다.
    excluded_dates: tuple[str, ...]

    def mean(self, field: str) -> float | None:
        """날짜별 값의 평균. 제외된 날은 애초에 `per_date` 에 없습니다.

        **전체 후보를 한 풀로 합쳐 계산하지 않습니다** (§5.4) — 운영은 하루 단위로
        돌기 때문입니다. 풀을 합치면 후보 719건인 날이 350건인 날을 삼킵니다.
        """
        if not self.per_date:
            return None
        return sum(getattr(row, field) for row in self.per_date) / len(self.per_date)


def evaluate_condition(
    condition: str,
    rankings: Mapping[str, Sequence[str]],
    gold: Mapping[str, Iterable[str]],
    candidate_counts: Mapping[str, int],
) -> ConditionResult:
    """날짜별로 지표를 계산하고 (평균은 `ConditionResult.mean`) 제외된 날을 보고합니다."""
    rows: list[DateResult] = []
    excluded: list[str] = []
    for date in sorted(rankings):
        gold_ids = set(gold.get(date, ()))
        ranked = list(rankings[date])
        hit = hit_at_k(ranked, gold_ids)
        if hit is None:  # 정답 0건 — 그날은 평가에서 제외 (§5.4)
            excluded.append(date)
            continue
        reciprocal = mrr(ranked, gold_ids)
        gain = ndcg_at_10(ranked, gold_ids)
        if reciprocal is None or gain is None:  # 도달 불가. 도달했다면 조용히 넘기지 않습니다
            raise RuntimeError(f"{date}: 지표가 정의되지 않았습니다 (정답 {len(gold_ids)}건)")
        rows.append(
            DateResult(
                date=date,
                candidates=candidate_counts.get(date, len(ranked)),
                relevant=len(gold_ids),
                hit_at_5=hit,
                mrr=reciprocal,
                ndcg_at_10=gain,
                gold_missing=len(gold_ids - set(ranked)),
            )
        )
    return ConditionResult(condition=condition, per_date=tuple(rows), excluded_dates=tuple(excluded))


# ══════════════════════════════════════════════════════════════════════════
# 베이스라인 — 프로파일 명사구 매칭 (§5.5)
# ══════════════════════════════════════════════════════════════════════════

#: 문법 기능어만 뺍니다. "model", "evaluation" 같은 **내용어를 빼면 베이스라인이
#: 약해지고, 약한 베이스라인은 개선폭을 과장합니다** (§5.5). 정직한 하한선이어야 합니다.
STOPWORDS: frozenset[str] = frozenset(
    """
    the and for that with from this are was were its their such than not but into over under
    between within without while when where which who whom whose what how why can could may
    might will would shall should must have has had been being does did doing also only other
    others more most less least much many some any all both each every either neither one two
    three instead rather therefore thus hence however although though because since upon across
    against among about after before during through toward towards per via them they there these
    those then our your his her
    """.split()
)

#: 3자 미만 토큰은 버립니다 (§5.5). `of`·`in`·`to` 같은 기능어가 대부분이고,
#: 남는 2자 약어는 매칭 잡음이 큽니다.
MIN_TOKEN_LEN = 3

_WORD_RE = re.compile(r"[a-z0-9]+")


def content_tokens(text: str) -> list[str]:
    """소문자 정규화 → 영숫자 토큰화 → 기능어·2자 이하 제거.

    질의(프로파일 text)와 문서(제목+초록)에 **똑같이** 적용합니다. 한쪽만 다르게
    처리하면 매칭이 조용히 어긋납니다.
    """
    return [
        token
        for token in _WORD_RE.findall(text.lower())
        if len(token) >= MIN_TOKEN_LEN and token not in STOPWORDS
    ]


def noun_phrase_terms(text: str) -> set[str]:
    """단어 + 인접 2단어 구. "명사구" 의 근사입니다 (§5.5).

    형태소 분석기를 새로 들이지 않습니다 (의존성 추가 금지 — §9.10). 기능어를 걷어낸
    뒤의 인접 2-gram 은 `"cross encoder"`, `"identity preserving"` 처럼 실제 명사구를
    잘 잡습니다. 문서 쪽에도 같은 변환을 적용하므로 부분 문자열 오탐이 없습니다
    (`"art"` 가 `"start"` 에 걸리는 문제).
    """
    tokens = content_tokens(text)
    terms = set(tokens)
    terms.update(f"{first} {second}" for first, second in zip(tokens, tokens[1:]))
    return terms


def baseline_scores(
    items: Sequence[Any], interests: Sequence[Mapping[str, Any]]
) -> dict[str, float]:
    """`Σ_i weight_i × (관심사 i 의 명사구 중 문서에 등장한 개수)` (§5.5).

    - **등장 횟수가 아니라 서로 다른 term 수**를 셉니다. 같은 단어가 초록에 5번
      나온다고 근거가 5배가 되지는 않습니다. 긴 초록이 무조건 이기는 것도 막습니다.
    - 정규화하지 않습니다 — §5.5 가 "매칭 건수 × weight 가중합" 이라고 못박았고,
      15개 관심사의 `text` 길이가 서로 비슷해(45~70단어) 정규화의 효과가 작습니다.
    - `exclude.soft` 감점은 적용하지 않습니다. §5.2 의 감점은 임베딩 코사인 기반이고
      §5.5 의 베이스라인 정의에는 감점이 없습니다. **한쪽에만 장치를 임의로 붙이거나
      빼면 조건 비교가 성립하지 않습니다.**

    문서 텍스트는 `src.rank.retrieve.document_text` 를 씁니다 — 임베딩 조건과 **같은
    입력**을 봐야 비교가 공정합니다.
    """
    interest_terms = [
        (noun_phrase_terms(str(row["text"])), float(row.get("weight", 1.0))) for row in interests
    ]
    scores: dict[str, float] = {}
    for item in items:
        doc_terms = noun_phrase_terms(document_text(item))
        item_id = str(item["id"] if isinstance(item, Mapping) else getattr(item, "id"))
        scores[item_id] = sum(weight * len(terms & doc_terms) for terms, weight in interest_terms)
    return scores


# ══════════════════════════════════════════════════════════════════════════
# 골드셋
# ══════════════════════════════════════════════════════════════════════════

_LABELING_GUIDE = """골드셋이 없으면 평가할 수 없습니다 (기획안_2 §5.8 DoD).
랭커 상위 N건만 라벨링해서 대신하지 마세요 — 평가 대상이 정답셋의 범위를 정하게 되어
개선폭이 실제보다 좋게 나옵니다 (§9.11 풀링 편향).

  python -m eval.label triage  --date <날짜>   # 1패스 — 제목만 보고 전수 트리아지
  python -m eval.label review  --date <날짜>   # 2패스 — 보류분 초록 판정
  python -m eval.label recheck --date <날짜>   # 1패스 누락률 실측
  python -m eval.label status                  # 진행률 (3일치 필요)
  python -m eval.label export                  # eval/goldset.yaml 생성"""


@dataclass(frozen=True, slots=True)
class Goldset:
    path: Path
    dates: tuple[str, ...]
    gold: Mapping[str, frozenset[str]]
    labeled: Mapping[str, int]
    generated_at: str
    recheck_sampled: int
    recheck_flipped: int
    #: 1패스 누락률. `recheck` 를 안 돌렸으면 `None` 입니다 — **안 한 것은 안 했다고**
    #: 적습니다 (작업규약 §7.2). 0.0 으로 채우면 "재검토했는데 누락이 없었다"가 됩니다.
    miss_rate: float | None


def load_goldset(path: Path | None = None) -> Goldset:
    """`eval/goldset.yaml` 을 읽습니다. 없으면 라벨링 명령을 안내하며 거부합니다."""
    path = path or GOLDSET_PATH
    if not path.exists():
        raise SystemExit(f"{path} 가 없습니다.\n\n{_LABELING_GUIDE}")
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not document.get("labels"):
        raise SystemExit(f"{path} 에 labels 가 없습니다.\n\n{_LABELING_GUIDE}")

    gold: dict[str, set[str]] = {}
    labeled: dict[str, int] = {}
    for row in document["labels"]:
        date = str(row["date"])
        if row.get("relevant") is None:
            # export 가 막지만 손으로 고친 파일이 들어올 수 있습니다. 보류를 무관으로
            # 접으면 정답 수가 조용히 줄어 Hit@5 의 상한이 내려갑니다.
            raise SystemExit(f"{path}: 보류(relevant: null) 라벨이 남아 있습니다 — {row['item_id']}")
        labeled[date] = labeled.get(date, 0) + 1
        if row["relevant"]:
            gold.setdefault(date, set()).add(str(row["item_id"]))

    summary = document.get("summary") or {}
    recheck = summary.get("title_pass_recheck") or {}
    dates = tuple(sorted(labeled))
    return Goldset(
        path=path,
        dates=dates,
        gold={date: frozenset(gold.get(date, ())) for date in dates},
        labeled=labeled,
        generated_at=str(document.get("generated_at", "미상")),
        recheck_sampled=int(recheck.get("sampled", 0) or 0),
        recheck_flipped=int(recheck.get("flipped_to_relevant", 0) or 0),
        miss_rate=recheck.get("miss_rate"),
    )


# ══════════════════════════════════════════════════════════════════════════
# 조건 러너 — 랭킹은 src/rank 를 그대로 부릅니다
# ══════════════════════════════════════════════════════════════════════════


def embedding_ranking(
    items: Sequence[Any],
    embedder: Embedder,
    queryset: Any,
    *,
    method: str | None = None,
    penalty_weight: float | None = None,
    reranker: Reranker | None = None,
    profile: Mapping[str, Any] | None = None,
) -> list[str]:
    """한 날짜의 후보를 랭킹해 `item_id` 리스트로 돌려줍니다.

    ★ 리랭킹은 1차 상위 N만 재정렬하고 **나머지를 버리지 않습니다.** 꼬리를 버리면
      정답이 31위인 날의 MRR 이 "정답이 아예 없는 날"과 같은 0이 되어, 리랭커가
      실제로 얼마나 끌어올렸는지 측정할 수 없습니다. 운영은 top-5 만 발행하지만
      **평가는 전체 순위를 봐야 합니다.**

    `rerank_stage2` 의 `final_n` 은 기본값이 5(발행 건수)이므로, 평가에서는 1차 통과
    건수를 그대로 넘겨 **자르지 않게** 합니다.
    """
    ranked = score_items(items, embedder, queryset, method=method, penalty_weight=penalty_weight)
    if reranker is None:
        return [row.item_id for row in ranked]

    head = select_stage1(ranked, stage1_settings(profile))
    reranked = rerank_stage2(head, reranker, queryset, final_n=len(head))
    head_ids = [row.item_id for row in reranked]
    seen = set(head_ids)
    # 1차 순서를 유지한 꼬리. `select_stage1` 이 임계값(§5.7)으로 걸러낸 것도 여기 들어옵니다.
    return head_ids + [row.item_id for row in ranked if row.item_id not in seen]


def run_conditions(
    conditions: Sequence[str] | None = None,
    *,
    device: str | None = None,
    goldset_path: Path | None = None,
    candidates_dir: Path | None = None,
    models_path: Path | None = None,
    profile_root: Path | None = None,
    cache_dir: Path | None = None,
    out: Path | None = None,
    embedder: Embedder | None = None,
    reranker: Reranker | None = None,
    embedder_kind: str | None = None,
    reranker_kind: str | None = None,
    penalty_weight: float | None = None,
) -> Path:
    """골드셋의 모든 날짜에 대해 조건들을 돌리고 `eval/results.md` 를 씁니다.

    경로·설정 인자의 기본값은 전부 `None` 이고 **호출 시점에** 해석합니다 (§9.1 / R7).

    `embedder`/`reranker` 를 직접 넘기면 생성을 건너뜁니다 — 테스트가 스텁을 주입하는
    통로입니다. **기본값은 스텁이 아니라 실모델입니다**: `hash_stub` 으로 낸 Hit@5 는
    거짓이고(작업규약 §8-9), 모델이 없어서 죽는 게 가짜로 성공하는 것보다 낫습니다(§4.2).
    """
    selected = tuple(conditions or DEFAULT_CONDITIONS)
    unknown = [name for name in selected if name not in CONDITIONS]
    if unknown:
        raise SystemExit(f"알 수 없는 조건: {unknown}. 가능한 값: {sorted(CONDITIONS)}")

    goldset = load_goldset(goldset_path)
    models = load_models_config(models_path)
    resolved_device = device or str((models.get("device") or {}).get("preferred", "cpu"))

    items_by_date: dict[str, list[Any]] = {
        date: list(load_candidates(date, candidates_dir)) for date in goldset.dates
    }
    candidate_counts = {date: len(items) for date, items in items_by_date.items()}

    profile, profile_path = _load(PROFILE_STEM, profile_root)
    settings = stage1_settings(profile)

    needs_embedder = any(name in EMBEDDING_CONDITIONS for name in selected)
    if needs_embedder and embedder is None:
        kind = embedder_kind or str((models.get("embedder") or {}).get("id") or "bge_m3")
        embedder = build_embedder(
            kind, cache_dir=cache_dir, device=resolved_device, models_path=models_path
        )
    if "bge_m3_rerank" in selected and reranker is None:
        kind = reranker_kind or str((models.get("reranker") or {}).get("id") or "bge_reranker_v2_m3")
        reranker = build_reranker(kind, device=resolved_device, models_path=models_path)

    querysets: dict[str, Any] = {}
    ko_path: Path | None = None
    if needs_embedder:
        if embedder is None:  # 도달 불가. 조용히 통과시키지 않습니다
            raise RuntimeError("임베딩 조건인데 Embedder 가 없습니다")
        querysets["en"] = build_queryset(
            embedder, profile=profile, profile_path=profile_path, penalty_weight=penalty_weight
        )
        if "bge_m3_ko" in selected:
            # 한국어 프로파일은 **평가 대조군**입니다. 운영 파이프라인에 로드하면 안 되고
            # (CLAUDE.md §1-3) 여기서만 씁니다. 경로를 문자열로 쓰지 않습니다 (R1).
            ko_profile, ko_path = _load(PROFILE_STEM_KO, profile_root)
            querysets["ko"] = build_queryset(
                embedder, profile=ko_profile, profile_path=ko_path, penalty_weight=penalty_weight
            )

    results: list[ConditionResult] = []
    for name in selected:
        if name == "baseline":
            interests = profile.get("interests") or ()
            if not interests:
                raise SystemExit(f"{profile_path} 에 interests 가 없습니다")
            rankings = {
                date: rank_items(baseline_scores(items, interests))
                for date, items in items_by_date.items()
            }
        else:
            rankings = {
                date: embedding_ranking(
                    items,
                    embedder,  # type: ignore[arg-type]  — 위에서 None 을 배제했습니다
                    querysets["ko"] if name == "bge_m3_ko" else querysets["en"],
                    method="weighted_mean" if name == "bge_m3_wmean" else "weighted_max",
                    penalty_weight=penalty_weight,
                    reranker=reranker if name == "bge_m3_rerank" else None,
                    profile=profile,
                )
                for date, items in items_by_date.items()
            }
        results.append(evaluate_condition(name, rankings, goldset.gold, candidate_counts))

    meta = {
        "device": resolved_device,
        "models": models,
        "profile_path": str(profile_path),
        "ko_profile_path": str(ko_path) if ko_path else None,
        "composition": "가중 최대 (§5.2 기본값)",
        # `build_queryset` 이 인자 > 프로파일 `selection.penalty_weight` > 기본값 순으로
        # 해석한 **실제 값**을 적습니다. 인자를 그대로 적으면 None 이 남습니다.
        "penalty_weight": querysets["en"].penalty_weight if "en" in querysets else penalty_weight,
        "top_n": settings.top_n,
        "min_score": settings.min_score,
        "embedder": _describe_model(embedder),
        "reranker": _describe_model(reranker),
        # ★ 표의 `임베더 revision` 열은 **실제로 쓴** 구현체의 리비전입니다.
        #   models.yaml 의 카탈로그 값을 찍으면 스텁으로 돌린 표가 실모델 표로 보입니다.
        "embedder_revision": getattr(embedder, "revision", None) if embedder is not None else None,
        "stub_models": [
            _describe_model(model) for model in (embedder, reranker) if is_stub_model(model)
        ],
        "max_seq_length": _max_seq_length(embedder),
    }
    out = out or RESULTS_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_results(results, goldset, meta), encoding="utf-8")
    return out


def _load(stem: str, root: Path | None) -> tuple[dict[str, Any], Path]:
    """R1 — stem 으로 최고 번호를 해석합니다. `root` 는 테스트 주입구입니다 (§9.1)."""
    return load_profile(stem) if root is None else load_profile(stem, root=root)


def _describe_model(model: Any) -> str:
    """`구현체(model_id@revision)`. 어느 가중치로 낸 수치인지 표에 남기기 위함입니다 (§9.3)."""
    if model is None:
        return "미사용"
    model_id = getattr(model, "model_id", "") or "?"
    revision = str(getattr(model, "revision", "") or "?")
    return f"{type(model).__name__}({model_id}@{revision[:8]})"


def is_stub_model(model: Any) -> bool:
    """배선 확인용 구현체(`hash_stub`·`fake_reranker`)인가.

    `models.yaml` 의 리비전이 아니라 **실제로 쓴 구현체**로 판정합니다.
    `--embedder hash_stub` 으로 돌려놓고 표에는 카탈로그의 bge-m3 리비전이 찍히면,
    그 표를 읽는 사람은 실모델 수치로 착각합니다. 작업규약 §8-9 가 금지한 것이
    정확히 그것입니다 — 스텁으로 낸 Hit@5 는 거짓입니다.
    """
    if model is None:
        return False
    model_id = str(getattr(model, "model_id", "") or "").lower()
    return "stub" in model_id or "fake" in model_id


def _max_seq_length(embedder: Any) -> int | None:
    """§5.6 이 요구하는 `max_seq_length`. 못 알아내면 **넘겨짚지 않고 None** 입니다.

    `CachedEmbedder` 는 이 값을 노출하지 않습니다. 감싸인 내부 객체를 `_inner` 로
    뒤지지 않는 이유는 그게 사유 속성이라서입니다 — 모르면 results.md 에 "미상"이라고
    적는 게 맞습니다 (작업규약 §7.2 "안 한 것은 안 했다고 적는다").
    """
    value = getattr(embedder, "max_seq_length", None)
    if value is not None:
        return int(value)
    describe = getattr(embedder, "describe", None)
    if callable(describe):
        described = describe().get("max_seq_length")
        return None if described is None else int(described)
    return None


# ══════════════════════════════════════════════════════════════════════════
# results.md — §5.6 / 작업규약 §7.3. **열을 빼지 마세요**
# ══════════════════════════════════════════════════════════════════════════

#: §5.6 과 작업규약 §7.3 이 각각 요구한 열의 합집합입니다. `후보 수` 가 빠지면
#: 표가 무효입니다 (§9.12 — 첫날 719건 / 이후 ~350건이라 평균 해석이 왜곡됩니다).
RESULT_COLUMNS: tuple[str, ...] = (
    "조건",
    "날짜",
    "후보 수",
    "정답 수",
    "Hit@5",
    "MRR",
    "nDCG@10",
    "장치",
    "임베더 revision",
)


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def render_results(
    results: Sequence[ConditionResult], goldset: Goldset, meta: Mapping[str, Any]
) -> str:
    models = meta.get("models") or {}
    embedder_cfg = models.get("embedder") or {}
    reranker_cfg = models.get("reranker") or {}
    device_cfg = models.get("device") or {}
    catalog_revision = str(embedder_cfg.get("revision", "미상"))
    # 표에는 **실제로 쓴** 임베더의 리비전을 찍습니다 (카탈로그 값이 아니라).
    used_revision = meta.get("embedder_revision")
    revision = str(used_revision) if used_revision else "미사용"
    device = str(meta.get("device", "미상"))
    max_seq = meta.get("max_seq_length")
    max_seq_text = "미상 — 임베더가 노출하지 않음" if max_seq is None else str(max_seq)
    stubs = list(meta.get("stub_models") or ())

    lines: list[str] = [
        "# M1 랭킹 평가 결과 (기획안_2 §5.4~5.6)",
        "",
        "`python -m eval.run_eval` 산출물입니다. **손으로 고치지 마세요** — 다시 돌리면 덮어씁니다.",
        "",
    ]

    if stubs:
        # ★ 작업규약 §8-9 — `hash_stub`·`fake` 로 낸 Hit@5 는 거짓입니다. 배선 확인용으로
        #   돌린 표가 나중에 실측치로 인용되는 것을 막습니다. 표를 지우지 않는 이유는,
        #   배선 확인 자체는 정당한 용도이고 결과물을 남길 이유가 있기 때문입니다.
        lines += [
            f"> **⚠ 이 표의 수치는 무효입니다.** 배선 확인용 스텁으로 돌렸습니다: "
            f"{', '.join(f'`{name}`' for name in stubs)}.",
            ">",
            "> `hash_stub`·`FakeReranker` 는 의미를 전혀 모르는 결정적 함수입니다. "
            "여기서 나온 Hit@5·MRR·nDCG 를 실측치로 인용하지 마세요 (작업규약 §8-9). "
            "실측은 `models.yaml` 의 실모델로 다시 돌리세요.",
            "",
        ]

    lines += [
        "| " + " | ".join(RESULT_COLUMNS) + " |",
        "|:--|:--|--:|--:|--:|--:|--:|:--|:--|",
    ]

    for result in results:
        for row in result.per_date:
            lines.append(
                "| "
                + " | ".join(
                    [
                        result.condition,
                        row.date,
                        str(row.candidates),
                        str(row.relevant),
                        _fmt(row.hit_at_5),
                        _fmt(row.mrr),
                        _fmt(row.ndcg_at_10),
                        device,
                        revision[:8],
                    ]
                )
                + " |"
            )
        lines.append(
            "| "
            + " | ".join(
                [
                    f"**{result.condition} 평균**",
                    f"{len(result.per_date)}일 평균",
                    str(sum(row.candidates for row in result.per_date)),
                    str(sum(row.relevant for row in result.per_date)),
                    _fmt(result.mean("hit_at_5")),
                    _fmt(result.mean("mrr")),
                    _fmt(result.mean("ndcg_at_10")),
                    device,
                    revision[:8],
                ]
            )
            + " |"
        )

    lines += ["", "## 조건", ""]
    for result in results:
        lines.append(f"- `{result.condition}` — {CONDITIONS[result.condition]}")

    # 베이스라인 대비 개선폭 — §5.8 DoD "수치로 기록 (개선이 작아도 그대로)".
    baseline = next((r for r in results if r.condition == "baseline"), None)
    if baseline is not None and len(results) > 1:
        lines += [
            "",
            "## 베이스라인 대비 개선폭",
            "",
            "| 조건 | ΔHit@5 | ΔMRR | ΔnDCG@10 |",
            "|:--|--:|--:|--:|",
        ]
        for result in results:
            if result.condition == "baseline":
                continue
            deltas = []
            for field in ("hit_at_5", "mrr", "ndcg_at_10"):
                mine, base = result.mean(field), baseline.mean(field)
                deltas.append("—" if mine is None or base is None else f"{mine - base:+.4f}")
            lines.append(f"| {result.condition} | " + " | ".join(deltas) + " |")

    excluded = sorted({date for result in results for date in result.excluded_dates})
    gold_missing = sum(row.gold_missing for result in results for row in result.per_date)
    pool_sizes = [row.candidates for result in results for row in result.per_date]

    lines += [
        "",
        "## 실행 조건 (§5.6 — 표 아래에 반드시 기록)",
        "",
        f"- 생성 시각: {datetime.now(UTC).isoformat(timespec='seconds')}",
        f"- 장치: `{device}` (models.yaml `device.verified_equivalent: "
        f"{device_cfg.get('verified_equivalent')}` — cpu↔mps 수치 동일 확인)",
        f"- 임베더: 실제 사용 `{meta.get('embedder')}` / models.yaml 카탈로그 "
        f"`{embedder_cfg.get('repo', '미상')}` @ `{catalog_revision}`",
        f"- 리랭커: 실제 사용 `{meta.get('reranker')}` / models.yaml 카탈로그 "
        f"`{reranker_cfg.get('repo', '미상')}` @ `{reranker_cfg.get('revision', '미상')}`, "
        f"`max_length={reranker_cfg.get('max_length', '미상')}`, "
        f"`use_raw_logits={reranker_cfg.get('use_raw_logits')}` (§9.4 — raw 로짓 계약)",
        f"- 임베더 `max_seq_length`: {max_seq_text}",
        f"- 점수 합성: {meta.get('composition')}, `penalty_weight={meta.get('penalty_weight')}` "
        "(§5.2 — M1에서 정하도록 열려 있던 값. 여기 적힌 값이 실제로 쓴 값입니다)",
        f"- 1차 선별: `stage1_top_n={meta.get('top_n')}`, "
        f"`min_score_threshold={meta.get('min_score')}` "
        "(§5.7 — 0.42 는 잠정치입니다. 정답/오답 점수 분포를 보고 정답의 95%를 살리는 "
        "지점으로 다시 잡은 뒤 그 근거를 이 문서에 남기세요)",
        f"- 프로파일: `{meta.get('profile_path')}`"
        + (f" / 한국어 대조군 `{meta.get('ko_profile_path')}`" if meta.get("ko_profile_path") else ""),
        f"- 골드셋: `{goldset.path}` (생성 {goldset.generated_at}, {len(goldset.dates)}일치)",
    ]

    if goldset.miss_rate is None:
        lines.append(
            "- **1패스 라벨링 누락률: 미실측** — `python -m eval.label recheck` 를 돌리지 "
            "않았습니다. 제목만 보고 버린 것 중 관련 논문이 얼마나 섞였는지 모르는 상태이고, "
            "그만큼 Hit@5 의 상한이 낙관적일 수 있습니다."
        )
    else:
        lines.append(
            f"- 1패스 라벨링 누락률: **{goldset.miss_rate:.4f}** "
            f"(표본 {goldset.recheck_sampled}건 중 {goldset.recheck_flipped}건이 관련으로 뒤집힘)"
        )

    if excluded:
        lines.append(
            f"- **평가에서 제외한 날: {', '.join(excluded)}** — 정답이 0건이라 Hit@5·MRR·nDCG 가 "
            "정의되지 않습니다 (§5.4). 0.0 으로 넣으면 '다 놓친 날'과 구분이 사라집니다."
        )
    else:
        lines.append("- 정답 0건으로 제외한 날: 없음")

    if gold_missing:
        lines.append(
            f"- **주의: 정답인데 후보 풀에 없는 건수 {gold_missing}건** — 랭킹의 문제가 아니라 "
            "수집·중복제거 쪽 문제입니다. 지표를 구조적으로 끌어내리므로 원인을 먼저 확인하세요."
        )

    lines += [
        "",
        f"- 후보 수 열은 뺄 수 없습니다 (§9.12): 날짜별 풀 크기가 "
        f"{min(pool_sizes, default=0)}~{max(pool_sizes, default=0)}건으로 다릅니다. "
        "후보 수 없이 날짜 평균을 내면 큰 풀의 날이 구조적으로 불리합니다.",
        "- 동점은 (점수 내림차순, item_id 오름차순)으로 고정했습니다. 같은 명령을 두 번 "
        "돌리면 같은 수치가 나와야 합니다 (§5.8).",
        "",
    ]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m eval.run_eval",
        description="M1 랭킹 평가 — Hit@5 / MRR / nDCG@10 조건 비교 (기획안_2 §5.4~5.6)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="조건:\n" + "\n".join(f"  {key:<14} {value}" for key, value in CONDITIONS.items()),
    )
    parser.add_argument("--conditions", nargs="+", choices=sorted(CONDITIONS), default=None)
    parser.add_argument("--device", choices=["mps", "cpu"], default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--goldset", type=Path, default=None)
    parser.add_argument("--candidates-dir", type=Path, default=None)
    parser.add_argument("--models", type=Path, default=None, help="기본 eval/models.yaml")
    parser.add_argument("--cache-dir", type=Path, default=None, help="§5.3 임베딩 캐시 위치")
    parser.add_argument("--penalty-weight", type=float, default=None, help="§5.2 감점 계수")
    parser.add_argument(
        "--embedder",
        default=None,
        help="models.yaml 의 임베더 id (기본값). 'hash_stub' 은 배선 확인 전용이며 "
        "이걸로 낸 수치는 거짓입니다 (작업규약 §8-9)",
    )
    parser.add_argument("--reranker", default=None, help="models.yaml 의 리랭커 id (기본값)")
    args = parser.parse_args(argv)

    path = run_conditions(
        args.conditions,
        device=args.device,
        goldset_path=args.goldset,
        candidates_dir=args.candidates_dir,
        models_path=args.models,
        cache_dir=args.cache_dir,
        out=args.out,
        embedder_kind=args.embedder,
        reranker_kind=args.reranker,
        penalty_weight=args.penalty_weight,
    )
    print(f"{path} 를 썼습니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
