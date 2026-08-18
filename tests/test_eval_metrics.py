"""M1 평가 지표와 조건 러너 게이트 (기획안_2 §5.4~5.6, §5.8).

★ 이 파일의 지표 테스트를 비활성화하지 마세요. Hit@5 / MRR / nDCG@10 수치가
  이 프로젝트의 포트폴리오 가치 전부입니다 (기획안_2 §1). 정의가 조용히 어긋나면
  수치는 계속 나오는데 아무 의미가 없어집니다 — 빨간불로 위장하지 않는 종류의 사고입니다.

기대값은 전부 **손으로 계산했습니다.** 각 테스트 docstring 에 계산 과정이 있습니다.
프로덕션 코드에서 같은 식을 다시 부르면 그건 검증이 아니라 동어반복입니다.

스텁 임베더에 대하여
--------------------
`PlantedEmbedder` 는 **배선 확인 전용**입니다. 작업규약 §8-9 는 `hash_stub` 으로 랭킹
**품질**을 재는 것을 금지합니다 — 그걸로 낸 Hit@5 는 거짓입니다. 여기서 하는 일은
"프로파일이 실제로 로드되는가 · 감점이 실제로 적용되는가 · 리랭커가 실제로 불리는가"
같은 배선 확인이며, 심어둔 신호가 그대로 나오는지만 봅니다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pytest
import yaml

from eval import run_eval
from eval.run_eval import (
    CONDITIONS,
    ConditionResult,
    baseline_scores,
    content_tokens,
    embedding_ranking,
    evaluate_condition,
    hit_at_k,
    load_goldset,
    mrr,
    ndcg_at_10,
    noun_phrase_terms,
    rank_items,
    run_conditions,
)
from src.rank.embed import build_embedder
from src.rank.profile import build_queryset
from src.rank.retrieve import stage1_settings

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_YAML = REPO_ROOT / "eval" / "models.yaml"


# ══════════════════════════════════════════════════════════════════════════
# 지표 — 손으로 계산한 소형 케이스 (§5.4)
# ══════════════════════════════════════════════════════════════════════════

#: 5건 중 정답 2건이 **순위 2와 4**. 아래 세 테스트가 전부 이 케이스를 씁니다.
RANKED_5 = ["r1", "r2", "r3", "r4", "r5"]
GOLD_2_AND_4 = {"r2", "r4"}


def test_hit_at_5_hand_computed():
    """★ Hit@5 = `1 if R[:5] ∩ G ≠ ∅ else 0` (§5.4).

    손계산: 정답 r2 가 순위 2 → 상위 5에 있음 → 1.0.
            정답이 순위 6뿐이면 상위 5 밖 → 0.0.

    깨뜨리는 법: run_eval.hit_at_k 의 `ranked[:k]` 를 `ranked` 로 바꾸면
    두 번째 assert 가 1.0 을 받아 빨간불.
    확인일: 2026-08-18
    """
    assert hit_at_k(RANKED_5, GOLD_2_AND_4) == 1.0
    assert hit_at_k(["a", "b", "c", "d", "e", "정답"], {"정답"}) == 0.0
    assert hit_at_k(["a", "b", "c", "d", "정답"], {"정답"}) == 1.0


def test_mrr_hand_computed():
    """★ MRR = 1 / 첫 정답의 순위 (§5.4).

    손계산: 정답이 순위 2와 4 → 첫 정답은 2 → 1/2 = 0.5.
            정답이 순위 1이면 1/1 = 1.0, 순위 3이면 1/3.

    깨뜨리는 법: run_eval.mrr 의 `enumerate(window, start=1)` 을 `start=0` 으로
    바꾸면 ZeroDivisionError 또는 1.0 이 나와 빨간불.
    확인일: 2026-08-18
    """
    assert mrr(RANKED_5, GOLD_2_AND_4) == pytest.approx(0.5)
    assert mrr(RANKED_5, {"r1"}) == pytest.approx(1.0)
    assert mrr(RANKED_5, {"r3", "r5"}) == pytest.approx(1 / 3)


def test_mrr_is_zero_not_infinity_when_gold_is_absent():
    """★ 정답이 랭킹에 하나도 없으면 **0** 입니다 — `1/∞` 가 아닙니다 (§5.4).

    후보 풀에서 사라진 정답(수집·중복제거 사고)이 있는 날, 이 규칙이 없으면
    예외나 inf 가 평균을 오염시킵니다.

    깨뜨리는 법: run_eval.mrr 의 마지막 `return 0.0` 을 `return float("inf")` 로
    바꾸면 빨간불.
    확인일: 2026-08-18
    """
    assert mrr(RANKED_5, {"랭킹에-없는-정답"}) == 0.0
    assert mrr([], {"정답"}) == 0.0


def test_ndcg_at_10_hand_computed():
    """★ nDCG@10, 이진 relevance, IDCG 는 `min(|G|,10)` 기준 (§5.4).

    손계산 (정답 2건이 순위 2, 4):
      DCG  = 1/log2(2+1) + 1/log2(4+1) = 0.6309297536 + 0.4306765581 = 1.0616063117
      IDCG = 1/log2(1+1) + 1/log2(2+1) = 1.0          + 0.6309297536 = 1.6309297536
      nDCG = 1.0616063117 / 1.6309297536 = 0.6509209298

    깨뜨리는 법: run_eval.ndcg_at_10 의 `math.log2(position + 1)` 을 `log2(position)`
    으로 바꾸면 순위 1에서 ZeroDivisionError, 값도 달라져 빨간불.
    확인일: 2026-08-18
    """
    assert ndcg_at_10(RANKED_5, GOLD_2_AND_4) == pytest.approx(0.6509209298, abs=1e-9)
    # 정답 2건이 순위 1, 2 (완벽한 랭킹) → DCG == IDCG → 1.0
    assert ndcg_at_10(RANKED_5, {"r1", "r2"}) == pytest.approx(1.0)


def test_ndcg_idcg_is_capped_at_ten_gold_items():
    """★ 정답이 10건보다 많아도 IDCG 는 10건 기준입니다 (§5.4 `min(|G_d|,10)`).

    상한을 `|G|` 로 잡으면 정답이 많은 날의 nDCG 가 구조적으로 1.0 에 못 미쳐
    "정답이 많은 날일수록 랭커가 못한다"는 착시가 생깁니다.

    손계산: 정답 15건 중 15건이 전부 상위 10위 안 → DCG == IDCG@10 → 1.0.

    깨뜨리는 법: run_eval.ndcg_at_10 의 `min(len(gold), k)` 를 `len(gold)` 로
    바꾸면 0.7 대가 나와 빨간불.
    확인일: 2026-08-18
    """
    ranked = [f"g{i:02d}" for i in range(15)]
    assert ndcg_at_10(ranked, set(ranked)) == pytest.approx(1.0)


def test_zero_gold_day_returns_none_from_every_metric():
    """★ 정답 0건인 날은 세 지표 모두 `None` — 계산에서 제외한다는 뜻입니다 (§5.4).

    0.0 을 돌려주면 "정답이 없는 날"과 "전부 놓친 날"이 같은 값이 되어 평균이 왜곡되고,
    nDCG 는 IDCG=0 이라 정의 자체가 안 됩니다.

    깨뜨리는 법: run_eval.hit_at_k 의 `if not gold: return None` 을 `return 0.0` 으로
    바꾸면 빨간불 (그리고 아래 제외 테스트도 같이 빨간불).
    확인일: 2026-08-18
    """
    assert hit_at_k(RANKED_5, set()) is None
    assert mrr(RANKED_5, set()) is None
    assert ndcg_at_10(RANKED_5, set()) is None


def test_zero_gold_day_is_excluded_and_reported():
    """★ 정답 0건인 날은 평균에서 빠지고 **제외 사실이 보고**돼야 합니다 (§5.4).

    손계산: 12일 Hit@5=1, 13일은 정답 0건.
            제외하면 평균 = 1/1 = 1.0. 제외하지 않고 0으로 넣으면 0.5 가 됩니다.

    깨뜨리는 법: run_eval.evaluate_condition 의 `if hit is None: excluded.append(...)`
    분기를 지우면 TypeError 또는 평균 0.5 로 빨간불.
    확인일: 2026-08-18
    """
    result = evaluate_condition(
        "t",
        rankings={"2026-08-12": ["a", "b"], "2026-08-13": ["c", "d"]},
        gold={"2026-08-12": {"a"}, "2026-08-13": set()},
        candidate_counts={"2026-08-12": 2, "2026-08-13": 2},
    )
    assert [row.date for row in result.per_date] == ["2026-08-12"]
    assert result.excluded_dates == ("2026-08-13",)
    assert result.mean("hit_at_5") == pytest.approx(1.0)


def test_metrics_are_averaged_per_date_not_pooled():
    """★ **날짜별로 계산한 뒤 평균**입니다. 전체를 한 풀로 합치면 안 됩니다 (§5.4).

    손계산: 12일 정답 1건을 1위로 맞춤(Hit@5=1) / 13일 정답 3건을 전부 놓침(Hit@5=0).
            날짜 평균 = (1 + 0) / 2 = 0.5.
            정답 수로 가중하면 (1×1 + 0×3) / 4 = 0.25 — 이게 "풀을 합친" 값입니다.
    운영이 하루 단위로 도는데 풀을 합치면 후보 719건인 날이 350건인 날을 삼킵니다.

    깨뜨리는 법: run_eval.ConditionResult.mean 을 `relevant` 가중평균으로 바꾸면
    0.25 가 나와 빨간불.
    확인일: 2026-08-18
    """
    result = evaluate_condition(
        "t",
        rankings={
            "2026-08-12": ["맞춤", "x", "y"],
            "2026-08-13": ["a", "b", "c", "d", "e", "정답1", "정답2", "정답3"],
        },
        gold={"2026-08-12": {"맞춤"}, "2026-08-13": {"정답1", "정답2", "정답3"}},
        candidate_counts={"2026-08-12": 3, "2026-08-13": 8},
    )
    assert result.mean("hit_at_5") == pytest.approx(0.5)


def test_ties_break_by_item_id_and_are_stable_across_runs():
    """★ 동점은 (점수 내림차순, item_id 오름차순) 고정 (§5.4).

    정렬이 불안정하면 같은 조건을 두 번 돌렸을 때 수치가 달라집니다 (§5.8 재현성).
    dict 삽입 순서에 기대면 후보 파일 줄 순서가 바뀌는 것만으로 Hit@5 가 흔들립니다.

    손계산: b·a·c 가 전부 0.5 로 동점 → a, b, c 순. z 는 0.9 라 맨 앞.

    깨뜨리는 법: run_eval.rank_items 의 key 에서 `kv[0]` 을 빼고 `-kv[1]` 만 남기면
    삽입 순서(b, a, c)가 그대로 나와 빨간불.
    확인일: 2026-08-18
    """
    scores = {"b": 0.5, "a": 0.5, "z": 0.9, "c": 0.5}
    assert rank_items(scores) == ["z", "a", "b", "c"]
    # 삽입 순서를 뒤집어도 같은 결과여야 합니다
    assert rank_items(dict(reversed(list(scores.items())))) == ["z", "a", "b", "c"]


# ══════════════════════════════════════════════════════════════════════════
# 베이스라인 — §5.5. 약한 베이스라인은 개선폭을 과장합니다
# ══════════════════════════════════════════════════════════════════════════


def test_baseline_tokenizer_drops_short_tokens_and_stopwords():
    """소문자 정규화 + 3자 이상 토큰 (§5.5 / 과제 지시).

    깨뜨리는 법: run_eval.MIN_TOKEN_LEN 을 1 로 내리면 `we`·`of` 가 남아 빨간불.
    확인일: 2026-08-18
    """
    tokens = content_tokens("We study Cross-Encoder RERANKING for the retrieval of documents")
    assert tokens == ["study", "cross", "encoder", "reranking", "retrieval", "documents"]


def test_baseline_extracts_two_word_phrases_not_only_words():
    """명사구 근사 = 단어 + 인접 2단어 구 (§5.5).

    손계산: "identity preserving edit" →
      단어 {identity, preserving, edit} + 구 {"identity preserving", "preserving edit"} = 5개.
    같은 단어를 순서만 바꿔 나열한 문서는 구가 안 맞아 3개만 걸립니다.

    깨뜨리는 법: run_eval.noun_phrase_terms 의 bigram 갱신 줄을 지우면 두 문서가
    똑같이 3점이 되어 빨간불.
    확인일: 2026-08-18
    """
    interests = [{"weight": 1.0, "text": "identity preserving edit"}]
    scores = baseline_scores(
        [
            {"id": "구까지일치", "title": "identity preserving edit", "abstract": ""},
            {"id": "단어만일치", "title": "edit. preserving. identity.", "abstract": ""},
        ],
        interests,
    )
    assert scores["구까지일치"] == pytest.approx(5.0)
    assert scores["단어만일치"] == pytest.approx(3.0)


def test_baseline_counts_matches_and_applies_interest_weight():
    """★ `Σ weight_i × 매칭 term 수` (§5.5).

    손계산 — 관심사 a(weight 1.0) = "cross encoder reranking for retrieval evaluation"
      terms: {cross, encoder, reranking, retrieval, evaluation,
              "cross encoder", "encoder reranking", "reranking retrieval",
              "retrieval evaluation"}
      문서 hit ("Cross encoder reranking" + "We study retrieval evaluation.") 와의 교집합:
      단어 5개 + 구 3개("cross encoder", "encoder reranking", "retrieval evaluation") = 8
      → 8 × 1.0 = 8.0
    관심사 b(weight 0.5) = "protein folding structure prediction"
      문서 other ("Protein folding" + "We predict structure.") 와 교집합:
      단어 3개(protein, folding, structure) + 구 1개("protein folding") = 4 → 4 × 0.5 = 2.0
    아무것도 안 걸리는 문서는 0.0.

    깨뜨리는 법: run_eval.baseline_scores 에서 `weight *` 를 지우면 other 가 4.0 이
    되어 빨간불.
    확인일: 2026-08-18
    """
    interests = [
        {"weight": 1.0, "text": "cross encoder reranking for retrieval evaluation"},
        {"weight": 0.5, "text": "protein folding structure prediction"},
    ]
    scores = baseline_scores(
        [
            {"id": "hit", "title": "Cross encoder reranking", "abstract": "We study retrieval evaluation."},
            {"id": "other", "title": "Protein folding", "abstract": "We predict structure."},
            {"id": "none", "title": "Quantum widget", "abstract": "Nothing at all."},
        ],
        interests,
    )
    assert scores["hit"] == pytest.approx(8.0)
    assert scores["other"] == pytest.approx(2.0)
    assert scores["none"] == pytest.approx(0.0)


def test_baseline_is_not_a_straw_man():
    """★ 베이스라인이 무의미하면 개선폭이 과장됩니다 (§5.5).

    실제 운영 프로파일(`data/profile.papers_1.yaml`)의 관심사 문장으로, 관심사에
    정면으로 해당하는 초록이 무관한 초록보다 높은 점수를 받아야 합니다.
    프로파일은 R1 대로 `load_profile()` 로 읽습니다 — 경로를 하드코딩하면
    `profile.papers_2.yaml` 이 생기는 순간 구버전을 읽습니다 (§9.9).

    깨뜨리는 법: run_eval.baseline_scores 가 항상 0.0 을 돌려주게 하면 빨간불.
    확인일: 2026-08-18
    """
    from src.core.config import load_profile

    profile, _ = load_profile("profile.papers", root=REPO_ROOT / "data")
    scores = baseline_scores(
        [
            {
                "id": "관련",
                "title": "Cross-encoder reranking for retrieval-augmented generation",
                "abstract": (
                    "We evaluate hybrid retrieval combining sparse BM25 with dense embeddings "
                    "and cross-encoder reranking, reporting Hit@k, MRR and nDCG."
                ),
            },
            {
                "id": "무관",
                "title": "A convergence proof for stochastic mirror descent",
                "abstract": "We establish asymptotic convergence rates under mild assumptions.",
            },
        ],
        profile["interests"],
    )
    assert scores["관련"] > scores["무관"]


# ══════════════════════════════════════════════════════════════════════════
# 랭킹 위임 — 점수 합성 자체는 src/rank/profile.py 의 테스트가 봅니다.
# 여기서 보는 것은 **평가가 그 결과를 잘라먹지 않는가** 입니다.
# ══════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════
# 조건 러너 end-to-end — 심은 신호로 배선만 확인 (작업규약 §8-9)
# ══════════════════════════════════════════════════════════════════════════

#: 날짜별 (후보 수, 정답 수). 후보 수를 일부러 다르게 뒀습니다 — §9.12 의
#: "첫날 719건 / 이후 ~350건" 상황을 축소한 것이고, results.md 의 `후보 수` 열이
#: 실제로 날짜마다 다른 값을 싣는지 확인합니다. 마지막 날은 정답 0건(제외 대상).
FIXTURE_DATES: dict[str, tuple[int, int]] = {
    "2026-08-12": (8, 2),
    "2026-08-13": (6, 1),
    "2026-08-14": (4, 0),
}

#: 심은 신호. 정답 문서에만 들어갑니다.
SIGNAL = "signal"


class PlantedEmbedder:
    """배선 확인용 결정적 임베더 (품질 측정 금지 — 작업규약 §8-9).

    `src.rank.embed.Embedder` 포트를 만족합니다 (`model_id`·`revision`·`encode`).
    3차원 직교 벡터로 텍스트를 분류합니다:
      "signal"  포함 → [1,0,0]   (영어 프로파일의 핵심 관심사 · 정답 문서)
      "exclude" 포함 → [0,1,0]   (영어 프로파일의 감점 대상)
      그 외            → [0,0,1]   (무관 문서 · 한국어 프로파일 전체)

    한국어 프로파일에는 ASCII 마커가 없으므로 관심사가 전부 [0,0,1] 이 됩니다.
    즉 **영어 프로파일을 쓰면 정답이 1위, 한국어 프로파일을 쓰면 정답이 꼴찌**가
    되도록 심어 뒀습니다. 조건 러너가 두 프로파일을 실제로 갈아끼우는지 확인하는
    장치입니다 (교차언어 손실을 "측정"한 것이 아닙니다).
    """

    model_id = "planted_stub"
    revision = "planted0"
    max_seq_length = 512

    def __init__(self, device: str | None = None) -> None:
        self.device = device
        self.calls = 0

    def encode(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        self.calls += 1
        vectors: list[tuple[float, ...]] = []
        for text in texts:
            lowered = text.lower()
            if SIGNAL in lowered:
                vectors.append((1.0, 0.0, 0.0))
            elif "exclude" in lowered:
                vectors.append((0.0, 1.0, 0.0))
            else:
                vectors.append((0.0, 0.0, 1.0))
        return vectors


class PlantedReranker:
    """정답 문서에 큰 raw 로짓을 주는 스텁 (`src.rank.rerank.Reranker` 포트).

    **sigmoid 를 씌우지 않습니다** — 음수 로짓이 나오는 것이 의도입니다 (§9.4).
    반환값을 확률로 착각하는 코드가 있으면 여기서 드러나야 합니다.
    """

    model_id = "planted_reranker"
    revision = "planted0"

    def __init__(self, device: str | None = None) -> None:
        self.device = device
        self.pairs_seen = 0

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        self.pairs_seen += len(pairs)
        return [3.0 if SIGNAL in doc.lower() else -7.0 for _query, doc in pairs]


def _write_profiles(root: Path) -> None:
    """tmp 프로파일 2종. `_1` 접미사를 붙여 **번호 해석까지** 태웁니다 (R1 / §9.9)."""
    root.mkdir(parents=True, exist_ok=True)
    english = {
        "version": 1,
        "channel": "papers",
        "language": "en",
        "interests": [
            {"id": "core", "label": "핵심", "weight": 1.0, "text": f"radar eval {SIGNAL} interest"},
            {"id": "side", "label": "주변", "weight": 0.5, "text": "radar eval other interest"},
        ],
        "exclude": {"soft": [{"text": "radar eval exclude topic", "penalty": 0.5}]},
        "selection": {"stage1_top_n": 3, "final_n": 2},
    }
    korean = {
        "version": 1,
        "channel": "papers",
        "language": "ko",
        "role": "control_group",
        "interests": [
            {"id": "core", "label": "핵심", "weight": 1.0, "text": "레이더 평가 핵심 관심사"},
            {"id": "side", "label": "주변", "weight": 0.5, "text": "레이더 평가 주변 관심사"},
        ],
        "exclude": {"soft": [{"text": "레이더 평가 감점 대상", "penalty": 0.5}]},
    }
    (root / "profile.papers_1.yaml").write_text(
        yaml.safe_dump(english, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    (root / "profile.papers.ko.yaml").write_text(
        yaml.safe_dump(korean, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


def _write_candidates_and_goldset(candidates_dir: Path, goldset_path: Path) -> None:
    candidates_dir.mkdir(parents=True, exist_ok=True)
    labels: list[dict[str, Any]] = []
    per_date: dict[str, dict[str, int]] = {}
    for date, (total, gold_count) in FIXTURE_DATES.items():
        rows = []
        for index in range(total):
            is_gold = index < gold_count
            item_id = f"arxiv:{date}-{index:02d}"
            rows.append(
                {
                    "id": item_id,
                    "source": "arxiv",
                    "channel": "papers",
                    "title": f"paper {index}",
                    "abstract": f"this abstract carries the {SIGNAL} marker"
                    if is_gold
                    else "this abstract is about unrelated quantum widgets",
                    "url": f"https://arxiv.org/abs/{date}-{index:02d}",
                    "published": f"{date}T00:00:00+00:00",
                    "publish_scope": "public",
                    "categories": ["cs.IR"],
                }
            )
            labels.append(
                {
                    "date": date,
                    "item_id": item_id,
                    "relevant": is_gold,
                    "basis": "abstract" if is_gold else "title",
                    "title": f"paper {index}",
                }
            )
        (candidates_dir / f"{date}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )
        per_date[date] = {"candidates": total, "relevant": gold_count}

    goldset = {
        "version": 1,
        "channel": "papers",
        "criteria_doc": "docs/라벨링_기준.md",
        "generated_at": "2026-08-18T00:00:00+00:00",
        "summary": {
            "dates": sorted(FIXTURE_DATES),
            "total": len(labels),
            "relevant": sum(count for _, count in FIXTURE_DATES.values()),
            "per_date": per_date,
            "title_pass_recheck": {"sampled": 40, "flipped_to_relevant": 2, "miss_rate": 0.05},
        },
        "labels": labels,
    }
    goldset_path.parent.mkdir(parents=True, exist_ok=True)
    goldset_path.write_text(
        yaml.safe_dump(goldset, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


@pytest.fixture
def eval_env(tmp_path: Path) -> dict[str, Path]:
    _write_profiles(tmp_path / "data")
    _write_candidates_and_goldset(tmp_path / "candidates", tmp_path / "goldset.yaml")
    return {
        "profile_root": tmp_path / "data",
        "candidates_dir": tmp_path / "candidates",
        "goldset": tmp_path / "goldset.yaml",
        "out": tmp_path / "results.md",
        # 테스트가 실제 `data/cache/emb/` 에 쓰지 않게 합니다. 픽스처 텍스트의 벡터가
        # 운영 캐시에 섞이면 지울 이유를 아무도 모르는 쓰레기가 남습니다.
        "cache_dir": tmp_path / "emb",
    }


def _run(eval_env: dict[str, Path], **overrides: Any) -> Path:
    kwargs: dict[str, Any] = dict(
        conditions=["baseline", "bge_m3", "bge_m3_rerank", "bge_m3_ko"],
        device="cpu",
        goldset_path=eval_env["goldset"],
        candidates_dir=eval_env["candidates_dir"],
        models_path=MODELS_YAML,
        profile_root=eval_env["profile_root"],
        cache_dir=eval_env["cache_dir"],
        out=eval_env["out"],
        embedder=PlantedEmbedder(),
        reranker=PlantedReranker(),
    )
    kwargs.update(overrides)
    conditions = kwargs.pop("conditions")
    return run_conditions(conditions, **kwargs)


def test_run_conditions_writes_every_condition_and_date(eval_env):
    """★ 조건 4개 × 날짜(정답 0건인 날 제외)가 전부 표에 있어야 합니다 (§5.8 DoD).

    깨뜨리는 법: run_eval.run_conditions 의 `for name in selected:` 루프에서
    `results.append(...)` 를 조건부로 만들면(예: baseline 만) 빨간불.
    확인일: 2026-08-18
    """
    table = _run(eval_env).read_text(encoding="utf-8")
    for condition in ("baseline", "bge_m3", "bge_m3_rerank", "bge_m3_ko"):
        assert f"| {condition} | 2026-08-12 |" in table, condition
        assert f"| {condition} | 2026-08-13 |" in table, condition
        # 2026-08-14 는 정답 0건 → 표에 행이 없어야 합니다 (§5.4)
        assert f"| {condition} | 2026-08-14 |" not in table, condition
        assert f"**{condition} 평균**" in table
    assert CONDITIONS["bge_m3_rerank"] in table


def test_results_table_keeps_the_candidate_count_column(eval_env):
    """★ `후보 수` 열이 없으면 표가 무효입니다 (§9.12 / 작업규약 §7.3).

    첫날 719건 · 이후 ~350건으로 풀 크기가 2배 다릅니다. 이 열 없이 날짜 평균을
    내면 큰 풀의 날이 구조적으로 불리하다는 사실이 표에서 사라집니다.

    손계산: 픽스처는 12일 8건 / 13일 6건 → 두 값이 그대로 실려야 합니다.

    깨뜨리는 법: run_eval.RESULT_COLUMNS 에서 "후보 수" 를 빼고 render_results 의
    `str(row.candidates)` 를 지우면 빨간불.
    확인일: 2026-08-18
    """
    table = _run(eval_env).read_text(encoding="utf-8")
    header = next(line for line in table.splitlines() if line.startswith("| 조건 |"))
    assert "후보 수" in header
    assert "정답 수" in header
    assert "| baseline | 2026-08-12 | 8 | 2 |" in table
    assert "| baseline | 2026-08-13 | 6 | 1 |" in table


def test_results_records_run_conditions_below_the_table(eval_env):
    """★ 표 아래 실행 조건 — 장치·리비전·max_seq_length·합성 방식·1패스 누락률 (§5.6).

    조건 없는 Hit@5 는 3개월 뒤 재현이 불가능합니다.

    깨뜨리는 법: render_results 의 "실행 조건" 블록에서 아무 줄이나 지우면 빨간불.
    확인일: 2026-08-18
    """
    table = _run(eval_env).read_text(encoding="utf-8")
    models = yaml.safe_load(MODELS_YAML.read_text(encoding="utf-8"))
    assert "- 장치: `cpu`" in table
    assert models["embedder"]["revision"] in table  # R8 — 해시를 타이핑하지 않고 읽어옵니다
    assert str(models["reranker"]["revision"]) in table
    assert "`max_seq_length`: 512" in table
    assert "penalty_weight=1.0" in table
    assert "stage1_top_n=3" in table
    assert "1패스 라벨링 누락률: **0.0500**" in table
    assert "제외한 날: 2026-08-14" in table


def test_stub_run_is_marked_invalid_and_never_shows_the_catalog_revision(eval_env):
    """★ 스텁으로 돌린 표가 실모델 표로 보이면 안 됩니다 (작업규약 §8-9).

    `hash_stub`·`FakeReranker` 로 낸 Hit@5 는 거짓입니다. 그런데 `임베더 revision` 열에
    models.yaml 의 카탈로그 해시를 찍으면, 몇 달 뒤 이 표를 보는 사람은 bge-m3 실측치로
    인용합니다. **표에는 실제로 쓴 구현체의 리비전이 들어가야 합니다.**

    손계산: 픽스처는 `PlantedEmbedder(revision="planted0")` 로 돕니다 →
            열에는 `planted0` 이, 카탈로그 해시 `5617a9f6…` 는 열에 없어야 합니다.

    깨뜨리는 법: run_eval.render_results 의 `revision` 을 `catalog_revision` 으로
    되돌리면 표 행에 카탈로그 해시가 찍혀 빨간불. 스텁 경고 블록을 지워도 빨간불.
    확인일: 2026-08-18
    """
    table = _run(eval_env).read_text(encoding="utf-8")
    models = yaml.safe_load(MODELS_YAML.read_text(encoding="utf-8"))
    catalog = models["embedder"]["revision"][:8]

    assert "이 표의 수치는 무효입니다" in table
    assert "planted_stub" in table and "planted_reranker" in table

    rows = [line for line in table.splitlines() if line.startswith("| baseline |")]
    assert rows, "표 행을 찾지 못했습니다"
    for row in rows:
        assert row.rstrip().endswith("| planted0 |"), row
        assert catalog not in row, f"카탈로그 해시가 표 행에 찍혔습니다: {row}"


def test_ko_condition_actually_loads_the_korean_control_profile(eval_env):
    """★ 한국어 조건이 영어 프로파일을 그대로 재사용하면 교차언어 대조가 무의미해집니다.

    심은 신호(§ 클래스 docstring): 영어 프로파일이면 정답이 1·2위, 한국어 프로파일이면
    정답이 꼴찌. 따라서 bge_m3 는 Hit@5=1.0, bge_m3_ko 는 Hit@5=0.0 이어야 합니다.
    (교차언어 성능을 **측정**한 것이 아니라 배선을 확인한 것입니다 — 작업규약 §8-9)

    손계산 (12일, 후보 8건 중 정답 2건):
      영어: 정답 벡터 [1,0,0]·핵심관심사 [1,0,0] → base 1.0, 감점 0 → 1위·2위 → Hit@5=1
      한국어: 관심사가 전부 [0,0,1] → 무관 문서(6건)가 0.5, 정답(2건)이 0.0 →
              정답은 7·8위 → Hit@5=0
    깨뜨리는 법: run_eval.run_conditions 에서 `ko_profile if name == "bge_m3_ko" else profile`
    을 `profile` 로 바꾸면 두 조건이 같아져 빨간불.
    확인일: 2026-08-18
    """
    table = _run(eval_env).read_text(encoding="utf-8")
    english_row = next(line for line in table.splitlines() if line.startswith("| bge_m3 | 2026-08-12 |"))
    korean_row = next(line for line in table.splitlines() if line.startswith("| bge_m3_ko | 2026-08-12 |"))
    assert english_row.split("|")[5].strip() == "1.0000"  # Hit@5
    assert korean_row.split("|")[5].strip() == "0.0000"


def test_reranker_is_actually_called_on_stage1_top_n(eval_env):
    """★ `bge_m3_rerank` 조건이 리랭커를 실제로 부르는지. 안 부르면 조건이 하나 사라집니다.

    손계산: tmp 프로파일의 `stage1_top_n` 은 3, 평가 대상 날짜는 3일 →
            3 + 3 + 3 = 9쌍 (정답 0건인 날도 랭킹은 계산합니다).

    깨뜨리는 법: run_conditions 의 `reranker=reranker if name == "bge_m3_rerank" else None`
    을 `None` 으로 바꾸면 pairs_seen 이 0 이 되어 빨간불.
    확인일: 2026-08-18
    """
    reranker = PlantedReranker()
    _run(eval_env, reranker=reranker)
    assert reranker.pairs_seen == 9


def test_runner_uses_the_cached_embedder_so_conditions_share_vectors(eval_env, tmp_path):
    """★ §5.3 — 조건마다 719건을 다시 인코딩하면 조건 비교를 끝까지 못 돌립니다.

    "조용히 망가지는 것은 속도가 아니라 **비교 조건의 수**" 입니다 (§5.3). 캐시가
    없으면 사람이 4조건을 다 안 돌리고 한두 개로 끝냅니다.

    캐시는 `src.rank.embed.CachedEmbedder` 의 책임이고, 평가 러너는 **그 캐시가
    붙은 임베더를 그대로 써야** 합니다. 임베딩 조건 3개가 같은 문서를 보므로
    2·3번째 조건은 전부 캐시 적중이어야 합니다.

    이 테스트는 `src/rank` 의 실제 `HashStubEmbedder` 로 조건 러너를 end-to-end 로
    태우는 역할도 합니다 (포트 적합성 확인). 여기서 나온 수치는 의미가 없습니다 —
    `hash_stub` 으로 랭킹 품질을 재는 것은 금지입니다 (작업규약 §8-9).

    깨뜨리는 법: run_eval.run_conditions 에서 조건마다 `build_embedder(...)` 를 새로
    만들도록 바꾸면 (캐시 디렉터리는 같아도 hits 가 0인 인스턴스가 생겨) 빨간불.
    확인일: 2026-08-18
    """
    embedder = build_embedder("hash_stub", cache_dir=tmp_path / "emb", dim=32)
    _run(
        eval_env,
        conditions=["bge_m3", "bge_m3_wmean", "bge_m3_rerank"],
        embedder=embedder,
        reranker=PlantedReranker(),
    )
    assert embedder.hits > 0, "조건 간 임베딩 캐시가 전혀 먹지 않았습니다 (§5.3)"
    assert embedder.misses > 0  # 첫 조건은 당연히 미스여야 합니다 (캐시가 비어 있었으므로)


class IndexReranker:
    """문서 제목의 번호로 raw 로짓을 주는 스텁 (`src.rank.rerank.Reranker` 포트).

    d5·d6 의 로짓 순서를 **1차 순서와 뒤집어** 뒀습니다. 리랭킹이 상위 5건에서
    잘리면 d5·d6 이 재정렬 결과가 아니라 1차 순서로 되돌아가므로, 잘림이 순서에
    드러납니다. 길이만 재면 안 보입니다 — 잘린 것도 꼬리에 다시 붙기 때문입니다.
    """

    model_id = "index_reranker"
    revision = "index000"

    _LOGITS = {0: 10.0, 1: 9.0, 2: 8.0, 3: 7.0, 4: 6.0, 5: 0.0, 6: 1.0}

    def score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        # document_text 는 "paper {i}\n\n{초록}" 이므로 두 번째 토큰이 번호입니다.
        return [self._LOGITS[int(document.split()[1])] for _query, document in pairs]


def test_rerank_reorders_the_whole_stage1_head_not_only_the_final_five():
    """★ 평가는 발행 top-5 가 아니라 **1차 통과 전체**를 재정렬해야 합니다.

    `rerank_stage2` 의 `final_n` 기본값은 **발행 건수 5**입니다. 평가에서 그대로 쓰면
    6위 이하가 재정렬을 못 받고 1차 순서로 남습니다. nDCG@10 은 10위까지 보므로
    6~10위가 조용히 1차 결과인 채로 "리랭킹 조건"의 수치가 됩니다.

    손계산 (후보 8건, 전부 같은 벡터 → 1차 동점 → item_id 사전순 d0..d7,
             `stage1_top_n=7` → 1차 통과 d0..d6):
      리랭커 로짓 d0..d6 = 10,9,8,7,6,0,1 → 재정렬 d0,d1,d2,d3,d4,d6,d5
      전체 재정렬  : [d0,d1,d2,d3,d4,d6,d5] + 꼬리 [d7]
      5건만 재정렬 : [d0,d1,d2,d3,d4] + 남은 것을 1차 순서로 [d5,d6,d7]  ← d5·d6 이 뒤바뀜
    길이는 둘 다 8이라 **길이 검사로는 못 잡습니다.**

    깨뜨리는 법: run_eval.embedding_ranking 의 `final_n=len(head)` 를 지우면
    d5·d6 순서가 뒤바뀌어 빨간불. `return head_ids + [...]` 의 꼬리를 지우면
    길이 검사가 빨간불.
    확인일: 2026-08-18
    """
    profile = {
        "interests": [{"id": "core", "text": f"radar eval {SIGNAL} interest", "weight": 1.0}],
        "selection": {"stage1_top_n": 7},
    }
    assert stage1_settings(profile).top_n == 7  # 5(발행 건수)보다 커야 잘림이 드러납니다
    embedder = PlantedEmbedder()
    queryset = build_queryset(embedder, profile=profile, profile_path="tmp")
    # 전부 같은 벡터 → 1차 점수 동점 → 순서는 item_id 사전순 (§5.4)
    items = [
        {"id": f"d{index}", "title": f"paper {index}", "abstract": "unrelated quantum widgets"}
        for index in range(8)
    ]

    ranking = embedding_ranking(
        items, embedder, queryset, reranker=IndexReranker(), profile=profile
    )
    assert ranking == ["d0", "d1", "d2", "d3", "d4", "d6", "d5", "d7"]
    assert len(set(ranking)) == 8, "중복이 생겼습니다 — 꼬리에 1차 통과분이 다시 들어갔습니다"


def test_two_runs_produce_identical_numbers(eval_env, tmp_path):
    """★ 같은 명령을 두 번 돌리면 같은 수치 (§5.8 DoD 재현성).

    깨뜨리는 법: run_eval.rank_items 의 정렬 key 에서 `kv[0]` 을 빼고
    `random.shuffle` 을 끼우면 빨간불.
    확인일: 2026-08-18
    """
    first = _run(eval_env).read_text(encoding="utf-8")
    second = _run(eval_env, out=tmp_path / "results2.md").read_text(encoding="utf-8")

    def table_only(text: str) -> list[str]:
        return [line for line in text.splitlines() if line.startswith("|")]

    assert table_only(first) == table_only(second)


def test_run_conditions_refuses_without_a_goldset(eval_env, tmp_path):
    """★ 골드셋 없이는 거부하고 **라벨링 명령을 안내**합니다.

    골드셋이 없다고 랭커 상위 N건으로 대신하면 평가 대상이 정답셋의 범위를 정하게 되어
    개선폭이 실제보다 좋게 나옵니다 (§9.11 풀링 편향).

    깨뜨리는 법: run_eval.load_goldset 의 `if not path.exists(): raise SystemExit` 을
    지우면 다른 예외가 나거나 빈 결과가 나와 빨간불.
    확인일: 2026-08-18
    """
    with pytest.raises(SystemExit) as error:
        _run(eval_env, goldset_path=tmp_path / "없는파일.yaml")
    message = str(error.value)
    assert "python -m eval.label triage" in message
    assert "python -m eval.label export" in message


def test_pending_labels_in_goldset_are_rejected(eval_env):
    """보류(`relevant: null`)가 남은 골드셋은 거부합니다.

    보류를 무관으로 접으면 정답 수가 조용히 줄어 Hit@5 의 상한이 내려갑니다.
    `eval/label.py export` 가 1차로 막지만, 손으로 고친 파일이 들어올 수 있습니다.

    깨뜨리는 법: run_eval.load_goldset 의 `if row.get("relevant") is None: raise`
    를 지우면 통과해버려 빨간불.
    확인일: 2026-08-18
    """
    goldset = yaml.safe_load(eval_env["goldset"].read_text(encoding="utf-8"))
    goldset["labels"][0]["relevant"] = None
    eval_env["goldset"].write_text(
        yaml.safe_dump(goldset, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="보류"):
        _run(eval_env)


def test_unknown_condition_is_rejected(eval_env):
    """오타난 조건 이름이 조용히 무시되면 "4조건 돌렸다"가 거짓이 됩니다 (§5.8 DoD).

    깨뜨리는 법: run_conditions 의 `unknown` 검사를 지우면 통과해버려 빨간불.
    확인일: 2026-08-18
    """
    with pytest.raises(SystemExit, match="알 수 없는 조건"):
        _run(eval_env, conditions=["baseline", "bge_m4"])


def test_goldset_loader_reports_recheck_miss_rate(eval_env):
    """1패스 누락률은 골드셋에서 읽어 옵니다 — 안 했으면 `None` 이어야 합니다.

    0.0 으로 채우면 "재검토했는데 누락이 0이었다"가 되어, 안 한 일을 한 것처럼
    기록하게 됩니다 (작업규약 §7.2).

    깨뜨리는 법: run_eval.load_goldset 의 `recheck.get("miss_rate")` 를
    `recheck.get("miss_rate", 0.0) or 0.0` 으로 바꾸면 두 번째 assert 가 빨간불.
    확인일: 2026-08-18
    """
    loaded = load_goldset(eval_env["goldset"])
    assert loaded.miss_rate == pytest.approx(0.05)
    assert loaded.gold["2026-08-12"] == {"arxiv:2026-08-12-00", "arxiv:2026-08-12-01"}
    assert loaded.gold["2026-08-14"] == frozenset()

    goldset = yaml.safe_load(eval_env["goldset"].read_text(encoding="utf-8"))
    goldset["summary"]["title_pass_recheck"] = {
        "sampled": 0,
        "flipped_to_relevant": 0,
        "miss_rate": None,
    }
    eval_env["goldset"].write_text(
        yaml.safe_dump(goldset, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    assert load_goldset(eval_env["goldset"]).miss_rate is None


def test_default_embedder_is_not_a_stub(eval_env):
    """★ 임베더를 안 주면 **실모델을 만듭니다. 스텁으로 조용히 폴백하지 않습니다.**

    `hash_stub` 은 의미를 전혀 모르는 해싱 트릭이라 그걸로 낸 Hit@5 는 거짓입니다
    (작업규약 §8-9). 모델이 없어서 죽는 게 가짜로 성공하는 것보다 낫습니다 (§4.2).

    여기서는 `models.yaml` 에 없는 id 를 주어 "조용히 hash_stub 으로 물러서지 않는다"를
    확인합니다 — 2.3GB 가중치를 받지 않고도 확인할 수 있는 지점입니다.

    깨뜨리는 법: run_eval.run_conditions 의 `build_embedder(kind, ...)` 를
    `build_embedder("hash_stub", ...)` 로 바꾸면 통과해버려 빨간불.
    확인일: 2026-08-18
    """
    with pytest.raises((KeyError, ValueError)):
        _run(eval_env, conditions=["bge_m3"], embedder=None, embedder_kind="없는_모델_id")


def test_baseline_condition_builds_no_model_at_all(eval_env, monkeypatch):
    """베이스라인만 돌릴 땐 모델을 **만들지도 않아야** 합니다 (CLAUDE.md §4 — 키 없이 완주).

    "돌아가더라" 로는 부족합니다. 실모델이 로컬 캐시에 있으면 2.3GB 를 로드하고도
    테스트는 초록이 됩니다 — 실제로 이 테스트의 첫 버전이 그랬습니다. 그래서 모델
    **생성 함수 자체를 폭탄으로 바꿔** 부르지 않는 것을 확인합니다.

    깨뜨리는 법: run_conditions 의 `needs_embedder` 판정을 `True` 로 고정하면
    build_embedder 가 불려 빨간불.
    확인일: 2026-08-18
    """

    def explode(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("베이스라인 조건인데 모델을 만들었습니다 (CLAUDE.md §4)")

    monkeypatch.setattr(run_eval, "build_embedder", explode)
    monkeypatch.setattr(run_eval, "build_reranker", explode)

    out = _run(eval_env, conditions=["baseline"], embedder=None, reranker=None)
    table = out.read_text(encoding="utf-8")
    assert "| baseline | 2026-08-12 | 8 | 2 |" in table
    # 안 쓴 것은 안 썼다고 적습니다 (작업규약 §7.2)
    assert "실제 사용 `미사용`" in table
    assert table.count("| 미사용 |") >= 1, "임베더 revision 열이 '미사용' 이어야 합니다"


def test_condition_result_mean_is_none_when_every_date_is_excluded():
    """모든 날이 제외되면 평균은 `None` 입니다 — 0.0 으로 적으면 "전부 놓쳤다"가 됩니다.

    깨뜨리는 법: ConditionResult.mean 의 `if not self.per_date: return None` 을
    `return 0.0` 으로 바꾸면 빨간불.
    확인일: 2026-08-18
    """
    empty = ConditionResult(condition="t", per_date=(), excluded_dates=("2026-08-14",))
    assert empty.mean("hit_at_5") is None


def test_noun_phrase_terms_are_shared_between_query_and_document():
    """질의와 문서에 **같은** 변환을 씁니다. 한쪽만 다르면 매칭이 조용히 어긋납니다.

    깨뜨리는 법: baseline_scores 가 문서 쪽에만 `noun_phrase_terms` 를 쓰고 질의 쪽은
    `set(text.split())` 을 쓰게 바꾸면 대소문자·구두점 때문에 점수가 0이 되어 빨간불.
    확인일: 2026-08-18
    """
    assert noun_phrase_terms("Cross-Encoder, reranking!") == noun_phrase_terms(
        "cross encoder reranking"
    )
