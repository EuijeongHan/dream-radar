"""검증 게이트 (기획안 §8 M2·§10, 델타 §D6.4).

★ `test_verifier_differs` 는 비활성화하지 마세요. 동일가족 편향으로 검증이 성립하지
   않게 되는 걸 막는 게이트입니다.

이 파일에는 **아직 판정할 수 없는 DoD 테스트**가 하나 있습니다
(`test_llm_verifier_catches_all_three_distortions`). 키가 없어 skip 상태이며,
키가 도착하면 그날 바로 판정됩니다. 왜곡 샘플 3건과 하네스는 이미 준비돼 있습니다.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from src.summarize.schema import (
    FAITHFULNESS_THRESHOLD,
    VERIFIABLE_FIELDS,
    PaperSummary,
    UnsupportedClaim,
    Verdict,
)
from src.verify.faithfulness import RuleBasedFakeVerifier, Verifier
from src.verify.gates import (
    GateViolation,
    ScopeViolation,
    assert_faithfulness_flagged,
    assert_public_scope,
    assert_verifier_differs,
    model_family,
)

FIXTURE = Path(__file__).parent / "fixtures" / "distorted_summaries.yaml"


@pytest.fixture(scope="module")
def samples() -> dict:
    return yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))


def _summary(item_id: str, data: dict) -> PaperSummary:
    return PaperSummary.from_dict(data, item_id=item_id)


# ── 검증자 분리 게이트 ★ ─────────────────────────────────────────────────


def test_verifier_differs_rejects_identical():
    with pytest.raises(GateViolation, match="동일"):
        assert_verifier_differs("claude-opus-5", "claude-opus-5")


def test_verifier_differs_rejects_same_family():
    """★ ID만 다르고 계열이 같으면 편향이 그대로입니다.

    이걸 잡지 못하면 `claude-opus-5` 생성 / `claude-haiku-4-5` 검증 같은 구성이
    게이트를 통과합니다. ID 비교만으로는 부족합니다.
    """
    for generator, verifier in [
        ("claude-opus-5", "claude-haiku-4-5"),
        ("anthropic/claude-opus-5", "claude-sonnet-5"),
        ("gpt-5", "openai/gpt-4.1-mini"),
        ("google/gemini-3-pro", "gemma-3-27b"),
    ]:
        with pytest.raises(GateViolation, match="같은 계열"):
            assert_verifier_differs(generator, verifier)


def test_verifier_differs_accepts_cross_family():
    assert_verifier_differs("claude-opus-5", "openai/gpt-5-mini")
    assert_verifier_differs("anthropic/claude-opus-5", "qwen/qwen3-max")
    assert_verifier_differs("claude-opus-5", "rulebased_fake/v1")


def test_verifier_differs_rejects_empty():
    with pytest.raises(GateViolation, match="비어"):
        assert_verifier_differs("claude-opus-5", "")


def test_model_family_detection():
    assert model_family("anthropic/claude-opus-5") == "anthropic"
    assert model_family("gpt-5-mini") == "openai"
    assert model_family("meta-llama/llama-4-70b") == "meta"
    assert model_family("mystery-model-9") == "mystery-model-9"


# ── 규칙 검증자 (하한선 베이스라인) ──────────────────────────────────────


def test_rulebased_verifier_passes_faithful_summary(samples):
    """정상 요약을 통과시키지 못하면 위양성이 너무 높은 것입니다."""
    verifier = RuleBasedFakeVerifier()
    verdict = verifier.verify(
        samples["source"]["abstract"],
        _summary(samples["source"]["item_id"], samples["faithful"]["summary"]),
    )
    assert verdict.unsupported_claims == [], verdict.unsupported_claims
    assert verdict.faithfulness == 1.0
    assert verdict.passed


def test_rulebased_verifier_catches_numeric_distortion(samples):
    """① 수치 조작 — 규칙 검증자가 잡아야 하는 유일한 유형입니다."""
    case = next(d for d in samples["distortions"] if d["id"] == "numeric_inflation")
    verdict = RuleBasedFakeVerifier().verify(
        samples["source"]["abstract"],
        _summary(samples["source"]["item_id"], case["summary"]),
    )
    assert verdict.unsupported_claims, "부풀린 수치를 못 잡았습니다"
    assert all(c.type == "numeric" for c in verdict.unsupported_claims)
    assert not verdict.passed


@pytest.mark.parametrize("case_id", ["fabricated_causality", "limitation_reversal"])
def test_rulebased_verifier_misses_semantic_distortions(samples, case_id):
    """②③은 규칙 검증자가 **못 잡는 게 정상**입니다.

    이 테스트는 실패를 고정합니다. 나중에 누가 규칙으로 ②③을 흉내 내면 여기가 깨지고,
    그때 "그 규칙이 내가 만든 샘플에 과적합된 게 아닌가"를 반드시 되묻게 됩니다.
    하한선 베이스라인은 정직할 때만 베이스라인입니다.
    """
    case = next(d for d in samples["distortions"] if d["id"] == case_id)
    verdict = RuleBasedFakeVerifier().verify(
        samples["source"]["abstract"],
        _summary(samples["source"]["item_id"], case["summary"]),
    )
    assert verdict.faithfulness == 1.0, (
        "규칙 검증자가 의미 왜곡을 잡았습니다. 의도치 않은 동작이거나, "
        "샘플에 과적합된 규칙이 추가된 것입니다"
    )
    assert "llm" in case["detectable_by"]
    assert "rulebased_fake" not in case["detectable_by"]


def test_rulebased_verifier_satisfies_protocol():
    assert isinstance(RuleBasedFakeVerifier(), Verifier)


def test_rulebased_verifier_ignores_years():
    """연도는 성능 수치가 아니라 식별자입니다. 오탐이 나면 하한선 자격을 잃습니다."""
    verdict = RuleBasedFakeVerifier().verify(
        "We evaluate on the standard benchmark.",
        PaperSummary(item_id="x", problem="2024년에 제안된 방법", method="m"),
    )
    assert verdict.unsupported_claims == []


@pytest.mark.parametrize(
    "sentence",
    [
        "모델 6종을 비교했다",  # 초록은 "six" 라고 단어로 씀
        "2단계 검출기가 여전히 경쟁력이 있다",
        "YOLOv5, YOLOv8, YOLO11, YOLO26 을 평가했다",
    ],
)
def test_rulebased_verifier_ignores_prose_integers(sentence):
    """★ 교차언어 오탐 방지.

    영어 초록은 작은 수를 단어로 쓰고("six models") 한국어 요약은 숫자로 씁니다("6종").
    맨정수를 전부 대조하면 정상 요약이 감점됩니다 — 실제로 그렇게 짰다가 정상 요약이
    0.3점을 받았습니다.
    """
    verdict = RuleBasedFakeVerifier().verify(
        "We compare six detectors including YOLO variants in two stages.",
        PaperSummary(item_id="x", problem="p", method=sentence),
    )
    assert verdict.unsupported_claims == [], verdict.unsupported_claims


@pytest.mark.parametrize("sentence", ["500장으로 학습했다", "정확도가 42.7% 향상됐다"])
def test_rulebased_verifier_still_catches_measured_numbers(sentence):
    """오탐을 줄이면서 진짜 수치 조작은 여전히 잡아야 합니다."""
    verdict = RuleBasedFakeVerifier().verify(
        "We train on 3,382 images and observe a 12.1% improvement.",
        PaperSummary(item_id="x", problem="p", method=sentence),
    )
    assert verdict.unsupported_claims, "성능 수치 조작을 놓쳤습니다"


# ── 왜곡 샘플 자체의 건전성 ──────────────────────────────────────────────


def test_three_distortion_types_are_distinct(samples):
    """기획안 §8 M2 DoD — 왜곡 유형 3개가 서로 달라야 합니다 (델타 §D6.4)."""
    distortions = samples["distortions"]
    assert len(distortions) == 3
    types = [t for d in distortions for t in d["expect_types"]]
    assert sorted(types) == ["causal", "numeric", "reversal"]


def test_distortion_samples_share_one_source(samples):
    """원문이 고정돼야 왜곡만이 유일한 변수가 됩니다."""
    assert samples["source"]["abstract"]
    assert samples["source"]["item_id"].startswith("arxiv:")
    for case in samples["distortions"]:
        assert case["expect_pass"] is False


# ── 스키마 ──────────────────────────────────────────────────────────────


def test_connection_is_excluded_from_verification(samples):
    """★ `connection`은 초록에 근거가 있을 수 없는 필드입니다.

    채점에 포함하면 모든 요약이 감점되어 지표가 무의미해집니다.
    """
    assert "connection" not in VERIFIABLE_FIELDS
    summary = _summary(samples["source"]["item_id"], samples["faithful"]["summary"])
    assert summary.connection
    assert summary.connection not in summary.verifiable_text()


def test_summary_requires_core_fields():
    with pytest.raises(ValueError, match="필수 필드"):
        PaperSummary.from_dict({"problem": "", "method": "m"}, item_id="x")


def test_verdict_rejects_out_of_range():
    with pytest.raises(ValueError, match="0.0~1.0"):
        Verdict(faithfulness=1.5)


def test_unsupported_claim_rejects_unknown_type():
    with pytest.raises(ValueError, match="왜곡 유형"):
        UnsupportedClaim(claim="c", field="method", type="vibes", why="w")


def test_verdict_serialises_for_ledger(samples):
    verdict = Verdict(
        faithfulness=0.91,
        unsupported_claims=[UnsupportedClaim("c", "method", "numeric", "w")],
        verifier_model="openai/gpt-5-mini",
    )
    row = verdict.to_dict()
    assert set(row) == {"faithfulness", "unsupported_claims", "verifier_model"}
    assert row["faithfulness"] == 0.91


# ── 발행 게이트 ──────────────────────────────────────────────────────────


def test_faithfulness_threshold_gate():
    low = Verdict(faithfulness=0.4)
    with pytest.raises(GateViolation, match="플래그"):
        assert_faithfulness_flagged(low, flagged=False)
    assert_faithfulness_flagged(low, flagged=True)  # 플래그가 있으면 발행 허용
    assert_faithfulness_flagged(Verdict(faithfulness=FAITHFULNESS_THRESHOLD), flagged=False)


def test_publish_scope_gate():
    class FakeItem:
        def __init__(self, id_, scope):
            self.id = id_
            self.publish_scope = scope

    assert_public_scope([FakeItem("a", "public")])
    with pytest.raises(ScopeViolation, match="b"):
        assert_public_scope([FakeItem("a", "public"), FakeItem("b", "private")])


# ── 키가 오면 판정되는 DoD ───────────────────────────────────────────────


@pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="검증자 키 미발급. 키가 오면 이 테스트가 M2 DoD를 판정합니다",
)
def test_llm_verifier_catches_all_three_distortions(samples):
    """기획안 §8 M2 DoD — 왜곡 3건을 검증자가 **모두** 잡아내야 합니다.

    구현체(LLMVerifier)는 M2 나머지에서 만듭니다. 샘플·하네스·게이트는 준비 완료입니다.
    """
    pytest.fail("LLMVerifier 미구현 — M2 나머지 (델타 §D6.5 순서 7)")
