"""LLM 검증자 + 재생성 루프 게이트 (기획안_2 §6.2·§6.3, docs/03_검증자_선정.md).

★ 비활성화하지 마세요:
  - `test_verify_sends_only_verifiable_fields` — `connection` 이 검증자에게 넘어가면
    모든 요약이 감점되어 지표 자체가 무의미해집니다 (기획안_2 §9.8 / R4).
  - `test_regeneration_asks_verifier_differs_right_before_call` — 검증자 계열 분리
    게이트를 **실제로 부르는지** 확인합니다. 게이트는 아무도 부르지 않으면 통과한 적이
    없는 것과 같습니다 (R5).

키 없이 전부 통과합니다. `OPENROUTER_API_KEY` 는 monkeypatch 로 가짜 값을 넣고,
HTTP 는 transport 훅으로 대체합니다 (델타 §D6.2) — 네트워크에 나가지 않습니다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from src.core.models import Item
from src.summarize.generator import FakeGenerator
from src.summarize.orchestrator import (
    RegenerationUnsupportedError,
    ledger_summary_entry,
    render_regeneration_feedback,
    summarize_with_verification,
)
from src.summarize.schema import (
    FAITHFULNESS_THRESHOLD,
    VERIFIABLE_FIELDS,
    PaperSummary,
    UnsupportedClaim,
    Verdict,
)
from src.verify.faithfulness import RuleBasedFakeVerifier, Verifier
from src.verify.gates import GateViolation
from src.verify.llm_verifier import (
    DEFAULT_VERIFIER_MODEL,
    FALLBACK_VERIFIER_MODEL,
    MissingAPIKeyError,
    OpenRouterVerifier,
    VerdictParseError,
    VerificationUsage,
    VerifierError,
    load_verify_template,
    parse_verdict,
    summary_json,
)

FIXTURE = Path(__file__).parent / "fixtures" / "distorted_summaries.yaml"

#: 실제 키가 아닙니다. 키를 코드에 남기지 않습니다 (CLAUDE.md §3-2).
_FAKE_KEY = "test-key-not-a-real-credential"


@pytest.fixture(scope="module")
def samples() -> dict:
    return yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def fake_api_key(monkeypatch) -> None:
    """키가 있는 상태를 흉내 냅니다. 실제 호출은 transport 훅이 가로챕니다."""
    monkeypatch.setenv("OPENROUTER_API_KEY", _FAKE_KEY)


@pytest.fixture
def item(samples) -> Item:
    src = samples["source"]
    return Item(
        id=src["item_id"],
        source="arxiv",
        channel="papers",
        title=src["title"],
        abstract=src["abstract"],
        url=src["url"],
        published="2026-08-12T04:11:02+00:00",
        updated="2026-08-12T04:11:02+00:00",
        publish_scope="public",
        categories=("cs.CV",),
    )


def _summary(samples, key: str) -> PaperSummary:
    """픽스처의 요약 하나를 `PaperSummary` 로. `key`는 'faithful' 또는 왜곡 id."""
    if key == "faithful":
        data = samples["faithful"]["summary"]
    else:
        data = next(d for d in samples["distortions"] if d["id"] == key)["summary"]
    return PaperSummary.from_dict(data, item_id=samples["source"]["item_id"])


# ── 테스트용 대역 ────────────────────────────────────────────────────────


class _RecordingTransport:
    """OpenRouter 응답을 흉내 내고 보낸 payload 를 보관합니다."""

    def __init__(self, content: str = "", *, status: int = 200, body: dict | None = None):
        self.status = status
        self.body = body if body is not None else _chat_body(content)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, url, headers, payload, timeout_sec):
        self.calls.append(
            {"url": url, "headers": headers, "payload": payload, "timeout": timeout_sec}
        )
        return self.status, self.body

    @property
    def sent_text(self) -> str:
        """마지막 요청의 메시지 전문. '무엇을 보냈나'를 검사하는 데 씁니다."""
        messages = self.calls[-1]["payload"]["messages"]
        return "\n".join(str(m["content"]) for m in messages)


def _chat_body(content: str, *, prompt_tokens: int = 1200, completion_tokens: int = 180) -> dict:
    return {
        "choices": [
            {"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


class _ScriptedGenerator:
    """대본대로 요약을 내는 생성자. `feedback` 인자를 받습니다 (기획안_2 §6.3).

    대본이 떨어지면 마지막 요약을 계속 냅니다 — "고쳐도 계속 미달"을 흉내 냅니다.
    `cost_per_call` 을 주면 호출마다 `last_usage` 를 갱신해 실제 생성자를 흉내 냅니다.
    """

    def __init__(
        self,
        summaries,
        model_id: str = "anthropic/claude-opus-5",
        cost_per_call: float | None = None,
    ):
        self.model_id = model_id
        self._summaries = list(summaries)
        self._cost_per_call = cost_per_call
        #: 호출마다 받은 feedback (첫 호출은 None). 주입 여부를 여기서 검사합니다.
        self.feedbacks: list[str | None] = []
        self.last_usage: Any = None

    def generate(self, item: Item, output_language: str, *, feedback: str | None = None):
        self.feedbacks.append(feedback)
        if self._cost_per_call is not None:
            self.last_usage = VerificationUsage(
                input_tokens=1000, output_tokens=300, cost_usd=self._cost_per_call
            )
        index = min(len(self.feedbacks) - 1, len(self._summaries) - 1)
        return self._summaries[index]


class _TwoArgGenerator:
    """`Generator` 프로토콜의 **최소 계약**(2인자)만 만족하는 생성자.

    `generator.py` 의 구현체 둘은 `feedback` 을 받지만 프로토콜은 요구하지 않습니다.
    재생성이 필요해졌을 때 이런 구현체를 만나면 오케스트레이터가 멈춰야 합니다.
    """

    model_id = "anthropic/claude-opus-5"

    def __init__(self, summary: PaperSummary):
        self._summary = summary
        self.calls = 0

    def generate(self, item: Item, output_language: str) -> PaperSummary:
        self.calls += 1
        return self._summary


class _CountingVerifier:
    """항상 같은 점수를 내는 검증자. 몇 번 불렸는지 셉니다."""

    def __init__(self, faithfulness: float, model_id: str = "openai/gpt-5.6-luna"):
        self.model_id = model_id
        self._faithfulness = faithfulness
        self.calls = 0

    def verify(self, abstract: str, summary: PaperSummary) -> Verdict:
        self.calls += 1
        return Verdict(faithfulness=self._faithfulness, verifier_model=self.model_id)


# ══ OpenRouterVerifier ═══════════════════════════════════════════════════


def test_verifier_requires_api_key(monkeypatch):
    """키가 없으면 조용히 폴백하지 않고 죽습니다 (기획안_2 §4.2).

    깨뜨리는 법: llm_verifier.OpenRouterVerifier.__init__ 의 api_key 검사를 지우고
    `os.environ.get(..., "")` 값을 그대로 쓰면 빨간불 (예외가 안 납니다).
    확인일: 2026-08-18
    """
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError, match="OPENROUTER_API_KEY"):
        OpenRouterVerifier()


def test_verifier_defaults_to_documented_model():
    """모델 id 는 docs/03_검증자_선정.md 의 결정입니다. 계열은 생성자(Claude)와 달라야 합니다.

    깨뜨리는 법: DEFAULT_VERIFIER_MODEL 을 "anthropic/claude-..." 로 바꾸면
    `test_verifier_differs_*` 계열과 함께 여기서도 빨간불.
    확인일: 2026-08-18
    """
    from src.verify.gates import model_family

    verifier = OpenRouterVerifier(transport=_RecordingTransport())
    assert verifier.model_id == DEFAULT_VERIFIER_MODEL == "openai/gpt-5.6-luna"
    assert FALLBACK_VERIFIER_MODEL == "deepseek/deepseek-v4-flash"
    # 생성자는 Claude 계열 확정 (기획안 v1.0 §5) — 검증자는 다른 계열이어야 합니다.
    assert model_family(DEFAULT_VERIFIER_MODEL) != "anthropic"
    assert model_family(FALLBACK_VERIFIER_MODEL) != "anthropic"

    explicit = OpenRouterVerifier("qwen/qwen3-max", transport=_RecordingTransport())
    assert explicit.model_id == "qwen/qwen3-max"


def test_verifier_satisfies_protocol():
    assert isinstance(OpenRouterVerifier(transport=_RecordingTransport()), Verifier)


def test_verify_sends_only_verifiable_fields(samples, item):
    """★ `connection` 이 검증자에게 넘어가면 안 됩니다 (기획안_2 §9.8 / R4).

    초록에 근거가 있을 수 없는 필드라, 넘기면 **모든 요약이 감점**되어 지표가
    무의미해집니다.

    깨뜨리는 법: llm_verifier.summary_json 의 `verifiable_dict()` 를 `to_dict()` 로
    바꾸면 빨간불 (connection 문장이 payload 에 들어갑니다).
    확인일: 2026-08-18
    """
    summary = _summary(samples, "faithful")
    assert summary.connection, "픽스처의 정상 요약에 connection 이 있어야 검사가 성립합니다"

    transport = _RecordingTransport('{"faithfulness": 1.0, "unsupported_claims": []}')
    OpenRouterVerifier(transport=transport).verify(item.abstract, summary)

    sent = transport.sent_text
    assert summary.connection not in sent, "connection 이 검증자에게 넘어갔습니다 (§9.8)"
    assert "connection" not in summary_json(summary)
    for name in VERIFIABLE_FIELDS:
        assert name in summary_json(summary), f"채점 대상 필드 {name} 가 빠졌습니다"
    assert summary.problem in sent
    assert item.abstract[:60] in sent


def test_verify_survives_latex_braces(item):
    """★ 초록의 10%에 LaTeX 중괄호가 있습니다 (기획안_2 §9.15).

    치환을 `.format()` 으로 하거나 치환 결과를 다시 format 하면 `KeyError: '6π'` 로
    죽고, 그 아이템은 원장에서 이유 없이 사라집니다.

    깨뜨리는 법: llm_verifier.render_verify_prompt 의 `_substitute` 를
    `part.format(abstract=abstract, summary_json=summary_json_text)` 로 바꾸면 빨간불.
    확인일: 2026-08-18
    """
    abstract = (
        r"We tighten the bounds to \[ \frac{6π}{11} \;\le\; K_G \;\le\; 1 \] and "
        r"introduce \textbf{meta-detection} for AI-generated text."
    )
    summary = PaperSummary(
        item_id=item.id,
        problem=r"경계를 \frac{6π}{11} 로 좁히는 문제",
        method=r"\textbf{meta-detection} 기법",
    )
    transport = _RecordingTransport('{"faithfulness": 0.9, "unsupported_claims": []}')
    verdict = OpenRouterVerifier(transport=transport).verify(abstract, summary)

    assert verdict.faithfulness == 0.9
    assert r"\frac{6π}{11}" in transport.sent_text, "초록이 원문 그대로 들어가야 합니다"
    assert "{abstract}" not in transport.sent_text
    assert "{summary_json}" not in transport.sent_text


def test_verify_returns_verdict_and_records_usage(item, samples):
    """토큰 사용량이 없으면 원장 `cost_usd` 가 실측이 아니라 추정이 됩니다 (기획안_2 §12).

    깨뜨리는 법: llm_verifier.verify 에서 `self.last_usage = ...` 줄을 지우면 빨간불.
    확인일: 2026-08-18
    """
    body = _chat_body(
        """```json
        {"faithfulness": 0.42,
         "unsupported_claims": [
           {"claim": "precision 0.82", "field": "key_results",
            "type": "numeric", "why": "초록은 0.768"}]}
        ```""",
        prompt_tokens=2000,
        completion_tokens=500,
    )
    verifier = OpenRouterVerifier(
        transport=_RecordingTransport(body=body),
        usd_per_1m_input_tokens=0.20,  # docs/03_검증자_선정.md 정가 (프로모 아님)
        usd_per_1m_output_tokens=1.20,
    )
    verdict = verifier.verify(item.abstract, _summary(samples, "numeric_inflation"))

    assert verdict.faithfulness == 0.42
    assert not verdict.passed
    assert verdict.verifier_model == DEFAULT_VERIFIER_MODEL
    assert [c.type for c in verdict.unsupported_claims] == ["numeric"]
    assert verifier.last_usage is not None
    assert (verifier.last_usage.input_tokens, verifier.last_usage.output_tokens) == (2000, 500)
    assert verifier.last_usage.cost_usd == pytest.approx(2000 * 0.2e-6 + 500 * 1.2e-6)


def test_verify_usage_is_recorded_even_when_parse_fails(item, samples):
    """파싱이 실패해도 그 호출의 비용은 실제로 발생했습니다 (기획안_2 §12).

    깨뜨리는 법: llm_verifier.verify 에서 last_usage 대입을 파싱 뒤로 옮기면 빨간불.
    확인일: 2026-08-18
    """
    verifier = OpenRouterVerifier(
        transport=_RecordingTransport("여기 요약은 괜찮아 보입니다."),
        usd_per_1m_input_tokens=0.20,
        usd_per_1m_output_tokens=1.20,
    )
    with pytest.raises(VerdictParseError):
        verifier.verify(item.abstract, _summary(samples, "faithful"))
    assert verifier.last_usage is not None
    assert verifier.last_usage.input_tokens == 1200


def test_verify_without_unit_prices_reports_none_cost(item, samples):
    """단가를 모르면 0.0 이 아니라 None 입니다. 0.0 은 "공짜였다"는 거짓말입니다."""
    verifier = OpenRouterVerifier(transport=_RecordingTransport('{"faithfulness": 1.0}'))
    verifier.verify(item.abstract, _summary(samples, "faithful"))
    assert verifier.last_usage is not None
    assert verifier.last_usage.cost_usd is None


def test_verify_rejects_empty_abstract(samples):
    """대조할 원문이 없으면 faithfulness 는 성립하지 않습니다 (CLAUDE.md §3-8)."""
    verifier = OpenRouterVerifier(transport=_RecordingTransport('{"faithfulness": 1.0}'))
    with pytest.raises(VerifierError, match="초록"):
        verifier.verify("   ", _summary(samples, "faithful"))


def test_http_error_does_not_leak_key(item, samples):
    """★ 예외 메시지에 키가 들어가면 안 됩니다 (CLAUDE.md §3-2).

    깨뜨리는 법: llm_verifier.verify 의 오류 메시지에 `self._api_key` 를 넣으면 빨간불.
    확인일: 2026-08-18
    """
    transport = _RecordingTransport(status=401, body={"error": {"message": "invalid key"}})
    verifier = OpenRouterVerifier(transport=transport)
    with pytest.raises(VerifierError) as excinfo:
        verifier.verify(item.abstract, _summary(samples, "faithful"))
    assert "HTTP 401" in str(excinfo.value)
    assert _FAKE_KEY not in str(excinfo.value)
    assert transport.calls[0]["headers"]["Authorization"] == f"Bearer {_FAKE_KEY}"


def test_empty_response_raises_parse_error(item, samples):
    verifier = OpenRouterVerifier(transport=_RecordingTransport(""))
    with pytest.raises(VerdictParseError, match="비어"):
        verifier.verify(item.abstract, _summary(samples, "faithful"))


# ── 판정 파싱 ────────────────────────────────────────────────────────────


def test_parse_verdict_tolerates_prose_around_json():
    """모델이 설명을 붙여 보내는 일이 있습니다. JSON 객체만 잘라 씁니다."""
    text = '판정 결과입니다.\n{"faithfulness": 0.8, "unsupported_claims": []}\n이상입니다.'
    verdict = parse_verdict(text, verifier_model="openai/gpt-5.6-luna")
    assert verdict.faithfulness == 0.8


def test_parse_verdict_rejects_unknown_distortion_type():
    """★ 왜곡 유형은 프롬프트가 정한 5개뿐입니다 (schema.DISTORTION_TYPES).

    임의 유형을 받아 주면 원장의 왜곡 유형 분포가 조용히 거짓이 됩니다.

    깨뜨리는 법: llm_verifier.parse_verdict 에서 UnsupportedClaim 생성을
    try/except 없이 dict 그대로 담게 바꾸면 빨간불.
    확인일: 2026-08-18
    """
    text = (
        '{"faithfulness": 0.5, "unsupported_claims": ['
        '{"claim": "c", "field": "method", "type": "vibes", "why": "w"}]}'
    )
    with pytest.raises(VerdictParseError, match="스키마"):
        parse_verdict(text, verifier_model="openai/gpt-5.6-luna")


@pytest.mark.parametrize(
    "text, match",
    [
        ("검증에 실패했습니다", "JSON"),
        ("[1, 2, 3]", "객체가 아닙니다"),
        ('{"unsupported_claims": []}', "faithfulness 가 없습니다"),
        ('{"faithfulness": "높음"}', "수치가 아닙니다"),
        ('{"faithfulness": 1.4}', "범위 밖"),
        ('{"faithfulness": 0.5, "unsupported_claims": "많음"}', "배열이 아닙니다"),
    ],
)
def test_parse_verdict_rejects_malformed(text, match):
    """파싱 실패는 조용히 넘어가지 않습니다 — 그날 검증 불능입니다."""
    with pytest.raises(VerdictParseError, match=match):
        parse_verdict(text, verifier_model="openai/gpt-5.6-luna")


def test_prompt_template_has_placeholders():
    """확정 프롬프트(기획안_2 §6.1)의 자리표시자 검사는 **치환 전**에만 합니다."""
    template = load_verify_template()
    assert "{abstract}" in template and "{summary_json}" in template
    assert "## USER" in template


def test_prompt_template_missing_placeholder_is_loud(tmp_path):
    bad = tmp_path / "verify.md"
    bad.write_text("## SYSTEM\n검사하세요\n## USER\n{abstract}\n", encoding="utf-8")
    with pytest.raises(VerifierError, match="summary_json"):
        load_verify_template(bad)


# ══ 재생성 루프 (기획안_2 §6.3) ═══════════════════════════════════════════


def test_passing_summary_needs_no_retry(item):
    """통과하면 재생성이 없습니다. Fake 조합으로 전 구간 배선을 확인합니다 (델타 §D6.2).

    깨뜨리는 법: orchestrator.summarize_with_verification 의 `if verdict.passed: break`
    를 지우면 시도가 3회로 늘어 빨간불.
    확인일: 2026-08-18
    """
    generator = FakeGenerator()
    verifier = RuleBasedFakeVerifier()
    result = summarize_with_verification(item, generator, verifier)

    assert result.retries == 0
    assert len(result.attempts) == 1
    assert result.flagged is False
    assert result.verdict.passed
    assert result.generator_model == generator.model_id
    assert result.verdict.verifier_model == verifier.model_id


def test_regenerates_until_pass(item, samples):
    """★ 미달 → 재생성 → 통과. `retries` 가 원장 지표입니다 (기획안_2 §6.3).

    실제 규칙 검증자를 씁니다 — 왜곡 요약(수치 부풀림)은 0.3, 정상 요약은 1.0 입니다.

    깨뜨리는 법: orchestrator 의 `retries=len(attempts) - 1` 을 `retries=0` 으로
    바꾸면 빨간불.
    확인일: 2026-08-18
    """
    generator = _ScriptedGenerator(
        [_summary(samples, "numeric_inflation"), _summary(samples, "faithful")]
    )
    result = summarize_with_verification(item, generator, RuleBasedFakeVerifier())

    assert result.retries == 1
    assert len(result.attempts) == 2
    assert result.attempts[0].faithfulness < FAITHFULNESS_THRESHOLD
    assert result.attempts[1].faithfulness == 1.0
    assert result.flagged is False
    assert result.summary.key_results == _summary(samples, "faithful").key_results


def test_regeneration_injects_previous_claims(item, samples):
    """★ 같은 프롬프트로 다시 부르면 같은 실수가 나옵니다 (기획안_2 §6.3).

    직전 판정의 `unsupported_claims` 가 재생성 호출에 실려야 합니다.

    깨뜨리는 법: orchestrator 루프의 `feedback = render_regeneration_feedback(...)`
    줄을 지우면(=None 유지) 빨간불.
    확인일: 2026-08-18
    """
    generator = _ScriptedGenerator(
        [_summary(samples, "numeric_inflation"), _summary(samples, "faithful")]
    )
    summarize_with_verification(item, generator, RuleBasedFakeVerifier())

    assert generator.feedbacks[0] is None, "최초 생성에는 피드백이 없습니다"
    feedback = generator.feedbacks[1]
    assert feedback, "재생성인데 피드백이 비어 있습니다"
    assert "0.82" in feedback, "직전에 지적된 수치가 프롬프트에 들어가야 합니다"
    assert "key_results" in feedback and "numeric" in feedback


def test_flags_after_max_retries(item, samples):
    """★ 2회 재생성 후에도 미달이면 flagged=True 로 **발행합니다** (기획안_2 §6.3).

    미달 자체는 발행을 막지 않습니다. 막는 건 플래그 누락입니다.

    깨뜨리는 법: orchestrator 의 `flagged = not verdict.passed` 를 `flagged = False`
    로 바꾸면 assert_faithfulness_flagged 가 GateViolation 을 던져 빨간불.
    확인일: 2026-08-18
    """
    generator = _ScriptedGenerator([_summary(samples, "numeric_inflation")])
    result = summarize_with_verification(item, generator, RuleBasedFakeVerifier())

    assert result.retries == 2, "재생성은 최대 2회입니다 (schema.MAX_REGENERATIONS)"
    assert len(result.attempts) == 3
    assert result.flagged is True
    assert not result.verdict.passed
    assert result.faithfulness_history == (
        pytest.approx(0.3),
        pytest.approx(0.3),
        pytest.approx(0.3),
    )
    assert generator.feedbacks[1] and generator.feedbacks[2], "재생성마다 피드백을 넣습니다"


def test_max_retries_zero_flags_immediately(item, samples):
    """재생성 0회 설정이면 첫 판정으로 끝납니다."""
    generator = _ScriptedGenerator([_summary(samples, "numeric_inflation")])
    result = summarize_with_verification(item, generator, RuleBasedFakeVerifier(), 0)

    assert result.retries == 0
    assert result.flagged is True
    assert generator.feedbacks == [None]


def test_negative_max_retries_is_rejected(item, samples):
    with pytest.raises(ValueError, match="max_retries"):
        summarize_with_verification(
            item, _ScriptedGenerator([_summary(samples, "faithful")]), RuleBasedFakeVerifier(), -1
        )


def test_regeneration_asks_verifier_differs_right_before_call(item, samples):
    """★ 검증자 계열 분리 게이트를 **요청 직전에** 부릅니다 (기획안_2 §6.2 / R5).

    생성은 이미 일어났고 검증 요청은 아직 나가지 않은 상태에서 터져야 합니다.
    같은 계열이면 동일가족 편향으로 검증이 성립하지 않습니다 (CLAUDE.md §3-9).

    깨뜨리는 법: orchestrator 루프의 `assert_verifier_differs(...)` 호출을 지우면
    빨간불 (GateViolation 이 안 납니다).
    확인일: 2026-08-18
    """
    generator = _ScriptedGenerator([_summary(samples, "faithful")], model_id="claude-opus-5")
    verifier = _CountingVerifier(1.0, model_id="claude-haiku-4-5")  # ID 는 다르지만 같은 계열

    with pytest.raises(GateViolation, match="같은 계열"):
        summarize_with_verification(item, generator, verifier)

    assert len(generator.feedbacks) == 1, "생성은 한 번 일어났어야 합니다"
    assert verifier.calls == 0, "게이트가 검증 요청보다 먼저 터져야 합니다"


def test_generator_without_feedback_support_stops_loudly(item, samples):
    """★ 피드백을 못 받는 생성자로 재생성하면 멈춥니다 (기획안_2 §6.3).

    같은 프롬프트로 조용히 재호출하면 같은 실수가 그대로 나오고, 원장의 `retries` 는
    "고쳐 봤다"는 거짓말이 됩니다. `Generator` 프로토콜의 최소 계약은 2인자라
    피드백을 못 받는 구현체가 들어올 수 있습니다.

    깨뜨리는 법: orchestrator._generate 의 `_accepts_feedback` 검사를 지우면
    TypeError 로 바뀌거나(=메시지가 무의미) 조용히 재호출됩니다.
    확인일: 2026-08-18
    """
    generator = _TwoArgGenerator(_summary(samples, "numeric_inflation"))
    with pytest.raises(RegenerationUnsupportedError, match="feedback"):
        summarize_with_verification(item, generator, _CountingVerifier(0.4))
    assert generator.calls == 1, "최초 생성은 되고, 재생성 직전에 멈춰야 합니다"


def test_feedback_text_lists_claims_without_formatting():
    """피드백 문장은 그대로 넘깁니다. 다시 format 하면 LaTeX 중괄호에서 터집니다 (§9.15)."""
    claims = [
        UnsupportedClaim(
            claim=r"경계는 \frac{6π}{11} 이다", field="method", type="numeric", why="초록에 없음"
        )
    ]
    text = render_regeneration_feedback(claims)
    assert r"\frac{6π}{11}" in text
    assert "method" in text and "numeric" in text

    # 지목된 문장이 없어도 재생성 지시는 비지 않습니다 (§6.3 — 같은 프롬프트 금지).
    assert render_regeneration_feedback([]).strip()


# ── 비용 (기획안_2 §12) ──────────────────────────────────────────────────


def test_cost_counts_every_attempt(item, samples):
    """★ 재생성분의 비용이 사라지면 안 됩니다 (기획안_2 §12).

    루프가 끝난 뒤 `generator.last_usage` 를 한 번 읽는 방식은 **마지막 시도분만**
    잡아서, 재생성한 아이템일수록 비용이 적게 기록되는 방향으로 조용히 틀립니다.

    깨뜨리는 법: orchestrator 의 `cost.collect(generator)` 를 루프 밖(반환 직전)으로
    옮기면 0.003 이 아니라 0.001 이 나와 빨간불.
    확인일: 2026-08-18
    """
    generator = _ScriptedGenerator([_summary(samples, "numeric_inflation")], cost_per_call=0.001)
    result = summarize_with_verification(item, generator, RuleBasedFakeVerifier())

    assert result.retries == 2
    assert result.cost_usd == pytest.approx(0.003)


def test_cost_is_unknown_rather_than_zero(item, samples):
    """단가를 모르면 0.0 이 아니라 None 입니다 — 0.0 은 "공짜였다"는 거짓말입니다."""
    fake_only = summarize_with_verification(item, FakeGenerator(), RuleBasedFakeVerifier())
    assert fake_only.cost_usd is None, "사용량을 보고하지 않는 Fake 조합은 0.0 이 아닙니다"

    unpriced = _ScriptedGenerator([_summary(samples, "faithful")], cost_per_call=None)
    unpriced.last_usage = VerificationUsage(input_tokens=10, output_tokens=5, cost_usd=None)
    result = summarize_with_verification(item, unpriced, RuleBasedFakeVerifier())
    assert result.cost_usd is None, "단가 없이 보고된 호출이 있으면 합계는 '모름'입니다"


# ── 원장 항목 (기획안_2 §4.4) ────────────────────────────────────────────


def test_ledger_entry_uses_spec_field_names(item, samples):
    """★ 원장 `summaries[]` 필드명은 기획안_2 §4.4 그대로여야 합니다.

    이름이 갈리면 이미 기록된 줄과 비교가 불가능해집니다 (원장은 고쳐 쓰지 않습니다).

    깨뜨리는 법: orchestrator.ledger_summary_entry 의 키 하나를 빼거나 이름을 바꾸면
    빨간불 (예: "flagged" → "flag").
    확인일: 2026-08-18
    """
    generator = _ScriptedGenerator([_summary(samples, "numeric_inflation")])
    result = summarize_with_verification(item, generator, RuleBasedFakeVerifier())
    entry = ledger_summary_entry(result, rank_score_stage1=0.71, rank_score_stage2=-2.51)

    assert list(entry)[:9] == [
        "item_id",
        "rank_score_stage1",
        "rank_score_stage2",
        "generator_model",
        "verifier_model",
        "faithfulness",
        "retries",
        "unsupported_claims",
        "flagged",
    ]
    assert entry["item_id"] == samples["source"]["item_id"]
    assert entry["generator_model"] == "anthropic/claude-opus-5"
    assert entry["verifier_model"] == "rulebased_fake/v1"
    assert entry["retries"] == 2
    assert entry["flagged"] is True
    assert entry["unsupported_claims"] and entry["unsupported_claims"][0]["type"] == "numeric"
    # 2차 랭킹 점수는 raw 로짓입니다 — sigmoid 를 씌우면 원장을 읽을 수 없습니다 (§9.4)
    assert entry["rank_score_stage2"] == -2.51


def test_ledger_entry_keeps_every_attempt_score(item, samples):
    """기획안_2 §6.3 — "매 시도의 faithfulness 를 전부 원장에 남기세요. 분포가 DoD."

    깨뜨리는 법: ledger_summary_entry 의 "attempts" 키를 지우면 빨간불.
    확인일: 2026-08-18
    """
    generator = _ScriptedGenerator(
        [_summary(samples, "numeric_inflation"), _summary(samples, "faithful")]
    )
    result = summarize_with_verification(item, generator, RuleBasedFakeVerifier())
    entry = ledger_summary_entry(result)

    assert [a["index"] for a in entry["attempts"]] == [0, 1]
    assert entry["attempts"][0]["faithfulness"] == pytest.approx(0.3)
    assert entry["attempts"][1]["faithfulness"] == 1.0
    # 랭킹을 돌리지 않았으면 키 삭제가 아니라 null 입니다 (§4.4 규칙 2)
    assert entry["rank_score_stage1"] is None and entry["rank_score_stage2"] is None
