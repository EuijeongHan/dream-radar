"""요약 생성자 (기획안_2 §6, 델타 §D6.2).

이 파일이 지키는 것 4가지:

1. **키 없이 전부 통과합니다.** 실제 API 를 부르는 테스트는 딱 하나이고
   `ANTHROPIC_API_KEY` + `RADAR_GENERATOR_MODEL` 이 **둘 다** 있을 때만 돕니다.
   나머지는 `FakeGenerator` 와 주입 transport 로 돕니다 (델타 §D6.2).
2. **★ LaTeX 중괄호가 프롬프트 치환을 깨지 않는 것** (기획안_2 §9.15). 초록 719건 중
   72건(10.0%)에 `\\frac{6π}{11}` 같은 중괄호가 있고, 여기서 터지면 10편 중 1편이
   원장에 **이유 없이** 사라집니다. 이 파일에서 가장 중요한 테스트입니다.
3. **키가 없으면 조용히 폴백하지 않고 죽는 것** (기획안_2 §4.2).
4. **프롬프트의 `connection` 제외 규칙이 살아 있는 것** (기획안_2 §9.8).

테스트 작성 절차는 작업규약 §4.2 를 따랐습니다 — 각 테스트에 `깨뜨리는 법` 을
남겼고, 전부 실제로 심어서 빨간불을 확인한 뒤 되돌렸습니다.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from src.core.models import Item
from src.summarize import generator as G
from src.summarize.generator import (
    AnthropicGenerator,
    FakeGenerator,
    GenerationUsage,
    Generator,
    GeneratorError,
    MissingAPIKeyError,
    SummaryParseError,
    load_prompt_template,
    parse_summary,
    render_prompt,
)
from src.summarize.schema import OUTPUT_LANGUAGES, VERIFIABLE_FIELDS, PaperSummary
from src.verify.faithfulness import RuleBasedFakeVerifier
from src.verify.gates import assert_verifier_differs, model_family

# ── 픽스처 ───────────────────────────────────────────────────────────────

#: ★ 기획안_2 §9.15 의 실측 사례를 그대로 옮긴 초록.
#: 세 종류의 함정이 한 문자열에 다 들어 있습니다:
#:   ① LaTeX 수식 중괄호  `\frac{6\pi}{11}`, `\textbf{meta-detection}`
#:   ② 산문 중괄호        `{a, b, c}`
#:   ③ **자리표시자와 글자가 같은 문자열** `{title}` · `{abstract}`
#: ③이 핵심입니다. 치환 순서가 틀리면 초록 안의 `{title}` 이 제목으로 바뀌는데,
#: 예외가 안 나므로 아무도 모릅니다.
LATEX_ABSTRACT = (
    "We tighten the bounds to \\[ \\frac{6\\pi}{11} \\;\\le\\; K_G \\;\\le\\; 1 \\] and are "
    "the first to introduce \\textbf{meta-detection} into AI-generated image forensics. "
    "The candidate set {a, b, c} is closed under the operator. "
    "Ablations over the {title} and {abstract} placeholders confirm a 28.4% gain. "
    "We release code and the 3,382-image benchmark."
)

_API_KEY_ENV = "ANTHROPIC_API_KEY"
_FAKE_KEY = "sk-ant-test-not-a-real-key"
_TEST_MODEL = "anthropic/claude-test-model"


@pytest.fixture
def item() -> Item:
    return Item(
        id="arxiv:2608.01234",
        source="arxiv",
        channel="papers",
        title="Meta-Detection of AI-Generated Images",
        abstract=LATEX_ABSTRACT,
        url="https://arxiv.org/abs/2608.01234",
        published="2026-08-12T00:00:00Z",
        updated="2026-08-12T00:00:00Z",
        publish_scope="public",
        categories=("cs.CV", "cs.LG"),
    )


@pytest.fixture
def plain_item() -> Item:
    return Item(
        id="arxiv:2608.09999",
        source="arxiv",
        channel="papers",
        title="A Plain Paper",
        abstract="We study X. We propose Y. Y improves Z by 12.1%. Code is available.",
        url="https://arxiv.org/abs/2608.09999",
        published="2026-08-12T00:00:00Z",
        updated="2026-08-12T00:00:00Z",
        publish_scope="public",
        categories=("cs.CV",),
    )


VALID_RESPONSE_JSON = json.dumps(
    {
        "problem": "AI 생성 이미지 탐지기의 일반화가 약하다.",
        "method": "meta-detection 을 도입해 탐지기를 메타 수준에서 결합한다.",
        "key_results": ["28.4% 향상"],
        "limitations": [],
        "connection": "",
    },
    ensure_ascii=False,
)


class RecordingTransport:
    """주입 transport. 네트워크·키 없이 요청 본문을 그대로 붙잡습니다.

    `arxiv.py` 의 Transport 훅과 같은 목적입니다.
    """

    def __init__(self, status: int = 200, body: dict[str, Any] | None = None) -> None:
        self.status = status
        self.body = body if body is not None else _message_body(VALID_RESPONSE_JSON)
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, url: str, headers: dict[str, str], payload: dict[str, Any], timeout_sec: float
    ) -> tuple[int, dict[str, Any]]:
        self.calls.append(
            {"url": url, "headers": headers, "payload": payload, "timeout_sec": timeout_sec}
        )
        return self.status, self.body

    @property
    def payload(self) -> dict[str, Any]:
        assert self.calls, "transport 가 한 번도 호출되지 않았습니다"
        return self.calls[-1]["payload"]

    @property
    def user_text(self) -> str:
        return self.payload["messages"][0]["content"]

    @property
    def system_text(self) -> str:
        return self.payload["system"]


def _message_body(
    text: str, *, input_tokens: int = 1200, output_tokens: int = 300, stop_reason: str = "end_turn"
) -> dict[str, Any]:
    """Anthropic Messages API 응답 형태 (문서 형식 그대로)."""
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


def _anthropic(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> AnthropicGenerator:
    """키를 가짜로 심고 생성자를 만듭니다. transport 가 주입돼 있어 네트워크는 없습니다."""
    monkeypatch.setenv(_API_KEY_ENV, _FAKE_KEY)
    kwargs.setdefault("transport", RecordingTransport())
    kwargs.setdefault("interests", "멀티모달 검색, 랭킹")
    return AnthropicGenerator(_TEST_MODEL, **kwargs)


# ── FakeGenerator — 키 없이 도는 배선 ────────────────────────────────────


def test_fake_generator_satisfies_protocol():
    """★ `Generator` 포트를 만족해야 파이프라인이 Fake/실물을 갈아끼울 수 있습니다.

    `model_id` 가 특히 중요합니다 — 검증자 분리 게이트가 이 값을 비교합니다.

    깨뜨리는 법: `FakeGenerator.model_id` 속성을 지우면 빨간불.
    확인일: 2026-08-18
    """
    fake = FakeGenerator()
    assert isinstance(fake, Generator)
    assert fake.model_id


def test_fake_generator_roundtrips_schema(plain_item: Item):
    """생성 → `to_dict` → `from_dict` 왕복에서 필드가 살아남아야 합니다 (R4).

    깨뜨리는 법: `FakeGenerator.generate` 의 `key_results` 를 `[]` 로 고정하면 빨간불.
    확인일: 2026-08-18
    """
    summary = FakeGenerator().generate(plain_item, "ko")
    assert isinstance(summary, PaperSummary)
    assert summary.item_id == plain_item.id
    assert summary.problem == "We study X."
    assert summary.method == "We propose Y."
    assert summary.key_results == ["Y improves Z by 12.1%."]

    restored = PaperSummary.from_dict(summary.to_dict())
    assert restored.to_dict() == summary.to_dict()


def test_fake_generator_is_deterministic(plain_item: Item):
    """같은 입력에 같은 출력. 배선 테스트가 흔들리면 회귀를 못 봅니다.

    깨뜨리는 법: `generate` 에 `random.shuffle(sentences)` 를 넣으면 빨간불.
    확인일: 2026-08-18
    """
    first = FakeGenerator().generate(plain_item, "ko").to_dict()
    second = FakeGenerator().generate(plain_item, "ko").to_dict()
    assert first == second


def test_fake_generator_copies_abstract_verbatim(plain_item: Item):
    """★ Fake 는 초록 문장을 **그대로** 옮깁니다. 지어내면 배선 테스트가 거짓이 됩니다.

    `RuleBasedFakeVerifier` 와 짝지었을 때 faithfulness 1.0 이 나와야, 배선 실패와
    요약 품질 문제가 구분됩니다 (작업규약 §8-9 — fake 로 낸 지표는 거짓입니다).

    깨뜨리는 법: `problem` 에 `"요약: " + sentences[0]` 처럼 접두어를 붙이면
    아래 `in abstract` 가 빨간불.
    확인일: 2026-08-18
    """
    summary = FakeGenerator().generate(plain_item, "ko")
    for sentence in [summary.problem, summary.method, *summary.key_results]:
        assert sentence in plain_item.abstract, f"초록에 없는 문장을 만들었습니다: {sentence!r}"

    verdict = RuleBasedFakeVerifier().verify(plain_item.abstract, summary)
    assert verdict.faithfulness == 1.0, verdict.unsupported_claims


def test_fake_generator_leaves_connection_empty(plain_item: Item):
    """`connection` 은 초록에 근거가 없는 필드입니다 (기획안_2 §9.8). Fake 는 비웁니다.

    깨뜨리는 법: `connection="관련 있어 보임"` 으로 채우면 빨간불.
    확인일: 2026-08-18
    """
    assert FakeGenerator().generate(plain_item, "ko").connection == ""


def test_fake_generator_handles_empty_abstract():
    """초록이 비어도 스키마 필수 필드(`problem`·`method`)는 채워져야 합니다.

    빈 값이면 `PaperSummary.from_dict` 가 `ValueError` 를 냅니다 — 즉 이 아이템은
    원장에 이유 없이 사라집니다.

    깨뜨리는 법: `fallback` 을 `""` 로 두면 아래 `from_dict` 가 빨간불.
    확인일: 2026-08-18
    """
    empty = Item(
        id="arxiv:2608.00000",
        source="arxiv",
        channel="papers",
        title="Title Only",
        abstract="",
        url="https://arxiv.org/abs/2608.00000",
        published="",
        updated="",
        publish_scope="public",
    )
    summary = FakeGenerator().generate(empty, "ko")
    assert summary.problem and summary.method
    PaperSummary.from_dict(summary.to_dict())  # 예외가 나면 실패


@pytest.mark.parametrize("language", OUTPUT_LANGUAGES)
def test_fake_generator_accepts_every_declared_language(plain_item: Item, language: str):
    """`schema.OUTPUT_LANGUAGES` 에 있는 값은 전부 받아야 합니다 (§6.4 한/영 비교 실험).

    깨뜨리는 법: `_LANGUAGE_NAMES` 에서 `"en"` 을 지우면… 통과합니다(폴백이 있음).
    대신 `_require_known_language` 가 `{"ko"}` 만 허용하게 하면 빨간불.
    확인일: 2026-08-18
    """
    assert FakeGenerator().generate(plain_item, language).problem


def test_fake_generator_rejects_unknown_language(plain_item: Item):
    """알 수 없는 언어 코드는 조용히 한국어로 폴백하지 않고 죽어야 합니다.

    깨뜨리는 법: `_require_known_language` 의 `raise` 를 `return output_language` 로
    바꾸면 빨간불.
    확인일: 2026-08-18
    """
    with pytest.raises(ValueError, match="출력 언어"):
        FakeGenerator().generate(plain_item, "jp")


def test_fake_generator_model_id_is_not_a_vendor_family():
    """★ Fake 의 `model_id` 가 실제 벤더 계열로 읽히면 안 됩니다 (기획안_2 §9.13).

    이 값은 두 곳으로 갑니다 — 원장의 `generator_model`, 그리고 검증자 분리 게이트.
    `"claude-fake"` 같은 이름을 쓰면 ① 원장만 보고는 fake 실행인지 실제 실행인지
    구분이 안 되고(작업규약 §8-9 — fake 로 낸 지표는 거짓입니다) ② 게이트가 이걸
    anthropic 계열로 읽어, 실제 Claude 생성자와 짝지어야 할 검증자 조합 판정이
    엉뚱해집니다.

    처음엔 `assert_verifier_differs` 통과만 봤는데, `model_id` 를 `"claude-fake"` 로
    바꿔도 초록이었습니다(상대가 `rulebased_fake/v1` 이라 계열이 어차피 다름).
    그래서 계열 판정 자체를 봅니다.

    깨뜨리는 법: `FakeGenerator.model_id` 를 `"claude-fake"` 로 바꾸면 빨간불.
    확인일: 2026-08-18
    """
    fake_id = FakeGenerator().model_id
    assert "fake" in fake_id, "원장에서 fake 실행임이 드러나야 합니다"
    assert model_family(fake_id) == fake_id, (
        f"Fake 의 model_id 가 벤더 계열({model_family(fake_id)})로 읽힙니다: {fake_id}"
    )
    # model_id 가 실제로 게이트의 입력이 됩니다 (R5).
    assert_verifier_differs(fake_id, RuleBasedFakeVerifier().model_id)


# ── 프롬프트 템플릿 무결성 ───────────────────────────────────────────────


def test_prompt_template_has_all_placeholders():
    """확정 프롬프트(자산_인벤토리 §2.9)의 자리표시자 5개가 살아 있어야 합니다.

    검사를 **치환 전** 템플릿에서만 하는 이유는 §9.15 입니다.

    기대 목록을 여기에 **다시 적은** 것은 의도입니다. `REQUIRED_PLACEHOLDERS` 를 순회하면
    상수를 줄이는 것만으로 테스트가 통과합니다 — 실제로 §4.2 1차 시도에서 그렇게
    가짜 통과가 났습니다. 확정본이 기준이므로 기준을 테스트가 들고 있습니다.

    깨뜨리는 법: `REQUIRED_PLACEHOLDERS` 에서 `"{output_language}"` 를 지우면 빨간불.
    확인일: 2026-08-18
    """
    template = load_prompt_template()
    expected = ("{output_language}", "{interests}", "{title}", "{categories}", "{abstract}")
    for placeholder in expected:
        assert placeholder in template, placeholder
    assert set(G.REQUIRED_PLACEHOLDERS) == set(expected), (
        "load_prompt_template 이 검사하는 자리표시자 목록이 확정본(자산_인벤토리 §2.9)과 "
        f"어긋납니다: {G.REQUIRED_PLACEHOLDERS}"
    )
    assert "## USER" in template


def test_prompt_keeps_connection_exclusion_rule():
    """★ 프롬프트에서 `connection` 채점 제외 규칙이 사라지면 안 됩니다 (기획안_2 §9.8).

    코드(`VERIFIABLE_FIELDS`)와 프롬프트가 **같은 결정을 말하고 있어야** 합니다.
    한쪽만 바뀌면 모든 요약이 "초록에 없는 주장"으로 감점되고, 지표가 무의미해집니다.

    깨뜨리는 법: `prompts/paper.md` 에서 "채점 대상에서 제외" 문장을 지우면 빨간불.
    (다른 에이전트가 프롬프트를 만지는 중이라, 확인은 tmp 사본에 `load_prompt_template`
    을 물려서 했습니다 — 저장소 파일은 건드리지 않았습니다.)
    확인일: 2026-08-18
    """
    template = load_prompt_template()
    assert "connection" in template
    assert "채점 대상에서 제외" in template
    assert "VERIFIABLE_FIELDS" in template
    assert "connection" not in VERIFIABLE_FIELDS


def test_prompt_template_rejects_missing_placeholder(tmp_path: Path):
    """자리표시자가 빠진 템플릿은 로드 단계에서 죽어야 합니다.

    치환은 조용히 성공하므로(`replace` 는 없는 문자열을 그냥 넘어감), 여기서 안 잡으면
    초록 없는 프롬프트가 API 로 나갑니다.

    깨뜨리는 법: `load_prompt_template` 의 `missing` 검사를 지우면 빨간불.
    확인일: 2026-08-18
    """
    broken = tmp_path / "paper.md"
    broken.write_text(load_prompt_template().replace("{abstract}", "(초록 생략)"), encoding="utf-8")
    with pytest.raises(GeneratorError, match="자리표시자"):
        load_prompt_template(broken)


def test_prompt_template_rejects_missing_user_marker(tmp_path: Path):
    """`## USER` 마커가 없으면 SYSTEM/USER 분리가 성립하지 않습니다.

    깨뜨리는 법: `load_prompt_template` 의 `_USER_MARKER_RE` 검사를 지우면 빨간불.
    확인일: 2026-08-18
    """
    broken = tmp_path / "paper.md"
    broken.write_text(load_prompt_template().replace("## USER", "## 사용자"), encoding="utf-8")
    with pytest.raises(GeneratorError, match="USER"):
        load_prompt_template(broken)


def test_prompt_path_is_resolved_at_call_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """★ 경로 기본값이 정의 시점에 바인딩되면 monkeypatch 가 무효가 됩니다 (§9.1 / R7).

    라벨링 테스트 17개가 실제로 이 상태였습니다 — 통과하면서 아무것도 검증하지 않았습니다.

    깨뜨리는 법: 시그니처를 `def load_prompt_template(path: Path = _PROMPT_PATH)` 로
    되돌리면 빨간불 (실제 `prompts/paper.md` 를 읽어 마커 문장이 안 나옴).
    확인일: 2026-08-18
    """
    stub = tmp_path / "paper.md"
    stub.write_text(
        "SYSTEM 자리 {output_language}\n## USER\n{interests} {title} {categories} {abstract}\n"
        "TMP-SENTINEL",
        encoding="utf-8",
    )
    monkeypatch.setattr(G, "_PROMPT_PATH", stub)
    assert "TMP-SENTINEL" in load_prompt_template()


# ── ★ 치환 — LaTeX 중괄호 (기획안_2 §9.15) ──────────────────────────────


def test_format_would_break_on_this_template():
    """★ 왜 `.format()` 을 쓰지 않는지 **실행으로** 남깁니다.

    확정 프롬프트에는 출력 형식 JSON 예시의 리터럴 중괄호가 있습니다. `.format()` 은
    그걸 자리표시자로 읽고 즉시 터집니다. 이 테스트가 빨간불이면 누군가 치환 방식을
    `.format()` 으로 되돌려도 안전하다고 착각한 것이므로, 그때 이 파일을 읽으세요.

    깨뜨리는 법: — (프로덕션 코드가 아니라 템플릿의 성질을 고정하는 테스트입니다.
    `prompts/paper.md` 의 JSON 예시 블록을 지우면 이 테스트가 빨간불이 됩니다.)
    확인일: 2026-08-18
    """
    template = load_prompt_template()
    with pytest.raises((KeyError, IndexError, ValueError)):
        template.format(
            output_language="한국어",
            interests="i",
            title="t",
            categories="c",
            abstract="a",
        )


def test_render_prompt_survives_latex_braces(item: Item):
    """★★ 이 파일에서 가장 중요한 테스트 (기획안_2 §9.15).

    초록 10%에 LaTeX 중괄호가 있습니다. 여기서 터지면 10편 중 1편이 요약 단계에서
    예외로 죽고, 어느 제외 사유에도 안 잡혀 **원장에 이유 없이 사라집니다.**

    깨뜨리는 법: `render_prompt._substitute` 를
    `part.format(output_language=..., interests=..., ...)` 로 바꾸면 빨간불
    (`KeyError: '6\\\\pi'`).
    확인일: 2026-08-18
    """
    system_text, user_text = render_prompt(
        load_prompt_template(),
        interests="멀티모달 검색",
        title=item.title,
        categories=", ".join(item.categories),
        abstract=item.abstract,
        output_language="ko",
    )
    assert "\\frac{6\\pi}{11}" in user_text
    assert "\\textbf{meta-detection}" in user_text
    assert "{a, b, c}" in user_text
    assert "28.4%" in user_text
    assert system_text


def test_render_prompt_does_not_rescan_substituted_values(item: Item):
    """★ 치환값 안의 자리표시자가 **다시 치환되면 안 됩니다.**

    초록에 `{title}` 이라는 글자가 있으면(있습니다 — 이 논문은 자리표시자 ablation 을
    합니다) 순서가 틀린 구현은 그걸 논문 제목으로 바꿔 버립니다. 예외가 안 나므로
    아무도 모릅니다. 초록이 조용히 오염된 채 API 로 나갑니다.

    깨뜨리는 법: `_substitute` 에서 `{abstract}` 치환을 **맨 앞으로** 옮기면 빨간불.
    확인일: 2026-08-18
    """
    _system_text, user_text = render_prompt(
        load_prompt_template(),
        interests="멀티모달 검색",
        title=item.title,
        categories=", ".join(item.categories),
        abstract=item.abstract,
        output_language="ko",
    )
    assert "the {title} and {abstract} placeholders" in user_text
    assert user_text.count(item.title) == 1, "초록 안의 {title} 이 제목으로 치환됐습니다"


def test_render_prompt_splits_system_and_user(item: Item):
    """SYSTEM 에는 규칙이, USER 에는 관심사와 논문이 들어가야 합니다.

    깨뜨리는 법: `render_prompt` 가 `template` 전체를 (system, user) 양쪽에 그대로
    돌려주게 하면 빨간불.
    확인일: 2026-08-18
    """
    system_text, user_text = render_prompt(
        load_prompt_template(),
        interests="멀티모달 검색, 랭킹",
        title=item.title,
        categories="cs.CV",
        abstract=item.abstract,
        output_language="ko",
    )
    assert "초록에 있는 것만" in system_text
    assert "멀티모달 검색, 랭킹" not in system_text
    assert item.abstract not in system_text

    assert "멀티모달 검색, 랭킹" in user_text
    assert item.title in user_text
    assert "cs.CV" in user_text


def test_render_prompt_uses_human_readable_language_name(item: Item):
    """`{output_language}` 에 코드("ko")를 그대로 넣으면 지시가 약해집니다.

    깨뜨리는 법: `_require_known_language` 가 `output_language` 를 그대로 반환하게
    하면 빨간불.
    확인일: 2026-08-18
    """
    system_ko, _ = render_prompt(
        load_prompt_template(),
        interests="i",
        title="t",
        categories="c",
        abstract="a",
        output_language="ko",
    )
    assert "출력 언어는 한국어 입니다" in system_ko

    system_en, _ = render_prompt(
        load_prompt_template(),
        interests="i",
        title="t",
        categories="c",
        abstract="a",
        output_language="en",
    )
    assert "출력 언어는 English 입니다" in system_en


# ── 응답 파싱 ────────────────────────────────────────────────────────────


def test_parse_summary_strips_code_fences():
    """프롬프트가 "코드펜스를 붙이지 마세요"라고 해도 모델은 종종 붙입니다.

    깨뜨리는 법: `parse_summary` 에서 `_strip_code_fences` 호출을 지우면 빨간불.
    확인일: 2026-08-18
    """
    fenced = f"```json\n{VALID_RESPONSE_JSON}\n```"
    summary = parse_summary(fenced, item_id="arxiv:1")
    assert summary.item_id == "arxiv:1"
    assert summary.key_results == ["28.4% 향상"]


def test_parse_summary_keeps_braces_inside_json_values():
    """본문의 중괄호·백슬래시를 펜스 제거가 건드리면 안 됩니다.

    깨뜨리는 법: `_strip_code_fences` 를 `text.replace("`", "")` 로 바꾸면… 통과합니다.
    `re.sub(r"[{}]", "", ...)` 로 바꾸면 json 파싱이 깨져 빨간불.
    확인일: 2026-08-18
    """
    payload = json.dumps(
        {"problem": "경계 \\frac{6\\pi}{11} 를 좁힌다", "method": "m"}, ensure_ascii=False
    )
    summary = parse_summary(f"```json\n{payload}\n```", item_id="arxiv:1")
    assert summary.problem == "경계 \\frac{6\\pi}{11} 를 좁힌다"


def test_parse_summary_raises_on_non_json():
    """JSON 이 아니면 조용히 빈 요약을 만들지 않고 죽어야 합니다.

    깨뜨리는 법: `json.JSONDecodeError` 를 잡아 `PaperSummary(item_id, "", "")` 를
    돌려주게 하면 빨간불.
    확인일: 2026-08-18
    """
    with pytest.raises(SummaryParseError, match="JSON"):
        parse_summary("죄송합니다, 초록을 요약할 수 없습니다.", item_id="arxiv:1")


def test_parse_summary_raises_on_missing_required_field():
    """`problem`·`method` 가 비면 스키마 위반입니다 (schema.from_dict 계약).

    깨뜨리는 법: `parse_summary` 의 `except ValueError` 블록을 지우면 예외 타입이
    `ValueError` 로 새어 나가 빨간불.
    확인일: 2026-08-18
    """
    with pytest.raises(SummaryParseError, match="스키마"):
        parse_summary(json.dumps({"problem": "", "method": "m"}), item_id="arxiv:1")


def test_parse_summary_raises_on_json_array():
    """배열을 돌려주는 모델도 있습니다. `data["problem"]` 이 TypeError 를 내기 전에 잡습니다.

    깨뜨리는 법: `isinstance(data, dict)` 검사를 지우면 `TypeError` 가 새어 나가 빨간불.
    확인일: 2026-08-18
    """
    with pytest.raises(SummaryParseError, match="객체가 아닙니다"):
        parse_summary("[1, 2, 3]", item_id="arxiv:1")


# ── AnthropicGenerator — 키·모델·요청 ────────────────────────────────────


def test_anthropic_generator_requires_api_key(monkeypatch: pytest.MonkeyPatch):
    """★ 키가 없으면 **생성자에서** 죽습니다. 조용한 폴백 금지 (기획안_2 §4.2).

    Fake 로 조용히 넘어가면, 매일 도는 파이프라인이 초록불을 내면서 가짜 요약을
    발행합니다. 키가 없어 죽는 게 가짜로 성공하는 것보다 낫습니다.

    깨뜨리는 법: `__init__` 의 `MissingAPIKeyError` 를
    `self._api_key = api_key or "dummy"` 로 바꾸면 빨간불.
    확인일: 2026-08-18
    """
    monkeypatch.delenv(_API_KEY_ENV, raising=False)
    with pytest.raises(MissingAPIKeyError, match=_API_KEY_ENV):
        AnthropicGenerator(_TEST_MODEL, interests="i")


def test_anthropic_generator_rejects_blank_api_key(monkeypatch: pytest.MonkeyPatch):
    """Actions 는 미설정 시크릿을 **빈 문자열**로 주입합니다. 있는 걸로 치면 안 됩니다.

    깨뜨리는 법: `__init__` 에서 `.strip()` 을 빼고 `os.environ.get(...) is None` 만
    보게 하면 빨간불.
    확인일: 2026-08-18
    """
    monkeypatch.setenv(_API_KEY_ENV, "   ")
    with pytest.raises(MissingAPIKeyError):
        AnthropicGenerator(_TEST_MODEL, interests="i")


def test_anthropic_generator_has_no_default_model(monkeypatch: pytest.MonkeyPatch):
    """★ 모델 기본값이 없습니다. 호출부(파이프라인)가 정합니다 (기획안_2 §6.2).

    기본값을 두면 단가·가용성을 확인하지 않은 모델이 조용히 운영에 들어갑니다.

    깨뜨리는 법: `__init__(self, model_id: str = "claude-...")` 로 기본값을 주면
    아래 `TypeError` 가 안 나서 빨간불.
    확인일: 2026-08-18
    """
    monkeypatch.setenv(_API_KEY_ENV, _FAKE_KEY)
    with pytest.raises(TypeError):
        AnthropicGenerator(interests="i")  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="model_id"):
        AnthropicGenerator("  ", interests="i")


def test_anthropic_generator_satisfies_protocol(monkeypatch: pytest.MonkeyPatch):
    """깨뜨리는 법: `model_id` 를 `self._model_id` 로 이름만 바꾸면 빨간불.
    확인일: 2026-08-18
    """
    gen = _anthropic(monkeypatch)
    assert isinstance(gen, Generator)
    assert gen.model_id == _TEST_MODEL


def test_anthropic_generator_sends_expected_request(monkeypatch: pytest.MonkeyPatch, item: Item):
    """요청 형태 — 모델·max_tokens·system/user 분리·인증 헤더.

    깨뜨리는 법: `payload["model"]` 을 리터럴 문자열로 하드코딩하면 빨간불.
    확인일: 2026-08-18
    """
    transport = RecordingTransport()
    gen = _anthropic(monkeypatch, transport=transport, max_tokens=777, timeout_sec=12.5)
    gen.generate(item, "ko")

    assert transport.payload["model"] == _TEST_MODEL
    assert transport.payload["max_tokens"] == 777
    assert transport.calls[-1]["timeout_sec"] == 12.5
    assert transport.calls[-1]["headers"]["x-api-key"] == _FAKE_KEY
    assert transport.calls[-1]["headers"]["anthropic-version"]
    assert transport.payload["messages"][0]["role"] == "user"


def test_anthropic_generator_sends_latex_abstract_verbatim(
    monkeypatch: pytest.MonkeyPatch, item: Item
):
    """★ §9.15 의 끝단 확인 — 실제 요청 본문에 초록이 **원문 그대로** 실려야 합니다.

    `render_prompt` 단위 테스트가 통과해도, generate 경로가 초록을 다시 만지면
    (`.format`, f-string, 이스케이프) 여기서 터집니다.

    깨뜨리는 법: `generate_with_usage` 에서 `abstract=item.abstract` 를
    `abstract=item.abstract.format()` 으로 바꾸면 빨간불.
    확인일: 2026-08-18
    """
    transport = RecordingTransport()
    gen = _anthropic(monkeypatch, transport=transport)
    gen.generate(item, "ko")
    assert item.abstract in transport.user_text
    assert "cs.CV, cs.LG" in transport.user_text


def test_anthropic_generator_parses_response(monkeypatch: pytest.MonkeyPatch, item: Item):
    """깨뜨리는 법: `content` 블록 합치기에서 `block.get("type") == "text"` 조건을
    `!= "text"` 로 뒤집으면 빨간불.
    확인일: 2026-08-18
    """
    body = _message_body(VALID_RESPONSE_JSON)
    body["content"].insert(0, {"type": "thinking", "thinking": "무시돼야 합니다"})
    gen = _anthropic(monkeypatch, transport=RecordingTransport(body=body))

    summary = gen.generate(item, "ko")
    assert summary.item_id == item.id
    assert summary.key_results == ["28.4% 향상"]
    assert "무시돼야" not in summary.problem


# ── 토큰·비용 (기획안_2 §12) ────────────────────────────────────────────


def test_usage_is_returned_with_summary(monkeypatch: pytest.MonkeyPatch, item: Item):
    """★ 원장 `cost_usd` 는 실측이어야 합니다. 사용량이 요약과 함께 나와야 합니다.

    깨뜨리는 법: `generate_with_usage` 가 `GenerationUsage(0, 0, 0.0)` 을 돌려주게
    하면 빨간불.
    확인일: 2026-08-18
    """
    gen = _anthropic(
        monkeypatch,
        usd_per_1m_input_tokens=3.0,
        usd_per_1m_output_tokens=15.0,
    )
    summary, usage = gen.generate_with_usage(item, "ko")
    assert isinstance(usage, GenerationUsage)
    assert summary.problem
    assert usage.input_tokens == 1200
    assert usage.output_tokens == 300
    # 1200/1e6*3 + 300/1e6*15 = 0.0036 + 0.0045
    assert usage.cost_usd == pytest.approx(0.0081)
    assert gen.last_usage == usage


def test_cost_is_none_when_pricing_not_given(monkeypatch: pytest.MonkeyPatch, item: Item):
    """★ 단가를 모르면 `None` 입니다. `0.0` 으로 적으면 원장의 비용이 **거짓**이 됩니다.

    기획안_2 §12 는 "구현 시점에 실제 단가를 확인"이 원칙입니다. 0원짜리 실행이
    원장에 쌓이면 가동 비용 검증이 통째로 무의미해집니다.

    깨뜨리는 법: `_cost_usd` 가 단가 없을 때 `0.0` 을 돌려주게 하면 빨간불.
    확인일: 2026-08-18
    """
    _summary, usage = _anthropic(monkeypatch).generate_with_usage(item, "ko")
    assert usage.cost_usd is None
    assert usage.input_tokens == 1200


def test_usage_is_recorded_even_when_parsing_fails(monkeypatch: pytest.MonkeyPatch, item: Item):
    """★ 파싱에 실패해도 그 호출의 **비용은 이미 발생했습니다.**

    파싱 뒤에 기록하면 실패한 호출의 토큰이 원장에서 사라지고, 재생성이 많은 날의
    비용이 실제보다 싸게 보입니다.

    깨뜨리는 법: `self.last_usage = usage` 를 `parse_summary` 호출 **뒤로** 옮기면
    빨간불.
    확인일: 2026-08-18
    """
    gen = _anthropic(
        monkeypatch,
        transport=RecordingTransport(body=_message_body("설명을 곁들인 산문 응답입니다.")),
        usd_per_1m_input_tokens=3.0,
        usd_per_1m_output_tokens=15.0,
    )
    with pytest.raises(SummaryParseError):
        gen.generate(item, "ko")
    assert gen.last_usage is not None
    assert gen.last_usage.input_tokens == 1200
    assert gen.last_usage.cost_usd == pytest.approx(0.0081)


# ── 실패 경로 ────────────────────────────────────────────────────────────


def test_http_error_raises_and_never_leaks_key(monkeypatch: pytest.MonkeyPatch, item: Item):
    """★ 예외 메시지에 API 키가 들어가면 안 됩니다 (CLAUDE.md §3-2).

    예외는 로그·Actions 요약으로 흘러갑니다. 이 저장소는 public 입니다.

    깨뜨리는 법: `GeneratorError` 메시지에 `self._api_key` 를 넣으면 빨간불.
    확인일: 2026-08-18
    """
    body = {"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}}
    gen = _anthropic(monkeypatch, transport=RecordingTransport(status=429, body=body))
    with pytest.raises(GeneratorError) as excinfo:
        gen.generate(item, "ko")
    message = str(excinfo.value)
    assert "429" in message
    assert "slow down" in message
    assert _FAKE_KEY not in message


def test_empty_text_block_raises(monkeypatch: pytest.MonkeyPatch, item: Item):
    """텍스트가 없는 응답을 빈 요약으로 통과시키면 안 됩니다.

    깨뜨리는 법: `if not text.strip(): raise` 를 지우면 `SummaryParseError` 대신
    JSON 파싱 실패 경로로 빠지는데, 메시지가 원인을 못 가리킵니다.
    (`match="text 블록"` 이 빨간불)
    확인일: 2026-08-18
    """
    body = _message_body("")
    gen = _anthropic(monkeypatch, transport=RecordingTransport(body=body))
    with pytest.raises(SummaryParseError, match="text 블록"):
        gen.generate(item, "ko")


def test_truncated_response_names_max_tokens(monkeypatch: pytest.MonkeyPatch, item: Item):
    """★ `stop_reason=max_tokens` 는 재요청으로 안 풀립니다 — 같은 자리에서 또 잘립니다.

    "JSON 이 아닙니다"로 넘기면 호출부가 1회 재요청을 낭비하고 같은 실패를 봅니다.

    깨뜨리는 법: `stop_reason == "max_tokens"` 분기를 지우면 메시지가
    "응답이 JSON 이 아닙니다"로 바뀌어 빨간불.
    확인일: 2026-08-18
    """
    truncated = _message_body('{"problem": "잘린 응', stop_reason="max_tokens")
    gen = _anthropic(monkeypatch, transport=RecordingTransport(body=truncated), max_tokens=64)
    with pytest.raises(SummaryParseError, match="max_tokens"):
        gen.generate(item, "ko")


def test_generator_does_not_retry_internally(monkeypatch: pytest.MonkeyPatch, item: Item):
    """★ 재시도는 파이프라인 책임입니다 (기획안_2 §6.3 — `retries` 는 원장 필드).

    생성자가 몰래 재시도하면 원장의 `retries` 가 실제 호출 수와 어긋나고, 비용도
    맞지 않게 됩니다.

    깨뜨리는 법: `generate_with_usage` 에 `for _ in range(2):` 재시도 루프를 넣으면
    빨간불.
    확인일: 2026-08-18
    """
    transport = RecordingTransport(body=_message_body("JSON 이 아닌 응답"))
    gen = _anthropic(monkeypatch, transport=transport)
    with pytest.raises(SummaryParseError):
        gen.generate(item, "ko")
    assert len(transport.calls) == 1, f"내부 재시도 {len(transport.calls)}회"


# ── 재생성 피드백 (기획안_2 §6.3) ──────────────────────────────────────

#: 검증자가 지목한 문장을 인용한 지적 사항. **초록에서 인용하므로 LaTeX 중괄호가
#: 그대로 들어 있습니다** — 재생성 경로가 §9.15 의 두 번째 지뢰밭인 이유입니다.
FEEDBACK_WITH_BRACES = (
    "직전 시도의 요약은 검증에서 다음 지적을 받았습니다.\n"
    "1. [key_results / numeric] 경계를 \\frac{6\\pi}{11} 로 좁혔다고 썼습니다\n"
    "   → 초록은 {a, b, c} 집합에 대해서만 말합니다. {title} 항목도 근거가 없습니다."
)


@pytest.mark.parametrize("generator_factory", ["fake", "anthropic"])
def test_generators_accept_feedback_keyword(generator_factory: str, monkeypatch, item: Item):
    """★ 재생성 루프의 계약 — 구현체는 `feedback` 키워드를 받아야 합니다 (기획안_2 §6.3).

    오케스트레이터는 `inspect.signature` 로 지원 여부를 보고, **지원하지 않으면 조용히
    재호출하지 않고 멈춥니다.** 즉 여기가 어긋나면 재생성이 아예 못 돌고, M2 DoD 의
    "재생성률" 이 영원히 0 으로 기록됩니다.

    시그니처로 검사하는 이유: 오케스트레이터 모듈을 import 해서 확인하면 그쪽 파일이
    바뀔 때 이 테스트가 같이 흔들립니다. 계약만 여기서 고정합니다.

    깨뜨리는 법: `FakeGenerator.generate` 에서 `*, feedback: str | None = None` 을
    지우면 빨간불.
    확인일: 2026-08-18
    """
    import inspect

    if generator_factory == "fake":
        generate = FakeGenerator().generate
    else:
        generate = _anthropic(monkeypatch).generate

    parameters = inspect.signature(generate).parameters
    assert "feedback" in parameters, "재생성 피드백을 받지 못합니다 (기획안_2 §6.3)"
    assert parameters["feedback"].default is None, "피드백은 선택 인자여야 합니다"
    assert parameters["feedback"].kind is inspect.Parameter.KEYWORD_ONLY

    # 실제로 호출도 돼야 합니다 — 시그니처만 맞고 안에서 터지면 의미가 없습니다.
    assert generate(item, "ko", feedback="지적 사항").problem


def test_feedback_is_appended_to_user_message(monkeypatch: pytest.MonkeyPatch, item: Item):
    """지적 사항이 실제 요청에 실려야 합니다. 안 실리면 같은 실수가 반복됩니다.

    확정 프롬프트(`paper.md`)는 건드리지 않고 USER 메시지 끝에 붙입니다.

    깨뜨리는 법: `generate_with_usage` 의 `if feedback:` 블록을 지우면 빨간불.
    확인일: 2026-08-18
    """
    transport = RecordingTransport()
    gen = _anthropic(monkeypatch, transport=transport)
    gen.generate(item, "ko", feedback=FEEDBACK_WITH_BRACES)

    user_text = transport.user_text
    assert FEEDBACK_WITH_BRACES in user_text
    assert "직전 시도에 대한 검증 지적" in user_text
    # 원래 프롬프트가 지적 사항에 밀려 사라지면 안 됩니다.
    assert item.abstract in user_text
    assert user_text.index(item.abstract) < user_text.index(FEEDBACK_WITH_BRACES)


def test_feedback_with_latex_braces_is_not_reformatted(
    monkeypatch: pytest.MonkeyPatch, item: Item
):
    """★ §9.15 의 두 번째 지뢰밭 — 지적 사항에는 인용된 초록의 중괄호가 들어 있습니다.

    최초 생성이 아니라 **재생성일 때만** 터지므로, faithfulness 가 낮은 아이템만 골라서
    죽습니다. 즉 요약이 어려운 논문일수록 조용히 사라집니다.

    깨뜨리는 법: `user_text = "\\n\\n".join([...])` 를
    `user_text = f"{user_text}\\n\\n{_FEEDBACK_HEADING}\\n\\n".format(feedback)` 처럼
    format 을 거치게 하면 빨간불.
    확인일: 2026-08-18
    """
    transport = RecordingTransport()
    gen = _anthropic(monkeypatch, transport=transport)
    gen.generate(item, "ko", feedback=FEEDBACK_WITH_BRACES)

    user_text = transport.user_text
    assert "\\frac{6\\pi}{11}" in user_text
    assert "{a, b, c} 집합" in user_text
    assert "{title} 항목도 근거가 없습니다" in user_text


def test_no_feedback_section_on_first_attempt(monkeypatch: pytest.MonkeyPatch, item: Item):
    """최초 생성에는 지적 구획이 없어야 합니다. 빈 구획을 붙이면 모델이 혼란스러워합니다.

    깨뜨리는 법: `if feedback:` 을 `if feedback is not None:` 로 두고 호출부가 `""` 를
    넘기면 빈 구획이 붙습니다. 아래 두 번째 assert 가 그걸 잡습니다.
    확인일: 2026-08-18
    """
    transport = RecordingTransport()
    gen = _anthropic(monkeypatch, transport=transport)

    gen.generate(item, "ko")
    assert "직전 시도에 대한 검증 지적" not in transport.user_text

    gen.generate(item, "ko", feedback="")
    assert "직전 시도에 대한 검증 지적" not in transport.user_text


def test_fake_generator_ignores_feedback_content(plain_item: Item):
    """★ Fake 는 피드백을 **받되 반영하지 않습니다** (작업규약 §8-9).

    반영하는 척하면 fake 실행에서 "재생성이 점수를 올린다"는 거짓 신호가 나옵니다.
    fake 로는 재생성 효과를 측정할 수 없다는 게 정직한 상태입니다.

    깨뜨리는 법: `FakeGenerator.generate` 가 `feedback` 을 `problem` 앞에 붙이게 하면
    빨간불.
    확인일: 2026-08-18
    """
    fake = FakeGenerator()
    first = fake.generate(plain_item, "ko").to_dict()
    regenerated = fake.generate(plain_item, "ko", feedback=FEEDBACK_WITH_BRACES).to_dict()
    assert first == regenerated


# ── 키가 있을 때만 도는 실 API 테스트 ───────────────────────────────────


@pytest.mark.skipif(
    not (os.environ.get(_API_KEY_ENV) and os.environ.get("RADAR_GENERATOR_MODEL")),
    reason=(
        "생성자 키/모델 미설정. ANTHROPIC_API_KEY 와 RADAR_GENERATOR_MODEL 이 둘 다 "
        "있을 때만 실제 API 를 부릅니다 (모델 기본값 금지 — 기획안_2 §6.2)"
    ),
)
def test_real_api_returns_parsable_summary(plain_item: Item):
    """실제 Messages API 왕복 1건. 유료 호출이므로 1건·짧은 초록으로 고정합니다.

    깨뜨리는 법: `api_url` 을 잘못된 엔드포인트로 바꾸면 빨간불.
    확인일: — (키 미발급. 키가 오면 이 테스트가 실제 왕복을 판정합니다)
    """
    gen = AnthropicGenerator(
        os.environ["RADAR_GENERATOR_MODEL"],
        interests="멀티모달 검색, 랭킹",
        max_tokens=512,
    )
    summary, usage = gen.generate_with_usage(plain_item, "ko")
    assert summary.item_id == plain_item.id
    assert summary.problem and summary.method
    assert usage.input_tokens > 0 and usage.output_tokens > 0
