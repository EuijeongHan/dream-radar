"""요약 생성자 (기획안_2 §6, 델타 §D6.2).

지금 있는 것
-----------
    Generator           포트 (프로토콜). `model_id` 속성이 필수입니다 —
                        검증자 분리 게이트(`gates.assert_verifier_differs`)가
                        생성자·검증자의 계열을 이 값으로 비교합니다 (기획안_2 §6.2).
    FakeGenerator       키 불필요. 초록 첫 문장들로 결정적 요약을 만듭니다.
                        `RADAR_MODE=fake` 배선·테스트 전용 (델타 §D6.2).
    AnthropicGenerator  키 필요. Messages API 를 requests 로 직접 호출합니다.
                        SDK 를 쓰지 않는 이유: Actions 설치 시간 + 기획안_2 §9.10
                        (requirements 는 표준 라이브러리 + PyYAML·requests 뿐).

프롬프트 치환 규칙 ★ (기획안_2 §9.15)
------------------------------------
arXiv 초록의 10%에 LaTeX 중괄호(`\\frac{6π}{11}`)가 있고, 템플릿 자체에도 JSON 예시의
리터럴 중괄호가 있습니다. 따라서:

- 치환은 **str.replace 1회씩, 단 한 번만.** `.format()` 은 템플릿의 JSON 예시
  중괄호에서 즉시 KeyError 를 냅니다.
- 치환한 결과를 다시 format 하거나 f-string 에 넣지 않습니다.
- **치환 후 텍스트에 미치환 검사를 하지 않습니다** — 초록의 `{...}` 를 오탐합니다.
  자리표시자 존재 검사는 치환 **전** 템플릿(`load_prompt_template`)에서만 합니다.

호출부(파이프라인)가 지켜야 할 것
--------------------------------
- 모델 id 는 호출부가 결정합니다. 이 모듈에 기본 모델이 없습니다 (기획안_2 §6.2 —
  "구체 모델은 구현 시점에 가용성과 단가를 확인하고 고른다").
- 검증 요청을 보내기 직전에 `gates.assert_verifier_differs(generator.model_id,
  verifier.model_id)` 를 호출하세요 (기획안_2 §6.2, CLAUDE.md §3-9).
- 단가는 인자로 넘기세요 (`usd_per_1m_*`). 하드코딩하지 않는 이유: 기획안_2 §12 —
  "구현 시작 시점에 실제 단가를 확인"이 원칙이고, 단가는 변동합니다.
- **재생성일 때는 직전 판정의 지적 사항을 `feedback=` 로 넘기세요** (기획안_2 §6.3).
  안 넘기면 프롬프트가 직전과 완전히 같아서 같은 실수가 그대로 나옵니다.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

import requests

from src.core.models import Item
from src.summarize.schema import OUTPUT_LANGUAGES, PaperSummary

#: 확정 프롬프트 (자산_인벤토리 §2.9). CWD 가 아니라 이 모듈 기준으로 해석합니다 —
#: 상대경로 `Path("data")` 계약(저장소 루트 실행)과 달리 프롬프트는 코드의 일부입니다.
_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "paper.md"

#: 치환 전 템플릿에서 존재를 검사하는 자리표시자 5개 (자산_인벤토리 §2.9 확정).
REQUIRED_PLACEHOLDERS: tuple[str, ...] = (
    "{output_language}",
    "{interests}",
    "{title}",
    "{categories}",
    "{abstract}",
)

#: SYSTEM/USER 분리 마커. 템플릿의 `## USER` 헤딩 줄입니다.
_USER_MARKER_RE = re.compile(r"^## USER[ \t]*$", re.MULTILINE)

#: `{output_language}` 에 넣을 사람이 읽는 언어 이름. 코드 그대로("ko") 넣으면
#: "출력 언어는 ko 입니다"가 되어 지시가 약해집니다.
_LANGUAGE_NAMES: dict[str, str] = {"ko": "한국어", "en": "English"}

#: 재생성 시 USER 메시지 **끝에 이어 붙이는** 구획 제목 (기획안_2 §6.3).
#:
#: 확정 프롬프트(`prompts/paper.md`)에 자리표시자를 새로 뚫지 않는 이유: 그 파일은
#: `schema.py` 와 함께 확정본이고, 자리표시자를 늘리면 `REQUIRED_PLACEHOLDERS` 와
#: 원장 스키마까지 같이 흔들립니다. 지적 사항은 템플릿의 일부가 아니라 **이번 시도에만
#: 붙는 추가 지시**이므로 메시지 수준에서 이어 붙입니다.
_FEEDBACK_HEADING = "### 직전 시도에 대한 검증 지적"

_API_KEY_ENV = "ANTHROPIC_API_KEY"
_DEFAULT_API_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"

#: transport(url, headers, payload, timeout_sec) -> (status_code, 응답 JSON).
#: arxiv.py 의 Transport 훅과 같은 목적 — 네트워크·키 없이 테스트하기 위한 주입 지점.
Transport = Callable[[str, dict[str, str], dict[str, Any], float], tuple[int, dict[str, Any]]]


class GeneratorError(RuntimeError):
    """생성자 실패. 조용히 넘어가지 않습니다 (작업규약 §8-3)."""


class MissingAPIKeyError(GeneratorError):
    """API 키 부재. 키가 없어 죽는 게 가짜로 성공하는 것보다 낫습니다 (기획안_2 §4.2)."""


class SummaryParseError(GeneratorError):
    """응답을 `PaperSummary` 로 파싱하지 못했습니다.

    호출부는 이 예외를 받으면 **1회에 한해** 재요청할 수 있습니다 — 여기서 내부
    재시도를 하지 않는 이유는 재시도 횟수·기록(`retries`)이 파이프라인의 원장 책임이기
    때문입니다 (기획안_2 §6.3).
    """


@runtime_checkable
class Generator(Protocol):
    """생성자 포트 (델타 §D6.2). `Verifier` 프로토콜과 같은 형태입니다."""

    #: 원장 `generator_model` 로 기록되고, `assert_verifier_differs` 가 비교하는 값.
    model_id: str

    def generate(self, item: Item, output_language: str) -> PaperSummary: ...

    # 재생성(기획안_2 §6.3)을 지원하는 구현체는 여기에 키워드 인자 하나를 더 받습니다:
    #
    #     def generate(self, item, output_language, *, feedback: str | None = None)
    #
    # 프로토콜에는 넣지 않습니다. 최소 계약은 2인자이고, 오케스트레이터가
    # `inspect.signature` 로 지원 여부를 보고 **지원하지 않으면 조용히 재호출하지 않고
    # 멈추도록** 짜여 있기 때문입니다. 이 모듈의 구현체 둘은 모두 받습니다.


def _require_known_language(output_language: str) -> str:
    if output_language not in OUTPUT_LANGUAGES:
        raise ValueError(
            f"알 수 없는 출력 언어: {output_language!r} (허용: {list(OUTPUT_LANGUAGES)}) — "
            "schema.OUTPUT_LANGUAGES 참조"
        )
    return _LANGUAGE_NAMES.get(output_language, output_language)


def load_prompt_template(path: Path | None = None) -> str:
    """확정 프롬프트를 읽고 **치환 전** 무결성을 검사합니다.

    자리표시자 존재 검사를 여기서만 하는 이유: 치환 후 텍스트에는 초록의 LaTeX
    중괄호가 들어 있어 미치환 검사가 오탐을 냅니다 (기획안_2 §9.15).

    `path` 기본값이 모듈 상수가 아니라 `None` 인 이유: 기본 인자는 정의 시점에
    바인딩되어 테스트의 monkeypatch 가 무효가 됩니다 (기획안_2 §9.1 / R7).
    """
    resolved = path if path is not None else _PROMPT_PATH
    template = resolved.read_text(encoding="utf-8")
    missing = [ph for ph in REQUIRED_PLACEHOLDERS if ph not in template]
    if missing:
        raise GeneratorError(f"프롬프트 템플릿에 자리표시자가 없습니다: {missing} ({resolved})")
    if not _USER_MARKER_RE.search(template):
        raise GeneratorError(f"프롬프트 템플릿에 '## USER' 마커가 없습니다 ({resolved})")
    return template


def render_prompt(
    template: str,
    *,
    interests: str,
    title: str,
    categories: str,
    abstract: str,
    output_language: str,
) -> tuple[str, str]:
    """템플릿을 (system, user) 로 나누고 자리표시자를 채웁니다.

    - 분리를 치환보다 **먼저** 합니다 — 치환값에 우연히 '## USER' 가 들어 있어도
      분리가 흔들리지 않습니다.
    - 치환은 자리표시자당 `str.replace` 1회. `.format()` 금지 (기획안_2 §9.15 —
      템플릿의 JSON 예시 중괄호와 초록의 LaTeX 중괄호에서 KeyError).
    - `{abstract}` 를 **마지막에** 치환합니다. 다른 값 치환이 끝난 뒤라, 초록 본문이
      다시 스캔되어 재치환될 일이 없습니다 (str.replace 는 치환값을 재스캔하지
      않지만, 뒤이은 다른 replace 호출은 전체를 다시 훑기 때문입니다).
    - 치환 후 검사는 하지 않습니다. 검사는 `load_prompt_template` 에서 끝났습니다.
    """
    language_name = _require_known_language(output_language)
    marker = _USER_MARKER_RE.search(template)
    if marker is None:
        raise GeneratorError("프롬프트 템플릿에 '## USER' 마커가 없습니다")
    system_part = template[: marker.start()].strip()
    user_part = template[marker.end() :].strip()

    def _substitute(part: str) -> str:
        part = part.replace("{output_language}", language_name)
        part = part.replace("{interests}", interests)
        part = part.replace("{title}", title)
        part = part.replace("{categories}", categories)
        return part.replace("{abstract}", abstract)  # ★ 마지막 — 위 docstring 참조

    return _substitute(system_part), _substitute(user_part)


def _strip_code_fences(text: str) -> str:
    """앞뒤 코드펜스(```json ... ```)만 벗깁니다. 본문 중괄호는 건드리지 않습니다."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[^\n]*\n?", "", stripped)
        stripped = re.sub(r"\n?```\s*$", "", stripped)
    return stripped.strip()


def parse_summary(text: str, *, item_id: str) -> PaperSummary:
    """모델 응답 텍스트 → `PaperSummary` (R4 — 스키마는 schema.py 재사용).

    실패는 전부 `SummaryParseError` 로 올립니다. 호출부가 1회 재요청 여부를
    결정합니다 (내부 재시도 없음 — 클래스 docstring 참조).
    """
    cleaned = _strip_code_fences(text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise SummaryParseError(
            f"응답이 JSON 이 아닙니다 ({item_id}): {exc}. 호출부는 1회 재요청할 수 "
            f"있습니다. 응답 앞부분: {cleaned[:200]!r}"
        ) from exc
    if not isinstance(data, dict):
        raise SummaryParseError(
            f"응답 JSON 이 객체가 아닙니다 ({item_id}): {type(data).__name__}. "
            "호출부는 1회 재요청할 수 있습니다"
        )
    try:
        return PaperSummary.from_dict(data, item_id=item_id)
    except ValueError as exc:
        raise SummaryParseError(
            f"응답 JSON 이 요약 스키마에 맞지 않습니다 ({item_id}): {exc}. "
            "호출부는 1회 재요청할 수 있습니다"
        ) from exc


#: 문장 경계. 결정성만 필요하므로 정교하지 않아도 됩니다 (Fake 는 배선 테스트 전용).
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


class FakeGenerator:
    """초록 첫 문장들을 그대로 옮기는 결정적 생성자. 키가 필요 없습니다.

    `RADAR_MODE=fake` 전 구간 배선(델타 §D6.2)과 테스트 전용입니다.
    초록 문장을 그대로 옮기므로 왜곡이 없어 `RuleBasedFakeVerifier` 와 짝지으면
    배선 테스트에서 faithfulness 1.0 이 나옵니다.

    **요약 품질 평가에 쓰지 마세요** — fake 로 낸 지표는 거짓입니다 (작업규약 §8-9).
    """

    model_id = "fake_generator/v1"

    def generate(
        self, item: Item, output_language: str, *, feedback: str | None = None
    ) -> PaperSummary:
        """`feedback` 은 **받기만 하고 쓰지 않습니다** (기획안_2 §6.3).

        받는 이유: 재생성 루프가 fake 배선에서도 돌아야 합니다. 안 받으면 오케스트레이터가
        `RegenerationUnsupportedError` 로 멈춰, 키 없는 전 구간 배선 테스트(델타 §D6.2)가
        재생성 경로만 못 밟습니다.

        쓰지 않는 이유: Fake 는 초록 문장을 그대로 옮길 뿐이라 지적을 반영할 수단이
        없습니다. 지적을 받은 척 다른 문장을 내놓으면 **재생성이 점수를 올린다는 거짓
        신호**가 fake 실행에서 나옵니다 (작업규약 §8-9). 재생성해도 같은 요약이 나오고,
        그래서 fake 로는 재생성 효과를 측정할 수 없다는 게 정직한 상태입니다.
        """
        _require_known_language(output_language)
        sentences = [
            s.strip() for s in _SENTENCE_SPLIT_RE.split(item.abstract.strip()) if s.strip()
        ]
        # problem·method 는 스키마 필수 필드라 비울 수 없습니다 (schema.from_dict 계약).
        fallback = item.title.strip() or item.id
        problem = sentences[0] if sentences else fallback
        method = sentences[1] if len(sentences) > 1 else fallback
        key_results = [sentences[2]] if len(sentences) > 2 else []
        return PaperSummary(
            item_id=item.id,
            problem=problem,
            method=method,
            key_results=key_results,
            limitations=[],  # 초록에 한계 언급이 없으면 빈 배열 — 프롬프트 규칙 4와 동일
            connection="",  # 채점 제외 필드 (기획안_2 §9.8). fake 는 지어내지 않습니다
        )


@dataclass(frozen=True)
class GenerationUsage:
    """1회 생성의 토큰 사용량. 원장 `cost_usd` 실측 누적용 (기획안_2 §12)."""

    input_tokens: int
    output_tokens: int
    #: 단가 미지정이면 None. 0.0 으로 적으면 원장의 비용 실측이 거짓이 됩니다.
    cost_usd: float | None


def _requests_transport(
    url: str, headers: dict[str, str], payload: dict[str, Any], timeout_sec: float
) -> tuple[int, dict[str, Any]]:
    response = requests.post(url, headers=headers, json=payload, timeout=timeout_sec)
    try:
        body = response.json()
    except ValueError:
        body = {"raw_text": response.text[:500]}
    return response.status_code, body


class AnthropicGenerator:
    """Anthropic Messages API 를 `requests` 로 직접 호출하는 생성자.

    - 키는 환경변수 `ANTHROPIC_API_KEY` 에서만 읽습니다. 없으면 생성자가
      `MissingAPIKeyError` 로 죽습니다 — 조용한 폴백 금지 (기획안_2 §4.2).
      키 없이 돌려야 하면 `FakeGenerator` 를 쓰세요 (델타 §D6.2).
    - `model_id` 에 기본값이 없습니다. 호출부(파이프라인)가 결정합니다 (기획안_2 §6.2).
    - 단가(`usd_per_1m_*`)도 인자입니다. 하드코딩 금지 — 기획안_2 §12.
    - 파싱 실패는 `SummaryParseError`. 재요청(최대 1회) 여부는 호출부 책임입니다.
    - 원장에 `cost_usd` 를 적으려면 `generate_with_usage` 를 쓰세요. `generate` 는
      프로토콜을 만족시키기 위한 얇은 래퍼입니다.
    """

    def __init__(
        self,
        model_id: str,
        interests: str,
        *,
        max_tokens: int = 1024,
        timeout_sec: float = 60.0,
        usd_per_1m_input_tokens: float | None = None,
        usd_per_1m_output_tokens: float | None = None,
        prompt_path: Path | None = None,
        api_url: str | None = None,
        transport: Transport | None = None,
    ) -> None:
        if not model_id or not model_id.strip():
            raise ValueError(
                "model_id 가 비어 있습니다. 기본 모델은 없습니다 — 호출부(파이프라인)가 "
                "결정합니다 (기획안_2 §6.2)"
            )
        api_key = os.environ.get(_API_KEY_ENV, "").strip()
        if not api_key:
            raise MissingAPIKeyError(
                f"환경변수 {_API_KEY_ENV} 가 없습니다. 조용히 폴백하지 않습니다 — "
                "키가 없어 죽는 게 가짜로 성공하는 것보다 낫습니다 (기획안_2 §4.2). "
                "키 없이 돌리려면 FakeGenerator (델타 §D6.2)"
            )
        self.model_id = model_id.strip()
        self._api_key = api_key  # 로그·예외 메시지에 절대 넣지 않습니다 (CLAUDE.md §3-2)
        self._interests = interests
        self._max_tokens = max_tokens
        self._timeout_sec = timeout_sec
        self._usd_per_1m_input = usd_per_1m_input_tokens
        self._usd_per_1m_output = usd_per_1m_output_tokens
        # 호출 시점 해석 (기획안_2 §9.1 / R7) — 모듈 상수를 기본 인자로 쓰지 않습니다.
        self._template = load_prompt_template(prompt_path)
        self._api_url = api_url if api_url is not None else _DEFAULT_API_URL
        self._transport: Transport = transport if transport is not None else _requests_transport
        #: 마지막 호출의 토큰 사용량. 파싱 실패여도 기록됩니다 — 실패한 호출도 비용이
        #: 들었고, 원장 cost_usd 는 실측이어야 합니다 (기획안_2 §12).
        #: 성공 경로에서는 `generate_with_usage` 의 반환값을 쓰세요. 이 속성은 예외가
        #: 난 호출의 비용을 건지기 위한 것입니다.
        self.last_usage: GenerationUsage | None = None

    def _cost_usd(self, input_tokens: int, output_tokens: int) -> float | None:
        if self._usd_per_1m_input is None or self._usd_per_1m_output is None:
            return None
        return (
            input_tokens * self._usd_per_1m_input / 1_000_000
            + output_tokens * self._usd_per_1m_output / 1_000_000
        )

    def generate(
        self, item: Item, output_language: str, *, feedback: str | None = None
    ) -> PaperSummary:
        """`Generator` 프로토콜 구현. 토큰 사용량이 필요하면 `generate_with_usage`."""
        summary, _usage = self.generate_with_usage(item, output_language, feedback=feedback)
        return summary

    def generate_with_usage(
        self, item: Item, output_language: str, *, feedback: str | None = None
    ) -> tuple[PaperSummary, GenerationUsage]:
        """요약과 이번 호출의 토큰 사용량을 **함께** 돌려줍니다.

        `generate` 뒤에 `last_usage` 를 읽는 것과 결과는 같지만, 사용량이 어느 아이템의
        것인지가 반환값으로 묶입니다. 원장은 아이템별로 `cost_usd` 를 적으므로
        (기획안_2 §4.4·§12) 파이프라인은 이쪽을 쓰세요 — 속성을 나중에 읽는 방식은
        호출 순서가 조금만 어긋나도 **다른 아이템의 비용을 적으면서 조용히 통과**합니다.

        예외로 나가는 경우(HTTP 오류·파싱 실패)에는 반환값이 없으므로 그때만
        `last_usage` 를 읽으세요. 실패한 호출도 비용이 발생했고, 원장의 비용은
        실측이어야 합니다.

        `feedback` 은 재생성 시 직전 판정의 지적 사항입니다 (기획안_2 §6.3 —
        "같은 프롬프트로 다시 부르면 같은 실수가 나옵니다"). USER 메시지 끝에
        이어 붙습니다.
        """
        system_text, user_text = render_prompt(
            self._template,
            interests=self._interests,
            title=item.title,
            categories=", ".join(item.categories),
            abstract=item.abstract,
            output_language=output_language,
        )
        if feedback:
            # ★ **이어 붙이기만** 합니다 (기획안_2 §9.15). 지적 사항에는 왜곡으로 지목된
            # 요약·초록 문장이 인용돼 있어 LaTeX 중괄호가 그대로 들어 있습니다.
            # join 은 값을 다시 스캔하지 않습니다 — 여기서 .format 이나 f-string 을 쓰면
            # 재생성이 필요한 아이템만 골라서 터집니다.
            user_text = "\n\n".join([user_text, _FEEDBACK_HEADING, feedback])
        payload: dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": self._max_tokens,
            "system": system_text,
            "messages": [{"role": "user", "content": user_text}],
        }
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        status, body = self._transport(self._api_url, headers, payload, self._timeout_sec)
        if status != 200:
            # 키는 예외 메시지에 넣지 않습니다 (CLAUDE.md §3-2). 응답 본문은 서버발이라
            # 키가 들어 있지 않습니다.
            detail = ""
            if isinstance(body, dict):
                error = body.get("error")
                if isinstance(error, dict):
                    detail = str(error.get("message", ""))[:300]
                if not detail:
                    detail = str(body)[:300]
            raise GeneratorError(
                f"Anthropic Messages API 오류 (HTTP {status}, model={self.model_id}, "
                f"item={item.id}): {detail}"
            )

        raw_usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        input_tokens = int(raw_usage.get("input_tokens") or 0)
        output_tokens = int(raw_usage.get("output_tokens") or 0)
        # 파싱보다 먼저 기록 — 파싱이 실패해도 이 호출의 비용은 실제로 발생했습니다.
        usage = GenerationUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=self._cost_usd(input_tokens, output_tokens),
        )
        self.last_usage = usage

        # ★ max_tokens 에서 잘린 응답은 JSON 이 미완성입니다. 같은 max_tokens 로 재요청하면
        # 같은 자리에서 다시 잘리므로, "JSON 이 아닙니다"로 넘기면 호출부가 1회 재요청을
        # 낭비하고 같은 실패를 봅니다. 원인을 메시지에 박아 둡니다.
        if body.get("stop_reason") == "max_tokens":
            raise SummaryParseError(
                f"응답이 max_tokens({self._max_tokens})에서 잘렸습니다 ({item.id}, "
                f"model={self.model_id}). 같은 max_tokens 로 재요청하면 같은 자리에서 다시 "
                "잘립니다 — 호출부는 max_tokens 를 올려 재요청하세요"
            )

        text = "".join(
            block.get("text", "")
            for block in body.get("content", [])
            if isinstance(block, dict) and block.get("type") == "text"
        )
        if not text.strip():
            raise SummaryParseError(
                f"응답에 text 블록이 없습니다 ({item.id}, model={self.model_id}). "
                f"stop_reason={body.get('stop_reason')!r}. 호출부는 1회 재요청할 수 있습니다"
            )
        return parse_summary(text, item_id=item.id), usage
