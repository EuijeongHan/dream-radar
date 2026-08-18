"""OpenRouter 경유 LLM faithfulness 검증자 (기획안_2 §6.2, docs/03_검증자_선정.md).

`faithfulness.py` 가 "M2 나머지에서 구현"이라고 남겨 둔 그 구현체입니다.
`RuleBasedFakeVerifier` 는 수치만 봅니다 — 없는 인과(②)와 한계 반전(③)은 규칙으로
판정할 수 없어서 기획안 §8 M2 DoD("왜곡 3건을 모두 잡는다")는 이 구현체로만
판정됩니다.

모델 선정 (docs/03_검증자_선정.md, 2026-08-16 확정)
--------------------------------------------------
    기본  openai/gpt-5.6-luna       $0.20 / $1.20 per 1M (정가)
    폴백  deepseek/deepseek-v4-flash $0.14 / $0.28 per 1M (정가)

생성자는 Claude 계열이므로 둘 다 `assert_verifier_differs` 를 통과합니다
(CLAUDE.md §3-9 / 기획안_2 §9.13 — ID 가 아니라 **계열**로 비교합니다).
위 슬러그는 **호출 직전 재확인**해야 합니다. 프로바이더 라우팅·이름이 바뀝니다.
단가는 생성자와 같은 이유로 하드코딩하지 않습니다 (인자, 기획안_2 §12) — 현재
프로모 할인가로 예산을 잡으면 원장 `cost_usd` 가 거짓이 됩니다.

프롬프트 치환 규칙 ★ (기획안_2 §9.15)
------------------------------------
초록의 10%에 LaTeX 중괄호(`\\frac{6π}{11}`)가 있고, 템플릿 자체에도 JSON 예시의
리터럴 중괄호가 있습니다. generator.py 와 **같은 규약**을 씁니다:

- 치환은 자리표시자당 `str.replace` 1회. `.format()` 은 템플릿의 JSON 예시
  중괄호에서 즉시 KeyError 를 냅니다.
- 치환 결과를 다시 format 하거나 f-string 에 넣지 않습니다.
- **치환 후 미치환 검사를 하지 않습니다** — 초록의 `{...}` 를 오탐합니다. 자리표시자
  존재 검사는 치환 **전** 템플릿(`load_verify_template`)에서만 합니다.

★ 검증자에게 넘기는 것은 `summary.verifiable_dict()` 뿐입니다
------------------------------------------------------------
`connection`("내 프로젝트와의 연결점")은 독자의 맥락에서 나오는 판단이라 초록에 근거가
있을 수 없습니다. 통째로 넘기면 **모든 요약이 감점**되어 지표가 무의미해집니다
(기획안_2 §9.8 / 재사용 규칙 R4). 필드 목록을 여기서 다시 타이핑하지 않고
`schema.PaperSummary.verifiable_dict()` 에 위임합니다.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import requests

from src.summarize.schema import PaperSummary, UnsupportedClaim, Verdict

#: 확정 프롬프트 (기획안_2 §6.1 — 재설계 금지). CWD 가 아니라 이 모듈 기준으로
#: 해석합니다 — 프롬프트는 데이터가 아니라 코드의 일부입니다 (generator.py 와 같은 규약).
_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent / "summarize" / "prompts" / "verify_faithfulness.md"
)

#: docs/03_검증자_선정.md 의 결정. **기본 인자로 직접 쓰지 않습니다** — 정의 시점에
#: 바인딩되면 테스트의 monkeypatch 가 무효가 됩니다 (기획안_2 §9.1 / R7).
#: `model_id=None` 으로 받아 호출 시점에 해석합니다.
DEFAULT_VERIFIER_MODEL = "openai/gpt-5.6-luna"
FALLBACK_VERIFIER_MODEL = "deepseek/deepseek-v4-flash"

#: 치환 **전** 템플릿에서 존재를 검사하는 자리표시자.
REQUIRED_PLACEHOLDERS: tuple[str, ...] = ("{abstract}", "{summary_json}")

#: SYSTEM/USER 분리 마커. generator.py 와 같은 규칙입니다 (템플릿의 `## USER` 헤딩 줄).
#: 마커 뒤의 부록 절("채점 대상에서 제외되는 필드 ★")도 USER 메시지에 함께 들어갑니다 —
#: 그 절은 검증자에게 주는 지시이므로 의도된 동작입니다.
_USER_MARKER_RE = re.compile(r"^## USER[ \t]*$", re.MULTILINE)

_API_KEY_ENV = "OPENROUTER_API_KEY"
_DEFAULT_API_URL = "https://openrouter.ai/api/v1/chat/completions"

#: transport(url, headers, payload, timeout_sec) -> (status_code, 응답 JSON).
#: generator.py·arxiv.py 의 훅과 같은 형태 — 네트워크·키 없이 테스트하기 위한 주입 지점.
Transport = Callable[[str, dict[str, str], dict[str, Any], float], tuple[int, dict[str, Any]]]


class VerifierError(RuntimeError):
    """검증자 실패. 조용히 넘어가지 않습니다 (작업규약 §8-3)."""


class MissingAPIKeyError(VerifierError):
    """API 키 부재. 키가 없어 죽는 게 가짜로 성공하는 것보다 낫습니다 (기획안_2 §4.2)."""


class VerdictParseError(VerifierError):
    """응답을 `Verdict` 로 파싱하지 못했습니다.

    파싱 실패 = 그날 검증 불능입니다 (docs/03_검증자_선정.md 선정 근거 2). 여기서
    내부 재시도를 하지 않는 이유는 재시도 횟수·기록이 오케스트레이터의 책임이기
    때문입니다 (기획안_2 §6.3, `src/summarize/orchestrator.py`).
    """


def load_verify_template(path: Path | None = None) -> str:
    """검증 프롬프트를 읽고 **치환 전** 무결성을 검사합니다.

    자리표시자 존재 검사를 여기서만 하는 이유: 치환 후 텍스트에는 초록의 LaTeX
    중괄호가 들어 있어 미치환 검사가 오탐을 냅니다 (기획안_2 §9.15).

    `path` 기본값이 모듈 상수가 아니라 `None` 인 이유는 기획안_2 §9.1 / R7 입니다.
    """
    resolved = path if path is not None else _PROMPT_PATH
    template = resolved.read_text(encoding="utf-8")
    missing = [ph for ph in REQUIRED_PLACEHOLDERS if ph not in template]
    if missing:
        raise VerifierError(f"검증 프롬프트에 자리표시자가 없습니다: {missing} ({resolved})")
    if not _USER_MARKER_RE.search(template):
        raise VerifierError(f"검증 프롬프트에 '## USER' 마커가 없습니다 ({resolved})")
    return template


def summary_json(summary: PaperSummary) -> str:
    """검증자에게 넘길 요약 JSON. ★ `verifiable_dict()` 만입니다.

    `to_dict()` 를 쓰면 `connection` 이 들어가고, 초록에 근거가 있을 수 없는 필드라
    **모든 요약이 감점**됩니다 (기획안_2 §9.8 / R4). 필드 목록을 여기서 다시
    타이핑하지 않는 것도 같은 이유입니다.
    """
    return json.dumps(summary.verifiable_dict(), ensure_ascii=False, indent=2)


def render_verify_prompt(
    template: str, *, abstract: str, summary_json_text: str
) -> tuple[str, str]:
    """템플릿을 (system, user) 로 나누고 자리표시자를 채웁니다.

    - 분리를 치환보다 **먼저** 합니다 — 치환값에 우연히 '## USER' 가 들어 있어도
      분리가 흔들리지 않습니다.
    - 치환은 자리표시자당 `str.replace` 1회 (기획안_2 §9.15). `{abstract}` 를
      **마지막에** 치환합니다 — 초록이 LaTeX 중괄호를 가장 많이 들고 있어, 뒤이은
      replace 호출이 그 텍스트를 다시 훑는 일이 없게 합니다.
    - 치환 후 검사는 하지 않습니다. 검사는 `load_verify_template` 에서 끝났습니다.
    """
    marker = _USER_MARKER_RE.search(template)
    if marker is None:
        raise VerifierError("검증 프롬프트에 '## USER' 마커가 없습니다")
    system_part = template[: marker.start()].strip()
    user_part = template[marker.end() :].strip()

    def _substitute(part: str) -> str:
        part = part.replace("{summary_json}", summary_json_text)
        return part.replace("{abstract}", abstract)  # ★ 마지막 — 위 docstring 참조

    return _substitute(system_part), _substitute(user_part)


def _strip_code_fences(text: str) -> str:
    """앞뒤 코드펜스(```json ... ```)만 벗깁니다. 본문 중괄호는 건드리지 않습니다."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[^\n]*\n?", "", stripped)
        stripped = re.sub(r"\n?```\s*$", "", stripped)
    return stripped.strip()


def _json_object_slice(text: str) -> str:
    """설명 문장을 앞뒤로 붙여 보낸 응답에서 JSON 객체만 잘라냅니다.

    프롬프트가 "설명을 붙이지 마세요"라고 지시하지만 모델은 종종 붙입니다. 첫 `{` 부터
    마지막 `}` 까지를 자릅니다 — 왜곡 인용문 안의 LaTeX 중괄호는 JSON 문자열 **내부**라
    이 절단에 영향을 주지 않습니다.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return text
    return text[start : end + 1]


def parse_verdict(text: str, *, verifier_model: str) -> Verdict:
    """모델 응답 텍스트 → `Verdict` (R4 — 스키마는 schema.py 재사용).

    `UnsupportedClaim` 이 왜곡 유형을 검증하므로(`DISTORTION_TYPES`) 프롬프트가 정한
    5개 유형 밖의 값이 오면 여기서 시끄럽게 죽습니다. 임의로 다른 유형에 끼워 맞추면
    원장의 왜곡 유형 분포가 조용히 거짓이 됩니다.
    """
    cleaned = _strip_code_fences(text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            data = json.loads(_json_object_slice(cleaned))
        except json.JSONDecodeError as exc:
            raise VerdictParseError(
                f"검증자 응답이 JSON 이 아닙니다 ({verifier_model}): {exc}. "
                f"응답 앞부분: {cleaned[:200]!r}"
            ) from exc
    if not isinstance(data, dict):
        raise VerdictParseError(
            f"검증자 응답 JSON 이 객체가 아닙니다 ({verifier_model}): {type(data).__name__}"
        )
    if "faithfulness" not in data:
        raise VerdictParseError(
            f"검증자 응답에 faithfulness 가 없습니다 ({verifier_model}): {sorted(data)}"
        )
    try:
        faithfulness = float(data["faithfulness"])
    except (TypeError, ValueError) as exc:
        raise VerdictParseError(
            f"faithfulness 가 수치가 아닙니다 ({verifier_model}): {data['faithfulness']!r}"
        ) from exc

    raw_claims = data.get("unsupported_claims") or []
    if not isinstance(raw_claims, list):
        raise VerdictParseError(
            f"unsupported_claims 가 배열이 아닙니다 ({verifier_model}): {type(raw_claims).__name__}"
        )
    claims: list[UnsupportedClaim] = []
    for raw in raw_claims:
        if not isinstance(raw, dict):
            raise VerdictParseError(
                f"unsupported_claims 항목이 객체가 아닙니다 ({verifier_model}): {raw!r}"
            )
        try:
            claims.append(
                UnsupportedClaim(
                    claim=str(raw.get("claim", "")),
                    field=str(raw.get("field", "")),
                    type=str(raw.get("type", "")),
                    why=str(raw.get("why", "")),
                )
            )
        except ValueError as exc:
            raise VerdictParseError(
                f"unsupported_claims 항목이 스키마에 맞지 않습니다 ({verifier_model}): {exc}"
            ) from exc

    try:
        return Verdict(
            faithfulness=faithfulness,
            unsupported_claims=claims,
            verifier_model=verifier_model,
        )
    except ValueError as exc:  # 0.0~1.0 범위 위반 — 스키마가 잡습니다 (R4)
        raise VerdictParseError(f"검증자가 범위 밖 점수를 냈습니다 ({verifier_model}): {exc}") from exc


@dataclass(frozen=True)
class VerificationUsage:
    """1회 검증의 토큰 사용량. 원장 `cost_usd` 실측 누적용 (기획안_2 §12).

    필드명은 `generator.GenerationUsage` 와 같습니다 — 파이프라인이 생성·검증 비용을
    같은 방식으로 더할 수 있어야 합니다.
    """

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


def _message_text(body: dict[str, Any]) -> str:
    choices = body.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # 일부 프로바이더는 블록 배열로 돌려줍니다
        return "".join(
            str(part.get("text", "")) for part in content if isinstance(part, dict)
        )
    return ""


class OpenRouterVerifier:
    """OpenRouter chat completions 를 `requests` 로 직접 호출하는 검증자.

    - 키는 환경변수 `OPENROUTER_API_KEY` 에서만 읽습니다. 없으면 생성자가
      `MissingAPIKeyError` 로 죽습니다 — 조용한 폴백 금지 (기획안_2 §4.2).
      키 없이 돌려야 하면 `RuleBasedFakeVerifier` 를 쓰세요 (델타 §D6.2).
    - SDK 를 쓰지 않는 이유: Actions 설치 시간 + 기획안_2 §9.10 (운영 requirements 는
      표준 라이브러리 + PyYAML·requests 뿐).
    - `model_id` 기본값은 docs/03_검증자_선정.md 의 결정(`openai/gpt-5.6-luna`)이지만
      **호출 시점에** 해석합니다 (기획안_2 §9.1 / R7). 폴백은
      `FALLBACK_VERIFIER_MODEL`.
    - 단가(`usd_per_1m_*`)는 인자입니다. 하드코딩 금지 — 기획안_2 §12.
    - `assert_verifier_differs` 는 여기서 부르지 않습니다. 게이트는 **요청 직전**에
      호출해야 하고(기획안_2 §6.2), 그 지점을 아는 건 오케스트레이터입니다
      (`src/summarize/orchestrator.py`).
    """

    def __init__(
        self,
        model_id: str | None = None,
        *,
        max_tokens: int = 1024,
        timeout_sec: float = 60.0,
        temperature: float | None = None,
        request_json_mode: bool = True,
        usd_per_1m_input_tokens: float | None = None,
        usd_per_1m_output_tokens: float | None = None,
        prompt_path: Path | None = None,
        api_url: str | None = None,
        transport: Transport | None = None,
    ) -> None:
        api_key = os.environ.get(_API_KEY_ENV, "").strip()
        if not api_key:
            raise MissingAPIKeyError(
                f"환경변수 {_API_KEY_ENV} 가 없습니다. 조용히 폴백하지 않습니다 — "
                "키가 없어 죽는 게 가짜로 성공하는 것보다 낫습니다 (기획안_2 §4.2). "
                "키 없이 돌리려면 RuleBasedFakeVerifier (델타 §D6.2)"
            )
        # 호출 시점 해석 (기획안_2 §9.1 / R7) — 모듈 상수를 기본 인자로 쓰지 않습니다.
        resolved_model = DEFAULT_VERIFIER_MODEL if model_id is None else model_id.strip()
        if not resolved_model:
            raise ValueError(
                "model_id 가 비어 있습니다. 생략하면 docs/03_검증자_선정.md 의 기본값을 씁니다"
            )
        self.model_id = resolved_model
        self._api_key = api_key  # 로그·예외 메시지에 절대 넣지 않습니다 (CLAUDE.md §3-2)
        self._max_tokens = max_tokens
        self._timeout_sec = timeout_sec
        self._temperature = temperature
        self._request_json_mode = request_json_mode
        self._usd_per_1m_input = usd_per_1m_input_tokens
        self._usd_per_1m_output = usd_per_1m_output_tokens
        self._template = load_verify_template(prompt_path)
        self._api_url = api_url if api_url is not None else _DEFAULT_API_URL
        self._transport: Transport = transport if transport is not None else _requests_transport
        #: 마지막 verify() 의 토큰 사용량. 파싱 실패여도 기록됩니다 — 실패한 호출도
        #: 비용이 들었고, 원장 cost_usd 는 실측이어야 합니다 (기획안_2 §12).
        self.last_usage: VerificationUsage | None = None

    def _cost_usd(self, input_tokens: int, output_tokens: int) -> float | None:
        if self._usd_per_1m_input is None or self._usd_per_1m_output is None:
            return None
        return (
            input_tokens * self._usd_per_1m_input / 1_000_000
            + output_tokens * self._usd_per_1m_output / 1_000_000
        )

    def verify(self, abstract: str, summary: PaperSummary) -> Verdict:
        """초록과 요약을 대조해 `Verdict` 를 냅니다.

        `abstract` 가 유일한 근거입니다. 비어 있으면 대조할 원문이 없으므로 판정
        자체가 성립하지 않습니다 — jobs 채널에 faithfulness 를 적용하지 않는 것과
        같은 이유입니다 (CLAUDE.md §3-8).
        """
        if not abstract or not abstract.strip():
            raise VerifierError(
                f"초록이 비어 있어 대조할 원문이 없습니다 ({summary.item_id}). "
                "대조 원문 없는 faithfulness 는 성립하지 않습니다 (CLAUDE.md §3-8)"
            )

        system_text, user_text = render_verify_prompt(
            self._template,
            abstract=abstract,
            summary_json_text=summary_json(summary),  # ★ verifiable_dict() 만 (§9.8/R4)
        )
        payload: dict[str, Any] = {
            "model": self.model_id,
            "max_tokens": self._max_tokens,
            "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ],
        }
        if self._request_json_mode:
            # docs/03_검증자_선정.md 선정 근거 2 — strict JSON. 프로바이더가 거부하면
            # (HTTP 400) `request_json_mode=False` 로 끄세요. 프롬프트가 이미 JSON
            # 하나만 요구하고 parse_verdict 가 코드펜스를 벗깁니다.
            payload["response_format"] = {"type": "json_object"}
        if self._temperature is not None:
            # 일부 추론 모델은 temperature 지정 자체를 거부합니다. 지정했을 때만 넣습니다.
            payload["temperature"] = self._temperature

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
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
            raise VerifierError(
                f"OpenRouter 오류 (HTTP {status}, model={self.model_id}, "
                f"item={summary.item_id}): {detail}"
            )

        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        input_tokens = int(usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("completion_tokens") or 0)
        # 파싱보다 먼저 기록 — 파싱이 실패해도 이 호출의 비용은 실제로 발생했습니다.
        self.last_usage = VerificationUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=self._cost_usd(input_tokens, output_tokens),
        )

        text = _message_text(body)
        if not text.strip():
            choices = body.get("choices") or [{}]
            finish = choices[0].get("finish_reason") if isinstance(choices[0], dict) else None
            raise VerdictParseError(
                f"검증자 응답이 비어 있습니다 ({summary.item_id}, model={self.model_id}). "
                f"finish_reason={finish!r}"
            )
        return parse_verdict(text, verifier_model=self.model_id)


#: `faithfulness.py` 주석과 `tests/test_verify.py` 가 부르는 이름. 구현체는 하나입니다.
LLMVerifier = OpenRouterVerifier
