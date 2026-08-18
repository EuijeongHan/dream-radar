"""텔레그램 푸시 싱크 (기획안_2 §8.1 M4, v1.0 §5.1(2)).

무엇을 하는가
------------
    render_message(item, summary)   아이템 1건 → 텔레그램 메시지(본문 + 인라인 버튼 3개)
    send_messages(messages, ...)    Bot API `sendMessage` 로 실제 전송
    FakeTransport                   키·네트워크 없이 전송 경로를 도는 대체재 (델타 §D6.2)

★ scope 게이트를 부르지 않습니다 — 의도된 부재입니다
---------------------------------------------------
`gates.assert_public_scope()` 는 **공개 싱크**(Hugo 사이트)가 비공개 아이템을 흘리는 것을
막는 장치입니다 (R5 / 사람인 약관 조항 6 — 채용공고 공개 게시 금지, 무료여도 위반).

텔레그램은 본인 `chat_id` 하나로만 보내는 **비공개 채널**이라 그 게이트의 전제가
성립하지 않습니다. 기획안_2 §4.1 표와 §8.2 가 jobs 채널의 발행 경로를 "텔레그램 +
비공개만" 으로 확정했습니다 — 즉 `publish_scope="private"` 아이템이 여기로 나가는 것이
**설계상 정상**입니다. 여기에 `assert_public_scope` 를 넣으면 채용 채널 전체가 발행
불가가 됩니다. `src/publish/markdown.py`(공개 사이트)에는 반드시 있어야 하고, 여기에는
반드시 없어야 합니다. 두 파일의 차이는 실수가 아닙니다.

parse_mode 를 쓰지 않습니다 ★
-----------------------------
논문 제목·요약에는 LaTeX 와 마크다운 특수문자(`_`, `*`, `[`, `` ` ``)가 그대로 들어
있습니다 (기획안_2 §9.15 — 초록의 10%에 중괄호). `parse_mode=Markdown|HTML` 을 켜면
그 10%가 텔레그램에서 `400 Bad Request: can't parse entities` 로 죽습니다. §9.15 와 같은
계열의 함정이라 아예 켜지 않습니다. 링크는 맨 URL 이라 텔레그램이 알아서 걸어 줍니다.

원문 링크는 하드 게이트입니다
----------------------------
- papers: `https://arxiv.org/abs/...` 가 없으면 **예외**. arXiv 이용약관 권고
  "Direct users to arXiv.org to retrieve e-print content" (기획안_2 §3.1 / 델타 §D1).
  PDF·전문 링크는 대체물이 될 수 없습니다 (CLAUDE.md §3-4).
- jobs: 원문 링크가 없으면 **예외**. "판단은 사람이 링크를 열어서 합니다" (기획안_2 §8.2).

키 취급
-------
`TELEGRAM_BOT_TOKEN` 은 **URL 경로에 들어갑니다** (`/bot<token>/sendMessage`). 그래서
예외 메시지에 URL 을 그대로 넣으면 그 순간 토큰이 Actions 로그에 남습니다
(CLAUDE.md §3-2). 이 모듈은 URL 을 예외에 넣지 않고, 서버 응답 문자열도
`redact_token()` 을 거쳐 내보냅니다.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import requests

from src.summarize.schema import ALL_FIELDS, PaperSummary

#: 인라인 버튼 3개 (기획안_2 §8.1). 순서가 곧 화면 순서입니다.
ACTIONS: tuple[str, ...] = ("read", "skip", "save")

#: 버튼에 찍히는 한국어 라벨. 키는 `ACTIONS` 그대로 — 여기 없는 액션이 생기면
#: KeyError 로 시끄럽게 죽는 것이 의도입니다.
ACTION_LABELS: dict[str, str] = {"read": "읽음", "skip": "스킵", "save": "저장"}

#: Bot API 제한. `callback_data` 는 1~64 **바이트**입니다 (문자 수가 아닙니다).
#: 넘기면 텔레그램이 `BUTTON_DATA_INVALID` 로 거절하는데, 그 실패는 전송 시점에
#: 나므로 여기서 미리 막습니다.
CALLBACK_DATA_MAX_BYTES = 64

#: `sendMessage` 본문 길이 제한. 넘기면 `400 Bad Request: message is too long`.
#: 텔레그램은 UTF-16 코드 단위로 세지만, 한글·영문은 1이라 `len()` 과 같습니다.
#: 이모지(2)를 본문에 넣지 않는 이유이기도 합니다 — 채널 표시는 `[논문]`·`[채용]` 입니다.
MESSAGE_MAX_CHARS = 4096

#: 잘렸다는 사실을 사람이 알아야 합니다. 잘린 줄 모르고 "요약이 이상하다"로 오해하면
#: 원인을 찾는 데 하루가 갑니다.
TRUNCATION_MARK = "\n…(길어서 줄임 — 아래 링크에서 원문 확인)"

BOT_TOKEN_ENV = "TELEGRAM_BOT_TOKEN"
CHAT_ID_ENV = "TELEGRAM_CHAT_ID"
DEFAULT_API_BASE = "https://api.telegram.org"

#: transport(url, payload, timeout_sec) -> (status_code, 응답 JSON).
#: `summarize/generator.py`·`sources/arxiv.py` 의 주입 훅과 같은 목적입니다 —
#: 키도 네트워크도 없이 전송 경로를 테스트하기 위한 지점 (델타 §D6.2).
Transport = Callable[[str, dict[str, Any], float], tuple[int, dict[str, Any]]]

#: 채널 표시. `render_message` 가 모르는 채널을 받으면 KeyError 가 아니라 명시적
#: ValueError 를 내도록 아래에서 존재 검사를 먼저 합니다.
CHANNEL_PREFIX: dict[str, str] = {"papers": "[논문] ", "jobs": "[채용] "}

#: 논문 요약 필드 라벨. 키는 `schema.ALL_FIELDS` 그대로입니다 (R4 — 필드명을 다시
#: 타이핑하지 않습니다). 스키마에 필드가 추가되면 여기서 KeyError 로 죽습니다.
#: `markdown.py` 의 절 제목과 값이 겹치지만 형태가 다릅니다(`## 문제` vs `문제:`) —
#: 텔레그램은 마크다운을 끈 평문이라 헤딩 문법을 쓸 수 없습니다.
PAPER_FIELD_LABELS: dict[str, str] = {
    "problem": "문제",
    "method": "방법",
    "key_results": "핵심결과",
    "limitations": "한계",
    "connection": "연결점",
}

#: jobs 채널의 요약 필드 (기획안_2 §8.2 — LLM 역할이 "내용 요약"이 아니라
#: **"적합도 판단 + 추천 사유"** 입니다. 사람인 API 가 직무내용 본문을 주지 않아
#: 요약할 원문 자체가 없습니다).
#:
#: ★ M5 에서 jobs 요약 스키마가 `src/summarize/schema.py` 에 확정되면 이 두 상수를
#:   거기서 import 하도록 바꾸세요 (R4). 지금은 그 스키마가 없어서 여기 둡니다.
JOB_FIELDS: tuple[str, ...] = ("fit_score", "fit_reason", "cautions")
JOB_REQUIRED_FIELDS: tuple[str, ...] = ("fit_reason",)
JOB_FIELD_LABELS: dict[str, str] = {
    "fit_score": "적합도",
    "fit_reason": "추천 사유",
    "cautions": "주의",
}

#: arXiv abs 링크. `/abs/` 뒤는 `math.GT/0309136` 처럼 슬래시를 포함할 수 있습니다
#: (기획안_2 §9.5 — 구형 ID 의 아카이브 접두사).
_ARXIV_ABS_RE = re.compile(r"^https?://(?:www\.)?arxiv\.org/abs/\S+$", re.IGNORECASE)


class TelegramError(RuntimeError):
    """텔레그램 싱크 실패. 조용히 넘어가지 않습니다 (작업규약 §8-3)."""


class MissingTelegramCredentialsError(TelegramError):
    """봇 토큰·chat_id 부재. 키가 없어 죽는 게 가짜로 성공하는 것보다 낫습니다 (기획안_2 §4.2)."""


class TelegramSendError(TelegramError):
    """`sendMessage` 가 실패했습니다.

    `sent` 에 **그때까지 실제로 나간 건수**가 들어 있습니다. 5건 중 3건이 나간 뒤
    실패한 것과 0건 나간 것은 다릅니다 — 호출부가 원장에 그대로 남길 수 있게
    예외에 붙여 보냅니다 (기획안_2 §4.4).
    """

    def __init__(self, message: str, *, sent: Sequence["SendResult"] = ()) -> None:
        super().__init__(message)
        self.sent: list[SendResult] = list(sent)


@dataclass(frozen=True, slots=True)
class TelegramMessage:
    """전송 직전 형태. 렌더링과 전송을 나눠 두면 키 없이 렌더링만 테스트할 수 있습니다."""

    item_id: str
    text: str
    reply_markup: dict[str, Any] = field(default_factory=dict)

    def to_payload(self, chat_id: str) -> dict[str, Any]:
        """`sendMessage` 요청 본문.

        - `parse_mode` 를 **넣지 않습니다** (모듈 docstring 참조).
        - `reply_markup` 은 중첩 객체 그대로 둡니다. 폼 전송이면 JSON 문자열이어야
          하지만 이 모듈은 `application/json` 으로 보냅니다.
        - 미리보기를 끕니다. 하루 5건이 전부 큰 카드로 펼쳐지면 버튼이 화면 밖으로
          밀립니다.
        """
        return {
            "chat_id": chat_id,
            "text": self.text,
            "disable_web_page_preview": True,
            "reply_markup": self.reply_markup,
        }


@dataclass(frozen=True, slots=True)
class SendResult:
    item_id: str
    message_id: int | None


# ── 콜백 데이터 · 버튼 ────────────────────────────────────────────────────


def build_callback_data(action: str, item_id: str) -> str:
    """`"{action}:{item_id}"` (기획안_2 §8.1).

    64 **바이트** 제한입니다. 문자 수로 세면 한글 item_id 에서 3배를 놓칩니다.
    제한을 넘으면 여기서 `ValueError` 입니다 — 텔레그램에 보내면 전송 시점에
    `BUTTON_DATA_INVALID` 가 나고, 그때는 이미 다른 메시지가 나간 뒤입니다.
    """
    if action not in ACTIONS:
        raise ValueError(f"알 수 없는 액션: {action!r} (허용: {list(ACTIONS)})")
    if not item_id:
        raise ValueError("item_id 가 비어 있습니다 — 콜백을 아이템에 되돌릴 수 없습니다")
    data = f"{action}:{item_id}"
    size = len(data.encode("utf-8"))
    if size > CALLBACK_DATA_MAX_BYTES:
        raise ValueError(
            f"callback_data 가 {size}바이트로 한도 {CALLBACK_DATA_MAX_BYTES}바이트를 "
            f"넘습니다 (action={action}, item_id={item_id!r}). "
            "텔레그램이 BUTTON_DATA_INVALID 로 거절합니다"
        )
    return data


def build_keyboard(item_id: str) -> dict[str, Any]:
    """인라인 버튼 3개를 한 줄로 (기획안_2 §8.1 — 읽음 / 스킵 / 저장)."""
    return {
        "inline_keyboard": [
            [
                {
                    "text": ACTION_LABELS[action],
                    "callback_data": build_callback_data(action, item_id),
                }
                for action in ACTIONS
            ]
        ]
    }


# ── 렌더링 ───────────────────────────────────────────────────────────────


def _require_arxiv_abs_url(url: str, item_id: str) -> None:
    """papers 는 abs 링크가 없으면 발행하지 않습니다 (기획안_2 §3.1 / 델타 §D1)."""
    if not url:
        raise ValueError(
            f"arXiv abs 링크가 없습니다 ({item_id}). arXiv 이용약관 권고 "
            '"Direct users to arXiv.org" — 링크 없는 아이템은 푸시하지 않습니다 (기획안_2 §3.1)'
        )
    if "/pdf/" in url or url.lower().endswith(".pdf"):
        raise ValueError(
            f"PDF 링크로는 푸시할 수 없습니다 ({item_id}): {url}. "
            "전문은 대부분 재배포 불가 라이선스입니다 (CLAUDE.md §3-4)"
        )
    if not _ARXIV_ABS_RE.match(url):
        raise ValueError(
            f"arXiv abs 링크가 아닙니다 ({item_id}): {url}. "
            "형식: https://arxiv.org/abs/<id> (기획안_2 §3.1)"
        )


def _render_lines(labels: Mapping[str, str], values: Iterable[tuple[str, Any]]) -> str:
    lines: list[str] = []
    for name, value in values:
        if not value:  # 빈 절은 만들지 않습니다 (markdown.py 와 같은 규칙)
            continue
        label = labels[name]
        if isinstance(value, (list, tuple)):
            lines.append(f"{label}:")
            lines.extend(f"- {entry}" for entry in value)
        else:
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


def _render_paper_body(summary: Mapping[str, Any], item_id: str) -> str:
    """요약 필드명·필수값 검증을 스키마에 위임합니다 (R4)."""
    paper = PaperSummary.from_dict(dict(summary), item_id=item_id)
    return _render_lines(
        PAPER_FIELD_LABELS, ((name, getattr(paper, name)) for name in ALL_FIELDS)
    )


def _render_job_body(summary: Mapping[str, Any], item_id: str) -> str:
    """jobs 는 요약이 아니라 **적합도 판단 + 추천 사유** 입니다 (기획안_2 §8.2).

    faithfulness 검증을 붙이지 않습니다 — 사람인 API 가 본문을 주지 않아 대조할
    원문 자체가 없습니다 (CLAUDE.md §3-8). 그래서 판단 근거를 사람이 확인할 수
    있도록 원문 링크가 더 중요해집니다.
    """
    missing = [name for name in JOB_REQUIRED_FIELDS if not summary.get(name)]
    if missing:
        raise ValueError(
            f"jobs 요약에 필수 필드가 비어 있습니다 ({item_id}): {missing} "
            "(기획안_2 §8.2 — 적합도 사유 없이는 푸시할 이유가 없습니다)"
        )
    return _render_lines(JOB_FIELD_LABELS, ((name, summary.get(name)) for name in JOB_FIELDS))


def _compose(header: str, body: str, url: str) -> str:
    """`header` + `body` + 원문 링크. 한도를 넘으면 **본문만** 줄입니다.

    링크는 절대 자르지 않습니다 — 링크가 이 메시지의 존재 이유입니다 (기획안_2 §8.2).
    """
    tail = f"\n\n{url}"
    room = MESSAGE_MAX_CHARS - len(tail)
    if room <= len(TRUNCATION_MARK):
        raise ValueError(
            f"원문 링크만으로 텔레그램 한도({MESSAGE_MAX_CHARS}자)에 닿습니다: {url}"
        )
    head = f"{header}\n\n" if header else ""
    if len(head) + len(body) <= room:
        return f"{head}{body}{tail}"
    budget = room - len(TRUNCATION_MARK)
    if len(head) >= budget:  # 제목만으로 한도를 넘는 병적인 경우
        return f"{head[:budget]}{TRUNCATION_MARK}{tail}"
    return f"{head}{body[: budget - len(head)]}{TRUNCATION_MARK}{tail}"


def render_message(item: Mapping[str, Any], summary: Mapping[str, Any]) -> TelegramMessage:
    """아이템 1건 → 텔레그램 메시지 (기획안_2 §8.1).

    구성: `[논문]/[채용] 제목` → 한국어 요약(또는 적합도 사유) → 원문 링크 → 버튼 3개.

    `item` 은 후보 JSONL 한 줄과 같은 dict 입니다 (`markdown.py` 와 동일한 계약).
    원문 링크가 없거나 채널을 모르면 `ValueError` 입니다 — 모르면 막는 쪽입니다.
    """
    item_id = str(item.get("id") or "")
    if not item_id:
        raise ValueError("item.id 가 비어 있습니다 — 콜백을 아이템에 되돌릴 수 없습니다")

    channel = str(item.get("channel") or "")
    if channel not in CHANNEL_PREFIX:
        raise ValueError(
            f"알 수 없는 channel: {channel!r} ({item_id}). "
            f"허용: {sorted(CHANNEL_PREFIX)} (기획안_2 §4.1)"
        )

    url = str(item.get("url") or "").strip()
    if channel == "papers":
        _require_arxiv_abs_url(url, item_id)
        body = _render_paper_body(summary, item_id)
    else:
        if not url:
            raise ValueError(
                f"원문 링크가 없습니다 ({item_id}). 기획안_2 §8.2 — "
                "판단은 사람이 링크를 열어서 합니다"
            )
        body = _render_job_body(summary, item_id)

    header = f"{CHANNEL_PREFIX[channel]}{str(item.get('title') or '').strip()}"
    return TelegramMessage(
        item_id=item_id,
        text=_compose(header, body, url),
        # ★ 여기서 scope 게이트를 부르지 않는 이유는 모듈 docstring 참조.
        reply_markup=build_keyboard(item_id),
    )


# ── 전송 ─────────────────────────────────────────────────────────────────


def redact_token(text: str, token: str) -> str:
    """토큰 문자열을 가립니다. 예외·로그로 나가는 모든 문자열이 이걸 거칩니다.

    토큰이 URL 경로에 들어가는 API 라, 응답 본문이 요청 URL 을 되돌려주는 경우가
    실제로 있습니다 (`Bad Request` 계열). CLAUDE.md §3-2.
    """
    if not token:
        return text
    return text.replace(token, "<TELEGRAM_BOT_TOKEN>")


def resolve_bot_token() -> str:
    """`TELEGRAM_BOT_TOKEN`. 없으면 예외입니다 — 조용한 폴백 금지 (기획안_2 §4.2)."""
    token = os.environ.get(BOT_TOKEN_ENV, "").strip()
    if not token:
        raise MissingTelegramCredentialsError(
            f"환경변수 {BOT_TOKEN_ENV} 가 없습니다. 조용히 폴백하지 않습니다 — "
            "키가 없어 죽는 게 가짜로 성공하는 것보다 낫습니다 (기획안_2 §4.2). "
            "테스트·배선 확인은 FakeTransport 를 쓰세요 (델타 §D6.2). "
            "★ forG 봇을 재사용하지 마세요 (v1.0 §13) — webhook 충돌"
        )
    return token


def resolve_chat_id() -> str:
    """`TELEGRAM_CHAT_ID`. 없으면 예외입니다."""
    chat_id = os.environ.get(CHAT_ID_ENV, "").strip()
    if not chat_id:
        raise MissingTelegramCredentialsError(
            f"환경변수 {CHAT_ID_ENV} 가 없습니다. 조용히 폴백하지 않습니다 (기획안_2 §4.2)"
        )
    return chat_id


def api_url(token: str, method: str, api_base: str | None = None) -> str:
    """`{base}/bot{token}/{method}`.

    ★ 이 반환값을 예외 메시지·로그에 넣지 마세요. 토큰이 경로에 들어 있습니다
      (CLAUDE.md §3-2). 사람이 읽을 주소가 필요하면 `safe_api_url()` 을 쓰세요.
    """
    base = DEFAULT_API_BASE if api_base is None else api_base
    return f"{base.rstrip('/')}/bot{token}/{method}"


def safe_api_url(method: str, api_base: str | None = None) -> str:
    """토큰 자리를 가린 주소. 예외 메시지용입니다."""
    base = DEFAULT_API_BASE if api_base is None else api_base
    return f"{base.rstrip('/')}/bot<TELEGRAM_BOT_TOKEN>/{method}"


def requests_transport(
    url: str, payload: dict[str, Any], timeout_sec: float
) -> tuple[int, dict[str, Any]]:
    """실전송. SDK 를 쓰지 않는 이유는 기획안_2 §9.10 (Actions 설치 시간)."""
    response = requests.post(url, json=payload, timeout=timeout_sec)
    try:
        body = response.json()
    except ValueError:
        body = {"ok": False, "description": response.text[:500]}
    return response.status_code, body


class FakeTransport:
    """키·네트워크 없이 Bot API 를 흉내냅니다 (델타 §D6.2).

    `calls` 에 `(url, payload)` 가 순서대로 쌓입니다 — 테스트는 이걸로 "무엇을 보냈나"를
    확인합니다. `status`/`description` 을 바꾸면 실패 경로도 그대로 돌 수 있습니다.

    **전송 성공 여부 판정에 쓰지 마세요.** fake 로 초록이 났다고 실제 봇이 동작하는 건
    아닙니다 (작업규약 §8-9).
    """

    def __init__(
        self,
        *,
        status: int = 200,
        ok: bool = True,
        description: str = "",
        error_code: int | None = None,
        first_message_id: int = 1000,
        result: Any = None,
    ) -> None:
        self.status = status
        self.ok = ok
        self.description = description
        self.error_code = error_code
        self.result = result
        self._next_message_id = first_message_id
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(
        self, url: str, payload: dict[str, Any], timeout_sec: float
    ) -> tuple[int, dict[str, Any]]:
        self.calls.append((url, payload))
        if not self.ok:
            body: dict[str, Any] = {"ok": False, "description": self.description}
            if self.error_code is not None:
                body["error_code"] = self.error_code
            return self.status, body
        if self.result is not None:
            return self.status, {"ok": True, "result": self.result}
        message_id = self._next_message_id
        self._next_message_id += 1
        return self.status, {
            "ok": True,
            "result": {"message_id": message_id, "chat": {"id": payload.get("chat_id")}},
        }


def send_messages(
    messages: Iterable[TelegramMessage],
    *,
    transport: Transport | None = None,
    api_base: str | None = None,
    timeout_sec: float = 30.0,
) -> list[SendResult]:
    """메시지를 순서대로 보냅니다 (기획안_2 §8.1).

    - 토큰·chat_id 는 **환경변수에서만** 읽습니다. 인자로 받지 않는 이유: 인자를 열면
      호출부가 어딘가에서 리터럴을 넘기게 되고, 그게 로그·원장에 남습니다
      (CLAUDE.md §3-2). 키 없이 돌려야 하면 `transport=FakeTransport()` 입니다.
    - `transport`·`api_base` 기본값은 **호출 시점에** 해석합니다 (기획안_2 §9.1 / R7).
    - 한 건이라도 실패하면 `TelegramSendError` 이고, 예외에 그때까지 나간 건수가
      들어 있습니다. 하루 5건이라 초당 30건 제한(Bot API)에 걸리지 않으므로
      호출 사이에 대기를 넣지 않습니다.
    """
    queued = list(messages)
    token = resolve_bot_token()
    chat_id = resolve_chat_id()
    url = api_url(token, "sendMessage", api_base)
    send: Transport = requests_transport if transport is None else transport

    sent: list[SendResult] = []
    for message in queued:
        # 서버가 응답을 준 경우(HTTP 오류)는 아래에서 다루지만, **연결 자체가
        # 실패하는 경로**는 requests 예외가 URL 을 통째로 담아 전파합니다.
        # URL 경로에 봇 토큰이 있습니다 — Actions 로그로 새면 §3-2 위반입니다.
        try:
            status, body = send(url, message.to_payload(chat_id), timeout_sec)
        except TelegramError:
            raise
        except Exception as exc:  # noqa: BLE001 — 전파 전에 반드시 지웁니다
            raise TelegramSendError(
                f"sendMessage 전송 실패 (item={message.item_id}, "
                f"{len(sent)}/{len(queued)}건 전송 후, "
                f"{safe_api_url('sendMessage', api_base)}): "
                f"{redact_token(f'{type(exc).__name__}: {exc}', token)}",
                sent=sent,
            ) from None
        if status != 200 or not (isinstance(body, dict) and body.get("ok")):
            detail = ""
            if isinstance(body, dict):
                detail = str(body.get("description") or body)[:300]
            raise TelegramSendError(
                # URL 을 넣지 않습니다 — 토큰이 경로에 있습니다 (CLAUDE.md §3-2).
                f"sendMessage 실패 (HTTP {status}, item={message.item_id}, "
                f"{len(sent)}/{len(queued)}건 전송 후, {safe_api_url('sendMessage', api_base)}): "
                f"{redact_token(detail, token)}",
                sent=sent,
            )
        result = body.get("result") if isinstance(body.get("result"), dict) else {}
        sent.append(SendResult(item_id=message.item_id, message_id=result.get("message_id")))
    return sent
