"""텔레그램 인라인 버튼 콜백 수집 (기획안_2 §8.1 M4 · §8.3 M6).

왜 webhook 이 아니라 `getUpdates` 인가
--------------------------------------
GitHub Actions 는 **단발 실행**이라 webhook 을 받을 서버가 없습니다. 텔레그램은 확인되지
않은 업데이트를 24시간 보관하므로, 다음날 실행이 `getUpdates` 로 지난 24시간치를 한 번에
걷어 갑니다 (기획안_2 §8.1 / v1.0 §5.1(2)). 피드백은 랭킹 개선용이라 하루 지연이 문제되지
않습니다.

★ forG 봇을 재사용하면 이 모듈은 영원히 동작하지 않습니다 (v1.0 §13).
webhook 이 걸린 봇에 `getUpdates` 를 부르면 텔레그램이 **409 Conflict** 를 돌려줍니다.
`WebhookConflictError` 가 그 사실을 그대로 말하도록 만들어 두었습니다 — 이 실패는
"네트워크가 이상한가?" 로 하루를 태우기 딱 좋은 형태로 옵니다.

중복 수집을 막는 것은 두 겹입니다 ★
-----------------------------------
1. **서버 측 확인(confirm)** — `getUpdates(offset=N+1)` 을 부르는 순간 텔레그램이
   `update_id ≤ N` 을 큐에서 지웁니다. 이게 **주 방어선**입니다.
2. **로컬 offset 파일** (`data/cache/telegram_offset`) — 같은 머신에서 다시 돌릴 때의
   보조 방어선입니다.

`feedback.jsonl` 을 스캔해 offset 을 복원하지 않습니다. 파일이 커질수록 비싸지고,
§8.1 의 피드백 스키마는 `{ts, item_id, action, channel}` **4필드 고정**이라 `update_id`
자체가 파일에 없습니다.

★ 그런데 Actions 러너는 매 실행 새 파일시스템이고 `data/cache/` 는 gitignore 입니다.
  즉 러너에서 2번은 **항상 비어 있습니다.** 그래서 수집 직후 `confirm=True` 로
  `getUpdates(offset=next)` 를 한 번 더 불러 서버 측에서 확실히 지웁니다. 이걸 빼면,
  피드백을 파일에 쓴 뒤 커밋 전에 실행이 죽었을 때 다음날 같은 콜백이 한 번 더
  `feedback.jsonl` 에 들어가고, M6 의 가중치 갱신이 그 항목만 2배로 셉니다.
  (확인 호출이 새 업데이트를 지우지는 않습니다 — 확인 대상은 `offset` **미만**입니다.)

유실은 감지해서 원장에 남깁니다
------------------------------
실행이 하루 실패하면 그날 피드백은 24시간 보관을 넘겨 사라집니다. 막을 방법이 없으므로
**유실을 기록하고 넘어갑니다** (기획안_2 §8.1). `detect_gap()` 이 그 판정만 합니다 —
원장 쓰기는 하지 않습니다. 원장은 `ledger.RunRecord` 로만 씁니다 (R3). 호출부가
`CollectResult.to_params()` 를 `RunRecord.params["feedback"]` 에 넣으세요.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from src.publish.telegram import (
    ACTIONS,
    TelegramError,
    Transport,
    api_url,
    redact_token,
    requests_transport,
    resolve_bot_token,
    safe_api_url,
)

#: `src/core/pipeline.py` 의 KST 와 같은 값입니다. 상수 하나 때문에 수집 파이프라인
#: 전체(arxiv·state·sqlite)를 import 하지는 않습니다.
KST = timezone(timedelta(hours=9))

#: offset 상태 파일. `data/cache/` 는 이미 `.gitignore` 에 있습니다 — 상태 파일을
#: 새로 만들면서 `.gitignore` 를 고칠 필요가 없도록 여기에 둡니다.
DEFAULT_OFFSET_PATH = Path("data/cache/telegram_offset")

#: 피드백 원장. **공개 커밋 대상입니다** (기획안_2 §4.3). item_id 와 read/skip/save 만
#: 들어가고 공고 본문·제목은 들어가지 않습니다 — 그래서 jobs 콜백도 안전합니다
#: (CLAUDE.md §3-3: 채용공고 공개 게시 금지).
DEFAULT_FEEDBACK_PATH = Path("data/feedback.jsonl")

#: 기획안_2 §8.1 — 텔레그램은 확인되지 않은 업데이트를 24시간 보관합니다.
RETENTION_HOURS = 24.0

#: `feedback.jsonl` 한 줄의 키. **4개 고정** (기획안_2 §8.1). 늘리고 싶으면 문서를
#: 먼저 고치세요 — M6 가 이 스키마를 읽습니다.
FEEDBACK_FIELDS: tuple[str, ...] = ("ts", "item_id", "action", "channel")

#: item_id 접두사(= 소스) → 채널. `callback_data` 는 64바이트라 채널을 실을 자리가
#: 없어서 (`{action}:{item_id}` 고정, §8.1) item_id 에서 되짚습니다.
#: ★ 새 소스 어댑터를 만들면 여기 등록하세요. 등록하지 않으면 그 소스의 피드백은
#:   `errors` 로 빠지고 파일에 들어가지 않습니다 — 조용히 "papers" 로 기본값을 주면
#:   M6 가 엉뚱한 프로파일의 가중치를 올립니다.
SOURCE_CHANNELS: dict[str, str] = {
    "arxiv": "papers",
    "hf_papers": "papers",
    "saramin": "jobs",
    "worknet": "jobs",
    "fake": "papers",
}


class FeedbackError(TelegramError):
    """피드백 수집 실패."""


class WebhookConflictError(FeedbackError):
    """`getUpdates` 409 — 이 봇에 webhook 이 걸려 있습니다 (v1.0 §13)."""


class FeedbackStateError(FeedbackError):
    """offset 상태 파일을 읽을 수 없습니다."""


class FeedbackParseError(FeedbackError):
    """업데이트 1건을 피드백으로 해석하지 못했습니다.

    이 예외는 **배치 전체를 멈추지 않습니다.** `collect()` 가 잡아서 `errors` 에
    모으고 offset 은 그대로 전진시킵니다. 전진시키지 않으면 해석 불가능한 업데이트
    하나가 큐 맨 앞에 남아 다음 실행부터 영원히 같은 지점에서 막힙니다.
    """


@dataclass(frozen=True, slots=True)
class FeedbackEvent:
    """`feedback.jsonl` 한 줄 (기획안_2 §8.1)."""

    ts: str
    item_id: str
    action: str
    channel: str
    #: 파일에는 **쓰지 않습니다** (스키마 4필드 고정). 중복 판정과 offset 계산에만 씁니다.
    update_id: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts": self.ts,
            "item_id": self.item_id,
            "action": self.action,
            "channel": self.channel,
        }


@dataclass(frozen=True, slots=True)
class OffsetState:
    """`data/cache/telegram_offset` 의 내용."""

    next_offset: int | None = None
    collected_at: str | None = None


@dataclass(frozen=True, slots=True)
class FeedbackGap:
    """직전 수집 이후 벌어진 간격. 원장 `params` 에 그대로 들어갑니다."""

    last_collected_at: str | None
    gap_hours: float | None
    lost: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "last_collected_at": self.last_collected_at,
            "gap_hours": self.gap_hours,
            "lost": self.lost,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class CollectResult:
    events: list[FeedbackEvent]
    next_offset: int | None
    updates_received: int
    duplicates: int
    ignored: int
    gap: FeedbackGap
    confirmed: bool
    feedback_path: Path
    errors: list[str] = field(default_factory=list)

    def to_params(self) -> dict[str, Any]:
        """원장용 요약. 호출부가 `RunRecord.params["feedback"]` 에 넣습니다 (R3).

        이 모듈은 원장을 직접 쓰지 않습니다 — `ledger.RunRecord` 로만 씁니다.
        """
        return {
            "updates_received": self.updates_received,
            "events": len(self.events),
            "duplicates": self.duplicates,
            "ignored": self.ignored,
            "next_offset": self.next_offset,
            "confirmed": self.confirmed,
            "errors": list(self.errors),
            "gap": self.gap.to_dict(),
        }


# ── 파싱 ─────────────────────────────────────────────────────────────────


def parse_callback_data(data: str) -> tuple[str, str]:
    """`"{action}:{item_id}"` → `(action, item_id)` (기획안_2 §8.1).

    ★ `data.split(":")` 로 자르면 안 됩니다. item_id 자체에 콜론이 있습니다
      (`arxiv:2608.11053`, `saramin:49123456`). **최대 1회만** 자릅니다.
      구형 arXiv ID(`arxiv:math.GT/0309136`, §9.5)도 그대로 살아남습니다.
    """
    action, separator, item_id = data.partition(":")
    if not separator or not item_id:
        raise FeedbackParseError(f"callback_data 형식이 아닙니다 ('{{action}}:{{item_id}}'): {data!r}")
    if action not in ACTIONS:
        raise FeedbackParseError(f"알 수 없는 액션: {action!r} (허용: {list(ACTIONS)}) — data={data!r}")
    return action, item_id


def channel_for_item_id(item_id: str) -> str:
    """item_id 접두사로 채널을 되짚습니다 (`SOURCE_CHANNELS` 주석 참조)."""
    source = item_id.split(":", 1)[0]
    channel = SOURCE_CHANNELS.get(source)
    if channel is None:
        raise FeedbackParseError(
            f"소스 {source!r} 의 채널을 모릅니다 (item_id={item_id!r}). "
            f"collector.SOURCE_CHANNELS 에 등록하세요 (현재: {sorted(SOURCE_CHANNELS)})"
        )
    return channel


def parse_update(update: Mapping[str, Any], *, ts: str) -> FeedbackEvent | None:
    """업데이트 1건 → 피드백 1건. 콜백이 아니면 `None` 입니다.

    콜백이 아닌 업데이트(일반 메시지 등)가 섞이는 것은 정상입니다. `allowed_updates`
    는 호출 **이전에** 만들어진 업데이트에는 적용되지 않는다고 Bot API 문서가
    명시합니다. 그러니 무시 경로는 예외가 아니라 평시 동작입니다.
    """
    update_id = update.get("update_id")
    if not isinstance(update_id, int):
        raise FeedbackParseError(f"update_id 가 없습니다: {str(update)[:200]}")

    callback = update.get("callback_query")
    if not isinstance(callback, dict):
        return None

    data = callback.get("data")
    if not isinstance(data, str) or not data:
        raise FeedbackParseError(f"callback_query 에 data 가 없습니다 (update_id={update_id})")

    action, item_id = parse_callback_data(data)
    return FeedbackEvent(
        # 텔레그램 콜백에는 **클릭 시각이 없습니다.** `callback_query.message.date` 는
        # 봇이 메시지를 보낸 시각이지 사람이 누른 시각이 아닙니다. 그래서 ts 는
        # **수집 시각**이고, 최대 24시간 늦습니다 (기획안_2 §8.1). M6 가 시간 순서로
        # 뭔가를 판단하려 하면 이 한계를 먼저 보세요.
        ts=ts,
        item_id=item_id,
        action=action,
        channel=channel_for_item_id(item_id),
        update_id=update_id,
    )


def parse_updates(
    updates: Iterable[Mapping[str, Any]],
    *,
    ts: str,
    next_offset: int | None = None,
) -> tuple[list[FeedbackEvent], int | None, dict[str, int], list[str]]:
    """업데이트 목록 → `(events, next_offset, stats, errors)`.

    중복 제거는 `update_id` 로 합니다 (기획안_2 §8.1):
    - 이미 수집한 구간(`update_id < next_offset`)은 건너뜁니다
    - 같은 응답 안에서 같은 `update_id` 가 두 번 오면 첫 건만 받습니다

    반환하는 `next_offset` 은 **받은 모든 업데이트의 최댓값 + 1** 입니다.
    무시한 메시지·해석 실패도 포함합니다 — 빼면 그 업데이트가 큐에 남아 다음
    실행에서 다시 옵니다. 마지막 원소의 update_id 로 계산하는 지름길도 안 됩니다:
    텔레그램이 순서를 보장하지만 우리 픽스처처럼 재전송분이 뒤에 붙을 수 있습니다.
    """
    events: list[FeedbackEvent] = []
    errors: list[str] = []
    seen: set[int] = set()
    stats = {"received": 0, "duplicates": 0, "ignored": 0}
    highest: int | None = None

    for update in updates:
        stats["received"] += 1
        update_id = update.get("update_id")
        if isinstance(update_id, int):
            if highest is None or update_id > highest:
                highest = update_id
            if next_offset is not None and update_id < next_offset:
                stats["duplicates"] += 1
                continue
            if update_id in seen:
                stats["duplicates"] += 1
                continue
            seen.add(update_id)

        try:
            event = parse_update(update, ts=ts)
        except FeedbackParseError as exc:
            errors.append(str(exc))
            continue
        if event is None:
            stats["ignored"] += 1
            continue
        events.append(event)

    resolved = next_offset if highest is None else highest + 1
    if next_offset is not None and resolved is not None:
        resolved = max(resolved, next_offset)
    return events, resolved, stats, errors


# ── offset 상태 파일 ─────────────────────────────────────────────────────


def read_offset(path: Path | None = None) -> OffsetState:
    """offset 상태를 읽습니다. 파일이 없으면 빈 상태입니다 (첫 실행 / Actions 러너).

    `path` 기본값이 모듈 상수가 아니라 `None` 인 이유: 기본 인자는 정의 시점에
    바인딩되어 테스트의 monkeypatch 가 무효가 됩니다 (기획안_2 §9.1 / R7).
    """
    resolved = DEFAULT_OFFSET_PATH if path is None else Path(path)
    if not resolved.exists():
        return OffsetState()
    try:
        data = json.loads(resolved.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise FeedbackStateError(
            f"offset 상태 파일을 읽을 수 없습니다 ({resolved}): {exc}. "
            "조용히 0 으로 되돌리면 지난 24시간치가 통째로 중복 수집됩니다. "
            "내용을 확인하고 지운 뒤 다시 실행하세요"
        ) from exc
    if not isinstance(data, dict):
        raise FeedbackStateError(f"offset 상태 파일이 객체가 아닙니다 ({resolved}): {type(data).__name__}")
    next_offset = data.get("next_offset")
    if next_offset is not None and not isinstance(next_offset, int):
        raise FeedbackStateError(f"next_offset 이 정수가 아닙니다 ({resolved}): {next_offset!r}")
    collected_at = data.get("collected_at")
    return OffsetState(next_offset=next_offset, collected_at=collected_at)


def write_offset(state: OffsetState, path: Path | None = None) -> Path:
    resolved = DEFAULT_OFFSET_PATH if path is None else Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(
            {"next_offset": state.next_offset, "collected_at": state.collected_at},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return resolved


# ── 유실 감지 ────────────────────────────────────────────────────────────


def detect_gap(
    last_collected_at: str | None,
    now: datetime,
    *,
    retention_hours: float | None = None,
) -> FeedbackGap:
    """직전 수집 이후 24시간이 지났으면 그 사이 피드백은 사라졌습니다 (기획안_2 §8.1).

    막을 방법이 없으므로 **판정만 하고 원장에 남깁니다.** 조용히 넘어가면 나중에
    "왜 이 날만 피드백이 0건이지" 를 실행 로그로 역추적하게 됩니다.

    `last_collected_at` 을 호출부가 넘길 수 있게 열어 둔 이유 ★: Actions 러너의
    `data/cache/` 는 매 실행 비어 있어 offset 파일의 시각을 믿을 수 없습니다.
    러너에서는 커밋되는 원장(`data/runs.jsonl`)의 직전 실행 시각을 넘기세요.
    """
    limit = RETENTION_HOURS if retention_hours is None else retention_hours
    if not last_collected_at:
        return FeedbackGap(
            last_collected_at=None,
            gap_hours=None,
            lost=False,
            reason="직전 수집 기록이 없습니다 (첫 실행이거나 상태 파일이 없는 러너)",
        )
    try:
        previous = datetime.fromisoformat(last_collected_at)
    except ValueError as exc:
        raise FeedbackStateError(
            f"last_collected_at 이 ISO 시각이 아닙니다: {last_collected_at!r} ({exc})"
        ) from exc
    if previous.tzinfo is None:
        raise FeedbackStateError(
            f"last_collected_at 에 시간대가 없습니다: {last_collected_at!r}. "
            "KST 오프셋(+09:00)을 포함해 기록하세요"
        )
    gap_hours = round((now - previous).total_seconds() / 3600.0, 3)
    if gap_hours > limit:
        return FeedbackGap(
            last_collected_at=last_collected_at,
            gap_hours=gap_hours,
            lost=True,
            reason=(
                f"직전 수집 이후 {gap_hours}시간 — 텔레그램 보관 한도 {limit}시간을 "
                "넘겨 그 사이 콜백은 유실되었습니다 (기획안_2 §8.1)"
            ),
        )
    return FeedbackGap(
        last_collected_at=last_collected_at,
        gap_hours=gap_hours,
        lost=False,
        reason=f"직전 수집 이후 {gap_hours}시간 — 보관 한도 {limit}시간 이내",
    )


# ── 파일 쓰기 ────────────────────────────────────────────────────────────


def append_feedback(events: Sequence[FeedbackEvent], path: Path | None = None) -> Path:
    """`feedback.jsonl` 에 append. 기존 줄은 건드리지 않습니다 (원장과 같은 규칙)."""
    resolved = DEFAULT_FEEDBACK_PATH if path is None else Path(path)
    if not events:
        return resolved  # 빈 파일을 새로 만들지 않습니다
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("a", encoding="utf-8") as fp:
        for event in events:
            fp.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
    return resolved


# ── 수집 ─────────────────────────────────────────────────────────────────


def _check_response(status: int, body: Any, token: str, api_base: str | None) -> list[Any]:
    """`getUpdates` 응답 검사. 409 는 별도 예외입니다."""
    error_code = body.get("error_code") if isinstance(body, dict) else None
    description = str(body.get("description") or "")[:300] if isinstance(body, dict) else ""

    if status == 409 or error_code == 409:
        raise WebhookConflictError(
            "getUpdates 가 409 Conflict 입니다 — 이 봇에 webhook 이 걸려 있습니다. "
            "webhook 과 getUpdates 는 동시에 쓸 수 없습니다. "
            "★ forG 봇 재사용 금지 (v1.0 §13): forG 봇에는 webhook 이 설정돼 있어 "
            "이 파이프라인의 피드백 수집이 영구히 동작하지 않습니다. Radar 전용 봇을 "
            "새로 만들고 TELEGRAM_BOT_TOKEN 을 교체하세요 (기획안_2 §11.2-6). "
            f"텔레그램 응답: {redact_token(description, token)}"
        )
    if status != 200 or not (isinstance(body, dict) and body.get("ok")):
        raise FeedbackError(
            f"getUpdates 실패 (HTTP {status}, error_code={error_code}, "
            f"{safe_api_url('getUpdates', api_base)}): {redact_token(description, token)}"
        )
    result = body.get("result")
    if not isinstance(result, list):
        raise FeedbackError(f"getUpdates 응답의 result 가 배열이 아닙니다: {type(result).__name__}")
    return result


def collect(
    *,
    transport: Transport | None = None,
    offset_path: Path | None = None,
    feedback_path: Path | None = None,
    api_base: str | None = None,
    timeout_sec: float = 30.0,
    limit: int = 100,
    clock: Callable[[], datetime] | None = None,
    last_collected_at: str | None = None,
    confirm: bool = True,
) -> CollectResult:
    """지난 24시간의 콜백을 걷어 `feedback.jsonl` 에 append 합니다 (기획안_2 §8.1).

    경로·시계·transport 기본값은 전부 **호출 시점에** 해석합니다 (기획안_2 §9.1 / R7).
    `transport=FakeTransport()` 면 키·네트워크 없이 전 구간이 돕니다 (델타 §D6.2).

    순서가 계약입니다: **응답 검사 → 파싱 → 파일 append → offset 저장 → 서버 확인.**
    파일에 쓰기 전에 offset 을 저장하면, 그 사이에 죽었을 때 피드백이 영구 유실됩니다.
    (반대 순서의 위험은 중복이고, 중복은 offset 으로 다시 막을 수 있습니다 —
    기획안_2 §9.6 과 같은 판단입니다.)
    """
    now = datetime.now(KST) if clock is None else clock()
    ts = now.isoformat(timespec="seconds")

    token = resolve_bot_token()
    send: Transport = requests_transport if transport is None else transport
    url = api_url(token, "getUpdates", api_base)

    state = read_offset(offset_path)
    previous_collected_at = (
        last_collected_at if last_collected_at is not None else state.collected_at
    )
    gap = detect_gap(previous_collected_at, now)

    payload: dict[str, Any] = {
        "limit": limit,
        "timeout": 0,  # 롱폴링 금지. Actions 는 단발 실행이고 대기는 요금입니다
        # 문서상 이 필터는 호출 **이전에** 생성된 업데이트에는 적용되지 않습니다.
        # 그래서 파서에 무시 경로가 여전히 필요합니다 (parse_update 참조).
        "allowed_updates": ["callback_query"],
    }
    if state.next_offset is not None:
        payload["offset"] = state.next_offset

    # 연결 자체가 실패하는 경로도 토큰을 흘립니다 — _check_response 는 서버가
    # 응답을 준 경우만 다룹니다. requests 예외의 URL 을 그대로 전파하면
    # Actions 로그에 토큰이 찍힙니다 (CLAUDE.md §3-2).
    try:
        status, body = send(url, payload, timeout_sec)
    except Exception as exc:  # noqa: BLE001 — 전파 전에 반드시 지웁니다
        raise FeedbackError(
            f"getUpdates 요청 실패 ({safe_api_url('getUpdates', api_base)}): "
            f"{type(exc).__name__}: {redact_token(str(exc), token)}"
        ) from None
    updates = _check_response(status, body, token, api_base)

    events, next_offset, stats, errors = parse_updates(
        updates, ts=ts, next_offset=state.next_offset
    )

    written_path = append_feedback(events, feedback_path)
    write_offset(OffsetState(next_offset=next_offset, collected_at=ts), offset_path)

    confirmed = False
    if confirm and next_offset is not None:
        # 서버 측 확인 (모듈 docstring ★). 확인 대상은 `offset` **미만**이라 이 호출이
        # 새 업데이트를 지우지는 않습니다. 실패해도 수집 자체는 이미 끝났으므로
        # 예외로 올리지 않고 결과에 남깁니다 — 호출부가 원장에 기록합니다.
        try:
            confirm_status, confirm_body = send(
                url, {"offset": next_offset, "limit": 1, "timeout": 0}, timeout_sec
            )
            confirmed = confirm_status == 200 and bool(
                isinstance(confirm_body, dict) and confirm_body.get("ok")
            )
            if not confirmed:
                errors.append(f"offset 서버 확인 실패 (HTTP {confirm_status}) — 다음 실행에서 재시도")
        except Exception as exc:  # noqa: BLE001 — 확인 실패로 수집 결과를 버리지 않습니다
            # ★ redact 필수. requests 예외 메시지는 실패한 URL 을 통째로 담고,
            #   그 URL 경로에 봇 토큰이 들어 있습니다(/bot<TOKEN>/getUpdates).
            #   이 문자열은 CollectResult.errors → to_params() → RunRecord.params →
            #   data/runs.jsonl 로 흐르고, runs.jsonl 은 **git 추적 대상이며
            #   저장소는 public** 입니다. 지우지 마세요 (CLAUDE.md §3-2).
            errors.append(
                f"offset 서버 확인 중 예외: {type(exc).__name__}: "
                f"{redact_token(str(exc), token)}"
            )

    return CollectResult(
        events=events,
        next_offset=next_offset,
        updates_received=stats["received"],
        duplicates=stats["duplicates"],
        ignored=stats["ignored"],
        gap=gap,
        confirmed=confirmed,
        feedback_path=written_path,
        errors=errors,
    )
