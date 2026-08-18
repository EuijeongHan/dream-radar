"""텔레그램 푸시 싱크 + 피드백 수집 게이트 (기획안_2 §8.1 M4).

★ 다음 셋은 비활성화하지 마세요.
  - `test_paper_message_rejects_*` : arXiv 이용약관 권고 "Direct users to arXiv.org"
    (기획안_2 §3.1). 링크 없는 푸시는 만들지 않습니다.
  - `test_private_scope_is_not_blocked` : jobs 채널이 텔레그램으로 나가는 것은
    **설계상 정상**입니다 (기획안_2 §8.2 "발행: 텔레그램 + 비공개만"). 여기에 공개
    싱크용 scope 게이트를 넣으면 채용 채널 전체가 막힙니다.
  - `test_send_error_does_not_leak_token` : 봇 토큰이 URL 경로에 들어가는 API 라
    예외 메시지 하나로 Actions 로그에 키가 남습니다 (CLAUDE.md §3-2).

키 없이 전부 돌아야 합니다. 전송은 `telegram.FakeTransport`, 수집은 아래
`ScriptedTransport` 로 주입합니다 (델타 §D6.2).
논문 아이템·요약은 `tests/fixtures/distorted_summaries.yaml` 의 faithful 케이스를
재사용합니다 (새 픽스처를 만들지 않습니다).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.feedback import collector as C
from src.publish import telegram as T

KST = timezone(timedelta(hours=9))

FIXTURES = Path(__file__).parent / "fixtures"
UPDATES_FIXTURE = FIXTURES / "telegram_updates.json"
SUMMARY_FIXTURE = FIXTURES / "distorted_summaries.yaml"


# ── 공통 픽스처 ──────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def updates() -> dict[str, Any]:
    return json.loads(UPDATES_FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def samples() -> dict[str, Any]:
    return yaml.safe_load(SUMMARY_FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture
def paper_item(samples) -> dict[str, Any]:
    source = samples["source"]
    return {
        "id": source["item_id"],
        "source": "arxiv",
        "channel": "papers",
        "title": source["title"],
        "abstract": source["abstract"],
        "url": source["url"],
        "published": "2026-08-12T04:11:02+00:00",
        "publish_scope": "public",
        "categories": ["cs.CV"],
    }


@pytest.fixture
def paper_summary(samples) -> dict[str, Any]:
    return dict(samples["faithful"]["summary"])


@pytest.fixture
def job_item() -> dict[str, Any]:
    """★ 가상 공고입니다. 실제 공고 본문은 테스트에도 넣지 않습니다 (CLAUDE.md §3-3)."""
    return {
        "id": "saramin:49123456",
        "source": "saramin",
        "channel": "jobs",
        "title": "(픽스처) AI 엔지니어",
        "url": "https://example.invalid/jobs/49123456",
        "published": "2026-08-17T09:00:00+09:00",
        "publish_scope": "private",  # jobs 는 리터럴 고정 (기획안_2 §8.2)
        "categories": [],
    }


@pytest.fixture
def job_summary() -> dict[str, Any]:
    return {
        "fit_score": "높음",
        "fit_reason": "RAG·평가 파이프라인 경험을 직접 요구합니다.",
        "cautions": ["파견 여부 확인 필요"],
    }


@pytest.fixture
def bot_env(monkeypatch) -> None:
    monkeypatch.setenv(T.BOT_TOKEN_ENV, "111222333:FIXTURE-TOKEN-NOT-REAL")
    monkeypatch.setenv(T.CHAT_ID_ENV, "500000001")


class ScriptedTransport:
    """`(status, body)` 를 순서대로 돌려주는 transport. 마지막 응답을 계속 반복합니다.

    `collect()` 는 수집 1회 + 서버 확인 1회로 **두 번** 호출하므로, 두 호출을 구분해
    검사하려면 응답을 두 개 넘기세요.
    """

    def __init__(self, *responses: tuple[int, dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(
        self, url: str, payload: dict[str, Any], timeout_sec: float
    ) -> tuple[int, dict[str, Any]]:
        self.calls.append((url, payload))
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return self.responses[index]


def _ok(updates_fixture: dict[str, Any], case: str) -> tuple[int, dict[str, Any]]:
    return 200, updates_fixture[case]


# ─────────────────────────────────────────────────────────────────────────
# 1. 렌더링 — 원문 링크 하드 게이트
# ─────────────────────────────────────────────────────────────────────────


def test_paper_message_carries_arxiv_abs_link(paper_item, paper_summary):
    """★ 논문 푸시에는 abs 링크가 반드시 실립니다 (기획안_2 §3.1 / 델타 §D1).

    깨뜨리는 법: telegram._compose 의 `tail` 을 빈 문자열로 바꾸면 빨간불.
    확인일: 2026-08-18
    """
    message = T.render_message(paper_item, paper_summary)
    assert message.text.rstrip().endswith("https://arxiv.org/abs/2608.11053")
    assert "문제:" in message.text and "핵심결과:" in message.text
    assert message.item_id == "arxiv:2608.11053"
    # 전문·PDF 경로는 어디에도 없어야 합니다 (CLAUDE.md §3-4).
    assert "/pdf/" not in message.text


@pytest.mark.parametrize("bad_url", ["", None])
def test_paper_message_rejects_missing_url(paper_item, paper_summary, bad_url):
    """★ 링크 없는 논문은 푸시하지 않습니다.

    깨뜨리는 법: telegram.render_message 의 `_require_arxiv_abs_url(...)` 호출을 지우면 빨간불.
    확인일: 2026-08-18
    """
    paper_item["url"] = bad_url
    with pytest.raises(ValueError, match="링크"):
        T.render_message(paper_item, paper_summary)


@pytest.mark.parametrize(
    "pdf_url",
    ["https://arxiv.org/pdf/2608.11053", "https://arxiv.org/pdf/2608.11053v1.pdf"],
)
def test_paper_message_rejects_pdf_link(paper_item, paper_summary, pdf_url):
    """PDF 는 abs 의 대체물이 아닙니다 — 전문은 대부분 재배포 불가 (CLAUDE.md §3-4).

    깨뜨리는 법: _require_arxiv_abs_url 의 "/pdf/" 검사를 지우면 메시지가 "abs 링크가
    아닙니다" 로 바뀌어 match="PDF" 가 빨간불.
    확인일: 2026-08-18
    """
    paper_item["url"] = pdf_url
    with pytest.raises(ValueError, match="PDF"):
        T.render_message(paper_item, paper_summary)


def test_paper_message_rejects_non_arxiv_link(paper_item, paper_summary):
    """미러·요약 사이트 링크로 대체할 수 없습니다 (기획안_2 §3.1).

    깨뜨리는 법: _require_arxiv_abs_url 의 정규식 검사를 지우면 빨간불.
    확인일: 2026-08-18
    """
    paper_item["url"] = "https://huggingface.co/papers/2608.11053"
    with pytest.raises(ValueError, match="abs"):
        T.render_message(paper_item, paper_summary)


def test_unknown_channel_rejected(paper_item, paper_summary):
    """모르는 채널은 막습니다 — 어느 링크 규칙을 적용할지 알 수 없습니다.

    깨뜨리는 법: render_message 의 channel 검사를 지우고 else 로 흘리면 빨간불.
    확인일: 2026-08-18
    """
    paper_item["channel"] = "newsletter"
    with pytest.raises(ValueError, match="channel"):
        T.render_message(paper_item, paper_summary)


def test_summary_missing_core_fields_rejected(paper_item, paper_summary):
    """요약 검증은 `schema.PaperSummary.from_dict` 에 위임합니다 (R4 — 재구현 금지).

    깨뜨리는 법: _render_paper_body 에서 PaperSummary.from_dict 대신 summary.get 을
    직접 쓰면 빈 method 가 그대로 푸시되어 빨간불.
    확인일: 2026-08-18
    """
    paper_summary["method"] = ""
    with pytest.raises(ValueError, match="필수 필드"):
        T.render_message(paper_item, paper_summary)


# ─────────────────────────────────────────────────────────────────────────
# 2. 인라인 버튼 · callback_data 64바이트
# ─────────────────────────────────────────────────────────────────────────


def test_keyboard_has_three_actions(paper_item, paper_summary):
    """인라인 버튼 3개 read/skip/save, `{action}:{item_id}` (기획안_2 §8.1).

    깨뜨리는 법: telegram.ACTIONS 에서 "save" 를 빼면 빨간불.
    확인일: 2026-08-18
    """
    row = T.render_message(paper_item, paper_summary).reply_markup["inline_keyboard"][0]
    assert [button["callback_data"] for button in row] == [
        "read:arxiv:2608.11053",
        "skip:arxiv:2608.11053",
        "save:arxiv:2608.11053",
    ]
    assert [button["text"] for button in row] == ["읽음", "스킵", "저장"]


def test_callback_data_keeps_colon_in_item_id():
    """item_id 자체에 콜론이 있습니다. 왕복(생성→파싱)이 같은 값을 돌려줘야 합니다.

    깨뜨리는 법: collector.parse_callback_data 의 partition 을 `data.split(":")` 로
    바꾸면 ValueError(unpack) 또는 item_id 절단으로 빨간불.
    확인일: 2026-08-18
    """
    data = T.build_callback_data("read", "arxiv:math.GT/0309136")
    assert data == "read:arxiv:math.GT/0309136"
    assert C.parse_callback_data(data) == ("read", "arxiv:math.GT/0309136")


def test_callback_data_rejects_over_64_bytes():
    """★ 64바이트를 넘으면 텔레그램이 BUTTON_DATA_INVALID 로 거절합니다.
    전송 시점이 아니라 **렌더링 시점에** 막습니다.

    깨뜨리는 법: build_callback_data 의 길이 검사를 지우면 빨간불.
    확인일: 2026-08-18
    """
    long_id = "arxiv:" + "0" * 60
    with pytest.raises(ValueError, match="64"):
        T.build_callback_data("read", long_id)


def test_callback_data_counts_bytes_not_characters():
    """★ 한글 item_id 는 문자 수의 3배입니다. 문자로 세면 조용히 통과합니다.

    깨뜨리는 법: build_callback_data 의 `len(data.encode("utf-8"))` 를 `len(data)` 로
    바꾸면 이 테스트가 빨간불 (25자 = 68바이트가 통과해버림).
    확인일: 2026-08-18
    """
    item_id = "saramin:" + "가" * 20  # 8 + 20자 → 8 + 60바이트, 액션까지 더해 68바이트
    data = f"read:{item_id}"
    assert len(data) <= T.CALLBACK_DATA_MAX_BYTES < len(data.encode("utf-8"))
    with pytest.raises(ValueError, match="바이트"):
        T.build_callback_data("read", item_id)


def test_callback_data_rejects_unknown_action():
    """깨뜨리는 법: build_callback_data 의 action 검사를 지우면 빨간불.
    확인일: 2026-08-18
    """
    with pytest.raises(ValueError, match="액션"):
        T.build_callback_data("archive", "arxiv:2608.11053")


# ─────────────────────────────────────────────────────────────────────────
# 3. 길이 제한 · parse_mode
# ─────────────────────────────────────────────────────────────────────────


def test_long_summary_is_truncated_but_link_survives(paper_item, paper_summary):
    """★ 4096자를 넘으면 본문만 줄입니다. 링크는 절대 자르지 않습니다 (기획안_2 §8.2).

    깨뜨리는 법: _compose 에서 길이 분기(`if len(head) + len(body) <= room`)를 지우고
    항상 그대로 반환하게 하면 빨간불.
    확인일: 2026-08-18
    """
    paper_summary["problem"] = "가" * 6000
    message = T.render_message(paper_item, paper_summary)
    assert len(message.text) <= T.MESSAGE_MAX_CHARS
    assert message.text.rstrip().endswith(paper_item["url"])
    assert T.TRUNCATION_MARK.strip() in message.text


def test_payload_has_no_parse_mode(paper_item, paper_summary):
    """★ 마크다운을 켜면 제목의 `_`·`*` 에서 400 Bad Request 가 납니다 (§9.15 계열).

    깨뜨리는 법: TelegramMessage.to_payload 에 "parse_mode": "Markdown" 을 추가하면 빨간불.
    확인일: 2026-08-18
    """
    paper_item["title"] = "GAN_v2 *tricks* [ablation](x) `code`"
    payload = T.render_message(paper_item, paper_summary).to_payload("500000001")
    assert "parse_mode" not in payload
    assert "GAN_v2 *tricks* [ablation](x) `code`" in payload["text"]  # 그대로 실려야 합니다
    assert payload["chat_id"] == "500000001"
    assert payload["reply_markup"]["inline_keyboard"][0][0]["callback_data"].startswith("read:")


# ─────────────────────────────────────────────────────────────────────────
# 4. jobs 채널 — 요약이 아니라 적합도 사유, 그리고 scope 게이트 부재 ★
# ─────────────────────────────────────────────────────────────────────────


def test_jobs_message_uses_fit_reason(job_item, job_summary):
    """jobs 는 내용 요약이 아니라 적합도 판단 + 추천 사유입니다 (기획안_2 §8.2).

    깨뜨리는 법: render_message 의 jobs 분기를 _render_paper_body 로 바꾸면
    "요약에 필수 필드가 비어 있습니다"(problem/method)로 빨간불.
    확인일: 2026-08-18
    """
    message = T.render_message(job_item, job_summary)
    assert "추천 사유: RAG" in message.text
    assert "적합도: 높음" in message.text
    assert "- 파견 여부 확인 필요" in message.text
    assert message.text.rstrip().endswith(job_item["url"])
    # papers 전용 절 제목이 섞이면 안 됩니다.
    assert "문제:" not in message.text


def test_jobs_message_requires_link(job_item, job_summary):
    """"판단은 사람이 링크를 열어서 합니다" (기획안_2 §8.2).

    깨뜨리는 법: render_message 의 jobs 분기에서 `if not url` 검사를 지우면 빨간불.
    확인일: 2026-08-18
    """
    job_item["url"] = ""
    with pytest.raises(ValueError, match="링크"):
        T.render_message(job_item, job_summary)


def test_jobs_missing_fit_reason_rejected(job_item, job_summary):
    """사유 없는 추천은 푸시할 이유가 없습니다.

    깨뜨리는 법: _render_job_body 의 필수 필드 검사를 지우면 빨간불.
    확인일: 2026-08-18
    """
    job_summary["fit_reason"] = ""
    with pytest.raises(ValueError, match="필수 필드"):
        T.render_message(job_item, job_summary)


def test_private_scope_is_not_blocked(bot_env, job_item, job_summary):
    """★ `publish_scope="private"` 아이템이 텔레그램으로 나가는 것은 정상입니다.

    텔레그램은 본인 chat_id 하나로만 가는 비공개 채널이라 공개 싱크용
    `gates.assert_public_scope()` 의 전제가 성립하지 않습니다. 기획안_2 §8.2 가
    jobs 발행 경로를 "텔레그램 + 비공개만" 으로 확정했습니다.

    깨뜨리는 법: telegram.render_message(또는 send_messages)에
    `gates.assert_public_scope([...])` 를 추가하면 ScopeViolation 으로 빨간불 —
    그게 바로 넣으면 안 되는 이유입니다.
    확인일: 2026-08-18
    """
    transport = T.FakeTransport()
    sent = T.send_messages([T.render_message(job_item, job_summary)], transport=transport)
    assert [result.item_id for result in sent] == ["saramin:49123456"]
    assert len(transport.calls) == 1


# ─────────────────────────────────────────────────────────────────────────
# 5. 전송 — 키 부재, 토큰 누출, 부분 전송
# ─────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("missing", [T.BOT_TOKEN_ENV, T.CHAT_ID_ENV])
def test_send_requires_env_credentials(bot_env, monkeypatch, paper_item, paper_summary, missing):
    """★ 키가 없으면 조용히 폴백하지 않고 죽습니다 (기획안_2 §4.2).

    깨뜨리는 법: resolve_bot_token / resolve_chat_id 가 빈 문자열을 그대로 돌려주게
    하면 빨간불 (예외 대신 FakeTransport 로 전송이 성공해버림).
    확인일: 2026-08-18
    """
    monkeypatch.delenv(missing, raising=False)
    message = T.render_message(paper_item, paper_summary)
    with pytest.raises(T.MissingTelegramCredentialsError, match=missing):
        T.send_messages([message], transport=T.FakeTransport())


def test_send_posts_to_bot_url_with_keyboard(bot_env, paper_item, paper_summary):
    """전송 URL·본문·버튼이 그대로 실려야 합니다.

    깨뜨리는 법: TelegramMessage.to_payload 에서 "reply_markup" 을 빼면 빨간불.
    확인일: 2026-08-18
    """
    transport = T.FakeTransport()
    message = T.render_message(paper_item, paper_summary)
    sent = T.send_messages([message], transport=transport)

    url, payload = transport.calls[0]
    assert url == "https://api.telegram.org/bot111222333:FIXTURE-TOKEN-NOT-REAL/sendMessage"
    assert payload["text"] == message.text
    assert payload["reply_markup"] == message.reply_markup
    assert payload["disable_web_page_preview"] is True
    assert sent[0].message_id == 1000


def test_send_error_does_not_leak_token(bot_env, paper_item, paper_summary):
    """★ 토큰이 URL 경로에 들어가는 API 입니다. 예외 메시지가 그걸 되풀이하면
    Actions 로그에 키가 남습니다 (CLAUDE.md §3-2).

    깨뜨리는 법: send_messages 의 예외 메시지에 `safe_api_url(...)` 대신 `url` 을
    넣거나 `redact_token(...)` 을 벗기면 빨간불.
    확인일: 2026-08-18
    """
    # 서버가 요청 URL 을 되돌려주는 응답을 흉내냅니다 — 실제로 있는 형태입니다.
    transport = T.FakeTransport(
        status=400,
        ok=False,
        description=(
            "Bad Request at https://api.telegram.org"
            "/bot111222333:FIXTURE-TOKEN-NOT-REAL/sendMessage"
        ),
    )
    with pytest.raises(T.TelegramSendError) as excinfo:
        T.send_messages([T.render_message(paper_item, paper_summary)], transport=transport)
    text = str(excinfo.value)
    assert "111222333:FIXTURE-TOKEN-NOT-REAL" not in text
    assert "<TELEGRAM_BOT_TOKEN>" in text


def test_send_error_reports_partial_progress(
    bot_env, paper_item, paper_summary, job_item, job_summary
):
    """5건 중 2건이 나간 뒤 실패한 것과 0건은 다릅니다. 예외가 그 수를 들고 옵니다.

    깨뜨리는 법: TelegramSendError 생성에서 `sent=sent` 를 빼면 빨간불.
    확인일: 2026-08-18
    """

    class FailSecond:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, url, payload, timeout_sec):
            self.calls += 1
            if self.calls == 1:
                return 200, {"ok": True, "result": {"message_id": 7}}
            return 429, {"ok": False, "description": "Too Many Requests: retry after 30"}

    messages = [
        T.render_message(paper_item, paper_summary),
        T.render_message(job_item, job_summary),
    ]
    with pytest.raises(T.TelegramSendError) as excinfo:
        T.send_messages(messages, transport=FailSecond())
    assert [result.item_id for result in excinfo.value.sent] == ["arxiv:2608.11053"]
    assert "1/2건" in str(excinfo.value)


# ─────────────────────────────────────────────────────────────────────────
# 6. 피드백 수집 — 파싱 · 중복 제거 · offset
# ─────────────────────────────────────────────────────────────────────────


def test_collect_parses_two_callbacks_from_fixture(bot_env, tmp_path, updates):
    """콜백 2건만 피드백이 됩니다 (무관 메시지·중복은 제외).

    깨뜨리는 법: channel_for_item_id 가 항상 "papers" 를 돌려주게 하면 jobs 콜백의
    채널이 어긋나 빨간불. parse_update 의 `return None`(콜백 아님)을 예외로 바꿔도
    errors 가 늘어 빨간불.
    확인일: 2026-08-18
    """
    result = C.collect(
        transport=ScriptedTransport(_ok(updates, "ok_with_callbacks")),
        offset_path=tmp_path / "telegram_offset",
        feedback_path=tmp_path / "feedback.jsonl",
        clock=lambda: datetime(2026, 8, 18, 7, 0, tzinfo=KST),
    )
    assert [(event.action, event.item_id, event.channel) for event in result.events] == [
        ("read", "arxiv:2608.11053", "papers"),
        ("save", "saramin:49123456", "jobs"),
    ]
    assert result.updates_received == 4
    assert result.ignored == 1
    assert result.errors == []


def test_feedback_line_schema_is_four_keys(bot_env, tmp_path, updates):
    """★ `feedback.jsonl` 한 줄은 {ts, item_id, action, channel} 4필드입니다 (기획안_2 §8.1).

    update_id 를 여기에 넣고 싶어지지만 넣지 않습니다 — M6 가 이 스키마를 읽습니다.
    중복 방지는 offset 파일과 서버 확인이 담당합니다.

    깨뜨리는 법: FeedbackEvent.to_dict 에 "update_id" 를 추가하면 빨간불.
    확인일: 2026-08-18
    """
    feedback_path = tmp_path / "feedback.jsonl"
    C.collect(
        transport=ScriptedTransport(_ok(updates, "ok_with_callbacks")),
        offset_path=tmp_path / "telegram_offset",
        feedback_path=feedback_path,
        clock=lambda: datetime(2026, 8, 18, 7, 0, tzinfo=KST),
    )
    lines = [json.loads(line) for line in feedback_path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    assert list(lines[0]) == list(C.FEEDBACK_FIELDS) == ["ts", "item_id", "action", "channel"]
    assert lines[0]["ts"] == "2026-08-18T07:00:00+09:00"
    assert lines[1]["channel"] == "jobs"


def test_duplicate_update_id_recorded_once(bot_env, tmp_path, updates):
    """★ 같은 응답에 같은 update_id 가 두 번 오면 한 번만 기록합니다 (기획안_2 §8.1).

    깨뜨리는 법: parse_updates 의 `seen` 집합 검사를 지우면 3건이 되어 빨간불.
    확인일: 2026-08-18
    """
    result = C.collect(
        transport=ScriptedTransport(_ok(updates, "ok_with_callbacks")),
        offset_path=tmp_path / "telegram_offset",
        feedback_path=tmp_path / "feedback.jsonl",
        clock=lambda: datetime(2026, 8, 18, 7, 0, tzinfo=KST),
    )
    assert result.duplicates == 1
    assert [event.item_id for event in result.events].count("arxiv:2608.11053") == 1


def test_offset_advances_past_ignored_message(bot_env, tmp_path, updates):
    """★ offset 은 **받은 모든 업데이트의 최댓값 + 1** 입니다.

    무시한 일반 메시지(900002)를 빼면 그 업데이트가 큐에 남아 매 실행 다시 옵니다.
    마지막 원소로 계산해도 안 됩니다 — 픽스처는 재전송분(900001)이 맨 뒤에 붙어 있어
    그 지름길이면 900002 가 나옵니다.

    깨뜨리는 법: parse_updates 의 `highest` 계산을 `updates[-1]["update_id"] + 1` 로
    바꾸면 빨간불.
    확인일: 2026-08-18
    """
    offset_path = tmp_path / "telegram_offset"
    result = C.collect(
        transport=ScriptedTransport(_ok(updates, "ok_with_callbacks")),
        offset_path=offset_path,
        feedback_path=tmp_path / "feedback.jsonl",
        clock=lambda: datetime(2026, 8, 18, 7, 0, tzinfo=KST),
    )
    assert result.next_offset == 900004
    assert C.read_offset(offset_path).next_offset == 900004


def test_second_run_sends_offset_and_skips_old_updates(bot_env, tmp_path, updates):
    """★ 두 번째 실행은 offset 을 보내고, 그래도 옛 업데이트가 오면 건너뜁니다.

    깨뜨리는 법: collect 에서 `payload["offset"] = state.next_offset` 을 지우면
    (또는 parse_updates 의 `update_id < next_offset` 검사를 지우면) 같은 콜백이
    feedback.jsonl 에 두 번 들어가 빨간불.
    확인일: 2026-08-18
    """
    offset_path = tmp_path / "telegram_offset"
    feedback_path = tmp_path / "feedback.jsonl"
    clock = lambda: datetime(2026, 8, 18, 7, 0, tzinfo=KST)  # noqa: E731
    common = dict(offset_path=offset_path, feedback_path=feedback_path, clock=clock)

    C.collect(transport=ScriptedTransport(_ok(updates, "ok_with_callbacks")), **common)
    second_transport = ScriptedTransport(_ok(updates, "ok_with_callbacks"))
    second = C.collect(transport=second_transport, **common)

    assert second_transport.calls[0][1]["offset"] == 900004
    assert second.events == []
    assert second.duplicates == 4
    assert len(feedback_path.read_text(encoding="utf-8").splitlines()) == 2


def test_confirm_call_sends_next_offset(bot_env, tmp_path, updates):
    """★ 수집 직후 `getUpdates(offset=next)` 를 한 번 더 불러 서버 큐에서 지웁니다.

    Actions 러너의 `data/cache/` 는 매 실행 비어 있어 로컬 offset 파일이 다음 실행에
    남지 않습니다. 이 확인 호출이 없으면 커밋 전에 죽었을 때 같은 콜백이 다음날
    한 번 더 들어옵니다.

    깨뜨리는 법: collect 의 `if confirm and next_offset is not None:` 블록을 지우면 빨간불.
    확인일: 2026-08-18
    """
    transport = ScriptedTransport(
        _ok(updates, "ok_with_callbacks"), _ok(updates, "empty")
    )
    result = C.collect(
        transport=transport,
        offset_path=tmp_path / "telegram_offset",
        feedback_path=tmp_path / "feedback.jsonl",
        clock=lambda: datetime(2026, 8, 18, 7, 0, tzinfo=KST),
    )
    assert len(transport.calls) == 2
    assert transport.calls[1][1]["offset"] == 900004
    assert result.confirmed is True


def test_unrelated_message_is_ignored_not_error(bot_env, tmp_path, updates):
    """콜백이 아닌 업데이트는 평시 동작입니다 — `allowed_updates` 는 호출 이전에
    생성된 업데이트에 적용되지 않습니다 (Bot API 문서).

    깨뜨리는 법: parse_update 가 콜백이 아닐 때 None 대신 FeedbackParseError 를
    던지게 하면 errors 가 늘어 빨간불.
    확인일: 2026-08-18
    """
    result = C.collect(
        transport=ScriptedTransport(_ok(updates, "ok_with_callbacks")),
        offset_path=tmp_path / "telegram_offset",
        feedback_path=tmp_path / "feedback.jsonl",
        clock=lambda: datetime(2026, 8, 18, 7, 0, tzinfo=KST),
    )
    assert result.ignored == 1
    assert result.errors == []


def test_unknown_source_becomes_error_not_wrong_channel():
    """★ 모르는 소스는 "papers" 로 기본값을 주지 않습니다 — M6 가 엉뚱한 프로파일의
    가중치를 올립니다. 그리고 그 실패가 배치 전체를 멈추지도 않습니다.

    깨뜨리는 법: channel_for_item_id 를 `SOURCE_CHANNELS.get(source, "papers")` 로
    바꾸면 빨간불.
    확인일: 2026-08-18
    """
    events, next_offset, stats, errors = C.parse_updates(
        [{"update_id": 5, "callback_query": {"data": "read:notion:abc123"}}],
        ts="2026-08-18T07:00:00+09:00",
    )
    assert events == []
    assert len(errors) == 1 and "notion" in errors[0]
    assert next_offset == 6, "해석 실패한 업데이트도 offset 은 전진해야 합니다"
    assert stats["received"] == 1


# ─────────────────────────────────────────────────────────────────────────
# 7. webhook 409 · 상태 파일
# ─────────────────────────────────────────────────────────────────────────


def test_webhook_conflict_names_forG_bot(bot_env, tmp_path, updates):
    """★ 409 는 "네트워크가 이상한가?" 로 하루를 태우기 딱 좋은 형태로 옵니다.
    메시지가 원인(webhook)과 조치(전용 봇 생성)를 그대로 말해야 합니다 (v1.0 §13).

    깨뜨리는 법: _check_response 의 409 분기를 지우면 일반 FeedbackError 가 되어
    match="forG" 가 빨간불.
    확인일: 2026-08-18
    """
    with pytest.raises(C.WebhookConflictError) as excinfo:
        C.collect(
            transport=ScriptedTransport((409, updates["webhook_conflict"])),
            offset_path=tmp_path / "telegram_offset",
            feedback_path=tmp_path / "feedback.jsonl",
            clock=lambda: datetime(2026, 8, 18, 7, 0, tzinfo=KST),
        )
    text = str(excinfo.value)
    assert "409" in text and "webhook" in text and "forG" in text
    assert not (tmp_path / "feedback.jsonl").exists(), "실패했는데 파일이 남았습니다"


def test_offset_state_file_is_gitignored():
    """★ offset 상태 파일은 커밋되면 안 됩니다. `data/cache/` 가 이미 gitignore 라
    그 아래에 둡니다 — 상태 파일 하나 때문에 게이트 파일을 고치지 않습니다.

    깨뜨리는 법: collector.DEFAULT_OFFSET_PATH 를 `Path("data/telegram_offset")` 로
    바꾸면 빨간불.
    확인일: 2026-08-18
    """
    assert C.DEFAULT_OFFSET_PATH == Path("data/cache/telegram_offset")
    gitignore = (Path(__file__).resolve().parents[1] / ".gitignore").read_text(encoding="utf-8")
    assert "data/cache/" in gitignore.splitlines()


def test_corrupt_offset_file_is_loud(bot_env, tmp_path):
    """상태 파일이 깨졌을 때 조용히 0 으로 돌아가면 24시간치가 중복 수집됩니다.

    깨뜨리는 법: read_offset 의 예외 처리를 `return OffsetState()` 로 바꾸면 빨간불.
    확인일: 2026-08-18
    """
    offset_path = tmp_path / "telegram_offset"
    offset_path.write_text("{not json", encoding="utf-8")
    with pytest.raises(C.FeedbackStateError, match="offset"):
        C.read_offset(offset_path)


# ─────────────────────────────────────────────────────────────────────────
# 8. 유실 감지 — 원장 params 용
# ─────────────────────────────────────────────────────────────────────────


def test_detect_gap_flags_loss_over_24h():
    """★ 하루 실패하면 그날 피드백은 사라집니다. 막을 수 없으니 판정해서 남깁니다
    (기획안_2 §8.1).

    깨뜨리는 법: detect_gap 의 `gap_hours > limit` 를 `> limit * 10` 으로 바꾸면 빨간불.
    확인일: 2026-08-18
    """
    now = datetime(2026, 8, 18, 7, 0, tzinfo=KST)
    gap = C.detect_gap("2026-08-16T07:00:00+09:00", now)
    assert gap.lost is True
    assert gap.gap_hours == 48.0
    assert "유실" in gap.reason


def test_detect_gap_ok_within_24h():
    """깨뜨리는 법: detect_gap 이 항상 lost=True 를 돌려주게 하면 빨간불.
    확인일: 2026-08-18
    """
    now = datetime(2026, 8, 18, 7, 0, tzinfo=KST)
    gap = C.detect_gap("2026-08-17T07:00:00+09:00", now)
    assert gap.lost is False
    assert gap.gap_hours == 24.0


def test_detect_gap_requires_timezone():
    """시간대 없는 문자열을 KST 로 가정하면 9시간이 조용히 어긋납니다.

    깨뜨리는 법: detect_gap 의 tzinfo 검사를 지우면 TypeError(offset-naive 비교)로
    어긋난 실패가 나며 match="시간대" 가 빨간불.
    확인일: 2026-08-18
    """
    with pytest.raises(C.FeedbackStateError, match="시간대"):
        C.detect_gap("2026-08-17T07:00:00", datetime(2026, 8, 18, 7, 0, tzinfo=KST))


def test_collect_params_carry_gap_for_ledger(bot_env, tmp_path, updates):
    """유실 판정이 원장 params 로 넘어갈 수 있어야 합니다 (기획안_2 §8.1).
    이 모듈은 원장을 직접 쓰지 않습니다 — `ledger.RunRecord` 로만 씁니다 (R3).

    깨뜨리는 법: CollectResult.to_params 에서 "gap" 키를 빼면 빨간불.
    확인일: 2026-08-18
    """
    offset_path = tmp_path / "telegram_offset"
    C.write_offset(
        C.OffsetState(next_offset=1, collected_at="2026-08-16T07:00:00+09:00"), offset_path
    )
    result = C.collect(
        transport=ScriptedTransport(_ok(updates, "empty")),
        offset_path=offset_path,
        feedback_path=tmp_path / "feedback.jsonl",
        clock=lambda: datetime(2026, 8, 18, 7, 0, tzinfo=KST),
    )
    params = result.to_params()
    assert params["gap"]["lost"] is True
    assert params["gap"]["gap_hours"] == 48.0
    assert params["events"] == 0
    assert params["updates_received"] == 0


def test_collect_uses_caller_supplied_last_collected_at(bot_env, tmp_path, updates):
    """★ Actions 러너의 `data/cache/` 는 매 실행 비어 있어 offset 파일의 시각을 믿을 수
    없습니다. 호출부가 커밋되는 원장의 직전 실행 시각을 넘길 수 있어야 합니다.

    깨뜨리는 법: collect 에서 `last_collected_at` 인자를 무시하고 항상
    `state.collected_at` 을 쓰면 빨간불 (상태 파일이 없어 lost=False 가 됨).
    확인일: 2026-08-18
    """
    result = C.collect(
        transport=ScriptedTransport(_ok(updates, "empty")),
        offset_path=tmp_path / "telegram_offset",
        feedback_path=tmp_path / "feedback.jsonl",
        clock=lambda: datetime(2026, 8, 18, 7, 0, tzinfo=KST),
        last_collected_at="2026-08-16T07:00:00+09:00",
    )
    assert result.gap.lost is True
