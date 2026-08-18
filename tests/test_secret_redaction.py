"""★ 키 노출 회귀 방지 (CLAUDE.md §3-2, 사람인 약관 조항 5).

이 저장소는 **public** 이고 `data/runs.jsonl` 은 **git 추적 대상**입니다.
따라서 "예외 메시지가 원장에 들어간다"는 곧 "키가 공개 저장소에 커밋된다"입니다.

실제로 났던 사고 (2026-08-18, 적대 검토가 적발):
    requests 예외 메시지는 실패한 URL 을 통째로 담는다.
    텔레그램 URL 경로에는 봇 토큰이 들어 있다 — /bot<TOKEN>/getUpdates.
    그 문자열이 CollectResult.errors → to_params() → RunRecord.params →
    data/runs.jsonl 로 흘러 공개 저장소에 커밋될 뻔했다.

`test_no_secret_leak` 은 **파일에 하드코딩된** 키만 잡습니다. 런타임에 예외를 타고
흘러나가는 키는 못 잡습니다. 그 구멍을 이 파일이 막습니다.
"""

from __future__ import annotations

import json
import pathlib
import tempfile

import pytest
import requests

from src.feedback import collector as C
from src.publish import telegram as T
from src.sources.saramin import _scrub

TOKEN = "123456:AAHsecretTOKENvalue"
SECRET = "AAHsecretTOKENvalue"

CALLBACK_UPDATE = {
    "update_id": 1,
    "callback_query": {
        "id": "c1",
        "data": "read:arxiv:2608.01234",
        "message": {"chat": {"id": 42}},
        "from": {"id": 7},
    },
}


@pytest.fixture(autouse=True)
def _creds(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "42")


def _connection_error(url, *_args, **_kwargs):
    """requests 가 실제로 내는 형태 — 실패한 URL 이 메시지에 통째로 들어갑니다."""
    raise requests.exceptions.ConnectionError(
        f"HTTPSConnectionPool(host='api.telegram.org', port=443): "
        f"Max retries exceeded with url: {url}"
    )


def test_offset_confirm_failure_does_not_leak_token_into_ledger(tmp_path):
    """★ 이게 실제로 났던 사고입니다.

    깨뜨리는 법: collector.py 의 offset 확인 except 절에서 redact_token() 을 빼면
    빨간불. 확인일 2026-08-18.
    """
    calls = {"n": 0}

    def transport(url, params, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            return 200, {"ok": True, "result": [CALLBACK_UPDATE]}
        _connection_error(url)

    result = C.collect(
        transport=transport,
        offset_path=tmp_path / "offset",
        feedback_path=tmp_path / "feedback.jsonl",
        confirm=True,
    )
    assert result.errors, "확인 실패가 errors 에 기록되지 않았습니다 (테스트가 무의미해짐)"
    blob = json.dumps(result.to_params(), ensure_ascii=False)
    assert SECRET not in blob, f"봇 토큰이 원장 params 에 유출됩니다: {blob[:200]}"


def test_getupdates_connection_failure_does_not_leak_token(tmp_path):
    """연결 자체가 실패하는 경로 — 서버 응답 검사(_check_response)로는 못 막습니다."""
    with pytest.raises(C.FeedbackError) as excinfo:
        C.collect(
            transport=_connection_error,
            offset_path=tmp_path / "offset",
            feedback_path=tmp_path / "feedback.jsonl",
        )
    assert SECRET not in str(excinfo.value)


def test_send_message_connection_failure_does_not_leak_token():
    """전송 실패가 Actions 로그로 나갑니다. URL 이 그대로 전파되면 토큰이 찍힙니다."""
    item = {
        "id": "arxiv:2608.01234",
        "title": "T",
        "url": "https://arxiv.org/abs/2608.01234",
        "channel": "papers",
        "abstract": "a",
    }
    summary = {"problem": "p", "method": "m", "key_results": [], "limitations": [], "connection": ""}
    message = T.render_message(item, summary)
    with pytest.raises(T.TelegramError) as excinfo:
        T.send_messages([message], transport=_connection_error)
    assert SECRET not in str(excinfo.value)


@pytest.mark.parametrize(
    "key",
    ["abc+def/ghi==", "plain-alnum-key", "with space+plus"],
)
def test_saramin_scrub_handles_url_encoded_keys(key):
    """★ requests 가 쿼리스트링을 퍼센트 인코딩하므로 원문 매칭만으로는 부족합니다.

    사람인 키가 아직 미발급이라 어떤 문자가 올지 모릅니다 — 모르면 막는 쪽입니다.
    깨뜨리는 법: _scrub 에서 quote/quote_plus 변형 루프를 빼면 첫 케이스가 빨간불.
    """
    import urllib.parse

    for form in (key, urllib.parse.quote(key, safe=""), urllib.parse.quote_plus(key)):
        message = f"Max retries exceeded with url: /job-search?access-key={form}&count=110"
        assert form not in _scrub(message, key), f"키 형태 {form!r} 가 남았습니다"


def test_ledger_is_tracked_so_redaction_matters():
    """이 파일들의 전제 — runs.jsonl 이 추적 대상이 아니라면 위 테스트의 심각도가 다릅니다.

    추적에서 빠지는 변경이 생기면 이 테스트가 알려줍니다 (전제가 바뀐 것이지 결함은 아님).
    """
    import subprocess

    repo = pathlib.Path(__file__).resolve().parents[1]
    tracked = subprocess.run(
        ["git", "ls-files", "data/runs.jsonl"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    assert tracked == "data/runs.jsonl", (
        "runs.jsonl 이 더 이상 추적되지 않습니다. 키 유출의 심각도가 바뀌었으니 "
        "tests/test_secret_redaction.py 의 전제를 다시 검토하세요"
    )
