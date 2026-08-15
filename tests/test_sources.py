"""소스 어댑터 게이트.

★ `test_arxiv_interval` / `test_arxiv_single_connection` 은 델타 §D2에 직결됩니다.
   비활성화하지 마세요. 위반 시 IP 차단이고, 제약은 본인 통제 하의 모든 머신에
   합산 적용됩니다.
"""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.core.ratelimit import SingleConnectionRateLimiter
from src.sources.arxiv import ArxivAdapter
from src.sources.base import SourceAdapter, SourceUnavailable
from src.sources.hf_papers import HFDailyPapersAdapter

FIXTURE = Path(__file__).parent / "fixtures" / "arxiv_atom_sample.xml"
EMPTY_FEED = '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'

CONFIG = {
    "endpoint": "http://export.arxiv.org/api/query",
    "categories": ["cs.CL", "cs.CV", "cs.LG", "cs.IR", "cs.AI"],
    "max_results": 100,
    "min_interval_sec": 3.0,
    "max_connections": 1,
    "window_hours": 48,
}


class FakeClock:
    """수동으로 흐르는 시계. 실제로 3초를 기다리지 않고 간격을 검증합니다."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _adapter(tmp_path, transport, *, now=None, window_hours=None, sleeper=None, clock=None):
    limiter = SingleConnectionRateLimiter(
        min_interval_sec=3.0,
        lock_path=tmp_path / ".arxiv_lock",
        sleeper=sleeper or (lambda _s: None),
        clock=clock or time.monotonic,
    )
    return ArxivAdapter(
        CONFIG,
        window_hours=window_hours,
        limiter=limiter,
        transport=transport,
        now=now or (lambda: datetime(2026, 8, 12, 12, 0, tzinfo=UTC)),
    )


# ── 레이트리밋 ★ 델타 §D2 ────────────────────────────────────────────────


def test_arxiv_interval(tmp_path):
    """연속 요청 사이에 3초 이상이 강제되는지."""
    clock = FakeClock()
    limiter = SingleConnectionRateLimiter(
        min_interval_sec=3.0,
        lock_path=tmp_path / ".arxiv_lock",
        sleeper=clock.sleep,
        clock=clock,
    )

    with limiter.slot():
        pass
    first_exit = clock.now

    with limiter.slot():
        entered = clock.now

    assert entered - first_exit >= 3.0, "두 번째 요청이 3초를 기다리지 않았습니다"
    assert clock.slept and clock.slept[0] == pytest.approx(3.0)


def test_arxiv_interval_accounts_for_request_duration(tmp_path):
    """요청 자체가 오래 걸렸으면 그만큼 덜 기다립니다 (완료 시각 기준)."""
    clock = FakeClock()
    limiter = SingleConnectionRateLimiter(
        min_interval_sec=3.0,
        lock_path=tmp_path / ".arxiv_lock",
        sleeper=clock.sleep,
        clock=clock,
    )
    with limiter.slot():
        pass
    clock.advance(10.0)  # 다른 일을 10초 하고 왔다면
    with limiter.slot():
        pass
    assert clock.slept == [], "이미 10초가 지났는데 추가로 기다렸습니다"


def test_arxiv_interval_below_three_seconds_is_rejected(tmp_path):
    """설정으로 3초 미만을 넣는 걸 코드가 거부합니다. 델타 §D2는 협상 대상이 아닙니다."""
    with pytest.raises(ValueError, match="3.0초 미만"):
        SingleConnectionRateLimiter(min_interval_sec=1.0, lock_path=tmp_path / ".lock")


def test_arxiv_single_connection(tmp_path):
    """요청이 동시에 2개 이상 열리면 실패 (델타 §D2 / §D8).

    스레드 8개가 동시에 슬롯을 잡으려 해도, 락을 쥔 하나만 transport에 진입해야 합니다.
    요청이 슬롯 **밖에서** 일어나도록 코드가 바뀌면 이 테스트가 깨집니다.
    """
    in_flight = 0
    max_in_flight = 0
    guard = threading.Lock()

    def transport(url, params):
        nonlocal in_flight, max_in_flight
        with guard:
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
        time.sleep(0.03)  # 겹칠 기회를 실제로 준다
        with guard:
            in_flight -= 1
        return EMPTY_FEED

    limiter = SingleConnectionRateLimiter(
        min_interval_sec=3.0,
        lock_path=tmp_path / ".arxiv_lock",
        sleeper=lambda _s: None,  # 간격은 위 테스트에서 검증. 여기선 동시성만 본다
    )

    def worker():
        with limiter.slot():
            transport("http://example.invalid", {})

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert max_in_flight == 1, f"동시 커넥션이 {max_in_flight}개 열렸습니다"


def test_arxiv_adapter_rejects_multi_connection_config():
    with pytest.raises(ValueError, match="max_connections"):
        ArxivAdapter({**CONFIG, "max_connections": 4})


# ── 파싱 ────────────────────────────────────────────────────────────────


def test_arxiv_parses_fixture(tmp_path):
    body = FIXTURE.read_text(encoding="utf-8")
    adapter = _adapter(tmp_path, lambda url, params: body)
    items = adapter.collect()

    assert len(items) == 2, "창 밖 항목이 걸러지지 않았습니다"
    first = items[0]
    assert first.id == "arxiv:2608.01234"
    assert first.url == "https://arxiv.org/abs/2608.01234"
    assert first.title == "Cross-Encoder Reranking for Domain-Specific Retrieval Evaluation"
    assert first.abstract.startswith("We study reranking on domain corpora")
    assert first.authors == ("Jane Researcher", "Kim Minsu")
    assert set(first.categories) == {"cs.IR", "cs.CL"}


def test_arxiv_id_strips_version(tmp_path):
    """개정판이 새 아이템으로 보이면 중복 제거가 무력해집니다 (기획안 §7 예시도 버전 없음)."""
    body = FIXTURE.read_text(encoding="utf-8")
    adapter = _adapter(tmp_path, lambda url, params: body)
    items = adapter.collect()
    revised = next(i for i in items if i.id == "arxiv:2608.05678")
    assert revised.raw["version"] == "v3"


def test_arxiv_never_stores_pdf_link(tmp_path):
    """델타 §D1 — 저장해두면 언젠가 누가 받습니다. 전문은 대부분 재배포 불가입니다."""
    body = FIXTURE.read_text(encoding="utf-8")
    adapter = _adapter(tmp_path, lambda url, params: body)
    for item in adapter.collect():
        serialised = str(item.to_dict())
        assert ".pdf" not in serialised
        assert "/pdf/" not in serialised


def test_arxiv_item_is_public_scope(tmp_path):
    """초록은 CC0라 공개 가능. 어댑터가 리터럴로 고정합니다 (델타 §D1·§D4)."""
    body = FIXTURE.read_text(encoding="utf-8")
    adapter = _adapter(tmp_path, lambda url, params: body)
    assert all(item.publish_scope == "public" for item in adapter.collect())


def test_arxiv_fulltext_ok_defaults_false(tmp_path):
    """Search API에 라이선스 필드가 없으므로 확인 전에는 False입니다 (델타 §D1)."""
    body = FIXTURE.read_text(encoding="utf-8")
    adapter = _adapter(tmp_path, lambda url, params: body)
    assert all(item.fulltext_ok is False for item in adapter.collect())


def test_arxiv_stops_paging_when_window_exhausted(tmp_path):
    """창 밖 항목이 섞이면 다음 페이지를 더 받지 않습니다. 불필요한 호출은 곧 약관 리스크입니다."""
    calls: list[dict] = []

    def transport(url, params):
        calls.append(params)
        return FIXTURE.read_text(encoding="utf-8")

    adapter = _adapter(tmp_path, transport)
    adapter.collect()
    assert len(calls) == 1
    assert adapter.pages_fetched == 1


def test_arxiv_stops_on_empty_feed(tmp_path):
    adapter = _adapter(tmp_path, lambda url, params: EMPTY_FEED)
    assert adapter.collect() == []
    assert adapter.pages_fetched == 1


def test_arxiv_query_covers_configured_categories(tmp_path):
    captured: dict = {}

    def transport(url, params):
        captured.update(params)
        return EMPTY_FEED

    _adapter(tmp_path, transport).collect()
    for category in CONFIG["categories"]:
        assert f"cat:{category}" in captured["search_query"]
    assert captured["sortBy"] == "submittedDate"
    assert captured["sortOrder"] == "descending"


# ── 인터페이스 / 보류 소스 ────────────────────────────────────────────────


def test_arxiv_adapter_satisfies_protocol():
    assert isinstance(ArxivAdapter(CONFIG), SourceAdapter)


def test_hf_papers_is_a_deliberate_stub():
    """공식 문서화된 API가 아니라 보류했습니다 (CLAUDE.md §3-5, 결정_M0 §3).

    스텁이 조용히 빈 리스트를 돌려주면 "왜 안 도나"로 시간을 씁니다. 예외로 죽습니다.
    """
    with pytest.raises(SourceUnavailable, match="공식 문서화된 API"):
        HFDailyPapersAdapter().collect()


def test_arxiv_keeps_legacy_archive_prefix(tmp_path):
    """구형 ID(`math.GT/0309136`)에서 아카이브 접두사가 날아가면 다른 아카이브의
    같은 번호와 충돌합니다. 신규 제출엔 안 나오지만 조용히 틀리는 종류의 버그입니다."""
    body = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/math.GT/0309136v2</id>
    <updated>2026-08-12T00:00:00Z</updated>
    <published>2026-08-12T00:00:00Z</published>
    <title>Legacy identifier paper</title>
    <summary>Body.</summary>
    <author><name>Old Author</name></author>
    <category term="math.GT"/>
  </entry>
</feed>"""
    adapter = _adapter(tmp_path, lambda url, params: body)
    item = adapter.collect()[0]
    assert item.id == "arxiv:math.GT/0309136"
    assert item.url == "https://arxiv.org/abs/math.GT/0309136"
    assert item.raw["version"] == "v2"
