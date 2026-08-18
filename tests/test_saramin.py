"""jobs 채널 게이트 — 사람인·워크넷 어댑터와 하드 필터 검사 (기획안_2 §8.2).

★ 이 파일의 테스트는 대부분 **법적 제약을 코드로 고정한 것**입니다. 비활성화하지
  마세요. CLAUDE.md §3의 5·6·7·8·9번이 전부 여기 걸립니다.

  - 일 500회 하드 한도 · 에러코드 4 즉시중단 · 재시도 금지 (§3-7)
  - access-key 비노출 (§3-2)
  - publish_scope="private" 리터럴 (§3-3)
  - jobs 채널에 faithfulness 미적용 (§3-8)

키 없이 전부 통과합니다. 실제 API를 부르는 테스트는 없습니다 — transport를 주입해
`tests/fixtures/saramin_response.json`(가공 데이터)을 돌려줍니다.

★ 아래 `PROFILE`은 **가공**입니다. 실제 `data/profile.jobs_1.yaml`을 복사해 오지
  마세요. 이 파일은 커밋되므로 본인의 희망 연차·지역·제외 조건이 그대로
  노출됩니다 (CLAUDE.md §3-1).
"""

from __future__ import annotations

import copy
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from src.core.models import Item
from src.sources.base import SourceAdapter
from src.sources.saramin import (
    ACCESS_KEY_ENV,
    HARD_DAILY_LIMIT,
    KST,
    CallBudgetExceeded,
    DailyCallCounter,
    DailyLimitReached,
    MissingAccessKey,
    SaraminAdapter,
    SaraminAPIError,
    SaraminError,
    _build_error_policy,
)
from src.sources.worknet import (
    SERVICE_KEY_ENV,
    MissingServiceKey,
    WorknetAdapter,
)
from src.verify.filter_check import (
    RULE_ACTIVE,
    RULE_EXCLUDE_KEYWORD,
    RULE_EXPERIENCE_CODE,
    RULE_EXPERIENCE_YEARS,
    RULE_LOCATION,
    RULE_MISSING_EVIDENCE,
    FilterViolationError,
    JobFilters,
    assert_no_filter_violations,
    check_filters,
    load_filters,
)

FIXTURE = Path(__file__).parent / "fixtures" / "saramin_response.json"

#: 가짜 키. 실제 키 형태가 아니고 어디에도 저장되지 않습니다.
FAKE_KEY = "fake-access-key-for-tests-only"

EMPTY_BODY = json.dumps({"jobs": {"count": 0, "start": 0, "total": 0, "job": []}})

#: 픽스처에서 하드 필터를 통과해야 하는 공고 (id 앞의 `saramin:` 접두사 포함).
PASSING_IDS = {"saramin:40000001", "saramin:40000007"}

PROFILE: dict[str, Any] = {
    "version": 1,
    "channel": "jobs",
    "visibility": "private",
    "filters": {
        # 가공 값입니다. 실제 프로파일과 무관합니다.
        "location": {"include": ["서울", "경기"]},
    },
    "exclude_keywords": {
        "hard": ["단순 라벨링"],
        "soft": [{"text": "파견", "penalty": 0.5}],
    },
    "sources": {
        "saramin": {
            "enabled": True,
            "endpoint": "https://oapi.saramin.co.kr/job-search",
            "fixed_params": {"sr": "directhire", "sort": "pd", "count": 110},
            "job_mid_cd": ["22"],
            "job_cd": ["2248"],
            "keyword_queries": ["가상키워드1", "가상키워드2"],
            "published_window_days": 1,
            "daily_call_budget": 60,
            "rank_text_fields": [
                "position.title",
                "position.job-code",
                "keyword",
                "position.industry",
                "company.name",
                "position.location",
            ],
            "response_filters": {
                "active": 1,
                "experience_level_code": [0, 1, 3],
                "experience_max_years": 3,
            },
            "error_handling": {"abort_on": [1, 2, 3, 4], "retry_on": [99], "max_retries": 3},
        }
    },
}


# ── 하네스 ──────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def fake_access_key(monkeypatch):
    """모든 테스트에 가짜 키를 넣습니다. 실제 키 없이 전 구간이 돕니다 (델타 §D6.2)."""
    monkeypatch.setenv(ACCESS_KEY_ENV, FAKE_KEY)


@pytest.fixture(scope="module")
def body() -> str:
    return FIXTURE.read_text(encoding="utf-8")


class FakeTransport:
    """호출을 기록하고 미리 정한 본문을 돌려줍니다. 네트워크에 나가지 않습니다."""

    def __init__(self, *bodies: str, repeat_last: bool = True) -> None:
        self.bodies = list(bodies)
        self.repeat_last = repeat_last
        self.calls: list[dict[str, Any]] = []

    def __call__(self, url: str, params: dict[str, Any]) -> str:
        self.calls.append(dict(params))
        index = len(self.calls) - 1
        if index < len(self.bodies):
            return self.bodies[index]
        if self.repeat_last and self.bodies:
            return self.bodies[-1]
        return EMPTY_BODY

    @property
    def count(self) -> int:
        return len(self.calls)


def _profile(**saramin_overrides: Any) -> dict[str, Any]:
    profile = copy.deepcopy(PROFILE)
    profile["sources"]["saramin"].update(saramin_overrides)
    return profile


def _adapter(tmp_path, transport, *, profile=None, counter=None, now=None) -> SaraminAdapter:
    profile = profile or copy.deepcopy(PROFILE)
    return SaraminAdapter(
        profile["sources"]["saramin"],
        profile=profile,
        counter=counter,
        cache_dir=tmp_path / "cache",
        transport=transport,
        now=now or (lambda: datetime(2026, 8, 18, 9, 0, tzinfo=KST)),
        sleeper=lambda _seconds: None,
    )


def _error_body(code: Any, message: str = "오류") -> str:
    """공식 가이드의 에러 응답: `result{code, message}`."""
    return json.dumps({"result": {"code": code, "message": message}})


# ── ★ publish_scope 리터럴 (CLAUDE.md §3-3) ─────────────────────────────


def test_saramin_publish_scope_is_private():
    """★ 채용공고는 공개 발행할 수 없습니다 (사람인 약관 — 무료여도 위반).

    깨뜨리는 법: SaraminAdapter.publish_scope 를 "public" 으로 바꾸면 빨간불.
    확인일: 2026-08-18
    """
    assert SaraminAdapter.publish_scope == "private"


def test_saramin_items_are_private_scope(tmp_path, body):
    """★ 클래스 속성뿐 아니라 **만들어진 Item** 이 private 이어야 합니다.

    깨뜨리는 법: _to_item 의 publish_scope=self.publish_scope 를 "public" 으로
    바꾸면 빨간불. (클래스 속성만 검사하면 이 경로가 비어 있는 채로 통과합니다)
    확인일: 2026-08-18
    """
    items = _adapter(tmp_path, FakeTransport(body)).collect()
    assert items
    assert {item.publish_scope for item in items} == {"private"}


def test_job_source_modules_never_mention_public_scope():
    """★ test_gates.py 의 소스 텍스트 검사와 같은 것을 여기서도 봅니다.

    깨뜨리는 법: saramin.py 나 worknet.py 어디든 `publish_scope = "public"` 을
    적으면 빨간불.
    확인일: 2026-08-18
    """
    for name in ("saramin", "worknet"):
        text = (Path(__file__).parents[1] / "src" / "sources" / f"{name}.py").read_text("utf-8")
        assert 'publish_scope = "private"' in text
        assert 'publish_scope = "public"' not in text


# ── 키워드 합집합 (keywords 는 AND — 기획안_2 §8.2) ─────────────────────


def test_each_keyword_gets_its_own_call(tmp_path, body):
    """keywords 가 AND라 OR 검색이 없습니다 → 키워드마다 개별 호출.

    깨뜨리는 법: collect() 의 `for keyword in self.keyword_queries` 루프를 지우고
    " ".join(keyword_queries) 를 한 번만 호출하게 바꾸면 빨간불 (호출 1회).
    확인일: 2026-08-18
    """
    transport = FakeTransport(body)
    adapter = _adapter(tmp_path, transport)
    adapter.collect()

    sent = [call["keywords"] for call in transport.calls]
    assert sent == ["가상키워드1", "가상키워드2"]
    for call in transport.calls:
        assert " " not in call["keywords"], "키워드를 합쳐 보내면 AND 검색이 됩니다"


def test_ids_are_unioned_not_duplicated(tmp_path, body):
    """같은 공고가 두 키워드에 걸려도 1건입니다 (id 합집합).

    깨뜨리는 법: collect() 의 merged.setdefault 를 list.append 로 바꾸면
    2배로 늘어 빨간불.
    확인일: 2026-08-18
    """
    items = _adapter(tmp_path, FakeTransport(body)).collect()
    ids = [item.id for item in items]
    assert len(ids) == len(set(ids)), "합집합이 아니라 이어붙였습니다"
    assert set(ids) == PASSING_IDS


def test_union_covers_keywords_with_disjoint_results(tmp_path, body):
    """키워드마다 다른 공고가 나오면 **합쳐져야** 합니다.

    깨뜨리는 법: collect() 가 마지막 키워드 결과만 반환하도록 merged 를 매
    키워드마다 초기화하면 빨간불.
    확인일: 2026-08-18
    """
    payload = json.loads(body)
    jobs = payload["jobs"]["job"]
    first = _slice_body(payload, [job for job in jobs if job["id"] == "40000001"])
    second = _slice_body(payload, [job for job in jobs if job["id"] == "40000007"])

    items = _adapter(tmp_path, FakeTransport(first, second, repeat_last=False)).collect()
    assert {item.id for item in items} == PASSING_IDS


def _slice_body(payload: dict[str, Any], jobs: list[dict[str, Any]]) -> str:
    block = dict(payload["jobs"])
    block["job"] = jobs
    block["count"] = len(jobs)
    block["total"] = len(jobs)
    return json.dumps({"jobs": block})


# ── ★ 일 500회 한도 (CLAUDE.md §3-7) ────────────────────────────────────


def test_budget_refuses_call_before_it_is_made(tmp_path, body):
    """★ 예산을 넘기면 **네트워크에 나가기 전에** 막힙니다.

    깨뜨리는 법: _request() 의 self.counter.reserve() 호출을 transport 호출
    뒤로 옮기면 3회째가 실제로 나가서 빨간불.
    확인일: 2026-08-18
    """
    transport = FakeTransport(body)
    adapter = _adapter(tmp_path, transport, profile=_profile(daily_call_budget=1))

    with pytest.raises(CallBudgetExceeded) as excinfo:
        adapter.collect()

    assert transport.count == 1, "예산을 넘긴 호출이 실제로 나갔습니다"
    assert "1" in str(excinfo.value)


def test_hard_limit_500_cannot_be_raised_by_config(tmp_path):
    """★ 설정으로 일 500회 한도를 올릴 수 없습니다 (CLAUDE.md §3-7).

    깨뜨리는 법: DailyCallCounter.limit 을 `self._budget` 만 보게 바꾸면
    9999 회까지 허용되어 빨간불.
    확인일: 2026-08-18
    """
    counter = DailyCallCounter(9999, cache_dir=tmp_path, now=lambda: datetime(2026, 8, 18, tzinfo=KST))
    assert counter.limit == HARD_DAILY_LIMIT

    counter.path.parent.mkdir(parents=True, exist_ok=True)
    counter.path.write_text(json.dumps({"calls": HARD_DAILY_LIMIT}), encoding="utf-8")
    with pytest.raises(CallBudgetExceeded):
        counter.reserve()


def test_counter_file_is_keyed_by_kst_date(tmp_path):
    """카운터는 **KST** 자정에 리셋됩니다 — 파일명이 KST 날짜로 갈립니다.

    시계를 UTC로 주는 것이 핵심입니다. KST 시각을 주면 astimezone(KST)를 지워도
    같은 값이 나와 아무것도 검증하지 못합니다 (Actions 러너는 UTC입니다).

    깨뜨리는 법: DailyCallCounter.date 에서 .astimezone(KST) 를 빼면 UTC 날짜가
    되어 파일명이 2026-08-17 이 되고 빨간불.
    확인일: 2026-08-18
    """
    # 2026-08-17 23:00 UTC = 2026-08-18 08:00 KST. KST 기준이면 18일입니다.
    counter = DailyCallCounter(60, cache_dir=tmp_path, now=lambda: datetime(2026, 8, 17, 23, 0, tzinfo=UTC))
    assert counter.path.name == "saramin_calls_2026-08-18.json"

    counter.reserve()
    later = DailyCallCounter(60, cache_dir=tmp_path, now=lambda: datetime(2026, 8, 18, 23, 0, tzinfo=UTC))
    assert later.path.name == "saramin_calls_2026-08-19.json"
    assert later.used() == 0, "날짜가 바뀌었는데 카운터가 이어졌습니다"


def test_counter_survives_a_new_process(tmp_path):
    """재실행해도 오늘 쓴 횟수가 남아 있어야 합니다.

    깨뜨리는 법: DailyCallCounter._write 를 no-op 으로 만들면 빨간불.
    확인일: 2026-08-18
    """
    now = lambda: datetime(2026, 8, 18, 9, 0, tzinfo=KST)  # noqa: E731
    first = DailyCallCounter(60, cache_dir=tmp_path, now=now)
    first.reserve()
    first.reserve()

    second = DailyCallCounter(60, cache_dir=tmp_path, now=now)
    assert second.used() == 2
    assert second.remaining() == 58


def test_corrupt_counter_is_not_silently_reset(tmp_path):
    """★ 카운터를 못 읽으면 0으로 리셋하지 않고 죽습니다.

    리셋하면 그날 500회를 넘길 수 있습니다 — 조용히 위반하는 경로입니다.

    깨뜨리는 법: DailyCallCounter._read 의 except 절에서 raise 대신
    `return {"calls": 0}` 을 돌려주면 빨간불.
    확인일: 2026-08-18
    """
    counter = DailyCallCounter(60, cache_dir=tmp_path, now=lambda: datetime(2026, 8, 18, tzinfo=KST))
    counter.path.parent.mkdir(parents=True, exist_ok=True)
    counter.path.write_text("{망가진 JSON", encoding="utf-8")

    with pytest.raises(SaraminError, match="카운터"):
        counter.reserve()


def test_calls_are_counted_across_keywords(tmp_path, body):
    """호출 수가 원장에 남을 수 있도록 실측됩니다 (기획안_2 §7 params).

    깨뜨리는 법: _request 의 self.calls_made += 1 을 지우면 빨간불.
    확인일: 2026-08-18
    """
    transport = FakeTransport(body)
    adapter = _adapter(tmp_path, transport)
    adapter.collect()
    assert adapter.calls_made == transport.count == 2
    assert adapter.counter.used() == 2


# ── ★ 에러코드 4 — 즉시 중단 · 재시도 금지 (CLAUDE.md §3-7) ────────────


def test_error_code_4_aborts_immediately_without_retry(tmp_path):
    """★ 일일 한도 초과(코드 4)는 즉시 중단이고 재시도하지 않습니다.

    깨뜨리는 법: _raise_or_retry 의 `if code in ALWAYS_ABORT_CODES` 블록을 지우면
    코드 4가 일반 에러 경로로 내려가 재시도되고 빨간불(호출 4회).
    확인일: 2026-08-18
    """
    transport = FakeTransport(_error_body(4, "일일 최대 요청 가능 횟수 초과"))
    adapter = _adapter(tmp_path, transport)

    with pytest.raises(DailyLimitReached) as excinfo:
        adapter.collect()

    assert transport.count == 1, "에러코드 4 이후에 재시도했습니다"
    assert excinfo.value.code == 4
    assert excinfo.value.ledger_fields["retried"] is False
    assert excinfo.value.ledger_fields["calls_made"] == 1
    assert excinfo.value.ledger_fields["source"] == "saramin"


def test_error_code_4_marks_the_day_exhausted(tmp_path):
    """★ 코드 4를 받으면 같은 날 추가 호출도 거부됩니다.

    깨뜨리는 법: _raise_or_retry 의 self.counter.exhaust(...) 를 지우면
    두 번째 어댑터가 그대로 호출해서 빨간불.
    확인일: 2026-08-18
    """
    adapter = _adapter(tmp_path, FakeTransport(_error_body(4, "초과")))
    with pytest.raises(DailyLimitReached):
        adapter.collect()

    transport = FakeTransport(EMPTY_BODY)
    again = _adapter(tmp_path, transport)
    with pytest.raises(CallBudgetExceeded):
        again.collect()
    assert transport.count == 0


def test_error_code_4_is_never_retried_even_if_config_says_so(tmp_path):
    """★ 프로파일이 `retry_on: [4]` 라고 해도 재시도하지 않습니다.

    약관 제약이라 설정으로 열 수 없습니다 (CLAUDE.md §3-7).

    깨뜨리는 법: _build_error_policy 의 `retry_on=retry_on - ALWAYS_ABORT_CODES`
    에서 뺄셈을 지우면 4가 재시도 대상이 되어 빨간불.
    확인일: 2026-08-18
    """
    profile = _profile(
        error_handling={"abort_on": [1, 2, 3], "retry_on": [4, 99], "max_retries": 3}
    )
    transport = FakeTransport(_error_body(4, "초과"))
    adapter = _adapter(tmp_path, transport, profile=profile)

    with pytest.raises(DailyLimitReached):
        adapter.collect()
    assert transport.count == 1


@pytest.mark.parametrize("code", [1, 2, 3])
def test_setup_error_codes_abort_without_retry(tmp_path, code):
    """1·2·3 은 설정 오류라 재시도해도 같은 답이 옵니다.

    깨뜨리는 법: _raise_or_retry 의 `if code in self._error_policy.abort_on`
    블록을 지우면 (retry_on 에 없으므로) 여전히 raise 지만, retry_on 에 1을 넣은
    프로파일에서는 재시도됩니다. 확실히 보려면 abort_on 검사를 지우고
    error_handling.retry_on 에 [1,2,3,99] 를 넣어 실행하세요.
    확인일: 2026-08-18
    """
    transport = FakeTransport(_error_body(code, "설정 오류"))
    adapter = _adapter(tmp_path, transport)

    with pytest.raises(SaraminAPIError) as excinfo:
        adapter.collect()
    assert excinfo.value.code == code
    assert transport.count == 1


def test_error_policy_can_never_put_code_4_in_retry_on(tmp_path):
    """★ 이중 방어 — 정책을 만드는 단계에서 코드 4를 재시도 목록에서 뺍니다.

    `_raise_or_retry`의 ALWAYS_ABORT 분기가 1차 방어이고, 이건 2차입니다.
    한쪽만 남으면 나중에 누가 다른 쪽을 "중복"이라며 지웁니다 (델타 §D9와 같은
    이중 방어 논리 — `sr=directhire` 와 exclude_keywords "파견"의 관계).

    깨뜨리는 법: _build_error_policy 의 `retry_on - ALWAYS_ABORT_CODES` 에서
    뺄셈을 지우면 빨간불.
    확인일: 2026-08-18
    """
    policy = _build_error_policy({"abort_on": [1], "retry_on": [4, 99], "max_retries": 3})
    assert 4 not in policy.retry_on, "설정이 코드 4를 재시도 목록에 넣었습니다 (CLAUDE.md §3-7)"
    assert 4 in policy.abort_on
    assert 99 in policy.retry_on


def test_generic_error_99_retries_then_gives_up(tmp_path):
    """99(오류 발생)만 재시도 대상입니다. max_retries 를 넘기면 죽습니다.

    깨뜨리는 법: _raise_or_retry 의 `attempt <= max_retries` 를 무조건 True 로
    바꾸면 영원히 재시도하다 예산에서 막혀 다른 예외가 나서 빨간불.
    확인일: 2026-08-18
    """
    transport = FakeTransport(_error_body(99, "오류 발생"))
    adapter = _adapter(tmp_path, transport)

    with pytest.raises(SaraminAPIError) as excinfo:
        adapter.collect()
    assert excinfo.value.code == 99
    # 첫 호출 + 재시도 3회
    assert transport.count == 4


def test_retry_consumes_budget(tmp_path):
    """재시도도 호출입니다. 예산에서 빠져야 합니다.

    깨뜨리는 법: _request 의 while 루프 밖으로 reserve() 를 빼면 재시도가
    공짜가 되어 빨간불.
    확인일: 2026-08-18
    """
    adapter = _adapter(tmp_path, FakeTransport(_error_body(99, "오류")))
    with pytest.raises(SaraminAPIError):
        adapter.collect()
    assert adapter.counter.used() == 4


# ── ★ access-key 비노출 (CLAUDE.md §3-2) ────────────────────────────────


def test_missing_access_key_raises_instead_of_returning_nothing(tmp_path, monkeypatch):
    """★ 키가 없으면 빈 결과가 아니라 예외입니다 (델타 §D6.2).

    빈 결과로 폴백하면 "오늘은 공고가 없었다"와 구분되지 않고, 원장에 0건이
    정상처럼 남습니다.

    깨뜨리는 법: _require_access_key 가 raise 대신 "" 를 돌려주게 하면
    collect() 가 조용히 돌아 빨간불.
    확인일: 2026-08-18
    """
    monkeypatch.delenv(ACCESS_KEY_ENV, raising=False)
    transport = FakeTransport(EMPTY_BODY)
    adapter = _adapter(tmp_path, transport)

    with pytest.raises(MissingAccessKey):
        adapter.collect()
    assert transport.count == 0


def test_access_key_never_appears_in_exception_message(tmp_path):
    """★ transport 예외에 키가 섞여 있어도 밖으로 나가지 않습니다.

    requests 는 예외 메시지에 쿼리스트링(=키)을 통째로 넣습니다.

    깨뜨리는 법: _call_transport 의 except 절을 지우고 원본 예외를 그대로
    올려보내면 키가 메시지에 남아 빨간불.
    확인일: 2026-08-18
    """

    def exploding(url: str, params: dict[str, Any]) -> str:
        raise RuntimeError(f"HTTP 500 for url: {url}?access-key={params['access-key']}")

    adapter = _adapter(tmp_path, exploding)
    with pytest.raises(SaraminError) as excinfo:
        adapter.collect()

    message = str(excinfo.value)
    assert FAKE_KEY not in message, "예외 메시지에 access-key 가 남았습니다"
    assert "***" in message


def test_access_key_never_reaches_the_log(tmp_path, body, caplog):
    """★ 로그에도 남지 않습니다. public repo 의 Actions 로그는 공개됩니다.

    깨뜨리는 법: _request 나 _params 에 log.info("%s", params) 같은 줄을 넣으면
    빨간불.
    확인일: 2026-08-18
    """
    with caplog.at_level(logging.DEBUG):
        _adapter(tmp_path, FakeTransport(body)).collect()
    assert FAKE_KEY not in caplog.text


def test_access_key_is_not_stored_on_items(tmp_path, body):
    """★ 키가 후보 파일(`data/candidates/*.jsonl`)로 흘러가지 않습니다.

    깨뜨리는 법: _to_item 의 raw 에 "params": params 같은 키를 추가하면 빨간불.
    확인일: 2026-08-18
    """
    items = _adapter(tmp_path, FakeTransport(body)).collect()
    serialised = json.dumps([item.to_dict() for item in items], ensure_ascii=False)
    assert FAKE_KEY not in serialised


def test_log_does_not_leak_posting_titles(tmp_path, body, caplog):
    """★ 공고 제목·기업명을 로그에 남기지 않습니다 (CLAUDE.md §3-3).

    public repo 라 Actions 로그가 공개됩니다. 로그에 공고를 쓰면 그 자체가
    "채용공고 공개 게시"입니다.

    깨뜨리는 법: _to_item 의 url 없음 경고에 `job.get("position")` 을 넣으면
    "AI 평가 엔지니어"(픽스처의 url 없는 공고 제목)가 로그에 나와 빨간불.
    확인일: 2026-08-18
    """
    with caplog.at_level(logging.DEBUG):
        _adapter(tmp_path, FakeTransport(body)).collect()
    assert "AI 평가 엔지니어" not in caplog.text
    assert "링크없는예시" not in caplog.text


# ── 하드 필터 (experience-level@code 등) ────────────────────────────────


def test_experience_code_2_is_dropped(tmp_path, body):
    """★ 경력직(코드 2) 공고는 수집 단계에서 빠집니다 (기획안_2 §8.2).

    깨뜨리는 법: _collect_keyword 의 passes_hard_filters(...) 분기를 지우고
    전부 append 하면 40000002 가 들어와 빨간불.
    확인일: 2026-08-18
    """
    items = _adapter(tmp_path, FakeTransport(body)).collect()
    assert "saramin:40000002" not in {item.id for item in items}


def test_experience_max_years_is_applied(tmp_path, body):
    """경력코드가 통과해도 연차 상한(@max)을 넘으면 빠집니다.

    40000004 는 code=1(신입)이라 코드 검사만으로는 통과합니다.

    깨뜨리는 법: filter_check.check_item 의 experience_max_years 블록을 지우면
    40000004 가 들어와 빨간불.
    확인일: 2026-08-18
    """
    items = _adapter(tmp_path, FakeTransport(body)).collect()
    assert "saramin:40000004" not in {item.id for item in items}


def test_location_filter_is_applied(tmp_path, body):
    """근무지가 include 목록 밖이면 빠집니다.

    깨뜨리는 법: check_item 의 location 블록을 지우면 부산 공고(40000003)가
    들어와 빨간불.
    확인일: 2026-08-18
    """
    items = _adapter(tmp_path, FakeTransport(body)).collect()
    assert "saramin:40000003" not in {item.id for item in items}


def test_hard_exclude_keyword_is_applied(tmp_path, body):
    """즉시 제외 키워드가 제목에 있으면 빠집니다.

    깨뜨리는 법: check_item 의 exclude_keywords 루프를 지우면 40000005 가
    들어와 빨간불.
    확인일: 2026-08-18
    """
    items = _adapter(tmp_path, FakeTransport(body)).collect()
    assert "saramin:40000005" not in {item.id for item in items}


def test_closed_posting_is_dropped(tmp_path, body):
    """마감(active=0) 공고는 추천해도 쓸모가 없습니다.

    깨뜨리는 법: check_item 의 active 블록을 지우면 40000006 이 들어와 빨간불.
    확인일: 2026-08-18
    """
    items = _adapter(tmp_path, FakeTransport(body)).collect()
    assert "saramin:40000006" not in {item.id for item in items}


def test_posting_without_url_is_skipped(tmp_path, body):
    """원문 링크가 없으면 사람이 판단할 수 없습니다 (기획안_2 §8.2).

    깨뜨리는 법: _to_item 의 url 검사를 지우면 Item 생성자가 ValueError 를 던져
    수집 전체가 죽습니다 (빨간불의 형태가 다를 뿐 여전히 빨간불).
    확인일: 2026-08-18
    """
    adapter = _adapter(tmp_path, FakeTransport(body))
    items = adapter.collect()
    assert "saramin:40000008" not in {item.id for item in items}
    assert adapter.skipped_no_url == 2  # 키워드 2개 × 1건


def test_filtered_out_is_counted(tmp_path, body):
    """제외 건수가 실측돼야 원장에서 "왜 5건뿐인가"에 답할 수 있습니다.

    깨뜨리는 법: _collect_keyword 의 self.filtered_out += 1 을 지우면 빨간불.
    확인일: 2026-08-18
    """
    adapter = _adapter(tmp_path, FakeTransport(body))
    adapter.collect()
    # 픽스처 8건 중 통과 2 · url 없음 1 · 필터 제외 5, 키워드 2개
    assert adapter.filtered_out == 10


# ── 요청 파라미터 (프로파일 확정값을 그대로) ────────────────────────────


def test_fixed_params_from_profile_are_sent(tmp_path, body):
    """★ `sr=directhire` — 헤드헌팅·파견을 API 레벨에서 제외 (기획안_2 §8.2).

    깨뜨리는 법: _params 의 `**self.fixed_params` 를 지우면 빨간불.
    확인일: 2026-08-18
    """
    transport = FakeTransport(body)
    _adapter(tmp_path, transport).collect()
    call = transport.calls[0]
    assert call["sr"] == "directhire"
    assert call["sort"] == "pd"
    assert call["count"] == 110
    assert call["job_mid_cd"] == "22"


def test_published_window_comes_from_profile(tmp_path, body):
    """전날 신규만 가져옵니다 (published_window_days).

    깨뜨리는 법: _params 의 published_min 계산에서 timedelta 를 지우면
    두 값이 같아져 빨간불.
    확인일: 2026-08-18
    """
    transport = FakeTransport(body)
    _adapter(tmp_path, transport).collect()
    call = transport.calls[0]
    assert call["published_max"] - call["published_min"] == 86400


def test_start_is_a_page_number_not_an_offset(tmp_path, body):
    """공식 가이드: "start: 검색 결과의 페이지 번호".

    오프셋으로 보내면 2페이지째가 110번째 결과부터가 아니라 110페이지째가 됩니다.

    깨뜨리는 법: _params 의 "start": page 를 page * self.page_size 로 바꾸면
    빨간불.
    확인일: 2026-08-18
    """
    payload = json.loads(body)
    full = dict(payload["jobs"])
    full["count"] = 2
    full["total"] = 4
    page_body = json.dumps({"jobs": {**full, "job": payload["jobs"]["job"][:2]}})

    profile = _profile(fixed_params={"sr": "directhire", "sort": "pd", "count": 2})
    transport = FakeTransport(page_body)
    _adapter(tmp_path, transport, profile=profile).collect()

    starts = [call["start"] for call in transport.calls]
    assert starts[:2] == [0, 1], f"start 가 페이지 번호가 아닙니다: {starts}"


def test_paging_stops_when_total_is_reached(tmp_path, body):
    """총건수를 다 읽었으면 더 부르지 않습니다 (불필요 호출 = 약관 리스크).

    깨뜨리는 법: _collect_keyword 의 total 종료 조건을 지우면 페이지 상한까지
    돌아 호출이 늘고 빨간불.
    확인일: 2026-08-18
    """
    payload = json.loads(body)
    two = payload["jobs"]["job"][:2]
    page_body = json.dumps({"jobs": {"count": 2, "start": 0, "total": 2, "job": two}})

    profile = _profile(fixed_params={"sr": "directhire", "sort": "pd", "count": 2})
    transport = FakeTransport(page_body)
    _adapter(tmp_path, transport, profile=profile).collect()
    assert transport.count == 2, "키워드당 1회씩만 불러야 합니다"


def test_empty_page_stops_paging(tmp_path):
    """빈 응답이면 즉시 멈춥니다 (불필요 호출 = 약관 리스크).

    깨뜨리는 법: _collect_keyword 의 종료 조건 **두 개를 모두** 지우면 빨간불
    (`len(jobs) < self.page_size` / `total 소진`). 하나만 지우면 다른 하나가
    잡습니다 — 의도한 이중 방어입니다. 각각을 단독으로 검증하는 것은
    test_short_page_stops_paging / test_paging_stops_when_total_is_reached 입니다.
    확인일: 2026-08-18
    """
    transport = FakeTransport(EMPTY_BODY)
    items = _adapter(tmp_path, transport).collect()
    assert items == []
    assert transport.count == 2


def test_short_page_stops_paging(tmp_path):
    """한 페이지를 덜 채워 왔으면 마지막 페이지입니다 — total 이 커도 멈춥니다.

    total 이 1000이라 total 기반 종료 조건은 걸리지 않습니다. 이 테스트는
    `len(jobs) < page_size` 종료 조건만 검증합니다.

    깨뜨리는 법: _collect_keyword 의 `if len(jobs) < self.page_size: break` 를
    지우면 키워드마다 페이지 상한(3회)까지 돌아 빨간불.
    확인일: 2026-08-18
    """
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    short = json.dumps(
        {"jobs": {"count": 2, "start": 0, "total": 1000, "job": payload["jobs"]["job"][:2]}}
    )
    transport = FakeTransport(short)
    _adapter(tmp_path, transport).collect()
    assert transport.count == 2, "덜 채워진 페이지 뒤에도 계속 불렀습니다"


def test_empty_job_cd_warns(tmp_path, caplog):
    """job_cd 는 미확정입니다 (docs/사람인_직무코드_조사.md §2).

    추측으로 채우면 조용히 엉뚱한 직무를 수집합니다. 비어 있으면 경고합니다.

    깨뜨리는 법: __init__ 의 `if not self.job_cd:` 경고를 지우면 빨간불.
    확인일: 2026-08-18
    """
    with caplog.at_level(logging.WARNING):
        _adapter(tmp_path, FakeTransport(EMPTY_BODY), profile=_profile(job_cd=[]))
    assert "job_cd" in caplog.text
    assert "사람인_직무코드_조사" in caplog.text


# ── Item 변환 ───────────────────────────────────────────────────────────


def test_rank_text_is_composed_from_profile_fields(tmp_path, body):
    """본문이 없으므로 랭킹 입력 텍스트를 응답 필드로 합성합니다 (기획안_2 §8.2).

    깨뜨리는 법: _rank_text 가 position.title 만 돌려주게 바꾸면 빨간불.
    확인일: 2026-08-18
    """
    items = {item.id: item for item in _adapter(tmp_path, FakeTransport(body)).collect()}
    text = items["saramin:40000001"].abstract
    for expected in ("AI 엔지니어", "데이터 사이언티스트", "PyTorch", "가상테크랩", "서울 > 강남구"):
        assert expected in text, f"랭킹 텍스트에 {expected} 가 없습니다"


def test_item_keeps_filter_evidence_in_raw(tmp_path, body):
    """filter_check 가 읽는 판정 근거가 raw 에 남아야 합니다.

    깨뜨리는 법: _to_item 의 raw 에서 "experience_level" 을 지우면
    filter_check 가 근거 없음으로 판정해 수집이 0건이 되고 빨간불.
    확인일: 2026-08-18
    """
    items = {item.id: item for item in _adapter(tmp_path, FakeTransport(body)).collect()}
    raw = items["saramin:40000001"].raw
    assert raw["experience_level"]["code"] == "3"
    assert raw["location"]["name"] == "서울 > 강남구"
    assert raw["active"] == 1


def test_adapter_satisfies_source_protocol(tmp_path):
    """SourceAdapter Protocol 이탈 방지 (재사용 규칙 R6).

    깨뜨리는 법: SaraminAdapter 에서 channel 속성을 지우면 빨간불.
    확인일: 2026-08-18
    """
    adapter = _adapter(tmp_path, FakeTransport(EMPTY_BODY))
    assert isinstance(adapter, SourceAdapter)
    assert adapter.name == "saramin"
    assert adapter.channel == "jobs"


# ── filter_check — jobs 채널의 faithfulness 대체물 (CLAUDE.md §3-8) ─────


def _item(item_id: str, **raw: Any) -> Item:
    base: dict[str, Any] = {
        "active": 1,
        "experience_level": {"code": "3", "min": "0", "max": "2"},
        "location": {"code": "101150", "name": "서울 > 강남구"},
    }
    base.update(raw)
    return Item(
        id=item_id,
        source="saramin",
        channel="jobs",
        title="가공 공고",
        abstract="가공 공고 · 서울 > 강남구",
        url="https://example.invalid/1",
        published="2026-08-17T09:30:00+09:00",
        updated="2026-08-17T09:30:00+09:00",
        publish_scope="private",
        raw=base,
    )


@pytest.fixture
def filters() -> JobFilters:
    return load_filters(PROFILE)


def test_clean_set_passes_the_gate(filters):
    """정상 집합에는 위반이 없습니다.

    깨뜨리는 법: check_item 이 무조건 위반 1건을 돌려주게 하면 빨간불.
    확인일: 2026-08-18
    """
    assert_no_filter_violations([_item("saramin:1")], filters=filters)


@pytest.mark.parametrize(
    "raw, rule",
    [
        ({"active": 0}, RULE_ACTIVE),
        # @max 는 상한 안이므로 **경력코드만** 걸려야 합니다 (규칙 분리 확인).
        ({"experience_level": {"code": "2", "min": "1", "max": "3"}}, RULE_EXPERIENCE_CODE),
        ({"experience_level": {"code": "1", "min": "0", "max": "10"}}, RULE_EXPERIENCE_YEARS),
        ({"location": {"code": "106000", "name": "부산 > 해운대구"}}, RULE_LOCATION),
        ({"experience_level": None}, RULE_MISSING_EVIDENCE),
    ],
)
def test_gate_flags_each_hard_filter_violation(filters, raw, rule):
    """★ 발행 직전 게이트가 하드 필터 위반을 잡습니다 (CLAUDE.md §3-8 대체 검증).

    깨뜨리는 법: check_item 의 해당 블록을 지우면 그 파라미터 케이스가 빨간불.
    확인일: 2026-08-18
    """
    violations = check_filters([_item("saramin:1", **raw)], filters=filters)
    assert [violation.rule for violation in violations] == [rule]

    with pytest.raises(FilterViolationError):
        assert_no_filter_violations([_item("saramin:1", **raw)], filters=filters)


def test_gate_flags_hard_exclude_keyword(filters):
    """제목의 즉시 제외 키워드도 위반입니다.

    깨뜨리는 법: check_item 의 exclude_keywords 루프를 지우면 빨간불.
    확인일: 2026-08-18
    """
    item = Item(
        id="saramin:9",
        source="saramin",
        channel="jobs",
        title="AI 학습데이터 단순 라벨링 모집",
        abstract="가공",
        url="https://example.invalid/9",
        published="2026-08-17T09:30:00+09:00",
        updated="2026-08-17T09:30:00+09:00",
        publish_scope="private",
        raw={
            "active": 1,
            "experience_level": {"code": "0", "min": "0", "max": "0"},
            "location": {"code": "101150", "name": "서울 > 강남구"},
        },
    )
    violations = check_filters([item], filters=filters)
    assert [violation.rule for violation in violations] == [RULE_EXCLUDE_KEYWORD]


def test_violation_report_carries_no_posting_text(filters):
    """★ 위반 리포트에 공고 제목·기업명·제외 키워드를 넣지 않습니다.

    이 저장소는 public 이고 Actions 로그도 공개됩니다. 위반 메시지에 공고를
    쓰면 그 자체가 "채용공고 공개 게시"이고(CLAUDE.md §3-3), 제외 키워드는
    본인의 구직 조건입니다(§3-1).

    깨뜨리는 법: check_item 의 RULE_EXCLUDE_KEYWORD 위반 detail 을
    f"{keyword}" 로 바꾸거나, RULE_LOCATION 의 detail 에 location@name 을 넣으면
    빨간불.
    확인일: 2026-08-18
    """
    item = Item(
        id="saramin:9",
        source="saramin",
        channel="jobs",
        title="AI 학습데이터 단순 라벨링 모집",
        abstract="무명컴퍼니 · 부산 > 해운대구",
        url="https://example.invalid/9",
        published="2026-08-17T09:30:00+09:00",
        updated="2026-08-17T09:30:00+09:00",
        publish_scope="private",
        raw={
            "active": 1,
            "experience_level": {"code": "0", "min": "0", "max": "0"},
            "location": {"code": "106000", "name": "부산 > 해운대구"},
        },
    )
    with pytest.raises(FilterViolationError) as excinfo:
        assert_no_filter_violations([item], filters=filters)

    message = str(excinfo.value)
    for secret in ("단순 라벨링", "무명컴퍼니", "부산 > 해운대구", "AI 학습데이터"):
        assert secret not in message, f"위반 메시지에 {secret!r} 가 남았습니다"
    assert "saramin:9" in message


def test_profile_without_filters_is_rejected():
    """★ 조건이 없으면 게이트가 "전부 통과"가 됩니다 — 그 상태를 거부합니다.

    깨뜨리는 법: JobFilters.from_profile 의 missing 검사를 지우고 빈 기본값을
    쓰면 예외가 안 나서 빨간불.
    확인일: 2026-08-18
    """
    empty: dict[str, Any] = {"sources": {"saramin": {}}}
    with pytest.raises(ValueError, match="하드 필터"):
        JobFilters.from_profile(empty)


def test_jobs_channel_has_no_faithfulness_verification():
    """★ CLAUDE.md §3-8 — jobs 채널에 faithfulness 를 붙이지 않습니다.

    사람인 API 가 본문을 주지 않아 대조할 원문이 없습니다. 원문 없이 매긴
    faithfulness 는 숫자만 있고 의미가 없습니다.

    깨뜨리는 법: filter_check.py 에서 src.verify.faithfulness 를 import 하거나
    faithfulness 를 계산하는 함수를 추가하면 빨간불.
    확인일: 2026-08-18
    """
    text = (Path(__file__).parents[1] / "src" / "verify" / "filter_check.py").read_text("utf-8")
    assert "from src.verify.faithfulness" not in text
    assert "import faithfulness" not in text
    assert "RuleBasedFakeVerifier" not in text


# ── 워크넷 ──────────────────────────────────────────────────────────────


def test_worknet_scope_is_private():
    """★ 워크넷도 비공개입니다. 재배포 조건 확인 전에는 열지 않습니다.

    깨뜨리는 법: WorknetAdapter.publish_scope 를 "public" 으로 바꾸면 빨간불.
    확인일: 2026-08-18
    """
    assert WorknetAdapter.publish_scope == "private"


def test_worknet_collect_refuses_to_guess_the_schema(monkeypatch):
    """★ 응답 스키마가 미확정이면 빈 리스트가 아니라 NotImplementedError 입니다.

    빈 리스트를 돌려주면 원장에 "수집 0건"이 정상처럼 남습니다.

    깨뜨리는 법: WorknetAdapter.collect 가 `return []` 하게 바꾸면 빨간불.
    확인일: 2026-08-18
    """
    monkeypatch.setenv(SERVICE_KEY_ENV, "fake-service-key-for-tests-only")
    with pytest.raises(NotImplementedError, match="스키마"):
        WorknetAdapter({"dataset": "3038225"}).collect()


def test_worknet_missing_key_raises(monkeypatch):
    """키가 없으면 조용히 폴백하지 않습니다 (델타 §D6.2).

    깨뜨리는 법: _require_service_key 가 "" 를 돌려주게 하면 NotImplementedError
    가 먼저 나서 이 테스트가 빨간불.
    확인일: 2026-08-18
    """
    monkeypatch.delenv(SERVICE_KEY_ENV, raising=False)
    with pytest.raises(MissingServiceKey):
        WorknetAdapter().collect()


def test_worknet_key_is_not_echoed(monkeypatch):
    """★ 예외 메시지에 서비스키를 넣지 않습니다 (CLAUDE.md §3-2).

    깨뜨리는 법: _require_service_key 의 메시지에 키 값을 f-string 으로 넣으면
    (키가 있을 때 경로가 다르므로) 이 테스트를 키 있는 경우로 바꿔야 잡힙니다.
    NotImplementedError 메시지에 키를 넣어도 빨간불.
    확인일: 2026-08-18
    """
    secret = "fake-service-key-for-tests-only"
    monkeypatch.setenv(SERVICE_KEY_ENV, secret)
    with pytest.raises(NotImplementedError) as excinfo:
        WorknetAdapter().collect()
    assert secret not in str(excinfo.value)
