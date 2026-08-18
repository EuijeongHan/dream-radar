"""사람인 채용공고 API 수집기 (기획안_2 §8.2 — 법적으로 가장 빡빡한 구간).

**papers 파이프라인을 복사한 것이 아닙니다.** 사람인 API는 직무내용 본문을 주지
않습니다. 반환 필드는 제목·키워드·코드값 수준이고, 그래서 이 채널은 요약 대신
적합도 판단을, faithfulness 대신 `src/verify/filter_check.py`를 씁니다.

지키는 것 — 전부 CLAUDE.md §3의 조항입니다
-------------------------------------------
1. **`publish_scope = "private"` 리터럴** (§3-3). 설정으로 바꿀 수 없습니다.
   공고를 공개 게시하면 사람인 약관의 제휴 관계 오인 조항에 걸립니다. **무료여도**
   걸립니다. `tests/test_gates.py::test_publish_scope_literal_on_job_sources` 가
   이 파일의 **소스 텍스트**를 검사합니다.
2. **공식 API만** (§3-5). 웹페이지 파싱 코드는 이 파일에 없고, 앞으로도 넣지
   않습니다 — 사람인 웹페이지 스크래핑은 DB제작자 권리 침해 판례가 있습니다.
3. **일 500회 하드 한도** (§3-7). `DailyCallCounter`가 호출 **직전에** 거부합니다.
   에러코드 4(일일 최대 요청 횟수 초과)를 받으면 **즉시 중단하고 재시도하지
   않습니다.** 설정으로 재시도하게 만들 수도 없습니다.
4. **access-key를 로그·예외·산출물에 남기지 않습니다** (§3-2). 이 저장소는
   public이고 Actions 로그도 공개됩니다. `requests`가 던지는 예외에는 쿼리스트링이
   통째로 들어가므로(= 키가 들어가므로) 원본 예외를 그대로 올려보내지 않습니다.

파라미터를 코드에 쓰지 않는 이유
--------------------------------
검색 조건은 전부 `data/profile.jobs_1.yaml`의 `sources.saramin`에 확정돼 있고
이 파일은 그것을 읽기만 합니다 (기획안_2 §8.2 "임의 변경 금지"). 프로파일은
gitignore라 git 이력이 없고 **파일 번호가 유일한 이력**이므로, 경로가 아니라
`load_profile("profile.jobs")`로 읽습니다 (재사용 규칙 R1).

`keywords`는 AND입니다
----------------------
공식 가이드: "하나의 매개변수에서 공백으로 구분한 여러 값은 모두 포함하여
검색되며", "와일드카드(*, ?)와 AND/OR 는 지원하지 않습니다." 즉 OR 검색이
없습니다. → `keyword_queries`의 키워드마다 **개별 호출 후 id 합집합**을 취합니다.
"""

from __future__ import annotations

import urllib.parse

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import requests

from src.core.config import load_profile
from src.core.models import Item
from src.verify.filter_check import JobFilters, load_filters, passes_hard_filters

log = logging.getLogger(__name__)

#: 카운터 파일명이 KST 날짜로 갈립니다 = 자정 리셋 (기획안_2 §8.2).
#: `src.core.pipeline.KST`를 import 하지 않는 이유: pipeline이 소스 어댑터를
#: import 하므로 순환이 됩니다. 상수 한 줄을 복제하는 쪽이 쌉니다.
KST = timezone(timedelta(hours=9))

#: 공식 가이드의 요청 URI. 프로파일에 `endpoint`가 없을 때만 씁니다.
DEFAULT_ENDPOINT = "https://oapi.saramin.co.kr/job-search"

#: access-key는 **환경변수에서만** 옵니다. 파일·프로파일·인자로 받지 않습니다.
ACCESS_KEY_ENV = "SARAMIN_ACCESS_KEY"

#: ★ 사람인 약관/가이드의 일일 최대 요청 횟수 (CLAUDE.md §3-7). 설정으로 올릴 수
#: 없습니다. 프로파일의 `daily_call_budget`은 이 값 **아래에서만** 유효합니다.
HARD_DAILY_LIMIT = 500

#: 카운터 파일 위치. `data/cache/`는 이미 gitignore 입니다 (기획안_2 §5.3).
CACHE_DIR = Path("data/cache")

#: 키워드 1개당 페이지 상한. 창(published_window_days)이 1일이라 실제로는 1~2페이지에서
#: 끝나지만, 응답이 이상할 때 무한히 두드리는 것보다 덜 수집하고 멈추는 게 낫습니다.
MAX_PAGES_PER_KEYWORD = 3

#: 에러코드 (공식 가이드 "에러 출력 결과").
ERROR_NO_KEY = 1
ERROR_INVALID_KEY = 2
ERROR_INVALID_PARAM = 3
ERROR_DAILY_LIMIT = 4
ERROR_GENERIC = 99

#: ★ 이 코드는 **설정과 무관하게** 즉시 중단입니다 (CLAUDE.md §3-7).
ALWAYS_ABORT_CODES = frozenset({ERROR_DAILY_LIMIT})

USER_AGENT = "radar/0.1 (personal job digest; https://github.com/EuijeongHan/dream-radar)"

Transport = Callable[[str, dict[str, Any]], str]


# ── 예외 ────────────────────────────────────────────────────────────────


class SaraminError(RuntimeError):
    """사람인 수집 실패. 메시지에 access-key가 들어가지 않습니다 (CLAUDE.md §3-2)."""


class MissingAccessKey(SaraminError):
    """키가 없습니다. **조용히 빈 결과로 폴백하지 않습니다** (델타 §D6.2)."""


class CallBudgetExceeded(SaraminError):
    """우리 쪽 카운터가 호출을 거부했습니다. 네트워크에 나가기 전에 막힙니다."""


class SaraminAPIError(SaraminError):
    """API가 에러코드를 반환했습니다.

    `ledger_fields`는 `ledger.RunRecord`의 `params`/`gates`에 그대로 넣을 수 있는
    형태입니다. 재사용 규칙 R3 — 원장은 `RunRecord`로만 쓰므로, 여기서 dict를
    만들어 넘기고 기록은 호출자가 합니다.
    """

    def __init__(self, code: int, message: str, *, calls_made: int, date: str) -> None:
        super().__init__(f"사람인 API 에러코드 {code}: {message}")
        self.code = code
        self.api_message = message
        self.ledger_fields: dict[str, Any] = {
            "source": "saramin",
            "error_code": code,
            "error_message": message,
            "calls_made": calls_made,
            "kst_date": date,
            "retried": False,
        }


class DailyLimitReached(SaraminAPIError):
    """★ 에러코드 4. 즉시 중단 · 원장 기록 · **재시도 금지** (CLAUDE.md §3-7)."""


# ── 일일 호출 카운터 ────────────────────────────────────────────────────


class DailyCallCounter:
    """KST 날짜별 호출 카운터.

    `data/cache/saramin_calls_{YYYY-MM-DD}.json` — 파일명이 날짜로 갈리므로
    자정 리셋이 자동입니다. 기획안_2 §8.2는 "카운터를 state.db에 두고"라고 썼지만
    `state.db` 스키마는 `src/core/state.py` 소유이고 M5가 손대면 안 됩니다.
    파일 하나로 같은 보장(프로세스 재시작 후에도 유지 · 자정 리셋)을 얻습니다.

    프로세스가 여러 개면 경합이 있습니다. 지금 실행 계약은 "하루 1회 Actions
    단발 실행"이라 경합이 없고, 있더라도 **적게 세는 쪽이 아니라 많이 세는 쪽으로**
    틀리게 설계했습니다 — 요청 **전에** 증가시킵니다.
    """

    def __init__(
        self,
        budget: int,
        *,
        hard_limit: int | None = None,
        cache_dir: Path | str | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        # 기본값을 모듈 상수로 바인딩하지 않습니다 (기획안_2 §9.1).
        self._budget = int(budget)
        self._hard_limit = HARD_DAILY_LIMIT if hard_limit is None else int(hard_limit)
        self._cache_dir = cache_dir
        self._now = now or (lambda: datetime.now(KST))

    # ── 경로·상태 ──

    @property
    def cache_dir(self) -> Path:
        return Path(self._cache_dir) if self._cache_dir is not None else CACHE_DIR

    @property
    def date(self) -> str:
        """KST 기준 오늘. 호출 시점에 해석하므로 자정을 넘기면 파일이 바뀝니다."""
        return self._now().astimezone(KST).date().isoformat()

    @property
    def path(self) -> Path:
        return self.cache_dir / f"saramin_calls_{self.date}.json"

    @property
    def limit(self) -> int:
        """★ 예산과 하드 한도 중 **작은 쪽**. 설정으로 500을 넘길 수 없습니다."""
        return min(self._budget, self._hard_limit)

    def used(self) -> int:
        return int(self._read().get("calls", 0))

    def remaining(self) -> int:
        return max(0, self.limit - self.used())

    # ── 동작 ──

    def reserve(self) -> int:
        """호출 1회를 **미리** 기록합니다. 한도를 넘기면 호출 전에 예외입니다."""
        state = self._read()
        if state.get("exhausted"):
            raise CallBudgetExceeded(
                f"오늘({self.date}) 사람인 호출이 소진 표시돼 있습니다: "
                f"{state.get('reason', '')} (CLAUDE.md §3-7)"
            )
        used = int(state.get("calls", 0))
        if used + 1 > self.limit:
            raise CallBudgetExceeded(
                f"오늘({self.date}) 사람인 호출 한도 도달: {used}/{self.limit} "
                f"(예산 {self._budget}, 하드 한도 {self._hard_limit}, CLAUDE.md §3-7)"
            )
        state["calls"] = used + 1
        self._write(state)
        return state["calls"]

    def exhaust(self, reason: str) -> None:
        """에러코드 4를 받았을 때. 오늘은 더 이상 호출하지 않습니다."""
        state = self._read()
        state["exhausted"] = True
        state["reason"] = reason
        self._write(state)

    # ── 파일 ──

    def _read(self) -> dict[str, Any]:
        path = self.path
        if not path.exists():
            return {"date": self.date, "calls": 0, "exhausted": False}
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            # ★ 조용히 0으로 리셋하면 그날 500회를 넘길 수 있습니다.
            raise SaraminError(
                f"사람인 호출 카운터를 읽을 수 없습니다: {path} ({exc.__class__.__name__}). "
                "0으로 리셋하지 않습니다 — 일 500회 한도를 넘길 수 있습니다 (CLAUDE.md §3-7)"
            ) from None
        if not isinstance(state, dict):
            raise SaraminError(f"사람인 호출 카운터가 매핑이 아닙니다: {path}")
        return state

    def _write(self, state: dict[str, Any]) -> None:
        path = self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        state.setdefault("date", self.date)
        state["budget"] = self._budget
        state["hard_limit"] = self._hard_limit
        temp = path.parent / f"{path.name}.tmp"
        temp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        # 원자적 교체. 쓰다가 죽어도 카운터가 반쯤 쓰인 상태로 남지 않습니다.
        temp.replace(path)


# ── 어댑터 ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _ErrorPolicy:
    """프로파일 `error_handling`. 코드 4는 이 정책과 **무관하게** 중단입니다."""

    abort_on: frozenset[int]
    retry_on: frozenset[int]
    max_retries: int


class SaraminAdapter:
    name = "saramin"
    channel = "jobs"
    # ★ CLAUDE.md §3-3 — 채용공고 공개 게시 금지(무료여도 위반). 어댑터가 리터럴로
    # 고정하므로 설정 파일을 고쳐도 공개 싱크로 흘러갈 수 없습니다 (델타 §D4).
    publish_scope = "private"

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        profile: Mapping[str, Any] | None = None,
        filters: JobFilters | None = None,
        counter: DailyCallCounter | None = None,
        cache_dir: Path | str | None = None,
        transport: Transport | None = None,
        now: Callable[[], datetime] | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        if config is None or filters is None:
            # R1 — 경로가 아니라 stem으로. 호출 시점에 해석합니다.
            if profile is None:
                profile, _path = load_profile("profile.jobs")
            if config is None:
                config = profile.get("sources", {}).get("saramin", {})
        self._config: Mapping[str, Any] = config
        self._filters = filters if filters is not None else load_filters(profile)

        self.endpoint = str(config.get("endpoint") or DEFAULT_ENDPOINT)
        self.fixed_params: dict[str, Any] = dict(config.get("fixed_params") or {})
        self.job_mid_cd: list[str] = [str(code) for code in (config.get("job_mid_cd") or [])]
        self.job_cd: list[str] = [str(code) for code in (config.get("job_cd") or [])]
        self.keyword_queries: list[str] = [str(kw) for kw in (config.get("keyword_queries") or [])]
        self.rank_text_fields: list[str] = [
            str(field) for field in (config.get("rank_text_fields") or [])
        ]
        self.published_window_days = float(config.get("published_window_days", 1))
        self.page_size = int(self.fixed_params.get("count", 110))

        if self.page_size < 1:
            # 0이면 "덜 받은 페이지" 판정이 성립하지 않아 페이지 상한까지 두드립니다.
            # 불필요한 호출은 그 자체로 약관 리스크입니다 (CLAUDE.md §3-7).
            raise ValueError(
                f"sources.saramin.fixed_params.count 는 1 이상이어야 합니다: {self.page_size}"
            )
        if not self.keyword_queries:
            raise ValueError(
                "sources.saramin.keyword_queries 가 비어 있습니다. "
                "keywords 는 AND라 키워드별 개별 호출이 유일한 OR 경로입니다 (기획안_2 §8.2)"
            )
        if not self.job_cd:
            # 미확정 — `docs/사람인_직무코드_조사.md` §2의 3회 호출로 먼저 가려야 합니다.
            # 추측으로 채우면 조용히 엉뚱한 직무를 수집합니다. 경고만 남기고 진행합니다
            # (job_mid_cd=22 만으로도 수집은 됩니다. 넓게 가져와 랭킹으로 좁힙니다).
            log.warning(
                "sources.saramin.job_cd 가 비어 있습니다 — 직무코드 체계가 미확정입니다. "
                "docs/사람인_직무코드_조사.md §2 의 3회 호출로 확정한 뒤 "
                "profile.jobs_N.yaml 에 채우세요. 지금은 job_mid_cd 만으로 수집합니다."
            )

        self._error_policy = _build_error_policy(config.get("error_handling"))
        self._transport = transport or self._http_get
        self._now = now or (lambda: datetime.now(KST))
        self._sleep = sleeper or time.sleep
        self._session: requests.Session | None = None

        self.counter = counter or DailyCallCounter(
            int(config.get("daily_call_budget", 60)),
            cache_dir=cache_dir,
            now=self._now,
        )

        #: 원장 `params`에 넣을 실측치.
        self.calls_made = 0
        self.filtered_out = 0
        self.skipped_no_url = 0
        self.pages_fetched: dict[str, int] = {}

    # ── 수집 ────────────────────────────────────────────────────────────

    def collect(self) -> list[Item]:
        """키워드별로 호출하고 id 합집합을 돌려줍니다.

        `keywords`가 AND라 OR 검색이 없습니다 (기획안_2 §8.2). 같은 공고가 여러
        키워드에 걸리는 것이 정상이고, **먼저 만난 쪽을 남깁니다** — 어느 키워드로
        걸렸는지는 랭킹에 영향을 주지 않습니다.
        """
        access_key = _require_access_key()
        self.calls_made = 0
        self.filtered_out = 0
        self.skipped_no_url = 0
        self.pages_fetched = {}

        merged: dict[str, Item] = {}
        for keyword in self.keyword_queries:
            for item in self._collect_keyword(keyword, access_key):
                merged.setdefault(item.id, item)
        log.info(
            "사람인 수집 완료: %d건 (호출 %d회, 필터 제외 %d건)",
            len(merged),
            self.calls_made,
            self.filtered_out,
        )
        return list(merged.values())

    def _collect_keyword(self, keyword: str, access_key: str) -> list[Item]:
        items: list[Item] = []
        pages = 0
        for page in range(MAX_PAGES_PER_KEYWORD):
            payload = self._request(self._params(keyword, page, access_key), access_key)
            pages = page + 1
            jobs, total = _extract_jobs(payload)

            for job in jobs:
                item = self._to_item(job)
                if item is None:
                    continue
                if passes_hard_filters(item, self._filters):
                    items.append(item)
                else:
                    self.filtered_out += 1

            # 공식 가이드: `start`는 오프셋이 아니라 **검색 결과 페이지 번호**입니다
            # ("start: 검색 결과의 페이지 번호"). 그래서 소진 판정도 페이지 단위입니다.
            # 빈 응답(0건)도 여기서 걸립니다 — `page_size >= 1`이 생성자에서 보장됩니다.
            if len(jobs) < self.page_size:
                break
            if total is not None and (page + 1) * self.page_size >= total:
                break
        else:
            log.warning(
                "키워드 페이지 상한(%d)에 도달했습니다. 창이 넓거나 응답이 이상합니다.",
                MAX_PAGES_PER_KEYWORD,
            )
        self.pages_fetched[keyword] = pages
        return items

    # ── 요청 ────────────────────────────────────────────────────────────

    def _params(self, keyword: str, page: int, access_key: str) -> dict[str, Any]:
        published_max = self._now()
        published_min = published_max - timedelta(days=self.published_window_days)
        params: dict[str, Any] = {
            "access-key": access_key,
            "keywords": keyword,
            # 프로파일에 확정된 값만 넣습니다. `sr=directhire`(헤드헌팅·파견 제외)가
            # 여기 포함됩니다 — API 레벨 1차 방어이고, `exclude_keywords.soft`의
            # "파견"은 실제 응답으로 동작을 확인할 때까지 **지우지 않습니다**
            # (기획안_2 §8.2 / 델타 §D9 — 이중 방어).
            **self.fixed_params,
            # 공식 가이드의 datetime|timestamp 중 timestamp를 씁니다. 문자열 날짜
            # 형식은 로캘·타임존 해석 여지가 있는데 timestamp에는 없습니다.
            "published_min": int(published_min.timestamp()),
            "published_max": int(published_max.timestamp()),
            "start": page,
        }
        if self.job_mid_cd:
            # 복수검색은 공백 또는 콤마 구분 (공식 가이드).
            params["job_mid_cd"] = ",".join(self.job_mid_cd)
        if self.job_cd:
            params["job_cd"] = ",".join(self.job_cd)
        return params

    def _request(self, params: dict[str, Any], access_key: str) -> dict[str, Any]:
        """1회 호출 + 에러코드 처리. 재시도는 `retry_on`에 한합니다."""
        attempt = 0
        while True:
            # ★ 예산 확인이 먼저입니다. 넘으면 네트워크에 나가지 않습니다.
            self.counter.reserve()
            self.calls_made += 1
            body = self._call_transport(params, access_key)
            try:
                payload = json.loads(body)
            except json.JSONDecodeError as exc:
                raise SaraminError(
                    f"사람인 응답을 JSON으로 읽을 수 없습니다: {exc.__class__.__name__}"
                ) from None
            if not isinstance(payload, dict):
                raise SaraminError("사람인 응답이 매핑이 아닙니다")

            error = _extract_error(payload)
            if error is None:
                return payload

            code, message = error
            attempt += 1
            self._raise_or_retry(code, message, attempt)

    def _raise_or_retry(self, code: int, message: str, attempt: int) -> None:
        """에러코드 정책. **코드 4는 정책과 무관하게 즉시 중단입니다.**"""
        date = self.counter.date
        if code in ALWAYS_ABORT_CODES:
            # ★ CLAUDE.md §3-7 — 일 500회 초과. 재시도하면 위반이 반복됩니다.
            # 프로파일의 `retry_on`에 4를 넣어도 여기서 막힙니다.
            self.counter.exhaust(f"error_code={code}")
            raise DailyLimitReached(code, message, calls_made=self.calls_made, date=date)
        if code in self._error_policy.abort_on:
            # 1·2·3 = 설정 오류입니다. 재시도해도 같은 답이 옵니다.
            raise SaraminAPIError(code, message, calls_made=self.calls_made, date=date)
        if code in self._error_policy.retry_on and attempt <= self._error_policy.max_retries:
            delay = float(2 ** (attempt - 1))
            log.warning("사람인 에러코드 %d — %.0f초 후 재시도 (%d/%d)",
                        code, delay, attempt, self._error_policy.max_retries)
            self._sleep(delay)
            return
        raise SaraminAPIError(code, message, calls_made=self.calls_made, date=date)

    def _call_transport(self, params: dict[str, Any], access_key: str) -> str:
        """transport 호출을 감싸 **모든 예외 메시지에서 키를 지웁니다**.

        `requests`가 던지는 예외에는 쿼리스트링이 통째로 들어갑니다 = access-key가
        그대로 들어갑니다. 원본 예외를 체인으로 남기면 트레이스백에도 남으므로
        `from None`으로 끊습니다 (CLAUDE.md §3-2).
        """
        try:
            return self._transport(self.endpoint, params)
        except SaraminError:
            raise
        except Exception as exc:  # noqa: BLE001 — 어떤 예외든 키가 섞일 수 있습니다
            raise SaraminError(
                f"사람인 요청 실패: {_scrub(f'{exc.__class__.__name__}: {exc}', access_key)}"
            ) from None

    def _http_get(self, url: str, params: dict[str, Any]) -> str:
        if self._session is None:
            session = requests.Session()
            session.headers["User-Agent"] = USER_AGENT
            session.headers["Accept"] = "application/json"
            self._session = session
        response = self._session.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.text

    # ── 변환 ────────────────────────────────────────────────────────────

    def _to_item(self, job: Mapping[str, Any]) -> Item | None:
        job_id = str(job.get("id") or "").strip()
        url = str(job.get("url") or "").strip()
        if not job_id or not url:
            # 원문 링크 없이는 사람이 판단할 수 없습니다 (기획안_2 §8.2 "텔레그램
            # 메시지에 원문 링크 필수"). `Item`도 빈 url을 거부합니다.
            # 로그에 제목·기업명을 남기지 않습니다 (CLAUDE.md §3-3).
            self.skipped_no_url += 1
            log.warning("사람인 공고에 id 또는 url이 없어 건너뜁니다 (id=%r)", job_id)
            return None

        position = job.get("position") or {}
        title = _label(position.get("title"))
        posting_ts = _as_int(job.get("posting-timestamp"))
        modified_ts = _as_int(job.get("modification-timestamp"))
        published = str(job.get("posting-date") or "") or _iso(posting_ts)

        return Item(
            id=f"saramin:{job_id}",
            source=self.name,
            channel=self.channel,
            title=title,
            # 사람인은 본문을 주지 않습니다. 랭킹 입력 텍스트를 프로파일의
            # `rank_text_fields` 순서로 합성합니다 (기획안_2 §8.2).
            abstract=self._rank_text(job),
            url=url,
            published=published,
            updated=_iso(modified_ts) or published,
            # ★ 리터럴. 인스턴스마다 다를 수 없습니다.
            publish_scope=self.publish_scope,
            authors=(),
            categories=_categories(position),
            # 전문이 애초에 없습니다.
            fulltext_ok=False,
            # 하드 필터 판정 근거만 남깁니다. `filter_check`가 이 키들을 읽습니다.
            raw={
                "active": _as_int(job.get("active")),
                "experience_level": _code_block(position.get("experience-level")),
                "location": _code_block(position.get("location")),
                "job_type": _code_block(position.get("job-type")),
                "industry": _code_block(position.get("industry")),
                "job_mid_code": _code_block(position.get("job-mid-code")),
                "job_code": _code_block(position.get("job-code")),
                "required_education_level": _code_block(position.get("required-education-level")),
                "company": _company_name(job.get("company")),
                "keyword": _label(job.get("keyword")),
                "salary": _code_block(job.get("salary")),
                "posting_timestamp": posting_ts,
                "expiration_timestamp": _as_int(job.get("expiration-timestamp")),
                "close_type": _code_block(job.get("close-type")),
            },
        )

    def _rank_text(self, job: Mapping[str, Any]) -> str:
        parts: list[str] = []
        for field in self.rank_text_fields:
            value = _dig(job, *field.split("."))
            text = _company_name(value) if field == "company.name" else _label(value)
            if text and text not in parts:
                parts.append(text)
        return " · ".join(parts)


# ── 모듈 헬퍼 ───────────────────────────────────────────────────────────


def _require_access_key() -> str:
    """★ 키가 없으면 **명확한 예외**입니다. 조용히 빈 결과를 내지 않습니다.

    빈 결과로 폴백하면 "오늘은 공고가 없었다"와 구분되지 않고, 원장에는 0건이
    정상처럼 남습니다 (델타 §D6.2 / 작업규약 C17).
    """
    key = os.environ.get(ACCESS_KEY_ENV, "").strip()
    if not key:
        raise MissingAccessKey(
            f"환경변수 {ACCESS_KEY_ENV} 가 없습니다. 사람인 API는 access-key가 필수입니다. "
            "키를 코드·프로파일·커밋에 넣지 마세요 (CLAUDE.md §3-2)"
        )
    return key


def _scrub(text: str, secret: str) -> str:
    """메시지에서 access-key를 지웁니다 (CLAUDE.md §3-2 · 사람인 약관 조항 5).

    ★ 원문 매칭만으로는 부족합니다. requests 가 쿼리스트링을 퍼센트 인코딩하므로,
    키에 `+ / =` 가 있으면 예외 메시지에는 `abc%2Bdef%2F...` 로 들어가 원문이
    매칭되지 않습니다. 키가 아직 미발급이라 어떤 문자가 들어올지 알 수 없으므로
    인코딩 변형까지 함께 지웁니다 — 모르면 막는 쪽입니다.
    """
    if not secret:
        return text
    for form in (secret, urllib.parse.quote(secret, safe=""), urllib.parse.quote_plus(secret)):
        text = text.replace(form, "***")
    return text


def _build_error_policy(raw: Any) -> _ErrorPolicy:
    block = raw if isinstance(raw, Mapping) else {}
    abort_on = frozenset(
        code for value in (block.get("abort_on") or []) if (code := _as_int(value)) is not None
    )
    retry_on = frozenset(
        code for value in (block.get("retry_on") or []) if (code := _as_int(value)) is not None
    )
    # ★ 설정이 코드 4를 재시도하라고 해도 무시합니다 (CLAUDE.md §3-7).
    return _ErrorPolicy(
        abort_on=abort_on | ALWAYS_ABORT_CODES,
        retry_on=retry_on - ALWAYS_ABORT_CODES,
        max_retries=int(block.get("max_retries", 3)),
    )


def _extract_error(payload: Mapping[str, Any]) -> tuple[int, str] | None:
    """공식 가이드의 에러 응답: `result{code, message}`.

    코드가 문자열로 올 수 있어 정수로 맞춥니다. `code`가 최상위로 오는 변종도
    받아둡니다 — 실제 JSON 응답은 키 발급 후 확인해야 합니다.
    """
    block = payload.get("result")
    if not isinstance(block, Mapping):
        block = payload if "code" in payload and "jobs" not in payload else None
    if not isinstance(block, Mapping):
        return None
    code = _as_int(block.get("code"))
    if code is None:
        return None
    return code, str(block.get("message") or "")


def _extract_jobs(payload: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], int | None]:
    """`jobs{count, start, total, job[]}` (공식 가이드 "출력 결과")."""
    block = payload.get("jobs")
    if not isinstance(block, Mapping):
        return [], None
    jobs = block.get("job")
    if isinstance(jobs, Mapping):
        # XML→JSON 변환기가 1건일 때 리스트를 벗기는 경우가 있습니다.
        jobs = [jobs]
    if not isinstance(jobs, list):
        return [], _as_int(block.get("total"))
    return [job for job in jobs if isinstance(job, Mapping)], _as_int(block.get("total"))


def _code_block(value: Any) -> dict[str, Any] | None:
    """`{"code": "2", "min": "6", "max": "10", "name": "경력 6~10년"}` 형태를 보존합니다."""
    if isinstance(value, Mapping):
        return {str(key): value[key] for key in value}
    if value in (None, ""):
        return None
    return {"name": str(value)}


def _label(value: Any) -> str:
    """코드 블록에서 사람이 읽는 이름만 뽑습니다."""
    if isinstance(value, Mapping):
        return str(value.get("name") or value.get("#text") or "").strip()
    return str(value or "").strip()


def _company_name(value: Any) -> str:
    """가이드의 응답 스키마는 `company > name`(+ `name@href`)입니다.

    JSON 응답에서 `company > detail > name`으로 오는 형태도 받아둡니다 — 어느
    쪽인지는 키 발급 후 실제 응답으로 확정해야 합니다. 둘 다 못 찾으면 빈 문자열이고,
    회사명은 하드 필터 판정에 쓰이지 않으므로 수집을 막지 않습니다.
    """
    if not isinstance(value, Mapping):
        return _label(value)
    name = _label(value.get("name"))
    if name:
        return name
    return _label(_dig(value, "detail", "name"))


def _categories(position: Mapping[str, Any]) -> tuple[str, ...]:
    """직무명은 콤마로 구분돼 옵니다 (`"게임개발,기술지원"`)."""
    names: list[str] = []
    for key in ("job-mid-code", "job-code"):
        for token in _label(position.get(key)).split(","):
            token = token.strip()
            if token and token not in names:
                names.append(token)
    return tuple(names)


def _iso(timestamp: int | None) -> str:
    if timestamp is None:
        return ""
    return datetime.fromtimestamp(timestamp, KST).isoformat()


def _dig(mapping: Any, *keys: str) -> Any:
    node = mapping
    for key in keys:
        if not isinstance(node, Mapping):
            return None
        node = node.get(key)
    return node


def _as_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None
