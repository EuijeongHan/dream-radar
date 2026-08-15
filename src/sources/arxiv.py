"""arXiv Search API 수집기.

인증이 없습니다. 키 없이 지금 실동작합니다 (델타 §D6.1).

지키는 것
---------
1. **3초 1회 + 단일 커넥션** — `SingleConnectionRateLimiter.slot()` 안에서만 요청을
   엽니다. 병렬화 코드를 넣지 마세요 (델타 §D2). 이건 성능 튜닝 여지가 아니라
   약관 요구사항이고 위반 시 IP 차단입니다.
2. **초록까지만** — PDF 링크는 파싱해서 버립니다. 다운로드도 캐시도 하지 않습니다.
   arXiv 전문은 대부분 재배포 불가 라이선스입니다 (델타 §D1).
3. **원문 abs 링크 필수** — arXiv 이용약관 권고 "Direct users to arXiv.org to
   retrieve e-print content."

라이선스 필드는 Search API 스키마에 **없습니다.** OAI-PMH에만 있습니다. 그래서
`fulltext_ok`는 여기서 항상 기본값 False이고, 선정된 top-N에 한해 M2 이후 OAI-PMH로
확인합니다 (델타 §D1). 100건 전부 조회하면 3초 × 100 = 300초입니다.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta
from typing import Any, Callable

import requests

from src.core.models import Item
from src.core.ratelimit import SingleConnectionRateLimiter

log = logging.getLogger(__name__)

ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"

USER_AGENT = "radar/0.1 (personal paper digest; https://github.com/EuijeongHan/dream-radar)"

#: 페이징 안전장치. 창(window) 밖으로 나가면 자연히 멈추지만, 응답이 이상할 때
#: 무한 루프로 arXiv를 두드리는 것보다 덜 수집하고 멈추는 쪽이 낫습니다.
MAX_PAGES = 12

Transport = Callable[[str, dict[str, Any]], str]


class ArxivAdapter:
    name = "arxiv"
    channel = "papers"
    # 델타 §D1 — 초록은 CC0이므로 공개 발행이 가능합니다. 어댑터가 리터럴로 고정합니다.
    publish_scope = "public"

    def __init__(
        self,
        config: dict[str, Any],
        *,
        window_hours: float | None = None,
        limiter: SingleConnectionRateLimiter | None = None,
        transport: Transport | None = None,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.endpoint = config.get("endpoint", "http://export.arxiv.org/api/query")
        self.categories: list[str] = list(config["categories"])
        self.page_size = int(config.get("max_results", 100))
        self.sort_by = config.get("sort_by", "submittedDate")
        self.sort_order = config.get("sort_order", "descending")
        self.window_hours = float(
            window_hours if window_hours is not None else config.get("window_hours", 48)
        )

        max_connections = int(config.get("max_connections", 1))
        if max_connections != 1:
            raise ValueError(
                f"arXiv max_connections 는 1이어야 합니다 (델타 §D2): {max_connections}"
            )

        self.limiter = limiter or SingleConnectionRateLimiter(
            min_interval_sec=float(config.get("min_interval_sec", 3.0))
        )
        self._transport = transport or self._http_get
        self._now = now
        self._session: requests.Session | None = None
        self.pages_fetched = 0

    # ── 수집 ────────────────────────────────────────────────────────────

    def collect(self) -> list[Item]:
        cutoff = self._now() - timedelta(hours=self.window_hours)
        query = " OR ".join(f"cat:{category}" for category in self.categories)

        items: list[Item] = []
        self.pages_fetched = 0
        for page in range(MAX_PAGES):
            params = {
                "search_query": query,
                "start": page * self.page_size,
                "max_results": self.page_size,
                "sortBy": self.sort_by,
                "sortOrder": self.sort_order,
            }
            # ★ 요청은 반드시 슬롯 안에서. 슬롯 = 배타 락 + 최소 간격 보장.
            with self.limiter.slot():
                body = self._transport(self.endpoint, params)
            self.pages_fetched += 1

            entries = self._parse(body)
            if not entries:
                break

            page_items = [self._to_item(entry) for entry in entries]
            in_window = [item for item in page_items if self._published_at(item) >= cutoff]
            items.extend(in_window)

            # submittedDate 내림차순이므로, 창 밖 항목이 섞이기 시작하면 이후 페이지는
            # 전부 창 밖입니다. 여기서 멈추는 게 정상 종료 경로입니다.
            if len(in_window) < len(page_items):
                break
            if len(entries) < self.page_size:
                break
        else:
            log.warning(
                "MAX_PAGES(%d) 도달. 창(%.0fh) 밖으로 나가기 전에 멈췄습니다.",
                MAX_PAGES,
                self.window_hours,
            )

        return items

    # ── HTTP ────────────────────────────────────────────────────────────

    def _http_get(self, url: str, params: dict[str, Any]) -> str:
        if self._session is None:
            session = requests.Session()
            session.headers["User-Agent"] = USER_AGENT
            # 단일 커넥션을 커넥션 풀 수준에서도 명시합니다. 실제 직렬화는 파일락이
            # 하지만, 풀 크기를 1로 두면 의도가 코드에 남습니다 (델타 §D2).
            adapter = requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=1)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            self._session = session
        response = self._session.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.text

    # ── 파싱 ────────────────────────────────────────────────────────────

    @staticmethod
    def _parse(body: str) -> list[ET.Element]:
        root = ET.fromstring(body)
        return root.findall(f"{ATOM_NS}entry")

    def _to_item(self, entry: ET.Element) -> Item:
        raw_id = _text(entry, f"{ATOM_NS}id")  # http://arxiv.org/abs/2608.01234v1
        # `/abs/` 뒤 **전체**를 씁니다. 마지막 슬래시 뒤만 잘라내면 구형 ID
        # (`math.GT/0309136v1`)에서 아카이브 접두사가 날아가 다른 아카이브의 같은 번호와
        # 충돌합니다. 2007년 이전 ID라 신규 제출에는 안 나오지만, 충돌하면 중복 제거가
        # 조용히 틀립니다 — 조용히 틀리는 쪽이 비쌉니다.
        tail = raw_id.split("/abs/", 1)[-1] if "/abs/" in raw_id else raw_id.rsplit("/", 1)[-1]
        arxiv_id, version = _split_version(tail)

        # abs 페이지 링크. 델타 §D1 — 모든 산출물에 필수입니다.
        abs_url = f"https://arxiv.org/abs/{arxiv_id}"

        authors = tuple(
            _text(node, f"{ATOM_NS}name")
            for node in entry.findall(f"{ATOM_NS}author")
            if _text(node, f"{ATOM_NS}name")
        )
        categories = tuple(
            term
            for node in entry.findall(f"{ATOM_NS}category")
            if (term := node.attrib.get("term"))
        )
        primary = entry.find(f"{ARXIV_NS}primary_category")

        return Item(
            id=f"arxiv:{arxiv_id}",
            source=self.name,
            channel=self.channel,
            title=_normalise(_text(entry, f"{ATOM_NS}title")),
            abstract=_normalise(_text(entry, f"{ATOM_NS}summary")),
            url=abs_url,
            published=_text(entry, f"{ATOM_NS}published"),
            updated=_text(entry, f"{ATOM_NS}updated"),
            publish_scope=self.publish_scope,
            authors=authors,
            categories=categories,
            # 델타 §D1 — Search API에는 라이선스 필드가 없습니다. 확인 전에는 False.
            fulltext_ok=False,
            raw={
                "version": version,
                "primary_category": primary.attrib.get("term") if primary is not None else None,
                "comment": _text(entry, f"{ARXIV_NS}comment") or None,
                "doi": _text(entry, f"{ARXIV_NS}doi") or None,
                # PDF 링크는 의도적으로 저장하지 않습니다. 저장해두면 언젠가 누가
                # 받습니다. 전문은 대부분 재배포 불가입니다 (델타 §D1).
            },
        )

    @staticmethod
    def _published_at(item: Item) -> datetime:
        return datetime.fromisoformat(item.published.replace("Z", "+00:00"))


# ── 헬퍼 ────────────────────────────────────────────────────────────────


def _text(node: ET.Element, path: str) -> str:
    found = node.find(path)
    return (found.text or "").strip() if found is not None else ""


def _normalise(text: str) -> str:
    """Atom은 제목·초록을 여러 줄로 접어서 줍니다. 한 줄로 폅니다."""
    return " ".join(text.split())


def _split_version(tail: str) -> tuple[str, str | None]:
    """`'2608.01234v2'` → `('2608.01234', 'v2')`.

    id에서 버전을 떼는 이유는 `models.Item` 주석 참조 — 개정판이 새 아이템으로
    보이면 중복 제거가 무력해집니다.
    """
    base, sep, version = tail.rpartition("v")
    if sep and version.isdigit():
        return base, f"v{version}"
    return tail, None
