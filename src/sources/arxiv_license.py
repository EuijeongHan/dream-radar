"""arXiv 라이선스 게이트 — OAI-PMH (델타 §D1 / 기획안_2 §3.1).

왜 이 파일이 따로 있는가
------------------------
라이선스 필드는 **Search API 스키마에 없습니다. OAI-PMH 출력에만 있습니다.** 그래서
수집기가 2계통입니다.

    Search API   → 검색·랭킹용 (초록까지)   — 후보 전체 (`src/sources/arxiv.py`)
    OAI-PMH      → 라이선스 확인용           — **선정된 top-N 에만** (이 파일)

**전수 조회는 금지입니다.** 3초 간격이 요구사항(델타 §D2)이라 100건이면 300초,
하루 719건이면 36분입니다. 그래서 `apply()`는 넘어온 건수가 `max_lookups`를 넘으면
요청을 **한 번도 열지 않고** 거부합니다. "선정된 5건에만"이 주석이 아니라 코드입니다.

이 게이트가 지키는 것
--------------------
- arXiv **전문**은 논문별로 라이선스가 다르고 **대부분 재배포 불가**입니다
  (arXiv 비독점 라이선스). 초록(서술 메타데이터)만 CC0입니다.
- `Item.fulltext_ok`는 **기본 False**이고, 여기서 **확인된 라이선스에만** True가 됩니다.
  확인하지 못한 것(비-arXiv 소스 · OAI 에러 · license 원소 없음)은 전부 False입니다.
  모르면 막는 쪽입니다.
- 현재 파이프라인에 전문을 읽는 코드는 없습니다. 이 필드는 미래에 "전문을 읽고
  요약하면 더 좋겠는데"가 될 자기 자신을 막는 장치입니다 (델타 §D1).

레이트리미터는 **arXiv 어댑터와 같은 것**을 씁니다 — 기본 락 파일이 같습니다
(`data/.arxiv_lock`). arXiv는 하나이므로 락도 하나여야 합니다 (기획안_2 §9.14).
게이트 전용 락을 새로 만들면 Search API 요청과 OAI-PMH 요청이 동시에 열립니다.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import replace
from typing import Any, Iterable, Sequence

import requests

from src.core.models import Item
from src.core.ratelimit import SingleConnectionRateLimiter
from src.sources.arxiv import USER_AGENT, Transport

log = logging.getLogger(__name__)

OAI_ENDPOINT = "http://export.arxiv.org/oai2"

OAI_NS = "{http://www.openarchives.org/OAI/2.0/}"
ARXIV_OAI_NS = "{http://arxiv.org/OAI/arXiv/}"

#: OAI-PMH metadataPrefix. `arXiv` 포맷에만 `<license>` 원소가 있습니다.
METADATA_PREFIX = "arXiv"

#: OAI 식별자 접두사. `oai:arXiv.org:2608.01234`
OAI_ID_PREFIX = "oai:arXiv.org:"

#: 한 번의 `apply()`에서 허용하는 최대 조회 건수 (델타 §D1 "선정된 top-N 에만").
#: 프로파일의 `selection.final_n`은 5입니다. 여유를 두되 stage1_top_n(30)에는 한참
#: 못 미치게 잡습니다 — 30건이면 90초, 후보 전체(719건)면 36분입니다.
DEFAULT_MAX_LOOKUPS = 10

#: arXiv 최소 호출 간격. 리미터 생성자가 3.0 미만을 거부합니다 (델타 §D2).
MIN_INTERVAL_SEC = 3.0


class ArxivLicenseGate:
    """선정된 아이템의 라이선스를 OAI-PMH로 확인해 `fulltext_ok`를 채웁니다.

    `Item`이 `frozen=True`라 제자리 수정이 불가능합니다 — 그게 의도입니다
    (기획안_2 §2.2). `dataclasses.replace()`로 **새 Item**을 만들어 돌려줍니다.
    """

    def __init__(
        self,
        config: dict[str, Any],
        *,
        limiter: SingleConnectionRateLimiter | None = None,
        transport: Transport | None = None,
        endpoint: str | None = None,
        max_lookups: int | None = None,
    ) -> None:
        """`config`는 프로파일의 `license_gate` 절입니다 (R1 — `load_profile()`로 읽으세요).

        설정으로 게이트를 무력화할 수 없도록 세 값을 생성자가 검사합니다.
        `max_connections != 1`을 `ArxivAdapter`가 거부하는 것과 같은 이유입니다.
        """
        allowed = config.get("fulltext_ok_licenses")
        if not allowed:
            # 조용히 빈 집합으로 넘어가면 "전부 False"라 안전하긴 하지만, 설정이
            # 깨진 채로 매일 도는 걸 아무도 모릅니다. 시끄럽게 죽는 쪽입니다.
            raise ValueError("license_gate.fulltext_ok_licenses 가 비어 있습니다 (델타 §D1)")
        # 정규화하지 않습니다. https 변형이나 끝 슬래시 유무를 관대하게 맞추면
        # **게이트가 느슨해집니다.** 못 맞추면 False로 떨어지고, 그건 안전한 방향입니다.
        self.allowed_licenses = frozenset(str(uri).strip() for uri in allowed)

        if config.get("default_fulltext_ok", False):
            # 기본값 True는 라이선스 확인 없이 전문 이용을 허용하는 것과 같습니다.
            # CLAUDE.md §3-4 는 설정으로 뒤집을 수 있는 항목이 아닙니다.
            raise ValueError(
                "license_gate.default_fulltext_ok 는 false 여야 합니다 (CLAUDE.md §3-4 / 델타 §D1)"
            )

        lookup = config.get("lookup", "oai-pmh")
        if lookup != "oai-pmh":
            raise ValueError(f"license_gate.lookup 은 'oai-pmh' 여야 합니다: {lookup!r}")

        scope = config.get("lookup_scope", "selected_only")
        if scope != "selected_only":
            # 전수 조회를 설정으로 켤 수 있으면 3초 × N 이 그대로 실행시간이 됩니다.
            raise ValueError(
                f"license_gate.lookup_scope 는 'selected_only' 여야 합니다 (델타 §D1): {scope!r}"
            )

        self.endpoint = endpoint or OAI_ENDPOINT
        self.max_lookups = DEFAULT_MAX_LOOKUPS if max_lookups is None else int(max_lookups)

        # ★ 락 파일을 지정하지 않습니다 = arXiv 어댑터와 같은 기본 락
        #   (`data/.arxiv_lock`). arXiv는 하나이므로 락도 하나입니다 (기획안_2 §9.14).
        self.limiter = limiter or SingleConnectionRateLimiter(min_interval_sec=MIN_INTERVAL_SEC)
        self._transport = transport or self._http_get
        self._session: requests.Session | None = None

        #: 실제로 연 OAI-PMH 요청 수. 원장 `params`에 남기세요.
        self.lookups = 0
        #: item_id → 라이선스 URI (확인 실패 시 None). 원장·results.md 기록용.
        self.licenses: dict[str, str | None] = {}

    # ── 게이트 ──────────────────────────────────────────────────────────

    def apply(self, items: Sequence[Item] | Iterable[Item]) -> list[Item]:
        """선정된 아이템들의 `fulltext_ok`를 확인된 값으로 갱신한 **새 리스트**를 냅니다.

        입력 Item은 건드리지 않습니다 (frozen). 순서는 보존합니다.
        """
        selected = list(items)

        lookup_targets = [item for item in selected if self._is_arxiv(item)]
        if len(lookup_targets) > self.max_lookups:
            # 요청을 한 번도 열지 않고 거부합니다. 절반 조회하다 죽으면 그게 최악입니다.
            raise ValueError(
                f"라이선스 조회는 선정된 top-N 에만 허용됩니다 (델타 §D1): "
                f"{len(lookup_targets)}건 > 상한 {self.max_lookups}건. "
                f"3초 간격이 요구사항이라 전수 조회는 수십 분입니다 (델타 §D2)"
            )

        updated: list[Item] = []
        for item in selected:
            if not self._is_arxiv(item):
                # 라이선스를 확인할 수단이 없는 아이템입니다. 확인 못 한 것은 False.
                self.licenses[item.id] = None
                updated.append(_deny(item))
                continue

            license_uri = self._fetch_license(item)
            self.licenses[item.id] = license_uri
            fulltext_ok = license_uri is not None and license_uri in self.allowed_licenses
            updated.append(replace(item, fulltext_ok=fulltext_ok))
        return updated

    @staticmethod
    def _is_arxiv(item: Item) -> bool:
        return item.source == "arxiv" and item.id.startswith("arxiv:")

    def _fetch_license(self, item: Item) -> str | None:
        params = {
            "verb": "GetRecord",
            # `arxiv:` 뒤 **전체**를 씁니다. 구형 ID(`math.GT/0309136`)의 아카이브
            # 접두사를 자르면 다른 아카이브의 같은 번호를 조회하게 됩니다 (§9.5).
            "identifier": OAI_ID_PREFIX + item.id.split(":", 1)[1],
            "metadataPrefix": METADATA_PREFIX,
        }
        # ★ 요청은 반드시 슬롯 안에서 (R2 / §9.14). 슬롯 = 배타 락 + 최소 간격 보장.
        #   슬롯 밖에서 열면 Search API 요청과 동시에 두 커넥션이 열립니다.
        with self.limiter.slot():
            body = self._transport(self.endpoint, params)
        self.lookups += 1
        return parse_license(body)

    # ── HTTP ────────────────────────────────────────────────────────────

    def _http_get(self, url: str, params: dict[str, Any]) -> str:
        if self._session is None:
            session = requests.Session()
            session.headers["User-Agent"] = USER_AGENT  # 어댑터와 같은 UA를 씁니다
            adapter = requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=1)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            self._session = session
        response = self._session.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.text


def parse_license(body: str) -> str | None:
    """OAI-PMH `GetRecord` 응답에서 `<license>` URI를 뽑습니다. 없으면 None.

    None이 되는 경우 3가지 — 전부 `fulltext_ok=False`로 떨어집니다.

    1. `<error code="idDoesNotExist">` 등 OAI 에러 응답
    2. 삭제된 레코드 (`<header status="deleted">` — metadata 자체가 없음)
    3. 2007년 이전 등록물처럼 `<license>` 원소가 아예 없는 레코드
       (arXiv 문서: license 원소가 없으면 arXiv 비독점 라이선스로 간주)
    """
    root = ET.fromstring(body)

    error = root.find(f"{OAI_NS}error")
    if error is not None:
        log.warning(
            "OAI-PMH 에러 — 라이선스 확인 불가 (fulltext_ok=False 유지): code=%s %s",
            error.attrib.get("code"),
            (error.text or "").strip(),
        )
        return None

    node = root.find(
        f"{OAI_NS}GetRecord/{OAI_NS}record/{OAI_NS}metadata/"
        f"{ARXIV_OAI_NS}arXiv/{ARXIV_OAI_NS}license"
    )
    if node is None or not (node.text or "").strip():
        return None
    return node.text.strip()


def _deny(item: Item) -> Item:
    """확인하지 못한 아이템은 `fulltext_ok=False`로 낮춥니다 (기본값 유지, 델타 §D1)."""
    if not item.fulltext_ok:
        return item
    return replace(item, fulltext_ok=False)
