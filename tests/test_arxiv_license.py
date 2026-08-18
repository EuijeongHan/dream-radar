"""arXiv 라이선스 게이트 (델타 §D1 / 기획안_2 §3.1).

★ 이 파일의 테스트를 비활성화하지 마세요. 여기서 막는 것은 편의 기능이 아니라
  CLAUDE.md §3-4(arXiv 전문 다운로드·저장·서빙 금지)와 §3-6(3초 1회 + 단일 커넥션)입니다.

네트워크를 쓰지 않습니다 — `transport` 주입 지점으로 OAI-PMH 응답 픽스처를 먹입니다.
키도 필요 없습니다 (arXiv API 는 인증이 없습니다).
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from src.core.models import Item
from src.core.ratelimit import DEFAULT_LOCK_PATH, SingleConnectionRateLimiter
from src.sources.arxiv_license import (
    DEFAULT_MAX_LOOKUPS,
    OAI_ID_PREFIX,
    ArxivLicenseGate,
    parse_license,
)

FIXTURES = Path(__file__).parent / "fixtures"

CC_BY = (FIXTURES / "oai_getrecord_cc_by.xml").read_text(encoding="utf-8")
NONEXCLUSIVE = (FIXTURES / "oai_getrecord_nonexclusive.xml").read_text(encoding="utf-8")

#: 프로파일 `license_gate` 절 그대로 (data/profile.papers_1.yaml).
LICENSE_GATE = {
    "fulltext_ok_licenses": [
        "http://creativecommons.org/publicdomain/zero/1.0/",
        "http://creativecommons.org/licenses/by/4.0/",
        "http://creativecommons.org/licenses/by-sa/4.0/",
    ],
    "default_fulltext_ok": False,
    "lookup": "oai-pmh",
    "lookup_scope": "selected_only",
}

_NO_LICENSE_ELEMENT = """<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
  <GetRecord><record>
    <header><identifier>oai:arXiv.org:0704.00001</identifier></header>
    <metadata><arXiv xmlns="http://arxiv.org/OAI/arXiv/">
      <id>0704.00001</id><title>Old submission without a license element</title>
    </arXiv></metadata>
  </record></GetRecord>
</OAI-PMH>"""

_OAI_ERROR = """<?xml version="1.0" encoding="UTF-8"?>
<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/">
  <responseDate>2026-08-18T02:11:07Z</responseDate>
  <error code="idDoesNotExist">No matching identifier</error>
</OAI-PMH>"""


def make_item(item_id: str = "arxiv:2608.11053", **overrides) -> Item:
    fields = {
        "id": item_id,
        "source": "arxiv",
        "channel": "papers",
        "title": "t",
        "abstract": "a",
        "url": f"https://arxiv.org/abs/{item_id.split(':', 1)[1]}",
        "published": "2026-08-12T04:11:02+00:00",
        "updated": "2026-08-12T04:11:02+00:00",
        "publish_scope": "public",
    }
    fields.update(overrides)
    return Item(**fields)


class RecordingLimiter:
    """`slot()` 안에서 요청이 열렸는지 기록하는 가짜 리미터 (R2 검사용)."""

    def __init__(self) -> None:
        self.slots = 0
        self.inside = False

    @contextmanager
    def slot(self):
        self.slots += 1
        self.inside = True
        try:
            yield
        finally:
            self.inside = False


class FakeTransport:
    """identifier → 응답 본문. 호출 시점의 리미터 상태를 함께 기록합니다."""

    def __init__(self, bodies: dict[str, str], limiter: RecordingLimiter | None = None) -> None:
        self.bodies = bodies
        self.limiter = limiter
        self.calls: list[dict] = []
        self.inside_slot: list[bool] = []

    def __call__(self, url: str, params: dict) -> str:
        self.calls.append({"url": url, **params})
        if self.limiter is not None:
            self.inside_slot.append(self.limiter.inside)
        return self.bodies[params["identifier"]]


def build_gate(bodies: dict[str, str], **kwargs) -> tuple[ArxivLicenseGate, FakeTransport]:
    limiter = kwargs.pop("limiter", None) or RecordingLimiter()
    transport = FakeTransport(bodies, limiter if isinstance(limiter, RecordingLimiter) else None)
    gate = ArxivLicenseGate(LICENSE_GATE, limiter=limiter, transport=transport, **kwargs)
    return gate, transport


# ── 라이선스 판정 ★ ──────────────────────────────────────────────────────


def test_cc_by_enables_fulltext_ok():
    """★ 허용 목록(CC-BY 4.0)에 있는 라이선스만 fulltext_ok=True 가 됩니다 (델타 §D1).

    깨뜨리는 법: arxiv_license.py 의 `license_uri in self.allowed_licenses` 를
    `license_uri is not None` 으로 바꾸면 비독점 케이스가 통과해 빨간불.
    확인일: 2026-08-18
    """
    item = make_item("arxiv:2608.11053")
    gate, _ = build_gate({OAI_ID_PREFIX + "2608.11053": CC_BY})

    (updated,) = gate.apply([item])

    assert updated.fulltext_ok is True
    assert gate.licenses["arxiv:2608.11053"] == "http://creativecommons.org/licenses/by/4.0/"


def test_nonexclusive_license_keeps_fulltext_ok_false():
    """★ arXiv 비독점 라이선스는 재배포 불가입니다 — 다수가 이것입니다 (델타 §D1).

    깨뜨리는 법: 위와 동일. 허용 목록 대조를 없애면 빨간불.
    확인일: 2026-08-18
    """
    item = make_item("arxiv:2608.01234")
    gate, _ = build_gate({OAI_ID_PREFIX + "2608.01234": NONEXCLUSIVE})

    (updated,) = gate.apply([item])

    assert updated.fulltext_ok is False
    assert gate.licenses["arxiv:2608.01234"] == "http://arxiv.org/licenses/nonexclusive-distrib/1.0/"


def test_missing_license_element_keeps_default_false():
    """license 원소가 없는 구형 레코드 → 기본값 False 유지 (델타 §D1 "기본값 False").

    깨뜨리는 법: parse_license 가 None 대신 빈 문자열을 반환하고 게이트가
    `license_uri is not None` 으로 판정하게 바꾸면 빨간불.
    확인일: 2026-08-18
    """
    item = make_item("arxiv:0704.00001")
    gate, _ = build_gate({OAI_ID_PREFIX + "0704.00001": _NO_LICENSE_ELEMENT})

    (updated,) = gate.apply([item])

    assert updated.fulltext_ok is False
    assert gate.licenses["arxiv:0704.00001"] is None


def test_oai_error_response_keeps_false_and_is_logged(caplog):
    """OAI 에러(idDoesNotExist 등)는 "확인 못 함"이고, 확인 못 한 것은 False 입니다.
    다만 **조용히** 지나가면 안 됩니다 — 엔드포인트가 통째로 망가져도 전부 False 라
    아무도 눈치채지 못합니다. 안전한 방향이지만 로그에는 남깁니다.

    깨뜨리는 법: parse_license 의 `if error is not None:` 분기를 지우면 경고가
    사라져 빨간불 (판정 자체는 우연히 False 로 같습니다 — 그래서 로그까지 봅니다).
    확인일: 2026-08-18
    """
    item = make_item("arxiv:2608.99999")
    gate, _ = build_gate({OAI_ID_PREFIX + "2608.99999": _OAI_ERROR})

    with caplog.at_level("WARNING", logger="src.sources.arxiv_license"):
        (updated,) = gate.apply([item])

    assert updated.fulltext_ok is False
    assert gate.licenses["arxiv:2608.99999"] is None
    assert "idDoesNotExist" in caplog.text


def test_https_variant_is_not_matched():
    """URI 를 정규화하지 않습니다. 관대하게 맞추면 게이트가 느슨해지고, 못 맞추면
    False(안전한 방향)로 떨어집니다.

    깨뜨리는 법: 생성자에서 URI 를 https→http 로 정규화하면 이 테스트가 빨간불.
    확인일: 2026-08-18
    """
    body = CC_BY.replace(
        "http://creativecommons.org/licenses/by/4.0/",
        "https://creativecommons.org/licenses/by/4.0/",
    )
    gate, _ = build_gate({OAI_ID_PREFIX + "2608.11053": body})

    (updated,) = gate.apply([make_item("arxiv:2608.11053")])

    assert updated.fulltext_ok is False


# ── 전수 조회 금지 ★ ─────────────────────────────────────────────────────


def test_refuses_bulk_lookup_without_opening_any_request():
    """★ 라이선스 조회는 **선정된 top-N 에만** (델타 §D1). 3초 간격이 요구사항이라
    전수 조회(719건)는 36분입니다. 절반 조회하다 죽지 않도록 요청 전에 거부합니다.

    깨뜨리는 법: apply() 의 max_lookups 검사를 지우면 빨간불
    (11건 전부 조회되어 transport.calls 가 11).
    확인일: 2026-08-18
    """
    items = [make_item(f"arxiv:2608.{n:05d}") for n in range(DEFAULT_MAX_LOOKUPS + 1)]
    bodies = {OAI_ID_PREFIX + item.id.split(":", 1)[1]: NONEXCLUSIVE for item in items}
    gate, transport = build_gate(bodies)

    with pytest.raises(ValueError, match="top-N"):
        gate.apply(items)

    assert transport.calls == [], "거부하면서 요청을 열었습니다"
    assert gate.lookups == 0


def test_allows_exactly_max_lookups():
    """경계값: 상한과 같은 건수는 통과합니다 (거부는 초과일 때만).

    깨뜨리는 법: 검사를 `>=` 로 바꾸면 빨간불.
    확인일: 2026-08-18
    """
    items = [make_item(f"arxiv:2608.{n:05d}") for n in range(DEFAULT_MAX_LOOKUPS)]
    bodies = {OAI_ID_PREFIX + item.id.split(":", 1)[1]: NONEXCLUSIVE for item in items}
    gate, transport = build_gate(bodies)

    updated = gate.apply(items)

    assert len(updated) == DEFAULT_MAX_LOOKUPS
    assert len(transport.calls) == DEFAULT_MAX_LOOKUPS
    assert gate.lookups == DEFAULT_MAX_LOOKUPS


# ── 레이트리미터 ★ ───────────────────────────────────────────────────────


def test_request_opens_only_inside_limiter_slot():
    """★ OAI-PMH 요청도 `SingleConnectionRateLimiter.slot()` **안에서만** 열립니다
    (R2 / CLAUDE.md §3-6). 슬롯 밖에서 열면 Search API 요청과 동시에 커넥션 2개입니다.

    깨뜨리는 법: _fetch_license 에서 `with self.limiter.slot():` 을 제거하고
    transport 를 직접 호출하면 빨간불 (inside_slot 이 [False]).
    확인일: 2026-08-18
    """
    limiter = RecordingLimiter()
    gate, transport = build_gate({OAI_ID_PREFIX + "2608.11053": CC_BY}, limiter=limiter)

    gate.apply([make_item("arxiv:2608.11053")])

    assert transport.inside_slot == [True], "슬롯 밖에서 요청이 열렸습니다"
    assert limiter.slots == 1


def test_one_slot_per_lookup():
    """조회 N건이면 슬롯도 N개. 한 슬롯에 여러 요청을 몰면 3초 간격이 무너집니다.

    깨뜨리는 법: apply() 에서 슬롯을 루프 바깥으로 빼면 빨간불 (slots == 1).
    확인일: 2026-08-18
    """
    limiter = RecordingLimiter()
    bodies = {
        OAI_ID_PREFIX + "2608.11053": CC_BY,
        OAI_ID_PREFIX + "2608.01234": NONEXCLUSIVE,
    }
    gate, _ = build_gate(bodies, limiter=limiter)

    gate.apply([make_item("arxiv:2608.11053"), make_item("arxiv:2608.01234")])

    assert limiter.slots == 2


def test_default_limiter_shares_the_arxiv_lock_file():
    """★ 락 파일은 arXiv 어댑터와 **같은 것**입니다 — arXiv 는 하나입니다 (§9.14).
    게이트 전용 락을 만들면 두 계통이 서로를 막지 못합니다.

    깨뜨리는 법: 생성자에서 `SingleConnectionRateLimiter(..., lock_path=...)` 로
    다른 경로를 주면 빨간불.
    확인일: 2026-08-18
    """
    gate = ArxivLicenseGate(LICENSE_GATE, transport=lambda url, params: CC_BY)

    assert isinstance(gate.limiter, SingleConnectionRateLimiter)
    assert gate.limiter.lock_path == DEFAULT_LOCK_PATH
    assert gate.limiter.min_interval_sec >= 3.0


# ── 설정 게이트 ★ ────────────────────────────────────────────────────────


def test_rejects_default_fulltext_ok_true():
    """★ 설정으로 게이트를 무력화할 수 없습니다 (CLAUDE.md §3-4 / 델타 §D1).

    깨뜨리는 법: 생성자의 default_fulltext_ok 검사를 지우면 빨간불.
    확인일: 2026-08-18
    """
    with pytest.raises(ValueError, match="default_fulltext_ok"):
        ArxivLicenseGate({**LICENSE_GATE, "default_fulltext_ok": True})


def test_rejects_full_scan_scope():
    """`lookup_scope` 를 설정으로 전수 조회로 바꿀 수 없습니다 (델타 §D1).

    깨뜨리는 법: 생성자의 lookup_scope 검사를 지우면 빨간불.
    확인일: 2026-08-18
    """
    with pytest.raises(ValueError, match="lookup_scope"):
        ArxivLicenseGate({**LICENSE_GATE, "lookup_scope": "all_candidates"})


def test_rejects_unknown_lookup_method():
    """깨뜨리는 법: 생성자의 lookup 검사를 지우면 빨간불.
    확인일: 2026-08-18
    """
    with pytest.raises(ValueError, match="lookup"):
        ArxivLicenseGate({**LICENSE_GATE, "lookup": "search-api"})


def test_rejects_empty_allowlist():
    """허용 목록이 비면 설정이 깨진 것입니다. "전부 False"로 조용히 넘어가지 않습니다.

    깨뜨리는 법: 생성자의 fulltext_ok_licenses 검사를 지우면 빨간불.
    확인일: 2026-08-18
    """
    with pytest.raises(ValueError, match="fulltext_ok_licenses"):
        ArxivLicenseGate({**LICENSE_GATE, "fulltext_ok_licenses": []})


# ── Item 취급 ────────────────────────────────────────────────────────────


def test_returns_new_items_and_leaves_input_untouched():
    """Item 은 frozen 입니다 (기획안_2 §2.2). dataclasses.replace 로 **새 Item** 을 냅니다.

    깨뜨리는 법: apply() 가 replace 대신 object.__setattr__ 로 제자리 수정하게 바꾸면
    원본 item.fulltext_ok 가 True 로 바뀌어 빨간불.
    확인일: 2026-08-18
    """
    item = make_item("arxiv:2608.11053")
    gate, _ = build_gate({OAI_ID_PREFIX + "2608.11053": CC_BY})

    (updated,) = gate.apply([item])

    assert updated is not item
    assert item.fulltext_ok is False, "입력 Item 이 바뀌었습니다"
    assert updated.fulltext_ok is True
    # 나머지 필드는 그대로여야 합니다.
    assert updated.id == item.id and updated.url == item.url
    with pytest.raises(FrozenInstanceError):
        updated.fulltext_ok = True  # type: ignore[misc]


def test_order_is_preserved():
    """깨뜨리는 법: apply() 가 dict 로 모아 재정렬하면 빨간불.
    확인일: 2026-08-18
    """
    ids = ["arxiv:2608.11053", "arxiv:2608.01234", "arxiv:0704.00001"]
    bodies = {
        OAI_ID_PREFIX + "2608.11053": CC_BY,
        OAI_ID_PREFIX + "2608.01234": NONEXCLUSIVE,
        OAI_ID_PREFIX + "0704.00001": _NO_LICENSE_ELEMENT,
    }
    gate, _ = build_gate(bodies)

    updated = gate.apply([make_item(i) for i in ids])

    assert [item.id for item in updated] == ids
    assert [item.fulltext_ok for item in updated] == [True, False, False]


def test_non_arxiv_items_are_not_looked_up_and_forced_false():
    """확인할 수단이 없는 아이템은 조회하지도, 허용하지도 않습니다.

    깨뜨리는 법: apply() 의 `_is_arxiv` 분기에서 `_deny(item)` 대신 `item` 을 그대로
    넣으면 fulltext_ok=True 가 살아남아 빨간불.
    확인일: 2026-08-18
    """
    jobs_item = make_item(
        "saramin:12345",
        source="saramin",
        channel="jobs",
        publish_scope="private",
        url="https://www.saramin.co.kr/x/12345",
        fulltext_ok=True,  # 어디선가 True 로 들어온 아이템
    )
    gate, transport = build_gate({})

    (updated,) = gate.apply([jobs_item])

    assert updated.fulltext_ok is False
    assert transport.calls == [], "arXiv 가 아닌 아이템을 OAI-PMH 로 조회했습니다"
    assert gate.lookups == 0


def test_legacy_id_keeps_archive_prefix_in_identifier():
    """구형 ID(`math.GT/0309136`)의 아카이브 접두사를 자르면 **다른 아카이브의 같은
    번호**를 조회하게 됩니다 (§9.5).

    깨뜨리는 법: _fetch_license 의 `item.id.split(":", 1)[1]` 을
    `item.id.rsplit("/", 1)[-1]` 로 바꾸면 빨간불.
    확인일: 2026-08-18
    """
    item = make_item("arxiv:math.GT/0309136", url="https://arxiv.org/abs/math.GT/0309136")
    identifier = OAI_ID_PREFIX + "math.GT/0309136"
    gate, transport = build_gate({identifier: NONEXCLUSIVE})

    gate.apply([item])

    assert transport.calls[0]["identifier"] == identifier


def test_request_params_are_getrecord_with_arxiv_prefix():
    """`<license>` 원소는 metadataPrefix=arXiv 출력에만 있습니다.

    깨뜨리는 법: METADATA_PREFIX 를 "oai_dc" 로 바꾸면 빨간불
    (oai_dc 에는 license 원소가 없어 전부 False 가 됩니다).
    확인일: 2026-08-18
    """
    gate, transport = build_gate({OAI_ID_PREFIX + "2608.11053": CC_BY})

    gate.apply([make_item("arxiv:2608.11053")])

    call = transport.calls[0]
    assert call["verb"] == "GetRecord"
    assert call["metadataPrefix"] == "arXiv"
    assert call["url"].endswith("/oai2")


# ── 파서 단위 ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "body,expected",
    [
        (CC_BY, "http://creativecommons.org/licenses/by/4.0/"),
        (NONEXCLUSIVE, "http://arxiv.org/licenses/nonexclusive-distrib/1.0/"),
        (_NO_LICENSE_ELEMENT, None),
        (_OAI_ERROR, None),
    ],
)
def test_parse_license(body, expected):
    """깨뜨리는 법: parse_license 의 XPath 에서 네임스페이스를 빼면 전부 None 이 되어 빨간불.
    확인일: 2026-08-18
    """
    assert parse_license(body) == expected
