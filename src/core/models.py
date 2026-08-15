"""파이프라인 전 구간을 흐르는 아이템 모델.

M0에서 아직 아무도 읽지 않는 필드가 두 개 있습니다 — `publish_scope`, `fulltext_ok`.
둘 다 **소스 어댑터가 생성 시점에 채워야 하는** 필드라 나중에 소급하면 어댑터를
전부 고쳐야 합니다. 그래서 지금 넣습니다.

  publish_scope  델타 §D4 — jobs 어댑터는 "private" 리터럴만 낼 수 있고 설정으로
                 바꿀 수 없습니다. 공개 싱크가 이 값을 검사해 유출을 막습니다.
  fulltext_ok    델타 §D1 — arXiv 초록(서술 메타데이터)은 CC0지만 **전문은 논문별로
                 다르고 대부분 재배포 불가**입니다. 현재 파이프라인은 이 필드를
                 읽기만 하고 쓰지 않습니다. 전문을 쓰는 코드가 없기 때문입니다.
                 미래에 "전문을 읽고 요약하면 더 좋겠는데"가 될 자기 자신을 막는
                 장치입니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

PublishScope = Literal["private", "public"]

_VALID_SCOPES: frozenset[str] = frozenset({"private", "public"})


@dataclass(frozen=True, slots=True)
class Item:
    """수집된 후보 1건.

    `id`는 **버전 없는 안정 식별자**입니다 (기획안 §7의 `"arxiv:2608.01234"`).
    arXiv는 개정판마다 v2, v3가 붙는데 버전을 포함하면 개정 때마다 새 아이템으로
    보여 중복 제거가 무력해집니다. 버전은 `raw["version"]`에 남깁니다.
    """

    id: str
    source: str
    channel: str
    title: str
    abstract: str
    url: str
    published: str
    updated: str
    publish_scope: PublishScope
    authors: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    # 델타 §D1 — 기본값 False. 라이선스를 확인하기 전에는 전문 이용 불가입니다.
    fulltext_ok: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Item.id 는 비어 있을 수 없습니다")
        if self.publish_scope not in _VALID_SCOPES:
            raise ValueError(
                f"publish_scope 는 {sorted(_VALID_SCOPES)} 중 하나여야 합니다: {self.publish_scope!r}"
            )
        # 델타 §D1 — arXiv 이용약관 권고 "Direct users to arXiv.org to retrieve
        # e-print content." 원문 링크 없는 아이템은 발행 산출물을 만들 수 없습니다.
        if not self.url:
            raise ValueError(f"원문 링크가 없습니다 (델타 §D1): {self.id}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "channel": self.channel,
            "title": self.title,
            "abstract": self.abstract,
            "url": self.url,
            "published": self.published,
            "updated": self.updated,
            "publish_scope": self.publish_scope,
            "authors": list(self.authors),
            "categories": list(self.categories),
            "fulltext_ok": self.fulltext_ok,
            "raw": self.raw,
        }
