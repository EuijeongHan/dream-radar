"""소스 어댑터 인터페이스.

기획안 §2의 핵심 판단 — 논문과 채용은 **수집 소스와 출력 형식만 다르고 중간 4단계가
동일**합니다. 그 경계가 이 프로토콜입니다.

`publish_scope`를 어댑터가 정하는 게 중요합니다 (델타 §D4). 설정이 아니라 어댑터가
리터럴로 냅니다 — `saramin.py`/`worknet.py`는 `"private"` 외의 값을 낼 수 없어야 하고,
그래서 설정 파일 수정으로는 공고를 공개 싱크에 밀어 넣을 수 없습니다.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.core.models import Item, PublishScope


@runtime_checkable
class SourceAdapter(Protocol):
    #: 원장 `sources` 키에 쓰이는 이름. 예: `"arxiv"`
    name: str
    #: 이 어댑터가 속한 채널. `"papers"` | `"jobs"`
    channel: str
    #: 이 어댑터가 만드는 아이템의 공개 범위. 어댑터가 고정합니다.
    publish_scope: PublishScope

    def collect(self) -> list[Item]:
        """후보를 수집합니다. 중복 제거·랭킹은 하지 않습니다.

        네트워크 제약(간격·커넥션 수)은 **어댑터 안에서** 지킵니다. 호출자가 지키게
        하면 호출 경로가 늘어날 때마다 빠집니다.
        """
        ...


class SourceUnavailable(RuntimeError):
    """어댑터를 지금 쓸 수 없습니다 (키 없음, 보류 결정 등)."""
