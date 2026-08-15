"""키·네트워크 없이 파이프라인 배선을 돌리기 위한 가짜 소스 (델타 §D6.2).

`RADAR_MODE=fake`일 때만 쓰입니다. 랭킹 품질 평가에는 쓰지 마세요 — 내용이 없습니다.
배선과 게이트 동작 확인 전용입니다.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.core.models import Item


class FakeSourceAdapter:
    name = "fake"
    channel = "papers"
    publish_scope = "public"

    def __init__(self, count: int = 320, channel: str = "papers") -> None:
        self.count = count
        self.channel = channel

    def collect(self) -> list[Item]:
        now = datetime.now(UTC)
        return [
            Item(
                id=f"fake:{i:05d}",
                source=self.name,
                channel=self.channel,
                title=f"Fake paper {i} on retrieval evaluation",
                abstract=(
                    f"Placeholder abstract {i}. Generated offline for wiring tests; "
                    "carries no semantic signal."
                ),
                url=f"https://example.invalid/abs/{i:05d}",
                published=(now - timedelta(minutes=i)).isoformat(),
                updated=(now - timedelta(minutes=i)).isoformat(),
                publish_scope=self.publish_scope,
                categories=("cs.IR",),
            )
            for i in range(self.count)
        ]
