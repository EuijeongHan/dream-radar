"""처리 완료 ID 저장소 (`data/state.db`).

M0 DoD의 "재실행 시 중복 0건"을 담당합니다. Actions 단발 실행이라 SQLite로 충분합니다
(기획안 §5).

중복 제거는 **버전 없는 id** 기준입니다. `models.Item` 주석 참조.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from contextlib import closing
from pathlib import Path

from src.core.models import Item

DEFAULT_STATE_PATH = Path("data/state.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS seen_items (
    item_id       TEXT PRIMARY KEY,
    channel       TEXT NOT NULL,
    source        TEXT NOT NULL,
    first_seen_ts TEXT NOT NULL,
    first_run_id  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_seen_channel ON seen_items (channel);
"""


class State:
    def __init__(self, path: Path | str = DEFAULT_STATE_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.executescript(_SCHEMA)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def filter_unseen(self, items: Iterable[Item]) -> tuple[list[Item], int]:
        """(처음 보는 것, 이미 본 건수)를 돌려줍니다.

        같은 실행 안에서 소스가 같은 id를 두 번 준 경우도 중복으로 셉니다 —
        `test_no_duplicates`가 선정 결과의 id 중복을 실패로 보기 때문에 여기서 미리 접습니다.
        """
        items = list(items)
        if not items:
            return [], 0
        with closing(self._connect()) as conn:
            rows = conn.execute(
                f"SELECT item_id FROM seen_items WHERE item_id IN "  # noqa: S608 - 자리표시자만 생성
                f"({','.join('?' * len(items))})",
                [item.id for item in items],
            ).fetchall()
        known = {row[0] for row in rows}

        unseen: list[Item] = []
        duplicates = 0
        for item in items:
            if item.id in known:
                duplicates += 1
                continue
            known.add(item.id)  # 같은 배치 내 중복도 여기서 걸립니다
            unseen.append(item)
        return unseen, duplicates

    def mark_seen(self, items: Sequence[Item], run_id: str, ts: str) -> int:
        if not items:
            return 0
        with closing(self._connect()) as conn:
            cursor = conn.executemany(
                "INSERT OR IGNORE INTO seen_items "
                "(item_id, channel, source, first_seen_ts, first_run_id) VALUES (?, ?, ?, ?, ?)",
                [(i.id, i.channel, i.source, ts, run_id) for i in items],
            )
            conn.commit()
            return cursor.rowcount

    def count(self, channel: str | None = None) -> int:
        with closing(self._connect()) as conn:
            if channel is None:
                return conn.execute("SELECT COUNT(*) FROM seen_items").fetchone()[0]
            return conn.execute(
                "SELECT COUNT(*) FROM seen_items WHERE channel = ?", (channel,)
            ).fetchone()[0]
