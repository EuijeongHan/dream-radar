"""처리 상태 저장소 (`data/state.db`).

M0 DoD의 "재실행 시 중복 0건"과, M3부터의 **2단계 수명주기**를 담당합니다
(기획안_2 §7.1, 함정 9.7).

    collected   수집됨. 후보 파일에 기록됐고 발행 대기 중
    published   발행 성공. 다시는 후보에 오르지 않음

왜 2단계인가: `collect` 직후를 "끝"으로 마킹하면, 전 구간 실행에서 발행이 실패한 날의
아이템이 영영 후보에 오르지 못합니다 — 하루치가 조용히 사라집니다. Actions 크론은
반드시 실패하는 날이 오므로(v1.0 §11) 수집과 발행을 분리합니다.

★ 의미론 주의 — 두 질문은 다른 집합을 봅니다:
    filter_unseen()   "후보 파일에 새로 쓸 것" = **한 번도 못 본 것** (status 무관).
                      이걸 published 기준으로 바꾸면 미발행분이 매일 후보 파일에
                      다시 쌓여 날짜별 골드셋 풀이 오염됩니다.
    unpublished()     "오늘 랭킹에 올릴 것" = collected ∧ ¬published.
                      과거에 수집됐지만 발행 못 한 아이템이 여기서 되살아납니다.

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
    first_run_id  TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'collected',
    published_ts  TEXT
);
CREATE INDEX IF NOT EXISTS idx_seen_channel ON seen_items (channel);
"""
# idx_seen_status 는 여기 넣지 마세요 — 구 스키마 DB에서는 CREATE TABLE IF NOT EXISTS
# 가 건너뛰어 status 컬럼이 아직 없고, 인덱스 생성이 마이그레이션보다 먼저 실행되면
# "no such column" 으로 죽습니다. _migrate() 가 컬럼을 보장한 뒤 만듭니다.

_VALID_STATUS = ("collected", "published")


class State:
    def __init__(self, path: Path | str = DEFAULT_STATE_PATH) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.executescript(_SCHEMA)
            self._migrate(conn)
            conn.commit()

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        """구 스키마(M0, status 없음) DB를 제자리 승격합니다.

        기존 행은 전부 'collected'가 됩니다 — 정확합니다. M0~M1 시점에는 아무것도
        발행된 적이 없습니다.
        """
        cols = {row[1] for row in conn.execute("PRAGMA table_info(seen_items)")}
        if "status" not in cols:
            conn.execute(
                "ALTER TABLE seen_items ADD COLUMN status TEXT NOT NULL DEFAULT 'collected'"
            )
            conn.execute("ALTER TABLE seen_items ADD COLUMN published_ts TEXT")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_seen_status ON seen_items (channel, status)"
        )

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

    # ── 2단계 수명주기 (기획안_2 §7.1) ──────────────────────────────────

    def mark_published(self, item_ids: Sequence[str], ts: str) -> int:
        """발행 성공을 기록합니다. **발행이 실제로 끝난 뒤에만** 부르세요.

        멱등입니다 — 같은 id를 두 번 승격해도 안전하고, 이미 published 면
        published_ts 를 덮지 않습니다 (첫 발행 시각이 진실입니다).
        """
        if not item_ids:
            return 0
        with closing(self._connect()) as conn:
            cursor = conn.executemany(
                "UPDATE seen_items SET status = 'published', published_ts = ? "
                "WHERE item_id = ? AND status != 'published'",
                [(ts, item_id) for item_id in item_ids],
            )
            conn.commit()
            return cursor.rowcount

    def unpublished(self, channel: str) -> list[str]:
        """랭킹 후보 풀 — 수집됐지만 아직 발행되지 않은 id.

        발행이 실패한 날의 아이템이 다음 실행에서 여기로 되살아납니다.
        이게 함정 9.7의 해법입니다.
        """
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT item_id FROM seen_items "
                "WHERE channel = ? AND status = 'collected' ORDER BY item_id",
                (channel,),
            ).fetchall()
        return [row[0] for row in rows]

    def status_counts(self, channel: str | None = None) -> dict[str, int]:
        """원장·대시보드용 상태 분포."""
        query = "SELECT status, COUNT(*) FROM seen_items"
        params: tuple = ()
        if channel is not None:
            query += " WHERE channel = ?"
            params = (channel,)
        query += " GROUP BY status"
        with closing(self._connect()) as conn:
            return dict(conn.execute(query, params).fetchall())
