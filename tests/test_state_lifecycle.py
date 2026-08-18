"""state.db 2단계 수명주기 게이트 (기획안_2 §7.1, 함정 9.7).

★ 가장 중요한 두 가지:
  1. 발행 실패한 아이템이 `unpublished()` 풀에 살아 있는가 — 하루치 유실 방지
  2. `filter_unseen()` 의미론이 **변하지 않았는가** — 이게 published 기준으로 바뀌면
     미발행분이 매일 후보 파일에 다시 쌓여 날짜별 골드셋 풀이 오염됩니다.
     내일 3일차 수집이 이 의미론 위에서 돕니다.
"""

from __future__ import annotations

import sqlite3

from src.core.models import Item
from src.core.state import State


def _item(n: int, channel: str = "papers") -> Item:
    return Item(
        id=f"arxiv:2608.{n:05d}",
        source="arxiv",
        channel=channel,
        title=f"t{n}",
        abstract="a",
        url=f"https://arxiv.org/abs/2608.{n:05d}",
        published="2026-08-18T00:00:00Z",
        updated="2026-08-18T00:00:00Z",
        publish_scope="public",
    )


def test_collect_marks_collected_status(tmp_path):
    state = State(tmp_path / "s.db")
    items = [_item(1), _item(2)]
    state.mark_seen(items, "run/1", "2026-08-18T07:00:00+09:00")
    assert state.status_counts("papers") == {"collected": 2}
    assert state.unpublished("papers") == [items[0].id, items[1].id]


def test_publish_failure_keeps_items_in_pool(tmp_path):
    """★ 함정 9.7 재현 방지 — 발행이 안 된 아이템은 풀에 남는다."""
    state = State(tmp_path / "s.db")
    items = [_item(1), _item(2), _item(3)]
    state.mark_seen(items, "run/1", "ts")

    # 3건 중 2건만 발행 성공 (1건은 발행 단계에서 실패했다고 가정)
    state.mark_published([items[0].id, items[1].id], "2026-08-18T07:05:00+09:00")

    assert state.unpublished("papers") == [items[2].id], "발행 실패분이 풀에서 사라졌습니다"
    assert state.status_counts("papers") == {"collected": 1, "published": 2}


def test_filter_unseen_semantics_unchanged_by_lifecycle(tmp_path):
    """★ 후보 파일 중복 제거는 status 와 무관해야 합니다.

    미발행 아이템이 다음 수집에서 '신규'로 나오면 내일 후보 파일이 오늘 것으로
    오염되고 골드셋의 날짜별 풀 구분이 무너집니다.
    """
    state = State(tmp_path / "s.db")
    item = _item(1)
    state.mark_seen([item], "run/1", "ts")
    # 발행되지 않은 상태로 다음날 같은 아이템이 다시 수집됨
    unseen, duplicates = state.filter_unseen([item])
    assert unseen == [] and duplicates == 1, "미발행 아이템이 후보 파일에 다시 들어갑니다"
    # 발행 후에도 동일
    state.mark_published([item.id], "ts2")
    unseen, duplicates = state.filter_unseen([item])
    assert unseen == [] and duplicates == 1


def test_mark_published_is_idempotent_and_keeps_first_ts(tmp_path):
    state = State(tmp_path / "s.db")
    item = _item(1)
    state.mark_seen([item], "run/1", "ts")
    assert state.mark_published([item.id], "2026-08-18T07:00:00+09:00") == 1
    assert state.mark_published([item.id], "2026-08-19T07:00:00+09:00") == 0  # 이미 발행됨

    with sqlite3.connect(tmp_path / "s.db") as conn:
        ts = conn.execute(
            "SELECT published_ts FROM seen_items WHERE item_id = ?", (item.id,)
        ).fetchone()[0]
    assert ts == "2026-08-18T07:00:00+09:00", "첫 발행 시각이 덮였습니다"


def test_migration_from_m0_schema(tmp_path):
    """M0 시절 DB(=status 컬럼 없음)가 열릴 때 제자리 승격되는지."""
    db = tmp_path / "old.db"
    with sqlite3.connect(db) as conn:
        conn.executescript(
            """
            CREATE TABLE seen_items (
                item_id TEXT PRIMARY KEY, channel TEXT NOT NULL, source TEXT NOT NULL,
                first_seen_ts TEXT NOT NULL, first_run_id TEXT NOT NULL);
            INSERT INTO seen_items VALUES ('arxiv:2608.00001','papers','arxiv','ts','run/0');
            """
        )
    state = State(db)  # 열면서 마이그레이션
    assert state.status_counts("papers") == {"collected": 1}
    assert state.unpublished("papers") == ["arxiv:2608.00001"]
    # 기존 행 보존 + 재수집 시 여전히 중복 처리
    unseen, duplicates = state.filter_unseen([_item(1)])
    assert unseen == [] and duplicates == 1


def test_channels_are_isolated(tmp_path):
    state = State(tmp_path / "s.db")
    state.mark_seen([_item(1, "papers"), _item(2, "jobs")], "run/1", "ts")
    state.mark_published([_item(1, "papers").id], "ts")
    assert state.unpublished("papers") == []
    assert state.unpublished("jobs") == [_item(2, "jobs").id]
