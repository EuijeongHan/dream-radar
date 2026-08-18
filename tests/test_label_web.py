"""모바일 라벨링 API 게이트.

HTTP 계층이 아니라 API 코어 함수를 직접 검사합니다. 핵심은 **터미널 도구와
의미론이 같은가**입니다 — 같은 원장, 같은 basis 규칙, 같은 순서. 어긋나면
폰으로 찍은 라벨과 맥으로 찍은 라벨이 다른 종류의 데이터가 됩니다.
"""

from __future__ import annotations

import pytest

from eval import label as L
from eval import label_web as W
from tests.test_labeling import feed, write_candidates, workspace  # noqa: F401 — 픽스처 재사용


def test_next_follows_terminal_shuffle_order(workspace):
    """폰과 맥에서 같은 순서로 나와야 합니다. 섞어 쓰면 이어져야 하니까요."""
    write_candidates(workspace, "2026-08-12", 10)
    terminal_order = [i["id"] for i in L.load_candidates("2026-08-12")]
    first = W.api_next("2026-08-12")
    assert first["phase"] == "triage"
    assert first["item"]["id"] == terminal_order[0]


def test_submit_records_same_journal_shape(workspace):
    write_candidates(workspace, "2026-08-12", 3)
    item = W.api_next("2026-08-12")["item"]
    W.api_submit({"date": "2026-08-12", "item_id": item["id"], "relevant": False, "basis": "title"})
    rows = L.load_journal()
    assert len(rows) == 1
    assert rows[0]["item_id"] == item["id"]
    assert rows[0]["relevant"] is False
    assert rows[0]["basis"] == "title"


def test_triage_then_review_phase_transition(workspace):
    """트리아지가 끝나면 보류분이 2패스로 이어집니다 — 터미널의 triage→review와 동일."""
    write_candidates(workspace, "2026-08-12", 3)
    for expected_rel in (None, False, False):
        item = W.api_next("2026-08-12")["item"]
        W.api_submit(
            {"date": "2026-08-12", "item_id": item["id"], "relevant": expected_rel, "basis": "title"}
        )
    r = W.api_next("2026-08-12")
    assert r["phase"] == "review"
    W.api_submit({"date": "2026-08-12", "item_id": r["item"]["id"], "relevant": True, "basis": "abstract"})
    assert W.api_next("2026-08-12")["phase"] == "done"

    decided = L.latest_by_item(L.load_journal())
    relevant = [r for r in decided.values() if r["relevant"] is True]
    assert len(relevant) == 1 and relevant[0]["basis"] == "abstract"


def test_defer_is_always_title_basis(workspace):
    """보류는 basis=title 로 남아야 recheck·2패스 의미론이 유지됩니다 (터미널 a→k 동일)."""
    write_candidates(workspace, "2026-08-12", 1)
    item = W.api_next("2026-08-12")["item"]
    W.api_submit({"date": "2026-08-12", "item_id": item["id"], "relevant": None, "basis": "abstract"})
    assert L.load_journal()[0]["basis"] == "title"


def test_submit_rejects_unknown_item_and_bad_values(workspace):
    write_candidates(workspace, "2026-08-12", 1)
    with pytest.raises(ValueError, match="후보에 없는"):
        W.api_submit({"date": "2026-08-12", "item_id": "arxiv:nope", "relevant": False, "basis": "title"})
    item = W.api_next("2026-08-12")["item"]
    with pytest.raises(ValueError, match="basis"):
        W.api_submit({"date": "2026-08-12", "item_id": item["id"], "relevant": False, "basis": "vibes"})
    with pytest.raises(ValueError, match="relevant"):
        W.api_submit({"date": "2026-08-12", "item_id": item["id"], "relevant": "yes", "basis": "title"})


def test_undo_matches_terminal_undo(workspace):
    write_candidates(workspace, "2026-08-12", 2)
    item = W.api_next("2026-08-12")["item"]
    W.api_submit({"date": "2026-08-12", "item_id": item["id"], "relevant": False, "basis": "title"})
    result = W.api_undo()
    assert result["removed"] == item["id"]
    assert L.load_journal() == []
    assert W.api_next("2026-08-12")["item"]["id"] == item["id"]  # 같은 항목이 다시 나옴


def test_mixed_terminal_and_web_labeling(workspace, monkeypatch):
    """맥 터미널로 찍다가 폰으로 넘어가도 한 원장에서 이어져야 합니다."""
    write_candidates(workspace, "2026-08-12", 4)
    feed(monkeypatch, "nnq")
    L.run_triage("2026-08-12", workspace["journal"])

    r = W.api_next("2026-08-12")
    assert r["progress"]["done"] == 2
    W.api_submit({"date": "2026-08-12", "item_id": r["item"]["id"], "relevant": False, "basis": "title"})
    W.api_submit(
        {"date": "2026-08-12", "item_id": W.api_next("2026-08-12")["item"]["id"], "relevant": False, "basis": "title"}
    )
    assert W.api_next("2026-08-12")["phase"] == "done"
    assert len(L.latest_by_item(L.load_journal())) == 4


def test_status_lists_days(workspace):
    write_candidates(workspace, "2026-08-12", 2)
    write_candidates(workspace, "2026-08-13", 3)
    days = {d["date"]: d for d in W.api_status()["days"]}
    assert days["2026-08-12"]["total"] == 2
    assert days["2026-08-13"]["total"] == 3
