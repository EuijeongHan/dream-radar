"""파이프라인 배선 + M0 DoD 게이트.

네트워크를 쓰지 않습니다. `RADAR_MODE=fake` 경로로 전 구간을 돌립니다 (델타 §D6.2).
실제 arXiv 수집으로 DoD를 판정하는 테스트는 `-m network` 로 분리돼 있습니다 —
로컬 테스트가 크론과 겹치면 델타 §D2의 합산 제약을 위반하기 때문입니다.
"""

from __future__ import annotations

import pytest

from src.core import ledger
from src.core.pipeline import run_collect
from src.core.state import State


@pytest.fixture
def fake_mode(monkeypatch):
    monkeypatch.setenv("RADAR_MODE", "fake")


@pytest.fixture
def paths(tmp_path):
    return {
        "state_path": tmp_path / "state.db",
        "ledger_path": tmp_path / "runs.jsonl",
        "candidates_dir": tmp_path / "candidates",
    }


def test_collect_nonzero(fake_mode, paths):
    """수집 0건이면 실패 (기획안 §10)."""
    result = run_collect("papers", **paths)
    assert result["collected"] > 0
    assert result["gates"]["passed"], result["gates"]["failures"]


def test_collect_writes_one_ledger_line(fake_mode, paths):
    """M0 DoD — runs.jsonl 에 1줄 기록."""
    run_collect("papers", **paths)
    rows = ledger.read_all(paths["ledger_path"])
    assert len(rows) == 1
    assert rows[0]["stage"] == "collect"
    assert rows[0]["schema"] == 1
    assert rows[0]["summaries"] is None


def test_rerun_yields_zero_duplicates(fake_mode, paths):
    """★ M0 DoD — 재실행 시 중복 0건.

    두 번째 실행은 `collected`가 그대로여도 `after_dedup`이 0이어야 합니다.
    후보 파일에도 줄이 늘지 않습니다.
    """
    first = run_collect("papers", **paths)
    assert first["after_dedup"] == first["collected"]

    candidates = paths["candidates_dir"].glob("*.jsonl")
    lines_after_first = sum(len(p.read_text(encoding="utf-8").splitlines()) for p in candidates)

    second = run_collect("papers", **paths)
    assert second["after_dedup"] == 0, "재실행에서 신규 아이템이 나왔습니다"
    assert second["duplicates"] == second["collected"]

    candidates = paths["candidates_dir"].glob("*.jsonl")
    lines_after_second = sum(len(p.read_text(encoding="utf-8").splitlines()) for p in candidates)
    assert lines_after_second == lines_after_first, "재실행이 후보 파일에 줄을 더 썼습니다"


def test_no_duplicates_within_single_run(fake_mode, paths, monkeypatch):
    """같은 실행 안에서 소스가 같은 id를 두 번 줘도 접혀야 합니다."""
    from src.sources import fake as fake_module

    original = fake_module.FakeSourceAdapter.collect

    def doubled(self):
        items = original(self)
        return items + items

    monkeypatch.setattr(fake_module.FakeSourceAdapter, "collect", doubled)

    result = run_collect("papers", **paths)
    assert result["after_dedup"] * 2 == result["collected"]
    assert result["gates"]["passed"], result["gates"]["failures"]


def test_min_collected_gate_fails_loudly(fake_mode, paths):
    """DoD 미달은 조용히 넘어가지 않습니다 (기획안 §10)."""
    result = run_collect("papers", min_collected=10_000, **paths)
    assert not result["gates"]["passed"]
    assert any("min_collected" in f for f in result["gates"]["failures"])
    # 실패해도 원장에는 남습니다. 실패를 지우면 가동률이 거짓이 됩니다.
    rows = ledger.read_all(paths["ledger_path"])
    assert rows[0]["gates"]["passed"] is False


def test_ledger_records_window_and_mode(fake_mode, paths):
    """연휴로 창을 72시간으로 넓히면 그 사실이 원장에 남아야 합니다 (결정_M0 §6)."""
    run_collect("papers", window_hours=72, **paths)
    row = ledger.read_all(paths["ledger_path"])[0]
    assert row["params"]["window_hours"] == 72
    assert row["params"]["mode"] == "fake"


def test_state_survives_across_instances(tmp_path):
    from src.core.models import Item

    item = Item(
        id="arxiv:2608.00001",
        source="arxiv",
        channel="papers",
        title="t",
        abstract="a",
        url="https://arxiv.org/abs/2608.00001",
        published="2026-08-12T00:00:00Z",
        updated="2026-08-12T00:00:00Z",
        publish_scope="public",
    )
    state = State(tmp_path / "state.db")
    unseen, dupes = state.filter_unseen([item])
    assert len(unseen) == 1 and dupes == 0
    state.mark_seen(unseen, "run/1", "2026-08-12T00:00:00+09:00")

    reopened = State(tmp_path / "state.db")
    unseen, dupes = reopened.filter_unseen([item])
    assert unseen == [] and dupes == 1


def test_jobs_channel_is_not_m0_scope(paths, monkeypatch):
    monkeypatch.delenv("RADAR_MODE", raising=False)
    with pytest.raises(NotImplementedError, match="M5"):
        run_collect("jobs", **paths)


# ── 실제 arXiv 수집 (기본 실행에서 제외) ──────────────────────────────────


@pytest.mark.network
def test_arxiv_live_collect_meets_dod(paths):
    """M0 DoD 판정 — 실제 arXiv에서 300건 이상.

    `pytest -m network` 로만 돕니다. 실행 전 Actions 스케줄이 꺼져 있는지 확인하세요.
    겹치면 arXiv 약관 위반이며 IP 차단 대상입니다 (델타 §D2).
    """
    result = run_collect("papers", **paths)
    assert result["collected"] >= 300, result
    assert result["gates"]["passed"], result["gates"]["failures"]
