"""골드셋 라벨링 도구 게이트 (기획안 §9).

이 도구가 조용히 틀리면 M1 수치가 통째로 거짓이 됩니다. 특히
`test_export_rejects_pending`과 `test_triage_covers_every_candidate` 는
**커버리지 편향**을 막는 게이트입니다 — 일부만 라벨링하고 넘어가면 랭커가
놓친 논문은 애초에 정답이 될 기회가 없습니다.
"""

from __future__ import annotations

import io
import json

import pytest
import yaml

from eval import label as L


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    candidates = tmp_path / "candidates"
    candidates.mkdir()
    monkeypatch.setattr(L, "CANDIDATES_DIR", candidates)
    journal = tmp_path / "labels.jsonl"
    monkeypatch.setattr(L, "JOURNAL_PATH", journal)
    monkeypatch.setattr(L, "GOLDSET_PATH", tmp_path / "goldset.yaml")
    return {"candidates": candidates, "journal": journal, "goldset": tmp_path / "goldset.yaml"}


def write_candidates(workspace, date: str, n: int) -> None:
    path = workspace["candidates"] / f"{date}.jsonl"
    with path.open("w", encoding="utf-8") as fp:
        for i in range(n):
            fp.write(
                json.dumps(
                    {
                        "id": f"arxiv:{date}.{i:04d}",
                        "title": f"Paper {i}",
                        "abstract": f"Abstract for paper {i}.",
                        "url": f"https://arxiv.org/abs/{date}.{i:04d}",
                        "categories": ["cs.IR"],
                        "authors": ["A", "B"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def feed(monkeypatch, keys: str) -> None:
    """비대화형 입력. read_key 는 TTY가 아니면 한 줄에서 첫 글자를 씁니다."""
    monkeypatch.setattr("sys.stdin", io.StringIO("\n".join(keys) + "\n"))


# ── 원장 ────────────────────────────────────────────────────────────────


def test_journal_last_write_wins(workspace):
    path = workspace["journal"]
    L.append_label(L.Label("a", "2026-08-12", "T", None, "title"), path)
    L.append_label(L.Label("a", "2026-08-12", "T", True, "abstract"), path)
    latest = L.latest_by_item(L.load_journal(path))
    assert len(latest) == 1
    assert latest["a"]["relevant"] is True
    assert latest["a"]["basis"] == "abstract"


def test_undo_removes_only_last(workspace):
    path = workspace["journal"]
    L.append_label(L.Label("a", "2026-08-12", "A", False, "title"), path)
    L.append_label(L.Label("b", "2026-08-12", "B", False, "title"), path)
    removed = L.undo_last(path)
    assert removed["item_id"] == "b"
    assert set(L.latest_by_item(L.load_journal(path))) == {"a"}


def test_undo_on_empty_journal_is_safe(workspace):
    assert L.undo_last(workspace["journal"]) is None


# ── 후보 로딩 ───────────────────────────────────────────────────────────


def test_shuffle_is_deterministic_per_date(workspace):
    write_candidates(workspace, "2026-08-12", 50)
    first = [i["id"] for i in L.load_candidates("2026-08-12")]
    second = [i["id"] for i in L.load_candidates("2026-08-12")]
    assert first == second, "중단 후 재개하면 순서가 달라집니다"


def test_shuffle_actually_shuffles(workspace):
    write_candidates(workspace, "2026-08-12", 50)
    shuffled = [i["id"] for i in L.load_candidates("2026-08-12")]
    original = [f"arxiv:2026-08-12.{i:04d}" for i in range(50)]
    assert shuffled != original
    assert sorted(shuffled) == sorted(original), "섞다가 아이템을 잃었습니다"


def test_missing_candidates_file_explains_next_step(workspace):
    with pytest.raises(FileNotFoundError, match="pipeline"):
        L.load_candidates("2026-01-01")


# ── 1패스 ───────────────────────────────────────────────────────────────


def test_triage_covers_every_candidate(workspace, monkeypatch, capsys):
    """★ 전수 커버리지. 일부만 라벨링하면 골드셋에 커버리지 편향이 생깁니다."""
    write_candidates(workspace, "2026-08-12", 12)
    feed(monkeypatch, "n" * 12)
    L.run_triage("2026-08-12", workspace["journal"])

    decided = L.latest_by_item(L.load_journal(workspace["journal"]))
    assert len(decided) == 12
    assert all(r["relevant"] is False and r["basis"] == "title" for r in decided.values())


def test_triage_resumes_where_it_stopped(workspace, monkeypatch):
    write_candidates(workspace, "2026-08-12", 10)
    feed(monkeypatch, "nnnq")
    L.run_triage("2026-08-12", workspace["journal"])
    assert len(L.load_journal(workspace["journal"])) == 3

    feed(monkeypatch, "n" * 7)
    L.run_triage("2026-08-12", workspace["journal"])
    assert len(L.latest_by_item(L.load_journal(workspace["journal"]))) == 10


def test_triage_keep_defers_to_second_pass(workspace, monkeypatch):
    write_candidates(workspace, "2026-08-12", 5)
    feed(monkeypatch, "kknnn")
    L.run_triage("2026-08-12", workspace["journal"])
    decided = L.latest_by_item(L.load_journal(workspace["journal"]))
    pending = [r for r in decided.values() if r["relevant"] is None]
    assert len(pending) == 2


def test_triage_ignores_unknown_keys(workspace, monkeypatch):
    """오타로 라벨이 찍히면 안 됩니다."""
    write_candidates(workspace, "2026-08-12", 3)
    feed(monkeypatch, "xzn" + "nn")
    L.run_triage("2026-08-12", workspace["journal"])
    decided = L.latest_by_item(L.load_journal(workspace["journal"]))
    assert len(decided) == 3


def test_triage_undo(workspace, monkeypatch):
    write_candidates(workspace, "2026-08-12", 4)
    feed(monkeypatch, "nnu" + "nnn")
    L.run_triage("2026-08-12", workspace["journal"])
    assert len(L.latest_by_item(L.load_journal(workspace["journal"]))) == 4


# ── 2패스 ───────────────────────────────────────────────────────────────


def test_review_only_touches_deferred(workspace, monkeypatch):
    write_candidates(workspace, "2026-08-12", 6)
    feed(monkeypatch, "kknnnn")
    L.run_triage("2026-08-12", workspace["journal"])

    feed(monkeypatch, "yn")
    L.run_review("2026-08-12", workspace["journal"])

    decided = L.latest_by_item(L.load_journal(workspace["journal"]))
    assert all(r["relevant"] is not None for r in decided.values())
    assert sum(1 for r in decided.values() if r["relevant"]) == 1
    by_basis = {r["basis"] for r in decided.values() if r["relevant"] is not None}
    assert by_basis == {"title", "abstract"}


def test_review_records_note(workspace, monkeypatch):
    write_candidates(workspace, "2026-08-12", 2)
    feed(monkeypatch, "kn")
    L.run_triage("2026-08-12", workspace["journal"])

    monkeypatch.setattr("sys.stdin", io.StringIO("m\n검색 구조는 일반적이라 관련\ny\n"))
    L.run_review("2026-08-12", workspace["journal"])

    decided = L.latest_by_item(L.load_journal(workspace["journal"]))
    noted = [r for r in decided.values() if r["note"]]
    assert len(noted) == 1
    assert "검색 구조" in noted[0]["note"]


# ── 내보내기 ────────────────────────────────────────────────────────────


def test_export_rejects_pending(workspace, monkeypatch):
    """★ 보류가 남은 채로 골드셋을 만들면 그 논문들은 정답이 될 기회를 잃습니다."""
    write_candidates(workspace, "2026-08-12", 3)
    feed(monkeypatch, "knn")
    L.run_triage("2026-08-12", workspace["journal"])
    with pytest.raises(SystemExit, match="보류"):
        L.run_export(workspace["journal"], workspace["goldset"])


def test_export_shape_matches_spec(workspace, monkeypatch):
    """기획안 §9 — {date, item_id, relevant} 가 반드시 들어갑니다."""
    write_candidates(workspace, "2026-08-12", 4)
    feed(monkeypatch, "knnn")
    L.run_triage("2026-08-12", workspace["journal"])
    feed(monkeypatch, "y")
    L.run_review("2026-08-12", workspace["journal"])

    out = L.run_export(workspace["journal"], workspace["goldset"])
    doc = yaml.safe_load(out.read_text(encoding="utf-8"))

    assert doc["version"] == 1
    assert doc["criteria_doc"] == "docs/라벨링_기준.md"
    assert doc["summary"]["total"] == 4
    assert doc["summary"]["relevant"] == 1
    for row in doc["labels"]:
        assert set(row) >= {"date", "item_id", "relevant", "basis", "title"}
        assert isinstance(row["relevant"], bool)


def test_export_warns_below_three_days(workspace, monkeypatch, capsys):
    write_candidates(workspace, "2026-08-12", 2)
    feed(monkeypatch, "nn")
    L.run_triage("2026-08-12", workspace["journal"])
    L.run_export(workspace["journal"], workspace["goldset"])
    assert "3일치" in capsys.readouterr().out


def test_export_empty_journal_fails(workspace):
    with pytest.raises(SystemExit, match="라벨이 없습니다"):
        L.run_export(workspace["journal"], workspace["goldset"])


# ── 재검토 (1패스 누락률) ────────────────────────────────────────────────


def test_recheck_measures_title_pass_miss_rate(workspace, monkeypatch):
    """★ docs/라벨링_기준.md §6 한계 2를 숫자로 만듭니다.

    1패스에서 제목만 보고 버린 것 중 표본을 초록까지 읽고, 몇 건이 뒤집히는지 잽니다.
    """
    write_candidates(workspace, "2026-08-12", 10)
    feed(monkeypatch, "n" * 10)
    L.run_triage("2026-08-12", workspace["journal"])

    feed(monkeypatch, "ynnn")
    L.run_recheck("2026-08-12", sample=4, journal_path=workspace["journal"])

    decided = L.latest_by_item(L.load_journal(workspace["journal"]))
    rechecked = [r for r in decided.values() if r["basis"] == "recheck"]
    assert len(rechecked) == 4
    assert sum(1 for r in rechecked if r["relevant"]) == 1


def test_recheck_sample_is_deterministic(workspace, monkeypatch):
    write_candidates(workspace, "2026-08-12", 20)
    feed(monkeypatch, "n" * 20)
    L.run_triage("2026-08-12", workspace["journal"])

    feed(monkeypatch, "nnq")
    L.run_recheck("2026-08-12", sample=5, journal_path=workspace["journal"])
    first = [r["item_id"] for r in L.load_journal(workspace["journal"]) if r["basis"] == "recheck"]

    feed(monkeypatch, "nnn")
    L.run_recheck("2026-08-12", sample=5, journal_path=workspace["journal"])
    after = [r["item_id"] for r in L.load_journal(workspace["journal"]) if r["basis"] == "recheck"]
    assert after[: len(first)] == first, "재개했더니 표본이 바뀌었습니다"
    assert len(set(after)) == 5


def test_recheck_flips_label_in_export(workspace, monkeypatch):
    write_candidates(workspace, "2026-08-12", 6)
    feed(monkeypatch, "n" * 6)
    L.run_triage("2026-08-12", workspace["journal"])
    feed(monkeypatch, "ynn")
    L.run_recheck("2026-08-12", sample=3, journal_path=workspace["journal"])

    out = L.run_export(workspace["journal"], workspace["goldset"])
    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert doc["summary"]["relevant"] == 1
    stats = doc["summary"]["title_pass_recheck"]
    assert stats["sampled"] == 3
    assert stats["flipped_to_relevant"] == 1
    assert stats["miss_rate"] == pytest.approx(1 / 3, abs=1e-4)


def test_export_records_when_recheck_was_skipped(workspace, monkeypatch):
    """재검토를 안 한 것도 사실로 남습니다. 안 한 걸 안 했다고 적어야 정직한 골드셋입니다."""
    write_candidates(workspace, "2026-08-12", 3)
    feed(monkeypatch, "nnn")
    L.run_triage("2026-08-12", workspace["journal"])
    out = L.run_export(workspace["journal"], workspace["goldset"])
    stats = yaml.safe_load(out.read_text(encoding="utf-8"))["summary"]["title_pass_recheck"]
    assert stats == {"sampled": 0, "flipped_to_relevant": 0, "miss_rate": None}


def test_recheck_without_triage_is_a_noop(workspace, capsys):
    write_candidates(workspace, "2026-08-12", 3)
    L.run_recheck("2026-08-12", sample=3, journal_path=workspace["journal"])
    assert "triage" in capsys.readouterr().out


def test_load_candidates_dedupes_repeated_lines(workspace):
    """수집이 후보 기록과 state.db 마킹 사이에서 죽으면 같은 줄이 두 번 쓰입니다.

    그 상태로 라벨링하면 같은 논문이 두 번 나오고, 두 판정이 엇갈리면 골드셋이 오염됩니다.
    """
    write_candidates(workspace, "2026-08-12", 5)
    path = workspace["candidates"] / "2026-08-12.jsonl"
    path.write_text(path.read_text(encoding="utf-8") * 2, encoding="utf-8")

    items = L.load_candidates("2026-08-12")
    assert len(items) == 5
    assert len({i["id"] for i in items}) == 5


# ── 1패스 초록 펼치기 (a 키) ─────────────────────────────────────────────


def test_triage_abstract_peek_records_abstract_basis(workspace, monkeypatch):
    """`a`로 초록을 읽고 내린 판정은 basis=abstract 로 기록돼야 합니다.

    basis=title 로 남기면 recheck 의 "제목만 보고 놓쳤는가" 측정이 오염됩니다.
    """
    write_candidates(workspace, "2026-08-12", 3)
    feed(monkeypatch, "ayann")  # a→y (관련) · a→n (무관) · n (제목 기각)
    L.run_triage("2026-08-12", workspace["journal"])

    decided = L.latest_by_item(L.load_journal(workspace["journal"]))
    assert len(decided) == 3
    outcomes = sorted((r["relevant"], r["basis"]) for r in decided.values())
    assert outcomes == [(False, "abstract"), (False, "title"), (True, "abstract")]


def test_triage_abstract_peek_can_still_defer(workspace, monkeypatch):
    """초록을 보고도 애매하면 k — 보류는 basis=title 로 남아 2패스 대상이 됩니다."""
    write_candidates(workspace, "2026-08-12", 2)
    feed(monkeypatch, "akn")
    L.run_triage("2026-08-12", workspace["journal"])
    decided = L.latest_by_item(L.load_journal(workspace["journal"]))
    pending = [r for r in decided.values() if r["relevant"] is None]
    assert len(pending) == 1
    assert pending[0]["basis"] == "title"


def test_triage_abstract_peek_ignores_stray_keys(workspace, monkeypatch):
    """펼친 상태에서 y/n/k/q 외의 키는 무시됩니다."""
    write_candidates(workspace, "2026-08-12", 1)
    feed(monkeypatch, "axzy")  # a → 잘못된 키 2번 → y
    L.run_triage("2026-08-12", workspace["journal"])
    decided = L.latest_by_item(L.load_journal(workspace["journal"]))
    assert list(decided.values())[0]["relevant"] is True


def test_recheck_pool_excludes_abstract_peeked_rejects(workspace, monkeypatch):
    """`a`로 초록까지 읽고 기각한 항목은 재검토 표본에 들어가면 안 됩니다."""
    write_candidates(workspace, "2026-08-12", 4)
    feed(monkeypatch, "an" + "nnn")  # 1건은 초록 보고 기각, 3건은 제목 기각
    L.run_triage("2026-08-12", workspace["journal"])
    feed(monkeypatch, "nnn")
    L.run_recheck("2026-08-12", sample=10, journal_path=workspace["journal"])
    rechecked = [r for r in L.load_journal(workspace["journal"]) if r["basis"] == "recheck"]
    assert len(rechecked) == 3, "초록 기각분이 재검토 표본에 섞였습니다"
