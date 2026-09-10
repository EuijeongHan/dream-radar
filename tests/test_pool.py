"""라벨링 풀 게이트 (docs/05_골드셋_풀링.md).

★ test_pool_depth_must_cover_metric_depth — 풀 깊이가 지표 깊이(10)보다 얕으면 기여한
  조건의 상위 10 에도 미판정 항목이 생겨 풀링의 전제가 무너집니다.
★ test_pending_outside_pool_still_reaches_review — 풀 밖으로 밀려난 보류가 2패스에 안
  나오면 영원히 보류로 남아 export 가 막힙니다.
★ test_results_warn_when_a_condition_was_not_pooled — 풀에 기여하지 않은 조건의 수치가
  경고 없이 표에 실리면, 불리하게 측정된 값이 공정한 비교처럼 인용됩니다.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import eval.pool as P
from eval import label as L
from eval import label_web as W
from eval import run_eval as R
from tests.test_labeling import feed, workspace, write_candidates  # noqa: F401 — 픽스처 재사용

DATE = "2026-08-12"


def ids(n: int, prefix: str = DATE) -> list[str]:
    return [f"arxiv:{prefix}.{i:04d}" for i in range(n)]


def random_only(pool) -> set[str]:
    return {i for i, s in pool["members"].items() if s == [P.RANDOM_SOURCE]}


# ── 풀 만들기 (순수 함수) ────────────────────────────────────────────────


def test_pool_is_union_of_top_k_with_provenance():
    cands = ids(40)
    pool = P.build_pool(DATE, {"a": cands[0:12], "b": cands[5:17]}, cands, depth=10, random_n=0)
    members = pool["members"]
    assert set(members) == set(cands[0:15])
    assert members[cands[7]] == ["a", "b"], "두 조건이 겹친 항목의 출처가 둘 다 남아야 합니다"
    assert members[cands[0]] == ["a"]
    assert members[cands[14]] == ["b"]
    assert pool["conditions"] == ["a", "b"]


def test_random_sample_is_outside_ranked_pool_and_deterministic():
    cands = ids(100)
    first = P.build_pool(DATE, {"a": cands[:10]}, cands, depth=10, random_n=20)
    second = P.build_pool(DATE, {"a": cands[:10]}, cands, depth=10, random_n=20)
    assert len(random_only(first)) == 20
    assert random_only(first) == random_only(second), "같은 날짜인데 표본이 달라졌습니다"
    assert random_only(first).isdisjoint(cands[:10])


def test_random_sample_is_capped_by_remainder():
    cands = ids(15)
    pool = P.build_pool(DATE, {"a": cands[:10]}, cands, depth=10, random_n=50)
    assert len(random_only(pool)) == 5


def test_pool_depth_must_cover_metric_depth():
    """깨뜨리는 법: build_pool 의 METRIC_DEPTH 검사를 지우면 빨간불."""
    with pytest.raises(ValueError, match="10"):
        P.build_pool(DATE, {"a": ids(20)}, ids(20), depth=5, random_n=0)


def test_pool_rejects_ids_outside_candidates():
    with pytest.raises(ValueError, match="후보"):
        P.build_pool(DATE, {"a": ["arxiv:nope"] + ids(20)}, ids(20), depth=10, random_n=0)


def test_rebuild_is_additive_and_keeps_random_sample():
    """새 조건을 덧붙여도 기존 멤버와 무작위 표본은 그대로여야 합니다."""
    cands = ids(100)
    first = P.build_pool(DATE, {"a": cands[:10]}, cands, depth=10, random_n=20)
    second = P.build_pool(DATE, {"b": cands[50:60]}, cands, depth=10, random_n=20, existing=first)
    assert set(first["members"]) <= set(second["members"])
    before = {i for i, s in first["members"].items() if P.RANDOM_SOURCE in s}
    after = {i for i, s in second["members"].items() if P.RANDOM_SOURCE in s}
    assert before == after, "덧붙이기가 무작위 표본을 다시 뽑았습니다"
    assert second["conditions"] == ["a", "b"]


def test_pool_roundtrip(tmp_path):
    pool = P.build_pool(DATE, {"a": ids(20)}, ids(20), depth=10, random_n=0)
    P.write_pool(pool, tmp_path)
    assert P.load_pool(DATE, tmp_path)["members"] == pool["members"]
    assert P.pool_member_ids(DATE, tmp_path) == frozenset(pool["members"])
    assert P.pool_member_ids("2099-01-01", tmp_path) is None, "풀이 없으면 None(전수 모드)이어야 합니다"


def test_build_pools_writes_every_date_and_appends_history(tmp_path):
    items = {DATE: [{"id": i} for i in ids(40)], "2026-08-18": [{"id": i} for i in ids(40, "b")]}
    first_rank = {"a": {d: [i["id"] for i in v][:15] for d, v in items.items()}}
    P.build_pools(items, first_rank, depth=10, random_n=5, pools_dir=tmp_path, build_meta={"profile": "p"})
    second_rank = {"b": {d: [i["id"] for i in v][20:35] for d, v in items.items()}}
    P.build_pools(items, second_rank, depth=10, random_n=5, pools_dir=tmp_path)
    pool = P.load_pool(DATE, tmp_path)
    assert pool["conditions"] == ["a", "b"]
    assert len(pool["builds"]) == 2 and pool["builds"][0]["profile"] == "p"
    assert P.load_pool("2026-08-18", tmp_path) is not None


# ── 라벨링 도구 연동 ────────────────────────────────────────────────────


def _pool_for(candidate_ids: list[str], rank_n: int, random_n: int = 0):
    pool = P.build_pool(DATE, {"a": candidate_ids[:rank_n]}, candidate_ids, depth=rank_n, random_n=random_n)
    P.write_pool(pool)  # conftest 가 POOLS_DIR 을 임시 디렉터리로 돌려둡니다
    return pool


def test_work_items_without_pool_is_every_candidate(workspace):
    write_candidates(workspace, DATE, 30)
    assert len(L.work_items(DATE)) == 30


def test_work_items_with_pool_keeps_shuffle_order(workspace):
    """터미널과 모바일이 같은 순서여야 섞어 써도 이어집니다."""
    write_candidates(workspace, DATE, 30)
    shuffled = [i["id"] for i in L.load_candidates(DATE)]
    _pool_for(sorted(shuffled), 10)
    work = [i["id"] for i in L.work_items(DATE)]
    assert len(work) == 10
    assert work == [i for i in shuffled if i in set(work)], "풀 필터가 셔플 순서를 바꿨습니다"


def test_triage_only_walks_the_pool(workspace, monkeypatch):
    """깨뜨리는 법: run_triage 가 work_items 대신 load_candidates 를 쓰면 20건이 찍혀 빨간불.

    키를 풀 크기보다 많이 넣어야 합니다 — 딱 10개면 후보 전체를 돌아도 10건에서 멈춥니다.
    """
    write_candidates(workspace, DATE, 30)
    _pool_for(sorted(i["id"] for i in L.load_candidates(DATE)), 10)
    feed(monkeypatch, "n" * 20)
    L.run_triage(DATE, workspace["journal"])
    assert len(L.latest_by_item(L.load_journal(workspace["journal"]))) == 10


def test_labels_outside_pool_are_kept(workspace, monkeypatch):
    """풀을 만들기 전에 찍은 라벨은 풀 밖이어도 유효하고, 작업 목록에서 사라지면 안 됩니다."""
    write_candidates(workspace, DATE, 30)
    feed(monkeypatch, "nnq")
    L.run_triage(DATE, workspace["journal"])
    before = set(L.latest_by_item(L.load_journal(workspace["journal"])))
    rest = sorted({i["id"] for i in L.load_candidates(DATE)} - before)
    _pool_for(rest, 10)
    assert before <= {i["id"] for i in L.work_items(DATE)}


def test_pending_outside_pool_still_reaches_review(workspace):
    """★ 깨뜨리는 법: work_items 에서 `| decided` 를 빼면 빨간불."""
    write_candidates(workspace, DATE, 30)
    all_ids = sorted(i["id"] for i in L.load_candidates(DATE))
    outsider = all_ids[-1]
    L.append_label(L.Label(outsider, DATE, "t", None, "title"))
    _pool_for(all_ids[:10], 10)
    for item_id in all_ids[:10]:
        L.append_label(L.Label(item_id, DATE, "t", False, "title"))
    nxt = W.api_next(DATE)
    assert nxt["phase"] == "review" and nxt["item"]["id"] == outsider


def test_web_status_reports_pool(workspace):
    write_candidates(workspace, DATE, 30)
    _pool_for(sorted(i["id"] for i in L.load_candidates(DATE)), 10)
    day = W.api_status()["days"][0]
    assert day["pooled"] is True and day["total"] == 10 and day["candidates"] == 30


def test_export_records_pooling_and_missed_estimate(workspace):
    write_candidates(workspace, DATE, 60)
    all_ids = sorted(i["id"] for i in L.load_candidates(DATE))
    pool = P.build_pool(DATE, {"a": all_ids[:10]}, all_ids, depth=10, random_n=10)
    P.write_pool(pool)
    first_random = sorted(random_only(pool))[0]
    for item_id in pool["members"]:
        L.append_label(L.Label(item_id, DATE, "t", item_id == first_random, "abstract"))
    out = L.run_export(workspace["journal"], workspace["goldset"])
    info = yaml.safe_load(out.read_text(encoding="utf-8"))["summary"]["pooling"][DATE]
    assert info["depth"] == 10 and info["pool_size"] == 20 and info["candidates"] == 60
    assert info["random_only_labeled"] == 10 and info["random_only_relevant"] == 1
    # 무작위 표본 1/10 이 관련 → 풀 밖(랭킹 기여 제외) 50건 중 약 5건을 놓쳤다고 추정
    assert info["estimated_missed"] == pytest.approx(5.0)


def test_export_points_to_latest_criteria_doc():
    """파일 번호 규칙 — 기준 문서가 _1, _2 로 늘어도 골드셋이 최신을 가리켜야 합니다."""
    highest = max(
        (0 if p.stem == "라벨링_기준" else int(p.stem.rsplit("_", 1)[1]), p.name)
        for p in Path("docs").glob("라벨링_기준*.md")
    )[1]
    assert L._criteria_doc() == f"docs/{highest}"


# ── 평가 연동 — 판정률(judged@10) ───────────────────────────────────────


def test_judged_at_10_is_reported_per_date():
    gold = {DATE: {"x1"}}
    judged = {DATE: {f"x{i}" for i in range(1, 11)}}
    full = R.evaluate_condition("a", {DATE: [f"x{i}" for i in range(1, 11)]}, gold, {DATE: 100}, judged=judged)
    part = R.evaluate_condition(
        "b", {DATE: ["y1", "x1"] + [f"y{i}" for i in range(2, 10)]}, gold, {DATE: 100}, judged=judged
    )
    assert full.per_date[0].judged_at_10 == 1.0
    assert part.per_date[0].judged_at_10 == pytest.approx(0.1)


def test_judged_at_10_is_none_without_judged_mapping():
    result = R.evaluate_condition("a", {DATE: ["x1"]}, {DATE: {"x1"}}, {DATE: 1})
    assert result.per_date[0].judged_at_10 is None


def _meta() -> dict:
    return {
        "device": "cpu", "models": {}, "profile_path": "p", "ko_profile_path": None,
        "composition": "c", "penalty_weight": 0.5, "top_n": 30, "min_score": 0.42,
        "embedder": "E", "reranker": "미사용", "embedder_revision": None,
        "stub_models": [], "max_seq_length": None,
    }


def test_results_warn_when_a_condition_was_not_pooled(tmp_path):
    """★ 깨뜨리는 법: render_results 의 판정률 경고 블록을 지우면 빨간불."""
    gold = {DATE: frozenset({"x1"})}
    judged = {DATE: frozenset(f"x{i}" for i in range(1, 11))}
    pooled = R.evaluate_condition("bge_m3", {DATE: [f"x{i}" for i in range(1, 11)]}, gold, {DATE: 10}, judged=judged)
    unpooled = R.evaluate_condition(
        "bge_m3_wmean", {DATE: ["x1"] + [f"y{i}" for i in range(9)]}, gold, {DATE: 10}, judged=judged
    )
    goldset = R.Goldset(
        path=tmp_path / "g.yaml", dates=(DATE,), gold=gold, labeled={DATE: 10}, generated_at="t",
        recheck_sampled=0, recheck_flipped=0, miss_rate=None, judged=judged, pooling=None,
    )
    text = R.render_results([pooled, unpooled], goldset, _meta())
    assert "judged@10" in text
    warning = text.split("판정률 1.0 미만", 1)[1]
    assert "bge_m3_wmean" in warning and "`bge_m3`" not in warning.split("\n", 1)[0]
