"""S3 증분 적재 게이트.

이 테스트가 **무엇을 잡는지**:
  · 두 번 돌렸을 때 두 번째가 네트워크를 타면 실패 (증분 적재가 죽은 것)
  · 매니페스트를 지웠는데 전량 재업로드하면 실패 (복구가 죽은 것)
  · jobs 후보나 private 아이템이 올라가면 실패 (CLAUDE.md §3-3)

`LocalObjectStore` 를 쓰므로 AWS 계정 없이 돕니다.
"""

from __future__ import annotations

import json

import pytest

from src.storage.local import LocalObjectStore
from src.storage.sync import (
    SyncRefused,
    assert_uploadable,
    build_key,
    load_manifest,
    plan,
    sync,
)


def _row(item_id: str, channel: str = "papers", scope: str = "public") -> str:
    return json.dumps(
        {
            "id": item_id,
            "source": "arxiv",
            "channel": channel,
            "title": "t",
            "abstract": "a",
            "publish_scope": scope,
        },
        ensure_ascii=False,
    )


@pytest.fixture
def env(tmp_path):
    cand = tmp_path / "candidates"
    cand.mkdir()
    (cand / "2026-08-18.jsonl").write_text(_row("arxiv:1") + "\n", encoding="utf-8")
    store = LocalObjectStore(tmp_path / "bucket")
    return {
        "cand": cand,
        "store": store,
        "manifest": tmp_path / "s3_manifest.json",
        "prefix": "radar",
    }


def test_키_구조는_날짜_파티션이다():
    assert build_key("papers", "2026-08-18", "radar") == (
        "radar/channel=papers/dt=2026-08-18/candidates.jsonl"
    )


def test_첫_실행은_올리고_두번째는_건너뛴다(env):
    first = sync(env["store"], "papers", env["prefix"], env["cand"], env["manifest"])
    assert first["uploaded"] == 1

    second = sync(env["store"], "papers", env["prefix"], env["cand"], env["manifest"])
    assert second["planned"] == 0, "증분 적재가 죽었습니다 — 변경 없는데 다시 올립니다"
    assert second["uploaded"] == 0


def test_파일이_자라면_다시_올린다(env):
    sync(env["store"], "papers", env["prefix"], env["cand"], env["manifest"])
    path = env["cand"] / "2026-08-18.jsonl"
    with path.open("a", encoding="utf-8") as fp:
        fp.write(_row("arxiv:2") + "\n")

    result = sync(env["store"], "papers", env["prefix"], env["cand"], env["manifest"])
    assert result["uploaded"] == 1
    manifest = load_manifest(env["manifest"])
    key = build_key("papers", "2026-08-18", "radar")
    assert manifest[key]["rows"] == 2


def test_매니페스트를_잃어도_전량_재업로드하지_않는다(env):
    sync(env["store"], "papers", env["prefix"], env["cand"], env["manifest"])
    env["manifest"].unlink()

    result = sync(env["store"], "papers", env["prefix"], env["cand"], env["manifest"])
    assert result["uploaded"] == 0, "복구가 죽었습니다 — 원격에 있는데 다시 올립니다"
    assert result["recovered"] == 1
    assert load_manifest(env["manifest"]), "복구 후 매니페스트가 다시 기록되어야 합니다"


def test_원격이_비면_매니페스트를_잃었을_때_다시_올린다(tmp_path, env):
    sync(env["store"], "papers", env["prefix"], env["cand"], env["manifest"])
    env["manifest"].unlink()
    empty_store = LocalObjectStore(tmp_path / "other-bucket")

    result = sync(empty_store, "papers", env["prefix"], env["cand"], env["manifest"])
    assert result["uploaded"] == 1


def test_jobs_채널은_거부한다(env):
    with pytest.raises(SyncRefused, match="적재 대상이 아닙니다"):
        plan(env["store"], "jobs", env["prefix"], env["cand"], {})


def test_private_아이템이_섞이면_거부한다(env):
    body = (_row("arxiv:1") + "\n" + _row("saramin:9", scope="private") + "\n").encode()
    with pytest.raises(Exception) as exc:
        assert_uploadable(body, "papers")
    assert "channel" in str(exc.value) or "비공개" in str(exc.value)


def test_다른_채널_줄이_섞이면_거부한다():
    body = (_row("arxiv:1") + "\n" + _row("saramin:9", channel="jobs") + "\n").encode()
    with pytest.raises(SyncRefused, match="다른 채널이 섞였습니다"):
        assert_uploadable(body, "papers")


def test_깨진_JSON은_통과하지_못한다():
    body = (_row("arxiv:1") + "\n" + "{not json" + "\n").encode()
    with pytest.raises(SyncRefused, match="JSON이 아닙니다"):
        assert_uploadable(body, "papers")


def test_빈_줄은_건너뛴다():
    body = (_row("arxiv:1") + "\n\n" + _row("arxiv:2") + "\n").encode()
    assert assert_uploadable(body, "papers") == 2


def test_dry_run은_올리지도_기록하지도_않는다(env):
    result = sync(
        env["store"], "papers", env["prefix"], env["cand"], env["manifest"], dry_run=True
    )
    assert result["planned"] == 1
    assert result["uploaded"] == 0
    assert not env["manifest"].exists(), "dry-run이 매니페스트를 건드렸습니다"
    key = build_key("papers", "2026-08-18", "radar")
    assert env["store"].head(key) is None, "dry-run이 실제로 올렸습니다"


def test_후보_디렉터리가_없으면_빈_계획(tmp_path):
    store = LocalObjectStore(tmp_path / "bucket")
    assert plan(store, "papers", "radar", tmp_path / "nope", {}) == []


def test_백엔드가_다르면_다시_올린다(tmp_path, env):
    """★ RADAR_MODE=fake 로 로컬에 적재한 기록이 실제 S3 적재로 오인되면 안 된다.

    이 테스트가 없으면: fake 로 한 번 돌린 머신에서 진짜 버킷으로 바꿨을 때
    매니페스트가 "이미 올렸다"고 답해 **버킷이 영원히 비어 있게** 된다.
    """
    sync(env["store"], "papers", env["prefix"], env["cand"], env["manifest"])
    other = LocalObjectStore(tmp_path / "real-bucket")

    result = sync(other, "papers", env["prefix"], env["cand"], env["manifest"])
    assert result["uploaded"] == 1, "백엔드가 바뀌었는데 건너뛰었습니다"
