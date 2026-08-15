"""설정 로더 게이트.

`test_profile_resolution` 이 잡는 실제 버그: `data/profile.jobs.yaml`을 경로로
하드코딩하면 이미 존재하는 `profile.jobs_1.yaml`(최신)을 못 읽고 구버전을 읽습니다.
(결정_M0 §4)
"""

from __future__ import annotations

import pytest

from src.core.config import DATA_DIR, load_profile, resolve_profile


def _touch(directory, name: str) -> None:
    (directory / name).write_text("version: 1\n", encoding="utf-8")


def test_profile_resolution_picks_highest_number(tmp_path):
    for name in ("x.yaml", "x_1.yaml", "x_2.yaml"):
        _touch(tmp_path, name)
    assert resolve_profile("x", root=tmp_path).name == "x_2.yaml"


def test_profile_resolution_is_numeric_not_lexical(tmp_path):
    """`_10` > `_9`. 문자열 정렬이면 `_9`가 뽑혀서 최신 파일을 건너뜁니다."""
    for name in ("x.yaml", "x_9.yaml", "x_10.yaml"):
        _touch(tmp_path, name)
    assert resolve_profile("x", root=tmp_path).name == "x_10.yaml"


def test_profile_resolution_ignores_non_numeric_suffix(tmp_path):
    """`profile.papers.ko.yaml`은 평가 대조군이라 운영 조회에 걸리면 안 됩니다."""
    _touch(tmp_path, "x.yaml")
    _touch(tmp_path, "x.ko.yaml")
    _touch(tmp_path, "x_draft.yaml")
    assert resolve_profile("x", root=tmp_path).name == "x.yaml"


def test_profile_resolution_unnumbered_is_oldest(tmp_path):
    """접미사 없는 파일은 0. 관측된 현실(profile.jobs.yaml 이 구버전)과 일치합니다."""
    _touch(tmp_path, "x.yaml")
    _touch(tmp_path, "x_1.yaml")
    assert resolve_profile("x", root=tmp_path).name == "x_1.yaml"


def test_profile_resolution_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_profile("nope", root=tmp_path)


def test_real_papers_profile_resolves_to_latest():
    """저장소의 실제 상태. hf_daily_papers 를 끈 `_1`이 선택돼야 합니다."""
    profile, path = load_profile("profile.papers")
    assert path.name == "profile.papers_1.yaml"
    assert profile["sources"]["hf_daily_papers"]["enabled"] is False
    assert profile["sources"]["arxiv"]["enabled"] is True


def test_real_jobs_profile_resolves_to_latest():
    """구버전 `profile.jobs.yaml`이 아니라 `_1`을 읽는지. 지금 당장 틀릴 수 있던 지점입니다."""
    path = resolve_profile("profile.jobs")
    assert path.name == "profile.jobs_1.yaml"


def test_profile_parity():
    """영/한 프로파일의 id 집합·weight·순서가 같아야 합니다 (델타 §D7·§D8).

    하나라도 다르면 M1에서 재는 게 교차언어 손실이 아니라 프로파일 차이가 됩니다.
    """
    en, _ = load_profile("profile.papers")
    # ko 쪽도 경로를 하드코딩하지 않습니다. 나중에 profile.papers.ko_1.yaml 이 생기면
    # 하드코딩된 경로는 구버전을 읽고, 이 테스트는 통과하는데 실제 대조군은 다른 파일이
    # 되는 최악의 상태가 됩니다. 결정_M0 §4가 지적한 함정이 여기에도 있었습니다.
    ko, ko_path = load_profile("profile.papers.ko")
    assert ko_path.parent == DATA_DIR

    en_ids = [i["id"] for i in en["interests"]]
    ko_ids = [i["id"] for i in ko["interests"]]
    assert en_ids == ko_ids, "id 집합 또는 순서가 다릅니다"

    en_weights = [i["weight"] for i in en["interests"]]
    ko_weights = [i["weight"] for i in ko["interests"]]
    assert en_weights == ko_weights, "weight 가 다릅니다"

    en_penalties = [e["penalty"] for e in en["exclude"]["soft"]]
    ko_penalties = [e["penalty"] for e in ko["exclude"]["soft"]]
    assert en_penalties == ko_penalties, "exclude penalty 가 다릅니다"

    assert en["language"] == "en"
    assert ko["language"] == "ko"
