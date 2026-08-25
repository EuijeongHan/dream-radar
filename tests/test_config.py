"""설정 로더 게이트.

`test_profile_resolution` 이 잡는 실제 버그: `data/profile.jobs.yaml`을 경로로
하드코딩하면 이미 존재하는 `profile.jobs_1.yaml`(최신)을 못 읽고 구버전을 읽습니다.
(결정_M0 §4)
"""

from __future__ import annotations

import re

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


def _highest_numbered(stem: str, ext: str = ".yaml") -> str:
    """`resolve_profile` 과 **독립적으로** 최고 번호 파일을 구합니다.

    같은 구현을 두 번 부르면 순환 검증이 됩니다. 여기서는 정규식으로 직접
    번호를 뽑아 비교합니다 — 해석기가 틀리면 이 둘이 갈라집니다.
    """
    best_n, best = -1, None
    for candidate in DATA_DIR.glob(f"{stem}*{ext}"):
        rest = candidate.name[len(stem) : -len(ext)]
        if rest == "":
            n = 0
        elif (m := re.fullmatch(r"_(\d+)", rest)):
            n = int(m.group(1))
        else:
            continue
        if n > best_n:
            best_n, best = n, candidate.name
    return best


def test_real_papers_profile_resolves_to_latest():
    """저장소의 실제 상태 — **파일명을 못박지 않습니다.**

    프로파일이 갱신되면(_1 → _2 → …) 번호가 올라갑니다. 파일명을 하드코딩하면
    갱신할 때마다 이 테스트가 깨지고, 그러면 사람이 숫자만 고쳐 통과시키게 됩니다.
    그건 해석기를 검증하는 게 아니라 통과시키는 것입니다.
    대신 "해석기가 고른 것 == 실제 최고 번호" 를 검사합니다.
    """
    profile, path = load_profile("profile.papers")
    assert path.name == _highest_numbered("profile.papers"), (
        "resolve_profile 이 최고 번호를 고르지 못했습니다"
    )
    # 번호와 무관하게 유지돼야 하는 설정 (결정_M0 §3)
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
