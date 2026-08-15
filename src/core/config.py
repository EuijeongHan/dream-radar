"""설정 로더.

★ 이 저장소의 파일 번호 규칙(CLAUDE.md §2)이 설정 로더를 깨뜨립니다.

규칙은 "파일을 고치면 덮어쓰지 않고 `_N`을 붙여 새로 저장, 읽을 땐 최고 번호"인데
코드가 설정을 **경로로** 로드합니다. `data/profile.jobs.yaml`을 하드코딩하면 이미
존재하는 `profile.jobs_1.yaml`(최신)을 못 읽고 구버전을 읽습니다.

→ 로더가 최고 번호를 해석합니다. (결정_M0 §4)

설정 파일을 번호 규칙에서 제외하는 대안도 있었지만, `profile.jobs*`는 gitignore라
git 이력이 없습니다. **번호가 유일한 이력**이므로 제외하면 안 됩니다.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

DATA_DIR = Path("data")

# 접미사는 `_` + 숫자만 인정합니다. `.ko` 같은 변종이 걸리면 안 됩니다 —
# profile.papers.ko.yaml 은 평가 대조군이라 운영 파이프라인에 로드하면 안 됩니다.
_SUFFIX_RE = re.compile(r"^_(\d+)$")


def resolve_profile(stem: str, ext: str = ".yaml", root: Path | str = DATA_DIR) -> Path:
    """`'profile.jobs'` → `data/profile.jobs_2.yaml` (최고 번호).

    접미사 없는 파일은 **0**으로 취급합니다 — 관측된 현실과 일치합니다
    (`profile.jobs.yaml`이 구버전, `_1`이 최신).

    번호는 문자열이 아니라 정수로 비교합니다. 문자열 정렬이면 `_10 < _9`가 되어
    최신 파일을 건너뜁니다.
    """
    root = Path(root)
    candidates: list[tuple[int, Path]] = []
    for path in root.glob(f"{stem}*{ext}"):
        rest = path.name[len(stem) : -len(ext)]
        if rest == "":
            candidates.append((0, path))
            continue
        matched = _SUFFIX_RE.match(rest)
        if matched:
            candidates.append((int(matched.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"프로파일을 찾을 수 없습니다: {root}/{stem}*{ext}")
    return max(candidates, key=lambda pair: pair[0])[1]


def load_profile(stem: str, root: Path | str = DATA_DIR) -> tuple[dict[str, Any], Path]:
    """최신 프로파일을 읽고 **실제로 읽은 경로를 함께** 돌려줍니다.

    경로를 반환하는 게 핵심입니다. 어느 프로파일로 돈 결과인지 원장에 남기지 않으면
    기획안 §9 평가가 재현 불가가 됩니다. (결정_M0 §4)
    """
    path = resolve_profile(stem, root=root)
    with path.open(encoding="utf-8") as fp:
        data = yaml.safe_load(fp)
    if not isinstance(data, dict):
        raise ValueError(f"프로파일이 매핑이 아닙니다: {path}")
    return data, path


def radar_mode() -> str:
    """`RADAR_MODE=fake` 면 외부 의존 없이 전 구간이 돕니다. (델타 §D6.2)

    기본값이 `fake`가 아니라 `live`인 이유: 기본값을 fake로 두면 운영에서 조용히
    가짜 데이터를 발행할 수 있습니다. 키가 없어 죽는 게 가짜로 성공하는 것보다 낫습니다.
    """
    return os.environ.get("RADAR_MODE", "live").strip().lower()
