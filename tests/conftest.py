"""pytest 공용 설정.

`pyproject.toml` 의 `--strict-markers` 때문에 등록되지 않은 마커를 쓰면 **수집 단계에서
에러**입니다. `slow` 를 여기서 등록합니다 — `pyproject.toml` 을 고치지 않고 마커만
추가하기 위한 최소 파일입니다.

`slow` 는 실모델(bge-m3 2.3GB / bge-reranker-v2-m3)을 실제로 로드하는 테스트에 붙입니다.
그런 테스트에는 `network` 도 같이 붙이세요 — 가중치를 HuggingFace 에서 받는 것은
실제 네트워크 접근이고, `network` 는 이미 `addopts = "-m 'not network'"` 로 기본
실행에서 빠집니다. `slow` 만으로는 기본 실행에서 빠지지 않습니다.
"""

from __future__ import annotations

import pytest

from eval import pool as _pool


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers",
        "slow: 실모델 가중치를 로드하는 테스트. 기본 실행에서 빼려면 network 마커를 함께 붙인다",
    )


@pytest.fixture(autouse=True)
def _isolate_label_pools(tmp_path_factory, monkeypatch):
    """라벨링 풀 디렉터리를 테스트마다 빈 곳으로 돌립니다 (함정 9.1).

    `eval/pools/<날짜>.json` 은 저장소에 커밋되는 실제 풀입니다. 격리하지 않으면 라벨링
    도구 테스트가 **진짜 풀**을 읽고 — 테스트 날짜(2026-08-12)가 실제 날짜와 같아서 —
    가짜 후보 30건이 실제 풀의 item_id 로 걸러져 0건이 됩니다.
    """
    monkeypatch.setattr(_pool, "POOLS_DIR", tmp_path_factory.mktemp("pools"))
