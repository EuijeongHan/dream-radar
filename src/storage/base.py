"""오브젝트 스토어 추상화.

왜 Protocol을 두는가 ★
----------------------
S3를 직접 부르는 코드를 `sync.py`에 박으면 **테스트가 AWS 계정 없이는 못 돕니다.**
이 저장소는 `RADAR_MODE=fake`로 외부 의존 없이 전 구간이 도는 것을 전제로 하고
(델타 §D6.2), pytest 게이트도 키 없이 통과해야 합니다.

그래서 `sync.py`는 이 Protocol만 알고, 구현체는 둘입니다.

    LocalObjectStore   파일시스템. 테스트 · RADAR_MODE=fake
    S3ObjectStore      boto3. 운영

**어기면 조용히 망가지는 것**: 스토어를 직접 호출하면 테스트가 네트워크를 타게 되고,
계정이 없는 머신에서 pytest가 빨간불이 됩니다. 그러면 게이트를 skip 처리하고 싶어지는데
`docs/작업규약.md` §3-T8이 금지하는 바로 그 상황입니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class ObjectMeta:
    """원격에 이미 있는 객체의 메타.

    `sha256`은 **우리가 업로드할 때 붙인 메타데이터**입니다. S3의 ETag는 멀티파트
    업로드에서 MD5가 아니게 되므로 무결성 비교에 쓰면 안 됩니다 — 작은 파일에서는
    맞다가 파일이 커지는 순간 조용히 어긋납니다.
    """

    key: str
    size: int
    sha256: str | None = None


@runtime_checkable
class ObjectStore(Protocol):
    """put / head 둘이면 증분 적재에 충분합니다.

    delete를 넣지 않은 것은 의도입니다. 이 파이프라인은 **누적만** 합니다
    (`data/runs.jsonl`이 append-only인 것과 같은 이유). 지우는 기능이 있으면
    언젠가 재처리 스크립트가 그걸 부릅니다.
    """

    #: 이 스토어의 신원. 매니페스트가 **어느 백엔드에 올렸는지** 기억하기 위함입니다.
    #: 없으면 `RADAR_MODE=fake` 로 로컬에 적재한 기록이 실제 S3 적재로 오인되어,
    #: 진짜 버킷에는 아무것도 올라가지 않은 채 매번 "건너뜀"이 됩니다.
    @property
    def backend_id(self) -> str: ...

    def put(self, key: str, body: bytes, *, sha256: str) -> None: ...

    def head(self, key: str) -> ObjectMeta | None:
        """없으면 **None**. 예외를 던지지 않습니다 — 존재 확인은 정상 흐름입니다."""
        ...
