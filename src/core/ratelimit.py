"""arXiv 호출 제약 강제기.

델타 §D2 — 이건 권장이 아니라 **API 이용약관상 요구사항**이고, 위반 시 IP 차단입니다.

  > "Make no more than one request every three seconds, and limit requests to a
  >  single connection at a time."
  > 이 제약은 "본인 통제 하의 모든 머신에 합산 적용"됩니다.

왜 `time.sleep(3)` 루프로는 부족한가
------------------------------------
슬립 루프는 **한 프로세스 안의 순차 호출만** 지킵니다. 제약은 두 개인데 하나만
막습니다.

  (a) 3초 간격        → 슬립으로 지켜짐
  (b) 동시 커넥션 1개  → 못 지킴. 로컬 테스트와 Actions 크론이 겹치거나
                        프로세스를 두 개 띄우면 그 순간 커넥션이 2개입니다.

→ 파일락으로 바꿉니다. **잠금을 쥔 상태에서만 HTTP 요청을 엽니다.** 순서가 핵심이고,
   그래서 요청 자체가 `slot()` 컨텍스트 안에서만 일어나도록 어댑터를 짭니다.
   프로세스가 몇 개 뜨든 락을 가진 하나만 커넥션을 열 수 있습니다.

한계 (알고도 남기는 것)
----------------------
파일락은 **같은 파일시스템을 보는 프로세스에만** 유효합니다. 로컬 맥과 Actions
러너는 서로 다른 머신이라 이 락으로 못 묶습니다. 그건 코드가 아니라 운영 규칙으로
막습니다 — README 참조. 구조적으로 막을 수 없는 걸 막았다고 쓰지 않습니다.
"""

from __future__ import annotations

import fcntl
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

DEFAULT_LOCK_PATH = Path("data/.arxiv_lock")


class SingleConnectionRateLimiter:
    """최소 간격 + 단일 커넥션을 파일락으로 동시에 강제합니다.

    락 파일 본문에 마지막 요청 **완료** 시각(epoch)을 적습니다. 완료 시각 기준으로
    간격을 재면 시작 시각 기준보다 항상 보수적입니다 — 요청이 2초 걸렸다면 다음
    요청까지 실제로 5초가 벌어집니다. arXiv 쪽에 유리한 방향으로 틀리는 게 맞습니다.
    """

    def __init__(
        self,
        min_interval_sec: float,
        lock_path: Path | str = DEFAULT_LOCK_PATH,
        *,
        sleeper=time.sleep,
        clock=time.monotonic,
    ) -> None:
        if min_interval_sec < 3.0:
            # 설정 파일이 실수로 낮춰지는 걸 코드에서 거부합니다. 델타 §D2는
            # 협상 대상이 아닙니다.
            raise ValueError(
                f"arXiv 최소 간격은 3.0초 미만일 수 없습니다 (델타 §D2): {min_interval_sec}"
            )
        self.min_interval_sec = float(min_interval_sec)
        self.lock_path = Path(lock_path)
        self._sleep = sleeper
        self._clock = clock

    @contextmanager
    def slot(self) -> Iterator[None]:
        """호출 슬롯 하나. **이 블록 안에서만** 네트워크 요청을 여세요.

        블록 진입 = 배타 락 획득 + 직전 요청으로부터 min_interval 경과 보장.
        블록 이탈 = 완료 시각 기록 + 락 해제.
        """
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        # 슬롯마다 새 파일 디스크립터를 엽니다. flock은 open file description에
        # 걸리므로, 같은 프로세스의 스레드 두 개도 이렇게 해야 서로 막힙니다.
        fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)  # 락을 못 잡으면 여기서 대기합니다
            try:
                self._wait_for_interval(fd)
                yield
            finally:
                self._stamp(fd)
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def _wait_for_interval(self, fd: int) -> None:
        last = self._read_stamp(fd)
        if last is None:
            return
        elapsed = self._clock() - last
        if elapsed < 0:
            # 재부팅으로 monotonic이 리셋됐거나 다른 clock으로 기록된 값입니다.
            # 간격을 믿을 수 없으므로 한 주기를 통째로 기다립니다. 모르면 기다리는 쪽입니다.
            self._sleep(self.min_interval_sec)
        elif elapsed < self.min_interval_sec:
            self._sleep(self.min_interval_sec - elapsed)

    def _read_stamp(self, fd: int) -> float | None:
        os.lseek(fd, 0, os.SEEK_SET)
        raw = os.read(fd, 64).decode("utf-8", "replace").strip()
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    def _stamp(self, fd: int) -> None:
        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        os.write(fd, f"{self._clock():.6f}".encode())
