"""실행 원장 `data/runs.jsonl` (AdNova 패턴 계승, 기획안 §7).

한 줄 = 한 실행. **기록된 줄은 절대 고치지 않습니다.**

schema 버전에 대하여 (결정_M0 §1)
--------------------------------
`schema`는 **1**입니다. 2로 올리지 않습니다. 기획안 §7의 "스키마 변경 시 버전을
올린다"는 이미 기록된 줄이 있을 때 적용되는데, 이 파일을 쓰는 시점에 `runs.jsonl`은
존재하지 않았습니다. 즉 아래 형태가 schema 1의 원형입니다. 2로 시작했다면 세상에
없는 schema 1을 영원히 처리하는 리더 코드를 짜게 됩니다.

기획안 §7 대비 추가된 키는 3개이고, **삭제·개명한 키는 없습니다.**

  stage    실행 구간. `summaries: []`가 "수집만 했다"인지 "전 구간 돌았는데 0건
           선정"인지 구분하기 위함입니다. 이 둘을 섞으면 기획안 §12의 가동률
           수치가 거짓이 됩니다.
  profile  실제로 로드한 프로파일 경로. 파일 번호 규칙(결정_M0 §4) 때문에 경로가
           실행마다 달라질 수 있어, 남기지 않으면 기획안 §9 평가가 재현 불가입니다.
  params   그 실행에 실제로 적용된 수집 파라미터. 연휴로 48시간 창이 미달해
           72시간으로 넓히면 그 사실이 여기 남습니다 (결정_M0 §6).

미실행 단계는 **키 삭제가 아니라 `null`**입니다. 리더가 `row["selected"]`에서
KeyError를 내지 않습니다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

SCHEMA_VERSION = 1
DEFAULT_LEDGER_PATH = Path("data/runs.jsonl")

Stage = Literal["collect", "rank", "summarize", "publish", "full"]

#: 허용값을 지금 고정합니다. 안 하면 `collect_only`, `COLLECT` 변종이 섞입니다.
STAGES: tuple[str, ...] = ("collect", "rank", "summarize", "publish", "full")


@dataclass
class RunRecord:
    run_id: str
    channel: str
    stage: str
    profile: str
    sources: dict[str, int]
    collected: int
    after_dedup: int
    duration_sec: float
    params: dict[str, Any] = field(default_factory=dict)
    stage1_top_n: int | None = None
    selected: int | None = None
    summaries: list[dict[str, Any]] | None = None
    gates: dict[str, Any] = field(default_factory=lambda: {"passed": True, "failures": []})
    cost_usd: float = 0.0

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise ValueError(f"stage 는 {list(STAGES)} 중 하나여야 합니다: {self.stage!r}")

    def to_dict(self) -> dict[str, Any]:
        # 기획안 §7의 키 순서를 그대로 유지합니다. 사람이 원장을 눈으로 읽습니다.
        return {
            "schema": SCHEMA_VERSION,
            "run_id": self.run_id,
            "channel": self.channel,
            "stage": self.stage,
            "profile": self.profile,
            "params": self.params,
            "sources": self.sources,
            "collected": self.collected,
            "after_dedup": self.after_dedup,
            "stage1_top_n": self.stage1_top_n,
            "selected": self.selected,
            "summaries": self.summaries,
            "gates": self.gates,
            "duration_sec": round(self.duration_sec, 3),
            "cost_usd": self.cost_usd,
        }


def append(record: RunRecord, path: Path | str = DEFAULT_LEDGER_PATH) -> Path:
    """원장에 한 줄 덧붙입니다. 기존 줄은 건드리지 않습니다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
    return path


def read_all(path: Path | str = DEFAULT_LEDGER_PATH) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]
