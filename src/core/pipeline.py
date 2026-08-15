"""파이프라인 오케스트레이션.

    python -m src.core.pipeline --channel papers --stage collect

M0에서 구현된 구간은 `collect` 하나입니다. 나머지 stage는 M1~M3에서 채웁니다.
구현되지 않은 stage를 요청하면 조용히 넘어가지 않고 죽습니다 — 기획안 §10의
"게이트 실패는 빨간불로 남아야 한다"와 같은 이유입니다.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.core import ledger
from src.core.config import load_profile, radar_mode
from src.core.state import State
from src.sources.arxiv import ArxivAdapter
from src.sources.base import SourceAdapter
from src.sources.fake import FakeSourceAdapter

KST = timezone(timedelta(hours=9))
CANDIDATES_DIR = Path("data/candidates")

#: M0 DoD (기획안 §8). 측정 지점은 **창 적용 후, 중복 제거 전**입니다 (결정_M0 §6).
#: 연휴로 미달하면 실패가 아니라 --window-hours 72 로 넓히고 원장에 남기면 됩니다.
DEFAULT_MIN_COLLECTED = 300

log = logging.getLogger("radar")


def build_sources(channel: str, profile: dict[str, Any], window_hours: float | None) -> list:
    if radar_mode() == "fake":
        log.warning("RADAR_MODE=fake — 가짜 소스로 실행합니다. 결과를 발행하지 마세요.")
        return [FakeSourceAdapter(channel=channel)]

    if channel != "papers":
        raise NotImplementedError(
            f"채널 {channel!r}은 M0 범위가 아닙니다. jobs 채널은 M5입니다 (기획안 §8)."
        )

    sources_config = profile.get("sources", {})
    adapters: list[SourceAdapter] = []

    arxiv_config = sources_config.get("arxiv", {})
    if arxiv_config.get("enabled", False):
        adapters.append(ArxivAdapter(arxiv_config, window_hours=window_hours))

    # hf_daily_papers 는 enabled: false 입니다. 공식 문서화된 API가 아니라서
    # 보류했습니다 (결정_M0 §3). src/sources/hf_papers.py 주석 참조.
    if sources_config.get("hf_daily_papers", {}).get("enabled", False):
        raise NotImplementedError(
            "hf_daily_papers 가 enabled 로 켜져 있지만 구현이 보류 상태입니다. "
            "src/sources/hf_papers.py 를 읽고 이용약관을 먼저 확인하세요."
        )

    if not adapters:
        raise RuntimeError(f"{channel} 채널에 활성화된 소스가 없습니다")
    return adapters


def run_collect(
    channel: str,
    *,
    window_hours: float | None = None,
    min_collected: int = DEFAULT_MIN_COLLECTED,
    state_path: Path | None = None,
    ledger_path: Path | None = None,
    candidates_dir: Path = CANDIDATES_DIR,
) -> dict[str, Any]:
    started = time.monotonic()
    now = datetime.now(KST)
    run_id = f"{now.isoformat(timespec='seconds')}/{channel}"

    profile, profile_path = load_profile(f"profile.{channel}")
    adapters = build_sources(channel, profile, window_hours)

    per_source: dict[str, int] = {}
    collected: list = []
    for adapter in adapters:
        items = adapter.collect()
        per_source[adapter.name] = len(items)
        collected.extend(items)
        log.info("%s: %d건 수집", adapter.name, len(items))

    state = State(state_path) if state_path else State()
    unseen, duplicates = state.filter_unseen(collected)
    log.info("중복 제거: %d건 신규, %d건 기수집", len(unseen), duplicates)

    candidates_path = _write_candidates(unseen, now, candidates_dir)
    state.mark_seen(unseen, run_id, now.isoformat())

    effective_window = window_hours
    if effective_window is None:
        effective_window = float(
            profile.get("sources", {}).get("arxiv", {}).get("window_hours", 48)
        )

    gates = _check_gates(collected, unseen, min_collected)

    record = ledger.RunRecord(
        run_id=run_id,
        channel=channel,
        stage="collect",
        profile=str(profile_path),
        params={
            "window_hours": effective_window,
            "min_collected": min_collected,
            "mode": radar_mode(),
            "pages_fetched": {
                a.name: getattr(a, "pages_fetched", None)
                for a in adapters
                if getattr(a, "pages_fetched", None) is not None
            },
        },
        sources=per_source,
        collected=len(collected),
        after_dedup=len(unseen),
        # 이 실행에서 돌지 않은 단계는 키 삭제가 아니라 null 입니다 (결정_M0 §1).
        stage1_top_n=None,
        selected=None,
        summaries=None,
        gates=gates,
        duration_sec=time.monotonic() - started,
        cost_usd=0.0,
    )
    ledger_file = ledger.append(record, ledger_path or ledger.DEFAULT_LEDGER_PATH)

    return {
        "run_id": run_id,
        "collected": len(collected),
        "after_dedup": len(unseen),
        "duplicates": duplicates,
        "gates": gates,
        "candidates_path": str(candidates_path) if candidates_path else None,
        "ledger_path": str(ledger_file),
        "profile": str(profile_path),
    }


def _write_candidates(items: list, now: datetime, candidates_dir: Path) -> Path | None:
    """신규 아이템만 `data/candidates/{date}.jsonl`에 덧붙입니다.

    재실행하면 신규가 0건이라 줄이 늘지 않습니다. M0 DoD "재실행 시 중복 0건"의
    가시적 증거가 이 파일입니다. (`data/candidates/` 는 gitignore)
    """
    if not items:
        return None
    candidates_dir.mkdir(parents=True, exist_ok=True)
    path = candidates_dir / f"{now.date().isoformat()}.jsonl"
    with path.open("a", encoding="utf-8") as fp:
        for item in items:
            fp.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
    return path


def _check_gates(collected: list, unseen: list, min_collected: int) -> dict[str, Any]:
    failures: list[str] = []

    # test_collect_nonzero — 수집 0건이면 실패 (기획안 §10)
    if not collected:
        failures.append("collect_nonzero: 수집 결과가 0건입니다")

    # M0 DoD — 창 적용 후·중복 제거 전 기준 (결정_M0 §6)
    if len(collected) < min_collected:
        failures.append(
            f"min_collected: {len(collected)}건 < {min_collected}건. "
            "연휴 등으로 창이 비었으면 --window-hours 72 로 넓히고 원장에 남기세요"
        )

    # test_no_duplicates — 결과에 동일 item_id 중복 시 실패 (기획안 §10)
    ids = [item.id for item in unseen]
    if len(ids) != len(set(ids)):
        failures.append("no_duplicates: 신규 아이템에 중복 id가 있습니다")

    return {"passed": not failures, "failures": failures}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.core.pipeline")
    parser.add_argument("--channel", required=True, choices=["papers", "jobs"])
    parser.add_argument("--stage", default="full", choices=list(ledger.STAGES))
    parser.add_argument(
        "--window-hours",
        type=float,
        default=None,
        help="수집 창. 기본값은 프로파일의 window_hours (48). 연휴엔 72로 넓히세요",
    )
    parser.add_argument("--min-collected", type=int, default=DEFAULT_MIN_COLLECTED)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
    )

    if args.stage != "collect":
        parser.error(
            f"stage {args.stage!r}는 아직 구현되지 않았습니다. M0 범위는 'collect' 뿐입니다."
        )

    result = run_collect(
        args.channel,
        window_hours=args.window_hours,
        min_collected=args.min_collected,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if not result["gates"]["passed"]:
        # 기획안 §10 — 게이트 실패는 조용히 넘어가지 않고 빨간불로 남습니다.
        for failure in result["gates"]["failures"]:
            log.error("GATE FAIL — %s", failure)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
