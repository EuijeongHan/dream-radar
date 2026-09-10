"""골드셋 라벨링 풀 — 다중 시스템 풀링 (docs/05_골드셋_풀링.md).

    python -m eval.pool                                  # 모든 날짜 · 기본 4조건 · 깊이 30 · 무작위 50
    python -m eval.pool --date 2026-08-12
    python -m eval.pool --conditions bge_m3_wmean        # 새 조건을 기존 풀에 덧붙이기

하루 후보 700~1,100건을 전부 라벨링하는 대신, **비교할 랭킹 조건 전부의 상위 k 합집합**
과 **풀 밖에서 뽑은 무작위 표본**만 라벨링합니다. TREC 이 수십 년 써 온 방식입니다.

§9.11 과 무엇이 다른가
---------------------
§9.11 이 금지한 것은 **단일 랭커**의 상위 N 만 라벨링하는 것입니다. 평가 대상 하나가
정답셋의 범위를 정하면 그 랭커가 놓친 논문은 정답이 될 기회가 없습니다.

1. 비교하는 조건 **전부**가 같은 깊이로 풀에 기여합니다 — 한 조건이 범위를 정하지 않습니다
2. 풀 깊이(30)가 지표 깊이(Hit@5 · nDCG@10)보다 깊어서, 기여한 조건의 **상위 10 은
   전부 판정됩니다.** 그래서 조건 간 비교는 편향되지 않습니다
3. 풀 밖 무작위 표본으로 **풀이 놓친 정답의 비율을 추정**합니다 — 한계를 숫자로 남깁니다

한계 — 숨기지 않습니다
---------------------
- 풀에 기여하지 않은 **새 조건**은 상위 10 에 미판정 항목이 섞여 불리하게 측정됩니다.
  평가 전에 `--conditions` 로 풀에 덧붙이고 새 항목을 라벨링하세요.
  `eval.run_eval` 이 조건별 판정률(judged@10)을 찍어 이걸 드러냅니다.
- MRR 은 첫 정답이 풀 깊이 밖이면 0 으로 셉니다. 그 기여는 1/30 미만입니다.
- README·이력서에 "전수 라벨링" 이라고 쓰면 거짓입니다.
  "N조건 풀링(깊이 30) + 무작위 50" 이라고 씁니다.

풀은 **덧붙이기만** 합니다. 다시 돌리면 새 조건의 상위 k 가 더해지고, 무작위 표본은
처음 뽑은 것을 그대로 둡니다 (날짜 시드 고정). 이미 라벨된 항목은 풀 밖이어도 유효합니다.

이 모듈의 최상위는 표준 라이브러리만 import 합니다. `eval.label` 이 이 모듈을 import
하므로, 여기서 `eval.label`·`eval.run_eval` 을 최상위로 부르면 순환 import 가 됩니다.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

POOLS_DIR = Path("eval/pools")
DEFAULT_DEPTH = 30
DEFAULT_RANDOM_N = 50
#: 지표가 보는 깊이(nDCG@10). 풀 깊이가 이보다 얕으면 기여한 조건의 상위 10 에도
#: 미판정 항목이 생겨 "기여한 조건끼리는 공정하다"는 전제가 무너집니다.
METRIC_DEPTH = 10
RANDOM_SOURCE = "random"
SEED_BASE = "radar-pool"
POOL_VERSION = 1


# ── 파일 ────────────────────────────────────────────────────────────────
# 경로 인자 기본값은 None 이고 호출 시점에 POOLS_DIR 을 읽습니다 (함정 9.1 / R7).


def pool_path(date: str, pools_dir: Path | None = None) -> Path:
    return (pools_dir or POOLS_DIR) / f"{date}.json"


def load_pool(date: str, pools_dir: Path | None = None) -> dict[str, Any] | None:
    path = pool_path(date, pools_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_pool(pool: Mapping[str, Any], pools_dir: Path | None = None) -> Path:
    path = pool_path(str(pool["date"]), pools_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pool, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return path


def pool_member_ids(date: str, pools_dir: Path | None = None) -> frozenset[str] | None:
    """그날 풀의 item_id. **풀이 없으면 None** — 전수 라벨링 모드입니다 (빈 집합과 다릅니다)."""
    pool = load_pool(date, pools_dir)
    if pool is None:
        return None
    return frozenset(pool.get("members") or {})


def ranked_members(pool: Mapping[str, Any]) -> frozenset[str]:
    """랭킹 조건이 기여한 항목 (무작위 표본으로만 들어온 것은 제외)."""
    return frozenset(
        item_id
        for item_id, sources in (pool.get("members") or {}).items()
        if any(source != RANDOM_SOURCE for source in sources)
    )


def random_only_members(pool: Mapping[str, Any]) -> frozenset[str]:
    """무작위 표본으로**만** 들어온 항목 — 풀이 놓친 정답 비율의 추정에 씁니다."""
    return frozenset(
        item_id
        for item_id, sources in (pool.get("members") or {}).items()
        if list(sources) == [RANDOM_SOURCE]
    )


# ── 풀 만들기 ───────────────────────────────────────────────────────────


def build_pool(
    date: str,
    rankings: Mapping[str, Sequence[str]],
    candidate_ids: Sequence[str],
    *,
    depth: int = DEFAULT_DEPTH,
    random_n: int = DEFAULT_RANDOM_N,
    existing: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """풀 1일치를 만듭니다. 순수 함수입니다 — 파일을 읽거나 쓰지 않습니다.

    `rankings` 는 `{조건: 그날의 전체 랭킹(item_id 리스트)}` 입니다.
    `existing` 이 있으면 거기에 덧붙이고, 무작위 표본은 다시 뽑지 않습니다.
    """
    if depth < METRIC_DEPTH:
        raise ValueError(
            f"풀 깊이 {depth} 는 지표 깊이 {METRIC_DEPTH}(nDCG@10)보다 얕습니다. "
            "기여한 조건의 상위 10 에도 미판정 항목이 생겨 풀링의 전제가 무너집니다"
        )
    candidates = list(dict.fromkeys(candidate_ids))
    candidate_set = set(candidates)
    base = existing or {}

    members: dict[str, list[str]] = {
        item_id: list(sources) for item_id, sources in (base.get("members") or {}).items()
    }
    conditions = list(base.get("conditions") or [])

    for condition in sorted(rankings):
        top = list(rankings[condition])[:depth]
        stray = [item_id for item_id in top if item_id not in candidate_set]
        if stray:
            raise ValueError(f"{date} 조건 {condition!r} 의 랭킹에 후보에 없는 항목: {stray[:3]}")
        for item_id in top:
            sources = members.setdefault(item_id, [])
            if condition not in sources:
                sources.append(condition)
        if condition not in conditions:
            conditions.append(condition)

    has_random = any(RANDOM_SOURCE in sources for sources in members.values())
    if not has_random and random_n > 0:
        ranked = {i for i, s in members.items() if any(x != RANDOM_SOURCE for x in s)}
        # sorted — 집합 순서에 기대면 같은 시드로도 표본이 달라집니다 (재현성).
        remainder = sorted(candidate_set - ranked)
        rng = random.Random(f"{SEED_BASE}:{date}")
        for item_id in rng.sample(remainder, min(random_n, len(remainder))):
            members.setdefault(item_id, []).append(RANDOM_SOURCE)

    return {
        "version": POOL_VERSION,
        "date": date,
        "candidates": len(candidates),
        # 보장 깊이 — 여러 번 덧붙였다면 가장 얕은 깊이가 "모든 조건에 대해" 성립합니다.
        "depth": depth if not base else min(depth, int(base.get("depth", depth))),
        "random_n": int(base.get("random_n", random_n)) if has_random else random_n,
        "seed": f"{SEED_BASE}:{date}",
        "conditions": conditions,
        "builds": list(base.get("builds") or []),
        "members": {item_id: members[item_id] for item_id in sorted(members)},
    }


def build_pools(
    items_by_date: Mapping[str, Sequence[Any]],
    rankings: Mapping[str, Mapping[str, Sequence[str]]],
    *,
    depth: int = DEFAULT_DEPTH,
    random_n: int = DEFAULT_RANDOM_N,
    build_meta: Mapping[str, Any] | None = None,
    pools_dir: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """날짜별 풀을 만들어(기존 풀에 덧붙여) 씁니다. `rankings` 는 `{조건: {날짜: [ids]}}`."""
    written: dict[str, dict[str, Any]] = {}
    for date, items in items_by_date.items():
        pool = build_pool(
            date,
            {condition: by_date[date] for condition, by_date in rankings.items()},
            [_item_id(item) for item in items],
            depth=depth,
            random_n=random_n,
            existing=load_pool(date, pools_dir),
        )
        pool["builds"].append(
            {
                "at": datetime.now(UTC).isoformat(timespec="seconds"),
                "conditions": sorted(rankings),
                "depth": depth,
                **dict(build_meta or {}),
            }
        )
        write_pool(pool, pools_dir)
        written[date] = pool
    return written


def _item_id(item: Any) -> str:
    return str(item["id"] if isinstance(item, Mapping) else getattr(item, "id"))


def _describe(model: Any) -> str | None:
    if model is None:
        return None
    revision = str(getattr(model, "revision", "") or "")
    return f"{getattr(model, 'model_id', '?')}@{revision[:8]}"


# ── CLI ─────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    # 순환 import 방지 — eval.label 이 이 모듈을 최상위에서 import 합니다.
    from eval.label import available_dates, load_candidates  # noqa: PLC0415
    from eval.run_eval import DEFAULT_CONDITIONS, compute_rankings  # noqa: PLC0415

    parser = argparse.ArgumentParser(prog="python -m eval.pool")
    parser.add_argument("--date", action="append", help="날짜 (반복 가능). 없으면 수집된 전 날짜")
    parser.add_argument("--conditions", nargs="+", default=list(DEFAULT_CONDITIONS))
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    parser.add_argument("--random", dest="random_n", type=int, default=DEFAULT_RANDOM_N)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    dates = args.date or available_dates()
    if not dates:
        raise SystemExit(
            "수집된 후보가 없습니다. python -m src.core.pipeline --channel papers --stage collect"
        )
    items_by_date = {date: list(load_candidates(date)) for date in dates}
    # 평가와 **같은 랭킹 코드**로 풀을 만듭니다 — 따로 구현하면 둘이 갈라집니다.
    run = compute_rankings(tuple(args.conditions), items_by_date, device=args.device)

    pools = build_pools(
        items_by_date,
        run.rankings,
        depth=args.depth,
        random_n=args.random_n,
        build_meta={
            "profile": str(run.profile_path),
            "ko_profile": str(run.ko_path) if run.ko_path else None,
            "embedder": _describe(run.embedder),
            "reranker": _describe(run.reranker),
            "device": run.device,
        },
    )
    for date, pool in pools.items():
        members = pool["members"]
        print(
            f"{date}: 후보 {pool['candidates']}건 → 풀 {len(members)}건 "
            f"(랭킹 {len(ranked_members(pool))} + 무작위 {len(random_only_members(pool))})"
        )
        for condition in pool["conditions"]:
            alone = sum(1 for sources in members.values() if sources == [condition])
            print(f"   {condition:<16} 단독 기여 {alone}건")
    return 0


if __name__ == "__main__":
    sys.exit(main())
