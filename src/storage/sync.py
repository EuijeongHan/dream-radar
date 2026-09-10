"""`data/candidates/*.jsonl` → 오브젝트 스토리지 증분 적재.

    python -m src.storage.sync --channel papers
    python -m src.storage.sync --channel papers --dry-run

무엇을 푸는 문제인가
--------------------
`data/candidates/` 는 gitignore입니다. 즉 **이 머신이 죽으면 수집분이 사라집니다.**
`state.db`에 id는 남지만 초록·제목은 후보 파일에만 있습니다. 재수집하면 되지 않느냐면
안 됩니다 — arXiv 창은 48시간이라 과거분은 같은 쿼리로 다시 오지 않습니다.

키 구조 (Hive 스타일 파티션)
---------------------------
    {prefix}/channel=papers/dt=2026-08-18/candidates.jsonl

날짜가 키에 있으므로 **같은 날 다시 올리면 같은 키를 덮어씁니다.** 후보 파일은
append-only라 재실행 시 줄이 늘 수 있고, 그때만 다시 올라갑니다 (해시 비교).

멱등성과 복구 ★
--------------
매니페스트(`data/s3_manifest.json`)에 키별 sha256을 남깁니다. 세 경우가 다릅니다.

    로컬 해시 == 매니페스트 해시              → 건너뜀 (증분 적재)
    로컬 해시 != 매니페스트 해시              → 재업로드 (파일이 자랐음)
    매니페스트에 없음                         → **원격 head 확인 후** 결정 (실패 복구)

마지막 줄이 핵심입니다. 매니페스트를 잃어버려도 원격을 보고 복구하므로, 전량
재업로드가 일어나지 않습니다. 매니페스트만 믿으면 파일 하나 지워진 걸로 매일
수 MB를 다시 올립니다.

유출 게이트 ★
------------
CLAUDE.md §3-3 — 채용공고를 공개 게시 금지. jobs 채널 후보 파일이 섞여 올라가면
버킷 정책이 하루만 잘못돼도 약관 위반입니다. 그래서 **업로드 직전에 줄 단위로**
`gates.assert_public_scope()`를 태웁니다. 파일명이나 인자가 아니라 내용을 봅니다 —
인자는 오타가 나고 내용은 거짓말을 하지 않습니다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from src.core.config import radar_mode
from src.storage.base import ObjectStore
from src.verify.gates import assert_public_scope

CANDIDATES_DIR = Path("data/candidates")
MANIFEST_PATH = Path("data/s3_manifest.json")
FAKE_STORE_DIR = Path("data/_fake_s3")

#: 이 채널만 올립니다. jobs 를 넣으려면 CLAUDE.md §3-3 부터 다시 읽어야 합니다.
ALLOWED_CHANNELS = frozenset({"papers"})

log = logging.getLogger("radar.storage")


class SyncRefused(RuntimeError):
    """게이트 위반. 조용히 건너뛰지 않고 죽습니다."""


@dataclass(frozen=True, slots=True)
class Upload:
    path: Path
    key: str
    sha256: str
    size: int
    reason: str  # "new" | "changed" | "manifest-lost"


def sha256_bytes(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def build_key(channel: str, date: str, prefix: str) -> str:
    return f"{prefix}/channel={channel}/dt={date}/candidates.jsonl"


def load_manifest(path: Path | None = None) -> dict[str, dict[str, Any]]:
    # R7 — 기본값을 인자 디폴트에 박지 않고 여기서 해석합니다.
    path = MANIFEST_PATH if path is None else path
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # 매니페스트가 깨졌으면 빈 것으로 취급합니다. 원격 head가 복구해 줍니다.
        log.warning("매니페스트가 깨졌습니다. 원격 확인으로 복구합니다: %s", path)
        return {}
    return data.get("objects", {}) if isinstance(data, dict) else {}


def save_manifest(objects: dict[str, dict[str, Any]], path: Path | None = None) -> None:
    path = MANIFEST_PATH if path is None else path
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": 1, "objects": objects}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def assert_uploadable(body: bytes, channel: str) -> int:
    """줄 단위 스코프 게이트. 통과하면 줄 수를 돌려줍니다.

    빈 줄은 건너뛰되, **깨진 JSON은 통과시키지 않습니다** — 파싱 실패를 무시하면
    게이트가 검사하지 못한 줄이 그대로 올라갑니다.
    """
    rows: list[SimpleNamespace] = []
    for lineno, raw in enumerate(body.decode("utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SyncRefused(f"{lineno}번째 줄이 JSON이 아닙니다: {exc}") from exc
        if row.get("channel") != channel:
            raise SyncRefused(
                f"{lineno}번째 줄의 channel이 {row.get('channel')!r} 입니다 "
                f"(기대: {channel!r}). 다른 채널이 섞였습니다."
            )
        rows.append(SimpleNamespace(id=row.get("id"), publish_scope=row.get("publish_scope")))
    # R5 — 게이트는 "존재"가 아니라 "호출"되어야 의미가 있습니다.
    assert_public_scope(rows)
    return len(rows)


def plan(
    store: ObjectStore,
    channel: str,
    prefix: str,
    candidates_dir: Path | None = None,
    manifest: dict[str, dict[str, Any]] | None = None,
) -> list[Upload]:
    candidates_dir = CANDIDATES_DIR if candidates_dir is None else candidates_dir
    manifest = {} if manifest is None else manifest

    if channel not in ALLOWED_CHANNELS:
        raise SyncRefused(
            f"채널 {channel!r}은 적재 대상이 아닙니다 (허용: {sorted(ALLOWED_CHANNELS)}). "
            "CLAUDE.md §3-3 — 채용공고는 외부 스토리지에 올리지 않습니다."
        )
    if not candidates_dir.exists():
        return []

    uploads: list[Upload] = []
    for path in sorted(candidates_dir.glob("*.jsonl")):
        date = path.stem
        body = path.read_bytes()
        digest = sha256_bytes(body)
        key = build_key(channel, date, prefix)

        entry = manifest.get(key, {})
        # ★ 백엔드가 다르면 "기록 없음"으로 봅니다. RADAR_MODE=fake 로 로컬에 적재한
        #   기록 때문에 실제 버킷으로 바꿨을 때 전부 건너뛰는 사고를 막습니다.
        recorded = entry.get("sha256") if entry.get("backend") == store.backend_id else None
        if recorded == digest:
            continue
        if recorded is None:
            remote = store.head(key)
            if remote is not None and remote.sha256 == digest:
                # 원격엔 이미 있는데 매니페스트만 잃은 경우. 올리지 않고 기록만 되살립니다.
                uploads.append(Upload(path, key, digest, len(body), "manifest-lost"))
                continue
            reason = "new"
        else:
            reason = "changed"
        uploads.append(Upload(path, key, digest, len(body), reason))
    return uploads


def sync(
    store: ObjectStore,
    channel: str,
    prefix: str,
    candidates_dir: Path | None = None,
    manifest_path: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    uploads = plan(store, channel, prefix, candidates_dir, manifest)

    uploaded = 0
    skipped_recovered = 0
    total_bytes = 0
    for up in uploads:
        body = up.path.read_bytes()
        rows = assert_uploadable(body, channel)
        if up.reason == "manifest-lost":
            # 원격에 동일 해시가 이미 있음 — 네트워크를 태우지 않고 기록만 복구합니다.
            skipped_recovered += 1
        elif dry_run:
            log.info("[dry-run] %s ← %s (%d줄)", up.key, up.path, rows)
        else:
            store.put(up.key, body, sha256=up.sha256)
            uploaded += 1
            total_bytes += up.size
        if not dry_run:
            manifest[up.key] = {
                "backend": store.backend_id,
                "sha256": up.sha256,
                "size": up.size,
                "rows": rows,
                "source": str(up.path),
            }

    if not dry_run and uploads:
        save_manifest(manifest, manifest_path)

    result = {
        "planned": len(uploads),
        "uploaded": uploaded,
        "recovered": skipped_recovered,
        "bytes": total_bytes,
        "dry_run": dry_run,
    }
    log.info("동기화 완료: %s", result)
    return result


def build_store(prefix_hint: str = "") -> ObjectStore:
    """`RADAR_MODE=fake` 면 로컬 스토어. 기본값은 S3입니다.

    기본값을 로컬로 두지 않는 이유는 `local.py` 상단 참조 — 올라간 줄 알았는데
    디스크에만 있는 상태가 조용히 만들어집니다.
    """
    if radar_mode() == "fake":
        log.warning("RADAR_MODE=fake — 로컬 스토어에 적재합니다. S3가 아닙니다.")
        from src.storage.local import LocalObjectStore  # noqa: PLC0415

        return LocalObjectStore(FAKE_STORE_DIR)
    from src.storage.s3 import S3ObjectStore  # noqa: PLC0415

    return S3ObjectStore()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.storage.sync")
    parser.add_argument("--channel", default="papers", choices=sorted(ALLOWED_CHANNELS))
    parser.add_argument("--prefix", default=None, help="기본값: RADAR_S3_PREFIX 또는 'radar'")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    prefix = args.prefix if args.prefix is not None else os.environ.get("RADAR_S3_PREFIX", "radar").strip().strip("/")
    try:
        store = build_store()
        result = sync(store, args.channel, prefix, dry_run=args.dry_run)
    except SyncRefused as exc:
        log.error("게이트 거부: %s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001 — 크론에서 조용히 죽지 않도록 exit 1 로 알린다
        log.error("동기화 실패: %s", exc)
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
