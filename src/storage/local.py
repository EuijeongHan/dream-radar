"""파일시스템 오브젝트 스토어 — 테스트와 `RADAR_MODE=fake` 용.

S3의 의미론을 **필요한 만큼만** 흉내 냅니다.

  · 키의 `/`는 디렉터리가 됩니다 (S3는 평면이지만 결과는 같습니다)
  · `sha256`은 사이드카 파일에 남깁니다. S3의 object metadata에 대응합니다
  · 같은 키에 다시 put하면 덮어씁니다 (S3와 동일)

**이것을 운영에 쓰지 마세요.** `sync.py`가 스토어를 만들 때 `RADAR_MODE`를 보고
고르며, 기본값은 live(S3)입니다 — `config.radar_mode()`가 fake를 기본값으로 두지
않는 것과 같은 이유입니다. 기본값이 로컬이면 "S3에 올라간 줄 알았는데 디스크에만
있는" 상태가 조용히 만들어집니다.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.storage.base import ObjectMeta

_META_SUFFIX = ".meta.json"


class LocalObjectStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    @property
    def backend_id(self) -> str:
        return f"local:{self.root}"

    def _path(self, key: str) -> Path:
        # 키에 .. 가 들어오면 루트 밖으로 나갑니다. 스토어는 키를 신뢰하지 않습니다.
        parts = [p for p in key.split("/") if p not in ("", ".", "..")]
        if len(parts) != len([p for p in key.split("/") if p != ""]):
            raise ValueError(f"키에 상위 경로 참조가 있습니다: {key!r}")
        return self.root.joinpath(*parts)

    def put(self, key: str, body: bytes, *, sha256: str) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        path.with_name(path.name + _META_SUFFIX).write_text(
            json.dumps({"sha256": sha256, "size": len(body)}), encoding="utf-8"
        )

    def head(self, key: str) -> ObjectMeta | None:
        path = self._path(key)
        if not path.exists():
            return None
        meta_path = path.with_name(path.name + _META_SUFFIX)
        digest = None
        if meta_path.exists():
            digest = json.loads(meta_path.read_text(encoding="utf-8")).get("sha256")
        return ObjectMeta(key=key, size=path.stat().st_size, sha256=digest)
