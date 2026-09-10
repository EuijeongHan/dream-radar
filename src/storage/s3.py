"""AWS S3 오브젝트 스토어.

자격증명은 코드에 없습니다 ★
---------------------------
CLAUDE.md §3-2 — "API 키를 코드·로그·원장·커밋에 남기기" 금지. 이 저장소는 public입니다.

boto3의 기본 자격증명 체인을 **그대로** 씁니다. 즉 아래 중 하나로 주입합니다.

    환경변수      AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION
    공유 설정     ~/.aws/credentials  (로컬 · Airflow)
    역할          EC2/ECS 인스턴스 프로파일

**키를 인자로 받는 생성자를 만들지 않았습니다.** 만들면 언젠가 누가 설정 파일에
적어 넣고, 그 파일이 커밋됩니다. 받을 수 없으면 적을 수도 없습니다.

boto3는 지연 임포트입니다
------------------------
`requirements.txt`에 boto3를 넣지 않았습니다. botocore가 100MB에 가깝고, GitHub
Actions의 daily-papers는 수집만 하므로 매일 그걸 받을 이유가 없습니다
(`requirements-eval.txt`를 Actions에 설치하지 않는 것과 같은 판단 — README §개발 환경).

S3 동기화를 쓰는 환경에서만 `pip install -r requirements-s3.txt` 합니다.
"""

from __future__ import annotations

import os
from typing import Any

from src.storage.base import ObjectMeta

#: 우리가 붙이는 무결성 해시의 메타데이터 키. ETag를 쓰지 않는 이유는 base.py 참조.
SHA256_METADATA_KEY = "radar-sha256"


class S3ConfigError(RuntimeError):
    """버킷 미설정 등 설정 오류. **조용히 건너뛰지 않고 죽습니다.**"""


def _load_boto3() -> Any:
    try:
        import boto3  # noqa: PLC0415 — 지연 임포트가 의도입니다
    except ModuleNotFoundError as exc:  # pragma: no cover - 환경 의존
        raise S3ConfigError(
            "boto3가 없습니다. S3 동기화를 쓰려면:\n"
            "    pip install -r requirements-s3.txt"
        ) from exc
    return boto3


def bucket_from_env() -> str:
    """`RADAR_S3_BUCKET` 미설정이면 **죽습니다.**

    빈 문자열을 반환해 호출부가 "설정 안 됐으니 건너뛰자"로 처리하게 두면,
    크론이 매일 조용히 아무것도 안 올리고 초록불을 냅니다. `config.radar_mode()`가
    기본값을 fake로 두지 않는 것과 같은 판단입니다.
    """
    bucket = os.environ.get("RADAR_S3_BUCKET", "").strip()
    if not bucket:
        raise S3ConfigError(
            "RADAR_S3_BUCKET 이 비어 있습니다. 버킷 이름을 환경변수로 주세요.\n"
            "    export RADAR_S3_BUCKET=your-bucket-name"
        )
    return bucket


def prefix_from_env() -> str:
    """기본 `radar`. 앞뒤 `/`는 떼서 키 조립이 `//`가 되지 않게 합니다."""
    return os.environ.get("RADAR_S3_PREFIX", "radar").strip().strip("/")


class S3ObjectStore:
    def __init__(self, bucket: str | None = None, client: Any = None) -> None:
        # R7 — 기본값에 모듈 상수를 박지 않고 호출 시점에 해석합니다.
        self.bucket = bucket if bucket is not None else bucket_from_env()
        self._client = client

    @property
    def backend_id(self) -> str:
        return f"s3:{self.bucket}"

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = _load_boto3().client("s3")
        return self._client

    def put(self, key: str, body: bytes, *, sha256: str) -> None:
        self.client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=body,
            Metadata={SHA256_METADATA_KEY: sha256},
        )

    def head(self, key: str) -> ObjectMeta | None:
        try:
            resp = self.client.head_object(Bucket=self.bucket, Key=key)
        except Exception as exc:  # noqa: BLE001 - botocore 예외 타입을 지연 임포트하지 않기 위해
            if _is_not_found(exc):
                return None
            raise
        meta = resp.get("Metadata") or {}
        return ObjectMeta(
            key=key,
            size=int(resp.get("ContentLength", 0)),
            sha256=meta.get(SHA256_METADATA_KEY),
        )


def _is_not_found(exc: Exception) -> bool:
    """404/NoSuchKey만 "없음"으로 봅니다.

    ★ 광범위 except로 모든 예외를 "없음" 처리하면 **권한 오류(403)가 "객체 없음"으로
    둔갑**해 매번 다시 올리려 시도하고, 그 시도도 실패하는데 로그만 늘어납니다.
    """
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False
    error = response.get("Error", {})
    code = str(error.get("Code", ""))
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return code in {"404", "NoSuchKey", "NotFound"} or status == 404
