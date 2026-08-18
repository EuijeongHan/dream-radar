"""임베딩 포트 — 프로토콜 · 해시 스텁 · bge-m3 어댑터 · 캐시 (기획안_2 §5.1 · §5.3).

이 모듈이 지키는 것 4가지:

1. **모듈 최상단에서 torch·sentence-transformers 를 import 하지 않습니다** (§9.10).
   운영 `requirements.txt` 에는 PyYAML·requests 만 있고, 로컬 평가만
   `requirements-eval.txt` 로 무거운 것을 깝니다. `import src.rank.embed` 가
   Actions 러너에서 죽으면 랭킹 단계가 통째로 멈춥니다. 실모델 어댑터는 지연 import.

2. **리비전을 코드에 타이핑하지 않습니다** (재사용 규칙 R8 / §9.3).
   `eval/models.yaml` 에서 읽습니다. 실제로 밟은 함정입니다 — 리비전 없이
   `SentenceTransformer("BAAI/bge-m3")` 를 호출했더니 sentence-transformers 5.x 가
   safetensors 를 선호해 **머지되지 않은 PR(`refs/pr/130`)의 가중치**를 받아갔습니다.
   PR은 닫히거나 바뀌므로 그 상태로 낸 Hit@5 는 재현되지 않습니다.
   여기서는 한 걸음 더 나가 **40자 커밋 해시가 아니면 로드를 거부**합니다.

3. **캐시 키는 §5.3 규칙 그대로**입니다:
   `data/cache/emb/{model_id}_{revision[:8]}/{sha256(text)[:16]}.npy`.
   모델·리비전이 바뀌면 디렉터리가 갈리고, 텍스트가 바뀌면 파일명이 갈립니다.
   무효화 로직을 따로 두지 않는 것이 핵심입니다 — 무효화는 잊습니다.

4. **경로 기본값을 정의 시점에 바인딩하지 않습니다** (§9.1 / R7).
   `cache_dir=None` + 호출 시점 해석. 모듈 상수를 기본 인자로 쓰면 테스트가
   monkeypatch 해도 실제 `data/` 를 읽습니다 (라벨링 테스트 17개가 이 상태였습니다).

`.npy` 를 numpy 없이 쓰는 이유
------------------------------
캐시 형식은 §5.3이 `.npy` 로 못박았지만 numpy 는 **운영 의존성이 아닙니다**
(`requirements.txt` 에 없음). npy v1.0 은 헤더가 ASCII dict 인 단순 포맷이라
표준 라이브러리(`struct`)만으로 쓰고 읽을 수 있고, 그렇게 쓴 파일을 numpy 가 그대로
읽습니다. 의존성을 늘리지 않으면서 형식 호환을 지키는 쪽을 택했습니다.

저장 dtype 은 `<f4`(float32) 입니다. 실모델 출력이 float32 이고 파일이 절반이며,
왕복 오차는 1e-7 수준입니다 — `eval/models.yaml` 이 실측한 MPS/CPU 최대 절대오차
5.22e-07 과 같은 자릿수라 코사인 순위를 바꾸지 않습니다.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

import yaml

#: 모델 카탈로그. **함수 기본 인자에 직접 쓰지 마세요** (§9.1 — 호출 시점 해석).
DEFAULT_MODELS_PATH = Path("eval/models.yaml")

#: 임베딩 캐시 루트 (§5.3). `data/cache/` 는 이미 gitignore 입니다.
DEFAULT_CACHE_DIR = Path("data/cache/emb")

#: 벡터 표현. tuple 인 이유는 캐시 값을 실수로 제자리 수정하지 못하게 하기 위함입니다.
Vector = tuple[float, ...]

#: 리비전은 40자 커밋 해시만 인정합니다 (§9.3). `"main"` 은 재현성이 없습니다.
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")

#: 해시 스텁의 토큰화. 유니코드 단어 문자 — 한국어 프로파일(교차언어 대조군)도 통과합니다.
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


# ── 모델 스펙 (eval/models.yaml) ─────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """`eval/models.yaml` 한 항목. 리비전이 해시가 아니면 생성 자체를 거부합니다."""

    id: str
    repo: str
    revision: str
    raw: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.repo:
            raise ValueError(f"{self.id}: repo 가 비어 있습니다 (eval/models.yaml)")
        # ★ §9.3. 여기서 막지 않으면 미머지 PR 가중치로 평가가 돌고, 그 수치는
        #   3개월 뒤에 재현되지 않습니다. M1 수치가 이 프로젝트의 가치 전부입니다(§1).
        if not _REVISION_RE.match(self.revision or ""):
            raise ValueError(
                f"{self.id}({self.repo}) 의 revision 이 40자 커밋 해시가 아닙니다: "
                f"{self.revision!r} — eval/models.yaml 에 해시를 고정하세요 (기획안_2 §9.3)"
            )


def load_models_config(models_path: Path | str | None = None) -> dict[str, Any]:
    """`eval/models.yaml` 전체를 dict 로 읽습니다 (기본 경로는 **호출 시점** 해석)."""
    path = Path(models_path) if models_path is not None else DEFAULT_MODELS_PATH
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError(f"모델 카탈로그가 매핑이 아닙니다: {path}")
    return dict(data)


def load_model_spec(kind: str = "embedder", models_path: Path | str | None = None) -> ModelSpec:
    """`"embedder"` · `"reranker"` · `optional_embedders` 의 `id` 로 스펙을 읽습니다 (R8).

    코드에 커밋 해시를 타이핑하는 대신 항상 이 함수를 거치세요. 두 곳이 어긋나면
    **어느 가중치로 낸 수치인지 영영 알 수 없습니다.**
    """
    config = load_models_config(models_path)
    entry: Any = config.get(kind) if isinstance(config.get(kind), Mapping) else None
    if entry is None:
        # 최상위 키(`embedder`/`reranker`)로 못 찾으면 **id** 로 찾습니다.
        # `build_embedder("bge_m3")` 처럼 부르는 쪽은 카탈로그의 키가 아니라
        # id 를 알고 있는 게 자연스럽습니다.
        pool: list[Mapping[str, Any]] = []
        for key in ("embedder", "reranker"):
            candidate = config.get(key)
            if isinstance(candidate, Mapping):
                pool.append(candidate)
        pool.extend(
            item for item in (config.get("optional_embedders") or []) if isinstance(item, Mapping)
        )
        for candidate in pool:
            if candidate.get("id") == kind:
                entry = candidate
                break
    if not isinstance(entry, Mapping):
        raise KeyError(f"모델 카탈로그에 {kind!r} 항목이 없습니다 (eval/models.yaml)")
    return ModelSpec(
        id=str(entry.get("id") or kind),
        repo=str(entry.get("repo") or ""),
        revision=str(entry.get("revision") or ""),
        raw=dict(entry),
    )


def resolve_device(
    device: str | None = None,
    models_path: Path | str | None = None,
    *,
    torch_module: Any = None,
) -> str:
    """`None` 이면 `models.yaml` 의 `device.preferred` → 사용 불가면 `fallback`.

    MPS와 CPU가 수치적으로 동일한 것은 이미 확인됐습니다 (models.yaml — 코사인 최소
    1.000000, 최대 절대오차 5.22e-07, top-5 순위 완전 일치). 그래서 장치 선택은
    **속도 문제일 뿐 결과를 바꾸지 않습니다.** 다만 어느 장치로 돌렸는지는
    `eval/results.md` 에 반드시 기록하세요 (§5.6).
    """
    if device:
        return device
    config = load_models_config(models_path)
    block = config.get("device") or {}
    preferred = str(block.get("preferred") or "cpu")
    fallback = str(block.get("fallback") or "cpu")
    if preferred == "cpu":
        return preferred

    torch = torch_module
    if torch is None:
        # 지연 import — Actions 러너에는 torch 가 없습니다 (§9.10).
        try:
            import torch as _torch  # noqa: PLC0415
        except ImportError:
            return fallback
        torch = _torch
    if preferred == "mps":
        backend = getattr(getattr(torch, "backends", None), "mps", None)
        available = bool(backend and backend.is_available())
    elif preferred == "cuda":
        available = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    else:
        available = True
    return preferred if available else fallback


# ── 벡터 연산 (numpy 없이) ───────────────────────────────────────────────────


def l2_norm(vector: Sequence[float]) -> float:
    return math.sqrt(math.fsum(float(value) * float(value) for value in vector))


def normalize(vector: Sequence[float]) -> Vector:
    """L2 정규화. 영벡터는 그대로 돌려줍니다 (0으로 나누지 않습니다)."""
    norm = l2_norm(vector)
    if norm == 0.0:
        return tuple(float(value) for value in vector)
    return tuple(float(value) / norm for value in vector)


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """코사인 유사도. 한쪽이라도 영벡터면 **0.0**.

    길이가 다르면 `ValueError` 입니다 — 서로 다른 모델의 벡터를 섞으면 조용히
    잘린 채 계산되는 대신 시끄럽게 죽어야 합니다.
    """
    if len(left) != len(right):
        raise ValueError(f"벡터 차원이 다릅니다: {len(left)} vs {len(right)}")
    left_norm = l2_norm(left)
    right_norm = l2_norm(right)
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    dot = math.fsum(float(a) * float(b) for a, b in zip(left, right))
    return dot / (left_norm * right_norm)


# ── npy v1.0 (표준 라이브러리만) ─────────────────────────────────────────────

_NPY_MAGIC = b"\x93NUMPY"
_NPY_VERSION = b"\x01\x00"
_NPY_ALIGN = 64  # npy 사양: magic + version + HEADER_LEN + header 가 64의 배수
_NPY_DESCR = "<f4"
_NPY_HEADER_RE = re.compile(
    r"'descr':\s*'(?P<descr>[^']+)'.*?'fortran_order':\s*(?P<order>True|False)"
    r".*?'shape':\s*\((?P<shape>[^)]*)\)",
    re.DOTALL,
)


def npy_dumps(vector: Sequence[float]) -> bytes:
    """1차원 float32 배열을 npy v1.0 바이트로 직렬화합니다 (numpy 불필요)."""
    values = [float(value) for value in vector]
    header = f"{{'descr': '{_NPY_DESCR}', 'fortran_order': False, 'shape': ({len(values)},), }}"
    prefix = len(_NPY_MAGIC) + len(_NPY_VERSION) + 2
    padding = (_NPY_ALIGN - (prefix + len(header) + 1) % _NPY_ALIGN) % _NPY_ALIGN
    header = header + " " * padding + "\n"
    return (
        _NPY_MAGIC
        + _NPY_VERSION
        + struct.pack("<H", len(header))
        + header.encode("latin-1")
        + struct.pack(f"<{len(values)}f", *values)
    )


def npy_loads(blob: bytes) -> Vector:
    """`npy_dumps` 의 역연산. 형식이 어긋나면 `ValueError`.

    깨진 캐시 파일(쓰다 만 것)을 조용히 반쯤 읽는 것보다 예외가 낫습니다 —
    캐시 미스로 처리하는 쪽은 `EmbeddingCache` 가 판단합니다.
    """
    if not blob.startswith(_NPY_MAGIC):
        raise ValueError("npy 매직 바이트가 아닙니다")
    if blob[6:8] != _NPY_VERSION:
        raise ValueError(f"npy 버전이 1.0 이 아닙니다: {blob[6:8]!r}")
    (header_len,) = struct.unpack("<H", blob[8:10])
    header = blob[10 : 10 + header_len].decode("latin-1")
    matched = _NPY_HEADER_RE.search(header)
    if not matched:
        raise ValueError(f"npy 헤더를 해석할 수 없습니다: {header!r}")
    if matched.group("descr") != _NPY_DESCR:
        raise ValueError(f"float32 리틀엔디언만 지원합니다: {matched.group('descr')!r}")
    shape = [part for part in matched.group("shape").split(",") if part.strip()]
    if len(shape) != 1:
        raise ValueError(f"1차원 배열만 지원합니다: shape=({matched.group('shape')})")
    count = int(shape[0])
    payload = blob[10 + header_len :]
    if len(payload) != count * 4:
        raise ValueError(f"본문 길이가 shape 와 다릅니다: {len(payload)} != {count * 4}")
    return tuple(struct.unpack(f"<{count}f", payload))


# ── 포트 ────────────────────────────────────────────────────────────────────


@runtime_checkable
class Embedder(Protocol):
    """임베딩 포트 (§4.2). 구현체는 `model_id`·`revision`·`encode` 셋을 냅니다.

    `model_id`·`revision` 은 장식이 아니라 **캐시 키의 일부**입니다 (§5.3).
    모델을 바꾸면 캐시가 자동으로 갈려야 하므로 구현체는 자기 정체를 정확히 밝혀야 합니다.
    """

    @property
    def model_id(self) -> str: ...

    @property
    def revision(self) -> str: ...

    def encode(self, texts: Sequence[str]) -> list[Vector]: ...


class HashStubEmbedder:
    """키·모델·네트워크 없이 도는 결정적 임베더 — **배선 테스트 전용**.

    ★ 랭킹 품질 평가에 쓰지 마세요 (`src/sources/fake.py` 와 같은 경고입니다).
      의미를 전혀 모르는 해싱 트릭이라 Hit@5 를 재면 숫자가 거짓이 됩니다.

    구현: 토큰마다 sha256 → 버킷 index + 부호를 뽑아 누적하고 L2 정규화.
    문서 전체를 한 번에 해싱하지 않는 이유는, 그러면 **모든 문서가 서로 직교**해서
    "가중 최대 vs 가중 평균"(§5.2) 같은 점수 합성 차이를 테스트로 드러낼 수 없기
    때문입니다. 토큰 단위 해싱은 공유 토큰이 많을수록 코사인이 올라갑니다.

    `revision` 에 차원을 넣습니다. 차원을 바꾸면 벡터가 완전히 달라지는데 캐시 키가
    같으면 이전 차원의 캐시를 읽고 `cosine()` 에서 차원 불일치로 죽습니다.
    """

    model_id = "hash_stub"

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError(f"dim 은 양수여야 합니다: {dim}")
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def revision(self) -> str:
        return f"stub{self._dim:04d}"

    def encode(self, texts: Sequence[str]) -> list[Vector]:
        return [self._encode_one(text) for text in texts]

    def _encode_one(self, text: str) -> Vector:
        buckets = [0.0] * self._dim
        for token in _TOKEN_RE.findall(text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self._dim
            sign = 1.0 if digest[4] & 1 else -1.0
            buckets[index] += sign
        return normalize(buckets)


class EmbeddingCache:
    """§5.3 캐시. 키 규칙을 이 클래스 밖에서 다시 조립하지 마세요.

        data/cache/emb/{model_id}_{revision[:8]}/{sha256(text)[:16]}.npy

    캐시가 없으면 조건 하나 바꿀 때마다 719건 × 3일치를 다시 인코딩합니다(CPU 3.1분).
    그러면 사람이 조건을 4개 다 안 돌리고 한두 개로 끝냅니다 — **캐시가 없을 때
    조용히 망가지는 것은 속도가 아니라 비교 조건의 수입니다** (§5.3).
    """

    def __init__(self, model_id: str, revision: str, cache_dir: Path | str | None = None) -> None:
        self._model_id = model_id
        self._revision = revision
        # ★ 모듈 상수를 여기서 붙잡지 않습니다. `dir` 에서 호출 시점에 해석합니다 (§9.1).
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None

    @property
    def dir(self) -> Path:
        root = self._cache_dir if self._cache_dir is not None else DEFAULT_CACHE_DIR
        return Path(root) / f"{self._model_id}_{self._revision[:8]}"

    def path_for(self, text: str) -> Path:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        return self.dir / f"{digest}.npy"

    def get(self, text: str) -> Vector | None:
        path = self.path_for(text)
        if not path.exists():
            return None
        try:
            return npy_loads(path.read_bytes())
        except (ValueError, OSError):
            # 쓰다 만 파일 · 다른 형식. 조용히 반쯤 읽느니 미스로 취급하고 다시 인코딩합니다.
            return None

    def put(self, text: str, vector: Sequence[float]) -> Path:
        path = self.path_for(text)
        path.parent.mkdir(parents=True, exist_ok=True)
        # 임시 파일 → os.replace 로 원자적 교체. 중간에 죽어도 반쪽 파일이 캐시에
        # 남지 않습니다 (후보 파일 중복 줄 §9.6 과 같은 종류의 사고를 막습니다).
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_bytes(npy_dumps(vector))
        os.replace(tmp, path)
        return path


class CachedEmbedder:
    """다른 임베더를 감싸 §5.3 캐시를 붙입니다. 그 자체로 `Embedder` 입니다.

    - 캐시 적중분은 내부 임베더를 **호출하지 않습니다** (테스트가 이걸 검사합니다)
    - 한 배치 안의 중복 텍스트는 한 번만 인코딩합니다
    - `hits`/`misses` 는 평가 로그용입니다. 캐시가 실제로 먹었는지 눈으로 확인하세요
    """

    def __init__(self, inner: Embedder, cache_dir: Path | str | None = None) -> None:
        self._inner = inner
        self._cache_dir = cache_dir
        self.hits = 0
        self.misses = 0

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    @property
    def revision(self) -> str:
        return self._inner.revision

    @property
    def cache(self) -> EmbeddingCache:
        # 내부 임베더의 정체를 매번 다시 읽습니다 — 지연 로딩 임베더는 스펙 해석이
        # 늦게 끝날 수 있고, 그때 캐시 디렉터리가 바뀌어야 합니다.
        return EmbeddingCache(self.model_id, self.revision, self._cache_dir)

    def encode(self, texts: Sequence[str]) -> list[Vector]:
        texts = list(texts)
        cache = self.cache
        results: list[Vector | None] = [None] * len(texts)
        pending: list[str] = []
        positions: dict[str, list[int]] = {}

        for index, text in enumerate(texts):
            cached = cache.get(text)
            if cached is not None:
                results[index] = cached
                self.hits += 1
                continue
            if text not in positions:
                positions[text] = []
                pending.append(text)
            positions[text].append(index)

        if pending:
            encoded = self._inner.encode(pending)
            if len(encoded) != len(pending):
                raise RuntimeError(
                    f"임베더가 {len(pending)}건을 받아 {len(encoded)}건을 냈습니다 — "
                    "입력과 출력 순서가 어긋나면 문서와 벡터가 뒤섞입니다"
                )
            for text, vector in zip(pending, encoded):
                vector = tuple(float(value) for value in vector)
                cache.put(text, vector)
                for index in positions[text]:
                    results[index] = vector
                    self.misses += 1

        filled: list[Vector] = []
        for index, vector in enumerate(results):
            if vector is None:  # 도달할 수 없는 상태. 도달했다면 조용히 건너뛰지 않습니다
                raise RuntimeError(f"임베딩이 비었습니다: index={index}, text={texts[index]!r}")
            filled.append(vector)
        return filled


class BgeM3Embedder:
    """`BAAI/bge-m3` 어댑터 (로컬 CPU/MPS, 키 불필요 — CLAUDE.md §4).

    - `sentence_transformers` 는 **`encode` 시점에** import 합니다. 모듈 import 만으로
      죽으면 Actions 러너(§9.10)에서 랭킹 단계가 통째로 멈춥니다
    - repo·revision 은 `eval/models.yaml` 에서 읽습니다 (R8 / §9.3)
    - `factory` 는 테스트 주입구입니다. 실모델 2.3GB 를 받지 않고도 "리비전을 정말
      넘기는가"를 검사할 수 있어야 합니다

    실측 (models.yaml): CPU 0.255초/건, MPS 0.094초/건. 719건이면 각각 3.1분 / 1.1분.
    """

    def __init__(
        self,
        spec: ModelSpec | None = None,
        *,
        device: str | None = None,
        models_path: Path | str | None = None,
        batch_size: int = 16,
        factory: Callable[..., Any] | None = None,
        model_kind: str = "embedder",
    ) -> None:
        self._spec = spec
        self._device_arg = device
        self._models_path = models_path
        self._batch_size = batch_size
        self._factory = factory
        self._model_kind = model_kind
        self._model: Any = None
        self._device: str | None = None

    @property
    def spec(self) -> ModelSpec:
        if self._spec is None:
            self._spec = load_model_spec(self._model_kind, self._models_path)
        return self._spec

    @property
    def model_id(self) -> str:
        return self.spec.id

    @property
    def revision(self) -> str:
        return self.spec.revision

    @property
    def device(self) -> str:
        if self._device is None:
            self._device = resolve_device(self._device_arg, self._models_path)
        return self._device

    def _load(self) -> Any:
        if self._model is None:
            factory = self._factory or _default_sentence_transformer
            # ★ revision 을 반드시 넘깁니다 (§9.3). 빼면 미머지 PR 가중치가 옵니다.
            self._model = factory(self.spec.repo, revision=self.spec.revision, device=self.device)
        return self._model

    def encode(self, texts: Sequence[str]) -> list[Vector]:
        texts = list(texts)
        if not texts:
            return []
        model = self._load()
        vectors = model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [tuple(float(value) for value in row) for row in vectors]

    def describe(self) -> dict[str, Any]:
        """`eval/results.md` 에 적을 실행 조건 (§5.6 — 장치·리비전·max_seq_length).

        모델을 **로드하지 않습니다.** `max_seq_length` 는 이미 로드된 경우에만 채워집니다
        (인코딩을 마친 뒤 호출하세요). 기록용 함수가 2.3GB 다운로드를 유발하면 안 됩니다.
        """
        return {
            "model_id": self.model_id,
            "repo": self.spec.repo,
            "revision": self.spec.revision,
            "device": self._device,  # 아직 해석 전이면 None — 넘겨짚지 않습니다
            "batch_size": self._batch_size,
            "max_seq_length": getattr(self._model, "max_seq_length", None),
        }


def _default_sentence_transformer(repo: str, *, revision: str, device: str) -> Any:
    """실모델 생성. **지연 import** 입니다 (§9.10)."""
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    return SentenceTransformer(repo, revision=revision, device=device)


def build_embedder(
    kind: str = "hash_stub",
    *,
    cache_dir: Path | str | None = None,
    device: str | None = None,
    models_path: Path | str | None = None,
    dim: int = 256,
) -> Embedder:
    """이름으로 임베더를 만들고 캐시를 씌웁니다 (§4.2 포트/어댑터 표).

    `"hash_stub"` 은 배선용, `"bge_m3"`(또는 models.yaml 의 다른 id)은 실모델입니다.
    **키가 없다고 조용히 스텁으로 폴백하지 않습니다** — 알 수 없는 이름은 예외입니다.
    """
    if kind == "hash_stub":
        inner: Embedder = HashStubEmbedder(dim=dim)
    else:
        inner = BgeM3Embedder(device=device, models_path=models_path, model_kind=kind)
    return CachedEmbedder(inner, cache_dir=cache_dir)
