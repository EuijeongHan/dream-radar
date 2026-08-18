"""랭킹 레이어 게이트 (기획안_2 §5.1~§5.4 · §5.7 · §9.1 · §9.3 · §9.4).

이 파일이 막는 것 6가지:

1. **점수 합성 방식이 조용히 바뀌는 것** (§5.2) — 가중 최대와 가중 평균은 서로 다른
   논문을 1등으로 올립니다. 기본값이 뒤집히면 Hit@5 가 통째로 달라지는데, 예외는
   나지 않습니다
2. **`exclude.soft` 가 하드 배제로 바뀌는 것** (§5.2 경고) — 경계 사례가 후보에서
   사라지면 Hit@5 의 상한 자체가 내려갑니다
3. **동점 정렬이 불안정해지는 것** (§5.4) — 같은 명령을 두 번 돌려 다른 수치가 나오면
   M1 DoD(§5.8)의 재현성 항목이 거짓이 됩니다
4. **캐시가 안 먹는 것** (§5.3) — 느려서 비교 조건을 4개가 아니라 2개만 돌리게 됩니다.
   증상이 "틀림"이 아니라 "덜 함"이라 눈에 띄지 않습니다
5. **리비전 미고정** (§9.3) — 미머지 PR 가중치로 낸 수치는 재현되지 않습니다
6. **CrossEncoder sigmoid** (§9.4) — 원장의 `rank_score_stage2` 를 사람이 못 읽습니다

전 경로를 `HashStubEmbedder`/`CannedEmbedder` 로 돌립니다. 키도, 2.3GB 다운로드도,
네트워크도 필요 없습니다 (델타 §D6.2). 실모델 테스트는 `slow` + `network` 입니다.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest
import yaml

from src.core.config import load_profile
from src.rank import embed as embed_module
from src.rank.embed import (
    BgeM3Embedder,
    CachedEmbedder,
    EmbeddingCache,
    Embedder,
    HashStubEmbedder,
    ModelSpec,
    cosine,
    load_model_spec,
    npy_dumps,
    npy_loads,
    normalize,
)
from src.rank.profile import (
    DEFAULT_SCORING_METHOD,
    Exclusion,
    Interest,
    QuerySet,
    build_queryset,
    parse_exclusions,
    parse_interests,
    score_document,
)
from src.rank.rerank import (
    BgeReranker,
    FakeReranker,
    RerankedItem,
    Reranker,
    looks_like_probabilities,
    rerank_stage2,
)
from src.rank.retrieve import (
    UNSET,
    RankedItem,
    document_text,
    run_stage1,
    score_items,
    select_stage1,
    stage1_settings,
)

REPO = Path(__file__).resolve().parents[1]
DIM = 8


# ── 테스트용 임베더/리랭커 ───────────────────────────────────────────────────


def basis(index: int, dim: int = DIM) -> tuple[float, ...]:
    """단위 기저 벡터. 코사인이 정확히 0/1 이라 점수 수식을 소수점까지 검사할 수 있습니다."""
    return tuple(1.0 if position == index else 0.0 for position in range(dim))


class CannedEmbedder:
    """텍스트 → 미리 정한 벡터. 표에 없으면 해시 스텁으로 떨어집니다.

    해시 기하(우연한 충돌)에 기대지 않고 §5.2 의 수식을 정확한 숫자로 검사하기 위한
    도구입니다. `encode_calls` 로 **무엇을 임베딩했는지**도 검사합니다.
    """

    model_id = "canned"
    revision = "canned000000"

    def __init__(self, table: dict[str, Any] | None = None, dim: int = DIM) -> None:
        self.table = dict(table or {})
        self.dim = dim
        self.encode_calls: list[list[str]] = []
        self._fallback = HashStubEmbedder(dim=dim)

    def encode(self, texts):
        texts = list(texts)
        self.encode_calls.append(texts)
        out = []
        for text in texts:
            if text in self.table:
                out.append(tuple(float(value) for value in self.table[text]))
            else:
                out.append(self._fallback.encode([text])[0])
        return out


class CannedReranker:
    """쌍 순서대로 미리 정한 점수를 냅니다. 어떤 쌍을 받았는지 `seen` 에 남깁니다."""

    model_id = "canned_reranker"
    revision = "canned000000"

    def __init__(self, scores: list[float]) -> None:
        self.scores = list(scores)
        self.seen: list[tuple[str, str]] = []

    def score_pairs(self, pairs):
        self.seen = list(pairs)
        return self.scores[: len(self.seen)]


def make_profile(
    interests: list[tuple[str, str, float]],
    exclude: list[tuple[str, float]] | None = None,
    selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "interests": [
            {"id": name, "label": f"레이블-{name}", "text": text, "weight": weight}
            for name, text, weight in interests
        ]
    }
    if exclude:
        profile["exclude"] = {"soft": [{"text": text, "penalty": p} for text, p in exclude]}
    if selection:
        profile["selection"] = selection
    return profile


def make_item(item_id: str, title: str, abstract: str = "") -> dict[str, Any]:
    """후보 JSONL 한 줄 모양 (`Item.to_dict()` 의 부분집합)."""
    return {"id": item_id, "title": title, "abstract": abstract}


# ── 해시 스텁 임베더 ─────────────────────────────────────────────────────────


def test_hash_stub_is_deterministic_across_instances():
    """스텁이 결정적이 아니면 같은 조건을 두 번 돌려 다른 수치가 나옵니다 (§5.8 재현성).

    깨뜨리는 법: HashStubEmbedder._encode_one 의 sha256 을 `hash(token)` 으로 바꾸면
    PYTHONHASHSEED 때문에 프로세스마다 달라져 빨간불 (같은 프로세스 안이면 통과하므로
    두 인스턴스가 아니라 두 실행에서 확인).
    확인일: 2026-08-18
    """
    first = HashStubEmbedder(dim=64).encode(["retrieval augmented generation"])[0]
    second = HashStubEmbedder(dim=64).encode(["retrieval augmented generation"])[0]
    assert first == second
    assert len(first) == 64


def test_hash_stub_ranks_shared_tokens_higher():
    """토큰을 공유할수록 코사인이 높아야 합니다.

    문서 전체를 한 번에 해싱하면 모든 문서가 직교해서 §5.2 의 합성 방식 차이를
    테스트로 드러낼 수 없습니다. 그래서 토큰 단위 해싱입니다.

    깨뜨리는 법: _encode_one 에서 토큰 루프 대신 `_TOKEN_RE.findall` 결과를
    `[text]` 로 바꾸면(문서 전체를 토큰 1개로 취급) near/far 가 둘 다 0이 되어 빨간불.
    확인일: 2026-08-18
    """
    embedder = HashStubEmbedder(dim=512)
    query, near, far = embedder.encode(
        [
            "diffusion image editing identity preservation",
            "diffusion image editing with masks",
            "quantum entanglement distillation protocol",
        ]
    )
    assert cosine(query, near) > cosine(query, far)


def test_hash_stub_vectors_are_unit_norm():
    """정규화가 빠지면 긴 문서가 코사인과 무관하게 유리해집니다(내적으로 계산될 때).

    깨뜨리는 법: _encode_one 의 `return normalize(buckets)` 를 `return tuple(buckets)`
    로 바꾸면 노름이 1이 아니라 빨간불.
    확인일: 2026-08-18
    """
    (vector,) = HashStubEmbedder(dim=128).encode(["retrieval evaluation with Hit@5 and MRR"])
    assert sum(value * value for value in vector) == pytest.approx(1.0, abs=1e-9)


def test_hash_stub_revision_tracks_dimension():
    """차원이 캐시 키에 반영되지 않으면, 차원을 바꾼 뒤 이전 차원의 캐시를 읽습니다.

    깨뜨리는 법: HashStubEmbedder.revision 을 상수 문자열로 고정하면 두 revision 이
    같아져 빨간불.
    확인일: 2026-08-18
    """
    assert HashStubEmbedder(dim=64).revision != HashStubEmbedder(dim=128).revision


def test_cosine_rejects_dimension_mismatch():
    """다른 모델의 벡터를 섞으면 조용히 잘리는 대신 죽어야 합니다.

    깨뜨리는 법: cosine() 의 길이 검사를 지우면 zip 이 짧은 쪽에서 멈춰 조용히
    통과하고 빨간불.
    확인일: 2026-08-18
    """
    with pytest.raises(ValueError, match="차원"):
        cosine((1.0, 0.0), (1.0, 0.0, 0.0))


def test_cosine_of_zero_vector_is_zero():
    """빈 텍스트(초록 없는 후보)가 ZeroDivisionError 로 파이프라인을 죽이면 안 됩니다."""
    assert cosine((0.0, 0.0), (1.0, 0.0)) == 0.0


# ── npy 직렬화 (numpy 없이) ──────────────────────────────────────────────────


def test_npy_roundtrip_preserves_values():
    """깨뜨리는 법: npy_dumps 의 struct 포맷을 `<{n}d`(float64)로 바꾸면 npy_loads 의
    본문 길이 검사에서 빨간불.
    확인일: 2026-08-18
    """
    original = normalize([0.5, -0.25, 0.125, 2.0])
    restored = npy_loads(npy_dumps(original))
    assert restored == pytest.approx(original, abs=1e-6)


def test_npy_header_follows_numpy_v1_format():
    """numpy 가 읽을 수 있는 형식이어야 합니다 — 캐시를 평가 스크립트가 읽습니다.

    numpy 를 운영 의존성으로 추가하지 않으려고(§9.10) 직접 씁니다. 형식을 지키는지는
    numpy 없이도 검사할 수 있습니다: 매직 · 버전 · 64바이트 정렬 · dtype.

    깨뜨리는 법: npy_dumps 의 `padding` 계산에서 `+ 1`(개행)을 지우면 전체 헤더가
    64의 배수가 아니게 되어 빨간불.
    확인일: 2026-08-18
    """
    blob = npy_dumps([0.0] * 10)
    assert blob[:6] == b"\x93NUMPY"
    assert blob[6:8] == b"\x01\x00"
    header_len = int.from_bytes(blob[8:10], "little")
    assert (10 + header_len) % 64 == 0, "npy 사양: 헤더 전체가 64의 배수여야 합니다"
    header = blob[10 : 10 + header_len].decode("latin-1")
    assert "'descr': '<f4'" in header
    assert "'fortran_order': False" in header
    assert "'shape': (10,)" in header
    assert header.endswith("\n")
    assert len(blob) == 10 + header_len + 10 * 4


def test_npy_loads_rejects_truncated_file():
    """쓰다 만 캐시 파일을 반쯤 읽는 것보다 예외가 낫습니다 (§9.6 과 같은 종류).

    깨뜨리는 법: npy_loads 의 본문 길이 검사를 지우면 struct.error 대신 조용한 통과
    또는 다른 예외가 나서 빨간불.
    확인일: 2026-08-18
    """
    blob = npy_dumps([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="본문 길이"):
        npy_loads(blob[:-4])


# ── 임베딩 캐시 (§5.3) ───────────────────────────────────────────────────────


def _spy_embedder(model_id: str = "spy", revision: str = "abcdef0123456789") -> Any:
    """`encode` 호출 횟수를 세는 mock 임베더."""
    spy = mock.Mock(spec_set=["model_id", "revision", "encode"])
    spy.model_id = model_id
    spy.revision = revision
    stub = HashStubEmbedder(dim=DIM)
    spy.encode.side_effect = lambda texts: stub.encode(texts)
    return spy


def test_cache_path_follows_spec_key_rule(tmp_path):
    """★ §5.3 키 규칙: `{model_id}_{revision[:8]}/{sha256(text)[:16]}.npy`.

    키에 모델 id 와 리비전이 들어가야 모델을 바꿀 때 캐시가 **자동으로** 갈립니다.
    무효화를 코드로 관리하면 잊습니다.

    깨뜨리는 법: EmbeddingCache.dir 의 `revision[:8]` 을 `revision` 으로 바꾸거나
    path_for 의 `[:16]` 을 지우면 경로가 달라져 빨간불.
    확인일: 2026-08-18
    """
    cache = EmbeddingCache("bge_m3", "5617a9f61b028005a4858fdac845db406aefb181", tmp_path)
    text = "retrieval augmented generation"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    assert cache.dir == tmp_path / "bge_m3_5617a9f6"
    assert cache.path_for(text) == tmp_path / "bge_m3_5617a9f6" / f"{digest}.npy"
    assert len(digest) == 16


def test_cache_hit_does_not_call_the_encoder(tmp_path):
    """★ 두 번째 호출은 인코더를 부르지 않아야 합니다 (§5.3).

    이게 안 되면 조건 하나 바꿀 때마다 719건 × 3일치를 다시 인코딩하고(CPU 3.1분),
    결국 비교 조건을 4개가 아니라 1~2개만 돌리게 됩니다. 증상이 "틀림"이 아니라
    "덜 함"이라 아무도 눈치채지 못합니다.

    깨뜨리는 법: CachedEmbedder.encode 의 `cached = cache.get(text)` 를
    `cached = None` 으로 바꾸면 두 번째 호출도 인코더를 불러 call_count 2 로 빨간불.
    확인일: 2026-08-18
    """
    spy = _spy_embedder()
    cached = CachedEmbedder(spy, cache_dir=tmp_path)

    first = cached.encode(["hello world"])
    second = cached.encode(["hello world"])

    assert spy.encode.call_count == 1, "캐시 적중인데 인코더를 다시 불렀습니다"
    assert first[0] == pytest.approx(second[0], abs=1e-6)
    assert (cached.hits, cached.misses) == (1, 1)


def test_cache_dedupes_repeated_texts_within_one_batch(tmp_path):
    """한 배치에 같은 텍스트가 여러 번 오면 한 번만 인코딩합니다.

    후보 파일에는 중복 줄이 생길 수 있습니다 (§9.6 — collect 가 쓰고 죽으면 다음
    실행이 같은 아이템을 한 번 더 씁니다).

    깨뜨리는 법: CachedEmbedder.encode 의 `if text not in positions:` 를 지우고
    무조건 pending 에 append 하면 인코더가 3건을 받아 빨간불.
    확인일: 2026-08-18
    """
    spy = _spy_embedder()
    cached = CachedEmbedder(spy, cache_dir=tmp_path)

    vectors = cached.encode(["dup", "dup", "other"])

    assert spy.encode.call_args.args[0] == ["dup", "other"]
    assert len(vectors) == 3
    assert vectors[0] == vectors[1]


def test_cache_separates_by_revision(tmp_path):
    """리비전이 바뀌면 캐시가 갈려야 합니다 (§5.3 — 무효화가 자동).

    깨뜨리는 법: EmbeddingCache.dir 에서 revision 을 빼면 두 디렉터리가 같아져
    두 번째 인코더 호출이 사라지고 빨간불.
    확인일: 2026-08-18
    """
    old = _spy_embedder(revision="1111111111111111")
    new = _spy_embedder(revision="2222222222222222")

    CachedEmbedder(old, cache_dir=tmp_path).encode(["same text"])
    CachedEmbedder(new, cache_dir=tmp_path).encode(["same text"])

    assert old.encode.call_count == 1
    assert new.encode.call_count == 1, "리비전이 달라졌는데 이전 캐시를 읽었습니다"
    assert {path.name for path in tmp_path.iterdir()} == {"spy_11111111", "spy_22222222"}


def test_cache_dir_default_is_resolved_at_call_time(tmp_path, monkeypatch):
    """★ 함정 9.1 — 경로 기본값을 정의 시점에 바인딩하면 monkeypatch 가 무효입니다.

    라벨링 테스트 17개가 통과하면서 실제로는 `data/candidates/` 의 719건을 읽고
    있었던 그 함정입니다. 여기서는 캐시가 저장소의 `data/cache/emb` 를 오염시킵니다.

    깨뜨리는 법: EmbeddingCache.__init__ 시그니처를
    `cache_dir: Path = DEFAULT_CACHE_DIR` 로 바꾸면(정의 시점 바인딩) 이 테스트가
    tmp 대신 data/cache/emb 를 가리켜 빨간불.
    확인일: 2026-08-18
    """
    monkeypatch.setattr(embed_module, "DEFAULT_CACHE_DIR", tmp_path / "emb")
    cached = CachedEmbedder(_spy_embedder())  # cache_dir 미지정 — 기본값 경로

    cached.encode(["hello"])

    assert cached.cache.dir == tmp_path / "emb" / "spy_abcdef01"
    assert list((tmp_path / "emb" / "spy_abcdef01").glob("*.npy"))


def test_cache_recovers_from_corrupt_file(tmp_path):
    """쓰다 만 파일이 남아 있으면 미스로 처리하고 다시 인코딩합니다.

    깨뜨리는 법: EmbeddingCache.get 의 try/except 를 지우면 ValueError 가 새어나와
    빨간불.
    확인일: 2026-08-18
    """
    spy = _spy_embedder()
    cached = CachedEmbedder(spy, cache_dir=tmp_path)
    cached.encode(["hello"])
    cached.cache.path_for("hello").write_bytes(b"\x93NUMPY broken")

    vectors = cached.encode(["hello"])

    assert spy.encode.call_count == 2
    assert len(vectors) == 1


# ── 모델 카탈로그 (R8 / §9.3) ────────────────────────────────────────────────


def test_load_model_spec_reads_pinned_revision_from_catalog():
    """★ 리비전은 `eval/models.yaml` 에서 읽습니다 (R8 / §9.3).

    깨뜨리는 법: load_model_spec 이 revision 을 `"main"` 으로 되돌리게 하면
    ModelSpec.__post_init__ 이 ValueError 를 내 빨간불.
    확인일: 2026-08-18
    """
    catalog = yaml.safe_load((REPO / "eval" / "models.yaml").read_text(encoding="utf-8"))
    spec = load_model_spec("embedder")
    assert spec.repo == catalog["embedder"]["repo"]
    assert spec.revision == catalog["embedder"]["revision"]
    assert len(spec.revision) == 40

    by_id = load_model_spec(catalog["reranker"]["id"])  # 최상위 키가 아니라 id 로도 찾습니다
    assert by_id.repo == catalog["reranker"]["repo"]
    assert by_id.raw["max_length"] == catalog["reranker"]["max_length"]


def test_model_spec_rejects_unpinned_revision(tmp_path):
    """★ §9.3 — `"main"` 은 재현성이 없습니다. 미머지 PR 가중치가 올 수 있습니다.

    깨뜨리는 법: ModelSpec.__post_init__ 의 _REVISION_RE 검사를 지우면 `"main"` 이
    통과해 빨간불.
    확인일: 2026-08-18
    """
    catalog = tmp_path / "models.yaml"
    catalog.write_text(
        yaml.safe_dump({"embedder": {"id": "x", "repo": "BAAI/bge-m3", "revision": "main"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="40자 커밋 해시"):
        load_model_spec("embedder", catalog)


def test_rank_sources_do_not_hardcode_revisions():
    """★ R8 — 코드에 해시를 타이핑하면 카탈로그와 어긋나도 아무도 모릅니다.

    깨뜨리는 법: src/rank/embed.py 어디든 models.yaml 의 40자 해시를 문자열로
    적어 넣으면 빨간불.
    확인일: 2026-08-18
    """
    catalog = yaml.safe_load((REPO / "eval" / "models.yaml").read_text(encoding="utf-8"))
    entries = [catalog["embedder"], catalog["reranker"], *catalog.get("optional_embedders", [])]
    revisions = {entry["revision"] for entry in entries}
    for source in (REPO / "src" / "rank").glob("*.py"):
        text = source.read_text(encoding="utf-8")
        for revision in revisions:
            assert revision not in text, f"{source.name} 에 리비전이 하드코딩돼 있습니다"


def test_bge_embedder_passes_pinned_revision_to_factory():
    """★ §9.3 — `SentenceTransformer(repo)` 만 부르면 미머지 PR 가중치가 옵니다.

    실모델 2.3GB 를 받지 않고 "리비전을 정말 넘기는가"만 검사합니다.

    깨뜨리는 법: BgeM3Embedder._load 의 `revision=self.spec.revision` 인자를 지우면
    TypeError(또는 revision 누락)로 빨간불.
    확인일: 2026-08-18
    """
    calls: list[dict[str, Any]] = []

    def factory(repo, *, revision, device):
        calls.append({"repo": repo, "revision": revision, "device": device})
        return SimpleNamespace(
            encode=lambda texts, **kwargs: [[0.1, 0.2] for _ in texts],
            max_seq_length=8192,
        )

    embedder = BgeM3Embedder(factory=factory, device="cpu")
    vectors = embedder.encode(["hello"])

    assert calls[0]["repo"] == "BAAI/bge-m3"
    assert len(calls[0]["revision"]) == 40
    assert calls[0]["device"] == "cpu"
    assert vectors == [(0.1, 0.2)]
    assert embedder.describe()["max_seq_length"] == 8192


def test_bge_embedder_does_not_import_torch_at_module_import():
    """§9.10 — Actions 러너에는 torch·sentence-transformers 가 없습니다.

    `import src.rank.embed` 만으로 죽으면 랭킹 단계가 통째로 멈춥니다.

    깨뜨리는 법: src/rank/embed.py 최상단에 `import torch` 를 추가하면
    (torch 없는 환경에서) import 자체가 실패하고, 있는 환경에서도 이 검사가 빨간불.
    확인일: 2026-08-18
    """
    source = (REPO / "src" / "rank" / "embed.py").read_text(encoding="utf-8")
    header = source.split("# ── 모델 스펙", 1)[0]
    assert "import torch" not in header
    assert "import sentence_transformers" not in header
    assert "from sentence_transformers" not in header


def test_embedder_protocol_conformance():
    """스텁과 캐시 래퍼가 포트 계약(§4.2)을 만족하는지."""
    assert isinstance(HashStubEmbedder(), Embedder)
    assert isinstance(CachedEmbedder(HashStubEmbedder()), Embedder)
    assert isinstance(FakeReranker(), Reranker)


# ── 점수 합성 (§5.2) ─────────────────────────────────────────────────────────


def _orthogonal_profile(count: int = DIM) -> tuple[dict[str, Any], CannedEmbedder]:
    """관심사 8개를 서로 직교시킨 프로파일 + 그 벡터를 내는 임베더."""
    interests = [(f"q{index}", f"interest text {index}", 1.0) for index in range(count)]
    table = {f"interest text {index}": basis(index) for index in range(count)}
    return make_profile(interests), CannedEmbedder(table)


def test_weighted_max_and_mean_choose_different_documents():
    """★ §5.2 — 합성 방식이 1등을 바꿉니다. 기본값은 **가중 최대**입니다.

    구성: 관심사 8개(직교).
      - `focused` 는 관심사 1개와 완전 일치 → 최대 1.0 / 평균 0.125
      - `diffuse` 는 8개 전부에 약하게 걸침 → 최대 0.354 / 평균 0.354
    가중 평균을 쓰면 "모든 관심사에 고르게 애매한" diffuse 가 이깁니다. 관심사가
    15개인 실제 프로파일에서는 희석이 더 심합니다.

    깨뜨리는 법: profile.py 의 DEFAULT_SCORING_METHOD 를 "weighted_mean" 으로 바꾸면
    기본 경로의 1등이 뒤집혀 빨간불.
    확인일: 2026-08-18
    """
    profile, embedder = _orthogonal_profile()
    queryset = build_queryset(embedder, profile=profile)

    focused = basis(0)
    diffuse = normalize([1.0] * DIM)

    assert DEFAULT_SCORING_METHOD == "weighted_max"
    max_focused = score_document(focused, queryset)
    max_diffuse = score_document(diffuse, queryset)
    assert max_focused.score == pytest.approx(1.0)
    assert max_diffuse.score == pytest.approx(1 / DIM**0.5)
    assert max_focused.score > max_diffuse.score
    assert max_focused.best_interest == "q0"

    mean_focused = score_document(focused, queryset, method="weighted_mean")
    mean_diffuse = score_document(diffuse, queryset, method="weighted_mean")
    assert mean_focused.score == pytest.approx(1 / DIM)
    assert mean_diffuse.score == pytest.approx(1 / DIM**0.5)
    assert mean_diffuse.score > mean_focused.score, "가중 평균에서는 희석된 문서가 이겨야 합니다"


def test_weighted_max_applies_interest_weight():
    """가중치가 곱해지지 않으면 0.6짜리 주변 관심사가 1.0짜리 핵심 관심사와 동등해집니다.

    깨뜨리는 법: score_document 의 `* interest.weight` 를 지우면 두 점수가 같아져 빨간불.
    확인일: 2026-08-18
    """
    profile = make_profile([("core", "core text", 1.0), ("side", "side text", 0.6)])
    embedder = CannedEmbedder({"core text": basis(0), "side text": basis(1)})
    queryset = build_queryset(embedder, profile=profile)

    assert score_document(basis(0), queryset).score == pytest.approx(1.0)
    side = score_document(basis(1), queryset)
    assert side.score == pytest.approx(0.6)
    assert side.best_interest == "side"


def test_exclude_soft_is_a_penalty_not_a_hard_filter():
    """★ §5.2 경고 — 감점이지 배제가 아닙니다.

    하드 배제로 바꾸면 경계 사례가 후보에서 통째로 사라지고, 골드셋에 관련으로
    라벨된 논문이 빠져 **Hit@5 의 상한 자체가 내려갑니다.** 프로파일의 `survey`
    (penalty 0.3)에 "완전 배제하지 않음 — 분야 입문에 유용" 이라고 적혀 있습니다.

    검사 2가지: (1) 점수가 정확히 `base - penalty_weight × cos×p` 만큼 깎인다
                (2) 감점을 맞고도 **결과 목록에 남는다**

    깨뜨리는 법: score_document 에서 `score=base - weight * penalty` 를
    `score=base` 로 바꾸면 (1)이, run_stage1 에 `penalty > 0` 을 버리는 필터를
    넣으면 (2)가 빨간불.
    확인일: 2026-08-18
    """
    profile = make_profile(
        [("core", "core text", 1.0)],
        exclude=[("excluded text", 0.5)],
    )
    embedder = CannedEmbedder({"core text": basis(0), "excluded text": basis(1)})
    queryset = build_queryset(embedder, profile=profile)

    # 관심사에 0.6, 배제어에 0.8 로 걸리는 문서
    doc = (0.6, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    breakdown = score_document(doc, queryset)

    assert breakdown.base == pytest.approx(0.6)
    assert breakdown.penalty == pytest.approx(0.4)  # 0.8 × 0.5
    assert breakdown.score == pytest.approx(0.2)  # 0.6 − 1.0 × 0.4
    assert breakdown.best_exclusion == "excluded text"

    # (2) 하드 배제가 아니므로 결과에 남아 있어야 합니다
    item = make_item("arxiv:1", "penalised")
    ranked = run_stage1(
        [item], CannedEmbedder({"penalised": doc}), queryset, top_n=5, min_score=None
    )
    assert [row.item_id for row in ranked] == ["arxiv:1"]


def test_penalty_weight_scales_and_zero_disables():
    """penalty_weight 는 M1에서 확정할 값입니다 (§5.2). 0이면 감점이 사라져야 합니다.

    깨뜨리는 법: score_document 의 `weight` 해석에서 `penalty_weight` 인자를 무시하고
    항상 queryset.penalty_weight 를 쓰게 하면 빨간불.
    확인일: 2026-08-18
    """
    profile = make_profile([("core", "core text", 1.0)], exclude=[("excluded text", 1.0)])
    embedder = CannedEmbedder({"core text": basis(0), "excluded text": basis(1)})
    queryset = build_queryset(embedder, profile=profile)
    doc = (0.6, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    assert score_document(doc, queryset, penalty_weight=0.0).score == pytest.approx(0.6)
    assert score_document(doc, queryset, penalty_weight=0.5).score == pytest.approx(0.2)


def test_penalty_is_clamped_at_zero():
    """★ 음수 감점은 **가점**이 됩니다 — "배제어와 정반대인 문서"에 보너스를 주는 꼴입니다.

    깨뜨리는 법: score_document 의 `penalty = max(0.0, penalty)` 를 지우면 점수가
    1.0 이 아니라 1.5 가 되어 빨간불.
    확인일: 2026-08-18
    """
    profile = make_profile([("core", "core text", 1.0)], exclude=[("excluded text", 0.5)])
    embedder = CannedEmbedder(
        {"core text": basis(0), "excluded text": tuple(-1.0 if i == 0 else 0.0 for i in range(DIM))}
    )
    queryset = build_queryset(embedder, profile=profile)

    breakdown = score_document(basis(0), queryset)
    assert breakdown.penalty == 0.0
    assert breakdown.score == pytest.approx(1.0)


def test_score_document_rejects_unknown_method():
    """오타 난 조건 이름이 조용히 기본값으로 처리되면 results.md 의 행이 거짓이 됩니다."""
    profile, embedder = _orthogonal_profile(2)
    queryset = build_queryset(embedder, profile=profile)
    with pytest.raises(ValueError, match="점수 합성"):
        score_document(basis(0), queryset, method="top3_mean")


# ── 프로파일 로딩 (R1 / §9.9) ────────────────────────────────────────────────


def test_build_queryset_resolves_highest_numbered_profile(tmp_path):
    """★ R1 / §9.9 — 경로를 하드코딩하면 `_1` 이 생기는 순간 구버전을 읽습니다.

    깨뜨리는 법: build_queryset 에서 load_profile 대신
    `yaml.safe_load(Path(root)/"profile.papers.yaml")` 로 읽게 하면 OLD 를 읽어 빨간불.
    확인일: 2026-08-18
    """
    (tmp_path / "profile.papers.yaml").write_text(
        yaml.safe_dump(make_profile([("old", "OLD interest", 1.0)]), allow_unicode=True),
        encoding="utf-8",
    )
    (tmp_path / "profile.papers_1.yaml").write_text(
        yaml.safe_dump(make_profile([("new", "NEW interest", 1.0)]), allow_unicode=True),
        encoding="utf-8",
    )

    queryset = build_queryset(HashStubEmbedder(dim=DIM), stem="profile.papers", root=tmp_path)

    assert [interest.id for interest in queryset.interests] == ["new"]
    assert queryset.profile_path.endswith("profile.papers_1.yaml")


def test_build_queryset_embeds_text_not_label():
    """★ label 은 사람이 읽는 필드입니다 — 프로파일 주석이 "임베딩하지 마세요" 라고 명시.

    한국어 label 을 붙여 임베딩하면 영어 초록과의 정렬이 흐려지고, 교차언어 대조
    실험(§5.5 조건 4)의 의미가 사라집니다.

    깨뜨리는 법: build_queryset 의 texts 를 `f"{i.id} {i.text}"` 같은 조합으로 바꾸면
    임베더가 받은 텍스트가 달라져 빨간불.
    확인일: 2026-08-18
    """
    profile = make_profile([("core", "core text", 1.0)], exclude=[("excluded text", 0.5)])
    embedder = CannedEmbedder()

    build_queryset(embedder, profile=profile)

    assert embedder.encode_calls == [["core text", "excluded text"]]
    assert not any("레이블" in text for call in embedder.encode_calls for text in call)


def test_parse_helpers_reject_broken_profiles():
    """프로파일이 망가졌으면 조용히 0건 관심사로 도는 대신 죽어야 합니다."""
    with pytest.raises(ValueError, match="interests"):
        parse_interests({})
    with pytest.raises(ValueError, match="text"):
        parse_interests({"interests": [{"id": "x", "text": "   "}]})
    with pytest.raises(ValueError, match="exclude.soft"):
        parse_exclusions({"exclude": {"soft": "not a list"}})
    assert parse_exclusions({}) == ()


def test_queryset_rejects_vector_count_mismatch():
    """임베더가 건수를 흘리면 문서와 벡터가 어긋납니다 — 조용히 zip 되면 안 됩니다."""
    with pytest.raises(ValueError, match="벡터"):
        QuerySet(
            interests=(Interest(id="a", text="a"), Interest(id="b", text="b")),
            interest_vectors=(basis(0),),
        )
    with pytest.raises(ValueError, match="exclude"):
        QuerySet(
            interests=(Interest(id="a", text="a"),),
            interest_vectors=(basis(0),),
            exclusions=(Exclusion(text="x"),),
            exclusion_vectors=(),
        )


# ── 1차 랭킹 (§5.4 / §5.7) ───────────────────────────────────────────────────


def test_document_text_combines_title_and_abstract():
    """제목만 쓰면 신호가 짧고 초록만 쓰면 제목의 단서를 버립니다.

    ★ 여기서 재치환(.format)을 하면 초록의 LaTeX 중괄호(10%)에서 터집니다 (§9.15).

    깨뜨리는 법: document_text 가 abstract 를 빼고 title 만 돌려주게 하면 빨간불.
    확인일: 2026-08-18
    """
    item = make_item("arxiv:1", "Title", r"bounds to \[ \frac{6\pi}{11} \]")
    text = document_text(item)
    assert text.startswith("Title")
    assert r"\frac{6\pi}{11}" in text, "LaTeX 중괄호가 그대로 살아 있어야 합니다"
    assert document_text(SimpleNamespace(title="T", abstract="A")) == "T\n\nA"
    assert document_text(make_item("arxiv:2", "", "only abstract")) == "only abstract"


def test_ties_break_by_item_id_lexicographically():
    """★ §5.4 — 동점은 `item_id` 사전순. 안 그러면 같은 조건이 다른 수치를 냅니다.

    깨뜨리는 법: retrieve.sort_ranked 의 정렬 키에서 `row[0]`(item_id)을 빼면
    입력 순서가 그대로 남아 두 번째 assert 가 빨간불.
    확인일: 2026-08-18
    """
    profile, embedder = _orthogonal_profile(2)
    queryset = build_queryset(embedder, profile=profile)

    same = "identical document"
    items = [make_item("arxiv:zz", same), make_item("arxiv:aa", same), make_item("arxiv:mm", same)]
    scorer = CannedEmbedder({same: basis(0)})

    forward = score_items(items, scorer, queryset)
    backward = score_items(list(reversed(items)), scorer, queryset)

    assert [row.item_id for row in forward] == ["arxiv:aa", "arxiv:mm", "arxiv:zz"]
    assert [row.item_id for row in backward] == [row.item_id for row in forward]
    assert [row.rank for row in forward] == [1, 2, 3]


def test_ranking_is_reproducible_across_runs():
    """M1 DoD (§5.8) — 같은 명령을 두 번 돌려 같은 수치가 나와야 합니다."""
    profile, embedder = _orthogonal_profile()
    queryset = build_queryset(embedder, profile=profile)
    items = [make_item(f"arxiv:{index}", f"doc {index}") for index in range(10)]

    first = [(row.item_id, round(row.score, 9)) for row in score_items(items, embedder, queryset)]
    second = [(row.item_id, round(row.score, 9)) for row in score_items(items, embedder, queryset)]
    assert first == second


def test_stage1_top_n_and_threshold_come_from_profile():
    """§5.1 / §5.7 — 설정에서 읽습니다. 코드에 30·0.42 를 박지 않습니다.

    깨뜨리는 법: stage1_settings 가 selection 을 무시하고 DEFAULT_STAGE1_TOP_N 을
    돌려주게 하면 top_n 2 를 못 읽어 빨간불.
    확인일: 2026-08-18
    """
    settings = stage1_settings(
        {"selection": {"stage1_top_n": 2, "final_n": 1, "min_score_threshold": 0.5}}
    )
    assert (settings.top_n, settings.final_n, settings.min_score) == (2, 1, 0.5)

    empty = stage1_settings(None)
    assert empty.top_n == 30 and empty.final_n == 5
    assert empty.min_score is None, "임계값 기본은 '없음' — 잠정치 0.42 를 코드에 박지 않습니다"


def test_none_means_disabled_not_unspecified():
    """★ `None`(끈다)과 미지정(`UNSET`)을 구분합니다.

    §5.7 은 "정답/오답 점수 분포를 그려 임계값을 정하라"고 요구합니다. 그 실행은
    **프로파일에 0.42가 있는 상태에서 임계값을 꺼야** 가능합니다. 두 뜻을 `None`
    하나로 뭉치면 그 조건을 표현할 수 없고, 분포를 못 보고 잠정치를 그대로 씁니다.

    깨뜨리는 법: stage1_settings 에서 `isinstance(min_score, _Unset)` 판정을
    `min_score is None` 으로 되돌리면 명시적 None 이 프로파일 값으로 덮여 빨간불.
    확인일: 2026-08-18
    """
    profile = {"selection": {"stage1_top_n": 7, "final_n": 3, "min_score_threshold": 0.42}}

    assert stage1_settings(profile).min_score == pytest.approx(0.42)
    assert stage1_settings(profile, min_score=None).min_score is None
    assert stage1_settings(profile, min_score=0.9).min_score == pytest.approx(0.9)
    assert stage1_settings(profile, top_n=None).top_n is None
    assert stage1_settings(profile, top_n=UNSET).top_n == 7


def test_real_profile_exposes_the_selection_keys_we_read():
    """운영 프로파일의 키 이름이 바뀌면 랭킹이 조용히 기본값으로 돕니다 (R1).

    값 자체는 검사하지 않습니다 — `min_score_threshold` 는 M1 평가로 바뀔 값입니다(§5.7).

    ★ 실제 프로파일의 `stage1_top_n` 은 30이고 모듈 기본값도 30입니다. 그래서
      "프로파일에서 읽었다"와 "기본값으로 떨어졌다"가 구별되지 않습니다 —
      작업규약 §4.2로 실측했더니 설정을 통째로 무시해도 이 테스트가 통과했습니다.
      그래서 기본값과 겹치지 않는 탐침 값으로 한 번 더 확인합니다.

    깨뜨리는 법: retrieve.stage1_settings 가 selection 을 무시하게 하면 탐침 검사가
    빨간불. 읽는 키 이름을 `top_n` 등으로 바꿔도 빨간불.
    확인일: 2026-08-18
    """
    profile, path = load_profile("profile.papers")
    selection = profile["selection"]
    assert path.name.startswith("profile.papers")
    assert {"stage1_top_n", "final_n", "min_score_threshold"} <= set(selection)

    settings = stage1_settings(profile)
    assert settings.top_n == selection["stage1_top_n"]
    assert settings.final_n == selection["final_n"]
    assert settings.min_score == pytest.approx(selection["min_score_threshold"])

    probe = {
        **profile,
        "selection": {**selection, "stage1_top_n": 7, "final_n": 3, "min_score_threshold": 0.99},
    }
    probed = stage1_settings(probe)
    assert (probed.top_n, probed.final_n) == (7, 3)
    assert probed.min_score == pytest.approx(0.99)


def test_select_stage1_filters_before_truncating():
    """§5.7 — 거른 뒤 자릅니다. 자른 뒤 거르면 "왜 5건이 안 나오나"를 추적할 수 없습니다.

    ★ 참고: 프로파일의 0.42 는 잠정치이고 bge-m3 실측 스케일(무관한 텍스트도 0.5~0.6)
      에서는 거의 전부를 통과시킵니다. 임계값은 M1에서 정답 95% 지점으로 다시 잡습니다.

    깨뜨리는 법: select_stage1 에서 자르기(`kept[:limit]`)를 임계 필터보다 앞으로
    옮기면 결과가 ["a"] 하나가 되어 빨간불.
    확인일: 2026-08-18
    """
    rows = [
        RankedItem("a", 0.90, 0.90, 0.0, "q0", 1, None),
        RankedItem("b", 0.30, 0.30, 0.0, "q0", 2, None),
        RankedItem("c", 0.80, 0.80, 0.0, "q0", 3, None),
    ]
    kept = select_stage1(rows, top_n=2, min_score=0.5)
    assert [row.item_id for row in kept] == ["a", "c"]

    assert [row.item_id for row in select_stage1(rows, top_n=2, min_score=None)] == ["a", "b"]


def test_score_items_rejects_vector_count_mismatch():
    """임베더가 건수를 흘리면 문서와 벡터가 어긋난 채 순위가 나옵니다 — 죽어야 합니다."""

    class LosingEmbedder:
        model_id = "losing"
        revision = "0" * 12

        def encode(self, texts):
            return HashStubEmbedder(dim=DIM).encode(list(texts)[:-1])

    profile, embedder = _orthogonal_profile(2)
    queryset = build_queryset(embedder, profile=profile)
    with pytest.raises(RuntimeError, match="어긋납니다"):
        score_items([make_item("a", "x"), make_item("b", "y")], LosingEmbedder(), queryset)


def test_score_items_requires_item_id():
    """id 없는 후보는 원장·골드셋과 연결할 수 없습니다."""
    profile, embedder = _orthogonal_profile(2)
    queryset = build_queryset(embedder, profile=profile)
    with pytest.raises(ValueError, match="id"):
        score_items([{"title": "no id"}], embedder, queryset)


# ── 2차 랭킹 (§9.4 raw 로짓) ────────────────────────────────────────────────


def test_fake_reranker_returns_raw_logits_not_probabilities():
    """★ §9.4 — 리랭커 점수는 raw 로짓입니다. 무관한 쌍은 크게 음수입니다.

    sigmoid 를 씌우면 무관한 쌍이 전부 0 근처로 뭉쳐 원장(`rank_score_stage2`)을
    사람이 읽을 수 없고 임계값도 못 정합니다. 실측: 로짓 -10.08~-2.51 / sigmoid
    0.0000~0.0751.

    깨뜨리는 법: FakeReranker._INTERCEPT 를 0.0 으로, _SLOPE 를 1.0 으로 바꾸면
    점수가 [0,1] 안에 들어와 빨간불.
    확인일: 2026-08-18
    """
    scores = FakeReranker().score_pairs(
        [
            ("identical query", "identical query"),
            ("identical query", "완전히 다른 문서"),
            ("identical query", "identical document text"),
        ]
    )
    assert scores[0] > scores[2] > scores[1]
    assert scores[1] < 0.0, "무관한 쌍은 음수 로짓이어야 합니다"
    assert not looks_like_probabilities(scores)
    assert looks_like_probabilities([0.0001, 0.03, 0.0751])
    assert not looks_like_probabilities([0.2, 0.3]), "표본 2건으로는 판정하지 않습니다"


def test_bge_reranker_passes_identity_activation_and_pinned_revision():
    """★★ §9.4 + §9.3 — CrossEncoder 는 기본으로 sigmoid 를 씌웁니다.

    실모델을 받지 않고 생성 인자만 검사합니다: Identity 활성화 · 고정 리비전 ·
    models.yaml 의 max_length(512).

    깨뜨리는 법: BgeReranker._load 의 kwargs 에서 `"activation_fn"` 줄을 지우면
    (= 기본 sigmoid) 빨간불. `max_length` 를 512 리터럴로 박아도 카탈로그와
    어긋나는 순간 빨간불.
    확인일: 2026-08-18
    """
    sentinel = object()
    calls: list[dict[str, Any]] = []

    def factory(repo, **kwargs):
        calls.append({"repo": repo, **kwargs})
        return SimpleNamespace(predict=lambda pairs, **kw: [-3.5 for _ in pairs])

    reranker = BgeReranker(factory=factory, activation=sentinel, device="cpu")
    scores = reranker.score_pairs([("q", "d")])

    catalog = yaml.safe_load((REPO / "eval" / "models.yaml").read_text(encoding="utf-8"))
    assert calls[0]["repo"] == catalog["reranker"]["repo"]
    assert calls[0]["revision"] == catalog["reranker"]["revision"]
    assert calls[0]["max_length"] == catalog["reranker"]["max_length"]
    assert calls[0]["activation_fn"] is sentinel, "Identity 를 넘기지 않으면 sigmoid 가 씌워집니다"
    assert scores == [-3.5]


def test_bge_reranker_retries_with_legacy_activation_kwarg():
    """sentence-transformers 4.x 이하는 이름이 `activation_fct` 였습니다.

    이름이 안 맞을 때 **인자를 빼고 재시도하면 조용히 sigmoid** 가 됩니다 — 그게
    §9.4 그 자체입니다. 이름만 바꿔 다시 넘겨야 합니다.

    깨뜨리는 법: _load 의 except 절에서 `kwargs["activation_fct"] = kwargs.pop(...)`
    를 그냥 `kwargs.pop("activation_fn")` 으로 바꾸면 활성화가 사라져 빨간불.
    확인일: 2026-08-18
    """
    sentinel = object()
    calls: list[dict[str, Any]] = []

    def picky_factory(repo, **kwargs):
        calls.append(dict(kwargs))
        if "activation_fn" in kwargs:
            raise TypeError("__init__() got an unexpected keyword argument 'activation_fn'")
        return SimpleNamespace(predict=lambda pairs, **kw: [-2.0 for _ in pairs])

    BgeReranker(factory=picky_factory, activation=sentinel, device="cpu").score_pairs([("q", "d")])

    assert len(calls) == 2
    assert calls[1]["activation_fct"] is sentinel
    assert "activation_fn" not in calls[1]


def test_bge_reranker_dies_when_scores_look_like_sigmoid():
    """★ §9.4 2중 방어 — 누가 Identity 를 되돌려도 sigmoid 값이 원장에 들어가면 안 됩니다.

    깨뜨리는 법: score_pairs 의 `looks_like_probabilities` 검사 블록을 지우면
    sigmoid 값이 그대로 통과해 빨간불.
    확인일: 2026-08-18
    """

    def sigmoid_factory(repo, **kwargs):
        return SimpleNamespace(predict=lambda pairs, **kw: [0.0001, 0.0312, 0.0751, 0.0002])

    reranker = BgeReranker(factory=sigmoid_factory, activation=object(), device="cpu")
    with pytest.raises(RuntimeError, match="sigmoid"):
        reranker.score_pairs([("q", f"d{index}") for index in range(4)])


def test_rerank_stage2_reorders_truncates_and_uses_best_interest_as_query():
    """§5.1 — 1차 상위 N 을 cross-encoder 로 재정렬해 final_n 건만 남깁니다.

    질의는 1차에서 그 문서를 끌어올린 관심사의 text 입니다 (모듈 독스트링 참조).

    깨뜨리는 법: rerank_stage2 의 정렬 키 `-triple[2]` 에서 부호를 빼면 순서가
    뒤집혀 빨간불. query_for 가 best_interest 를 무시하고 첫 관심사를 쓰게 해도 빨간불.
    확인일: 2026-08-18
    """
    profile = make_profile([("q0", "first interest", 1.0), ("q1", "second interest", 1.0)])
    embedder = CannedEmbedder({"first interest": basis(0), "second interest": basis(1)})
    queryset = build_queryset(embedder, profile=profile)

    stage1 = [
        RankedItem("arxiv:a", 0.9, 0.9, 0.0, "q0", 1, make_item("arxiv:a", "doc a")),
        RankedItem("arxiv:b", 0.8, 0.8, 0.0, "q1", 2, make_item("arxiv:b", "doc b")),
        RankedItem("arxiv:c", 0.7, 0.7, 0.0, "q1", 3, make_item("arxiv:c", "doc c")),
    ]
    reranker = CannedReranker([-8.0, -1.0, -4.0])

    result = rerank_stage2(stage1, reranker, queryset, final_n=2)

    assert [row.item_id for row in result] == ["arxiv:b", "arxiv:c"]
    assert [row.rank for row in result] == [1, 2]
    assert result[0].stage1_score == 0.8 and result[0].stage2_score == -1.0
    assert result[0].query_id == "q1"
    assert reranker.seen[0] == ("first interest", "doc a")
    assert reranker.seen[1] == ("second interest", "doc b")


def test_rerank_stage2_breaks_ties_by_item_id():
    """§5.4 — 2차도 같은 계약입니다.

    깨뜨리는 법: rerank_stage2 의 정렬 키에서 `triple[0].item_id` 를 빼면 입력 순서가
    남아 빨간불.
    확인일: 2026-08-18
    """
    profile = make_profile([("q0", "only interest", 1.0)])
    queryset = build_queryset(CannedEmbedder({"only interest": basis(0)}), profile=profile)
    stage1 = [
        RankedItem(item_id, 0.5, 0.5, 0.0, "q0", index, make_item(item_id, "doc"))
        for index, item_id in enumerate(["arxiv:zz", "arxiv:aa", "arxiv:mm"], start=1)
    ]

    result = rerank_stage2(stage1, CannedReranker([-3.0, -3.0, -3.0]), queryset, final_n=3)

    assert [row.item_id for row in result] == ["arxiv:aa", "arxiv:mm", "arxiv:zz"]


def test_rerank_stage2_rejects_score_count_mismatch():
    """리랭커가 점수를 흘리면 문서와 점수가 어긋난 채 발행됩니다 — 죽어야 합니다."""
    profile = make_profile([("q0", "only interest", 1.0)])
    queryset = build_queryset(CannedEmbedder({"only interest": basis(0)}), profile=profile)
    stage1 = [
        RankedItem("arxiv:a", 0.5, 0.5, 0.0, "q0", 1, make_item("arxiv:a", "doc a")),
        RankedItem("arxiv:b", 0.4, 0.4, 0.0, "q0", 2, make_item("arxiv:b", "doc b")),
    ]
    with pytest.raises(RuntimeError, match="흘렸습니다"):
        rerank_stage2(stage1, CannedReranker([-1.0]), queryset)


def test_rerank_stage2_is_empty_for_empty_stage1():
    """조용한 날(1차 통과 0건)에 예외가 아니라 빈 목록이어야 합니다 — 원장의
    `summaries: []` 와 `null` 구분(§4.4 규칙 3)이 여기서 갈립니다."""
    profile = make_profile([("q0", "only interest", 1.0)])
    queryset = build_queryset(CannedEmbedder({"only interest": basis(0)}), profile=profile)
    assert rerank_stage2([], FakeReranker(), queryset) == []


# ── 전 구간 배선 ─────────────────────────────────────────────────────────────


def test_full_stage1_to_stage2_pipeline_with_stubs(tmp_path):
    """키 0개 · 네트워크 0회로 수집→1차→2차가 도는지 (델타 §D6.2).

    깨뜨리는 법: run_stage1 이 select_stage1 을 거치지 않게 하면 top_n 이 안 먹어
    빨간불.
    확인일: 2026-08-18
    """
    profile, path = load_profile("profile.papers")
    embedder = CachedEmbedder(HashStubEmbedder(dim=256), cache_dir=tmp_path)
    queryset = build_queryset(embedder, profile=profile, profile_path=path)

    items = [
        make_item(
            "arxiv:2608.00001",
            "Cross-encoder reranking for retrieval augmented generation",
            "We evaluate hybrid retrieval with Hit@k, MRR and nDCG on domain corpora.",
        ),
        make_item(
            "arxiv:2608.00002",
            "Quantum entanglement distillation",
            "A quantum computing protocol with convergence proofs and no empirical system.",
        ),
        make_item(
            "arxiv:math.GT/0309136",
            "Legacy identifier paper",
            "Old-style arXiv id must survive ranking untouched.",
        ),
    ]

    ranked = run_stage1(items, embedder, queryset, profile=profile, top_n=3, min_score=None)
    assert len(ranked) == 3
    assert ranked[0].best_interest  # 어느 관심사가 끌어올렸는지 항상 남습니다
    assert "arxiv:math.GT/0309136" in [row.item_id for row in ranked], "§9.5 구형 ID 보존"
    # top_n 이 실제로 먹는지 — run_stage1 이 select_stage1 을 거치지 않으면 3건이 나옵니다
    assert len(run_stage1(items, embedder, queryset, top_n=1, min_score=None)) == 1

    final = rerank_stage2(ranked, FakeReranker(), queryset, final_n=2)
    assert len(final) == 2
    assert all(isinstance(row, RerankedItem) for row in final)
    assert all(row.stage2_score <= 2.0 for row in final)

    # 두 번째 실행은 전부 캐시 적중 — 조건을 여러 개 돌릴 수 있는 이유입니다 (§5.3)
    misses_before = embedder.misses
    run_stage1(items, embedder, queryset, profile=profile, top_n=3, min_score=None)
    assert embedder.misses == misses_before


# ── 실모델 (기본 실행에서 제외) ──────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.network
def test_real_models_load_with_pinned_revision():
    """실모델 배선 확인 — 가중치 2.3GB 를 받으므로 `slow` + `network` 입니다.

    `network` 를 함께 붙인 이유: pyproject 의 `addopts = "-m 'not network'"` 가
    기본 실행에서 빼 주고, 가중치 다운로드는 실제 네트워크 접근이기 때문입니다.

    실행: RADAR_PY -m pytest tests/test_ranking.py -m slow -p no:cacheprovider
    확인일: 2026-08-18 (기본 실행에서 제외되는 것만 확인)
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("sentence_transformers")

    embedder = BgeM3Embedder(device="cpu")
    vectors = embedder.encode(["retrieval augmented generation", "quantum entanglement"])
    assert len(vectors[0]) == load_model_spec("embedder").raw["dim"]
    assert cosine(vectors[0], vectors[1]) < 1.0

    reranker = BgeReranker(device="cpu")
    scores = reranker.score_pairs(
        [
            ("retrieval augmented generation", "hybrid retrieval with BM25 and dense embeddings"),
            ("retrieval augmented generation", "a quantum computing convergence proof"),
            ("retrieval augmented generation", "protein structure prediction"),
        ]
    )
    assert scores[0] > scores[1], "관련 쌍이 더 높아야 합니다"
    assert min(scores) < 0.0, "raw 로짓이면 음수가 나옵니다 (§9.4)"
    assert isinstance(reranker._identity_activation(), torch.nn.Identity)


def test_model_spec_requires_repo():
    """카탈로그에 repo 가 비면 조용히 빈 이름으로 로드를 시도하지 않습니다."""
    with pytest.raises(ValueError, match="repo"):
        ModelSpec(id="x", repo="", revision="0" * 40)
