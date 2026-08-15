"""발행·법적 게이트 (기획안 §10 + 델타 §D8).

★ 표시된 게이트는 법적·설계 원칙에 직결됩니다. 비활성화하지 마세요.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from src.core import ledger
from src.core.models import Item

REPO = Path(__file__).resolve().parents[1]


def _tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
    )
    return out.stdout.splitlines()


# ── 법적 게이트 ★ ────────────────────────────────────────────────────────


def test_jobs_never_committed():
    """★ `data/jobs/` 하위 파일이 git에 추적되면 실패 (기획안 §8 M5).

    채용공고 원문을 공개 저장소에 커밋하면 사람인 약관 조항 6(제휴 관계 오인)에
    걸립니다. 무료여도 걸립니다 (델타 §D4).
    """
    leaked = [f for f in _tracked_files() if f.startswith("data/jobs/")]
    assert not leaked, f"채용공고 원문이 git에 추적되고 있습니다: {leaked}"


def test_jobs_profile_ignored():
    """★ `profile.jobs*.yaml`이 추적되면 실패. 구직 조건이 그대로 노출됩니다 (기획안 §7)."""
    leaked = [f for f in _tracked_files() if "profile.jobs" in f]
    assert not leaked, f"구직 프로파일이 git에 추적되고 있습니다: {leaked}"


def test_no_fulltext_stored():
    """★ `data/` 하위에 .pdf가 존재하면 실패 (델타 §D1·§D8).

    arXiv 전문은 대부분 재배포 불가 라이선스입니다. 다운로드도 캐시도 금지입니다.
    """
    pdfs = list((REPO / "data").rglob("*.pdf"))
    assert not pdfs, f"전문 PDF가 저장돼 있습니다: {pdfs}"


def test_actions_requirements_have_no_local_models():
    """★ 기획안 §5.1-(1) — Actions 러너에서 로컬 임베딩을 돌리지 않습니다.

    러너가 4 vCPU·GPU 없음이라 500건 CPU 인코딩이 실행시간의 대부분을 잡아먹습니다.
    `requirements.txt`에 torch가 들어가면 **매일 도는 워크플로가 매번 1GB를 설치**합니다.

    M1 평가용 로컬 모델은 `requirements-eval.txt`로 분리돼 있습니다 (델타 §D6.3).
    두 파일을 합치려는 시도를 여기서 막습니다.
    """
    runtime = (REPO / "requirements.txt").read_text(encoding="utf-8")
    banned = ["torch", "sentence-transformers", "transformers", "FlagEmbedding"]
    lines = [
        line.strip()
        for line in runtime.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    offenders = [line for line in lines if any(pkg in line for pkg in banned)]
    assert not offenders, (
        f"운영 requirements.txt 에 로컬 모델 의존성이 있습니다: {offenders}. "
        "requirements-eval.txt 로 옮기세요 (기획안 §5.1-(1))"
    )


def test_eval_requirements_exist_and_are_pinned():
    """평가 재현성 — 버전이 고정돼 있지 않으면 나중에 같은 수치가 안 나옵니다."""
    path = REPO / "requirements-eval.txt"
    assert path.exists(), "requirements-eval.txt 가 없습니다"
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert lines, "평가 의존성이 비어 있습니다"
    unpinned = [line for line in lines if "==" not in line]
    assert not unpinned, f"버전이 고정되지 않았습니다: {unpinned}"


def test_no_secret_leak():
    """★ 사람인 약관 조항 5 — access-key 공개·공유 금지 (델타 §D3).

    public repo라 유출이 곧 약관 위반입니다. 추적 중인 파일에 키 형태 문자열이
    없는지 확인합니다.
    """
    patterns = [
        "sk-ant-",
        "sk-or-",
        "ANTHROPIC_API_KEY=",
        "SARAMIN_ACCESS_KEY=",
        "TELEGRAM_BOT_TOKEN=",
    ]
    offenders: list[str] = []
    for name in _tracked_files():
        path = REPO / name
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        # 이 테스트 파일 자신은 패턴 목록을 들고 있으므로 제외합니다.
        if path.resolve() == Path(__file__).resolve():
            continue
        for pattern in patterns:
            if pattern in text:
                offenders.append(f"{name}: {pattern}")
    assert not offenders, f"키로 보이는 문자열이 추적 파일에 있습니다: {offenders}"


def test_publish_scope_literal_on_job_sources():
    """★ 델타 §D4 — jobs 소스는 "private" 외의 publish_scope를 낼 수 없어야 합니다.

    M5에서 saramin.py / worknet.py 가 추가되면 이 테스트가 자동으로 검사합니다.
    설정 수정만으로 공고가 공개 싱크에 흘러가는 경로를 막습니다.
    """
    job_modules = [
        p for p in (REPO / "src" / "sources").glob("*.py") if p.stem in {"saramin", "worknet"}
    ]
    for module in job_modules:
        text = module.read_text(encoding="utf-8")
        assert 'publish_scope = "private"' in text, (
            f"{module.name} 이 publish_scope를 'private' 리터럴로 고정하지 않았습니다"
        )
        assert 'publish_scope = "public"' not in text


def test_item_rejects_missing_link():
    """델타 §D1 — arXiv 이용약관 권고에 따라 원문 링크는 필수입니다."""
    with pytest.raises(ValueError, match="원문 링크"):
        Item(
            id="arxiv:1",
            source="arxiv",
            channel="papers",
            title="t",
            abstract="a",
            url="",
            published="2026-08-12T00:00:00Z",
            updated="2026-08-12T00:00:00Z",
            publish_scope="public",
        )


def test_item_rejects_unknown_scope():
    with pytest.raises(ValueError, match="publish_scope"):
        Item(
            id="arxiv:1",
            source="arxiv",
            channel="papers",
            title="t",
            abstract="a",
            url="https://arxiv.org/abs/1",
            published="2026-08-12T00:00:00Z",
            updated="2026-08-12T00:00:00Z",
            publish_scope="internal",  # type: ignore[arg-type]
        )


# ── 원장 스키마 (결정_M0 §1) ─────────────────────────────────────────────


def _record(**overrides) -> ledger.RunRecord:
    base = dict(
        run_id="2026-08-12T07:00:00+09:00/papers",
        channel="papers",
        stage="collect",
        profile="data/profile.papers_1.yaml",
        sources={"arxiv": 480},
        collected=480,
        after_dedup=471,
        duration_sec=18.2,
    )
    base.update(overrides)
    return ledger.RunRecord(**base)


def test_ledger_schema_is_one():
    """2로 시작하면 세상에 없는 schema 1을 처리하는 리더 코드를 짜게 됩니다."""
    assert _record().to_dict()["schema"] == 1


def test_ledger_rejects_unknown_stage():
    """허용값을 고정하지 않으면 `collect_only`, `COLLECT` 변종이 섞입니다."""
    for bad in ("collect_only", "COLLECT", "", "ingest"):
        with pytest.raises(ValueError, match="stage"):
            _record(stage=bad)


def test_ledger_accepts_every_declared_stage():
    for stage in ledger.STAGES:
        assert _record(stage=stage).to_dict()["stage"] == stage


def test_ledger_unrun_stages_are_null_not_missing():
    """리더가 row["selected"] 에서 KeyError를 내면 안 됩니다 (결정_M0 §1-(2))."""
    row = _record().to_dict()
    for key in ("stage1_top_n", "selected", "summaries"):
        assert key in row
        assert row[key] is None


def test_ledger_keeps_original_schema_keys():
    """기획안 §7의 키를 삭제하거나 개명하지 않았는지."""
    row = _record().to_dict()
    for key in (
        "schema",
        "run_id",
        "channel",
        "sources",
        "collected",
        "after_dedup",
        "stage1_top_n",
        "selected",
        "summaries",
        "gates",
        "duration_sec",
        "cost_usd",
    ):
        assert key in row, f"기획안 §7 키가 사라졌습니다: {key}"


def test_ledger_records_profile_path():
    """어느 프로파일로 돈 결과인지 못 밝히면 기획안 §9 평가가 재현 불가입니다."""
    assert _record().to_dict()["profile"] == "data/profile.papers_1.yaml"


def test_ledger_append_and_read(tmp_path):
    path = tmp_path / "runs.jsonl"
    ledger.append(_record(), path)
    ledger.append(_record(stage="full", selected=5, summaries=[]), path)
    rows = ledger.read_all(path)
    assert len(rows) == 2
    # 수집만 한 실행과 전 구간 돌고 0건 선정한 실행이 구분됩니다.
    assert rows[0]["stage"] == "collect" and rows[0]["summaries"] is None
    assert rows[1]["stage"] == "full" and rows[1]["summaries"] == []


# ── M1 평가 재현성 (eval/models.yaml) ────────────────────────────────────


def test_eval_models_are_revision_pinned():
    """★ 모델 리비전이 40자 커밋 해시로 고정돼 있는지 (기획안 §9 재현성).

    실제로 밟은 함정입니다. `SentenceTransformer("BAAI/bge-m3")` 를 리비전 없이
    호출했더니 sentence-transformers 가 safetensors 를 선호해 **머지되지 않은
    PR(`refs/pr/130`)에서 가중치를 받아갔습니다.** PR은 닫히거나 바뀔 수 있으므로
    그 상태로 낸 Hit@5는 나중에 재현되지 않습니다.
    """
    import re

    import yaml as _yaml

    cfg = _yaml.safe_load((REPO / "eval" / "models.yaml").read_text(encoding="utf-8"))
    entries = [cfg["embedder"], cfg["reranker"], *cfg.get("optional_embedders", [])]
    for entry in entries:
        revision = entry.get("revision", "")
        assert re.fullmatch(r"[0-9a-f]{40}", revision), (
            f"{entry['repo']} 의 revision 이 커밋 해시로 고정돼 있지 않습니다: {revision!r}"
        )
        assert entry.get("license"), f"{entry['repo']} 라이선스가 기록되지 않았습니다 (CLAUDE.md §3)"


def test_eval_models_not_committed_as_weights():
    """★ 모델 가중치가 저장소에 들어오면 실패. 캐시는 ~/.cache/huggingface 에 둡니다."""
    weights = [f for f in _tracked_files() if f.endswith((".bin", ".safetensors", ".pt", ".onnx"))]
    assert not weights, f"모델 가중치가 git에 추적되고 있습니다: {weights}"
