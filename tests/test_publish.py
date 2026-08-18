"""Hugo 발행기 게이트 (기획안_2 §7.2 · §10).

★ `test_frontmatter_schema` 와 scope 게이트 테스트는 비활성화하지 마세요.
  전자는 arXiv 이용약관 권고(abs 링크 필수, §3.1)를, 후자는 사람인 약관 조항 6
  (채용공고 공개 게시 금지 — 무료여도 위반)을 코드로 고정한 것입니다.

summary/verdict 는 아직 실호출이 없으므로 dict 픽스처를 씁니다.
정상 요약은 `tests/fixtures/distorted_summaries.yaml` 의 faithful 케이스를 재사용합니다.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.publish.markdown import post_filename, render_post, write_posts
from src.summarize.schema import FAITHFULNESS_THRESHOLD
from src.verify.gates import ScopeViolation

FIXTURE = Path(__file__).parent / "fixtures" / "distorted_summaries.yaml"


@pytest.fixture(scope="module")
def samples() -> dict:
    return yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture
def item(samples) -> dict:
    src = samples["source"]
    return {
        "id": src["item_id"],
        "source": "arxiv",
        "channel": "papers",
        "title": src["title"],
        "abstract": src["abstract"],
        "url": src["url"],
        "published": "2026-08-12T04:11:02+00:00",
        "updated": "2026-08-12T04:11:02+00:00",
        "publish_scope": "public",
        "authors": [],
        "categories": ["cs.CV", "cs.AI"],
        "fulltext_ok": False,
        "raw": {},
    }


@pytest.fixture
def summary(samples) -> dict:
    return dict(samples["faithful"]["summary"])


@pytest.fixture
def verdict() -> dict:
    return {"faithfulness": 0.93, "unsupported_claims": [], "verifier_model": "openai/gpt-5.6-luna"}


def _frontmatter(post: str) -> dict:
    """포스트 문자열에서 프론트매터만 YAML 로 파싱합니다."""
    assert post.startswith("---\n"), "프론트매터 구분선이 없습니다"
    _, front, _body = post.split("---\n", 2)
    return yaml.safe_load(front)


# ── 프론트매터 스키마 ★ ──────────────────────────────────────────────────


def test_frontmatter_schema(item, summary, verdict):
    """★ 필수 7필드가 전부, 계약된 이름으로 들어가야 합니다 (기획안_2 §7.2).

    깨뜨리는 법: markdown.py 의 front dict 에서 아무 키나 하나 빼면 (예: arxiv_id)
    RuntimeError("프론트매터 키가 계약과 다릅니다")로 빨간불.
    확인일: 2026-08-18
    """
    front = _frontmatter(render_post(item, summary, verdict))
    assert list(front) == [
        "title",
        "date",
        "arxiv_id",
        "arxiv_abs_url",
        "categories",
        "faithfulness",
        "flagged",
    ]
    assert front["title"] == item["title"]
    assert front["date"] == item["published"]
    assert front["arxiv_id"] == "arxiv:2608.11053"
    assert front["arxiv_abs_url"] == "https://arxiv.org/abs/2608.11053"
    assert front["categories"] == ["cs.CV", "cs.AI"]
    assert front["faithfulness"] == 0.93
    assert front["flagged"] is False


@pytest.mark.parametrize("missing", ["title", "published", "id", "url", "categories"])
def test_frontmatter_schema_rejects_missing_item_field(item, summary, verdict, missing):
    """★ 프론트매터를 채울 수 없는 아이템은 ValueError (기획안 §10 test_frontmatter_schema).

    깨뜨리는 법: _require_item_fields 의 해당 검사 튜플에서 한 줄을 지우면
    그 파라미터 케이스가 빨간불 (ValueError 대신 KeyError 또는 조용한 통과).
    확인일: 2026-08-18
    """
    item.pop(missing)
    with pytest.raises(ValueError, match="프론트매터"):
        render_post(item, summary, verdict)


def test_frontmatter_rejects_missing_faithfulness(item, summary):
    """깨뜨리는 법: _verdict_from_dict 의 None 검사를 지우면 TypeError 로 어긋난 실패.
    확인일: 2026-08-18
    """
    with pytest.raises(ValueError, match="faithfulness"):
        render_post(item, summary, {"verifier_model": "openai/gpt-5.6-luna"})


def test_summary_missing_core_fields_rejected(item, summary, verdict):
    """요약 검증은 schema.PaperSummary.from_dict 에 위임합니다 (R4).

    깨뜨리는 법: render_post 에서 PaperSummary.from_dict 호출을 dict 직접 접근으로
    바꾸면 빨간불 (빈 method 가 그대로 발행됨).
    확인일: 2026-08-18
    """
    summary["method"] = ""
    with pytest.raises(ValueError, match="필수 필드"):
        render_post(item, summary, verdict)


# ── arXiv 링크 하드 게이트 ★ ─────────────────────────────────────────────


@pytest.mark.parametrize("bad_url", ["", None])
def test_refuses_to_publish_without_abs_url(item, summary, verdict, bad_url):
    """★ arxiv_abs_url 이 비어 있으면 발행 거부 — arXiv 이용약관 권고
    "Direct users to arXiv.org" (기획안_2 §3.1). 함정이 아니라 하드 게이트입니다.

    깨뜨리는 법: _require_item_fields 의 ("url", "arxiv_abs_url") 줄을 지우면 빨간불.
    확인일: 2026-08-18
    """
    item["url"] = bad_url
    with pytest.raises(ValueError):
        render_post(item, summary, verdict)


@pytest.mark.parametrize(
    "pdf_url",
    ["https://arxiv.org/pdf/2608.11053", "https://arxiv.org/pdf/2608.11053v1.pdf"],
)
def test_refuses_pdf_links(item, summary, verdict, pdf_url):
    """PDF 링크는 abs 링크의 대체물이 아닙니다 — 전문은 대부분 재배포 불가
    (CLAUDE.md §3-4 / 델타 §D1).

    깨뜨리는 법: _require_item_fields 의 "/pdf/" 검사를 지우면 빨간불.
    확인일: 2026-08-18
    """
    item["url"] = pdf_url
    with pytest.raises(ValueError, match="PDF"):
        render_post(item, summary, verdict)


# ── scope 게이트 ★ ───────────────────────────────────────────────────────


def test_write_posts_rejects_private_scope(tmp_path, item, summary, verdict):
    """★ publish_scope != "public" 아이템은 ScopeViolation — 사람인 약관 조항 6
    (제휴 관계 오인 금지, 무료여도 위반). gates.assert_public_scope 를 그대로
    호출해야 합니다 (R5). 재구현 금지.

    깨뜨리는 법: write_posts 의 assert_public_scope(...) 호출을 지우면 빨간불.
    확인일: 2026-08-18
    """
    item["publish_scope"] = "private"
    with pytest.raises(ScopeViolation):
        write_posts([(item, summary, verdict)], site_dir=tmp_path)


def test_scope_gate_fires_before_any_write(tmp_path, item, summary, verdict):
    """★ 섞인 배치에서 한 건이라도 비공개면 **아무 파일도 쓰지 않아야** 합니다.
    공개분 먼저 쓰고 나서 터지면 이미 유출 경로가 생긴 뒤입니다.

    깨뜨리는 법: write_posts 에서 assert_public_scope 호출을 파일 쓰기 루프
    뒤로 옮기면 빨간불 (공개 1건이 이미 디스크에 남음).
    확인일: 2026-08-18
    """
    private_item = dict(item, id="arxiv:9999.00001", publish_scope="private")
    with pytest.raises(ScopeViolation):
        write_posts([(item, summary, verdict), (private_item, summary, verdict)], site_dir=tmp_path)
    assert list(tmp_path.rglob("*.md")) == [], "게이트 위반인데 파일이 남았습니다"


def test_write_posts_rejects_scope_missing(tmp_path, summary, verdict):
    """publish_scope 키가 아예 없는 dict 도 통과시키지 않습니다 (모르면 막는 쪽).

    깨뜨리는 법: _ScopedRef 생성 시 publish_scope 기본값을 "public" 으로 채우면 빨간불.
    확인일: 2026-08-18
    """
    bare = {
        "id": "arxiv:2608.11053",
        "title": "t",
        "url": "https://arxiv.org/abs/2608.11053",
        "published": "2026-08-12T00:00:00+00:00",
        "categories": [],
    }
    with pytest.raises(ScopeViolation):
        write_posts([(bare, summary, verdict)], site_dir=tmp_path)


# ── flagged 경고 블록 ────────────────────────────────────────────────────


def test_flagged_post_carries_warning_block(item, summary):
    """임계 미달 요약은 flagged=true + 본문 상단 경고 블록 (기획안 §8 M2 —
    미달이어도 발행은 하되 경고와 함께).

    깨뜨리는 법: render_post 의 `if flagged:` 블록을 지우면 빨간불.
    확인일: 2026-08-18
    """
    post = render_post(item, summary, {"faithfulness": 0.55, "verifier_model": "x/y"})
    front = _frontmatter(post)
    assert front["flagged"] is True
    assert front["faithfulness"] == 0.55

    body = post.split("---\n", 2)[2]
    assert "검증 미달" in body
    # 경고는 본문 맨 위 — 요약 첫 절보다 앞서야 합니다.
    assert body.index("검증 미달") < body.index("## 문제")


def test_unflagged_post_has_no_warning(item, summary, verdict):
    """깨뜨리는 법: render_post 의 flagged 유도를 `flagged = True` 상수로 바꾸면 빨간불.
    확인일: 2026-08-18
    """
    post = render_post(item, summary, verdict)
    assert _frontmatter(post)["flagged"] is False
    assert "검증 미달" not in post


def test_flag_threshold_is_schema_constant(item, summary):
    """경계값: 정확히 임계면 미달이 아닙니다 (Verdict.passed 와 같은 방향, R4).

    깨뜨리는 법: render_post 의 비교를 `<=` 로 바꾸면 빨간불.
    확인일: 2026-08-18
    """
    post = render_post(item, summary, {"faithfulness": FAITHFULNESS_THRESHOLD})
    assert _frontmatter(post)["flagged"] is False


# ── 본문 구성 ────────────────────────────────────────────────────────────


def test_body_has_summary_abstract_and_link(item, summary, verdict):
    """본문 = 한국어 요약(스키마 필드 순서) → 영어 초록 원문(CC0, §3.1) → abs 링크.

    깨뜨리는 법: render_post 에서 abstract 절 또는 링크 줄을 지우면 빨간불.
    확인일: 2026-08-18
    """
    post = render_post(item, summary, verdict)
    body = post.split("---\n", 2)[2]

    for heading in ("## 문제", "## 방법", "## 핵심결과", "## 연결점"):
        assert heading in body, f"{heading} 절이 없습니다"
    # limitations 는 이 픽스처에서 빈 배열 — 빈 절은 만들지 않습니다.
    assert "## 한계" not in body

    assert "## Abstract" in body
    assert "significant potential for improving crop monitoring" in body  # 초록 원문 그대로
    assert "[arXiv에서 원문 보기](https://arxiv.org/abs/2608.11053)" in body

    # 순서: 요약 → 초록 → 링크
    assert body.index("## 문제") < body.index("## Abstract") < body.rindex("[arXiv에서 원문 보기]")

    # CLAUDE.md §3-4 — 발행물 어디에도 PDF 경로가 없어야 합니다.
    assert "/pdf/" not in post and ".pdf" not in post


# ── 파일명 규칙 ──────────────────────────────────────────────────────────


def test_filename_rule(item):
    """`{YYYY-MM-DD}-{arxiv_id에서 : 제거}.md` (기획안_2 §7.2).

    깨뜨리는 법: post_filename 의 replace(":", "") 를 지우면 빨간불.
    확인일: 2026-08-18
    """
    assert post_filename(item) == "2026-08-12-arxiv2608.11053.md"


def test_filename_keeps_legacy_prefix_without_subdir(tmp_path, item, summary, verdict):
    """구형 ID(§9.5)의 아카이브 접두사는 보존하되, `/` 가 하위 디렉터리가 되면 안 됩니다.

    깨뜨리는 법: post_filename 의 replace("/", "-") 를 지우면 빨간불
    (math.GT 하위 디렉터리에 파일이 생김).
    확인일: 2026-08-18
    """
    item["id"] = "arxiv:math.GT/0309136"
    assert post_filename(item) == "2026-08-12-arxivmath.GT-0309136.md"

    paths = write_posts([(item, summary, verdict)], site_dir=tmp_path)
    assert paths == [tmp_path / "content" / "posts" / "2026-08-12-arxivmath.GT-0309136.md"]
    assert paths[0].parent == tmp_path / "content" / "posts", "파일명이 디렉터리를 만들었습니다"


def test_write_posts_writes_rendered_content(tmp_path, item, summary, verdict):
    """write_posts 산출물 == render_post 결과. 경로 반환도 계약입니다.

    깨뜨리는 법: write_posts 가 빈 문자열을 쓰도록 바꾸면 빨간불.
    확인일: 2026-08-18
    """
    paths = write_posts([(item, summary, verdict)], site_dir=tmp_path)
    assert len(paths) == 1
    assert paths[0].read_text(encoding="utf-8") == render_post(item, summary, verdict)


def test_write_posts_rejects_bad_entry_before_writing(tmp_path, item, summary, verdict):
    """배치 중 한 건이 ValueError 면 나머지 정상분도 쓰지 않습니다 (부분 발행 금지).

    깨뜨리는 법: write_posts 에서 렌더링과 쓰기를 한 루프로 합치면 빨간불
    (첫 건이 이미 디스크에 남음).
    확인일: 2026-08-18
    """
    broken = dict(item, id="arxiv:9999.00001", url="")
    with pytest.raises(ValueError):
        write_posts([(item, summary, verdict), (broken, summary, verdict)], site_dir=tmp_path)
    assert list(tmp_path.rglob("*.md")) == []
