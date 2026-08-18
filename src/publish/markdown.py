"""Hugo 포스트 발행기 (기획안_2 §7.2, M3).

발행은 이 파이프라인에서 유일하게 **공개 인터넷으로 나가는 단계**입니다. 그래서
이 모듈의 검사는 편의 기능이 아니라 하드 게이트입니다:

- `arxiv_abs_url` 없이는 발행하지 않습니다 — arXiv 이용약관 권고
  "Direct users to arXiv.org to retrieve e-print content" (기획안_2 §3.1 / 델타 §D1)
- PDF·전문 링크는 싣지 않습니다 — 전문은 대부분 재배포 불가 (CLAUDE.md §3-4)
- 쓰기 전에 `gates.assert_public_scope()`를 호출합니다 — 채용공고가 공개 싱크로
  흐르면 사람인 약관 조항 6(제휴 관계 오인, 무료여도 위반)에 걸립니다 (재사용 규칙 R5)
- 영어 초록 원문은 그대로 싣습니다 — arXiv 서술 메타데이터(제목·초록)는 CC0 1.0
  이라 재배포 제약이 없습니다 (v1.0 §3.3 / 기획안_2 §3.1)

요약 필드명은 `src.summarize.schema`의 것을 재사용합니다 (재사용 규칙 R4).
여기서 필드명을 다시 타이핑하면 스키마가 바뀔 때 발행기만 조용히 어긋납니다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from src.summarize.schema import (
    ALL_FIELDS,
    FAITHFULNESS_THRESHOLD,
    PaperSummary,
    Verdict,
)
from src.verify.gates import assert_faithfulness_flagged, assert_public_scope

#: 기본 사이트 루트. 함수 기본 인자에 직접 쓰지 않습니다 (기획안_2 §9.1 —
#: 정의 시점 바인딩 때문에 monkeypatch가 무효가 됩니다. `None` 기본값 + 호출 시점 해석).
DEFAULT_SITE_DIR = Path("site")

#: 프론트매터 필수 필드 (기획안_2 §7.2). 하나라도 못 채우면 ValueError 입니다.
FRONTMATTER_REQUIRED_FIELDS: tuple[str, ...] = (
    "title",
    "date",
    "arxiv_id",
    "arxiv_abs_url",
    "categories",
    "faithfulness",
    "flagged",
)

#: 본문 절 제목. 키는 schema.ALL_FIELDS 의 필드명 그대로입니다 (R4).
#: 스키마에 필드가 추가되면 여기서 KeyError 로 시끄럽게 죽는 것이 의도입니다.
_SECTION_HEADINGS: dict[str, str] = {
    "problem": "문제",
    "method": "방법",
    "key_results": "핵심결과",
    "limitations": "한계",
    "connection": "연결점",
}

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")


@dataclass(frozen=True, slots=True)
class _ScopedRef:
    """dict 아이템을 `gates.assert_public_scope()`에 넘기기 위한 최소 어댑터.

    게이트는 `publish_scope` **속성**을 읽습니다 (Item 객체 기준). 발행 단계는
    후보 JSONL에서 읽은 dict를 다루므로 여기서 속성 형태로 감쌉니다.
    검사 로직 자체는 재구현하지 않고 게이트를 그대로 호출합니다 (R5).
    """

    id: str
    publish_scope: Any


def _require_item_fields(item: dict[str, Any]) -> None:
    """프론트매터를 채울 수 없는 아이템을 거부합니다 (기획안_2 §7.2)."""
    missing = [
        source_key
        for source_key, front_key in (
            ("title", "title"),
            ("published", "date"),
            ("id", "arxiv_id"),
            ("url", "arxiv_abs_url"),
        )
        if not item.get(source_key)
    ]
    if item.get("categories") is None:
        missing.append("categories")
    if missing:
        raise ValueError(f"프론트매터 필수 필드를 채울 수 없습니다 (기획안_2 §7.2): {missing}")

    url = str(item["url"])
    # arXiv 이용약관 권고 "Direct users to arXiv.org to retrieve e-print content"
    # (기획안_2 §3.1). abs 링크가 없으면 발행 자체를 거부합니다 — 하드 게이트입니다.
    # PDF 링크는 abs 링크의 대체물이 될 수 없습니다: 전문은 대부분 재배포 불가
    # 라이선스입니다 (CLAUDE.md §3-4 / 델타 §D1).
    if "/pdf/" in url or url.lower().endswith(".pdf"):
        raise ValueError(f"PDF 링크로는 발행할 수 없습니다 (CLAUDE.md §3-4): {url}")

    if not _DATE_RE.match(str(item["published"])):
        raise ValueError(
            f"published 가 ISO 날짜(YYYY-MM-DD…)가 아닙니다: {item['published']!r}"
        )


def _verdict_from_dict(verdict: dict[str, Any]) -> Verdict:
    """verdict dict → `Verdict`. 범위(0.0~1.0) 검증은 스키마가 합니다 (R4)."""
    faithfulness = verdict.get("faithfulness")
    if faithfulness is None:
        raise ValueError("verdict 에 faithfulness 가 없습니다 — 프론트매터를 채울 수 없습니다")
    return Verdict(
        faithfulness=float(faithfulness),
        verifier_model=str(verdict.get("verifier_model", "")),
    )


def _render_section(field_name: str, value: Any) -> str:
    heading = _SECTION_HEADINGS[field_name]
    if isinstance(value, list):
        body = "\n".join(f"- {line}" for line in value)
    else:
        body = str(value)
    return f"## {heading}\n\n{body}\n"


def render_post(item: dict[str, Any], summary: dict[str, Any], verdict: dict[str, Any]) -> str:
    """아이템 1건을 Hugo 포스트(프론트매터 + 본문)로 렌더링합니다.

    구성 (기획안_2 §7.2): [flagged 경고] → 한국어 요약 5필드 → 영어 초록 원문(CC0)
    → arXiv abs 링크.

    프론트매터 필수 필드를 하나라도 못 채우면 `ValueError` 입니다.
    """
    _require_item_fields(item)

    # R4 — 필드명·필수값 검증을 스키마에 위임합니다. problem/method 가 비면 ValueError.
    paper = PaperSummary.from_dict(summary, item_id=str(item["id"]))
    verdict_obj = _verdict_from_dict(verdict)

    # 기획안 §8 M2 — 임계 미달은 발행을 막지 않습니다. 막는 건 **플래그 누락**입니다.
    # 플래그를 임계값에서 직접 유도하고, 그 계약을 게이트로 한 번 더 못박습니다 (R5).
    flagged = verdict_obj.faithfulness < FAITHFULNESS_THRESHOLD
    assert_faithfulness_flagged(verdict_obj, flagged)

    front = {
        "title": str(item["title"]),
        "date": str(item["published"]),
        "arxiv_id": str(item["id"]),
        "arxiv_abs_url": str(item["url"]),
        "categories": [str(c) for c in item["categories"]],
        "faithfulness": round(verdict_obj.faithfulness, 4),  # Verdict.to_dict 와 같은 자릿수
        "flagged": flagged,
    }
    if tuple(front) != FRONTMATTER_REQUIRED_FIELDS:  # 계약과 구현이 어긋나면 즉사
        raise RuntimeError(f"프론트매터 키가 계약과 다릅니다: {tuple(front)}")

    # 제목의 따옴표·LaTeX 중괄호(초록의 10%가 포함, 기획안_2 §9.15)는 yaml 라이브러리가
    # 이스케이프합니다. 문자열 조립으로 프론트매터를 만들지 마세요.
    frontmatter = yaml.safe_dump(front, allow_unicode=True, sort_keys=False).strip()

    sections: list[str] = []

    if flagged:
        sections.append(
            "> **경고: 검증 미달** — faithfulness "
            f"{verdict_obj.faithfulness:.2f} < {FAITHFULNESS_THRESHOLD}. "
            "요약이 초록과 어긋날 수 있습니다. 아래 원문 초록을 함께 확인하세요.\n"
        )

    for field_name in ALL_FIELDS:  # R4 — 순서·이름을 스키마에서 가져옵니다
        value = getattr(paper, field_name)
        if value:
            sections.append(_render_section(field_name, value))

    # 초록 원문 — arXiv 서술 메타데이터는 CC0 1.0 (기획안_2 §3.1). 전문·PDF 는 싣지
    # 않습니다 (CLAUDE.md §3-4). 초록은 이미 치환된 값이므로 다시 .format() 하지
    # 않습니다 (기획안_2 §9.15 — LaTeX 중괄호 재치환 금지).
    abstract = str(item.get("abstract") or "")
    if abstract:
        sections.append(f"## Abstract\n\n{abstract}\n")

    # arXiv 약관 권고 — 독자를 arXiv.org 로 보냅니다 (기획안_2 §3.1).
    sections.append(f"[arXiv에서 원문 보기]({item['url']})\n")

    body = "\n".join(sections)
    return f"---\n{frontmatter}\n---\n\n{body}"


def post_filename(item: dict[str, Any]) -> str:
    """`{YYYY-MM-DD}-{arxiv_id에서 : 제거}.md` (기획안_2 §7.2).

    구형 ID(`arxiv:math.GT/0309136`, §9.5)의 `/` 는 `-` 로 바꿉니다 — 파일명의 `/` 는
    하위 디렉터리가 되어 포스트가 조용히 엉뚱한 곳에 쓰입니다. 아카이브 접두사
    자체는 보존합니다 (§9.5 — 접두사를 지우면 타 아카이브와 충돌).
    """
    date = str(item["published"])[:10]
    stem = str(item["id"]).replace(":", "").replace("/", "-")
    return f"{date}-{stem}.md"


def write_posts(
    entries: Iterable[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]],
    site_dir: Path | None = None,
) -> list[Path]:
    """(item, summary, verdict) 목록을 `{site_dir}/content/posts/` 에 기록합니다.

    - 쓰기 **전에** `assert_public_scope()` 를 호출합니다 (R5). `publish_scope` 가
      `"public"` 이 아닌 아이템이 하나라도 있으면 `ScopeViolation` 이고,
      **아무 파일도 쓰지 않습니다** — 사람인 약관 조항 6(제휴 관계 오인 금지).
    - 렌더링을 전부 마친 뒤에 씁니다. 한 건이라도 `ValueError` 면 아무것도 남지 않습니다.
    - `site_dir` 기본값은 호출 시점에 해석합니다 (기획안_2 §9.1).
    """
    triples = list(entries)

    # ★ 쓰기 전에, 렌더링보다도 먼저. 게이트가 파일 시스템에 손대기 전에 터져야
    #   부분 발행이 남지 않습니다 (R5 — 함수 존재만으로는 아무것도 막지 못합니다).
    # `publish_scope` 키가 아예 없는 dict 도 위반입니다 — .get() 이 None 을 주고
    # 게이트가 "public 아님"으로 잡습니다. 기본값을 "public" 으로 채우면 안 됩니다.
    assert_public_scope(
        [_ScopedRef(str(item.get("id", "?")), item.get("publish_scope")) for item, _, _ in triples]
    )

    rendered = [
        (post_filename(item), render_post(item, summary, verdict))
        for item, summary, verdict in triples
    ]

    root = DEFAULT_SITE_DIR if site_dir is None else Path(site_dir)
    posts_dir = root / "content" / "posts"
    posts_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for filename, text in rendered:
        path = posts_dir / filename
        path.write_text(text, encoding="utf-8")
        written.append(path)
    return written
