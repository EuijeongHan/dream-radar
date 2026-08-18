"""jobs 채널 검증 — 하드 필터 위반 검사 (기획안_2 §8.2, CLAUDE.md §3-8).

**jobs 채널에는 faithfulness 검증을 적용하지 않습니다.**
사람인 API는 직무내용 본문을 주지 않습니다 (`profile.jobs_1.yaml` 주석 참조 —
반환 필드는 title/keyword/job-code/industry/location 수준). 대조할 원문이 없는데
faithfulness를 매기면 0.0~1.0 사이의 숫자는 나오지만 **그 숫자가 무엇을 재는지
아무도 말할 수 없습니다.** 그래서 CLAUDE.md §3-8이 금지합니다.

대신 재는 것은 **하드 필터가 실제로 걸렸는가**입니다. 프로파일의
`filters.experience` · `filters.location` · `exclude_keywords.hard` 는 "이런 공고는
보고 싶지 않다"는 선언이고, 그 선언이 산출물에서 지켜졌는지는 원문 없이도 검사할 수
있습니다.

판정 함수를 왜 여기 두는가
--------------------------
`saramin.py`가 수집 시점에 거르고, 이 모듈이 발행 직전에 다시 검사합니다.
두 벌의 판정 로직을 두면 **둘이 함께 틀립니다** — 어댑터의 버그가 게이트의 같은
버그에 가려집니다. 그래서 판정은 여기 한 벌만 두고 어댑터가 `passes_hard_filters()`를
import 합니다 (자산_인벤토리 R6의 정신). 게이트는 "다른 코드로 다시 재는 것"이 아니라
"수집 이후 어딘가에서 필터를 통과하지 않은 아이템이 섞여 들어왔는지"를 잡습니다.

★ 위반 리포트에 공고 원문을 넣지 않습니다
------------------------------------------
이 저장소는 public이고 Actions 로그도 공개됩니다. 위반 메시지에 공고 제목·기업명을
넣으면 그 순간 채용공고를 공개 게시한 것이 됩니다 (CLAUDE.md §3-3, 무료여도 위반).
제외 키워드는 본인의 구직 조건이라 역시 비공개입니다 (CLAUDE.md §3-1).
→ `FilterViolation.detail`에는 **코드값과 인덱스만** 넣습니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from src.core.config import load_profile
from src.verify.gates import GateViolation

#: 프로파일 stem. 경로를 문자열로 쓰지 않습니다 (재사용 규칙 R1 / 기획안_2 §9.9).
JOBS_PROFILE_STEM = "profile.jobs"

#: 위반 규칙 이름. 문자열을 여기저기 다시 타이핑하면 오타가 조용히 통과합니다.
RULE_ACTIVE = "active"
RULE_EXPERIENCE_CODE = "experience_level_code"
RULE_EXPERIENCE_YEARS = "experience_max_years"
RULE_LOCATION = "location"
RULE_EXCLUDE_KEYWORD = "exclude_keyword_hard"
RULE_MISSING_EVIDENCE = "missing_filter_evidence"

#: 경력코드 (사람인 API 가이드 "응답 결과" — experience-level@code).
EXPERIENCE_LEVEL_NAMES: dict[int, str] = {
    0: "경력무관",
    1: "신입",
    2: "경력",
    3: "신입/경력",
}


class FilterViolationError(GateViolation):
    """하드 필터를 통과하지 않은 공고가 산출물에 남아 있습니다 (기획안_2 §8.2)."""


@dataclass(frozen=True)
class FilterViolation:
    """위반 1건.

    `detail`은 **공고 원문을 담지 않습니다** (모듈 docstring 참조). 코드값·인덱스만
    담습니다. 원문이 필요하면 `item_id`로 `data/candidates/`(gitignore)를 보세요.
    """

    item_id: str
    rule: str
    detail: str

    def __str__(self) -> str:
        return f"{self.item_id} [{self.rule}] {self.detail}"


@dataclass(frozen=True)
class JobFilters:
    """프로파일에서 뽑은 하드 필터 조건.

    프로파일 dict를 그대로 들고 다니지 않는 이유: 어느 키를 읽는지가 여기 한 곳에
    고정됩니다. 프로파일이 `_2`로 갱신되면서 키가 빠지면 `from_profile`이 시끄럽게
    죽습니다 — 조용히 "검사할 조건이 없음 = 전부 통과"가 되는 것보다 낫습니다.
    """

    active_only: bool
    experience_level_codes: frozenset[int]
    experience_max_years: int | None
    location_include: tuple[str, ...]
    exclude_keywords_hard: tuple[str, ...]

    @classmethod
    def from_profile(cls, profile: Mapping[str, Any]) -> "JobFilters":
        missing: list[str] = []

        response_filters = _dig(profile, "sources", "saramin", "response_filters")
        if not isinstance(response_filters, Mapping):
            missing.append("sources.saramin.response_filters")
            response_filters = {}

        codes = response_filters.get("experience_level_code")
        if not codes:
            missing.append("sources.saramin.response_filters.experience_level_code")
            codes = []

        location = _dig(profile, "filters", "location", "include")
        if not location:
            missing.append("filters.location.include")
            location = []

        # 빈 리스트는 허용합니다 — "제외할 키워드가 없다"는 명시적 선언일 수 있습니다.
        # 키 자체가 없는 것과는 다릅니다.
        exclude_block = _dig(profile, "exclude_keywords")
        if not isinstance(exclude_block, Mapping) or "hard" not in exclude_block:
            missing.append("exclude_keywords.hard")
            hard = []
        else:
            hard = exclude_block.get("hard") or []

        if missing:
            raise ValueError(
                "jobs 프로파일에 하드 필터 조건이 없습니다: "
                + ", ".join(missing)
                + ". 조건 없이 게이트를 돌리면 전부 통과합니다 (기획안_2 §8.2)"
            )

        max_years = response_filters.get("experience_max_years")
        return cls(
            active_only=_as_int(response_filters.get("active")) == 1,
            experience_level_codes=frozenset(
                code for value in codes if (code := _as_int(value)) is not None
            ),
            experience_max_years=_as_int(max_years),
            location_include=tuple(str(token) for token in location),
            exclude_keywords_hard=tuple(str(token) for token in hard),
        )


def load_filters(profile: Mapping[str, Any] | None = None) -> JobFilters:
    """프로파일을 받거나(주입) 최신 jobs 프로파일을 읽어(R1) 필터를 만듭니다.

    기본 인자에 모듈 상수를 쓰지 않습니다 — 호출 시점 해석 (기획안_2 §9.1).
    """
    if profile is None:
        profile, _path = load_profile(JOBS_PROFILE_STEM)
    return JobFilters.from_profile(profile)


# ── 판정 ────────────────────────────────────────────────────────────────


def check_item(item: Any, filters: JobFilters) -> list[FilterViolation]:
    """아이템 1건의 하드 필터 위반 목록. 위반이 없으면 빈 리스트입니다."""
    item_id = str(_attr(item, "id") or "")
    raw = _attr(item, "raw") or {}
    violations: list[FilterViolation] = []

    if not isinstance(raw, Mapping):
        return [
            FilterViolation(item_id, RULE_MISSING_EVIDENCE, "raw 가 매핑이 아닙니다")
        ]

    # ① 진행 여부 — 마감된 공고를 추천하면 그 자리에서 쓸모가 없습니다.
    if filters.active_only:
        active = _as_int(raw.get("active"))
        if active is None:
            violations.append(
                FilterViolation(item_id, RULE_MISSING_EVIDENCE, "active 없음")
            )
        elif active != 1:
            violations.append(
                FilterViolation(item_id, RULE_ACTIVE, f"active={active}")
            )

    # ② 경력 — experience-level@code (0=경력무관 1=신입 2=경력 3=신입/경력)
    experience = raw.get("experience_level")
    if not isinstance(experience, Mapping):
        violations.append(
            FilterViolation(item_id, RULE_MISSING_EVIDENCE, "experience_level 없음")
        )
    else:
        code = _as_int(experience.get("code"))
        if code is None:
            violations.append(
                FilterViolation(item_id, RULE_MISSING_EVIDENCE, "experience_level@code 없음")
            )
        elif code not in filters.experience_level_codes:
            violations.append(
                FilterViolation(
                    item_id,
                    RULE_EXPERIENCE_CODE,
                    f"experience-level@code={code}({EXPERIENCE_LEVEL_NAMES.get(code, '?')})",
                )
            )

        # 연차 상한. 프로파일이 "experience-level@max 기준"이라고 확정했으므로
        # @max 로 판정합니다. @min 기준으로 바꾸면 "신입/경력 0~10년" 공고가
        # 통과합니다 — 어느 쪽이 맞는지는 취향이 아니라 프로파일의 결정이고,
        # 바꾸려면 코드가 아니라 `profile.jobs_N.yaml`을 고쳐야 합니다
        # (기획안_2 §8.2 "파라미터는 프로파일에 확정. 임의 변경 금지").
        if filters.experience_max_years is not None:
            max_years = _as_int(experience.get("max"))
            if max_years is not None and max_years > filters.experience_max_years:
                violations.append(
                    FilterViolation(
                        item_id,
                        RULE_EXPERIENCE_YEARS,
                        f"experience-level@max={max_years}>{filters.experience_max_years}",
                    )
                )

    # ③ 지역 — location@name 만 봅니다.
    # 제목까지 뒤지면 "서울대학교 산학협력단"이 "서울"로 통과합니다.
    location = raw.get("location")
    if not isinstance(location, Mapping) or not str(location.get("name") or "").strip():
        violations.append(
            FilterViolation(item_id, RULE_MISSING_EVIDENCE, "location@name 없음")
        )
    else:
        name = str(location["name"])
        if not any(token in name for token in filters.location_include):
            violations.append(
                FilterViolation(
                    item_id,
                    RULE_LOCATION,
                    # 지역명도 공고 메타데이터이므로 코드만 남깁니다.
                    f"location@code={location.get('code')}",
                )
            )

    # ④ 즉시 제외 키워드. 어느 키워드가 걸렸는지는 **인덱스로만** 남깁니다
    # (제외 키워드 = 본인의 구직 조건, CLAUDE.md §3-1).
    haystack = f"{_attr(item, 'title') or ''} {_attr(item, 'abstract') or ''}"
    for index, keyword in enumerate(filters.exclude_keywords_hard):
        if keyword and keyword in haystack:
            violations.append(
                FilterViolation(
                    item_id, RULE_EXCLUDE_KEYWORD, f"exclude_keywords.hard[{index}]"
                )
            )

    return violations


def passes_hard_filters(item: Any, filters: JobFilters) -> bool:
    """수집 시점 필터. 어댑터가 이 함수를 씁니다 — 판정 로직은 한 벌뿐입니다."""
    return not check_item(item, filters)


def check_filters(
    items: Iterable[Any],
    *,
    profile: Mapping[str, Any] | None = None,
    filters: JobFilters | None = None,
) -> list[FilterViolation]:
    """산출물 전체의 위반 목록. **jobs 채널의 faithfulness 대체물입니다.**"""
    resolved = filters if filters is not None else load_filters(profile)
    violations: list[FilterViolation] = []
    for item in items:
        violations.extend(check_item(item, resolved))
    return violations


def assert_no_filter_violations(
    items: Iterable[Any],
    *,
    profile: Mapping[str, Any] | None = None,
    filters: JobFilters | None = None,
) -> None:
    """★ 발행 직전 게이트 (재사용 규칙 R5 — 호출하지 않으면 없는 것과 같습니다).

    하드 필터를 통과하지 않은 공고가 산출물에 남아 있으면 예외입니다.
    어댑터가 이미 걸렀으므로 여기서 걸리면 **수집 이후 어딘가가 잘못된 것**입니다.
    """
    violations = check_filters(items, profile=profile, filters=filters)
    if violations:
        raise FilterViolationError(
            f"하드 필터 위반 {len(violations)}건 (기획안_2 §8.2): "
            + "; ".join(str(violation) for violation in violations)
        )


# ── 헬퍼 ────────────────────────────────────────────────────────────────


def _attr(item: Any, name: str) -> Any:
    """`Item` 객체와 `Item.to_dict()` 결과를 모두 받습니다.

    후보 파일(`data/candidates/*.jsonl`)에서 읽으면 dict이고, 어댑터에서 바로 오면
    dataclass입니다. 게이트가 둘 중 하나만 받으면 호출자가 변환을 잊는 순간
    조용히 통과합니다.
    """
    if isinstance(item, Mapping):
        return item.get(name)
    return getattr(item, name, None)


def _dig(mapping: Any, *keys: str) -> Any:
    node = mapping
    for key in keys:
        if not isinstance(node, Mapping):
            return None
        node = node.get(key)
    return node


def _as_int(value: Any) -> int | None:
    """사람인 응답은 코드값을 문자열로도 정수로도 줍니다 (`"2"` / `2`)."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None
