"""M6 피드백 환류 — feedback.jsonl 라벨로 프로파일 weight 갱신 (기획안_2 §8.3).

무엇을 하는가
------------
텔레그램 인라인 버튼이 남긴 `read` / `skip` / `save` 라벨을 모아서, **읽거나 저장한
아이템과 가장 가까운 관심사의 weight를 올리고, 스킵이 반복되는 관심사의 weight를
내립니다.** 결과는 프로파일의 **새 번호 파일**로 씁니다.

세 가지 제동장치 — 전부 §8.3이 요구한 것입니다
---------------------------------------------
1. **라벨 50건 미만이면 거부합니다.** 며칠치 잡음으로 프로파일을 흔들지 않습니다.
2. **1회 갱신 폭은 ±0.1을 넘지 않고**, 결과 weight는 `[0.1, 1.5]`로 클램프합니다.
   프로파일이 며칠치 라벨로 뒤집히면 랭킹 평가가 무의미해집니다. 관심사 하나가
   1.0에서 0으로 사라지는 데 최소 열 번의 갱신 주기가 걸리도록 만든 값입니다.
3. **스킵은 반복돼야 반영됩니다** (`MIN_SKIP_REPEAT`). 한 번 안 읽은 것은 관심이
   식었다는 근거가 못 됩니다 — 그날 바빴을 수도 있습니다.

기존 파일을 **덮어쓰지 않습니다** (CLAUDE.md §2). `profile.papers_1.yaml`이 최신이면
`profile.papers_2.yaml`을 새로 만들고, 이미 있으면 `FileExistsError`로 죽습니다.
갱신 전 프로파일이 남아 있어야 §8.3의 "갱신 전후를 골드셋으로 재측정"이 가능합니다.

임베딩은 **주입받습니다.** 여기서 유사도 모델을 자체 구현하지 않습니다 — `src/rank`의
Embedder 포트를 그대로 씁니다 (기획안_2 §4.2). 이 모듈은 순위·클램프 규칙만 압니다.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import yaml

from src.core import config

# 유사도는 `src/rank`의 것을 그대로 씁니다 — 여기서 임베딩도 코사인도 다시 만들지
# 않습니다 (기획안_2 §4.2 · 자산 인벤토리). 랭킹과 환류가 서로 다른 공간에서 거리를
# 재면 "읽은 논문과 가까운 관심사"가 랭킹이 쓰는 뜻과 달라집니다.
# `src.rank.embed`는 최상단에서 torch·sentence-transformers 를 import 하지 않으므로
# (§9.10) 운영 의존성만으로 이 모듈을 import 할 수 있습니다.
from src.rank.embed import Embedder, cosine, l2_norm

#: `data/feedback.jsonl` (기획안_2 §8.1). 함수 기본 인자에 직접 쓰지 않습니다 (§9.1).
DEFAULT_FEEDBACK_PATH = Path("data/feedback.jsonl")

#: 이 건수 미만이면 갱신을 거부합니다 (§8.3 "50건 이상 누적 후").
MIN_LABELS = 50

#: 1회 갱신에서 weight가 움직일 수 있는 최대 폭 (§8.3 "갱신 폭에 상한을 두세요").
MAX_DELTA = 0.1

#: 결과 weight 클램프 범위. 하한이 0이 아닌 이유 — 0이면 관심사가 사실상 삭제되고,
#: 한 번 사라진 관심사는 다시 올라올 근거(피드백)를 얻을 기회조차 없습니다.
WEIGHT_MIN = 0.1
WEIGHT_MAX = 1.5

#: 스킵이 이 횟수 이상 쌓인 관심사만 내립니다 (§8.3 "skip이 **반복되는**").
MIN_SKIP_REPEAT = 3

#: 텔레그램 인라인 버튼 3개 (§8.1). 그 외 값은 수집기 버그이므로 시끄럽게 죽습니다.
POSITIVE_ACTIONS = frozenset({"read", "save"})
NEGATIVE_ACTIONS = frozenset({"skip"})
KNOWN_ACTIONS = POSITIVE_ACTIONS | NEGATIVE_ACTIONS

_SUFFIX_RE = re.compile(r"^_(\d+)$")


@dataclass(frozen=True, slots=True)
class InterestChange:
    """관심사 1개의 갱신 내역. `results.md`에 전/후를 적기 위한 구조체입니다 (§8.3)."""

    id: str
    before: float
    after: float
    delta: float
    read: int
    save: int
    skip: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "before": self.before,
            "after": self.after,
            "delta": self.delta,
            "read": self.read,
            "save": self.save,
            "skip": self.skip,
        }


@dataclass(frozen=True, slots=True)
class ProfileUpdate:
    """갱신 1회의 전량. `changes`는 **모든** 관심사를 담습니다 (delta 0 포함).

    변하지 않은 관심사를 빼면 `results.md`의 전/후 표가 반쪽이 됩니다 — 무엇이 안
    움직였는지도 결과입니다.
    """

    changes: tuple[InterestChange, ...]
    labels_used: int
    unknown_items: int
    source_path: Path | None = None
    out_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "labels_used": self.labels_used,
            "unknown_items": self.unknown_items,
            "source_path": str(self.source_path) if self.source_path else None,
            "out_path": str(self.out_path) if self.out_path else None,
            "changes": [change.to_dict() for change in self.changes],
        }

    def markdown_table(self) -> str:
        """`eval/results.md`에 그대로 붙일 수 있는 전/후 표 (§8.3)."""
        lines = [
            "| 관심사 | before | after | Δ | read | save | skip |",
            "|:--|--:|--:|--:|--:|--:|--:|",
        ]
        for change in self.changes:
            lines.append(
                f"| {change.id} | {change.before:.3f} | {change.after:.3f} | "
                f"{change.delta:+.3f} | {change.read} | {change.save} | {change.skip} |"
            )
        return "\n".join(lines)


# ── 입력 로딩 ───────────────────────────────────────────────────────────


def load_feedback(path: Path | None = None) -> list[dict[str, Any]]:
    """`data/feedback.jsonl`을 읽습니다. 파일이 없으면 빈 리스트입니다.

    경로 기본값은 **호출 시점에** 해석합니다 (§9.1 — 정의 시점에 바인딩하면 테스트가
    monkeypatch해도 실제 `data/`를 읽습니다).
    """
    path = path or DEFAULT_FEEDBACK_PATH
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def latest_by_item(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """같은 `item_id`가 여러 줄이면 **마지막 줄이 유효**합니다.

    한 아이템을 읽고 나서 저장하면 두 줄이 남습니다. 둘 다 세면 부지런한 사람의
    아이템 하나가 두 표가 되어, 갱신이 소수의 아이템에 끌려갑니다.
    """
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        item_id = row.get("item_id")
        if not item_id:
            raise ValueError(f"feedback 줄에 item_id 가 없습니다: {dict(row)!r}")
        latest[str(item_id)] = dict(row)
    return latest


def _index_items(items: Iterable[Any]) -> dict[str, str]:
    """아이템 → `{id: 임베딩할 텍스트}`. dict(후보 JSONL)와 `Item` 둘 다 받습니다.

    id로 접으므로 후보 파일의 중복 줄(§9.6 — 수집이 파일 기록과 state 마킹 사이에서
    죽으면 생깁니다)이 같은 아이템을 두 번 세게 만들지 않습니다.
    """
    indexed: dict[str, str] = {}
    for obj in items:
        if isinstance(obj, Mapping):
            item_id = obj.get("id")
            title = obj.get("title") or ""
            abstract = obj.get("abstract") or ""
        else:
            item_id = getattr(obj, "id", None)
            title = getattr(obj, "title", "") or ""
            abstract = getattr(obj, "abstract", "") or ""
        if not item_id:
            raise ValueError(f"id 없는 아이템입니다: {obj!r}")
        text = f"{title}\n\n{abstract}".strip()
        if not text:
            raise ValueError(f"제목·초록이 모두 비어 있습니다: {item_id}")
        indexed[str(item_id)] = text
    return indexed


# ── 유사도 ──────────────────────────────────────────────────────────────


def _encode(embedder: Embedder | Any, texts: Sequence[str]) -> list[Sequence[float]]:
    """주입된 Embedder로 인코딩합니다. 임베딩을 여기서 구현하지 않습니다.

    포트를 만족하지 않는 객체면 **즉시 TypeError**입니다 — 조용히 대체 구현으로
    떨어지지 않습니다.
    """
    fn: Callable[[Sequence[str]], Sequence[Sequence[float]]] | None = getattr(
        embedder, "encode", None
    )
    if fn is None:
        raise TypeError(
            f"Embedder 는 encode(texts) 를 제공해야 합니다 (기획안_2 §4.2): "
            f"{type(embedder).__name__}"
        )
    vectors = [list(vector) for vector in fn(list(texts))]
    if len(vectors) != len(texts):
        raise ValueError(f"임베딩 개수가 입력과 다릅니다: {len(vectors)} != {len(texts)}")
    return vectors


def _similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """`src.rank.embed.cosine` 을 그대로 씁니다. 차원 불일치는 거기서 ValueError 입니다.

    영벡터만 여기서 따로 막습니다. `cosine`은 영벡터에 0.0을 주는데(랭킹에서는
    합당합니다 — 점수가 0인 문서일 뿐), 환류에서는 **가장 가까운 관심사 하나**를
    골라야 합니다. 전부 0이면 사전순 첫 관심사가 조용히 표를 받습니다.
    """
    if l2_norm(a) == 0.0 or l2_norm(b) == 0.0:
        raise ValueError("임베딩이 영벡터입니다 — 가장 가까운 관심사를 정할 수 없습니다")
    return cosine(a, b)


def _nearest_interest(
    item_vector: Sequence[float],
    interest_vectors: Sequence[tuple[str, Sequence[float]]],
) -> str:
    """가장 유사한 관심사 id. 동점이면 **id 사전순**으로 고정합니다.

    동점 처리를 정하지 않으면 같은 피드백으로 두 번 돌렸을 때 다른 프로파일이
    나옵니다 (기획안_2 §5.4의 동점 규칙과 같은 이유).
    """
    scored = [
        (-_similarity(item_vector, vector), interest_id)
        for interest_id, vector in interest_vectors
    ]
    return min(scored)[1]


# ── 갱신 계산 ───────────────────────────────────────────────────────────


def _interests(profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    interests = profile.get("interests")
    if not interests:
        raise ValueError("프로파일에 interests 가 없습니다")
    for interest in interests:
        for key in ("id", "text", "weight"):
            if key not in interest:
                raise ValueError(f"관심사에 {key} 가 없습니다: {interest.get('id', interest)!r}")
    return list(interests)


def compute_updates(
    profile: Mapping[str, Any],
    feedback_rows: Iterable[Mapping[str, Any]],
    items: Iterable[Any],
    embedder: Any,
) -> ProfileUpdate:
    """순수 계산 — 파일을 읽지도 쓰지도 않습니다.

    라벨 1건 = 아이템 1건입니다 (같은 아이템의 중복 줄은 마지막 것만 셉니다).
    각 아이템은 **가장 가까운 관심사 하나**에만 표를 던집니다. 여러 관심사에 나눠
    주면 15개 전체가 조금씩 같은 방향으로 움직여 순위가 그대로입니다.
    """
    interests = _interests(profile)
    indexed_items = _index_items(items)
    latest = latest_by_item(feedback_rows)

    tallies: dict[str, dict[str, int]] = {
        str(interest["id"]): {"read": 0, "save": 0, "skip": 0} for interest in interests
    }

    usable: list[tuple[str, str]] = []  # (item_id, action)
    unknown = 0
    for item_id, row in latest.items():
        action = str(row.get("action", "")).strip().lower()
        if action not in KNOWN_ACTIONS:
            # 수집기(§8.1)가 쓰는 값은 3개뿐입니다. 그 외가 있으면 수집기 버그이고,
            # 무시하면 갱신이 조용히 일부 라벨만 반영합니다.
            raise ValueError(f"알 수 없는 피드백 action 입니다: {action!r} ({item_id})")
        if item_id not in indexed_items:
            # 후보 파일이 정리됐거나 다른 채널의 아이템입니다. 텍스트가 없으면
            # 어느 관심사에 넣을지 알 수 없으므로 세지 않습니다.
            unknown += 1
            continue
        usable.append((item_id, action))

    if len(usable) < MIN_LABELS:
        raise ValueError(
            f"라벨이 {MIN_LABELS}건 미만이라 갱신을 거부합니다 (기획안_2 §8.3): "
            f"{len(usable)}건 (매칭 실패 {unknown}건)"
        )

    # 임베딩은 두 번에 몰아서 호출합니다 — 한 건씩 부르면 API 임베더에서 비용·지연이
    # 그대로 곱해집니다. 관심사는 `text`만 씁니다. `label`은 사람용이라 임베딩하지
    # 않습니다 (data/profile.papers_1.yaml 머리말).
    interest_ids = [str(interest["id"]) for interest in interests]
    interest_vectors = list(
        zip(interest_ids, _encode(embedder, [str(interest["text"]) for interest in interests]))
    )
    ordered_item_ids = [item_id for item_id, _ in usable]
    item_vectors = dict(
        zip(ordered_item_ids, _encode(embedder, [indexed_items[i] for i in ordered_item_ids]))
    )

    for item_id, action in usable:
        nearest = _nearest_interest(item_vectors[item_id], interest_vectors)
        tallies[nearest][action] += 1

    changes: list[InterestChange] = []
    for interest in interests:
        interest_id = str(interest["id"])
        tally = tallies[interest_id]
        before = float(interest["weight"])
        delta = _delta(tally)
        after = min(WEIGHT_MAX, max(WEIGHT_MIN, before + delta))
        changes.append(
            InterestChange(
                id=interest_id,
                before=round(before, 4),
                after=round(after, 4),
                # 클램프 후 실제로 움직인 폭을 적습니다. 요청한 delta를 적으면
                # before + delta != after 인 표가 되어 읽는 사람이 속습니다.
                delta=round(after - before, 4),
                read=tally["read"],
                save=tally["save"],
                skip=tally["skip"],
            )
        )

    return ProfileUpdate(
        changes=tuple(changes), labels_used=len(usable), unknown_items=unknown
    )


def _delta(tally: Mapping[str, int]) -> float:
    """관심사 1개의 갱신 폭. **항상 `|delta| <= MAX_DELTA`** 입니다 (§8.3).

    순 신호를 비율로 만듭니다: `(긍정 - 부정) / 전체`. 비율이므로 표 100개가 몰려도
    상한을 넘지 못합니다. 절대 건수를 쓰면 하루 많이 읽은 날 프로파일이 뒤집힙니다.
    """
    positive = tally["read"] + tally["save"]
    negative = tally["skip"]
    total = positive + negative
    if total == 0:
        return 0.0
    if positive <= negative and negative < MIN_SKIP_REPEAT:
        # 스킵 1~2건으로는 내리지 않습니다 (§8.3 "skip이 **반복되는** 관심사").
        return 0.0
    ratio = (positive - negative) / total
    return round(MAX_DELTA * ratio, 6)


def apply_changes(profile: Mapping[str, Any], changes: Sequence[InterestChange]) -> dict[str, Any]:
    """weight만 바꾼 **새 프로파일 dict**. 입력 dict는 건드리지 않습니다."""
    by_id = {change.id: change for change in changes}
    # deepcopy 입니다. json 왕복으로 복사하면 YAML이 날짜로 파싱한 값(`date`)이
    # 문자열로 바뀌거나 TypeError 가 납니다 — 프로파일에 그런 키가 생기는 순간
    # 조용히 형이 달라집니다.
    updated: dict[str, Any] = copy.deepcopy(dict(profile))
    for interest in updated["interests"]:
        change = by_id.get(str(interest["id"]))
        if change is not None:
            interest["weight"] = change.after
    return updated


# ── 파일 번호 규칙 ──────────────────────────────────────────────────────


def next_profile_path(stem: str, root: Path | str | None = None, ext: str = ".yaml") -> Path:
    """현재 **최고 번호 + 1** 경로. 기존 파일은 절대 건드리지 않습니다 (CLAUDE.md §2).

    최고 번호는 `config.resolve_profile()`이 정합니다 (R1) — 번호를 여기서 다시 세면
    `_10 < _9` 같은 문자열 정렬 버그가 두 곳에 따로 생깁니다.
    """
    root = Path(root) if root is not None else config.DATA_DIR
    current = config.resolve_profile(stem, ext=ext, root=root)
    rest = current.name[len(stem) : -len(ext)]
    matched = _SUFFIX_RE.match(rest)
    number = int(matched.group(1)) if matched else 0
    return root / f"{stem}_{number + 1}{ext}"


def _header(update: ProfileUpdate, stem: str, now: datetime) -> str:
    """생성 근거를 파일 머리에 남깁니다. YAML 덤프는 원본 주석을 보존하지 못하므로,
    최소한 **무엇을 근거로 이 숫자가 되었는지**는 새 파일이 스스로 들고 있어야 합니다.
    """
    lines = [
        f"# {update.out_path}",
        "# ★ 자동 생성 — src/feedback/update_profile.py (기획안_2 §8.3 M6 피드백 환류)",
        f"# 생성 시각: {now.isoformat(timespec='seconds')}",
        f"# 원본: {update.source_path} (원본은 고치지 않습니다 — CLAUDE.md §2)",
        f"# 반영한 라벨: {update.labels_used}건 (매칭 실패 {update.unknown_items}건)",
        f"# 갱신 상한: 1회 ±{MAX_DELTA}, weight 범위 [{WEIGHT_MIN}, {WEIGHT_MAX}]",
        f"# 프로파일 stem: {stem}",
        "#",
        "# 갱신 내역 (id / before → after / read·save·skip)",
    ]
    for change in update.changes:
        lines.append(
            f"#   {change.id:<32} {change.before:.3f} → {change.after:.3f} "
            f"({change.delta:+.3f})  r{change.read} s{change.save} k{change.skip}"
        )
    lines.append(
        "#\n# 갱신 전후를 골드셋으로 재측정해 eval/results.md 에 기록하세요 (§8.3). "
        "악화해도 그대로 씁니다."
    )
    return "\n".join(lines) + "\n\n"


def update_profile(
    stem: str = "profile.papers",
    *,
    embedder: Any,
    items: Iterable[Any],
    feedback_path: Path | None = None,
    root: Path | str | None = None,
    now: Callable[[], datetime] | None = None,
) -> ProfileUpdate:
    """피드백으로 프로파일을 갱신해 **새 번호 파일**로 쓰고, 갱신 내역을 돌려줍니다.

    - 프로파일은 `config.load_profile()`로만 읽습니다 (R1 — 경로 하드코딩 금지).
    - 출력은 `{stem}_{N+1}.yaml`. 이미 있으면 `FileExistsError`로 죽습니다.
    - 라벨 50건 미만이면 아무것도 쓰지 않고 `ValueError`입니다 (§8.3).
    """
    root_path = Path(root) if root is not None else config.DATA_DIR
    profile, source_path = config.load_profile(stem, root=root_path)

    update = compute_updates(profile, load_feedback(feedback_path), items, embedder)

    out_path = next_profile_path(stem, root=root_path)
    if out_path.exists():
        # 여기까지 왔다는 건 resolve_profile 이 최고 번호를 잘못 골랐다는 뜻입니다.
        # 덮어쓰면 이력이 사라집니다 (CLAUDE.md §2).
        raise FileExistsError(f"이미 있는 파일을 덮어쓸 수 없습니다 (CLAUDE.md §2): {out_path}")

    update = replace(update, source_path=source_path, out_path=out_path)

    body = yaml.safe_dump(
        apply_changes(profile, update.changes),
        allow_unicode=True,
        sort_keys=False,
        width=88,
    )
    stamp = (now or (lambda: datetime.now(UTC)))()
    out_path.write_text(_header(update, stem, stamp) + body, encoding="utf-8")
    return update
