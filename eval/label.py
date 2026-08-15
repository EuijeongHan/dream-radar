"""골드셋 라벨링 도구 (기획안 §9, 델타 §D6.5 순서 2).

    python -m eval.label triage  --date 2026-08-12    # 1패스: 제목만 보고 빠르게 거른다
    python -m eval.label review  --date 2026-08-12    # 2패스: 남은 것만 초록까지 읽고 판정
    python -m eval.label recheck --date 2026-08-12    # 1패스 누락률 측정 (표본)
    python -m eval.label status                       # 진행률
    python -m eval.label export                       # eval/goldset.yaml 생성

왜 2패스인가
------------
하루 후보가 700건대인데 관련 논문은 3~7건입니다 (기획안 §9). 700건 초록을 다 읽는 건
불가능하고, 그렇다고 **랭커로 상위 N건을 뽑아 그것만 라벨링하면 평가가 무너집니다.**
평가 대상인 랭커가 정답셋의 범위를 정하게 되어, 랭커가 놓친 논문은 애초에 정답이 될
기회가 없습니다. 개선폭이 실제보다 좋게 나옵니다 (풀링 편향).

→ **전수 커버리지를 유지합니다.** 다만 1패스에서는 제목·카테고리만 보고 초 단위로
   거릅니다. 대부분은 제목만으로 무관이 확실합니다. 애매하거나 관련 있어 보이는 것만
   2패스로 넘겨 초록을 읽습니다.

   1패스에서 거른 것도 `relevant: false`로 **정식 라벨**입니다. 다만 무엇을 근거로
   판정했는지(`basis: title` / `abstract`)를 남겨서, 나중에 "제목만 보고 놓친 게
   아닌가"를 검증할 수 있게 합니다.

저장 구조
--------
    eval/labels.jsonl   append-only 작업 원장. 크래시·중단에 안전하고 이어서 할 수 있음
    eval/goldset.yaml   export 산출물. 기획안 §9가 요구하는 형식

원장은 **덧붙이기만** 합니다. 같은 item_id가 여러 줄이면 마지막 줄이 유효합니다.
undo는 마지막 줄을 지웁니다.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

CANDIDATES_DIR = Path("data/candidates")
EVAL_DIR = Path("eval")
JOURNAL_PATH = EVAL_DIR / "labels.jsonl"
GOLDSET_PATH = EVAL_DIR / "goldset.yaml"

#: 라벨링 순서를 날짜별로 결정적으로 섞습니다.
#: arXiv 제출 순서 그대로 라벨링하면 뒤로 갈수록 집중력이 떨어지는 효과가 **특정
#: 카테고리에 체계적으로** 몰립니다(제출 시각과 분야가 상관이 있으므로). 섞으면
#: 피로 효과가 무작위로 흩어져 편향이 아니라 잡음이 됩니다. 시드는 날짜에서 뽑으므로
#: 중단 후 재개해도 순서가 같습니다.
SHUFFLE_SEED_BASE = "radar-goldset"


# ── 원장 ────────────────────────────────────────────────────────────────


@dataclass
class Label:
    item_id: str
    date: str
    title: str
    relevant: bool | None  # None = 2패스로 보류
    basis: str  # "title" | "abstract"
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "date": self.date,
            "title": self.title,
            "relevant": self.relevant,
            "basis": self.basis,
            "note": self.note,
            "labeled_at": datetime.now(UTC).isoformat(timespec="seconds"),
        }


# 아래 함수들의 경로 인자는 기본값을 `None`으로 두고 호출 시점에 모듈 상수를
# 해석합니다. 기본 인자에 상수를 직접 쓰면 정의 시점에 바인딩되어, 테스트가 상수를
# 갈아끼워도 **실제 data/ 를 읽습니다.** 실제로 그렇게 짰다가 테스트가 진짜 후보
# 파일을 읽는 걸 발견했습니다.


def load_journal(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or JOURNAL_PATH
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def latest_by_item(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """같은 item_id가 여러 번 나오면 마지막 판정이 유효합니다."""
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        latest[row["item_id"]] = row
    return latest


def append_label(label: Label, path: Path | None = None) -> None:
    path = path or JOURNAL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(label.to_dict(), ensure_ascii=False) + "\n")


def undo_last(path: Path | None = None) -> dict[str, Any] | None:
    path = path or JOURNAL_PATH
    rows = load_journal(path)
    if not rows:
        return None
    removed = rows.pop()
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
    return removed


# ── 후보 로딩 ───────────────────────────────────────────────────────────


def load_candidates(date: str, candidates_dir: Path | None = None) -> list[dict[str, Any]]:
    path = (candidates_dir or CANDIDATES_DIR) / f"{date}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 가 없습니다. 먼저 수집하세요:\n"
            f"  python -m src.core.pipeline --channel papers --stage collect"
        )
    with path.open(encoding="utf-8") as fp:
        rows = [json.loads(line) for line in fp if line.strip()]

    # 후보 파일은 append-only 입니다. 수집이 후보 기록과 state.db 마킹 **사이에서**
    # 죽으면 다음 실행이 같은 아이템을 한 번 더 씁니다. 그 상태로 라벨링하면 같은
    # 논문이 두 번 나오고, 두 판정이 엇갈리면 골드셋이 조용히 오염됩니다.
    # 읽는 쪽에서 접습니다 — 후보 파일 자체는 이력이므로 고치지 않습니다.
    items: dict[str, dict[str, Any]] = {}
    for row in rows:
        items[row["id"]] = row
    ordered = list(items.values())
    random.Random(f"{SHUFFLE_SEED_BASE}:{date}").shuffle(ordered)
    return ordered


def available_dates(candidates_dir: Path | None = None) -> list[str]:
    candidates_dir = candidates_dir or CANDIDATES_DIR
    if not candidates_dir.exists():
        return []
    return sorted(p.stem for p in candidates_dir.glob("*.jsonl"))


# ── 터미널 입력 ─────────────────────────────────────────────────────────


def read_key() -> str:
    """키 하나를 엔터 없이 받습니다. 700건을 다루려면 타건 수가 절반이 됩니다."""
    if not sys.stdin.isatty():
        line = sys.stdin.readline()
        return line.strip()[:1].lower() if line else "q"
    import termios
    import tty

    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        char = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
    if char in ("\x03", "\x04"):  # Ctrl-C / Ctrl-D
        raise KeyboardInterrupt
    return char.lower()


def read_line(prompt: str) -> str:
    sys.stdout.write(prompt)
    sys.stdout.flush()
    return (sys.stdin.readline() or "").strip()


BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
YELLOW = "\033[33m"
GREEN = "\033[32m"


def clear() -> None:
    sys.stdout.write("\033[2J\033[H")


def progress_bar(done: int, total: int, width: int = 40) -> str:
    filled = int(width * done / total) if total else width
    pct = 100 * done / total if total else 100.0
    return f"[{'█' * filled}{'·' * (width - filled)}] {done}/{total} ({pct:.1f}%)"


# ── 1패스: 제목 트리아지 ────────────────────────────────────────────────

TRIAGE_HELP = f"""{BOLD}1패스 — 제목만 보고 거릅니다{RESET}
  {BOLD}n{RESET} / {BOLD}space{RESET}  무관     (relevant=false, basis=title)
  {BOLD}k{RESET}            보류     (2패스에서 초록을 읽고 판정)
  {BOLD}u{RESET}            직전 취소
  {BOLD}q{RESET}            저장하고 종료 (다시 실행하면 이어서 합니다)

{DIM}확신이 없으면 k 입니다. 1패스의 목적은 판정이 아니라 '초록을 읽을 가치가 있는가'입니다.
제목만으로 무관이 확실한 것만 n 하세요.{RESET}
"""


def run_triage(date: str, journal_path: Path | None = None) -> None:
    journal_path = journal_path or JOURNAL_PATH
    items = load_candidates(date)
    total = len(items)
    # 판정 상태는 메모리에 들고 갑니다. 키 입력마다 원장 전체를 다시 읽으면
    # 700건 후반부에서 눈에 띄게 느려지고, 느린 도구는 라벨 품질을 떨어뜨립니다.
    decided = latest_by_item(load_journal(journal_path))

    while True:
        todo = [i for i in items if i["id"] not in decided]
        if not todo:
            clear()
            print(f"{GREEN}1패스 완료{RESET} — {date}, {total}건 전부 처리했습니다.")
            deferred = sum(
                1
                for row in decided.values()
                if row["date"] == date and row["relevant"] is None
            )
            print(f"보류 {deferred}건 → 다음: python -m eval.label review --date {date}")
            return

        item = todo[0]
        clear()
        print(f"{BOLD}골드셋 라벨링 · 1패스 · {date}{RESET}")
        print(progress_bar(total - len(todo), total))
        print()
        print(TRIAGE_HELP)
        print("─" * 72)
        print(f"{BOLD}{item['title']}{RESET}")
        print()
        print(f"{DIM}{', '.join(item['categories'])} · {len(item['authors'])}인 · {item['id']}{RESET}")
        print("─" * 72)

        key = read_key()
        if key == "q":
            print("\n저장했습니다. 같은 명령으로 이어서 하세요.")
            return
        if key == "u":
            undo_last(journal_path)
            # 취소는 드물게 일어나므로 여기서만 원장을 다시 읽습니다.
            decided = latest_by_item(load_journal(journal_path))
            continue
        if key == "k":
            label = Label(item["id"], date, item["title"], None, "title")
        elif key in ("n", " ", "\r", "\n"):
            label = Label(item["id"], date, item["title"], False, "title")
        else:
            continue  # 그 외 키는 무시 — 오타로 라벨이 찍히는 걸 막습니다
        append_label(label, journal_path)
        decided[item["id"]] = label.to_dict()


# ── 2패스: 초록 판정 ────────────────────────────────────────────────────

REVIEW_HELP = f"""{BOLD}2패스 — 초록을 읽고 판정합니다{RESET}
  {BOLD}y{RESET}  관련 있음  (relevant=true,  basis=abstract)
  {BOLD}n{RESET}  관련 없음  (relevant=false, basis=abstract)
  {BOLD}m{RESET}  메모를 남기고 판정  (경계 사례는 꼭 남기세요)
  {BOLD}u{RESET}  직전 취소     {BOLD}q{RESET}  저장하고 종료
"""


def run_review(date: str, journal_path: Path | None = None) -> None:
    journal_path = journal_path or JOURNAL_PATH
    items = {i["id"]: i for i in load_candidates(date)}
    decided = latest_by_item(load_journal(journal_path))

    while True:
        pending = [
            item_id
            for item_id, row in decided.items()
            if row["date"] == date and row["relevant"] is None and item_id in items
        ]
        orphaned = [
            item_id
            for item_id, row in decided.items()
            if row["date"] == date and row["relevant"] is None and item_id not in items
        ]
        if orphaned and not pending:
            clear()
            print(
                f"{YELLOW}보류 {len(orphaned)}건이 후보 파일에 없습니다.{RESET}\n"
                f"data/candidates/{date}.jsonl 이 라벨링 이후에 바뀐 것 같습니다.\n"
                "해당 날짜를 다시 수집했다면 그 라벨은 버리고 다시 라벨링하세요."
            )
            return
        if not pending:
            clear()
            relevant = [
                row
                for row in decided.values()
                if row["date"] == date and row["relevant"] is True
            ]
            print(f"{GREEN}2패스 완료{RESET} — {date}")
            print(f"관련 {len(relevant)}건:")
            for row in relevant:
                print(f"  · {row['title'][:66]}")
            print(f"\n다음: python -m eval.label export")
            return

        item_id = pending[0]
        item = items[item_id]
        done = len(
            [
                r
                for r in decided.values()
                if r["date"] == date and r["basis"] == "abstract"
            ]
        )
        clear()
        print(f"{BOLD}골드셋 라벨링 · 2패스 · {date}{RESET}")
        print(progress_bar(done, done + len(pending)))
        print()
        print(REVIEW_HELP)
        print("─" * 72)
        print(f"{BOLD}{item['title']}{RESET}")
        print()
        print(_wrap(item["abstract"], 72))
        print()
        print(f"{DIM}{', '.join(item['categories'])} · {item['url']}{RESET}")
        print("─" * 72)

        key = read_key()
        if key == "q":
            print("\n저장했습니다. 같은 명령으로 이어서 하세요.")
            return
        if key == "u":
            undo_last(journal_path)
            decided = latest_by_item(load_journal(journal_path))
            continue

        note = ""
        if key == "m":
            print()
            note = read_line(f"{YELLOW}메모(왜 이렇게 판정했는가): {RESET}")
            key = ""
            while key not in ("y", "n"):
                sys.stdout.write("y(관련) / n(무관): ")
                sys.stdout.flush()
                key = read_key()
        if key in ("y", "n"):
            label = Label(item_id, date, item["title"], key == "y", "abstract", note)
            append_label(label, journal_path)
            decided[item_id] = label.to_dict()


def _wrap(text: str, width: int) -> str:
    import textwrap

    return "\n".join(textwrap.wrap(text, width))


# ── 재검토: 1패스 누락률 측정 ★ ──────────────────────────────────────────

RECHECK_HELP = f"""{BOLD}재검토 — 1패스에서 제목만 보고 버린 것 중 일부를 초록까지 읽습니다{RESET}
  {BOLD}y{RESET}  사실은 관련 있었음 (라벨을 뒤집습니다)
  {BOLD}n{RESET}  무관이 맞음
  {BOLD}u{RESET}  직전 취소     {BOLD}q{RESET}  저장하고 종료

{DIM}목적은 라벨 수정이 아니라 **1패스 누락률 측정**입니다. 표본에서 몇 건이 뒤집히는지가
'제목만 보고 놓쳤을 수 있다'는 한계를 숫자로 만들어 줍니다. 뒤집힌 게 0건이면
1패스가 신뢰할 만하다는 근거가 되고, 많으면 1패스 기준을 느슨하게 바꿔야 합니다.{RESET}
"""


def run_recheck(date: str, sample: int = 40, journal_path: Path | None = None) -> None:
    """1패스에서 `basis: title` 로 버린 것 중 표본을 뽑아 초록까지 읽고 재판정합니다.

    전수를 다시 보면 1패스를 한 의미가 없으므로 **표본**입니다. 표본에서의 누락률이
    전체 누락률의 추정치가 됩니다. 표본 추출도 날짜 고정 시드라 재개해도 같습니다.
    """
    journal_path = journal_path or JOURNAL_PATH
    items = {i["id"]: i for i in load_candidates(date)}
    decided = latest_by_item(load_journal(journal_path))

    rejected = sorted(
        item_id
        for item_id, row in decided.items()
        if row["date"] == date and row["basis"] == "title" and row["relevant"] is False
    )
    if not rejected:
        print(f"{date}: 1패스에서 버린 항목이 없습니다. 먼저 triage 를 하세요.")
        return

    picked = rejected[:]
    random.Random(f"{SHUFFLE_SEED_BASE}:recheck:{date}").shuffle(picked)
    picked = picked[: min(sample, len(picked))]

    while True:
        todo = [i for i in picked if decided[i]["basis"] == "title"]
        if not todo:
            clear()
            flipped = [
                i
                for i in picked
                if decided[i]["basis"] == "recheck" and decided[i]["relevant"]
            ]
            rate = len(flipped) / len(picked) if picked else 0.0
            print(f"{GREEN}재검토 완료{RESET} — {date}")
            print(f"표본 {len(picked)}건 중 뒤집힘 {len(flipped)}건 (누락률 {rate:.1%})")
            if flipped:
                print(f"\n{YELLOW}1패스에서 놓친 논문:{RESET}")
                for item_id in flipped:
                    print(f"  · {decided[item_id]['title'][:64]}")
                print(
                    "\n놓친 이유를 docs/라벨링_기준.md §5에 남기세요. "
                    "1패스 기준을 고쳐야 할 수도 있습니다."
                )
            else:
                print("\n제목 트리아지가 이 표본에서는 놓친 게 없습니다.")
            return

        item_id = todo[0]
        item = items[item_id]
        clear()
        print(f"{BOLD}골드셋 라벨링 · 재검토 · {date}{RESET}")
        print(progress_bar(len(picked) - len(todo), len(picked)))
        print()
        print(RECHECK_HELP)
        print("─" * 72)
        print(f"{BOLD}{item['title']}{RESET}")
        print()
        print(_wrap(item["abstract"], 72))
        print()
        print(f"{DIM}{', '.join(item['categories'])} · {item['url']}{RESET}")
        print("─" * 72)

        key = read_key()
        if key == "q":
            print("\n저장했습니다. 같은 명령으로 이어서 하세요.")
            return
        if key == "u":
            undo_last(journal_path)
            decided = latest_by_item(load_journal(journal_path))
            continue
        if key in ("y", "n"):
            label = Label(
                item_id,
                date,
                item["title"],
                key == "y",
                "recheck",
                "1패스 제목 기각분 재검토" if key == "y" else "",
            )
            append_label(label, journal_path)
            decided[item_id] = label.to_dict()


# ── 상태 / 내보내기 ─────────────────────────────────────────────────────


def run_status(journal_path: Path | None = None) -> None:
    decided = latest_by_item(load_journal(journal_path))
    dates = available_dates()
    if not dates:
        print("수집된 후보가 없습니다. python -m src.core.pipeline --channel papers --stage collect")
        return

    print(f"{BOLD}골드셋 진행 상황{RESET}\n")
    print(f"{'날짜':<12} {'후보':>6} {'판정':>6} {'보류':>6} {'관련':>6}  진행")
    print("─" * 68)
    for date in dates:
        total = len(load_candidates(date))
        rows = [r for r in decided.values() if r["date"] == date]
        counts = Counter(
            "pending" if r["relevant"] is None else ("yes" if r["relevant"] else "no")
            for r in rows
        )
        settled = counts["yes"] + counts["no"]
        print(
            f"{date:<12} {total:>6} {settled:>6} {counts['pending']:>6} "
            f"{counts['yes']:>6}  {progress_bar(settled, total, 20)}"
        )
    print()
    labeled_dates = {r["date"] for r in decided.values()}
    if len(labeled_dates) < 3:
        print(f"{YELLOW}기획안 §9는 3일치를 요구합니다. 현재 {len(labeled_dates)}일치.{RESET}")


def run_export(journal_path: Path | None = None, out: Path | None = None) -> Path:
    out = out or GOLDSET_PATH
    decided = latest_by_item(load_journal(journal_path))
    pending = [r for r in decided.values() if r["relevant"] is None]
    if pending:
        raise SystemExit(
            f"보류 {len(pending)}건이 남아 있습니다. 먼저 review 를 끝내세요.\n"
            "  python -m eval.label review --date <날짜>"
        )
    if not decided:
        raise SystemExit("라벨이 없습니다.")

    rows = sorted(decided.values(), key=lambda r: (r["date"], r["item_id"]))
    by_date = Counter(r["date"] for r in rows)
    relevant = Counter(r["date"] for r in rows if r["relevant"])
    recheck_rows = [r for r in rows if r["basis"] == "recheck"]
    recheck_total = len(recheck_rows)
    recheck_flipped = sum(1 for r in recheck_rows if r["relevant"])

    document = {
        "version": 1,
        "channel": "papers",
        # 라벨링 기준 문서 없이는 재현이 불가능합니다 (기획안 §9-4).
        "criteria_doc": "docs/라벨링_기준.md",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "summary": {
            "dates": sorted(by_date),
            "total": len(rows),
            "relevant": sum(relevant.values()),
            "per_date": {
                date: {"candidates": by_date[date], "relevant": relevant[date]}
                for date in sorted(by_date)
            },
            # docs/라벨링_기준.md §6 한계 2를 숫자로 만드는 항목입니다.
            # 1패스에서 제목만 보고 버린 것 중 표본을 재검토했을 때 몇 건이 뒤집혔는가.
            # recheck 를 안 돌렸으면 sampled=0 이고, 그 사실 자체가 기록됩니다.
            "title_pass_recheck": {
                "sampled": recheck_total,
                "flipped_to_relevant": recheck_flipped,
                "miss_rate": round(recheck_flipped / recheck_total, 4)
                if recheck_total
                else None,
            },
        },
        "labels": [
            {
                "date": r["date"],
                "item_id": r["item_id"],
                "relevant": bool(r["relevant"]),
                # basis 는 기획안 §9 형식에 없는 추가 필드입니다. 제목만 보고 내린
                # 판정과 초록까지 읽고 내린 판정을 구분하지 않으면, 나중에 "1패스에서
                # 놓친 게 아닌가"를 검증할 방법이 없습니다.
                "basis": r["basis"],
                "title": r["title"],
                **({"note": r["note"]} if r.get("note") else {}),
            }
            for r in rows
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(document, fp, allow_unicode=True, sort_keys=False, width=100)

    print(f"{GREEN}{out}{RESET} — {len(rows)}건, 관련 {sum(relevant.values())}건")
    for date in sorted(by_date):
        print(f"  {date}: 후보 {by_date[date]}건 중 관련 {relevant[date]}건")
    if recheck_total:
        print(f"  1패스 재검토: 표본 {recheck_total}건 중 {recheck_flipped}건 뒤집힘")
    else:
        print(
            f"  {YELLOW}1패스 재검토 미실시{RESET} — "
            "python -m eval.label recheck --date <날짜> 로 누락률을 재두세요"
        )
    if len(by_date) < 3:
        print(f"\n{YELLOW}경고: {len(by_date)}일치입니다. 기획안 §9는 3일치를 요구합니다.{RESET}")
    return out


# ── CLI ─────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.label")
    sub = parser.add_subparsers(dest="command", required=True)

    triage = sub.add_parser("triage", help="1패스 — 제목만 보고 빠르게 거른다")
    triage.add_argument("--date", required=True)

    review = sub.add_parser("review", help="2패스 — 보류분의 초록을 읽고 판정")
    review.add_argument("--date", required=True)

    recheck = sub.add_parser(
        "recheck", help="1패스 기각분 표본 재검토 — 제목만 보고 놓친 비율을 잰다"
    )
    recheck.add_argument("--date", required=True)
    recheck.add_argument("--sample", type=int, default=40)

    sub.add_parser("status", help="진행률")
    sub.add_parser("export", help="eval/goldset.yaml 생성")

    args = parser.parse_args(argv)
    try:
        if args.command == "triage":
            run_triage(args.date)
        elif args.command == "review":
            run_review(args.date)
        elif args.command == "recheck":
            run_recheck(args.date, args.sample)
        elif args.command == "status":
            run_status()
        elif args.command == "export":
            run_export()
    except KeyboardInterrupt:
        print("\n중단했습니다. 여기까지는 저장돼 있습니다.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
