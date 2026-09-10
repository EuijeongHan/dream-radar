"""Radar papers 채널 DAG — collect → rank → summarize → publish.

무엇을 하지 않는가 ★
--------------------
이 DAG는 **파이프라인 로직을 재구현하지 않습니다.** `python -m src.core.pipeline`
CLI를 그대로 호출합니다. 이유는 재사용 규칙입니다:

  R2  arXiv 요청은 `SingleConnectionRateLimiter.slot()` 안에서만 열어야 한다
  R3  원장은 `ledger.RunRecord` + `ledger.append()` 로만 써야 한다

DAG가 소스 어댑터나 원장을 직접 건드리면 두 규칙이 조용히 깨집니다. CLI를 부르면
슬롯도 원장도 기존 경로를 그대로 탑니다.

동시성 — CLAUDE.md §3-6 이 설계를 강제합니다 ★
-----------------------------------------------
    "arXiv 요청 병렬화 금지. 3초 1회 + 단일 커넥션.
     권장이 아니라 요구사항. **본인 통제 머신 전체에 합산 적용**"

`SingleConnectionRateLimiter`는 한 프로세스 안에서만 유효합니다. Airflow가 태스크를
병렬로 띄우면 프로세스가 여러 개가 되고, 각자 자기 슬롯을 지키면서 **합산으로는
3초 1회를 위반**합니다. 테스트는 통과하는데 IP가 차단되는 종류의 실패입니다.

그래서 세 겹으로 막습니다:

  1. `max_active_runs=1`   — DAG 런이 겹치지 않음
  2. `max_active_tasks=1`  — 이 DAG 안에서 태스크가 동시에 안 돎
  3. `catchup=False`       — 며칠치가 한꺼번에 밀려 들어오지 않음

**GitHub Actions의 `daily-papers.yml`과 동시에 돌리지 마세요.** 프로세스가 분리돼
있어 위 셋으로도 막을 수 없습니다. Airflow로 옮기면 Actions 쪽을 끄십시오.

미구현 스테이지
--------------
현재 파이프라인은 `collect`만 구현돼 있고, 나머지를 요청하면 argparse가 죽습니다
(기획안 §10 — 조용히 넘어가지 않는다). DAG를 빨간불로 두면 쓸모가 없으므로,
아직 구현되지 않은 스테이지는 자리만 잡아 둡니다. 구현되면 `IMPLEMENTED_STAGES`에
이름을 옮기면 실제 실행으로 바뀝니다 — DAG 모양은 그대로입니다.
"""

from __future__ import annotations

import os
import pendulum
from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator

# ── 저장소 경로 ──────────────────────────────────────────────────────────
# Airflow는 dags/ 만 스캔하므로 저장소 루트를 명시적으로 잡는다.
# 다른 머신에서 돌릴 때는 RADAR_HOME 으로 덮어쓴다.
RADAR_HOME = os.environ.get("RADAR_HOME", os.path.expanduser("~/radar"))

CHANNEL = "papers"
STAGES = ("collect", "rank", "summarize", "publish")

#: 파이프라인이 실제로 처리하는 스테이지. 구현되면 여기에 추가한다.
#: (`src/core/pipeline.py` main()이 collect 외에는 parser.error 로 죽는다)
IMPLEMENTED_STAGES = {"collect"}

#: CLI 호출 — 저장소 루트에서 모듈로 실행해야 상대 경로(data/, eval/)가 맞는다
RUN_STAGE = (
    "cd {home} && "
    "python -m src.core.pipeline --channel {channel} --stage {stage}"
)

#: 수집분을 오브젝트 스토리지로 증분 적재한다 (`src/storage/sync.py`).
#: 스테이지가 아니라 별도 태스크인 이유: 원장(`ledger.STAGES`)은 확정 스키마이고
#: (작업규약 §3-T7), 적재는 파이프라인 산출물이 아니라 그 **보존** 이기 때문이다.
SYNC_S3 = "cd {home} && python -m src.storage.sync --channel {channel}"

with DAG(
    dag_id="radar_papers",
    description="arXiv 논문 수집·랭킹·요약·발행 (CLAUDE.md §3-6 준수: 직렬 실행)",
    schedule="0 8 * * *",  # 매일 08:00 KST
    start_date=pendulum.datetime(2026, 8, 12, tz="Asia/Seoul"),
    catchup=False,          # ★ 밀린 날짜를 한꺼번에 돌리면 arXiv 레이트리밋 위반
    max_active_runs=1,      # ★ 런 중첩 금지
    max_active_tasks=1,     # ★ 태스크 병렬 금지 — 프로세스가 여러 개가 되면 슬롯이 무의미
    default_args={
        "owner": "euijeong",
        "retries": 2,
        "retry_delay": pendulum.duration(minutes=10),
        "retry_exponential_backoff": True,
        "max_retry_delay": pendulum.duration(hours=1),
        "execution_timeout": pendulum.duration(minutes=45),
    },
    tags=["radar", "papers", "M0"],
    doc_md=__doc__,
) as dag:

    previous = None
    collect_task = None

    for stage in STAGES:
        if stage in IMPLEMENTED_STAGES:
            task = BashOperator(
                task_id=stage,
                bash_command=RUN_STAGE.format(
                    home=RADAR_HOME, channel=CHANNEL, stage=stage
                ),
                doc_md=(
                    f"`python -m src.core.pipeline --channel {CHANNEL} --stage {stage}`\n\n"
                    "파이프라인 CLI를 그대로 부른다 (R2 슬롯 · R3 원장 유지)."
                ),
            )
        else:
            task = EmptyOperator(
                task_id=stage,
                doc_md=(
                    f"**미구현.** `{stage}` 는 아직 파이프라인에 없다.\n\n"
                    f"구현되면 `IMPLEMENTED_STAGES` 에 `{stage}` 를 추가하면 "
                    "실제 실행으로 바뀐다 — DAG 모양은 그대로다."
                ),
            )

        if stage == "collect":
            collect_task = task

        if previous is not None:
            previous >> task
        previous = task

    # ── 수집 직후 적재 ────────────────────────────────────────────────────
    # collect 뒤에 두는 이유: 랭킹·요약이 실패해도 **수집분은 이미 보존**된다.
    # 파이프라인 끝에 두면 요약 단계가 죽는 날의 후보 파일이 이 머신에만 남는데,
    # `data/candidates/` 는 gitignore라 머신이 죽으면 같이 사라진다. arXiv 창은
    # 48시간이라 과거분은 재수집으로 복구되지 않는다.
    sync_task = BashOperator(
        task_id="sync_s3",
        bash_command=SYNC_S3.format(home=RADAR_HOME, channel=CHANNEL),
        doc_md=(
            "`python -m src.storage.sync --channel papers`\n\n"
            "`data/candidates/*.jsonl` 를 `s3://$RADAR_S3_BUCKET/"
            "{prefix}/channel=papers/dt=YYYY-MM-DD/` 로 증분 적재한다.\n\n"
            "- 해시가 같으면 건너뛴다 (증분)\n"
            "- 매니페스트를 잃으면 원격 head 로 복구한다 (전량 재업로드 방지)\n"
            "- 업로드 직전 줄 단위로 `assert_public_scope()` 를 태운다 (CLAUDE.md §3-3)\n\n"
            "**환경변수**: `RADAR_S3_BUCKET` 필수, `RADAR_S3_PREFIX` 선택(기본 `radar`).\n"
            "미설정이면 조용히 건너뛰지 않고 **exit 1** 이다."
        ),
    )
    if collect_task is not None:
        collect_task >> sync_task
