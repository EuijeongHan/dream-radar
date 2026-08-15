"""Hugging Face Daily Papers — **보류 중인 스텁입니다.**

붙이지 않은 이유 (결정_M0 §3)
-----------------------------
HF Daily Papers에는 공식 문서화된 API 엔드포인트가 없습니다. CLAUDE.md §3-5는
"공식 API만 사용"이고, 문서화되지 않은 내부 엔드포인트는 예고 없이 바뀌어도 항의할
근거가 없습니다. 추측으로 붙이지 않습니다.

이 파일이 빈 채로 존재하는 이유는, 나중에 누가 "왜 hf_papers.py가 없지"라며 즉흥적으로
스크래퍼를 짜는 걸 막기 위해서입니다. 붙이려면 **이용약관과 엔드포인트 안정성을 먼저
확인**받으세요.

`data/profile.papers_1.yaml`의 `sources.hf_daily_papers.enabled`도 `false`입니다.
설정과 코드가 어긋난 채로 두면 다음 사람이 "왜 안 도나"로 시간을 씁니다.
"""

from __future__ import annotations

from src.core.models import Item
from src.sources.base import SourceUnavailable


class HFDailyPapersAdapter:
    name = "hf_daily"
    channel = "papers"
    publish_scope = "public"

    def collect(self) -> list[Item]:
        raise SourceUnavailable(
            "HF Daily Papers는 공식 문서화된 API가 없어 보류 상태입니다 "
            "(CLAUDE.md §3-5, 결정_M0 §3). 이용약관·엔드포인트 안정성 확인 후 구현하세요."
        )
