"""워크넷 채용정보 수집기 — **골격만**. 응답 스키마 미확정 (기획안_2 §8.2, M5).

공공데이터포털 "한국고용정보원_워크넷 채용정보" (dataset 3038225).
`profile.jobs_1.yaml`의 `sources.worknet.dataset` 이 이 번호입니다.

왜 지금 구현이 없는가
---------------------
활용신청이 승인되기 전에는 **응답 스키마를 볼 수 없습니다.** 공공데이터포털의
채용정보 계열 API는 오퍼레이션마다 필드명·중첩·인코딩이 제각각이라, 가이드 문서만
보고 파서를 미리 쓰면 두 가지 중 하나가 됩니다.

- 실제 응답과 어긋나 예외로 죽는다 → 그나마 낫습니다
- 필드가 없어 **전부 빈 문자열로 채워진 Item**이 만들어진다 → 하드 필터가
  판정 근거를 못 찾고, 원장에는 "수집 N건"이 정상처럼 남습니다

두 번째가 이 저장소가 반복해서 밟은 함정입니다 (기획안_2 §9 — "전부 초록불로
위장했다"). 그래서 **추측으로 채우지 않고** `NotImplementedError`로 막아둡니다.
`collect()`가 빈 리스트를 돌려주는 스텁으로 만들지 않은 것도 같은 이유입니다
(`src/sources/hf_papers.py`의 `test_hf_papers_is_a_deliberate_stub` 참조).

승인 후 확정할 것 — 이 3개를 실제 응답으로 확인하기 전에는 파서를 쓰지 마세요
-------------------------------------------------------------------------
1. **응답 포맷과 루트 경로**: XML/JSON 중 무엇으로 오는지, 목록의 경로가
   `response > body > items > item`인지. 총건수 필드명(`totalCount`)과 페이징
   파라미터명(`pageNo`/`numOfRows`)도 함께.
2. **하드 필터 판정 근거의 필드명**: 경력(신입/경력무관 구분과 연차),
   근무지역, 마감 여부. `src/verify/filter_check.py`가 읽는 `raw` 키
   (`active` · `experience_level{code,min,max}` · `location{code,name}`)로
   **매핑이 가능한지**를 먼저 봅니다. 매핑이 안 되면 필터를 못 겁니다.
3. **원문 링크 필드**: `Item`은 빈 url을 거부합니다. 워크넷 공고 상세 URL을
   응답에서 직접 주는지, 공고번호로 조립해야 하는지.

키
--
`DATA_GO_KR_SERVICE_KEY` (환경변수). 인코딩/디코딩 키 두 종류가 발급되는데
`requests`가 파라미터를 다시 인코딩하므로 **디코딩 키**를 넣습니다.
키가 없으면 조용히 폴백하지 않고 예외로 죽습니다 (델타 §D6.2).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Mapping

log = logging.getLogger(__name__)

#: 공공데이터포털 서비스키. 코드·프로파일·커밋에 넣지 않습니다 (CLAUDE.md §3-2).
SERVICE_KEY_ENV = "DATA_GO_KR_SERVICE_KEY"

#: 활용신청 대상 데이터셋 (한국고용정보원_워크넷 채용정보).
DATASET_ID = "3038225"

Transport = Callable[[str, dict[str, Any]], str]


class WorknetError(RuntimeError):
    """워크넷 수집 실패. 메시지에 서비스키가 들어가지 않습니다 (CLAUDE.md §3-2)."""


class MissingServiceKey(WorknetError):
    """키가 없습니다. **조용히 빈 결과로 폴백하지 않습니다** (델타 §D6.2)."""


class WorknetAdapter:
    name = "worknet"
    channel = "jobs"
    # ★ CLAUDE.md §3-3 — 채용공고 공개 게시 금지. 사람인과 같은 이유로 리터럴입니다.
    # 워크넷은 공공데이터라 재배포 조건이 다를 수 있지만, **확인 전에는 비공개**입니다.
    # 확인되기 전에 열어두면 되돌릴 수 없습니다.
    publish_scope = "private"

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        transport: Transport | None = None,
    ) -> None:
        self._config: Mapping[str, Any] = config or {}
        self._transport = transport
        dataset = str(self._config.get("dataset") or DATASET_ID)
        if dataset != DATASET_ID:
            log.warning(
                "worknet dataset 이 %s 입니다 (기대값 %s). 스키마 확인 대상이 바뀌었는지 확인하세요.",
                dataset,
                DATASET_ID,
            )
        self.dataset = dataset

    def collect(self) -> list[Any]:
        """★ 아직 구현하지 않았습니다. 응답 스키마 미확정 (모듈 docstring 참조).

        키 확인을 **먼저** 하는 것은 의도입니다. 자격증명 경로가 실제로 연결돼
        있는지가 스키마보다 먼저 드러나야, 승인 직후 파서만 채우면 됩니다.
        """
        _require_service_key()
        raise NotImplementedError(
            "워크넷 응답 스키마가 미확정입니다 (기획안_2 §8.2). "
            f"공공데이터포털 {self.dataset} 활용신청 승인 후 실제 응답으로 "
            "① 목록 루트 경로 ② 경력·지역·마감 필드명 ③ 원문 링크 필드를 확인하고 "
            "src/sources/worknet.py 를 채우세요. 추측으로 채우면 빈 Item 이 "
            "정상처럼 수집됩니다"
        )


def _require_service_key() -> str:
    """키가 없으면 명확한 예외. 키 값 자체는 어떤 메시지에도 넣지 않습니다."""
    key = os.environ.get(SERVICE_KEY_ENV, "").strip()
    if not key:
        raise MissingServiceKey(
            f"환경변수 {SERVICE_KEY_ENV} 가 없습니다. 공공데이터포털 활용신청 후 "
            "발급되는 디코딩 키를 넣으세요. 키를 코드·프로파일·커밋에 넣지 "
            "마세요 (CLAUDE.md §3-2)"
        )
    return key
