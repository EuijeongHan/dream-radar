[논문 한 편, 5장으로 읽기 · PAPER RADAR 002]

AI 에이전트에게 H100 한 장과 두 시간을 주면, 사람보다 더 기발하게 LLM 추론 서버를 최적화할까요?

InferenceBench는 에이전트가 정해진 레시피를 따라가는 대신 프레임워크, 양자화, 어텐션 백엔드와 실행 설정을 직접 선택하게 합니다. 완성된 서버는 속도뿐 아니라 품질·무결성 게이트와 깨끗한 환경에서의 재실행까지 통과해야 합니다.

15개 프런티어 에이전트 구성을 평가한 결과, 최고 에이전트는 순진한 PyTorch 기준보다 종합 8.08배 빨랐습니다. 하지만 같은 두 시간을 쓴 단순 하이퍼파라미터 탐색은 11.53배로 모든 시나리오에서 에이전트를 앞섰습니다.

흥미로운 지점은 ‘모르는 것’보다 ‘덜 탐색하는 것’이었습니다. 180번의 실행 중 93.9%가 최종적으로 vLLM을 선택했고, 두 시간 동안 시험한 비기본 vLLM 설정 수의 중앙값은 단 하나였습니다.

에이전트 연구 자동화에는 더 긴 추론만이 아니라 서로 다른 가설을 강제로 만들고, 병렬로 비교하며, 가장 좋았던 정상 작동 상태를 보존하는 실행 구조가 필요해 보입니다.

이 게시물은 논문 본문과 공식 공개 저장소를 바탕으로 직접 요약했습니다. 수치는 Mistral-7B-Instruct-v0.3, NVIDIA H100 1장, 실행당 2시간이라는 논문의 평가 조건 안에서 해석해야 합니다.

📄 InferenceBench: A Benchmark for Open-Ended LLM Inference Optimization by AI Agents

✍️ Jehyeok Yeon, Ben Rank, Maksym Andriushchenko

🔗 원문
https://arxiv.org/abs/2607.20468

💻 공식 코드 (Apache-2.0)
https://github.com/aisa-group/InferenceBench

#AI #AI에이전트 #LLMInference #논문요약 #한해원의관측소
