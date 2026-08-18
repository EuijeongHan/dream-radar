"""랭킹 레이어 (기획안_2 §5).

    embed.py     임베딩 포트 — 프로토콜 · 해시 스텁 · bge-m3 어댑터 · 캐시 (§5.1, §5.3)
    profile.py   프로파일 관심사 → 질의 벡터 · 점수 합성 (§5.2)
    retrieve.py  1차 랭킹 — 임베딩 유사도 → stage1_top_n (§5.1, §5.7)
    rerank.py    2차 랭킹 — cross-encoder raw 로짓 → final_n (§5.1, §9.4)

이 패키지는 **무거운 의존성을 모듈 최상단에서 import 하지 않습니다.** torch·
sentence-transformers 는 `requirements-eval.txt` 에만 있고 Actions 러너에는 없습니다
(§9.10). `import src.rank.embed` 만으로 죽으면 운영 파이프라인이 통째로 멈춥니다.
실모델 어댑터는 전부 지연 import 입니다.
"""
