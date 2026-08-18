"""M6 피드백 환류 — 프로파일 weight 갱신 (기획안_2 §8.3).

★ 이 파일이 막는 것은 "프로파일이 며칠치 라벨로 뒤집히는 것"입니다. 상한(±0.1)·
  클램프([0.1, 1.5])·50건 하한·**기존 파일 불변**(CLAUDE.md §2) 네 가지가 핵심이고,
  넷 중 하나라도 빠지면 랭킹 평가의 기준선이 조용히 사라집니다.

임베딩은 주입합니다 — 이 테스트는 모델도 키도 네트워크도 쓰지 않습니다 (델타 §D6.2).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.core import config
from src.feedback.update_profile import (
    MAX_DELTA,
    MIN_LABELS,
    WEIGHT_MAX,
    WEIGHT_MIN,
    compute_updates,
    load_feedback,
    next_profile_path,
    update_profile,
)

STEM = "profile.test"

#: 고정 어휘. 임베딩을 흉내 내되 **결정적**이어야 테스트가 재현됩니다.
VOCAB = (
    "diffusion",
    "editing",
    "identity",
    "retrieval",
    "rag",
    "evaluation",
    "inference",
    "quantization",
    "misc",
)

TOPIC_TEXT = {
    "alpha": "diffusion editing identity",
    "beta": "retrieval rag evaluation",
    "gamma": "inference quantization",
}


class BagOfWordsEmbedder:
    """주입되는 Embedder 스텁 — `src/rank/embed.py`의 `Embedder` 포트를 만족합니다.

    `HashStubEmbedder`를 쓰지 않는 이유: 해시 스텁은 방향이 사실상 무작위라 어느
    관심사가 가장 가까운지 **테스트가 통제할 수 없습니다.** "alpha 를 읽으면 alpha 가
    오른다"를 검사하려면 근접 관계가 결정적이어야 합니다. 단어 빈도 벡터가 그 역할만
    합니다 — 랭킹 품질과는 아무 관계가 없습니다.
    """

    model_id = "bag_of_words_test_stub"
    revision = "test"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, texts):
        batch = list(texts)
        self.calls.append(batch)
        return [
            tuple(float(text.lower().split().count(word)) for word in VOCAB) for text in batch
        ]


# ── 픽스처 ───────────────────────────────────────────────────────────────


def write_profile(root: Path, name: str, weights: dict[str, float]) -> Path:
    """`data/profile.papers_1.yaml`과 같은 모양의 최소 프로파일을 씁니다."""
    profile = {
        "version": 1,
        "channel": "papers",
        "visibility": "public",
        "language": "en",
        "interests": [
            {
                "id": interest_id,
                "label": f"사람이 읽는 이름 {interest_id}",  # 임베딩 대상이 아닙니다
                "weight": weight,
                "origin": "테스트",
                "text": TOPIC_TEXT[interest_id],
            }
            for interest_id, weight in weights.items()
        ],
        "exclude": {"soft": [{"text": "misc", "penalty": 0.3}]},
        "sources": {"arxiv": {"enabled": True, "min_interval_sec": 3.0, "max_connections": 1}},
        "license_gate": {
            "fulltext_ok_licenses": ["http://creativecommons.org/publicdomain/zero/1.0/"],
            "default_fulltext_ok": False,
            "lookup": "oai-pmh",
            "lookup_scope": "selected_only",
        },
        "selection": {"stage1_top_n": 30, "final_n": 5, "min_score_threshold": 0.42},
    }
    path = root / name
    path.write_text(yaml.safe_dump(profile, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """`data/` 대역. 실제 `data/`를 절대 건드리지 않습니다."""
    write_profile(tmp_path, f"{STEM}.yaml", {"alpha": 0.5, "beta": 0.5, "gamma": 0.5})
    write_profile(tmp_path, f"{STEM}_1.yaml", {"alpha": 1.0, "beta": 0.9, "gamma": 0.6})
    return tmp_path


@pytest.fixture
def profile(root: Path) -> dict:
    data, path = config.load_profile(STEM, root=root)
    assert path.name == f"{STEM}_1.yaml", "최고 번호를 못 골랐습니다"
    return data


def corpus(spec: list[tuple[int, str, str]]) -> tuple[list[dict], list[dict]]:
    """`[(건수, action, topic), ...]` → (items, feedback_rows). item_id는 전부 다릅니다."""
    items: list[dict] = []
    rows: list[dict] = []
    counter = 0
    for count, action, topic in spec:
        for _ in range(count):
            counter += 1
            item_id = f"arxiv:2608.{counter:05d}"
            items.append(
                {
                    "id": item_id,
                    "title": TOPIC_TEXT[topic],
                    "abstract": f"{TOPIC_TEXT[topic]} misc",
                }
            )
            rows.append(
                {
                    "ts": "2026-08-18T07:00:00+09:00",
                    "item_id": item_id,
                    "action": action,
                    "channel": "papers",
                }
            )
    return items, rows


def by_id(update) -> dict:
    return {change.id: change for change in update.changes}


def write_feedback(path: Path, rows: list[dict]) -> Path:
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


# ── 50건 하한 ★ ──────────────────────────────────────────────────────────


def test_refuses_fewer_than_50_labels(profile):
    """★ 라벨 50건 미만이면 갱신하지 않습니다 (§8.3 "50건 이상 누적 후").

    깨뜨리는 법: update_profile.py 의 MIN_LABELS 를 1로 낮추면 빨간불.
    확인일: 2026-08-18
    """
    items, rows = corpus([(MIN_LABELS - 1, "read", "alpha")])

    with pytest.raises(ValueError, match="50건 미만"):
        compute_updates(profile, rows, items, BagOfWordsEmbedder())


def test_accepts_exactly_50_labels(profile):
    """경계값: 정확히 50건은 통과합니다.

    깨뜨리는 법: 검사를 `<=` 로 바꾸면 빨간불.
    확인일: 2026-08-18
    """
    items, rows = corpus([(MIN_LABELS, "read", "alpha")])

    update = compute_updates(profile, rows, items, BagOfWordsEmbedder())

    assert update.labels_used == MIN_LABELS
    assert by_id(update)["alpha"].delta > 0


def test_refused_update_writes_nothing(root, tmp_path):
    """거부하면 파일도 남기지 않습니다 — 반쪽 갱신이 최악입니다.

    깨뜨리는 법: update_profile() 에서 compute_updates 호출을 파일 쓰기 뒤로 옮기면 빨간불.
    확인일: 2026-08-18
    """
    items, rows = corpus([(10, "read", "alpha")])
    feedback = write_feedback(tmp_path / "feedback.jsonl", rows)

    with pytest.raises(ValueError, match="50건 미만"):
        update_profile(
            STEM, embedder=BagOfWordsEmbedder(), items=items, feedback_path=feedback, root=root
        )

    assert not (root / f"{STEM}_2.yaml").exists()


# ── 갱신 폭 상한 · 클램프 ★ ──────────────────────────────────────────────


def test_delta_never_exceeds_max_delta(profile):
    """★ 표가 아무리 몰려도 1회 갱신 폭은 ±0.1 을 넘지 않습니다 (§8.3).

    깨뜨리는 법: _delta() 가 비율 대신 건수를 쓰게 하면(`MAX_DELTA * (pos - neg)`)
    빨간불 — 200건이면 delta 가 20 이 됩니다.
    확인일: 2026-08-18
    """
    items, rows = corpus([(200, "read", "alpha")])

    update = compute_updates(profile, rows, items, BagOfWordsEmbedder())

    alpha = by_id(update)["alpha"]
    assert alpha.delta == pytest.approx(MAX_DELTA)
    assert alpha.after == pytest.approx(1.1)
    assert all(abs(change.delta) <= MAX_DELTA + 1e-9 for change in update.changes)


def test_weight_is_clamped_at_ceiling(root, tmp_path):
    """★ 상한 1.5 를 넘지 않습니다 (§8.3).

    깨뜨리는 법: compute_updates 의 min(WEIGHT_MAX, ...) 를 지우면 1.55 가 되어 빨간불.
    확인일: 2026-08-18
    """
    write_profile(root, f"{STEM}_2.yaml", {"alpha": 1.45, "beta": 0.9, "gamma": 0.6})
    data, _ = config.load_profile(STEM, root=root)
    items, rows = corpus([(60, "save", "alpha")])

    update = compute_updates(data, rows, items, BagOfWordsEmbedder())

    alpha = by_id(update)["alpha"]
    assert alpha.after == pytest.approx(WEIGHT_MAX)
    assert alpha.delta == pytest.approx(0.05)  # 요청한 0.1 이 아니라 실제로 움직인 폭


def test_weight_is_clamped_at_floor(root):
    """★ 하한 0.1 아래로 내려가지 않습니다. 0 이 되면 그 관심사는 다시 올라올
    피드백을 받을 기회조차 없습니다 (§8.3).

    깨뜨리는 법: max(WEIGHT_MIN, ...) 를 지우면 0.05 가 되어 빨간불.
    확인일: 2026-08-18
    """
    write_profile(root, f"{STEM}_2.yaml", {"alpha": 1.0, "beta": 0.9, "gamma": 0.15})
    data, _ = config.load_profile(STEM, root=root)
    items, rows = corpus([(50, "read", "alpha"), (10, "skip", "gamma")])

    update = compute_updates(data, rows, items, BagOfWordsEmbedder())

    gamma = by_id(update)["gamma"]
    assert gamma.after == pytest.approx(WEIGHT_MIN)
    assert gamma.delta == pytest.approx(-0.05)


# ── 라벨 해석 ★ ──────────────────────────────────────────────────────────


def test_read_and_save_both_raise_weight(profile):
    """`save` 도 긍정 신호입니다 (§8.1 버튼 3개 중 둘).

    깨뜨리는 법: POSITIVE_ACTIONS 에서 "save" 를 빼면 KNOWN_ACTIONS 에서도 빠져
    ValueError 로 빨간불.
    확인일: 2026-08-18
    """
    items, rows = corpus([(30, "read", "alpha"), (30, "save", "alpha")])

    update = compute_updates(profile, rows, items, BagOfWordsEmbedder())

    alpha = by_id(update)["alpha"]
    assert (alpha.read, alpha.save, alpha.skip) == (30, 30, 0)
    assert alpha.delta == pytest.approx(MAX_DELTA)


def test_single_skip_does_not_lower_weight(profile):
    """★ 스킵 1건으로는 내리지 않습니다 — §8.3 은 "skip이 **반복되는** 관심사"입니다.
    그날 바빴던 것과 관심이 식은 것을 구분하지 못하면 프로파일이 잡음을 학습합니다.

    깨뜨리는 법: _delta() 의 MIN_SKIP_REPEAT 분기를 지우면 beta 가 -0.1 이 되어 빨간불.
    확인일: 2026-08-18
    """
    items, rows = corpus([(55, "read", "alpha"), (1, "skip", "beta")])

    update = compute_updates(profile, rows, items, BagOfWordsEmbedder())

    beta = by_id(update)["beta"]
    assert beta.skip == 1
    assert beta.delta == 0.0
    assert beta.after == beta.before


def test_repeated_skips_lower_weight(profile):
    """반복된 스킵은 반영합니다 (§8.3).

    깨뜨리는 법: _delta() 가 negative 를 무시하게 하면 delta 0 이 되어 빨간불.
    확인일: 2026-08-18
    """
    items, rows = corpus([(50, "read", "alpha"), (5, "skip", "beta")])

    update = compute_updates(profile, rows, items, BagOfWordsEmbedder())

    beta = by_id(update)["beta"]
    assert beta.skip == 5
    assert beta.delta == pytest.approx(-MAX_DELTA)
    assert beta.after == pytest.approx(0.8)


def test_untouched_interest_stays_and_is_reported(profile):
    """피드백이 없는 관심사는 그대로 두되 **표에는 남깁니다** — 안 움직인 것도 결과입니다.

    깨뜨리는 법: compute_updates 가 delta 0 인 관심사를 changes 에서 빼면 빨간불.
    확인일: 2026-08-18
    """
    items, rows = corpus([(60, "read", "alpha")])

    update = compute_updates(profile, rows, items, BagOfWordsEmbedder())

    gamma = by_id(update)["gamma"]
    assert gamma.delta == 0.0 and gamma.after == gamma.before
    assert {change.id for change in update.changes} == {"alpha", "beta", "gamma"}


def test_latest_label_per_item_wins(profile):
    """같은 아이템의 두 줄은 마지막이 유효합니다 (읽고 나서 저장하면 두 줄이 남습니다).

    깨뜨리는 법: latest_by_item 을 쓰지 않고 rows 를 그대로 세면 labels_used 가 70 이 되어 빨간불.
    확인일: 2026-08-18
    """
    items, rows = corpus([(60, "read", "alpha")])
    later = [dict(row, action="save", ts="2026-08-18T09:00:00+09:00") for row in rows[:10]]

    update = compute_updates(profile, rows + later, items, BagOfWordsEmbedder())

    alpha = by_id(update)["alpha"]
    assert update.labels_used == 60
    assert (alpha.read, alpha.save) == (50, 10)


def test_unknown_action_is_rejected(profile):
    """수집기(§8.1)가 쓰는 값은 read/skip/save 3개뿐입니다. 그 외는 수집기 버그이므로
    조용히 무시하지 않습니다.

    깨뜨리는 법: compute_updates 의 KNOWN_ACTIONS 검사를 `continue` 로 바꾸면
    ValueError 가 안 나 빨간불.
    확인일: 2026-08-18
    """
    items, rows = corpus([(60, "read", "alpha")])
    rows[0]["action"] = "bookmark"

    with pytest.raises(ValueError, match="action"):
        compute_updates(profile, rows, items, BagOfWordsEmbedder())


def test_unmatched_items_do_not_count_toward_the_50(profile):
    """텍스트를 못 찾은 아이템은 어느 관심사에 넣을지 알 수 없으므로 세지 않습니다.
    이걸 세면 하한 50건이 실제로는 30건이 됩니다.

    깨뜨리는 법: compute_updates 에서 `unknown += 1; continue` 대신 usable 에 넣으면
    (텍스트 없이) KeyError 또는 잘못된 통과로 빨간불.
    확인일: 2026-08-18
    """
    items, rows = corpus([(60, "read", "alpha")])
    known = items[:40]

    with pytest.raises(ValueError, match="매칭 실패 20건"):
        compute_updates(profile, rows, known, BagOfWordsEmbedder())


# ── 임베더 주입 ★ ────────────────────────────────────────────────────────


def test_embedder_is_injected_and_label_is_never_embedded(profile):
    """유사도는 주입된 Embedder 로만 계산합니다 (자체 구현 금지, 기획안_2 §4.2).
    관심사는 `text` 만 임베딩합니다 — `label` 은 사람용입니다 (프로파일 머리말).

    깨뜨리는 법: compute_updates 가 interest["text"] 대신 interest["label"] 을
    임베딩하면 빨간불.
    확인일: 2026-08-18
    """
    items, rows = corpus([(60, "read", "alpha")])
    embedder = BagOfWordsEmbedder()

    compute_updates(profile, rows, items, embedder)

    embedded = [text for batch in embedder.calls for text in batch]
    assert TOPIC_TEXT["alpha"] in embedded, "관심사 text 를 임베딩하지 않았습니다"
    assert not any("사람이 읽는 이름" in text for text in embedded), "label 을 임베딩했습니다"
    # 배치 호출 — 아이템 60건을 한 건씩 부르면 API 임베더에서 비용이 60배입니다.
    assert len(embedder.calls) == 2


def test_embedder_without_encode_dies_loudly(profile):
    """포트를 만족하지 않는 객체는 즉시 TypeError. 조용히 대체 구현으로 떨어지지 않습니다.
    포트는 `src/rank/embed.py`의 `Embedder`(`encode(texts)`)입니다.

    깨뜨리는 법: _encode() 의 None 검사를 지우면 TypeError 대신 다른 예외로 빨간불.
    확인일: 2026-08-18
    """
    items, rows = corpus([(60, "read", "alpha")])

    with pytest.raises(TypeError, match="encode"):
        compute_updates(profile, rows, items, object())


def test_zero_vector_embedding_is_rejected(profile):
    """★ 영벡터는 "가장 가까운 관심사"를 정할 수 없습니다. `src.rank.embed.cosine` 은
    영벡터에 0.0 을 주는데(랭킹에서는 합당합니다), 여기서 그대로 쓰면 그 아이템의 표가
    **사전순 첫 관심사**로 조용히 흘러갑니다.

    깨뜨리는 법: _similarity 의 l2_norm 검사를 `if False:` 로 바꾸면 예외 대신
    alpha 가 표를 받아 빨간불.
    확인일: 2026-08-18
    """
    items, rows = corpus([(60, "read", "alpha")])
    for item in items[:1]:  # 어휘에 없는 텍스트 → 전 성분 0
        item["title"] = "zzz"
        item["abstract"] = "zzz"

    with pytest.raises(ValueError, match="영벡터"):
        compute_updates(profile, rows, items, BagOfWordsEmbedder())


def test_embedder_returning_wrong_count_is_rejected(profile):
    """개수가 어긋나면 아이템과 벡터가 밀려서 **엉뚱한 관심사**에 표가 갑니다.

    깨뜨리는 법: _encode 의 개수 검사를 지우면 zip 이 조용히 짧은 쪽에서 끊겨 빨간불
    (예외 없이 통과).
    확인일: 2026-08-18
    """

    class TruncatingEmbedder(BagOfWordsEmbedder):
        def encode(self, texts):
            return super().encode(texts)[:-1]

    items, rows = corpus([(60, "read", "alpha")])

    with pytest.raises(ValueError, match="임베딩 개수"):
        compute_updates(profile, rows, items, TruncatingEmbedder())


def test_same_input_gives_same_numbers(profile):
    """재현성 — 같은 피드백으로 두 번 돌리면 같은 프로파일이어야 합니다 (§5.8 DoD).

    깨뜨리는 법: _nearest_interest 의 동점 처리(id 사전순)를 없애고 set 순회로
    바꾸면 동점 케이스에서 빨간불.
    확인일: 2026-08-18
    """
    items, rows = corpus([(40, "read", "alpha"), (20, "skip", "beta")])

    first = compute_updates(profile, rows, items, BagOfWordsEmbedder())
    second = compute_updates(profile, rows, items, BagOfWordsEmbedder())

    assert first.to_dict() == second.to_dict()


# ── 파일 번호 규칙 ★ ─────────────────────────────────────────────────────


def test_next_profile_path_is_max_plus_one(root):
    """★ `_1` 이 있으면 `_2` 를 만듭니다 (CLAUDE.md §2).

    깨뜨리는 법: next_profile_path 에서 접미사 파싱(`int(matched.group(1))`)을 0 으로
    고정하면 이미 있는 `_1` 을 다시 가리켜 빨간불.
    확인일: 2026-08-18
    """
    assert next_profile_path(STEM, root=root) == root / f"{STEM}_2.yaml"

    write_profile(root, f"{STEM}_10.yaml", {"alpha": 1.0, "beta": 0.9, "gamma": 0.6})
    assert next_profile_path(STEM, root=root) == root / f"{STEM}_11.yaml"


def test_writes_new_numbered_file_and_leaves_source_untouched(root, tmp_path):
    """★ 기존 파일은 이동·삭제·수정하지 않습니다 (CLAUDE.md §2). 갱신 전 프로파일이
    남아야 §8.3 의 "갱신 전후를 골드셋으로 재측정"이 가능합니다.

    깨뜨리는 법: update_profile() 이 out_path 대신 source_path 에 쓰게 하면 빨간불.
    확인일: 2026-08-18
    """
    source = root / f"{STEM}_1.yaml"
    before_bytes = source.read_bytes()
    items, rows = corpus([(60, "read", "alpha")])
    feedback = write_feedback(tmp_path / "feedback.jsonl", rows)

    update = update_profile(
        STEM, embedder=BagOfWordsEmbedder(), items=items, feedback_path=feedback, root=root
    )

    assert update.out_path == root / f"{STEM}_2.yaml"
    assert update.source_path == source
    assert update.out_path.exists()
    assert source.read_bytes() == before_bytes, "원본이 바뀌었습니다"


def test_refuses_to_overwrite_an_existing_file(root, tmp_path, monkeypatch):
    """★ 번호를 잘못 골랐더라도 덮어쓰지 않습니다 — 덮어쓰면 이력이 사라집니다
    (CLAUDE.md §2). 번호 해석과 쓰기 사이의 경합에 대한 마지막 방어선이라
    `next_profile_path` 가 기존 파일을 가리키는 상황을 만들어 검사합니다.

    깨뜨리는 법: update_profile() 의 out_path.exists() 검사를 지우면 원본이
    덮어써져 빨간불.
    확인일: 2026-08-18
    """
    import src.feedback.update_profile as M

    items, rows = corpus([(60, "read", "alpha")])
    feedback = write_feedback(tmp_path / "feedback.jsonl", rows)
    source = root / f"{STEM}_1.yaml"
    before_bytes = source.read_bytes()
    monkeypatch.setattr(M, "next_profile_path", lambda stem, root=None, ext=".yaml": source)

    with pytest.raises(FileExistsError):
        update_profile(
            STEM, embedder=BagOfWordsEmbedder(), items=items, feedback_path=feedback, root=root
        )
    assert source.read_bytes() == before_bytes, "기존 파일이 덮어써졌습니다"


def test_output_is_a_loadable_profile_with_updated_weights(root, tmp_path):
    """산출물은 `load_profile()` 로 그대로 읽히는 프로파일이어야 합니다 (R1).
    weight 외의 키(interests 순서·text·license_gate·selection)는 보존합니다.

    깨뜨리는 법: apply_changes 가 interests 를 새로 만들면서 `text` 를 빼면 빨간불.
    확인일: 2026-08-18
    """
    items, rows = corpus([(50, "read", "alpha"), (10, "skip", "beta")])
    feedback = write_feedback(tmp_path / "feedback.jsonl", rows)

    update = update_profile(
        STEM, embedder=BagOfWordsEmbedder(), items=items, feedback_path=feedback, root=root
    )

    data, path = config.load_profile(STEM, root=root)
    assert path == update.out_path, "새 파일이 최고 번호로 잡히지 않았습니다"

    weights = {interest["id"]: interest["weight"] for interest in data["interests"]}
    assert weights["alpha"] == pytest.approx(1.1)
    assert weights["beta"] == pytest.approx(0.8)
    assert weights["gamma"] == pytest.approx(0.6)

    assert [interest["id"] for interest in data["interests"]] == ["alpha", "beta", "gamma"]
    assert data["interests"][0]["text"] == TOPIC_TEXT["alpha"]
    assert data["license_gate"]["default_fulltext_ok"] is False
    assert data["selection"]["final_n"] == 5


def test_output_header_records_provenance(root, tmp_path):
    """새 파일은 무엇을 근거로 이 숫자가 됐는지 스스로 들고 있어야 합니다
    (YAML 덤프는 원본 주석을 보존하지 못합니다).

    깨뜨리는 법: update_profile() 에서 _header(...) 를 빼면 빨간불.
    확인일: 2026-08-18
    """
    items, rows = corpus([(60, "read", "alpha")])
    feedback = write_feedback(tmp_path / "feedback.jsonl", rows)

    update = update_profile(
        STEM, embedder=BagOfWordsEmbedder(), items=items, feedback_path=feedback, root=root
    )

    text = update.out_path.read_text(encoding="utf-8")
    assert "§8.3" in text
    assert f"{STEM}_1.yaml" in text  # 원본 경로
    assert "반영한 라벨: 60건" in text
    assert "alpha" in text and "1.000 → 1.100" in text


# ── 갱신 내역 반환 ★ ─────────────────────────────────────────────────────


def test_result_reports_before_and_after(profile):
    """★ results.md 에 전/후를 적을 수 있어야 합니다 (§8.3).

    깨뜨리는 법: InterestChange 에서 before 를 빼면 AttributeError 로 빨간불.
    확인일: 2026-08-18
    """
    items, rows = corpus([(50, "read", "alpha"), (10, "skip", "beta")])

    update = compute_updates(profile, rows, items, BagOfWordsEmbedder())

    alpha = by_id(update)["alpha"]
    assert (alpha.before, alpha.after) == (1.0, 1.1)
    assert alpha.before + alpha.delta == pytest.approx(alpha.after)

    table = update.markdown_table()
    assert table.splitlines()[0].startswith("| 관심사 |")
    for interest_id in ("alpha", "beta", "gamma"):
        assert f"| {interest_id} |" in table

    payload = update.to_dict()
    assert payload["labels_used"] == 60
    assert {change["id"] for change in payload["changes"]} == {"alpha", "beta", "gamma"}


# ── feedback.jsonl 로딩 ──────────────────────────────────────────────────


def test_load_feedback_resolves_path_at_call_time(tmp_path, monkeypatch):
    """경로 기본값을 정의 시점에 바인딩하면 테스트가 실제 `data/feedback.jsonl` 을
    읽습니다 (기획안_2 §9.1 — 라벨링 테스트 17개가 실제로 그 상태였습니다).

    깨뜨리는 법: load_feedback 의 기본 인자를 `path=DEFAULT_FEEDBACK_PATH` 로
    되돌리면 monkeypatch 가 안 먹어 빨간불.
    확인일: 2026-08-18
    """
    import src.feedback.update_profile as M

    _, rows = corpus([(3, "read", "alpha")])
    write_feedback(tmp_path / "feedback.jsonl", rows)
    monkeypatch.setattr(M, "DEFAULT_FEEDBACK_PATH", tmp_path / "feedback.jsonl")

    assert len(load_feedback()) == 3


def test_load_feedback_returns_empty_when_missing(tmp_path):
    """파일이 없으면 빈 리스트입니다 (첫 실행). 50건 하한이 그다음에 걸립니다.

    깨뜨리는 법: 존재 검사를 지우면 FileNotFoundError 로 빨간불.
    확인일: 2026-08-18
    """
    assert load_feedback(tmp_path / "nope.jsonl") == []
