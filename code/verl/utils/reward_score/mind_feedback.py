"""
verl/utils/reward_score/mind_feedback.py

MIND 실제 클릭 로그 기반 무감독 reward 함수 (A안: Re-ranking).

verl reward 인터페이스를 따른다:
    compute_score(data_source, solution_str, ground_truth, extra_info=None) -> float

설계 (지금까지 논의 반영):
    1. solution_str에서 <think>...</think><answer>...</answer> 파싱, 포맷 보상/페널티
    2. <answer> 안의 쿼리로, "이 impression의 후보군"(ground_truth["candidates"])에 대해
       즉석(in-memory) BM25 재랭킹 수행 (rank_bm25) — impression마다 후보가 다르므로
       고정 코퍼스 인덱스가 아니라 매번 가볍게 생성한다.
    3. reward는 "전역 인기도"가 아니라 "이 impression 내에서 실제 클릭된 항목의 상대 순위"로
       계산한다 (reward hacking 방지 — 인기 카테고리로 쏠리는 shortcut을 차단하기 위함).
    4. 클릭된 항목이 속한 카테고리의 baseline CTR이 높을수록(=원래도 잘 클릭되는 카테고리)
       reward를 소폭 할인한다 (popularity-adjusted reward).
"""
import json
import re
from functools import lru_cache

from rank_bm25 import BM25Okapi

# ---- 하이퍼파라미터 ----
FORMAT_BONUS = 0.5
FORMAT_PENALTY = -1.0
MISSING_CLICK_PENALTY = -2.0
POPULARITY_LAMBDA = 2.0  # 카테고리 baseline CTR 보정 강도

MRR_BUCKETS = [
    (0.8, 5.0),
    (0.5, 3.0),
    (0.3, 1.0),
    (0.1, 0.3),
]
MRR_FLOOR_REWARD = -2.0

_THINK_ANSWER_RE = re.compile(r"<think>(.*?)</think>\s*<answer>(.*?)</answer>", re.DOTALL)
_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list:
    return _TOKEN_RE.findall(text.lower())


def parse_solution(solution_str: str):
    """<think>/<answer> 파싱. 실패 시 (None, format_ok=False)."""
    m = _THINK_ANSWER_RE.search(solution_str or "")
    if not m:
        return None, False
    query = m.group(2).strip()
    if not query:
        return None, False
    return query, True


@lru_cache(maxsize=4096)
def _build_bm25_cached(candidates_key: tuple):
    """impression(candidate 집합) 단위로 BM25 인덱스를 캐싱.

    candidates_key: ((news_id, text), (news_id, text), ...) 형태의 튜플 -- dict는
    hashable하지 않으므로 캐시 키로 쓰기 위해 튜플화한다.
    """
    news_ids = [nid for nid, _ in candidates_key]
    tokenized = [_tokenize(text) for _, text in candidates_key]
    return BM25Okapi(tokenized), news_ids


def _rank_candidates(candidates: dict, query: str) -> list:
    key = tuple(sorted(candidates.items()))
    bm25, news_ids = _build_bm25_cached(key)
    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(zip(news_ids, scores), key=lambda x: -x[1])
    return [nid for nid, _ in ranked]


def _mrr_to_reward(mrr: float) -> float:
    for threshold, reward in MRR_BUCKETS:
        if mrr >= threshold:
            return reward
    return MRR_FLOOR_REWARD


def _popularity_discount(clicked_category_ctrs: list) -> float:
    """클릭된 항목이 속한 카테고리들의 평균 baseline CTR이 높을수록 reward를 할인.

    category_ctr는 대략 0~0.1(0~10%) 스케일이므로, LAMBDA로 스케일을 맞춰 소폭 보정만 가함.
    """
    if not clicked_category_ctrs:
        return 0.0
    avg_ctr = sum(clicked_category_ctrs) / len(clicked_category_ctrs)
    return -POPULARITY_LAMBDA * avg_ctr


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None) -> float:
    gt = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    candidates: dict = gt["candidates"]
    clicked_ids: list = gt["clicked_ids"]
    category_of: dict = gt.get("category_of", {})
    category_ctr: dict = gt.get("category_ctr", {})

    query, format_ok = parse_solution(solution_str)
    format_reward = FORMAT_BONUS if format_ok else FORMAT_PENALTY

    if not format_ok or not candidates:
        return format_reward + MRR_FLOOR_REWARD

    ranked_ids = _rank_candidates(candidates, query)
    clicked_in_pool = [c for c in clicked_ids if c in ranked_ids]

    if not clicked_in_pool:
        # ground_truth 파싱/전처리 이상 등으로 클릭된 항목이 후보 풀에 없는 예외 케이스
        return format_reward + MISSING_CLICK_PENALTY

    reciprocal_ranks = [1.0 / (ranked_ids.index(c) + 1) for c in clicked_in_pool]
    mrr = max(reciprocal_ranks)  # 클릭된 항목 중 가장 상위로 올라온 것을 기준
    base_reward = _mrr_to_reward(mrr)

    best_clicked = clicked_in_pool[reciprocal_ranks.index(mrr)]
    best_category = category_of.get(best_clicked)
    ctr_of_best = category_ctr.get(best_category, 0.0) if best_category else 0.0
    popularity_adj = _popularity_discount([ctr_of_best])

    total = format_reward + base_reward + popularity_adj
    return round(total, 4)


# verl RewardManager는 보통 data_source로 라우팅하므로, 필요시 다음과 같이 등록:
#   from verl.utils.reward_score import mind_feedback
#   REWARD_FN_MAP["mind_feedback"] = mind_feedback.compute_score
