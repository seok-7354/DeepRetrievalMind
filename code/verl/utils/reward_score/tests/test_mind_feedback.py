"""
reward_score/mind_feedback.py 단위 테스트

실행:
    cd code/verl/utils/reward_score
    python -m pytest tests/test_mind_feedback.py -v
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mind_feedback import compute_score, parse_solution  # noqa: E402


def make_ground_truth(clicked_ids, extra_candidates=0, category_ctr=None):
    candidates = {
        "N1": "Trade War Tariffs China US politics economy",
        "N2": "Local Weather Forecast Sunny Skies This Week",
        "N3": "Celebrity Wedding Photos Entertainment News",
    }
    for i in range(extra_candidates):
        candidates[f"NX{i}"] = f"Filler article number {i} random topic content"

    category_of = {"N1": "news", "N2": "weather", "N3": "entertainment"}
    for i in range(extra_candidates):
        category_of[f"NX{i}"] = "misc"

    return json.dumps({
        "candidates": candidates,
        "clicked_ids": clicked_ids,
        "category_of": category_of,
        "category_ctr": category_ctr or {"news": 0.04, "weather": 0.03, "entertainment": 0.05, "misc": 0.02},
    })


# ---------- parse_solution ----------

def test_parse_solution_valid():
    text = "<think>user likes politics</think><answer>trade war tariffs</answer>"
    query, ok = parse_solution(text)
    assert ok is True
    assert query == "trade war tariffs"


def test_parse_solution_missing_tags():
    query, ok = parse_solution("just some text without tags")
    assert ok is False
    assert query is None


def test_parse_solution_empty_answer():
    text = "<think>hmm</think><answer>   </answer>"
    query, ok = parse_solution(text)
    assert ok is False


# ---------- compute_score: format ----------

def test_bad_format_gets_penalized():
    gt = make_ground_truth(clicked_ids=["N1"])
    score = compute_score("mind_feedback", "no tags at all", gt)
    assert score < 0  # format penalty + floor


def test_good_format_and_perfect_query_scores_high():
    gt = make_ground_truth(clicked_ids=["N1"])
    solution = "<think>interested in politics</think><answer>trade war tariffs china us</answer>"
    score = compute_score("mind_feedback", solution, gt)
    assert score > 3.0  # format bonus + high MRR bucket


def test_irrelevant_query_scores_low_but_not_missing_click():
    gt = make_ground_truth(clicked_ids=["N1"], extra_candidates=20)
    # 전혀 관련 없는 쿼리를 던져서 클릭 항목이 하위로 밀려나는 상황
    solution = "<think>random</think><answer>filler random topic content</answer>"
    score = compute_score("mind_feedback", solution, gt)
    assert score < 3.0


def test_clicked_id_not_in_pool_triggers_missing_click_penalty():
    gt = make_ground_truth(clicked_ids=["N_NOT_IN_POOL"])
    solution = "<think>x</think><answer>trade war</answer>"
    score = compute_score("mind_feedback", solution, gt)
    assert score == 0.5 + (-2.0)  # FORMAT_BONUS + MISSING_CLICK_PENALTY


# ---------- popularity adjustment ----------

def test_popularity_adjustment_penalizes_high_ctr_category():
    # 동일한 rank(1위)를 맞추더라도 카테고리 baseline CTR이 높으면 reward가 더 낮아야 함
    gt_high_ctr = make_ground_truth(clicked_ids=["N3"], category_ctr={
        "news": 0.04, "weather": 0.03, "entertainment": 0.20, "misc": 0.02,
    })
    gt_low_ctr = make_ground_truth(clicked_ids=["N3"], category_ctr={
        "news": 0.04, "weather": 0.03, "entertainment": 0.01, "misc": 0.02,
    })
    solution = "<think>x</think><answer>celebrity wedding entertainment</answer>"
    score_high = compute_score("mind_feedback", solution, gt_high_ctr)
    score_low = compute_score("mind_feedback", solution, gt_low_ctr)
    assert score_high < score_low


# ---------- multi-click ----------

def test_multi_click_uses_best_rank():
    # N1, N2 둘 다 클릭됐다고 가정. 쿼리가 N1과 훨씬 더 잘 매칭되면 N1 기준으로 reward 계산.
    gt = make_ground_truth(clicked_ids=["N1", "N2"])
    solution = "<think>x</think><answer>trade war tariffs china us politics</answer>"
    score = compute_score("mind_feedback", solution, gt)
    assert score > 3.0


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
