"""The variance measures and output parsers behind every number in the paper."""

import json
import math

import pytest

from analyze_sembench_repeats import pair_disagreement, parse, subset_flip, wilson


def test_flip_rate_is_zero_when_every_execution_agrees():
    assert subset_flip([True] * 5) == 0.0


def test_flip_rate_counts_one_differing_execution():
    assert subset_flip([True, True, True, True, False]) == 1.0


def test_flip_rate_is_the_exact_expectation_over_subsets_of_five():
    # Nine executions say True and one says False: a subset of five is unanimous
    # only if it misses the False, with probability C(9,5) / C(10,5) = 1/2.
    assert subset_flip([True] * 9 + [False]) == pytest.approx(0.5)


def test_flip_rate_is_undefined_with_fewer_executions_than_the_subset_size():
    assert math.isnan(subset_flip([True] * 4))


def test_pairwise_disagreement_draws_two_distinct_executions():
    # Pairs (a,a), (b,b) agree: 2 * (2 * 1) of 4 * 3 ordered pairs.
    assert pair_disagreement(["a", "a", "b", "b"]) == pytest.approx(2 / 3)
    assert pair_disagreement(["a"] * 6) == 0.0


def test_wilson_interval_is_clipped_and_symmetric_at_one_half():
    low, high = wilson(0, 10)
    assert low == 0.0 and 0.0 < high < 0.35
    low, high = wilson(5, 10)
    assert low == pytest.approx(0.2366, abs=1e-4)
    assert high == pytest.approx(0.7634, abs=1e-4)


def test_boolean_outputs_follow_lotus_filter_rule():
    assert parse("bool", "The review praises the film.\nAnswer: True") is True
    assert parse("bool", "Reasoning first. Answer: False") is False
    # LOTUS keeps a row whose reply names neither value.
    assert parse("bool", "Answer: unclear") is True


def test_scores_outside_one_to_five_default_to_three_as_in_sembench():
    assert parse("score", "4") == 4.0
    assert parse("score", "7") == 3.0
    assert parse("score", "four") == 3.0


def test_extractions_follow_lotus_json_rule():
    assert parse("extract", json.dumps({"text_diagnosis": "asthma"})) == "asthma"
    assert parse("extract", "not json") is None


def test_missing_output_stays_missing():
    assert parse("bool", None) is None
