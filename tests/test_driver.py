"""Bookkeeping of the SemBench driver: cell names, request keys, and the attempt limit."""

import json

from sembench_repeats import MAX_ATTEMPTS, _message_key, exhausted, model_tag, policy_rejected, setting_tag


def test_setting_names_keep_probes_apart_from_the_main_matrix():
    assert setting_tag(20, None, False, False) == "w20"
    assert setting_tag(20, None, False, True) == "w20_lp"
    assert setting_tag(20, "high", False, False) == "w20_r-high"
    assert setting_tag(20, None, True, True) == "w20_shuffle_lp"


def test_model_directory_drops_the_provider_prefix():
    assert model_tag("openai/Azure/gpt-4o") == "Azure-gpt-4o"


def test_request_key_ignores_dictionary_order():
    first = [{"role": "user", "content": "x"}]
    second = [{"content": "x", "role": "user"}]
    assert _message_key(first) == _message_key(second)
    assert _message_key(first) != _message_key([{"role": "user", "content": "y"}])


def _fail(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def test_a_cell_is_given_up_after_the_attempt_limit(tmp_path):
    failures = tmp_path / "repeat-1" / "Q3.failures.jsonl"
    assert not exhausted(failures)
    _fail(failures, [{"error": "x"}] * (MAX_ATTEMPTS - 1))
    assert not exhausted(failures)
    _fail(failures, [{"error": "x"}] * MAX_ATTEMPTS)
    assert exhausted(failures)


def test_a_query_is_skipped_only_if_a_content_policy_rejected_every_attempt(tmp_path):
    rejection = {"call_error_samples": ["ContentPolicyViolationError: rejected"]}
    _fail(tmp_path / "repeat-1" / "Q1.failures.jsonl", [rejection, {"error": "RateLimitError"}, rejection])
    assert not policy_rejected(tmp_path, 1)
    _fail(tmp_path / "repeat-2" / "Q1.failures.jsonl", [rejection] * MAX_ATTEMPTS)
    assert policy_rejected(tmp_path, 1)
