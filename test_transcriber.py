"""Algorithm tests use fixtures and do not download or load models."""
import json
import math
import tempfile
import unittest
from pathlib import Path

from transcriber import demo, read_rows, transcribe


class TranscriberTests(unittest.TestCase):
    def run_scores(self, scores, **kwargs):
        values = iter(scores)
        return transcribe("original", lambda _: "candidate", lambda _a, _b: next(values), **kwargs)

    def test_lower_bound_is_exclusive(self):
        result = self.run_scores([0.85] * 3)
        self.assertTrue(result["fallback"])
        self.assertEqual(result["output"], "original")
        self.assertEqual(len(result["trace"]), 3)

    def test_upper_bound_is_inclusive(self):
        result = self.run_scores([0.95])
        self.assertTrue(result["accepted"])
        self.assertEqual(len(result["trace"]), 1)

    def test_first_valid_candidate_is_returned(self):
        candidates = iter(["too close", "valid", "unused"])
        scores = iter([0.99, 0.90])
        result = transcribe("original", lambda _: next(candidates), lambda _a, _b: next(scores))
        self.assertEqual(result["output"], "valid")
        self.assertEqual(len(result["trace"]), 2)

    def test_each_attempt_rewrites_original(self):
        calls = []
        def rewrite(text):
            calls.append(text)
            return "failed"
        result = transcribe("original", rewrite, lambda _a, _b: 0.1)
        self.assertEqual(calls, ["original"] * 3)
        self.assertTrue(result["fallback"])

    def test_nonfinite_scores_are_rejected_and_json_safe(self):
        result = self.run_scores([math.nan, math.inf, -math.inf])
        self.assertTrue(result["fallback"])
        self.assertTrue(all(t["similarity"] is None for t in result["trace"]))
        json.dumps(result, allow_nan=False)

    def test_empty_candidates_skip_similarity(self):
        def score(_a, _b):
            self.fail("empty candidate must not be encoded")
        result = transcribe("original", lambda _: "  ", score)
        self.assertTrue(result["fallback"])
        self.assertEqual(result["trace"][0]["rejection_reason"], "empty_candidate")

    def test_unconstrained_accepts_first_candidate_outside_gate(self):
        result = self.run_scores([0.99], mode="unconstrained")
        self.assertTrue(result["accepted"])
        self.assertEqual(len(result["trace"]), 1)

    def test_unconstrained_empty_result_falls_back_after_one_attempt(self):
        result = transcribe("original", lambda _: "", lambda _a, _b: 1, mode="unconstrained")
        self.assertTrue(result["fallback"])
        self.assertEqual(len(result["trace"]), 1)

    def test_invalid_arguments(self):
        for instruction in ["", "  ", None]:
            with self.assertRaises(ValueError):
                transcribe(instruction, str, lambda _a, _b: 0.9)
        for kwargs in [
            {"lower": 0.95, "upper": 0.85}, {"lower": math.nan},
            {"max_attempts": 0}, {"mode": "other"},
        ]:
            with self.assertRaises(ValueError):
                self.run_scores([0.9], **kwargs)

    def test_annotations_are_not_passed_to_defense(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.jsonl"
            path.write_text(json.dumps({
                "id": "x", "instruction": "original",
                "trigger": "secret", "clean_reference": "hidden",
            }), encoding="utf-8")
            self.assertEqual(list(read_rows(path)), [{"id": "x", "instruction": "original"}])

    def test_demo_is_explicitly_synthetic(self):
        result = demo()
        self.assertIn("Synthetic", result["demo_notice"])
        self.assertTrue(result["accepted"])
        self.assertEqual(len(result["trace"]), 2)


if __name__ == "__main__":
    unittest.main()
