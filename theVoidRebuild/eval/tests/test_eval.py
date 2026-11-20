"""Behavioral checks for scoring, data leakage boundaries and token constraints."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, SCHEMA, TAGS
from data.generate import CachedAPI, judge_row, make_specs, near_duplicate
from tagging.metrics import accuracy, percentile, summarize, validate_output
from tagging.run import load_rows


class ScoringTests(unittest.TestCase):
    def test_invalid_and_secondary_predictions_are_scored_differently(self):
        rows = [
            {"label": TAGS[0], "schema_valid": True, "prediction": {"tags": [TAGS[1], TAGS[0]]}},
            {"label": TAGS[0], "schema_valid": False, "prediction": {"tags": [TAGS[0]]}},
            {"label": TAGS[1], "schema_valid": True, "prediction": {"tags": [TAGS[1]]}},
        ]
        result = accuracy(rows)
        self.assertEqual(result["top1"], 1 / 3)
        self.assertEqual(result["any_tag"], 2 / 3)
        self.assertEqual(result["confusion"][TAGS[0]]["invalid"], 1)
        self.assertEqual(result["by_quadrant"][TAGS[0]]["top1"], 0)
        self.assertEqual(result["by_quadrant"][TAGS[2]]["n"], 0)

    def test_strict_schema_and_raw_json(self):
        good = {"title": "A" * 60, "tags": TAGS[:2]}
        self.assertFalse(validate_output(json.dumps(good))[1])
        bad = [dict(good, title="A" * 61), dict(good, tags=[]), dict(good, tags=TAGS[:3]),
               dict(good, tags=["happy"]), dict(good, extra=1), dict(good, title="")]
        for value in bad:
            self.assertTrue(validate_output(json.dumps(value))[1], value)
        self.assertTrue(validate_output('```json\n{}\n```')[1])
        self.assertTrue(validate_output('{"title":"a","title":"b","tags":["low_pleasant"]}')[1])

    def test_percentile_is_nearest_rank(self):
        self.assertEqual(percentile(range(1, 11), .95), 10)
        self.assertEqual(percentile(range(1, 101), .95), 95)

    def test_optional_split_is_supported(self):
        rows = [{"label": TAGS[0], "schema_valid": False, "prediction": None, "split": None},
                {"label": TAGS[1], "schema_valid": False, "prediction": None, "split": "clean"}]
        result = summarize(rows)
        self.assertEqual(result["by_split"]["unassigned"]["n"], 1)
        self.assertEqual(result["all"]["top1"], 0)

    def test_bad_input_is_rejected_before_inference(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".cache") as directory:
            path = Path(directory) / "input.jsonl"
            row = {"id": "same", "transcript": "hello", "label": TAGS[0]}
            path.write_text(json.dumps(row) + "\n" + json.dumps(row))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_rows(path, None, "all")


class DataTests(unittest.TestCase):
    def test_judge_has_no_access_to_generation_metadata(self):
        class SpyAPI:
            def call(self, model, messages, schema, seed, temperature):
                self.messages = messages
                return {"title": "A moment", "tags": [TAGS[1]]}, "cached.json", model
        api = SpyAPI()
        row = {"id": "secret-id", "label": TAGS[0], "transcript": "I'm quietly enjoying my tea", "topic": "secret-topic", "mixed": True}
        judged = judge_row(api, row, "judge-model", 7)
        self.assertEqual(json.loads(api.messages[1]["content"]), {"transcript": row["transcript"]})
        self.assertNotIn("secret-id", json.dumps(api.messages))
        self.assertNotIn("secret-topic", json.dumps(api.messages))
        self.assertEqual(judged["label"], TAGS[0])
        self.assertEqual(judged["judge_label"], TAGS[1])
        self.assertEqual(judged["split"], "hard")

    def test_seeded_balance_and_mixed_quota(self):
        specs = make_specs(300, 7)
        self.assertEqual(specs, make_specs(300, 7))
        for tag in TAGS:
            self.assertEqual(sum(s["label"] == tag for s in specs), 75)
            self.assertEqual(sum(s["label"] == tag and s["mixed"] for s in specs), 15)

    def test_duplicates_include_case_punctuation_and_long_overlap(self):
        self.assertTrue(near_duplicate("Um, I feel fine!", "um i feel fine"))
        self.assertTrue(near_duplicate("yes", "YES!"))
        self.assertFalse(near_duplicate("I like green tea", "I hate cold soup"))

    def test_offline_cache_hit_never_constructs_api_client(self):
        from common import digest
        messages = [{"role": "user", "content": "fictional memo"}]
        request = dict(model="test", messages=messages, temperature=0, seed=7, max_tokens=7500,
                       response_format={"type": "json_schema", "json_schema": {"name": "voice_memos", "strict": True, "schema": SCHEMA}})
        response = {"model": "snapshot", "choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps({"title": "Tea", "tags": [TAGS[1]]})}}]}
        with tempfile.TemporaryDirectory(dir=ROOT / ".cache") as directory:
            (Path(directory) / (digest(request) + ".json")).write_text(json.dumps({"response": response}))
            api = CachedAPI(directory, offline=True)
            with patch("openai.OpenAI", side_effect=AssertionError("No API client should be built")):
                result, _, _ = api.call("test", messages, SCHEMA, 7, 0)
            self.assertEqual(result["title"], "Tea")
            self.assertEqual(api.hits, 1)
            with self.assertRaisesRegex(RuntimeError, "cache miss"):
                api.call("other-model", messages, SCHEMA, 7, 0)


class ConstraintTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from lmformatenforcer import TokenEnforcerTokenizerData
        # Character tokens plus combined tokens test transitions across enum boundaries.
        cls.token_text = [chr(c) for c in range(32, 127)] + ['high_pleasant"]}', 'bad_quadrant', 'A' * 61, 'A' * 60]
        cls.eos = len(cls.token_text)
        cls.data = TokenEnforcerTokenizerData([(i, text, False) for i, text in enumerate(cls.token_text)],
                    lambda ids: "".join(cls.token_text[i] for i in ids if i != cls.eos), cls.eos, False, cls.eos + 1)

    def allows(self, text):
        from lmformatenforcer import JsonSchemaParser, TokenEnforcer
        enforcer = TokenEnforcer(self.data, JsonSchemaParser(SCHEMA))
        sequence = [999]  # Untokenized prompt sentinel, ignored by the root parser.
        allowed = enforcer.get_allowed_tokens(sequence).allowed_tokens
        for character in text:
            token = self.token_text.index(character)
            if token not in allowed:
                return False
            sequence.append(token)
            allowed = enforcer.get_allowed_tokens(sequence).allowed_tokens
        return self.eos in allowed

    def test_schema_boundary_enforcement(self):
        for title_length in [1, 60]:
            for tags in [TAGS[:1], TAGS[:2]]:
                self.assertTrue(self.allows(json.dumps({"title": "A" * title_length, "tags": tags})))
        for value in [{"title": "A" * 61, "tags": TAGS[:1]}, {"title": "x", "tags": []},
                      {"title": "x", "tags": TAGS[:3]}, {"title": "x", "tags": ["bad_quadrant"]},
                      {"title": "", "tags": TAGS[:1]}, {"title": "x", "tags": TAGS[:1], "extra": True}]:
            self.assertFalse(self.allows(json.dumps(value)), value)

    def test_logits_mask_blocks_high_scoring_illegal_tokens(self):
        import mlx.core as mx
        from tagging.constrained import SchemaLogitsProcessor
        processor = SchemaLogitsProcessor(self.data, SCHEMA)
        # Give the illegal 'X' a much larger score than legal JSON opening tokens.
        logits = mx.zeros((1, self.eos + 1))
        logits[0, self.token_text.index("X")] = 1000
        masked = processor(mx.array([999]), logits)
        selected = mx.argmax(masked, axis=-1).item()
        self.assertIn(self.token_text[selected], ["{", " "])
        self.assertEqual(masked[0, self.token_text.index("X")].item(), -float("inf"))


if __name__ == "__main__":
    unittest.main()
