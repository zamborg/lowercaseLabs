"""Check that the mask permits every quadrant with each real cached tokenizer.

This catches vocabulary-adapter bugs that schema validity alone cannot detect:
a broken adapter could generate valid JSON while making one enum unreachable.
No weights are loaded and no downloads or API calls are made.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, SCHEMA, TAGS, MODEL_CACHE, MODEL_PATTERNS, configure_caches
configure_caches()


class RealTokenizerTests(unittest.TestCase):
    def test_all_quadrants_and_schema_boundaries_for_each_cached_candidate(self):
        import yaml
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import LocalEntryNotFoundError
        from lmformatenforcer import JsonSchemaParser, TokenEnforcer
        from mlx_lm.tokenizer_utils import load
        from tagging.constrained import tokenizer_data

        candidates = yaml.safe_load((ROOT / "tagging" / "candidates.yaml").read_text())["candidates"]
        checked = 0
        for candidate in candidates:
            with self.subTest(model=candidate["id"]):
                try:
                    snapshot = Path(snapshot_download(candidate["repo"], revision=candidate["revision"],
                                    cache_dir=MODEL_CACHE, allow_patterns=MODEL_PATTERNS, local_files_only=True))
                except LocalEntryNotFoundError as exc:
                    cache = MODEL_CACHE / ("models--" + candidate["repo"].replace("/", "--"))
                    if cache.exists():
                        self.fail(f"Cached candidate snapshot is incomplete: {exc}")
                    continue
                config = json.loads((snapshot / "config.json").read_text())
                tokenizer = load(snapshot, eos_token_ids=config.get("eos_token_id"))
                data = tokenizer_data(tokenizer)

                def accepts(value):
                    raw = json.dumps(value, ensure_ascii=True)
                    enforcer = TokenEnforcer(data, JsonSchemaParser(SCHEMA))
                    enforcer.root_parser.config.force_json_field_order = True
                    sequence = [999999999]
                    allowed = enforcer.get_allowed_tokens(sequence).allowed_tokens
                    for token in tokenizer.encode(raw, add_special_tokens=False):
                        if token not in allowed:
                            return False
                        sequence.append(token)
                        allowed = enforcer.get_allowed_tokens(sequence).allowed_tokens
                    return any(token in allowed for token in tokenizer.eos_token_ids)

                for index, tag in enumerate(TAGS):
                    self.assertTrue(accepts({"title": "Tea after work", "tags": [tag]}), tag)
                    self.assertTrue(accepts({"title": "A" * 60, "tags": [tag, TAGS[(index + 1) % 4]]}), tag)
                for value in [{"title": "A" * 61, "tags": TAGS[:1]}, {"title": "Tea", "tags": []},
                              {"title": "Tea", "tags": TAGS[:3]}, {"title": "Tea", "tags": ["bad_quadrant"]}]:
                    self.assertFalse(accepts(value), value)
                checked += 1
        if checked == 0:
            self.skipTest("No candidate tokenizer snapshots cached; run the sample benchmark first")
        self.assertEqual(checked, len(candidates), "Only some candidates cached; run the full sample benchmark")


if __name__ == "__main__":
    unittest.main()
