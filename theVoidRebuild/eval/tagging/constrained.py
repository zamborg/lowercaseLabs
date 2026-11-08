"""MLX logits masking backed by LM Format Enforcer's JSON Schema parser.

Tokenizer handling follows LMFE's Transformers integration without importing
PyTorch: https://github.com/noamgat/lm-format-enforcer (MIT).
"""
from __future__ import annotations

import time

from lmformatenforcer import JsonSchemaParser, TokenEnforcer, TokenEnforcerTokenizerData


def tokenizer_data(tokenizer):
    underlying = getattr(tokenizer, "_tokenizer", tokenizer)
    special = set(underlying.all_special_ids)
    zero = underlying.encode("0", add_special_tokens=False)[-1]
    zero_text = underlying.decode([zero], clean_up_tokenization_spaces=False)
    vocab_size = len(underlying)
    regular = []
    for index in range(vocab_size):
        if index in special:
            continue
        after_zero = underlying.decode([zero, index], clean_up_tokenization_spaces=False)[len(zero_text):]
        alone = underlying.decode([index], clean_up_tokenization_spaces=False)
        if after_zero:
            regular.append((index, after_zero, len(after_zero) > len(alone)))

    def decode(tokens):
        return underlying.decode(tokens, clean_up_tokenization_spaces=False).rstrip("\ufffd")

    eos_ids = sorted(tokenizer.eos_token_ids)
    return TokenEnforcerTokenizerData(regular, decode, eos_ids, False, vocab_size)


class SchemaLogitsProcessor:
    def __init__(self, data, schema):
        self.enforcer = TokenEnforcer(data, JsonSchemaParser(schema))
        # Canonical field order and bounded formatting reduce wasted output tokens.
        self.enforcer.root_parser.config.force_json_field_order = True
        self.enforcer.root_parser.config.max_consecutive_whitespaces = 2
        self.calls = 0
        self.cpu_seconds = 0.0
        self.initial_prefix_length = None

    def __call__(self, tokens, logits):
        import mlx.core as mx
        start = time.perf_counter()
        sequence = tokens.tolist()
        if self.initial_prefix_length is None:
            # MLX versions differ in how much of the prefill reaches processors.
            # LMFE initializes the first sequence as its root, ignoring prompt text.
            self.initial_prefix_length = len(sequence)
        allowed = self.enforcer.get_allowed_tokens(sequence).allowed_tokens
        allowed = [token for token in allowed if 0 <= token < logits.shape[-1]]
        if not allowed:
            raise RuntimeError("Schema parser has no legal token; never fall back to unconstrained decoding")
        mask = mx.full((logits.shape[-1],), -float("inf"), dtype=logits.dtype)
        mask[mx.array(allowed, dtype=mx.int32)] = 0
        result = logits + mask
        self.calls += 1
        self.cpu_seconds += time.perf_counter() - start
        return result
