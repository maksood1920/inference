"""Small random models catch math and masking errors without a large checkpoint."""
import unittest

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from engine import QwenEngine
from validate_engine import check_causality


class EngineTests(unittest.TestCase):
    def make_pair(self, tied=True, kv_heads=2):
        torch.manual_seed(42)
        config = Qwen2Config(vocab_size=67, hidden_size=32, intermediate_size=48,
                             num_hidden_layers=2, num_attention_heads=4,
                             num_key_value_heads=kv_heads, max_position_embeddings=128,
                             tie_word_embeddings=tied, use_sliding_window=False)
        config._attn_implementation = "eager"
        reference = Qwen2ForCausalLM(config).eval()
        return reference, QwenEngine(config.to_dict(), reference.state_dict())

    def test_all_position_logits_match_for_tied_untied_and_grouped_heads(self):
        for tied, heads in [(True, 1), (True, 2), (False, 4)]:
            reference, engine = self.make_pair(tied, heads)
            for ids in [torch.tensor([[3]]), torch.tensor([[3, 9, 12, 5, 8]])]:
                with torch.inference_mode():
                    expected = reference(input_ids=ids, use_cache=False).logits
                torch.testing.assert_close(engine.forward(ids), expected, atol=1e-6, rtol=1e-5)

    def test_future_tokens_cannot_change_earlier_logits(self):
        _, engine = self.make_pair()
        prefix = torch.tensor([[3, 9, 12]])
        longer = torch.tensor([[3, 9, 12, 5, 8]])
        torch.testing.assert_close(engine.forward(prefix), engine.forward(longer)[:, :3], atol=1e-6, rtol=1e-5)

    def test_generation_honors_eos_and_length_cap(self):
        _, engine = self.make_pair()
        ids = torch.tensor([[3, 9]])
        first = engine.forward(ids)[:, -1].argmax().item()
        self.assertEqual(engine.generate(ids, 5, [first]), [first])
        self.assertEqual(len(engine.generate(ids, 3, [])), 3)

    def test_causality_validator_detects_future_leakage(self):
        _, engine = self.make_pair()
        ids = torch.tensor([[3, 9, 12, 5, 8, 7, 4, 2, 6, 1]])
        valid = check_causality(engine.forward, ids, engine.forward(ids), 67, 1e-6, 1e-5)
        self.assertTrue(valid["same_shape_future_independence"])

        # Deliberately corrupt every logit with information from the last token.
        def leaky(tokens):
            return engine.forward(tokens) + tokens[:, -1:].float().unsqueeze(-1)

        invalid = check_causality(leaky, ids, leaky(ids), 67, 1e-6, 1e-5)
        self.assertFalse(invalid["same_shape_future_independence"])

    def test_causality_validator_separates_shape_drift(self):
        ids = torch.tensor([[3, 9, 12, 5, 8, 7, 4, 2, 6, 1]])

        # A synthetic causal oracle with sequence-length-dependent numerical
        # drift demonstrates why the two measurements must be separate.
        def shape_drift(tokens):
            return tokens.float().unsqueeze(-1) + tokens.shape[1] * 0.01

        checks = check_causality(shape_drift, ids, shape_drift(ids), 67, 1e-6, 1e-5)
        self.assertTrue(checks["same_shape_future_independence"])
        self.assertFalse(checks["cross_length_prefix_close"])

    def test_cache_matches_full_pass_for_multiple_chunk_sizes(self):
        _, engine = self.make_pair()
        ids = torch.tensor([[3, 9, 12, 5, 8, 2]])
        expected = engine.forward(ids)
        for chunks in [(6,), (3, 1, 1, 1), (2, 2, 2)]:
            past, start = None, 0
            for length in chunks:
                stop = start + length
                logits, past = engine.forward_cached(ids[:, start:stop], past)
                torch.testing.assert_close(logits, expected[:, start:stop],
                                           atol=1e-6, rtol=1e-5)
                self.assertEqual(len(past), 2)
                for key, value in past:
                    self.assertEqual(key.shape, (1, 2, stop, 8))
                    self.assertEqual(value.shape, key.shape)
                start = stop

    def test_cached_generation_matches_uncached_and_resets_between_requests(self):
        _, engine = self.make_pair()
        for ids in [torch.tensor([[3, 9]]), torch.tensor([[8, 1]])]:
            expected = engine.generate(ids, 5, [])
            self.assertEqual(engine.generate(ids, 5, [], use_cache=True), expected)
            self.assertEqual(engine.generate(ids, 5, [expected[0]], use_cache=True),
                             expected[:1])

    def test_batch_matches_individual_across_padding_and_early_eos(self):
        _, engine = self.make_pair()
        short = torch.tensor([3, 9])
        long = torch.tensor([6, 7, 8, 10, 12])
        eos = engine.generate(short[None], 1, [], use_cache=True)[0]
        prompts = [long, short]
        expected = [engine.generate(p[None], 6, [eos], use_cache=True) for p in prompts]
        self.assertEqual(len(expected[1]), 1)
        self.assertGreater(len(expected[0]), 1)
        for pad_id in (0, 34):
            self.assertEqual(engine.generate_batch(prompts, 6, [eos], pad_id=pad_id), expected)
            self.assertEqual(engine.generate_batch(prompts[::-1], 6, [eos], pad_id=pad_id), expected[::-1])

    def test_continuous_batch_refills_a_slot_before_long_request_finishes(self):
        _, engine = self.make_pair()
        short = torch.tensor([3, 9])
        long = torch.tensor([6, 7, 8, 10, 12])
        queued = torch.tensor([1, 2, 4])
        prompts = [short, long, queued, short]
        eos = [engine.generate(short[None], 1, [], use_cache=True)[0]]
        expected = [engine.generate(p[None], 6, eos, use_cache=True) for p in prompts]
        outputs, schedule = engine.generate_continuous(prompts, 6, eos, pad_id=0)
        self.assertEqual(outputs, expected)
        self.assertEqual(schedule[0], {"admitted_at": 0, "finished_at": 0})
        self.assertEqual(schedule[2]["admitted_at"], 1)
        self.assertLess(schedule[2]["admitted_at"], schedule[1]["finished_at"])
        self.assertGreater(schedule[3]["admitted_at"], schedule[2]["admitted_at"])

    def test_rejects_unsupported_inputs_and_configs(self):
        _, engine = self.make_pair()
        for ids in [torch.ones(2, 3, dtype=torch.long), torch.ones(1, 0, dtype=torch.long),
                    torch.ones(1, 3, dtype=torch.float32)]:
            with self.assertRaises(ValueError):
                engine.forward(ids)
        with self.assertRaises(ValueError):
            QwenEngine({**engine.config, "use_sliding_window": True}, engine.weights)


if __name__ == "__main__":
    unittest.main()
