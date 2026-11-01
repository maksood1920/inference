"""Compare our forward pass against the pinned Transformers reference offline."""
import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine import QwenEngine
from run import MODEL, ROOT, synchronize


@torch.inference_mode()
def check_causality(forward, ids, full_logits, vocab_size, atol, rtol):
    """Change future token content without changing tensor shapes.

    A short-prefix run is also recorded, but is not the causality gate: changing
    sequence length can change floating-point kernels and rounding on a device.
    """
    cut = min(7, ids.shape[1] - 1)
    if cut < 1:
        raise ValueError("Causality check requires at least two tokens")
    original_prefix = full_logits[:, :cut].float()
    variants = []
    for offset in (1, 17):
        changed = ids.clone()
        changed[:, cut:] = (changed[:, cut:] + offset) % vocab_size
        changed_logits = forward(changed)[:, :cut].float()
        variants.append({"suffix_token_offset": offset,
                         "prefix_logits_close": torch.allclose(original_prefix, changed_logits, atol=atol, rtol=rtol),
                         "max_absolute_error": (original_prefix - changed_logits).abs().max().item()})
    short_logits = forward(ids[:, :cut]).float()
    return {"prefix_length": cut,
            "same_shape_future_independence": all(v["prefix_logits_close"] for v in variants),
            "suffix_mutations": variants,
            "cross_length_prefix_close": torch.allclose(short_logits, original_prefix, atol=atol, rtol=rtol),
            "cross_length_max_absolute_error": (short_logits - original_prefix).abs().max().item(),
            "short_logits": short_logits}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--output", type=str, default=str(ROOT / "results" / "engine-validation.json"))
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 64:
        parser.error("Validation generates between 1 and 64 tokens per case")
    device = ("mps" if torch.backends.mps.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS unavailable; use --device cpu")
    dtype = torch.float16 if device == "mps" else torch.float32
    lock = json.loads((ROOT / "model-lock.json").read_text())
    snapshot = ROOT / ".cache" / "huggingface" / "hub" / ("models--" + MODEL.replace("/", "--")) / "snapshots" / lock["revision"]
    print(f"Loading pinned reference on {device}; all model access is offline", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    reference = AutoModelForCausalLM.from_pretrained(
        snapshot, dtype=dtype, attn_implementation="eager", local_files_only=True,
        trust_remote_code=False,
    ).to(device).eval()
    # Share read-only weight tensors rather than holding two model copies.
    # Our engine never calls the reference model's layers or forward method.
    engine = QwenEngine(reference.config.to_dict(), reference.state_dict())
    eos = reference.generation_config.eos_token_id
    eos_ids = eos if isinstance(eos, list) else [eos]
    # Different numerical kernels can introduce floating-point differences.
    atol, rtol = (0.05, 0.005) if device == "mps" else (0.0002, 0.0001)
    cases = []
    prompts = ["What is the capital of France? Answer with only the city name.",
               "Name five colors.",
               "Explain why a KV cache reduces repeated computation in two sentences."]
    with torch.inference_mode():
        for prompt in prompts:
            text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
            ids = inputs["input_ids"]
            expected_logits = reference(**inputs, use_cache=False).logits
            actual_logits = engine.forward(ids)
            delta = (actual_logits.float() - expected_logits.float()).abs()
            allclose = torch.allclose(actual_logits.float(), expected_logits.float(), atol=atol, rtol=rtol)
            # Isolate future-token dependence from sequence-length rounding.
            ours = check_causality(engine.forward, ids, actual_logits,
                                   reference.config.vocab_size, atol, rtol)
            reference_checks = check_causality(
                lambda tokens: reference(input_ids=tokens, use_cache=False).logits,
                ids, expected_logits, reference.config.vocab_size, atol, rtol)
            short_logits = ours.pop("short_logits")
            ref_short_logits = reference_checks.pop("short_logits")
            short_matches = torch.allclose(short_logits, ref_short_logits, atol=atol, rtol=rtol)
            short_error = (short_logits - ref_short_logits).abs().max().item()
            causal = ours["same_shape_future_independence"]
            del short_logits, ref_short_logits
            ref_ids = ids.clone()
            expected_tokens = []
            for _ in range(args.max_new_tokens):
                token = reference(input_ids=ref_ids, use_cache=False).logits[:, -1].argmax(-1, keepdim=True)
                token_id = token.item()
                expected_tokens.append(token_id)
                if token_id in eos_ids:
                    break
                ref_ids = torch.cat((ref_ids, token), dim=1)
            synchronize(device)
            start = time.perf_counter()
            actual_tokens = engine.generate(ids, args.max_new_tokens, eos_ids)
            synchronize(device)
            elapsed = time.perf_counter() - start
            case = {"prompt": prompt, "input_tokens": ids.shape[1],
                    "logits_shape": list(actual_logits.shape),
                    "max_absolute_logit_error": delta.max().item(),
                    "mean_absolute_logit_error": delta.mean().item(),
                    "all_position_logits_close": allclose,
                    "causal_future_independence": causal,
                    "engine_causality": ours,
                    "reference_causality": reference_checks,
                    "short_prefix_matches_reference": short_matches,
                    "short_prefix_reference_max_error": short_error,
                    "generated_tokens_match": actual_tokens == expected_tokens,
                    "reference_token_ids": expected_tokens, "engine_token_ids": actual_tokens,
                    "engine_text": tokenizer.decode(actual_tokens, skip_special_tokens=True),
                    "engine_generation_seconds": elapsed,
                    "engine_tokens_per_second": len(actual_tokens) / elapsed}
            cases.append(case)
            print(f"{prompt}\n  logits close={allclose}; causal={causal}; short reference match={short_matches}; tokens match={case['generated_tokens_match']}; max error={case['max_absolute_logit_error']:.6g}", flush=True)
            print(f"  Cross-length diagnostic (not a gate): engine={ours['cross_length_prefix_close']}, reference={reference_checks['cross_length_prefix_close']}; max errors={ours['cross_length_max_absolute_error']:.6g}/{reference_checks['cross_length_max_absolute_error']:.6g}", flush=True)
    report = {"model_id": MODEL, "revision": lock["revision"], "device": device,
              "dtype": str(dtype), "torch_version": torch.__version__,
              "atol": atol, "rtol": rtol, "use_cache": False,
              "validation_version": 2,
              "note": "Causality changes future token content at fixed shape. Cross-length comparisons are diagnostics; same-length and short-prefix reference parity remain required. Numerical tolerances are unchanged. Timings are educational.",
              "cases": cases}
    from pathlib import Path
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {output}")
    if not all(c["all_position_logits_close"] and c["causal_future_independence"]
               and c["reference_causality"]["same_shape_future_independence"]
               and c["short_prefix_matches_reference"] and c["generated_tokens_match"] for c in cases):
        raise SystemExit("Validation failed; inspect the report before trusting this implementation")


if __name__ == "__main__":
    main()
