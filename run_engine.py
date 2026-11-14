"""Generate with our own forward pass and the already-downloaded model weights."""
import argparse
import json
import time
from pathlib import Path

from run import MODEL, ROOT, synchronize
import torch
from transformers import AutoTokenizer
from engine import QwenEngine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="Name five colors.")
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "own-engine.json")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--kv-cache", action="store_true", help="Use our own KV cache")
    mode.add_argument("--compare-cache", action="store_true", help="Time both modes and compare output IDs")
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 256:
        parser.error("--max-new-tokens must be between 1 and 256")
    device = ("mps" if torch.backends.mps.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS unavailable; use --device cpu")
    dtype = torch.float16 if device == "mps" else torch.float32
    lock = json.loads((ROOT / "model-lock.json").read_text())
    if lock["model_id"] != MODEL:
        raise ValueError("Unexpected model lock")
    snapshot = ROOT / ".cache" / "huggingface" / "hub" / ("models--" + MODEL.replace("/", "--")) / "snapshots" / lock["revision"]
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    text = tokenizer.apply_chat_template([{"role": "user", "content": args.prompt}], tokenize=False, add_generation_prompt=True)
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    if ids.shape[1] > 512:
        parser.error("Prompt including template exceeds 512 tokens")
    print(f"Loading our engine on {device} ({dtype})", flush=True)
    engine = QwenEngine.from_snapshot(snapshot, device, dtype)
    generation = json.loads((snapshot / "generation_config.json").read_text())
    eos = generation["eos_token_id"]
    eos_ids = eos if isinstance(eos, list) else [eos]
    def measure(use_cache):
        engine.generate(ids, min(2, args.max_new_tokens), eos_ids, use_cache=use_cache)
        synchronize(device)
        start = time.perf_counter()
        tokens = engine.generate(ids, args.max_new_tokens, eos_ids, use_cache=use_cache)
        synchronize(device)
        elapsed = time.perf_counter() - start
        return tokens, elapsed

    use_cache = args.kv_cache or args.compare_cache
    tokens, elapsed = measure(use_cache)
    report = {"model_id": MODEL, "revision": lock["revision"], "device": device,
              "dtype": str(dtype), "prompt": args.prompt, "input_token_ids": ids[0].cpu().tolist(),
              "output_token_ids": tokens, "text": tokenizer.decode(tokens, skip_special_tokens=True),
              "use_cache": use_cache, "elapsed_seconds": elapsed,
              "tokens_per_second": len(tokens) / elapsed,
              "stop_reason": "eos" if tokens[-1] in eos_ids else "token_limit"}
    if args.compare_cache:
        baseline, duration = measure(False)
        report["uncached"] = {"output_token_ids": baseline,
                              "elapsed_seconds": duration,
                              "tokens_per_second": len(baseline) / duration}
        report["cached_tokens_match"] = tokens == baseline
        print(f"Uncached: {len(baseline) / duration:.2f} tokens/s; matching tokens: {tokens == baseline}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(report["text"])
    print(f"\n{len(tokens)} tokens in {elapsed:.2f}s; {report['tokens_per_second']:.2f} tokens/s; stop={report['stop_reason']}")
    print(f"Saved {args.output}")
    if args.compare_cache and not report["cached_tokens_match"]:
        raise SystemExit("Cached and uncached greedy tokens differ; see saved token IDs.")


if __name__ == "__main__":
    main()
