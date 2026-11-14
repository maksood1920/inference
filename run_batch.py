"""Compare static batching with sequential cached generation."""
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
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--prompt", action="append", help="Repeat for each request")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "batch-comparison.json")
    args = parser.parse_args()
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS unavailable")
    prompts = args.prompt or ["What is the capital of France? Answer with only the city name.",
                              "Name five colors."]
    lock = json.loads((ROOT / "model-lock.json").read_text())
    if lock["model_id"] != MODEL:
        raise ValueError("Unexpected model lock")
    snapshot = ROOT / ".cache/huggingface/hub" / ("models--" + MODEL.replace("/", "--")) / "snapshots" / lock["revision"]
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    engine = QwenEngine.from_snapshot(snapshot, args.device,
                                      torch.float16 if args.device == "mps" else torch.float32)
    eos = json.loads((snapshot / "generation_config.json").read_text())["eos_token_id"]
    eos_ids = [eos] if isinstance(eos, int) else eos
    ids = []
    for prompt in prompts:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        ids.append(tokenizer(rendered, return_tensors="pt", add_special_tokens=False)
                   ["input_ids"][0].to(args.device))

    def batch(limit):
        return engine.generate_batch(ids, limit, eos_ids, pad_id=tokenizer.pad_token_id)

    def serial(limit):
        return [engine.generate(row[None], limit, eos_ids, use_cache=True) for row in ids]

    def measure(fn):
        fn(min(2, args.max_new_tokens))
        synchronize(args.device)
        start = time.perf_counter()
        output = fn(args.max_new_tokens)
        synchronize(args.device)
        seconds = time.perf_counter() - start
        return output, {"seconds": seconds, "tokens_per_second": sum(map(len, output)) / seconds}

    actual, batched = measure(batch)
    expected, sequential = measure(serial)
    rows = [{"prompt": prompt, "prompt_tokens": len(tokens), "output_ids": output,
             "sequential_ids": baseline, "matches": output == baseline,
             "stop": "eos" if output[-1] in eos_ids else "token_limit",
             "text": tokenizer.decode(output, skip_special_tokens=True)}
            for prompt, tokens, output, baseline in zip(prompts, ids, actual, expected)]
    report = {"model_revision": lock["revision"], "device": args.device, "dtype": str(engine.dtype),
              "max_new_tokens": args.max_new_tokens, "batched": batched,
              "sequential": sequential, "requests": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for row in rows:
        print(f"{row['prompt']} → {row['text']} (stop={row['stop']}, matches={row['matches']})")
    print(f"Batch: {batched['tokens_per_second']:.2f} tokens/s; "
          f"sequential: {sequential['tokens_per_second']:.2f} tokens/s")
    print(f"Saved {args.output}")
    if not all(row["matches"] for row in rows):
        raise SystemExit("Batch and sequential greedy tokens differ; see the saved report.")


if __name__ == "__main__":
    main()
