"""Milestone one: a reference model plus an explicit greedy decoding loop."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import platform
import resource
import time

import torch
import transformers
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def synchronize(device):
    if device == "mps":
        torch.mps.synchronize()


@torch.inference_mode()
def decode(model, inputs, limit, eos_ids, use_cache, device):
    """Use the library forward pass, but own token selection and cache plumbing."""
    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    cache = None
    generated = []
    first_logits = None
    synchronize(device)
    start = time.perf_counter()
    ttft = None
    for _ in range(limit):
        step_ids = ids[:, -1:] if use_cache and cache is not None else ids
        result = model(input_ids=step_ids, attention_mask=mask,
                       past_key_values=cache, use_cache=use_cache)
        logits = result.logits[:, -1, :]
        if first_logits is None:
            values, indices = logits[0].float().topk(20)
            first_logits = {"token_ids": indices.cpu().tolist(),
                            "logits": values.cpu().tolist()}
        token = logits.argmax(dim=-1, keepdim=True)
        token_id = token.item()  # Synchronizes device result for EOS handling.
        if ttft is None:
            synchronize(device)
            ttft = time.perf_counter() - start
        generated.append(token_id)
        ids = torch.cat((ids, token), dim=1)
        mask = torch.cat((mask, torch.ones_like(token)), dim=1)
        cache = result.past_key_values if use_cache else None
        if token_id in eos_ids:
            break
    synchronize(device)
    elapsed = time.perf_counter() - start
    return {"output_token_ids": generated, "first_step_top20": first_logits,
            "elapsed_seconds": elapsed, "time_to_first_token_seconds": ttft,
            "generated_tokens_per_second": len(generated) / elapsed,
            "use_cache": use_cache}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="Explain batch inference in three short sentences.")
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--compare-no-cache", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "reference.json")
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 256:
        parser.error("--max-new-tokens must be between 1 and 256")
    device = ("mps" if torch.backends.mps.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable in this process; use --device cpu")
    dtype = torch.float16 if device == "mps" else torch.float32
    torch.manual_seed(0)
    lock_path = ROOT / "model-lock.json"
    lock = json.loads(lock_path.read_text()) if lock_path.exists() else None
    if lock and lock["model_id"] != MODEL:
        raise ValueError("Model lock does not match the configured model")
    print(f"Loading {MODEL} on {device} ({dtype})", flush=True)
    load_start = time.perf_counter()
    snapshot = snapshot_download(
        MODEL, revision=lock["revision"] if lock else "main",
        local_files_only=args.offline,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
    )
    revision = Path(snapshot).name
    if not lock:
        lock_path.write_text(json.dumps({"model_id": MODEL, "revision": revision}, indent=2) + "\n")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot, dtype=dtype, attn_implementation="eager",
        local_files_only=True, trust_remote_code=False,
    ).to(device).eval()
    synchronize(device)
    load_seconds = time.perf_counter() - load_start
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}], tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to(device)
    if inputs["input_ids"].shape[1] > 512:
        raise ValueError("Prompt including template exceeds the 512 token educational limit")
    eos = model.generation_config.eos_token_id
    eos_ids = eos if isinstance(eos, list) else [eos]
    config = GenerationConfig(
        do_sample=False, num_beams=1, max_new_tokens=args.max_new_tokens,
        use_cache=True, eos_token_id=eos, pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
    )
    print("Running library reference and warming the device...", flush=True)
    with torch.inference_mode():
        reference = model.generate(**inputs, generation_config=config)
    expected = reference[0, inputs["input_ids"].shape[1]:].cpu().tolist()
    print("Running explicit cached decoding...", flush=True)
    cached = decode(model, inputs, args.max_new_tokens, eos_ids, True, device)
    cached["text"] = tokenizer.decode(cached["output_token_ids"], skip_special_tokens=True)
    checks = {"manual_matches_library_token_for_token": cached["output_token_ids"] == expected}
    no_cache = None
    if args.compare_no_cache:
        print("Running explicit decoding without a KV cache...", flush=True)
        no_cache = decode(model, inputs, args.max_new_tokens, eos_ids, False, device)
        no_cache["text"] = tokenizer.decode(no_cache["output_token_ids"], skip_special_tokens=True)
        checks["cached_matches_uncached_token_for_token"] = cached["output_token_ids"] == no_cache["output_token_ids"]
    report = {
        "model_id": MODEL, "revision": revision, "device": device, "dtype": str(dtype),
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "platform": platform.platform(), "prompt": args.prompt, "rendered_prompt": rendered,
        "input_token_ids": inputs["input_ids"][0].cpu().tolist(),
        "max_new_tokens": args.max_new_tokens, "load_seconds_including_download": load_seconds,
        "reference_output_token_ids": expected,
        "reference_text": tokenizer.decode(expected, skip_special_tokens=True),
        "cached": cached, "uncached": no_cache, "checks": checks,
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "memory_note": "macOS process peak RSS; not a peak GPU or total unified memory measurement",
        "timing_note": "Warm explicit loop; includes token selection and diagnostic top20 capture; excludes load and final text decoding",
    }
    if device == "mps":
        report["mps_current_allocated_bytes"] = torch.mps.current_allocated_memory()
        report["mps_driver_allocated_bytes"] = torch.mps.driver_allocated_memory()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("\n" + cached["text"])
    print(f"\nDevice: {device}; generated tokens: {len(expected)}")
    print(f"Warm loop: {cached['generated_tokens_per_second']:.2f} tokens/s")
    print(f"Checks: {checks}\nSaved: {args.output}")
    if not all(checks.values()):
        raise SystemExit("Token mismatch: inspect the saved report. Floating point differences can change greedy choices.")


if __name__ == "__main__":
    main()
