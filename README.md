# Mac inference lab

An educational inference engine on an M3 Mac with 16 GB memory.
Uses Qwen2.5-0.5B-Instruct, the instruction-tuned variant of the proposed model,
so the initial prompt produces an answer rather than a text continuation.

## Run

From this folder, after setup:

```sh
.venv/bin/python run.py --compare-no-cache
.venv/bin/python run.py --offline --prompt "Why does a KV cache help decoding?" --output results/kv-cache.json
```

The default chooses MPS when available, otherwise CPU. Force either with
`--device mps` or `--device cpu`. CPU uses float32; MPS uses float16.
The first run downloads about 1 GB of model weights and pins the exact model
revision in `model-lock.json`. Model files remain in this folder's `.cache`.
After downloading, `--offline` prohibits further model network access.
No paid inference API is called; generation runs locally.

To recreate the environment using a compatible Python 3.12 installation:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
```

## What is implemented

Milestone two is now implemented in `engine.py`: the model forward pass itself,
including embeddings, RMS normalization, rotary positions, grouped-query causal
attention, SwiGLU feed-forward layers, residual additions, and the vocabulary head.
It uses PyTorch operations and pretrained weights; it does not call Transformers
model layers. See `FORWARD_PASS.md` for a beginner walkthrough.

Validate it against the reference, then run it independently:

```sh
.venv/bin/python -m unittest -v test_engine.py
.venv/bin/python validate_engine.py --device mps --output results/engine-validation-mps.json
.venv/bin/python run_engine.py --device mps --prompt "Name five colors."
```

Both commands use the downloaded checkpoint offline. Use `--device cpu` if MPS
is unavailable. `validate_engine.py` compares all prompt-position logits within
explicit numerical tolerances, verifies future-token independence, and checks greedy
token agreement on three prompts. It saves a report and exits nonzero on failure.
MPS uses looser float16 tolerances than the CPU float32 path. Those tolerances
are acceptance thresholds, not claims of bitwise equality or exhaustive proof.

Validator version 2 changes future token IDs while keeping tensor shapes fixed
to test causality. It also compares the short-prefix run directly to Transformers.
The original short-versus-long prefix comparison remains a diagnostic, including
the reference's own error. Shape-dependent rounding is possible, so this
cross-length diagnostic alone no longer determines success. Tolerances have not
been increased. Preserve earlier reports by choosing a new output filename.

This milestone deliberately has **no KV cache**. It recomputes the full growing
sequence. The existing milestone-one `run.py` still demonstrates Transformers'
KV cache. Our own cache is the next optimization, after forward-pass correctness.

### Milestone one reference

`run.py` loads the tokenizer and model forward pass from Transformers. It first
uses `model.generate` as a reference, then runs its own explicit greedy loop:

1. Run a forward pass on the prompt.
2. Select the largest last-position logit.
3. Append the selected token and extend the attention mask.
4. Feed only the new token when a KV cache is available.
5. Stop at an end token or the output cap.

`--compare-no-cache` reruns the loop on the full growing sequence each time.
Both paths are checked against the reference tokens. A mismatch is reported
and exits nonzero; it is never silently accepted. Numerical differences can
cause greedy outputs to diverge, especially between devices and dtypes.

This reference script owns the generation loop but delegates its forward pass
to Transformers. The new `run_engine.py` instead uses our own model calculations.

## Results and measurement

`results/reference.json` stores the model revision, library versions, token IDs,
generated text, first-step top logits, timings, and correctness checks.
Use separate output filenames for experiments to preserve the original.

Timing excludes model loading and final text decoding. The library reference
warms the device before the explicit loop. The first-token timing includes a
small diagnostic top-logit capture. These are educational measurements, not
production throughput benchmarks. Cache/no-cache order is fixed, so thermal
effects and warmup may affect comparisons. Repeat runs before drawing conclusions.

Peak RSS is the process high-water mark on macOS, including loading and all
comparison paths. MPS allocation counters, when available, are current values,
not peak GPU usage. Neither is a measurement of total Mac memory use.

Limits: one sequence; at most 512 prompt tokens including the chat template;
1 to 256 output tokens, default 64; eager attention for transparency.

## References

- Model and license: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
- Mac GPU backend: https://docs.pytorch.org/docs/stable/notes/mps.html
- Generation interface: https://huggingface.co/docs/transformers/en/main_classes/text_generation
