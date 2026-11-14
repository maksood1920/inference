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

### Milestone three: our own KV cache

The default runner remains uncached. Use `--kv-cache` to cache each layer's
keys and values, or `--compare-cache` to time both modes with the same prompt
and verify that their greedy token IDs match.

```sh
.venv/bin/python run_engine.py --device mps --compare-cache --output results/own-cache-mps.json
```

The first call processes the whole prompt. Later calls process one token at a
time with its absolute position and attend to the saved history. Each layer
stores the original KV heads, before expanding them to match the query heads.
The cache is local to a generation request and grows through concatenation.
It does not implement paging or batching. Timing includes prefill and decoding,
excludes model loading, and reflects a single run per mode. Small floating-point
differences between sequence shapes can change a greedy selection.

### Milestone four: static batching

```sh
.venv/bin/python run_batch.py --device mps --output results/batch-mps.json
```

The default prompts ask for a city name and five colors. Both share a prefill
call, then share each decode call while both are active. Each prompt has a
separate row in the KV cache. Right padding is masked out, and each row uses its
own token positions. When a row emits EOS, it leaves the batch; remaining rows
continue. New prompts cannot join a running batch yet.

Repeat `--prompt` to supply up to eight prompts. The runner compares every
output token ID against sequential cached generation and reports aggregate
tokens per second for each mode. Timings include prefill and decode but exclude
loading and tokenization. Small floating-point differences can change greedy
choices across batch shapes, so mismatches are saved rather than hidden.

### Milestone five: continuous batching

```sh
.venv/bin/python run_batch.py --continuous --device mps --output results/continuous-mps.json
```

Three requests start in a FIFO queue with two active slots. The city and
colors prompts start together. When the city finishes, the KV-cache question
takes its slot while colors is still generating. Each request starts with
prefill. Subsequent decode calls batch the active requests using their own
cache histories and token positions. The runner prints the zero-based round
when each request starts and finishes and checks its output against sequential
cached generation.

The lab simulates a queue of already submitted requests. It uses a fixed slot
limit and copies compact per-request caches into temporary padded decode
batches. Prefill runs as a separate model call in the admission round. Packing
cost can outweigh batching gains for small workloads.

### Milestone six: local HTTP inference

Start the server on your Mac.

```sh
.venv/bin/python serve.py --device mps
```

Open http://127.0.0.1:8000/generate to submit a prompt in your browser,
or use another terminal.

```sh
curl -s http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Name five colors.","max_new_tokens":32}'
```

The server binds to localhost and loads the pinned checkpoint from disk. One
worker owns the model. Up to two requests decode at once, and up to 16 can wait
in the FIFO queue. A full queue returns HTTP 503. To see live admission, submit
a long story request with a 128-token limit, then submit a short question from
a separate client while the story is still running. Compare the admission and
finish rounds in the two responses.

Each response includes the text, generated token IDs, stop reason, scheduling
rounds, and server-side timings. Time to first token runs from the start of the
HTTP handler through first-token selection and includes queue time. The browser requests Server-Sent Events (SSE) to display text during generation.
Send `"stream": true` in the JSON body and use `curl -N` to see the events in a
terminal. Each `text` event carries the current decoded text, and `done` carries
the final response with timing. An `error` event ends a stream if generation
fails after headers are sent. The default API response remains a single JSON
object after generation finishes. Use `--device cpu`
to run the same server without MPS.

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
