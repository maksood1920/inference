# Milestone two validation

Device: cpu; precision: torch.float32. MPS has not been validated from this execution process.

All four small-model tests passed. All three real-checkpoint prompts passed all-position logit comparison, causal-prefix checks, and generated-token comparison. Maximum CPU logit error was zero on these three prompts.

| Prompt | Maximum logit error | Generated tokens match |
|---|---:|---|
| What is the capital of France? Answer with only the city name. | 0.0 | True |
| Name five colors. | 0.0 | True |
| Explain why a KV cache reduces repeated computation in two sentences. | 0.0 | True |

These checks use the exact pinned checkpoint and identical read-only weights. They do not establish exhaustive correctness or model factual accuracy. The custom engine has no KV cache yet.

Run on your Mac GPU:

```sh
.venv/bin/python validate_engine.py --device mps --output results/engine-validation-mps.json
.venv/bin/python run_engine.py --device mps --prompt "Name five colors."
```

The validator uses dtype-specific numerical tolerances and exits nonzero if any check fails. Preserve the CPU report.
