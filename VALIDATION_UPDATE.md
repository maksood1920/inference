# Investigating the first GPU validation failure

The original MPS report shows zero difference between our logits and the
Transformers reference for each full prompt, and identical generated tokens.
However, all three short-versus-long prefix comparisons failed. The old report
does not include the difference magnitude or run that comparison on the reference,
so it cannot establish the cause.

One plausible cause is float16 numerical differences from changing matrix
shapes. PyTorch notes that mathematically equivalent batched and sliced
computations need not produce identical floating-point results:
https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html

Validator version 2 separates the questions:

1. Do full-prompt logits match Transformers? Required.
2. Do short-prefix logits match Transformers at that same shape? Required.
3. Do earlier logits stay invariant when future token content changes at fixed
   shape? Required for both our engine and the reference; two suffix mutations.
4. Do generated tokens match? Required.
5. Do prefix logits match across different sequence lengths? Recorded for both
   implementations, with error magnitudes, as a numerical diagnostic.

No engine calculations or numerical tolerance thresholds were changed.
Six CPU tests pass, including a test that intentionally introduces future-token
leakage and must be rejected. Three real-checkpoint CPU cases pass the revised
validator, with zero custom-versus-reference full-prompt logit error. Both engine
and reference have the same small cross-length CPU error (about 0.0001024).

MPS is unavailable to the agent's execution process. The revised GPU validation
is still pending. Run from this folder:

```sh
.venv/bin/python validate_engine.py --device mps --output results/engine-validation-mps-v2.json
```

This preserves the original failed GPU report. If the fixed-shape or same-shape
reference checks fail, inspect their error values before trusting the engine.
Matching natural-language output alone is insufficient.
