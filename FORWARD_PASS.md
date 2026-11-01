# Understanding our own model forward pass

We now own the calculations inside the model box, in addition to the generation
loop. We still use an existing tokenizer, pretrained weights, and PyTorch tensor
operations. We have not trained a model or written GPU kernels.

## The files

| File | Role |
|---|---|
| `run.py` | Milestone-one reference using Transformers' forward pass and KV cache |
| `engine.py` | Our own model forward pass and uncached generation loop |
| `run_engine.py` | Load local weights and generate using our engine |
| `validate_engine.py` | Compare our engine against the pinned reference |
| `test_engine.py` | Small random-model checks for math, causality, and stopping |

## Read the code in this order

Start with `QwenEngine.forward`, then `attention`, then `generate`. The helper
functions implement normalization, linear projections, and positional rotations.

The checkpoint has 24 layers, hidden size 896, 14 query heads, 2 key/value heads,
and vocabulary size 151,936. A tensor is an array; its shape tells you how many
numbers it holds along each axis. `T` below means the number of input tokens.

```text
Token IDs                         [1, T]
   ↓ embedding lookup
Token representations             [1, T, 896]
   ↓ 24 transformer layers
Final token representations       [1, T, 896]
   ↓ vocabulary projection
Logits                            [1, T, 151936]
   ↓ take last position, then argmax
One next-token ID                 [1, 1]
```

## Embeddings

The input token ID selects a row in a learned table. That row is an 896-number
vector. The table comes from the pretrained checkpoint. We use `F.embedding`.
This step is a lookup, not token generation.

## One transformer layer

Each of the 24 layers performs:

```text
x → RMS normalization → attention → add original x
  → RMS normalization → feed-forward network → add incoming representation
```

RMS normalization controls the scale of vectors using their root-mean-square
and a learned per-dimension weight. We accumulate squares in float32 to reduce
rounding problems, then return to the model's precision.

Residual additions preserve the incoming vector while adding a learned update.
The two residual additions happen at different points and use their respective
incoming representations.

## Attention

Learned linear projections calculate queries, keys, and values from normalized
representations. A linear projection is a matrix multiplication, sometimes with
a bias added. For this checkpoint:

```text
Q: [1, 14, T, 64]
K: [1,  2, T, 64]
V: [1,  2, T, 64]
```

RoPE rotates Q and K using token positions. It gives attention position
information. It does not rotate V. This checkpoint uses the default RoPE scheme;
our implementation explicitly rejects scaled RoPE and sliding-window attention.

Seven query heads share each KV head. This is grouped-query attention. Our
simple implementation expands K and V logically to match the query heads for
the calculation; this is not a memory-optimized production attention kernel.

The main attention calculation is:

```text
scores = Q @ transpose(K) / sqrt(64)
scores = scores + causal_mask
weights = softmax(scores)
attention_result = weights @ V
```

The causal mask blocks future positions. For three positions:

```text
               May read position
               0    1    2
Position 0     yes  no   no
Position 1     yes  yes  no
Position 2     yes  yes  yes
```

Softmax turns the allowed scores into weights that sum to approximately one
along each attention row. The weights combine V vectors. These attention
weights are not the vocabulary logits used to choose the final output token.

The head outputs are merged and projected back to 896 dimensions.

## Feed-forward network

The layer then transforms each position's representation independently using
a gated feed-forward network:

```text
gate = SiLU(gate_projection(x))
expanded = up_projection(x)
update = down_projection(gate * expanded)
```

The expanded size is 4,864. The down projection returns to 896 dimensions.
The learned gate controls the update; it is not a user quota or scheduler gate.

## Final logits

After all layers, final normalization is followed by a projection into the
vocabulary. This checkpoint shares the input embedding and output-head weights.
For each position we get 151,936 scores. Only the last position's scores are
needed to select the next token during generation.

## Generation without a cache

`generate` calls our `forward`, chooses the highest final-position logit, appends
the token, and repeats. It stops at an EOS ID or the output limit.

Every iteration recomputes the growing sequence. This is intentionally slower
than cached decoding, but keeps the first correctness milestone straightforward.
Queries, keys, and values are computed inside attention; they are not yet retained
between calls. Our next milestone can add a separate KV cache to every layer.

## How we check correctness

The small-model tests compare different head groupings and tied/untied output
weights against Transformers. They also check that adding future tokens cannot
change earlier-position logits, and verify EOS and length stopping.

The real-checkpoint validator compares every logit at every prompt position for
three prompts, plus greedy output tokens. Both implementations use identical
read-only weight tensors, precision, and eager attention. Sharing weights avoids
loading two copies; our forward calculations remain separate.

Validator version 2 checks causality by changing only future token IDs while
keeping the tensor shapes fixed. Earlier-position logits must stay close. It
also compares a short-prefix run directly to the reference. A separate diagnostic
compares short and long runs for both implementations; floating-point kernels
can yield different rounding when tensor sizes change. That diagnostic alone is
not a failure, and it does not replace the fixed-shape causality check.

The regression tests include an intentionally leaky forward function, which the
validator must reject, and a synthetic shape-dependent offset, which is recorded
separately from future-token dependence. Numerical tolerances remain unchanged.

Passing these checks validates the exercised configurations. It does not prove
all inputs work or that generated answers are factually accurate. CPU and MPS
must be checked separately because their kernels and precision differ.

## Scope

One unpadded sequence, up to 512 prompt tokens, up to 256 generated tokens, and
at most 768 positions. No batching, KV cache, paged allocation, sampling, HTTP
API, or training. The fixed bounds keep explicit attention and logits tensors
manageable on the learning machine. This is an educational engine, not a
production serving replacement.

Implementation reference: the installed Transformers 4.57.6 Qwen2 model code.
The project dependency lock and model revision identify the comparison baseline.
