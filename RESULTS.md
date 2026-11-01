# First run results

Validated on CPU because this execution process reports MPS unavailable even though PyTorch was built with MPS support. GPU performance has not been measured.

- Model: Qwen/Qwen2.5-0.5B-Instruct
- Revision: 7ae557604adf67be50417f59c2c2f167def9a775
- Device: cpu, torch.float32
- Input tokens: 38
- Generated tokens: 64
- Cached loop: 2.399 seconds, 26.68 tokens/s
- Uncached loop: 8.447 seconds, 7.58 tokens/s
- Process peak RSS: 2.84 GiB (not peak GPU memory)
- Cached manual loop matches library reference: True
- Cached and uncached loops match: True

This single warm run is an educational measurement. It does not establish M3 GPU performance or a reliable general cache speedup. The default output hits the 64-token cap and is incomplete.

## Model quality observation

The tiny model's response to the batch inference question incorrectly conflates inference with training. Successful token parity validates implementation consistency, not factual accuracy. Batch inference applies a trained model to a dataset asynchronously; it does not train the model. Use the saved response as a numerical reference, not as learning material about batch inference.

## Next experiment

Run `.venv/bin/python run.py --device mps --offline --compare-no-cache --output results/mps-reference.json` from this folder in Terminal. This explicitly requires MPS and will report an error if it remains unavailable. Preserve the CPU reference because different devices and precision can produce different token choices.
