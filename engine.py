"""A small Qwen2 forward pass written with PyTorch operations.

No Transformers model code is called here. We reuse pretrained weights and
PyTorch's tensor primitives. Scope: single sequences, static and continuous batching, full causal attention,
default RoPE, greedy generation, optional KV cache and no training.
"""
from dataclasses import dataclass, field
import json
from pathlib import Path

import torch
import torch.nn.functional as F


@dataclass
class _Request:
    prompt: torch.Tensor
    output: list[int] = field(default_factory=list)
    past: tuple = ()
    pending: int = 0
    admitted_at: int = -1
    finished_at: int = -1


class QwenEngine:
    def __init__(self, config, weights):
        self.config = config
        if config.get("model_type") != "qwen2":
            raise ValueError("Only the Qwen2 architecture is implemented")
        if config.get("use_sliding_window", False) or config.get("rope_scaling"):
            raise ValueError("Sliding attention and scaled RoPE are not implemented")
        if config.get("hidden_act", "silu") != "silu":
            raise ValueError("Only SiLU gated feed-forward layers are implemented")
        self.weights = weights
        self.hidden = config["hidden_size"]
        self.heads = config["num_attention_heads"]
        self.kv_heads = config["num_key_value_heads"]
        self.head_dim = config.get("head_dim", self.hidden // self.heads)
        if self.heads % self.kv_heads or self.head_dim % 2:
            raise ValueError("Heads must group evenly and head dimension must be even")
        self.eps = config["rms_norm_eps"]
        self.device = weights["model.embed_tokens.weight"].device
        self.dtype = weights["model.embed_tokens.weight"].dtype
        # Half as many frequencies as head dimensions; duplicated after the
        # position/frequency product to implement Qwen's rotate-half convention.
        exponents = torch.arange(0, self.head_dim, 2, device=self.device).float() / self.head_dim
        self.inv_freq = 1.0 / (config["rope_theta"] ** exponents)

    @classmethod
    def from_snapshot(cls, snapshot, device="cpu", dtype=torch.float32):
        """Load the already-downloaded, single-file safetensors checkpoint."""
        from safetensors.torch import load_file
        snapshot = Path(snapshot)
        config = json.loads((snapshot / "config.json").read_text())
        weights = {name: value.to(device=device, dtype=dtype)
                   for name, value in load_file(snapshot / "model.safetensors").items()}
        return cls(config, weights)

    def linear(self, x, name):
        # F.linear performs x @ weight.T + bias. Not every projection has bias.
        return F.linear(x, self.weights[name + ".weight"], self.weights.get(name + ".bias"))

    def rms_norm(self, x, name):
        # Accumulate squares in float32 even when model weights use float16.
        normalized = x.float()
        normalized = normalized * torch.rsqrt(normalized.square().mean(-1, keepdim=True) + self.eps)
        return self.weights[name + ".weight"] * normalized.to(x.dtype)

    @staticmethod
    def rotate_half(x):
        left, right = x.chunk(2, dim=-1)
        return torch.cat((-right, left), dim=-1)

    def position_factors(self, length, start=0):
        positions = torch.arange(start, start + length, device=self.device).float()
        return self.rotary_factors(positions[None, :])

    def rotary_factors(self, positions):
        # Float32 position calculations match the reference's default RoPE.
        angles = (self.inv_freq[None, :, None] @ positions[:, None, :].float()).transpose(1, 2)
        angles = torch.cat((angles, angles), dim=-1)
        return angles.cos().to(self.dtype).unsqueeze(1), angles.sin().to(self.dtype).unsqueeze(1)

    def attention(self, x, prefix, cos, sin, causal_mask, past=None):
        batch, length, _ = x.shape
        # [batch, positions, heads * head_dim] -> [batch, heads, positions, head_dim]
        q = self.linear(x, prefix + ".q_proj").view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        k = self.linear(x, prefix + ".k_proj").view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.linear(x, prefix + ".v_proj").view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        # Rotary position information is applied to queries and keys, not values.
        q = q * cos + self.rotate_half(q) * sin
        k = k * cos + self.rotate_half(k) * sin
        if past is not None:
            k = torch.cat((past[0], k), dim=2)
            v = torch.cat((past[1], v), dim=2)
        # Keep the compact two-head representation in the cache.
        present = (k, v)
        key_length = k.shape[2]
        # Grouped-query attention: several query heads share one KV head.
        repeats = self.heads // self.kv_heads
        k = k[:, :, None].expand(batch, self.kv_heads, repeats, key_length, self.head_dim).reshape(batch, self.heads, key_length, self.head_dim)
        v = v[:, :, None].expand(batch, self.kv_heads, repeats, key_length, self.head_dim).reshape(batch, self.heads, key_length, self.head_dim)
        scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        scores = scores + causal_mask
        probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        mixed_values = probabilities @ v
        merged = mixed_values.transpose(1, 2).contiguous().reshape(batch, length, self.heads * self.head_dim)
        return self.linear(merged, prefix + ".o_proj"), present

    @torch.inference_mode()
    def forward(self, input_ids):
        logits, _ = self._forward(input_ids, None, False)
        return logits

    @torch.inference_mode()
    def forward_cached(self, input_ids, past=None):
        """Process new tokens and return logits plus an updated per-layer cache."""
        return self._forward(input_ids, past, True)

    def _forward(self, input_ids, past, use_cache):
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("Expected one unpadded sequence shaped [1, length]")
        length = input_ids.shape[1]
        layer_count = self.config["num_hidden_layers"]
        start = 0
        if past is not None:
            if len(past) != layer_count:
                raise ValueError("Expected one cache pair per layer")
            start = past[0][0].shape[2]
        if length < 1 or start + length > min(768, self.config["max_position_embeddings"]):
            raise ValueError("Sequence must contain 1 to 768 tokens within model context")
        if input_ids.dtype != torch.long or input_ids.device != self.device:
            raise ValueError("Token IDs must be int64 and on the model device")
        positions = torch.arange(start + length, device=self.device)
        future = positions[None, :] > positions[start:, None]
        return self._layers(input_ids, self.position_factors(length, start),
                            future, past, use_cache)

    def _layers(self, input_ids, rotary, blocked, past, use_cache, last=None):
        layer_count = self.config["num_hidden_layers"]
        # Token IDs -> [1, length, hidden_size] learned embeddings.
        x = F.embedding(input_ids, self.weights["model.embed_tokens.weight"])
        cos, sin = rotary
        causal_mask = torch.zeros_like(blocked, dtype=x.dtype)
        causal_mask.masked_fill_(blocked, torch.finfo(x.dtype).min)
        cache = []
        for index in range(layer_count):
            name = f"model.layers.{index}"
            # Residual addition preserves the incoming representation.
            normalized = self.rms_norm(x, name + ".input_layernorm")
            update, present = self.attention(
                normalized, name + ".self_attn", cos, sin, causal_mask,
                past[index] if past is not None else None,
            )
            if use_cache:
                cache.append(present)
            x = x + update
            normalized = self.rms_norm(x, name + ".post_attention_layernorm")
            # SwiGLU: activated gate times an expanded representation, projected back.
            gated = F.silu(self.linear(normalized, name + ".mlp.gate_proj"))
            expanded = self.linear(normalized, name + ".mlp.up_proj")
            x = x + self.linear(gated * expanded, name + ".mlp.down_proj")
        if last is not None:
            x = x[torch.arange(x.shape[0], device=self.device), last].unsqueeze(1)
        x = self.rms_norm(x, "model.norm")
        # This checkpoint ties its output head to the input embedding weights.
        if self.config.get("tie_word_embeddings", False):
            output_weight = self.weights["model.embed_tokens.weight"]
        else:
            output_weight = self.weights["lm_head.weight"]
        return F.linear(x, output_weight), tuple(cache)  # logits: [1, length, vocabulary_size]

    @torch.inference_mode()
    def generate(self, input_ids, max_new_tokens, eos_ids, *, use_cache=False):
        if not 1 <= max_new_tokens <= 256:
            raise ValueError("max_new_tokens must be between 1 and 256")
        if input_ids.ndim != 2 or not 1 <= input_ids.shape[1] <= 512:
            raise ValueError("Prompt must contain 1 to 512 tokens")
        tokens = []
        past = None
        for _ in range(max_new_tokens):
            if use_cache:
                logits, past = self.forward_cached(input_ids, past)
            else:
                logits = self.forward(input_ids)
            next_token = logits[:, -1, :].argmax(-1, keepdim=True)
            token_id = next_token.item()
            tokens.append(token_id)
            if token_id in eos_ids:
                break
            input_ids = next_token if use_cache else torch.cat((input_ids, next_token), dim=1)
        return tokens

    @torch.inference_mode()
    def generate_batch(self, prompts, max_new_tokens, eos_ids, *, pad_id):
        """Greedy decode a fixed batch, dropping rows as they finish."""
        if not 1 <= len(prompts) <= 8 or not 1 <= max_new_tokens <= 256:
            raise ValueError("Expected 1 to 8 prompts and 1 to 256 new tokens")
        self._validate_batch_prompts(prompts, max_new_tokens)
        logits, past, lengths, valid = self._prefill_batch(prompts, pad_id)
        outputs = [[] for _ in prompts]
        active = list(range(len(prompts)))
        for step in range(max_new_tokens):
            picks = logits[:, 0].argmax(-1)
            keep = []
            for slot, (request, token) in enumerate(zip(active, picks.tolist())):
                outputs[request].append(token)
                if token not in eos_ids:
                    keep.append(slot)
            if not keep or step + 1 == max_new_tokens:
                break
            if len(keep) != len(active):
                rows = torch.tensor(keep, device=self.device)
                active = [active[i] for i in keep]
                picks = picks.index_select(0, rows)
                lengths = lengths.index_select(0, rows)
                valid = valid.index_select(0, rows)
                past = tuple((k.index_select(0, rows), v.index_select(0, rows)) for k, v in past)
            # Physical KV slots include padding; RoPE uses each row's logical length.
            valid = torch.cat((valid, torch.ones_like(valid[:, :1])), dim=1)
            logits, past = self._layers(
                picks[:, None], self.rotary_factors((lengths + step)[:, None]),
                ~valid[:, None, None, :], past, True,
            )
        return outputs

    def _validate_batch_prompts(self, prompts, max_new_tokens):
        if any(p.ndim != 1 or not 1 <= p.numel() <= 512
               or p.dtype != torch.long or p.device != self.device for p in prompts):
            raise ValueError("Each prompt must be a 1-D tensor of 1 to 512 token IDs on the model device")
        if max(p.numel() for p in prompts) + max_new_tokens - 1 > min(768, self.config["max_position_embeddings"]):
            raise ValueError("Request exceeds the model context limit")

    def _prefill_batch(self, prompts, pad_id):
        width = max(p.numel() for p in prompts)
        lengths = torch.tensor([p.numel() for p in prompts], device=self.device)
        columns = torch.arange(width, device=self.device)
        ids = torch.full((len(prompts), width), pad_id, device=self.device, dtype=torch.long)
        for row, prompt in zip(ids, prompts):
            row[:prompt.numel()] = prompt
        valid = columns[None, :] < lengths[:, None]
        # Right padding makes prompt positions start at zero for every row.
        # Select logits from each row's last real prompt token.
        blocked = (columns[None, :] > columns[:, None])[None, None] | ~valid[:, None, None, :]
        logits, past = self._layers(ids, self.rotary_factors(columns[None, :]),
                                    blocked, None, True, last=lengths - 1)
        return logits, past, lengths, valid

    @torch.inference_mode()
    def generate_continuous(self, prompts, max_new_tokens, eos_ids, *, pad_id, max_active=2):
        """Serve a FIFO backlog, refilling active slots between decode rounds.

        Return output IDs and per-request scheduling rounds in input order.
        """
        if not prompts or not 1 <= max_active <= 8 or not 1 <= max_new_tokens <= 256:
            raise ValueError("Expected prompts, 1 to 8 active slots and 1 to 256 new tokens")
        self._validate_batch_prompts(prompts, max_new_tokens)
        requests = [_Request(prompt) for prompt in prompts]
        active = []
        next_request = 0
        round_number = 0
        while next_request < len(requests) or active:
            # Requests already decoding each get one decode step this round.
            # New requests get their first output token from prefill.
            decode = active
            end = min(len(requests), next_request + max_active - len(decode))
            admitted = requests[next_request:end]
            next_request = end
            if admitted:
                logits, past, _, _ = self._prefill_batch([r.prompt for r in admitted], pad_id)
                picks = logits[:, 0].argmax(-1).tolist()
                for row, (request, token) in enumerate(zip(admitted, picks)):
                    request.admitted_at = round_number
                    request.output.append(token)
                    request.pending = token
                    request.past = self._cache_row(past, row, request.prompt.numel())
                del past
            if decode:
                self._decode_requests(decode)
            active = []
            for request in decode + admitted:
                if request.pending in eos_ids or len(request.output) == max_new_tokens:
                    request.finished_at = round_number
                    request.past = ()
                else:
                    active.append(request)
            round_number += 1

        return [r.output for r in requests], [
            {"admitted_at": r.admitted_at, "finished_at": r.finished_at} for r in requests
        ]

    @staticmethod
    def _cache_row(past, row, length, appended_position=None):
        def extract(tensor):
            prefix = tensor[row:row + 1, :, :length]
            if appended_position is None:
                return prefix.clone()
            new_slot = tensor[row:row + 1, :, appended_position:appended_position + 1]
            return torch.cat((prefix, new_slot), dim=2)

        return tuple((extract(k), extract(v)) for k, v in past)

    def _decode_requests(self, requests):
        lengths = [r.past[0][0].shape[2] for r in requests]
        width = max(lengths)
        positions = torch.tensor(lengths, device=self.device)[:, None]
        columns = torch.arange(width + 1, device=self.device)
        # The final column is the new token for every row. Mask the gap
        # between that column and each request's compact cache.
        blocked = (columns >= positions) & (columns < width)

        def pack(tensors):
            return torch.cat([F.pad(tensor, (0, 0, 0, width - length))
                              for tensor, length in zip(tensors, lengths)], dim=0)

        past = tuple((pack(keys), pack(values))
                     for keys, values in (zip(*layer) for layer in zip(*(r.past for r in requests))))
        ids = torch.tensor([[r.pending] for r in requests], device=self.device)
        logits, past = self._layers(ids, self.rotary_factors(positions),
                                    blocked[:, None, None, :], past, True)
        picks = logits[:, 0].argmax(-1).tolist()
        for row, (request, token, length) in enumerate(zip(requests, picks, lengths)):
            request.output.append(token)
            request.pending = token
            request.past = self._cache_row(past, row, length, appended_position=width)
