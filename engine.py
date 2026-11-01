"""A small Qwen2 forward pass written with PyTorch operations.

No Transformers model code is called here. We reuse pretrained weights and
PyTorch's tensor primitives. Scope: one unpadded sequence, full causal attention,
default RoPE, greedy generation, no KV cache and no training.
"""
import json
from pathlib import Path

import torch
import torch.nn.functional as F


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

    def position_factors(self, length):
        positions = torch.arange(length, device=self.device).float()
        # Float32 position calculations match the reference's default RoPE.
        angles = (self.inv_freq[None, :, None] @ positions[None, None, :]).transpose(1, 2)
        angles = torch.cat((angles, angles), dim=-1)
        return angles.cos().to(self.dtype).unsqueeze(1), angles.sin().to(self.dtype).unsqueeze(1)

    def attention(self, x, prefix, cos, sin, causal_mask):
        batch, length, _ = x.shape
        # [batch, positions, heads * head_dim] -> [batch, heads, positions, head_dim]
        q = self.linear(x, prefix + ".q_proj").view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        k = self.linear(x, prefix + ".k_proj").view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.linear(x, prefix + ".v_proj").view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        # Rotary position information is applied to queries and keys, not values.
        q = q * cos + self.rotate_half(q) * sin
        k = k * cos + self.rotate_half(k) * sin
        # Grouped-query attention: several query heads share one KV head.
        repeats = self.heads // self.kv_heads
        k = k[:, :, None].expand(batch, self.kv_heads, repeats, length, self.head_dim).reshape(batch, self.heads, length, self.head_dim)
        v = v[:, :, None].expand(batch, self.kv_heads, repeats, length, self.head_dim).reshape(batch, self.heads, length, self.head_dim)
        scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        scores = scores + causal_mask
        probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        mixed_values = probabilities @ v
        merged = mixed_values.transpose(1, 2).contiguous().reshape(batch, length, self.heads * self.head_dim)
        return self.linear(merged, prefix + ".o_proj")

    @torch.inference_mode()
    def forward(self, input_ids):
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("Expected one unpadded sequence shaped [1, length]")
        length = input_ids.shape[1]
        if not 1 <= length <= min(768, self.config["max_position_embeddings"]):
            raise ValueError("Sequence must contain 1 to 768 tokens within model context")
        if input_ids.dtype != torch.long or input_ids.device != self.device:
            raise ValueError("Token IDs must be int64 and on the model device")
        # Token IDs -> [1, length, hidden_size] learned embeddings.
        x = F.embedding(input_ids, self.weights["model.embed_tokens.weight"])
        cos, sin = self.position_factors(length)
        # At position i, attention may read positions <= i, never future tokens.
        future = torch.ones(length, length, dtype=torch.bool, device=self.device).triu(1)
        causal_mask = torch.zeros(length, length, dtype=x.dtype, device=self.device)
        causal_mask.masked_fill_(future, torch.finfo(x.dtype).min)
        for index in range(self.config["num_hidden_layers"]):
            name = f"model.layers.{index}"
            # Residual addition preserves the incoming representation.
            normalized = self.rms_norm(x, name + ".input_layernorm")
            x = x + self.attention(normalized, name + ".self_attn", cos, sin, causal_mask)
            normalized = self.rms_norm(x, name + ".post_attention_layernorm")
            # SwiGLU: activated gate times an expanded representation, projected back.
            gated = F.silu(self.linear(normalized, name + ".mlp.gate_proj"))
            expanded = self.linear(normalized, name + ".mlp.up_proj")
            x = x + self.linear(gated * expanded, name + ".mlp.down_proj")
        x = self.rms_norm(x, "model.norm")
        # This checkpoint ties its output head to the input embedding weights.
        if self.config.get("tie_word_embeddings", False):
            output_weight = self.weights["model.embed_tokens.weight"]
        else:
            output_weight = self.weights["lm_head.weight"]
        return F.linear(x, output_weight)  # [1, length, vocabulary_size]

    @torch.inference_mode()
    def generate(self, input_ids, max_new_tokens, eos_ids):
        if not 1 <= max_new_tokens <= 256:
            raise ValueError("max_new_tokens must be between 1 and 256")
        if input_ids.ndim != 2 or not 1 <= input_ids.shape[1] <= 512:
            raise ValueError("Prompt must contain 1 to 512 tokens")
        tokens = []
        for _ in range(max_new_tokens):
            logits = self.forward(input_ids)
            next_token = logits[:, -1, :].argmax(-1, keepdim=True)
            token_id = next_token.item()
            tokens.append(token_id)
            if token_id in eos_ids:
                break
            input_ids = torch.cat((input_ids, next_token), dim=1)
        return tokens
