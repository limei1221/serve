# llama
# Adapted from https://github.com/meta-llama/llama/blob/main/llama/model.py
import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F
from torch import distributed as dist

from serve.kernels.cache_write import write_multiple_positions_triton, write_single_position_triton
from serve.kernels.attn_decode import attn_decode_triton
from serve.kernels.attn_prefill import attn_prefill_triton
from serve.kv_cache_paged import BlockTableData, KVCacheIndexPaged




@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 8
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: float | None = None
    norm_eps: float = 1e-5
    rope_base: float = 10000
    max_batch_size: int = 32
    max_seq_len: int = 2048
    rope_scaling: dict | None = None


@dataclass
class Metadata:
    seqlens_k: torch.Tensor
    freqs_cis: torch.Tensor
    block_table_data: BlockTableData
    is_prefill: bool
    model_config: ModelArgs
    max_seqlen: int


class RotaryEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q: torch.Tensor, k: torch.Tensor, metadata: Metadata) -> tuple[torch.Tensor, torch.Tensor]:
        freqs_cis = metadata.freqs_cis
        q_out = apply_rotary_emb(q, freqs_cis)
        k_out = apply_rotary_emb(k, freqs_cis)
        return q_out, k_out


class LlamaRMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class LlamaMLP(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        dim = config.dim
        hidden_dim = int(2 * 4 * dim / 3)
        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)

        hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class LlamaAttention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.head_dim = config.dim // config.n_heads
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads

        self.wqkv = nn.Linear(config.dim, (self.n_heads + 2 * self.n_kv_heads) * self.head_dim, bias=False)
        self.rotary_emb = RotaryEmbedding()
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        self.n_local_heads = self.n_heads // self.world_size
        self.n_local_kv_heads = self.n_kv_heads // self.world_size

    def forward(self, x: torch.Tensor, kv_cache: tuple[torch.Tensor, torch.Tensor], metadata: Metadata):
        qkv = self.wqkv(x)

        kv_hd = self.head_dim * self.n_local_kv_heads
        q_hd = self.head_dim * self.n_local_heads
        q, k, v = qkv.split([q_hd, kv_hd, kv_hd], dim=-1)

        q = q.view(*x.shape[:-1], self.n_local_heads, self.head_dim)
        k = k.view(*x.shape[:-1], self.n_local_kv_heads, self.head_dim)
        v = v.view(*x.shape[:-1], self.n_local_kv_heads, self.head_dim)

        if not metadata.is_prefill:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
        q, k = self.rotary_emb(q, k, metadata)

        k_cache, v_cache = kv_cache
        write_to_cache_fn = write_multiple_positions_triton if metadata.is_prefill else write_single_position_triton
        with torch.autograd.profiler.record_function("write_to_cache"):
            write_to_cache_fn(k_cache, k, metadata.seqlens_k, metadata.block_table_data)
            write_to_cache_fn(v_cache, v, metadata.seqlens_k, metadata.block_table_data)

        attn_fn = attn_prefill_triton if metadata.is_prefill else attn_decode_triton
        with torch.autograd.profiler.record_function("attention"):
            attn_out = attn_fn(q, k_cache, v_cache, metadata.block_table_data, metadata.seqlens_k, metadata.max_seqlen, metadata.model_config)

        if not metadata.is_prefill:
            attn_out = attn_out.transpose(1, 2)
        attn_out = attn_out.contiguous().view(*x.shape[:-1], -1)
        output = self.wo(attn_out)
        return output


class LlamaDecoderBlock(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.mlp = LlamaMLP(config)
        self.attn = LlamaAttention(config)
        self.attention_norm = LlamaRMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = LlamaRMSNorm(config.dim, eps=config.norm_eps)

    def forward(self, x: torch.Tensor, kv_cache: tuple[torch.Tensor, torch.Tensor], metadata: Metadata):
        x = x + self.attn(self.attention_norm(x), kv_cache, metadata)
        x = x + self.mlp(self.ffn_norm(x))
        return x


class LlamaModel(nn.Module):
    def __init__(self, config: ModelArgs, dtype: torch.dtype):
        super().__init__()
        self.config = config
        self.norm = LlamaRMSNorm(config.dim, eps=config.norm_eps)
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([LlamaDecoderBlock(config) for _ in range(config.n_layers)])
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.dtype = dtype
        self.kv_caches_init = False

    def compute_freqs_cis(self, device: str):
        self.freqs_cis = precompute_freqs_cis(
            self.config.max_seq_len, self.config.dim // self.config.n_heads, base=self.config.rope_base, rope_scaling=self.config.rope_scaling
        ).to(device)

    def create_kv_cache(self, blocksize_k: int, max_batch_size: int, device: str):
        available_memory = torch.cuda.mem_get_info()[0]
        block_size_memory = 2 * self.config.n_layers * blocksize_k * (self.config.dim // self.config.n_heads) * self.dtype.itemsize
        # todo: improve this heuristic
        num_kv_blocks = int((available_memory * 0.6) // block_size_memory)
        print(f"KV cache using {block_size_memory * num_kv_blocks / 1024**3:.2f}GB")
        print(f"allocating {num_kv_blocks} blocks, of size {block_size_memory / 1024**2:.2f}MB")

        kv_caches = [
            (
                torch.empty(num_kv_blocks, blocksize_k, self.config.dim // self.config.n_heads, dtype=self.dtype, device=device),
                torch.empty(num_kv_blocks, blocksize_k, self.config.dim // self.config.n_heads, dtype=self.dtype, device=device),
            )
            for _ in range(self.config.n_layers)
        ]
        kv_cache_ix = KVCacheIndexPaged(
            num_kv_blocks, blocksize_k, self.config.n_kv_heads, self.config.n_layers, self.config.max_seq_len, max_batch_size, device
        )
        # todo: does this make things better?
        for layer in kv_caches:
            for _cache in layer:
                torch._dynamo.mark_dynamic(_cache, 0)

        self.kv_caches = kv_caches
        self.kv_cache_ix = kv_cache_ix
        self.num_kv_blocks = num_kv_blocks
        self.kv_caches_init = True

    def forward(
        self,
        tokens: torch.Tensor,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]],
        positions: torch.Tensor,
        seqlens_k: torch.Tensor,
        block_table_data: BlockTableData,
        max_seqlen: int,
        mode: Literal["decode", "prefill"],
    ) -> torch.Tensor:
        if self.freqs_cis is None:
            self.compute_freqs_cis()
        freqs_cis = self.freqs_cis[positions]
        freqs_cis = freqs_cis.unsqueeze(1)

        metadata = Metadata(
            seqlens_k=seqlens_k,
            freqs_cis=freqs_cis,
            block_table_data=block_table_data,
            is_prefill=(mode == "prefill"),
            model_config=self.config,
            max_seqlen=max_seqlen,
        )

        x = self.tok_embeddings(tokens)
        for layer, kv_cache in zip(self.layers, kv_caches):
            x = layer(x, kv_cache, metadata)
        x = self.norm(x)

        # todo: clean this up (will do with chunked prefill)
        if mode == "decode":
            return self.output(x)
        else:
            assert mode == "prefill", f"invalid mode: {mode}"
            pos_to_sample = torch.cumsum(seqlens_k, dim=-1) - 1
            return self.output(x[pos_to_sample]).view(seqlens_k.shape[0], -1)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )
    x_out2 = x_out2.flatten(-2)
    return x_out2.type_as(x)


def apply_rope_scaling(freqs: torch.Tensor, rope_scaling: dict) -> torch.Tensor:
    factor = rope_scaling["factor"]
    low_freq_factor = rope_scaling["low_freq_factor"]
    high_freq_factor = rope_scaling["high_freq_factor"]
    old_context_len = rope_scaling["original_max_position_embeddings"]

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            new_freqs.append((1 - smooth) * freq / factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def precompute_freqs_cis(
    seq_len: int,
    n_elem: int,
    base: int = 10000,
    dtype: torch.dtype = torch.float32,
    rope_scaling: dict | None = None,
) -> torch.Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    if rope_scaling is not None:
        freqs = apply_rope_scaling(freqs, rope_scaling)
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=dtype)
