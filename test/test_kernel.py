import os
import sys

import torch

# for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from serve.kernels.cache_write import (
    write_to_cache_torch_decode_paged,
    write_to_cache_torch_prefill_paged,
    write_single_position_triton,
    write_multiple_positions_triton,
)
from serve.model import apply_rotary_emb, precompute_freqs_cis
from serve.kv_cache_paged import BlockTableData

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def create_mask(seqlens_k: torch.Tensor, max_len_input_ids: int, batch_size: int, device: str):
    max_seqlen_k = int(seqlens_k.max())
    mask = (
        torch.tril(
            torch.ones(max_len_input_ids, max_seqlen_k),
            diagonal=max_seqlen_k - max_len_input_ids,
        )
        .view(1, 1, max_len_input_ids, max_seqlen_k)
        .repeat(batch_size, 1, 1, 1)
        .to(device)
        .to(torch.bool)
    )
    for i, seqlen in enumerate(seqlens_k):
        mask[i, :, :, seqlen:] = 0
        mask[i, :, seqlen:, :] = 0
    return mask


# todo: pytest with different sizes + seeds
def test_write_to_cache_single_position():
    B = 4
    num_heads = 8
    N = 4  # number of blocks
    blocksize_k = 16
    D = 128

    torch.manual_seed(0)
    num_blocks = 2 * B * num_heads * N
    block_table = torch.randperm(num_blocks, dtype=torch.int32, device=DEVICE).reshape(1, 2, B, num_heads, N)
    block_table_index = torch.randperm(B, device=DEVICE)
    block_table_data = BlockTableData(block_table, block_table_index, blocksize_k)

    x_cache = torch.zeros(num_blocks, blocksize_k, D, device=DEVICE)
    x_cache_copy = x_cache.clone()
    x = torch.randn(B, num_heads, 1, D, device=DEVICE)
    seqlens_k = torch.randint(1, N * blocksize_k, (B,), device=DEVICE)

    write_to_cache_torch_decode_paged(x_cache, x, seqlens_k, block_table_data, 0, 0)
    write_single_position_triton(x_cache_copy, x, seqlens_k, block_table_data, 0, 0)

    torch.testing.assert_close(x_cache, x_cache_copy)


def test_write_to_cache_multiple_positions():
    B = 4
    num_heads = 4
    N = 4  # number of blocks
    blocksize_k = 64
    D = 128

    torch.manual_seed(1)
    num_blocks = 2 * B * num_heads * N
    block_table = torch.randperm(num_blocks, dtype=torch.int32, device=DEVICE).reshape(1, 2, B, num_heads, N)
    block_table_index = torch.randperm(B, device=DEVICE)
    block_table_data = BlockTableData(block_table, block_table_index, blocksize_k)

    x_cache = torch.zeros(num_blocks, blocksize_k, D, device=DEVICE)
    x_cache_copy = x_cache.clone()
    x = torch.randn(B, num_heads, N * blocksize_k, D, device=DEVICE)
    seqlens_k = torch.randint(2, N * blocksize_k, (B,), device=DEVICE)
    print(seqlens_k)
    write_to_cache_torch_prefill_paged(x_cache, x, seqlens_k, block_table_data, 0, 0)

    x_stacked = torch.cat([x[i, :, : seqlens_k[i]] for i in range(B)], dim=1).transpose(0, 1)
    write_multiple_positions_triton(x_cache_copy, x_stacked, seqlens_k, block_table_data, 0, 0)

    torch.testing.assert_close(x_cache, x_cache_copy)
    print("match!")


### testing rope for "flat" prefill (unpadded) vs padded


def test_rope():
    max_seqlen = 1024
    head_dim = 128
    freqs_cis = precompute_freqs_cis(max_seqlen, head_dim).to(DEVICE)

    b = 10
    s_max = 100
    h = 8
    d = head_dim
    seqlens_k = torch.randint(1, s_max, (b,), device=DEVICE)

    x_tensors = [torch.randn(h, seqlens_k[i], d, device=DEVICE) for i in range(b)]
    x_stacked = torch.cat(x_tensors, dim=1).transpose(0, 1)  # bs, h, d

    x_padded = torch.zeros(b, h, s_max, d, device=DEVICE)
    for i in range(b):
        x_padded[i, :, : seqlens_k[i], :] = x_tensors[i]

    # regular, padded rope:
    regular_positions = torch.zeros(b, s_max, dtype=torch.int)
    for i in range(b):
        regular_positions[i, : seqlens_k[i]] = torch.arange(seqlens_k[i], device=DEVICE)

    regular_freqs_cis = freqs_cis[regular_positions].unsqueeze(1)

    regular_rope_out = apply_rotary_emb(x_padded, regular_freqs_cis)

    # flat rope
    flat_positions = torch.cat([torch.arange(seqlens_k[i], device=DEVICE) for i in range(b)], dim=0)
    flat_freqs_cis = freqs_cis[flat_positions].unsqueeze(1)

    flat_rope_out = apply_rotary_emb(x_stacked, flat_freqs_cis)

    flat_rope_compare = torch.zeros_like(regular_rope_out)
    start_pos = torch.cumsum(seqlens_k, dim=0) - seqlens_k
    for i in range(b):
        flat_rope_compare[i, :, : seqlens_k[i], :] = flat_rope_out[start_pos[i] : start_pos[i] + seqlens_k[i], :, :].transpose(0, 1)
    # compare
    delta = (regular_rope_out - flat_rope_compare).abs().max()
    print(f"{delta=}")
    torch.testing.assert_close(regular_rope_out, flat_rope_compare, atol=1e-3, rtol=1e-4)
    print("rope agrees!")


# TODO: tests for prefill / decode kernels

if __name__ == "__main__":
    test_write_to_cache_single_position()
    test_write_to_cache_multiple_positions()
    test_rope()
