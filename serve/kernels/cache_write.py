import triton  # type: ignore
import triton.language as tl  # type: ignore
import torch

from serve.kv_cache_paged import BlockTableData

#####


def write_to_cache_torch_prefill_paged(
    x_cache: torch.Tensor, x: torch.Tensor, seqlens_k: torch.Tensor, block_table_data: BlockTableData, layer_ix: int, kv_ix: int
):
    b, h, _, d = x.shape
    block_table = block_table_data.block_table[layer_ix, kv_ix]
    block_table_index = block_table_data.block_table_index
    block_size = block_table_data.block_table_size

    for b_ix in range(b):
        block_table_index_ix = block_table_index[b_ix]
        size_xb = seqlens_k[b_ix]
        top_block_ix = size_xb // block_size
        for h_ix in range(h):
            for n_ix in range(top_block_ix):
                block_ix = block_table[block_table_index_ix, h_ix, n_ix]
                x_cache[block_ix, :, :] = x[b_ix, h_ix, n_ix * block_size : (n_ix + 1) * block_size]
            block_ix = block_table[block_table_index_ix, h_ix, top_block_ix]
            x_cache[block_ix, : (size_xb % block_size)] = x[
                b_ix, h_ix, top_block_ix * block_size : top_block_ix * block_size + (size_xb % block_size)
            ]


def write_to_cache_torch_decode_paged(
    x_cache: torch.Tensor, x: torch.Tensor, seqlens_k: torch.Tensor, block_table_data: BlockTableData, layer_ix: int, kv_ix: int
):
    b, h, _, d = x.shape
    block_table = block_table_data.block_table[layer_ix, kv_ix]
    block_table_index = block_table_data.block_table_index
    block_size = block_table_data.block_table_size

    for b_ix in range(b):
        block_table_index_ix = block_table_index[b_ix]
        size_xb = seqlens_k[b_ix]
        top_block_ix = (size_xb - 1) // block_size
        for h_ix in range(h):
            block_ix = block_table[block_table_index_ix, h_ix, top_block_ix]
            x_cache[block_ix, (size_xb - 1) % block_size] = x[b_ix, h_ix, 0]


@triton.jit
def write_single_position_kernel(
    cache,
    stride_cb,
    stride_cs,
    stride_cd,
    x,
    stride_xb,
    stride_xh,
    stride_xs,
    stride_xd,
    seqlens_k,
    stride_sb,
    block_table,
    stride_tb,
    stride_th,
    stride_tn,
    block_table_index,
    stride_tb_batch,
    block_size: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
):
    p_id = tl.program_id(0)
    b_ix = p_id // H
    block_table_index_ix = tl.load(block_table_index + b_ix * stride_tb_batch)
    h_ix = p_id % H
    seq_pos_ix = tl.load(seqlens_k + b_ix * stride_sb) - 1
    n_ix = seq_pos_ix // block_size
    block_ix = tl.load(block_table + block_table_index_ix * stride_tb + h_ix * stride_th + n_ix * stride_tn)

    x_offset = x + b_ix * stride_xb + h_ix * stride_xh
    x_ptrs = x_offset + tl.arange(0, D) * stride_xd
    x_val = tl.load(x_ptrs)
    cache_offset = cache + block_ix * stride_cb + (seq_pos_ix % block_size) * stride_cs
    cache_ptrs = cache_offset + tl.arange(0, D) * stride_cd
    tl.store(cache_ptrs, x_val)


def write_single_position_triton(x_cache: torch.Tensor, x: torch.Tensor, seqlens_k: torch.Tensor, block_table_data: BlockTableData):
    b, h, _, d = x.shape

    grid = (b * h,)

    block_table = block_table_data.block_table
    block_table_index = block_table_data.block_table_index
    block_size = block_table_data.block_table_size

    write_single_position_kernel[grid](
        x_cache,
        x_cache.stride(0),
        x_cache.stride(1),
        x_cache.stride(2),
        x,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        seqlens_k,
        seqlens_k.stride(0),
        block_table,
        block_table.stride(0),
        block_table.stride(1),
        block_table.stride(2),
        block_table_index,
        block_table_index.stride(0),
        block_size,
        ###
        H=h,
        D=d,
    )


@triton.jit
def write_multiple_positions_kernel(
    cache,
    stride_cb,
    stride_cs,
    stride_cd,
    x,
    stride_xn,
    stride_xh,
    stride_xd,
    seqlens_k,
    stride_sb,
    x_start_ix,
    x_start_ix_stride,
    block_table,
    stride_tb,
    stride_th,
    stride_tn,
    block_table_index,
    stride_tb_batch,
    block_size: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
):
    p_id = tl.program_id(0)
    b_ix = p_id // H
    block_table_index_ix = tl.load(block_table_index + b_ix * stride_tb_batch)
    h_ix = p_id % H
    seq_pos_ix = tl.load(seqlens_k + b_ix * stride_sb)
    n_ix = (seq_pos_ix - 1) // block_size
    cur_start_ix = tl.load(x_start_ix + b_ix * x_start_ix_stride)

    for n_iter in range(n_ix + 1):
        block_ix = tl.load(block_table + block_table_index_ix * stride_tb + h_ix * stride_th + n_iter * stride_tn)

        x_offset = x + h_ix * stride_xh + (cur_start_ix + n_iter * block_size) * stride_xn
        x_ptrs = x_offset + (tl.arange(0, block_size) * stride_xn)[:, None] + (tl.arange(0, D) * stride_xd)[None, :]
        mask = (n_iter * block_size + tl.arange(0, block_size) < seq_pos_ix)[:, None]
        x_val = tl.load(x_ptrs, mask=mask, other=0.0)  # other is not strictly necessary if we mask properly on the write

        cache_offset = cache + block_ix * stride_cb
        cache_ptrs = cache_offset + tl.arange(0, block_size)[:, None] * stride_cs + (tl.arange(0, D) * stride_cd)[None, :]
        mask = (n_iter * block_size + tl.arange(0, block_size) < seq_pos_ix)[:, None]
        tl.store(cache_ptrs, x_val, mask=mask)


def write_multiple_positions_triton(x_cache: torch.Tensor, x: torch.Tensor, seqlens_k: torch.Tensor, block_table_data: BlockTableData):
    n, h, d = x.shape
    b = seqlens_k.shape[0]

    grid = (b * h,)

    block_table = block_table_data.block_table
    block_table_index = block_table_data.block_table_index
    block_size = block_table_data.block_table_size

    x_start_ix = torch.cumsum(seqlens_k, dim=0) - seqlens_k

    write_multiple_positions_kernel[grid](
        x_cache,
        x_cache.stride(0),
        x_cache.stride(1),
        x_cache.stride(2),
        x,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        seqlens_k,
        seqlens_k.stride(0),
        x_start_ix,
        x_start_ix.stride(0),
        block_table,
        block_table.stride(0),
        block_table.stride(1),
        block_table.stride(2),
        block_table_index,
        block_table_index.stride(0),
        block_size,
        ###
        H=h,
        D=d,
    )


def write_to_cache(
    x_cache: torch.Tensor,
    x: torch.Tensor,
    seqlens_k: torch.Tensor,
    block_table_data: BlockTableData,
    is_prefill: bool,
):
    if is_prefill:
        return write_multiple_positions_triton(x_cache, x, seqlens_k, block_table_data)
    return write_single_position_triton(x_cache, x, seqlens_k, block_table_data)
