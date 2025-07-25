# ruff: noqa: E471

import torch
import torch.nn.functional as F

import triton  # type: ignore
import triton.language as tl  # type: ignore

from serve.kv_cache_paged import BlockTableData


# very slow paged attention implementation in pure torch
def naive_attn_fn(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table_data: BlockTableData,
    layer_ix: int,
    seqlens_k: torch.Tensor,
    mask: torch.Tensor,
):
    block_table = block_table_data.block_table[layer_ix]
    block_table_index = block_table_data.block_table_index
    block_size = block_table_data.block_table_size

    b, hq, sq, d = q.shape
    hk = block_table.size(2)
    max_seqlen_k = int(torch.max(seqlens_k))
    # todo: torch.sdpa annoying if k/v are init as empty
    # since it gets mixed inf/nan and softmax goes wrong
    k = torch.zeros(b, hk, max_seqlen_k, d, dtype=q.dtype, device=q.device)
    v = torch.zeros(b, hk, max_seqlen_k, d, dtype=q.dtype, device=q.device)

    for b_ix in range(b):
        block_table_index_ix = block_table_index[b_ix].to(torch.int64)
        size_xb = int(seqlens_k[b_ix])
        num_blocks = (size_xb + block_size - 1) // block_size
        for h_ix in range(hk):
            for n_ix in range(num_blocks):
                block_ix_k = block_table[0, block_table_index_ix, h_ix, n_ix]
                block_ix_v = block_table[1, block_table_index_ix, h_ix, n_ix]
                upper_ix = min(size_xb, (n_ix + 1) * block_size)
                k[b_ix, h_ix, n_ix * block_size : upper_ix, :] = kv_cache[block_ix_k, : upper_ix - n_ix * block_size]
                v[b_ix, h_ix, n_ix * block_size : upper_ix, :] = kv_cache[block_ix_v, : upper_ix - n_ix * block_size]

    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
    return out


############################################################
#### decode kernel
############################################################
@triton.jit
def attn_decode_one_block_inner(
    q_block,
    k_cache,
    v_cache,
    stride_kb,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vs,
    stride_vd,
    m_block,
    o_block,
    l_block,
    lo,
    hi,
    batch,
    head_group,
    block_table_index_ix,
    current_seqlen_k,
    block_table,
    stride_btb,
    stride_bth,
    stride_btn,
    block_size: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    sm_scale: tl.constexpr,
    mask_block: tl.constexpr,
):
    offset_d = tl.arange(0, BLOCK_DMODEL)

    for ki in tl.range(lo, hi):
        kv_block_ix = tl.load(block_table + block_table_index_ix * stride_btb + head_group * stride_bth + ki * stride_btn)
        k_offset = kv_block_ix * stride_kb
        v_offset = kv_block_ix * stride_vb

        offset_kvs = tl.arange(0, block_size)
        k_ptrs = k_cache + k_offset + stride_ks * offset_kvs[:, None] + stride_kd * offset_d[None, :]
        v_ptrs = v_cache + v_offset + stride_vs * offset_kvs[:, None] + stride_vd * offset_d[None, :]
        if mask_block:
            s_mask = ki * block_size + offset_kvs[:, None] < current_seqlen_k
            k_block = tl.load(k_ptrs, mask=s_mask, other=0.0)
            v_block = tl.load(v_ptrs, mask=s_mask, other=0.0)
        else:
            k_block = tl.load(k_ptrs)
            v_block = tl.load(v_ptrs)

        qkt = tl.dot(q_block, tl.trans(k_block)) * sm_scale
        if mask_block:
            qkt = tl.where(ki * block_size + offset_kvs[None, :] < current_seqlen_k, qkt, -float("inf"))

        m_new = tl.maximum(tl.max(qkt, 1), m_block)
        p = tl.exp2(qkt - m_new[:, None])
        alpha = tl.exp2(m_block - m_new)
        m_block = m_new
        l_block = alpha * l_block + tl.sum(p, 1)
        o_block = alpha[:, None] * o_block

        p = p.to(v_block.dtype)
        o_block = tl.dot(p, v_block, o_block)

    return m_block, o_block, l_block


@triton.jit
def attn_decode_one_block(
    q,
    k_cache,
    v_cache,
    o,
    l,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vs,
    stride_vd,
    stride_oz,
    stride_ob,
    stride_oh,
    stride_od,
    stride_lz,
    stride_lb,
    stride_lh,
    split_ix,
    batch,
    head_group,
    block_table_index_ix,
    seqlens_k,
    stride_sk,
    block_table,
    stride_btb,
    stride_bth,
    stride_btn,
    block_size: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    sm_scale: tl.constexpr,
    num_splits: tl.constexpr,
    HEAD_GROUP_SIZE: tl.constexpr,
    PADDED_HEAD_GROUP_SIZE: tl.constexpr,
):
    q_offset = head_group * HEAD_GROUP_SIZE * stride_qh + batch * stride_qb
    offset_d = tl.arange(0, BLOCK_DMODEL)
    offset_h = tl.arange(0, PADDED_HEAD_GROUP_SIZE)
    q_ptrs = q + q_offset + stride_qd * offset_d[None, :] + stride_qh * offset_h[:, None]
    mask_h = offset_h[:, None] < HEAD_GROUP_SIZE
    q_block = tl.load(q_ptrs, mask=mask_h, other=0.0)

    m_block = tl.full([PADDED_HEAD_GROUP_SIZE], -float("inf"), dtype=tl.float32)
    o_block = tl.zeros([PADDED_HEAD_GROUP_SIZE, BLOCK_DMODEL], dtype=tl.float32)
    l_block = tl.zeros([PADDED_HEAD_GROUP_SIZE], dtype=tl.float32)

    current_seqlen_k = tl.load(seqlens_k + batch * stride_sk)
    NUM_K_BLOCKS = tl.cdiv(current_seqlen_k, block_size)
    NUM_K_BLOCKS_PER_SPLIT = tl.cdiv(NUM_K_BLOCKS, num_splits)

    lo = min(NUM_K_BLOCKS, split_ix * NUM_K_BLOCKS_PER_SPLIT)
    mid = min(NUM_K_BLOCKS, (split_ix + 1) * NUM_K_BLOCKS_PER_SPLIT - 1)
    hi = min(NUM_K_BLOCKS, (split_ix + 1) * NUM_K_BLOCKS_PER_SPLIT)
    # todo: can do the unrolling even more efficiently
    # we only need to unroll + mask if split_ix == num_splits - 1

    m_block, o_block, l_block = attn_decode_one_block_inner(
        q_block,
        k_cache,
        v_cache,
        stride_kb,
        stride_ks,
        stride_kd,
        stride_vb,
        stride_vs,
        stride_vd,
        m_block,
        o_block,
        l_block,
        lo,
        mid,
        batch,
        head_group,
        block_table_index_ix,
        current_seqlen_k,
        block_table,
        stride_btb,
        stride_bth,
        stride_btn,
        block_size,
        BLOCK_DMODEL=BLOCK_DMODEL,
        sm_scale=sm_scale,
        mask_block=False,
    )

    m_block, o_block, l_block = attn_decode_one_block_inner(
        q_block,
        k_cache,
        v_cache,
        stride_kb,
        stride_ks,
        stride_kd,
        stride_vb,
        stride_vs,
        stride_vd,
        m_block,
        o_block,
        l_block,
        mid,
        hi,
        batch,
        head_group,
        block_table_index_ix,
        current_seqlen_k,
        block_table,
        stride_btb,
        stride_bth,
        stride_btn,
        block_size,
        BLOCK_DMODEL=BLOCK_DMODEL,
        sm_scale=sm_scale,
        mask_block=True,
    )

    # epilogue
    o_block = o_block / l_block[:, None]

    offset_h = tl.arange(0, PADDED_HEAD_GROUP_SIZE)
    mask_oh = offset_h[:, None] < HEAD_GROUP_SIZE
    o_offset = head_group * HEAD_GROUP_SIZE * stride_oh + batch * stride_ob + split_ix * stride_oz
    o_ptrs = o + o_offset + stride_od * offset_d + stride_oh * offset_h[:, None]
    tl.store(o_ptrs, o_block, mask=mask_oh)

    l_offset = head_group * HEAD_GROUP_SIZE * stride_lh + batch * stride_lb + split_ix * stride_lz
    l_ptrs = l + l_offset + stride_lh * offset_h
    mask_lh = offset_h < HEAD_GROUP_SIZE
    tl.store(l_ptrs, l_block * tl.exp2(m_block), mask=mask_lh)


autotune_configs = [triton.Config({}, num_stages=s, num_warps=w) for s in [2] for w in [4]]


@triton.autotune(autotune_configs, key=[])
@triton.jit
def attn_decode_kernel(
    q,
    k_cache,
    v_cache,
    po,
    pl,
    stride_qb,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vs,
    stride_vd,
    stride_oz,
    stride_ob,
    stride_oh,
    stride_od,
    stride_lz,
    stride_lb,
    stride_lh,
    seqlens_k,
    stride_sk,
    block_table,
    stride_btb,
    stride_bth,
    stride_btn,
    block_table_index,
    stride_tb_batch,
    block_size: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    sm_scale: tl.constexpr,
    num_splits: tl.constexpr,
    HEAD_GROUP_SIZE: tl.constexpr,
    PADDED_HEAD_GROUP_SIZE: tl.constexpr,
):
    split_ix = tl.program_id(0)
    batch = tl.program_id(1)
    head_group = tl.program_id(2)

    block_table_index_ix = tl.load(block_table_index + batch * stride_tb_batch)

    attn_decode_one_block(
        q,
        k_cache,
        v_cache,
        po,
        pl,
        stride_qb,
        stride_qh,
        stride_qd,
        stride_kb,
        stride_ks,
        stride_kd,
        stride_vb,
        stride_vs,
        stride_vd,
        stride_oz,
        stride_ob,
        stride_oh,
        stride_od,
        stride_lz,
        stride_lb,
        stride_lh,
        split_ix,
        batch,
        head_group,
        block_table_index_ix,
        seqlens_k,
        stride_sk,
        block_table,
        stride_btb,
        stride_bth,
        stride_btn,
        block_size,
        BLOCK_DMODEL=BLOCK_DMODEL,
        sm_scale=sm_scale,
        num_splits=num_splits,
        HEAD_GROUP_SIZE=HEAD_GROUP_SIZE,
        PADDED_HEAD_GROUP_SIZE=PADDED_HEAD_GROUP_SIZE,
    )


# todo: implement in triton
# make more numerically stable
def merge_attn(po, pl, num_splits):
    if num_splits > 1:
        pl = pl.unsqueeze(-1)
        l = pl.sum(0)
        o = (po * pl).sum(0) / l
    else:
        o, l = po.squeeze(0), pl.squeeze(0)
    return o, l


def attn_decode_triton(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table_data: BlockTableData,
    seqlens_k: torch.Tensor,
    max_seqlen: int,
    model_config,
) -> torch.Tensor:
    # todo: fix num_splits > 1
    num_splits = 1

    B, Hq, SEQLEN_Q, D = q.shape
    assert SEQLEN_Q == 1
    q = q.squeeze(-2)
    HEAD_GROUP_SIZE = model_config.n_heads // model_config.n_kv_heads
    Hk = Hq // HEAD_GROUP_SIZE

    PADDED_HEAD_GROUP_SIZE = max(16, HEAD_GROUP_SIZE)  # need >= 16 to use tl.dot / tensor cores

    BLOCK_DMODEL = D
    log2e = 1.44269504089
    sm_scale = log2e * D ** (-0.5)

    grid = (num_splits, B, Hk)

    block_table = block_table_data.block_table
    block_table_index = block_table_data.block_table_index
    block_size = block_table_data.block_table_size

    po = torch.empty(num_splits, B, Hq, D, dtype=q.dtype, device=q.device)
    pl = torch.empty(num_splits, B, Hq, dtype=q.dtype, device=q.device)

    attn_decode_kernel[grid](
        q,
        k_cache,
        v_cache,
        po,
        pl,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        po.stride(0),
        po.stride(1),
        po.stride(2),
        po.stride(3),
        pl.stride(0),
        pl.stride(1),
        pl.stride(2),
        seqlens_k,
        seqlens_k.stride(0),
        block_table,
        block_table.stride(0),
        block_table.stride(1),
        block_table.stride(2),
        block_table_index,
        block_table_index.stride(0),
        block_size,
        BLOCK_DMODEL=BLOCK_DMODEL,
        sm_scale=sm_scale,
        num_splits=num_splits,
        HEAD_GROUP_SIZE=HEAD_GROUP_SIZE,
        PADDED_HEAD_GROUP_SIZE=PADDED_HEAD_GROUP_SIZE,
    )

    # reduce o using l
    o, l = merge_attn(po, pl, num_splits)
    return o.unsqueeze(-2)
