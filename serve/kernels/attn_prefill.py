import triton  # type: ignore
import triton.language as tl  # type: ignore
import torch

from serve.kv_cache_paged import BlockTableData


@triton.jit
def attn_prefill_one_block_inner(
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
    BLOCK_Q_IX,
    head,
    batch,
    head_group_ix,
    block_table_index_ix,
    seqlen,
    block_table,
    stride_btb,
    stride_bth,
    stride_btn,
    BLOCK_SEQLEN_Q: tl.constexpr,
    BLOCK_SEQLEN_K: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    sm_scale: tl.constexpr,
    mask_block: tl.constexpr,
):
    offset_d = tl.arange(0, BLOCK_DMODEL)

    for ki in tl.range(lo, hi):
        kv_block_offset = block_table + block_table_index_ix * stride_btb + head_group_ix * stride_bth + ki * stride_btn
        kv_block_ix = tl.load(kv_block_offset).to(tl.int64)
        k_offset = kv_block_ix * stride_kb
        v_offset = kv_block_ix * stride_vb
        offset_ks = tl.arange(0, BLOCK_SEQLEN_K)
        k_ptrs = k_cache + k_offset + stride_ks * offset_ks[None, :] + stride_kd * offset_d[:, None]
        v_ptrs = v_cache + v_offset + stride_vs * offset_ks[:, None] + stride_vd * offset_d[None, :]

        k_mask = ki * BLOCK_SEQLEN_K + offset_ks[None, :] < seqlen
        k_block = tl.load(k_ptrs, mask=k_mask, other=0.0)
        qkt = tl.dot(q_block, k_block)

        if mask_block:
            causal_mask = BLOCK_Q_IX * BLOCK_SEQLEN_Q + tl.arange(0, BLOCK_SEQLEN_Q)[:, None] >= ki * BLOCK_SEQLEN_K + offset_ks[None, :]
            qkt = tl.where(causal_mask, qkt * sm_scale, -float("inf"))
            m_new = tl.maximum(m_block, tl.max(qkt, axis=1))
            p = tl.exp2(qkt - m_new[:, None])
        else:
            m_new = tl.maximum(m_block, sm_scale * tl.max(qkt, axis=1))
            p = tl.exp2(qkt * sm_scale - m_new[:, None])

        alpha = tl.exp2(m_block - m_new)
        m_block = m_new
        l_block = alpha * l_block + tl.sum(p, axis=1)
        o_block = alpha[:, None] * o_block

        v_mask = ki * BLOCK_SEQLEN_K + offset_ks[:, None] < seqlen
        v_block = tl.load(v_ptrs, mask=v_mask, other=0.0)
        p = p.to(v_block.dtype)
        o_block = tl.dot(p, v_block, o_block)

    return m_block, o_block, l_block


@triton.jit
def attn_prefill_one_block(
    q,
    k_cache,
    v_cache,
    o,
    l,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vs,
    stride_vd,
    stride_on,
    stride_oh,
    stride_od,
    stride_ln,
    stride_lh,
    BLOCK_Q_IX,
    head,
    batch,
    head_group_ix,
    block_table_index_ix,
    seqlens_k,
    stride_sk,
    q_start_ix,
    stride_qs_ix,
    block_table,
    stride_btb,
    stride_bth,
    stride_btn,
    BLOCK_SEQLEN_Q: tl.constexpr,
    BLOCK_SEQLEN_K: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    sm_scale: tl.constexpr,
):
    cur_seqlen = tl.load(seqlens_k + batch * stride_sk)
    cur_q_start_ix = tl.load(q_start_ix + batch * stride_qs_ix)
    qo_offset_n = cur_q_start_ix + BLOCK_Q_IX * BLOCK_SEQLEN_Q + tl.arange(0, BLOCK_SEQLEN_Q)

    offset_d = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = q + head * stride_qh + stride_qn * qo_offset_n[:, None] + stride_qd * offset_d[None, :]

    qo_mask = BLOCK_Q_IX * BLOCK_SEQLEN_Q + tl.arange(0, BLOCK_SEQLEN_Q)[:, None] < cur_seqlen
    q_block = tl.load(q_ptrs, mask=qo_mask, other=0.0)

    m_block = tl.full([BLOCK_SEQLEN_Q], -float("inf"), dtype=tl.float32)
    o_block = tl.zeros((BLOCK_SEQLEN_Q, BLOCK_DMODEL), dtype=tl.float32)
    l_block = tl.zeros((BLOCK_SEQLEN_Q,), dtype=tl.float32)

    q_start_pos = BLOCK_Q_IX * BLOCK_SEQLEN_Q
    q_end_pos = q_start_pos + BLOCK_SEQLEN_Q
    total_num_k_blocks = tl.cdiv(q_end_pos, BLOCK_SEQLEN_K)
    NUM_K_BLOCKS_UNMASKED = q_start_pos // BLOCK_SEQLEN_K

    # attn between q and blocks of k/v that don't need masking
    lo, hi = 0, NUM_K_BLOCKS_UNMASKED
    m_block, o_block, l_block = attn_prefill_one_block_inner(
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
        BLOCK_Q_IX,
        head,
        batch,
        head_group_ix,
        block_table_index_ix,
        cur_seqlen,
        block_table,
        stride_btb,
        stride_bth,
        stride_btn,
        BLOCK_SEQLEN_Q=BLOCK_SEQLEN_Q,
        BLOCK_SEQLEN_K=BLOCK_SEQLEN_K,
        BLOCK_DMODEL=BLOCK_DMODEL,
        sm_scale=sm_scale,
        mask_block=False,
    )

    #  masked attn between q and last k/v blocks
    lo, hi = NUM_K_BLOCKS_UNMASKED, total_num_k_blocks
    m_block, o_block, l_block = attn_prefill_one_block_inner(
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
        BLOCK_Q_IX,
        head,
        batch,
        head_group_ix,
        block_table_index_ix,
        cur_seqlen,
        block_table,
        stride_btb,
        stride_bth,
        stride_btn,
        BLOCK_SEQLEN_Q=BLOCK_SEQLEN_Q,
        BLOCK_SEQLEN_K=BLOCK_SEQLEN_K,
        BLOCK_DMODEL=BLOCK_DMODEL,
        sm_scale=sm_scale,
        mask_block=True,
    )

    # epilogue
    o_block = o_block / l_block[:, None]

    o_ptrs = o + head * stride_oh + stride_on * qo_offset_n[:, None] + stride_od * offset_d[None, :]
    tl.store(o_ptrs, o_block, mask=qo_mask)

    l_ptrs = l + head * stride_lh + stride_ln * qo_offset_n
    l_mask = BLOCK_Q_IX * BLOCK_SEQLEN_Q + tl.arange(0, BLOCK_SEQLEN_Q) < cur_seqlen
    tl.store(l_ptrs, l_block, mask=l_mask)


# optional autotuning
autotune_configs = [
    triton.Config({"BLOCK_SEQLEN_Q": BSQ}, num_stages=s, num_warps=w) for BSQ in [32, 64, 128] for s in ([1, 2, 3, 4]) for w in [4, 8]
]

heuristic_configs = [triton.Config({"BLOCK_SEQLEN_Q": 32}, num_warps=8, num_stages=3)]


@triton.autotune(autotune_configs, key=[])
@triton.jit
def attn_prefill_kernel(
    q,
    k_cache,
    v_cache,
    o,
    l,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vs,
    stride_vd,
    stride_on,
    stride_oh,
    stride_od,
    stride_ln,
    stride_lh,
    seqlens_k,
    stride_sk,
    q_start_ix,
    stride_qs_ix,
    block_table,
    stride_btb,
    stride_bth,
    stride_btn,
    block_table_index,
    stride_tb_batch,
    head_group_size: tl.constexpr,
    BLOCK_SEQLEN_Q: tl.constexpr,
    BLOCK_SEQLEN_K: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    sm_scale: tl.constexpr,
):
    BLOCK_Q_IX = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)

    head_group_ix = head // head_group_size

    block_table_index_ix = tl.load(block_table_index + batch * stride_tb_batch)

    attn_prefill_one_block(
        q,
        k_cache,
        v_cache,
        o,
        l,
        stride_qn,
        stride_qh,
        stride_qd,
        stride_kb,
        stride_ks,
        stride_kd,
        stride_vb,
        stride_vs,
        stride_vd,
        stride_on,
        stride_oh,
        stride_od,
        stride_ln,
        stride_lh,
        BLOCK_Q_IX,
        head,
        batch,
        head_group_ix,
        block_table_index_ix,
        seqlens_k,
        stride_sk,
        q_start_ix,
        stride_qs_ix,
        block_table,
        stride_btb,
        stride_bth,
        stride_btn,
        BLOCK_SEQLEN_Q=BLOCK_SEQLEN_Q,
        BLOCK_SEQLEN_K=BLOCK_SEQLEN_K,
        BLOCK_DMODEL=BLOCK_DMODEL,
        sm_scale=sm_scale,
    )


def attn_prefill_triton(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table_data: BlockTableData,
    seqlens_k: torch.Tensor,
    max_seqlen: int,
    model_config,
) -> torch.Tensor:
    B = seqlens_k.shape[0]
    N, Hq, D = q.shape
    Hq = model_config.n_heads
    Hk = model_config.n_kv_heads
    head_group_size = Hq // Hk
    BLOCK_DMODEL = D
    q_start_ix = torch.cumsum(seqlens_k, dim=0) - seqlens_k
    # todo: don't use as many blocks with more careful grid launch + indexing
    # num_blocks_per_batch = torch.ceil(seqlens_k / BLOCK_SEQLEN_Q)

    block_table = block_table_data.block_table
    block_table_index = block_table_data.block_table_index
    block_size = block_table_data.block_table_size

    log2e = 1.44269504089
    sm_scale = log2e * D ** (-0.5)

    grid = lambda meta: (triton.cdiv(int(max_seqlen), meta["BLOCK_SEQLEN_Q"]), Hq, B)  # noqa: E731

    o = torch.empty_like(q)
    l = torch.empty(N, Hq, dtype=q.dtype, device=q.device)
    attn_prefill_kernel[grid](
        q,
        k_cache,
        v_cache,
        o,
        l,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        l.stride(0),
        l.stride(1),
        seqlens_k,
        seqlens_k.stride(0),
        q_start_ix,
        q_start_ix.stride(0),
        block_table,
        block_table.stride(0),
        block_table.stride(1),
        block_table.stride(2),
        block_table_index,
        block_table_index.stride(0),
        head_group_size=head_group_size,
        BLOCK_SEQLEN_K=block_size,
        BLOCK_DMODEL=BLOCK_DMODEL,
        sm_scale=sm_scale,
    )

    return o
