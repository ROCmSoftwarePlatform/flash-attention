import torch
import triton
import triton.language as tl
from .utils import DEBUG, DROPOUT_USE_PYTORCH, DROPOUT_DUMP, get_shape_from_layout, get_strides_from_layout, write_dropout_mask, create_dropout_mask

# NOTE: triton fails to import tl.constexprs so create them here for the file
tl_DROPOUT_USE_PYTORCH: tl.constexpr = DROPOUT_USE_PYTORCH
tl_DROPOUT_DUMP: tl.constexpr = DROPOUT_DUMP

# This function computes delta given output Out and gradient DO
# Here is the I/O shape:
# Out: (batch, nhead_q, max_seqlens_q, headDim)
# DO: (batch, nhead_q, max_seqlens_q, headDim)
# Delta: (batch, nheads_q, max_seqlens_q), same as softmax_lse defined at fwd_prefill.py line 607
@triton.jit
def _bwd_preprocess(
    Out,
    DO,
    Delta,
    stride_ob, stride_oh, stride_om, stride_ok,
    stride_dob, stride_doh, stride_dom, stride_dok,
    stride_deltab, stride_deltah, stride_deltam,
    cu_seqlens_q,
    max_seqlen_q,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    ACTUAL_BLOCK_DMODEL: tl.constexpr,
    N_CTX_Q: tl.constexpr,
    H: tl.constexpr,
    IS_VARLEN: tl.constexpr
):
    # Compute batch and head indices
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    off_b = pid_bh // H
    off_h = pid_bh % H

    if IS_VARLEN:
        # Compute actual sequence length
        q_start = tl.load(cu_seqlens_q + off_b)
        q_end = tl.load(cu_seqlens_q + off_b + 1)
        N_CTX_Q = q_end - q_start
    else:
        q_start = 0
        N_CTX_Q = max_seqlen_q

    # compute offsets
    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, BLOCK_DMODEL)
    o_offset = Out + off_b * stride_ob + off_h * stride_oh + q_start * stride_om
    do_offset = DO + off_b * stride_ob + off_h * stride_oh + q_start * stride_om
    # create masks
    mask_m = off_m < N_CTX_Q
    mask_d = off_d < ACTUAL_BLOCK_DMODEL
    mask_md = mask_m[:, None] & mask_d[None, :]
    # compute pointers
    out_ptrs = o_offset + off_m[:, None] * stride_om + off_d[None, :] * stride_ok
    do_ptrs = do_offset + off_m[:, None] * stride_dom + off_d[None, :] * stride_dok
    # load
    # TODO: tutorial only has do converted to f32 but not o, check if it is the same
    o = tl.load(out_ptrs, mask=mask_md, other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=mask_md, other=0.0).to(tl.float32)
    # compute and write-back to delta
    delta = tl.sum(o * do, axis=1)
    delta_offset = Delta + off_b * stride_deltab + off_h * stride_deltah + q_start * stride_deltam
    delta_ptrs = delta_offset + off_m * stride_deltam
    tl.store(delta_ptrs, delta, mask=mask_m)


@triton.jit
def _bwd_kernel(
    Q, K, V, sm_scale, Out, DO, DQ, DK, DV,
    L, Delta,
    dropout_mask,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_deltab, stride_deltah, stride_deltam,
    stride_dropoutb, stride_dropouth, stride_dropoutm, stride_dropoutn,
    B, HQ, HK,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    dropout_mask, dropout_p, philox_seed, philox_offset_base,
    BLOCK_M1: tl.constexpr,
    BLOCK_N1: tl.constexpr,
    BLOCK_M2: tl.constexpr,
    BLOCK_N2: tl.constexpr,
    BLK_SLICE_FACTOR: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    ACTUAL_BLOCK_DMODEL: tl.constexpr,
    CAUSAL: tl.constexpr,
    DROPOUT: tl.constexpr,
    USE_EXP2: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    # program ids
    off_bh = tl.program_id(0)
    off_b = off_bh // HQ
    off_hq = off_bh % HQ
    off_hk = off_hq

    if IS_VARLEN:
        # Compute actual sequence lengths
        q_start = tl.load(cu_seqlens_q + off_b)
        q_end = tl.load(cu_seqlens_q + off_b + 1)
        k_start = tl.load(cu_seqlens_k + off_b)
        k_end = tl.load(cu_seqlens_k + off_b + 1)
        N_CTX_Q = q_end - q_start
        N_CTX_K = k_end - k_start
    else:
        q_start = 0
        k_start = 0
        N_CTX_Q = max_seqlen_q
        N_CTX_K = max_seqlen_k

    # input tensor offsets
    q_offset = Q + off_b * stride_qb + off_hq * stride_qh + q_start * stride_qm
    k_offset = K + off_b * stride_kb + off_hk * stride_kh + k_start * stride_kn
    v_offset = V + off_b * stride_vb + off_hk * stride_vh + k_start * stride_vn
    do_offset = DO + off_b * stride_qb + off_hq * stride_qh + q_start * stride_qm
    l_offset = L + off_b * stride_deltab + off_hq * stride_deltah + q_start * stride_deltam
    delta_offset = Delta + off_b * stride_deltab + off_hq * stride_deltah + q_start * stride_deltam

    if DROPOUT:
        batch_philox_offset = philox_offset_base + off_b * stride_dropoutb + off_hq * stride_dropouth #+ q_start * stride_dropoutm
        dropout_offset = dropout_mask + off_b * stride_dropoutb + off_hq * stride_dropouth #+ q_start * stride_dropoutm
    else:
        batch_philox_offset = 0
        dropout_offset = 0


    # output tensor offsets
    dq_offset = DQ + off_b * stride_qb + off_hq * stride_qh + q_start * stride_qm
    dk_offset = DK + off_b * stride_kb + off_hk * stride_kh + k_start * stride_kn
    dv_offset = DV + off_b * stride_vb + off_hk * stride_vh + k_start * stride_vn


# NOTE: smaller blocks have lower accuracy. more accumlation error probably 128 * 128 seems good but leads to oom. 64 * 64 has accumlation errors but no oom.
def attention_prefill_backward_triton_split_impl(
    do,
    q,
    k,
    v,
    o,
    softmax_lse,
    dq,
    dk,
    dv,
    sm_scale: float,
    alibi_slopes,
    causal,
    layout: str,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p,
    philox_seed,
    philox_offset,
    use_exp2: bool,
    sequence_parallel = True,
):
    # make contigious
    # TODO: why making this continguous????
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    softmax_lse = softmax_lse.contiguous()
    do = do.contiguous()

    # get strides and shape
    batch, nheads_q, nheads_k, head_size, max_seqlen_q, max_seqlen_k = \
        get_shape_from_layout(q, k, layout,
                              cu_seqlens_q, cu_seqlens_k,
                              max_seqlen_q, max_seqlen_k)
    q_strides, k_strides, v_strides, o_strides = \
        get_strides_from_layout(q, k, v, o, layout)
    stride_qb, stride_qh, stride_qm, stride_qk =  q_strides
    stride_kb, stride_kh, stride_kn, stride_kk = k_strides
    stride_vb, stride_vh, stride_vn, stride_vk = v_strides
    stride_ob, stride_oh, stride_om, stride_ok = o_strides
    IS_VARLEN = layout == "thd"
    use_dropout = (dropout_p > 0.0)

    # get closest power of 2 over or equal to 32.
    # TODO: is this 32 or 16? why are we padding to power of 2 (instead of being power of 2 directly)
    padded_d_model = 1 << (head_size - 1).bit_length()
    padded_d_model = max(padded_d_model, 16)
    BLOCK_DMODEL = padded_d_model
    ACTUAL_BLOCK_DMODEL = head_size
    # meta-parameters
    # TODO: fix num_stages later
    NUM_WARPS, NUM_STAGES = 4, 1
    WAVES_PER_EU = 1
    PRE_BLOCK = 128
    BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 128, 128, 32
    BLK_SLICE_FACTOR = 2

    # init delta
    delta = torch.empty_like(softmax_lse)
    if IS_VARLEN:
        stride_deltab = 0
        stride_deltam, stride_deltah = delta.stride()
    else:
        stride_deltab, stride_deltah, stride_deltam = delta.stride()
    pre_grid = (triton.cdiv(max_seqlen_q, PRE_BLOCK), batch * nheads_q)
    _bwd_preprocess[pre_grid](
        o,
        do,
        delta,
        stride_ob, stride_oh, stride_om, stride_ok,
        stride_ob, stride_oh, stride_om, stride_ok,
        stride_deltab, stride_deltah, stride_deltam,
        cu_seqlens_q,
        max_seqlen_q,
        BLOCK_M=PRE_BLOCK,
        BLOCK_DMODEL=BLOCK_DMODEL,
        ACTUAL_BLOCK_DMODEL=ACTUAL_BLOCK_DMODEL,
        N_CTX_Q=max_seqlen_q,
        H=nheads_q,
        IS_VARLEN=IS_VARLEN
    )

    # dropout mask tensor for debugging. We dump the dropout mask created in the kernel for testing
    if use_dropout:
        if DROPOUT_USE_PYTORCH:
            dropout_mask = create_dropout_mask(dropout_p, (batch, nheads_q, max_seqlen_q, max_seqlen_k), seed = philox_seed)
        else:
            dropout_mask = torch.zeros((batch, nheads_q, max_seqlen_q, max_seqlen_k), device=q.device,
                                        dtype=torch.float32)
        stride_dropoutb, stride_dropouth, stride_dropoutm, stride_dropoutn = (dropout_mask.stride(0), dropout_mask.stride(1), dropout_mask.stride(2), dropout_mask.stride(3))
    else:
        dropout_mask = None
        stride_dropoutb, stride_dropouth, stride_dropoutm, stride_dropoutn = (0, 0 , 0 , 0)

    # TODO: why the 2nd dim is 1? necessary???
    grid = (max_seqlen_k // BLOCK_N1, 1, batch * nheads_q)
    _bwd_kernel[grid](
        q, k, v, sm_scale, o, do, dq, dk, dv,
        softmax_lse, delta,
        stride_qb, stride_qh, stride_qm, stride_qk,
        stride_kb, stride_kh, stride_kn, stride_kk,
        stride_vb, stride_vh, stride_vn, stride_vk,
        stride_deltab, stride_deltah, stride_deltam,
        stride_dropoutb, stride_dropouth, stride_dropoutm, stride_dropoutn,
        batch, nheads_q, nheads_k,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        dropout_mask, dropout_p, philox_seed, philox_offset,
        BLOCK_M1=BLOCK_M1, BLOCK_N1=BLOCK_N1,
        BLOCK_M2=BLOCK_M2, BLOCK_N2=BLOCK_N2,
        BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,
        BLOCK_DMODEL=BLOCK_DMODEL,
        ACTUAL_BLOCK_DMODEL=ACTUAL_BLOCK_DMODEL,
        CAUSAL=causal,
        DROPOUT=use_dropout,
        USE_EXP2=use_exp2,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
        waves_per_eu = WAVES_PER_EU,
        IS_VARLEN=IS_VARLEN,
    )

    return dq, dk, dv, delta, None, None
