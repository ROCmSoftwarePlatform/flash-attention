import torch
import triton
import triton.language as tl
from .utils import DEBUG, DROPOUT_USE_PYTORCH, DROPOUT_DUMP, \
    get_shape_from_layout, get_strides_from_layout, write_dropout_mask, \
    create_dropout_mask

# NOTE: triton fails to import tl.constexprs so create them here for the file
tl_DROPOUT_USE_PYTORCH: tl.constexpr = DROPOUT_USE_PYTORCH
tl_DROPOUT_DUMP: tl.constexpr = DROPOUT_DUMP

# This function computes delta given output Out and gradient DO
# Here is the I/O shape:
# Out: (batch, nhead_q, max_seqlens_q, headDim)
# DO: (batch, nhead_q, max_seqlens_q, headDim)
# Delta: (batch, nheads_q, max_seqlens_q), same as softmax_lse defined at
#   fwd_prefill.py line 607
@triton.jit
def _bwd_preprocess(
    O, DO,
    Delta,
    stride_ob, stride_oh, stride_om, stride_ok,
    stride_deltab, stride_deltah, stride_deltam,
    cu_seqlens_q, max_seqlen_q,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ACTUAL_HEAD_DIM: tl.constexpr,
    IS_VARLEN: tl.constexpr
):
    pid_m = tl.program_id(0)
    bid = tl.program_id(1)
    hid = tl.program_id(2)
    # Handle varlen
    q_start = 0
    seqlen_q = max_seqlen_q
    if IS_VARLEN:
        q_start = tl.load(cu_seqlens_q + bid)
        q_end = tl.load(cu_seqlens_q + bid + 1)
        seqlen_q = q_end - q_start
    # Compute offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, HEAD_DIM)
    # Offset O/DO by batch, head and q_start
    O += bid * stride_ob + hid * stride_oh + q_start * stride_om
    DO += bid * stride_ob + hid * stride_oh + q_start * stride_om
    # create masks
    mask_m = offs_m < seqlen_q
    mask_md = mask_m[:, None]
    PADDED_HEAD: tl.constexpr = (ACTUAL_HEAD_DIM != HEAD_DIM)
    if PADDED_HEAD:
        mask_md &= offs_k[None, :] < ACTUAL_HEAD_DIM
    # compute pointers
    offs_o = offs_m[:, None] * stride_om + offs_k[None, :] * stride_ok
    out_ptrs = O + offs_o
    do_ptrs = DO + offs_o
    # load
    o = tl.load(out_ptrs, mask=mask_md, other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=mask_md, other=0.0).to(tl.float32)
    # compute and write-back to delta
    delta = tl.sum(o * do, axis=1)
    delta_offset = Delta + bid * stride_deltab + hid * stride_deltah + q_start
    tl.store(delta_offset + offs_m, delta, mask=mask_m)


# The main inner-loop logic for computing dK and dV.
@triton.jit
def _bwd_dkdv_inner(
    dk, dv,  # output
    Q, k, v, DO, M, D, qk_scale,  # input tensor
    stride_qm, stride_qk,  # shared by Q/DO.
    stride_dropoutm, stride_dropoutn,  #
    BLOCK_M: tl.constexpr,  # 16
    BLOCK_N: tl.constexpr,  # 128
    HEAD_DIM: tl.constexpr,  #
    ACTUAL_HEAD_DIM: tl.constexpr,  #
    dropout_p, philox_seed, batch_philox_offset, dropout_offset,  #
    seqlen_q, seqlen_k,  # max sequence length for q and k
    # Filled in by the wrapper.
    start_n, start_m, num_steps,  # iteration numbers
    MASK: tl.constexpr,  # causal masking, only apply to tiles on mask diagonal
    ENABLE_DROPOUT: tl.constexpr  # activate dropout
):
    offs_m = start_m + tl.arange(0, BLOCK_M)  # start_m + (0, 15)
    offs_n = start_n + tl.arange(0, BLOCK_N)  # start_m + (0, 127)
    offs_k = tl.arange(0, HEAD_DIM)
    # mask to make sure not OOB of seqlen_q
    mask_m = offs_m < seqlen_q
    mask_n = offs_n < seqlen_k
    mask_qT = mask_m[None, :]
    mask_do = mask_m[:, None]
    PADDED_HEAD: tl.constexpr = (ACTUAL_HEAD_DIM != HEAD_DIM)
    # if HEAD_DIM is padded
    if PADDED_HEAD:
        mask_qT &= offs_k[:, None] < ACTUAL_HEAD_DIM
        mask_do &= offs_k[None, :] < ACTUAL_HEAD_DIM
    # Q and DO are (seqlen_q, head_dim)
    # qT_ptrs = (1, BLOCK_M) + (HEAD_DIM, 1), transpose of q
    qT_ptrs = Q + offs_m[None, :] * stride_qm + offs_k[:, None] * stride_qk
    # do_ptrs = (BLOCK_M, 1) + (1, HEAD_DIM), NOT transposed
    do_ptrs = DO + offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    # BLOCK_N must be a multiple of BLOCK_M, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N % BLOCK_M == 0)
    curr_m = start_m
    step_m = BLOCK_M
    curr_philox_offset = batch_philox_offset
    curr_dropout_offset = dropout_offset
    RCP_LN2: tl.constexpr = 1.4426950408889634  # = 1.0 / ln(2)
    for blk_idx in range(num_steps):
        qT = tl.load(qT_ptrs, mask=mask_qT, other=0.0)
        # generate dropout mask
        if ENABLE_DROPOUT:
            # NOTE: dropout is transposed because it is used to mask pT
            philox_offs = curr_philox_offset + \
                          offs_m[None, :] * stride_dropoutm + \
                          offs_n[:, None] * stride_dropoutn
            if tl_DROPOUT_USE_PYTORCH:
                dropout_offs = offs_m[None, :] * stride_dropoutm + \
                               offs_n[:, None] * stride_dropoutn
                dropout_mask = tl.load(
                    curr_dropout_offset + dropout_offs,
                    mask=mask_m[:, None] & mask_n[None, :]
                )
            else:
                rand_vals = tl.rand(philox_seed, philox_offs)
                dropout_mask = rand_vals > dropout_p
            dropout_scale = 1 / (1 - dropout_p)

        # Load m before computing qk to reduce pipeline stall.
        offs_m = curr_m + tl.arange(0, BLOCK_M)
        # update the mask because offs_m advanced
        mask_m = offs_m < seqlen_q
        m = tl.load(M + offs_m, mask=offs_m < seqlen_q)
        qkT = tl.dot(k, qT)
        # TODO: remove the scaling of m later when we removed re-scaling in fwd
        pT = tl.math.exp2(qkT * qk_scale - m[None, :] * RCP_LN2)
        # Autoregressive masking.
        if MASK:
            mask_nm = mask_n[:, None] & (offs_m[None, :] < seqlen_q)
            mask = (offs_m[None, :] >= offs_n[:, None]) & mask_nm
            pT = tl.where(mask, pT, 0.0)
        do = tl.load(do_ptrs, mask=mask_do, other=0.0)
        # Compute dV.
        if ENABLE_DROPOUT:
            pT = tl.where(dropout_mask, pT, 0.0) * dropout_scale
        pT = pT.to(tl.float16)
        dv += tl.dot(pT, do)
        # D (= delta) is pre-divided by ds_scale.
        Di = tl.load(D + offs_m, mask=offs_m < seqlen_q)
        # Compute dP and dS.
        dpT = tl.dot(v, tl.trans(do)).to(tl.float32)
        if ENABLE_DROPOUT:
            dpT = tl.where(dropout_mask, dpT, 0.0) * dropout_scale
        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(tl.float16)
        dk += tl.dot(dsT, tl.trans(qT))
        # Increment pointers.
        curr_m += step_m
        qT_ptrs += step_m * stride_qm
        do_ptrs += step_m * stride_qm
        if ENABLE_DROPOUT:
            curr_dropout_offset += step_m * stride_qm
            curr_philox_offset += step_m * stride_qm
        mask_qT = mask_m[None, :]
        mask_do = mask_m[:, None]
        if PADDED_HEAD:
            mask_qT &= offs_k[:, None] < ACTUAL_HEAD_DIM
            mask_do &= offs_k[None, :] < ACTUAL_HEAD_DIM
    return dk, dv


# grid = (max_seqlen_k // BLOCK_N, batch, nheads_q)
@triton.jit
def _bwd_kernel_dkdv(
    Q, K, V, sm_scale, Out, DO, DK, DV,
    M, Delta,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_deltab, stride_deltah, stride_deltam,
    stride_dropoutb, stride_dropouth, stride_dropoutm, stride_dropoutn,
    HQ, HK,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    dropout_mask, dropout_p, philox_seed, philox_offset_base,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLK_SLICE_FACTOR: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ACTUAL_HEAD_DIM: tl.constexpr,
    CAUSAL: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    # program ids
    pid = tl.program_id(0)
    bid = tl.program_id(1)
    hkid = tl.program_id(2)
    # figure out varlen start and end
    q_start = 0
    k_start = 0
    seqlen_q = max_seqlen_q
    seqlen_k = max_seqlen_k
    if IS_VARLEN:
        # Compute actual sequence lengths
        q_start = tl.load(cu_seqlens_q + bid)
        q_end = tl.load(cu_seqlens_q + bid + 1)
        k_start = tl.load(cu_seqlens_k + bid)
        k_end = tl.load(cu_seqlens_k + bid + 1)
        seqlen_q = q_end - q_start
        seqlen_k = k_end - k_start

    dk = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, HEAD_DIM], dtype=tl.float32)
    # Figure out causal starting block since we have seqlen_q >=< seqlen_k.
    # Unlike forward pass where we tile on M dim and iterate on N dim, so that
    # we can skip some M blocks, in backward pass, we tile on the N dim for kv
    # and iterate over the M. In this way, we cannot skip N blocks, but only to
    # determine the starting M blocks to skip some initial blocks masked by
    # causal.
    seqlen_delta = seqlen_q - seqlen_k
    start_delta = (seqlen_delta // BLOCK_M) * BLOCK_M
    start_n = pid * BLOCK_N
    # seqlen_q > seqlen_k: skip additional blocks at the beginning for every
    #   N-tile, i.e. add offset to start_m
    # seqlen_q < seqlen_k: some initial N-tiles will have nothing to skip,
    #   i.e. subtract offset to start_m and cap to 0. These tiles will not have
    #   the first _bwd_dkdv_inner() with MASK=True performed.

    offs_k = tl.arange(0, HEAD_DIM)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    # Mask for loading K and V
    mask_kv = offs_n[:, None] < seqlen_k
    PADDED_HEAD: tl.constexpr = (ACTUAL_HEAD_DIM != HEAD_DIM)
    if PADDED_HEAD:
        mask_k = offs_k < ACTUAL_HEAD_DIM
        mask_kv &= mask_k[None, :]
    offs_kv = offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk

    GROUP_SIZE = HQ // HK
    # K/V tensors not changed for the group
    adj_kv = bid * stride_kb + hkid * stride_kh + k_start * stride_kn
    # load K and V: they stay in SRAM throughout the inner loop.
    k = tl.load(K + adj_kv + offs_kv, mask=mask_kv, other=0.0)
    v = tl.load(V + adj_kv + offs_kv, mask=mask_kv, other=0.0)
    # If MQA / GQA, set the K and V head offsets appropriately.
    for hqid in range(hkid * GROUP_SIZE, hkid * GROUP_SIZE + GROUP_SIZE):
        start_m = start_n + start_delta
        # offset input and output tensor by batch and Q/K heads
        adj_q = bid * stride_qb + hqid * stride_qh + q_start * stride_qm
        Q_ptr = Q + adj_q
        DO_ptr = DO + adj_q
        adj_delta = bid * stride_deltab + hqid * stride_deltah + q_start
        M_ptr = M + adj_delta
        Delta_ptr = Delta + adj_delta

        # batch_philox_offset is the ACTUALLY dropout offset
        # dropout_offset is for debug purpose and will be removed later
        batch_philox_offset = 0
        dropout_offset = 0
        if ENABLE_DROPOUT:
            batch_philox_offset = philox_offset_base + bid * stride_dropoutb + \
                                  hqid * stride_dropouth
            dropout_offset = dropout_mask + bid * stride_dropoutb + \
                             hqid * stride_dropouth

        MASK_BLOCK_M: tl.constexpr = BLOCK_M // BLK_SLICE_FACTOR
        num_steps = BLOCK_N // MASK_BLOCK_M

        # scaling factor
        RCP_LN2: tl.constexpr = 1.4426950408889634  # = 1.0 / ln(2)
        qk_scale = sm_scale * RCP_LN2

        # if start_m is negative, the current N-tile has no block on the
        #   diagonal of causal mask, so everything have no causal mask
        if start_m >= 0:
            dk, dv = _bwd_dkdv_inner(
                dk, dv,  # output tensors
                Q_ptr, k, v, DO_ptr, M_ptr, Delta_ptr, qk_scale, # input tensors
                stride_qm, stride_qk,  # strides for q
                stride_dropoutm, stride_dropoutn,  # strides for dropout
                MASK_BLOCK_M, BLOCK_N,  # block dim
                HEAD_DIM, ACTUAL_HEAD_DIM,  # head dim
                dropout_p, philox_seed, batch_philox_offset, dropout_offset,  #
                seqlen_q, seqlen_k,  # max sequence length for q and k
                start_n, start_m, num_steps,  # iteration numbers
                MASK=True,  # causal masking
                ENABLE_DROPOUT=ENABLE_DROPOUT,  # activate dropout
            )
            start_m += num_steps * MASK_BLOCK_M
        else:
            start_m = 0

        num_steps = (seqlen_q - start_m) // BLOCK_M
        # only the blocks on the causal mask diagonal needs to mask
        dk, dv = _bwd_dkdv_inner(
            dk, dv,  # output tensors
            Q_ptr, k, v, DO_ptr, M_ptr, Delta_ptr, qk_scale, # input tensors
            stride_qm, stride_qk,  # strides for q
            stride_dropoutm, stride_dropoutn,  # strides for dropout
            BLOCK_M, BLOCK_N,  # block dim
            HEAD_DIM, ACTUAL_HEAD_DIM,  # head dim
            dropout_p, philox_seed, batch_philox_offset, dropout_offset,  #
            seqlen_q, seqlen_k,  # max sequence length for q and k
            start_n, start_m, num_steps,  # iteration numbers
            MASK=False,  # causal masking
            ENABLE_DROPOUT=ENABLE_DROPOUT,  # activate dropout
        )

    # Write back dV and dK.
    tl.store(DV + adj_kv + offs_kv, dv, mask=mask_kv)
    dk *= sm_scale
    tl.store(DK + adj_kv + offs_kv, dk, mask=mask_kv)


# the main inner-loop logic for computing dQ
@triton.jit
def _bwd_dq_inner(
    dq,  # output
    q, K, V, do, m, Delta, qk_scale, # input
    # shared by Q/K/V/DO.
    stride_qm, stride_qk, stride_kn,
    stride_dropoutm, stride_dropoutn,  # stride for dropout
    seqlen_q, seqlen_k,  #
    BLOCK_M2: tl.constexpr,  #
    BLOCK_N2: tl.constexpr,  #
    HEAD_DIM: tl.constexpr,
    ACTUAL_HEAD_DIM: tl.constexpr,  #
    dropout_p, philox_seed, batch_philox_offset, dropout_offset,  #
    # Filled in by the wrapper.
    start_m, start_n, num_steps,  #
    MASK: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr
):
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    offs_k = tl.arange(0, HEAD_DIM)

    # mask to make sure not OOB of seqlen_q
    mask_m = offs_m < seqlen_q
    mask_n = offs_n < seqlen_k
    mask_kT = mask_n[None, :]
    PADDED_HEAD: tl.constexpr = (ACTUAL_HEAD_DIM != HEAD_DIM)
    # if HEAD_DIM is padded
    if PADDED_HEAD:
        mask_kT &= offs_k[:, None] < ACTUAL_HEAD_DIM

    kT_ptrs = K + offs_n[None, :] * stride_kn + offs_k[:, None] * stride_qk
    vT_ptrs = V + offs_n[None, :] * stride_kn + offs_k[:, None] * stride_qk
    # D (= delta) is pre-divided by ds_scale.
    Di = tl.load(Delta + offs_m, mask=mask_m, other=0.0)
    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    curr_philox_offset = batch_philox_offset
    curr_dropout_offset = dropout_offset
    RCP_LN2: tl.constexpr = 1.4426950408889634  # = 1.0 / ln(2)
    for blk_idx in range(num_steps):
        kT = tl.load(kT_ptrs, mask=mask_kT, other=0.0)
        vT = tl.load(vT_ptrs, mask=mask_kT, other=0.0)

        if ENABLE_DROPOUT:
            # NOTE: dropout is transposed because it is used to mask pT
            philox_offs = curr_philox_offset + \
                          offs_m[None, :] * stride_dropoutm + \
                          offs_n[:, None] * stride_dropoutn
            if tl_DROPOUT_USE_PYTORCH:
                dropout_offs = offs_m[None, :] * stride_dropoutm + \
                               offs_n[:, None] * stride_dropoutn
                dropout_mask = tl.load(
                    curr_dropout_offset + dropout_offs,
                    mask=mask_m[:, None] & mask_n[None, :])
            else:
                rand_vals = tl.rand(philox_seed, philox_offs)
                dropout_mask = rand_vals > dropout_p
            dropout_scale = 1 / (1 - dropout_p)

        qk = tl.dot(q, kT)
        p = tl.math.exp2(qk * qk_scale - m * RCP_LN2)
        # Autoregressive masking.
        if MASK:
            offs_n = curr_n + tl.arange(0, BLOCK_N2)
            mask_mn = mask_m[:, None] & (offs_n[None, :] < seqlen_k)
            mask = (offs_m[:, None] >= offs_n[None, :]) & mask_mn
            p = tl.where(mask, p, 0.0)
        # Compute dP and dS.
        dp = tl.dot(do, vT).to(tl.float32)
        if ENABLE_DROPOUT:
            p = tl.where(dropout_mask, p, 0.0) * dropout_scale
            dp = tl.where(dropout_mask, dp, 0.0) * dropout_scale
        ds = p * (dp - Di[:, None])
        ds = ds.to(tl.float16)
        # Compute dQ.
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        dq += tl.dot(ds, tl.trans(kT))
        # Increment pointers.
        curr_n += step_n
        kT_ptrs += step_n * stride_kn
        vT_ptrs += step_n * stride_kn
        curr_dropout_offset += step_n * stride_kn
        curr_philox_offset += step_n * stride_kn
    return dq


# grid = (tl.cdiv(max_seqlen_q // BLOCK_M2), batch, nheads_q)
@triton.jit
def _bwd_kernel_dq(
    Q, K, V, sm_scale, Out, DO, DQ,
    M, Delta,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kn, stride_kk,
    stride_deltab, stride_deltah, stride_deltam,
    stride_dropoutb, stride_dropouth, stride_dropoutm, stride_dropoutn,
    HQ, HK,
    cu_seqlens_q, cu_seqlens_k,
    max_seqlen_q, max_seqlen_k,
    dropout_mask, dropout_p, philox_seed, philox_offset_base,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLK_SLICE_FACTOR: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ACTUAL_HEAD_DIM: tl.constexpr,
    CAUSAL: tl.constexpr,
    ENABLE_DROPOUT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    # program ids
    pid = tl.program_id(0)
    bid = tl.program_id(1)
    hkid = tl.program_id(2)
    # figure out varlen start and end
    q_start = 0
    k_start = 0
    seqlen_q = max_seqlen_q
    seqlen_k = max_seqlen_k
    if IS_VARLEN:
        # Compute actual sequence lengths
        q_start = tl.load(cu_seqlens_q + bid)
        q_end = tl.load(cu_seqlens_q + bid + 1)
        k_start = tl.load(cu_seqlens_k + bid)
        k_end = tl.load(cu_seqlens_k + bid + 1)
        seqlen_q = q_end - q_start
        seqlen_k = k_end - k_start

    # Figure out causal starting block since we have seqlen_q >=< seqlen_k.
    # Unlike forward pass where we tile on M dim and iterate on N dim, so that
    # we can skip some M blocks, in backward pass, we tile on the N dim for kv
    # and iterate over the M. In this way, we cannot skip N blocks, but only to
    # determine the starting M blocks to skip some initial blocks masked by
    # causal.
    # DQ tiles on M dim and iterate on N dim, so we there could be some tiles we
    # can simply skip and we need to adjust starting position.
    start_m = pid * BLOCK_M
    # seqlen_q > seqlen_k, no need to process these tile for dq
    seqlen_delta = seqlen_q - seqlen_k
    if start_m + BLOCK_M < seqlen_delta:
        return

    offs_k = tl.arange(0, HEAD_DIM)
    offs_m = start_m + tl.arange(0, BLOCK_M)
    # Mask for loading K and V
    mask_q = offs_m[:, None] < seqlen_q
    PADDED_HEAD: tl.constexpr = (ACTUAL_HEAD_DIM != HEAD_DIM)
    if PADDED_HEAD:
        mask_k = offs_k < ACTUAL_HEAD_DIM
        mask_q &= mask_k[None, :]
    offs_q = offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
    adj_kv = bid * stride_kb + hkid * stride_kh + k_start * stride_kn
    K +=  adj_kv
    V +=  adj_kv
    # If MQA / GQA, set the K and V head offsets appropriately.
    GROUP_SIZE = HQ // HK
    for hqid in range(hkid * GROUP_SIZE, hkid * GROUP_SIZE + GROUP_SIZE):
        # seqlen_q < seqlen_k: seqlen_delta more tokens are added at the front
        #   for every M-tile
        end_n = start_m + BLOCK_M + seqlen_delta
        # offset input and output tensor by batch and Q/K heads
        adj_q = bid * stride_qb + hqid * stride_qh + q_start * stride_qm
        adj_delta = \
            bid * stride_deltab + hqid * stride_deltah + q_start * stride_deltam
        Delta_ptr = Delta + adj_delta

        # batch_philox_offset is the ACTUALLY dropout offset
        # dropout_offset is for debug purpose and will be removed later
        batch_philox_offset = 0
        dropout_offset = 0
        if ENABLE_DROPOUT:
            batch_philox_offset = philox_offset_base + \
                                  bid * stride_dropoutb + \
                                  hqid * stride_dropouth
            dropout_offset = \
                dropout_mask + bid * stride_dropoutb + hqid * stride_dropouth

        q = tl.load(Q + adj_q + offs_q, mask=mask_q, other=0.0)
        do = tl.load(DO + adj_q + offs_q, mask=mask_q, other=0.0)
        m = tl.load(M + adj_delta + offs_m, mask=offs_m < seqlen_q)
        m = m[:, None]

        MASK_BLOCK_N: tl.constexpr = BLOCK_N // BLK_SLICE_FACTOR
        num_steps = BLOCK_M // MASK_BLOCK_N

        dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        # scaling factor
        RCP_LN2: tl.constexpr = 1.4426950408889634  # = 1.0 / ln(2)
        qk_scale = sm_scale * RCP_LN2

        # Compute dQ for masked (diagonal) blocks.
        # NOTE: This code scans each row of QK^T backward (from right to left,
        # but inside each call to _bwd_dq_inner, from left to right), but that's
        # not due to anything important.  I just wanted to reuse the loop
        # structure for dK & dV above as much as possible.
        dq = _bwd_dq_inner(
            dq,
            q, K, V, do, m, Delta_ptr, qk_scale, #
            stride_qm, stride_qk, stride_kn,
            stride_dropoutm, stride_dropoutn,  #
            seqlen_q, seqlen_k,  #
            BLOCK_M, MASK_BLOCK_N,  #
            HEAD_DIM, ACTUAL_HEAD_DIM,  #
            dropout_p, philox_seed, batch_philox_offset, dropout_offset,  #
            start_m, end_n - num_steps * MASK_BLOCK_N, num_steps,  #
            MASK=True,  #
            ENABLE_DROPOUT=ENABLE_DROPOUT,
        )
        end_n -= num_steps * MASK_BLOCK_N
        # stage 2
        num_steps = end_n // BLOCK_N
        dq = _bwd_dq_inner(
            dq,  #
            q, K, V, do, m, Delta_ptr, qk_scale, #
            stride_qm, stride_qk, stride_kn,  #
            stride_dropoutm, stride_dropoutn,  #
            seqlen_q, seqlen_k,  #
            BLOCK_M, BLOCK_N,  #
            HEAD_DIM, ACTUAL_HEAD_DIM,  #
            dropout_p, philox_seed, batch_philox_offset, dropout_offset,  #
            start_m, end_n - num_steps * BLOCK_N, num_steps,  #
            MASK=False,  #
            ENABLE_DROPOUT=ENABLE_DROPOUT,
        )
        # Write back dQ.
        offs_dq = offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk
        dq *= sm_scale
        tl.store(DQ + adj_q + offs_dq, dq, mask=mask_q)


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
):
    # make contigious
    # TODO: why making this continguous????
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    softmax_lse = softmax_lse.contiguous()  # (batch, head_q, seqlen_q)
    do = do.contiguous()

    # get strides and shape
    batch, nheads_q, nheads_k, head_size, max_seqlen_q, max_seqlen_k = \
        get_shape_from_layout(
            q, k, layout,
            cu_seqlens_q, cu_seqlens_k,
            max_seqlen_q, max_seqlen_k
        )
    q_strides, k_strides, v_strides, o_strides = \
        get_strides_from_layout(q, k, v, o, layout)
    stride_qb, stride_qh, stride_qm, stride_qk =  q_strides
    stride_kb, stride_kh, stride_kn, stride_kk = k_strides
    stride_vb, stride_vh, stride_vn, stride_vk = v_strides
    stride_ob, stride_oh, stride_om, stride_ok = o_strides
    IS_VARLEN = layout == "thd"
    use_dropout = (dropout_p > 0.0)

    # get closest power of 2 over or equal to 32.
    padded_d_model = 1 << (head_size - 1).bit_length()
    padded_d_model = max(padded_d_model, 16)
    HEAD_DIM = padded_d_model
    ACTUAL_HEAD_DIM = head_size
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
    pre_grid = (triton.cdiv(max_seqlen_q, PRE_BLOCK), batch, nheads_q)
    _bwd_preprocess[pre_grid](
        o, do,
        delta,
        stride_ob, stride_oh, stride_om, stride_ok,
        stride_deltab, stride_deltah, stride_deltam,
        cu_seqlens_q, max_seqlen_q,
        BLOCK_M=PRE_BLOCK,
        HEAD_DIM=HEAD_DIM,
        ACTUAL_HEAD_DIM=ACTUAL_HEAD_DIM,
        IS_VARLEN=IS_VARLEN
    )

    # dropout mask tensor for debugging. We dump the dropout mask created in
    #   the kernel for testing
    dropout_mask = None
    stride_dropoutb, stride_dropouth, stride_dropoutm, stride_dropoutn = \
        (0, 0 , 0 , 0)
    if use_dropout:
        if DROPOUT_USE_PYTORCH:
            dropout_mask = create_dropout_mask(
                dropout_p,
                (batch, nheads_q, max_seqlen_q, max_seqlen_k),
                seed = philox_seed
            )
        else:
            dropout_mask = torch.zeros(
                (batch, nheads_q, max_seqlen_q, max_seqlen_k),
                device=q.device,
                dtype=torch.float32
            )
        stride_dropoutb, stride_dropouth, stride_dropoutm, stride_dropoutn = ( \
            dropout_mask.stride(0),
            dropout_mask.stride(1),
            dropout_mask.stride(2),
            dropout_mask.stride(3)
        )

    grid = ((max_seqlen_k + BLOCK_N1 - 1) // BLOCK_N1, batch, nheads_k)
    _bwd_kernel_dkdv[grid](
        q, k, v, sm_scale, o, do, dk, dv,
        softmax_lse, delta,
        stride_qb, stride_qh, stride_qm, stride_qk,
        stride_kb, stride_kh, stride_kn, stride_kk,
        stride_deltab, stride_deltah, stride_deltam,
        stride_dropoutb, stride_dropouth, stride_dropoutm, stride_dropoutn,
        nheads_q, nheads_k,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        dropout_mask, dropout_p, philox_seed, philox_offset,
        BLOCK_M1, BLOCK_N1, BLK_SLICE_FACTOR,
        HEAD_DIM, ACTUAL_HEAD_DIM,
        CAUSAL=causal,
        ENABLE_DROPOUT=use_dropout,
        IS_VARLEN=IS_VARLEN,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
        waves_per_eu = WAVES_PER_EU,
    )

    grid = ((max_seqlen_q + BLOCK_M2 - 1) // BLOCK_M2, batch, nheads_k)
    _bwd_kernel_dq[grid](
        q, k, v, sm_scale, o, do, dq,
        softmax_lse, delta,
        stride_qb, stride_qh, stride_qm, stride_qk,
        stride_kb, stride_kh, stride_kn, stride_kk,
        stride_deltab, stride_deltah, stride_deltam,
        stride_dropoutb, stride_dropouth, stride_dropoutm, stride_dropoutn,
        nheads_q, nheads_k,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        dropout_mask, dropout_p, philox_seed, philox_offset,
        BLOCK_M2, BLOCK_N2, BLK_SLICE_FACTOR,
        HEAD_DIM, ACTUAL_HEAD_DIM,
        CAUSAL=causal,
        ENABLE_DROPOUT=use_dropout,
        IS_VARLEN=IS_VARLEN,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
        waves_per_eu = WAVES_PER_EU,
    )

    return dq, dk, dv, delta, None, None
