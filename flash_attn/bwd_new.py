import torch

import triton
import triton.language as tl

from triton import cdiv


DEBUG = True

@triton.jit
def _bwd_preprocess(
    Out,
    DO,
    Delta,
    BLOCK_M: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, D_HEAD)
    # load
    o = tl.load(Out + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    do = tl.load(DO + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    # compute
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_m, delta)


@triton.jit
def _bwd_kernel_one_col_block(
    Q,
    K,
    V,
    sm_scale,
    Out,
    DO,
    DQ,
    DK,
    DV,
    L,
    D,
    q_offset,
    k_offset,
    v_offset,
    do_offset,
    dq_offset,
    dk_offset,
    dv_offset,
    stride_dqa,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    Z,
    H,
    N_CTX_Q,
    N_CTX_K,
    off_h,
    off_z,
    off_hz,
    start_n,
    num_block_n,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    ACTUAL_BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SEQUENCE_PARALLEL: tl.constexpr,
    CAUSAL: tl.constexpr,
    USE_EXP2: tl.constexpr,
):
    if CAUSAL:
        lo = start_n * BLOCK_M
    else:
        lo = 0

    if SEQUENCE_PARALLEL:
        dq_offset += stride_dqa * start_n
    # initialize row/col offsets
    offs_n = start_n * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_m = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Apply masking for head dimension
    mask_d = offs_d < ACTUAL_BLOCK_DMODEL

    # pointer to row-wise quantities in value-like data
    D_ptrs = D + off_hz * N_CTX_Q
    l_ptrs = L + off_hz * N_CTX_Q

    # initialize dv and dk accumulators
    dv = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    dk = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # load k and v once per column block
    k_ptrs = k_offset + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    v_ptrs = v_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
    k = tl.load(k_ptrs, mask_d[None, :], other=0.0)
    v = tl.load(v_ptrs, mask_d[None, :], other=0.0)

    # loop over rows
    # num_block_m = tl.cdiv(N_CTX, BLOCK_M)
    for start_m in range(lo, num_block_n * BLOCK_M, BLOCK_M):
        offs_m_curr = start_m + offs_m
        q_ptrs = q_offset + offs_m_curr[:, None] * stride_qm + offs_d[None, :] * stride_qk
        dq_ptrs = dq_offset + offs_m_curr[:, None] * stride_qm + offs_d[None, :] * stride_qk
        do_ptrs = do_offset + offs_m_curr[:, None] * stride_qm + offs_d[None, :] * stride_qk

        # load q, k, v, do on-chip
        q = tl.load(q_ptrs, mask=mask_d[None, :], other=0.0)
        do = tl.load(do_ptrs, mask=mask_d[None, :], other=0.0)

        # recompute p = softmax(qk, dim=-1).T
        # NOTE: `do` is pre-divided by `l`; no normalization here
        if CAUSAL:
            qk = tl.where(
                offs_m_curr[:, None] >= offs_n[None, :], float(0.0), float("-inf")
            )
        else:
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        l_i = tl.load(l_ptrs + offs_m_curr)
       
        if USE_EXP2:
            RCP_LN2: tl.constexpr = 1.4426950408889634
            qk *= sm_scale * RCP_LN2
            l_i *= RCP_LN2
        else:
            qk *= sm_scale

        p = tl.math.exp2(qk - l_i[:, None])
        # compute dv
        dv += tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do)
        # compute dp = dot(v, do)
        Di = tl.load(D_ptrs + offs_m_curr)
        # dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[:, None]
        dp = tl.dot(do, tl.trans(v))
        # compute ds = p * (dp - delta[:, None])
        ds = (p * (dp - Di[:, None]) * sm_scale).to(Q.dtype.element_ty)
        # compute dk = dot(ds.T, q)
        dk += tl.dot(tl.trans(ds), q)

        # compute dq
        if not SEQUENCE_PARALLEL:
            dq = tl.load(dq_ptrs, mask_d[None, :], other=0.0)
            dq += tl.dot(ds, k)
            tl.store(dq_ptrs, dq.to(Q.dtype.element_ty), mask_d[None, :])
        elif SEQUENCE_PARALLEL:
            if False: # path for MMA_V3 in oai kernel
                dq = tl.dot(ds, k)
            else:
                # not work with mma v3, because M % 64 != 0
                dq = tl.trans(tl.dot(tl.trans(k), tl.trans(ds)))
            tl.store(dq_ptrs, dq.to(Q.dtype.element_ty), mask_d[None, :])

    # write-back dv and dk
    dv_ptrs = dv_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
    dk_ptrs = dk_offset + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    # write-back
    tl.store(dv_ptrs, dv.to(V.dtype.element_ty), mask=mask_d[None, :])
    tl.store(dk_ptrs, dk.to(K.dtype.element_ty), mask=mask_d[None, :])


@triton.jit
def _bwd_kernel(
    Q,
    K,
    V,
    sm_scale,
    Out,
    DO,
    DQ,
    DK,
    DV,
    L,
    D,
    stride_dqa,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    Z,
    H,
    N_CTX_Q,
    N_CTX_K,
    Z_H_N_CTX,
    SQ_Z_H_N_CTX,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    ACTUAL_BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SEQUENCE_PARALLEL: tl.constexpr,
    CAUSAL: tl.constexpr,
    USE_EXP2: tl.constexpr,
):
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    off_h = off_hz % H

    q_offset = Q + off_z * stride_qz + off_h * stride_qh
    k_offset = K + off_z * stride_kz + off_h * stride_kh
    v_offset = V + off_z * stride_vz + off_h * stride_vh
    do_offset = DO + off_z * stride_qz + off_h * stride_qh
    if SEQUENCE_PARALLEL:
        dq_offset = DQ + off_z * stride_qz + off_h * stride_qh # use SQ_Z_H_N_CTX
    else:
        dq_offset = DQ + off_z * stride_qz + off_h * stride_qh # use Z_H_N_CTX
    dk_offset = DK + off_z * stride_kz + off_h * stride_kh
    dv_offset = DV + off_z * stride_vz + off_h * stride_vh

    num_block_n = tl.cdiv(N_CTX_K, BLOCK_N)
    if not SEQUENCE_PARALLEL:
        for start_n in range(0, num_block_n):
            _bwd_kernel_one_col_block(
                Q,
                K,
                V,
                sm_scale,
                Out,
                DO,
                DQ,
                DK,
                DV,
                L,
                D,
                q_offset,
                k_offset,
                v_offset,
                do_offset,
                dq_offset,
                dk_offset,
                dv_offset,
                stride_dqa,
                stride_qz,
                stride_qh,
                stride_qm,
                stride_qk,
                stride_kz,
                stride_kh,
                stride_kn,
                stride_kk,
                stride_vz,
                stride_vh,
                stride_vn,
                stride_vk,
                Z,
                H,
                N_CTX_Q,
                N_CTX_K,
                off_h,
                off_z,
                off_hz,
                start_n,
                num_block_n,
                BLOCK_M=BLOCK_M,
                BLOCK_DMODEL=BLOCK_DMODEL,
                ACTUAL_BLOCK_DMODEL=ACTUAL_BLOCK_DMODEL,
                BLOCK_N=BLOCK_N,
                SEQUENCE_PARALLEL=SEQUENCE_PARALLEL,
                CAUSAL=CAUSAL,
                USE_EXP2=USE_EXP2
            )
    else:
        start_n = tl.program_id(1)
        _bwd_kernel_one_col_block(
            Q,
            K,
            V,
            sm_scale,
            Out,
            DO,
            DQ,
            DK,
            DV,
            L,
            D,
            q_offset,
            k_offset,
            v_offset,
            do_offset,
            dq_offset,
            dk_offset,
            dv_offset,
            stride_dqa,
            stride_qz,
            stride_qh,
            stride_qm,
            stride_qk,
            stride_kz,
            stride_kh,
            stride_kn,
            stride_kk,
            stride_vz,
            stride_vh,
            stride_vn,
            stride_vk,
            Z,
            H,
            N_CTX_Q,
            N_CTX_K,
            off_h,
            off_z,
            off_hz,
            start_n,
            num_block_n,
            BLOCK_M=BLOCK_M,
            BLOCK_DMODEL=BLOCK_DMODEL,
            ACTUAL_BLOCK_DMODEL=ACTUAL_BLOCK_DMODEL,
            BLOCK_N=BLOCK_N,
            SEQUENCE_PARALLEL=SEQUENCE_PARALLEL,
            CAUSAL=CAUSAL,
            USE_EXP2=USE_EXP2
        )




def attention_prefill_backward_new_impl(do, q, k, v, o, softmax_lse, sm_scale, head_size, alibi_slopes, causal, layout, use_exp2):
    if DEBUG:
        print()
        print("attention_prefill_backward_new_impl")
        print("do:", do, do.shape)
        print("q:", q, q.shape)
        print("k:", k, k.shape)
        print("v:", v, v.shape)
        print("o:", o, o.shape)
        print("softmax_lse:", softmax_lse, softmax_lse.shape)
        print("sm_scale", sm_scale)
        print("head_size", head_size)
        print("alibi_slopes", alibi_slopes)
        print("layout", layout)

    # the kernel wants bhsd
    if layout == "bshd":
        do = do.transpose(1, 2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        o = o.transpose(1, 2)
        # TODO: does L/M need to be transposed

        if DEBUG:
            print("After layout change")
            print("do:", do, do.shape)
            print("q:", q, q.shape)
            print("k:", k, k.shape)
            print("v:", v, v.shape)
            print("o:", o, o.shape)
    elif layout == "bhsd":
        pass
    else:
        raise ValueError(f"Unknown layout {layout}")

    sequence_parallel = False
    causal = False

    # OOM issue on hip # TODO: use autotune
    if torch.version.hip is not None:
        BLOCK_M = 64
        BLOCK_N = 64
    else:
        BLOCK_M = 128
        BLOCK_N = 128

    batch_q, heads_q, N_CTX_Q, head_size_q = q.shape
    batch_k, heads_k, N_CTX_K, head_size_k = k.shape

    assert (batch_q == batch_k)
    assert (heads_q == heads_k) # just for now
    assert (head_size_q == head_size_q == head_size)

    batch = batch_q
    # get closest power of 2 over or equal to 32.
    padded_d_model = 1 << (head_size - 1).bit_length()
    padded_d_model = max(padded_d_model, 16)
    BLOCK_DMODEL = padded_d_model
    ACTUAL_BLOCK_DMODEL = head_size

    do = do.contiguous()
    if sequence_parallel:
        replicas = cdiv(N_CTX_K, BLOCK_N)
        new_dq_shape = (replicas,) + q.shape
        dq = torch.zeros(new_dq_shape, device=q.device, dtype=q.dtype)
    else:
        dq = torch.zeros_like(q, dtype=q.dtype)

    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    delta = torch.empty_like(softmax_lse)
    
    batch_headsize = batch * heads_q
    num_blocks_m = cdiv(N_CTX_Q, BLOCK_M)
    num_blocks_n = cdiv(N_CTX_K, BLOCK_N)

    _bwd_preprocess[(num_blocks_m * batch_headsize,)](
        o,
        do,
        delta,
        BLOCK_M=BLOCK_M,
        D_HEAD=BLOCK_DMODEL,
    )

    _bwd_kernel[(batch_headsize, num_blocks_n if sequence_parallel else 1)](
        q,
        k,
        v,
        sm_scale,
        o,
        do,
        dq,
        dk,
        dv,
        softmax_lse,
        delta,
        o.numel(),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        q.shape[0],
        q.shape[1],
        N_CTX_Q,
        N_CTX_K,
        batch_q * head_size_q * N_CTX_Q,
        num_blocks_n * batch_q * head_size_q * N_CTX_Q,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        ACTUAL_BLOCK_DMODEL=ACTUAL_BLOCK_DMODEL,
        SEQUENCE_PARALLEL=sequence_parallel,
        CAUSAL=causal,
        num_warps=8,
        num_stages=1,
        USE_EXP2=use_exp2
    )

    if len(dq.shape) == 5:
        dq = dq.sum(dim=0)

    # go back to original layout
    if layout == "bshd":
        dq = dq.transpose(1, 2)
        dk = dk.transpose(1, 2)
        dv = dv.transpose(1, 2)
    elif layout == "bhsd":
        pass
    else:
        raise ValueError(f"Unknown layout {layout}")

    return dq, dk, dv, None, None, None
