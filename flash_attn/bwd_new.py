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
    BLOCK_DMODEL: tl.constexpr,
    ACTUAL_BLOCK_DMODEL: tl.constexpr,
    N_CTX_Q: tl.constexpr
):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, BLOCK_DMODEL)
    
    # create masks
    # mask_m = off_m < N_CTX_Q
    mask_d = off_d < ACTUAL_BLOCK_DMODEL
    # o_mask = None
    # o_mask = mask_m[:, None]
    o_mask = mask_d[None, :]
    # o_mask = mask_m[:, None] & mask_d[None, :]

    # load
    o = tl.load(Out + off_m[:, None] * ACTUAL_BLOCK_DMODEL + off_d[None, :], mask=o_mask).to(tl.float32)
    do = tl.load(DO + off_m[:, None] * ACTUAL_BLOCK_DMODEL + off_d[None, :], mask=o_mask).to(tl.float32)
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
    d_ptrs,
    l_ptrs,
    stride_dq_all,
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
    num_block_m,
    num_block_n,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    ACTUAL_BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SEQUENCE_PARALLEL: tl.constexpr,
    CAUSAL: tl.constexpr,
    USE_EXP2: tl.constexpr,
    PREPROCESSING: tl.constexpr,
):
    if CAUSAL:
        lo = start_n * BLOCK_M
    else:
        lo = 0

    # initialize col and head offsets
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # masks
    mask_n = offs_n < N_CTX_K
    mask_d = offs_d < ACTUAL_BLOCK_DMODEL
    k_mask = mask_n[:, None] & mask_d[None, :]
    v_mask = mask_n[:, None] & mask_d[None, :]
    

    # initialize grad accumulators
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)

    # load k and v once per column block
    k_ptrs = k_offset + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    v_ptrs = v_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
    k = tl.load(k_ptrs, mask=k_mask, other=0.0)
    v = tl.load(v_ptrs, mask=v_mask, other=0.0)
    # print("k:", k)
    # print("v:", v)

    # loop over rows
    for start_m in range(lo, num_block_m * BLOCK_M, BLOCK_M):
        offs_m = start_m + tl.arange(0, BLOCK_M)
        q_ptrs = q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
        dq_ptrs = dq_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
        do_ptrs = do_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
        
        # update mask as row block changes
        mask_m = offs_m < N_CTX_Q
        q_mask = mask_m[:, None] & mask_d[None, :]

        # load q, k, v, do on-chip
        q = tl.load(q_ptrs, mask=q_mask, other=0.0)
        do = tl.load(do_ptrs, mask=q_mask, other=0.0)
        # print("q:", q)
        # print("do:", do)

        # recompute p = softmax(qk, dim=-1).T
        # NOTE: `do` is pre-divided by `l`; no normalization here
        if CAUSAL:
            qk = tl.where(
                offs_m[:, None] >= offs_n[None, :], float(0.0), float("-inf")
            )
        else:
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        # print("qk:", qk)
        l_i = tl.load(l_ptrs + offs_m, mask=mask_m)
        # print("l_i:", l_i)

        # compute p
        if USE_EXP2:
            RCP_LN2: tl.constexpr = 1.4426950408889634
            qk *= sm_scale * RCP_LN2
            l_i *= RCP_LN2
            p = tl.math.exp2(qk - l_i[:, None])
        else:
            qk *= sm_scale
            p = tl.math.exp(qk - l_i[:, None])
        # print("p:", p)
        # mask block in the cases where the data is smaller the block size
        p_mask = mask_m[:, None] & mask_n[None, :]
        p = tl.where(p_mask, p, 0.0)
        # print("p masked:", p)
        
        # compute dv
        dv += tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do)
        # print("dv:", dv)

        # compute dp
        dp = tl.dot(do, tl.trans(v))
        # print("dp:", dp)

        # compute ds , ds = p * (dp - delta[:, None])
        if PREPROCESSING:
            Di = tl.load(d_ptrs + offs_m, mask=mask_m)
            ds = (p * (dp - Di[:, None])) * sm_scale
        else:
            delta = tl.sum(p * dp, axis=1)
            delta = tl.where(mask_m, delta, 0.0)
            ds = (p * (dp - delta[:, None])) * sm_scale
        # print("ds:", ds)
        ds = tl.where(p_mask, ds, 0.0).to(Q.dtype.element_ty)
        # print("ds masked:", ds)
        
        # compute dk = dot(ds.T, q)
        dk += tl.dot(tl.trans(ds), q)
        # print("dk:", dk)

        # compute dq
        if SEQUENCE_PARALLEL:
            if True: # path for MMA_V3 in oai kernel
                dq = tl.dot(ds, k)
            else:
                # not work with mma v3, because M % 64 != 0
                dq = tl.trans(tl.dot(tl.trans(k), tl.trans(ds)))
            tl.store(dq_ptrs, dq.to(Q.dtype.element_ty), mask=q_mask)
        else:
            dq = tl.load(dq_ptrs, mask=q_mask, other=0.0)
            # print("dq load:", dq)
            dq += tl.dot(ds, k)
            # print("dq after dot:", dq)
            tl.store(dq_ptrs, dq.to(Q.dtype.element_ty), mask=q_mask)

    # write-back dv and dk
    dk_ptrs = dk_offset + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    dv_ptrs = dv_offset + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
    # write-back
    # print("dv:", dv)
    tl.store(dk_ptrs, dk.to(K.dtype.element_ty), mask=k_mask)
    tl.store(dv_ptrs, dv.to(V.dtype.element_ty), mask=v_mask)

# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'waves_per_eu': 1, 'PRE_LOAD_V': False}, num_stages=1,
#                       num_warps=4),
#     ],
#     key=['IS_CAUSAL', 'dropout_p', 'BLOCK_DMODEL'],
#     use_cuda_graph=True,
# )
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
    stride_dq_all,
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
    num_block_m,
    num_block_n,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    ACTUAL_BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SEQUENCE_PARALLEL: tl.constexpr,
    CAUSAL: tl.constexpr,
    USE_EXP2: tl.constexpr,
    PREPROCESSING: tl.constexpr
):
    # program ids
    off_hz = tl.program_id(0)
    if SEQUENCE_PARALLEL:
        start_n = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    # input tensor offsets
    q_offset = Q + off_z * stride_qz + off_h * stride_qh
    k_offset = K + off_z * stride_kz + off_h * stride_kh
    v_offset = V + off_z * stride_vz + off_h * stride_vh
    do_offset = DO + off_z * stride_qz + off_h * stride_qh
    l_ptrs = L + off_hz * N_CTX_Q # softmax lse from forward pass. used to recompute attention from forward
    d_ptrs = D + off_hz * N_CTX_Q # delta(o*do summed for each row) TODO: explain delta

    # output tensor offsets
    dk_offset = DK + off_z * stride_kz + off_h * stride_kh
    dv_offset = DV + off_z * stride_vz + off_h * stride_vh
    if SEQUENCE_PARALLEL:
        dq_offset = DQ + stride_dq_all * start_n + off_z * stride_qz + off_h * stride_qh
    else:
        dq_offset = DQ + off_z * stride_qz + off_h * stride_qh

    # inner loop 
    if SEQUENCE_PARALLEL:
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
            d_ptrs,
            l_ptrs,
            stride_dq_all,
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
            num_block_m,
            num_block_n,
            BLOCK_M=BLOCK_M,
            BLOCK_DMODEL=BLOCK_DMODEL,
            ACTUAL_BLOCK_DMODEL=ACTUAL_BLOCK_DMODEL,
            BLOCK_N=BLOCK_N,
            SEQUENCE_PARALLEL=SEQUENCE_PARALLEL,
            CAUSAL=CAUSAL,
            USE_EXP2=USE_EXP2,
            PREPROCESSING=PREPROCESSING
        )
    else:
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
                d_ptrs,
                l_ptrs,
                stride_dq_all,
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
                num_block_m,
                num_block_n,
                BLOCK_M=BLOCK_M,
                BLOCK_DMODEL=BLOCK_DMODEL,
                ACTUAL_BLOCK_DMODEL=ACTUAL_BLOCK_DMODEL,
                BLOCK_N=BLOCK_N,
                SEQUENCE_PARALLEL=SEQUENCE_PARALLEL,
                CAUSAL=CAUSAL,
                USE_EXP2=USE_EXP2,
                PREPROCESSING=PREPROCESSING
            )

def attention_prefill_backward_triton_new_impl(do, q, k, v, o, softmax_lse, dq, dk, dv, sm_scale, head_size, alibi_slopes, causal, layout, use_exp2, preprocessing):

    DEBUG_INPUT=False

    if DEBUG:
        print()
        print("attention_prefill_backward_triton_new_impl")
        print("do:", do, do.shape)
        print("q:", q, q.shape)
        print("k:", k, k.shape)
        print("v:", v, v.shape)
        print("o:", o, o.shape)
        print("softmax_lse:", softmax_lse, softmax_lse.shape)
        print("dq:", dq, dq.shape if dq is not None else None)
        print("dk:", dk, dk.shape if dk is not None else None)
        print("dv:", dv, dv.shape if dv is not None else None)
        print("sm_scale:", sm_scale)
        print("head_size:", head_size)
        print("alibi_slopes:", alibi_slopes)
        print("layout:", layout)
        print("use_exp2:", use_exp2)
        print("preprocessing:", preprocessing)

    # the kernel wants bhsd
    if layout == "bshd":
        do = do.transpose(1, 2).contiguous()
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        o = o.transpose(1, 2).contiguous()
        # TODO: does L/M need to be transposed. possible to use strides

        if False:
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

    # divide up the problem
    num_blocks_m = cdiv(N_CTX_Q, BLOCK_M)
    num_blocks_n = cdiv(N_CTX_K, BLOCK_N)

    # get closest power of 2 over or equal to 32.
    padded_d_model = 1 << (head_size - 1).bit_length()
    padded_d_model = max(padded_d_model, 16)
    BLOCK_DMODEL = padded_d_model
    ACTUAL_BLOCK_DMODEL = head_size

    do = do.contiguous()
    if sequence_parallel:
        # replicate q for each parallel sequence
        replicas = num_blocks_n
        new_dq_shape = (replicas,) + q.shape
        if dq is None: 
            dq = torch.zeros(new_dq_shape, device=q.device, dtype=q.dtype)
    else:
        if dq is None:
            dq = torch.zeros_like(q, dtype=q.dtype)

    # NOTE: the kernel does inplace accumlation so dq has to be zeros. This avoids the case where we are passed empty dq and it is not all zeros
    dq.zero_()
    
    if dk is None:
        if DEBUG_INPUT:
            dk = torch.zeros_like(k)
        else:
            dk = torch.empty_like(k)

    if dv is None:
        if DEBUG_INPUT:
            dv = torch.zeros_like(v)
        else:
            dv = torch.empty_like(v)
        

    # assert contigious
    assert do.is_contiguous()
    assert q.is_contiguous()
    assert k.is_contiguous()
    assert v.is_contiguous()
    assert o.is_contiguous()
    assert softmax_lse.is_contiguous()
    assert dq.is_contiguous()
    assert dk.is_contiguous()
    assert dv.is_contiguous()
    
    batch_headsize = batch * heads_q

    if preprocessing:
        if True:
            delta = torch.zeros_like(softmax_lse)
        else:
            delta = torch.empty_like(softmax_lse)
        _bwd_preprocess[(batch_headsize * num_blocks_m,)](
            o,
            do,
            delta,
            BLOCK_M=BLOCK_M,
            BLOCK_DMODEL=BLOCK_DMODEL,
            ACTUAL_BLOCK_DMODEL=ACTUAL_BLOCK_DMODEL,
            N_CTX_Q=N_CTX_Q
        )

    stride_dq_all = dq.numel()
    stride_qz, stride_qh, stride_qm, stride_qk =  q.stride(0),  q.stride(1), q.stride(2),  q.stride(3)
    stride_kz, stride_kh, stride_kn, stride_kk = k.stride(0),  k.stride(1), k.stride(2),  k.stride(3)
    stride_vz, stride_vh, stride_vn, stride_vk = v.stride(0),  v.stride(1), v.stride(2),  v.stride(3)
    num_warps = 4 # NOTE: originial is 8. changing it to 1 caused issues be careful
    num_stages = 1

    if True:
        print("_bwd_kernel inputs")
        print("do:", do, do.shape)
        print("q:", q, q.shape)
        print("k:", k, k.shape)
        print("v:", v, v.shape)
        print("sm_scale", sm_scale)
        print("o:", o, o.shape)
        print("dq:", dq, dq.shape)
        print("dk:", dk, dk.shape)
        print("dv:", dv, dv.shape)
        print("L:", softmax_lse, softmax_lse.shape)
        print("delta:", delta, delta.shape)
        print("stride_qz, stride_qh, stride_qm, stride_qk:",  stride_qz, stride_qh, stride_qm, stride_qk)
        print("stride_kz, stride_kh, stride_kn, stride_kk:",  stride_kz, stride_kh, stride_kn, stride_kk)
        print("stride_vz, stride_vh, stride_vn, stride_vk:",  stride_vz, stride_vh, stride_vn, stride_vk)
        print("batch_q:", batch_q)
        print("heads_q:",heads_q)
        print("N_CTX_Q:",N_CTX_Q)
        print("N_CTX_K:",N_CTX_K)
        print("batch_q * head_size_q * N_CTX_Q:",batch_q * head_size_q * N_CTX_Q)
        print("num_blocks_n * batch_q * head_size_q * N_CTX_Q:",num_blocks_n * batch_q * head_size_q * N_CTX_Q)
        print("BLOCK_M:",BLOCK_M)
        print("BLOCK_N:",BLOCK_M)
        print("BLOCK_DMODEL:",BLOCK_DMODEL)
        print("ACTUAL_BLOCK_DMODEL:",ACTUAL_BLOCK_DMODEL)
        print("SEQUENCE_PARALLEL:",sequence_parallel)
        print("CAUSAL:",causal)
        print("num_warps:",num_warps)
        print("num_stages:", num_stages)
        print("USE_EXP2:", use_exp2)
        print("preprocessing:", preprocessing)

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
        stride_dq_all,
        stride_qz, stride_qh, stride_qm, stride_qk,
        stride_kz, stride_kh, stride_kn, stride_kk,
        stride_vz, stride_vh, stride_vn, stride_vk,
        batch_q,
        heads_q,
        N_CTX_Q,
        N_CTX_K,
        num_blocks_m,
        num_blocks_n,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        ACTUAL_BLOCK_DMODEL=ACTUAL_BLOCK_DMODEL,
        SEQUENCE_PARALLEL=sequence_parallel,
        CAUSAL=causal,
        USE_EXP2=use_exp2,
        PREPROCESSING=preprocessing,
        num_warps=num_warps,
        num_stages=num_stages,
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

    return dq, dk, dv, delta, None, None
