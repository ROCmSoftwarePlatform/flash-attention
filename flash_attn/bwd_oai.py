import torch
import triton

from triton import cdiv, jit
from triton import language as tl


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


@jit
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


@jit
def _bwd_kernel_one_col_block(
    Q,
    K,
    V,
    sm_scale,
    qk_scale,  #
    Out,
    DO,  #
    DQ,
    DK,
    DV,  #
    L,  #
    D,  #
    Q_block_ptr,
    K_block_ptr,
    V_block_ptr,  #
    DO_block_ptr,
    DQ_block_ptr,
    DK_block_ptr,
    DV_block_ptr,  #
    stride_dqa,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,  #
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,  #
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,  #
    Z,
    H,
    N_CTX,  #
    off_h,
    off_z,
    off_hz,
    start_n,
    num_block,  #
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    SEQUENCE_PARALLEL: tl.constexpr,  #
    CAUSAL: tl.constexpr,  #
    MMA_V3: tl.constexpr,  #
):
    if CAUSAL:
        lo = start_n * BLOCK_M
    else:
        lo = 0

    Q_offset = (off_z * stride_qz + off_h * stride_qh) // stride_qm
    DQ_offset = off_z * stride_qz + off_h * stride_qh
    K_offset = (off_z * stride_kz + off_h * stride_kh) // stride_kn
    V_offset = (off_z * stride_vz + off_h * stride_vh) // stride_vn
    if SEQUENCE_PARALLEL:
        DQ_offset += stride_dqa * start_n
    DQ_offset = DQ_offset // stride_qm

    Q_block_ptr = tl.advance(Q_block_ptr, (lo + Q_offset, 0))
    K_block_ptr = tl.advance(K_block_ptr, (start_n * BLOCK_M + K_offset, 0))
    V_block_ptr = tl.advance(V_block_ptr, (start_n * BLOCK_M + V_offset, 0))
    DO_block_ptr = tl.advance(DO_block_ptr, (lo + Q_offset, 0))
    DQ_block_ptr = tl.advance(DQ_block_ptr, (lo + DQ_offset, 0))
    DK_block_ptr = tl.advance(DK_block_ptr, (start_n * BLOCK_M + K_offset, 0))
    DV_block_ptr = tl.advance(DV_block_ptr, (start_n * BLOCK_M + V_offset, 0))

    # initialize row/col offsets
    offs_n = start_n * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_m = tl.arange(0, BLOCK_N)
    # pointer to row-wise quantities in value-like data
    D_ptrs = D + off_hz * N_CTX
    l_ptrs = L + off_hz * N_CTX
    # initialize dv amd dk
    dv = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    dk = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # k and v stay in SRAM throughout
    k = tl.load(K_block_ptr)
    v = tl.load(V_block_ptr)
    # loop over rows
    for start_m in range(lo, num_block * BLOCK_M, BLOCK_M):
        offs_m_curr = start_m + offs_m
        # load q, k, v, do on-chip
        q = tl.load(Q_block_ptr)
        # recompute p = softmax(qk, dim=-1).T
        # NOTE: `do` is pre-divided by `l`; no normalization here
        if CAUSAL:
            qk = tl.where(
                offs_m_curr[:, None] >= (offs_n[None, :]), float(0.0), float("-inf")
            )
        else:
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        qk *= qk_scale
        l_i = tl.load(l_ptrs + offs_m_curr)
        p = tl.math.exp2(qk - l_i[:, None])
        # compute dv
        do = tl.load(DO_block_ptr)
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
            dq = tl.load(DQ_block_ptr)
            dq += tl.dot(ds, k)
            tl.store(DQ_block_ptr, dq.to(Q.dtype.element_ty))
        elif SEQUENCE_PARALLEL:
            if MMA_V3:
                dq = tl.dot(ds, k)
            else:
                # not work with mma v3, because M % 64 != 0
                dq = tl.trans(tl.dot(tl.trans(k), tl.trans(ds)))
            tl.store(DQ_block_ptr, dq.to(Q.dtype.element_ty))

        # increment pointers
        DQ_block_ptr = tl.advance(DQ_block_ptr, (BLOCK_M, 0))
        Q_block_ptr = tl.advance(Q_block_ptr, (BLOCK_M, 0))
        DO_block_ptr = tl.advance(DO_block_ptr, (BLOCK_M, 0))
    # write-back
    tl.store(DV_block_ptr, dv.to(V.dtype.element_ty))
    tl.store(DK_block_ptr, dk.to(K.dtype.element_ty))


@jit
def _bwd_kernel(
    Q,
    K,
    V,
    sm_scale,  #
    Out,
    DO,  #
    DQ,
    DK,
    DV,  #
    L,  #
    D,  #
    stride_dqa,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,  #
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,  #
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,  #
    Z,
    H,
    N_CTX,  #
    Z_H_N_CTX,  #
    SQ_Z_H_N_CTX,  #
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,  #
    BLOCK_N: tl.constexpr,  #
    SEQUENCE_PARALLEL: tl.constexpr,  #
    CAUSAL: tl.constexpr,  #
    MMA_V3: tl.constexpr,  #
):
    qk_scale = sm_scale * 1.44269504
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    off_h = off_hz % H

    Q_block_ptr = tl.make_block_ptr(
        base=Q,
        shape=(Z_H_N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K,
        shape=(Z_H_N_CTX, BLOCK_DMODEL),
        strides=(stride_kn, stride_kk),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V,
        shape=(Z_H_N_CTX, BLOCK_DMODEL),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    DO_block_ptr = tl.make_block_ptr(
        base=DO,
        shape=(Z_H_N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qk),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    if SEQUENCE_PARALLEL:
        DQ_block_ptr = tl.make_block_ptr(
            base=DQ,
            shape=(SQ_Z_H_N_CTX, BLOCK_DMODEL),
            strides=(stride_qm, stride_qk),
            offsets=(0, 0),
            block_shape=(BLOCK_M, BLOCK_DMODEL),
            order=(1, 0),
        )
    else:
        DQ_block_ptr = tl.make_block_ptr(
            base=DQ,
            shape=(Z_H_N_CTX, BLOCK_DMODEL),
            strides=(stride_qm, stride_qk),
            offsets=(0, 0),
            block_shape=(BLOCK_M, BLOCK_DMODEL),
            order=(1, 0),
        )

    DK_block_ptr = tl.make_block_ptr(
        base=DK,
        shape=(Z_H_N_CTX, BLOCK_DMODEL),
        strides=(stride_kn, stride_kk),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )
    DV_block_ptr = tl.make_block_ptr(
        base=DV,
        shape=(Z_H_N_CTX, BLOCK_DMODEL),
        strides=(stride_vn, stride_vk),
        offsets=(0, 0),
        block_shape=(BLOCK_M, BLOCK_DMODEL),
        order=(1, 0),
    )

    num_block_n = tl.cdiv(N_CTX, BLOCK_N)
    if not SEQUENCE_PARALLEL:
        for start_n in range(0, num_block_n):
            _bwd_kernel_one_col_block(
                Q,
                K,
                V,
                sm_scale,
                qk_scale,
                Out,
                DO,  #
                DQ,
                DK,
                DV,  #
                L,  #
                D,  #
                Q_block_ptr,
                K_block_ptr,
                V_block_ptr,  #
                DO_block_ptr,
                DQ_block_ptr,
                DK_block_ptr,
                DV_block_ptr,  #
                stride_dqa,
                stride_qz,
                stride_qh,
                stride_qm,
                stride_qk,  #
                stride_kz,
                stride_kh,
                stride_kn,
                stride_kk,  #
                stride_vz,
                stride_vh,
                stride_vn,
                stride_vk,  #
                Z,
                H,
                N_CTX,  #
                off_h,
                off_z,
                off_hz,
                start_n,
                num_block_n,  #
                BLOCK_M=BLOCK_M,
                BLOCK_DMODEL=BLOCK_DMODEL,  #
                BLOCK_N=BLOCK_N,  #
                SEQUENCE_PARALLEL=SEQUENCE_PARALLEL,  #
                CAUSAL=CAUSAL,  #
                MMA_V3=MMA_V3,  #
            )
    else:
        start_n = tl.program_id(1)
        _bwd_kernel_one_col_block(
            Q,
            K,
            V,
            sm_scale,
            qk_scale,
            Out,
            DO,  #
            DQ,
            DK,
            DV,  #
            L,  #
            D,  #
            Q_block_ptr,
            K_block_ptr,
            V_block_ptr,  #
            DO_block_ptr,
            DQ_block_ptr,
            DK_block_ptr,
            DV_block_ptr,  #
            stride_dqa,
            stride_qz,
            stride_qh,
            stride_qm,
            stride_qk,  #
            stride_kz,
            stride_kh,
            stride_kn,
            stride_kk,  #
            stride_vz,
            stride_vh,
            stride_vn,
            stride_vk,  #
            Z,
            H,
            N_CTX,  #
            off_h,
            off_z,
            off_hz,
            start_n,
            num_block_n,  #
            BLOCK_M=BLOCK_M,
            BLOCK_DMODEL=BLOCK_DMODEL,  #
            BLOCK_N=BLOCK_N,  #
            SEQUENCE_PARALLEL=SEQUENCE_PARALLEL,  #
            CAUSAL=CAUSAL,  #
            MMA_V3=MMA_V3,  #
        )


def attention_prefill_backward_triton_oai_impl(do, q, k, v, o, L, sm_scale, head_size, alibi_slopes, causal, layout, sequence_parallel=False):
    DEBUG = True
    
    if DEBUG:
        print()
        print("attention_prefill_backward_triton_oai_impl")
        print("do:", do, do.shape)
        print("q:", q, q.shape)
        print("k:", k, k.shape)
        print("v:", v, v.shape)
        print("o:", o, o.shape)
        print("L:", L, L.shape)
        print("sm_scale", sm_scale)
        print("head_size", head_size)
        print("alibi_slopes", alibi_slopes)
        print("layout", layout)

    if layout != "bhsd":
        raise ValueError("OAI kernel expects bhsd")
    capability = torch.cuda.get_device_capability()
    MMA_V3 = capability[0] >= 9
    BLOCK = 128

    if is_hip():
        # Bwd pass runs out of shared memory on HIP with larger block size.
        BLOCK = 64

    # my changes
    grid = (cdiv(q.shape[2], BLOCK), q.shape[0] * q.shape[1], 1)
    BLOCK_DMODEL = head_size

    sequence_parallel = sequence_parallel
    seq_len_kv = k.shape[2]
    do = do.contiguous()
    if sequence_parallel:
        replicas = cdiv(seq_len_kv, BLOCK)
        new_dq_shape = (replicas,) + q.shape
        dq = torch.zeros(new_dq_shape, device=q.device, dtype=q.dtype)
    else:
        dq = torch.zeros_like(q, dtype=q.dtype)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    delta = torch.empty_like(L)
    _bwd_preprocess[(cdiv(q.shape[2], BLOCK) * grid[1],)](
        o,
        do,
        delta,
        BLOCK_M=BLOCK,
        D_HEAD=BLOCK_DMODEL,
    )
    # if True:
    #     print("after _bwd_preprocess")
    #     print("o:", o, o.shape)
    #     print("do:", do, do.shape)
    #     print("delta:", delta, delta.shape)
    #     print("BLOCK_M:", BLOCK)
    #     print("D_HEAD:", BLOCK_DMODEL)


    # if True:
    #     print("before _bwd_kernel")
    #     print("q:", q, q.shape)
    #     print("k:", k, k.shape)
    #     print("v:", v, v.shape)
    #     print("sm_scale", sm_scale)
    #     print("o:", o, o.shape)
    #     print("do:", do, do.shape)
    #     print("dq:", dq, dq.shape)
    #     print("dk:", dk, dk.shape)
    #     print("dv:", dv, dv.shape)
    #     print("L:", L, L.shape)
    #     print("delta:", delta, delta.shape)
    _bwd_kernel[(grid[1], cdiv(seq_len_kv, BLOCK) if sequence_parallel else 1)](
        q,
        k,
        v,
        sm_scale,  #
        o,
        do,  #
        dq,
        dk,
        dv,  #
        L,  #
        delta,  #
        o.numel(),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),  #
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),  #
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),  #
        q.shape[0],
        q.shape[1],
        q.shape[2],  #
        q.shape[0] * q.shape[1] * q.shape[2],  #
        cdiv(seq_len_kv, BLOCK) * q.shape[0] * q.shape[1] * q.shape[2],  #
        BLOCK_M=BLOCK,
        BLOCK_N=BLOCK,  #
        BLOCK_DMODEL=BLOCK_DMODEL,  #
        SEQUENCE_PARALLEL=sequence_parallel,  #
        CAUSAL=causal,  #
        MMA_V3=MMA_V3,  #
        num_warps=8,  #
        num_stages=1,  #
    )

    if len(dq.shape) == 5:
        dq = dq.sum(dim=0)
    return dq, dk, dv, None, None, None