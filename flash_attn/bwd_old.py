
import torch

import triton
import triton.language as tl

from triton import cdiv

RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
LN2= 0.6931471824645996  # = ln(2)

@triton.jit
def _attn_bwd_preprocess(
    Out,
    DO,
    Delta,
    stride_oz,
    stride_oh,
    stride_om,
    stride_on,
    stride_doz,
    stride_doh,
    stride_dom,
    stride_don,
    seqlen_q,
    head_dim,
    BLOCK_M: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    # off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    # off_n = tl.arange(0, D_HEAD)
    off_m = tl.program_id(0) * BLOCK_M
    off_h = tl.program_id(1)  # head index
    off_z = tl.program_id(2)  # batch index
    num_h = tl.num_programs(1)
    o_offset = off_h * stride_oh + off_z * stride_oz
    O_block_ptr = tl.make_block_ptr(base=Out + o_offset, shape=(seqlen_q, head_dim), strides=(stride_om, stride_on),
                                    offsets=(off_m, 0), block_shape=(BLOCK_M, D_HEAD), order=(1, 0))
    do_offset = off_h * stride_doh + off_z * stride_doz
    DO_block_ptr = tl.make_block_ptr(base=DO + do_offset, shape=(seqlen_q, head_dim), strides=(stride_dom, stride_don),
                                     offsets=(off_m, 0), block_shape=(BLOCK_M, D_HEAD), order=(1, 0))
    # load
    # o = tl.load(Out + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    # do = tl.load(DO + off_m[:, None] * D_HEAD + off_n[None, :]).to(tl.float32)
    o = tl.load(O_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    do = tl.load(DO_block_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
    # compute
    delta = tl.sum(o * do, axis=1)
    # write-back, shape (q.shape[0] * q.shape[1], q.shape[2])
    off_zh = off_z * num_h + off_h * 1
    # Check for OOB accesses
    delta_ptrs = Delta + off_zh * seqlen_q + off_m + tl.arange(0, BLOCK_M)
    overflow = off_m + BLOCK_M - seqlen_q
    if overflow > 0:
        boundary = tl.full((BLOCK_M, ), BLOCK_M - overflow, dtype=tl.int32)
        mask = boundary > tl.arange(0, BLOCK_M)
        tl.store(delta_ptrs, delta, mask=mask)
    else:
        tl.store(delta_ptrs, delta)


@triton.jit
def _bwd_kernel_dk_dv(dk, dv, Q, k, v, sm_scale, alibi_slope, DO, LSE, D,
                      # shared by Q/K/V/DO.
                      stride_tok, stride_d, H, N_CTX, BLOCK_M1: tl.constexpr, BLOCK_N1: tl.constexpr,
                      BLOCK_DMODEL: tl.constexpr,
                      # Filled in by the wrapper.
                      start_n, start_m, num_steps, MASK: tl.constexpr,
                      USE_EXP2: tl.constexpr, RCP_LN2: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M1)
    offs_n = start_n + tl.arange(0, BLOCK_N1)
    # offs_k = tl.arange(0, BLOCK_DMODEL)
    QT_block_ptr = tl.make_block_ptr(base=Q, shape=(BLOCK_DMODEL, N_CTX), strides=(stride_d, stride_tok),
                                     offsets=(0, start_m), block_shape=(BLOCK_DMODEL, BLOCK_M1), order=(0, 1))
    DO_block_ptr = tl.make_block_ptr(base=DO, shape=(N_CTX, BLOCK_DMODEL), strides=(stride_tok, stride_d),
                                     offsets=(start_m, 0), block_shape=(BLOCK_M1, BLOCK_DMODEL), order=(1, 0))
    # BLOCK_N1 must be a multiple of BLOCK_M1, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_N1 % BLOCK_M1 == 0)
    curr_m = start_m
    step_m = BLOCK_M1
    for blk_idx in range(num_steps):
        qT = tl.load(QT_block_ptr)
        # Load m before computing qk to reduce pipeline stall.
        offs_m = curr_m + tl.arange(0, BLOCK_M1)
        l = tl.load(LSE + offs_m)
        kqT = tl.dot(k, qT)
        if alibi_slope is not None:
            alibi_block = compute_alibi_block(alibi_slope, N_CTX, N_CTX, offs_m, offs_n, True)
            kqT += alibi_block
        
        if USE_EXP2:
            RCP_LN2: tl.constexpr = 1.4426950408889634
            kqT *= sm_scale * RCP_LN2
            l *= RCP_LN2
            pT = tl.math.exp2(kqT - l[None, :])
        else:
            kqT *= sm_scale
            pT = tl.math.exp(kqT - l[None, :])
        
        # if tl.program_id(0) == 0:
        #     print("pT:", pT)


        # Autoregressive masking.
        if MASK:
            mask = (offs_m[None, :] >= offs_n[:, None])
            # if tl.program_id(0) == 0:
            #     print("mask:", mask)
            pT = tl.where(mask, pT, 0.0)
        do = tl.load(DO_block_ptr)
        # Compute dV.
        ppT = pT
        ppT = ppT.to(tl.float16)
        dv += tl.dot(ppT, do)
        # D (= delta) is pre-divided by ds_scale.
        Di = tl.load(D + offs_m)
        # Compute dP and dS.
        dpT = tl.dot(v, tl.trans(do))
        dsT = pT * (dpT - Di[None, :])
        dsT = dsT.to(tl.float16)
        dk += tl.dot(dsT, tl.trans(qT))
        # Increment pointers.
        curr_m += step_m
        QT_block_ptr = tl.advance(QT_block_ptr, (0, step_m))
        DO_block_ptr = tl.advance(DO_block_ptr, (step_m, 0))
    return dk, dv


@triton.jit
def _bwd_kernel_dq(dq, q, K, V, do, lse, D, alibi_slope, sm_scale,
                   # shared by Q/K/V/DO.
                   stride_tok, stride_d, H, N_CTX, BLOCK_M2: tl.constexpr, BLOCK_N2: tl.constexpr,
                   BLOCK_DMODEL: tl.constexpr,
                   # Filled in by the wrapper.
                   start_m, start_n, num_steps, MASK: tl.constexpr,
                    USE_EXP2: tl.constexpr, RCP_LN2: tl.constexpr):
    offs_m = start_m + tl.arange(0, BLOCK_M2)
    offs_n = start_n + tl.arange(0, BLOCK_N2)
    # offs_k = tl.arange(0, BLOCK_DMODEL)
    KT_block_ptr = tl.make_block_ptr(base=K, shape=(BLOCK_DMODEL, N_CTX), strides=(stride_d, stride_tok),
                                     offsets=(0, start_n), block_shape=(BLOCK_DMODEL, BLOCK_N2), order=(0, 1))
    VT_block_ptr = tl.make_block_ptr(base=V, shape=(BLOCK_DMODEL, N_CTX), strides=(stride_d, stride_tok),
                                     offsets=(0, start_n), block_shape=(BLOCK_DMODEL, BLOCK_N2), order=(0, 1))
    # D (= delta) is pre-divided by ds_scale.
    Di = tl.load(D + offs_m)
    # BLOCK_M2 must be a multiple of BLOCK_N2, otherwise the code wouldn't work.
    tl.static_assert(BLOCK_M2 % BLOCK_N2 == 0)
    curr_n = start_n
    step_n = BLOCK_N2
    for blk_idx in range(num_steps):
        kT = tl.load(KT_block_ptr)
        qk = tl.dot(q, kT)
        if alibi_slope is not None:
            alibi_block = compute_alibi_block(alibi_slope, N_CTX, N_CTX, offs_m, offs_n)
            qk += alibi_block
        
        if USE_EXP2:
            RCP_LN2: tl.constexpr = 1.4426950408889634
            qk *= sm_scale * RCP_LN2
            lse *= RCP_LN2
            p = tl.math.exp2(qk - lse)
        else:
            qk *= sm_scale
            p = tl.math.exp(qk - lse)
        
        # Autoregressive masking.
        if MASK:
            offs_n = curr_n + tl.arange(0, BLOCK_N2)
            mask = (offs_m[:, None] >= offs_n[None, :])
            p = tl.where(mask, p, 0.0)
        # Compute dP and dS.
        vT = tl.load(VT_block_ptr)
        dp = tl.dot(do, vT).to(tl.float32)
        ds = p * (dp - Di[:, None])
        ds = ds.to(tl.float16)
        # Compute dQ.0.
        # NOTE: We need to de-scale dq in the end, because kT was pre-scaled.
        dq += tl.dot(ds, tl.trans(kT))
        # Increment pointers.
        curr_n += step_n
        KT_block_ptr = tl.advance(KT_block_ptr, (0, step_n))
        VT_block_ptr = tl.advance(VT_block_ptr, (0, step_n))
    return dq


@triton.jit
def _attn_bwd(Q, K, V, sm_scale, alibi_slopes, DO, DQ, DK, DV, LSE, D,
              # shared by Q/K/V/DO.
              stride_qz, stride_qh, stride_qm, stride_qd,
              # H = 16, N_CTX = 1024
              H, N_CTX, BLOCK_DMODEL: tl.constexpr, BLOCK_M1: tl.constexpr, BLOCK_N1: tl.constexpr,
              BLOCK_M2: tl.constexpr, BLOCK_N2: tl.constexpr, BLK_SLICE_FACTOR: tl.constexpr, USE_ALIBI: tl.constexpr,
              USE_EXP2: tl.constexpr, RCP_LN2: tl.constexpr, LN2: tl.constexpr):

    # grid is (num_blocks_n, 1, Batch * Heads)
    n_block_id = tl.program_id(0)
    bh_id = tl.program_id(2)
    off_chz = (bh_id * N_CTX).to(tl.int64)
    adj = (stride_qh * (bh_id % H) + stride_qz * (bh_id // H)).to(tl.int64)
    

    # offset pointers for batch/head
    Q += adj
    K += adj
    V += adj
    DO += adj
    DQ += adj
    DK += adj
    DV += adj
    LSE += off_chz
    D += off_chz

    # load K and V. they stay in SRAM throughout the inner loop for dk dv.
    start_n = n_block_id * BLOCK_N1
    K_block_ptr = tl.make_block_ptr(
        base=K,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qd), # kernel assumes seqlen_q and seqlen_k are equal
        offsets=(start_n, 0),
        block_shape=(BLOCK_N1, BLOCK_DMODEL),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V,
        shape=(N_CTX, BLOCK_DMODEL),
        strides=(stride_qm, stride_qd), # kernel assumes seqlen_q and seqlen_k are equal
        offsets=(start_n, 0),
        block_shape=(BLOCK_N1, BLOCK_DMODEL),
        order=(1, 0),
    )
    k = tl.load(K_block_ptr)
    v = tl.load(V_block_ptr)

    # dk and dv are accumlated here
    dv = tl.zeros([BLOCK_N1, BLOCK_DMODEL], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N1, BLOCK_DMODEL], dtype=tl.float32)

    # use alibi
    if USE_ALIBI:
        a_offset = bh_id
        alibi_slope = tl.load(alibi_slopes + a_offset)
    else:
        alibi_slope = None

    # This assignment is important. It is what allows us to pick the diagonal
    # blocks. Later, when we want to do the lower triangular, we update start_m
    # after the first dkdv call.
    # start_m = start_n
    # compute dK and dV for blocks close to the diagonal that need to be masked
    # MASK_BLOCK_M1: tl.constexpr = BLOCK_M1 // BLK_SLICE_FACTOR # BLK_SLICE_FACTOR is a magical number 2
    # num_steps = BLOCK_N1 // MASK_BLOCK_M1
    # dk, dv = _bwd_kernel_dk_dv(dk, dv, Q, k, v, sm_scale, alibi_slope, DO, M, D, stride_qm, stride_qd, H, N_CTX,
    #                            MASK_BLOCK_M1, BLOCK_N1, BLOCK_DMODEL, start_n, start_m, num_steps, MASK=True,
    #                            USE_EXP2=USE_EXP2, RCP_LN2=RCP_LN2)

    # compute dK and dV for blocks that don't need masking further from the diagonal
    # start_m += num_steps * MASK_BLOCK_M1
    # start_m += (BLOCK_N1 // (BLOCK_M1 // BLK_SLICE_FACTOR)) * (BLOCK_M1 // BLK_SLICE_FACTOR)
    # start_m += BLOCK_N1
    # start_m = start_m + BLOCK_N1
    # start_m = start_n + BLOCK_N1
    # start_m = (n_block_id * BLOCK_N1) + BLOCK_N1
    # start_m = (n_block_id + 1) * BLOCK_N1
    
    # start_m = (n_block_id + 1) * BLOCK_N1
    # num_steps = (N_CTX - start_m) // BLOCK_M1
    start_m = 0
    num_blocks_m = N_CTX // BLOCK_M1
    dk, dv = _bwd_kernel_dk_dv(dk, dv, Q, k, v, sm_scale, alibi_slope, DO, LSE, D, stride_qm, stride_qd, H, N_CTX,
                               BLOCK_M1, BLOCK_N1, BLOCK_DMODEL, start_n, start_m, num_blocks_m, MASK=False,
                               USE_EXP2=USE_EXP2, RCP_LN2=RCP_LN2)

    DV_block_ptrs = tl.make_block_ptr(base=DV, shape=(N_CTX, BLOCK_DMODEL), strides=(stride_qm, stride_qd),
                                      offsets=(start_n, 0), block_shape=(BLOCK_N1, BLOCK_DMODEL), order=(1, 0))
    # print("dv:", dv)
    tl.store(DV_block_ptrs, dv.to(v.dtype))

    # Write back dK.
    DK_block_ptrs = tl.make_block_ptr(base=DK, shape=(N_CTX, BLOCK_DMODEL), strides=(stride_qm, stride_qd),
                                      offsets=(start_n, 0), block_shape=(BLOCK_N1, BLOCK_DMODEL), order=(1, 0))
    tl.store(DK_block_ptrs, dk.to(k.dtype))

    # THIS BLOCK DOES DQ:
    start_m = n_block_id * BLOCK_M2
    # end_n = start_m + BLOCK_M2

    # MASK_BLOCK_N2: tl.constexpr = BLOCK_N2 // BLK_SLICE_FACTOR
    offs_m = start_m + tl.arange(0, BLOCK_M2)

    Q_block_ptr = tl.make_block_ptr(base=Q, shape=(N_CTX, BLOCK_DMODEL), strides=(stride_qm, stride_qd),
                                    offsets=(start_m, 0), block_shape=(BLOCK_M2, BLOCK_DMODEL), order=(1, 0))

    DO_block_ptr = tl.make_block_ptr(base=DO, shape=(N_CTX, BLOCK_DMODEL), strides=(stride_qm, stride_qd),
                                     offsets=(start_m, 0), block_shape=(BLOCK_M2, BLOCK_DMODEL), order=(1, 0))
    q = tl.load(Q_block_ptr)
    do = tl.load(DO_block_ptr)
    dq = tl.zeros([BLOCK_M2, BLOCK_DMODEL], dtype=tl.float32)

    l = tl.load(LSE + offs_m)
    l = l[:, None]

    # # Compute dQ for masked (diagonal) blocks.
    # # NOTE: This code scans each row of QK^T backward (from right to left,
    # # but inside each call to _attn_bwd_dq, from left to right), but that's
    # # not due to anything important.  I just wanted to reuse the loop
    # # structure for dK & dV above as much as possible.
    # num_steps = BLOCK_M2 // MASK_BLOCK_N2
    # dq = _bwd_kernel_dq(dq, q, K, V, do, m, D, alibi_slope, sm_scale, stride_qm, stride_qd, H, N_CTX, BLOCK_M2, MASK_BLOCK_N2,
    #                     BLOCK_DMODEL, start_m, end_n - num_steps * MASK_BLOCK_N2, num_steps, MASK=True,
    #                     USE_EXP2=USE_EXP2, RCP_LN2=RCP_LN2)
    # end_n -= num_steps * MASK_BLOCK_N2
    # stage 2
    # start_n = end_n - num_steps * BLOCK_N2
    # num_steps = end_n // BLOCK_N2
    # num_steps = end_n // BLOCK_N2
    start_n = 0
    num_blocks_m = N_CTX // BLOCK_M1
    dq = _bwd_kernel_dq(dq, q, K, V, do, l, D, alibi_slope, sm_scale, stride_qm, stride_qd, H, N_CTX, BLOCK_M2, BLOCK_N2,
                        BLOCK_DMODEL, start_m, start_n, num_blocks_m, MASK=False,
                        USE_EXP2=USE_EXP2, RCP_LN2=RCP_LN2)
    # Write back dQ.
    DQ_block_ptr = tl.make_block_ptr(base=DQ, shape=(N_CTX, BLOCK_DMODEL), strides=(stride_qm, stride_qd),
                                     offsets=(start_m, 0), block_shape=(BLOCK_M2, BLOCK_DMODEL), order=(1, 0))
    tl.store(DQ_block_ptr, dq.to(q.dtype))

def attention_prefill_backward_triton_old_impl(do, q, k, v, o, softmax_lse, sm_scale, head_size, alibi_slopes, causal, layout, use_exp2):
    if True:
        print()
        print("attention_prefill_backward_triton_old_impl")
        print("do:", do, do.shape)
        print("q:", q, q.shape)
        print("k:", k, k.shape)
        print("v:", v, v.shape)
        print("o:", o, o.shape)
        print("softmax_lse:", softmax_lse, softmax_lse.shape)
        print("sm_scale:", sm_scale)
        print("head_size:", head_size)
        print("alibi_slopes:", alibi_slopes)
        print("layout:", layout)
        print("use_exp2:", use_exp2)

    # the kernel wants bhsd
    if layout == "bhsd":
        pass
    elif layout == "bshd":
        do= do.transpose(1, 2)
        q= q.transpose(1, 2)
        k= q.transpose(1, 2)
        v= q.transpose(1, 2)
        o= q.transpose(1, 2)
    else:
        raise ValueError(f"Unknown layout {layout}")

    if torch.version.hip is not None:
        BLOCK = 64
    else:
        BLOCK = 128
    assert do.is_contiguous()
    assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()
    seqlen_q = q.shape[2]
    if True:
        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
    else:
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
    
    BATCH, N_HEAD, N_CTX = q.shape[:3]
    PRE_BLOCK = 128
    # NUM_WARPS, NUM_STAGES = 4, 1
    BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 64, 64, 32
    BLK_SLICE_FACTOR = 2
    assert N_CTX % PRE_BLOCK == 0
    delta = torch.empty_like(softmax_lse)
    _, Lk, _ = q.shape[-1], k.shape[-1], v.shape[-1]
    # padded_head = (Lk != ctx.BLOCK_DMODEL)
    grid_preprocess = (triton.cdiv(do.shape[2], BLOCK), do.shape[1], do.shape[0])

    _attn_bwd_preprocess[grid_preprocess](
        o,
        do,
        delta,
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        do.stride(0),
        do.stride(1),
        do.stride(2),
        do.stride(3),
        seqlen_q,
        head_dim=Lk,
        BLOCK_M=BLOCK,
        D_HEAD=head_size,
    )
    grid = lambda META: (triton.cdiv(N_CTX, META['BLOCK_N1']), 1, BATCH * N_HEAD)

    stride_qz, stride_qh, stride_qm, stride_qk =  q.stride(0),  q.stride(1), q.stride(2),  q.stride(3)
    use_alibi = False if alibi_slopes is None else True
    
    if True:
        print("_attn_bwd inputs")
        print("q:", q, q.shape)
        print("k:", k, k.shape)
        print("v:", v, v.shape)
        print("sm_scale", sm_scale)
        print("alibi_slopes:", alibi_slopes)
        print("o:", o, o.shape)
        print("do:", do, do.shape)
        print("dq:", dq, dq.shape)
        print("dk:", dk, dk.shape)
        print("dv:", dv, dv.shape)
        print("L:", softmax_lse, softmax_lse.shape)
        print("delta:", delta, delta.shape)
        print("stride_qz, stride_qh, stride_qm, stride_qk:", stride_qz, stride_qh, stride_qm, stride_qk)
        print("N_HEAD:",N_HEAD)
        print("N_CTX:",N_CTX)
        print("BLOCK_DMODEL:", head_size)
        print("BLOCK_M1:",BLOCK_M1)
        print("BLOCK_N1:",BLOCK_N1)
        print("BLOCK_M2:",BLOCK_M2)
        print("BLOCK_N2:",BLOCK_N2)
        print("BLK_SLICE_FACTOR:", BLK_SLICE_FACTOR)
        print("USE_ALIBI:", use_alibi)
        print("USE_EXP2:", use_exp2)
        print("RCP_LN2:", RCP_LN2)
        print("LN2:", LN2)

    _attn_bwd[grid](
        q,
        k,
        v,
        sm_scale,
        alibi_slopes,
        do,
        dq,
        dk,
        dv,
        softmax_lse,
        delta,
        stride_qz, stride_qh, stride_qm, stride_qk,
        N_HEAD,
        N_CTX,
        BLOCK_DMODEL=head_size,
        BLOCK_M1=BLOCK_M1,
        BLOCK_N1=BLOCK_N1,
        BLOCK_M2=BLOCK_M2,
        BLOCK_N2=BLOCK_N2,
        BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,
        USE_ALIBI=use_alibi,
        USE_EXP2=use_exp2, 
        RCP_LN2=RCP_LN2,
        LN2=LN2
    )

    return dq, dk, dv, softmax_lse, None, None