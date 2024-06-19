from .flash_attn_triton_kernel_amd import MetaData, attention, get_shape_from_layout, _attn_bwd_preprocess, _attn_bwd
import torch
import triton

DEBUG=False

def fwd(q,
        k,
        v,
        o,
        alibi_slopes,
        dropout_p,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        return_softmax,
        gen_):
    if DEBUG:
        print("flash_attn_triton_amd.py::fwd")
        print("q:", q.shape)
        print("k:", k.shape)
        print("v:", v.shape)
        print("alibi_slopes:", alibi_slopes)
        print("dropout_p:", dropout_p)
        print("softmax_scale:", softmax_scale)
        print("causal:", causal)
        print("window_size_left:", window_size_left)
        print("window_size_right:", window_size_right)
        print("return_softmax:", return_softmax)
        print("gen_:", gen_)

    if dropout_p != 0.0:
        raise ValueError("dropout is not supported on HIP")

    if o is None:
        o = torch.empty_like(q)

    # Setup metadata
    input_metadata = MetaData(sm_scale=softmax_scale)
    input_metadata.max_seqlens_q = q.shape[1]
    input_metadata.max_seqlens_k = k.shape[1]
    input_metadata.layout = "bshd"

    batch, nheads_q, nheads_k, head_size = get_shape_from_layout(q, k, input_metadata)
    
    if causal:
        input_metadata.need_causal()
    
    # if bias is not None:
    #     input_metadata.need_bias(bias, batch, nheads_q, input_metadata.max_seqlens_q, input_metadata.max_seqlens_k)
    
    if alibi_slopes is not None:
        input_metadata.need_alibi(alibi_slopes, batch, nheads_q)
    
    if dropout_p > 0.0:
        input_metadata.need_dropout(dropout_p, return_softmax)
    
    # Check arguments
    input_metadata.check_args(q, k, v, o)
    
    # Perform the forward attention computation
    tri_out, encoded_softmax = attention(q, k, v, o, input_metadata)

    softmax_lse = encoded_softmax
    softmax_p = encoded_softmax

    return tri_out, q , k , v, o, softmax_lse, softmax_p, torch.get_rng_state()

def varlen_fwd(
        q, 
        k, 
        v, 
        o,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_k,
        block_table_,
        alibi_slopes,\
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        zero_tensors,
        causal,
        window_size_left,
        window_size_right,
        return_softmax,
        gen_):
    
    if DEBUG:
        print("flash_attn_triton_amd.py::varlen_fwd")
        print("q:", q.shape)
        print("k:", k.shape)
        print("v:", v.shape)

    if dropout_p != 0.0:
        raise ValueError("dropout is not supported on HIP")
    
    if o is None:
        o = torch.empty_like(q)

    # Setup metadata
    input_metadata = MetaData(sm_scale=softmax_scale)
    input_metadata.set_varlen_params(cu_seqlens_q, cu_seqlens_k)

    # get shapes
    batch, nheads_q, nheads_k, head_size = get_shape_from_layout(q, k, input_metadata)

    if causal:
        input_metadata.need_causal()
    
    # if bias is not None:
    #     input_metadata.need_bias(bias, batch, nheads_q, q.shape[2], k.shape[2])
    
    if alibi_slopes is not None:
        input_metadata.need_alibi(alibi_slopes, batch, nheads_q)
    
    if dropout_p > 0.0:
        input_metadata.need_dropout(dropout_p, return_softmax)
    
    # Check arguments
    input_metadata.check_args(q, k, v, o)

    # Perform the forward attention computation
    tri_out, encoded_softmax = attention(q, k, v, o, input_metadata)

    softmax_lse = encoded_softmax
    softmax_p = encoded_softmax

    return tri_out, q , k , v, o, softmax_lse, softmax_p, torch.get_rng_state()

def fwd_kvcache( 
        q,
        k_cache,
        v_cache,
        k,
        v,
        cache_seqlens,
        rotary_cos,
        rotary_sin,
        cache_batch_idx,
        block_table,
        alibi_slopes,
        out,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        rotary_interleaved,
        num_splits):
    pass


def bwd(dout, q, k, v, out, softmax_lse, dq, dk, dv, alibi_slopes, dropout_p, softmax_scale,  causal, window_size_left,
        window_size_right, deterministic, gen_, rng_state):
    if DEBUG:
        print("flash_attn_triton_amd.py::bwd")
        print("q:", q.shape)
        print("k:", k.shape)
        print("v:", v.shape)
        print("softmax_lse:", softmax_lse)
        print("dq:", dq.shape)
        print("dk:", dk.shape)
        print("dv:", dv.shape)
        print("alibi_slopes:", alibi_slopes)
        print("dropout_p:", dropout_p)
        print("softmax_scale:", softmax_scale)
        print("causal:", causal)
        print("window_size_left:", window_size_left)
        print("window_size_right:", window_size_right)
        print("deterministic:", deterministic)
        print("gen_:", gen_)
        print("rng_state:", rng_state)
 
    if out is None:
        out = torch.empty_like(q)

    # Ensure the tensors have requires_grad=True
    q.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()
    out.requires_grad_()

    # Create metadata object
    metadata = MetaData(sm_scale=softmax_scale)
    metadata.max_seqlens_q = q.shape[1]
    metadata.max_seqlens_k = k.shape[1]
    metadata.layout = "bshd"

    if metadata == 'bshd':
        q = q.transpose(1, 2).clone()
        k = k.transpose(1, 2).clone()
        v = v.transpose(1, 2).clone()

    batch = q.shape[0]
    nheads_q = q.shape[1]
    BLOCK_DMODEL = q.shape[3]
    
    # Setup metadata
    if causal:
        metadata.need_causal()
    
    # if bias is not None:
    #     metadata.need_bias(bias, q.shape[0], q.shape[1], q.shape[2], k.shape[2])

    return_softmax = True
    if alibi_slopes is not None:
        metadata.need_alibi(alibi_slopes, batch, nheads_q)
    
    if dropout_p > 0.0:
        metadata.need_dropout(dropout_p, return_softmax)
    
    # Check arguments
    metadata.check_args(q, k, v, out)

    # write your own version backward
    M = torch.empty((batch, nheads_q, metadata.max_seqlens_q), device=q.device, dtype=torch.float32) # this passed from 

    if torch.version.hip is not None:
        BLOCK = 64
    else:
        BLOCK = 128
    o = out
    do = dout
    sm_scale = softmax_scale
    assert do.is_contiguous()
    assert q.stride() == k.stride() == v.stride() == o.stride() == do.stride()
    seqlen_q = q.shape[2]
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    BATCH, N_CTX, N_HEAD = q.shape[:3]
    PRE_BLOCK = 128
    # NUM_WARPS, NUM_STAGES = 4, 1
    BLOCK_M1, BLOCK_N1, BLOCK_M2, BLOCK_N2 = 32, 64, 64, 32
    BLK_SLICE_FACTOR = 2
    RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
    arg_k = k
    arg_k = arg_k * (sm_scale * RCP_LN2)
    if DEBUG:
        print("N_CTX:", N_CTX)
    # assert N_CTX % PRE_BLOCK == 0

    delta = torch.empty_like(M)
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
        D_HEAD=BLOCK_DMODEL,
    )
    grid = lambda META: (triton.cdiv(N_CTX, META['BLOCK_N1']), 1, BATCH * N_HEAD)
    _attn_bwd[grid](
        q,
        arg_k,
        v,
        sm_scale,
        alibi_slopes,
        do,
        dq,
        dk,
        dv,
        M,
        delta,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        N_HEAD,
        N_CTX,
        BLOCK_DMODEL= BLOCK_DMODEL,
        BLOCK_M1=BLOCK_M1,
        BLOCK_N1=BLOCK_N1,
        BLOCK_M2=BLOCK_M2,
        BLOCK_N2=BLOCK_N2,
        BLK_SLICE_FACTOR=BLK_SLICE_FACTOR,
        USE_ALIBI=False if alibi_slopes is None else True,
    )

    return dq, dk, dv, None


def varlen_bwd(dout, q, k, v, out, softmax_lse, dq, dk, dv, *args, **kwargs):
    pass