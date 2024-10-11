import torch
import triton
from .flash_attn_triton_kernel_prefill_amd import MetaData, get_shape_from_layout, attention_prefill_forward_triton_impl, attention_prefill_backward_triton_impl
from .flash_attn_triton_kernel_decode_amd import attention_decode

DEBUG = False

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
        softcap,
        return_softmax,
        gen_):
    
    if DEBUG:
        print()
        print("flash_attn_triton_amd.py::fwd")
        print("q:", q, q.shape)
        print("k:", k, k.shape)
        print("v:", v, v.shape)
        print("o:", o)
        print("alibi_slopes:", alibi_slopes)
        print("dropout_p:", dropout_p)
        print("softmax_scale:", softmax_scale)
        print("causal:", causal)
        print("window_size_left:", window_size_left)
        print("window_size_right:", window_size_right)
        print("softcap:", softcap)
        print("softcap:", softcap)
        print("return_softmax:", return_softmax)


    if dropout_p != 0.0:
        raise ValueError("dropout is not supported on AMD's Triton Backend yet")

    if o is None:
        o = torch.empty_like(q)

    # Setup metadata
    input_metadata = MetaData(sm_scale=softmax_scale)
    input_metadata.max_seqlens_q = q.shape[1]
    input_metadata.max_seqlens_k = k.shape[1]
    input_metadata.layout = "bshd"
    input_metadata.use_exp2 = False
    if return_softmax:
        input_metadata.return_encoded_softmax = True

    batch, nheads_q, nheads_k, head_size = get_shape_from_layout(q, k, input_metadata)
    
    if causal:
        input_metadata.need_causal()
    
    if alibi_slopes is not None:
        input_metadata.need_alibi(alibi_slopes, batch, nheads_q)
    
    if dropout_p > 0.0:
        input_metadata.need_dropout(dropout_p, return_softmax)
    
    # Check arguments
    input_metadata.check_args(q, k, v, o)
    o_triton, softmax_lse, exp_scores, grid, head_size, philox_seed, philox_offset, scores, scores_scaled_shifted = attention_prefill_forward_triton_impl(q, k, v, o, input_metadata)

    return o_triton, q , k , v, o, softmax_lse, exp_scores, None

def bwd(
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    dq,
    dk,
    dv,
    alibi_slopes,
    dropout_p,
    softmax_scale,
    causal,
    window_size_left,
    window_size_right,
    softcap,
    deterministic,
    gen_,
    rng_state,
):
    if DEBUG:
        print()
        print("flash_attn_triton_amd.py::bwd")
        print("dout:", dout, dout.shape)
        print("q:", q, q.shape)
        print("k:", k, k.shape)
        print("v:", v, v.shape)
        print("softmax_lse:", softmax_lse, softmax_lse.shape)
        print("dq:", dq, dq.shape)
        print("dk:", dk, dk.shape)
        print("dv:", dv, dv.shape)
        print("alibi_slopes:", alibi_slopes)
        print("dropout_p:", dropout_p)
        print("out:", out)
        print("softmax_scale:", softmax_scale)
        print("causal:", causal)
        print("window_size_left:", window_size_left)
        print("window_size_right:", window_size_right)
        print("deterministic:", deterministic)
        print("gen_:", gen_)
        print("rng_state:", rng_state)

    if dropout_p != 0.0:
        raise ValueError("dropout is not supported on AMD yet")

    batch, max_seqlens_q, nheads_q,  head_size = q.shape
    _, _, _, _, _, _ = attention_prefill_backward_triton_impl(
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        dq,
        dk,
        dv,
        softmax_scale,
        head_size,
        alibi_slopes,
        causal,
        "bshd",
        False,
        True,
        True,
    )

    softmax_d = None # fill this in
    if False:
        print()
        print("bwd output")
        print("dq:", dq, dq.shape)
        print("dk:", dk, dk.shape)
        print("dv:", dv, dv.shape)
        print("softmax_d:", softmax_d)
        print()
    return dq, dk, dv, softmax_d

def varlen_fwd(
        q, 
        k, 
        v, 
        o,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_k,
        leftpad_k,
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
        softcap,
        return_softmax,
        gen_):

    if dropout_p != 0.0:
        raise ValueError("dropout is not supported on AMD's Triton Backend yet")
    
    if o is None:
        o = torch.empty_like(q)

    # Setup metadata
    input_metadata = MetaData(sm_scale=softmax_scale)
    if return_softmax:
        input_metadata.return_encoded_softmax = True
    input_metadata.set_varlen_params(cu_seqlens_q, cu_seqlens_k)  # set layout to "thd" and other metdata

    # get shapes
    batch, nheads_q, nheads_k, head_size = get_shape_from_layout(q, k, input_metadata)

    if causal:
        input_metadata.need_causal()

    if alibi_slopes is not None:
        input_metadata.need_alibi(alibi_slopes, batch, nheads_q)
    
    if dropout_p > 0.0:
        input_metadata.need_dropout(dropout_p, return_softmax)
    
    # Check arguments
    input_metadata.check_args(q, k, v, o)

    tri_out, softmax_lse, softmax_dmask = attention_prefill_forward_triton_impl(q, k, v, o, input_metadata)

    return tri_out, q , k , v, o, softmax_lse, softmax_dmask, None

def varlen_bwd(
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    dq,
    dk,
    dv,
    cu_seqlens_q,
    cu_seqlens_k,
    alibi_slopes,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p,
    softmax_scale,
    zero_tensors,
    causal,
    window_size_left,
    window_size_right,
    softcap,
    deterministic,
    gen_,
    rng_state,
):
    raise ValueError("varlen_bwd is not supported on AMD's Triton Backend yet")

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
        cache_leftpad,
        block_table,
        alibi_slopes,
        out,
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        softcap,
        rotary_interleaved,
        num_splits):

    if out is None:
        out = torch.empty_like(q)

    # fill metadata
    input_metadata = MetaData(sm_scale=softmax_scale)
    input_metadata.layout = "bshd"
    input_metadata.max_seqlens_q = q.shape[1]
    input_metadata.max_seqlens_k = k_cache.shape[1]
    input_metadata.cache_seqlens = cache_seqlens
    input_metadata.cache_batch_idx = cache_batch_idx

    if k is not None and v is not None:
        input_metadata.new_kv = True
        input_metadata.seqlen_new = k.shape[1]
        input_metadata.k_new = k
        input_metadata.v_new = v

    if causal:
        input_metadata.need_causal()

    if alibi_slopes is not None:
        batch, _ , nheads_q, _= q.shape
        input_metadata.need_alibi(alibi_slopes, batch, nheads_q)

    # launch kernel
    tri_out, softmax_lse = attention_decode(q, k_cache, v_cache, input_metadata)
    return tri_out, softmax_lse
