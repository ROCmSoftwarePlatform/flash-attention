import torch
from .fwd_prefill import attention_prefill_forward_triton_impl
from .bwd_prefill import attention_prefill_backward_triton_impl
from .fwd_decode import attention_decode_forward_triton_impl
from .fwd_ref import attention_forward_pytorch_ref_impl
from .bwd_ref import attention_backward_pytorch_ref_impl
from .utils import MetaData, get_shape_from_layout

DEBUG = False
USE_REF = True

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
    metadata = MetaData(sm_scale=softmax_scale)
    metadata.max_seqlens_q = q.shape[1]
    metadata.max_seqlens_k = k.shape[1]
    metadata.layout = "bshd"
    metadata.use_exp2 = False
    if return_softmax:
        metadata.return_encoded_softmax = True

    batch, nheads_q, nheads_k, head_size, _, _ = get_shape_from_layout(q, k, metadata.layout)
    
    if causal:
        metadata.need_causal()
    
    if alibi_slopes is not None:
        metadata.need_alibi(alibi_slopes, batch, nheads_q)
    
    if dropout_p > 0.0:
        metadata.need_dropout(dropout_p, return_softmax)
    
    # Check arguments
    metadata.check_args(q, k, v, o)
    if USE_REF:
        (output, 
        softmax_lse, 
        exp_scores, 
        _, 
        _,
        _, 
        _) = attention_forward_pytorch_ref_impl(
                                                q, 
                                                k, 
                                                v,
                                                metadata.sm_scale, 
                                                metadata.causal,
                                                metadata.layout, 
                                                metadata.cu_seqlens_q, 
                                                metadata.cu_seqlens_k,
                                                metadata.max_seqlens_q, 
                                                metadata.max_seqlens_k,
                                                metadata.use_exp2)
        o.copy_(output)
    else:
        (output, 
        softmax_lse, 
        exp_scores, 
        _, 
        _, 
        _, 
        _, 
        _, 
        _) = attention_prefill_forward_triton_impl(
                                                q, 
                                                k, 
                                                v, 
                                                o, 
                                                metadata.sm_scale, 
                                                metadata.alibi_slopes, 
                                                metadata.causal, 
                                                metadata.bias, 
                                                metadata.dropout_p, 
                                                metadata.layout, 
                                                metadata.cu_seqlens_q, 
                                                metadata.cu_seqlens_k,
                                                metadata.max_seqlens_q, 
                                                metadata.max_seqlens_k, 
                                                metadata.return_scores, 
                                                metadata.use_exp2)

    if DEBUG:
        print("fwd outputs")
        print("output:", output, output.shape)
        print("q:", q, q.shape)
        print("k:", k, k.shape)
        print("v:", v, v.shape)
        print("o:", o, o.shape)
        print("softmax_lse:", softmax_lse, softmax_lse.shape)
        print("exp_scores:", exp_scores, exp_scores.shape)

    return output, q , k , v, o, softmax_lse, exp_scores, None

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
        print("out:", out, out.shape)
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

    if USE_REF:
        dq_ref, dk_ref, dv_ref, delta_ref = attention_backward_pytorch_ref_impl(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            softmax_scale,
            causal,
            "bshd",
            None,
            None,
            None,
            None,
            False,
        )
        dq.copy_(dq_ref)
        dk.copy_(dk_ref)
        dv.copy_(dv_ref)
    else:
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
            alibi_slopes,
            causal,
            "bshd",
            None,
            None,
            None,
            None,
            False,
        )

    return dq, dk, dv, None

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

    if DEBUG:
        print()
        print("flash_attn_triton_amd.py::varlen_fwd")
        print("q:", q, q.shape)
        print("k:", k, k.shape)
        print("v:", v, v.shape)
        print("cu_seqlens_q:", cu_seqlens_q, cu_seqlens_q.shape)
        print("cu_seqlens_k:", cu_seqlens_k, cu_seqlens_k.shape)
        print("alibi_slopes:", alibi_slopes)
        print("max_seqlen_q:", max_seqlen_q)
        print("max_seqlen_k:", max_seqlen_k)
        print("dropout_p:", dropout_p)
        print("softmax_scale:", softmax_scale)
        print("causal:", causal)
        print("window_size_left:", window_size_left)
        print("window_size_right:", window_size_right)
        print("gen_:", gen_)

    if dropout_p != 0.0:
        raise ValueError("dropout is not supported on AMD's Triton Backend yet")
    
    if o is None:
        o = torch.empty_like(q)

    # Setup metadata
    metadata = MetaData(sm_scale=softmax_scale)
    metadata.use_exp2 = False
    if return_softmax:
        metadata.return_encoded_softmax = True
    metadata.set_varlen_params(cu_seqlens_q, cu_seqlens_k)  # set layout to "thd" and other metdata

    # get shapes
    batch, nheads_q, nheads_k, head_size , seqlen_q, seqlen_k = get_shape_from_layout(q, k, metadata.layout, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)

    if causal:
        metadata.need_causal()

    if alibi_slopes is not None:
        metadata.need_alibi(alibi_slopes, batch, nheads_q)
    
    if dropout_p > 0.0:
        metadata.need_dropout(dropout_p, return_softmax)
    
    # Check arguments
    metadata.check_args(q, k, v, o)
    if o is None:
        o = torch.empty_like(q, dtype=v.dtype)

    if USE_REF:
        (o_triton, 
        softmax_lse, 
        exp_scores, 
        _, 
        _,
        _, 
        _) = attention_forward_pytorch_ref_impl(
                                                q, 
                                                k, 
                                                v,
                                                metadata.sm_scale, 
                                                metadata.causal,
                                                metadata.layout, 
                                                metadata.cu_seqlens_q, 
                                                metadata.cu_seqlens_k,
                                                metadata.max_seqlens_q, 
                                                metadata.max_seqlens_k,
                                                metadata.use_exp2)
    else:
        (o_triton, 
        softmax_lse, 
        exp_scores, 
        _, 
        _, 
        _, 
        _, 
        _, 
        _) = attention_prefill_forward_triton_impl(
                                                            q, 
                                                            k, 
                                                            v, 
                                                            o, 
                                                            metadata.sm_scale, 
                                                            metadata.alibi_slopes, 
                                                            metadata.causal, 
                                                            metadata.bias, 
                                                            metadata.dropout_p, 
                                                            metadata.layout, 
                                                            metadata.cu_seqlens_q, 
                                                            metadata.cu_seqlens_k,
                                                            metadata.max_seqlens_q, 
                                                            metadata.max_seqlens_k, 
                                                            metadata.return_scores, 
                                                            metadata.use_exp2)

    return o_triton, q , k , v, o, softmax_lse, exp_scores, None

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
    if DEBUG:
        print()
        print("varlen_bwd")
        print("dout:", dout, dout.shape)
        print("q:", q, q.shape)
        print("k:", k, k.shape)
        print("v:", v, v.shape)
        print("softmax_lse:", softmax_lse, softmax_lse.shape)
        print("dq:", dq, dq.shape)
        print("dk:", dk, dk.shape)
        print("dv:", dv, dv.shape)
        print("cu_seqlens_q:", cu_seqlens_q, cu_seqlens_q.shape)
        print("cu_seqlens_k:", cu_seqlens_k, cu_seqlens_k.shape)
        print("alibi_slopes:", alibi_slopes)
        print("max_seqlen_q:", max_seqlen_q)
        print("max_seqlen_k:", max_seqlen_k)
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

    if USE_REF:
        dq_ref, dk_ref, dv_ref, delta_ref = attention_backward_pytorch_ref_impl(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            softmax_scale,
            causal,
            "thd",
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            False,
        )
        dq.copy_(dq_ref)
        dk.copy_(dk_ref)
        dv.copy_(dv_ref)
    else:
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
            alibi_slopes,
            causal,
            "thd",
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            False,
        )

    if DEBUG:
        print("varlen_bwd outputs")
        print("delta_ref:", delta_ref, delta_ref.shape)
        print("dv:", dv, dv.shape)
        print("dk:", dk, dk.shape)
        print("dq:", dq, dq.shape)

    return dq, dk, dv, delta_ref

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
    tri_out, softmax_lse = attention_decode_forward_triton_impl(q, k_cache, v_cache, input_metadata)
    return tri_out, softmax_lse
