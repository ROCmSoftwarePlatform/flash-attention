import torch
import triton
from .flash_attn_triton_kernel_prefill_amd import MetaData, attention_prefill, get_shape_from_layout, _attn_bwd_preprocess, _attn_bwd
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
        raise ValueError("dropout is not supported on AMD yet")

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
    tri_out, encoded_softmax = attention_prefill(q, k, v, o, input_metadata)

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
        print("cu_seqlens_q:", cu_seqlens_q)
        print("cu_seqlens_k:", cu_seqlens_k)
        print("block_table_:", block_table_)
        print("alibi_slopes:", alibi_slopes)
        print("max_seqlen_q:", max_seqlen_q)
        print("max_seqlen_k:", max_seqlen_k)
        print("dropout_p:", dropout_p)
        print("softmax_scale:", softmax_scale)
        print("zero_tensors:", zero_tensors)
        print("causal:", causal)
        print("window_size_left:", window_size_left)
        print("window_size_right:", window_size_right)
        print("return_softmax:", return_softmax)
        print("gen_:", gen_)

    if dropout_p != 0.0:
        raise ValueError("dropout is not supported on AMD yet")
    
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
    tri_out, encoded_softmax = attention_prefill(q, k, v, o, input_metadata)

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
    
    if DEBUG:
        print()
        print("flash_attn_triton_amd.py::fwd_kvcache")
        print("q:", q, q.shape)
        print("k_cache:", k_cache, k_cache.shape)
        print("v_cache:", v_cache, v_cache.shape)
        print("k:", k, k.shape if k is not None else None)
        print("v:", v, v.shape if v is not None else None)
        print("cache_seqlens:", cache_seqlens, cache_seqlens.size())
        print("rotary_cos:", rotary_cos)
        print("rotary_sin:", rotary_sin)
        print("cache_batch_idx:", cache_batch_idx)
        print("block_table:", block_table, block_table.shape if block_table is not None else None)
        print("alibi_slopes:", alibi_slopes)
        print("out:", out)
        print("softmax_scale:", softmax_scale)
        print("causal:", causal)
        print("window_size_left:", window_size_left)
        print("window_size_right:", window_size_right)
        print("rotary_interleaved:", rotary_interleaved)
        print("num_splits:", num_splits)
    
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
    tri_out = attention_decode(q, k_cache, v_cache, input_metadata)

    if DEBUG:
        print()
        print("tri_out:", tri_out, tri_out.shape)

    return tri_out, None


def bwd(dout, q, k, v, out, softmax_lse, dq, dk, dv, alibi_slopes, dropout_p, softmax_scale,  causal, window_size_left,
        window_size_right, deterministic, gen_, rng_state):
    raise ValueError("bwd is not supported on AMD yet")


def varlen_bwd(dout, q, k, v, out, softmax_lse, dq, dk, dv, *args, **kwargs):
    raise ValueError("varlen_bwd is not supported on AMD yet")