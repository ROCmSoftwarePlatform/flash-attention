import torch
import triton
from .flash_attn_triton_kernel_prefill_amd import MetaData, get_shape_from_layout, _attention_prefill, attention_prefill
from .flash_attn_triton_kernel_decode_amd import attention_decode

DEBUG = False

class AttentionContext:
    def __init__(self, q, k, v, o, M, sm_scale, causal, alibi_slopes, dropout_p, BLOCK_DMODEL):
        self.saved_tensors = (q, k, v, o, M)
        self.sm_scale = sm_scale
        self.grid = lambda META: (triton.cdiv(q.shape[2], META['BLOCK_M']), q.shape[1], q.shape[0])
        self.causal = causal
        self.alibi_slopes = alibi_slopes
        self.dropout_p = dropout_p
        self.BLOCK_DMODEL = BLOCK_DMODEL
        self.philox_seed = 0x1BF52
        self.philox_offset = 0x1D4B42
        self.return_encoded_softmax = False

    def save_for_backward(self, q, k, v, o, M):
        self.saved_tensors = (q, k, v, o, M)

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
        print("alibi_slopes:", alibi_slopes)
        print("dropout_p:", dropout_p)
        print("softmax_scale:", softmax_scale)
        print("causal:", causal)
        print("window_size_left:", window_size_left)
        print("window_size_right:", window_size_right)
        print("softcap", softcap)
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
    tri_out, softmax_lse, softmax_dmask= attention_prefill(q, k, v, o, input_metadata)

    return tri_out, q , k , v, o, softmax_lse, softmax_dmask, None

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
    deterministic,
    gen_,
    rng_state,
):
    if DEBUG:
        print()
        print("flash_attn_triton_amd.py::bwd")
        print("dout:", dout, dout.shape, dout.stride())
        print("q:", q, q.shape, q.stride())
        print("k:", k, k.shape, k.stride())
        print("v:", v, v.shape, v.stride())
        print("softmax_lse:", softmax_lse)
        print("dq:", dq, dq.shape, dq.stride())
        print("dk:", dk, dk.shape, dk.stride())
        print("dv:", dv, dv.shape, dv.stride())
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

    if out is None:
        out = torch.empty_like(q)

    batch, max_seqlens_q, nheads_q,  head_size = q.shape

    # Transform inputs from bshd to bhsd layout
    dout_bhsd = dout.permute(0, 2, 1, 3).contiguous()
    q_bhsd = q.permute(0, 2, 1, 3).contiguous()
    k_bhsd = k.permute(0, 2, 1, 3).contiguous()
    v_bhsd = v.permute(0, 2, 1, 3).contiguous()
    out_bhsd = out.permute(0, 2, 1, 3).contiguous() if out is not None else None

    # Ensure all tensors have the same stride
    dout_bhsd = dout_bhsd.view(dout_bhsd.shape)
    q_bhsd = q_bhsd.view(q_bhsd.shape)
    k_bhsd = k_bhsd.view(k_bhsd.shape)
    v_bhsd = v_bhsd.view(v_bhsd.shape)
    out_bhsd = out_bhsd.view(out_bhsd.shape) if out_bhsd is not None else None


    ctx = AttentionContext(q_bhsd, k_bhsd, v_bhsd, out_bhsd, softmax_lse, softmax_scale, causal, alibi_slopes, dropout_p, head_size)
    dq, dk, dv, _, _ = _attention_prefill.backward(ctx, dout_bhsd, None) # expect bhsd

    softmax_d = None # not sure what softmax_d is supposed to be
    if DEBUG:
        print()
        print("bwd output")
        print("dq:", dq, dq.shape)
        print("dk:", dk, dk.shape)
        print("dv:", dv, dv.shape)
        print("softmax_d:", softmax_d)
        print()
    return dq, dk, dv, softmax_d
    


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
    raise ValueError("bwd is not supported on AMD yet")

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
        print("cu_seqlens_q:", cu_seqlens_q)
        print("cu_seqlens_k:", cu_seqlens_k)
        print("seqused_k:", seqused_k)
        print("leftpad_k:", leftpad_k)
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
        print("softcap", softcap)
        print("return_softmax:", return_softmax)
        print("gen_:", gen_)

    if dropout_p != 0.0:
        raise ValueError("dropout is not supported on AMD yet")
    
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

    tri_out, softmax_lse, softmax_dmask= attention_prefill(q, k, v, o, input_metadata)

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
    deterministic,
    gen_,
    rng_state,
):
    if DEBUG:
        print()
        print("flash_attn_triton_amd.py::varlen_bwd")

    raise ValueError("varlen_bwd is not supported on AMD yet")

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
    raise ValueError("varlen_bwd is not supported on AMD yet")

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
        print("cache_leftpad", cache_leftpad)
        print("block_table:", block_table, block_table.shape if block_table is not None else None)
        print("alibi_slopes:", alibi_slopes)
        print("out:", out)
        print("softmax_scale:", softmax_scale)
        print("causal:", causal)
        print("window_size_left:", window_size_left)
        print("window_size_right:", window_size_right)
        print("softcap", softcap)
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
    tri_out, softmax_lse = attention_decode(q, k_cache, v_cache, input_metadata)

    if DEBUG:
        print()
        print("tri_out:", tri_out, tri_out.shape)

    return tri_out, softmax_lse
