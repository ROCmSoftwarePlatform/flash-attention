"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm
See https://tridao.me/publications/flash2/flash2.pdf

Credits:
AMD Triton kernels team
OpenAI kernel team

Currently only the forward kernel is supported, and contains these features:

1) Fwd with causal masking
2) Arbitrary Q and KV sequence lengths
3) Arbitrary head sizes
4) Multi and grouped query attention
5) Variable sequence lengths
6) ALiBi and matrix bias

"""

import argparse
import math
import pytest
import sys
import torch

import triton
import triton.language as tl

from triton import cdiv
from .bwd_new import attention_prefill_backward_new_impl
from .bwd_oai import attention_prefill_backward_oai_impl


DEBUG = True
RCP_LN2 = 1.4426950408889634  # = 1.0 / ln(2)
LN2= 0.6931471824645996  # = ln(2)


class MetaData():
    cu_seqlens_q = None
    cu_seqlens_k = None
    max_seqlens_q = 0
    max_seqlens_k = 0
    bias = None
    alibi_slopes = None
    causal = False
    num_contexts = 0
    varlen = False
    layout = None
    cache_seqlens = None
    cache_batch_idx = None
    new_kv = False
    seqlen_new = None
    k_new = None
    v_new = None
    dropout_p, return_scores= 0.0, False
    # NOTE: scale sm_scale by log_2(e) and use 2^x in the loop as we do not have native e^x support in HW.
    use_exp2 = True
    

    def __repr__(self) -> str:
        return (f"MetaData(\n"
                f"  sm_scale={self.sm_scale},\n"
                f"  cu_seqlens_q={self.cu_seqlens_q},\n"
                f"  cu_seqlens_k={self.cu_seqlens_k},\n"
                f"  max_seqlens_q={self.max_seqlens_q},\n"
                f"  max_seqlens_k={self.max_seqlens_k},\n"
                f"  bias={self.bias},\n"
                f"  alibi_slopes={self.alibi_slopes},\n"
                f"  causal={self.causal},\n"
                f"  num_contexts={self.num_contexts},\n"
                f"  varlen={self.varlen},\n"
                f"  layout={self.layout},\n"
                f"  cache_seqlens={self.cache_seqlens},\n"
                f"  cache_batch_idx={self.cache_batch_idx},\n"
                f"  new_kv={self.new_kv},\n"
                f"  seqlen_new={self.seqlen_new},\n"
                f"  k_new={self.k_new},\n"
                f"  v_new={self.v_new},\n"
                f"  dropout_p={self.dropout_p},\n"
                f"  return_scores={self.return_scores}\n"
                f")")

    def __init__(self, sm_scale=1.0):
        self.sm_scale = sm_scale

    def set_varlen_params(self, cu_seqlens_q, cu_seqlens_k):
        self.varlen = True
        self.layout = 'thd'
        self.cu_seqlens_q = cu_seqlens_q
        self.cu_seqlens_k = cu_seqlens_k
        # Without "varlen", there should still be one sequence.
        assert len(cu_seqlens_q) >= 2
        assert len(cu_seqlens_q) == len(cu_seqlens_k)
        self.num_contexts = len(cu_seqlens_q) - 1
        for i in range(0, self.num_contexts):
            self.max_seqlens_q = max(cu_seqlens_q[i + 1].item() - cu_seqlens_q[i].item(), self.max_seqlens_q)
            self.max_seqlens_k = max(cu_seqlens_k[i + 1].item() - cu_seqlens_k[i].item(), self.max_seqlens_k)

    def need_bias(self, bias, batch, nheads, seqlen_q, seqlen_k):
        assert bias.is_cuda
        assert bias.dim() == 4
        assert bias.shape[0] == 1
        assert bias.shape[2:] == (seqlen_q, seqlen_k)
        self.bias = bias

    def need_alibi(self, alibi_slopes, batch, nheads):
        assert alibi_slopes.is_cuda
        assert alibi_slopes.dim() == 2
        assert alibi_slopes.shape[0] == batch
        assert alibi_slopes.shape[1] == nheads
        self.alibi_slopes = alibi_slopes

    def need_causal(self):
        self.causal = True

    def need_dropout(self, dropout_p, return_scores):
        self.dropout_p = dropout_p
        self.return_scores = return_scores

    def check_args(self, q, k, v, o):
        assert q.dim() == k.dim() and q.dim() == v.dim()

        batch, nheads_q, nheads_k, head_size = get_shape_from_layout(q, k, self)
        if self.varlen:
            assert q.dim() == 3
            assert self.cu_seqlens_q is not None
            assert self.cu_seqlens_k is not None
            assert len(self.cu_seqlens_q) == len(self.cu_seqlens_k)
            # TODO: Remove once bias is supported with varlen
            assert self.bias is None
            # TODO:Remove once dropout is supported with varlen
            assert self.dropout_p == 0.0
            # assert not self.return_scores
        else:
            assert q.dim() == 4
            assert self.max_seqlens_q > 0 and self.max_seqlens_k > 0
            assert self.cu_seqlens_q is None and self.cu_seqlens_k is None
        assert k.shape == v.shape
        assert q.shape[-1] == k.shape[-1] and q.shape[-1] == v.shape[-1]
        # TODO: Change assert if we support qkl f8 and v f16
        assert q.dtype == k.dtype and q.dtype == v.dtype
        assert head_size <= 256
        assert o.shape == q.shape
        assert (nheads_q % nheads_k) == 0
        assert self.layout is not None
        assert self.layout == 'thd' or not self.varlen


@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y


@triton.jit
def max_fn(x, y):
    return tl.math.max(x, y)


@triton.jit
def dropout_offsets(philox_seed, philox_offset, dropout_p, m, n, stride):
    ms = tl.arange(0, m)
    ns = tl.arange(0, n)
    return philox_offset + ms[:, None] * stride + ns[None, :]


@triton.jit
def dropout_rng(philox_seed, philox_offset, dropout_p, m, n, stride):
    rng_offsets = dropout_offsets(philox_seed, philox_offset, dropout_p, m, n, stride).to(tl.uint32)
    # TODO: use tl.randint for better performance
    return tl.rand(philox_seed, rng_offsets)


@triton.jit
def dropout_mask(philox_seed, philox_offset, dropout_p, m, n, stride):
    rng_output = dropout_rng(philox_seed, philox_offset, dropout_p, m, n, stride)
    rng_keep = rng_output > dropout_p
    return rng_keep


# Convenience function to load with optional boundary checks.
# "First" is the major dim, "second" is the minor dim.
@triton.jit
def load_fn(ptrs, offset_first, offset_second, boundary_first, boundary_second):
    if offset_first is not None and offset_second is not None:
        mask = (offset_first[:, None] < boundary_first) & \
               (offset_second[None, :] < boundary_second)
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    elif offset_first is not None:
        mask = offset_first[:, None] < boundary_first
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    elif offset_second is not None:
        mask = offset_second[None, :] < boundary_second
        tensor = tl.load(ptrs, mask=mask, other=0.0)
    else:
        tensor = tl.load(ptrs)
    return tensor


@triton.jit
def compute_alibi_block(alibi_slope, seqlen_q, seqlen_k, offs_m, offs_n, transpose=False):
    # when seqlen_k and seqlen_q are different we want the diagonal to stick to the bottom right of the attention matrix
    # for casual mask we want something like this where (1 is kept and 0 is masked)
    # seqlen_q = 2 and seqlen_k = 5
    #   1 1 1 1 0
    #   1 1 1 1 1
    # seqlen_q = 5 and seqlen_k = 2
    #        0 0
    #        0 0
    #        0 0
    #        1 0
    #        1 1
    # for alibi the diagonal is 0 indicating no penalty for attending to that spot and increasing penalty for attending further from the diagonal
    # e.g. alibi_slope = 1, seqlen_q = 2, seqlen_k = 5, offs_m = [0, 1, 2, 3], offs_n = [0, 1, 2, 3, 4], transpose = False
    # 1. offs_m[:,None] = [[0],
    #                       [1],
    # 2. offs_m[:,None] + seqlen_k = [[5],
    #                                  [6],
    # 3. offs_m[:,None] + seqlen_k - seqlen_q = [[3],
    #                                             [4],
    # 4. offs_m[:,None] + seqlen_k - seqlen_q - offs_n[None,:] = [[3], - [[0, 1, 2, 3, 4]] =  [[ 3, 2, 1, 0,-1],
    #                                                            [4],                           [ 4, 3, 2, 1, 0]]
    # 5. -1 * alibi_slope * tl.abs(relative_pos_block) = [[ -3, -2, -1, 0,-1],
    #                                                     [ -4, -3, -2, -1, 0]],
    relative_pos_block = offs_m[:, None] + seqlen_k - seqlen_q - offs_n[None, :]
    alibi_block = -1 * alibi_slope * tl.abs(relative_pos_block)
    if transpose:
        return alibi_block.T
    else:
        return alibi_block


def compute_alibi_tensor(alibi_slopes, seqlen_q, seqlen_k):
    q_idx = torch.arange(seqlen_q, dtype=torch.int32, device="cuda").unsqueeze(-1)  # (N_CTX_Q, 1)
    k_idx = torch.arange(seqlen_k, dtype=torch.int32, device="cuda").unsqueeze(0)  # (1, N_CTX_K)
    relative_pos = torch.abs(q_idx + seqlen_k - seqlen_q - k_idx)  # (N_CTX_Q, N_CTX_K)
    return -1 * alibi_slopes.unsqueeze(-1).unsqueeze(-1) * relative_pos  # (Z, H, N_CTX_Q, N_CTX_K)


@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q, k_ptrs, v_ptrs, bias_ptrs, stride_kn, stride_vk, stride_bn, start_m,
                    actual_seqlen_k, actual_seqlen_q, dropout_p, philox_seed, batch_philox_offset, exp_scores_ptrs,
                    block_min, block_max, offs_n_causal, masked_blocks, n_extra_tokens, alibi_slope, score_ptrs, scores_scaled_shifted_ptrs,
                    IS_CAUSAL: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
                    OFFS_M: tl.constexpr, OFFS_N: tl.constexpr, PRE_LOAD_V: tl.constexpr, MASK_STEPS: tl.constexpr,
                    ENABLE_DROPOUT: tl.constexpr, PADDED_HEAD: tl.constexpr,
                    ACTUAL_BLOCK_DMODEL: tl.constexpr, sm_scale: tl.constexpr, USE_EXP2: tl.constexpr, RCP_LN2: tl.constexpr,
                    RETURN_SCORES: tl.constexpr):
    # loop over k, v, and update accumulator
    for start_n in range(block_min, block_max, BLOCK_N):
        # For padded blocks, we will overrun the tensor size if
        # we load all BLOCK_N. For others, the blocks are all within range.
        if MASK_STEPS:
            k_offs_n = start_n + tl.arange(0, BLOCK_N)
        else:
            k_offs_n = None
        k_offs_k = None if not PADDED_HEAD else tl.arange(0, BLOCK_DMODEL)
        k = load_fn(k_ptrs, k_offs_k, k_offs_n, ACTUAL_BLOCK_DMODEL, actual_seqlen_k)
        if PRE_LOAD_V:
            # We can use the same offsets as k, just with dims transposed.
            v = load_fn(v_ptrs, k_offs_n, k_offs_k, actual_seqlen_k, ACTUAL_BLOCK_DMODEL)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        # We start from end of seqlen_k so only the first iteration would need
        # to be checked for padding if it is not a multiple of block_n
        # TODO: This can be optimized to only be true for the padded block.
        if MASK_STEPS:
            # If this is the last block / iteration, we want to
            # mask if the sequence length is not a multiple of block size
            # a solution is to always do BLOCK_M // BLOCK_N + 1 steps if not is_modulo_mn.
            # last step might get wasted but that is okay. check if this masking works For
            # that case.
            if (start_n + BLOCK_N == block_max) and (n_extra_tokens != 0):
                boundary_m = tl.full([BLOCK_M], actual_seqlen_k, dtype=tl.int32)
                size_n = start_n + OFFS_N[None, :]
                mask = size_n < boundary_m[:, None]
                qk = tl.where(mask, qk, float("-inf"))
        
        # -- compute qk ----
        qk += tl.dot(q, k)
        if RETURN_SCORES:
            score_mask = (OFFS_M[:, None] < actual_seqlen_q) & ((start_n + tl.arange(0, BLOCK_N))[None, :] < actual_seqlen_k)
            tl.store(score_ptrs, qk, mask=score_mask)

        if IS_CAUSAL:
            causal_boundary = start_n + offs_n_causal
            causal_mask = OFFS_M[:, None] >= causal_boundary[None, :]
            qk = tl.where(causal_mask, qk, float("-inf"))
        if bias_ptrs is not None:
            bias_offs_n = start_n + tl.arange(0, BLOCK_N) if MASK_STEPS else None
            bias = load_fn(bias_ptrs, OFFS_M, bias_offs_n, actual_seqlen_q, actual_seqlen_k)
            if USE_EXP2:
                # While bias is added after multiplying qk with sm_scale,
                # our optimization to use 2^x instead of e^x results in an additional
                # scale factor of log2(e) which we must also multiply the bias with.
                qk += (bias * RCP_LN2)
            else:
                qk += bias

        if alibi_slope is not None:
            # Compute the global position of each token within the sequence
            global_m_positions = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
            global_n_positions = start_n + tl.arange(0, BLOCK_N)
            alibi_block = compute_alibi_block(alibi_slope, actual_seqlen_q, actual_seqlen_k, global_m_positions,
                                              global_n_positions)
            if USE_EXP2:
                 qk += (alibi_block * RCP_LN2)  # scale factor of log2(e)
            else:
                qk += alibi_block
        
        # get max scores so far
        m_ij = tl.maximum(m_i, tl.max(qk, 1))

        # scale and subtract max
        qk = qk - m_ij[:, None]
        if RETURN_SCORES: 
            # NOTE: the returned score is not the same as the reference because we need to adjust as we find new maxes per block. We are not doing that
            scores_scaled_shifted_mask = (OFFS_M[:, None] < actual_seqlen_q) & ((start_n + tl.arange(0, BLOCK_N))[None, :] < actual_seqlen_k)
            tl.store(scores_scaled_shifted_ptrs, qk, mask=scores_scaled_shifted_mask)
        
        # shift scores to 0 and take exponential of scores
        if USE_EXP2:
            p = tl.math.exp2(qk)
        else:
            p = tl.math.exp(qk)

        # CAVEAT: Must update l_ij before applying dropout
        l_ij = tl.sum(p, 1)
        if ENABLE_DROPOUT:
            philox_offset = batch_philox_offset + start_m * BLOCK_M * actual_seqlen_k + start_n - BLOCK_N
            keep = dropout_mask(philox_seed, philox_offset, dropout_p, BLOCK_M, BLOCK_N, actual_seqlen_k)
            if RETURN_SCORES:
                # NOTE: the returned score is not the same as the reference because we need to adjust as we find new maxes per block. We are not doing that
                exp_score_mask = (OFFS_M[:, None] < actual_seqlen_q) & ((start_n + tl.arange(0, BLOCK_N))[None, :] < actual_seqlen_k)
                tl.store(exp_scores_ptrs, tl.where(keep, p, -p), mask=exp_score_mask)
            p = tl.where(keep, p, 0.0)
        elif RETURN_SCORES:
            # NOTE: the returned score is not the same as the reference because we need to adjust as we find new maxes per block. We are not doing that
            exp_score_mask = (OFFS_M[:, None] < actual_seqlen_q) & ((start_n + tl.arange(0, BLOCK_N))[None, :] < actual_seqlen_k)
            tl.store(exp_scores_ptrs, p, mask=exp_score_mask)
        
        # -- update output accumulator --
        # alpha is an adjustment factor for acc and li as we loop and find new maxes
        # store the diff in maxes to adjust acc and li as we discover new maxes
        m_diff = m_i - m_ij
        if USE_EXP2:
            alpha = tl.math.exp2(m_diff)
        else:
            alpha = tl.math.exp(m_diff)
        acc = acc * alpha[:, None]
        if not PRE_LOAD_V:
            v = load_fn(v_ptrs, k_offs_n, k_offs_k, actual_seqlen_k, ACTUAL_BLOCK_DMODEL)
        # -- update m_i and l_i
        l_i = l_i * alpha + l_ij
        # update m_i and l_i
        m_i = m_ij
        acc += tl.dot(p.to(v.type.element_ty), v)
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vk
        if bias_ptrs is not None:
            bias_ptrs += BLOCK_N * stride_bn
        if RETURN_SCORES:
            score_ptrs += BLOCK_N
            scores_scaled_shifted_ptrs += BLOCK_N
            exp_scores_ptrs += BLOCK_N
    return acc, l_i, m_i


@triton.autotune(
    configs=[
        # triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'waves_per_eu': 2, 'PRE_LOAD_V': False}, num_stages=1,
        #               num_warps=8),
        # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'waves_per_eu': 2, 'PRE_LOAD_V': False}, num_stages=1,
        #               num_warps=4),
        # triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'waves_per_eu': 2, 'PRE_LOAD_V': False}, num_stages=1,
        #               num_warps=8),
        # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 3, 'PRE_LOAD_V': True}, num_stages=1,
        #               num_warps=4),
        # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 3, 'PRE_LOAD_V': False}, num_stages=1,
        #               num_warps=4),
        # triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'waves_per_eu': 4, 'PRE_LOAD_V': False}, num_stages=1,
        #               num_warps=8),
        # triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 1, 'PRE_LOAD_V': False}, num_stages=1,
        #               num_warps=4),
        # triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'waves_per_eu': 4, 'PRE_LOAD_V': False}, num_stages=1,
        #               num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'waves_per_eu': 1, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=4),
        # TODO: This configs fails with head_size not pow2 with data mismatches. figure out why
        # triton.Config({'BLOCK_M': 32, 'BLOCK_N': 16, 'waves_per_eu': 1, 'PRE_LOAD_V': False}, num_stages=1, num_warps=4),
        # triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'waves_per_eu': 1, 'PRE_LOAD_V': False}, num_stages=1, num_warps=4),
    ],
    key=['IS_CAUSAL', 'dropout_p', 'BLOCK_DMODEL'],
    use_cuda_graph=True,
)
@triton.jit
def attn_fwd(Q, K, V, bias, sm_scale, LSE, Out, stride_qz, stride_qh, stride_qm, stride_qk, 
             stride_kz, stride_kh, stride_kn, stride_kk, stride_vz, stride_vh, stride_vk, stride_vn, 
             stride_oz, stride_oh, stride_om, stride_on, stride_bz, stride_bh, stride_bm, stride_bn, stride_az, stride_ah,
             stride_sz, stride_sh, stride_sm, stride_sn, cu_seqlens_q, cu_seqlens_k,
             dropout_p, philox_seed, philox_offset_base, scores, scores_scaled_shifted, exp_scores, alibi_slopes,  HQ: tl.constexpr,
             HK: tl.constexpr, ACTUAL_BLOCK_DMODEL: tl.constexpr, MAX_SEQLENS_Q: tl.constexpr,
             MAX_SEQLENS_K: tl.constexpr, VARLEN: tl.constexpr, IS_CAUSAL: tl.constexpr, BLOCK_M: tl.constexpr,
             BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr, PRE_LOAD_V: tl.constexpr, USE_BIAS: tl.constexpr,
             ENABLE_DROPOUT: tl.constexpr, return_scores: tl.constexpr, USE_ALIBI: tl.constexpr,
             USE_EXP2: tl.constexpr, LN2: tl.constexpr, RCP_LN2: tl.constexpr, RETURN_SCORES: tl.constexpr):
    start_m = tl.program_id(0)
    off_h_q = tl.program_id(1)
    off_z = tl.program_id(2)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    if VARLEN:
        cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
        cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)
        seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
        # We have a one-size-fits-all grid in id(0). Some seqlens might be too
        # small for all start_m so for those we return early.
        if start_m * BLOCK_M > seqlen_q:
            return
        cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
        cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)
        seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
    else:
        cu_seqlens_q_start = 0
        cu_seqlens_k_start = 0
        seqlen_q = MAX_SEQLENS_Q
        seqlen_k = MAX_SEQLENS_K

    # Now we compute whether we need to exit early due to causal masking.
    # This is because for seqlen_q > seqlen_k, M rows of the attn scores
    # are completely masked, resulting in 0s written to the output, and
    # inf written to LSE. We don't need to do any GEMMs in this case.
    # This block of code determines what N is, and if this WG is operating
    # on those M rows.
    n_blocks = cdiv_fn(seqlen_k, BLOCK_N)
    if (IS_CAUSAL):
        # If seqlen_q == seqlen_k, the attn scores are a square matrix.
        # If seqlen_q != seqlen_k, attn scores are rectangular which means
        # the causal mask boundary is bottom right aligned, and ends at either
        # the top edge (seqlen_q < seqlen_k) or left edge.
        # This captures the decrease in n_blocks if we have a rectangular attn matrix
        n_blocks_seqlen = cdiv_fn((start_m + 1) * BLOCK_M + seqlen_k - seqlen_q, BLOCK_N)
        # This is what adjusts the block_max for the current WG, only
        # if IS_CAUSAL. Otherwise we want to always iterate through all n_blocks
        n_blocks = min(n_blocks, n_blocks_seqlen)
        # If we have no blocks after adjusting for seqlen deltas, this WG is part of
        # the blocks that are all 0. We exit early.
        if n_blocks <= 0:
            o_offset = Out + off_z * stride_oz + off_h_q * stride_oh + cu_seqlens_q_start * stride_om
            o_ptrs = o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
            acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=Out.type.element_ty)
            o_ptrs_mask = offs_m[:, None] < seqlen_q
            # We still need to write 0s to the result
            tl.store(o_ptrs, acc, mask=o_ptrs_mask)
            # The tensor allocated for L is based on MAX_SEQLENS_Q as that is
            # statically known.
            l_ptrs = LSE + off_z * HQ * MAX_SEQLENS_Q + off_h_q * MAX_SEQLENS_Q + offs_m
            # We store inf to LSE, not -inf because in the bwd pass, we subtract this
            # from qk which makes it -inf, such that exp(qk - inf) = 0 for these masked blocks.
            l = tl.full([BLOCK_M], value=float("inf"), dtype=tl.float32)
            l_ptrs_mask = offs_m < MAX_SEQLENS_Q
            tl.store(l_ptrs, l, mask=l_ptrs_mask)
            # TODO: Should dropout and return encoded softmax be handled here too?
            return

    # If MQA / GQA, set the K and V head offsets appropriately.
    GROUP_SIZE: tl.constexpr = HQ // HK
    if GROUP_SIZE != 1:
        off_h_k = off_h_q // GROUP_SIZE
    else:
        off_h_k = off_h_q

    n_extra_tokens = 0
    if seqlen_k < BLOCK_N:
        n_extra_tokens = BLOCK_N - seqlen_k
    elif seqlen_k % BLOCK_N:
        n_extra_tokens = seqlen_k % BLOCK_N
    PADDED_HEAD: tl.constexpr = (ACTUAL_BLOCK_DMODEL != BLOCK_DMODEL)

    # Compute pointers for all the tensors used in this kernel.
    q_offset = Q + off_z * stride_qz + off_h_q * stride_qh + cu_seqlens_q_start * stride_qm
    q_ptrs = q_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_offset = K + off_z * stride_kz + off_h_k * stride_kh + cu_seqlens_k_start * stride_kn
    k_ptrs = k_offset + offs_d[:, None] * stride_kk + offs_n[None, :] * stride_kn
    v_offset = V + off_z * stride_vz + off_h_k * stride_vh + cu_seqlens_k_start * stride_vk
    v_ptrs = v_offset + offs_n[:, None] * stride_vk + offs_d[None, :] * stride_vn
    if USE_BIAS:
        # Note: this might get large enough to overflow on some configs
        bias_offset = off_h_q * stride_bh
        bias_ptrs = bias + bias_offset + offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn
    else:
        bias_ptrs = None

    if USE_ALIBI:
        a_offset = off_z * stride_az + off_h_q * stride_ah
        alibi_slope = tl.load(alibi_slopes + a_offset)
    else:
        alibi_slope = None

    if RETURN_SCORES:
        scores_offset = scores + off_z * stride_sz + off_h_q * stride_sh + cu_seqlens_q_start * stride_sm
        score_ptrs = scores_offset + offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn

        scores_scaled_shifted_offset = scores_scaled_shifted + off_z * stride_sz + off_h_q * stride_sh + cu_seqlens_q_start * stride_sm
        scores_scaled_shifted_ptrs = scores_scaled_shifted_offset + offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn
    
        exp_scores_offset = exp_scores + off_z * stride_sz + off_h_q * stride_sh + cu_seqlens_q_start * stride_sm
        exp_scores_ptrs = exp_scores_offset + offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn
    else:
        score_ptrs = None
        scores_scaled_shifted_ptrs = None
        exp_scores_ptrs = None

    if ENABLE_DROPOUT:
        off_hz = off_z * HQ + off_h_q
        batch_philox_offset = philox_offset_base + off_hz * seqlen_q * seqlen_k
    else:
        batch_philox_offset = 0
    # initialize pointer to m and l
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    if USE_EXP2:
        # scale sm_scale by log_2(e) and use 2^x in the loop as we do not
        # have native e^x support in HW.
        QK_SCALE: tl.constexpr = sm_scale * 1.44269504089
    else:
        QK_SCALE: tl.constexpr = sm_scale
    # Q is loaded once at the beginning and shared by all N blocks.
    q_ptrs_mask = offs_m[:, None] < seqlen_q
    if PADDED_HEAD:
        q_ptrs_mask = q_ptrs_mask & (offs_d[None, :] < ACTUAL_BLOCK_DMODEL)
    q = tl.load(q_ptrs, mask=q_ptrs_mask, other=0.0)
    q = (q * QK_SCALE).to(q.type.element_ty)

    # Here we compute how many full and masked blocks we have.
    padded_block_k = n_extra_tokens != 0
    is_modulo_mn = not padded_block_k and (seqlen_q % BLOCK_M == 0)
    if IS_CAUSAL:
        # There are always at least BLOCK_M // BLOCK_N masked blocks.
        # Additionally there might be one more due to dissimilar seqlens.
        masked_blocks = BLOCK_M // BLOCK_N + (not is_modulo_mn)
    else:
        # Padding on Q does not need to be masked in the FA loop.
        masked_blocks = padded_block_k
    # if IS_CAUSAL, not is_modulo_mn does not always result in an additional block.
    # In this case we might exceed n_blocks so pick the min.
    masked_blocks = min(masked_blocks, n_blocks)
    n_full_blocks = n_blocks - masked_blocks
    block_min = 0
    block_max = n_blocks * BLOCK_N
    # Compute for full blocks. Here we set causal to false regardless of its actual
    # value because there is no masking. Similarly we do not need padding.
    if n_full_blocks > 0:
        block_max = (n_blocks - masked_blocks) * BLOCK_N
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, k_ptrs, v_ptrs, bias_ptrs, stride_kn, stride_vk, stride_bn,
                                        start_m, seqlen_k, seqlen_q, dropout_p, philox_seed, batch_philox_offset,
                                        exp_scores_ptrs,
                                        # _, _, offs_n_causal, masked_blocks, n_extra_tokens, _
                                        block_min, block_max, 0, 0, 0, alibi_slope, score_ptrs, scores_scaled_shifted_ptrs,
                                        # IS_CAUSAL, ....
                                        False, BLOCK_M, BLOCK_DMODEL, BLOCK_N, offs_m, offs_n,
                                        # _, MASK_STEPS, ...
                                        PRE_LOAD_V, False, ENABLE_DROPOUT, PADDED_HEAD,
                                        ACTUAL_BLOCK_DMODEL, sm_scale,  USE_EXP2=USE_EXP2, RCP_LN2=RCP_LN2, RETURN_SCORES=RETURN_SCORES)
        block_min = block_max
        block_max = n_blocks * BLOCK_N

    tl.debug_barrier()
    # Remaining blocks, if any, are full / not masked.
    if (masked_blocks > 0):
        if IS_CAUSAL:
            offs_n_causal = offs_n + (seqlen_q - seqlen_k)
        else:
            offs_n_causal = 0
        k_ptrs += n_full_blocks * BLOCK_N * stride_kn
        v_ptrs += n_full_blocks * BLOCK_N * stride_vk
        if USE_BIAS:
            bias_ptrs += n_full_blocks * BLOCK_N * stride_bn
        if RETURN_SCORES:
            score_ptrs += n_full_blocks * BLOCK_N
            scores_scaled_shifted_ptrs += n_full_blocks * BLOCK_N
            exp_scores_ptrs += n_full_blocks * BLOCK_N
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, k_ptrs, v_ptrs, bias_ptrs, stride_kn, stride_vk, stride_bn,
                                        start_m, seqlen_k, seqlen_q, dropout_p, philox_seed, batch_philox_offset,
                                        exp_scores_ptrs, block_min, block_max, offs_n_causal, masked_blocks,
                                        n_extra_tokens, alibi_slope, score_ptrs, scores_scaled_shifted_ptrs,
                                        IS_CAUSAL, BLOCK_M, BLOCK_DMODEL, BLOCK_N, offs_m, offs_n,
                                        # _, MASK_STEPS, ...
                                        PRE_LOAD_V, True, ENABLE_DROPOUT, PADDED_HEAD,
                                        ACTUAL_BLOCK_DMODEL, sm_scale, USE_EXP2=USE_EXP2, RCP_LN2=RCP_LN2, RETURN_SCORES=RETURN_SCORES)
    # epilogue
    acc = acc / l_i[:, None]
    if ENABLE_DROPOUT:
        acc = acc / (1 - dropout_p)
    # If seqlen_q > seqlen_k but the delta is not a multiple of BLOCK_M,
    # then we have one block with a row of all NaNs which come from computing
    # softmax over a row of all -infs (-inf - inf = NaN). We check for that here
    # and store 0s where there are NaNs as these rows should've been zeroed out.
    end_m_idx = (start_m + 1) * BLOCK_M
    start_m_idx = start_m * BLOCK_M
    causal_start_idx = seqlen_q - seqlen_k
    acc = acc.to(Out.type.element_ty)
    if IS_CAUSAL:
        if causal_start_idx > start_m_idx and causal_start_idx < end_m_idx:
            out_mask_boundary = tl.full((BLOCK_DMODEL, ), causal_start_idx, dtype=tl.int32)
            mask_m_offsets = start_m_idx + tl.arange(0, BLOCK_M)
            out_ptrs_mask = mask_m_offsets[:, None] >= out_mask_boundary[None, :]
            z = 0.0
            acc = tl.where(out_ptrs_mask, acc, z.to(acc.type.element_ty))
    
    # write back LSE(Log Sum Exponents), the log of the normalization constant
    l_ptrs = LSE + off_z * HQ * MAX_SEQLENS_Q + off_h_q * MAX_SEQLENS_Q + offs_m
    # If seqlen_q not multiple of BLOCK_M, we need to mask out the last few rows.
    # This is only true for the last M block. For others, overflow_size will be -ve
    overflow_size = end_m_idx - seqlen_q
    if USE_EXP2:
        softmax_lse = m_i + tl.math.log2(l_i)
        softmax_lse *= LN2
    else:
        softmax_lse = m_i + tl.math.log(l_i)
    if overflow_size > 0:
        boundary = tl.full((BLOCK_M, ), BLOCK_M - overflow_size, dtype=tl.int32)
        l_ptrs_mask = tl.arange(0, BLOCK_M) < boundary
        tl.store(l_ptrs, softmax_lse, mask=l_ptrs_mask) # the log of the normalization constant
    else:
        tl.store(l_ptrs, softmax_lse) # the log of the normalization constant

    # write back O
    o_offset = Out + off_z * stride_oz + off_h_q * stride_oh + cu_seqlens_q_start * stride_om
    o_ptrs = o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
    o_ptrs_mask = tl.full([BLOCK_M, BLOCK_DMODEL], 1, dtype=tl.int1)
    if overflow_size > 0:
        o_ptrs_mask = o_ptrs_mask & (offs_m[:, None] < seqlen_q)
    if PADDED_HEAD:
        o_ptrs_mask = o_ptrs_mask & (offs_d[None, :] < ACTUAL_BLOCK_DMODEL)
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=o_ptrs_mask)


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

def get_shape_from_layout(q, k, metadata):
    if metadata.layout == 'thd':
        nheads_q, nheads_k = q.shape[1], k.shape[1]
        head_size = q.shape[-1]
        batch = metadata.num_contexts
    elif metadata.layout == 'bhsd':
        batch, nheads_q, _, head_size = q.shape
        nheads_k = k.shape[1]
    elif metadata.layout == 'bshd':
        batch, _, nheads_q, head_size = q.shape
        nheads_k = k.shape[2]
    else:
        assert False, "Got unsupported layout."
    return batch, nheads_q, nheads_k, head_size


# TODO: This can probably optimized to have fewer lines of code.
def get_strides_from_layout(q, k, v, o, metadata):
    if metadata.layout == 'thd':
        q_strides = (0, q.stride(1), q.stride(0), q.stride(2))
        k_strides = (0, k.stride(1), k.stride(0), k.stride(2))
        v_strides = (0, v.stride(1), v.stride(0), v.stride(2))
        o_strides = (0, o.stride(1), o.stride(0), o.stride(2))
    elif metadata.layout == 'bhsd':
        q_strides = (q.stride(0), q.stride(1), q.stride(2), q.stride(3))
        k_strides = (k.stride(0), k.stride(1), k.stride(2), k.stride(3))
        v_strides = (v.stride(0), v.stride(1), v.stride(2), v.stride(3))
        o_strides = (o.stride(0), o.stride(1), o.stride(2), o.stride(3))
    elif metadata.layout == 'bshd':
        q_strides = (q.stride(0), q.stride(2), q.stride(1), q.stride(3))
        k_strides = (k.stride(0), k.stride(2), k.stride(1), k.stride(3))
        v_strides = (v.stride(0), v.stride(2), v.stride(1), v.stride(3))
        o_strides = (o.stride(0), o.stride(2), o.stride(1), o.stride(3))
    else:
        assert False, 'Got unsupported layout.'
    return q_strides, k_strides, v_strides, o_strides



def attention_prefill_forward_impl(q, k, v, o, metadata):
    # NOTE: a large bias tensor leads to overflow during pointer arithmetic
    if (metadata.bias is not None):
        assert (metadata.bias.numel() < 2**31)

    if o is None:
        o = torch.empty_like(q, dtype=v.dtype)
    metadata.check_args(q, k, v, o)

    batch, nheads_q, nheads_k, head_size = get_shape_from_layout(q, k, metadata)
    q_strides, k_strides, v_strides, o_strides = get_strides_from_layout(q, k, v, o, metadata)

    # Get closest power of 2 over or equal to 32.
    padded_d_model = 1 << (head_size - 1).bit_length()
    # Smallest head_dim supported is 16. If smaller, the tile in the
    # kernel is padded - there is no padding in memory for any dims.
    padded_d_model = max(padded_d_model, 16)

    grid = lambda META: (triton.cdiv(metadata.max_seqlens_q, META['BLOCK_M']), nheads_q, batch)

    if metadata.return_scores:
        scores = torch.zeros((batch, nheads_q, metadata.max_seqlens_q, metadata.max_seqlens_k), device=q.device,
                                        dtype=torch.float32)
        scores_scaled_shifted = torch.zeros((batch, nheads_q, metadata.max_seqlens_q, metadata.max_seqlens_k), device=q.device,
                                        dtype=torch.float32)
        scores_strides = (scores.stride(0), scores.stride(1), scores.stride(2), scores.stride(3))
    else:
        scores = None
        scores_scaled_shifted = None
        scores_strides = (0, 0 , 0 , 0)

    # exp_scores is used to validate dropout behavior vs the PyTorch SDPA math backend reference.  We zero this out
    # to give a consistent starting point and then populate it with the output of softmax with the sign bit set according
    # to the dropout mask. The resulting return allows this mask to be fed into the reference implementation for testing
    # only.  This return holds no useful output aside from debugging.
    if metadata.return_scores:
        exp_scores = torch.zeros((batch, nheads_q, metadata.max_seqlens_q, metadata.max_seqlens_k), device=q.device,
                                        dtype=torch.float32)
    else:
        exp_scores = None

    # stores LSE the log of the normalization constant / sum of expoential score(unnormalzied probablities)
    # if metadata.layout == "bhsd":
    softmax_lse = torch.empty((batch, nheads_q, metadata.max_seqlens_q), device=q.device, dtype=torch.float32)
    # elif metadata.layout == "bshd":
    #     softmax_lse = torch.empty((batch, metadata.max_seqlens_q, nheads_q), device=q.device, dtype=torch.float32)
    # else:
    #     print("metadat.layout:",  metadata.layout)
    #     raise ValueError("unknown layout")


    # Seed the RNG so we get reproducible results for testing.
    philox_seed = 0x1BF52
    philox_offset = 0x1D4B42

    if metadata.bias is not None:
        bias_strides = (metadata.bias.stride(0), metadata.bias.stride(1), metadata.bias.stride(2),
                        metadata.bias.stride(3))
    else:
        bias_strides = (0, 0, 0, 0)

    if metadata.alibi_slopes is not None:
        alibi_strides = (metadata.alibi_slopes.stride(0), metadata.alibi_slopes.stride(1))
    else:
        alibi_strides = (0, 0)

    attn_fwd[grid](q, k, v, metadata.bias, metadata.sm_scale, softmax_lse, o, *q_strides, *k_strides, *v_strides, *o_strides,
                    *bias_strides, *alibi_strides, *scores_strides, metadata.cu_seqlens_q, metadata.cu_seqlens_k,
                    dropout_p=metadata.dropout_p, philox_seed=philox_seed, philox_offset_base=philox_offset, scores=scores, 
                    scores_scaled_shifted=scores_scaled_shifted, exp_scores=exp_scores, alibi_slopes=metadata.alibi_slopes, 
                    HQ=nheads_q, HK=nheads_k, ACTUAL_BLOCK_DMODEL=head_size, MAX_SEQLENS_Q=metadata.max_seqlens_q,
                    MAX_SEQLENS_K=metadata.max_seqlens_k, IS_CAUSAL=metadata.causal, VARLEN=metadata.varlen,
                    BLOCK_DMODEL=padded_d_model, USE_BIAS=False if metadata.bias is None else True,
                    USE_ALIBI=False if metadata.alibi_slopes is None else True, ENABLE_DROPOUT=metadata.dropout_p
                    > 0.0, return_scores=metadata.return_scores,
                    USE_EXP2=metadata.use_exp2, LN2=LN2, RCP_LN2=RCP_LN2, RETURN_SCORES=metadata.return_scores)

    return o, softmax_lse, exp_scores, grid, head_size, philox_seed, philox_offset, scores, scores_scaled_shifted


def attention_prefill_backward_old_impl(do, q, k, v, o, softmax_lse, sm_scale, head_size, alibi_slopes, causal, layout, use_exp2):
    if True:
        print()
        print("attention_prefill_backward_old_impl")
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

def attention_prefill_backward_impl(do, q, k, v, o, softmax_lse,  dq, dk, dv, sm_scale, head_size, alibi_slopes, causal, layout, use_exp2, use_new):
    if False:
        if use_exp2:
            softmax_lse *= RCP_LN2 # oai kernel expects softmax_lse to be an intermediate result of using exp2
        else:
            raise ValueError("openai backward kernel assumes exp2")
        return attention_prefill_backward_oai_impl(do, q, k, v, o, softmax_lse, dq, dk, dv, sm_scale, head_size, alibi_slopes, causal, layout)
    elif True:
        # test pytorch impl
        dq_ref, dk_ref, dv_ref = attention_backward_pytorch_ref_impl(do, q, k, v, o, softmax_lse, sm_scale, causal, layout, use_exp2=use_exp2)
        if dq is not None:
            dq.copy_(dq_ref)
        else:
            dq = dq_ref
        
        if dk is not None:
            dk.copy_(dk_ref)
        else:
            dk = dk_ref

        if dv is not None:
            dv.copy_(dv_ref)
        else:
            dv = dv_ref
            
        return dq, dk, dv, None, None, None
    elif use_new:
        return attention_prefill_backward_new_impl(do, q, k, v, o, softmax_lse, dq, dk, dv, sm_scale, head_size, alibi_slopes, causal, layout, use_exp2)
    else:
        return attention_prefill_backward_old_impl(do, q, k, v, o, softmax_lse, dq, dk, dv, sm_scale, head_size, alibi_slopes, causal, layout, use_exp2)



class _attention_prefill(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, o, metadata):
        o, softmax_lse, exp_scores, grid, head_size, philox_seed, philox_offset, _, _ = attention_prefill_forward_impl(q, k, v, o, metadata)

        ctx.save_for_backward(q, k, v, o, softmax_lse)
        ctx.grid = grid
        ctx.sm_scale = metadata.sm_scale
        ctx.head_size = head_size
        ctx.causal = metadata.causal
        ctx.alibi_slopes = metadata.alibi_slopes
        ctx.dropout_p = metadata.dropout_p
        ctx.philox_seed = philox_seed
        ctx.philox_offset = philox_offset
        ctx.exp_scores = exp_scores
        ctx.return_scores = metadata.return_scores
        ctx.layout = metadata.layout
        ctx.use_exp2 = metadata.use_exp2
        return o, softmax_lse, exp_scores

    @staticmethod
    def backward(ctx, do, *args): # expects bhsd
        q, k, v, o, softmax_lse = ctx.saved_tensors
        return attention_prefill_backward_impl(do, q, k, v, o, softmax_lse, None, None, None, ctx.sm_scale, ctx.head_size, ctx.alibi_slopes, ctx.causal, ctx.layout, ctx.use_exp2, True)

attention_prefill = _attention_prefill.apply


def input_helper(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, layout):
    torch.manual_seed(20)

    # Initialize q, k, v
    if layout == 'bhsd':
        q_tensor_shape = (Z, HQ, N_CTX_Q, D_HEAD)
        k_tensor_shape = (Z, HK, N_CTX_K, D_HEAD)
    elif layout == 'bshd':
        q_tensor_shape = (Z, N_CTX_Q, HQ, D_HEAD)
        k_tensor_shape = (Z, N_CTX_K, HK, D_HEAD)
    else:
        assert False, 'Got unsupported tensor layout'
    q = torch.randn(q_tensor_shape, dtype=dtype, device="cuda", requires_grad=True)
    k = torch.randn(k_tensor_shape, dtype=dtype, device="cuda", requires_grad=True)
    v = torch.randn(k_tensor_shape, dtype=dtype, device="cuda", requires_grad=True)
    sm_scale = D_HEAD**-0.5
    input_metadata = MetaData(sm_scale=sm_scale)
    input_metadata.max_seqlens_q = N_CTX_Q
    input_metadata.max_seqlens_k = N_CTX_K
    input_metadata.layout = layout
    return q, k, v, input_metadata


def varlen_input_helper(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, equal_seqlens=False):
    torch.manual_seed(20)

    # Random sequence lengths. Using N_CTX as kind of max of sum of individual seqs
    if not equal_seqlens:
        max_seqlens_q = N_CTX_Q // Z
        max_seqlens_k = N_CTX_K // Z
        seqlens_q = torch.randint(1, max_seqlens_q + 1, (Z, ), dtype=torch.int32)
        seqlens_k = torch.randint(1, max_seqlens_k + 1, (Z, ), dtype=torch.int32)
    else:
        seqlens_q = torch.full((Z, ), N_CTX_Q // Z)
        seqlens_k = torch.full((Z, ), N_CTX_K // Z)

    # Calculate cumulative sequence lengths
    cu_seqlens_q = torch.cat([torch.tensor([0], dtype=torch.int32), seqlens_q.cumsum(dim=0, dtype=torch.int32)])
    cu_seqlens_k = torch.cat([torch.tensor([0], dtype=torch.int32), seqlens_k.cumsum(dim=0, dtype=torch.int32)])
    cu_seqlens_q = cu_seqlens_q.to(device="cuda")
    cu_seqlens_k = cu_seqlens_k.to(device="cuda")

    # Initialize q, k, v with variable lengths
    total_q = cu_seqlens_q[-1].item()
    total_k = cu_seqlens_k[-1].item()
    q = torch.randn((total_q, HQ, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    k = torch.randn((total_k, HK, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    v = torch.randn((total_k, HK, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0., std=0.5).requires_grad_()
    sm_scale = D_HEAD**-0.5
    input_metadata = MetaData(sm_scale=sm_scale)
    input_metadata.set_varlen_params(cu_seqlens_q, cu_seqlens_k)
    return q, k, v, input_metadata


@pytest.mark.parametrize('Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD', [
    (4, 48, 24, 1024, 1024, 64),
    (1, 24, 6, 8192, 8192, 64),
    (1, 4, 2, 16384, 16384, 128),
    (2, 16, 4, 1020, 987, 128),
    (2, 16, 4, 15498, 2, 128),
    (2, 16, 2, 7, 16219, 64),
    (4, 48, 12, 1, 1, 64),
    (4, 48, 48, 1, 1, 128),
    (4, 48, 24, 3, 3, 128),
    (4, 48, 48, 1001, 990, 64),
    (1, 8, 8, 8081, 7099, 64),
    (1, 4, 4, 16330, 15989, 128),
    (4, 4, 1, 1024, 1024, 33),
    (4, 4, 2, 65, 1018, 65),
    (4, 4, 4, 128, 128, 65),
    (4, 4, 4, 113, 123, 1),
])
@pytest.mark.parametrize('causal', [True, False])
@pytest.mark.parametrize('use_alibi', [True, False])
@pytest.mark.parametrize('layout', ['bshd', 'bhsd'])
def test_op_fwd(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, causal, use_alibi, layout, dtype=torch.float16):
    torch.manual_seed(20)
    q, k, v, input_metadata = input_helper(Z, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, layout)
    if causal:
        input_metadata.need_causal()

    if use_alibi:
        # for n heads the set of slopes is the geometric sequence that starts 2^(-8/n)
        alibi_slopes = torch.tensor([2**(-8 / HQ * i) for i in range(1, HQ + 1)], dtype=torch.float32,
                                    device="cuda").repeat(Z, 1)
        input_metadata.need_alibi(alibi_slopes, Z, HQ)
    else:
        alibi_slopes = None

    o = torch.empty_like(q)

    # triton implementation
    tri_out, _, _ = attention_prefill(q, k, v, o, input_metadata)

    # Transpose here if layout is bshd so we have same reference code for all layouts
    if layout == 'bshd':
        q = q.transpose(1, 2).clone()
        k = k.transpose(1, 2).clone()
        v = v.transpose(1, 2).clone()
    # Replicate K and V if using MQA/GQA
    if HQ != HK:
        k = k.view(k.shape[0], k.shape[1], -1, k.shape[2],
                   k.shape[3]).expand(-1, -1, HQ // HK, -1, -1).reshape(k.shape[0], -1, k.shape[2], k.shape[3])
        v = v.view(v.shape[0], v.shape[1], -1, v.shape[2],
                   v.shape[3]).expand(-1, -1, HQ // HK, -1, -1).reshape(v.shape[0], -1, v.shape[2], v.shape[3])

    scores = torch.einsum('bhqd,bhkd->bhqk', q, k).float() * input_metadata.sm_scale
    if causal:
        mask = torch.tril(torch.ones(N_CTX_Q, N_CTX_K, device="cuda"), diagonal=N_CTX_K - N_CTX_Q)
        scores[:, :, mask == 0] = float("-inf")
    if use_alibi:
        scores += compute_alibi_tensor(alibi_slopes, N_CTX_Q, N_CTX_K)

    p = torch.softmax(scores, dim=-1)
    if causal:
        # If N_CTX_Q > N_CTX_K, there is at least one row of all -infs going into
        # the softmax. This produces a row of NaNs as -inf - -inf == NaN. So we fix
        # this by converting the NaNs to 0s, which is what they should be out of the softmax.
        nan_mask = torch.isnan(p)
        p[nan_mask == 1] = 0
    ref_out = torch.einsum('bhqk,bhkd->bhqd', p.half(), v)
    # compare
    if layout == 'bshd':
        ref_out = ref_out.transpose(1, 2).clone()
    torch.testing.assert_close(ref_out, tri_out, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize('Z, H, N_CTX_Q, N_CTX_K, D_HEAD', [
    (4, 48, 1024, 1024, 64),
    (4, 12, 8192, 8192, 64),
    (2, 4, 16384, 16384, 128),
    (2, 16, 1020, 987, 128),
    (2, 16, 15498, 2, 128),
    (2, 4, 7, 16219, 64),
    (4, 48, 1, 1, 64),
    (4, 48, 1, 1, 128),
    (4, 48, 3, 3, 128),
    (4, 48, 1001, 990, 64),
    (1, 8, 8081, 7099, 64),
    (1, 8, 16330, 15989, 128),
    (4, 4, 1024, 1024, 33),
    (4, 4, 65, 1019, 65),
    (4, 4, 128, 128, 65),
    # TODO: This config fails. Disabled until triaged and fixed.
    #   (4, 4, 113, 123, 1),
])
@pytest.mark.parametrize('causal', [True, False])
@pytest.mark.parametrize('use_bias', [True])
def test_op_fwd_bias(Z, H, N_CTX_Q, N_CTX_K, D_HEAD, causal, use_bias, dtype=torch.float16):
    torch.manual_seed(20)
    sm_scale = D_HEAD**-0.5
    input_metadata = MetaData(sm_scale=sm_scale)
    q, k, v, input_metadata = input_helper(Z, H, H, N_CTX_Q, N_CTX_K, D_HEAD, dtype, layout='bhsd')
    if causal:
        input_metadata.need_causal()
    if use_bias:
        bias = torch.randn((1, H, N_CTX_Q, N_CTX_K), dtype=torch.float32, device="cuda")
        input_metadata.need_bias(bias, Z, H, N_CTX_Q, N_CTX_K)
    else:
        bias = None
    o = torch.empty_like(q)

    # triton implementation
    tri_out, _, _ = attention_prefill(q, k, v, o, input_metadata)
    # reference implementation:171

    scores = torch.einsum('bhqd,bhkd->bhqk', q, k).float() * sm_scale
    if causal:
        mask = torch.tril(torch.ones(N_CTX_Q, N_CTX_K, device="cuda"), diagonal=N_CTX_K - N_CTX_Q)
        scores[:, :, mask == 0] = float("-inf")
    if use_bias:
        scores += input_metadata.bias
    p = torch.softmax(scores, dim=-1)
    if causal:
        # If N_CTX_Q > N_CTX_K, there is at least one row of all -infs going into
        # the softmax. This produces a row of NaNs as -inf - -inf == NaN. So we fix
        # this by converting the NaNs to 0s, which is what they should be out of the softmax.
        nan_mask = torch.isnan(p)
        p[nan_mask == 1] = 0
    ref_out = torch.einsum('bhqk,bhkd->bhqd', p.half(), v)
    # compare
    torch.testing.assert_close(ref_out, tri_out, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize('Z, H, N_CTX, D_HEAD', [(4, 48, 8192, 64), (4, 48, 256, 64), (4, 48, 512, 64),
                                                 (4, 48, 1024, 64), (8, 48, 4096, 64), (4, 48, 8192, 64),
                                                 (4, 48, 128, 128), (4, 48, 4096, 128), (4, 48, 16384, 128),
                                                 (4, 16, 1024, 128), (4, 16, 8192, 128), (32, 48, 8192, 128)])
@pytest.mark.parametrize('causal', [True, False])
def test_op_varlen_fwd(Z, H, N_CTX, D_HEAD, causal, dtype=torch.float16):

    q, k, v, input_metadata = varlen_input_helper(Z, H, H, N_CTX, N_CTX, D_HEAD, dtype)

    tri_out = torch.empty_like(q)
    ref_out = torch.empty_like(q)

    for i in range(0, input_metadata.num_contexts):
        start_q, start_k = input_metadata.cu_seqlens_q[i], input_metadata.cu_seqlens_k[i]
        end_q, end_k = input_metadata.cu_seqlens_q[i + 1], input_metadata.cu_seqlens_k[i + 1]
        scores = torch.einsum('qhd,khd->qhk', q[start_q:end_q], k[start_k:end_k]).float()
        p = torch.softmax(scores * input_metadata.sm_scale, dim=-1).half()
        ref_out[start_q:end_q] = torch.einsum('qhk,khd->qhd', p, v[start_k:end_k])
    attention_prefill(q, k, v, tri_out, input_metadata)
    torch.testing.assert_close(ref_out, tri_out, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize('Z, HQ, HK, N_CTX, D_HEAD', [(2, 48, 24, 128, 64), (4, 48, 12, 256, 64), (4, 48, 4, 512, 64),
                                                      (4, 48, 2, 1024, 64), (8, 48, 6, 4096, 64), (4, 48, 8, 16384, 64),
                                                      (4, 64, 16, 128, 128), (4, 64, 4, 4096, 128),
                                                      (4, 64, 8, 16384, 128), (4, 16, 4, 1024, 128),
                                                      (4, 16, 2, 8192, 128), (32, 128, 32, 8192, 128)])
@pytest.mark.parametrize('causal', [False])
def test_op_varlen_mqa_fwd(Z, HQ, HK, N_CTX, D_HEAD, causal, dtype=torch.float16):
    q, k, v, input_metadata = varlen_input_helper(Z, HQ, HK, N_CTX, N_CTX, D_HEAD, dtype)
    ref_out = torch.empty_like(q)
    tri_out = torch.empty_like(q)
    # Make KV look like HQ/HK "groups" of HK. Later, we will reshape so the
    # size aligns with Q.
    k_ref = k.view(k.shape[0], k.shape[1], 1, k.shape[2]).expand(-1, -1, HQ // HK, -1)
    v_ref = v.view(v.shape[0], v.shape[1], 1, v.shape[2]).expand(-1, -1, HQ // HK, -1)
    for i in range(0, input_metadata.num_contexts):
        start_q, start_k = input_metadata.cu_seqlens_q[i], input_metadata.cu_seqlens_k[i]
        end_q, end_k = input_metadata.cu_seqlens_q[i + 1], input_metadata.cu_seqlens_k[i + 1]
        k_curr = k_ref[start_k:end_k]
        k_curr = k_curr.reshape(k_curr.shape[0], -1, k_curr.shape[3])
        v_curr = v_ref[start_k:end_k]
        v_curr = v_curr.reshape(v_curr.shape[0], -1, v_curr.shape[3])
        scores = torch.einsum('qhd,khd->qhk', q[start_q:end_q], k_curr).float()
        p = torch.softmax(scores * input_metadata.sm_scale, dim=-1).half()
        ref_out[start_q:end_q] = torch.einsum('qhk,khd->qhd', p, v_curr)
    attention_prefill(q, k, v, tri_out, input_metadata)
    torch.testing.assert_close(ref_out, tri_out, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize('Z, H, N_CTX_Q, N_CTX_K, D_HEAD', [
    # smallest config test
    (1, 1, 16, 16, 64), # pass on new # fail on old
    (1, 1, 32, 32, 64), # pass on new # fail on old
    (1, 1, 64, 64, 16), # pass # smallest head_size = 16
    (1, 1, 64, 64, 64), # pass # smallest seq len seems to be 64
    (1, 1, 128, 128, 64), # pass
    (1, 1, 256, 256, 64), # pass
    (1, 1, 512, 512, 64), # pass
    # failing FA
    (1, 1, 256, 512, 16),
    # old tests that work
    (4, 48, 1024, 1024, 64), # pass
    (4, 48, 2048, 2048, 64), # pass
    (2, 48, 4096, 4096, 64), # pass
    (1, 16, 1024, 1024, 64), # pass
    (1, 16, 1024, 1024, 128), # pass
    # old tests that were commented out
    # (1, 16, 8192, 8192, 63),
    # (1, 16, 1022, 1022, 64),
])
# @pytest.mark.parametrize('torch_sdpa_test', [False, True])
@pytest.mark.parametrize('torch_sdpa_test', [False])
# @pytest.mark.parametrize('causal', [True, False])
@pytest.mark.parametrize('causal', [False])
# @pytest.mark.parametrize('use_alibi', [False, True])
@pytest.mark.parametrize('use_alibi', [False])
def test_op_bwd(Z, H, N_CTX_Q, N_CTX_K, D_HEAD, causal, torch_sdpa_test, use_alibi, dtype=torch.float16):
    torch.manual_seed(20)

    DEBUG_INPUT = False # if DEBUG is True it fails

    # seqlens
    seqlen_q = N_CTX_Q
    seqlen_k = N_CTX_K

    # setup up metadata
    if DEBUG_INPUT:
        sm_scale = 1
    else:
        sm_scale = D_HEAD**-0.5
    input_metadata = MetaData(sm_scale=sm_scale)
    input_metadata.max_seqlens_q = seqlen_q
    input_metadata.max_seqlens_k = seqlen_k
    input_metadata.layout = "bhsd"

    dropout_p = 0
    if DEBUG_INPUT:
        q = torch.arange(seqlen_q, dtype=dtype, device="cuda").view(1, 1, seqlen_q, 1).expand(Z, H, seqlen_q, D_HEAD).requires_grad_()
        k = torch.arange(seqlen_k, dtype=dtype, device="cuda").view(1, 1, seqlen_k, 1).expand(Z, H, seqlen_k, D_HEAD).requires_grad_()
        v = torch.arange(seqlen_k, dtype=dtype, device="cuda").view(1, 1, seqlen_k, 1).expand(Z, H, seqlen_k, D_HEAD).requires_grad_()
        o = torch.zeros_like(q)
    else:
        # Generate random inputs
        q = torch.randn(Z, H, N_CTX_Q, D_HEAD, device='cuda', dtype=dtype, requires_grad=True)
        k = torch.randn(Z, H, N_CTX_K, D_HEAD, device='cuda', dtype=dtype, requires_grad=True)
        v = torch.randn(Z, H, N_CTX_K, D_HEAD, device='cuda', dtype=dtype, requires_grad=True)
        o = torch.empty_like(q)

    if causal:
        input_metadata.need_causal()

    if use_alibi and not torch_sdpa_test:
        # for n heads the set of slopes is the geometric sequence that starts 2^(-8/n)
        alibi_slopes = torch.tensor([2**(-8 / H * i) for i in range(1, H + 1)], dtype=torch.float32,
                                    device="cuda").repeat(Z, 1)
        input_metadata.need_alibi(alibi_slopes, Z, H)

    if DEBUG_INPUT:
        dout = torch.ones_like(q)
    else:
        dout = torch.randn_like(q)

    # reference implementation
    if torch_sdpa_test:
        ref_out, ref_softmax = torch.ops.aten._scaled_dot_product_attention_math(q, k, v, dropout_p=dropout_p,
                                                                                 is_causal=causal, scale=sm_scale,
                                                                                 dropout_mask=None)
        ref_out.backward(dout.to(device=ref_out.device, dtype=ref_out.dtype))
        ref_dv, v.grad = v.grad.clone(), None
        ref_dk, k.grad = k.grad.clone(), None
        ref_dq, q.grad = q.grad.clone(), None
    else:
        M = torch.tril(torch.ones((seqlen_q, seqlen_k), device="cuda"))
        p = torch.matmul(q, k.transpose(2, 3)) * sm_scale
        if use_alibi:
            p += compute_alibi_tensor(alibi_slopes, N_CTX_Q, N_CTX_K)
        if causal:
            p[:, :, M == 0] = float("-inf")

        p = torch.softmax(p.float(), dim=-1).type(dtype=p.dtype)
        ref_out = torch.matmul(p, v)
        ref_out.backward(dout)
        ref_dv, v.grad = v.grad.clone(), None
        ref_dk, k.grad = k.grad.clone(), None
        ref_dq, q.grad = q.grad.clone(), None

    # # triton implementation
    tri_out, _, _ = attention_prefill(q, k, v, o, input_metadata)
    tri_out.backward(dout)
    tri_dv, v.grad = v.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dq, q.grad = q.grad.clone(), None
    # compare
    if DEBUG:
        print("tri_out:", tri_out)
        print("ref_out:",ref_out )
    torch.testing.assert_close(ref_out, tri_out, atol=1e-2, rtol=0)
    # The current block size for MI200 series is 64x64. This results in
    # larger differences in float results due to rounding.

    if dtype == torch.bfloat16:
        ATOL = 1e-1 * max(1.0, (seqlen_q + D_HEAD) / 64.0)
    if dtype == torch.float32:
        ATOL = 1e-3 * max(1.0, (seqlen_q + D_HEAD) / 64.0)
    else:
        ATOL = 1e-1 * max(1.0, (seqlen_q + D_HEAD) / 64.0)

    RTOL = 0

    if DEBUG:
        print("ref_dv:", ref_dv)
        print("tri_dv:", tri_dv)
        print("ref_dk:", ref_dk)
        print("tri_dk:", tri_dk)
        print("ref_dq:", ref_dq)
        print("tri_dq:", tri_dq)

    torch.testing.assert_close(ref_dv, tri_dv, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(ref_dk, tri_dk, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(ref_dq, tri_dq, atol=ATOL, rtol=RTOL)


def attention_forward_pytorch_ref_impl(q, k, v, sm_scale, causal, layout, use_exp2=False):
    """compute reference output and softmax_lse using PyTorch's built-in function"""

    # expects bhsd layout
    if layout != "bhsd":
        raise ValueError("bhsd is the only layout supported")

    # get seqlens
    N_CTX_Q = q.shape[2]
    N_CTX_K = k.shape[2]

    # compute attention scores
    attention_scores = torch.matmul(q.to(torch.float32), k.transpose(-2, -1).to(torch.float32))


    # scale score
    attention_scaled_scores = sm_scale * attention_scores

    # ===========================  Softmax ========================================
    # compute max for numerical stability
    max_scores = torch.max(attention_scaled_scores, dim=-1, keepdim=True)[0]

    # shift scores to by subtracing max
    attention_shifted_scaled_scores = attention_scaled_scores - max_scores

    # subtract max and exponentiate
    if use_exp2:
        RCP_LN = 1/ math.log(2)
        exp2_scores = torch.exp2(RCP_LN * attention_shifted_scaled_scores)
    else:
        exp_scores = torch.exp(attention_shifted_scaled_scores)

    # sum of exponentials
    if use_exp2:
        sum_exp2_scores = torch.sum(exp2_scores, dim=-1, keepdim=True)
    else:
        sum_exp_scores = torch.sum(exp_scores, dim=-1, keepdim=True)

    # softmax probabilities
    if use_exp2:
        softmax_exp2 = exp2_scores / sum_exp2_scores
    else:
        softmax = exp_scores / sum_exp_scores
    
    # compute log-sum-exp and squeeze final dim which will be 1
    if use_exp2:
        LN2 = math.log(2)
        RCP_LN = 1/ math.log(2)
        # compute log-sum-exp in base 2 units
        max_scores_base2 = max_scores * RCP_LN
        softmax_exp2_lse_base2 = max_scores_base2 + torch.log2(sum_exp2_scores)
        # Convert back to natural units
        softmax_exp2_lse = softmax_exp2_lse_base2 * LN2
        softmax_exp2_lse.squeeze_(-1)
    else:
        softmax_lse = max_scores + torch.log(sum_exp_scores)
        softmax_lse.squeeze_(-1)
    
    # compute output
    if use_exp2:
        o_exp2 = torch.matmul(softmax_exp2, v.to(torch.float32)).to(torch.float16)
    else:
        o = torch.matmul(softmax, v.to(torch.float32)).to(torch.float16)


    if use_exp2:
        return o_exp2, softmax_exp2_lse, exp2_scores, softmax_exp2, attention_shifted_scaled_scores, attention_scores
    else:
        return o, softmax_lse, exp_scores, softmax, attention_shifted_scaled_scores, attention_scores

def attention_backward_pytorch_ref_impl(do, q, k, v, o, softmax_lse, sm_scale, causal, layout, use_exp2):
    # ensure the layout is 'bhsd'
    if layout == "bshd":
        print("Changing layout to bhsd!")
        do = do.transpose(1, 2).contiguous()
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        o = o.transpose(1, 2).contiguous()
        # softmax_lse = softmax_lse.transpose(1, 2).contiguous()
        # TODO: does L/M need to be transposed. possible to use strides
    elif layout == "bhsd":
        pass
    else:
        raise ValueError(f"Unknown layout {layout}")
    
    # recompute attention_scores
    attention_scores = torch.matmul(q, k.transpose(-2, -1))
    print("attention_scores:", attention_scores)

    # scale scores
    attention_scaled_scores = sm_scale * attention_scores
    print("attention_scaled_scores:", attention_scaled_scores)

    # compute probabilities using softmax_lse
    if use_exp2:
        RCP_LN = 1 / math.log(2)
        attention_scaled_scores_base2 = attention_scaled_scores * RCP_LN
        softmax_lse_base2 = softmax_lse * RCP_LN
        p = torch.exp2(attention_scaled_scores_base2 - softmax_lse_base2.unsqueeze(-1))
    else:
        p = torch.exp(attention_scaled_scores - softmax_lse.unsqueeze(-1))

    print("p:", p)
    # compute gradient wrt v
    dv = torch.matmul(p.transpose(-2, -1), do.to(torch.float32))

    # compute dp
    dp = torch.matmul(do, v.transpose(-2, -1))

    NO_PREPROCESSING = True
    # calculate ds
    if NO_PREPROCESSING:
        delta = torch.sum(p * dp, axis=-1).unsqueeze(-1) # what the math says you should use
    else:
        delta = torch.sum(o * do, axis=-1).unsqueeze(-1) # what oai kernel uses
    ds = (p * (dp - delta)) * sm_scale

    # compute gradient wrt k
    dk = torch.matmul(ds.transpose(-2, -1), q.to(torch.float32))

    # compute gradient wrt q
    dq = torch.matmul(ds, k.to(torch.float32))

    # cast back to original dtype
    dq = dq.to(q.dtype)
    dk = dk.to(k.dtype)
    dv = dv.to(v.dtype)

    # go back to original layout
    if layout == "bshd":
        print("Changing back to bshd!")
        dq = dq.transpose(1, 2)
        dk = dk.transpose(1, 2)
        dv = dv.transpose(1, 2)
    elif layout == "bhsd":
        pass
    else:
        raise ValueError(f"Unknown layout {layout}")

    return dq, dk, dv


# fp16 default is ATOL, RTOL = 1e-5, 1e-3. See table https://pytorch.org/docs/stable/testing.html
ATOL, RTOL = 1e-2, 1e-2 # old standard. maybe to lose. 
# ATOL, RTOL = 1e-3, 1e-3  # catchs fa mismatch issues
# ATOL, RTOL = 1e-4, 1e-3 # to strict. there will be small diffs

@pytest.mark.parametrize('Z, H, N_CTX_Q, N_CTX_K, D_HEAD', [
    (1, 1, 1, 1, 1),
    (1, 1, 4, 4, 16),
    (2, 2, 4, 4, 16),
    (1, 1, 2, 128, 1),
    (1, 1, 2, 128, 16),
    (1, 1, 256, 512, 16),
    (1, 1, 128, 128, 64),
    (2, 4, 1024, 1024, 64),
    (4, 8, 2048, 2048, 128),
    (4, 16, 4096, 4096, 64),
    (2, 4, 8192, 8192, 32),
])
@pytest.mark.parametrize('causal', [False])
@pytest.mark.parametrize('return_scores', [False])
@pytest.mark.parametrize('check_softmax', [True, False])
@pytest.mark.parametrize('use_exp2', [True, False]) # works when use_exp2 is false
@pytest.mark.parametrize('DEBUG_INPUT', [False]) # NOTE: debug input can overflow when the tensors are large. Just use to figure out issues
def test_op_fwd_impl(Z, H, N_CTX_Q, N_CTX_K, D_HEAD, causal, return_scores, check_softmax, use_exp2, DEBUG_INPUT):
    dtype = torch.float16
    torch.manual_seed(0)
    
    if DEBUG_INPUT:
        sm_scale = 1
    else:
        sm_scale =  D_HEAD ** -0.5
    layout = 'bhsd'
    alibi_slopes = None
    dropout_p = 0.0

    if DEBUG_INPUT:
        q = torch.arange(N_CTX_Q, dtype=dtype, device="cuda").view(1, 1, N_CTX_Q, 1).expand(Z, H, N_CTX_Q, D_HEAD).requires_grad_()
        k = torch.arange(N_CTX_K, dtype=dtype, device="cuda").view(1, 1, N_CTX_K, 1).expand(Z, H, N_CTX_K, D_HEAD).requires_grad_()
        v = torch.arange(N_CTX_K, dtype=dtype, device="cuda").view(1, 1, N_CTX_K, 1).expand(Z, H, N_CTX_K, D_HEAD).requires_grad_()
        o = torch.zeros_like(q)
    else:
        # Generate random inputs
        q = torch.randn(Z, H, N_CTX_Q, D_HEAD, device='cuda', dtype=dtype)
        k = torch.randn(Z, H, N_CTX_K, D_HEAD, device='cuda', dtype=dtype)
        v = torch.randn(Z, H, N_CTX_K, D_HEAD, device='cuda', dtype=dtype)
        o = torch.empty_like(q)

    # Set up metadata
    input_metadata = MetaData(sm_scale=sm_scale)
    input_metadata.max_seqlens_q = N_CTX_Q
    input_metadata.max_seqlens_k = N_CTX_K
    input_metadata.layout = layout
    input_metadata.use_exp2 = use_exp2
    if causal:
        input_metadata.need_causal()

     # NOTE: the returned score is not the same as the reference because we need to adjust as we find new maxes per block. We are not doing that
    if check_softmax or return_scores:
        input_metadata.return_scores = True
    
    # call Triton's forward implementation directly
    o, softmax_lse_triton, exp_scores_triton, grid, head_size, philox_seed, philox_offset, _, _ = attention_prefill_forward_impl(q, k, v, o, input_metadata)

    # compute reference
    o_ref, softmax_lse_ref, exp_scores_ref, softmax_ref, attention_shifted_scaled_scores_ref, attention_scores_ref = attention_forward_pytorch_ref_impl(q.clone(), k.clone(), v.clone(), sm_scale, causal, "bhsd", use_exp2=use_exp2)
    # ref output
    print("attention_scores_ref:", attention_scores_ref, attention_scores_ref.shape)
    print("attention_shifted_scaled_scores_ref:", attention_shifted_scaled_scores_ref, attention_shifted_scaled_scores_ref.shape)
    print("exp_scores_ref:", exp_scores_ref, exp_scores_ref.shape)
    print("softmax_lse_ref:", softmax_lse_ref, softmax_lse_ref.shape)
    print("softmax_ref:", softmax_ref, softmax_ref.shape)

    # compare the outputs with ref
    print("o:", o, o.shape)
    print("ref_o:", o_ref, o_ref.shape)
    torch.testing.assert_close(o, o_ref, atol=ATOL, rtol=RTOL)

    # compare with pytorch
    out_pytorch, softmax_pytorch = torch.ops.aten._scaled_dot_product_attention_math(q, k, v, dropout_p=dropout_p,
                                                                            is_causal=causal, scale=sm_scale,
                                                                            dropout_mask=None)
    print("o:", o, o.shape)
    print("out_pytorch:", out_pytorch, out_pytorch.shape)
    torch.testing.assert_close(o, out_pytorch, atol=ATOL, rtol=RTOL)

    if check_softmax:
        # compare softmax_lse with ref
        print("softmax_lse_triton:", softmax_lse_triton, softmax_lse_triton.shape)
        print("softmax_lse_ref:", softmax_lse_ref, softmax_lse_ref.shape)
        torch.testing.assert_close(softmax_lse_triton, softmax_lse_ref, atol=ATOL, rtol=RTOL)

        # use trick with lse to get the softmax. you need the scores but is it
        softmax_triton = torch.exp(sm_scale * attention_scores_ref - softmax_lse_triton.unsqueeze(-1))
        print("softmax_triton:", softmax_triton, softmax_triton.shape)
        print("softmax_ref:", softmax_ref, softmax_ref.shape)
        torch.testing.assert_close(softmax_triton, softmax_ref, atol=ATOL, rtol=RTOL)       
        
        # compare with pytorch output
        print("softmax_triton:", softmax_triton, softmax_triton.shape)
        print("softmax_pytorch:", softmax_pytorch, softmax_pytorch.shape)
        torch.testing.assert_close(softmax_triton, softmax_pytorch.to(torch.float32), atol=ATOL, rtol=RTOL)



@pytest.mark.parametrize('Z, H, N_CTX_Q, N_CTX_K, D_HEAD', [
    (1, 1, 1, 1, 1),
    (1, 1, 4, 4, 4),
    (1, 1, 4, 4, 16),
    (1, 1, 32, 32, 16),
    (1, 1, 64, 64, 16), # pass # smallest head_size = 16
    (1, 1, 64, 64, 64), # pass # smallest seq len seems to be 64
    (1, 1, 128, 128, 64),
    (1, 1, 128, 256, 45),
    (1, 1, 256, 256, 64),
    (1, 1, 256, 512, 16),
    (1, 1, 512, 512, 64), 
    (1, 1, 1024, 1024, 64),
    # old tests that work
    (4, 48, 1024, 1024, 73),
    (4, 48, 1024, 1024, 64),
    (4, 48, 2048, 2048, 64),
    (1, 24, 4096, 4096, 64),
    (1, 16, 1024, 1024, 64),
    (1, 16, 1024, 1024, 128),
    # # old tests that were commented out
    # (1, 16, 8192, 8192, 63),
    # (1, 16, 1022, 1022, 64),
    # bad fa configs
    # (1, 1, 256, 512, 16),
])
@pytest.mark.parametrize('causal', [False])
@pytest.mark.parametrize('use_exp2', [True, False])
@pytest.mark.parametrize('use_new', [True])
@pytest.mark.parametrize('DEBUG_INPUT', [False]) # debug output causes nans in both new and old backend
def test_op_bwd_impl(Z, H, N_CTX_Q, N_CTX_K, D_HEAD, causal, use_exp2, use_new, DEBUG_INPUT):
    dtype = torch.float16
    torch.manual_seed(20) # seed from test_op_bwd
    
    if DEBUG_INPUT:
        sm_scale = 1
    else:
        sm_scale =  D_HEAD ** -0.5
    head_size = D_HEAD
    layout = 'bhsd'
    alibi_slopes = None

    if DEBUG_INPUT:
        q = torch.arange(N_CTX_Q, dtype=dtype, device="cuda").view(1, 1, N_CTX_Q, 1).expand(Z, H, N_CTX_Q, D_HEAD).contiguous().requires_grad_()
        k = torch.arange(N_CTX_K, dtype=dtype, device="cuda").view(1, 1, N_CTX_K, 1).expand(Z, H, N_CTX_K, D_HEAD).contiguous().requires_grad_()
        v = torch.arange(N_CTX_K, dtype=dtype, device="cuda").view(1, 1, N_CTX_K, 1).expand(Z, H, N_CTX_K, D_HEAD).contiguous().requires_grad_()
        do = torch.ones_like(q).contiguous()
    else:
        # Generate random inputs
        q = torch.randn(Z, H, N_CTX_Q, D_HEAD, device='cuda', dtype=dtype, requires_grad=True)
        k = torch.randn(Z, H, N_CTX_K, D_HEAD, device='cuda', dtype=dtype, requires_grad=True)
        v = torch.randn(Z, H, N_CTX_K, D_HEAD, device='cuda', dtype=dtype, requires_grad=True)
        do = torch.randn_like(q)

    # =============================================== Reference ==============================================================
    q_ref = q.clone() 
    k_ref = k.clone()
    v_ref = v.clone()    
    o_ref, softmax_lse_ref, exp_scores_ref, softmax_ref, attention_shifted_scaled_scores_ref, attention_scores_ref = attention_forward_pytorch_ref_impl(q_ref, k_ref, v_ref, sm_scale, causal, layout, use_exp2=use_exp2)
    print("attention_scores_ref:", attention_scores_ref, attention_scores_ref.shape)
    print("attention_shifted_scaled_scores_ref:", attention_shifted_scaled_scores_ref, attention_shifted_scaled_scores_ref.shape)
    print("exp_scores_ref:", exp_scores_ref, exp_scores_ref.shape)
    print("softmax_lse_ref:", softmax_lse_ref, softmax_lse_ref.shape)
    print("softmax_ref:", softmax_ref, softmax_ref.shape)

    dq = torch.zeros_like(q, dtype=q.dtype) # NOTE: the kernel does inplace accumlation on dq so dq has to be zeros
    if DEBUG_INPUT:
        dk = torch.zeros_like(k, dtype=k.dtype)
        dv = torch.zeros_like(v, dtype=v.dtype)
    else:
        dk = torch.empty_like(k, dtype=k.dtype)
        dv = torch.empty_like(v, dtype=v.dtype)

    do_ref = do.clone()
    dq_ref, dk_ref, dv_ref = attention_backward_pytorch_ref_impl(do_ref, q_ref, k_ref, v_ref, o_ref, softmax_lse_ref, sm_scale, causal, layout, use_exp2=use_exp2)

    # =============================================== Triton ==============================================================
    o = o_ref.clone()
    softmax_lse = softmax_lse_ref.clone()
    dq, dk, dv, _, _, _ = attention_prefill_backward_impl(do, q, k, v, o, softmax_lse, dq, dk, dv, sm_scale, head_size, alibi_slopes, causal, layout, use_exp2=use_exp2, use_new=use_new)

    # =============================================== Check ==============================================================
    print()
    print("dv:", dv, dv.shape)
    print("dv_ref:", dv_ref, dv_ref.shape)
    torch.testing.assert_close(dv, dv_ref, atol=ATOL, rtol=RTOL)
    print("dk:", dk, dk.shape)
    print("dk_ref:", dk_ref, dk_ref.shape)
    torch.testing.assert_close(dk, dk_ref, atol=ATOL, rtol=RTOL)
    print("dq:", dq, dq.shape)
    print("dq_ref:", dq_ref, dq_ref.shape)
    torch.testing.assert_close(dq, dq_ref, atol=ATOL, rtol=RTOL)

def nonvarlen_benchmark_configs():
    configs = [
        (16, 16, 16, 1024, 1024),
        (8, 16, 16, 2048, 2048),
        (4, 16, 16, 4096, 4096),
        (2, 16, 16, 8192, 8192),
        (1, 16, 16, 16384, 16384),
        (2, 48, 48, 1024, 1024),
        (2, 48, 48, 2048, 1024),
        (2, 48, 48, 4096, 8192),
        (2, 48, 48, 8192, 4096),
        (2, 48, 48, 16384, 8192),
        (8, 16, 16, 1989, 15344),
        (4, 16, 16, 4097, 163),
        (2, 16, 16, 8122, 2159),
        (1, 16, 16, 16281, 7),
        (2, 48, 48, 1021, 1020),
        (2, 48, 48, 2001, 2048),
        (2, 48, 48, 3996, 9639),
        (2, 48, 48, 8181, 1021),
    ]
    return configs


def varlen_benchmark_configs():
    configs = [
        (2, 16, 4, 1024, 1024),
        (8, 16, 2, 2048, 2048),
        (4, 16, 8, 4096, 4096),
        (2, 16, 4, 8192, 8192),
        (2, 16, 8, 16384, 16384),
        (2, 48, 12, 1024, 1024),
        (2, 48, 24, 2048, 2048),
        (2, 48, 8, 4096, 4096),
        (2, 48, 4, 8192, 8192),
        (2, 48, 2, 16384, 16384),
        (2, 64, 32, 1024, 1024),
        (4, 64, 16, 2048, 2048),
        (4, 64, 8, 4096, 4096),
        (4, 64, 32, 8192, 8192),
        (4, 128, 16, 16384, 16384),
    ]
    return configs


def run_benchmark(custom, args):

    dtype = arg_to_torch_dtype[args.dtype]
    hk = args.hq if not args.hk else args.hk
    sk = args.sq if not args.sk else args.sk
    head_size = 128 if not args.d else args.d
    mode = 'fwd'
    x_names = ['BATCH', 'HQ', 'HK', 'N_CTX_Q', 'N_CTX_K']
    causal = args.causal
    varlen = args.layout == 'thd'
    configs = []
    if custom:
        x_vals_list = [(args.b, args.hq, hk, args.sq, sk)]
    else:
        if varlen:
            x_vals_list = varlen_benchmark_configs()
        else:
            x_vals_list = nonvarlen_benchmark_configs()
    print_time = args.return_time
    line_names = 'Time (ms)' if print_time else 'TFLOPS'
    configs.append(
        triton.testing.Benchmark(x_names=x_names, x_vals=x_vals_list, line_arg='provider', line_vals=['triton'],
                                 line_names=[line_names], styles=[('red', '-')], ylabel='ms',
                                 plot_name=f'fused-attention-{mode}-d{head_size}-layout{args.layout}',
                                 args={'D_HEAD': head_size, 'dtype': dtype, 'causal': causal, 'mode': mode}))

    @triton.testing.perf_report(configs)
    def bench_flash_attention(BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, causal, mode, provider, device="cuda"):
        assert mode in ["fwd", "bwd"]
        warmup = 25
        rep = 100
        # TODO: Enable bias after testing.
        # if use_bias:
        #     bias = torch.randn((1, H, N_CTX, N_CTX), dtype=torch.float32, device="cuda")
        #     input_metadata.need_bias(bias, BATCH, H, N_CTX, N_CTX)
        # else:
        #     bias = None
        # bias = None

        # Bwd pass only supports causal=True right now
        if mode == 'bwd':
            causal = True

        flops_per_matmul = 0
        if varlen:
            q, k, v, input_metadata = varlen_input_helper(BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype,
                                                          args.equal_seqlens)
            for i in range(0, input_metadata.num_contexts):
                seqlen_q = input_metadata.cu_seqlens_q[i + 1] - input_metadata.cu_seqlens_q[i]
                seqlen_k = input_metadata.cu_seqlens_k[i + 1] - input_metadata.cu_seqlens_k[i]
                # x2 for 2 GEMMs
                flops_per_matmul += seqlen_q.item() * seqlen_k.item() * HQ * D_HEAD * 2
        else:
            q, k, v, input_metadata = input_helper(BATCH, HQ, HK, N_CTX_Q, N_CTX_K, D_HEAD, dtype, args.layout)
            flops_per_matmul = 2.0 * BATCH * HQ * N_CTX_Q * N_CTX_K * D_HEAD
        if causal:
            input_metadata.need_causal()
        o = torch.empty_like(q)
        fn = lambda: attention_prefill(q, k, v, o, input_metadata)
        if mode == 'bwd':
            o, _, _= fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
        total_flops = 2 * flops_per_matmul
        # TODO: This needs to be fixed for unequal Q/K seqlens
        if causal:
            total_flops *= 0.5
        if mode == "bwd":
            total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
        if print_time:
            return ms
        else:
            return total_flops / ms * 1e-9

    bench_flash_attention.run(save_path=".", print_data=True)


def supported_layouts():
    layouts = \
        'bhsd: Q, K, V are individual tensors of [batch, num_heads, seqlen_q/k, head_size]' \
        'bshd: Q, K, V are individual tensors of [batch, seqlen_q/k, num_heads, head_size]' \
        'thd: Q, K, V are individual tensors of [total_q/k, num_heads, head_size]' \
        'This layout is sometimes called "varlen" or "grouped" layout.'
    return layouts


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Benchmark FlashAttention",
        allow_abbrev=False,
    )
    parser.add_argument("-b", type=int, default=0)
    parser.add_argument("-hq", type=int, default=0)
    parser.add_argument("-hk", type=int, default=0)
    parser.add_argument("-sq", type=int, default=0)
    parser.add_argument("-sk", type=int, default=0)
    parser.add_argument("-equal_seqlens", action='store_true', default=False,
                        help='If specified, each context within the thd layout' \
                            ' has same seqlen as sq and sk')
    parser.add_argument("-d", type=int, default=0)
    parser.add_argument("-causal", action='store_true', default=False)
    parser.add_argument("-dtype", default='fp16')
    parser.add_argument("-return_time", action='store_true', default=False)
    parser.add_argument("-layout", type=str, default='bhsd', help=supported_layouts())
    return parser.parse_args()


arg_to_torch_dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}


def main():
    args = parse_args()
    custom_config = False
    assert args.layout == 'thd' or not args.equal_seqlens, \
           "Equal sequence lengths arg must be used with the thd layout."
    if args.b or args.hq or args.hk or args.sq or args.sk or args.d:
        custom_config = True
        assert args.b and args.hq and args.sq and args.d, \
               "If custom config is specified, please provide \
                all of batch, number of Q heads, Q sequence length \
                and head size."

    assert args.dtype in arg_to_torch_dtype, \
           "Only fp16, bf16 and f32 types currently supported."

    run_benchmark(custom_config, args)


if __name__ == '__main__':
    sys.exit(main())