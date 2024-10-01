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
import pytest
import sys
import torch

import triton
import triton.language as tl

from triton import cdiv


DEBUG = True


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
    dropout_p, return_encoded_softmax = 0.0, False

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
                f"  return_encoded_softmax={self.return_encoded_softmax}\n"
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

    def need_dropout(self, dropout_p, return_encoded_softmax):
        self.dropout_p = dropout_p
        self.return_encoded_softmax = return_encoded_softmax

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
            # assert not self.return_encoded_softmax
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
                    actual_seqlen_k, actual_seqlen_q, dropout_p, philox_seed, batch_philox_offset, encoded_sm_ptrs,
                    block_min, block_max, offs_n_causal, masked_blocks, n_extra_tokens, alibi_slope,
                    IS_CAUSAL: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
                    OFFS_M: tl.constexpr, OFFS_N: tl.constexpr, PRE_LOAD_V: tl.constexpr, MASK_STEPS: tl.constexpr,
                    ENABLE_DROPOUT: tl.constexpr, RETURN_ENCODED_SOFTMAX: tl.constexpr, PADDED_HEAD: tl.constexpr,
                    ACTUAL_BLOCK_DMODEL: tl.constexpr):
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

        if IS_CAUSAL:
            causal_boundary = start_n + offs_n_causal
            causal_mask = OFFS_M[:, None] >= causal_boundary[None, :]
            qk = tl.where(causal_mask, qk, float("-inf"))
        if bias_ptrs is not None:
            bias_offs_n = start_n + tl.arange(0, BLOCK_N) if MASK_STEPS else None
            bias = load_fn(bias_ptrs, OFFS_M, bias_offs_n, actual_seqlen_q, actual_seqlen_k)
            # While bias is added after multiplying qk with sm_scale,
            # our optimization to use 2^x instead of e^x results in an additional
            # scale factor of log2(e) which we must also multiply the bias with.
            qk += (bias * 1.44269504089)

        if alibi_slope is not None:
            # Compute the global position of each token within the sequence
            global_m_positions = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
            global_n_positions = start_n + tl.arange(0, BLOCK_N)
            alibi_block = compute_alibi_block(alibi_slope, actual_seqlen_q, actual_seqlen_k, global_m_positions,
                                              global_n_positions)
            qk += (alibi_block * 1.44269504089)  # scale factor of log2(e)

        # get max scores so far
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        # shift scores to 0
        qk = qk - m_ij[:, None]
        # take exponential of scores
        p = tl.math.exp2(qk)

        # CAVEAT: Must update l_ij before applying dropout
        l_ij = tl.sum(p, 1)
        if ENABLE_DROPOUT:
            philox_offset = batch_philox_offset + start_m * BLOCK_M * actual_seqlen_k + start_n - BLOCK_N
            keep = dropout_mask(philox_seed, philox_offset, dropout_p, BLOCK_M, BLOCK_N, actual_seqlen_k)
            if RETURN_ENCODED_SOFTMAX:
                tl.store(encoded_sm_ptrs, tl.where(keep, p, -p))
            p = tl.where(keep, p, 0.0)
        elif RETURN_ENCODED_SOFTMAX:
            tl.store(encoded_sm_ptrs, p)
        
        # -- update output accumulator --
        alpha = tl.math.exp2(m_i - m_ij) # adjustment factor for acc and li as we loop and find new maxes
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
        if RETURN_ENCODED_SOFTMAX:
            encoded_sm_ptrs += BLOCK_N
    return acc, l_i, m_i


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'waves_per_eu': 2, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'waves_per_eu': 2, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'waves_per_eu': 2, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 3, 'PRE_LOAD_V': True}, num_stages=1,
                      num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 3, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'waves_per_eu': 4, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 1, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'waves_per_eu': 4, 'PRE_LOAD_V': False}, num_stages=1,
                      num_warps=8),
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
def attn_fwd(Q, K, V, bias, sm_scale, LSE, Out, stride_qz, stride_qh, stride_qm, stride_qk, stride_kz, stride_kh,
             stride_kn, stride_kk, stride_vz, stride_vh, stride_vk, stride_vn, stride_oz, stride_oh, stride_om,
             stride_on, stride_bz, stride_bh, stride_bm, stride_bn, stride_az, stride_ah,
             stride_sz, stride_sh, stride_sm, stride_sn, cu_seqlens_q, cu_seqlens_k,
             dropout_p, philox_seed, philox_offset_base, encoded_softmax, alibi_slopes, HQ: tl.constexpr,
             HK: tl.constexpr, ACTUAL_BLOCK_DMODEL: tl.constexpr, MAX_SEQLENS_Q: tl.constexpr,
             MAX_SEQLENS_K: tl.constexpr, VARLEN: tl.constexpr, IS_CAUSAL: tl.constexpr, BLOCK_M: tl.constexpr,
             BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr, PRE_LOAD_V: tl.constexpr, USE_BIAS: tl.constexpr,
             ENABLE_DROPOUT: tl.constexpr, RETURN_ENCODED_SOFTMAX: tl.constexpr, USE_ALIBI: tl.constexpr):
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

    if ENABLE_DROPOUT:
        off_hz = off_z * HQ + off_h_q
        batch_philox_offset = philox_offset_base + off_hz * seqlen_q * seqlen_k
    else:
        batch_philox_offset = 0
    # We can ask to return the dropout mask without actually doing any dropout. In
    # this case, we return an invalid pointer so indicate the mask is not valid.
    if RETURN_ENCODED_SOFTMAX:
        encoded_sm_offset = encoded_softmax + off_z * stride_sz + off_h_q * stride_sh + cu_seqlens_q_start * stride_sm
        encoded_sm_ptrs = encoded_sm_offset + offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn
    else:
        encoded_sm_ptrs = None
    # initialize pointer to m and l
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    
    # scale sm_scale by log_2(e) and use 2^x in the loop as we do not
    # have native e^x support in HW.
    qk_scale = sm_scale * 1.44269504089
    # Q is loaded once at the beginning and shared by all N blocks.
    q_ptrs_mask = offs_m[:, None] < seqlen_q
    if PADDED_HEAD:
        q_ptrs_mask = q_ptrs_mask & (offs_d[None, :] < ACTUAL_BLOCK_DMODEL)
    q = tl.load(q_ptrs, mask=q_ptrs_mask, other=0.0)
    q = (q * qk_scale).to(q.type.element_ty)

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
                                        encoded_sm_ptrs,
                                        # _, _, offs_n_causal, masked_blocks, n_extra_tokens, _
                                        block_min, block_max, 0, 0, 0, alibi_slope,
                                        # IS_CAUSAL, ....
                                        False, BLOCK_M, BLOCK_DMODEL, BLOCK_N, offs_m, offs_n,
                                        # _, MASK_STEPS, ...
                                        PRE_LOAD_V, False, ENABLE_DROPOUT, RETURN_ENCODED_SOFTMAX, PADDED_HEAD,
                                        ACTUAL_BLOCK_DMODEL)
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
        if RETURN_ENCODED_SOFTMAX:
            encoded_sm_ptrs += n_full_blocks * BLOCK_N
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, k_ptrs, v_ptrs, bias_ptrs, stride_kn, stride_vk, stride_bn,
                                        start_m, seqlen_k, seqlen_q, dropout_p, philox_seed, batch_philox_offset,
                                        encoded_sm_ptrs, block_min, block_max, offs_n_causal, masked_blocks,
                                        n_extra_tokens, alibi_slope, IS_CAUSAL, BLOCK_M, BLOCK_DMODEL, BLOCK_N, offs_m,
                                        offs_n,
                                        # _, MASK_STEPS, ...
                                        PRE_LOAD_V, True, ENABLE_DROPOUT, RETURN_ENCODED_SOFTMAX, PADDED_HEAD,
                                        ACTUAL_BLOCK_DMODEL)
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
    if overflow_size > 0:
        boundary = tl.full((BLOCK_M, ), BLOCK_M - overflow_size, dtype=tl.int32)
        l_ptrs_mask = tl.arange(0, BLOCK_M) < boundary
        tl.store(l_ptrs, m_i + tl.math.log2(l_i), mask=l_ptrs_mask) # the log of the normalization constant
    else:
        tl.store(l_ptrs, m_i + tl.math.log2(l_i)) # the log of the normalization constant

    # write back O
    o_offset = Out + off_z * stride_oz + off_h_q * stride_oh + cu_seqlens_q_start * stride_om
    o_ptrs = o_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
    o_ptrs_mask = tl.full([BLOCK_M, BLOCK_DMODEL], 1, dtype=tl.int1)
    if overflow_size > 0:
        o_ptrs_mask = o_ptrs_mask & (offs_m[:, None] < seqlen_q)
    if PADDED_HEAD:
        o_ptrs_mask = o_ptrs_mask & (offs_d[None, :] < ACTUAL_BLOCK_DMODEL)
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=o_ptrs_mask)


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

    # encoded_softmax is used to validate dropout behavior vs the PyTorch SDPA math backend reference.  We zero this out
    # to give a consistent starting point and then populate it with the output of softmax with the sign bit set according
    # to the dropout mask. The resulting return allows this mask to be fed into the reference implementation for testing
    # only.  This return holds no useful output aside from debugging.
    if metadata.return_encoded_softmax:
        encoded_softmax = torch.zeros((batch, nheads_q, metadata.max_seqlens_q, metadata.max_seqlens_k), device=q.device,
                                        dtype=torch.float32)
        softmax_strides = (encoded_softmax.stride(0), encoded_softmax.stride(1), encoded_softmax.stride(2),
                        encoded_softmax.stride(3))
    else:
        encoded_softmax = None
        softmax_strides = (0, 0 , 0 , 0)

    # stores LSE the log of the normalization constant / sum of expoential score(unnormalzied probablities)
    LSE = torch.empty((batch, nheads_q, metadata.max_seqlens_q), device=q.device, dtype=torch.float32)

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

    attn_fwd[grid](q, k, v, metadata.bias, metadata.sm_scale, LSE, o, *q_strides, *k_strides, *v_strides, *o_strides,
                    *bias_strides, *alibi_strides, *softmax_strides, metadata.cu_seqlens_q, metadata.cu_seqlens_k,
                    dropout_p=metadata.dropout_p, philox_seed=philox_seed, philox_offset_base=philox_offset,
                    encoded_softmax=encoded_softmax, alibi_slopes=metadata.alibi_slopes, HQ=nheads_q, HK=nheads_k,
                    ACTUAL_BLOCK_DMODEL=head_size, MAX_SEQLENS_Q=metadata.max_seqlens_q,
                    MAX_SEQLENS_K=metadata.max_seqlens_k, IS_CAUSAL=metadata.causal, VARLEN=metadata.varlen,
                    BLOCK_DMODEL=padded_d_model, USE_BIAS=False if metadata.bias is None else True,
                    USE_ALIBI=False if metadata.alibi_slopes is None else True, ENABLE_DROPOUT=metadata.dropout_p
                    > 0.0, RETURN_ENCODED_SOFTMAX=metadata.return_encoded_softmax)

    return o, LSE, encoded_softmax, q, k , v, grid, head_size, philox_seed, philox_offset

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
    qk_scale,
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
        qk *= qk_scale
        l_i = tl.load(l_ptrs + offs_m_curr)
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
    CAUSAL: tl.constexpr
):
    qk_scale = sm_scale * 1.44269504
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
                qk_scale,
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
                CAUSAL=CAUSAL
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
            CAUSAL=CAUSAL
        )




def attention_prefill_backward_baseline_impl(do, q, k, v, o, L, sm_scale, head_size, alibi_slopes, layout):
    if DEBUG:
        print()
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

    if DEBUG:
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        delta = torch.zeros_like(L)
    else:
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        delta = torch.empty_like(L)
    
    
    batch_headsize = batch * heads_q
    num_blocks_m = cdiv(N_CTX_Q, BLOCK_M)
    num_blocks_n = cdiv(N_CTX_K, BLOCK_N)


    if DEBUG:
        print("before _bwd_preprocess")
        print("o:", o, o.shape)
        print("do:", do, do.shape)
        print("delta:", delta, delta.shape)
        print("BLOCK_M:", BLOCK_M)
        print("D_HEAD:", BLOCK_DMODEL)

    _bwd_preprocess[(num_blocks_m * batch_headsize,)](
        o,
        do,
        delta,
        BLOCK_M=BLOCK_M,
        D_HEAD=BLOCK_DMODEL,
    )

    if DEBUG:
        print("after _bwd_preprocess")
        print("o:", o, o.shape)
        print("do:", do, do.shape)
        print("delta:", delta, delta.shape)
        print("BLOCK_M:", BLOCK_M)
        print("D_HEAD:", BLOCK_DMODEL)

    
    if DEBUG:
        print("before _bwd_kernel")
        print("q:", q, q.shape)
        print("k:", k, k.shape)
        print("v:", v, v.shape)
        print("sm_scale", sm_scale)
        print("o:", o, o.shape)
        print("do:", do, do.shape)
        print("dq:", dq, dq.shape)
        print("dk:", dk, dk.shape)
        print("dv:", dv, dv.shape)
        print("L:", L, L.shape)
        print("delta:", delta, delta.shape)

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
        L,
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

    return dq, dk, dv, None, None


class _attention_prefill(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, o, metadata):
        o, L, encoded_softmax, q, k , v, grid, head_size, philox_seed, philox_offset = attention_prefill_forward_impl(q, k, v, o, metadata)

        ctx.save_for_backward(q, k, v, o, L)
        ctx.grid = grid
        ctx.sm_scale = metadata.sm_scale
        ctx.BLOCK_DMODEL = head_size
        ctx.causal = metadata.causal
        ctx.alibi_slopes = metadata.alibi_slopes
        ctx.dropout_p = metadata.dropout_p
        ctx.philox_seed = philox_seed
        ctx.philox_offset = philox_offset
        ctx.encoded_softmax = encoded_softmax
        ctx.return_encoded_softmax = metadata.return_encoded_softmax
        ctx.layout = metadata.layout
        return o, L, encoded_softmax

    @staticmethod
    def backward(ctx, do, *args): # expects bhsd
        q, k, v, o, M = ctx.saved_tensors
        
        return attention_prefill_backward_baseline_impl(do, q, k, v, o, M, ctx.sm_scale, ctx.BLOCK_DMODEL, ctx.alibi_slopes, ctx.layout)

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
    torch.testing.assert_close(ref_out, tri_out, atol=1e-2, rtol=1e-2)


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
    torch.testing.assert_close(ref_out, tri_out, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize('Z, H, N_CTX_Q, N_CTX_K, D_HEAD', [
    # doesnot work
    # (1, 1, 1, 1,  1),
    # (1, 1, 1, 1,  16),
    (1, 1, 2, 2,  16),
    # (1, 1, 4, 4, 2),
    # (1, 1, 4, 3),
    # (1, 1, 3, 4),
    # (1, 1, 4, 4),
    # (1, 1, 4, 7),
    # (1, 1, 16, 16),
    # (1, 1, 32, 32, 32),
    # works
    # (1, 1, 64, 64, 64),
    # (1, 16, 1022, 1022, 64),
    # (4, 48, 1024, 1024, 64),
    # (4, 48, 2048, 2048, 64),
    # (2, 48, 4096, 4096, 64),
    # (1, 16, 1024, 1024, 64),
    # (1, 16, 1024, 1024, 128),
    # (1, 16, 8192, 8192, 63),
])
# @pytest.mark.parametrize('torch_sdpa_test', [False, True])
@pytest.mark.parametrize('torch_sdpa_test', [False])
# @pytest.mark.parametrize('causal', [True, False])
@pytest.mark.parametrize('causal', [False])
# @pytest.mark.parametrize('use_alibi', [False, True])
@pytest.mark.parametrize('use_alibi', [False])
def test_op_bwd(Z, H, N_CTX_Q, N_CTX_K, D_HEAD, causal, torch_sdpa_test, use_alibi, dtype=torch.float16):
    torch.manual_seed(20)

    # seqlens
    seqlen_q = N_CTX_Q
    seqlen_k = N_CTX_K

    # setup up metadata
    if DEBUG:
        sm_scale = 1
    else:
        sm_scale = D_HEAD**-0.5
    input_metadata = MetaData(sm_scale=sm_scale)
    input_metadata.max_seqlens_q = seqlen_q
    input_metadata.max_seqlens_k = seqlen_k
    input_metadata.layout = "bhsd"

    dropout_p = 0
    if DEBUG:
        q = torch.arange(seqlen_q, dtype=dtype, device="cuda").view(1, 1, seqlen_q, 1).expand(Z, H, seqlen_q, D_HEAD).requires_grad_()
        k = torch.arange(seqlen_k, dtype=dtype, device="cuda").view(1, 1, seqlen_k, 1).expand(Z, H, seqlen_k, D_HEAD).requires_grad_()
        v = torch.arange(seqlen_k, dtype=dtype, device="cuda").view(1, 1, seqlen_k, 1).expand(Z, H, seqlen_k, D_HEAD).requires_grad_()
        o = torch.zeros_like(q)
    else:
        q = (torch.empty((Z, H, seqlen_q, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
        k = (torch.empty((Z, H, seqlen_k, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
        v = (torch.empty((Z, H, seqlen_k, D_HEAD), dtype=dtype, device="cuda").normal_(mean=0.0, std=0.5).requires_grad_())
        o = torch.empty_like(q)       

    if causal:
        input_metadata.need_causal()

    if use_alibi and not torch_sdpa_test:
        # for n heads the set of slopes is the geometric sequence that starts 2^(-8/n)
        alibi_slopes = torch.tensor([2**(-8 / H * i) for i in range(1, H + 1)], dtype=torch.float32,
                                    device="cuda").repeat(Z, 1)
        input_metadata.need_alibi(alibi_slopes, Z, H)

    if DEBUG:
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
        # print("ref_p:", p )
        if use_alibi:
            p += compute_alibi_tensor(alibi_slopes, N_CTX_Q, N_CTX_K)
        if causal:
            p[:, :, M == 0] = float("-inf")

        ref_softmax = torch.softmax(p.float(), dim=-1).type(dtype=p.dtype)
        print("ref_softmax:", ref_softmax )
        ref_out = torch.matmul(ref_softmax, v)
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