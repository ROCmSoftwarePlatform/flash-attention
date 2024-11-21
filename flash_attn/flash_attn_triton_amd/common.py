import triton
import triton.language as tl
import torch


@triton.jit
def dropout_mask(philox_seed, philox_offset, dropout_p, m, n, stride):
    # calculate RNG offsets using the philox_offset and strides
    ms = tl.arange(0, m)
    ns = tl.arange(0, n)
    rng_offsets = (philox_offset + ms[:, None] * stride + ns[None, :]).to(tl.uint32)

    # get rng output
    rng_output = tl.rand(philox_seed, rng_offsets)  # TODO: use tl.randint for better performance

    # keep 1 - dropout elements
    rng_keep = rng_output > dropout_p

    return rng_keep


def generate_dropout_mask_ref(shape, dropout_p, seed, offset, device, dtype):
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    rand_vals = torch.rand(shape, generator=gen, device=device, dtype=dtype)
    return rand_vals >= dropout_p, (1.0 / (1 - dropout_p))


def compute_alibi_tensor_ref(alibi_slopes, seqlen_q, seqlen_k):
    q_idx = torch.arange(seqlen_q, dtype=torch.int32, device="cuda").unsqueeze(-1)  # (N_CTX_Q, 1)
    k_idx = torch.arange(seqlen_k, dtype=torch.int32, device="cuda").unsqueeze(0)  # (1, N_CTX_K)
    relative_pos = torch.abs(q_idx + seqlen_k - seqlen_q - k_idx)  # (N_CTX_Q, N_CTX_K)
    return -1 * alibi_slopes.unsqueeze(-1).unsqueeze(-1) * relative_pos  # (Z, H, N_CTX_Q, N_CTX_K)
