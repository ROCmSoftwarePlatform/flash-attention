import functools
import torch
import triton
import triton.language as tl

@triton.jit
def kernel_that_uses_dropout(
    output_ptr,
    philox_seed,
    philox_offset_base,
    dropout_p,
    stride_sz, stride_sh, stride_sm, stride_sn,
    seqlen_q,
    seqlen_k,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_h_q = tl.program_id(1)
    off_z = tl.program_id(2)

    # use for loop to iterate along seqlen_k dim
    start_n = 0

    # not varlen
    cu_seqlens_q_start = 0
    cu_seqlens_k_start = 0

    # Calculate the global offsets for the current block
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    n_blocks = tl.cdiv(seqlen_k, BLOCK_N)
    for start_n in range(0, n_blocks):
        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]

        batch_philox_offset = philox_offset_base + off_z * stride_sz + off_h_q * stride_sh + cu_seqlens_q_start * stride_sm
        philox_offset = batch_philox_offset + offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn

        # Generate the dropout mask
        rng_output = tl.rand(philox_seed, philox_offset)
        keep = rng_output > dropout_p
        
        # Store the result
        output_offset = output_ptr +  off_z * stride_sz + off_h_q * stride_sh + cu_seqlens_q_start * stride_sm
        output_ptrs = output_offset + offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn
        tl.store(output_ptrs, keep)

def tl_rand_ref(philox_seed, rng_offsets):
    device = rng_offsets.device
    # Ensure rng_offsets is contiguous
    rng_offsets = rng_offsets.contiguous()
    N = rng_offsets.numel()
    # Prepare output tensor
    output = torch.empty_like(rng_offsets, dtype=torch.float32)
    # Define block size
    BLOCK_SIZE = 1024

    @triton.jit
    def _tl_rand_kernel(output_ptr, offsets_ptr, philox_seed, N,
                        BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(0)
        idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = idx < N
        offsets = tl.load(offsets_ptr + idx, mask=mask, other=0)
        r = tl.rand(philox_seed, offsets)
        tl.store(output_ptr + idx, r, mask=mask)

    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)

    # Launch the Triton kernel
    _tl_rand_kernel[grid](
        output_ptr=output,
        offsets_ptr=rng_offsets,
        philox_seed=philox_seed,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output

# NOTE: cache result otherwise it is slow
@functools.cache 
def dropout_mask_ref(philox_seed, philox_offset, dropout_p, m, n, stride, device):
    # calculate RNG offsets (same as in Triton)
    ms = torch.arange(0, m, device=device)
    ns = torch.arange(0, n, device=device)
    rng_offsets = (philox_offset + ms[:, None] * stride + ns[None, :]).to(torch.uint32).to(device=device)

    rng_output = tl_rand_ref(philox_seed, rng_offsets) 

    # apply dropout mask
    rng_keep = rng_output > dropout_p

    return rng_keep

def generate_dropout_mask_ref(shape, dropout_p, philox_seed, philox_offset, device, dtype, BLOCK_M=128, BLOCK_N = 128):
    B, M, N = shape
    
    output =  torch.zeros(shape, dtype=torch.bool, device=device)
    for i in range(0, M, BLOCK_M):
        for j in range(0, N, BLOCK_N):
            m = min(BLOCK_M, M - i)
            n = min(BLOCK_N, N - j)
            # Generate the dropout mask for the current tile
            mask = dropout_mask_ref(
                philox_seed=philox_seed,
                philox_offset=philox_offset,
                dropout_p=dropout_p,
                m=m,
                n=n,
                stride=N,
                device=device,
            )
            # Store the result in the output tensor
            output[:, i : i + m, j : j + n] = mask
    
    return output, (1.0 / (1 - dropout_p))



def kernel_that_uses_dropout_ref(
    output_tensor,
    philox_seed,
    philox_offset,
    dropout_p,
    BLOCK_M,
    BLOCK_N,
    device,
):
    output = output_tensor
    for i in range(0, M, BLOCK_M):
        for j in range(0, N, BLOCK_N):
            m = min(BLOCK_M, M - i)
            n = min(BLOCK_N, N - j)
            # Generate the dropout mask for the current tile
            mask = dropout_mask_ref(
                philox_seed=philox_seed,
                philox_offset=philox_offset,
                dropout_p=dropout_p,
                m=m,
                n=n,
                stride=N,
                device=device,
            )
            # Store the result in the output tensor
            output[i : i + m, j : j + n] = mask
    return output


def test_dropout():
    # Set test parameters
    shape = (1, 1, 1024, 1024)
    batch, nheads_q, seqlen_q, seqlen_k = shape
    BLOCK_M, BLOCK_N = 32, 32
    dropout_p = 0.5
    philox_seed, philox_offset = 0x1BF58, 0x1D4B49
    device = "cuda"

    output = torch.zeros(shape, dtype=torch.bool, device=device)
    stride_sz, stride_sh, stride_sm, stride_sn = (output.stride(0), output.stride(1), output.stride(2), output.stride(3))

    # Run Triton implementation
    grid = lambda META: (triton.cdiv(seqlen_q, META['BLOCK_M']), nheads_q, batch)
    kernel_that_uses_dropout[grid](
        output_ptr=output,
        philox_seed=philox_seed,
        philox_offset_base=philox_offset,
        dropout_p=dropout_p,
        stride_sz=stride_sz,
        stride_sh=stride_sh,
        stride_sm=stride_sm,
        stride_sn=stride_sn,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    print("triton_output:", output)

    # Run PyTorch reference implementation
    # torch_output = torch.zeros(shape, dtype=torch.bool, device=device)
    # torch_output = kernel_that_uses_dropout_ref(
    #     output_tensor=torch_output,
    #     philox_seed=philox_seed,
    #     philox_offset=philox_offset,
    #     dropout_p=dropout_p,
    #     BLOCK_M=BLOCK_M,
    #     BLOCK_N=BLOCK_N,
    #     device=device,
    # )
    # print("torch_output:", torch_output)

    # Compare results
    print(f"Shape: {output.shape}")
    print(f"Expected ratio: {1 - dropout_p:.4f}")
    print(f"Triton keep ratio: {output.float().mean().item():.4f}")
    # print(f"PyTorch keep ratio: {torch_output.float().mean().item():.4f}")

    # Check if patterns match
    # matches = (output == torch_output).float().mean().item()
    # print(f"\nPattern match ratio: {matches:.4f}")

    # if matches > 0.99:  # Allow for small differences
    #     print("✓ Implementations match!")
    # else:
    #     print("✗ Implementations differ!")
    # return output, torch_output


def compute_alibi_tensor_ref(alibi_slopes, seqlen_q, seqlen_k):
    q_idx = torch.arange(seqlen_q, dtype=torch.int32, device="cuda").unsqueeze(-1)  # (N_CTX_Q, 1)
    k_idx = torch.arange(seqlen_k, dtype=torch.int32, device="cuda").unsqueeze(0)  # (1, N_CTX_K)
    relative_pos = torch.abs(q_idx + seqlen_k - seqlen_q - k_idx)  # (N_CTX_Q, N_CTX_K)
    return -1 * alibi_slopes.unsqueeze(-1).unsqueeze(-1) * relative_pos  # (Z, H, N_CTX_Q, N_CTX_K)

if __name__ == "__main__":
    test_dropout()
