import functools
import torch
import triton
import triton.language as tl

@triton.jit
def tl_rand(philox_seed, philox_offset):
    return tl.rand(philox_seed, philox_offset)

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

    # not varlen
    cu_seqlens_q_start = 0
    cu_seqlens_k_start = 0

    # Calculate the global offsets for the current block
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    n_blocks = tl.cdiv(seqlen_k, BLOCK_N)
    for start_n in range(0, n_blocks):
        offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]

        batch_philox_offset = philox_offset_base + off_z * stride_sz + off_h_q * stride_sh + cu_seqlens_q_start * stride_sm
        philox_offset = batch_philox_offset + offs_m * stride_sm + offs_n * stride_sn

        # print("philox_seed:", philox_seed)
        print("philox_offset:", philox_offset)

        # Generate the dropout mask
        rng_output = tl_rand(philox_seed, philox_offset)
        print("rng_output:", rng_output)
        # print("dropout_p:", dropout_p)
        keep = rng_output > dropout_p

        # print("keep:", keep)
        
        # Store the result
        output_offset = output_ptr +  off_z * stride_sz + off_h_q * stride_sh + cu_seqlens_q_start * stride_sm
        output_ptrs = output_offset + offs_m * stride_sm + offs_n * stride_sn
        tl.store(output_ptrs, keep)



def tl_rand_ref(philox_seed, philox_offset, BLOCK_M, BLOCK_N):
    @triton.jit
    def tl_rand_kernel(
        output_ptr,
        philox_seed,
        philox_offset_ptr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        # Calculate position in the output grid
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        
        # Calculate offsets for this block
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
        
        # Load philox offsets for this block
        philox_offset = tl.load(philox_offset_ptr + offs_m * BLOCK_N + offs_n)
        
        # Generate random numbers
        rng_output = tl.rand(philox_seed, philox_offset)
        
        # Store the result
        output_ptr = output_ptr + offs_m * BLOCK_N + offs_n
        tl.store(output_ptr, rng_output)


    # Get the shape of the philox_offset tensor
    shape = philox_offset.shape
    device = philox_offset.device
    
    # Create output tensor
    output = torch.zeros_like(philox_offset, dtype=torch.float32)
    
    # Define grid
    grid = (triton.cdiv(shape[0], BLOCK_M), triton.cdiv(shape[1], BLOCK_N))
    
    # Launch kernel
    tl_rand_kernel[grid](
        output_ptr=output,
        philox_seed=philox_seed,
        philox_offset_ptr=philox_offset,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    return output


def kernel_that_uses_dropout_ref(
    output_tensor,
    philox_seed,
    philox_offset_base,
    dropout_p,
    stride_sz, stride_sh, stride_sm, stride_sn,
    seqlen_q,
    seqlen_k,
    BLOCK_M,
    BLOCK_N,
    device,
):
    batch = output_tensor.size(0)
    nheads_q = output_tensor.size(1)
    
    # Iterate over the same program_id dimensions as Triton
    for start_m in range(0, seqlen_q, BLOCK_M):
        for off_h_q in range(nheads_q):
            for off_z in range(batch):
                # Iterate over seqlen_k dimension in blocks
                for start_n in range(0, seqlen_k, BLOCK_N):
                    
                    # Calculate global offsets matching Triton kernel
                    offs_m = start_m + torch.arange(0, BLOCK_M, device=device)[:, None]
                    offs_n = start_n + torch.arange(0, BLOCK_N, device=device)[None, :]
                    
                    # Calculate philox offsets
                    batch_philox_offset = (philox_offset_base + 
                                         off_z * stride_sz + 
                                         off_h_q * stride_sh)
                    philox_offset = (batch_philox_offset + 
                                   offs_m * stride_sm + 
                                   offs_n * stride_sn)

                    # print("philox_seed_ref:", philox_seed)
                    print("philox_offset_ref:", philox_offset)
                    
                    # Generate random values and apply dropout
                    rng_output = tl_rand_ref(philox_seed, philox_offset, BLOCK_M, BLOCK_N)
                    print("rng_output_ref:", rng_output)
                    # print("dropout_p_ref:", dropout_p)
                    keep = rng_output > dropout_p
                    # print("keep_ref:", keep)
                    
                    # Store results in the output tensor
                    output_tensor[off_z, off_h_q, 
                                offs_m, 
                                offs_n] = keep

    return output_tensor


def test_dropout():
    # Set test parameters
    shape = (1, 1, 32, 32)
    batch, nheads_q, seqlen_q, seqlen_k = shape
    BLOCK_M, BLOCK_N = 32, 32
    dropout_p = 0.5
    philox_seed, philox_offset = 0x1BF58, 0x1D4B49
    device = "cuda"

    triton_output = torch.zeros(shape, dtype=torch.bool, device=device)
    stride_sz, stride_sh, stride_sm, stride_sn = (triton_output.stride(0), triton_output.stride(1), triton_output.stride(2), triton_output.stride(3))

    # Run Triton implementation
    grid = lambda META: (triton.cdiv(seqlen_q, META['BLOCK_M']), nheads_q, batch)
    kernel_that_uses_dropout[grid](
        output_ptr=triton_output,
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
    print("triton_output:", triton_output)

    # Run PyTorch reference implementation
    torch_output = torch.zeros(shape, dtype=torch.bool, device=device)
    torch_output = kernel_that_uses_dropout_ref(
        output_tensor=torch_output,
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
        device=device,
    )
    print("torch_output:", torch_output)

    # Compare results
    print(f"Shape: {triton_output.shape}")
    print(f"Expected ratio: {1 - dropout_p:.4f}")
    print(f"Triton keep ratio: {triton_output.float().mean().item():.4f}")
    print(f"PyTorch keep ratio: {torch_output.float().mean().item():.4f}")

    # Check if patterns match
    matches = (triton_output == torch_output).float().mean().item()
    print(f"\nPattern match ratio: {matches:.4f}")

    if matches > 0.99:  # Allow for small differences
        print("✓ Implementations match!")
    else:
        print("✗ Implementations differ!")
    return triton_output, torch_output


def compute_alibi_tensor_ref(alibi_slopes, seqlen_q, seqlen_k):
    q_idx = torch.arange(seqlen_q, dtype=torch.int32, device="cuda").unsqueeze(-1)  # (N_CTX_Q, 1)
    k_idx = torch.arange(seqlen_k, dtype=torch.int32, device="cuda").unsqueeze(0)  # (1, N_CTX_K)
    relative_pos = torch.abs(q_idx + seqlen_k - seqlen_q - k_idx)  # (N_CTX_Q, N_CTX_K)
    return -1 * alibi_slopes.unsqueeze(-1).unsqueeze(-1) * relative_pos  # (Z, H, N_CTX_Q, N_CTX_K)

if __name__ == "__main__":
    test_dropout()
