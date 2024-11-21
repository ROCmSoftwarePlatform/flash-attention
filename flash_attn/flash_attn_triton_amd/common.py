import torch
import triton
import triton.language as tl

@triton.jit
def dropout_mask(philox_seed, philox_offset, dropout_p, m: tl.constexpr, n: tl.constexpr, stride):
    # calculate RNG offsets using the philox_offset and strides
    ms = tl.arange(0, m)
    ns = tl.arange(0, n)
    rng_offsets = (philox_offset + ms[:, None] * stride + ns[None, :]).to(tl.uint32)

    # get rng output
    rng_output = tl.rand(philox_seed, rng_offsets)  # TODO: use tl.randint for better performance

    # keep 1 - dropout elements
    rng_keep = rng_output > dropout_p

    return rng_keep

@triton.jit
def dropout_kernel_wrapper(
    output_ptr,
    philox_seed,
    philox_offset,
    dropout_p,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    mask = dropout_mask(
        philox_seed=philox_seed,
        philox_offset=philox_offset,
        dropout_p=dropout_p,
        m=BLOCK_M,
        n=BLOCK_N,
        stride=BLOCK_N
    )
    
    # Store the result
    output_ptrs = output_ptr + tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    tl.store(output_ptrs, mask)


def generate_dropout_mask_ref(shape, dropout_p, seed, offset, device, dtype):
    torch.manual_seed(seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    rand_vals = torch.rand(shape, generator=gen, device=device, dtype=dtype)
    return rand_vals >= dropout_p, (1.0 / (1 - dropout_p))

def test_dropout():
    # Set test parameters
    BLOCK_M, BLOCK_N = 8, 8
    dropout_p = 0.5
    philox_seed, philox_offset = 0x1BF58, 0x1D4B49
    device = 'cuda'
    
    # Run Triton implementation
    triton_output = torch.empty((BLOCK_M, BLOCK_N), dtype=torch.bool, device=device)
    dropout_kernel_wrapper[(1,)](
        output_ptr=triton_output,
        philox_seed=philox_seed,
        philox_offset=philox_offset,
        dropout_p=dropout_p,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    print("triton_output:", triton_output)
    
    # Run PyTorch reference implementation
    torch_output, scale = generate_dropout_mask_ref(
        shape=(BLOCK_M, BLOCK_N),
        dropout_p=dropout_p,
        seed=philox_seed,
        offset=philox_offset,
        device=device,
        dtype=torch.float32
    )
    print("torch_output:", torch_output)
    
    # Compare results
    print(f"Shape: {triton_output.shape}")
    print(f"Triton keep ratio: {triton_output.float().mean().item():.4f}")
    print(f"PyTorch keep ratio: {torch_output.float().mean().item():.4f}")
    print(f"Expected ratio: {1 - dropout_p:.4f}")
    
    # Check if patterns match
    matches = (triton_output == torch_output).float().mean().item()
    print(f"\nPattern match ratio: {matches:.4f}")
    
    if matches > 0.99:  # Allow for small floating point differences
        print("✓ Implementations match!")
    else:
        print("✗ Implementations differ!")
        # Print a small sample of differences if they exist
        if False and  matches < 1.0:
            diff_mask = triton_output != torch_output
            diff_indices = diff_mask.nonzero() 
            print("\nDifferences (row, col):")
            for idx in diff_indices:
                row, col = idx.cpu()[0].item(),idx.cpu()[1].item()
                print(f"Position ({row}, {col}): Triton={triton_output[row,col].item()}, PyTorch={torch_output[row,col].item()}")
    
    return triton_output, torch_output


def compute_alibi_tensor_ref(alibi_slopes, seqlen_q, seqlen_k):
    q_idx = torch.arange(seqlen_q, dtype=torch.int32, device="cuda").unsqueeze(-1)  # (N_CTX_Q, 1)
    k_idx = torch.arange(seqlen_k, dtype=torch.int32, device="cuda").unsqueeze(0)  # (1, N_CTX_K)
    relative_pos = torch.abs(q_idx + seqlen_k - seqlen_q - k_idx)  # (N_CTX_Q, N_CTX_K)
    return -1 * alibi_slopes.unsqueeze(-1).unsqueeze(-1) * relative_pos  # (Z, H, N_CTX_Q, N_CTX_K)


if __name__ == "__main__":
    test_dropout()
