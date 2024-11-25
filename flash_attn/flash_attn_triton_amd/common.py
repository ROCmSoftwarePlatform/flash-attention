import functools
import math
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
def kernel_that_uses_dropout(
    output_ptr,
    philox_seed,
    philox_offset,
    dropout_p,
    M,
    N,
    stride_out,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Get program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Calculate the global offsets for the current block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]

    # Create masks to handle boundaries
    mask_m = offs_m < M
    mask_n = offs_n < N
    mask = mask_m & mask_n

    # Adjust offsets to be within bounds
    offs_m = tl.where(mask_m, offs_m, 0)
    offs_n = tl.where(mask_n, offs_n, 0)

    # Generate the dropout mask
    block_mask = dropout_mask(
        philox_seed=philox_seed,
        philox_offset=philox_offset,
        dropout_p=dropout_p,
        m=BLOCK_M,
        n=BLOCK_N,
        stride=stride_out,
    )

    # Store the result
    output_ptrs = output_ptr + offs_m * stride_out + offs_n
    tl.store(output_ptrs, block_mask, mask=mask)


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
    shape,
    device,
):
    M, N = shape
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
    shape = (1024, 1024)
    BLOCK_M, BLOCK_N = 32, 32
    dropout_p = 0.5
    philox_seed, philox_offset = 0x1BF58, 0x1D4B49
    device = "cuda"

    # Run Triton implementation
    triton_output = torch.empty(shape, dtype=torch.bool, device=device)
    grid = lambda meta: (
        math.ceil(shape[0] / meta["BLOCK_M"]),
        math.ceil(shape[1] / meta["BLOCK_N"]),
    )
    kernel_that_uses_dropout[grid](
        output_ptr=triton_output,
        philox_seed=philox_seed,
        philox_offset=philox_offset,
        dropout_p=dropout_p,
        M=shape[0],
        N=shape[1],
        stride_out=shape[1],
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    print("triton_output:", triton_output)

    # Run PyTorch reference implementation
    torch_output = torch.empty(shape, dtype=torch.bool, device=device)
    torch_output = kernel_that_uses_dropout_ref(
        output_tensor=torch_output,
        philox_seed=philox_seed,
        philox_offset=philox_offset,
        dropout_p=dropout_p,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        shape=shape,
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
