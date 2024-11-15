import triton
import triton.language as tl

@triton.jit
def cdiv_fn(x, y):
    return (x + y - 1) // y

@triton.jit
def dropout_mask(philox_seed, philox_offset, dropout_p, m, n, stride):
    # calculate RNG offsets using the philox_offset and strides
    ms = tl.arange(0, m)
    ns = tl.arange(0, n)
    rng_offsets = (philox_offset + ms[:, None] * stride + ns[None, :]).to(tl.uint32)
    
    # get rng output
    rng_output = tl.rand(philox_seed, rng_offsets) # TODO: use tl.randint for better performance

    
    # keep 1 - dropout elements
    rng_keep = rng_output > dropout_p

    return rng_keep