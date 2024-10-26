Attention Fused Kernel
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


#### Triton Backend
FlashAttention-2 ROCm Triton backend is a work in progress. 
It current supports Forwards only. However some features like PagedAttention and Sliding Window are missing. It can run on both MI and Navi Machines. We are working on backwards.

Inorder to use the triton backend for rocm, follow the steps below.

First install the recommended Triton [commit](https://github.com/triton-lang/triton/commit/2e9f2c2d20601c24b91a4c32a7b97ad1f8a55d88).

```
git clone https://github.com/triton-lang/triton
cd triton
git checkout 2e9f2c2d20601c24b91a4c32a7b97ad1f8a55d88 
pip install --verbose -e python
```
Then install and test Flash Attention with the flag `FLASH_ATTENTION_USE_TRITON_AMD` set to `"TRUE"`.

```
export FLASH_ATTENTION_USE_TRITON_AMD="TRUE"
cd flash-attention
python setup.py install
pytest tests/test_flash_attn.py
```

