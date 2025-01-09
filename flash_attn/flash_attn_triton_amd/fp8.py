import os
from typing import Optional

import torch
from torch import Tensor

from .utils import get_shape_from_layout, MetaData


REMOVE_QUANTIZATION_SCALING: bool = os.environ.get("FLASH_ATTENTION_TRITON_AMD_REMOVE_QUANT_SCALE", "0").lower() in ("1", "true", "yes")

FP8_TYPES: set[torch.dtype] = {
    torch.float8_e4m3fnuz,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float8_e5m2fnuz,
}

FP8_MAX: dict[torch.dtype, float] = {
    dtype: torch.finfo(dtype).max
    for dtype in FP8_TYPES
}

DEFAULT_SCALE_PER_HEAD: bool = True


def check_is_fp8(x: Tensor, *xs: Tensor) -> bool:
    """
    Checks whether the given tensors are of FP8 data types.

    This function determines if any of the input tensors have a data type
    matching one of the FP8 types defined in `FP8_TYPES`. If the environment
    variable `FLASH_ATTENTION_TRITON_AMD_REMOVE_QUANT_SCALE` is set to a
    truthy value, the function always returns `False`,  effectively disabling FP8
    type detection.

    Args:
        x (Tensor): The primary tensor to check.
        *xs (Tensor): Additional tensors to check.

    Returns:
        bool:
            - `True` if any of the input tensors have an FP8 data type and
              quantization scaling is not disabled.
            - `False` otherwise.
    """
    if REMOVE_QUANTIZATION_SCALING:
        return False  # makes all methods believe they aren't working with fp8s, so no scaling is applied
    return any(y.dtype in FP8_TYPES for y in (x,) + xs)


def create_fp8_scale_tensors(
    q: Tensor, k: Tensor, v: Tensor, layout: str,
    cu_seqlens_q: Optional[Tensor] = None, cu_seqlens_k: Optional[Tensor] = None,
    max_seqlen_q: Optional[int] = None, max_seqlen_k: Optional[int] = None,
    scale_per_head: bool = DEFAULT_SCALE_PER_HEAD, eps: float = 1e-9,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Create scale tensors for q, k and v based on the scaling configuration.

    Args:
        q (torch.Tensor): Query tensor.
        k (torch.Tensor): Key tensor.
        v (torch.Tensor): Value tensor.
        layout (str): Tensor layout, can be `"bhsd"`, `"bshd"` or `"thd"`.
        cu_seqlens_q (Optional[torch.Tensor]): Cumulative Q sequence length.
            Used with `"thd"` varlen layout.
        cu_seqlens_k (Optional[torch.Tensor]): Cumulative KV sequence length.
            Used with `"thd"` varlen layout.
        max_seqlen_q (Optional[int]): Max. Q sequence length.
            Used with `"thd"` varlen layout.
        max_seqlen_k (Optional[int]): Max. KV sequence length.
            Used with `"thd"` varlen layout.
        scale_per_head (bool): Whether to compute scale per head or globally.
            Defaults to `DEFAULT_SCALE_PER_HEAD.`
        eps (float): If the maximum absolute value of a tensor is zero, this
            constant avoids division by zero while scaling. Defaults to 1e-9.

    Returns:
        tuple of 2D torch.Tensor: `(q_scale, k_scale, v_scale, p_scale)`.
        To perform fp8 quantization, you should multiply by scale factor `(x_quant = x * x_scale)`.
        To perform fp8 dequantization, you should multiply by reciprocal of scale factor
        `(x = x_quant * (1 / x_scale))`.
        `p_scale` is related to intermediate FA computation `p = softmax(matmul(q, transpose(k)))`.
        All scale tensors are `float32` ones.
        `q_scale` and `p_scale` have `(BATCH, HEADS_Q)` shape.
        `k_scale` and `v_scale` have `(BATCH, HEADS_K)` shape.
    """
    assert layout in ["bhsd", "bshd", "thd"], "Unknow layout."
    is_varlen = layout == "thd"
    if is_varlen:
        assert cu_seqlens_q is not None, "cu_seqlens_q is required for varlen layout."
        assert cu_seqlens_k is not None, "cu_seqlens_k is required for varlen layout."
        assert max_seqlen_q is not None, "max_seqlen_q is required for varlen layout."
        assert max_seqlen_k is not None, "max_seqlen_k is required for varlen layout."

    is_fp8 = check_is_fp8(q, k, v)
    batch, head_q, head_k, _, _, _ = get_shape_from_layout(
        q, k, layout,
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
    )

    if not is_fp8:
        # For non-float8 dtypes, use a default scale of 1.
        q_scale = torch.ones((batch, head_q), dtype=torch.float32, device=q.device)
        k_scale = torch.ones((batch, head_k), dtype=torch.float32, device=k.device)
        v_scale = torch.ones((batch, head_k), dtype=torch.float32, device=v.device)
        p_scale = torch.ones((batch, head_q), dtype=torch.float32, device="cuda")

    else:
        # Handle float8 dtype special case.

        # Convert to float32 for scale computation.
        q_float32 = q.detach().to(torch.float32)
        k_float32 = k.detach().to(torch.float32)
        v_float32 = v.detach().to(torch.float32)

        if not scale_per_head:
            # Handle global scaling.

            # Compute global max and create a tensor of that value.
            q_global_max = max(q_float32.abs().max().item(), eps)
            k_global_max = max(k_float32.abs().max().item(), eps)
            v_global_max = max(v_float32.abs().max().item(), eps)

            q_scale = torch.full((batch, head_q), q_global_max, dtype=torch.float32, device=q.device)
            k_scale = torch.full((batch, head_k), k_global_max, dtype=torch.float32, device=k.device)
            v_scale = torch.full((batch, head_k), v_global_max, dtype=torch.float32, device=v.device)

        else:
            # Handle per batch / head scaling.
            teps = torch.tensor(eps)

            if is_varlen:
                q_scale = torch.stack([torch.maximum(q_float32[s:e].abs().amax(dim=(0, 2)), teps) for s, e in zip(cu_seqlens_q[:-1], cu_seqlens_q[1:])])
                k_scale = torch.stack([torch.maximum(k_float32[s:e].abs().amax(dim=(0, 2)), teps) for s, e in zip(cu_seqlens_k[:-1], cu_seqlens_k[1:])])
                v_scale = torch.stack([torch.maximum(v_float32[s:e].abs().amax(dim=(0, 2)), teps) for s, e in zip(cu_seqlens_k[:-1], cu_seqlens_k[1:])])

            else:
                if layout == "bhsd":
                    seqlen_loc = 2
                    dim_loc = 3
                elif layout == "bshd":
                    seqlen_loc = 1
                    dim_loc = 3

                # Compute max for each batch-head pair.
                # Compute max across seqlen and dim.
                q_scale = torch.maximum(q_float32.abs().amax(dim=(seqlen_loc, dim_loc)), teps)
                k_scale = torch.maximum(k_float32.abs().amax(dim=(seqlen_loc, dim_loc)), teps)
                v_scale = torch.maximum(v_float32.abs().amax(dim=(seqlen_loc, dim_loc)), teps)

        # Divide max tensors by respective data type max.
        q_scale = FP8_MAX[q.dtype] / q_scale
        k_scale = FP8_MAX[k.dtype] / k_scale
        v_scale = FP8_MAX[v.dtype] / v_scale

        # Compute p_scale.
        p_scale = torch.full((batch, head_q), FP8_MAX[q.dtype], dtype=torch.float32, device="cuda")

    return q_scale, k_scale, v_scale, p_scale


def scale_fp8(
    x: Tensor, x_scale: Tensor, layout: str,
    cu_seqlens: Optional[Tensor] = None,
) -> Tensor:
    """
    Scales an FP8 tensor using a specified scaling factor.

    This function scales the input tensor `x` by dividing it by the scaling factor `x_scale`,
    while considering the specified tensor layout.

    If the input tensor `x` is not an FP8 tensor, the function returns it unchanged.
    For FP8 tensors, the result is clamped to the representable range of the FP8 data type.

    Args:
        x (Tensor): The input tensor to be scaled.
        x_scale (Tensor): The scaling factor tensor, broadcasted as needed based on `layout`.
        layout (str): The data layout of the tensor. Must be one of `"bhsd"`, `"bshd"`, or `"thd"`.
        cu_seqlens (Optional[Tensor], optional): Cumulative sequence lengths for variable-length
            sequences. Required when `layout` is `"thd"`.

    Returns:
        Tensor: The scaled tensor.
    """
    assert layout in ["bhsd", "bshd", "thd"], "Unknow layout."
    assert (layout == "thd" and cu_seqlens is not None) or layout != "thd", "cu_seqlens is required for varlen layout."
    if not check_is_fp8(x):
        return x
    # Fraction numerator is float32 version of x.
    n = x.detach().to(torch.float32)
    # Fraction denominator is the broadcasted scaled factor.
    x_scale = x_scale.detach()
    if layout == "bhsd":
        x_scaled = n * x_scale[:, :, None, None]
    elif layout == "bshd":
        x_scaled = n * x_scale[:, None, :, None]
    elif layout == "thd":
        x_scaled = torch.cat([
            n[s:e] * x_scale[z, :].unsqueeze(0).unsqueeze(-1)
            for z, (s, e) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:]))
        ], dim=0)
    # Clamp and convert back to float8.
    return torch.clamp(x_scaled, min=torch.finfo(x.dtype).min, max=torch.finfo(x.dtype).max).to(
        x.dtype).requires_grad_(x.requires_grad)


class Fp8MetaData:
    """
    Manages FP8 scaling metadata and scaled tensors for query, key, and value.

    Attributes:
        scale_per_head (bool): Indicates if scaling is applied per batch / head.
        q_scale (Tensor): Scaling factor for the query tensor.
        k_scale (Tensor): Scaling factor for the key tensor.
        v_scale (Tensor): Scaling factor for the value tensor.
        p_scale (Tensor): Scaling factor for intermediate FA computation
            `p = softmax(matmul(q, transpose(k)))`.
        q_inv_scale (Tensor): Inverse of `q_scale`.
        k_inv_scale (Tensor): Inverse of `k_scale`.
        v_inv_scale (Tensor): Inverse of `v_scale`.
        p_inv_scale (Tensor): Inverse of `p_scale`.
        q_scaled (Tensor): Scaled query tensor.
        k_scaled (Tensor): Scaled key tensor.
        v_scaled (Tensor): Scaled value tensor.

    Methods:
        __init__: Initializes scaling metadata and applies scaling to tensors.
    """

    scale_per_head: bool
    q_scale: Tensor
    k_scale: Tensor
    v_scale: Tensor
    p_scale: Tensor
    q_inv_scale: Tensor
    k_inv_scale: Tensor
    v_inv_scale: Tensor
    p_inv_scale: Tensor
    q_scaled: Tensor
    k_scaled: Tensor
    v_scaled: Tensor

    def __init__(
        self,
        q: Tensor, k: Tensor, v: Tensor, layout: str, metadata: MetaData,
        scale_per_head: bool = DEFAULT_SCALE_PER_HEAD,
    ) -> None:
        self.scale_per_head = scale_per_head
        self.q_scale, self.k_scale, self.v_scale, self.p_scale = create_fp8_scale_tensors(
            q, k, v, layout,
            cu_seqlens_q=metadata.cu_seqlens_q, cu_seqlens_k=metadata.cu_seqlens_k,
            max_seqlen_q=metadata.max_seqlens_q, max_seqlen_k=metadata.max_seqlens_k,
            scale_per_head=scale_per_head,
        )
        self.q_inv_scale = 1 / self.q_scale
        self.k_inv_scale = 1 / self.k_scale
        self.v_inv_scale = 1 / self.v_scale
        self.p_inv_scale = 1 / self.p_scale
        self.q_scaled = scale_fp8(q, self.q_scale, layout, cu_seqlens=metadata.cu_seqlens_q)
        self.k_scaled = scale_fp8(k, self.k_scale, layout, cu_seqlens=metadata.cu_seqlens_k)
        self.v_scaled = scale_fp8(v, self.v_scale, layout, cu_seqlens=metadata.cu_seqlens_k)
