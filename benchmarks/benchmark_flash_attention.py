# Install the newest triton version with
# pip install "git+https://github.com/openai/triton.git#egg=triton&subdirectory=python"
import pickle
import math
import argparse
import sys
import os
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from flash_attn.utils.benchmark import benchmark_all, benchmark_forward, benchmark_backward
from flash_attn.utils.benchmark import benchmark_fwd_bwd, benchmark_combined

from flash_attn import flash_attn_qkvpacked_func
import multiprocessing as mp

try:
    from triton.ops.flash_attention import attention as attention_triton
except ImportError:
    attention_triton = None

try:
    import xformers.ops as xops
except ImportError:
    xops = None

try:
    import pandas as pd
except ImportError:
    pd = None

def flops(batch, seqlen, headdim, nheads, causal, mode="fwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def efficiency(flop, time):
    return (flop / time / 10**12) if not math.isnan(time) else 0.0

def attention_pytorch(qkv, dropout_p=0.0, causal=True):
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, head_dim)
        dropout_p: float
    Output:
        output: (batch_size, seqlen, nheads, head_dim)
    """
    batch_size, seqlen, _, nheads, d = qkv.shape
    q, k, v = qkv.unbind(dim=2)
    q = rearrange(q, 'b t h d -> (b h) t d')
    k = rearrange(k, 'b s h d -> (b h) d s')
    softmax_scale = 1.0 / math.sqrt(d)
    # Preallocate attn_weights for `baddbmm`
    scores = torch.empty(batch_size * nheads, seqlen, seqlen, dtype=qkv.dtype, device=qkv.device)
    scores = rearrange(torch.baddbmm(scores, q, k, beta=0, alpha=softmax_scale),
                       '(b h) t s -> b h t s', h=nheads)
    if causal:
        # "triu_tril_cuda_template" not implemented for 'BFloat16'
        # So we have to construct the mask in float
        causal_mask = torch.triu(torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1)
        # TD [2022-09-30]: Adding is faster than masked_fill_ (idk why, just better kernel I guess)
        scores = scores + causal_mask.to(dtype=scores.dtype)
    attention = torch.softmax(scores, dim=-1)
    attention_drop = F.dropout(attention, dropout_p)
    output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    return output.to(dtype=qkv.dtype)


def time_fwd_bwd(func, *args, **kwargs):
    time_f, time_b = benchmark_fwd_bwd(func, *args, **kwargs)
    return time_f[1].mean, time_b[1].mean

def time_fwd(func, *args, **kwargs):
    time_f = benchmark_forward(func, *args, **kwargs)
    return time_f[1].mean

def time_bwd(func, *args, **kwargs):
    time_b = benchmark_backward(func, *args, **kwargs)
    return time_b[1].mean
def run_single_benchmark(args_dict, config, methods, return_dict):
    # Convert args_dict back to an object
    class Args:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    args = Args(**args_dict)

    os.environ['HIP_VISIBLE_DEVICES'] = str(args.gpu)
    torch.cuda.empty_cache()

    causal, headdim, batch_size, seqlen, nheads = config

    print(f"Warming up: {config}")
    device = torch.device(args.device)
    qkv = torch.randn(batch_size, seqlen, 3, nheads, headdim, device=device, dtype=getattr(torch, args.dtype),
                      requires_grad=True)
    for _ in range(5):  # 5 warm-up iterations
        for method in methods:
            if method == "Flash2":
                flash_attn_qkvpacked_func(qkv, args.dropout_p, causal=causal)
            elif method == "Pytorch":
                attention_pytorch(qkv, args.dropout_p, causal=causal)
            elif method == "Triton" and attention_triton is not None:
                q, k, v = [x.reshape(batch_size, nheads, seqlen, headdim) for x in qkv.unbind(dim=2)]
                attention_triton(q, k, v, causal, headdim**(-0.5), False)
            elif method in ["xformers.c", "xformers.f"] and xops is not None:
                q, k, v = [x.reshape(batch_size, seqlen, nheads, headdim) for x in qkv.unbind(dim=2)]
                op = (xops.fmha.cutlass.FwOp, xops.fmha.cutlass.BwOp) if method == "xformers.c" else (xops.fmha.flash.FwOp, xops.fmha.flash.BwOp)
                xops.memory_efficient_attention(q, k, v, attn_bias=xops.LowerTriangularMask() if causal else None, op=op)

    results = {}
    for method in methods:
        if method == "Flash2":
            if args.mode in ["fwd_bwd", "fwd"]:
                time_f = time_fwd(flash_attn_qkvpacked_func, qkv, args.dropout_p, causal=causal, repeats=args.repeats, verbose=False)
                results[f'{method} fwd (TFLOPs/s)'] = efficiency(flops(batch_size, seqlen, headdim, nheads, causal, mode="fwd"), time_f)
            if args.mode in ["fwd_bwd", "bwd"]:
                time_b = time_bwd(flash_attn_qkvpacked_func, qkv, args.dropout_p, causal=causal, repeats=args.repeats, verbose=False)
                results[f'{method} bwd (TFLOPs/s)'] = efficiency(flops(batch_size, seqlen, headdim, nheads, causal, mode="bwd"), time_b)
        elif method == "Pytorch":
            if args.mode in ["fwd_bwd", "fwd"]:
                time_f = time_fwd(attention_pytorch, qkv, args.dropout_p, causal=causal, repeats=args.repeats, verbose=False)
                results[f'{method} fwd (TFLOPs/s)'] = efficiency(flops(batch_size, seqlen, headdim, nheads, causal, mode="fwd"), time_f)
            if args.mode in ["fwd_bwd", "bwd"]:
                time_b = time_bwd(attention_pytorch, qkv, args.dropout_p, causal=causal, repeats=args.repeats, verbose=False)
                results[f'{method} bwd (TFLOPs/s)'] = efficiency(flops(batch_size, seqlen, headdim, nheads, causal, mode="bwd"), time_b)
        elif method == "Triton" and attention_triton is not None:
            q, k, v = [x.reshape(batch_size, nheads, seqlen, headdim) for x in qkv.unbind(dim=2)]
            if args.mode in ["fwd_bwd", "fwd"]:
                time_f = time_fwd(attention_triton, q, k, v, causal, headdim**(-0.5), False, repeats=args.repeats, verbose=False)
                results[f'{method} fwd (TFLOPs/s)'] = efficiency(flops(batch_size, seqlen, headdim, nheads, causal, mode="fwd"), time_f)
            if args.mode in ["fwd_bwd", "bwd"]:
                time_b = time_bwd(attention_triton, q, k, v, causal, headdim**(-0.5), False, repeats=args.repeats, verbose=False)
                results[f'{method} bwd (TFLOPs/s)'] = efficiency(flops(batch_size, seqlen, headdim, nheads, causal, mode="bwd"), time_b)
        elif method in ["xformers.c", "xformers.f"] and xops is not None:
            q, k, v = [x.reshape(batch_size, seqlen, nheads, headdim) for x in qkv.unbind(dim=2)]
            op = (xops.fmha.cutlass.FwOp, xops.fmha.cutlass.BwOp) if method == "xformers.c" else (xops.fmha.flash.FwOp, xops.fmha.flash.BwOp)
            if args.mode in ["fwd_bwd", "fwd"]:
                time_f = time_fwd(xops.memory_efficient_attention, q, k, v, attn_bias=xops.LowerTriangularMask() if causal else None, op=op)
                results[f'{method} fwd (TFLOPs/s)'] = efficiency(flops(batch_size, seqlen, headdim, nheads, causal, mode="fwd"), time_f)
            if args.mode in ["fwd_bwd", "bwd"]:
                time_b = time_bwd(xops.memory_efficient_attention, q, k, v, attn_bias=xops.LowerTriangularMask() if causal else None, op=op)
                results[f'{method} bwd (TFLOPs/s)'] = efficiency(flops(batch_size, seqlen, headdim, nheads, causal, mode="bwd"), time_b)

    return_dict[config] = results

def run_benchmark(args):
    print("args:", args)
    device = args.device
    dtype = getattr(torch, args.dtype)
    
    methods = args.methods
    if attention_triton is None and "Triton" in methods:
        methods.remove("Triton")
    if xops is None and ("xformers.c" in methods or "xformers.f" in methods):
        methods = [m for m in methods if not m.startswith("xformers")]

    configs = [(causal, headdim, batch_size, seqlen, nheads) 
               for causal in args.causal
               for headdim, nheads in zip(args.headdim, args.nheads)
               for batch_size, seqlen in zip(args.batch_size, args.seqlen)]

    all_results = []
    for pass_num in range(args.passes):
        print(f"Starting pass {pass_num + 1}/{args.passes}")
        random.shuffle(configs)
        for config in configs:
            manager = mp.Manager()
            return_dict = manager.dict()
            args_dict = vars(args)  # Convert args to a dictionary
            p = mp.Process(target=run_single_benchmark, args=(args_dict, config, methods, return_dict))
            p.start()
            p.join()
            
            if config not in return_dict:
                print(f"Error: No results returned for config {config}")
                continue

            results = return_dict[config]
            causal, headdim, batch_size, seqlen, nheads = config
            row = {'Pass': pass_num + 1, 'Batch Size': batch_size, 'Seq Len': seqlen, "Num of Heads": nheads, 'Dim of Head': headdim, 'Causal': causal}
            row.update(results)
            all_results.append(row)
            
            print(f"Completed: {config}")
            print(results)
            
            time.sleep(args.cooldown)

    df = pd.DataFrame(all_results)
    print(df.to_string(index=False))
    csv_filename = f'benchmark_results_{args.mode}.csv'
    df.to_csv(csv_filename, index=False)
    print(f"\nResults saved to {csv_filename}")

    print("\nAveraged Results:")
    df_avg = df.drop(columns=["Pass"]).groupby(['Batch Size', 'Seq Len', "Num of Heads", 'Dim of Head', 'Causal']).mean().reset_index()
    print(df_avg.to_string(index=False))
    csv_filename = f'benchmark_avg_results_{args.mode}.csv'
    df_avg.to_csv(csv_filename, index=False)
    print(f"\nAvg Results saved to {csv_filename}")
    

    return df

if __name__ == "__main__":
    mp.set_start_method('spawn')  # Set the start method to 'spawn'
    
    parser = argparse.ArgumentParser(description="Run attention benchmark with custom configurations.")
    parser.add_argument("--methods", nargs="+", default=["Flash2", "Pytorch"],
                        help="Attention methods to benchmark")
    parser.add_argument("--mode", choices=["fwd", "bwd", "fwd_bwd"], default="fwd",
                        help="Benchmark mode: forward, backward, or both")
    parser.add_argument("--device", default="cuda", help="Device to run the benchmark on")
    parser.add_argument("--dtype", default="float16", choices=["float16", "float32", "bfloat16"],
                        help="Data type for the tensors")
    parser.add_argument("--causal", nargs="+", type=lambda x: x.lower() == 'true', default=[False, True],
                        help="Whether to use causal attention")
    parser.add_argument("--dim", type=int, default=2048, help="Total dimension of the model")
    parser.add_argument("--nheads", nargs="+", type=int, help="Number of attention heads")
    parser.add_argument("--headdim", nargs="+", type=int, default=[64, 128], help="Dimension(s) of each attention head")
    parser.add_argument("--batch_size", nargs="+", type=int, default=[32, 16, 8, 4, 2, 1],
                        help="Batch sizes to benchmark")
    parser.add_argument("--seqlen", nargs="+", type=int, default=[512, 1024, 2048, 4096, 8192, 16384],
                        help="Sequence lengths to benchmark")
    parser.add_argument("--dropout_p", type=float, default=0.0, help="Dropout probability")
    parser.add_argument("--repeats", type=int, default=30, help="Number of repetitions for each benchmark")
    parser.add_argument("--passes", type=int, default=3, help="Number of passes through all configurations")
    parser.add_argument("--cooldown", type=float, default=2.0, help="Cool-down time between configurations (seconds)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device to use")

    args = parser.parse_args()

    # Check if all three parameters are provided
    if args.dim != parser.get_default('dim') and args.nheads is not None and args.headdim != parser.get_default('headdim'):
        print("Error: You have provided values for dim, nheads, and headdim. Please provide at most two of these parameters.")
        sys.exit(1)

    # Validate and adjust arguments
    if args.nheads is not None:
        if args.dim != parser.get_default('dim'):
            print(f"Running benchmark with dim={args.dim}, nheads={args.nheads}")
            args.headdim = [args.dim // nh for nh in args.nheads]
            print(f"Calculated headdim: {args.headdim}")
            # Check if any calculated headdim is 0
            if any(hd == 0 for hd in args.headdim):
                print("Error: Some number of heads are larger than the total dimension. Please adjust your input.")
                sys.exit(1)
        else:
            print(f"Running benchmark with nheads={args.nheads}, headdim={args.headdim}")
            args.dim = max(nh * hd for nh, hd in zip(args.nheads, args.headdim))
    elif args.dim != parser.get_default('dim'):
        print(f"Running benchmark with dim={args.dim}, headdim={args.headdim}")
        args.nheads = [args.dim // hd for hd in args.headdim]
        print(f"Calculated nheads: {args.nheads}")
        # Check if any calculated nheads is 0
        if any(nh == 0 for nh in args.nheads):
            print("Error: Some head dimensions are larger than the total dimension. Please adjust your input.")
            sys.exit(1)
    else:
        print(f"Running benchmark with default dim={args.dim}, headdim={args.headdim}")
        args.nheads = [args.dim // hd for hd in args.headdim]
        print(f"Calculated nheads: {args.nheads}")

    # Ensure nheads and headdim have the same length
    if len(args.nheads) != len(args.headdim):
        print("Error: The number of values for nheads and headdim must be the same.")
        sys.exit(1)

    results = run_benchmark(args)