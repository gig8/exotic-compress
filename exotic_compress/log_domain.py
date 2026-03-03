"""
Log-domain inference: restructure arithmetic without touching weights.

Core idea:
  Standard: y_j = Σ_i (w_ij * x_i)        — multiply then sum
  Log-domain: log(w*x) = log(w) + log(x)   — add instead of multiply

The expensive operation in neural nets is multiply-accumulate (MAC).
In log domain, the multiply becomes an add (cheap), but the accumulate
(summing in log domain) requires the log-sum-exp operation:

  log(a + b) = log(a) + log(1 + exp(log(b) - log(a)))
             = max(log(a), log(b)) + log(1 + exp(-|log(a) - log(b)|))

This is called the "Jacobian logarithm" or "log-add" operation.
It requires a lookup table or small approximation, but is still cheaper
than a full multiplier in hardware.

For GPU benchmarking: log-domain is NOT faster on GPU (GPUs are optimized
for multiply-add). The value is for:
1. FPGA/ASIC where adders are 3-5x cheaper than multipliers
2. Optical hardware where phase = log-amplitude naturally
3. Establishing that the math works (lossless verification)

This module demonstrates the LOSSLESS property: converting to log domain
and back produces bit-identical results (within floating-point precision).
"""

import json
import time
from pathlib import Path

import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from .verify import verify_lossless, print_verification


def to_log_domain(W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert a weight matrix to log-domain representation.

    Returns:
        sign: +1 or -1 for each element
        log_abs: log2(|w|) for each element (use -inf for zero)
    """
    sign = torch.sign(W)
    # Handle zeros: sign=0 stays 0, log_abs gets -inf
    abs_W = torch.abs(W)
    log_abs = torch.where(
        abs_W > 0,
        torch.log2(abs_W),
        torch.full_like(abs_W, float('-inf'))
    )
    return sign, log_abs


def from_log_domain(sign: torch.Tensor, log_abs: torch.Tensor) -> torch.Tensor:
    """Convert back from log-domain to standard representation."""
    abs_val = torch.where(
        log_abs > float('-inf'),
        torch.pow(2.0, log_abs),
        torch.zeros_like(log_abs)
    )
    return sign * abs_val


def log_domain_matmul_naive(
    x: torch.Tensor,
    W_sign: torch.Tensor,
    W_log_abs: torch.Tensor,
) -> torch.Tensor:
    """Matrix multiply in log domain (naive, for correctness verification).

    y_j = Σ_i (w_ij * x_i)

    In log domain:
    - Each product w_ij * x_i has sign = sign_w * sign_x, log = log_w + log_x
    - Summing signed log-domain values requires grouping by sign and using log-sum-exp

    This is a NAIVE implementation for correctness. Not optimized for speed.
    """
    # For correctness verification, we reconstruct and use standard matmul
    W = from_log_domain(W_sign, W_log_abs)
    return x @ W


def log_domain_matmul(
    x: torch.Tensor,
    W_sign: torch.Tensor,
    W_log_abs: torch.Tensor,
) -> torch.Tensor:
    """Matrix multiply using log-domain arithmetic.

    Computes y = x @ W where W is stored as (sign, log2|W|).

    The key operation: for each output element y_j:
    1. Compute log-products: lp_i = log2|x_i| + log2|W_ij| and sign_i = sign(x_i) * sign(W_ij)
    2. Separate into positive and negative groups by sign
    3. Use log-sum-exp within each group
    4. Subtract: y_j = exp2(lse_pos) - exp2(lse_neg)

    This is mathematically exact but numerically different from dense matmul
    due to floating-point ordering differences.
    """
    # x: (..., m), W: (m, n) stored as sign (m,n) and log_abs (m,n)
    orig_shape = x.shape[:-1]
    m = x.shape[-1]
    n = W_sign.shape[1]

    x_flat = x.reshape(-1, m)  # (batch, m)
    batch = x_flat.shape[0]

    # Log-domain products
    x_sign = torch.sign(x_flat)           # (batch, m)
    x_abs = torch.abs(x_flat)
    x_log = torch.where(
        x_abs > 0,
        torch.log2(x_abs),
        torch.full_like(x_abs, float('-inf'))
    )  # (batch, m)

    # For each (batch, j): log_products[i] = x_log[i] + W_log_abs[i, j]
    #                       prod_sign[i] = x_sign[i] * W_sign[i, j]
    # x_log: (batch, m, 1) + W_log_abs: (1, m, n) -> (batch, m, n)
    log_products = x_log.unsqueeze(-1) + W_log_abs.unsqueeze(0)  # (batch, m, n)
    prod_sign = x_sign.unsqueeze(-1) * W_sign.unsqueeze(0)       # (batch, m, n)

    # Separate positive and negative contributions
    pos_mask = prod_sign > 0  # (batch, m, n)
    neg_mask = prod_sign < 0

    # Log-sum-exp for positive terms: for each (batch, j), sum over m
    # We need to handle -inf properly
    log_pos = torch.where(pos_mask, log_products, torch.full_like(log_products, float('-inf')))
    log_neg = torch.where(neg_mask, log_products, torch.full_like(log_products, float('-inf')))

    # torch.logsumexp over the m dimension, converted from log2 to ln and back
    ln2 = np.log(2.0)
    # log2(sum(2^a_i)) = log2(sum(exp(a_i * ln2))) = logsumexp(a_i * ln2) / ln2
    pos_sum_log2 = torch.logsumexp(log_pos * ln2, dim=1) / ln2  # (batch, n)
    neg_sum_log2 = torch.logsumexp(log_neg * ln2, dim=1) / ln2  # (batch, n)

    # Convert back: y = 2^pos_sum_log2 - 2^neg_sum_log2
    pos_sum = torch.pow(2.0, pos_sum_log2)
    neg_sum = torch.pow(2.0, neg_sum_log2)

    # Handle infinities (when all terms in a group are zero)
    pos_sum = torch.where(torch.isinf(pos_sum_log2) & (pos_sum_log2 < 0), torch.zeros_like(pos_sum), pos_sum)
    neg_sum = torch.where(torch.isinf(neg_sum_log2) & (neg_sum_log2 < 0), torch.zeros_like(neg_sum), neg_sum)

    y = pos_sum - neg_sum  # (batch, n)

    return y.reshape(*orig_shape, n)


def verify_log_domain_roundtrip(model: GPT2LMHeadModel) -> dict:
    """Verify that weight -> log domain -> reconstruct is lossless."""
    results = {}
    max_error = 0.0
    total_exact = 0
    total_params = 0

    for name, param in model.named_parameters():
        W = param.detach().cpu()
        sign, log_abs = to_log_domain(W)
        W_reconstructed = from_log_domain(sign, log_abs)

        diff = torch.abs(W - W_reconstructed)
        max_diff = float(diff.max())
        exact_count = int((diff == 0).sum())

        results[name] = {
            "max_diff": max_diff,
            "exact_fraction": exact_count / W.numel(),
        }
        max_error = max(max_error, max_diff)
        total_exact += exact_count
        total_params += W.numel()

    results["__summary__"] = {
        "max_error_any_param": max_error,
        "overall_exact_fraction": total_exact / total_params,
        "lossless": max_error == 0.0,
    }
    return results


def benchmark_log_matmul(device: str = "cuda", n_iters: int = 500):
    """Benchmark standard vs log-domain matmul speed."""
    shapes = [
        ("768x2304 (c_attn)", 768, 2304),
        ("768x768 (c_proj)", 768, 768),
        ("768x3072 (c_fc)", 768, 3072),
        ("3072x768 (c_proj_mlp)", 3072, 768),
    ]

    print(f"{'Layer':<30} {'Dense(ms)':>10} {'Log-domain(ms)':>15} {'Ratio':>8} {'MaxDiff':>12}")
    print("-" * 80)

    for label, m, n in shapes:
        W = torch.randn(m, n, device=device)
        x = torch.randn(1, 128, m, device=device)

        W_sign, W_log_abs = to_log_domain(W)
        W_sign = W_sign.to(device)
        W_log_abs = W_log_abs.to(device)

        # Warmup
        for _ in range(20):
            _ = x @ W
            _ = log_domain_matmul(x, W_sign, W_log_abs)
        torch.cuda.synchronize() if device == "cuda" else None

        # Dense
        start = time.perf_counter()
        for _ in range(n_iters):
            y_dense = x @ W
        torch.cuda.synchronize() if device == "cuda" else None
        dense_ms = (time.perf_counter() - start) / n_iters * 1000

        # Log-domain
        start = time.perf_counter()
        for _ in range(n_iters):
            y_log = log_domain_matmul(x, W_sign, W_log_abs)
        torch.cuda.synchronize() if device == "cuda" else None
        log_ms = (time.perf_counter() - start) / n_iters * 1000

        max_diff = float((y_dense - y_log).abs().max().cpu())

        print(f"{label:<30} {dense_ms:>10.4f} {log_ms:>15.4f} {log_ms/dense_ms:>7.1f}x {max_diff:>12.4f}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Log-domain inference for GPT-2")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bench", action="store_true", help="Benchmark speed")
    args = parser.parse_args()

    output_dir = Path("artifacts/log_domain")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Log-Domain Inference ===\n")

    # 1. Verify roundtrip
    print("--- Roundtrip Verification (weight → log → reconstruct) ---\n")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    rt_results = verify_log_domain_roundtrip(model)
    summary = rt_results.pop("__summary__")
    print(f"  Max error across all params: {summary['max_error_any_param']:.2e}")
    print(f"  Exact fraction:              {summary['overall_exact_fraction']:.6f}")
    print(f"  Lossless roundtrip:          {'YES' if summary['lossless'] else 'NO'}")

    # 2. Storage analysis
    print("\n--- Storage Analysis ---\n")
    total_params = sum(p.numel() for p in model.parameters())
    # Log domain: sign (1 bit) + log2|w| (needs ~16 bits for FP32 range)
    # vs standard FP32 (32 bits)
    # Effective: 17 bits vs 32 bits = 0.53x compression in theory
    print(f"  Total parameters:     {total_params:,}")
    print(f"  FP32 storage:         {total_params * 4 / 1024**2:.1f} MB")
    print(f"  Log domain (1+16b):   {total_params * 17 / 8 / 1024**2:.1f} MB (theoretical)")
    print(f"  Log domain (1+8b):    {total_params * 9 / 8 / 1024**2:.1f} MB (8-bit log)")
    print(f"  Theoretical ratio:    {17/32:.3f} (16-bit log) / {9/32:.3f} (8-bit log)")

    # 3. Accuracy of log-domain matmul (not roundtrip, actual matmul)
    print("\n--- Log-Domain Matmul Accuracy ---\n")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    # Test on a single layer
    W = dict(model.named_parameters())["transformer.h.0.mlp.c_fc.weight"]
    W_data = W.detach().to(args.device)
    W_sign, W_log_abs = to_log_domain(W_data)

    x = torch.randn(1, 10, W_data.shape[0], device=args.device)
    y_dense = x @ W_data
    y_log = log_domain_matmul(x, W_sign, W_log_abs)

    abs_diff = (y_dense - y_log).abs()
    rel_diff = abs_diff / (y_dense.abs() + 1e-10)
    print(f"  Layer: transformer.h.0.mlp.c_fc.weight ({W_data.shape[0]}×{W_data.shape[1]})")
    print(f"  Max absolute diff:    {float(abs_diff.max()):.6f}")
    print(f"  Mean absolute diff:   {float(abs_diff.mean()):.6f}")
    print(f"  Max relative diff:    {float(rel_diff.max()):.6f}")
    print(f"  Mean relative diff:   {float(rel_diff.mean()):.6f}")

    # 4. Speed benchmark
    if args.bench:
        print("\n--- Speed Benchmark ---\n")
        benchmark_log_matmul(device=args.device)

    # Save results
    with open(output_dir / "log_domain_results.json", "w") as f:
        json.dump({
            "roundtrip": summary,
            "storage": {
                "total_params": total_params,
                "fp32_mb": total_params * 4 / 1024**2,
                "log16_mb": total_params * 17 / 8 / 1024**2,
                "log8_mb": total_params * 9 / 8 / 1024**2,
            },
        }, f, indent=2)


if __name__ == "__main__":
    main()
