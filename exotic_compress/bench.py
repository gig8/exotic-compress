"""
Benchmark: does restructured computation run FASTER, regardless of parameter count?

Key insight: even if SVD has more total parameters, the factored form
  y = (x @ Vt.T) * S @ U.T
may be faster than
  y = x @ W
because:
1. Two smaller matmuls can have better cache behavior than one large one
2. At reduced rank, FLOPs drop: r*(m+n) vs m*n
3. Memory bandwidth (not FLOPs) is often the bottleneck at batch=1

This script benchmarks dense vs factored forward passes at various ranks.
"""

import time
from dataclasses import dataclass

import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer


@dataclass
class BenchResult:
    label: str
    rank: int
    flops: int
    time_ms: float
    params: int
    max_logit_diff: float


def benchmark_dense(W: torch.Tensor, x: torch.Tensor, n_iters: int = 500) -> float:
    """Benchmark dense matmul: y = x @ W"""
    # Warmup
    for _ in range(50):
        _ = x @ W
    torch.cuda.synchronize() if x.is_cuda else None

    start = time.perf_counter()
    for _ in range(n_iters):
        y = x @ W
    torch.cuda.synchronize() if x.is_cuda else None
    elapsed = (time.perf_counter() - start) / n_iters
    return elapsed, y


def benchmark_svd_factored(
    U: torch.Tensor, S: torch.Tensor, Vt: torch.Tensor,
    x: torch.Tensor, n_iters: int = 500
) -> float:
    """Benchmark factored matmul for W = U @ diag(S) @ Vt.

    Dense: y = x @ W  where x is (..., m), W is (m, n), y is (..., n)
    SVD:   W = U @ diag(S) @ Vt, so y = x @ U @ diag(S) @ Vt
           = ((x @ U) * S) @ Vt
    where U is (m, r), S is (r,), Vt is (r, n)
    """
    # Warmup
    for _ in range(50):
        _ = ((x @ U) * S.unsqueeze(0)) @ Vt
    torch.cuda.synchronize() if x.is_cuda else None

    start = time.perf_counter()
    for _ in range(n_iters):
        y = ((x @ U) * S.unsqueeze(0)) @ Vt
    torch.cuda.synchronize() if x.is_cuda else None
    elapsed = (time.perf_counter() - start) / n_iters
    return elapsed, y


def benchmark_layer_at_ranks(
    name: str, W: torch.Tensor, x: torch.Tensor, device: str,
    ranks: list[int] = None, n_iters: int = 500
) -> list[BenchResult]:
    """Benchmark a single weight matrix at various SVD ranks."""
    m, n = W.shape
    full_rank = min(m, n)

    if ranks is None:
        # Test: full, 75%, 50%, 25%, 10%, 5%
        ranks = sorted(set([
            full_rank,
            int(full_rank * 0.75),
            int(full_rank * 0.50),
            int(full_rank * 0.25),
            int(full_rank * 0.10),
            int(full_rank * 0.05),
        ]), reverse=True)

    W_dev = W.to(device)
    x_dev = x.to(device)

    # Dense baseline: y = x @ W, x is (..., m), W is (m, n)
    dense_time, y_dense = benchmark_dense(W_dev, x_dev, n_iters)
    batch, seq = x.shape[0], x.shape[1]
    dense_flops = batch * seq * m * n  # one matmul

    results = [BenchResult(
        label=f"{name} (dense)",
        rank=full_rank,
        flops=dense_flops,
        time_ms=dense_time * 1000,
        params=m * n,
        max_logit_diff=0.0,
    )]

    # SVD: W = U @ diag(S) @ Vt, where U is (m,r), S is (r,), Vt is (r,n)
    # Factored: y = ((x @ U) * S) @ Vt
    U_full, S_full, Vt_full = torch.linalg.svd(W.float(), full_matrices=False)

    for r in ranks:
        U_r = U_full[:, :r].to(device)   # (m, r)
        S_r = S_full[:r].to(device)       # (r,)
        Vt_r = Vt_full[:r, :].to(device)  # (r, n)

        svd_time, y_svd = benchmark_svd_factored(U_r, S_r, Vt_r, x_dev, n_iters)

        # FLOPs: x @ U is (batch*seq*m*r), scale is free, @ Vt is (batch*seq*r*n)
        svd_flops = batch * seq * (m * r + r * n)

        # Accuracy check
        max_diff = float((y_svd.cpu() - y_dense.cpu()).abs().max())

        svd_params = m * r + r + r * n

        results.append(BenchResult(
            label=f"{name} (SVD r={r})",
            rank=r,
            flops=svd_flops,
            time_ms=svd_time * 1000,
            params=svd_params,
            max_logit_diff=max_diff,
        ))

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark dense vs factored computation")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--n-iters", type=int, default=500)
    parser.add_argument("--layers", nargs="*", default=None,
                        help="Specific layer names to benchmark (default: representative set)")
    args = parser.parse_args()

    print(f"=== Inference Speed Benchmark ===")
    print(f"Device: {args.device}")
    print(f"Batch: {args.batch_size}, Seq: {args.seq_len}, Iters: {args.n_iters}\n")

    # Load model
    print("Loading GPT-2 Small...")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    # Representative layers to benchmark
    target_layers = args.layers or [
        "transformer.h.0.attn.c_attn.weight",    # 768 × 2304 (QKV projection)
        "transformer.h.0.attn.c_proj.weight",     # 768 × 768  (output projection)
        "transformer.h.0.mlp.c_fc.weight",        # 768 × 3072 (FFN up)
        "transformer.h.0.mlp.c_proj.weight",      # 3072 × 768 (FFN down)
    ]

    all_results = []

    for name, param in model.named_parameters():
        if name not in target_layers:
            continue

        W = param.detach().cpu()
        if W.ndim != 2:
            continue

        m, n = W.shape
        # GPT-2 uses Conv1D: weight is (input_dim, output_dim), op is x @ W
        # So input last dim = m (input_dim)
        x = torch.randn(args.batch_size, args.seq_len, m)

        print(f"\n--- {name} ({m}×{n}) ---")
        results = benchmark_layer_at_ranks(
            name, W, x, args.device, n_iters=args.n_iters
        )
        all_results.extend(results)

        # Print table
        print(f"{'Config':<40} {'Rank':>6} {'FLOPs':>12} {'Time(ms)':>10} {'Speedup':>8} {'MaxDiff':>10} {'Params':>10}")
        print("-" * 100)
        base_time = results[0].time_ms
        for r in results:
            speedup = base_time / r.time_ms if r.time_ms > 0 else 0
            print(
                f"{r.label:<40} {r.rank:>6} {r.flops:>12,} {r.time_ms:>10.4f} "
                f"{speedup:>7.2f}x {r.max_logit_diff:>10.2e} {r.params:>10,}"
            )

    # Summary
    print("\n\n=== SUMMARY ===\n")
    print("Key question: At what rank does factored SVD become FASTER than dense?\n")

    for name in target_layers:
        layer_results = [r for r in all_results if name in r.label]
        if not layer_results:
            continue
        dense = layer_results[0]
        faster = [r for r in layer_results[1:] if r.time_ms < dense.time_ms]
        if faster:
            best = min(faster, key=lambda r: r.time_ms)
            print(f"{name}:")
            print(f"  Dense: {dense.time_ms:.4f}ms")
            print(f"  Best:  {best.time_ms:.4f}ms (rank={best.rank}, {dense.time_ms/best.time_ms:.2f}x faster, max_diff={best.max_logit_diff:.2e})")
        else:
            print(f"{name}: Dense is fastest (no speedup from SVD factoring)")


if __name__ == "__main__":
    main()
