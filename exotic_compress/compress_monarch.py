"""
Monarch matrix factorization for GPT-2 weight compression and speedup.

A Monarch matrix M factors a dense (n x n) matrix as:

    M = P_L @ L @ P_R @ R

where:
    - L is block-diagonal with 'b' blocks of size (n/b x n/b)
    - R is block-diagonal with 'b' blocks of size (n/b x n/b)
    - P_L, P_R are fixed permutation matrices (reshape + transpose)

For rectangular (m x n) matrices, the generalized form uses:
    - R: block-diagonal with sqrt(n) blocks
    - L: block-diagonal with sqrt(m) blocks
    - Permutations reshape between block structures

Storage: O(n^{3/2}) instead of O(n^2)
FLOPs:   O(n^{3/2}) instead of O(n^2)

The projection of a dense matrix W onto the Monarch manifold is done by:
1. Reshape W into blocks
2. Run low-rank decomposition on each block
3. Extract L and R block-diagonal factors

Reference: Dao et al., "Monarch: Expressive Structured Matrices for Efficient
and Accurate Training" (ICML 2022)
"""

import json
from pathlib import Path
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm

from .verify import verify_lossless, print_verification


@dataclass
class MonarchLayer:
    """A weight matrix stored in Monarch factored form."""
    name: str
    L_blocks: list[np.ndarray]  # List of (block_size, block_size) matrices
    R_blocks: list[np.ndarray]  # List of (block_size, block_size) matrices
    original_shape: tuple
    b: int  # number of blocks
    block_size_L: tuple  # (rows, cols) per L block
    block_size_R: tuple  # (rows, cols) per R block

    @property
    def original_params(self) -> int:
        return int(np.prod(self.original_shape))

    @property
    def compressed_params(self) -> int:
        l_params = sum(b.size for b in self.L_blocks)
        r_params = sum(b.size for b in self.R_blocks)
        return l_params + r_params

    @property
    def ratio(self) -> float:
        return self.compressed_params / self.original_params

    def reconstruct(self) -> np.ndarray:
        """Reconstruct dense matrix from Monarch factors."""
        return monarch_reconstruct(
            self.L_blocks, self.R_blocks, self.b, self.original_shape
        )


def find_block_size(n: int) -> int:
    """Find the best block count b such that b divides n and b ≈ sqrt(n).

    For Monarch factorization, we need n = b * block_size, so b must divide n.
    We want b close to sqrt(n) for optimal compression.
    """
    target = int(np.sqrt(n))
    # Search outward from sqrt(n) for a divisor
    for delta in range(target):
        if (target - delta) > 0 and n % (target - delta) == 0:
            return target - delta
        if (target + delta) <= n and n % (target + delta) == 0:
            return target + delta
    return 1  # fallback: single block (no compression)


def monarch_project(W: np.ndarray, b: int = None, rank: int = 1) -> tuple:
    """Project a dense matrix W onto the Monarch manifold.

    For a square matrix W (n x n):
        W ≈ P_L @ L @ P_R @ R

    The projection algorithm (from Dao et al.):
    1. View W as (b, n/b, b, n/b) by reshaping
    2. For each pair of block indices (i,j), extract W[i,:,j,:] which is (n/b, n/b)
    3. Do rank-k SVD on each such block to get the best L and R factors
    4. L_blocks[i] and R_blocks[j] accumulate these contributions

    For rectangular matrices, we adapt the block structure.

    With rank > 1, this becomes a sum of Monarch matrices (multi-rank Monarch).
    Each rank adds one set of L/R contributions. Storage scales linearly with rank.

    Args:
        W: Dense weight matrix (m, n)
        b: Number of blocks. If None, auto-detect ≈ sqrt(min(m,n)).
        rank: Number of singular values to keep per sub-block.
              rank=1 is standard Monarch. Higher rank = better accuracy, more params.

    Returns:
        (L_blocks, R_blocks, b, reconstruction_error)
    """
    m, n = W.shape

    if b is None:
        b = find_block_size(min(m, n))

    # We need both m and n divisible by b
    if m % b != 0 or n % b != 0:
        for candidate in range(b, 0, -1):
            if m % candidate == 0 and n % candidate == 0:
                b = candidate
                break
        else:
            return None, None, 0, float('inf')

    bm = m // b  # block height for L
    bn = n // b  # block width for R

    # Cap rank at the maximum possible
    max_rank = min(bm, bn)
    rank = min(rank, max_rank)

    # Reshape W into (b, bm, b, bn) — view as b×b grid of (bm, bn) sub-blocks
    W_blocks = W.reshape(b, bm, b, bn)

    # For multi-rank: L_blocks[i] is (bm, b*rank), R_blocks[j] is (b*rank, bn)
    L_blocks = [np.zeros((bm, b * rank), dtype=W.dtype) for _ in range(b)]
    R_blocks = [np.zeros((b * rank, bn), dtype=W.dtype) for _ in range(b)]

    for i in range(b):
        for j in range(b):
            sub = W_blocks[i, :, j, :]  # (bm, bn)
            U, S, Vt = np.linalg.svd(sub, full_matrices=False)
            for k in range(rank):
                sqrt_s = np.sqrt(S[k])
                L_blocks[i][:, j * rank + k] = U[:, k] * sqrt_s
                R_blocks[j][i * rank + k, :] = sqrt_s * Vt[k, :]

    # Compute reconstruction error
    W_approx = monarch_reconstruct(L_blocks, R_blocks, b, (m, n), rank)
    error = np.linalg.norm(W - W_approx) / np.linalg.norm(W)

    return L_blocks, R_blocks, b, float(error)


def monarch_reconstruct(
    L_blocks: list[np.ndarray],
    R_blocks: list[np.ndarray],
    b: int,
    shape: tuple,
    rank: int = None,
) -> np.ndarray:
    """Reconstruct dense matrix from Monarch factors.

    For multi-rank Monarch:
    W[i*bm:(i+1)*bm, j*bn:(j+1)*bn] = sum_{k=0}^{rank-1} L[i][:,j*rank+k] @ R[j][i*rank+k,:]

    Each (i,j) sub-block is a rank-k approximation via SVD.
    """
    m, n = shape
    bm = m // b
    bn = n // b

    if rank is None:
        # Infer rank from block shapes
        rank = L_blocks[0].shape[1] // b

    W_approx = np.zeros(shape, dtype=L_blocks[0].dtype)
    for i in range(b):
        for j in range(b):
            # Multi-rank: columns j*rank to j*rank+rank in L, rows i*rank to i*rank+rank in R
            L_sub = L_blocks[i][:, j * rank:(j + 1) * rank]  # (bm, rank)
            R_sub = R_blocks[j][i * rank:(i + 1) * rank, :]  # (rank, bn)
            W_approx[i * bm:(i + 1) * bm, j * bn:(j + 1) * bn] = L_sub @ R_sub

    return W_approx


def monarch_forward(
    L_blocks: list[torch.Tensor],
    R_blocks: list[torch.Tensor],
    x: torch.Tensor,
    b: int,
) -> torch.Tensor:
    """Fast Monarch forward pass: y = x @ M where M = P_L @ L @ P_R @ R.

    Instead of reconstructing the full matrix, compute in factored form:
    1. Reshape x from (..., m) to (..., b, bm)
    2. Apply L blocks: for each block i, z[..., i, :] = x[..., i, :] @ L_blocks[i]
       Result shape: (..., b, b) — the inner dim is now b
    3. Permute: transpose the last two dims
    4. Apply R blocks: for each block j, y[..., j, :] = z[..., j, :] @ R_blocks[j]
       Result shape: (..., b, bn)
    5. Reshape to (..., n)

    FLOPs: b * (bm * b) + b * (b * bn) = b²*bm + b²*bn = b²*(bm+bn)
    vs dense: m * n = b*bm * b*bn = b²*bm*bn

    Speedup factor: (bm*bn) / (bm+bn) ≈ sqrt(n)/2 for square matrices
    """
    orig_shape = x.shape[:-1]
    m = x.shape[-1]
    bm = m // b

    # Step 1: reshape x to (..., b, bm)
    x_blocked = x.reshape(*orig_shape, b, bm)

    # Step 2: apply L blocks — each L_blocks[i] is (bm, b)
    # z[..., i, :] = x_blocked[..., i, :] @ L_blocks[i]
    # Stack L blocks into (b, bm, b) for batched matmul
    L_stacked = torch.stack(L_blocks, dim=0)  # (b, bm, b)
    z = torch.einsum('...ij,ijk->...ik', x_blocked, L_stacked)  # (..., b, b)

    # Step 3: permute — transpose last two dims
    z = z.transpose(-2, -1)  # (..., b, b)

    # Step 4: apply R blocks — each R_blocks[j] is (b, bn)
    R_stacked = torch.stack(R_blocks, dim=0)  # (b, b, bn)
    y = torch.einsum('...ij,ijk->...ik', z, R_stacked)  # (..., b, bn)

    # Step 5: reshape to (..., n)
    n = b * R_blocks[0].shape[-1]
    y = y.reshape(*orig_shape, n)

    return y


def compress_model_monarch(
    model: GPT2LMHeadModel,
    skip_embeddings: bool = True,
) -> tuple[GPT2LMHeadModel, list[MonarchLayer], dict]:
    """Compress GPT-2 using Monarch matrix factorization.

    This projects each weight matrix onto the Monarch manifold — finding the
    closest Monarch matrix. This is NOT lossless (it's a rank-1-per-block
    approximation), but it reveals how much Monarch structure exists in the
    trained weights.
    """
    monarch_layers = []
    stats = {
        "layers": {},
        "total_original": 0,
        "total_compressed": 0,
        "skipped": [],
    }

    for name, param in tqdm(list(model.named_parameters()), desc="Monarch projection"):
        data = param.detach().cpu().numpy().astype(np.float64)

        if data.ndim != 2:
            stats["total_original"] += data.size
            stats["total_compressed"] += data.size
            stats["skipped"].append(name)
            continue

        if skip_embeddings and ("wte" in name or "wpe" in name):
            stats["total_original"] += data.size
            stats["total_compressed"] += data.size
            stats["skipped"].append(name)
            continue

        m, n = data.shape
        L_blocks, R_blocks, b, rel_error = monarch_project(data)

        if L_blocks is None:
            stats["skipped"].append(f"{name} (not factorizable)")
            stats["total_original"] += data.size
            stats["total_compressed"] += data.size
            continue

        ml = MonarchLayer(
            name=name,
            L_blocks=L_blocks,
            R_blocks=R_blocks,
            original_shape=data.shape,
            b=b,
            block_size_L=(m // b, b),
            block_size_R=(b, n // b),
        )
        monarch_layers.append(ml)

        # Reconstruct and write back
        reconstructed = ml.reconstruct().astype(np.float32)
        with torch.no_grad():
            param.copy_(torch.from_numpy(reconstructed))

        stats["layers"][name] = {
            "original_shape": list(data.shape),
            "blocks": b,
            "block_size_L": list(ml.block_size_L),
            "block_size_R": list(ml.block_size_R),
            "original_params": ml.original_params,
            "compressed_params": ml.compressed_params,
            "ratio": round(ml.ratio, 4),
            "relative_error": round(rel_error, 6),
        }
        stats["total_original"] += ml.original_params
        stats["total_compressed"] += ml.compressed_params

    stats["overall_ratio"] = round(
        stats["total_compressed"] / max(stats["total_original"], 1), 4
    )

    return model, monarch_layers, stats


def bench_monarch_speed(
    W: np.ndarray,
    name: str,
    device: str = "cuda",
    n_iters: int = 1000,
    batch: int = 1,
    seq_len: int = 128,
) -> dict:
    """Benchmark dense vs Monarch factored forward pass speed."""
    import time

    m, n = W.shape
    L_blocks_np, R_blocks_np, b, rel_error = monarch_project(W)

    if L_blocks_np is None:
        return {"error": "not factorizable"}

    W_t = torch.from_numpy(W.astype(np.float32)).to(device)
    x = torch.randn(batch, seq_len, m, device=device)

    L_blocks_t = [torch.from_numpy(lb.astype(np.float32)).to(device) for lb in L_blocks_np]
    R_blocks_t = [torch.from_numpy(rb.astype(np.float32)).to(device) for rb in R_blocks_np]

    # Warmup
    for _ in range(50):
        _ = x @ W_t
        _ = monarch_forward(L_blocks_t, R_blocks_t, x, b)
    torch.cuda.synchronize() if device == "cuda" else None

    # Dense
    start = time.perf_counter()
    for _ in range(n_iters):
        y_dense = x @ W_t
    torch.cuda.synchronize() if device == "cuda" else None
    dense_ms = (time.perf_counter() - start) / n_iters * 1000

    # Monarch
    start = time.perf_counter()
    for _ in range(n_iters):
        y_monarch = monarch_forward(L_blocks_t, R_blocks_t, x, b)
    torch.cuda.synchronize() if device == "cuda" else None
    monarch_ms = (time.perf_counter() - start) / n_iters * 1000

    max_diff = float((y_dense - y_monarch).abs().max().cpu())

    return {
        "name": name,
        "shape": f"{m}x{n}",
        "blocks": b,
        "dense_ms": round(dense_ms, 4),
        "monarch_ms": round(monarch_ms, 4),
        "speedup": round(dense_ms / monarch_ms, 2),
        "relative_error": round(rel_error, 6),
        "max_logit_diff": round(max_diff, 4),
        "dense_flops": batch * seq_len * m * n,
        "monarch_flops": batch * seq_len * b * b * (m // b + n // b),
        "param_ratio": round(
            (sum(lb.size for lb in L_blocks_np) + sum(rb.size for rb in R_blocks_np))
            / (m * n), 4
        ),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Monarch factorization of GPT-2")
    parser.add_argument("--bench-speed", action="store_true", help="Also benchmark speed")
    parser.add_argument("--rank-sweep", action="store_true", help="Sweep over ranks to find error/compression tradeoff")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    output_dir = Path("artifacts/monarch")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Monarch Matrix Factorization ===\n")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    if args.rank_sweep:
        print("--- Rank Sweep: Error vs Compression Tradeoff ---\n")
        # Test one representative layer at various ranks
        target = "transformer.h.0.mlp.c_fc.weight"
        for pname, param in model.named_parameters():
            if pname != target:
                continue
            W = param.detach().cpu().numpy().astype(np.float64)
            m, n = W.shape
            b = find_block_size(min(m, n))
            if m % b != 0 or n % b != 0:
                for c in range(b, 0, -1):
                    if m % c == 0 and n % c == 0:
                        b = c
                        break
            bm, bn = m // b, n // b
            max_rank = min(bm, bn)

            print(f"Layer: {pname} ({m}x{n}), blocks={b}, block_size=({bm},{bn}), max_rank={max_rank}\n")
            print(f"{'Rank':>6} {'ParamRatio':>11} {'Params':>10} {'RelError':>10} {'Lossless?':>10}")
            print("-" * 55)

            for rank in [1, 2, 4, 8, 12, 16, 24, max_rank]:
                if rank > max_rank:
                    continue
                _, _, _, rel_err = monarch_project(W, b=b, rank=rank)
                # Param count: b blocks of (bm, b*rank) + b blocks of (b*rank, bn)
                params = b * (bm * b * rank + b * rank * bn)
                ratio = params / (m * n)
                lossless = "YES" if rel_err < 1e-10 else "no"
                print(f"{rank:>6} {ratio:>11.4f} {params:>10,} {rel_err:>10.6f} {lossless:>10}")
        print()

    if args.bench_speed:
        print("--- Speed Benchmark: Dense vs Monarch ---\n")
        target_layers = [
            "transformer.h.0.attn.c_attn.weight",
            "transformer.h.0.attn.c_proj.weight",
            "transformer.h.0.mlp.c_fc.weight",
            "transformer.h.0.mlp.c_proj.weight",
        ]

        print(f"{'Layer':<42} {'Shape':<10} {'Blk':>4} {'Dense(ms)':>10} {'Monarch(ms)':>12} {'Speedup':>8} {'RelErr':>10} {'ParamRatio':>11}")
        print("-" * 115)

        for pname, param in model.named_parameters():
            if pname not in target_layers:
                continue
            W = param.detach().cpu().numpy().astype(np.float64)
            result = bench_monarch_speed(W, pname, device=args.device)
            if "error" in result:
                print(f"{pname:<42} SKIPPED: {result['error']}")
                continue
            print(
                f"{result['name']:<42} {result['shape']:<10} {result['blocks']:>4} "
                f"{result['dense_ms']:>10.4f} {result['monarch_ms']:>12.4f} "
                f"{result['speedup']:>7.2f}x {result['relative_error']:>10.6f} "
                f"{result['param_ratio']:>10.4f}"
            )

        print()

    # Full model compression + verification (rank-1)
    print("--- Full Model Monarch Projection (rank-1) ---\n")
    model2 = GPT2LMHeadModel.from_pretrained("gpt2")
    model2, monarch_layers, stats = compress_model_monarch(model2)

    with open(output_dir / "monarch_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nOverall param ratio: {stats['overall_ratio']:.4f}")
    print(f"Skipped: {len(stats['skipped'])} layers\n")

    print(f"{'Layer':<45} {'Blocks':>6} {'Ratio':>8} {'RelError':>10}")
    print("-" * 75)
    for name, info in stats["layers"].items():
        print(
            f"{name:<45} {info['blocks']:>6} "
            f"{info['ratio']:>8.4f} {info['relative_error']:>10.6f}"
        )

    # Verify
    print("\n=== Verification ===\n")
    results = verify_lossless(model2, tokenizer)
    print_verification(results)


if __name__ == "__main__":
    main()
