"""
Tensor Train (TT) decomposition for lossless weight compression.

A weight matrix W of shape (m, n) is reshaped into a higher-order tensor
and then decomposed into a train of small 3D cores:

    W[i1,i2,...,id, j1,j2,...,jd] = G1[i1] @ G2[i2] @ ... @ Gd[id,jd]

where each Gk is a small (rk-1, nk, rk) tensor and rk are the TT-ranks.

Key properties:
- At full TT-rank: EXACT reconstruction (lossless)
- Storage: sum of rk-1 * nk * rk for each core (vs m*n for dense)
- If the weight matrix has Kronecker-like structure, TT-ranks are naturally low

This is the tensor network approach from quantum physics (MPS = Matrix Product State).
"""

import json
from pathlib import Path
from dataclasses import dataclass, field

import torch
import numpy as np
import tensorly as tl
from tensorly.decomposition import tensor_train
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm

from .verify import verify_lossless, print_verification

tl.set_backend("numpy")


def factorize_shape(n: int, target_factors: int = 4) -> list[int]:
    """Factorize n into roughly equal factors for TT reshaping.

    GPT-2 dimensions: 768 = 4*4*6*8, 3072 = 4*4*8*24, 50257 (prime-ish)
    """
    factors = []
    d = 2
    remaining = n
    while d * d <= remaining and len(factors) < target_factors - 1:
        while remaining % d == 0 and len(factors) < target_factors - 1:
            factors.append(d)
            remaining //= d
        d += 1
    factors.append(remaining)

    # Merge small factors to get closer to target_factors count
    while len(factors) > target_factors:
        # Merge the two smallest
        factors.sort()
        factors = [factors[0] * factors[1]] + factors[2:]

    return sorted(factors, reverse=True)


@dataclass
class TTLayer:
    """A weight matrix stored in Tensor Train form."""
    name: str
    cores: list[np.ndarray]
    original_shape: tuple
    tt_shape: tuple  # reshaped tensor dimensions
    tt_ranks: list[int]

    @property
    def original_params(self) -> int:
        return int(np.prod(self.original_shape))

    @property
    def compressed_params(self) -> int:
        return sum(core.size for core in self.cores)

    @property
    def ratio(self) -> float:
        return self.compressed_params / self.original_params

    def reconstruct(self) -> np.ndarray:
        """Reconstruct the full tensor from TT cores."""
        result = tl.tt_to_tensor(self.cores)
        return result.reshape(self.original_shape)


def compress_layer_tt(
    name: str,
    weight: np.ndarray,
    max_rank: int = None,
) -> TTLayer:
    """Compress a single weight matrix using TT decomposition.

    Args:
        name: Layer name
        weight: Weight matrix (2D)
        max_rank: Maximum TT-rank. None = full rank (lossless).
    """
    original_shape = weight.shape

    if weight.ndim != 2:
        raise ValueError(f"Expected 2D weight, got {weight.ndim}D")

    m, n = weight.shape

    # Factorize dimensions for TT reshaping
    m_factors = factorize_shape(m)
    n_factors = factorize_shape(n)

    # Reshape to higher-order tensor: (m1, m2, ..., n1, n2, ...)
    tt_shape = tuple(m_factors + n_factors)

    # Check if reshape is possible
    if np.prod(m_factors) != m or np.prod(n_factors) != n:
        # Can't factorize cleanly — skip this layer
        return None

    tensor = weight.reshape(tt_shape)

    # TT decomposition
    if max_rank is not None:
        rank = [1] + [max_rank] * (len(tt_shape) - 1) + [1]
    else:
        # Full rank — compute max possible ranks at each position
        # TT-rank at position k is at most min(prod(shape[:k]), prod(shape[k:]))
        rank = [1]
        for k in range(1, len(tt_shape)):
            left = int(np.prod(tt_shape[:k]))
            right = int(np.prod(tt_shape[k:]))
            rank.append(min(left, right))
        rank.append(1)

    cores = tensor_train(tensor, rank=rank)

    tt_ranks = [1] + [core.shape[-1] for core in cores]

    return TTLayer(
        name=name,
        cores=[np.array(core) for core in cores],
        original_shape=original_shape,
        tt_shape=tt_shape,
        tt_ranks=tt_ranks,
    )


def compress_model_tt(
    model: GPT2LMHeadModel,
    max_rank: int = None,
    skip_embeddings: bool = True,
) -> tuple[GPT2LMHeadModel, list[TTLayer], dict]:
    """Compress GPT-2 using Tensor Train decomposition.

    Args:
        model: Original GPT-2 model
        max_rank: Max TT-rank per core. None = lossless.
        skip_embeddings: Skip embedding layers (large, hard to factorize).
    """
    tt_layers = []
    stats = {"layers": {}, "total_original": 0, "total_compressed": 0, "skipped": []}

    for name, param in tqdm(list(model.named_parameters()), desc="TT decomposition"):
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

        try:
            tt_layer = compress_layer_tt(name, data, max_rank=max_rank)
        except Exception as e:
            stats["skipped"].append(f"{name} (error: {e})")
            stats["total_original"] += data.size
            stats["total_compressed"] += data.size
            continue

        if tt_layer is None:
            stats["skipped"].append(f"{name} (unfactorizable shape)")
            stats["total_original"] += data.size
            stats["total_compressed"] += data.size
            continue

        tt_layers.append(tt_layer)

        # Reconstruct and verify
        reconstructed = tt_layer.reconstruct().astype(np.float32)
        reconstruction_error = np.max(np.abs(data.astype(np.float32) - reconstructed))

        with torch.no_grad():
            param.copy_(torch.from_numpy(reconstructed))

        stats["layers"][name] = {
            "original_shape": list(tt_layer.original_shape),
            "tt_shape": list(tt_layer.tt_shape),
            "tt_ranks": tt_layer.tt_ranks,
            "original_params": tt_layer.original_params,
            "compressed_params": tt_layer.compressed_params,
            "ratio": round(tt_layer.ratio, 4),
            "max_reconstruction_error": float(reconstruction_error),
        }
        stats["total_original"] += tt_layer.original_params
        stats["total_compressed"] += tt_layer.compressed_params

    stats["overall_ratio"] = round(
        stats["total_compressed"] / max(stats["total_original"], 1), 4
    )

    return model, tt_layers, stats


def analyze_tt_single_layer(name: str, weight: np.ndarray) -> dict:
    """Analyze TT structure of a single layer at various max ranks.

    Returns dict with rank sweep results showing error vs compression.
    """
    m, n = weight.shape
    m_factors = factorize_shape(m)
    n_factors = factorize_shape(n)
    tt_shape = tuple(m_factors + n_factors)

    if np.prod(m_factors) != m or np.prod(n_factors) != n:
        return {"error": "unfactorizable"}

    results = []

    # Full rank first (lossless baseline)
    tt_full = compress_layer_tt(name, weight, max_rank=None)
    if tt_full is None:
        return {"error": "decomposition failed"}

    results.append({
        "max_rank": "full",
        "tt_ranks": tt_full.tt_ranks,
        "compressed_params": tt_full.compressed_params,
        "ratio": round(tt_full.ratio, 4),
        "max_error": float(np.max(np.abs(
            weight.astype(np.float32) - tt_full.reconstruct().astype(np.float32)
        ))),
    })

    # Rank sweep
    for max_rank in [2, 4, 8, 16, 32, 64, 128]:
        try:
            tt = compress_layer_tt(name, weight, max_rank=max_rank)
            if tt is None:
                continue
            recon = tt.reconstruct().astype(np.float32)
            max_err = float(np.max(np.abs(weight.astype(np.float32) - recon)))
            rel_err = float(np.linalg.norm(weight - recon.astype(np.float64))
                           / np.linalg.norm(weight))
            results.append({
                "max_rank": max_rank,
                "tt_ranks": tt.tt_ranks,
                "compressed_params": tt.compressed_params,
                "ratio": round(tt.ratio, 4),
                "max_error": max_err,
                "rel_error": round(rel_err, 6),
            })
        except Exception:
            continue

    return {
        "name": name,
        "shape": [m, n],
        "tt_shape": list(tt_shape),
        "m_factors": m_factors,
        "n_factors": n_factors,
        "results": results,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Tensor Train compression of GPT-2")
    parser.add_argument(
        "--max-rank", type=int, default=None,
        help="Max TT-rank (None = full rank, lossless)"
    )
    parser.add_argument(
        "--rank-sweep", action="store_true",
        help="Analyze TT structure at various ranks for representative layers"
    )
    args = parser.parse_args()

    output_dir = Path("artifacts/tt")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Tensor Train Compression ===\n")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    if args.rank_sweep:
        print("--- TT Rank Sweep ---\n")
        target_layers = [
            "transformer.h.0.attn.c_attn.weight",
            "transformer.h.0.attn.c_proj.weight",
            "transformer.h.0.mlp.c_fc.weight",
            "transformer.h.0.mlp.c_proj.weight",
        ]

        for pname, param in model.named_parameters():
            if pname not in target_layers:
                continue
            W = param.detach().cpu().numpy().astype(np.float64)
            m, n = W.shape
            m_factors = factorize_shape(m)
            n_factors = factorize_shape(n)

            print(f"--- {pname} ({m}x{n}) ---")
            print(f"    Reshape: {m} = {'x'.join(str(f) for f in m_factors)}, "
                  f"{n} = {'x'.join(str(f) for f in n_factors)}")
            print(f"    TT-shape: ({', '.join(str(f) for f in m_factors + n_factors)})\n")

            analysis = analyze_tt_single_layer(pname, W)
            if "error" in analysis:
                print(f"    ERROR: {analysis['error']}\n")
                continue

            print(f"    {'MaxRank':<10} {'Ratio':<10} {'Params':<12} {'MaxErr':<12} {'RelErr':<10} {'TT-Ranks'}")
            print(f"    {'-'*80}")
            for r in analysis["results"]:
                rank_str = str(r["max_rank"])
                rel_err = r.get("rel_error", 0.0)
                lossless_mark = " <-- LOSSLESS" if r["max_error"] < 1e-5 else ""
                print(
                    f"    {rank_str:<10} {r['ratio']:<10.4f} {r['compressed_params']:<12,} "
                    f"{r['max_error']:<12.2e} {rel_err:<10.6f} "
                    f"{r['tt_ranks']}{lossless_mark}"
                )
            print()

        # Save analysis
        with open(output_dir / "tt_rank_sweep.json", "w") as f:
            all_analysis = {}
            for pname, param in model.named_parameters():
                if pname not in target_layers:
                    continue
                W = param.detach().cpu().numpy().astype(np.float64)
                all_analysis[pname] = analyze_tt_single_layer(pname, W)
            json.dump(all_analysis, f, indent=2)

    else:
        # Standard compression run
        print(f"Max rank: {args.max_rank}\n")
        model, tt_layers, stats = compress_model_tt(model, max_rank=args.max_rank)

        with open(output_dir / "tt_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        print(f"\nOverall ratio: {stats['overall_ratio']:.4f}")
        print(f"Skipped layers: {len(stats['skipped'])}")

        print(f"\n{'Layer':<45} {'TT-Shape':<25} {'Ranks':<30} {'Ratio':<10} {'MaxErr':<12}")
        print("-" * 125)
        for name, info in stats["layers"].items():
            shape_str = "x".join(str(s) for s in info["tt_shape"])
            rank_str = str(info["tt_ranks"])
            print(
                f"{name:<45} {shape_str:<25} {rank_str:<30} "
                f"{info['ratio']:<10.4f} {info['max_reconstruction_error']:<12.2e}"
            )

        # Verify
        print("\n=== Verification ===\n")
        results = verify_lossless(model, tokenizer)
        print_verification(results)


if __name__ == "__main__":
    main()
