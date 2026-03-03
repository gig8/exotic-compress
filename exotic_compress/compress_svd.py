"""
SVD-based lossless/near-lossless compression of GPT-2 weight matrices.

Strategy:
- Decompose each 2D weight matrix W (m×n) via SVD: W = U @ diag(S) @ Vt
- At full rank: this is EXACT (lossless), just restructured
- Truncated to rank r: W ≈ U[:,:r] @ diag(S[:r]) @ Vt[:r,:]
  - Original params: m*n
  - Compressed params: m*r + r + r*n = r*(m+n+1)
  - Compression ratio: r*(m+n+1) / (m*n)
  - Lossless when: all singular values are preserved

The key insight: if the effective rank is much less than min(m,n),
we can store U_r, S_r, Vt_r instead of W and reconstruct W = U_r @ diag(S_r) @ Vt_r
with bounded error.

For TRUE lossless: we store the full SVD but exploit numerical structure
(e.g., many singular values are nearly identical → encode deltas).
"""

import json
from pathlib import Path
from dataclasses import dataclass

import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm

from .verify import verify_lossless, print_verification


@dataclass
class SVDLayer:
    """A weight matrix stored in SVD form."""
    name: str
    U: np.ndarray       # (m, r)
    S: np.ndarray       # (r,)
    Vt: np.ndarray      # (r, n)
    original_shape: tuple
    rank: int

    @property
    def original_params(self) -> int:
        return int(np.prod(self.original_shape))

    @property
    def compressed_params(self) -> int:
        m, n = self.U.shape[0], self.Vt.shape[1]
        return self.U.shape[0] * self.rank + self.rank + self.rank * self.Vt.shape[1]

    @property
    def ratio(self) -> float:
        return self.compressed_params / self.original_params

    def reconstruct(self) -> np.ndarray:
        """Reconstruct the original weight matrix."""
        return (self.U * self.S[None, :]) @ self.Vt


def compress_model_svd(
    model: GPT2LMHeadModel,
    threshold: float = 0.0,
    min_rank: int = 1,
) -> tuple[GPT2LMHeadModel, list[SVDLayer], dict]:
    """Compress a GPT-2 model using SVD decomposition.

    Args:
        model: Original GPT-2 model
        threshold: Drop singular values below this fraction of the largest.
                   0.0 = keep all (fully lossless).
        min_rank: Minimum rank to keep per layer.

    Returns:
        (compressed_model, svd_layers, stats)
    """
    svd_layers = []
    stats = {"layers": {}, "total_original": 0, "total_compressed": 0}

    for name, param in tqdm(list(model.named_parameters()), desc="SVD decomposition"):
        data = param.detach().cpu().numpy().astype(np.float64)

        if data.ndim < 2:
            # Skip 1D params (biases, layer norms)
            stats["total_original"] += data.size
            stats["total_compressed"] += data.size
            continue

        # For 2D weight matrices, compute full SVD
        if data.ndim == 2:
            U, S, Vt = np.linalg.svd(data, full_matrices=False)
        else:
            # Reshape higher-dim to 2D
            orig_shape = data.shape
            data_2d = data.reshape(data.shape[0], -1)
            U, S, Vt = np.linalg.svd(data_2d, full_matrices=False)

        # Determine rank based on threshold
        if threshold > 0:
            keep = S > (threshold * S[0])
            rank = max(min_rank, int(np.sum(keep)))
        else:
            rank = len(S)  # Full rank = lossless

        # Truncate
        U_r = U[:, :rank]
        S_r = S[:rank]
        Vt_r = Vt[:rank, :]

        svd_layer = SVDLayer(
            name=name,
            U=U_r,
            S=S_r,
            Vt=Vt_r,
            original_shape=data.shape,
            rank=rank,
        )
        svd_layers.append(svd_layer)

        # Reconstruct and write back to model
        reconstructed = svd_layer.reconstruct().astype(np.float32)
        if data.ndim > 2:
            reconstructed = reconstructed.reshape(orig_shape)

        with torch.no_grad():
            param.copy_(torch.from_numpy(reconstructed))

        stats["layers"][name] = {
            "original_params": svd_layer.original_params,
            "compressed_params": svd_layer.compressed_params,
            "full_rank": int(len(S)),
            "kept_rank": rank,
            "ratio": round(svd_layer.ratio, 4),
            "dropped_energy_pct": round(
                float(1.0 - np.sum(S_r**2) / np.sum(S**2)) * 100, 6
            ),
        }
        stats["total_original"] += svd_layer.original_params
        stats["total_compressed"] += svd_layer.compressed_params

    stats["overall_ratio"] = round(
        stats["total_compressed"] / stats["total_original"], 4
    )

    return model, svd_layers, stats


def main():
    import argparse

    parser = argparse.ArgumentParser(description="SVD compression of GPT-2")
    parser.add_argument(
        "--threshold", type=float, default=0.0,
        help="SVD truncation threshold (0.0 = lossless, 0.01 = drop <1%% of max)"
    )
    parser.add_argument(
        "--verify", action="store_true", default=True,
        help="Verify lossless against baseline"
    )
    args = parser.parse_args()

    output_dir = Path("artifacts/svd")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== SVD Compression (threshold={args.threshold}) ===\n")

    # Load model
    print("Loading GPT-2 Small...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Compress
    model, svd_layers, stats = compress_model_svd(model, threshold=args.threshold)

    # Save stats
    with open(output_dir / "svd_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nOverall compression ratio: {stats['overall_ratio']:.4f}")
    print(f"Original params (2D): {stats['total_original']:,}")
    print(f"SVD params:           {stats['total_compressed']:,}")

    # Show top compressed layers
    print(f"\n{'Layer':<45} {'Rank':<10} {'Ratio':<10} {'Energy Lost':<12}")
    print("-" * 80)
    for name, info in sorted(
        stats["layers"].items(), key=lambda x: x[1]["ratio"]
    )[:20]:
        print(
            f"{name:<45} {info['kept_rank']:<10} "
            f"{info['ratio']:<10.4f} {info['dropped_energy_pct']:<12.6f}%"
        )

    # Verify
    if args.verify:
        print("\n=== Verification ===\n")
        results = verify_lossless(model, tokenizer)
        print_verification(results)


if __name__ == "__main__":
    main()
