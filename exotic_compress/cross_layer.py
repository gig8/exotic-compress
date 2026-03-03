"""
Cross-layer weight analysis for GPT-2.

If individual layers are full-rank and unstructured, maybe the structure
is BETWEEN layers. This module analyzes:

1. Weight similarity: Do different layers have similar weight matrices?
   If so, weight sharing or delta-encoding could compress.

2. Subspace alignment: Do different layers operate in similar subspaces?
   If U_i ≈ U_j (SVD left singular vectors), they share a basis.

3. Spectral similarity: Do layers have similar singular value distributions?
   If so, a shared spectrum + per-layer rotations could compress.

4. Weight deltas: If W_i ≈ W_0 + ΔW_i and ΔW_i is low-rank, we get
   a "base model + low-rank updates" factorization (like LoRA but discovered
   post-hoc rather than trained).
"""

import json
from pathlib import Path

import torch
import numpy as np
from transformers import GPT2LMHeadModel
from tqdm import tqdm


def extract_layer_weights(model: GPT2LMHeadModel) -> dict[str, dict[str, np.ndarray]]:
    """Extract weight matrices grouped by layer type across all transformer blocks."""
    groups = {
        "attn.c_attn": [],   # QKV projection (768 x 2304)
        "attn.c_proj": [],   # Attention output (768 x 768)
        "mlp.c_fc": [],      # FFN up (768 x 3072)
        "mlp.c_proj": [],    # FFN down (3072 x 768)
    }

    for name, param in model.named_parameters():
        for group_key in groups:
            if group_key + ".weight" in name and "transformer.h." in name:
                layer_idx = int(name.split(".")[2])
                W = param.detach().cpu().numpy().astype(np.float64)
                groups[group_key].append((layer_idx, W))

    # Sort by layer index
    for key in groups:
        groups[key].sort(key=lambda x: x[0])

    return groups


def cosine_similarity_matrices(weights: list[tuple[int, np.ndarray]]) -> np.ndarray:
    """Compute pairwise cosine similarity between flattened weight matrices."""
    n = len(weights)
    sims = np.zeros((n, n))
    flat = [w.ravel() for _, w in weights]
    norms = [np.linalg.norm(f) for f in flat]

    for i in range(n):
        for j in range(i, n):
            sim = np.dot(flat[i], flat[j]) / (norms[i] * norms[j])
            sims[i, j] = sim
            sims[j, i] = sim

    return sims


def subspace_alignment(weights: list[tuple[int, np.ndarray]], top_k: int = 50) -> np.ndarray:
    """Measure how much the top-k SVD subspaces align across layers.

    For each pair of layers, compute the principal angles between their
    top-k left singular subspaces. High alignment = shared computation basis.
    """
    n = len(weights)
    alignment = np.zeros((n, n))

    # Pre-compute SVD for each layer
    U_list = []
    for _, W in weights:
        if W.ndim == 2:
            U, _, _ = np.linalg.svd(W, full_matrices=False)
            U_list.append(U[:, :top_k])
        else:
            U_list.append(None)

    for i in range(n):
        for j in range(i, n):
            if U_list[i] is None or U_list[j] is None:
                continue
            # Subspace alignment: ||U_i^T @ U_j||_F / sqrt(k)
            # This is 1.0 if subspaces are identical, 0 if orthogonal
            overlap = np.linalg.svd(
                U_list[i].T @ U_list[j], compute_uv=False
            )
            score = float(np.mean(overlap))  # Average cosine of principal angles
            alignment[i, j] = score
            alignment[j, i] = score

    return alignment


def spectral_similarity(weights: list[tuple[int, np.ndarray]]) -> np.ndarray:
    """Compare singular value spectra across layers.

    If two layers have similar spectra, they might be "rotations" of each other.
    """
    n = len(weights)
    sims = np.zeros((n, n))

    # Pre-compute singular values
    spectra = []
    for _, W in weights:
        if W.ndim == 2:
            s = np.linalg.svd(W, compute_uv=False)
            s_norm = s / s[0] if s[0] > 0 else s
            spectra.append(s_norm)
        else:
            spectra.append(None)

    for i in range(n):
        for j in range(i, n):
            if spectra[i] is None or spectra[j] is None:
                continue
            # Truncate to same length
            min_len = min(len(spectra[i]), len(spectra[j]))
            si, sj = spectra[i][:min_len], spectra[j][:min_len]
            # Cosine similarity of normalized spectra
            sim = float(np.dot(si, sj) / (np.linalg.norm(si) * np.linalg.norm(sj)))
            sims[i, j] = sim
            sims[j, i] = sim

    return sims


def delta_analysis(weights: list[tuple[int, np.ndarray]]) -> list[dict]:
    """Analyze whether layers can be expressed as a base + low-rank delta.

    W_i = W_base + ΔW_i

    If ΔW_i is low-rank, this is a post-hoc LoRA-like decomposition.
    """
    # Use layer 0 as base, or the mean
    all_W = np.stack([w for _, w in weights])
    W_mean = np.mean(all_W, axis=0)

    results = []
    for layer_idx, W in weights:
        delta = W - W_mean
        delta_norm = np.linalg.norm(delta)
        W_norm = np.linalg.norm(W)
        relative_delta = delta_norm / W_norm

        # SVD of the delta — check if it's low-rank
        s = np.linalg.svd(delta, compute_uv=False)
        s_norm = s / s[0] if s[0] > 0 else s

        # Effective rank of delta at various thresholds
        eff_rank_1pct = int(np.sum(s_norm > 0.01))
        eff_rank_10pct = int(np.sum(s_norm > 0.10))

        # Energy in top-k singular values of delta
        total_energy = np.sum(s**2)
        energy_top10 = float(np.sum(s[:10]**2) / total_energy) if total_energy > 0 else 0
        energy_top50 = float(np.sum(s[:50]**2) / total_energy) if total_energy > 0 else 0

        full_rank = min(delta.shape)

        results.append({
            "layer": layer_idx,
            "delta_relative_norm": round(relative_delta, 4),
            "delta_eff_rank_1pct": eff_rank_1pct,
            "delta_eff_rank_10pct": eff_rank_10pct,
            "delta_full_rank": full_rank,
            "delta_energy_top10": round(energy_top10, 4),
            "delta_energy_top50": round(energy_top50, 4),
        })

    return results


def main():
    output_dir = Path("artifacts/cross_layer")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Cross-Layer Weight Analysis ===\n")

    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    groups = extract_layer_weights(model)

    all_stats = {}

    for group_name, weights in groups.items():
        if not weights:
            continue

        m, n = weights[0][1].shape
        print(f"\n{'='*80}")
        print(f"  {group_name} ({m}×{n}) — {len(weights)} layers")
        print(f"{'='*80}")

        # 1. Weight cosine similarity
        cos_sim = cosine_similarity_matrices(weights)
        print(f"\n  1. Weight Cosine Similarity (layer vs layer):")
        print(f"     Min off-diagonal: {np.min(cos_sim[np.triu_indices(len(weights), k=1)]):.4f}")
        print(f"     Max off-diagonal: {np.max(cos_sim[np.triu_indices(len(weights), k=1)]):.4f}")
        print(f"     Mean off-diagonal: {np.mean(cos_sim[np.triu_indices(len(weights), k=1)]):.4f}")

        # 2. Subspace alignment
        sub_align = subspace_alignment(weights, top_k=50)
        print(f"\n  2. Top-50 Subspace Alignment:")
        print(f"     Min: {np.min(sub_align[np.triu_indices(len(weights), k=1)]):.4f}")
        print(f"     Max: {np.max(sub_align[np.triu_indices(len(weights), k=1)]):.4f}")
        print(f"     Mean: {np.mean(sub_align[np.triu_indices(len(weights), k=1)]):.4f}")

        # 3. Spectral similarity
        spec_sim = spectral_similarity(weights)
        print(f"\n  3. Spectral Similarity:")
        print(f"     Min: {np.min(spec_sim[np.triu_indices(len(weights), k=1)]):.4f}")
        print(f"     Max: {np.max(spec_sim[np.triu_indices(len(weights), k=1)]):.4f}")
        print(f"     Mean: {np.mean(spec_sim[np.triu_indices(len(weights), k=1)]):.4f}")

        # 4. Delta analysis
        deltas = delta_analysis(weights)
        print(f"\n  4. Delta Analysis (W_i - W_mean):")
        print(f"     {'Layer':>5} {'||Δ||/||W||':>12} {'EffRank(1%)':>12} {'EffRank(10%)':>13} {'E_top10':>10} {'E_top50':>10}")
        print(f"     {'-'*65}")
        for d in deltas:
            print(
                f"     {d['layer']:>5} {d['delta_relative_norm']:>12.4f} "
                f"{d['delta_eff_rank_1pct']:>12} {d['delta_eff_rank_10pct']:>13} "
                f"{d['delta_energy_top10']:>10.4f} {d['delta_energy_top50']:>10.4f}"
            )

        # Store stats
        all_stats[group_name] = {
            "shape": [m, n],
            "n_layers": len(weights),
            "cosine_similarity": {
                "min": float(np.min(cos_sim[np.triu_indices(len(weights), k=1)])),
                "max": float(np.max(cos_sim[np.triu_indices(len(weights), k=1)])),
                "mean": float(np.mean(cos_sim[np.triu_indices(len(weights), k=1)])),
            },
            "subspace_alignment_top50": {
                "min": float(np.min(sub_align[np.triu_indices(len(weights), k=1)])),
                "max": float(np.max(sub_align[np.triu_indices(len(weights), k=1)])),
                "mean": float(np.mean(sub_align[np.triu_indices(len(weights), k=1)])),
            },
            "spectral_similarity": {
                "min": float(np.min(spec_sim[np.triu_indices(len(weights), k=1)])),
                "max": float(np.max(spec_sim[np.triu_indices(len(weights), k=1)])),
                "mean": float(np.mean(spec_sim[np.triu_indices(len(weights), k=1)])),
            },
            "delta_analysis": deltas,
        }

    # Check if deltas are low-rank (the key question)
    print(f"\n\n{'='*80}")
    print("  SUMMARY: Is delta-encoding viable?")
    print(f"{'='*80}\n")

    for group_name, stats in all_stats.items():
        deltas = stats["delta_analysis"]
        mean_delta_norm = np.mean([d["delta_relative_norm"] for d in deltas])
        mean_eff_rank = np.mean([d["delta_eff_rank_1pct"] for d in deltas])
        full_rank = deltas[0]["delta_full_rank"]
        mean_e50 = np.mean([d["delta_energy_top50"] for d in deltas])

        compressible = mean_eff_rank < full_rank * 0.5
        print(f"  {group_name}:")
        print(f"    Mean ||Δ||/||W||:     {mean_delta_norm:.4f}")
        print(f"    Mean EffRank(1%):     {mean_eff_rank:.0f} / {full_rank} ({mean_eff_rank/full_rank*100:.1f}%)")
        print(f"    Mean E_top50:         {mean_e50:.4f}")
        print(f"    Delta compressible?   {'YES — potential for weight sharing' if compressible else 'NO — deltas are full-rank'}")
        print()

    with open(output_dir / "cross_layer_stats.json", "w") as f:
        json.dump(all_stats, f, indent=2)

    print(f"Stats saved to {output_dir}/")


if __name__ == "__main__":
    main()
