"""
Download GPT-2 Small and generate reference outputs for lossless verification.

Usage:
    python -m exotic_compress.baseline
"""

import json
import hashlib
from pathlib import Path

import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer


REFERENCE_PROMPTS = [
    "The meaning of life is",
    "In the beginning, there was",
    "Mathematics is the language of",
    "The quick brown fox jumps over",
    "Neural networks learn by",
]

OUTPUT_DIR = Path("artifacts/baseline")


def get_model_fingerprint(model: GPT2LMHeadModel) -> dict:
    """Compute a fingerprint of all model weights for lossless verification."""
    fingerprint = {}
    total_params = 0
    for name, param in model.named_parameters():
        data = param.detach().cpu().numpy()
        fingerprint[name] = {
            "shape": list(data.shape),
            "params": int(np.prod(data.shape)),
            "md5": hashlib.md5(data.tobytes()).hexdigest(),
            "norm_l2": float(np.linalg.norm(data)),
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
        }
        total_params += fingerprint[name]["params"]
    fingerprint["__total_params__"] = total_params
    return fingerprint


def generate_reference_logits(
    model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer
) -> dict:
    """Generate reference logits for lossless verification.

    We store full logits (not just argmax) so we can verify the compressed
    model computes the EXACT same function, not just similar outputs.
    """
    model.eval()
    references = {}
    with torch.no_grad():
        for prompt in REFERENCE_PROMPTS:
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(model.device)
            outputs = model(input_ids)
            logits = outputs.logits.cpu().numpy()
            references[prompt] = {
                "input_ids": input_ids.cpu().numpy().tolist(),
                "logits_shape": list(logits.shape),
                "logits_md5": hashlib.md5(logits.tobytes()).hexdigest(),
                # Store last-position logits for quick comparison
                "last_logits_top10": {
                    "indices": np.argsort(logits[0, -1])[-10:][::-1].tolist(),
                    "values": np.sort(logits[0, -1])[-10:][::-1].tolist(),
                },
            }
    return references


def analyze_weight_structure(model: GPT2LMHeadModel) -> dict:
    """Analyze each weight matrix for compressibility signals."""
    analysis = {}
    for name, param in model.named_parameters():
        data = param.detach().cpu().numpy()
        if data.ndim < 2:
            analysis[name] = {"type": "1d", "size": data.shape[0]}
            continue

        # Reshape to 2D for SVD analysis
        if data.ndim > 2:
            data_2d = data.reshape(data.shape[0], -1)
        else:
            data_2d = data

        # SVD spectrum — how quickly singular values decay tells us compressibility
        try:
            s = np.linalg.svd(data_2d.astype(np.float64), compute_uv=False)
            s_normalized = s / s[0] if s[0] > 0 else s

            # Effective rank: number of singular values > 1% of the largest
            effective_rank_01 = int(np.sum(s_normalized > 0.01))
            effective_rank_10 = int(np.sum(s_normalized > 0.10))

            # Energy concentration: what fraction of total energy in top-k
            total_energy = np.sum(s**2)
            energy_top10 = float(np.sum(s[:10] ** 2) / total_energy)
            energy_top50 = float(np.sum(s[:50] ** 2) / total_energy)

            analysis[name] = {
                "shape": list(data.shape),
                "params": int(np.prod(data.shape)),
                "rank_full": int(min(data_2d.shape)),
                "effective_rank_1pct": effective_rank_01,
                "effective_rank_10pct": effective_rank_10,
                "energy_top10_pct": round(energy_top10 * 100, 2),
                "energy_top50_pct": round(energy_top50 * 100, 2),
                "compression_potential": round(
                    1.0 - effective_rank_01 / min(data_2d.shape), 3
                ),
                "singular_values_top5": s[:5].tolist(),
            }
        except Exception as e:
            analysis[name] = {"shape": list(data.shape), "error": str(e)}

    return analysis


def main():
    print("=== Exotic Compress: GPT-2 Baseline ===\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    print("Loading GPT-2 Small...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    # 1. Fingerprint
    print("Computing weight fingerprint...")
    fingerprint = get_model_fingerprint(model)
    total = fingerprint.pop("__total_params__")
    print(f"  Total parameters: {total:,}")
    with open(OUTPUT_DIR / "fingerprint.json", "w") as f:
        json.dump(fingerprint, f, indent=2)
    fingerprint["__total_params__"] = total

    # 2. Reference logits
    print("Generating reference logits...")
    references = generate_reference_logits(model, tokenizer)
    with open(OUTPUT_DIR / "reference_logits.json", "w") as f:
        json.dump(references, f, indent=2)

    # 3. Weight structure analysis
    print("Analyzing weight structure (SVD spectra)...")
    analysis = analyze_weight_structure(model)
    with open(OUTPUT_DIR / "weight_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)

    # Summary
    print("\n=== Compressibility Summary ===\n")
    print(f"{'Layer':<45} {'Shape':<20} {'Params':>10} {'EffRank(1%)':>12} {'Compress':>10}")
    print("-" * 100)
    for name, info in analysis.items():
        if "compression_potential" in info:
            shape_str = "x".join(str(s) for s in info["shape"])
            print(
                f"{name:<45} {shape_str:<20} {info['params']:>10,} "
                f"{info['effective_rank_1pct']:>12} "
                f"{info['compression_potential']:>9.1%}"
            )

    print(f"\nBaseline artifacts saved to {OUTPUT_DIR}/")
    print("These will be used to verify lossless compression.")


if __name__ == "__main__":
    main()
