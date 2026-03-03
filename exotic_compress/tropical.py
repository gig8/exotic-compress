"""
Tropical / functional analysis of GPT-2 MLP layers.

Core idea from tropical geometry:
  ReLU networks compute piecewise-linear (PWL) functions.
  In tropical algebra, these are "tropical rational maps."
  Two different weight configurations can compute the SAME PWL function.
  If we find such redundancy, we can simplify the network.

GPT-2 uses GELU (not ReLU), but GELU ≈ x * sigmoid(1.702x) and is
nearly piecewise-linear for |x| > 2. The analysis still applies approximately.

This module takes a PRACTICAL approach to functional analysis:

1. **Activation pattern analysis**: For a corpus of inputs, record which MLP
   neurons are "active" (output > threshold). If many neurons have identical
   or near-identical activation patterns, they're functionally redundant.

2. **Dead neuron detection**: Neurons that never activate (always ~0 output)
   across all inputs can be removed with zero loss.

3. **Correlated neuron detection**: Pairs of neurons whose activations are
   perfectly correlated can be merged (one neuron + scaling).

4. **Linear region counting**: Estimate the number of distinct linear regions
   the MLP implements. If far fewer than the theoretical max, there's
   structural redundancy in the weights.

5. **Effective dimensionality**: PCA on activation vectors — how many dimensions
   do the hidden activations actually use?

These are the PRACTICAL tests before full tropical algebra (which is PhD-level).
"""

import json
from pathlib import Path
from collections import Counter

import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm


def get_mlp_activations(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    texts: list[str],
    device: str = "cpu",
    layer_idx: int = 0,
) -> dict:
    """Run texts through the model and capture MLP intermediate activations.

    In GPT-2, each MLP block computes:
        h = GELU(x @ W_fc + b_fc)   # hidden activations (dim 3072)
        y = h @ W_proj + b_proj      # project back to dim 768

    We hook into the GELU output to get h.
    """
    model = model.to(device)
    model.eval()

    activations = []
    input_tokens = []

    # Hook to capture post-GELU activations
    hook_output = {}

    def hook_fn(module, input, output):
        hook_output["act"] = output.detach().cpu()

    # The MLP activation function in GPT-2 is applied inside the MLP forward.
    # We hook on c_fc (the up-projection) and apply GELU ourselves,
    # OR we hook the full MLP and extract the intermediate.
    # Actually, let's hook the c_fc linear layer and get its output (pre-GELU).
    mlp = model.transformer.h[layer_idx].mlp
    handle = mlp.c_fc.register_forward_hook(hook_fn)

    with torch.no_grad():
        for text in texts:
            tokens = tokenizer.encode(text, return_tensors="pt").to(device)
            input_tokens.append(tokens.cpu())
            _ = model(tokens)

            # c_fc output is pre-GELU, shape (1, seq_len, 3072)
            pre_gelu = hook_output["act"]
            # Apply GELU to get the actual activation values
            post_gelu = torch.nn.functional.gelu(pre_gelu)
            activations.append(post_gelu.squeeze(0))  # (seq_len, 3072)

    handle.remove()

    # Concatenate all activations: (total_tokens, 3072)
    all_activations = torch.cat(activations, dim=0)

    return {
        "activations": all_activations,  # (N, 3072) post-GELU values
        "n_tokens": all_activations.shape[0],
        "n_neurons": all_activations.shape[1],
    }


def analyze_dead_neurons(activations: torch.Tensor, threshold: float = 1e-6) -> dict:
    """Find neurons that never activate (always near zero).

    A dead neuron contributes nothing to the output and can be removed.
    """
    # activations: (N, d_ff)
    max_abs = activations.abs().max(dim=0).values  # (d_ff,)
    dead_mask = max_abs < threshold
    n_dead = int(dead_mask.sum())

    # Also check for "nearly dead" — neurons that rarely activate
    active_fraction = (activations.abs() > threshold).float().mean(dim=0)  # (d_ff,)
    rarely_active = (active_fraction < 0.01)  # active less than 1% of the time
    n_rarely_active = int(rarely_active.sum())

    return {
        "n_dead": n_dead,
        "n_rarely_active_1pct": n_rarely_active,
        "dead_fraction": n_dead / activations.shape[1],
        "rarely_active_fraction": n_rarely_active / activations.shape[1],
        "active_fraction_per_neuron": {
            "min": float(active_fraction.min()),
            "max": float(active_fraction.max()),
            "mean": float(active_fraction.mean()),
            "median": float(active_fraction.median()),
        },
    }


def analyze_activation_patterns(
    activations: torch.Tensor, threshold: float = 0.0
) -> dict:
    """Analyze binary activation patterns (on/off).

    For tropical analysis, what matters is the PATTERN of which neurons are active,
    not the exact values. Each distinct pattern defines a linear region.
    """
    # Binarize: neuron is "active" if output > threshold
    # For GELU: output can be negative (unlike ReLU), but small
    binary = (activations > threshold).int()  # (N, d_ff)

    n_tokens, d_ff = binary.shape

    # Count unique patterns (each pattern = a linear region)
    # We can't enumerate all 2^3072 patterns, so we hash them
    pattern_hashes = []
    for i in range(n_tokens):
        # Hash the binary pattern
        h = hash(binary[i].numpy().tobytes())
        pattern_hashes.append(h)

    unique_patterns = len(set(pattern_hashes))

    # Per-neuron activation rate
    activation_rate = binary.float().mean(dim=0)  # (d_ff,)

    # Neuron correlation: what fraction of neuron pairs have identical patterns?
    # Full pairwise is O(d_ff^2 * N) — sample instead
    n_sample = min(500, d_ff)
    sample_idx = torch.randperm(d_ff)[:n_sample]
    binary_sample = binary[:, sample_idx].float()  # (N, n_sample)

    # Correlation matrix
    centered = binary_sample - binary_sample.mean(dim=0, keepdim=True)
    stds = centered.std(dim=0, keepdim=True)
    stds = torch.clamp(stds, min=1e-8)
    normalized = centered / stds
    corr = (normalized.T @ normalized) / n_tokens  # (n_sample, n_sample)

    # Find highly correlated pairs (|corr| > 0.95)
    upper_tri = torch.triu(corr, diagonal=1)
    high_corr_mask = upper_tri.abs() > 0.95
    n_high_corr_pairs = int(high_corr_mask.sum())
    total_pairs = n_sample * (n_sample - 1) // 2

    # Anti-correlated pairs (corr < -0.95) — these are also redundant
    anti_corr_mask = upper_tri < -0.95
    n_anti_corr_pairs = int(anti_corr_mask.sum())

    return {
        "n_tokens_analyzed": n_tokens,
        "n_unique_patterns": unique_patterns,
        "theoretical_max_patterns": f"2^{d_ff}",
        "pattern_utilization": f"{unique_patterns} / 2^{d_ff}",
        "activation_rate": {
            "min": float(activation_rate.min()),
            "max": float(activation_rate.max()),
            "mean": float(activation_rate.mean()),
            "median": float(activation_rate.median()),
        },
        "correlation_analysis": {
            "n_sampled_neurons": n_sample,
            "high_corr_pairs_95pct": n_high_corr_pairs,
            "anti_corr_pairs_95pct": n_anti_corr_pairs,
            "total_pairs_sampled": total_pairs,
            "high_corr_fraction": n_high_corr_pairs / max(total_pairs, 1),
            "anti_corr_fraction": n_anti_corr_pairs / max(total_pairs, 1),
        },
    }


def analyze_effective_dimensionality(activations: torch.Tensor) -> dict:
    """PCA on activation vectors to find effective dimensionality.

    If 3072-dim activations live in a much lower-dimensional subspace,
    the MLP is not using its full capacity.
    """
    # activations: (N, d_ff) — center first
    mean = activations.mean(dim=0, keepdim=True)
    centered = activations - mean

    # Covariance via SVD (more numerically stable than explicit covariance)
    # For large N, compute on a subsample
    max_samples = 5000
    if centered.shape[0] > max_samples:
        idx = torch.randperm(centered.shape[0])[:max_samples]
        centered = centered[idx]

    # SVD of centered data matrix
    U, S, Vt = torch.linalg.svd(centered.float(), full_matrices=False)

    # Variance explained by each component
    variance = S ** 2 / (centered.shape[0] - 1)
    total_var = variance.sum()
    var_ratio = variance / total_var
    cumulative = torch.cumsum(var_ratio, dim=0)

    # Effective rank at various thresholds
    eff_dim_90 = int((cumulative < 0.90).sum()) + 1
    eff_dim_95 = int((cumulative < 0.95).sum()) + 1
    eff_dim_99 = int((cumulative < 0.99).sum()) + 1

    # Participation ratio (a smooth measure of effective dimensionality)
    # PR = (sum λ_i)^2 / sum(λ_i^2)
    pr = float((variance.sum() ** 2) / (variance ** 2).sum())

    return {
        "full_dimensionality": int(activations.shape[1]),
        "effective_dim_90pct": eff_dim_90,
        "effective_dim_95pct": eff_dim_95,
        "effective_dim_99pct": eff_dim_99,
        "participation_ratio": round(pr, 1),
        "top10_variance_explained": round(float(cumulative[9]) * 100, 2),
        "top50_variance_explained": round(float(cumulative[49]) * 100, 2),
        "top100_variance_explained": round(float(cumulative[99]) * 100, 2),
    }


def analyze_gelu_linearity(activations_pre_gelu: torch.Tensor) -> dict:
    """Analyze how much of the GELU activation is in its linear regime.

    GELU(x) ≈ x for x >> 0 and ≈ 0 for x << 0.
    The "interesting" nonlinear region is roughly |x| < 2.

    If most activations are in the linear regime (|x| > 2), the MLP
    is mostly computing a linear function and ReLU/GELU barely matters.
    This is directly relevant to tropical analysis — linear regions
    correspond to fixed activation patterns.
    """
    abs_vals = activations_pre_gelu.abs()

    # Fraction in each regime
    deep_off = (activations_pre_gelu < -3).float().mean()      # GELU ≈ 0
    transition_neg = ((activations_pre_gelu >= -3) & (activations_pre_gelu < -1)).float().mean()
    nonlinear = ((activations_pre_gelu >= -1) & (activations_pre_gelu <= 1)).float().mean()
    transition_pos = ((activations_pre_gelu > 1) & (activations_pre_gelu <= 3)).float().mean()
    deep_on = (activations_pre_gelu > 3).float().mean()        # GELU ≈ x

    linear_fraction = float(deep_off + deep_on)

    return {
        "deep_off_fraction": round(float(deep_off), 4),
        "transition_neg_fraction": round(float(transition_neg), 4),
        "nonlinear_fraction": round(float(nonlinear), 4),
        "transition_pos_fraction": round(float(transition_pos), 4),
        "deep_on_fraction": round(float(deep_on), 4),
        "linear_regime_fraction": round(linear_fraction, 4),
        "interpretation": (
            f"{linear_fraction*100:.1f}% of activations in linear regime "
            f"(GELU ≈ 0 or GELU ≈ x). "
            f"Network is {linear_fraction*100:.0f}% piecewise-linear."
        ),
    }


# Corpus for activation analysis
ANALYSIS_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In a hole in the ground there lived a hobbit.",
    "To be or not to be, that is the question.",
    "The only thing we have to fear is fear itself.",
    "I think, therefore I am.",
    "All happy families are alike; each unhappy family is unhappy in its own way.",
    "It was the best of times, it was the worst of times.",
    "Call me Ishmael.",
    "The world is a stage, and all the men and women merely players.",
    "Ask not what your country can do for you.",
    "Machine learning models compress representations of the training data.",
    "The transformer architecture uses self-attention to process sequences.",
    "Neural networks approximate continuous functions via composition.",
    "Gradient descent finds local minima in the loss landscape.",
    "The quick sort algorithm has O(n log n) average case complexity.",
    "Quantum computing leverages superposition and entanglement.",
    "The Fourier transform decomposes signals into frequency components.",
    "Category theory provides abstract frameworks for mathematical structure.",
    "Topological data analysis reveals shape in high-dimensional data.",
    "The central limit theorem explains why normal distributions arise.",
]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Tropical/functional analysis of GPT-2")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 5, 11],
                        help="Which transformer layers to analyze")
    args = parser.parse_args()

    output_dir = Path("artifacts/tropical")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== Tropical / Functional Analysis ===\n")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    all_results = {}

    for layer_idx in args.layers:
        print(f"\n{'='*80}")
        print(f"  Layer {layer_idx} MLP (768 → 3072 → 768)")
        print(f"{'='*80}")

        # Get activations
        print(f"\n  Collecting activations from {len(ANALYSIS_TEXTS)} texts...")
        act_data = get_mlp_activations(
            model, tokenizer, ANALYSIS_TEXTS,
            device=args.device, layer_idx=layer_idx
        )
        activations = act_data["activations"]
        print(f"  Total tokens: {act_data['n_tokens']}, Neurons: {act_data['n_neurons']}")

        # Also get pre-GELU for linearity analysis
        pre_gelu_data = {}
        def pre_gelu_hook(module, input, output):
            pre_gelu_data["pre"] = output.detach().cpu()

        handle = model.transformer.h[layer_idx].mlp.c_fc.register_forward_hook(pre_gelu_hook)
        pre_gelu_all = []
        with torch.no_grad():
            for text in ANALYSIS_TEXTS:
                tokens = tokenizer.encode(text, return_tensors="pt").to(args.device)
                _ = model(tokens)
                pre_gelu_all.append(pre_gelu_data["pre"].squeeze(0))
        handle.remove()
        pre_gelu_cat = torch.cat(pre_gelu_all, dim=0)

        # 1. Dead neuron analysis
        print(f"\n  --- 1. Dead Neuron Analysis ---")
        dead = analyze_dead_neurons(activations)
        print(f"  Dead neurons (never activate):    {dead['n_dead']} / {act_data['n_neurons']} ({dead['dead_fraction']*100:.1f}%)")
        print(f"  Rarely active (<1% of tokens):    {dead['n_rarely_active_1pct']} ({dead['rarely_active_fraction']*100:.1f}%)")
        print(f"  Activation rate: min={dead['active_fraction_per_neuron']['min']:.3f}, "
              f"mean={dead['active_fraction_per_neuron']['mean']:.3f}, "
              f"max={dead['active_fraction_per_neuron']['max']:.3f}")

        # 2. Activation pattern analysis
        print(f"\n  --- 2. Activation Pattern Analysis ---")
        patterns = analyze_activation_patterns(activations)
        print(f"  Unique activation patterns:  {patterns['n_unique_patterns']} (out of {patterns['n_tokens_analyzed']} tokens)")
        print(f"  (Theoretical max: {patterns['theoretical_max_patterns']})")
        print(f"  High-correlation neuron pairs (|r|>0.95): {patterns['correlation_analysis']['high_corr_pairs_95pct']} / {patterns['correlation_analysis']['total_pairs_sampled']}")
        print(f"  Anti-correlated pairs (r<-0.95):          {patterns['correlation_analysis']['anti_corr_pairs_95pct']}")

        # 3. Effective dimensionality
        print(f"\n  --- 3. Effective Dimensionality of Activations ---")
        eff_dim = analyze_effective_dimensionality(activations)
        print(f"  Full dimensionality:    {eff_dim['full_dimensionality']}")
        print(f"  90% variance in:        {eff_dim['effective_dim_90pct']} dims")
        print(f"  95% variance in:        {eff_dim['effective_dim_95pct']} dims")
        print(f"  99% variance in:        {eff_dim['effective_dim_99pct']} dims")
        print(f"  Participation ratio:    {eff_dim['participation_ratio']}")
        print(f"  Top 10 PCs explain:     {eff_dim['top10_variance_explained']}%")
        print(f"  Top 50 PCs explain:     {eff_dim['top50_variance_explained']}%")
        print(f"  Top 100 PCs explain:    {eff_dim['top100_variance_explained']}%")

        # 4. GELU linearity
        print(f"\n  --- 4. GELU Linearity Analysis ---")
        linearity = analyze_gelu_linearity(pre_gelu_cat)
        print(f"  Deep off (x < -3):      {linearity['deep_off_fraction']*100:.1f}%")
        print(f"  Transition (-3 to -1):  {linearity['transition_neg_fraction']*100:.1f}%")
        print(f"  Nonlinear (-1 to 1):    {linearity['nonlinear_fraction']*100:.1f}%")
        print(f"  Transition (1 to 3):    {linearity['transition_pos_fraction']*100:.1f}%")
        print(f"  Deep on (x > 3):        {linearity['deep_on_fraction']*100:.1f}%")
        print(f"  → {linearity['interpretation']}")

        all_results[f"layer_{layer_idx}"] = {
            "n_tokens": act_data["n_tokens"],
            "dead_neurons": dead,
            "activation_patterns": patterns,
            "effective_dimensionality": eff_dim,
            "gelu_linearity": linearity,
        }

    # Summary
    print(f"\n\n{'='*80}")
    print("  SUMMARY: Functional Redundancy in GPT-2 MLP Layers")
    print(f"{'='*80}\n")

    print(f"  {'Layer':<10} {'Dead':>8} {'Rarely':>10} {'EffDim90':>10} {'EffDim99':>10} {'PR':>8} {'Linear%':>10}")
    print(f"  {'-'*68}")
    for layer_key, stats in all_results.items():
        layer_num = layer_key.split("_")[1]
        dead = stats["dead_neurons"]
        eff = stats["effective_dimensionality"]
        lin = stats["gelu_linearity"]
        print(f"  {layer_num:>5}     {dead['n_dead']:>5}   {dead['n_rarely_active_1pct']:>7}   "
              f"{eff['effective_dim_90pct']:>7}    {eff['effective_dim_99pct']:>7}   "
              f"{eff['participation_ratio']:>6.0f}    {lin['linear_regime_fraction']*100:>6.1f}%")

    print(f"\n  Interpretation:")
    # Check for compressibility signals
    any_dead = any(r["dead_neurons"]["n_dead"] > 0 for r in all_results.values())
    any_low_dim = any(r["effective_dimensionality"]["effective_dim_99pct"] < 2000
                      for r in all_results.values())
    any_linear = any(r["gelu_linearity"]["linear_regime_fraction"] > 0.7
                     for r in all_results.values())

    if any_dead:
        print("  → Dead neurons found — trivially removable without any loss")
    if any_low_dim:
        print("  → Activations live in a lower-dimensional subspace — MLP is over-parameterized")
    if any_linear:
        print("  → Most activations are in GELU's linear regime — network is mostly piecewise-linear")
        print("    This validates the tropical analysis premise: the function IS piecewise-linear")
    if not any_dead and not any_low_dim:
        print("  → No obvious functional redundancy — GPT-2 Small uses its full capacity")

    with open(output_dir / "tropical_analysis.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n  Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
