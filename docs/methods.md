# Compression Methods

## 1. SVD Decomposition

**File:** `exotic_compress/compress_svd.py`

Any matrix W (m×n) can be decomposed as W = U @ diag(S) @ Vt where:
- U is (m × r) orthogonal
- S is (r,) diagonal (singular values, sorted descending)
- Vt is (r × n) orthogonal
- r = min(m, n) for full rank

**Storage:**
- Original: m × n parameters
- SVD: m×r + r + r×n = r(m + n + 1) parameters
- Breaks even when r < mn/(m+n+1)
- For 768×768: breaks even at r < 384 (50% of full rank)
- For 768×3072: breaks even at r < 615 (80% of full rank)

**Lossless guarantee:** Full-rank SVD reconstructs exactly (up to floating point).

**Near-lossless:** Truncate singular values below threshold. Error bound: ||W - W_r||_F = sqrt(sum of dropped s_i^2).

## 2. Tensor Train (TT) Decomposition

**File:** `exotic_compress/compress_tt.py`

Reshape a matrix into a higher-order tensor, then factorize into a "train" of small cores.

Example: W (768×768) → reshape to (4, 4, 6, 8, 4, 4, 6, 8) → TT decomposition

Each core G_k has shape (r_{k-1}, n_k, r_k) where r_k is the TT-rank at position k.

**Storage:** sum of r_{k-1} × n_k × r_k for each core.

**Key insight:** If W has Kronecker-like structure (W ≈ A ⊗ B ⊗ C), the TT-ranks will be 1, giving massive compression. Real weight matrices won't be rank-1 TT, but low TT-rank indicates exploitable structure.

## 3. Kronecker Factorization (planned)

Find A (p×q) and B (s×t) such that W = A ⊗ B where m=p×s, n=q×t.

**Storage:** p×q + s×t vs m×n = p×s × q×t. Exponential savings if it works.

**Challenge:** Exact Kronecker structure rarely exists. Need nearest Kronecker product (Van Loan & Pitsianis algorithm).

## 4. Monarch Matrices (planned)

Factorize W = P × M₁ × P^T × M₂ where:
- P is a fixed permutation matrix (free)
- M₁, M₂ are block-diagonal matrices

**Storage:** O(n^1.5) instead of O(n²).

**Why it works:** This is a generalized butterfly/Hadamard factorization. Many linear transforms (FFT, Hadamard, convolutions) have this structure. Trained weight matrices may approximately have it too.

## 5. Tropical / Functional Analysis

**File:** `exotic_compress/tropical.py`

ReLU networks compute tropical rational functions (piecewise-linear maps). Two different weight configurations can compute the SAME function. Tropical algebra may reveal when this happens, enabling algebraic simplification.

Our practical approach analyzes the FUNCTION the network computes rather than full tropical algebra:
1. **Dead neuron detection** — neurons that never activate can be removed
2. **Activation pattern analysis** — correlation between neuron pairs
3. **Effective dimensionality** — PCA on hidden activations to find actual subspace used
4. **GELU linearity analysis** — what fraction of activations are in the linear regime

**Key finding:** GPT-2's 3072-dim MLP activations are effectively ~130-180 dimensional (99% variance) — only 5% of hidden capacity is used on natural text. However, GELU (not ReLU) means classical tropical algebra doesn't directly apply.

## 6. Log-Domain Inference

**File:** `exotic_compress/log_domain.py`

Not compression per se, but a representation change:
- Store weights as log2(|w|) + sign bit
- Multiplications become additions (cheaper in hardware)
- The model computes the same function, just using different arithmetic

Relevant for: custom inference engines, FPGA/ASIC deployment, optical hardware interface.

---

## Experimental Status Summary

| Method | Implemented | Lossless? | Compression | Speed on GPU | Notes |
|--------|-------------|-----------|-------------|-------------|-------|
| SVD (full rank) | Yes | Bit-exact | 1.25x EXPANSION | 1.39x faster (tall matrices only) | Useful for speed, not size |
| SVD (truncated) | Yes | No (diff ~7) | 1.21x EXPANSION | - | GPT-2 is full rank; useless |
| Monarch (rank-1) | Yes | No (err 95%) | 25x compression | 3-5x slower | Way too lossy |
| Monarch (full rank) | Yes | Yes | 1.25x EXPANSION | Slower | No natural block structure |
| Tensor Train (full rank) | Yes | Yes | 1.3-2.6x EXPANSION | Not benchmarked | No tensor structure in GPT-2 |
| Tensor Train (truncated) | Yes | No (err 73-99%) | 0.001-0.22x | Not benchmarked | Smooth error decay, no elbow |
| Tropical (functional) | Yes | N/A | N/A | N/A | Activations use ~5% of 3072-dim hidden space |
| Kronecker | Planned | - | - | - | |
| Log-domain roundtrip | Yes | No (err 1.9e-6) | 0.53x theoretical | N/A | FP32 precision lost in log2/pow2 |
| Log-domain matmul | Yes | No (diff 4.5e-5) | 0.53x theoretical | 100-1000x slower | GPU worst case; designed for FPGA/optical |
| Kronecker | Planned | - | - | - | |
| Tropical | Planned | - | - | - | Functional-level, not parameter-level |

### Core Finding So Far
GPT-2 Small's weight matrices are "maximally dense" — nearly full rank, no block structure.
Standard matrix factorizations cannot compress them losslessly. Speed gains are possible
via restructured computation (SVD on tall matrices) independent of compression.

Log-domain arithmetic is mathematically sound (max relative error 1.6%) but not suitable
for GPU execution. Its value is for FPGA/ASIC hardware where adders are 3-5x cheaper
than multipliers, and for optical hardware where phase = log-amplitude naturally.

Functional analysis reveals a gap between weight-level and function-level structure:
weights are full-rank (no compression possible), but the functions they compute on
natural text use only ~5% of the available hidden dimensionality. This suggests
input-adaptive or activation-guided approaches as the true path to compression.
