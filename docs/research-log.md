# Research Log

Running log of experiments, findings, and next steps.

## 2026-03-02: Project Setup

### Goal
Explore lossless compression of GPT-2 Small (124M params) using exotic mathematical techniques.

### Initial Methods
1. **SVD decomposition** — decompose weight matrices into U, S, Vt. Full rank = lossless. Truncated = bounded error.
2. **Tensor Train (TT)** — reshape weights into higher-order tensors, decompose into chain of small cores. Full TT-rank = lossless.

### Next Steps
- [x] Run baseline to generate reference outputs
- [x] Run SVD analysis to understand compressibility of each layer
- [ ] Run TT decomposition and compare compression ratios
- [ ] Implement Kronecker factorization (W ≈ A ⊗ B)
- [ ] Implement Monarch matrix factorization (block-diagonal × permutation)
- [ ] Explore tropical algebra simplification (requires understanding the piecewise-linear structure)
- [ ] Implement log-domain inference (representation change, not weight compression)

### Key Questions
1. Which layers have the lowest effective rank? (Most compressible by SVD)
2. Do attention Q/K/V matrices have different compressibility than FFN matrices?
3. Does Kronecker structure naturally exist in GPT-2 weights?
4. What's the TT-rank distribution across layers?

---

## 2026-03-02: SVD Baseline Results

### Experiment 1: Full-rank SVD (threshold=0.0)
- **Result:** ALL PASS — bit-exact lossless reconstruction
- **Compression ratio:** 1.25 (EXPANSION, not compression)
- **Why:** Full-rank SVD stores U(m×r) + S(r) + Vt(r×n). For full rank r=min(m,n), this is LARGER than the original m×n matrix.
- **Conclusion:** SVD at full rank is lossless but useless for compression. Only helps if rank can be significantly reduced.

### Experiment 2: Truncated SVD (threshold=0.01)
- **Result:** ALL FAIL — max logit diff up to 6.93
- **Compression ratio:** 1.21 (still expansion!)
- **Why:** GPT-2 Small weight matrices are nearly full rank. Even at 1% threshold, removing a few singular values causes significant output drift, and we barely save any parameters.

### Key Finding
**GPT-2 Small's weight matrices are dense and nearly full rank.** The effective rank at 1% threshold is 690-768 out of 768 max. SVD-based compression is fundamentally the wrong approach for this model.

The only layers with any compressibility signal are the attention output projections (`attn.c_proj`), with effective ranks of 690-724 (5-10% below full). Everything else is full rank.

### Implication
This validates the exotic math thesis: we need **structural** approaches that exploit patterns in HOW the numbers are arranged, not low-rank approximations. The weights are dense but may still have:
- Kronecker structure (W ≈ A ⊗ B)
- Block-diagonal + permutation structure (Monarch)
- Piecewise-linear simplifiability (tropical)
- Redundancy across layers (weight sharing, cross-layer patterns)

---

## 2026-03-02: Speed Benchmark — Restructuring for Inference Speed

### Motivation
Compression ratio (parameter count) is only one axis. Even if the SVD form has MORE total
parameters, the restructured computation could be faster because:
1. Two smaller matmuls can beat one large one (better cache locality)
2. At reduced rank, FLOPs drop linearly with rank
3. Memory bandwidth is often the real bottleneck, not compute

### Experiment 3: Dense vs SVD-factored inference speed (CUDA, RTX 3090)
Benchmark: 1000 iterations, batch=1, seq_len=128, on 4 representative layer shapes.

**Dense operation:** `y = x @ W`
**Factored operation:** `y = ((x @ U) * S) @ Vt` where `W = U @ diag(S) @ Vt`

#### Results

| Layer | Shape | Dense (ms) | SVD r=384 (ms) | Speedup | SVD r=76 (ms) | Speedup |
|-------|-------|-----------|----------------|---------|---------------|---------|
| c_attn | 768×2304 | 0.0615 | 0.0764 | 0.80x (slower) | 0.0804 | 0.76x |
| c_proj (attn) | 768×768 | 0.0312 | 0.1203 | 0.26x (slower) | 0.0749 | 0.42x |
| c_fc | 768×3072 | 0.0480 | 0.0826 | 0.58x (slower) | 0.0875 | 0.55x |
| **c_proj (mlp)** | **3072×768** | **0.1035** | **0.0826** | **1.25x** | **0.0749** | **1.38x** |

#### Key Finding: Tall-Skinny Matrices Benefit from Factored Computation
- **mlp.c_proj (3072×768) is the ONLY layer where SVD factoring is faster**
  - At rank 384 (50%): 1.25x speedup
  - At rank 76 (10%): 1.38x speedup
  - At rank 38 (5%): 1.39x speedup
- All other shapes (square 768×768, wide 768×2304, wide 768×3072) are slower with SVD factoring

#### Why?
- **Tall matrices (m >> n):** The first matmul `x @ U` projects from high-dim (3072) down to rank `r`, making the second matmul `(...) @ Vt` much cheaper. The dimensionality reduction happens early.
- **Square/wide matrices:** Two CUDA kernel launches + intermediate allocation outweigh the FLOP savings. Dense matmul is already well-optimized for these shapes.

#### Implication
**Speed and size are independent axes.** A method can be:
- Smaller AND faster (ideal)
- Smaller but slower (classical quantization tradeoff)
- Same size but faster (restructured computation — this is the exotic math opportunity)
- Larger but faster (full-rank SVD on tall matrices)

The Monarch matrix approach (block-diagonal × permutation) is designed specifically for the "same expressivity but faster" case — O(n√n) FLOPs instead of O(n²), and block-diagonal matmuls are embarrassingly parallel on GPU.

#### Accuracy Note
The speed gains above come with significant accuracy loss (max_diff 11-21). For lossless speed gains, we need structural factorizations (Monarch, Kronecker) rather than rank truncation.

### Updated Next Steps
- [ ] Run TT decomposition and compare compression ratios
- [ ] **Implement Monarch matrix factorization** (priority — designed for speed)
- [ ] Implement Kronecker factorization (W ≈ A ⊗ B)
- [ ] Benchmark full model end-to-end inference with factored layers
- [ ] Explore tropical algebra simplification
- [ ] Implement log-domain inference
