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
- [x] **Implement Monarch matrix factorization** (priority — designed for speed)
- [ ] Implement Kronecker factorization (W ≈ A ⊗ B)
- [ ] Benchmark full model end-to-end inference with factored layers
- [ ] Explore tropical algebra simplification
- [ ] Implement log-domain inference

---

## 2026-03-02: Monarch Matrix Factorization Results

### What is Monarch?
Factor a dense matrix W (m×n) as: M = P_L @ L @ P_R @ R, where L and R are
block-diagonal matrices and P_L, P_R are fixed permutations (reshape + transpose).
Storage: O(n^{3/2}) instead of O(n²). FLOPs: same reduction.

### Experiment 4: Rank Sweep on c_fc layer (768×3072)
Blocks=24, block_size=(32,128), max_rank=32.

| Rank | ParamRatio | Params    | RelError | Lossless? |
|------|-----------|-----------|----------|-----------|
| 1    | 0.039     | 92,160    | 0.955    | no        |
| 2    | 0.078     | 184,320   | 0.916    | no        |
| 4    | 0.156     | 368,640   | 0.844    | no        |
| 8    | 0.312     | 737,280   | 0.714    | no        |
| 12   | 0.469     | 1,105,920 | 0.595    | no        |
| 16   | 0.625     | 1,474,560 | 0.483    | no        |
| 24   | 0.938     | 2,211,840 | 0.270    | no        |
| **32** | **1.250** | **2,949,120** | **0.000** | **YES** |

### Key Finding: Lossless Only at Full Block Rank
Monarch projection is lossless ONLY at max rank (rank=32 for block_size 32×128).
At max rank, the param ratio is 1.25 — same expansion as full-rank SVD. GPT-2's
weights have no natural Monarch structure; the sub-blocks are full-rank.

The error curve is roughly linear — no "elbow" suggesting hidden low-rank block structure.
This means the weights don't naturally decompose into block-diagonal form.

### Experiment 5: Monarch Rank-1 Speed Benchmark (CUDA, RTX 3090)

| Layer         | Shape     | Dense(ms) | Monarch(ms) | Speedup | RelErr |
|---------------|-----------|-----------|-------------|---------|--------|
| c_attn        | 768×2304  | 0.0508    | 0.1959      | 0.26x   | 0.942  |
| c_proj (attn) | 768×768   | 0.0387    | 0.2166      | 0.18x   | 0.791  |
| c_fc          | 768×3072  | 0.0471    | 0.1983      | 0.24x   | 0.955  |
| c_proj (mlp)  | 3072×768  | 0.0536    | 0.1987      | 0.27x   | 0.909  |

**Monarch rank-1 is 3-5x SLOWER than dense** on GPU. The einsum-based factored
forward pass has too much Python/kernel-launch overhead at these small matrix sizes.
Monarch's theoretical O(n√n) advantage only manifests at much larger n, or with
custom CUDA kernels that fuse the block-diagonal operations.

### Implications
1. **GPT-2 Small is too small for Monarch speedups.** The overhead of structured
   computation dominates at 768-dim. Monarch is designed for n ≥ 4096+.
2. **No natural block structure** in GPT-2 weights — rank sweep shows linear error
   decay, not an elbow. The weights are "maximally dense" relative to Monarch basis.
3. **Need either:** (a) much larger models where n is large enough, or (b) hardware-
   native structured ops (custom CUDA kernels, photonic block-diagonal).
4. **For lossless on GPT-2:** must look at approaches that exploit OTHER structure
   than rank or block-diagonal: cross-layer patterns, weight symmetries, or the
   tropical/functional view.

### Updated Next Steps
- [x] Run TT decomposition (different structural assumption)
- [ ] Analyze cross-layer weight similarity (do layers share structure?)
- [ ] Implement Kronecker factorization (different decomposition basis)
- [ ] Try on a larger model (Qwen3-0.6B) where n=1536+ may show Monarch benefit
- [ ] Explore tropical algebra simplification
- [ ] Implement log-domain inference

---

## 2026-03-02: Tensor Train Decomposition Results

### What TT Tests For
TT decomposition reshapes a 2D matrix into a high-order tensor and decomposes it into
a chain of small 3D cores (Matrix Product State in physics). This finds **Kronecker-like**
and **hierarchical tensor** structure that SVD and Monarch can't detect.

Example: 768×2304 → reshape to (96, 2, 2, 2, 288, 2, 2, 2) → 8 TT-cores.

### Experiment 6: TT Rank Sweep

#### c_attn (768×2304) — QKV projection
| MaxRank | Ratio  | Params    | RelErr | Lossless? |
|---------|--------|-----------|--------|-----------|
| 2       | 0.001  | 1,388     | 0.998  | no        |
| 8       | 0.011  | 19,668    | 0.974  | no        |
| 32      | 0.047  | 83,028    | 0.926  | no        |
| 128     | 0.223  | 394,324   | 0.728  | no        |
| full    | 1.443  | 2,552,916 | 0.000  | YES       |

Full TT-ranks at lossless: [1, 96, 192, 384, 768, 8, 4, 2, 1]

#### c_proj attn (768×768) — attention output
| MaxRank | Ratio  | Params    | RelErr | Lossless? |
|---------|--------|-----------|--------|-----------|
| 8       | 0.013  | 7,380     | 0.956  | no        |
| 64      | 0.136  | 79,956    | 0.768  | no        |
| 128     | 0.335  | 197,716   | 0.564  | no        |
| full    | 2.328  | 1,373,268 | 0.000  | YES       |

#### c_fc (768×3072) — FFN up-projection
| MaxRank | Ratio  | Params    | RelErr | Lossless? |
|---------|--------|-----------|--------|-----------|
| 32      | 0.046  | 107,604   | 0.943  | no        |
| 128     | 0.209  | 492,628   | 0.802  | no        |
| full    | 1.332  | 3,142,740 | 0.000  | YES       |

#### c_proj mlp (3072×768) — FFN down-projection
| MaxRank | Ratio  | Params    | RelErr | Lossless? |
|---------|--------|-----------|--------|-----------|
| 64      | 0.042  | 98,388    | 0.909  | no        |
| 128     | 0.104  | 245,844   | 0.863  | no        |
| full    | 2.563  | 6,045,780 | 0.000  | YES       |

### Key Findings

1. **Full-rank TT is even WORSE than SVD for expansion:** ratio 1.3-2.6x at lossless.
   The 8-core TT structure adds overhead from the chain of rank connections.

2. **No hidden tensor structure:** Error decays smoothly with rank, no elbow.
   The full TT-ranks reach 768 at the boundary between row/column factors — meaning
   the matrix has no exploitable Kronecker-like separability.

3. **Interesting asymmetry in the TT-ranks:** The ranks are large in the middle
   (where row and column factors meet) and small at the edges (where the binary
   factors are). This matches theory: the "entanglement" is at the row/column boundary.

4. **The reshape factorization matters:** 768 = 96×2×2×2 puts most information in
   the large factor (96). A different factorization (e.g., 768 = 4×4×6×8) might
   distribute entanglement differently and yield different compression behavior.

### Comparison Across Methods (Layer 0, c_proj 768×768)

| Method           | Lossless Ratio | At 10% params | RelErr @10% |
|------------------|----------------|---------------|-------------|
| SVD              | 1.250          | r=76 (0.20)   | ~0.42       |
| Monarch (rank-k) | 1.250          | rank=2 (0.06) | ~0.79       |
| TT               | 2.328          | rank=64 (0.14)| ~0.77       |
| Dense (original) | 1.000          | —             | 0.000       |

**None of the three decompositions can compress GPT-2 losslessly.** The weights are
genuinely dense/unstructured in all tested bases.

### Meta-Conclusion: The Weights Are Not the Right Target

Three independent structural analyses (SVD/rank, Monarch/block-diagonal, TT/tensor)
all agree: GPT-2 Small's weight matrices have no exploitable structure for lossless
compression. The weights are what they need to be — they've been optimized to use
all available capacity.

**The logical next step is to shift from parameter-level to function-level analysis:**
- Cross-layer redundancy: do different layers compute similar functions?
- Tropical simplification: can the piecewise-linear function be simplified?
- Log-domain: can we restructure the arithmetic without touching the weights?

### Updated Next Steps
- [ ] **Cross-layer analysis** — compare weight similarity across layers 0-11
- [ ] **Log-domain inference** — restructure arithmetic, not weights
- [ ] Explore tropical algebra simplification (function-level)
- [ ] Try on Qwen3-0.6B (larger model may have more redundancy)
