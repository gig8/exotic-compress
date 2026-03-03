# Research Log

Running log of experiments, findings, and next steps.

## 2026-03-02: Project Setup

### Goal
Explore lossless compression of GPT-2 Small (124M params) using exotic mathematical techniques.

### Initial Methods
1. **SVD decomposition** — decompose weight matrices into U, S, Vt. Full rank = lossless. Truncated = bounded error.
2. **Tensor Train (TT)** — reshape weights into higher-order tensors, decompose into chain of small cores. Full TT-rank = lossless.

### Next Steps
- [ ] Run baseline to generate reference outputs
- [ ] Run SVD analysis to understand compressibility of each layer
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
