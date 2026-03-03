# exotic-compress

Lossless neural network compression using exotic math. Starting with GPT-2 Small (124M).

## Thesis

Standard model compression (quantization, pruning, distillation) is lossy. We explore
mathematically exact restructuring of weight matrices to reduce parameter count and
compute cost **without changing the function the model computes**.

## Approaches

| Method | Idea | Lossless? |
|--------|------|-----------|
| SVD truncation | Decompose W = UΣV†, drop near-zero singular values | Near-lossless (ε-bounded) |
| Tensor Train (TT) | Factorize weight tensors into chains of small 3D cores | Exact at full rank |
| Kronecker factorization | W ≈ A ⊗ B, product of smaller matrices | Exact if structure exists |
| Monarch matrices | Block-diagonal × permutation factorization | Exact refactoring |
| Tropical simplification | Algebraic simplification of the piecewise-linear function | Exact (same function) |
| Log-domain arithmetic | Represent in log space; multiplies become adds | Exact (representation change) |

## Target Model

**GPT-2 Small** (124M parameters, 12 layers, d_model=768, 12 heads)

Why: fully understood architecture, small enough to inspect every weight, large enough to
be non-trivial, extensive tooling and baselines available.

## Quick Start

```bash
pip install -e ".[dev]"
python -m exotic_compress.baseline  # download GPT-2 and run reference outputs
python -m exotic_compress.analyze   # analyze weight matrix structure
```

## Hardware

Developed on RTX 3090 (24GB VRAM). All experiments designed to run on a single consumer GPU.
