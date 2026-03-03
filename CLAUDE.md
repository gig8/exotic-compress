# exotic-compress

Lossless neural network compression using exotic math. GPT-2 Small (124M) as test target. RTX 3090.

## Commands
```bash
pip install -e ".[dev]"                          # install
python -m exotic_compress.baseline               # generate reference outputs
python -m exotic_compress.compress_svd            # SVD compression
python -m exotic_compress.compress_svd --threshold 0.01  # near-lossless SVD
python -m exotic_compress.compress_tt             # Tensor Train compression
pytest tests/                                    # run tests
```

## Verify Your Work
Every compression method MUST be verified against baseline reference logits:
```python
from exotic_compress.verify import verify_lossless, print_verification
results = verify_lossless(compressed_model, tokenizer)
print_verification(results)
```
Lossless = bit-identical logits on all reference prompts. Near-lossless = bounded max diff.

## Project Structure
```
exotic_compress/          # Python package
  baseline.py             # Download GPT-2, generate reference fingerprints & logits
  verify.py               # Lossless verification against baseline
  compress_svd.py         # SVD-based decomposition
  compress_tt.py          # Tensor Train decomposition
artifacts/                # Generated data (gitignored)
  baseline/               # Reference fingerprints and logits
  svd/                    # SVD compression results
  tt/                     # TT compression results
docs/                     # Research notes, method descriptions
experiments/              # One-off experiment scripts
notebooks/                # Jupyter notebooks for analysis
tests/                    # pytest tests
```

## Key Docs
@docs/research-log.md -- running log of experiments and findings
@docs/methods.md -- description of each compression approach

## Code Conventions
- All compression functions return (model, layer_data, stats_dict)
- Stats dicts always include: original_params, compressed_params, ratio
- Weight analysis uses float64 for SVD/decomposition, float32 for model weights
- Reference logits are stored as MD5 hashes, not raw tensors

## Conditional Rules
- When adding a new compression method: follow the pattern in compress_svd.py
- When modifying model weights: ALWAYS verify against baseline after
- When reporting compression ratios: ratio = compressed_params / original_params (lower is better)
- When working with weight matrices: compute in float64, store in float32

## Do NOT
- NEVER modify baseline.py reference prompts (breaks all verification)
- NEVER commit artifacts/ directory (large binary data)
- NEVER claim "lossless" without bit-exact verification passing
- NEVER skip the reconstruction error check in compression methods
- Do NOT use GPU for decomposition math (SVD, TT) — NumPy on CPU is fine, model inference uses GPU

## Context Management
Preserve across compaction: current compression method being worked on, verification status, last experiment results.

## Research Methodology
1. **Hypothesis first**: State what you expect before running an experiment
2. **Measure baseline**: Always compare against the unmodified model
3. **One variable at a time**: Change one parameter per experiment
4. **Log everything**: Update docs/research-log.md with findings
5. **Verify lossless**: Run verification after every model modification
6. **Goldilocks changes**: 1-2 files, 50-200 LOC per change
7. **Leave breadcrumbs**: Document what was tried, what worked, what didn't
