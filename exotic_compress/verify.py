"""
Verify that a compressed model computes the EXACT same function as the original.

This is the core invariant: lossless means bit-identical logits.
"""

import json
import hashlib
from pathlib import Path

import torch
import numpy as np
from transformers import GPT2Tokenizer

BASELINE_DIR = Path("artifacts/baseline")


def load_references() -> dict:
    """Load baseline reference logits."""
    with open(BASELINE_DIR / "reference_logits.json") as f:
        return json.load(f)


def verify_lossless(model, tokenizer=None, atol=0.0, rtol=0.0) -> dict:
    """Verify a model produces identical outputs to the baseline.

    Args:
        model: The compressed/restructured model (must accept same inputs as GPT-2)
        tokenizer: GPT2Tokenizer (loaded from 'gpt2' if None)
        atol: Absolute tolerance. 0.0 = bit-exact lossless.
        rtol: Relative tolerance. 0.0 = bit-exact lossless.

    Returns:
        dict with verification results per prompt
    """
    if tokenizer is None:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    references = load_references()
    model.eval()
    results = {}

    with torch.no_grad():
        for prompt, ref in references.items():
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(next(model.parameters()).device)
            outputs = model(input_ids)
            logits = outputs.logits.cpu().numpy()

            # Check 1: Shape match
            shape_match = list(logits.shape) == ref["logits_shape"]

            # Check 2: MD5 match (bit-exact)
            md5 = hashlib.md5(logits.tobytes()).hexdigest()
            bit_exact = md5 == ref["logits_md5"]

            # Check 3: Numerical closeness (for near-lossless)
            top_indices = ref["last_logits_top10"]["indices"]
            top_values = ref["last_logits_top10"]["values"]
            actual_values = [float(logits[0, -1, i]) for i in top_indices]
            max_diff = max(
                abs(a - e) for a, e in zip(actual_values, top_values)
            )

            # Check 4: Same argmax (functional equivalence)
            same_argmax = np.argmax(logits[0, -1]) == top_indices[0]

            results[prompt] = {
                "shape_match": shape_match,
                "bit_exact": bit_exact,
                "max_logit_diff": max_diff,
                "same_argmax": same_argmax,
                "lossless": bit_exact if (atol == 0 and rtol == 0) else max_diff <= atol,
            }

    return results


def print_verification(results: dict):
    """Pretty-print verification results."""
    all_pass = True
    for prompt, r in results.items():
        status = "PASS" if r["lossless"] else "FAIL"
        if not r["lossless"]:
            all_pass = False
        exact_str = "bit-exact" if r["bit_exact"] else f"max_diff={r['max_logit_diff']:.2e}"
        print(f"  [{status}] \"{prompt[:40]}...\" — {exact_str}")

    print()
    if all_pass:
        print("  ALL CHECKS PASSED — compression is lossless")
    else:
        print("  SOME CHECKS FAILED — compression is NOT lossless")

    return all_pass
