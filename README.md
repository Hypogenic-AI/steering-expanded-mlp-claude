# Steering in the Expanded MLP Space

Systematic comparison of activation steering in the MLP intermediate space (3072-dim) vs. the residual stream (768-dim) in GPT-2 Small. Investigates whether the 4x wider MLP space is easier or harder to steer.

## Key Findings

- **Steering works in MLP intermediate space** but is ~2.6x less efficient than residual stream steering (sentiment shift per unit KL divergence)
- **MLP space is NOT more linearly separable**: Linear probe accuracy 76% (MLP intermediate) vs 86% (residual stream)
- **MLP steering vectors are remarkably sparse**: Just 2.8% of neurons carry 50% of the steering signal
- **MLP space degrades faster**: Steered generations become incoherent at lower alpha values
- **The MLP space is harder to steer because neurons encode specific, less universal information** — the residual stream's integrated representation is more robust

## Project Structure

```
├── REPORT.md              # Full research report with results and analysis
├── planning.md            # Research plan and experimental design
├── src/
│   ├── experiment.py      # Main experiment (linearity, sparsity, steering, geometry)
│   └── enhanced_eval.py   # Enhanced logit-based evaluation and generation analysis
├── results/
│   ├── all_results.json   # Raw results from main experiment
│   ├── enhanced_results.json  # Enhanced evaluation results
│   └── plots/             # All visualization plots
├── literature_review.md   # Literature review
├── resources.md           # Resource catalog
├── papers/                # Downloaded research papers
├── datasets/              # SST-2, TruthfulQA, OpenWebText samples
└── code/                  # Cloned reference repositories
```

## Reproduce

```bash
uv venv && source .venv/bin/activate
uv add transformer-lens torch numpy matplotlib scikit-learn scipy tqdm datasets 'transformers<5'
python src/experiment.py       # ~3 min on RTX A6000
python src/enhanced_eval.py    # ~1 min
```

## Full Report

See [REPORT.md](REPORT.md) for detailed methodology, results tables, and analysis.
