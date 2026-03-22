# Research Plan: Steering in the Expanded MLP Space

## Motivation & Novelty Assessment

### Why This Research Matters
Activation steering is a powerful technique for controlling LLM behavior at inference time, but nearly all work targets the residual stream. The MLP intermediate space is 4x wider (e.g., 3072 vs 768 in GPT-2 Small) and naturally sparse after the activation function — potentially offering a richer, more decomposable space for steering interventions. Understanding whether steering works better or worse here has direct implications for interpretability-based model control.

### Gap in Existing Work
The literature review reveals a critical gap: **no systematic comparison of direct activation steering in the MLP intermediate space vs the residual stream exists.** The SNMF paper (Shafran et al., 2025) shows SNMF-derived MLP features outperform SAEs at steering, but uses a feature-based approach rather than direct mean-difference steering. The ActAdd/CAA literature works exclusively in the residual stream. Nobody has asked the simple question: if you compute a mean-difference steering vector in the 3072-dim MLP intermediate space and add it there, how does it compare to the standard 768-dim residual stream approach?

### Our Novel Contribution
We conduct the first systematic comparison of direct activation steering across four intervention points in GPT-2 Small:
1. **Residual stream** (768-dim) — the standard approach
2. **MLP intermediate / post-activation** (3072-dim) — the expanded space
3. **MLP output** (768-dim, after down-projection) — projected MLP contribution

We also measure three properties that bear on *why* one space might be better:
- **Linearity**: How well do linear probes classify concepts in each space?
- **Sparsity**: How sparse are activations in the MLP intermediate space?
- **Reliability**: How consistent is the steering effect across different input prompts?

### Experiment Justification
- **Experiment 1 (Steering Effectiveness)**: Core test — does steering work in the expanded MLP space? Compared at matched KL divergence for fairness.
- **Experiment 2 (Linearity Analysis)**: Tests the hypothesis that the MLP space is "more linear" due to post-activation sparsity.
- **Experiment 3 (Sparsity Analysis)**: Quantifies the natural sparsity structure of the expanded space.
- **Experiment 4 (Reliability)**: Tests whether MLP-space steering is more or less consistent across inputs than residual-stream steering.

## Research Question
Can we steer model behavior by intervening in the expanded MLP intermediate space (post-activation, 4x wider than residual stream), and is it easier (due to linearity/sparsity) or harder (due to specificity) than residual stream steering?

## Hypothesis Decomposition
- **H1**: Steering vectors computed in the MLP intermediate space produce concept-aligned generations when applied there.
- **H2**: MLP intermediate space has higher linear separability for concepts than the residual stream.
- **H3**: MLP intermediate activations are naturally sparse, with concept information concentrated in few dimensions.
- **H4**: Steering in the MLP intermediate space is more/less reliable across different input prompts than residual stream steering.

## Proposed Methodology

### Approach
Use GPT-2 Small via TransformerLens. Compute mean-difference steering vectors from contrastive prompt pairs in both spaces. Apply steering at matched KL divergence budgets. Evaluate with automated metrics (sentiment classifier for SST-2 concept, perplexity for fluency).

### Model
- **GPT-2 Small** (12 layers, d_model=768, d_mlp=3072)
- Well-studied, fast, pre-trained SAEs available for comparison

### Concepts for Steering
1. **Sentiment** (positive vs negative) — using SST-2 dataset
2. We generate contrastive prompt pairs for activation collection

### Experimental Steps
1. Load GPT-2 Small, collect activations at residual stream and MLP intermediate for contrastive sentiment prompts
2. Compute mean-difference steering vectors in each space
3. Apply steering at multiple alpha values, measure KL divergence vs baseline
4. Generate text with steering, evaluate concept alignment and fluency
5. Train linear probes in both spaces, compare accuracy
6. Measure activation sparsity (L0, Gini coefficient) in both spaces
7. Measure steering effect variance across diverse prompts

### Baselines
- **No steering** (baseline generations)
- **Residual stream ActAdd** (standard approach from Turner et al.)
- **MLP output steering** (after down-projection, same dim as residual)

### Evaluation Metrics
- **Steering effectiveness**: Sentiment classifier score on steered generations
- **Fluency**: Perplexity of steered text (lower = more fluent)
- **Linear probe accuracy**: Classification accuracy in each space
- **Sparsity**: L0 norm, Gini coefficient of activations
- **Reliability**: Standard deviation of steering effect across prompts
- **KL divergence**: Measures intervention magnitude

### Statistical Analysis Plan
- Compare metrics across spaces using paired t-tests (same prompts, different intervention points)
- Report means ± standard deviations across prompts
- Use multiple alpha/KL budgets to show curves, not just point estimates
- Significance level: α = 0.05

## Expected Outcomes
- **If MLP space is better**: Higher concept alignment at same KL budget, higher linear probe accuracy, sparser activations
- **If MLP space is worse**: Lower concept alignment (features too specific/local), lower reliability, requiring higher KL to achieve same effect
- **Mixed outcome likely**: MLP space may be better for some concepts (those with dedicated neuron groups) but worse for others (distributed representations)

## Timeline
- Planning: 20 min ✓
- Environment setup: 10 min
- Implementation: 60 min
- Experiments: 60 min
- Analysis & visualization: 30 min
- Documentation: 20 min

## Potential Challenges
- MLP intermediate space is 4x larger — steering vectors may be noisier with limited data
- Need to ensure fair comparison (matched KL divergence)
- GPT-2 Small is small — results may not generalize to larger models
- Sentiment may be too simple a concept — but good for proof-of-concept

## Success Criteria
- Clear quantitative comparison of steering effectiveness across spaces
- Evidence for or against the linearity/sparsity hypotheses
- Reproducible results with documented code
