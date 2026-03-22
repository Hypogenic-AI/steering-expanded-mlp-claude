# Literature Review: Steering in the Expanded MLP Space

## Research Area Overview

This review covers research at the intersection of two areas in mechanistic interpretability: (1) **activation steering** — modifying model behavior by intervening on internal representations at inference time, and (2) **MLP interpretability** — understanding the intermediate (expanded) space within MLP sublayers of transformers. The central research question is whether the expanded MLP space (typically 4x wider than the residual stream) is a more effective or more interpretable target for behavioral steering than the residual stream.

## Key Concepts

### The Expanded MLP Space
In a standard transformer MLP, the input from the residual stream (dimension `d_model`) is projected up to a wider intermediate space (dimension `d_mlp`, typically 4x `d_model`), passed through a nonlinearity (ReLU, GELU, or SwiGLU), then projected back down. This intermediate space is the "expanded MLP space." Its higher dimensionality means features may be less superimposed than in the residual stream, potentially making it easier to identify and manipulate individual concepts.

### Activation Steering
Adding a "steering vector" to model activations during inference to change behavior without retraining. Most work has focused on steering in the residual stream. The key question is whether steering in the MLP intermediate space offers advantages.

---

## Core Papers

### 1. Transcoders Find Interpretable LLM Feature Circuits
**Dunefsky, Chlenski, Nanda (2024)** — NeurIPS 2024. [arXiv:2406.11944]

**Key contribution**: Introduces **transcoders** — wide, sparsely-activating approximations of MLP sublayers that map MLP input directly to MLP output. Unlike SAEs (which reconstruct activations), transcoders approximate the MLP's *computation*.

**Architecture**: `TC(x) = W_dec * ReLU(W_enc * x + b_enc) + b_dec`, where `d_features >> d_model`. This creates a sparse decomposition of MLP behavior in an expanded feature space.

**Critical finding for our research**: Transcoders enable **input-invariant** circuit analysis. The attribution between transcoder feature pairs factorizes cleanly into: `z_TC(x) * (f_dec · f_enc')` — an input-dependent activation times an input-invariant weight-based connection. SAEs cannot provide this because the MLP nonlinearity makes attributions always input-dependent.

**Comparison with SAEs**: Transcoders achieve equal or better sparsity-accuracy tradeoff compared to SAEs across GPT-2 Small, Pythia-410M, and Pythia-1.4B. The gap widens on larger models. Blind interpretability tests show 41/50 transcoder features are interpretable vs 38/50 for SAEs.

**Relevance**: Transcoders directly operate in the expanded MLP space and show it can be decomposed into sparse, interpretable features. This is foundational for understanding whether this space is suitable for steering.

**Code**: https://github.com/jacobdunefsky/transcoder_circuits

---

### 2. Decomposing MLP Activations into Interpretable Features via Semi-Nonnegative Matrix Factorization
**Shafran, Geiger, Geva (2025)** — [arXiv:2506.10920]

**Key contribution**: Uses **SNMF** to decompose MLP hidden activations (in the expanded space) directly, without training a neural network. Factorizes activation matrix `A ∈ R^{d_a × n}` into `Z ∈ R^{d_a × k}` (features as neuron combinations) and `Y ∈ R^{k × n}_≥0` (nonnegative coefficients mapping features to inputs).

**Critical findings for our research**:
- **SNMF features outperform SAEs AND supervised baselines (DiffMeans) on causal steering**, while performing comparably on concept detection. This is direct evidence that the MLP intermediate space is effective for steering.
- Features are **compositional**: recursive SNMF reveals hierarchical structure (e.g., individual weekdays → weekday/weekend → day of week).
- Semantically-related features **share core neurons** and are differentiated by exclusive neurons. Amplifying core neurons promotes all related concepts; amplifying exclusive neurons promotes one while suppressing others.
- Features are sparse linear combinations of co-activated neurons, making them directly interpretable.

**Models tested**: Llama 3.1-8B, Gemma 2 2B, GPT-2 Small.

**Relevance**: This is the most directly relevant paper — it demonstrates that steering in the expanded MLP space (via neuron-group features) outperforms residual stream methods. The hierarchical compositional structure also suggests the MLP space may encode information in a more structured way.

**Code**: https://github.com/ordavid-s/snmf-mlp-decomposition

---

### 3. Sparse Autoencoders Find Highly Interpretable Features in Language Models
**Cunningham, Ewart, Riggs, Huben, Sharkey (2023)** — [arXiv:2309.08600]

**Key contribution**: Foundational paper on training SAEs to decompose model activations into interpretable features. Shows that SAE features are more interpretable than individual neurons.

**Relevance**: Establishes the baseline approach (SAEs on residual stream) that MLP-space methods improve upon.

---

### 4. Activation Addition: Steering Language Models Without Optimization
**Turner et al. (2023)** — [arXiv:2308.10248]

**Key contribution**: Introduces "activation addition" (ActAdd) — adding a steering vector (computed as the mean activation difference between contrastive prompts) to the residual stream. Simple and training-free.

**Relevance**: Establishes the baseline for residual stream steering that MLP-space methods should be compared against.

---

### 5. Steering Language Model Refusal with Sparse Autoencoders
**(2024)** — [arXiv:2411.02886]

**Key contribution**: Uses SAE features to steer model refusal behavior, clamping specific SAE features to control whether models refuse harmful requests. Demonstrates feature-level steering via SAEs.

**Relevance**: Shows SAE-based steering in practice; our research would extend this to MLP-space features.

---

### 6. Analyzing the Generalization and Reliability of Steering Vectors
**Tan, Chanin, Lynch et al. (2024)** — [arXiv:2407.12404]

**Key findings**: Steering vectors have substantial limitations both in- and out-of-distribution. Steerability is highly variable across inputs. Spurious biases can contribute to steering effectiveness. Steering vectors are sometimes brittle to prompt changes.

**Relevance**: Establishes that residual-stream steering has reliability issues — our hypothesis is that MLP-space steering might be more reliable due to the space being more naturally decomposable.

**Code**: https://github.com/dtch1997/steering-bench

---

### 7. Language Model Circuits Are Sparse in the Neuron Basis
**(2026)** — [arXiv:2502.08148]

**Key finding**: MLP computations are sparse in the neuron basis — only a small fraction of neurons contribute meaningfully to any given computation. This supports the idea that the expanded MLP space is naturally sparse and decomposable.

**Relevance**: Direct evidence that the MLP intermediate space has natural sparsity structure that could be exploited for steering.

---

### 8. Representation Engineering: A Top-Down Approach to AI Transparency
**Zou et al. (2023)** — [arXiv:2310.01405]

**Key contribution**: Proposes representation engineering (RepE) — using linear probes and interventions on internal representations to understand and control model behavior. Works primarily in residual stream.

**Relevance**: Provides the broader framework for representation-level interventions; MLP-space steering extends this.

---

## Additional Relevant Papers

### Steering Methods
- **Programming Refusal with Conditional Activation Steering (CAST)** (Lee et al., 2024): Conditional steering based on input patterns in hidden states. Code: github.com/IBM/activation-steering
- **Multi-property Steering with Dynamic Activation Composition** (2024): Steers multiple properties simultaneously via composition.
- **Interpretable Steering with Feature-Guided Activation Additions** (2025): Uses SAE features to guide which directions to steer.
- **Scaling Laws for Activation Steering** (2025): Studies how steering effectiveness scales with model size.
- **Angular Steering** (2025): Rotation-based steering in activation space.
- **Beyond Linear Steering** (2025): Multi-attribute control moving beyond simple linear addition.
- **Steer2Edit** (2026): Component-level editing derived from steering insights.

### MLP Interpretability
- **Automatically Identifying Local and Global Circuits** (Marks et al., 2024): Uses transcoders/SAEs for circuit discovery through MLP layers.
- **The Knowledge Microscope** (2025): Features as better analytical lenses than neurons.
- **Mechanistic Interpretability for AI Safety** (Bereska & Gavves, 2024): Comprehensive review of the field.

### Linear Representation Hypothesis
- **Emergent Linear Representations in World Models** (Nanda et al., 2023): Shows linear board representations in Othello-GPT, controllable via vector arithmetic.
- **Extracting Latent Steering Vectors** (Subramani et al., 2022): Early work on finding steering vectors in LM hidden states.

---

## Common Methodologies

| Method | Space | Training Required | Key Advantage | Key Limitation |
|--------|-------|-------------------|---------------|----------------|
| ActAdd/CAA | Residual stream | No (contrast pairs) | Simple, fast | Input-dependent, unreliable OOD |
| SAE features | Residual stream | Yes (SAE training) | Interpretable features | Poor causal performance |
| SAE features | MLP output | Yes (SAE training) | Better grounded in MLP | Still in residual-stream space |
| Transcoders | MLP expanded space | Yes (transcoder training) | Input-invariant circuits | Approximation error |
| SNMF | MLP expanded space | No (matrix factorization) | Best steering, interpretable | Limited to k<500 features tested |
| DiffMeans | Residual/MLP output | No (supervised) | Strong baseline | Needs labeled data, noisy |

## Standard Baselines
- **ActAdd/CAA** (Turner et al., 2023; Rimsky et al., 2024): Mean activation difference steering
- **DiffMeans** (Marks & Tegmark, 2024): Supervised difference-in-means
- **SAE feature clamping** (Templeton et al., 2024): Steering via SAE feature activation

## Evaluation Metrics
- **Concept detection score**: Log-ratio of feature activation on concept-related vs neutral inputs
- **Concept steering score**: GPT-4 rated alignment of steered generations with target concept (0-2)
- **Fluency score**: GPT-4 rated coherence of steered generations (0-2)
- **KL divergence**: Measures perturbation magnitude during steering
- **Cross-entropy loss increase**: Measures faithfulness of SAE/transcoder approximation
- **L0 norm**: Sparsity of learned decomposition

## Datasets in the Literature
- **OpenWebText**: Standard corpus for SAE/transcoder training (GPT-2 family)
- **The Pile**: Evaluation corpus for Pythia models
- **TruthfulQA**: Evaluating factuality under steering
- **SST-2**: Sentiment steering evaluation
- **Concept-specific datasets**: Weekdays, languages, emotions (custom generated)

## Gaps and Opportunities

1. **No systematic comparison of steering in MLP expanded space vs residual stream**: The SNMF paper shows MLP features outperform SAEs at steering, but doesn't directly compare MLP-space ActAdd vs residual-stream ActAdd.

2. **Linearity of MLP expanded space is understudied**: The hypothesis that the expanded space is "more linear" (due to ReLU creating a naturally sparse basis) hasn't been formally tested.

3. **Transcoder features haven't been used for steering**: Transcoders were developed for circuit analysis, but their features could be directly used as steering vectors.

4. **Scale of MLP decomposition is limited**: SNMF tested with k<500 features. Transcoders use larger dictionaries but haven't been evaluated on steering tasks.

5. **No comparison of steering reliability across spaces**: Steering vectors in residual stream are unreliable (Tan et al., 2024). Is MLP-space steering more reliable?

6. **Interaction between MLP steering and attention**: How does steering in MLP space interact with the attention mechanism?

## Recommendations for Our Experiment

### Recommended Datasets
- **OpenWebText** (1000-sample for prototyping, full for training)
- **SST-2** (sentiment steering evaluation)
- **TruthfulQA** (factuality steering evaluation)
- Custom concept pairs (weekdays, languages, etc.) following SNMF paper methodology

### Recommended Baselines
- **ActAdd/CAA** in residual stream (simplest baseline)
- **DiffMeans** in residual stream (strong supervised baseline)
- **SAE feature steering** in residual stream
- **SNMF steering** in MLP space (current SOTA for MLP-space steering)

### Recommended Models
- **GPT-2 Small** (most studied, pre-trained SAEs/transcoders available)
- **Pythia-410M** (pre-trained transcoders available)
- **Gemma 2 2B** or **Llama 3.1 8B** (for showing scaling)

### Recommended Metrics
- Concept steering score + fluency (following SNMF paper)
- Concept detection score
- Steering reliability across different inputs (following steering-bench)
- Cross-entropy loss increase (measuring intervention distortion)

### Methodological Considerations
- Use TransformerLens for all model interventions (hook-based)
- Use SAELens for loading pre-trained SAEs
- Compare interventions at the same KL divergence budget for fair comparison
- Test both MLP-input and MLP-intermediate (post-activation) intervention points
