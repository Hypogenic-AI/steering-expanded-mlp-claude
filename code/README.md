# External Code Repositories for MLP Steering Research

This directory contains cloned repositories (shallow, `--depth 1`) relevant to
steering in expanded MLP spaces. Each repo addresses a different aspect of the
research pipeline: decomposing MLP layers into interpretable features, training
sparse autoencoders, building steering vectors, and evaluating their effects.

---

## 1. transcoder_circuits

- **Source:** https://github.com/jacobdunefsky/transcoder_circuits
- **Purpose:** Reverse-engineer LLM circuits using *transcoders* -- sparse
  autoencoders that decompose MLP sublayers into interpretable input-to-output
  feature mappings (as opposed to standard SAEs that only decompose a single
  activation space).
- **Key entry points:**
  - `setup.sh` -- install deps and download pre-trained GPT-2 transcoder weights
    from HuggingFace.
  - `walkthrough.ipynb` -- end-to-end demo of loading transcoders and tracing
    circuits through them.
  - `train_transcoder.py` -- script for training a new transcoder on a model.
  - `transcoder_circuits/` -- library code for circuit analysis (attribution,
    feature visualization).
  - `sae_training/` -- transcoder/SAE training loop (forked from an older
    SAELens).
- **Dependencies:** `torch 2.2`, `transformer_lens 1.11`, `einops`, `wandb`,
  `datasets`, `huggingface-hub`.
- **Relevance to MLP steering:** Transcoders give a *causal* decomposition of
  MLP computation (input features -> output features). This makes them a natural
  basis for targeted steering: activate or suppress specific transcoder output
  features to steer the MLP's contribution to the residual stream.

---

## 2. snmf-mlp-decomposition

- **Source:** https://github.com/ordavid-s/snmf-mlp-decomposition
- **Purpose:** Decompose MLP activations into interpretable concept-level
  features using Semi-Nonnegative Matrix Factorization (SNMF). Provides an
  alternative to SAEs that discovers concept-aligned directions in MLP space.
- **Key entry points:**
  - `snmf_tutorial.ipynb` -- train SNMF end-to-end: collect activations,
    factorize, inspect factors.
  - `hierarchial_nmf_tutorial.ipynb` -- recursive SNMF producing concept trees.
  - `experiments/` -- shell scripts for concept detection and concept steering
    experiments (`run_snmf_steering.sh`, `run_sae_steering.sh`,
    `run_diffmean_steering.sh`).
  - `experiments/train/` -- training code.
- **Dependencies:** `torch 2.8`, `transformer-lens 2.16`, `sae-lens 6.12`,
  `openai` (for LLM-based evaluation), `scipy`, `statsmodels`, `nltk`.
- **Relevance to MLP steering:** SNMF factors are concept-aligned directions in
  MLP activation space that can be directly used as steering vectors. The repo
  includes head-to-head comparisons of SNMF vs SAE vs DiffMeans steering,
  making it a ready-made evaluation framework for new decomposition methods.

---

## 3. SAELens

- **Source:** https://github.com/jbloomaus/SAELens
- **Purpose:** The standard library for training, loading, and analysing Sparse
  Autoencoders on language model internals. Supports many pre-trained SAEs
  (GPT-2, Gemma, Llama, etc.) via a model registry.
- **Key entry points:**
  - `sae_lens/training/` -- SAE training loop and configs.
  - `sae_lens/saes/` -- SAE model definitions.
  - `sae_lens/cache_activations_runner.py` -- cache model activations to disk.
  - `sae_lens/loading/` -- load pre-trained SAEs from HuggingFace / registry.
  - `sae_lens/analysis/` -- downstream analysis tools.
  - `sae_lens/evals.py` -- evaluation metrics for SAEs.
  - `tutorials/` -- Colab-ready notebooks (training, loading, logit lens).
- **Dependencies:** Managed via Poetry. Core: `torch`, `transformer-lens`,
  `transformers`, `datasets`, `wandb`, `safetensors`.
- **Install:** `pip install sae-lens` or `poetry install` from source.
- **Relevance to MLP steering:** SAELens provides the canonical SAE features
  that serve as a baseline decomposition of MLP (and other) activations.
  Steering can be implemented by clamping or scaling individual SAE feature
  activations during a forward pass. Pre-trained SAE weights avoid the cost of
  training from scratch.

---

## 4. steering-bench

- **Source:** https://github.com/dtch1997/steering-bench
- **Purpose:** Benchmark and evaluation framework for steering vectors, from the
  paper "Analyzing the Generalization and Reliability of Steering Vectors"
  (arXiv 2407.12404). Measures how well steering vectors generalize across
  prompt variations.
- **Key entry points:**
  - `steering_bench/core/pipeline.py` -- `Pipeline` wrapper around a
    (possibly-steered) model.
  - `steering_bench/core/hook.py` -- `SteeringHook` for applying steering
    vectors via the `steering-vectors` library.
  - `steering_bench/core/format.py` -- prompt formatting / scaffolding.
  - `steering_bench/core/metric.py` -- steerability metrics from the paper.
  - `experiments/layer_sweep/` -- sweep over layers to find best steering layer.
  - `experiments/steering_generalization/` -- evaluate generalization across
    prompt templates.
- **Dependencies:** `torch >=2.5`, `transformers >=4.46`, `steering-vectors
  >=0.12`, `datasets`, `bitsandbytes`, `accelerate`. Requires Python 3.12.
- **Install:** `pip install -e .`
- **Relevance to MLP steering:** This repo provides the evaluation harness for
  measuring whether steering interventions (including MLP-space interventions)
  actually change model behavior in a reliable and generalizable way. It can be
  adapted to benchmark expanded-MLP steering against standard residual-stream
  steering.

---

## 5. TransformerLens

- **Source:** https://github.com/TransformerLensOrg/TransformerLens
- **Purpose:** The foundational mechanistic interpretability library. Loads 50+
  open-source language models and exposes every internal activation via hooks.
  Supports activation caching, patching, and ablation.
- **Key entry points:**
  - `transformer_lens/HookedTransformer.py` -- main model class with hook
    points at every sublayer.
  - `transformer_lens/hook_points.py` -- the hook-point system for reading and
    modifying activations.
  - `transformer_lens/components/` -- individual transformer components (MLP,
    attention, layernorm, etc.).
  - `transformer_lens/ActivationCache.py` -- cache and retrieve activations.
  - `transformer_lens/patching.py` -- activation patching utilities.
  - `transformer_lens/loading_from_pretrained.py` -- model weight loading.
- **Dependencies:** `torch`, `einops`, `transformers`, `datasets`, `wandb`.
- **Install:** `pip install transformer_lens`
- **Relevance to MLP steering:** TransformerLens is the runtime substrate.
  Its hook system is what allows us to intercept MLP activations mid-forward-
  pass, project them into an expanded feature space (via SAE/transcoder/SNMF),
  manipulate specific features, and project back. All the other repos in this
  directory depend on or are compatible with TransformerLens.

---

## How the repos relate

```
TransformerLens          (runtime: load models, hook activations)
    |
    +-- SAELens          (train/load SAEs on hooked activations)
    +-- transcoder_circuits  (train/use transcoders on MLP sublayers)
    +-- snmf-mlp-decomposition (SNMF decomposition of MLP activations)
    |
    +-- steering-bench   (evaluate steering interventions end-to-end)
```

**Workflow for MLP steering research:**

1. Use **TransformerLens** to load a model and access MLP activations.
2. Use **SAELens**, **transcoder_circuits**, or **snmf-mlp-decomposition** to
   decompose MLP activations into interpretable features.
3. Construct steering vectors in the expanded feature space.
4. Apply the steering vectors using TransformerLens hooks.
5. Evaluate with **steering-bench** metrics and prompt-generalization tests.
