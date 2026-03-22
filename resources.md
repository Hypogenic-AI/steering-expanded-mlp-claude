# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project on steering in the expanded MLP space. The research investigates whether intervening in the intermediate (wider) space within MLPs is more effective for steering model behavior than intervening in the residual stream.

## Papers
Total papers downloaded: 20

| # | Title | Authors | Year | File | Key Info |
|---|-------|---------|------|------|----------|
| 1 | Transcoders Find Interpretable LLM Feature Circuits | Dunefsky et al. | 2024 | papers/2406.11944_*.pdf | Core: MLP-space decomposition via transcoders |
| 2 | Decomposing MLP Activations via SNMF | Shafran et al. | 2025 | papers/2506.10920.pdf | Core: SNMF outperforms SAEs at MLP-space steering |
| 3 | SAEs Find Highly Interpretable Features | Cunningham et al. | 2023 | papers/2309.08600_*.pdf | Foundational SAE paper |
| 4 | Activation Addition (ActAdd) | Turner et al. | 2023 | papers/2308.10248_*.pdf | Baseline residual-stream steering |
| 5 | Analyzing Steering Vector Reliability | Tan et al. | 2024 | papers/2407.12404_*.pdf | Steering limitations in residual stream |
| 6 | Representation Engineering | Zou et al. | 2023 | papers/2310.01405_*.pdf | RepE framework |
| 7 | Extracting Latent Steering Vectors | Subramani et al. | 2022 | papers/2205.05124_*.pdf | Early steering vector work |
| 8 | LM Circuits Sparse in Neuron Basis | — | 2026 | papers/2502.08148_*.pdf | MLP sparsity evidence |
| 9 | Emergent Linear Representations | Nanda et al. | 2023 | papers/2310.15154_*.pdf | Linear probes and control |
| 10 | Auto Circuit Discovery | Marks et al. | 2024 | papers/2407.14252_*.pdf | Circuit finding with transcoders |
| 11 | CAST (Conditional Activation Steering) | Lee et al. | 2024 | papers/2410.19454_*.pdf | Conditional steering |
| 12 | Improving Instruction-Following via Steering | — | 2024 | papers/2410.12877_*.pdf | Practical steering application |
| 13 | Feature-Guided Activation Additions | — | 2025 | papers/2503.01822_*.pdf | SAE-guided steering |
| 14 | Multi-property Steering | — | 2024 | papers/2406.12188_*.pdf | Multi-attribute control |
| 15 | Mechanistic Interp for AI Safety | Bereska et al. | 2024 | papers/2404.14082_*.pdf | Comprehensive review |
| 16 | Steer2Edit | — | 2026 | papers/2502.16762_*.pdf | Component-level editing |
| 17 | Knowledge Microscope | — | 2025 | papers/2502.00235_*.pdf | Features vs neurons |
| 18 | Transporting Activations | — | 2024 | papers/2410.23054_*.pdf | Optimal transport steering |
| 19 | Scaling Laws for Steering | — | 2025 | papers/2503.03280_*.pdf | Steering scaling behavior |
| 20 | Steering with SAE Refusal Features | — | 2024 | papers/2411.02886_*.pdf | SAE feature steering |

See papers/README.md for detailed descriptions.

## Datasets
Total datasets downloaded: 3

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| TruthfulQA | HuggingFace | 817 examples | Factuality evaluation | datasets/truthfulqa/ | Measuring steering effect on truthfulness |
| OpenWebText Sample | HuggingFace | 1,000 examples | SAE/transcoder training | datasets/openwebtext_sample/ | Sample for prototyping; full corpus available on-demand |
| SST-2 | HuggingFace | 872 val examples | Sentiment steering | datasets/sst2/ | Binary sentiment for concept steering evaluation |

See datasets/README.md for download instructions and detailed descriptions.

## Code Repositories
Total repositories cloned: 5

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| transcoder_circuits | github.com/jacobdunefsky/transcoder_circuits | Transcoder training & circuit analysis | code/transcoder_circuits/ | Core tool for MLP-space decomposition |
| snmf-mlp-decomposition | github.com/ordavid-s/snmf-mlp-decomposition | SNMF-based MLP feature finding | code/snmf-mlp-decomposition/ | Best MLP-space steering method |
| SAELens | github.com/jbloomaus/SAELens | SAE training & pre-trained models | code/SAELens/ | Standard SAE library, includes pre-trained models |
| steering-bench | github.com/dtch1997/steering-bench | Steering vector evaluation framework | code/steering-bench/ | Evaluation benchmark for steering reliability |
| TransformerLens | github.com/TransformerLensOrg/TransformerLens | Hooked model inference for mech interp | code/TransformerLens/ | Foundation library for all interventions |

See code/README.md for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
1. Used paper-finder service with diligent mode for two queries: (a) "steering model behavior MLP intermediate space activation engineering" and (b) "MLP neurons features superposition sparse autoencoder transformer interpretability"
2. Combined results (178 + 175 papers), deduplicated, and ranked by keyword relevance to MLP steering
3. Downloaded top 20 papers spanning core MLP-space methods, steering baselines, and interpretability foundations
4. Deep-read 2 core papers (Transcoders, SNMF) that directly work in the expanded MLP space

### Selection Criteria
- Papers directly about MLP intermediate space (transcoders, SNMF) — highest priority
- Papers about activation steering methods — needed as baselines
- Papers about SAE features and interpretability — context for understanding decomposition approaches
- Papers about steering reliability — motivates why MLP-space steering might be better

### Challenges Encountered
- Several arXiv IDs returned different papers than expected (likely ID reassignment over time)
- Some steering-specific datasets (Anthropic CAA prompts) are not publicly available on HuggingFace
- Pre-trained transcoders require specific model/layer combinations to use

### Gaps and Workarounds
- **Missing CAA contrast pairs**: Can be generated from templates following the ActAdd/RepE methodology
- **No direct MLP-space vs residual-stream steering comparison exists**: This is the core gap our research fills
- **Pre-trained transcoders only available for GPT-2/Pythia**: Limits initial experiments to these models

## Recommendations for Experiment Design

### 1. Primary Dataset(s)
- **OpenWebText** for activation collection and SAE/transcoder training
- **SST-2** and custom concept pairs for steering evaluation
- **TruthfulQA** for measuring steering side effects

### 2. Baseline Methods
- **ActAdd/CAA in residual stream**: Simplest steering baseline
- **DiffMeans in residual stream**: Strong supervised baseline
- **SAE feature steering in residual stream**: Feature-level baseline
- **SNMF in MLP space**: Current best MLP-space method

### 3. Evaluation Metrics
- Concept steering score (GPT-4 judged, 0-2)
- Fluency score (GPT-4 judged, 0-2)
- Concept detection score (log-ratio)
- KL divergence budget (for fair comparison)
- Steering reliability across inputs (variance of effect)

### 4. Code to Adapt/Reuse
- **TransformerLens**: All model loading and hook-based interventions
- **SAELens**: Loading pre-trained SAEs for baseline comparisons
- **snmf-mlp-decomposition**: SNMF decomposition and steering evaluation pipeline
- **transcoder_circuits**: Transcoder training and feature extraction
- **steering-bench**: Evaluation framework for steering reliability

### 5. Experimental Plan Outline
1. **Phase 1**: Replicate SNMF steering results on GPT-2 Small to validate setup
2. **Phase 2**: Implement direct MLP-space ActAdd (add steering vectors to MLP intermediate activations)
3. **Phase 3**: Compare steering effectiveness: residual stream vs MLP input vs MLP intermediate vs MLP output
4. **Phase 4**: Analyze linearity and sparsity of the MLP intermediate space
5. **Phase 5**: Test whether MLP-space steering is more reliable across inputs (using steering-bench framework)
