# Datasets

Data files are NOT committed to git due to size. Follow the download instructions below.

## Dataset 1: TruthfulQA

### Overview
- **Source**: HuggingFace `truthful_qa` (generation split)
- **Size**: 817 examples
- **Format**: HuggingFace Dataset
- **Task**: Evaluating truthfulness of model outputs; useful for measuring steering effects on factuality
- **License**: Apache 2.0

### Download Instructions
```python
from datasets import load_dataset
dataset = load_dataset("truthful_qa", "generation", split="validation")
dataset.save_to_disk("datasets/truthfulqa")
```

### Notes
- Used to evaluate whether steering interventions affect model truthfulness
- Contains questions with best/correct answers and incorrect answers

## Dataset 2: OpenWebText Sample

### Overview
- **Source**: HuggingFace `Skylion007/openwebtext`
- **Size**: 1,000 examples (sample from full 8M+ document corpus)
- **Format**: JSON
- **Task**: Training SAEs, transcoders, and SNMF decompositions; general text for activation collection
- **License**: MIT

### Download Instructions
```python
from datasets import load_dataset
ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
samples = [ex for i, ex in enumerate(ds) if i < 1000]
```

For full corpus (needed for SAE/transcoder training):
```python
ds = load_dataset("Skylion007/openwebtext", split="train")
```

### Notes
- The full dataset is ~12GB and is the standard corpus for training SAEs on GPT-2
- The 1000-example sample is sufficient for quick experiments and prototyping
- Pre-trained SAEs and transcoders are available (see code/README.md)

## Dataset 3: SST-2 (Stanford Sentiment Treebank)

### Overview
- **Source**: HuggingFace `stanfordnlp/sst2`
- **Size**: 872 validation examples
- **Format**: HuggingFace Dataset
- **Task**: Binary sentiment classification; concept steering evaluation
- **Splits**: train (67,349), validation (872), test (1,821)
- **License**: CC BY-SA 4.0

### Download Instructions
```python
from datasets import load_dataset
dataset = load_dataset("stanfordnlp/sst2", split="validation")
dataset.save_to_disk("datasets/sst2")
```

### Notes
- Useful for evaluating sentiment steering in MLP space
- Provides clear positive/negative contrast pairs for activation difference computation

## Additional Datasets (Not Downloaded - Available on Demand)

### Anthropic Model-Written Evals
- **Source**: https://github.com/anthropics/evals
- **Purpose**: Contrast pairs for contrastive activation steering (CAA)
- **Note**: Contains sycophancy, power-seeking, and other behavioral evaluation sets

### The Pile (Validation)
- **Source**: HuggingFace `EleutherAI/pile`
- **Purpose**: Standard evaluation corpus for Pythia models
- **Note**: Used by the transcoders paper for evaluation

### Pre-trained Models (Available via HuggingFace)
The experiments in this research use these models:
- `openai-community/gpt2` (GPT-2 Small, 124M params)
- `EleutherAI/pythia-410m` (Pythia 410M)
- `EleutherAI/pythia-1.4b` (Pythia 1.4B)
- `google/gemma-2-2b` (Gemma 2 2B)
- `meta-llama/Llama-3.1-8B` (Llama 3.1 8B)

These are loaded on-demand via TransformerLens/HuggingFace.
