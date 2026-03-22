"""
Steering in the Expanded MLP Space: Systematic Comparison
=========================================================
Compares steering effectiveness in the residual stream vs MLP intermediate space
in GPT-2 Small. Tests whether the 4x wider MLP space is easier or harder to steer.
"""

import json
import os
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

# Set seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda:0"
RESULTS_DIR = Path("/workspaces/steering-expanded-mlp-claude/results")
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ─── Contrastive prompt pairs for sentiment steering ───────────────────────────
POSITIVE_PROMPTS = [
    "I absolutely love this beautiful day, everything feels wonderful and",
    "The movie was fantastic, one of the best experiences I've ever had and",
    "This restaurant serves the most delicious food I've ever tasted and",
    "What an incredible achievement, I'm so proud and happy that",
    "The sunset was breathtaking, filling me with joy and",
    "I'm thrilled about the amazing progress we've made on",
    "The concert was absolutely magnificent, the music was",
    "She gave the most inspiring speech I've ever heard about",
    "The garden was gorgeous, with beautiful flowers blooming everywhere and",
    "I had the most wonderful time with my friends, we laughed and",
    "This book is a masterpiece, every chapter is captivating and",
    "The kindness of strangers never ceases to amaze me, today someone",
    "I feel incredibly grateful for all the wonderful things in",
    "The team's performance was outstanding, they exceeded every expectation and",
    "What a delightful surprise to find such a lovely place where",
    "The children were laughing and playing, their happiness was contagious and",
    "I'm overjoyed to announce that we've achieved our goal of",
    "The weather is perfect today, sunny and warm with a gentle",
    "This is the happiest I've been in years, everything is going",
    "The puppy was adorable, wagging its tail excitedly as",
    "I received the most wonderful news today about",
    "The view from the mountain top was absolutely spectacular and",
    "Everyone at the party was having an incredible time, dancing and",
    "The project turned out better than we ever imagined, with",
    "I love spending time in nature, it brings me such peace and",
]

NEGATIVE_PROMPTS = [
    "I absolutely hate this terrible day, everything feels awful and",
    "The movie was horrible, one of the worst experiences I've ever had and",
    "This restaurant serves the most disgusting food I've ever tasted and",
    "What a terrible failure, I'm so disappointed and upset that",
    "The weather was miserable, making me feel gloomy and",
    "I'm frustrated about the lack of progress we've made on",
    "The concert was absolutely dreadful, the music was",
    "She gave the most depressing speech I've ever heard about",
    "The garden was ugly, with dead plants rotting everywhere and",
    "I had the most terrible time with my enemies, we argued and",
    "This book is garbage, every chapter is boring and",
    "The cruelty of people never ceases to horrify me, today someone",
    "I feel incredibly resentful about all the terrible things in",
    "The team's performance was abysmal, they failed every expectation and",
    "What a horrible shock to find such a dreadful place where",
    "The children were crying and fighting, their misery was contagious and",
    "I'm devastated to announce that we've failed to achieve our goal of",
    "The weather is terrible today, cold and rainy with a harsh",
    "This is the worst I've felt in years, everything is going",
    "The dog was aggressive, baring its teeth menacingly as",
    "I received the most devastating news today about",
    "The view from the window was absolutely depressing and",
    "Everyone at the event was having a miserable time, complaining and",
    "The project turned out worse than we ever feared, with",
    "I hate being stuck indoors, it fills me with dread and",
]

# Neutral prompts for steering evaluation
NEUTRAL_PROMPTS = [
    "The weather today is",
    "I went to the store and",
    "The meeting started at noon and",
    "She opened the book and",
    "The cat sat on the",
    "Yesterday I decided to",
    "The train arrived at the station and",
    "He looked at the screen and",
    "The city was known for its",
    "They walked down the street and",
    "The report mentioned that",
    "After breakfast, she went to",
    "The new policy will affect",
    "He picked up the phone and",
    "The building on the corner was",
    "She thought about the situation and",
    "The river flowed through the valley and",
    "Last week, the team discussed",
    "The package arrived this morning and",
    "On Monday, we plan to",
]


def load_model():
    """Load GPT-2 Small with TransformerLens."""
    from transformer_lens import HookedTransformer
    print("Loading GPT-2 Small...")
    model = HookedTransformer.from_pretrained("gpt2", device=DEVICE)
    print(f"  d_model={model.cfg.d_model}, d_mlp={model.cfg.d_mlp}, n_layers={model.cfg.n_layers}")
    return model


def collect_activations(model, prompts, hook_points, batch_size=5):
    """
    Collect activations at specified hook points for given prompts.
    Returns dict mapping hook_point -> tensor of shape (n_prompts, d_hook).
    Uses the activation at the last token position.
    """
    all_acts = {hp: [] for hp in hook_points}

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        tokens = model.to_tokens(batch, prepend_bos=True)

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_points)

        for hp in hook_points:
            # Get last-token activation for each prompt
            acts = cache[hp][:, -1, :]  # (batch, d_hook)
            all_acts[hp].append(acts.cpu())

        del cache
        torch.cuda.empty_cache()

    return {hp: torch.cat(acts, dim=0) for hp, acts in all_acts.items()}


def compute_steering_vectors(pos_acts, neg_acts):
    """Compute mean-difference steering vector: mean(pos) - mean(neg)."""
    return pos_acts.mean(dim=0) - neg_acts.mean(dim=0)


# ─── Experiment 1: Steering Effectiveness ──────────────────────────────────────

def make_steering_hook(direction, alpha):
    """Create a hook that adds alpha * normalized_direction to activations."""
    direction_norm = direction / (direction.norm() + 1e-8)
    direction_norm = direction_norm.to(DEVICE)

    def hook_fn(value, hook):
        value[:, :, :] = value + alpha * direction_norm.unsqueeze(0).unsqueeze(0)
        return value
    return hook_fn


def compute_kl_divergence(model, prompt, hook_point, direction, alpha):
    """Compute KL divergence between baseline and steered logits at last token."""
    tokens = model.to_tokens(prompt, prepend_bos=True)

    with torch.no_grad():
        baseline_logits = model(tokens)[:, -1, :]
        baseline_probs = F.softmax(baseline_logits, dim=-1)

        hook_fn = make_steering_hook(direction, alpha)
        steered_logits = model.run_with_hooks(
            tokens, fwd_hooks=[(hook_point, hook_fn)]
        )[:, -1, :]
        steered_probs = F.softmax(steered_logits, dim=-1)

    kl = F.kl_div(steered_probs.log(), baseline_probs, reduction='batchmean').item()
    return kl


def find_alpha_for_kl(model, prompt, hook_point, direction, target_kl, tol=0.05, max_iter=30):
    """Binary search for alpha that produces target KL divergence."""
    low, high = 0.0, 500.0

    for _ in range(max_iter):
        mid = (low + high) / 2.0
        kl = compute_kl_divergence(model, prompt, hook_point, direction, mid)
        if abs(kl - target_kl) < tol:
            return mid
        if kl < target_kl:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def generate_steered_text(model, prompt, hook_point, direction, alpha, max_tokens=50):
    """Generate text with steering applied."""
    tokens = model.to_tokens(prompt, prepend_bos=True)
    hook_fn = make_steering_hook(direction, alpha)

    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model.run_with_hooks(
                tokens, fwd_hooks=[(hook_point, hook_fn)]
            )
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)

            # Stop at EOS
            if next_token.item() == model.tokenizer.eos_token_id:
                break

    return model.to_string(tokens[0])


def simple_sentiment_score(text):
    """
    Simple keyword-based sentiment scorer.
    Returns score in [-1, 1] range. Positive = positive sentiment.
    """
    positive_words = {
        'good', 'great', 'excellent', 'wonderful', 'amazing', 'fantastic',
        'beautiful', 'love', 'happy', 'joy', 'delightful', 'perfect',
        'brilliant', 'outstanding', 'superb', 'magnificent', 'lovely',
        'pleased', 'glad', 'cheerful', 'exciting', 'terrific', 'awesome',
        'best', 'incredible', 'marvelous', 'splendid', 'fortunate', 'grateful',
        'enjoy', 'enjoyed', 'enjoying', 'smile', 'laugh', 'fun', 'pleasant',
        'nice', 'kind', 'warm', 'bright', 'success', 'successful', 'triumph',
    }
    negative_words = {
        'bad', 'terrible', 'horrible', 'awful', 'disgusting', 'hate',
        'sad', 'angry', 'depressing', 'miserable', 'dreadful', 'worst',
        'ugly', 'painful', 'disappointing', 'frustrating', 'annoying',
        'unhappy', 'gloomy', 'boring', 'fail', 'failure', 'poor',
        'nasty', 'cruel', 'violence', 'violent', 'fear', 'fearful',
        'dislike', 'upset', 'devastated', 'tragic', 'suffer', 'suffering',
        'death', 'die', 'kill', 'destroy', 'ruin', 'damage', 'harm',
        'wrong', 'mistake', 'error', 'problem', 'trouble', 'crisis',
    }
    words = text.lower().split()
    pos_count = sum(1 for w in words if w.strip('.,!?;:') in positive_words)
    neg_count = sum(1 for w in words if w.strip('.,!?;:') in negative_words)
    total = pos_count + neg_count
    if total == 0:
        return 0.0
    return (pos_count - neg_count) / total


def compute_perplexity(model, text, max_len=100):
    """Compute perplexity of text under the model."""
    tokens = model.to_tokens(text, prepend_bos=True)[:, :max_len]
    with torch.no_grad():
        logits = model(tokens)
    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :]
    shift_labels = tokens[:, 1:]
    loss = F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)),
                           shift_labels.reshape(-1))
    return torch.exp(loss).item()


def run_steering_experiment(model, hook_points, steering_vectors, target_kls=[0.5, 1.0, 2.0, 5.0]):
    """
    Run steering experiment: for each hook point and KL budget, generate steered text
    and evaluate sentiment and fluency.
    """
    print("\n=== Experiment 1: Steering Effectiveness ===")
    results = {}

    for hp_name, hp in hook_points.items():
        print(f"\n--- {hp_name} ---")
        sv = steering_vectors[hp]
        results[hp_name] = {"kl_targets": [], "sentiment_scores": [], "perplexities": [],
                            "actual_kls": [], "alphas": [], "generations": []}

        for target_kl in target_kls:
            print(f"  KL target: {target_kl}")
            sentiments = []
            perplexities = []
            actual_kls = []
            alphas = []
            generations = []

            for prompt in tqdm(NEUTRAL_PROMPTS[:10], desc=f"    Generating"):
                # Find alpha for this KL budget
                alpha = find_alpha_for_kl(model, prompt, hp, sv, target_kl)
                actual_kl = compute_kl_divergence(model, prompt, hp, sv, alpha)

                # Generate steered text
                text = generate_steered_text(model, prompt, hp, sv, alpha, max_tokens=40)

                # Evaluate
                sentiment = simple_sentiment_score(text)
                ppl = compute_perplexity(model, text)

                sentiments.append(sentiment)
                perplexities.append(ppl)
                actual_kls.append(actual_kl)
                alphas.append(alpha)
                generations.append(text)

            results[hp_name]["kl_targets"].append(target_kl)
            results[hp_name]["sentiment_scores"].append(sentiments)
            results[hp_name]["perplexities"].append(perplexities)
            results[hp_name]["actual_kls"].append(actual_kls)
            results[hp_name]["alphas"].append(alphas)
            results[hp_name]["generations"].append(generations)

            mean_sent = np.mean(sentiments)
            mean_ppl = np.mean(perplexities)
            print(f"    Sentiment: {mean_sent:.3f} ± {np.std(sentiments):.3f}, PPL: {mean_ppl:.1f}")

    return results


# ─── Experiment 2: Linearity Analysis ──────────────────────────────────────────

def run_linearity_experiment(pos_acts_dict, neg_acts_dict, hook_points):
    """Train linear probes in each space and compare accuracy."""
    print("\n=== Experiment 2: Linearity (Linear Probe Accuracy) ===")
    results = {}

    for hp_name, hp in hook_points.items():
        pos = pos_acts_dict[hp].numpy()
        neg = neg_acts_dict[hp].numpy()

        X = np.concatenate([pos, neg], axis=0)
        y = np.array([1]*len(pos) + [0]*len(neg))

        clf = LogisticRegression(max_iter=1000, random_state=SEED)
        scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')

        results[hp_name] = {
            "mean_accuracy": float(scores.mean()),
            "std_accuracy": float(scores.std()),
            "dim": int(X.shape[1]),
        }
        print(f"  {hp_name}: accuracy={scores.mean():.4f} ± {scores.std():.4f} (dim={X.shape[1]})")

    return results


# ─── Experiment 3: Sparsity Analysis ──────────────────────────────────────────

def compute_sparsity_metrics(activations):
    """Compute sparsity metrics for activations tensor (n_samples, d)."""
    acts = activations.numpy()

    # L0: fraction of near-zero activations (|x| < threshold)
    threshold = 0.01 * np.abs(acts).max()
    l0_fraction = (np.abs(acts) < threshold).mean()

    # Gini coefficient (higher = sparser)
    abs_acts = np.abs(acts).mean(axis=0)  # mean across samples
    sorted_acts = np.sort(abs_acts)
    n = len(sorted_acts)
    index = np.arange(1, n + 1)
    gini = (2 * np.sum(index * sorted_acts) / (n * np.sum(sorted_acts))) - (n + 1) / n

    # Fraction of neurons that are exactly zero (for ReLU-activated spaces)
    exact_zero_fraction = (acts == 0).mean()

    # Kurtosis (higher = more peaked/sparse)
    kurtosis = float(np.mean(stats.kurtosis(acts, axis=0)))

    # Top-k concentration: what fraction of total activation is in top 10% of dimensions
    abs_mean = np.abs(acts).mean(axis=0)
    sorted_dims = np.sort(abs_mean)[::-1]
    top_10_pct = int(len(sorted_dims) * 0.1)
    top_10_concentration = sorted_dims[:top_10_pct].sum() / sorted_dims.sum()

    return {
        "l0_fraction": float(l0_fraction),
        "gini_coefficient": float(gini),
        "exact_zero_fraction": float(exact_zero_fraction),
        "kurtosis": float(kurtosis),
        "top_10pct_concentration": float(top_10_concentration),
        "dim": int(acts.shape[1]),
    }


def run_sparsity_experiment(pos_acts_dict, neg_acts_dict, hook_points):
    """Analyze sparsity in each activation space."""
    print("\n=== Experiment 3: Sparsity Analysis ===")
    results = {}

    for hp_name, hp in hook_points.items():
        all_acts = torch.cat([pos_acts_dict[hp], neg_acts_dict[hp]], dim=0)
        metrics = compute_sparsity_metrics(all_acts)
        results[hp_name] = metrics
        print(f"  {hp_name} (dim={metrics['dim']}):")
        print(f"    L0 fraction (near-zero): {metrics['l0_fraction']:.4f}")
        print(f"    Exact zero fraction: {metrics['exact_zero_fraction']:.4f}")
        print(f"    Gini coefficient: {metrics['gini_coefficient']:.4f}")
        print(f"    Kurtosis: {metrics['kurtosis']:.1f}")
        print(f"    Top-10% concentration: {metrics['top_10pct_concentration']:.4f}")

    return results


# ─── Experiment 4: Steering Reliability ────────────────────────────────────────

def run_reliability_experiment(model, hook_points, steering_vectors, target_kl=2.0):
    """Measure steering effect variance across different prompts."""
    print(f"\n=== Experiment 4: Steering Reliability (KL={target_kl}) ===")
    results = {}

    # Diverse prompts for reliability test
    test_prompts = NEUTRAL_PROMPTS

    for hp_name, hp in hook_points.items():
        sv = steering_vectors[hp]
        sentiments = []
        kl_divs = []

        for prompt in tqdm(test_prompts, desc=f"  {hp_name}"):
            alpha = find_alpha_for_kl(model, prompt, hp, sv, target_kl, tol=0.1)
            actual_kl = compute_kl_divergence(model, prompt, hp, sv, alpha)
            text = generate_steered_text(model, prompt, hp, sv, alpha, max_tokens=40)
            sentiment = simple_sentiment_score(text)
            sentiments.append(sentiment)
            kl_divs.append(actual_kl)

        results[hp_name] = {
            "mean_sentiment": float(np.mean(sentiments)),
            "std_sentiment": float(np.std(sentiments)),
            "cv_sentiment": float(np.std(sentiments) / (abs(np.mean(sentiments)) + 1e-8)),
            "mean_kl": float(np.mean(kl_divs)),
            "std_kl": float(np.std(kl_divs)),
            "sentiments": [float(s) for s in sentiments],
        }
        print(f"  {hp_name}: sentiment={np.mean(sentiments):.3f} ± {np.std(sentiments):.3f}, "
              f"CV={results[hp_name]['cv_sentiment']:.3f}")

    return results


# ─── Experiment 5: Steering Vector Geometry ────────────────────────────────────

def run_geometry_experiment(steering_vectors, hook_points):
    """Analyze the geometry of steering vectors in each space."""
    print("\n=== Experiment 5: Steering Vector Geometry ===")
    results = {}

    for hp_name, hp in hook_points.items():
        sv = steering_vectors[hp].numpy()

        # Norm of steering vector
        norm = float(np.linalg.norm(sv))

        # Sparsity of steering vector itself
        threshold = 0.01 * np.abs(sv).max()
        sv_sparsity = float((np.abs(sv) < threshold).mean())

        # Entropy of absolute values (higher = more distributed)
        abs_sv = np.abs(sv) / (np.abs(sv).sum() + 1e-8)
        entropy = float(-np.sum(abs_sv * np.log(abs_sv + 1e-12)))
        max_entropy = float(np.log(len(sv)))

        # Top-k dimensions needed for 90% of vector norm
        sorted_sq = np.sort(sv**2)[::-1]
        cumsum = np.cumsum(sorted_sq)
        total = cumsum[-1]
        dims_for_90 = int(np.searchsorted(cumsum, 0.9 * total) + 1)
        frac_for_90 = dims_for_90 / len(sv)

        results[hp_name] = {
            "norm": norm,
            "sparsity": sv_sparsity,
            "entropy": entropy,
            "max_entropy": max_entropy,
            "normalized_entropy": float(entropy / max_entropy),
            "dims_for_90pct_norm": dims_for_90,
            "frac_for_90pct_norm": float(frac_for_90),
            "dim": len(sv),
        }
        print(f"  {hp_name} (dim={len(sv)}):")
        print(f"    Norm: {norm:.4f}")
        print(f"    Sparsity: {sv_sparsity:.4f}")
        print(f"    Normalized entropy: {entropy/max_entropy:.4f}")
        print(f"    Dims for 90% norm: {dims_for_90} ({frac_for_90:.2%})")

    return results


# ─── Visualization ─────────────────────────────────────────────────────────────

def create_visualizations(steering_results, linearity_results, sparsity_results,
                          reliability_results, geometry_results, all_results):
    """Create all plots."""
    print("\n=== Creating Visualizations ===")

    # Colors for each hook point
    colors = {
        "resid_stream": "#2196F3",
        "mlp_intermediate": "#FF5722",
        "mlp_output": "#4CAF50",
    }

    # 1. Steering effectiveness: sentiment vs KL budget
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for hp_name, data in steering_results.items():
        kls = data["kl_targets"]
        mean_sents = [np.mean(s) for s in data["sentiment_scores"]]
        std_sents = [np.std(s) for s in data["sentiment_scores"]]
        axes[0].errorbar(kls, mean_sents, yerr=std_sents, marker='o',
                         label=hp_name, color=colors.get(hp_name, 'gray'), capsize=3)

    axes[0].set_xlabel("KL Divergence Budget")
    axes[0].set_ylabel("Sentiment Score (positive steering)")
    axes[0].set_title("Steering Effectiveness: Sentiment vs KL Budget")
    axes[0].legend()
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0].grid(True, alpha=0.3)

    for hp_name, data in steering_results.items():
        kls = data["kl_targets"]
        mean_ppls = [np.mean(p) for p in data["perplexities"]]
        std_ppls = [np.std(p) for p in data["perplexities"]]
        axes[1].errorbar(kls, mean_ppls, yerr=std_ppls, marker='s',
                         label=hp_name, color=colors.get(hp_name, 'gray'), capsize=3)

    axes[1].set_xlabel("KL Divergence Budget")
    axes[1].set_ylabel("Perplexity")
    axes[1].set_title("Fluency: Perplexity vs KL Budget")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "steering_effectiveness.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Linearity comparison (bar chart)
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(linearity_results.keys())
    accs = [linearity_results[n]["mean_accuracy"] for n in names]
    stds = [linearity_results[n]["std_accuracy"] for n in names]
    dims = [linearity_results[n]["dim"] for n in names]
    bar_colors = [colors.get(n, 'gray') for n in names]

    bars = ax.bar(names, accs, yerr=stds, color=bar_colors, capsize=5, alpha=0.8)
    ax.set_ylabel("Linear Probe Accuracy (5-fold CV)")
    ax.set_title("Linear Separability of Sentiment in Different Spaces")
    ax.set_ylim(0.5, 1.05)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    for bar, dim in zip(bars, dims):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'd={dim}', ha='center', fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "linearity_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Sparsity comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics_to_plot = [
        ("exact_zero_fraction", "Exact Zero Fraction", "Higher = Sparser"),
        ("gini_coefficient", "Gini Coefficient", "Higher = Sparser"),
        ("top_10pct_concentration", "Top-10% Concentration", "Higher = More Concentrated"),
    ]

    for ax, (metric, title, ylabel) in zip(axes, metrics_to_plot):
        vals = [sparsity_results[n][metric] for n in names]
        bars = ax.bar(names, vals, color=bar_colors, alpha=0.8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "sparsity_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Reliability comparison (box plot of sentiments)
    fig, ax = plt.subplots(figsize=(8, 5))
    data_for_box = [reliability_results[n]["sentiments"] for n in names]
    bp = ax.boxplot(data_for_box, labels=names, patch_artist=True)
    for patch, color in zip(bp['boxes'], bar_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Sentiment Score (positive steering)")
    ax.set_title("Steering Reliability: Sentiment Distribution Across Prompts")
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "reliability_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 5. Steering vector geometry
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Normalized entropy
    entropies = [geometry_results[n]["normalized_entropy"] for n in names]
    axes[0].bar(names, entropies, color=bar_colors, alpha=0.8)
    axes[0].set_ylabel("Normalized Entropy")
    axes[0].set_title("Steering Vector Entropy (Distribution of Information)")
    axes[0].grid(True, alpha=0.3, axis='y')

    # Fraction of dims for 90% norm
    fracs = [geometry_results[n]["frac_for_90pct_norm"] for n in names]
    axes[1].bar(names, fracs, color=bar_colors, alpha=0.8)
    axes[1].set_ylabel("Fraction of Dimensions")
    axes[1].set_title("Fraction of Dims for 90% of Steering Vector Norm")
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "geometry_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 6. Summary dashboard
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Steering in Expanded MLP Space: Summary Dashboard", fontsize=14, fontweight='bold')

    # Panel 1: Steering effectiveness at KL=2.0
    kl_idx = steering_results[names[0]]["kl_targets"].index(2.0) if 2.0 in steering_results[names[0]]["kl_targets"] else 1
    sents = [np.mean(steering_results[n]["sentiment_scores"][kl_idx]) for n in names]
    sent_stds = [np.std(steering_results[n]["sentiment_scores"][kl_idx]) for n in names]
    axes[0,0].bar(names, sents, yerr=sent_stds, color=bar_colors, capsize=5, alpha=0.8)
    axes[0,0].set_title(f"Sentiment at KL≈{steering_results[names[0]]['kl_targets'][kl_idx]}")
    axes[0,0].set_ylabel("Sentiment Score")
    axes[0,0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Panel 2: Linear probe accuracy
    axes[0,1].bar(names, accs, yerr=stds, color=bar_colors, capsize=5, alpha=0.8)
    axes[0,1].set_title("Linear Probe Accuracy")
    axes[0,1].set_ylabel("Accuracy")
    axes[0,1].set_ylim(0.5, 1.05)

    # Panel 3: Exact zero fraction
    zeros = [sparsity_results[n]["exact_zero_fraction"] for n in names]
    axes[0,2].bar(names, zeros, color=bar_colors, alpha=0.8)
    axes[0,2].set_title("Activation Sparsity (Exact Zeros)")
    axes[0,2].set_ylabel("Fraction of Zeros")

    # Panel 4: Reliability (CV of sentiment)
    cvs = [reliability_results[n]["cv_sentiment"] for n in names]
    axes[1,0].bar(names, cvs, color=bar_colors, alpha=0.8)
    axes[1,0].set_title("Steering Variance (CV)")
    axes[1,0].set_ylabel("Coefficient of Variation")

    # Panel 5: Steering vector concentration
    axes[1,1].bar(names, fracs, color=bar_colors, alpha=0.8)
    axes[1,1].set_title("Steering Vector Concentration")
    axes[1,1].set_ylabel("Frac dims for 90% norm")

    # Panel 6: Gini coefficient
    ginis = [sparsity_results[n]["gini_coefficient"] for n in names]
    axes[1,2].bar(names, ginis, color=bar_colors, alpha=0.8)
    axes[1,2].set_title("Gini Coefficient")
    axes[1,2].set_ylabel("Gini (higher = sparser)")

    for ax in axes.flat:
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "summary_dashboard.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Plots saved to {PLOTS_DIR}")


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    start_time = time.time()

    # Print environment info
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.cuda.is_available()}, Device: {DEVICE}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Load model
    model = load_model()

    # Define hook points for intervention comparison
    # We test layers 6-10 (middle layers, where steering is most effective)
    # and average results across them
    LAYERS_TO_TEST = [6, 7, 8, 9, 10]

    # For the main comparison, we use a single representative layer (layer 8, middle)
    MAIN_LAYER = 8

    hook_points = {
        "resid_stream": f"blocks.{MAIN_LAYER}.hook_resid_post",
        "mlp_intermediate": f"blocks.{MAIN_LAYER}.mlp.hook_post",
        "mlp_output": f"blocks.{MAIN_LAYER}.hook_mlp_out",
    }

    print(f"\nHook points (layer {MAIN_LAYER}):")
    for name, hp in hook_points.items():
        print(f"  {name}: {hp}")

    # ─── Collect activations ───────────────────────────────────────────────────
    print("\n=== Collecting Activations ===")
    hp_list = list(hook_points.values())

    print("  Positive prompts...")
    pos_acts = collect_activations(model, POSITIVE_PROMPTS, hp_list)
    print("  Negative prompts...")
    neg_acts = collect_activations(model, NEGATIVE_PROMPTS, hp_list)

    for hp in hp_list:
        print(f"  {hp}: pos={pos_acts[hp].shape}, neg={neg_acts[hp].shape}")

    # ─── Compute steering vectors ──────────────────────────────────────────────
    print("\n=== Computing Steering Vectors ===")
    steering_vectors = {}
    for hp in hp_list:
        steering_vectors[hp] = compute_steering_vectors(pos_acts[hp], neg_acts[hp])
        print(f"  {hp}: norm={steering_vectors[hp].norm():.4f}")

    # ─── Run experiments ───────────────────────────────────────────────────────
    # Experiment 1: Steering effectiveness
    steering_results = run_steering_experiment(
        model, hook_points, steering_vectors,
        target_kls=[0.5, 1.0, 2.0, 5.0]
    )

    # Experiment 2: Linearity
    linearity_results = run_linearity_experiment(pos_acts, neg_acts, hook_points)

    # Experiment 3: Sparsity
    sparsity_results = run_sparsity_experiment(pos_acts, neg_acts, hook_points)

    # Experiment 4: Reliability
    reliability_results = run_reliability_experiment(
        model, hook_points, steering_vectors, target_kl=2.0
    )

    # Experiment 5: Geometry
    geometry_results = run_geometry_experiment(steering_vectors, hook_points)

    # ─── Multi-layer analysis ──────────────────────────────────────────────────
    print("\n=== Multi-Layer Analysis ===")
    layer_linearity = {"resid_stream": {}, "mlp_intermediate": {}, "mlp_output": {}}

    for layer in LAYERS_TO_TEST:
        layer_hooks = {
            "resid_stream": f"blocks.{layer}.hook_resid_post",
            "mlp_intermediate": f"blocks.{layer}.mlp.hook_post",
            "mlp_output": f"blocks.{layer}.hook_mlp_out",
        }
        hp_list_layer = list(layer_hooks.values())

        pos_acts_l = collect_activations(model, POSITIVE_PROMPTS, hp_list_layer)
        neg_acts_l = collect_activations(model, NEGATIVE_PROMPTS, hp_list_layer)

        for hp_name, hp in layer_hooks.items():
            pos = pos_acts_l[hp].numpy()
            neg = neg_acts_l[hp].numpy()
            X = np.concatenate([pos, neg], axis=0)
            y = np.array([1]*len(pos) + [0]*len(neg))
            clf = LogisticRegression(max_iter=1000, random_state=SEED)
            scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
            layer_linearity[hp_name][layer] = float(scores.mean())

        print(f"  Layer {layer}: " + ", ".join(
            f"{n}={layer_linearity[n][layer]:.3f}" for n in layer_linearity
        ))

    # Plot multi-layer linearity
    fig, ax = plt.subplots(figsize=(10, 5))
    colors_map = {"resid_stream": "#2196F3", "mlp_intermediate": "#FF5722", "mlp_output": "#4CAF50"}
    for hp_name in layer_linearity:
        layers = sorted(layer_linearity[hp_name].keys())
        accs = [layer_linearity[hp_name][l] for l in layers]
        ax.plot(layers, accs, 'o-', label=hp_name, color=colors_map[hp_name])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Linear Probe Accuracy")
    ax.set_title("Linear Separability Across Layers")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "layer_linearity.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ─── Compile all results ───────────────────────────────────────────────────
    all_results = {
        "config": {
            "model": "gpt2",
            "main_layer": MAIN_LAYER,
            "layers_tested": LAYERS_TO_TEST,
            "n_positive_prompts": len(POSITIVE_PROMPTS),
            "n_negative_prompts": len(NEGATIVE_PROMPTS),
            "n_neutral_prompts": len(NEUTRAL_PROMPTS),
            "seed": SEED,
            "device": DEVICE,
        },
        "steering_effectiveness": {},
        "linearity": linearity_results,
        "sparsity": sparsity_results,
        "reliability": reliability_results,
        "geometry": geometry_results,
        "layer_linearity": layer_linearity,
    }

    # Convert steering results (remove generation text for JSON)
    for hp_name, data in steering_results.items():
        all_results["steering_effectiveness"][hp_name] = {
            "kl_targets": data["kl_targets"],
            "mean_sentiment": [float(np.mean(s)) for s in data["sentiment_scores"]],
            "std_sentiment": [float(np.std(s)) for s in data["sentiment_scores"]],
            "mean_perplexity": [float(np.mean(p)) for p in data["perplexities"]],
            "std_perplexity": [float(np.std(p)) for p in data["perplexities"]],
            "mean_alpha": [float(np.mean(a)) for a in data["alphas"]],
        }

    # Save example generations
    all_results["example_generations"] = {}
    for hp_name, data in steering_results.items():
        all_results["example_generations"][hp_name] = {}
        for i, kl in enumerate(data["kl_targets"]):
            all_results["example_generations"][hp_name][str(kl)] = data["generations"][i][:3]

    # Save results
    with open(RESULTS_DIR / "all_results.json", 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR / 'all_results.json'}")

    # Create visualizations
    create_visualizations(steering_results, linearity_results, sparsity_results,
                          reliability_results, geometry_results, all_results)

    elapsed = time.time() - start_time
    print(f"\n=== Total time: {elapsed/60:.1f} minutes ===")


if __name__ == "__main__":
    main()
