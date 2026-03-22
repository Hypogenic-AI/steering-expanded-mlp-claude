"""
Enhanced evaluation: logit-based steering measurement and generation analysis.
Measures how steering shifts next-token probabilities toward positive/negative words.
"""

import json
import os
import sys
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from tqdm import tqdm

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda:0"
RESULTS_DIR = Path("/workspaces/steering-expanded-mlp-claude/results")
PLOTS_DIR = RESULTS_DIR / "plots"

# Load previous results
with open(RESULTS_DIR / "all_results.json") as f:
    prev_results = json.load(f)

# ─── Positive/Negative token sets for logit-based evaluation ──────────────────
POSITIVE_WORDS = [
    "good", "great", "excellent", "wonderful", "amazing", "fantastic",
    "beautiful", "love", "happy", "joy", "delightful", "perfect",
    "brilliant", "outstanding", "superb", "magnificent", "lovely",
    "best", "incredible", "awesome", "nice", "pleasant", "positive",
    "success", "enjoy", "smile", "laugh", "fun", "bright", "warm",
]

NEGATIVE_WORDS = [
    "bad", "terrible", "horrible", "awful", "disgusting", "hate",
    "sad", "angry", "depressing", "miserable", "dreadful", "worst",
    "ugly", "painful", "disappointing", "frustrating", "annoying",
    "poor", "nasty", "cruel", "fear", "suffer", "dark", "cold",
    "fail", "damage", "wrong", "problem", "trouble", "crisis",
]

# Contrastive prompts (same as main experiment)
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
    from transformer_lens import HookedTransformer
    print("Loading GPT-2 Small...")
    model = HookedTransformer.from_pretrained("gpt2", device=DEVICE)
    return model


def get_token_ids(model, words):
    """Get token IDs for a list of words."""
    ids = []
    for w in words:
        # Try with space prefix (common in GPT-2 tokenizer)
        tokens = model.tokenizer.encode(f" {w}", add_special_tokens=False)
        if tokens:
            ids.append(tokens[0])
    return list(set(ids))


def collect_activations(model, prompts, hook_points, batch_size=5):
    all_acts = {hp: [] for hp in hook_points}
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        tokens = model.to_tokens(batch, prepend_bos=True)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_points)
        for hp in hook_points:
            acts = cache[hp][:, -1, :]
            all_acts[hp].append(acts.cpu())
        del cache
        torch.cuda.empty_cache()
    return {hp: torch.cat(acts, dim=0) for hp, acts in all_acts.items()}


def make_steering_hook(direction, alpha):
    direction_norm = direction / (direction.norm() + 1e-8)
    direction_norm = direction_norm.to(DEVICE)
    def hook_fn(value, hook):
        value[:, :, :] = value + alpha * direction_norm.unsqueeze(0).unsqueeze(0)
        return value
    return hook_fn


def compute_sentiment_logit_diff(model, prompt, hook_point, direction, alpha,
                                  pos_ids, neg_ids):
    """
    Compute the difference in mean log-probability between positive and negative
    sentiment words at the last token position, with and without steering.
    Returns (baseline_diff, steered_diff, shift).
    """
    tokens = model.to_tokens(prompt, prepend_bos=True)

    with torch.no_grad():
        # Baseline
        baseline_logits = model(tokens)[:, -1, :]
        baseline_probs = F.softmax(baseline_logits, dim=-1)
        baseline_pos = baseline_probs[0, pos_ids].mean().item()
        baseline_neg = baseline_probs[0, neg_ids].mean().item()
        baseline_diff = baseline_pos - baseline_neg

        # Steered
        hook_fn = make_steering_hook(direction, alpha)
        steered_logits = model.run_with_hooks(
            tokens, fwd_hooks=[(hook_point, hook_fn)]
        )[:, -1, :]
        steered_probs = F.softmax(steered_logits, dim=-1)
        steered_pos = steered_probs[0, pos_ids].mean().item()
        steered_neg = steered_probs[0, neg_ids].mean().item()
        steered_diff = steered_pos - steered_neg

        # KL divergence
        kl = F.kl_div(steered_probs.log(), baseline_probs, reduction='batchmean').item()

    return baseline_diff, steered_diff, steered_diff - baseline_diff, kl


def generate_steered_text(model, prompt, hook_point, direction, alpha, max_tokens=50):
    tokens = model.to_tokens(prompt, prepend_bos=True)
    hook_fn = make_steering_hook(direction, alpha)
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_point, hook_fn)])
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)
            if next_token.item() == model.tokenizer.eos_token_id:
                break
    return model.to_string(tokens[0])


def main():
    start_time = time.time()
    model = load_model()

    MAIN_LAYER = 8
    hook_points = {
        "resid_stream": f"blocks.{MAIN_LAYER}.hook_resid_post",
        "mlp_intermediate": f"blocks.{MAIN_LAYER}.mlp.hook_post",
        "mlp_output": f"blocks.{MAIN_LAYER}.hook_mlp_out",
    }

    # Get token IDs for sentiment words
    pos_ids = get_token_ids(model, POSITIVE_WORDS)
    neg_ids = get_token_ids(model, NEGATIVE_WORDS)
    print(f"Positive token IDs: {len(pos_ids)}, Negative token IDs: {len(neg_ids)}")

    # Collect activations and compute steering vectors
    print("\nCollecting activations...")
    hp_list = list(hook_points.values())
    pos_acts = collect_activations(model, POSITIVE_PROMPTS, hp_list)
    neg_acts = collect_activations(model, NEGATIVE_PROMPTS, hp_list)

    steering_vectors = {}
    for hp in hp_list:
        steering_vectors[hp] = (pos_acts[hp].mean(dim=0) - neg_acts[hp].mean(dim=0))

    # ─── Enhanced Experiment 1: Logit-based steering measurement ───────────────
    print("\n=== Enhanced Steering Measurement (Logit-Based) ===")

    alpha_range = [0, 5, 10, 20, 50, 100, 200, 500]
    logit_results = {}

    for hp_name, hp in hook_points.items():
        sv = steering_vectors[hp]
        logit_results[hp_name] = {"alphas": [], "mean_shift": [], "std_shift": [],
                                   "mean_kl": [], "individual_shifts": []}

        for alpha in alpha_range:
            shifts = []
            kls = []
            for prompt in NEUTRAL_PROMPTS[:15]:
                _, _, shift, kl = compute_sentiment_logit_diff(
                    model, prompt, hp, sv, alpha, pos_ids, neg_ids)
                shifts.append(shift)
                kls.append(kl)

            logit_results[hp_name]["alphas"].append(alpha)
            logit_results[hp_name]["mean_shift"].append(float(np.mean(shifts)))
            logit_results[hp_name]["std_shift"].append(float(np.std(shifts)))
            logit_results[hp_name]["mean_kl"].append(float(np.mean(kls)))
            logit_results[hp_name]["individual_shifts"].append([float(s) for s in shifts])

        print(f"  {hp_name}:")
        for i, alpha in enumerate(alpha_range):
            print(f"    alpha={alpha:>4d}: shift={logit_results[hp_name]['mean_shift'][i]:+.6f}, "
                  f"KL={logit_results[hp_name]['mean_kl'][i]:.4f}")

    # ─── Plot: Sentiment shift vs KL divergence (the key plot) ─────────────────
    print("\n=== Creating Enhanced Visualizations ===")

    colors = {"resid_stream": "#2196F3", "mlp_intermediate": "#FF5722", "mlp_output": "#4CAF50"}

    # Plot 1: Shift vs KL (efficiency curve)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for hp_name in hook_points:
        kls = logit_results[hp_name]["mean_kl"]
        shifts = logit_results[hp_name]["mean_shift"]
        shift_stds = logit_results[hp_name]["std_shift"]

        axes[0].errorbar(kls, shifts, yerr=shift_stds, marker='o',
                         label=hp_name, color=colors[hp_name], capsize=3)

    axes[0].set_xlabel("KL Divergence from Baseline")
    axes[0].set_ylabel("Positive Sentiment Probability Shift")
    axes[0].set_title("Steering Efficiency: Sentiment Shift per KL Budget")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Plot 2: Shift vs alpha
    for hp_name in hook_points:
        alphas = logit_results[hp_name]["alphas"]
        shifts = logit_results[hp_name]["mean_shift"]
        axes[1].plot(alphas, shifts, 'o-', label=hp_name, color=colors[hp_name])

    axes[1].set_xlabel("Steering Strength (alpha)")
    axes[1].set_ylabel("Positive Sentiment Probability Shift")
    axes[1].set_title("Steering Response: Sentiment Shift vs Alpha")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "enhanced_steering_efficiency.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ─── Plot 2: Reliability heatmap ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Steering Reliability: Per-Prompt Sentiment Shifts", fontsize=13)

    for ax, (hp_name, data) in zip(axes, logit_results.items()):
        # Use a subset of alphas for readability
        alpha_indices = [i for i, a in enumerate(data["alphas"]) if a in [10, 50, 100, 200, 500]]
        shifts_matrix = np.array([data["individual_shifts"][i] for i in alpha_indices])
        alpha_labels = [str(data["alphas"][i]) for i in alpha_indices]

        im = ax.imshow(shifts_matrix, aspect='auto', cmap='RdBu',
                       vmin=-shifts_matrix.max(), vmax=shifts_matrix.max())
        ax.set_xlabel("Prompt Index")
        ax.set_ylabel("Alpha")
        ax.set_yticks(range(len(alpha_labels)))
        ax.set_yticklabels(alpha_labels)
        ax.set_title(hp_name)
        plt.colorbar(im, ax=ax, label="Sentiment Shift")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "reliability_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ─── Compute efficiency metric: shift per unit KL ─────────────────────────
    print("\n=== Steering Efficiency (Shift per unit KL) ===")
    efficiency_results = {}
    for hp_name in hook_points:
        kls = np.array(logit_results[hp_name]["mean_kl"])
        shifts = np.array(logit_results[hp_name]["mean_shift"])

        # Find points where KL > 0.01 (avoid division by near-zero)
        valid = kls > 0.01
        if valid.sum() > 0:
            efficiency = shifts[valid] / kls[valid]
            best_idx = np.argmax(np.abs(efficiency))
            efficiency_results[hp_name] = {
                "best_efficiency": float(efficiency[best_idx]),
                "best_alpha": int(logit_results[hp_name]["alphas"][np.where(valid)[0][best_idx]]),
                "efficiencies": [float(e) for e in efficiency],
            }
            print(f"  {hp_name}: best efficiency={efficiency[best_idx]:.6f} at alpha={efficiency_results[hp_name]['best_alpha']}")

    # ─── Generate example texts for qualitative analysis ───────────────────────
    print("\n=== Example Generations (alpha=100, positive steering) ===")
    example_generations = {}
    test_prompt = "The weather today is"

    for hp_name, hp in hook_points.items():
        sv = steering_vectors[hp]

        # Baseline
        baseline_text = generate_steered_text(model, test_prompt, hp, sv, 0, max_tokens=40)

        # Positive steering
        pos_text = generate_steered_text(model, test_prompt, hp, sv, 100, max_tokens=40)

        # Negative steering (negate direction)
        neg_text = generate_steered_text(model, test_prompt, hp, -sv, 100, max_tokens=40)

        example_generations[hp_name] = {
            "baseline": baseline_text,
            "positive_100": pos_text,
            "negative_100": neg_text,
        }

        print(f"\n  {hp_name}:")
        print(f"    Baseline:  {baseline_text[:150]}")
        print(f"    Positive:  {pos_text[:150]}")
        print(f"    Negative:  {neg_text[:150]}")

    # More alpha values for qualitative comparison
    print("\n=== Generations at different strengths (resid vs mlp_intermediate) ===")
    multi_alpha_gens = {}
    test_prompts_short = ["The weather today is", "I went to the store and", "She thought about the situation and"]

    for prompt in test_prompts_short:
        multi_alpha_gens[prompt] = {}
        for hp_name in ["resid_stream", "mlp_intermediate"]:
            hp = hook_points[hp_name]
            sv = steering_vectors[hp]
            multi_alpha_gens[prompt][hp_name] = {}
            for alpha in [0, 50, 200]:
                text = generate_steered_text(model, prompt, hp, sv, alpha, max_tokens=30)
                multi_alpha_gens[prompt][hp_name][alpha] = text
                print(f"  [{hp_name}, a={alpha}] {prompt}... -> {text[len(prompt):len(prompt)+100]}")

    # ─── Cosine similarity between steering vectors ────────────────────────────
    print("\n=== Cross-Space Steering Vector Analysis ===")
    # Project MLP intermediate sv through W_out to residual stream for comparison
    W_out = model.blocks[MAIN_LAYER].mlp.W_out  # (d_mlp, d_model)
    mlp_sv = steering_vectors[hook_points["mlp_intermediate"]].to(DEVICE)
    projected_mlp_sv = mlp_sv @ W_out  # (d_model,)

    resid_sv = steering_vectors[hook_points["resid_stream"]].to(DEVICE)
    mlp_out_sv = steering_vectors[hook_points["mlp_output"]].to(DEVICE)

    cos_resid_projected = F.cosine_similarity(resid_sv.unsqueeze(0), projected_mlp_sv.unsqueeze(0)).item()
    cos_resid_mlpout = F.cosine_similarity(resid_sv.unsqueeze(0), mlp_out_sv.unsqueeze(0)).item()
    cos_projected_mlpout = F.cosine_similarity(projected_mlp_sv.unsqueeze(0), mlp_out_sv.unsqueeze(0)).item()

    print(f"  cos(resid_sv, projected_mlp_intermediate_sv): {cos_resid_projected:.4f}")
    print(f"  cos(resid_sv, mlp_output_sv): {cos_resid_mlpout:.4f}")
    print(f"  cos(projected_mlp_intermediate_sv, mlp_output_sv): {cos_projected_mlpout:.4f}")

    cross_space = {
        "cos_resid_projected_mlp": cos_resid_projected,
        "cos_resid_mlpout": cos_resid_mlpout,
        "cos_projected_mlp_mlpout": cos_projected_mlpout,
    }

    # ─── Neuron-level analysis of MLP intermediate steering vector ─────────────
    print("\n=== Neuron-Level Analysis of MLP Intermediate Steering Vector ===")
    mlp_sv_np = steering_vectors[hook_points["mlp_intermediate"]].numpy()

    # Sort neurons by absolute contribution
    sorted_indices = np.argsort(np.abs(mlp_sv_np))[::-1]
    cumsum = np.cumsum(mlp_sv_np[sorted_indices]**2)
    total_norm_sq = cumsum[-1]

    # How many neurons for 50%, 90%, 99% of norm
    for frac in [0.5, 0.9, 0.99]:
        n_needed = np.searchsorted(cumsum, frac * total_norm_sq) + 1
        print(f"  Neurons for {frac:.0%} of steering vector norm: {n_needed}/{len(mlp_sv_np)} ({n_needed/len(mlp_sv_np):.1%})")

    # Top 20 neurons
    print(f"\n  Top 20 neurons by |contribution|:")
    for i in range(20):
        idx = sorted_indices[i]
        print(f"    Neuron {idx}: {mlp_sv_np[idx]:+.6f} (cumulative: {cumsum[i]/total_norm_sq:.1%})")

    # Plot cumulative norm
    fig, ax = plt.subplots(figsize=(8, 5))
    fracs = cumsum / total_norm_sq
    ax.plot(range(1, len(fracs)+1), fracs, color='#FF5722')
    ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='90%')
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='50%')
    ax.set_xlabel("Number of MLP Neurons (sorted by |contribution|)")
    ax.set_ylabel("Cumulative Fraction of Steering Vector Norm²")
    ax.set_title("MLP Intermediate: Steering Vector Concentration")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "mlp_sv_concentration.png", dpi=150, bbox_inches='tight')
    plt.close()

    # ─── Save all enhanced results ─────────────────────────────────────────────
    enhanced_results = {
        "logit_based_steering": logit_results,
        "efficiency": efficiency_results,
        "example_generations": example_generations,
        "cross_space_cosines": cross_space,
        "multi_alpha_generations": multi_alpha_gens,
    }

    with open(RESULTS_DIR / "enhanced_results.json", 'w') as f:
        json.dump(enhanced_results, f, indent=2, default=str)

    elapsed = time.time() - start_time
    print(f"\n=== Enhanced evaluation completed in {elapsed/60:.1f} minutes ===")


if __name__ == "__main__":
    main()
