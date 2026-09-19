"""End-to-end benchmark: bare vs schema-grounded prompts for mT5-small.

Usage:
    python run_experiment.py
"""

import json
import os
import sys


def main():
    # ── Step 1: Generate data ────────────────────────────────────────────────
    print("=" * 70)
    print("STEP 1 / 4 — Generating dataset")
    print("=" * 70)
    from generate_data import generate_dataset

    generate_dataset()

    # ── Step 2: Train Condition A (bare prompt) ──────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 2 / 4 — Training: BARE prompt (no schema info)")
    print("=" * 70)
    from train import train

    loss_bare = train(
        data_path="data/train.json",
        prompt_mode="bare",
        output_dir="checkpoints/bare",
    )

    # ── Step 3: Train Condition B (schema-grounded prompt) ───────────────────
    print("\n" + "=" * 70)
    print("STEP 3 / 4 — Training: SCHEMA-GROUNDED prompt")
    print("=" * 70)
    loss_schema = train(
        data_path="data/train.json",
        prompt_mode="schema",
        output_dir="checkpoints/schema",
    )

    # ── Step 4: Evaluate ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 4 / 4 — Evaluating both conditions")
    print("=" * 70)
    from evaluate import evaluate

    os.makedirs("results", exist_ok=True)
    all_metrics = {}

    for condition in ["bare", "schema"]:
        model_dir = f"checkpoints/{condition}"
        for split in ["test_id", "test_ood"]:
            key = f"{condition}_{split}"
            print(f"\n  Evaluating: {condition} prompt → {split}...")
            metrics, preds = evaluate(
                model_dir=model_dir,
                test_data_path=f"data/{split}.json",
                prompt_mode=condition,
            )
            all_metrics[key] = metrics

            with open(f"results/predictions_{key}.json", "w") as f:
                json.dump(preds, f, indent=2)

    # ── Results table ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    header = f"  {'Prompt':<22} {'Test Split':<14} {'Exact Match':>14} {'Field Acc':>12}"
    print(f"\n{header}")
    print("  " + "─" * 62)
    for condition in ["bare", "schema"]:
        for split in ["test_id", "test_ood"]:
            m = all_metrics[f"{condition}_{split}"]
            em = f"{m['exact_match']:.1%} ({m['exact_match_count']}/{m['total']})"
            fa = f"{m['field_accuracy']:.1%}"
            label = "Bare" if condition == "bare" else "Schema-Grounded"
            split_label = "In-Dist" if split == "test_id" else "OOD"
            print(f"  {label:<22} {split_label:<14} {em:>14} {fa:>12}")

    # Key comparison
    ood_bare = all_metrics["bare_test_ood"]["exact_match"]
    ood_schema = all_metrics["schema_test_ood"]["exact_match"]
    delta = ood_schema - ood_bare
    print(f"\n  ➜ OOD Exact-Match delta: Schema-Grounded is {delta:+.1%} vs Bare")

    # Save metrics
    with open("results/metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    # ── Loss curve plot ──────────────────────────────────────────────────────
    _plot_loss_curves(loss_bare, loss_schema)

    # ── Sample OOD predictions ───────────────────────────────────────────────
    _print_ood_samples()

    print("\n✓ All results saved to results/")


def _plot_loss_curves(loss_bare, loss_schema):
    """Save a loss-curve comparison plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs("results", exist_ok=True)

    epochs = range(1, len(loss_bare) + 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, loss_bare, "o-", label="Bare Prompt", color="#e74c3c", markersize=5)
    ax.plot(
        epochs, loss_schema, "s-", label="Schema-Grounded Prompt", color="#2ecc71", markersize=5
    )
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Training Loss", fontsize=12)
    ax.set_title("mT5-small Training Loss: Bare vs Schema-Grounded Prompt", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/loss_curves.png", dpi=150)
    plt.close(fig)
    print("\n  Loss curve → results/loss_curves.png")


def _print_ood_samples():
    """Print a few OOD prediction samples for qualitative inspection."""
    print("\n── Sample OOD Predictions (first 5) ────────────────────────────────")
    for condition in ["bare", "schema"]:
        path = f"results/predictions_{condition}_test_ood.json"
        if not os.path.exists(path):
            continue
        with open(path) as f:
            preds = json.load(f)
        label = "BARE" if condition == "bare" else "SCHEMA-GROUNDED"
        print(f"\n  [{label}]")
        for p in preds[:5]:
            mark = "✓" if p["exact_match"] else "✗"
            print(f"    {mark}  Input:  {p['input']}")
            print(f"       Target: {p['target']}")
            print(f"       Pred:   {p['prediction']}")


if __name__ == "__main__":
    main()
