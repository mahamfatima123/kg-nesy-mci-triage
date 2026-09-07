import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from train.train_dqn import train
from src.eval.exhaustive_eval import evaluate_exhaustive


SEEDS = [42, 43, 44]

CONDITIONS = {
    "Vanilla DQN": dict(use_kg=False, penalize_kg=False),
    "Penalized DQN": dict(use_kg=False, penalize_kg=True),
    "KG-DQN": dict(use_kg=True, penalize_kg=True),
}


def summarize(results):
    n = len(results)

    acc = sum(r["correct"] for r in results) / n
    viol = sum(r["any_violation"] for r in results) / n
    mean_actions = sum(r["n_actions"] for r in results) / n

    return acc, viol, mean_actions


all_results = {
    name: {"acc": [], "viol": [], "actions": []}
    for name in CONDITIONS
}


# ============================================================
# Training and evaluation
# ============================================================

for seed in SEEDS:

    print(f"\n{'=' * 78}")
    print(f"SEED {seed}")
    print(f"{'=' * 78}")

    for name, kwargs in CONDITIONS.items():

        print(f"\nTraining {name}...", flush=True)

        *_, model = train(
            use_kg=kwargs["use_kg"],
            penalize_kg=kwargs["penalize_kg"],
            episodes=500,
            seed=seed,
        )

        results = evaluate_exhaustive(
            model,
            use_masking=kwargs["use_kg"],
        )

        acc, viol, mean_actions = summarize(results)

        all_results[name]["acc"].append(acc)
        all_results[name]["viol"].append(viol)
        all_results[name]["actions"].append(mean_actions)

        print(
            f"  Accuracy:       {acc * 100:6.1f}%"
        )
        print(
            f"  Violations:     {viol * 100:6.1f}%"
        )
        print(
            f"  Actions/patient:{mean_actions:6.2f}"
        )


# ============================================================
# Final summary
# ============================================================

print("\n")
print("=" * 95)
print("FINAL SUMMARY — 500 TRAINING EPISODES")
print("Mean ± standard deviation across seeds 42, 43, 44")
print("=" * 95)

print(
    f"{'Model':<22}"
    f"{'Accuracy':>18}"
    f"{'Violations':>18}"
    f"{'Actions/patient':>20}"
)

print("-" * 95)

for name, d in all_results.items():

    acc_mean = np.mean(d["acc"]) * 100
    acc_std = np.std(d["acc"]) * 100

    viol_mean = np.mean(d["viol"]) * 100
    viol_std = np.std(d["viol"]) * 100

    actions_mean = np.mean(d["actions"])
    actions_std = np.std(d["actions"])

    print(
        f"{name:<22}"
        f"{acc_mean:7.1f}% ± {acc_std:5.1f}%"
        f"{viol_mean:7.1f}% ± {viol_std:5.1f}%"
        f"{actions_mean:8.2f} ± {actions_std:5.2f}"
    )

print("=" * 95)