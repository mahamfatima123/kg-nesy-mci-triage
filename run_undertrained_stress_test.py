import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from train.train_dqn import train
from src.eval.exhaustive_eval import evaluate_exhaustive


SEEDS = [42, 43, 44]
EPISODE_COUNTS = [50, 150, 500]


def summarize(results):
    n = len(results)

    acc = sum(r["correct"] for r in results) / n
    viol = sum(r["any_violation"] for r in results) / n

    return acc, viol


# ============================================================
# Stress test
# ============================================================

for episodes in EPISODE_COUNTS:

    print("\n")
    print("=" * 95)
    print(f"TRAINING EPISODES: {episodes}")
    print("=" * 95)

    print(
        f"{'Model':<25}"
        f"{'Accuracy':>22}"
        f"{'Violations':>22}"
    )

    print("-" * 95)

    for cond_name, use_kg, penalize_kg in [
        ("Penalized DQN", False, True),
        ("KG-DQN", True, True),
    ]:

        accs = []
        viols = []

        for seed in SEEDS:

            *_, model = train(
                use_kg=use_kg,
                penalize_kg=penalize_kg,
                episodes=episodes,
                seed=seed,
            )

            results = evaluate_exhaustive(
                model,
                use_masking=use_kg,
            )

            acc, viol = summarize(results)

            accs.append(acc)
            viols.append(viol)

        # Convert to percentages
        acc_mean = np.mean(accs) * 100
        acc_std = np.std(accs) * 100

        viol_mean = np.mean(viols) * 100
        viol_std = np.std(viols) * 100

        print(
            f"{cond_name:<25}"
            f"{acc_mean:7.1f}% ± {acc_std:5.1f}%"
            f"{viol_mean:7.1f}% ± {viol_std:5.1f}%"
        )

        # Individual seed results
        print(
            f"{'  Individual seeds:':<25}"
            f"Acc = "
            f"{accs[0] * 100:5.1f}%, "
            f"{accs[1] * 100:5.1f}%, "
            f"{accs[2] * 100:5.1f}%"
        )

        print(
            f"{'':<25}"
            f"Viol = "
            f"{viols[0] * 100:5.1f}%, "
            f"{viols[1] * 100:5.1f}%, "
            f"{viols[2] * 100:5.1f}%"
        )

    print("=" * 95)