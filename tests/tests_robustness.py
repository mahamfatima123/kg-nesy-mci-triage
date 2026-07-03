import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.env.mci_env import MCIEnv
from train.train_dqn import train
from src.eval.exhaustive_eval import evaluate_exhaustive, run_single_patient

SEEDS          = [42, 43, 44]
PRIMARY_SEED   = SEEDS[0]
TRAIN_EPISODES = 500


def save_csv(results, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ambulatory", "breathing", "pulse", "follows_commands",
                          "decontaminated", "hazard_type", "tachypneic",
                          "prior_probability", "correct_tag", "kg_recommended",
                          "final_tag", "correct", "matches_kg", "any_violation",
                          "n_actions"])
        for r in results:
            amb, br, pulse, cmd, dec, hazard, tachy = r["profile"]
            writer.writerow([amb, br, pulse, cmd, dec, hazard, tachy,
                              f"{r['prior_probability']:.6f}", r["correct_tag"],
                              r["kg_recommended"], r["final_tag"], r["correct"],
                              r["matches_kg"], r["any_violation"], r["n_actions"]])


def run_one_seed(seed):
    print(f"  Training Vanilla DQN (seed {seed})...")
    *_, vanilla_model = train(use_kg=False, episodes=TRAIN_EPISODES, seed=seed)

    print(f"  Training KG-DQN (seed {seed})...")
    *_, kg_model = train(use_kg=True, episodes=TRAIN_EPISODES, seed=seed)

    vanilla_results = evaluate_exhaustive(vanilla_model, use_masking=False)
    kg_results      = evaluate_exhaustive(kg_model,      use_masking=True)

    return vanilla_results, kg_results, kg_model


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    van_accs, van_viols = [], []
    kg_accs,  kg_viols  = [], []

    primary_vanilla_results = None
    primary_kg_results      = None
    primary_kg_model        = None

    for seed in SEEDS:
        print(f"\n{'='*60}\nSEED {seed}\n{'='*60}")

        vanilla_results, kg_results, kg_model = run_one_seed(seed)

        n              = len(vanilla_results)
        van_correct    = sum(r["correct"]       for r in vanilla_results)
        van_violations = sum(r["any_violation"] for r in vanilla_results)
        kg_correct     = sum(r["correct"]       for r in kg_results)
        kg_violations  = sum(r["any_violation"] for r in kg_results)

        print(f"  Vanilla DQN: {van_correct}/{n} correct "
              f"({100*van_correct/n:.1f}%), "
              f"{van_violations}/{n} violations "
              f"({100*van_violations/n:.1f}%)")
        print(f"  KG-DQN    : {kg_correct}/{n} correct "
              f"({100*kg_correct/n:.1f}%), "
              f"{kg_violations}/{n} violations "
              f"({100*kg_violations/n:.1f}%)")

        van_acc       = van_correct    / n
        van_viol_rate = van_violations / n
        kg_acc        = kg_correct     / n
        kg_viol_rate  = kg_violations  / n

        assert kg_viol_rate == 0.0, (
            f"Seed {seed}: KG-DQN must have zero violations by construction.")

        van_accs.append(van_acc);   van_viols.append(van_viol_rate)
        kg_accs.append(kg_acc);     kg_viols.append(kg_viol_rate)

        if seed == PRIMARY_SEED:
            primary_vanilla_results = vanilla_results
            primary_kg_results      = kg_results
            primary_kg_model        = kg_model

    # ── Multi-seed summary (Table 1 in paper) ────────────────────────
    print("\n" + "="*80)
    print(f"            MULTI-SEED SUMMARY  ({len(SEEDS)} seeds: {SEEDS})")
    print("="*80)
    print(f"{'Agent':<16} {'Accuracy (mean ± std)':>26} " f"{'Violation rate (mean ± std)':>33}")
    print("-"*80)
    print(f"{'Vanilla DQN':<16} "
          f"{np.mean(van_accs)*100:>10.1f}% ± {np.std(van_accs)*100:<6.1f}%"
          f"{np.mean(van_viols)*100:>18.1f}% ± {np.std(van_viols)*100:<6.1f}%")
    print(f"{'KG-DQN':<16} "
          f"{np.mean(kg_accs)*100:>10.1f}% ± {np.std(kg_accs)*100:<6.1f}%"
          f"{np.mean(kg_viols)*100:>18.1f}% ± {np.std(kg_viols)*100:<6.1f}%")
    print("\nPASS: KG-DQN violation rate is exactly 0.0 on every seed.")

    # ── Per-profile CSVs (primary seed) ──────────────────────────────
    save_csv(primary_vanilla_results, "results/robustness_vanilla.csv")
    save_csv(primary_kg_results,      "results/robustness_kg.csv")
    print(f"\nSaved per-profile CSVs for seed {PRIMARY_SEED}.")

    # ── Figure 1: correctness and violations vs. profile rarity ──────
    def sort_by_rarity(results):
        return sorted(results, key=lambda r: r["prior_probability"])

    van_sorted = sort_by_rarity(primary_vanilla_results)
    kg_sorted  = sort_by_rarity(primary_kg_results)

    van_correct_flags   = [int(r["correct"])      for r in van_sorted]
    van_violation_flags = [int(r["any_violation"]) for r in van_sorted]
    kg_correct_flags    = [int(r["correct"])       for r in kg_sorted]

    def rolling(x, w=8):
        return np.convolve(x, np.ones(w) / w, mode="valid")

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = [
        "Libertinus Serif", "Linux Libertine",
        "Times New Roman", "Cambria", "Georgia",
        "Liberation Serif", "DejaVu Serif",
    ]
    plt.rcParams["mathtext.fontset"] = "stix"

    COLOR_VANILLA = "#7FA6D6"   # matches "breathing" blue from the other figure
    COLOR_KG      = "#F0A868"   # matches "pulse" orange
    COLOR_VIOL    = "#D98880"   # matches "ambulatory" red
    GRID_COLOR    = "#E5E5E5"
    SPINE_COLOR   = "#CCCCCC"
    TEXT_COLOR    = "#444444"

    def style_axis(ax):
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(SPINE_COLOR)
        ax.tick_params(axis="both", length=0, colors=TEXT_COLOR)
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3), dpi=150)

    ax = axes[0]
    ax.plot(rolling(van_correct_flags), color=COLOR_VANILLA, lw=2.2, label="Vanilla DQN")
    ax.plot(rolling(kg_correct_flags),  color=COLOR_KG,      lw=2.2, label="KG-DQN")
    ax.set_title("Correctness vs. profile rarity", fontsize=13,
                 fontweight="bold", color="#222222", pad=28, loc="left")
    ax.text(0, 1.06, "left = rarest, right = most common",
            transform=ax.transAxes, fontsize=9.5, color="#666666", ha="left")
    ax.set_xlabel("Profiles sorted by prior probability (rolling avg, w=8)",
                  fontsize=10, color=TEXT_COLOR, labelpad=8)
    ax.set_ylabel("Fraction correct", fontsize=10.5, color=TEXT_COLOR)
    ax.set_ylim(0, 1.15)
    style_axis(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2,
              frameon=False, fontsize=9.5, handlelength=1.4)

    ax = axes[1]
    ax.plot(rolling(van_violation_flags), color=COLOR_VIOL, lw=2.2, label="Vanilla DQN")
    ax.axhline(0, color=COLOR_KG, lw=2.2, linestyle="-", label="KG-DQN (always 0)")
    ax.set_title("Safety violations vs. profile rarity", fontsize=13,
                 fontweight="bold", color="#222222", pad=28, loc="left")
    ax.text(0, 1.06, "left = rarest, right = most common",
            transform=ax.transAxes, fontsize=9.5, color="#666666", ha="left")
    ax.set_xlabel("Profiles sorted by prior probability (rolling avg, w=8)",
                  fontsize=10, color=TEXT_COLOR, labelpad=8)
    ax.set_ylabel("Fraction with a violation", fontsize=10.5, color=TEXT_COLOR)
    ax.set_ylim(-0.05, 1.15)
    style_axis(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2,
              frameon=False, fontsize=9.5, handlelength=1.4)

    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    fig.subplots_adjust(wspace=0.28)
    fig.savefig("results/robustness_stress_test.png", dpi=150, bbox_inches="tight")
    print(f"Saved results/robustness_stress_test.png (Figure 1 in paper).")

    # ── KG-DQN failure traces (Table 6 in paper) ─────────────────────
    kg_failures = [r for r in primary_kg_results
                   if not r["correct"] or r["any_violation"]]

    if kg_failures:
        print(f"\nKG-DQN failure cases ({len(kg_failures)}/120) "
              f"— seed {PRIMARY_SEED}, action traces:")
        trace_env = MCIEnv(use_kg_constraint=True)
        for r in sorted(kg_failures, key=lambda r: r["prior_probability"]):
            amb, br, pulse, cmd, dec, hazard, tachy = r["profile"]
            run_single_patient(trace_env, primary_kg_model, True,
                               amb, br, pulse, cmd, dec, hazard, tachy)
            print(f"\n  Profile: amb={amb} br={br} pulse={pulse} cmd={cmd} "
                  f"dec={dec} hazard={hazard} tachy={tachy} "
                  f"(p={r['prior_probability']:.4f})")
            print(f"  correct_tag={r['correct_tag']}  "
                  f"final_tag={r['final_tag']}")
            for i, entry in enumerate(trace_env.episode_log):
                print(f"    step {i+1}: action={entry['action']:<14} "
                      f"reward={entry['reward']:+.1f}  "
                      f"kg_valid={entry['kg_valid']}")
    else:
        print(f"\nKG-DQN had zero failures on seed {PRIMARY_SEED}.")