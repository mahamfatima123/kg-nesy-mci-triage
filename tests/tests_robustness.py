import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import csv
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.env.mci_env import MCIEnv
from src.env.dqn import DQN
from train.train_dqn import train
from src.eval.exhaustive_eval import (
    evaluate_exhaustive, run_single_patient, enumerate_all_profiles
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Three independent training seeds. Both agents are trained and evaluated
# fresh for each seed, so the headline accuracy/violation numbers can be
# reported as mean +/- std instead of a single run. The KG-DQN violation
# rate is guaranteed to be 0.0 on every seed by construction (masking), so
# averaging it doesn't change the safety claim -- it just makes the
# *accuracy* comparison something a reviewer can actually trust.
#
# If this is too slow on your machine, drop to SEEDS = [42, 43] (training
# time roughly scales with len(SEEDS), since each seed trains BOTH agents
# from scratch).
SEEDS = [42, 43, 44]
PRIMARY_SEED = SEEDS[0]  # used for CSVs, plots, action traces, budget test

TRAIN_EPISODES = 500


def summarize(results, label):
    n = len(results)
    n_correct    = sum(r["correct"] for r in results)
    n_violations = sum(r["any_violation"] for r in results)
    print(f"\n{label}: {n_correct}/{n} correct ({100*n_correct/n:.1f}%), "
          f"{n_violations}/{n} profiles had at least one KG violation "
          f"({100*n_violations/n:.1f}%)")
    return n_correct / n, n_violations / n


def save_csv(results, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ambulatory", "breathing", "pulse", "follows_commands",
                          "decontaminated", "hazard_type", "prior_probability",
                          "correct_tag", "kg_recommended", "final_tag",
                          "correct", "matches_kg", "any_violation", "n_actions"])
        for r in results:
            amb, br, pulse, cmd, dec, hazard = r["profile"]
            writer.writerow([amb, br, pulse, cmd, dec, hazard,
                              f"{r['prior_probability']:.6f}", r["correct_tag"],
                              r["kg_recommended"], r["final_tag"], r["correct"],
                              r["matches_kg"], r["any_violation"], r["n_actions"]])


def run_one_seed(seed, van_ckpt, kg_ckpt):
    """Train both agents from scratch for one seed and return their
    exhaustive 80-profile evaluation results plus the loaded models."""
    probe_env = MCIEnv()
    state_dim  = probe_env.observation_space.shape[0]
    action_dim = probe_env.action_space.n

    print(f"\n--- Seed {seed}: training Vanilla DQN ---")
    train(use_kg=False, episodes=TRAIN_EPISODES, seed=seed, save_path=van_ckpt)
    print(f"\n--- Seed {seed}: training KG-DQN ---")
    train(use_kg=True, episodes=TRAIN_EPISODES, seed=seed, save_path=kg_ckpt)

    vanilla_model = DQN(state_dim=state_dim, action_dim=action_dim).to(device)
    vanilla_model.load_state_dict(torch.load(van_ckpt, map_location=device))
    vanilla_model.eval()

    kg_model = DQN(state_dim=state_dim, action_dim=action_dim).to(device)
    kg_model.load_state_dict(torch.load(kg_ckpt, map_location=device))
    kg_model.eval()

    vanilla_results = evaluate_exhaustive(vanilla_model, use_masking=False)
    kg_results      = evaluate_exhaustive(kg_model, use_masking=True)

    return vanilla_results, kg_results, vanilla_model, kg_model


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    van_accs, van_viols = [], []
    kg_accs,  kg_viols  = [], []

    # Kept aside for the single-seed deep dive below (CSVs, plots,
    # rarity analysis, action traces, budget test) -- multi-seed
    # averaging is great for the headline numbers but a qualitative
    # failure analysis only makes sense for one concrete run.
    primary_vanilla_results = None
    primary_kg_results      = None
    primary_kg_model        = None

    for seed in SEEDS:
        van_ckpt = f"results/vanilla_dqn_seed{seed}.pt"
        kg_ckpt  = f"results/kg_dqn_seed{seed}.pt"

        print("\n" + "=" * 70)
        print(f"SEED {seed}")
        print("=" * 70)

        vanilla_results, kg_results, vanilla_model, kg_model = run_one_seed(
            seed, van_ckpt, kg_ckpt
        )

        van_acc, van_viol_rate = summarize(
            vanilla_results, f"[seed {seed}] Vanilla DQN (no masking)")
        kg_acc, kg_viol_rate = summarize(
            kg_results, f"[seed {seed}] KG-DQN (masked)")

        assert kg_viol_rate == 0.0, (
            f"Seed {seed}: KG-DQN should have zero violations across ALL 80 "
            "profiles by construction (masking), not just the training distribution."
        )

        van_accs.append(van_acc);  van_viols.append(van_viol_rate)
        kg_accs.append(kg_acc);    kg_viols.append(kg_viol_rate)

        if seed == PRIMARY_SEED:
            primary_vanilla_results = vanilla_results
            primary_kg_results      = kg_results
            primary_kg_model        = kg_model
            # Also save under the canonical (no-seed-suffix) filenames,
            # so any other script that loads results/vanilla_dqn.pt /
            # results/kg_dqn.pt directly keeps working unmodified.
            torch.save(vanilla_model.state_dict(), "results/vanilla_dqn.pt")
            torch.save(kg_model.state_dict(), "results/kg_dqn.pt")

    print("\n" + "=" * 90)
    print(f"MULTI-SEED SUMMARY  ({len(SEEDS)} seeds: {SEEDS})")
    print("=" * 90)
    print(f"{'Agent':<16} {'Accuracy (mean ± std)':>26} {'Violation rate (mean ± std)':>30}")
    print("-" * 90)
    print(f"{'Vanilla DQN':<16} "
          f"{np.mean(van_accs)*100:>8.1f}% ± {np.std(van_accs)*100:<5.1f}%        "
          f"{np.mean(van_viols)*100:>8.1f}% ± {np.std(van_viols)*100:<5.1f}%")
    print(f"{'KG-DQN':<16} "
          f"{np.mean(kg_accs)*100:>8.1f}% ± {np.std(kg_accs)*100:<5.1f}%        "
          f"{np.mean(kg_viols)*100:>8.1f}% ± {np.std(kg_viols)*100:<5.1f}%")
    print("\nPASS: KG-DQN violation rate is exactly 0.0 on every seed, by construction.")

    # =================================================================
    # Everything below is the single-seed deep dive (primary seed only):
    # CSVs, plots, rarity analysis, qualitative failure listing, full
    # action traces for KG-DQN's failures, and a budget-sensitivity
    # follow-up on the same trained checkpoint.
    # =================================================================
    vanilla_results = primary_vanilla_results
    kg_results       = primary_kg_results

    save_csv(vanilla_results, "results/robustness_vanilla.csv")
    save_csv(kg_results, "results/robustness_kg.csv")
    print(f"\n[primary seed {PRIMARY_SEED}] Saved per-profile results to "
          "results/robustness_vanilla.csv and results/robustness_kg.csv")

    # ---- Correctness/safety vs. profile rarity ----
    def sort_by_rarity(results):
        return sorted(results, key=lambda r: r["prior_probability"])

    van_sorted = sort_by_rarity(vanilla_results)
    kg_sorted  = sort_by_rarity(kg_results)

    van_correct_flags   = [int(r["correct"]) for r in van_sorted]
    van_violation_flags = [int(r["any_violation"]) for r in van_sorted]
    kg_correct_flags    = [int(r["correct"]) for r in kg_sorted]

    def rolling(x, w=8):
        return np.convolve(x, np.ones(w) / w, mode="valid")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(rolling(van_correct_flags), color="steelblue", lw=2, label="Vanilla DQN")
    ax.plot(rolling(kg_correct_flags), color="darkorange", lw=2, label="KG-DQN")
    ax.set_title("Correctness vs. profile rarity\n(left = rarest, right = most common)")
    ax.set_xlabel("Profiles sorted by prior probability (rolling avg, w=8)")
    ax.set_ylabel("Fraction correct")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(rolling(van_violation_flags), color="crimson", lw=2, label="Vanilla DQN")
    ax.axhline(0, color="darkorange", lw=2, linestyle="--", label="KG-DQN (always 0)")
    ax.set_title("Safety violations vs. profile rarity")
    ax.set_xlabel("Profiles sorted by prior probability (rolling avg, w=8)")
    ax.set_ylabel("Fraction with a violation")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/robustness_stress_test.png", dpi=150, bbox_inches="tight")
    print(f"\n[primary seed {PRIMARY_SEED}] Saved results/robustness_stress_test.png")

    # ---- Worst-case profiles for Vanilla, rarest first ----
    worst = [r for r in vanilla_results if not r["correct"] or r["any_violation"]]
    worst_sorted = sorted(worst, key=lambda r: r["prior_probability"])
    print(f"\n[primary seed {PRIMARY_SEED}] Vanilla DQN failure cases "
          f"({len(worst)}/80), rarest profiles first:")
    for r in worst_sorted[:15]:
        amb, br, pulse, cmd, dec, hazard = r["profile"]
        print(f"  p={r['prior_probability']:.4f} | amb={amb} br={br} pulse={pulse} "
              f"cmd={cmd} dec={dec} hazard={hazard} | "
              f"correct_tag={r['correct_tag']} final_tag={r['final_tag']} "
              f"violation={r['any_violation']}")

    # =================================================================
    # NEW 1: full action-by-action traces for KG-DQN's failure cases.
    #
    # run_single_patient() doesn't return the trace, but it also doesn't
    # clear env.episode_log after it finishes -- reset_single_patient()
    # clears it at the START of each call, so reading env.episode_log
    # immediately after one run_single_patient() call gives exactly that
    # profile's full step-by-step log. We just re-run the (small number
    # of) failing profiles once more against a fresh env to capture it.
    # =================================================================
    kg_failures = [r for r in kg_results if not r["correct"] or r["any_violation"]]
    if kg_failures:
        print(f"\n[primary seed {PRIMARY_SEED}] KG-DQN failure cases "
              f"({len(kg_failures)}/80) -- full action traces:")
        trace_env = MCIEnv(use_kg_constraint=True)
        for r in sorted(kg_failures, key=lambda r: r["prior_probability"]):
            amb, br, pulse, cmd, dec, hazard = r["profile"]
            run_single_patient(trace_env, primary_kg_model, True,
                                amb, br, pulse, cmd, dec, hazard)
            trace = trace_env.episode_log
            print(f"\n  Profile: amb={amb} br={br} pulse={pulse} cmd={cmd} "
                  f"dec={dec} hazard={hazard} (p={r['prior_probability']:.4f})")
            print(f"  correct_tag={r['correct_tag']}  final_tag={r['final_tag']}")
            for i, entry in enumerate(trace):
                print(f"    step {i+1}: action={entry['action']:<14} "
                      f"reward={entry['reward']:+.1f} kg_valid={entry['kg_valid']}")
    else:
        print(f"\n[primary seed {PRIMARY_SEED}] KG-DQN had zero failures -- "
              "no action traces to show.")

    # =================================================================
    # NEW 2: budget-sensitivity check. Re-evaluate the SAME trained
    # KG-DQN checkpoint (no retraining at all) against a larger
    # per-patient action budget, purely at evaluation time. If the
    # failing profiles above resolve correctly with more steps, the
    # gap was a budget artifact; if they still never tag, it's a
    # genuine generalization gap on rare states.
    # =================================================================
    LARGER_BUDGET = 8
    print(f"\n[primary seed {PRIMARY_SEED}] Budget-sensitivity check: "
          f"re-evaluating the SAME checkpoint with max_actions_per_patient="
          f"{LARGER_BUDGET} (was 4), no retraining...")

    budget_env = MCIEnv(use_kg_constraint=True, max_actions_per_patient=LARGER_BUDGET)
    budget_results = [
        run_single_patient(budget_env, primary_kg_model, True, *p)
        for p in enumerate_all_profiles()
    ]
    summarize(budget_results, f"KG-DQN (masked, budget={LARGER_BUDGET})")
    save_csv(budget_results, "results/robustness_kg_budget8.csv")

    before_by_profile = {r["profile"]: r for r in kg_failures}
    after_by_profile  = {r["profile"]: r for r in budget_results
                          if r["profile"] in before_by_profile}

    resolved = 0
    for profile in sorted(before_by_profile,
                           key=lambda p: before_by_profile[p]["prior_probability"]):
        before = before_by_profile[profile]
        after  = after_by_profile[profile]
        newly_correct = after["correct"] and not before["correct"]
        resolved += int(newly_correct)
        print(f"  profile={profile} | budget=4 -> final_tag={before['final_tag']} "
              f"(correct={before['correct']}) | budget={LARGER_BUDGET} -> "
              f"final_tag={after['final_tag']} (correct={after['correct']})")

    if kg_failures:
        print(f"\n  => {resolved}/{len(kg_failures)} previously-failing profiles became "
              f"correct purely from a larger action budget (same weights, no retraining).")
        if resolved == len(kg_failures):
            print("  Conclusion: the gap was a BUDGET ARTIFACT, not a genuine policy gap.")
        elif resolved == 0:
            print("  Conclusion: the gap is NOT explained by budget -- looks like a "
                  "genuine generalization gap on rare states.")
        else:
            print("  Conclusion: partially budget-related -- some profiles resolve with "
                  "more steps, others don't.")