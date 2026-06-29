import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.env.mci_env import MCIEnv, INDEX_ACTION
from src.kg.constraint_guard import (
    get_full_kg_trace, get_kg_recommended_action, create_patient
)
from train.train_dqn import train


def run_random(n_episodes=200, use_kg=False, seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)

    label = "KG-constrained" if use_kg else "Vanilla"
    env   = MCIEnv(use_kg_constraint=use_kg)
    rewards, violations, correct_rates = [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done   = False
        ep_reward = ep_violations = ep_correct = ep_tags = 0

        while not done:
            action = env.action_space.sample()
            obs, reward, done, _, info = env.step(action)
            ep_reward += reward
            if not env.episode_log[-1]["kg_valid"]:
                ep_violations += 1

        for entry in env.episode_log:
            if entry["action"].startswith("tag_"):
                ep_tags += 1
                if entry["action"] == entry["correct"]:
                    ep_correct += 1

        rewards.append(ep_reward)
        violations.append(ep_violations)
        correct_rates.append(ep_correct / ep_tags if ep_tags > 0 else 0.0)

        if ep % 50 == 0:
            print(f"  [Random {label}] EP {ep:>3} | "
                  f"Reward: {ep_reward:>8.2f} | "
                  f"Correct: {(ep_correct/ep_tags if ep_tags>0 else 0):.2f} | "
                  f"Violations: {ep_violations}")

    print(f"  → Avg reward: {np.mean(rewards):.2f} | "
          f"Violations: {np.mean(violations):.2f} | "
          f"Correct: {np.mean(correct_rates)*100:.1f}%\n")

    return rewards, violations, correct_rates

# ── Run all agents ───────────────────────────────────────────────
print("\n" + "="*65)
print("=== RANDOM AGENT (Vanilla) ===")
rand_r, rand_v, rand_c = run_random(200, use_kg=False)

print("="*65)
print("=== RANDOM AGENT (KG-constrained) ===")
rand_kg_r, rand_kg_v, rand_kg_c = run_random(200, use_kg=True)

print("="*65)
print("=== TRAINING VANILLA DQN ===")
van_r, van_v, van_c, van_kg, van_exh_acc_log, van_exh_viol_log = train(
    use_kg=False, episodes=1500, save_path="results/_devtest_vanilla.pt")
print("\n" + "="*65)
print("=== TRAINING KG-CONSTRAINED DQN ===")
kg_r, kg_v, kg_c, kg_kg, kg_exh_acc_log, kg_exh_viol_log = train(
    use_kg=True, episodes=1500, save_path="results/_devtest_kg.pt")


# ── Summary table ────────────────────────────────────────────────
print("\n" + "="*75)
print(f"{'Agent':<28} {'Reward':>12} {'Correct%':>12} {'Violations':>12}")
print("-"*75)

print(f"{'Random (no KG)':<28} "
      f"{np.mean(rand_r):>12.2f} "
      f"{np.mean(rand_c)*100:>11.1f}% "
      f"{np.mean(rand_v):>12.2f}")

print(f"{'Random (KG)':<28} "
      f"{np.mean(rand_kg_r):>12.2f} "
      f"{np.mean(rand_kg_c)*100:>11.1f}% "
      f"{np.mean(rand_kg_v):>12.2f}")

print(f"{'Vanilla DQN':<28} "
      f"{np.mean(van_r):>12.2f} "
      f"{np.mean(van_c)*100:>11.1f}% "
      f"{np.mean(van_v):>12.2f}")

print(f"{'KG-DQN':<28} "
      f"{np.mean(kg_r):>12.2f} "
      f"{np.mean(kg_c)*100:>11.1f}% "
      f"{np.mean(kg_v):>12.2f}")

print("="*75)
print("Note: Vanilla's reward/correctness above looks favorable in part because")
print("      its reward function never penalizes tagging a still-contaminated")
print("      patient -- the -50 KG penalty only fires when use_kg_constraint=True.")
print("      Vanilla's violation rate (Table below) is the metric that exposes this.")

# ── Real, checkpoint-selected exhaustive accuracy (the trustworthy metric) ──
import torch
from src.env.dqn import DQN
from src.eval.exhaustive_eval import exhaustive_accuracy

probe_env = MCIEnv()
state_dim  = probe_env.observation_space.shape[0]
action_dim = probe_env.action_space.n

van_model = DQN(state_dim, action_dim)
van_model.load_state_dict(torch.load("results/_devtest_vanilla.pt"))
van_model.eval()
van_exh_acc, van_exh_viol = exhaustive_accuracy(van_model, use_masking=False)

kg_model = DQN(state_dim, action_dim)
kg_model.load_state_dict(torch.load("results/_devtest_kg.pt"))
kg_model.eval()
kg_exh_acc, kg_exh_viol = exhaustive_accuracy(kg_model, use_masking=True)

print("\n" + "="*75)
print("REAL CHECKPOINT ACCURACY (exhaustive 120-profile sweep, noise-free)")
print("-"*75)
print(f"{'Agent':<28} {'Exhaustive acc':>16} {'Exhaustive viol':>16}")
print(f"{'Vanilla DQN':<28} {van_exh_acc*100:>15.1f}% {van_exh_viol*100:>15.1f}%")
print(f"{'KG-DQN':<28} {kg_exh_acc*100:>15.1f}% {kg_exh_viol*100:>15.1f}%")
print("="*75)
print("Note: this is the noise-free, checkpoint-selected metric Table 1 reports.")
print("      The 'Correct%' figures above are the noisy TRAINING-distribution")
print("      signal (raw per-episode average, including label-escalation noise)")
print("      and should not be compared directly to Table 1's headline numbers.")

os.remove("results/_devtest_vanilla.pt")
os.remove("results/_devtest_kg.pt")

# ── KG-agreement summary ─────────────────────────────────────────
print("\n" + "="*55)
print(f"{'Agent':<28} {'KG-agree%':>15}")
print("-"*55)

print(f"{'Vanilla DQN':<28} {np.mean(van_kg)*100:>14.1f}%")
print(f"{'KG-DQN':<28} {np.mean(kg_kg)*100:>14.1f}%")

print("="*55)
print("Note: KG-agree diverges from Correct% due to 15% clinician-escalated")
print("      borderline cases where ground truth overrides START protocol.")


# ── Plots ─────────────────────────────────────────────────────────────────────
os.makedirs("results", exist_ok=True)

def smooth(x, w=20):
    return np.convolve(x, np.ones(w)/w, mode='valid')

rand_v_mean = np.mean(rand_v)
rand_kg_v_mean = np.mean(rand_kg_v)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle("KG-DQN vs Vanilla DQN — Full Comparison", fontsize=13, fontweight='bold')

# Plot 1: Exhaustive (noise-free) accuracy over training -- the metric
# that actually matters, sampled every eval_every episodes via checkpoint
# selection. Unlike the noisy training-distribution reward, this is
# directly comparable across agents.
ax = axes[0]
van_eval_eps = np.arange(len(van_exh_acc_log)) * 20
kg_eval_eps  = np.arange(len(kg_exh_acc_log)) * 20
ax.plot(van_eval_eps, van_exh_acc_log, color='steelblue',  lw=2, marker='o', ms=3, label='Vanilla DQN')
ax.plot(kg_eval_eps,  kg_exh_acc_log,  color='darkorange', lw=2, marker='o', ms=3, label='KG-DQN')
ax.set_title("Exhaustive accuracy\n(checkpoint metric)", fontweight='bold', fontsize=11)
ax.set_xlabel("Episode"); ax.set_ylabel("Fraction correct (120 profiles)")
ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Plot 2: Exhaustive (noise-free) violation rate over training -- same
# x-axis as Plot 1, showing the safety gap directly on the metric that
# matters, rather than the noisy training-distribution reward/correctness.
ax = axes[1]
ax.plot(van_eval_eps, van_exh_viol_log, color='steelblue',  lw=2, marker='o', ms=3, label='Vanilla DQN')
ax.plot(kg_eval_eps,  kg_exh_viol_log,  color='darkorange', lw=2, marker='o', ms=3, label='KG-DQN')
ax.set_title("Exhaustive violation rate\n(checkpoint metric)", fontweight='bold', fontsize=11)
ax.set_xlabel("Episode"); ax.set_ylabel("Fraction with a violation (120 profiles)")
ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Plot 3: KG violations per episode (training distribution)
ax = axes[2]
ax.plot(smooth(van_v), color='steelblue',  lw=2, label='Vanilla DQN')
ax.plot(smooth(kg_v),  color='darkorange', lw=2, label='KG-DQN')
ax.axhline(rand_v_mean,    color='steelblue', ls=':', lw=1.5, alpha=0.7, label=f'Random avg ({rand_v_mean:.1f})')
ax.axhline(rand_kg_v_mean, color='red',       ls='--', lw=1.5, alpha=0.8, label=f'Random KG avg ({rand_kg_v_mean:.1f})')
ax.set_title("KG violations per episode\n(training distribution)", fontweight='bold', fontsize=11)
ax.set_xlabel("Episode"); ax.set_ylabel("Violations")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Plot 4: KG-agree vs Correct% gap
ax = axes[3]
ax.plot(smooth(van_c),  color='steelblue',  lw=2, ls='-',  label='Vanilla correct%')
ax.plot(smooth(van_kg), color='steelblue',  lw=2, ls='--', label='Vanilla KG-agree%')
ax.plot(smooth(kg_c),   color='darkorange', lw=2, ls='-',  label='KG-DQN correct%')
ax.plot(smooth(kg_kg),  color='darkorange', lw=2, ls='--', label='KG-DQN KG-agree%')
ax.set_title("Correct% vs KG-agree%\n(gap = borderline escalation cases)", fontweight='bold')
ax.set_xlabel("Episode"); ax.set_ylabel("Fraction")
ax.set_ylim(0, 1.05); ax.legend(fontsize=7); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("results/full_comparison.png", dpi=150, bbox_inches='tight')
print("\nSaved results/full_comparison.png")


# ── Qualitative KG traces (paper Table 2) ───────────────────────
print("\n" + "="*65)
print("QUALITATIVE KG TRACES  (for paper Table 2)")
print("="*65)

PROFILES = [
    (True,  True,  True,  True,  True,  "walking wounded — ambulatory, clean"),
    (True,  True,  True,  True,  False, "walking wounded — ambulatory, CONTAMINATED"),
    (False, False, False, False, True,  "no breath no pulse — expectant"),
    (False, False, True,  False, True,  "not breathing, has pulse — immediate"),
    (False, True,  True,  False, True,  "no commands — immediate"),
    (False, True,  True,  True,  True,  "stable — delayed, clean"),
    (False, True,  True,  True,  False, "stable — delayed, CONTAMINATED"),
]

env_trace = MCIEnv(use_kg_constraint=True)
env_trace.reset()

for i, (amb, br, pls, cmd, dec, desc) in enumerate(PROFILES):
    pid = f"trace_p{i}"
    p = create_patient(env_trace.onto, pid,
                       ambulatory=amb, breathing=br,
                       pulse=pls, follows_commands=cmd,
                       decontaminated=dec)
    print(f"\nPatient {i+1}: {desc}")
    print(f"  vitals: amb={amb} br={br} pulse={pls} cmd={cmd} dec={dec}")
    print(get_full_kg_trace(env_trace.onto, p))