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
van_r, van_v, van_c, van_kg = train(use_kg=False, episodes=500)
print("\n" + "="*65)
print("=== TRAINING KG-CONSTRAINED DQN ===")
kg_r,  kg_v,  kg_c,  kg_kg  = train(use_kg=True,  episodes=500)


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

# ── KG-agreement summary (now meaningful — diverges from correct%) ─────────
# ── KG-agreement summary ─────────────────────────────────────────
print("\n" + "="*55)
print(f"{'Agent':<28} {'KG-agree%':>15}")
print("-"*55)

print(f"{'Vanilla DQN':<28} {np.mean(van_kg)*100:>14.1f}%")
print(f"{'KG-DQN':<28} {np.mean(kg_kg)*100:>14.1f}%")

print("="*55)
print("Note: KG-agree diverges from Correct% due to 15% clinician-escalated")
print("      borderline cases where ground truth overrides START protocol.")


# ── Convergence ──────────────────────────────────────────────────
def convergence_ep(rates, threshold=0.80, window=10):
    smoothed = np.convolve(rates, np.ones(window)/window, mode='valid')
    above = np.where(smoothed >= threshold)[0]
    return above[0] + window if len(above) > 0 else None

van_conv = convergence_ep(van_c)
kg_conv  = convergence_ep(kg_c)

print(f"\nConvergence to 80% correct tagging:")
print(f"  Vanilla DQN : episode {van_conv or 'not reached'}")
print(f"  KG-DQN      : episode {kg_conv  or 'not reached'}")
if van_conv and kg_conv:
    print(f"  KG-DQN converged {van_conv - kg_conv} episodes faster")


# ── Plots ─────────────────────────────────────────────────────────────────────
os.makedirs("results", exist_ok=True)

def smooth(x, w=20):
    return np.convolve(x, np.ones(w)/w, mode='valid')

rand_r_mean    = np.mean(rand_r)
rand_kg_r_mean = np.mean(rand_kg_r)
rand_v_mean    = np.mean(rand_v)
rand_kg_v_mean = np.mean(rand_kg_v)
rand_c_mean    = np.mean(rand_c)
rand_kg_c_mean = np.mean(rand_kg_c)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
fig.suptitle("KG-DQN vs Vanilla DQN — Full Comparison", fontsize=13, fontweight='bold')

# Plot 1: Episode reward
ax = axes[0]
ax.plot(smooth(van_r), color='steelblue',  lw=2, label='Vanilla DQN')
ax.plot(smooth(kg_r),  color='darkorange', lw=2, label='KG-DQN')
ax.axhline(rand_r_mean,    color='steelblue',  ls=':', alpha=0.6, label=f'Random ({rand_r_mean:.0f})')
ax.axhline(rand_kg_r_mean, color='darkorange', ls=':', alpha=0.6, label=f'Random KG ({rand_kg_r_mean:.0f})')
ax.set_title("Episode reward", fontweight='bold')
ax.set_xlabel("Episode"); ax.set_ylabel("Total reward")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Plot 2: Correct tag rate
ax = axes[1]
ax.plot(smooth(van_c), color='steelblue',  lw=2, label='Vanilla DQN')
ax.plot(smooth(kg_c),  color='darkorange', lw=2, label='KG-DQN')
ax.axhline(rand_c_mean,    color='steelblue',  ls=':', alpha=0.6, label=f'Random ({rand_c_mean*100:.0f}%)')
ax.axhline(rand_kg_c_mean, color='darkorange', ls=':', alpha=0.6, label=f'Random KG ({rand_kg_c_mean*100:.0f}%)')
ax.axhline(0.80, color='gray', ls='--', alpha=0.5, label='80% threshold')
if van_conv: ax.axvline(van_conv, color='steelblue',  ls='--', alpha=0.4, label=f'Van ep {van_conv}')
if kg_conv:  ax.axvline(kg_conv,  color='darkorange', ls='--', alpha=0.4, label=f'KG ep {kg_conv}')
ax.set_title("Correct tag rate", fontweight='bold')
ax.set_xlabel("Episode"); ax.set_ylabel("Fraction correct")
ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Plot 3: KG violations
ax = axes[2]
ax.plot(smooth(van_v), color='steelblue',  lw=2, label='Vanilla DQN')
ax.plot(smooth(kg_v),  color='darkorange', lw=2, label='KG-DQN')
ax.axhline(rand_v_mean,    color='steelblue', ls=':', lw=1.5, alpha=0.7, label=f'Random avg ({rand_v_mean:.1f})')
ax.axhline(rand_kg_v_mean, color='red',       ls='--', lw=1.5, alpha=0.8, label=f'Random KG avg ({rand_kg_v_mean:.1f})')
ax.set_title("KG violations per episode", fontweight='bold')
ax.set_xlabel("Episode"); ax.set_ylabel("Violations")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Plot 4: KG-agree vs Correct% gap — the new meaningful metric
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