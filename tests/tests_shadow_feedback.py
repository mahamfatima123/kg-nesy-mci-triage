import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.eval.exhaustive_eval import exhaustive_accuracy
from train.train_dqn import train

SEEDS = [45, 46, 47]   # new seeds, on top of the 42/43/44 you already have

print("\n" + "="*75)
print("SHADOW-FEEDBACK A/B CHECK -- ADDITIONAL SEEDS (45, 46, 47)")
print("="*75)

for seed in SEEDS:
    print(f"\n--- seed {seed} ---")

    print("Training KG-DQN, shadow_feedback=False (baseline)...")
    *_, base_model = train(use_kg=True, episodes=500, seed=seed, shadow_feedback=False)

    print("Training KG-DQN, shadow_feedback=True (with fix)...")
    *_, fix_model = train(use_kg=True, episodes=500, seed=seed, shadow_feedback=True)

    b_acc, b_viol, b_shadow = exhaustive_accuracy(base_model, use_masking=True)
    f_acc, f_viol, f_shadow = exhaustive_accuracy(fix_model, use_masking=True)

    print(f"  Baseline:             acc={b_acc*100:.1f}%  viol={b_viol*100:.1f}%  shadow={b_shadow*100:.1f}%")
    print(f"  With shadow_feedback: acc={f_acc*100:.1f}%  viol={f_viol*100:.1f}%  shadow={f_shadow*100:.1f}%")
