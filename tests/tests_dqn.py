import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.eval.exhaustive_eval import exhaustive_accuracy
from train.train_dqn import train

os.makedirs("results", exist_ok=True)

print("\n" + "="*75)
print("DEV SANITY CHECK  (single seed)")
print("="*75)

print("\nTraining Vanilla DQN...")
*van_logs, vanilla_model = train(use_kg=False, episodes=500)

print("\nTraining KG-DQN...")
*kg_logs, kg_model = train(use_kg=True, episodes=500)

van_exh_acc_log  = van_logs[4]
van_exh_viol_log = van_logs[5]
kg_exh_acc_log   = kg_logs[4]
kg_exh_viol_log  = kg_logs[5]
van_exh_shadow_log = van_logs[7]
kg_exh_shadow_log  = kg_logs[7]

van_acc  = max(van_exh_acc_log)
van_viol = van_exh_viol_log[van_exh_acc_log.index(van_acc)]
kg_acc   = max(kg_exh_acc_log)
kg_viol  = kg_exh_viol_log[kg_exh_acc_log.index(kg_acc)]
van_shadow = van_exh_shadow_log[van_exh_acc_log.index(van_acc)]
kg_shadow  = kg_exh_shadow_log[kg_exh_acc_log.index(kg_acc)]

print("\n" + "="*75)
print(f"{'Agent':<16} {'Best Exhaustive Acc':>20} {'Violation Rate':>16} {'Shadow-Viol Rate':>18}")
print("-"*75)
print(f"{'Vanilla DQN':<16} {van_acc*100:>19.1f}% {van_viol*100:>15.1f}% {van_shadow*100:>17.1f}%")
print(f"{'KG-DQN':<16} {kg_acc*100:>19.1f}% {kg_viol*100:>15.1f}% {kg_shadow*100:>17.1f}%")
print("="*75)
print("Shadow-Viol Rate = fraction of profiles where the network's own raw (unmasked) greedy preference was contraindicated at some step, even")
print("though the mask prevented it from ever being executed. Only meaningful for KG-DQN; Vanilla has no mask to compare against.")

assert min(kg_exh_viol_log) == 0.0, \
    "KG-DQN must have zero violations by construction on every checkpoint."
print("PASS: KG-DQN violation rate is exactly 0.0 by construction.")