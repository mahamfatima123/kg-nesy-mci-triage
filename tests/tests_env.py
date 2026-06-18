import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.env.mci_env import MCIEnv, ACTION_INDEX


def test_env():
    print("Testing environment (no KG constraint)...")
    env = MCIEnv(use_kg_constraint=False)
    obs, _ = env.reset()
    assert obs.shape == (11,), f"Wrong obs shape: {obs.shape}"
    assert env.action_space.n == 7

    done, steps, total_reward = False, 0, 0
    while not done:
        obs, reward, done, _, _ = env.step(env.action_space.sample())
        total_reward += reward
        steps += 1

    print(f"  Episode done: {steps} raw env.step() calls for "
          f"{env.patients_seen} patients, total reward: {total_reward:.1f}")
    assert env.patients_seen == env.EPISODE_LENGTH
    print("  PASS: unconstrained env — obs shape (11,), 10 patients processed")

    print("\nTesting environment (with KG constraint)...")
    env_kg = MCIEnv(use_kg_constraint=True)
    obs, _ = env_kg.reset()
    env_kg.render()
    mask = env_kg.get_valid_action_mask()
    print(f"  Valid action mask: {mask}")
    assert mask.dtype == bool and mask.any()
    print("  PASS: action mask works")

    for _ in range(50):
        obs, _ = env_kg.reset()
        if obs[0] == 1.0:
            break
    _, reward, _, _, _ = env_kg.step(ACTION_INDEX["tag_immediate"])
    print(f"  Ambulatory patient tagged Immediate → reward: {reward}")
    assert reward == env_kg.kg_penalty
    print("  PASS: KG constraint blocks wrong action (START Rule 1)")

    print("\nTesting CBRN Rule 5 (contaminated patient + treat action)...")
    for _ in range(200):
        obs, _ = env_kg.reset()
        if obs[0] == 0.0 and obs[4] == 0.0:  # non-ambulatory + contaminated
            break
    _, reward, _, _, _ = env_kg.step(ACTION_INDEX["treat"])
    print(f"  Contaminated patient + treat → reward: {reward}")
    assert reward == env_kg.kg_penalty
    print("  PASS: CBRN Rule 5 blocks treat on contaminated patient")

    print("\nAll environment tests passed.")


def test_decon_flow():
    print("\nTesting multi-step decontamination flow...")
    env_kg = MCIEnv(use_kg_constraint=True)

    # Find a non-ambulatory, contaminated patient
    obs = None
    for _ in range(300):
        obs, _ = env_kg.reset()
        if obs[0] == 0.0 and obs[4] == 0.0:
            break
    assert obs[4] == 0.0, "Could not find a contaminated patient in 300 tries"

    duration = env_kg.current_state["decon_duration"]
    hazard   = env_kg.current_state["hazard_type"]
    print(f"  Found contaminated patient: hazard={hazard}, decon_duration={duration}")

    # Apply `decontaminate` exactly `duration` times
    for i in range(duration):
        obs, reward, done, _, _ = env_kg.step(ACTION_INDEX["decontaminate"])
        print(f"  decontaminate step {i+1}/{duration} → "
              f"decontaminated={bool(obs[4])}, progress={obs[10]:.2f}, reward={reward}")

    assert obs[4] == 1.0, "Patient should be fully decontaminated now"
    assert obs[10] == 1.0, "decon_progress should read 1.0 once complete"
    print("  PASS: decontaminate progresses and completes correctly")

    # treat should now be KG-valid
    _, reward, _, _, _ = env_kg.step(ACTION_INDEX["treat"])
    print(f"  treat after decon → reward: {reward} (expect +1.0, not kg_penalty)")
    assert reward != env_kg.kg_penalty
    print("  PASS: treat becomes valid only after decontamination completes")

    # A second contaminated patient: treat attempted too early should
    # be blocked and NOT silently mark the patient as treated.
    for _ in range(300):
        obs, _ = env_kg.reset()
        if obs[0] == 0.0 and obs[4] == 0.0:
            break
    _, reward, _, _, _ = env_kg.step(ACTION_INDEX["treat"])
    print(f"  treat before any decon → reward: {reward} (expect kg_penalty)")
    assert reward == env_kg.kg_penalty
    assert env_kg.current_state["is_treated"] is False
    print("  PASS: premature treat is blocked and has no side effect")

    print("\nAll decon-flow tests passed.")


if __name__ == "__main__":
    test_env()
    test_decon_flow()