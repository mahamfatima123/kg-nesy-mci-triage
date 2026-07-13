import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.env.mci_env import MCIEnv, ACTION_INDEX


def section(n, title):
    print(f"\n{'='*150}")
    print(f"TEST {n}: {title}")
    print('='*150)


def new_patient_banner(env, why):
    """Explicitly announce that a fresh episode/patient was just generated,
    instead of letting it happen silently. `why` explains what kind of
    patient this test needed to go find."""
    print(f"\n  -- new patient generated ({why}) --")
    env.render()


# ── TEST 1 ──────────────────────────────────────────────────────────
def test_full_episode_no_kg():
    section(1, "Does a full 10-patient episode run without crashing? "
               "(no KG constraint)")
    env = MCIEnv(use_kg_constraint=False)
    obs, _ = env.reset()
    assert obs.shape == (12,), f"Wrong obs shape: {obs.shape}"
    assert env.action_space.n == 7

    done, steps, total_reward = False, 0, 0
    while not done:
        obs, reward, done, _, _ = env.step(env.action_space.sample())
        total_reward += reward
        steps += 1

    print(f"  Ran {steps} total env.step() calls to get through "
          f"{env.patients_seen} patients (random actions).")
    print(f"  Total reward across the whole episode: {total_reward:.1f}")
    print(f"  (This number will differ every run -- actions are random. "
          f"It is not a result, just proof the loop ran to completion.)")
    assert env.patients_seen == env.EPISODE_LENGTH
    print("  PASS: observation shape is correct, and all 10 patients were processed.")


# ── TEST 2 ──────────────────────────────────────────────────────────
def test_mask_is_well_formed():
    section(2, "For one example patient, does the action mask correctly "
               "mark some actions allowed and some blocked?")
    env_kg = MCIEnv(use_kg_constraint=True)
    obs, _ = env_kg.reset()
    new_patient_banner(env_kg, "just need any one example to check the mask on")

    mask = env_kg.get_valid_action_mask()
    print(f"\n  Action order:  [tag_immediate, tag_delayed, tag_minor, "
          f"tag_expectant, open_airway, treat, decontaminate]")
    print(f"  Valid mask:    {mask}")
    assert mask.dtype == bool and mask.any()
    print("  PASS: the mask is a real True/False array, and at least one "
          "action is allowed (so the agent is never stuck with zero options).")


# ── TEST 3 ──────────────────────────────────────────────────────────
def test_wrong_tag_gets_blocked():
    section(3, "If the agent deliberately picks a WRONG tag, does the "
               "penalty actually fire? (START Rule 1: ambulatory -> tag_minor only)")
    env_kg = MCIEnv(use_kg_constraint=True)

    found = False
    for _ in range(50):
        obs, _ = env_kg.reset()
        if obs[0] == 1.0:  # ambulatory
            found = True
            break
    assert found, "Could not find an ambulatory patient in 50 tries"
    new_patient_banner(env_kg, "needed an ambulatory patient specifically")

    print(f"\n  Deliberately forcing the WRONG action: tag_immediate "
          f"(should only ever be tag_minor for this patient)")
    _, reward, _, _, _ = env_kg.step(ACTION_INDEX["tag_immediate"])
    print(f"  Reward received: {reward}")
    assert reward == env_kg.kg_penalty
    print(f"  PASS: the forbidden action was punished with the KG penalty "
          f"({env_kg.kg_penalty}), confirming the rule has real teeth, "
          f"not just a label.")


# ── TEST 4 ──────────────────────────────────────────────────────────
def test_cbrn_rule5_blocks_treat():
    section(4, "If the agent tries to TREAT a contaminated patient before "
               "decontaminating them, does CBRN Rule 5 block it?")
    env_kg = MCIEnv(use_kg_constraint=True)

    found = False
    for _ in range(200):
        obs, _ = env_kg.reset()
        if obs[0] == 0.0 and obs[4] == 0.0:  # non-ambulatory + contaminated
            found = True
            break
    assert found, "Could not find a non-ambulatory contaminated patient in 200 tries"
    new_patient_banner(env_kg, "needed a contaminated patient specifically")

    print(f"\n  Deliberately forcing 'treat' BEFORE any decontamination")
    _, reward, _, _, _ = env_kg.step(ACTION_INDEX["treat"])
    print(f"  Reward received: {reward}")
    assert reward == env_kg.kg_penalty
    print(f"  PASS: treat was blocked with the KG penalty -- contamination "
          f"rule is enforced before any treatment is allowed.")


# ── TEST 5 ──────────────────────────────────────────────────────────
def test_decon_flow_completes_and_unlocks_treat():
    section(5, "Walking ONE contaminated patient through the full real "
               "sequence: decontaminate (N times) -> then treat should unlock")
    env_kg = MCIEnv(use_kg_constraint=True)

    found = False
    for _ in range(300):
        obs, _ = env_kg.reset()
        if obs[0] == 0.0 and obs[4] == 0.0:
            found = True
            break
    assert found, "Could not find a contaminated patient in 300 tries"
    new_patient_banner(env_kg, "needed a contaminated patient to walk through decon")

    duration = env_kg.current_state["decon_duration"]
    hazard   = env_kg.current_state["hazard_type"]
    print(f"\n  This patient's hazard is '{hazard}', which needs {duration} "
          f"decontaminate action(s) before they count as clean.")

    for i in range(duration):
        obs, reward, done, _, _ = env_kg.step(ACTION_INDEX["decontaminate"])
        print(f"    decontaminate {i+1}/{duration} -> "
              f"decontaminated={bool(obs[4])}, progress={obs[11]:.2f}, reward={reward}")

    assert obs[4] == 1.0, "Patient should be fully decontaminated now"
    assert obs[11] == 1.0, "decon_progress should read 1.0 once complete"
    print("  PASS: decontamination progress climbed correctly and finished exactly "
          "on the expected step.")

    print(f"\n  Now trying 'treat' on this SAME patient, who is freshly clean:")
    _, reward, _, _, _ = env_kg.step(ACTION_INDEX["treat"])
    print(f"  Reward received: {reward}")
    print(f"  (Note: if decontamination used most of this patient's 5-action "
          f"budget, this reward may also include a timeout penalty on top of "
          f"the normal +1.0 treat bonus -- what matters for this test is only "
          f"that it is NOT the KG penalty, i.e. treat was not blocked.)")
    assert reward != env_kg.kg_penalty
    print("  PASS: treat was allowed (not blocked) now that decontamination is done.")

    # A second, independent patient: confirm premature treat is still
    # blocked and has no silent side effect.
    found = False
    for _ in range(300):
        obs, _ = env_kg.reset()
        if obs[0] == 0.0 and obs[4] == 0.0:
            found = True
            break
    assert found, "Could not find a second contaminated patient in 300 tries"
    new_patient_banner(env_kg, "needed a SECOND, fresh contaminated patient "
                               "to confirm premature treat is still blocked")

    print(f"\n  Trying 'treat' immediately, with ZERO decontaminate actions taken:")
    _, reward, _, _, _ = env_kg.step(ACTION_INDEX["treat"])
    print(f"  Reward received: {reward}")
    assert reward == env_kg.kg_penalty
    assert env_kg.current_state["is_treated"] is False
    print("  PASS: premature treat was blocked, and did NOT silently mark "
          "the patient as treated behind the scenes.")


if __name__ == "__main__":
    test_full_episode_no_kg()
    test_mask_is_well_formed()
    test_wrong_tag_gets_blocked()
    test_cbrn_rule5_blocks_treat()
    test_decon_flow_completes_and_unlocks_treat()
    print("\n" + "="*150)
    print("ALL TESTS PASSED")
    print("="*150)
