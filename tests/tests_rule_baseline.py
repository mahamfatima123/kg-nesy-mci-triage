import sys, os, itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.env.mci_env import MCIEnv, ACTION_INDEX, INDEX_ACTION
from src.env.patient_generator import HAZARD_TYPES
from src.kg.constraint_guard import get_kg_recommended_action


def deterministic_correct_tag(ambulatory, breathing, pulse, follows_commands,
                               tachypneic=False):
    if ambulatory:
        return "tag_minor"
    elif not breathing and not pulse:
        return "tag_expectant"
    elif not breathing and pulse:
        return "tag_immediate"
    elif not pulse:
        return "tag_immediate"
    elif not follows_commands:
        return "tag_immediate"
    elif tachypneic:
        return "tag_immediate"
    else:
        return "tag_delayed"


def enumerate_all_profiles():
    profiles = []
    for amb, br, pulse, cmd in itertools.product([True, False], repeat=4):
        tachy_options = [False, True] if br else [False]
        for tachy in tachy_options:
            profiles.append((amb, br, pulse, cmd, True, None, tachy))       # clean
            for hazard in HAZARD_TYPES:                                     # contaminated
                profiles.append((amb, br, pulse, cmd, False, hazard, tachy))
    return profiles


def run_single_patient_ruleonly(env, amb, br, pulse, cmd, dec, hazard, tachy=False):
    correct_tag = deterministic_correct_tag(amb, br, pulse, cmd, tachy)
    obs = env.reset_single_patient(
        ambulatory=amb, breathing=br, pulse=pulse, follows_commands=cmd,
        decontaminated=dec, hazard_type=hazard, correct_tag=correct_tag,
        tachypneic=tachy,
    )
    kg_recommended = get_kg_recommended_action(env.onto, env.current_patient)

    done = False
    any_violation = False
    n_actions = 0
    final_tag = None

    while not done:
        mask = env.get_valid_action_mask()
        tag_idx = ACTION_INDEX[kg_recommended]

        if mask[tag_idx]:
            action = tag_idx
        else:
            action = ACTION_INDEX["decontaminate"]

        obs, reward, done, _, _ = env.step(action)
        n_actions += 1
        if not env.episode_log[-1]["kg_valid"]:
            any_violation = True
        action_name = INDEX_ACTION[action]
        if action_name.startswith("tag_"):
            final_tag = action_name

    return {
        "profile":        (amb, br, pulse, cmd, dec, hazard, tachy),
        "correct_tag":    correct_tag,
        "kg_recommended": kg_recommended,
        "final_tag":      final_tag,
        "correct":        final_tag == correct_tag,
        "any_violation":  any_violation,
        "n_actions":      n_actions,
    }


if __name__ == "__main__":
    env = MCIEnv(use_kg_constraint=True)
    profiles = enumerate_all_profiles()

    results = [run_single_patient_ruleonly(env, *p) for p in profiles]

    n = len(results)
    n_correct = sum(r["correct"] for r in results)
    n_violations = sum(r["any_violation"] for r in results)
    mean_actions = sum(r["n_actions"] for r in results) / n

    print(f"Rule-only baseline (no learning): {n_correct}/{n} correct "
          f"({100*n_correct/n:.1f}%), {n_violations}/{n} violations "
          f"({100*n_violations/n:.1f}%), mean actions/patient = {mean_actions:.2f}")

    failures = [r for r in results if not r["correct"] or r["any_violation"]]
    if failures:
        print(f"\n{len(failures)} failure(s):")
        for r in failures:
            print(f"  profile={r['profile']} correct_tag={r['correct_tag']} "
                  f"final_tag={r['final_tag']} violation={r['any_violation']}")
    else:
        print("Zero failures.")