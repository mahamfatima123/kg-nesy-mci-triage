import itertools
import numpy as np
import torch

from src.env.mci_env import MCIEnv, INDEX_ACTION
from src.env.patient_generator import HAZARD_TYPES
from src.kg.constraint_guard import get_kg_recommended_action

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Must match patient_generator.generate_patient()'s sampling probabilities.
P_AMBULATORY       = 0.3
P_BREATHING        = 0.8
P_PULSE            = 0.75
P_FOLLOWS_COMMANDS = 0.6
P_DECONTAMINATED   = 0.6
P_TACHYPNEIC_GIVEN_BREATHING = 0.10
P_HAZARD_GIVEN_CONTAMINATED = 1.0 / len(HAZARD_TYPES)


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


def profile_prior_probability(amb, br, pulse, cmd, dec, hazard, tachy=False):
    p = P_AMBULATORY if amb else 1 - P_AMBULATORY
    p *= P_BREATHING if br else 1 - P_BREATHING
    p *= P_PULSE if pulse else 1 - P_PULSE
    p *= P_FOLLOWS_COMMANDS if cmd else 1 - P_FOLLOWS_COMMANDS
    p *= P_DECONTAMINATED if dec else 1 - P_DECONTAMINATED
    if not dec:
        p *= P_HAZARD_GIVEN_CONTAMINATED
    if br:
        p *= P_TACHYPNEIC_GIVEN_BREATHING if tachy else 1 - P_TACHYPNEIC_GIVEN_BREATHING
    return p


def enumerate_all_profiles():

    profiles = []
    for amb, br, pulse, cmd in itertools.product([True, False], repeat=4):
        tachy_options = [False, True] if br else [False]
        for tachy in tachy_options:
            profiles.append((amb, br, pulse, cmd, True, None, tachy))       # clean
            for hazard in HAZARD_TYPES:                                     # contaminated
                profiles.append((amb, br, pulse, cmd, False, hazard, tachy))
    return profiles


def greedy_action(model, obs, mask=None):
 
    state_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
    q_values = model(state_t).detach().cpu().numpy()[0]

    unmasked_argmax = int(np.argmax(q_values))

    if mask is not None:
        masked_q = q_values.copy()
        masked_q[~mask] = -1e9
        action = int(np.argmax(masked_q))
        shadow_violation = not bool(mask[unmasked_argmax])
    else:
        action = unmasked_argmax
        shadow_violation = False

    return action, shadow_violation


def run_single_patient(env, model, use_masking, amb, br, pulse, cmd, dec, hazard, tachy=False):
    correct_tag = deterministic_correct_tag(amb, br, pulse, cmd, tachy)
    obs = env.reset_single_patient(
        ambulatory=amb, breathing=br, pulse=pulse, follows_commands=cmd,
        decontaminated=dec, hazard_type=hazard, correct_tag=correct_tag,
        tachypneic=tachy,
    )
    kg_recommended = get_kg_recommended_action(env.onto, env.current_patient)

    done = False
    any_violation = False
    n_shadow_violations = 0
    n_actions = 0
    final_tag = None

    while not done:
        mask = env.get_valid_action_mask() if use_masking else None
        action, shadow_violation = greedy_action(model, obs, mask)
        if shadow_violation:
            n_shadow_violations += 1
        obs, reward, done, _, _ = env.step(action)
        n_actions += 1
        if not env.episode_log[-1]["kg_valid"]:
            any_violation = True
        action_name = INDEX_ACTION[action]
        if action_name.startswith("tag_"):
            final_tag = action_name

    return {
        "profile":               (amb, br, pulse, cmd, dec, hazard, tachy),
        "correct_tag":           correct_tag,
        "kg_recommended":        kg_recommended,
        "final_tag":             final_tag,
        "correct":               final_tag == correct_tag,
        "matches_kg":            final_tag == kg_recommended,
        "any_violation":         any_violation,
        "n_shadow_violations":   n_shadow_violations,
        "any_shadow_violation":  n_shadow_violations > 0,
        "n_actions":             n_actions,
        "prior_probability":     profile_prior_probability(amb, br, pulse, cmd, dec, hazard, tachy),
    }


def evaluate_exhaustive(model, use_masking, owl_path="ontology/OWL_Ontology.rdf"):
    env = MCIEnv(owl_path=owl_path, use_kg_constraint=use_masking)
    return [run_single_patient(env, model, use_masking, *p) for p in enumerate_all_profiles()]


def exhaustive_accuracy(model, use_masking, owl_path="ontology/OWL_Ontology.rdf"):

    results = evaluate_exhaustive(model, use_masking, owl_path=owl_path)
    n = len(results)
    n_correct = sum(r["correct"] for r in results)
    n_violations = sum(r["any_violation"] for r in results)
    n_shadow = sum(r["any_shadow_violation"] for r in results)
    return n_correct / n, n_violations / n, n_shadow / n