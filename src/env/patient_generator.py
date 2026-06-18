import random
import numpy as np

HAZARD_TYPES = ["chemical", "biological", "radiological", "nuclear"]
HAZARD_ONE_HOT_ORDER = HAZARD_TYPES  # canonical order used in state_to_obs

OBS_DIM = 11  # 6 base features + 4 hazard one-hot + 1 decon progress


def generate_patient(seed=None):
    """
    Stochastic patient generator (IMPORTANT for RL learning).
    Removes deterministic patterns so KG actually matters.

    Contaminated patients (40% by default) are assigned one of 4 CBRN
    hazard types at random. The number of `decontaminate` actions
    required to clear them is read from the ontology by the environment
    (via get_decontamination_duration) once the patient's ontology
    individual is created -- the generator only decides WHICH hazard,
    not how long it takes to resolve.
    """

    if seed is not None:
        random.seed(seed)

    ambulatory       = random.random() < 0.3
    breathing        = random.random() < 0.8
    pulse            = random.random() < 0.75
    follows_commands = random.random() < 0.6
    decontaminated   = random.random() < 0.6   # 40% arrive still contaminated

    hazard_type = None
    if not decontaminated:
        hazard_type = random.choice(HAZARD_TYPES)

    # --- START-style heuristic labeling (ground truth)
    if ambulatory:
        correct_tag = "tag_minor"
        desc = "ambulatory patient"
    elif not breathing and not pulse:
        correct_tag = "tag_expectant"
        desc = "no breathing + no pulse"
    elif not breathing and pulse:
        correct_tag = "tag_immediate"
        desc = "not breathing but has pulse"
    elif not pulse:
        correct_tag = "tag_immediate"
        desc = "no pulse"
    elif not follows_commands:
        correct_tag = "tag_immediate"
        desc = "not following commands"
    else:
        if random.random() < 0.15:
            correct_tag = "tag_immediate"
            desc = "stable vitals but clinician-escalated to immediate"
        else:
            correct_tag = "tag_delayed"
            desc = "stable patient"

    if not decontaminated:
        desc += f" [CONTAMINATED:{hazard_type}]"

    state = {
        "ambulatory":       ambulatory,
        "breathing":        breathing,
        "pulse":            pulse,
        "follows_commands": follows_commands,
        "decontaminated":   decontaminated,
        "hazard_type":      hazard_type,
        "decon_steps_done": 0,
        "decon_duration":   1,     # placeholder; env overwrites via the ontology
        "is_treated":       False,
        "airway_opened":    False,
        "description":      desc,
    }

    return state, correct_tag


def state_to_obs(state):
    hazard_one_hot = [0, 0, 0, 0]
    hazard = state.get("hazard_type")
    if hazard in HAZARD_ONE_HOT_ORDER:
        hazard_one_hot[HAZARD_ONE_HOT_ORDER.index(hazard)] = 1

    if state["decontaminated"]:
        decon_progress = 1.0
    else:
        duration = state.get("decon_duration", 1) or 1
        decon_progress = min(1.0, state.get("decon_steps_done", 0) / duration)

    return np.array([
        int(state["ambulatory"]),
        int(state["breathing"]),
        int(state["pulse"]),
        int(state["follows_commands"]),
        int(state["decontaminated"]),
        int(state["is_treated"]),
        *hazard_one_hot,
        decon_progress,
    ], dtype=np.float32)