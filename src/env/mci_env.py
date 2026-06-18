import gymnasium as gym
from gymnasium import spaces
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.env.patient_generator import generate_patient, state_to_obs, OBS_DIM
from src.kg.constraint_guard import (
    load_ontology, create_patient, check_action, get_valid_actions,
    get_decontamination_duration, mark_decontaminated,
    ACTION_TAG_IMMEDIATE, ACTION_TAG_DELAYED, ACTION_TAG_MINOR,
    ACTION_TAG_EXPECTANT, ACTION_OPEN_AIRWAY, ACTION_TREAT,
    ACTION_DECONTAMINATE, ALL_ACTIONS
)

ACTION_INDEX = {a: i for i, a in enumerate(ALL_ACTIONS)}
INDEX_ACTION = {i: a for i, a in enumerate(ALL_ACTIONS)}

CORRECT_TAG = {
    "tag_minor":     ACTION_TAG_MINOR,
    "tag_immediate": ACTION_TAG_IMMEDIATE,
    "tag_delayed":   ACTION_TAG_DELAYED,
    "tag_expectant": ACTION_TAG_EXPECTANT,
}


class MCIEnv(gym.Env):
    """
    Observation (11 values -- see patient_generator.state_to_obs):
        ambulatory, breathing, pulse, follows_commands, decontaminated,
        is_treated, hazard_chemical, hazard_biological, hazard_radiological,
        hazard_nuclear, decon_progress

    Actions (7 -- see ALL_ACTIONS):
        tag_immediate, tag_delayed, tag_minor, tag_expectant,
        open_airway, treat, decontaminate

    Only tag_* actions end a patient's turn. decontaminate/open_airway/
    treat keep the same patient active (up to max_actions_per_patient
    non-tag actions), so a contaminated patient now requires a real
    sequence -- decontaminate (possibly more than once, depending on
    hazard type) before treat is KG-valid (CBRN Rule 5) -- rather than
    decontamination being a fixed, unchangeable attribute.
    """

    metadata = {"render_modes": ["human"]}
    EPISODE_LENGTH = 10

    def __init__(self, owl_path="ontology/triage_v2.rdf",
                 use_kg_constraint=False, kg_penalty=-50.0,
                 max_actions_per_patient=4, step_cost=0.5):
        super().__init__()

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(ALL_ACTIONS))

        self.use_kg_constraint       = use_kg_constraint
        self.kg_penalty              = kg_penalty
        self.max_actions_per_patient = max_actions_per_patient
        self.step_cost               = step_cost

        self.onto = load_ontology(owl_path)
        self.last_kg_violation = False

        self.current_patient      = None
        self.correct_tag          = None
        self.patients_seen        = 0
        self.patient_action_count = 0
        self.episode_log          = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._single_patient_mode = False
        self.patients_seen = 0
        self.episode_log   = []
        self._load_next_patient()
        return np.array(state_to_obs(self.current_state), dtype=np.float32), {}

    def reset_single_patient(self, ambulatory, breathing, pulse, follows_commands,
                              decontaminated, hazard_type, correct_tag,
                              description=None):
        """
        Bypasses the random generator to load one exact, caller-specified
        patient, and puts the env into single-patient mode: the episode
        ends as soon as this one patient is resolved (tagged, or the
        action budget runs out) instead of loading 9 more random patients.

        Used by exhaustive stress-test evaluation, which needs to
        enumerate every possible patient profile deterministically
        rather than relying on random sampling.
        """
        self._single_patient_mode = True
        self.patients_seen = 0
        self.episode_log   = []

        self.current_state = {
            "ambulatory":       ambulatory,
            "breathing":        breathing,
            "pulse":            pulse,
            "follows_commands": follows_commands,
            "decontaminated":   decontaminated,
            "hazard_type":      hazard_type,
            "decon_steps_done": 0,
            "decon_duration":   1,
            "is_treated":       False,
            "airway_opened":    False,
            "description":      description or "stress-test patient",
        }
        self.correct_tag = correct_tag

        patient_id = "stress_test_patient"
        self.current_patient = create_patient(
            self.onto, patient_id,
            ambulatory=ambulatory, breathing=breathing, pulse=pulse,
            follows_commands=follows_commands, decontaminated=decontaminated,
            hazard_type=hazard_type,
        )
        self.current_state["decon_duration"] = get_decontamination_duration(
            self.onto, self.current_patient
        )
        self.patient_action_count = 0

        return np.array(state_to_obs(self.current_state), dtype=np.float32)

    def _load_next_patient(self):
        self.current_state, self.correct_tag = generate_patient()
        patient_id = f"patient_{self.patients_seen}"
        self.current_patient = create_patient(
            self.onto, patient_id,
            ambulatory       = self.current_state["ambulatory"],
            breathing        = self.current_state["breathing"],
            pulse            = self.current_state["pulse"],
            follows_commands = self.current_state["follows_commands"],
            decontaminated   = self.current_state["decontaminated"],
            hazard_type      = self.current_state["hazard_type"],
        )
        self.current_state["decon_duration"] = get_decontamination_duration(
            self.onto, self.current_patient
        )
        self.patient_action_count = 0

    def _next_patient_or_done(self, last_reward):
        self.patients_seen += 1
        if getattr(self, "_single_patient_mode", False):
            done = True
        else:
            done = self.patients_seen >= self.EPISODE_LENGTH
            if not done:
                self._load_next_patient()
        obs = np.array(state_to_obs(self.current_state), dtype=np.float32)
        return obs, last_reward, done, False, {"episode_log": self.episode_log}

    def _same_patient_obs(self, reward):
        obs = np.array(state_to_obs(self.current_state), dtype=np.float32)
        return obs, reward, False, False, {"episode_log": self.episode_log}

    def step(self, action_idx):
        action_name = INDEX_ACTION[action_idx]
        self.last_kg_violation = False
        is_terminal = action_name.startswith("tag_")

        kg_valid, kg_reason = check_action(
            self.onto, self.current_patient, action_name)

        if self.use_kg_constraint and not kg_valid:
            reward = self.kg_penalty
            self.last_kg_violation = True
        elif is_terminal:
            reward = +20.0 if action_name == CORRECT_TAG.get(self.correct_tag) else -10.0
        else:
            reward = self._apply_intermediate_action(action_name)

        if is_terminal:
            # tag actions always end the patient's turn, valid or not
            self._log_step(action_name, reward, kg_valid, kg_reason)
            return self._next_patient_or_done(reward)

        # non-terminal action: stays on the same patient, unless the
        # per-patient action budget just ran out without a tag
        self.patient_action_count += 1
        timed_out = self.patient_action_count >= self.max_actions_per_patient
        if timed_out:
            reward += -10.0
            kg_reason = kg_reason + " [env: action budget exhausted without a tag]"

        self._log_step(action_name, reward, kg_valid, kg_reason)

        if timed_out:
            return self._next_patient_or_done(reward)
        return self._same_patient_obs(reward)

    def _apply_intermediate_action(self, action_name):
        """Mutates patient state for non-tag actions and shapes reward.
        A small step_cost discourages stalling; each action gives a
        one-time completion bonus and a small penalty if repeated after
        it's already done (no reward farming by spamming an action)."""
        reward = -self.step_cost

        if action_name == ACTION_DECONTAMINATE:
            if not self.current_state["decontaminated"]:
                self.current_state["decon_steps_done"] += 1
                duration = self.current_state["decon_duration"]
                if self.current_state["decon_steps_done"] >= duration:
                    self.current_state["decontaminated"] = True
                    mark_decontaminated(self.onto, self.current_patient)
                    reward += 2.0   # completed decontamination
                else:
                    reward += 1.0   # progress toward decontamination
            else:
                reward -= 1.0       # redundant: already clean

        elif action_name == ACTION_TREAT:
            if not self.current_state["is_treated"]:
                self.current_state["is_treated"] = True
                reward += 1.0
            else:
                reward -= 1.0       # redundant: already treated

        elif action_name == ACTION_OPEN_AIRWAY:
            if not self.current_state["airway_opened"]:
                self.current_state["airway_opened"] = True
                reward += 1.0
            else:
                reward -= 1.0       # redundant

        return reward

    def _log_step(self, action, reward, kg_valid, kg_reason):
        self.episode_log.append({
            "patient":        self.current_state["description"],
            "action":         action,
            "correct":        self.correct_tag,
            "reward":         reward,
            "kg_valid":       kg_valid,
            "kg_reason":      kg_reason,
            "kg_violation":   self.last_kg_violation,
            "decontaminated": self.current_state["decontaminated"],
            "hazard_type":    self.current_state.get("hazard_type"),
        })

    def get_valid_action_mask(self):
        valid = get_valid_actions(self.onto, self.current_patient)
        mask = np.zeros(len(ALL_ACTIONS), dtype=bool)
        for a in valid:
            mask[ACTION_INDEX[a]] = True
        return mask

    def render(self):
        p = self.current_state
        if p["decontaminated"]:
            dec_str = "clean"
        else:
            dec_str = (f"CONTAMINATED ({p['hazard_type']}, "
                       f"{p['decon_steps_done']}/{p['decon_duration']} decon steps)")
        print(f"\nPatient [{self.patients_seen+1}/{self.EPISODE_LENGTH}] "
              f"(action {self.patient_action_count}/{self.max_actions_per_patient}): "
              f"{p['description']}")
        print(f"  amb={p['ambulatory']} br={p['breathing']} "
              f"pls={p['pulse']} cmd={p['follows_commands']} dec={dec_str} "
              f"treated={p['is_treated']}")
        print(f"  Correct tag: {self.correct_tag}")